"""G1 叠布「数据增强回放」场景控制器（独立于遥操作采集场景）。

职责只有一件：读取数据集里已有 episode 的 action（16 维 = 双臂 7 关节 +
夹爪），在重新随机化的场景里重放同一段关节轨迹，并把重新渲染的帧追加
写回数据集。加载本场景即自动开始，不需要 PICO、不接收手柄输入。

增强项：
  - 场景外观：布料颜色/位置/朝向、桌面颜色、两盏灯的位置与色温
  - 第三人称机位：位置/朝向/视场角小幅随机
  - 图像后处理：对比度/亮度/色偏 + 高斯噪声/随机模糊
  - 布料初始位姿：随场景随机化一起重摆

世界 YAML 通过 robots[].args 配置：
    dataset:              源数据集目录（同时是输出目录，默认遥操作数据集）
    replays_per_episode:  每个 episode 回放次数（默认 3）
    augment_seed:         增强随机种子（默认 20261117）
"""
from __future__ import annotations

import pathlib
import sys
import time

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from g1_cloth_teleop import G1ClothTeleop  # noqa: E402

DEFAULT_REPLAYS = 3


def _look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """由「机位 + 瞄准点」构造 MuJoCo 相机四元数（wxyz）。

    MuJoCo 相机沿自身 -z 观察、+y 朝上；先按右手系解出 x/y/z 三根轴，
    再用 mju_mat2Quat 转四元数。抖动 quat 会让画面歪斜，因此位置变化后
    统一用瞄准点重新求解。
    """
    fwd = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, np.asarray(up, dtype=float))
    n = np.linalg.norm(x)
    if n < 1e-6:                       # 光轴与 up 平行：换一根参考轴
        x = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(x)
    x /= n
    y = np.cross(x, fwd)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, -fwd]).ravel())
    return q


class G1ClothAugment(G1ClothTeleop):
    """复用遥操作控制器的机械臂/夹爪/抓布/渲染逻辑，只跑自动回放。"""

    def __init__(self, simulation):
        super().__init__(simulation)
        # 本场景不需要 PICO：等基类服务器线程起来后立即关掉（避免占端口）
        deadline = time.monotonic() + 2.0
        while self.server._loop is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.server.stop()
        print("[augment] 本场景不需要 PICO 遥操作，已关闭页面服务器",
              flush=True)

        self.replays_per_episode = max(
            0, int(self.args.get("replays_per_episode", DEFAULT_REPLAYS)))
        self.augment_seed = int(self.args.get("augment_seed", 20261117))

        # 第三人称机位原值 + 光轴锚点（回放结束恢复）
        self.showcase_cam = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "showcase")
        if self.showcase_cam >= 0:
            self.cam_pos0 = self.model.cam_pos[self.showcase_cam].copy()
            self.cam_quat0 = self.model.cam_quat[self.showcase_cam].copy()
            self.cam_fovy0 = float(self.model.cam_fovy[self.showcase_cam])
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, self.cam_quat0)
            fwd = R.reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
            plane_z = float(self.args.get("cam_target_z", 0.82))
            self.cam_target0 = self.cam_pos0 + fwd * (
                (plane_z - self.cam_pos0[2]) / fwd[2])

        # 源 episode -> action 在启动时一次性快照，避免把本次新写的集当源
        self.sources = self._scan_source_episodes()
        self.total_jobs = len(self.sources) * self.replays_per_episode
        self.job_queue = [(ep, acts) for ep, acts in self.sources
                          for _ in range(self.replays_per_episode)]
        self.current: np.ndarray | None = None
        self.cur_i = 0
        self.replay_done = 0
        self.img_aug: dict = {}
        self.img_rng = np.random.default_rng()
        print(f"[augment] 源 episode {len(self.sources)} 集 × "
              f"{self.replays_per_episode} 次 = {self.total_jobs} 个回放任务"
              f"（数据集 {self.dataset_dir}）", flush=True)
        self.status.update({"recording": False, "msg": "增强回放：等待开始"})
        if self.total_jobs == 0:
            self._finish()

    # ------------------------------------------------------------ 主循环

    def advance(self) -> bool:
        if self._closed or self.done:
            return False
        if self.current is None:
            if not self._start_next():
                return False
            return True
        if self.cur_i >= len(self.current):
            self._end_current()
            return not self.done
        self._step_replay()
        return True

    def _step_replay(self) -> None:
        a = self.current[self.cur_i]
        self.cur_i += 1
        # action 记录的是每 tick 结束时的关节指令：q_cmd/q_goal 同时写回，
        # 8 个物理子步保持同一指令，等价于重放原始轨迹。
        for side, off in (("left", 0), ("right", 8)):
            self.q_goal[side] = a[off:off + 7].astype(float).copy()
            self.q_cmd[side] = a[off:off + 7].astype(float).copy()
            grip = float(np.clip(a[off + 7], 0.0, 1.0)) * 255.0
            self.grip_goal[side] = grip
            self.grip_cmd[side] = grip
        for _ in range(self.ctrl_period):
            self._substep()
        self.tick += 1
        self._record_aug()

    # ------------------------------------------------------------ 任务切换

    def _start_next(self) -> bool:
        if not self.job_queue:
            self._finish()
            return False
        src_ep, actions = self.job_queue.pop(0)
        self.rng = np.random.default_rng(
            self.augment_seed + 7919 * int(src_ep) + 104729 * self.replay_done)
        self._reset_episode(randomize=True)     # 布料位姿/颜色、桌面、灯光
        self._jitter_camera()
        self._sample_image_aug()
        try:
            ep = self.writer.start_episode()
        except Exception as e:                  # noqa: BLE001
            print(f"[augment] 开集失败: {e}", flush=True)
            try:
                self.writer.abort_episode()
            except Exception:                   # noqa: BLE001
                pass
            self.job_queue.clear()
            self._finish()
            return False
        self.current = actions
        self.cur_i = 0
        self.rec_frames = 0
        self.status.update({
            "recording": True, "episode": ep, "frames": 0,
            "msg": f"增强回放 {self.replay_done + 1}/{self.total_jobs}"
                   f"（源 episode {src_ep}）"})
        print(f"[augment] ↻ 回放 {self.replay_done + 1}/{self.total_jobs}"
              f"：episode {ep} <- 源 {src_ep}（{len(actions)} 帧）", flush=True)
        return True

    def _end_current(self) -> None:
        self.current = None
        self.replay_done += 1
        try:
            info = self.writer.end_episode()
            print(f"[augment] ■ 已保存: {info}", flush=True)
        except Exception as e:                  # noqa: BLE001
            print(f"[augment] 收尾失败: {e}", flush=True)
        if not self.job_queue:
            self._finish()

    def _finish(self) -> None:
        self.done = True
        self.success = True
        self.detail = f"增强回放新增 {self.replay_done} 集"
        self.current = None
        self.job_queue.clear()
        if self.showcase_cam >= 0:
            self.model.cam_pos[self.showcase_cam] = self.cam_pos0
            self.model.cam_quat[self.showcase_cam] = self.cam_quat0
            self.model.cam_fovy[self.showcase_cam] = self.cam_fovy0
        self.status.update({"recording": False,
                            "msg": f"增强回放完成：新增 {self.replay_done} 集"})
        print(f"[augment] 完成：本场景新增 {self.replay_done} 集"
              f"（数据集共 {self.writer.episode_count} 集）", flush=True)

    # ------------------------------------------------------------ 采集

    def _record_aug(self) -> None:
        d = self.data
        state = np.concatenate([
            d.qpos[self.arm_qadrs["left"]], [self._grip_state("left")],
            d.qpos[self.arm_qadrs["right"]], [self._grip_state("right")],
        ]).astype(np.float32)
        action = np.concatenate([
            self.q_cmd["left"], [self.grip_cmd["left"] / 255.0],
            self.q_cmd["right"], [self.grip_cmd["right"] / 255.0],
        ]).astype(np.float32)
        try:
            images = {k: self._augment_image(self._render(k))
                      for k in self.cam_specs}
            self.writer.add_frame(state, action, images)
        except Exception as e:                  # noqa: BLE001
            print(f"[augment] 渲染/写帧失败，停止回放: {e}", flush=True)
            try:
                self.writer.abort_episode()
            except Exception:                   # noqa: BLE001
                pass
            self._finish()
            return
        self.rec_frames += 1
        self.status["frames"] = self.rec_frames

    # ------------------------------------------------------------ 源数据

    def _scan_source_episodes(self) -> list[tuple[int, np.ndarray]]:
        root = self.dataset_dir / "data" / "chunk-000"
        out: list[tuple[int, np.ndarray]] = []
        for p in sorted(root.glob("episode_*.parquet")):
            try:
                ep = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            acts = self._load_actions(p)
            if acts is not None and len(acts) > 0:
                out.append((ep, acts))
        out.sort(key=lambda kv: kv[0])
        return out

    @staticmethod
    def _load_actions(path: pathlib.Path) -> np.ndarray | None:
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(path, columns=["action"])
            a = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)
        except Exception as e:                  # noqa: BLE001
            print(f"[augment] 读取 {path.name} 失败: {e}", flush=True)
            return None
        if a.ndim != 2 or a.shape[1] != 16:
            print(f"[augment] {path.name} action 维度 {a.shape}，需要 16，跳过",
                  flush=True)
            return None
        return a

    # ------------------------------------------------------------ 增强

    def _jitter_camera(self) -> None:
        """第三人称机位微扰：位置 ±3.5cm、瞄准点 ±2cm、视场 ±4°。"""
        cid = self.showcase_cam
        if cid < 0:
            return
        pos = self.cam_pos0 + self.rng.uniform(-0.035, 0.035, 3)
        target = self.cam_target0 + self.rng.uniform(-0.02, 0.02, 3)
        self.model.cam_pos[cid] = pos
        self.model.cam_quat[cid] = _look_at_quat(pos, target)
        self.model.cam_fovy[cid] = self.cam_fovy0 + self.rng.uniform(-4.0, 4.0)

    def _sample_image_aug(self) -> None:
        """整集固定的图像扰动参数；逐帧噪声用独立 rng（可复现）。"""
        rng = self.rng
        self.img_aug = {
            "gain": float(rng.uniform(0.85, 1.15)),      # 对比度
            "bias": float(rng.uniform(-16.0, 16.0)),     # 亮度
            "shift": rng.uniform(-5.0, 5.0, 3),          # 通道偏置（色偏）
            "noise": float(rng.uniform(0.0, 5.0)),       # 高斯噪声
            "blur": bool(rng.random() < 0.35),           # 3x3 均值模糊
        }
        self.img_rng = np.random.default_rng(int(rng.integers(1 << 30)))

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        a = self.img_aug
        x = img.astype(np.float32)
        x = (x - 127.5) * a["gain"] + 127.5 + a["bias"] + a["shift"]
        if a["noise"] > 0.0:
            x += self.img_rng.normal(0.0, a["noise"], x.shape)
        if a["blur"]:
            p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
            x = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
                 + p[1:-1, 1:-1]) * 0.2
        return np.clip(x, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------ 收尾

    def close(self) -> None:
        if self.current is not None and self.writer.episode_index >= 0:
            try:
                print(f"[augment] 收尾回放: {self.writer.end_episode()}",
                      flush=True)
            except Exception as e:              # noqa: BLE001
                print(f"[augment] 收尾回放失败: {e}", flush=True)
        self.current = None
        self.job_queue.clear()
        super().close()


def create_controller(simulation):
    return G1ClothAugment(simulation)
