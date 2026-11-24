"""G1 双臂叠布 π0.5 策略闭环评估场景控制器。

与 g1_cloth_teleop.py 的关系：完整复用基类的站姿、浮动基座锁定、2F85
夹爪、布料合规抓持、三路相机渲染；只把「PICO 手柄 + OSC」换成「π0.5
推理输出」。本文件同样以 shared_world_controller 身份跑在内核进程里，
这样锁基座、逐物理步给布料顶点施力才能生效。

推理链路：
    _render(cam_high, cam_left_wrist) + state16
        -> 子进程 controllers/pi05_infer_worker.py（openpi .venv，jax）
        -> 10 步 action chunk（绝对关节角 + 夹爪 0~1）
        -> 逐 tick 写 q_goal/grip_goal，chunk 用完再推理一次

世界 YAML 通过 robots[].args 配置：
    model:        openpi 配置名（默认 pi05_g1_cloth_lora）
    checkpoint:   训练产出的 checkpoint 目录（必填）
    python:       openpi .venv 的解释器（默认 ~/openpi/.venv/bin/python）
    device:       cpu / cuda（默认 cpu；5070 显存常被其他进程占用）
    prompt:       语言指令，必须与训练数据集 task 一致
    max_sim_seconds:  仿真时间上限（默认 30）
    success_box:  布料质心的目标区域 {min:[x,y,z], max:[x,y,z]}
    grip_open:    动作里夹爪 1.0 对应的 ctrl 值（默认 255）
    worker_log:   worker 的 stderr 日志文件
"""
from __future__ import annotations

import os
import pathlib
import select
import struct
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from g1_cloth_teleop import G1ClothTeleop  # noqa: E402

DEFAULT_PY = pathlib.Path.home() / "openpi" / ".venv" / "bin" / "python"
DEFAULT_CHECKPOINT = (pathlib.Path.home() / "openpi" / "checkpoints"
                      / "pi05_g1_cloth_lora" / "g1_cloth" / "30000")
DEFAULT_PROMPT = "put the blanket in the box"
DEFAULT_BOX = {"min": [0.285, 0.120, 0.745], "max": [0.475, 0.400, 0.790]}
INFER_TIMEOUT = 600.0     # 首次含 XLA 编译，CPU 上要几十秒


class G1ClothPolicyEval(G1ClothTeleop):
    """π0.5 闭环推理：动作完全来自策略网络，不再读手柄。"""

    def __init__(self, simulation):
        super().__init__(simulation)

        # 本场景不需要 PICO：等基类服务器线程起来后立刻关掉（避免占端口）
        deadline = time.monotonic() + 2.0
        while self.server._loop is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.server.stop()
        print("[policy_eval] 不需要 PICO 遥操作，已关闭页面服务器", flush=True)

        # 只渲染策略需要的两路（父类按 arms 生成，这里显式钉死顺序）
        size = int(self.args.get("cam_size", 256))
        self.cam_specs = {
            "observation.images.cam_high": ("showcase", size, size),
            "observation.images.cam_left_wrist":
                (f"{self.robot_name}/cam_left_wrist", size, size),
        }

        self.model_name = str(self.args.get("model", "pi05_g1_cloth_lora"))
        self.checkpoint = pathlib.Path(
            self.args.get("checkpoint", DEFAULT_CHECKPOINT)).expanduser()
        self.prompt = str(self.args.get("prompt", DEFAULT_PROMPT))
        self.grip_open = float(self.args.get("grip_open", 255.0))
        self.max_sim_seconds = float(self.args.get("max_sim_seconds", 30.0))
        box = self.args.get("success_box") or DEFAULT_BOX
        self.box_lo = np.asarray(box["min"], dtype=float)
        self.box_hi = np.asarray(box["max"], dtype=float)
        self.hold_seconds = float(self.args.get("success_hold_seconds", 0.5))
        self.grip_release = float(self.args.get("grip_release_threshold", 0.2))

        self.queue: list[np.ndarray] = []
        self.n_infer = 0
        self._stable_since: float | None = None
        self._last_status = 0.0

        if not self.checkpoint.exists():
            raise RuntimeError(f"checkpoint 不存在: {self.checkpoint}")
        py = pathlib.Path(self.args.get("python", DEFAULT_PY)).expanduser()
        if not py.exists():
            raise RuntimeError(
                f"找不到 openpi 解释器: {py}（世界 YAML args.python 可覆盖）")

        log_path = pathlib.Path(self.args.get(
            "worker_log", "/tmp/pi05_worker.log")).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_log = open(log_path, "w", encoding="utf-8", buffering=1)
        cmd = [str(py), str(HERE / "pi05_infer_worker.py"),
               "--config", self.model_name,
               "--checkpoint", str(self.checkpoint),
               "--prompt", self.prompt,
               "--device", str(self.args.get("device", "cpu")),
               "--mem-fraction", str(self.args.get("mem_fraction", 0.55))]
        print(f"[policy_eval] 启动推理 worker: {' '.join(cmd)}", flush=True)
        self.worker = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._worker_log, cwd=str(pathlib.Path.home() / "openpi"))
        self.status.update({"recording": False,
                            "msg": f"π0.5 策略评估：{self.checkpoint.name}"})
        print(f"[policy_eval] 就绪：{self.model_name} @ {self.checkpoint}"
              f"  (设备 {self.args.get('device', 'cpu')})", flush=True)

    # ------------------------------------------------------------ 观测/推理

    def _state16(self) -> np.ndarray:
        d = self.data
        return np.concatenate([
            d.qpos[self.arm_qadrs["left"]], [self._grip_state("left")],
            d.qpos[self.arm_qadrs["right"]], [self._grip_state("right")],
        ]).astype(np.float32)

    def _read_raw(self, n: int, timeout: float) -> bytes:
        buf = bytearray()
        fd = self.worker.stdout.fileno()
        while len(buf) < n:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise TimeoutError(f"推理 worker {timeout:.0f}s 无响应")
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                raise RuntimeError("推理 worker 已退出，详见 pi05_worker.log")
            buf.extend(chunk)
        return bytes(buf)

    def _infer(self) -> np.ndarray:
        images = [self._render(k) for k in self.cam_specs]
        state = self._state16()
        h, w = images[0].shape[:2]
        head = struct.pack("<IIII", len(state), h, w, len(images))
        payload = head + state.astype("<f4").tobytes() + b"".join(
            np.ascontiguousarray(img).tobytes() for img in images)
        self.worker.stdin.write(b"A" + struct.pack("<I", len(payload)) + payload)
        self.worker.stdin.flush()
        (nbytes,) = struct.unpack("<I", self._read_raw(4, INFER_TIMEOUT))
        data = self._read_raw(nbytes, INFER_TIMEOUT)
        return np.frombuffer(data, dtype="<f4").reshape(
            -1, len(state)).astype(np.float64)

    def _apply_action(self, a: np.ndarray) -> None:
        for side, off in (("left", 0), ("right", 8)):
            q = np.asarray(a[off:off + 7], dtype=float)
            self.q_goal[side] = np.clip(q, self.arm_lo[side], self.arm_hi[side])
            self.grip_goal[side] = float(
                np.clip(a[off + 7], 0.0, 1.0)) * self.grip_open

    # ------------------------------------------------------------ 主循环

    def _cloth_center(self) -> np.ndarray | None:
        if self.flex_id < 0 or self.vert_num == 0:
            return None
        v = self.data.flexvert_xpos[
            self.vert_adr:self.vert_adr + self.vert_num]
        return v.mean(axis=0) if len(v) else None

    def _finish(self, ok: bool, detail: str) -> None:
        self.done = True
        self.success = ok
        self.detail = detail
        self.status.update({"recording": False,
                            "msg": f"{'成功' if ok else '结束'}：{detail}"})
        print(f"[policy_eval] {'✔' if ok else '✖'} {detail}"
              f"（推理 {self.n_infer} 次）", flush=True)

    def _check_finish(self) -> bool:
        sim_t = self.tick * self.tick_dt
        center = self._cloth_center()
        inside = (center is not None
                  and np.all(center >= self.box_lo)
                  and np.all(center <= self.box_hi))
        # 必须已经松手：否则布料被夹爪夹着悬在盒口上方，质心照样落在判定区
        released = all(self._grip_state(s) < self.grip_release
                       for s in ("left", "right"))
        if inside and released:
            if self._stable_since is None:
                self._stable_since = sim_t
            elif sim_t - self._stable_since >= self.hold_seconds:
                self._finish(True, f"布料已落入盒内并松手，"
                                   f"质心 {np.round(center, 3)}")
                return True
        else:
            self._stable_since = None
        if sim_t >= self.max_sim_seconds:
            pos = "未知" if center is None else np.round(center, 3)
            grips = [round(self._grip_state(s), 2) for s in ("left", "right")]
            self._finish(False, f"到达 {self.max_sim_seconds:.0f}s 上限，"
                                f"布料质心 {pos}，夹爪开度 {grips}")
            return True
        return False

    def advance(self) -> bool:
        if self._closed or self.done:
            return False
        if self._check_finish():
            return False
        if not self.queue:
            t0 = time.monotonic()
            self.queue = list(self._infer())
            self.n_infer += 1
            print(f"[policy_eval] 推理 #{self.n_infer} "
                  f"{time.monotonic() - t0:.2f}s  t={self.tick * self.tick_dt:.2f}s  "
                  f"首步={np.round(self.queue[0][:8], 3)}", flush=True)
        self._apply_action(self.queue.pop(0))
        for _ in range(self.ctrl_period):
            self._substep()
        self.tick += 1
        if time.monotonic() - self._last_status > 0.5:
            self._last_status = time.monotonic()
            self.status["msg"] = (f"π0.5 推理 {self.n_infer} 次  "
                                  f"t={self.tick * self.tick_dt:.1f}s")
        return True

    # ------------------------------------------------------------ 收尾

    def close(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None and worker.poll() is None:
            try:
                worker.stdin.write(b"Q")
                worker.stdin.flush()
                worker.wait(timeout=5)
            except Exception:                      # noqa: BLE001
                worker.kill()
        log = getattr(self, "_worker_log", None)
        if log is not None:
            try:
                log.close()
            except OSError:
                pass
        super().close()


def create_controller(simulation):
    return G1ClothPolicyEval(simulation)
