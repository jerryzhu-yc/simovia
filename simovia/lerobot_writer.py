"""LeRobot v2.1 数据集写入器：直接产出 SmolVLA 训练格式的数据集。

目录结构（与 huggingface lerobot 数据集一致，from_pretrained 可直接读）：

    <root>/
      meta/
        info.json            # fps/features/统计（每集收尾时全量重写）
        episodes.jsonl       # 每行 {"episode_index", "tasks", "length"}
        tasks.jsonl          # 每行 {"task_index", "task"}
      data/chunk-000/
        episode_000000.parquet   # state/action/标量列 + 视频 struct 列
      videos/chunk-000/
        observation.images.<cam>/episode_000000.mp4

视频编码：帧流式 pipe 给 ffmpeg（libx264），不落原始帧、不占内存。
跨会话续写：启动时扫描已有 meta 恢复 episode/task 计数；同时对齐磁盘现状——
用户回看后手动删掉的 episode（任一相机视频或 parquet 缺失）会被整集剔除，
剩余 episode 按采集顺序从 0 连续重编号。
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:            # pragma: no cover
    pa = None

FORMAT_VERSION = "v2.1"
CHUNKS_SIZE = 1000
_VID_KEYS_IN_INFO = ("video.fps", "video.channels", "video.width",
                     "video.height", "video.codec", "video.pix_fmt",
                     "video.is_depth_map", "has_audio")


class LerobotWriter:
    """增量写 LeRobot v2.1 数据集（一次一个 episode）。

    用法：
        w = LerobotWriter("~/桌面/so101_smolvla_dataset",
                          fps=15.625, dim=6,
                          cameras={"observation.images.wrist_cam": (480, 640)},
                          task="把红色方块放进红盒")
        w.start_episode()
        for ...:
            w.add_frame(state=np.float32(6), action=np.float32(6),
                        images={"observation.images.wrist_cam": rgb_hwc})
        w.end_episode()          # flush mp4 + parquet + meta
    """

    def __init__(self, root, *, fps: float, dim: int,
                 cameras: dict[str, tuple[int, int]],
                 task: str, robot_type: str = "so101",
                 state_names: list[str] | None = None,
                 action_names: list[str] | None = None):
        if pa is None:
            raise RuntimeError(
                "写 LeRobot 数据集需要 pyarrow：pip install pyarrow")
        if not cameras:
            raise ValueError("SmolVLA 训练至少需要一个相机（cameras 为空）")
        self.root = pathlib.Path(root).expanduser()
        # 上次退出时已升级成 v3.0：换回 v2.1 备份，否则续写会写坏目录
        from .lerobot_v30 import restore_v21_if_needed
        restore_v21_if_needed(self.root)
        self.fps = float(fps)
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError(f"fps 必须为正，收到 {fps}")
        self.dim = int(dim)
        self.cameras = {str(k): (int(h), int(wv))
                        for k, (h, wv) in cameras.items()}
        self.robot_type = str(robot_type)
        self.state_names = state_names
        self.action_names = action_names or state_names

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in self.cameras:
            (self.root / "videos" / "chunk-000" / cam)\
                .mkdir(parents=True, exist_ok=True)

        # ---- 恢复已有计数（跨会话续写同一数据集）
        self._tasks: list[str] = []
        self._episodes: list[dict] = []
        self._next_index = 0
        tpath = self.root / "meta" / "tasks.jsonl"
        epath = self.root / "meta" / "episodes.jsonl"
        if tpath.exists():
            for line in tpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._tasks.append(json.loads(line)["task"])
        if epath.exists():
            for line in epath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ep = json.loads(line)
                    self._episodes.append(ep)
                    self._next_index += int(ep["length"])
        if task not in self._tasks:
            self._tasks.append(task)
        self.task = str(task)

        # ---- 当前 episode 状态
        self._ep: int | None = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._ff: dict[str, subprocess.Popen] = {}
        self._frame_i = 0

        # 对齐磁盘现状：用户回看后手删的视频/数据在此清掉并重编号
        self.sync()

    # ------------------------------------------------------------ sync

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def _episode_files_ok(self, idx: int) -> bool:
        """该 episode 的 parquet 与所有相机视频是否都还在且非空。"""
        pq_path = (self.root / "data" / "chunk-000" /
                   f"episode_{idx:06d}.parquet")
        if not pq_path.exists() or pq_path.stat().st_size == 0:
            return False
        for cam in self.cameras:
            p = (self.root / "videos" / "chunk-000" / cam /
                 f"episode_{idx:06d}.mp4")
            if not p.exists() or p.stat().st_size == 0:
                return False
        return True

    def sync(self) -> dict:
        """对齐磁盘现状：删除被手动删掉文件的 episode，剩余按采集顺序从 0 重编号。

        回看视频时删 mp4（可能只删某一路相机）后，用本方法把 data/parquet、
        其余相机视频、meta 三处一起清干净，避免 meta 与实际文件错位。任一相机
        视频或 parquet 缺失即整集丢弃；编号按原 episode_index（采集时间）排序
        后连续重排，parquet 里的 episode_index / index / video_path 同步改写。

        返回 {"removed": [旧编号], "renumbered": {旧: 新}}；无变化时不做任何
        写盘操作。
        """
        if not self._episodes:
            return {"removed": [], "renumbered": {}}
        kept: list[dict] = []
        removed: list[int] = []
        for ep in self._episodes:
            old = int(ep["episode_index"])
            if self._episode_files_ok(old):
                kept.append(ep)
            else:
                removed.append(old)
        kept.sort(key=lambda e: int(e["episode_index"]))   # 采集顺序
        renumbered = {int(e["episode_index"]): i
                      for i, e in enumerate(kept)
                      if int(e["episode_index"]) != i}
        if not removed and not renumbered:
            return {"removed": [], "renumbered": {}}

        # 1) 先清掉被丢弃 episode 的残留（parquet + 还在的那几路视频）
        for old in removed:
            for cam in self.cameras:
                (self.root / "videos" / "chunk-000" / cam /
                 f"episode_{old:06d}.mp4").unlink(missing_ok=True)
            (self.root / "data" / "chunk-000" /
             f"episode_{old:06d}.parquet").unlink(missing_ok=True)

        # 2) 剩余 episode 按顺序重排（目标编号一定 <= 原编号，升序处理不会覆盖）
        offset = 0
        for new, ep in enumerate(kept):
            old = int(ep["episode_index"])
            length = int(ep["length"])
            if old != new:
                self._renumber_episode(old, new, offset)
            offset += length

        self._episodes = [{"episode_index": i, "tasks": e["tasks"],
                           "length": int(e["length"])}
                          for i, e in enumerate(kept)]
        self._next_index = offset
        self._write_meta()
        print(f"[data] 数据集同步：剔除 {len(removed)} 集 {removed}，"
              f"剩余 {len(kept)} 集已从 0 连续编号", flush=True)
        return {"removed": removed, "renumbered": renumbered}

    def _renumber_episode(self, old: int, new: int, offset: int) -> None:
        """把一集从旧编号搬到新编号：parquet 改写三列后落新文件，视频改名。"""
        pq_old = (self.root / "data" / "chunk-000" /
                  f"episode_{old:06d}.parquet")
        t = pq.read_table(pq_old)
        n = t.num_rows
        cols = {name: t.column(name) for name in t.column_names}
        cols["episode_index"] = pa.array(np.full(n, new, dtype=np.int64))
        cols["index"] = pa.array(np.arange(offset, offset + n, dtype=np.int64))
        for cam in self.cameras:
            if cam not in cols:
                continue
            chunk = cols[cam].combine_chunks()
            rel = f"videos/chunk-000/{cam}/episode_{new:06d}.mp4"
            cols[cam] = pa.StructArray.from_arrays(
                [pa.array([rel] * n, type=pa.string()),
                 chunk.field("from_timestamp"),
                 chunk.field("to_timestamp")],
                names=["video_path", "from_timestamp", "to_timestamp"])
        pq.write_table(pa.table(cols),
                       self.root / "data" / "chunk-000" /
                       f"episode_{new:06d}.parquet")
        pq_old.unlink(missing_ok=True)
        for cam in self.cameras:
            (self.root / "videos" / "chunk-000" / cam /
             f"episode_{old:06d}.mp4").rename(
                self.root / "videos" / "chunk-000" / cam /
                f"episode_{new:06d}.mp4")

    # ------------------------------------------------------------ episode

    @property
    def episode_index(self) -> int:
        return -1 if self._ep is None else self._ep

    def start_episode(self) -> int:
        """开启新 episode（返回编号）。上一集未收尾会先强制收尾。"""
        if self._ep is not None:
            self.end_episode()
        self.sync()                 # 录新一集前先对齐磁盘（可能刚删过旧集）
        self._ep = len(self._episodes)
        self._states, self._actions = [], []
        self._frame_i = 0
        self._ff = {}
        for cam, (h, wv) in self.cameras.items():
            out = (self.root / "videos" / "chunk-000" / cam /
                   f"episode_{self._ep:06d}.mp4")
            self._ff[cam] = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{wv}x{h}", "-r", f"{self.fps:.6f}", "-i", "-",
                 "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "23", "-pix_fmt", "yuv420p", str(out)],
                stdin=subprocess.PIPE)
        return self._ep

    def add_frame(self, state, action, images: dict[str, np.ndarray]):
        """追加一帧。state/action 一维 dim 维；images 键与 cameras 一致。"""
        if self._ep is None:
            raise RuntimeError("先 start_episode() 再 add_frame()")
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if s.size != self.dim or a.size != self.dim:
            raise ValueError(f"state/action 需要 {self.dim} 维，"
                             f"收到 {s.size}/{a.size}")
        for cam, (h, wv) in self.cameras.items():
            img = images.get(cam)
            if img is None:
                raise ValueError(f"缺相机帧: {cam}")
            if img.shape[:2] != (h, wv):
                raise ValueError(f"{cam} 分辨率 {img.shape[:2]} != ({h},{wv})")
            data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
            try:
                self._ff[cam].stdin.write(data)
            except BrokenPipeError as e:
                raise RuntimeError(
                    f"ffmpeg 编码 {cam} 中断: {e}（磁盘满/无 ffmpeg?）") from e
        self._states.append(s)
        self._actions.append(a)
        self._frame_i += 1

    def end_episode(self) -> dict:
        """收尾：等 mp4 编码完、写 parquet + meta，返回 episode 摘要。"""
        if self._ep is None:
            raise RuntimeError("没有进行中的 episode")
        ep, n = self._ep, self._frame_i
        for cam, ff in self._ff.items():
            ff.stdin.close()
            rc = ff.wait(timeout=120)
            if rc != 0:
                raise RuntimeError(f"ffmpeg 编码 {cam} 退出码 {rc}")
        self._ff = {}
        if n == 0:
            # 空集：删掉 0 字节视频，不进 meta
            for cam in self.cameras:
                p = (self.root / "videos" / "chunk-000" / cam /
                     f"episode_{ep:06d}.mp4")
                if p.exists():
                    p.unlink()
            self._ep = None
            return {"episode_index": ep, "length": 0, "saved": False}

        dt = 1.0 / self.fps
        task_idx = self._tasks.index(self.task)
        cols = {
            "index": pa.array(
                np.arange(self._next_index, self._next_index + n,
                          dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "timestamp": pa.array(
                np.arange(n, dtype=np.float32) * np.float32(dt)),
            "task_index": pa.array(np.full(n, task_idx, dtype=np.int64)),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                np.stack(self._states).reshape(-1).astype(np.float32),
                self.dim),
            "action": pa.FixedSizeListArray.from_arrays(
                np.stack(self._actions).reshape(-1).astype(np.float32),
                self.dim),
        }
        vid_struct = pa.struct([("video_path", pa.string()),
                                ("from_timestamp", pa.float64()),
                                ("to_timestamp", pa.float64())])
        for cam in self.cameras:
            rel = f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4"
            cols[cam] = pa.array(
                [{"video_path": rel,
                  "from_timestamp": i * dt,
                  "to_timestamp": (i + 1) * dt} for i in range(n)],
                type=vid_struct)
        pq.write_table(
            pa.table(cols),
            self.root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")

        self._episodes.append({"episode_index": ep,
                               "tasks": [self._tasks[task_idx]],
                               "length": n})
        self._next_index += n
        self._write_meta()
        self._ep = None
        return {"episode_index": ep, "length": n, "saved": True,
                "task": self._tasks[task_idx]}

    def abort_episode(self) -> None:
        """丢弃当前 episode（录制误操作时用）。"""
        if self._ep is None:
            return
        for ff in self._ff.values():
            try:
                ff.stdin.close()
                ff.kill()
                ff.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._ff = {}
        for cam in self.cameras:
            p = (self.root / "videos" / "chunk-000" / cam /
                 f"episode_{self._ep:06d}.mp4")
            if p.exists():
                p.unlink()
        pq_path = (self.root / "data" / "chunk-000" /
                   f"episode_{self._ep:06d}.parquet")
        if pq_path.exists():
            pq_path.unlink()
        self._states, self._actions = [], []
        self._frame_i = 0
        self._ep = None

    # ------------------------------------------------------------ v3.0

    def finalize_v30(self) -> dict:
        """把整份数据集就地升级成 lerobot >= 0.6 可直接训练的 v3.0。

        采集结束后调用一次即可：原 v2.1 目录改名为 `<root>_v21` 保留，
        下次 new LerobotWriter 写同一路径时会自动换回来继续续采。
        """
        if self._ep is not None:
            self.end_episode()
        from .lerobot_v30 import convert_v21_to_v30
        return convert_v21_to_v30(self.root)

    # ------------------------------------------------------------ meta

    def _write_meta(self) -> None:
        total_frames = sum(e["length"] for e in self._episodes)
        features: dict = {
            "observation.state": {"dtype": "float32",
                                  "shape": [self.dim],
                                  "names": self.state_names},
            "action": {"dtype": "float32", "shape": [self.dim],
                       "names": self.action_names},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        }
        h, wv = next(iter(self.cameras.values()))
        # names 不能为 null：lerobot 的 dataset_to_policy_features 要读
        # names[2] 判断通道维位置，null 会在建 policy 时直接 TypeError
        for cam in self.cameras:
            features[cam] = {"dtype": "video", "shape": [h, wv, 3],
                             "names": ["height", "width", "channels"]}
        info = {
            "codebase_version": FORMAT_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": len(self._episodes) * len(self.cameras),
            "total_chunks": 1,          # 官方 v2.1 必填（转换脚本会 del 掉它）
            "chunks_size": CHUNKS_SIZE,
            "data_path": "data/chunk-{episode_chunk:03d}/"
                         "episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/"
                          "{video_key}/episode_{episode_index:06d}.mp4",
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "features": features,
        }
        (self.root / "meta" / "info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")
        with open(self.root / "meta" / "tasks.jsonl", "w",
                  encoding="utf-8") as f:
            for i, t in enumerate(self._tasks):
                f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        with open(self.root / "meta" / "episodes.jsonl", "w",
                  encoding="utf-8") as f:
            for e in self._episodes:
                f.write(json.dumps(e) + "\n")


def verify_dataset(root) -> dict:
    """读回校验数据集结构：parquet 行数/列、视频可解码、meta 计数一致。"""
    root = pathlib.Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = [json.loads(l) for l in
                (root / "meta" / "episodes.jsonl").read_text(
                    encoding="utf-8").splitlines() if l.strip()]
    problems: list[str] = []
    total = 0
    for e in episodes:
        pq_path = (root / "data" / "chunk-000" /
                   f"episode_{e['episode_index']:06d}.parquet")
        t = pq.read_table(pq_path)
        if t.num_rows != e["length"]:
            problems.append(f"episode {e['episode_index']}: parquet 行数 "
                            f"{t.num_rows} != meta 长度 {e['length']}")
        for feat in info["features"]:
            if feat not in t.column_names:
                problems.append(f"episode {e['episode_index']}: 缺列 {feat}")
        total += t.num_rows
        for cam in [k for k, v in info["features"].items()
                    if v["dtype"] == "video"]:
            rel = t.column(cam)[0].as_py()["video_path"]
            mp4 = root / rel
            if not mp4.exists():
                problems.append(f"episode {e['episode_index']}: 缺视频 {rel}")
                continue
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
                 str(mp4)], capture_output=True, text=True)
            frames = probe.stdout.strip()
            if frames.isdigit() and int(frames) != e["length"]:
                problems.append(f"episode {e['episode_index']}: {cam} 视频 "
                                f"{frames} 帧 != {e['length']}")
    if total != info["total_frames"]:
        problems.append(f"总帧数 {total} != info.json {info['total_frames']}")
    return {"ok": not problems, "episodes": len(episodes),
            "frames": total, "problems": problems,
            "columns": None if not episodes else
            pq.read_table(root / "data" / "chunk-000" /
                          f"episode_{episodes[0]['episode_index']:06d}.parquet"
                          ).column_names}

