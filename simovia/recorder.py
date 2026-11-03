"""Episode 录制：把一次仿真的观测/动作/传感器序列落盘成可训练的数据。

产物（一个 episode 一个目录）：

    episodes/episode_000001/
      meta.json     # 世界/契约/任务/统计等元信息
      data.npz      # time/tick + 每个机器人的观测与动作 + 物体位姿
      frames/       # 可选：相机帧 PNG（--record-images）

设计要点：
- 复用 kernel/task.py 的观测组装语义（关节顺序、传感器拼接与内核一致），
  录制数据与控制器看到的观测不会错位。
- 契约（contract.py）随 meta.json 一起落盘：训练端拿到的形状、关节名、
  动作模式、控制频率都能独立核对，不依赖 MuJoCo 实例。
- 默认 npz 纯 stdlib + numpy，不新增依赖；后续可在同一数据上再做
  LeRobot/HDF5 导出。
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib

import mujoco
import numpy as np

from simovia.contract import (RobotContract, contracts_for_world,
                              root_joint_addresses)
from simovia.kernel.task import robot_obs, sensors_obs
from simovia.render import encode_png, render_offscreen
from simovia.world.schema import WorldSpec

FORMAT = "simovia-episode-v1"


class EpisodeRecorder:
    """按控制 tick 采样，finish() 时统一写盘。

    典型用法（CLI 已封装）：

        rec = EpisodeRecorder(spec, sim.model, sim.index, "episodes/ep1")
        stats = sim.run(on_tick=lambda: rec.add(sim.tick, sim.data))
        rec.finish(stats=stats, extra={"world_path": "worlds/pick_cube.yaml"})

    images=True 时每个 image_stride 个采样帧渲染一张 PNG（离屏渲染，
    无显示器环境请确认 MUJOCO_GL 可用，Linux 默认 egl）。
    """

    def __init__(self, spec: WorldSpec, model: mujoco.MjModel, index, out_dir,
                 *, images: bool = False, camera: str | None = None,
                 width: int = 640, height: int = 480,
                 image_stride: int = 1):
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError(
                f"录制分辨率必须为正数，收到 {width}x{height}")
        if int(image_stride) < 1:
            raise ValueError(f"image_stride 必须 >= 1，收到 {image_stride}")
        self.spec = spec
        self.model = model
        self.index = index
        self.out_dir = pathlib.Path(out_dir)
        self.contracts: dict[str, RobotContract] = contracts_for_world(
            spec, model, index)
        self.images = bool(images)
        # 支持短名（YAML 传感器名）和全名（<robot>/<name>），统一解析成
        # MuJoCo 相机名，渲染时直接用全名；同时记下挂载信息供 meta 核对。
        self.camera, self.camera_short, self.camera_mount = _resolve_camera(
            spec, model, index, camera)
        self.width = int(width)
        self.height = int(height)
        self.image_stride = int(image_stride)

        self._robot_ids = {name: index.robots[name]
                           for name in self.contracts}
        self._root_adr = {name: root_joint_addresses(model, index.robots[name])
                          for name in self.contracts}
        self._rows: dict[str, dict[str, list[list[float]]]] = {
            name: {f.name: [] for f in c.obs_fields}
            for name, c in self.contracts.items()}
        self._actions: dict[str, list[list[float]]] = {
            name: [] for name in self.contracts}

        self._objects = list(index.objects.items())
        self._object_pos: list[list[float]] = []
        self._object_quat: list[list[float]] = []

        self._ticks: list[int] = []
        self._times: list[float] = []
        self._frame_paths: list[str] = []
        self._frame_ticks: list[int] = []
        self._sample = 0
        self._renderer: mujoco.Renderer | None = None

    # ------------------------------------------------------------ 采样

    def add(self, tick: int, data: mujoco.MjData) -> None:
        """记录当前 tick 的状态（在每控制 tick 推进完成后调用）。"""
        self._ticks.append(int(tick))
        self._times.append(float(data.time))

        for name, ridx in self._robot_ids.items():
            contract = self.contracts[name]
            qpos, qvel = robot_obs(ridx, self.model, data)
            row = {"joint_pos": qpos, "joint_vel": qvel}
            if contract.floating_base:
                qadr, vadr = self._root_adr[name]
                row["root_pos"] = list(map(float, data.qpos[qadr:qadr + 3]))
                row["root_quat"] = list(map(float, data.qpos[qadr + 3:qadr + 7]))
                row["root_lin_vel"] = list(map(float, data.qvel[vadr:vadr + 3]))
                row["root_ang_vel"] = list(map(float, data.qvel[vadr + 3:vadr + 6]))
            for sname, values in sensors_obs(ridx, data).items():
                row[f"sensor:{sname}"] = (
                    list(values) if isinstance(values, list) else [values])
            for field_name, values in row.items():
                self._rows[name][field_name].append(list(values))
            self._actions[name].append(list(map(
                float, data.ctrl[list(ridx.actuators.values())])))

        if self._objects:
            self._object_pos.append(
                [v for _, bid in self._objects
                 for v in map(float, data.xpos[bid])])
            self._object_quat.append(
                [v for _, bid in self._objects
                 for v in map(float, data.xquat[bid])])
        else:
            self._object_pos.append([])
            self._object_quat.append([])

        if self.images and self._sample % self.image_stride == 0:
            self._capture_frame(tick, data)
        self._sample += 1

    # ------------------------------------------------------------ 写盘

    def finish(self, stats: dict | None = None,
               extra: dict | None = None) -> pathlib.Path:
        """写 data.npz + meta.json，返回 episode 目录。"""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "tick": np.asarray(self._ticks, dtype=np.int64),
            "time": np.asarray(self._times, dtype=np.float64),
            "object_pos": _stack(self._object_pos),
            "object_quat": _stack(self._object_quat),
        }
        for name, rows in self._rows.items():
            for field_name, values in rows.items():
                arrays[f"{name}.{field_name}"] = _stack(values)
            arrays[f"{name}.action"] = _stack(self._actions[name])
        np.savez_compressed(self.out_dir / "data.npz", **arrays)

        meta = {
            "format": FORMAT,
            "world": self.spec.name,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "frames": len(self._ticks),
            "timestep": float(self.spec.timestep),
            "ctrl_period": int(self.spec.ctrl_period),
            "control_dt": float(self.spec.timestep * self.spec.ctrl_period),
            "gravity": list(map(float, self.spec.gravity)),
            "integrator": self.spec.integrator,
            "impratio": float(self.spec.impratio),
            "objects": [name for name, _ in self._objects],
            "contracts": {name: c.to_dict()
                          for name, c in self.contracts.items()},
            "contract_digests": {name: c.digest()
                                 for name, c in self.contracts.items()},
            "stats": _jsonable(stats or {}),
            "extra": _jsonable(extra or {}),
            "files": {"data": "data.npz",
                      "frames": "frames" if self._frame_paths else None},
            "image_frames": self._frame_ticks,
            "images": {
                "enabled": self.images,
                "camera": self.camera,
                "camera_short": self.camera_short,
                "camera_mount": self.camera_mount,
                "width": self.width,
                "height": self.height,
                "stride": self.image_stride,
                "count": len(self._frame_paths),
            },
        }
        (self.out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.close()
        return self.out_dir

    @property
    def frame_count(self) -> int:
        return len(self._frame_paths)

    @property
    def sample_count(self) -> int:
        return len(self._ticks)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ 相机

    def _capture_frame(self, tick: int, data: mujoco.MjData) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self.height, self.width)
        rgb = render_offscreen(self.model, data, self.width, self.height,
                               camera=self.camera, renderer=self._renderer)
        frame_dir = self.out_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        path = frame_dir / f"frame_{len(self._frame_paths):06d}.png"
        path.write_bytes(encode_png(rgb))
        self._frame_paths.append(str(path.relative_to(self.out_dir)))
        self._frame_ticks.append(int(tick))


# ---------------------------------------------------------------- 工具

def _resolve_camera(spec: WorldSpec, model: mujoco.MjModel, index,
                    camera: str | None):
    """把用户写的相机名解析成 MuJoCo 全名，并返回挂载信息。

    支持三种写法：
    - 短名：YAML 里 robots[].sensors[].name（如 wrist_cam）
    - 全名：<robot>/<name>（如 arm/wrist_cam）
    - 模型自带相机名（如 kinova_gen3 的 wrist）
    找不到或有歧义时直接报错，列出可用相机，避免录完才发现拍的是自由机位。
    """
    if camera is None:
        return None, None, None

    matches: dict[str, set[str]] = {}
    for rname, ridx in index.robots.items():
        for short in ridx.cameras:
            full = f"{rname}/{short}"
            matches.setdefault(short, set()).add(full)
            matches.setdefault(full, {full})
    # 模型自带的相机（不一定写进 YAML sensors）
    for cid in range(model.ncam):
        full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if full:
            matches.setdefault(full, {full})

    if camera not in matches:
        available = sorted({full for names in matches.values() for full in names})
        hint = "、".join(available) if available else "无（请在 robots[].sensors 里声明 camera）"
        raise ValueError(f"找不到录制相机 '{camera}'，可用相机: {hint}")
    full_names = matches[camera]
    if len(full_names) > 1:
        raise ValueError(
            f"相机短名 '{camera}' 在多台机器人上重名，请改用全名: "
            f"{'、'.join(sorted(full_names))}")
    full = next(iter(full_names))

    mount = None
    for rd in spec.robots:
        for s in rd.sensors:
            if s.type == "camera" and f"{rd.name}/{s.name}" == full:
                mount = {
                    "robot": rd.name,
                    "body": s.body,
                    "pos": list(s.pos),
                    "quat": list(s.quat) if s.quat is not None else None,
                }
    return full, camera, mount


def _stack(rows: list[list[float]]) -> np.ndarray:
    """list[list[float]] -> (T, dim) float64；空数据给 (0, 0)。"""
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _jsonable(value):
    """递归把 numpy / pathlib / tuple 转成 json 可写类型。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


__all__ = ["EpisodeRecorder", "FORMAT"]
