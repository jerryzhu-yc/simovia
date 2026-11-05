"""离屏渲染：当前物理状态 -> RGB 数组 / PNG bytes（MCP screenshot 实现层）。

纯 MuJoCo + stdlib，不 import PySide6——mcp_server 进程和测试也要能截图。
PNG 用 zlib 手写（签名/IHDR/IDAT/IEND），避免再引 Pillow。
"""
from __future__ import annotations

import struct
import zlib

import mujoco
import numpy as np

# 自由机位默认参数：与 GUI RenderModel 初始机位一致，截图与 GUI 所见对齐
_FREE_LOOKAT = (0.4, 0.15, 0.25)
_FREE_DISTANCE = 1.8
_FREE_AZIMUTH = 135.0
_FREE_ELEVATION = -30.0


def _sync_camlights(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """补齐相机与灯的派生位姿（mj_kinematics 只算 body/geom/site）。

    cam_xpos/cam_xmat 与 light_xpos/light_xdir 都由 mj_camlight 计算；刚
    load / reset / load_state / teleport 后没有步进就截图时，这些量还是全零，
    相机停在原点、灯全在地面里——画面机位错误且整片发暗。mj_camlight 只写
    这些派生量，不碰 qvel/warmstart，因此不影响物理状态。
    """
    mujoco.mj_camlight(model, data)


def probe_offscreen_gl(timeout: float = 60.0) -> tuple[bool, str]:
    """在独立子进程里试建离屏渲染上下文，返回 (可用, 说明)。

    MuJoCo 的 GL 后端在无显示/无驱动的环境里可能直接 abort（EGL 下是
    C 层 abort，Python 的 try/except 拦不住），会把探测方整个解释器带走。
    所以探测必须隔进程做：子进程崩了只表现为非零退出码。
    """
    import subprocess
    import sys

    code = (
        "import mujoco\n"
        "m = mujoco.MjModel.from_xml_string('<mujoco/>')\n"
        "r = mujoco.Renderer(m, 32, 32)\n"
        "r.update_scene(mujoco.MjData(m))\n"
        "r.render()\n"
        "r.close()\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=timeout)
    except Exception as e:                     # noqa: BLE001 探测不抛异常
        return False, f"{type(e).__name__}: {e}"
    if proc.returncode == 0:
        return True, "可用"
    err = (proc.stderr or b"").decode("utf-8", "replace").strip()
    last = err.splitlines()[-1] if err else f"退出码 {proc.returncode}"
    return False, last


def render_offscreen(model: mujoco.MjModel, data: mujoco.MjData,
                     width: int, height: int,
                     camera: str | None = None,
                     renderer: mujoco.Renderer | None = None,
                     free_cam: "mujoco.MjvCamera | None" = None,
                     depth: bool = False,
                     near: float | None = None,
                     far: float | None = None) -> np.ndarray:
    """渲染当前状态，返回 RGB (h,w,3 uint8) 或深度 (h,w float32 米)。

    - 先 mj_kinematics：刚 load / load_state 后 xpos 等派生量是过期/零值
    - camera=None：自由机位（GUI 同款默认视角）；str：模型里的相机名
      （manifest 短名需先解析成带机器人前缀的全名）
    - renderer：传入则复用（session 按 (w,h) 缓存），不传入则自建自毁
    - free_cam：自由机位的替代机位（遥操作时跟随末端 TCP 用）
    - depth=True：返回深度图（MuJoCo 深度渲染，单位米）；near/far 设置
      裁剪面（None 保持渲染器默认）
    """
    if near is not None or far is not None:
        _check_clip(near, far)
    mujoco.mj_kinematics(model, data)      # 只重算运动学，不碰 warmstart
    _sync_camlights(model, data)           # 相机/灯位姿不在 kinematics 里
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, height, width)
    if depth:
        renderer.enable_depth_rendering()
    # near/far 必须写 model.vis.map：mjv 用它同时推导光栅化 frustum 与深度
    # 反投影系数。只改 scene.camera.frustum_* 会让两者不一致，深度值按错误
    # 的 near/far 换算（0.8 m 的平面会被读成 0.11 m）。
    old_znear, old_zfar = model.vis.map.znear, model.vis.map.zfar
    scale = float(model.stat.extent) or 1.0
    if near is not None:
        model.vis.map.znear = float(near) / scale
    if far is not None:
        model.vis.map.zfar = float(far) / scale
    try:
        if camera is None:
            if free_cam is not None:
                renderer.update_scene(data, camera=free_cam)
            else:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = _FREE_LOOKAT
                cam.distance = _FREE_DISTANCE
                cam.azimuth = _FREE_AZIMUTH
                cam.elevation = _FREE_ELEVATION
                renderer.update_scene(data, camera=cam)
        else:
            renderer.update_scene(data, camera=camera)
        return renderer.render()
    finally:
        model.vis.map.znear = old_znear
        model.vis.map.zfar = old_zfar
        if depth:
            renderer.disable_depth_rendering()
        if own:
            renderer.close()


def _check_clip(near: float | None, far: float | None) -> None:
    """near/far 合法性：near>0 且 far>near（否则光栅化与深度换算都会失真）。"""
    n = float(near) if near is not None else None
    f = float(far) if far is not None else None
    if n is not None and (not np.isfinite(n) or n <= 0):
        raise ValueError(f"near 必须是正数，实际: {near}")
    if f is not None and (not np.isfinite(f) or f <= 0):
        raise ValueError(f"far 必须是正数，实际: {far}")
    if n is not None and f is not None and f <= n:
        raise ValueError(f"far 必须大于 near，实际: near={near}, far={far}")


def depth_to_rgb(depth: np.ndarray, near: float = 0.01,
                 far: float = 50.0) -> np.ndarray:
    """深度图 -> 近红远蓝伪彩（无效/超远像素涂黑），便于实时观察。"""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0) & (d < far)
    span = max(float(far) - float(near), 1e-6)
    t = np.clip((d - float(near)) / span, 0.0, 1.0)
    # 0=近(红) -> 1=远(蓝)
    r = np.clip(1.5 - 2.0 * t, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(2.0 * t - 0.5, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def encode_png(rgb: np.ndarray) -> bytes:
    """(h, w, 3) uint8 -> PNG bytes。8bit truecolor（colortype 2），无交织。

    PNG 结构：签名 + 各 chunk（len + type + data + crc32）。
    IDAT 前每行加 filter byte 0（None 滤波），再 zlib 压缩。
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"需要 (h, w, 3) RGB 数组，实际: {rgb.shape}")
    h, w = rgb.shape[:2]

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body)))

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    # IHDR：宽、高、位深 8、colortype 2(RGB)、压缩 0、滤波 0、交织 0
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw, 6)) +
            chunk(b"IEND", b""))
