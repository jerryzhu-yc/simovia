"""遥操作验收：IK 跟随 + 无控制器推进 + 相机流 + 采集录制。

守护点：
- 速度指令经 IK 真的推动 TCP（基座坐标系方向正确），位置伺服无控制器进程
- teleop_start 不启动控制器进程，stop 后 episode 落盘（data.npz + meta.json）
- camera_frame 返回 image content（base64 PNG），尺寸与请求一致
- 遥操作结束后再 run 会重建带控制器的仿真（_manual_sim 复位）
"""
import base64
import json
import pathlib
import time

import numpy as np
import pytest

from simovia.errors import WorldError
from simovia.mcp_server import SimoviaSession   # 导入即 setdefault MUJOCO_GL=egl
from simovia.kernel.sim import Simulation
from simovia.teleop import ArmTeleop
from simovia.world.schema import load_world

ROOT = pathlib.Path(__file__).resolve().parents[1]
PICK = ROOT / "worlds" / "pick_cube.yaml"


def _offscreen_ok() -> bool:
    # 隔子进程探测：无 GL 时 MuJoCo 会 abort，进程内 try/except 拦不住
    from simovia.render import probe_offscreen_gl
    return probe_offscreen_gl()[0]


GL = _offscreen_ok()


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def test_arm_teleop_moves_tcp_forward():
    """vx=1（基座 +x）：无控制器进程逐 tick 写 ctrl，TCP 真的前进。"""
    spec = load_world(str(PICK))
    sim = Simulation(spec, autostart_controllers=False)
    try:
        teleop = ArmTeleop(sim.model, sim.data, sim.index, spec, "arm")
        assert teleop.site_name.endswith("gripper_tcp")
        start = np.array(sim.data.site_xpos[teleop.site_id], dtype=float)
        teleop.set_command({"vx": 1.0})
        sim.run(max_ticks=10, ctrl_fn=teleop.step)
        end = np.array(sim.data.site_xpos[teleop.site_id], dtype=float)
        assert end[0] - start[0] > 0.05          # 明显前进
        assert abs(end[1] - start[1]) < 0.01     # 不横漂
    finally:
        sim.close()


def test_arm_teleop_sideways_and_gripper():
    """vy 走基座 +y；grip=0 让夹爪执行器朝闭合值走。"""
    spec = load_world(str(PICK))
    sim = Simulation(spec, autostart_controllers=False)
    try:
        teleop = ArmTeleop(sim.model, sim.data, sim.index, spec, "arm")
        assert teleop._grip_aid is not None      # pick_cube 带 2FG7
        start = np.array(sim.data.site_xpos[teleop.site_id], dtype=float)
        teleop.set_command({"vy": 1.0, "grip": 0.0})
        sim.run(max_ticks=10, ctrl_fn=teleop.step)
        end = np.array(sim.data.site_xpos[teleop.site_id], dtype=float)
        assert end[1] - start[1] > 0.02
        closed = float(sim.data.ctrl[teleop._grip_aid])
        assert closed < teleop._grip_open         # 朝 close（0.0）方向
    finally:
        sim.close()


def test_teleop_session_records_episode(tmp_path):
    """teleop_start -> 指令 -> stop：stats 带录制摘要，产物完整落盘。"""
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": str(PICK)})
        info = s.start_teleop({
            "robot": "arm", "camera": "", "width": 64, "height": 48,
            "fps": 5, "record_dir": str(tmp_path / "ep"),
            "record_images": GL,
        })
        assert info["robot"] == "arm"
        assert info["realtime"] is True
        assert s._manual_sim is True             # 不启动控制器进程
        assert not s.sim._procs
        s.call_tool("teleop_cmd", {"vx": 0.6})
        time.sleep(0.5)
        s.call_tool("stop", {})
        s._run_thread.join(timeout=15)
        assert not s.is_running

        stats = s._last_stats
        assert stats["stopped"] is True and stats["ticks"] > 0
        rec = stats["recording"]
        assert rec["samples"] == stats["ticks"]
        ep = pathlib.Path(rec["dir"])
        assert (ep / "data.npz").exists() and (ep / "meta.json").exists()
        meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        assert meta["extra"]["source"] == "teleop"
        assert meta["extra"]["teleop"]["robot"] == "arm"
        assert meta["stats"]["ticks"] == stats["ticks"]
        if GL:
            assert rec["images"] > 0
            assert meta["images"]["count"] == rec["images"]
    finally:
        s.close()


def test_teleop_cmd_requires_session():
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": str(PICK)})
        with pytest.raises(WorldError, match="遥操作"):
            s.call_tool("teleop_cmd", {"vx": 1.0})
    finally:
        s.close()


def test_camera_stream_frame_is_png():
    """camera_stream('' 自由机位) 立即出一帧；camera_frame 返回 image。"""
    if not GL:
        pytest.skip("离屏渲染不可用")
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": str(PICK)})
        r = s.call_tool("camera_stream",
                        {"camera": "", "width": 64, "height": 48, "fps": 5})
        assert r["streaming"] is True
        frame = s.call_tool("camera_frame", {})
        block = frame["content"][0]
        assert block["type"] == "image" and block["mimeType"] == "image/png"
        assert _png_size(base64.b64decode(block["data"])) == (64, 48)
        off = s.call_tool("camera_stream", {"camera": None})
        assert off["streaming"] is False
        pending = s.call_tool("camera_frame", {})
        assert pending["pending"] is True
    finally:
        s.close()
