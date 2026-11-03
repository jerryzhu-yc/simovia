"""P0 验收：Episode 录制（recorder.py）。

守护点：
- 输出目录含 data.npz + meta.json，字段维度与契约一致
- 同一世界、同一控制器两次录制逐位一致（训练数据可复现）
- go2 浮动基座录制包含 root 位姿/速度，meta 里的契约可独立核对
"""
import json
import pathlib

import numpy as np
import pytest

from simovia.kernel.sim import Simulation
from simovia.recorder import EpisodeRecorder
from simovia.world.schema import SensorSpec, load_world
from simovia.zoo import bundled_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
PICK = ROOT / "worlds" / "pick_cube.yaml"


def _record(spec, out_dir, ticks):
    with Simulation(spec) as sim:
        rec = EpisodeRecorder(spec, sim.model, sim.index, out_dir)
        stats = sim.run(max_ticks=ticks,
                        on_tick=lambda: rec.add(sim.tick, sim.data))
    return rec.finish(stats=stats, extra={"world_path": "test"}), stats


def _load(ep: pathlib.Path):
    with np.load(ep / "data.npz") as z:
        return {k: z[k] for k in z.files}


def test_pick_cube_record_layout(tmp_path):
    ep, stats = _record(load_world(str(PICK)), tmp_path / "ep", ticks=30)
    assert (ep / "data.npz").exists() and (ep / "meta.json").exists()

    data = _load(ep)
    assert data["tick"].tolist() == list(range(1, 31))
    assert data["time"][0] == pytest.approx(0.064)
    assert data["arm.joint_pos"].shape == (30, 9)
    assert data["arm.joint_vel"].shape == (30, 9)
    assert data["arm.action"].shape == (30, 8)
    assert data["object_pos"].shape == (30, 6)   # pick_cube 有 2 个物体
    assert data["object_quat"].shape == (30, 8)

    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "simovia-episode-v1"
    assert meta["world"] == "pick_cube"
    assert meta["frames"] == 30
    assert meta["contracts"]["arm"]["obs_dim"] == 18
    assert meta["contracts"]["arm"]["action_dim"] == 8
    digest = meta["contract_digests"]["arm"]
    assert isinstance(digest, str) and len(digest) == 64
    assert meta["stats"]["ticks"] == stats["ticks"] == 30
    assert meta["objects"] == ["table", "cube"]
    assert meta["files"]["frames"] is None


def test_pick_cube_record_bitwise_deterministic(tmp_path):
    spec = load_world(str(PICK))
    ep1, _ = _record(spec, tmp_path / "a", ticks=25)
    spec2 = load_world(str(PICK))
    ep2, _ = _record(spec2, tmp_path / "b", ticks=25)
    a, b = _load(ep1), _load(ep2)
    assert set(a) == set(b)
    for key in a:
        assert np.array_equal(a[key], b[key]), f"录制字段 {key} 两次不一致"


@pytest.mark.skipif(bundled_model("unitree_go2") is None,
                    reason="go2 资产缺失（仓库 models/zoo/ 不完整）")
def test_go2_floating_base_record(tmp_path):
    from simovia.zoo import generate
    r = generate("unitree_go2", out_dir=tmp_path / "go2", demo=True)
    spec = load_world(r["world"])
    ep, stats = _record(spec, tmp_path / "ep", ticks=40)

    data = _load(ep)
    assert data["unitree_go2.root_pos"].shape == (40, 3)
    assert data["unitree_go2.root_quat"].shape == (40, 4)
    assert data["unitree_go2.root_lin_vel"].shape == (40, 3)
    assert data["unitree_go2.root_ang_vel"].shape == (40, 3)
    assert data["unitree_go2.joint_pos"].shape == (40, 12)
    assert data["unitree_go2.action"].shape == (40, 12)
    # 四元数必须保持单位长度
    norms = np.linalg.norm(data["unitree_go2.root_quat"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    assert stats["crashed"] == []

    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    contract = meta["contracts"]["unitree_go2"]
    assert contract["floating_base"] is True
    assert contract["obs_dim"] == 37
    assert contract["control_dt"] == pytest.approx(0.016)
    assert {a["mode"] for a in contract["actuators"]} == {"torque"}


def _png_size(path: pathlib.Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def test_robot_mounted_camera_records_frames(tmp_path):
    """相机挂在机器人 body 上：短名解析、分辨率/步长、meta 挂载信息。"""
    spec = load_world(str(PICK))
    spec.robots[0].sensors.append(SensorSpec(
        name="wrist_cam", type="camera", body="link7", pos=(0.02, 0.0, 0.08)))

    with Simulation(spec) as sim:
        rec = EpisodeRecorder(
            spec, sim.model, sim.index, tmp_path / "ep",
            images=True, camera="wrist_cam", width=96, height=64,
            image_stride=2)
        stats = sim.run(max_ticks=6,
                        on_tick=lambda: rec.add(sim.tick, sim.data))
    ep = rec.finish(stats=stats)

    assert rec.camera == "arm/wrist_cam"
    assert rec.frame_count == 3
    frames = sorted((ep / "frames").glob("*.png"))
    assert [p.name for p in frames] == [
        "frame_000000.png", "frame_000001.png", "frame_000002.png"]
    assert all(_png_size(p) == (96, 64) for p in frames)

    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    assert meta["files"]["frames"] == "frames"
    assert meta["image_frames"] == [1, 3, 5]
    assert meta["images"] == {
        "enabled": True,
        "camera": "arm/wrist_cam",
        "camera_short": "wrist_cam",
        "camera_mount": {
            "robot": "arm", "body": "link7",
            "pos": [0.02, 0.0, 0.08], "quat": None},
        "width": 96,
        "height": 64,
        "stride": 2,
        "count": 3,
    }


def test_unknown_camera_fails_fast(tmp_path):
    spec = load_world(str(PICK))
    with Simulation(spec) as sim, pytest.raises(ValueError, match="找不到录制相机"):
        EpisodeRecorder(spec, sim.model, sim.index, tmp_path / "ep",
                        images=True, camera="nope")


def test_invalid_capture_options_rejected(tmp_path):
    spec = load_world(str(PICK))
    with Simulation(spec) as sim:
        with pytest.raises(ValueError, match="分辨率"):
            EpisodeRecorder(spec, sim.model, sim.index, tmp_path / "a",
                            width=0)
        with pytest.raises(ValueError, match="image_stride"):
            EpisodeRecorder(spec, sim.model, sim.index, tmp_path / "b",
                            image_stride=0)
