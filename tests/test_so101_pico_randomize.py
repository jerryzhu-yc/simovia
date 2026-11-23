"""采集随机化：worlds/so101_pico_sort.yaml 的 args.randomize 解析与采样。"""
import importlib.util
import pathlib
import random

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "so101_pico_teleop_test",
    ROOT / "controllers" / "so101_pico_teleop.py")
teleop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(teleop)

ARGS = yaml.safe_load(
    (ROOT / "worlds" / "so101_pico_sort.yaml").read_text(encoding="utf-8")
)["robots"][0]["args"]


def test_world_yaml_enables_randomization():
    """采集世界必须开到红蓝块都随机，否则策略又会学成位置先验。"""
    spec, min_sep = teleop._parse_randomize(ARGS)
    assert set(spec) == {"cube_red", "cube_blue"}
    assert min_sep == pytest.approx(0.06)
    for box in spec.values():
        assert box["x"][0] <= box["x"][1]
        assert box["y"][0] <= box["y"][1]
        assert box["z"] > 0


def test_missing_config_disables_randomization():
    assert teleop._parse_randomize({}) == ({}, 0.0)
    assert teleop._parse_randomize({"randomize": None}) == ({}, 0.0)


def test_bad_range_is_reported():
    bad = {"randomize": {"objects": {
        "cube_red": {"x": [0.26, 0.18], "y": [0.0, 0.0], "z": 0.18}}}}
    with pytest.raises(SystemExit, match="区间下限大于上限"):
        teleop._parse_randomize(bad)


def test_samples_stay_in_bounds_and_apart():
    spec, min_sep = teleop._parse_randomize(ARGS)
    rng = random.Random(0)
    for _ in range(300):
        poses = teleop._sample_positions(spec, min_sep, rng)
        assert set(poses) == set(spec)
        for name, (x, y, z) in poses.items():
            (lo_x, hi_x), (lo_y, hi_y) = spec[name]["x"], spec[name]["y"]
            assert lo_x <= x <= hi_x and lo_y <= y <= hi_y
            assert z == spec[name]["z"]
        red, blue = poses["cube_red"], poses["cube_blue"]
        dist = ((red[0] - blue[0]) ** 2 + (red[1] - blue[1]) ** 2) ** 0.5
        assert dist >= min_sep - 1e-12


def test_randomize_objects_resets_then_teleports():
    """先复位到 YAML 初始位姿，再按采样结果逐个瞬移。"""
    calls = []

    class FakeRobot:
        def reset_objects(self):
            calls.append(("reset",))

        def supervise(self, action, **params):
            calls.append((action, params))

    spec = {"cube_red": {"x": [0.2, 0.2], "y": [-0.03, -0.03], "z": 0.18},
            "cube_blue": {"x": [0.2, 0.2], "y": [0.03, 0.03], "z": 0.18}}
    teleop._randomize_objects(FakeRobot(), spec, 0.06)
    assert calls[0] == ("reset",)
    assert [c[0] for c in calls[1:]] == ["teleport", "teleport"]
    assert {c[1]["body"] for c in calls[1:]} == {"cube_red", "cube_blue"}
    for c in calls[1:]:
        assert len(c[1]["pos"]) == 3
