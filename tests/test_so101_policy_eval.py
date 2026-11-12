"""SmolVLA 成功停止：不加载模型，验证终止条件及真实控制循环。"""
import contextlib
import importlib.util
import pathlib
import types

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "so101_policy_eval_test", ROOT / "controllers" / "so101_policy_eval.py")
policy_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy_eval)
WORLD = yaml.safe_load((ROOT / "worlds" / "so101_policy_eval.yaml").read_text())
ARGS = WORLD["robots"][0]["args"]
ZONES = ARGS["success_zones"]
PLACED = {"cube_red": [0.178, -0.13, 0.184],
          "cube_blue": [0.178, 0.13, 0.184]}
OUTSIDE = {"cube_red": [0.22, -0.04, 0.18],
           "cube_blue": [0.22, 0.04, 0.18]}


def monitor(**kwargs):
    return policy_eval.SortCompletion(ZONES, **kwargs)


def advance(check, objects, grip=0.9, start=0.0, steps=40):
    return [check.update(objects, grip, start + i * 0.032) for i in range(steps)]


def test_both_placed_released_and_stable_latch_success():
    check = monitor()
    states = advance(check, PLACED)
    assert states[0] == "settling"
    assert states[15] != "success"  # 不够 0.5 秒
    assert states[-1] == "success"
    assert check.update(OUTSIDE, 0.0, 2.0) == "success"  # 本轮终态锁定


@pytest.mark.parametrize("objects", [
    {**OUTSIDE, "cube_red": PLACED["cube_red"]},
    {**OUTSIDE, "cube_blue": PLACED["cube_blue"]},
    {"cube_red": PLACED["cube_blue"], "cube_blue": PLACED["cube_red"]},
    {**PLACED, "cube_blue": [0.178, 0.13, 0.23]},  # 悬空
    {**PLACED, "cube_blue": [0.23, 0.13, 0.184]},  # 盒边
    {"cube_red": PLACED["cube_red"]},
    {**PLACED, "cube_blue": [float("nan"), 0.13, 0.184]},
])
def test_incomplete_wrong_box_or_invalid_positions_never_succeed(objects):
    assert set(advance(monitor(), objects)) == {"running"}


def test_closed_gripper_never_succeeds():
    assert set(advance(monitor(), PLACED, grip=0.1)) == {"running"}


def test_moving_cubes_do_not_count_as_stable():
    check = monitor()
    for i in range(50):
        objects = {**PLACED, "cube_blue": [0.17 + 0.01 * (i % 2), 0.13, 0.184]}
        assert check.update(objects, 0.9, i * 0.032) == "settling"


def test_leaving_zone_resets_stability_timer():
    check = monitor()
    assert advance(check, PLACED, steps=12)[-1] == "settling"
    assert check.update(OUTSIDE, 0.9, 0.4) == "running"
    assert advance(check, PLACED, start=0.432, steps=12)[-1] == "settling"
    assert advance(check, PLACED, start=0.816, steps=12)[-1] == "success"


def test_repeated_or_rewound_time_cannot_finish():
    check = monitor()
    for _ in range(100):
        assert check.update(PLACED, 0.9, 0.0) == "settling"
    advance(check, PLACED, steps=12)
    assert check.update(PLACED, 0.9, 0.0) == "settling"
    assert advance(check, PLACED, start=0.032, steps=12)[-1] == "settling"


@pytest.mark.parametrize("kwargs", [
    {"hold_seconds": 0}, {"hold_seconds": float("nan")},
    {"max_speed": -1}, {"gripper_open_min": float("inf")},
])
def test_invalid_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        monitor(**kwargs)


def test_missing_goals_rejected():
    with pytest.raises(ValueError, match="success_zones"):
        policy_eval.SortCompletion({})


class Chunk(np.ndarray):
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


class FakePolicy:
    config = types.SimpleNamespace(n_action_steps=50, chunk_size=50)
    def __init__(self): self.calls = 0
    def to(self, device): return self
    def eval(self): return self
    def predict_action_chunk(self, batch):
        self.calls += 1
        # 队列里仍有会闭爪的动作，完成检测必须抢在这些动作之前。
        return np.full((1, 50, 6), -0.1).view(Chunk)


class FakeRobot:
    def __init__(self, tmp_path, placed=True):
        (tmp_path / "config.json").write_text("{}")
        self.args = {**ARGS, "model": str(tmp_path), "max_seconds": 60}
        self.actuators = [f"a{i}" for i in range(6)]
        self.ranges = [[-3, 3]] * 6
        self.timestep, self.ctrl_period = 0.002, 16
        self.time = 0.0
        self.steps = 0
        self.placed = placed
        self.commands, self.results = [], []
        self.closed = False
        self.obs = {"qpos": [*policy_eval.PRE_Q5, 0.9], "objects": OUTSIDE}
    def observe(self): return self.obs
    def camera(self, *args, **kwargs): return np.zeros((2, 2, 3), dtype=np.uint8)
    def publish_camera(self, *args): pass
    def step(self):
        self.steps += 1
        self.time = self.steps * 0.032
        if self.steps > 60: return False
        self.obs = {"qpos": [*policy_eval.PRE_Q5, 0.9],
                    "objects": PLACED if self.placed and self.steps >= 4 else OUTSIDE}
        return self.obs
    def set_all(self, values): self.commands.append((self.steps, values))
    def report_result(self, success, detail): self.results.append((success, detail))
    def close(self): self.closed = True


def run_fake(monkeypatch, robot):
    policy = FakePolicy()
    torch = types.SimpleNamespace(no_grad=contextlib.nullcontext, float32=np.float32,
                                  tensor=lambda x, **kw: np.asarray(x))
    factory = lambda *a, **kw: (lambda x: x, lambda x: x)
    loader = types.SimpleNamespace(from_pretrained=lambda _: policy)
    monkeypatch.setattr(policy_eval, "_load_policy_runtime", lambda: (torch, factory, loader))
    monkeypatch.setattr(policy_eval, "Robot", lambda: robot)
    monkeypatch.setattr(policy_eval, "to_chw", lambda x: x)
    policy_eval.main()
    return policy


def test_main_stops_mid_chunk_without_more_policy_actions(monkeypatch, tmp_path):
    robot = FakeRobot(tmp_path)
    policy = run_fake(monkeypatch, robot)
    assert robot.results[0][0] is False
    assert robot.results[-1][0] is True
    assert robot.closed and robot.steps < 60
    assert policy.calls == 2  # 预热 + 一个 chunk；已入盒后不再调用模型
    for tick, command in robot.commands:
        if tick >= 4:
            np.testing.assert_allclose(command, [*policy_eval.PRE_Q5, 0.9])
    assert robot.commands[-1][0] < robot.steps  # 成功这一拍不再发动作


def test_timeout_still_reports_failure(monkeypatch, tmp_path):
    robot = FakeRobot(tmp_path, placed=False)
    robot.args["max_seconds"] = 2
    monkeypatch.setattr(policy_eval.time, "monotonic", lambda: float(robot.steps))
    run_fake(monkeypatch, robot)
    assert robot.closed and not any(success for success, _ in robot.results)
    assert "timeout" in robot.results[-1][1]


def test_world_uses_controller_result_instead_of_red_only_predicate():
    from simovia.world.schema import load_world
    from simovia.kernel.sim import Simulation
    spec = load_world(ROOT / "worlds" / "so101_policy_eval.yaml")
    assert spec.task.success is None
    with Simulation(spec, autostart_controllers=False) as sim:
        assert sim._task_success() is None
        sim._st["arm"] = {"crashed": False}  # 无进程测试补齐控制器状态
        sim.controller_results["arm"] = {"success": True, "detail": "both sorted"}
        assert sim._task_success() is True
        sim.controller_results["arm"] = {"success": False, "detail": "timeout"}
        assert sim._task_success() is False


@pytest.mark.parametrize("blue_placed, expected", [(True, True), (False, False)])
def test_physics_controller_reports_completion_for_both_cubes(tmp_path, blue_placed, expected):
    """真实 MuJoCo + ZMQ 控制器：落稳后报告成功；只放红块不能成功。"""
    from simovia.world.schema import load_world
    from simovia.kernel.sim import Simulation
    script = tmp_path / "check_sort.py"
    script.write_text(
        "from simovia.controller import Robot\n"
        "from controllers.so101_policy_eval import SortCompletion\n"
        "robot = Robot()\n"
        "check = SortCompletion(robot.args['success_zones'])\n"
        "robot.report_result(False, 'sorting')\n"
        "while robot.step():\n"
        "    q = robot.obs['qpos'][:6]\n"
        "    state = check.update(robot.obs['objects'], q[5], robot.time)\n"
        "    if state == 'success':\n"
        "        robot.report_result(True, 'both sorted and stable')\n"
        "        break\n"
        "    robot.set_all([0, 0, 0, 0, 0, 0.9])\n"
        "robot.close()\n")
    data = yaml.safe_load((ROOT / "worlds" / "so101_policy_eval.yaml").read_text())
    for obj in data['scene']['objects']:
        if obj['name'] == 'cube_red' or (blue_placed and obj['name'] == 'cube_blue'):
            obj['pos'] = PLACED[obj['name']]
    data['robots'][0]['controller'] = str(script)
    data['runtime']['realtime'] = False
    path = tmp_path / 'world.yaml'
    path.write_text(yaml.safe_dump(data))
    with Simulation(load_world(path)) as sim:
        stats = sim.run(max_ticks=100)
        assert stats['success'] is expected, stats
        assert not stats['crashed'] and not stats['nan']
        if expected:
            assert stats['ticks'] < 100
            assert stats['done'] == ['arm']
        else:
            assert stats['ticks'] == 100
