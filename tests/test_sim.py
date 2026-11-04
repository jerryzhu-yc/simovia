"""M0.2 内核测试：多进程通信、栅栏推进、超时与崩溃策略。

控制器脚本由测试动态生成（tmp_path），走真实子进程 + ZMQ 路径。
"""
import json
import pathlib
import subprocess

import mujoco
import numpy as np
import pytest

from simovia.errors import WorldError
from simovia.kernel.sim import Simulation
from simovia.world.builder import build_world
from simovia.world.schema import load_world, parse_world

ROOT = pathlib.Path(__file__).parent.parent
ARM_MODEL = str(ROOT / "models" / "demo_arm.xml")
WORLD = ROOT / "worlds" / "pick_cube.yaml"


def _world_dict(controller: str | None, **runtime) -> dict:
    d = {
        "name": "sim_test",
        "scene": {"objects": [
            {"name": "cube", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.3, 0, 0.051], "mass": 0.1, "color": "#F59E0B"}]},
        "robots": [{"name": "arm", "model": ARM_MODEL, "pos": [0, 0, 0.05],
                    "controller": controller}],
        "runtime": runtime or {"controller_timeout": 100.0},
    }
    return d


def _write_controller(tmp_path, code: str) -> str:
    p = tmp_path / "ctrl.py"
    p.write_text(code, encoding="utf-8")
    return str(p)


class TestAcceptance:
    def test_random_arm_1000_ticks(self):
        """验收：random_arm 跑 1000 tick，无死锁、无 NaN、无崩溃"""
        spec = load_world(WORLD)
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=1000)
        # 任务成功会提前终止运行（内核语义），两种结束方式都算通过
        assert stats["ticks"] == 1000 or stats["success"]
        assert not stats["nan"]
        assert stats["crashed"] == []
        assert stats["wall_time"] < 30        # 死锁保护网


class TestCtrlPlumbing:
    def test_controller_stdout_visible_with_prefix(self, tmp_path, capsys):
        """控制器 print 必须可见（转投 stderr + "[机器人名] " 前缀）。

        新手调试控制律基本靠 print；曾经是 DEVNULL，现象是"只有 stderr
        能看到、print 全是空的"，极易误判控制器没跑起来。
        """
        import time

        ctrl = _write_controller(tmp_path, """
from simovia.controller import Robot
robot = Robot()
n = 0
while robot.step():
    n += 1
    if n == 1:
        print("hello-from-controller", flush=True)
""")
        spec = parse_world(_world_dict(ctrl))
        with Simulation(spec) as sim:
            sim.run(max_ticks=5)
        text = ""
        deadline = time.time() + 3
        while time.time() < deadline:      # 读线程异步，轮询等它写完
            text += capsys.readouterr().err
            if "hello-from-controller" in text:
                break
            time.sleep(0.02)
        assert "[arm] hello-from-controller" in text

    def test_close_not_blocked_by_leaked_child(self, tmp_path):
        """控制器留下继承 stdout 的子进程时，close() 不能阻塞。

        收尾若从主线程关 stdout，会撞上读线程持有的缓冲区锁：控制器进程
        退出了、但孙进程还开着写端，就永远等不到 EOF——实测 close 卡死。
        """
        import time

        ctrl = _write_controller(tmp_path, """
import subprocess, sys
from simovia.controller import Robot
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
robot = Robot()
while robot.step():
    pass
""")
        spec = parse_world(_world_dict(ctrl))
        t0 = time.time()
        with Simulation(spec) as sim:
            sim.run(max_ticks=3)
        assert time.time() - t0 < 3      # 卡死时是无限等

    def test_ctrl_matches_reference(self, tmp_path):
        """控制量一致性：内核推进的轨迹与进程内参考实现逐位一致"""
        ctrl_path = _write_controller(tmp_path, f"""
from simovia.controller import Robot
robot = Robot()
robot.set_all({[0.5, 0.3, -0.2, -0.01, 0.01]})
while robot.step():
    pass
robot.close()
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            sim.run(max_ticks=20)
            kernel_qpos = sim.data.qpos.copy()

        # 进程内参考：同样模型、同样控制量、同样步数
        ref_model, _ = build_world(parse_world(_world_dict(None)))
        ref_data = mujoco.MjData(ref_model)
        spec2 = parse_world(_world_dict(None))
        for _ in range(20):
            ref_data.ctrl[:] = [0.5, 0.3, -0.2, -0.01, 0.01]
            for _ in range(spec2.ctrl_period):
                mujoco.mj_step(ref_model, ref_data)

        assert np.allclose(kernel_qpos, ref_data.qpos), \
            "内核路径与参考实现的物理轨迹不一致"
        assert np.all(np.isfinite(kernel_qpos))

    def test_obs_contains_names_and_state(self, tmp_path):
        """观测里带关节名/位置/body 位置，控制器侧可读"""
        ctrl_path = _write_controller(tmp_path, """
from simovia.controller import Robot
robot = Robot()
obs = robot.step()
assert "qpos" in obs and "bodies" in obs, f"观测缺字段: {obs.keys()}"
assert "wrist" in obs["bodies"], f"body 视图缺 wrist: {obs['bodies']}"
while robot.step():
    pass
robot.close()
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=5)
        assert stats["ticks"] == 5
        assert stats["crashed"] == []       # 断言没炸 = 控制器拿到了合法观测


class TestTimeoutPolicy:
    def test_slow_controller_last_ctrl(self, tmp_path):
        """慢控制器（0.1s/tick）+ 50ms 超时 + last_ctrl：内核补位推进，正常跑完"""
        ctrl_path = _write_controller(tmp_path, """
import time
from simovia.controller import Robot
robot = Robot()
while robot.step():
    time.sleep(0.1)
robot.close()
""")
        d = _world_dict(ctrl_path, controller_timeout=50.0)
        spec = parse_world(d)
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=10)
        assert stats["ticks"] == 10
        assert stats["lag"] > 0              # 确实发生了超时补位
        assert not stats["nan"]


class TestCrash:
    def test_crash_marks_inactive_no_deadlock(self, tmp_path):
        """控制器跑 2 tick 后 os._exit：内核标记崩溃并优雅结束，不死锁"""
        ctrl_path = _write_controller(tmp_path, """
import os
from simovia.controller import Robot
robot = Robot()
robot.step()
robot.step()
os._exit(1)
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=100)
        assert "arm" in stats["crashed"]
        assert stats["ticks"] >= 2           # 死掉前推进过
        assert not stats["nan"]


class TestNoController:
    def test_world_without_controllers_autoruns(self):
        """无控制器世界：空跑指定 tick（scene_gen 冒烟路径）"""
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=50)
        assert stats["ticks"] == 50
        assert not stats["nan"]


class TestStop:
    """stop_check：GUI 停止按钮 / MCP stop 工具的内核层。"""

    def test_stop_check_premature_finish_with_controller(self, tmp_path):
        """带控制器的世界：stop_check 为真时提前收尾——控制器进程随 stop
        正常退出（不泄漏、不算崩溃），已跑 tick 数小于上限"""
        ctrl_path = _write_controller(tmp_path, """
from simovia.controller import Robot
robot = Robot()
while robot.step():
    robot.set("a1", 0.3)
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=500, stop_check=lambda: sim.tick >= 5)
        assert 5 <= stats["ticks"] < 500     # 提前收尾且至少推进了几 tick
        assert stats["crashed"] == []        # 停止收尾不是崩溃
        assert not stats["nan"]
        proc = sim._procs["arm"]             # 控制器毫秒级退出（无泄漏）
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            raise AssertionError("停止后控制器未在 3s 内退出（泄漏）")

    def test_stop_check_no_controller_world(self):
        """无控制器世界：stop_check 同样生效（空跑循环也要能停）"""
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=10000, stop_check=lambda: sim.tick >= 7)
        assert 7 <= stats["ticks"] < 10000
        assert not stats["nan"]

    def test_stop_check_never_true_runs_to_end(self):
        """stop_check 恒 False：行为与不传完全一致（跑满）"""
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=20, stop_check=lambda: False)
        assert stats["ticks"] == 20


class TestSubstepSampling:
    """实时播放的子步采样：控制 tick 内按约 1/60s 切片出画面。

    pick_cube 这类世界一个控制 tick 跨 32 个物理步（64ms），只在 tick
    边界出帧时画面只有 15.6Hz；切片后 GUI 能拿到 60Hz 的中间态。
    """

    def test_sample_called_between_control_ticks(self):
        spec = parse_world(_world_dict(None))     # 32 步/tick，dt=2ms
        times = []
        with Simulation(spec) as sim:
            sim.run(max_ticks=2, realtime=True,
                    on_sample=lambda: times.append(float(sim.data.time)))
        # 每 tick 64ms，按 8 步（16ms）一片 -> 每 tick 4 次，2 tick 共 8 次
        assert len(times) == 8
        assert times[0] == pytest.approx(0.016, abs=1e-9)   # 首次在 tick 中间
        assert times[-1] == pytest.approx(0.128, abs=1e-9)  # 末次在 tick 末尾

    def test_sampling_does_not_change_trajectory(self):
        """采样只多读状态：物理轨迹与不采样逐位一致（reproducibility）。

        回调里跑 MCP 快照同款 mj_kinematics——派生量重算不得扰动 warmstart。
        """
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            sim.run(max_ticks=2, realtime=False)
            ref = sim.data.qpos.copy()
        with Simulation(spec) as sim2:
            def sample():
                mujoco.mj_kinematics(sim2.model, sim2.data)
            sim2.run(max_ticks=2, realtime=True, on_sample=sample)
            got = sim2.data.qpos.copy()
        assert np.array_equal(ref, got)

    def test_no_sampling_when_not_realtime(self):
        """全速跑批（realtime=False）不切片：只保留 on_tick 出帧。"""
        spec = parse_world(_world_dict(None))
        times = []
        with Simulation(spec) as sim:
            sim.run(max_ticks=2, realtime=False,
                    on_sample=lambda: times.append(float(sim.data.time)))
        assert times == []

    def test_sample_with_controller(self, tmp_path):
        """带控制器的世界（pick_cube 的实际路径）同样按子步采样。"""
        ctrl_path = _write_controller(tmp_path, """
from simovia.controller import Robot
robot = Robot()
while robot.step():
    pass
""")
        spec = parse_world(_world_dict(ctrl_path))
        times = []
        with Simulation(spec) as sim:
            sim.run(max_ticks=1, realtime=True,
                    on_sample=lambda: times.append(float(sim.data.time)))
        assert len(times) == 4        # 32 步/tick，按 8 步切片
        assert times[-1] == pytest.approx(0.064, abs=1e-9)


class TestHandshake:
    """握手等待期的鲁棒性（Windows 卡死病灶的回归守护）。

    控制器进程活着但迟迟不发 hello 时，原先的握手循环会永久转圈：
    run 不返回、stop_check 无人查、无任何报错。
    """

    def test_handshake_timeout_raises_and_kills(self, tmp_path, monkeypatch):
        """握手超时：run 抛 WorldError（带控制器名），进程被终止不泄漏"""
        monkeypatch.setattr("simovia.kernel.sim._HANDSHAKE_TIMEOUT", 0.3)
        ctrl_path = _write_controller(tmp_path, """
import time
time.sleep(60)          # 活着但永不握手
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            with pytest.raises(WorldError, match="握手超时.*arm"):
                sim.run(max_ticks=10)
            proc = sim._procs["arm"]
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                raise AssertionError("握手超时后控制器未被终止（泄漏）")

    def test_stop_during_handshake_terminates(self, tmp_path):
        """握手等待期停止：run 正常返回 stats（不抛），控制器被终止，
        不计入 crashed（用户主动停止不是故障）"""
        ctrl_path = _write_controller(tmp_path, """
import time
time.sleep(30)          # 握手前先睡：让 run 停在握手等待期
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=10, stop_check=lambda: True)
            assert stats["ticks"] == 0
            assert stats["crashed"] == []
            proc = sim._procs["arm"]
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                raise AssertionError("握手期停止后控制器未被终止（泄漏）")

    def test_controller_stderr_visible(self, tmp_path, capfd):
        """控制器 stderr 不再被吞（直通继承内核 stderr）：起不来的
        真实报错必须可见，这是 Windows 卡死可诊断的前提"""
        ctrl_path = _write_controller(tmp_path, """
import sys
print("stderr-passthrough-check", file=sys.stderr)
sys.exit(0)
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            sim.run(max_ticks=2)
        assert "stderr-passthrough-check" in capfd.readouterr().err


class TestJointStateAlignment:
    def test_floating_base_joint_state_not_shifted(self, tmp_path):
        """浮动机器人（命名 free 关节）的 joint_state 不能整体错位。

        回归点：命名 free 关节在 qpos 里占 7 个值（xyz+四元数）。以前
        joint_state 直接 dict(zip(joint_names, qpos))，free 关节之后的
        所有关节都会错位——人形/四足写完控制器读到的位置全不对。
        """
        from simovia.zoo import ensure_model

        try:
            g1 = ensure_model("unitree_g1")
        except Exception as e:                 # 模型库不可用
            pytest.skip(f"G1 模型不可用: {e}")

        out = tmp_path / "joint_state.json"
        ctrl_path = _write_controller(tmp_path, f"""
import json
from simovia.controller import Robot
robot = Robot()
for name in ("left_elbow_joint", "left_hip_pitch_joint"):
    robot.set(name, 0.5)
while robot.step():
    open({str(out)!r}, "w").write(json.dumps(robot.joint_state()))
robot.close()
""")
        d = {"name": "float_base", "timestep": 0.002, "ctrl_period": 8,
             "gravity": [0, 0, 0],
             "scene": {"floor": {"size": [2.0, 2.0]}},
             "robots": [{"name": "arm", "model": str(g1),
                         "controller": ctrl_path}]}
        with Simulation(parse_world(d)) as sim:
            stats = sim.run(max_ticks=80)
            truth = {n: float(sim.data.qpos[sim.model.jnt_qposadr[jid]])
                     for n, jid in sim.index.robots["arm"].joints.items()}
        assert stats["crashed"] == []
        js = json.loads(out.read_text())
        assert set(js) == set(truth)
        assert len(js["floating_base_joint"]) == 7
        assert js["left_elbow_joint"] == pytest.approx(truth["left_elbow_joint"])
        assert js["left_hip_pitch_joint"] == pytest.approx(
            truth["left_hip_pitch_joint"])
