"""Go2 行走 demo 的 sim2sim 目标端：同一份 CPG 控制律跑在 simovia 里。

源端对照：examples/sim2sim/mujoco_go2_trot.py（独立 mujoco 步进，不经过
simovia）；两端共用 simovia/sim2sim/go2_trot.py 里的参数与公式，差别只在
物理引擎配置，所以视频里的差异可以归因到仿真条件而不是两份实现。

日志统一走 stderr：MCP/GUI 模式下 stdout 是 JSON-RPC 协议流。
"""
import sys

import numpy as np

from simovia.controller import Robot
from simovia.sim2sim.go2_trot import ACTUATORS, JOINTS, Go2Trot


def _log(msg: str) -> None:
    print(f"[go2-sim2sim] {msg}", file=sys.stderr, flush=True)


robot = Robot()
if list(robot.actuators) != list(ACTUATORS):
    raise RuntimeError(
        f"执行器顺序与 go2_trot 契约不一致：{robot.actuators}")

dt = robot.ctrl_period * robot.timestep
gait = Go2Trot(dt)
_log(f"start dt={dt:.4f}s ({1.0 / dt:.1f}Hz) actuators={len(robot.actuators)}")

q_birth = None
start_x = 0.0
last_t = 0.0

while robot.step():
    obs = robot.obs
    q = np.asarray(obs["qpos"], dtype=float)
    qd = np.asarray(obs["qvel"], dtype=float)
    root = obs.get("root") or [0.0, 0.0, 0.0]
    if q_birth is None:
        # 首帧才能拿到 joint_names（hello 只下发执行器清单）
        names = obs.get("joint_names") or []
        if names != list(JOINTS):
            raise RuntimeError(f"关节顺序与契约不一致：{names}")
        if len(q) != len(ACTUATORS) or len(qd) != len(ACTUATORS):
            raise RuntimeError(
                f"观测维度不对：qpos={len(q)} qvel={len(qd)}，"
                f"期望 {len(ACTUATORS)}（free 关节不应进观测）")
        q_birth = q.copy()
        start_x = float(root[0])
        _log(f"birth qpos z={root[2]:.3f}")

    robot.set_all(gait.torques(q, qd, robot.tick, q_birth))
    last_t = robot.time

    if robot.tick % 50 == 0:
        dist = float(root[0]) - start_x
        speed = dist / last_t if last_t > 0 else 0.0
        _log(f"t={last_t:5.2f}s x={root[0]:5.2f} y={root[1]:6.2f} "
             f"z={root[2]:.3f} v={speed:4.2f}m/s")

root = robot.obs.get("root") or [0.0, 0.0, 0.0]
dist = float(root[0]) - start_x
speed = dist / last_t if last_t > 0 else 0.0
_log(f"done t={last_t:.2f}s distance={dist:.2f}m avg_v={speed:.2f}m/s "
     f"z={root[2]:.3f}")
