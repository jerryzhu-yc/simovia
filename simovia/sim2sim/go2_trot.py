"""Go2 对角小跑 CPG：与 zoo demo 完全相同的一条控制律，但不依赖仿真器。

用途：sim2sim 的两端共用本模块，避免"两个仿真器各写一份控制律"导致对比
其实在比两份实现：

- 源端：examples/sim2sim/mujoco_go2_trot.py（独立 mujoco 步进）
- 目标端：controllers/go2_trot_sim2sim.py（simovia 控制器）

控制律与 zoo 的 demo.py 逐项一致：
    des_j = home_j + amp_j·sin(2π·freq·t + phase_j + lag_j)
    τ_j   = kp_j·(des_j − q_j) − kd_j·qd_j
前 SETTLE tick 不摆腿，只做出生姿态渐变（RAMP=1 表示立即压到站姿）；
des 与 τ 都按执行器控制范围截断（Menagerie 的 hip/thigh ±23.7 N·m、
calf ±45.43 N·m，同时用作关节角限位）。

顺序约定：所有数组按 ACTUATORS 排列，与 Menagerie go2.xml 的执行器顺序
一致；调用方用名字校验（simovia 侧用 actuator 名字写入，不碰内部 id）。
"""
from __future__ import annotations

import math

import numpy as np

# Menagerie unitree_go2/go2.xml 的 12 个执行器（motor = 力矩控制）
ACTUATORS = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)
# 观测里的关节名（Menagerie 的 joint 名带 _joint 后缀，顺序与 ACTUATORS 一致）
JOINTS = tuple(f"{a}_joint" for a in ACTUATORS)

FREQ = 2.4      # 步频 Hz（隐式积分下 2.0 会趴地，2.4 实测最稳）
SETTLE = 40     # 先站稳的 tick 数
RAMP = 1        # 出生姿态渐变到站姿的 tick 数

HOME = np.array([0.0, 0.9, -1.8] * 4)
AMP = np.array([0.0, 0.35, 0.55] * 4)
# 对角腿反相：FL/RR 同相，FR/RL 反相
PHASE = np.array([0.0, 0.0, 0.0, math.pi, math.pi, math.pi,
                  math.pi, math.pi, math.pi, 0.0, 0.0, 0.0])
LAG = np.array([0.0, 0.0, 0.5] * 4)      # 膝滞后大腿，模拟推进弧线
KP = np.array([25.0, 60.0, 60.0] * 4)
KD = np.array([0.5, 1.0, 1.0] * 4)
LIMIT = np.array([23.7, 23.7, 45.43] * 4)


class Go2Trot:
    """按控制 tick 生成 12 个力矩；dt 由调用方的 timestep×ctrl_period 决定。"""

    def __init__(self, dt: float):
        if dt <= 0:
            raise ValueError(f"dt 必须为正，实际 {dt}")
        self.dt = float(dt)

    def desired(self, tick: int, q_birth: np.ndarray) -> np.ndarray:
        """当前 tick 的 12 个目标关节角（含出生姿态渐变与相位摆动）。"""
        t = (tick - SETTLE) * self.dt
        frac = min(1.0, tick / RAMP)
        des = np.asarray(q_birth, dtype=float) + (HOME - q_birth) * frac
        if t >= 0.0:
            des = des + AMP * np.sin(2.0 * math.pi * FREQ * t + PHASE + LAG)
        return np.clip(des, -LIMIT, LIMIT)

    def torques(self, q: np.ndarray, qd: np.ndarray, tick: int,
                q_birth: np.ndarray) -> np.ndarray:
        """力矩 PD：τ = kp·(des − q) − kd·qd，按执行器范围截断。"""
        des = self.desired(tick, q_birth)
        tau = KP * (des - np.asarray(q, dtype=float)) - KD * np.asarray(qd, dtype=float)
        return np.clip(tau, -LIMIT, LIMIT)
