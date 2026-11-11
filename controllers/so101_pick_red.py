"""SO-101 桌面分拣控制器：把红色方块抓起来放进红盒子里。

场景（worlds/so101_sort.yaml）：SO-101 固定在桌面上表面 (0,0,0.16)，
红方块 (0.22,-0.04,0.18)、蓝方块 (0.22,0.04,0.18)，红盒 (0.178,-0.13)、
蓝盒 (0.178,0.13)。本控制器只处理红色：抓取 -> 抬升 -> 平移到红盒上方
-> 下落 -> 松爪 -> 撤手。

几何依据（model: zoo://so101_lerobot）
- 工具轴 = `gripperframe` site 的 **x 轴**（site quat=(1,0,1,0) 旋转 90°，
  使 x 指向指尖、z 为平行夹爪开合方向）。顶抓 = site x 轴朝下。
- 指板沿工具轴跨 site-0.053 ~ site+0.005（site 在指尖附近），故抓取时
  site 放在方块中心下方 1.5cm（z=0.170），指板覆盖方块上 2.5cm。
- 夹爪 ctrl 与指间距实测：-0.175->4mm、0->16mm、0.32->41mm、0.9->84mm。
  张开 0.9 让 4cm 方块轻松入位，闭合 -0.10 由 position 伺服顶住方块
  （力矩饱和在 2.94 N·m，压力足够，撞不飞）。
- 航点由数值 IK 求出（位置 0mm + 工具轴 0°），并逐段热启动保证同族解，
  避免手腕 roll 冗余造成关节空间插值乱转。方块与盒子同半径（约 0.22m），
  抬升后的运输段只需肩部 pan，不切换肘部构型。

工作空间限制（实测）：方块上方 z<=0.245、盒子上方 z<=0.215 才精确可达，
故抬升高度取 0.245、跨盒高度 0.215，均在盒壁顶 (0.175) 之上。
"""
from simovia.controller import Robot

# (相位, 5 个臂关节伺服目标 rad, 夹爪 ctrl, 关节斜坡速度 rad/s)
PHASES = [
    ("pre",   (0.21400, -0.16915, 0.11906, 1.62081, 0.03055), 0.90, 0.40),
    ("d1",    (0.21399, -0.16802, 0.30328, 1.43563, 0.03034), 0.90, 0.40),
    ("d2",    (0.21398, -0.12726, 0.44792, 1.25023, 0.03022), 0.90, 0.40),
    ("grasp", (0.21398, -0.04620, 0.55452, 1.06255, 0.03016), 0.90, 0.25),
    ("close", (0.21398, -0.04620, 0.55452, 1.06255, 0.03016), -0.10, 0.25),
    ("up1",   (0.21398, -0.13885, 0.42231, 1.28727, 0.03025), -0.10, 0.35),
    ("up2",   (0.21399, -0.17341, 0.23481, 1.50933, 0.03043), -0.10, 0.35),
    ("carry", (0.50610, -0.14090, 0.12578, 1.58584, 0.23528), -0.10, 0.50),
    ("over",  (0.77248, -0.11504, 0.28290, 1.40301, 0.42429), -0.10, 0.50),
    ("open",  (0.77254, -0.08020, 0.39498, 1.25610, 0.42535), 0.90, 0.30),
    ("ret",   (0.77248, -0.11525, 0.28316, 1.40282, 0.42428), 0.90, 0.35),
    ("back",  (0.26376, -0.14795, 0.09630, 1.62237, -0.00500), 0.90, 0.50),
    ("hold",  (0.26376, -0.14795, 0.09630, 1.62237, -0.00500), 0.90, 0.00),
]
TOL = 0.03          # 5 关节到位容差 rad
STABLE = 25         # 连续 N tick 在容差内才算到位（约 1.6s，等伺服沉降）
TIMEOUT = 300       # 单相位 tick 上限
CLOSE_TICKS = 50    # 夹爪闭合/张开的硬超时（伺服被方块挡住，永远到不了目标）
HOLD_TICKS = 30
GRIP_SPEED = 0.70   # 夹爪斜坡速度 rad/s
TICK = 0.064        # 控制周期（ctrl_period 32 x timestep 0.002）

robot = Robot()
phase_i = stable = phase_ticks = 0
ramp_state: list[float] | None = None


def ramp(cur: float, tgt: float, step: float) -> float:
    return cur + max(-step, min(step, tgt - cur))


while robot.step():
    name, q5, grip, speed = PHASES[phase_i]
    q = robot.obs["qpos"]              # 本机器人 6 个关节：5 臂 + 夹爪
    if ramp_state is None:
        ramp_state = list(q[:6])       # 首 tick 从实际状态起步
    step_j = speed * TICK
    for k, tgt in enumerate(q5):
        ramp_state[k] = ramp(ramp_state[k], tgt, step_j)
    ramp_state[5] = ramp(ramp_state[5], grip, GRIP_SPEED * TICK)
    robot.set_all(ramp_state)

    # 到位判定只看 5 个臂关节（夹爪夹住方块后到不了闭合目标）
    err = max(abs(q[k] - g) for k, g in enumerate(q5))
    phase_ticks += 1
    stable = stable + 1 if err < TOL else 0

    done = stable >= STABLE or phase_ticks >= TIMEOUT
    # 松爪/夹紧：臂已在位，只需等夹爪行程走完
    if name in ("close", "open") and phase_ticks >= CLOSE_TICKS:
        done = True
    if name == "hold":
        if phase_ticks >= HOLD_TICKS or done:
            break
    if done:
        phase_i = min(phase_i + 1, len(PHASES) - 1)
        stable = phase_ticks = 0

robot.close()
