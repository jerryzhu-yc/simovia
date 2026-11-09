"""RRP 三自由度机械臂（2 转动 + 1 移动）· DH 参数表 + 解析逆运动学。

配套场景：worlds/rrp3_ik.yaml
运行：    python -m simovia run worlds/rrp3_ik.yaml
设备名：  python -m simovia inspect worlds/rrp3_ik.yaml

DH 参数表（标准 DH：T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)）
--------------------------------------------------------------------------
 i | θ      | d      | a     | α      | 关节       物理含义
---+--------+--------+-------+--------+-----------------------------------
 1 | q1     | 0.44   | 0.24  | 0°     | revolute   立柱 + 大臂（绕竖直轴）
 2 | q2     | 0      | 0.20  | 0°     | revolute   小臂（绕竖直轴）
 3 | 0      | q3     | 0     | 180°   | prismatic  末端滑台（沿 z2 升降）
--------------------------------------------------------------------------
α3=180° 让末端坐标系 z 轴朝下，法兰（长 0.10 m）自然垂向桌面。

正运动学
--------
    T = A1(q1) · A2(q2) · A3(q3)，A 即上面的 DH 齐次变换
    frame3 = T 的平移列（再叠加底座高度 base_height）
    工具点 tip = frame3 + 0.10 · z3        （z3 恒为世界 -z）

逆运动学（解析解，无迭代）
--------------------------
    工具点高度：  z_tip = base_height + d1 + q3 - 0.10
    ⇒            q3 = z_tip - 0.40

    水平面内退化成 2R 平面臂（连杆 a1=0.24、a2=0.20）：
        r  = hypot(x, y)，φ = atan2(y, x)
        c2 = (r² - a1² - a2²) / (2·a1·a2)
        q2 = +acos(c2)                    ← 肘上解；取 -acos 即肘下解
        q1 = φ - atan2(a2·sin q2, a1 + a2·cos q2)

    |c2| > 1 表示目标在工作空间外（内孔 r<|a1-a2| 或外沿 r>a1+a2）：
    夹到边界取最近可达点，不抛异常，日志里标 WARN 并计入失败。

控制流程
--------
按 args.targets 逐点做「余弦缓动直线插值 → IK → 位置伺服」，每段到点后
保持 hold_ticks 统计稳定误差。全部路点走完时用 report_result 上报成功
与否（场景未定义 task.success，内核会采用控制器自报结果）。
"""
import functools
import math

import numpy as np

from simovia.controller import Robot

# 内核收尾会 terminate 控制器进程，标准输出是块缓冲——逐行 flush 防丢日志
print = functools.partial(print, flush=True)

# ---------------------------------------------------------------- 几何常量
BASE_HEIGHT = 0.06          # 底座圆柱高度（与 YAML spec.base_height 一致）
D1 = 0.44                   # DH 第 1 行 d
A1 = 0.24                   # DH 第 1 行 a（大臂长）
A2 = 0.20                   # DH 第 2 行 a（小臂长）
FLANGE = 0.10               # 法兰从 frame3 沿 z3 伸出的长度
ALPHA3 = math.pi            # DH 第 3 行 α=180°

DEFAULT_TARGETS = [
    [0.28, -0.16, 0.45],
    [0.34, -0.10, 0.43],
    [0.30, 0.12, 0.43],
    [0.40, 0.00, 0.43],
]


# ---------------------------------------------------------------- 运动学

def _A(theta, d, a, alpha):
    """一行标准 DH 的齐次变换 T = Rz(θ)·Tz(d)·Tx(a)·Rx(α)。"""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk(q1, q2, q3):
    """正运动学 -> (T_3, frame3 原点, 工具点)。与仿真模型逐位同构。"""
    T = _A(q1, D1, A1, 0.0) @ _A(q2, 0.0, A2, 0.0) @ _A(0.0, q3, 0.0, ALPHA3)
    T[2, 3] += BASE_HEIGHT
    frame3 = T[:3, 3].copy()
    tip = frame3 + T[:3, 2] * FLANGE
    return T, frame3, tip


def ik(target):
    """解析逆解 -> (q1, q2, q3, reachable)。

    reachable=False 时返回值已夹到工作空间边界（最近可达点），
    调用方仍可安全下发给伺服。
    """
    x, y, z = (float(v) for v in target)
    q3 = z + FLANGE - (BASE_HEIGHT + D1)          # 竖直移动直接解出
    r = math.hypot(x, y)
    c2 = (r * r - A1 * A1 - A2 * A2) / (2.0 * A1 * A2)
    reachable = -1.0 <= c2 <= 1.0
    q2 = math.acos(min(1.0, max(-1.0, c2)))       # 肘上解
    q1 = math.atan2(y, x) - math.atan2(A2 * math.sin(q2),
                                       A1 + A2 * math.cos(q2))
    return q1, q2, q3, reachable


# ---------------------------------------------------------------- 小工具

def _resolve(names, short):
    """在带机器人前缀的名字里找到以 /short 结尾的那个（如 rrp3/joint1）。"""
    for n in names:
        if n == short or n.endswith("/" + short):
            return n
    raise KeyError(f"找不到 '{short}'，可用: {sorted(names)}")


def _clip(v, lo_hi, tag, state):
    lo, hi = lo_hi
    if v < lo or v > hi:
        state[tag] = True
        return min(hi, max(lo, v))
    return v


def _fmt(v):
    return "[" + " ".join(f"{x:+.3f}" for x in v) + "]"


# ---------------------------------------------------------------- 主循环

def main():
    robot = Robot()
    move_ticks = max(1, int(robot.args.get("move_ticks", 70)))
    hold_ticks = max(1, int(robot.args.get("hold_ticks", 45)))
    log_every = max(1, int(robot.args.get("log_every", 30)))
    waypoints = [np.asarray(t, dtype=float)
                 for t in robot.args.get("targets", DEFAULT_TARGETS)]

    act = {s: _resolve(robot.actuators, s)
           for s in ("joint1", "joint2", "joint3")}
    rng = {s: robot.range(act[s]) for s in act}

    print("=" * 74)
    print("RRP 三自由度机械臂 · DH 参数表")
    print("  i | theta |    d |    a | alpha | joint")
    print("  1 |    q1 | 0.44 | 0.24 |    0° | revolute  （竖直轴回转）")
    print("  2 |    q2 | 0.00 | 0.20 |    0° | revolute  （竖直轴回转）")
    print("  3 |     0 |   q3 | 0.00 |  180° | prismatic （竖直升降）")
    print(f"  base_height={BASE_HEIGHT} m  法兰长={FLANGE} m  "
          f"工具点高度 = 0.40 + q3")
    print(f"  actuators={robot.actuators}")
    print(f"  ranges: " + "  ".join(
        f"{s}=[{rng[s][0]:+.3f},{rng[s][1]:+.3f}]" for s in act))
    print("=" * 74)

    # ---- 时间线：move 段（缓动插值）+ hold 段（静止统计误差）
    _, _, p_start = fk(0.0, 0.0, 0.0)
    segs = []                       # (t0, t1, p0, p1, is_hold)
    hold_errors = []                # 每个 hold 段的最大稳定误差
    t0 = 0
    p_prev = np.array(p_start, dtype=float)
    for wp in waypoints:
        segs.append((t0, t0 + move_ticks, p_prev.copy(), wp.copy(), False))
        t0 += move_ticks
        segs.append((t0, t0 + hold_ticks, wp.copy(), wp.copy(), True))
        hold_errors.append(0.0)
        t0 += hold_ticks
        p_prev = wp.copy()
    total_ticks = t0

    # ---- 状态
    k = 0                           # 控制 tick 计数（从 0 开始）
    seg_i = 0
    clipped = {"q1": False, "q2": False, "q3": False}
    unreachable = 0
    reported = False
    q1 = q2 = q3 = 0.0          # 循环外兜底（循环至少跑一 tick）

    while robot.step():
        # 段推进（按 tick 计数而不是墙钟，保证可复现）
        while seg_i < len(segs) - 1 and k >= segs[seg_i][1]:
            seg_i += 1
        st0, st1, p0, p1, is_hold = segs[seg_i]
        u = 0.0 if st1 <= st0 else min(1.0, max(0.0, (k - st0) / (st1 - st0)))
        s = 0.5 - 0.5 * math.cos(math.pi * u)          # 余弦缓动
        cmd = p0 + s * (p1 - p0)

        q1, q2, q3, reachable = ik(cmd)
        if not reachable:
            unreachable += 1
        q1 = _clip(q1, rng["joint1"], "q1", clipped)
        q2 = _clip(q2, rng["joint2"], "q2", clipped)
        q3 = _clip(q3, rng["joint3"], "q3", clipped)
        robot.set(act["joint1"], q1)
        robot.set(act["joint2"], q2)
        robot.set(act["joint3"], q3)

        # ---- 用仿真实测校验 IK：body 位置 + 关节角
        bodies = robot.obs.get("bodies") or {}
        try:
            flange = np.asarray(bodies[_resolve(bodies, "flange")], dtype=float)
            tip_meas = flange + np.array([0.0, 0.0, -FLANGE])
        except (KeyError, ValueError):
            tip_meas = None
        err = float("inf") if tip_meas is None else float(
            np.linalg.norm(tip_meas - cmd))

        if is_hold and err != float("inf"):
            # hold 段序号 = seg_i // 2（move/hold 交替）
            hi_idx = seg_i // 2
            hold_errors[hi_idx] = max(hold_errors[hi_idx], err)

        if k % log_every == 0:
            js = {}
            try:
                js = robot.joint_state()
            except Exception:
                pass
            q_meas = []
            for short in ("joint1", "joint2", "joint3"):
                try:
                    q_meas.append(float(js[_resolve(js, short)]))
                except (KeyError, TypeError, ValueError):
                    q_meas.append(float("nan"))
            print(f"[tick {robot.tick:4d} | t={robot.time:6.2f}s] "
                  f"{'HOLD' if is_hold else 'MOVE'} 段{seg_i // 2}"
                  f"  cmd={_fmt(cmd)}  tip={_fmt(tip_meas) if tip_meas is not None else 'n/a'}"
                  f"  err={err * 1000:6.2f} mm")
            print(f"    IK  q=[{q1:+.4f} {q2:+.4f} {q3:+.4f}] rad"
                  f" ({math.degrees(q1):+7.2f}°, {math.degrees(q2):+7.2f}°)"
                  f"  实测 q=[{q_meas[0]:+.4f} {q_meas[1]:+.4f} {q_meas[2]:+.4f}]")

        k += 1
        if k >= total_ticks and not reported:
            reported = True
            worst = max(hold_errors)
            ok = (worst < 0.010 and unreachable == 0
                  and not any(clipped.values()))
            detail = (f"路点 {len(waypoints)} 个，最大稳定误差 "
                      f"{worst * 1000:.2f} mm；不可达点 {unreachable} 个"
                      f"{'；有控量夹限位' if any(clipped.values()) else ''}")
            print("-" * 74)
            print(f"验收: {'通过' if ok else '未通过'} —— {detail}")
            print("-" * 74)
            robot.report_result(ok, detail)

    print("=" * 74)
    for i, e in enumerate(hold_errors):
        print(f"  路点 {i} {_fmt(waypoints[i])}  最大稳定误差 {e * 1000:6.2f} mm")
    print(f"  最终 Q = "
          f"[{math.degrees(q1):+7.2f}°, {math.degrees(q2):+7.2f}°]  "
          f"q3={q3 * 1000:+6.2f} mm")
    print("=" * 74)
    robot.close()


if __name__ == "__main__":
    main()
