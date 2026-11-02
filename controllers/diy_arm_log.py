"""教学控制器：diy_arm 三段臂 —— 实时打印每个关节与连杆的旋转矩阵 / 位姿。

配合 worlds/diy_arm.yaml 使用：每 log_every 个控制 tick 打印一次

    [tick   50 | t=3.20s]
      joint turret   q=+0.312 rad (+17.88°)  目标=+0.298 rad
      turret
        R_world =
          [+0.000 -1.000 +0.000]
          [+1.000 +0.000 +0.000]
          [+0.000 +0.000 +1.000]
        p_world = [ +0.000 +0.000 +0.100 ]   仿真 = [ +0.000 +0.000 +0.100 ]

连杆树（parent / pos / rpy / axis）与 YAML 一一对应；控制器进程拿不到
模型的旋转数据（观测只下发 body 位置），所以这里的重心是亲手算一遍
正运动学，正好把公式和仿真位置对账：

    R_rel    = Rz(yaw) · Ry(pitch) · Rx(roll)      # 只由 rpy 决定
    R_joint  = 绕关节轴 axis 转 q（罗德里格斯公式）
    R_world(child) = R_world(parent) · R_rel(child) · R_joint(child)
    p_world(child) = p_world(parent) + R_world(parent) · p_rel(child)

执行器目标沿用 random_arm 的随机漫步，让关节真的动起来；日志节流由
世界 YAML 的 args.log_every 控制（默认 50 tick ≈ 3.2s）。
"""
import functools
import math
import random

import numpy as np

from agentsim.controller import Robot

# 内核在 run 收尾时会 terminate 控制器进程；标准输出是块缓冲，被 SIGTERM
# 打断时缓冲区会丢。这里逐行 flush，日志才能完整转到内核 stderr。
print = functools.partial(print, flush=True)

# 与 worlds/diy_arm.yaml 的 links 顺序/数值完全一致（前序遍历，父在前）
LINKS = [
    {"name": "base",   "parent": None,     "pos": [0.00, 0.0, 0.00],
     "rpy": [0, 0, 0],   "axis": None},
    {"name": "turret", "parent": "base",   "pos": [0.00, 0.0, 0.10],
     "rpy": [0, 0, 90],  "axis": [0, 0, 1]},
    {"name": "upper",  "parent": "turret", "pos": [0.16, 0.0, 0.00],
     "rpy": [0, -90, 0], "axis": [0, 1, 0]},
    {"name": "wrist",  "parent": "upper",  "pos": [0.22, 0.0, 0.00],
     "rpy": [90, 0, 0],  "axis": [1, 0, 0]},
]


# ------------------------------------------------------------------ 旋转工具

def rpy_to_matrix(rpy_deg):
    """R = Rz(yaw) · Ry(pitch) · Rx(roll)，与 MuJoCo euler xyz 一致。"""
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, q):
    """罗德里格斯：绕单位轴 axis 转 q 弧度的 3×3 旋转矩阵。"""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * (k @ k)


def forward_kinematics(joint_q):
    """按连杆树算每个连杆的 R_rel / R_joint / R_world / p_world。"""
    frames = {}
    for link in LINKS:
        r_rel = rpy_to_matrix(link["rpy"])
        p_rel = np.array(link["pos"], dtype=float)
        if link["parent"] is None:
            r_joint = np.eye(3)
            r_world, p_world = r_rel.copy(), p_rel
        else:
            parent = frames[link["parent"]]
            q = joint_q.get(link["name"], 0.0)
            r_joint = (axis_angle_matrix(link["axis"], q)
                       if link["axis"] is not None else np.eye(3))
            r_world = parent["r_world"] @ r_rel @ r_joint
            p_world = parent["p_world"] + parent["r_world"] @ p_rel
        frames[link["name"]] = {"r_rel": r_rel, "r_joint": r_joint,
                                "r_world": r_world, "p_world": p_world}
    return frames


# ------------------------------------------------------------------ 打印

def _fmt_mat(m):
    return "\n".join("        [" + " ".join(f"{v:+7.3f}" for v in row) + " ]"
                     for row in m)


def _fmt_vec(v):
    return "[" + " ".join(f"{x:+7.3f}" for x in v) + " ]"


def print_static():
    """启动时打一遍静态 R_rel（rpy 决定，不随关节角变化）。"""
    print("=" * 72)
    print("diy_arm 连杆树（R = Rz(yaw)·Ry(pitch)·Rx(roll)，rpy 单位为度）")
    for link in LINKS:
        tag = f"  ← parent: {link['parent']}" if link["parent"] else "  ← 根连杆"
        print(f"\n  {link['name']:<7}{tag}  rpy={link['rpy']}")
        print("    R_rel =")
        print(_fmt_mat(rpy_to_matrix(link["rpy"])))
    print("\n每 log_every 个 tick 打印一次实时 R_world 与 p_world "
          "（并与仿真 body 位置对比）")
    print("=" * 72)


def print_state(tick, t, joint_q, ctrl, frames, bodies):
    print(f"\n[tick {tick:5d} | t={t:6.2f}s]")
    for link in LINKS:
        name = link["name"]
        fr = frames[name]
        if link["axis"] is not None:
            q = joint_q.get(name, 0.0)
            print(f"  joint {name:<7} q={q:+.3f} rad ({math.degrees(q):+7.2f}°)"
                  f"   目标={ctrl.get(name, 0.0):+.3f} rad")
        print(f"  {name}  R_world =")
        print(_fmt_mat(fr["r_world"]))
        sim = bodies.get(name)
        sim_txt = _fmt_vec(sim) if sim is not None else "（无观测）"
        print(f"    p_world = {_fmt_vec(fr['p_world'])}   仿真 = {sim_txt}")


# ------------------------------------------------------------------ 主循环

def main():
    robot = Robot()
    log_every = max(1, int(robot.args.get("log_every", 50)))
    speed = float(robot.args.get("speed", 0.15))

    print_static()
    print(f"controller: robot={robot.name} actuators={robot.actuators} "
          f"log_every={log_every} speed={speed}")

    # 目标初始取范围中点（本臂都是 0），每次 step 后随机小步游走
    target = {n: sum(robot.range(n)) / 2 for n in robot.actuators}

    while robot.step():
        for name in robot.actuators:
            lo, hi = robot.range(name)
            target[name] = min(hi, max(lo, target[name]
                                       + random.uniform(-speed, speed)))
            robot.set(name, target[name])

        if robot.tick % log_every == 0:
            joint_q = dict(zip(robot.obs.get("joint_names", []),
                               robot.obs.get("qpos", [])))
            frames = forward_kinematics(joint_q)
            print_state(robot.tick, robot.time, joint_q, target,
                        frames, robot.obs.get("bodies", {}))

    robot.close()


if __name__ == "__main__":
    main()
