"""任务评估与观测组装（纯函数层）。

内核 Simulation 与 RL 环境（rl_env.py）共用这里——两个消费方、
一份语义：success/reward 的定义、qpos/qvel/sensors 的组装规则只允许有
一处实现（两份实现迟早漂移，reward 是任务语义）。
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from simovia.world.builder import ResolvedTask, RobotIndex

# 关节类型 -> qpos/qvel 占用宽度（hinge/slide 等默认 1）。
# qvel 宽度与 qpos 不同：free 关节 qpos 7（位置3+四元数4）但 qvel 6（角速度3+线速度3），
# ball 关节 qpos 4（四元数）但 qvel 3（角速度）——此前共用一张表会让带命名
# 自由关节的机器人（spot/g1 等）观测 qvel 错位一位。
_QPOS_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): 7,
               int(mujoco.mjtJoint.mjJNT_BALL): 4}
_QVEL_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): 6,
               int(mujoco.mjtJoint.mjJNT_BALL): 3}


def obs_layout(model: mujoco.MjModel, ridx: RobotIndex) -> dict[str, tuple[int, int, int]]:
    """观测拼接布局：{关节名: (qpos 偏移, qvel 偏移, qpos 宽度)}。

    robot_obs 按此布局拼接；zoo demo 生成器注入控制器用的偏移也取自这里——
    一份布局，两个消费方，杜绝错位。
    """
    qp = qv = 0
    out = {}
    for jname, jid in ridx.joints.items():
        w = _QPOS_WIDTH.get(int(model.jnt_type[jid]), 1)
        out[jname] = (qp, qv, w)
        qp += w
        qv += _QVEL_WIDTH.get(int(model.jnt_type[jid]), 1)
    return out


def joint_width(model: mujoco.MjModel, ridx: RobotIndex) -> tuple[int, int]:
    """单机器人 qpos/qvel 拼接宽度（观测空间构造用）。"""
    qpos_w = qvel_w = 0
    for jid in ridx.joints.values():
        qpos_w += _QPOS_WIDTH.get(int(model.jnt_type[jid]), 1)
        qvel_w += _QVEL_WIDTH.get(int(model.jnt_type[jid]), 1)
    return qpos_w, qvel_w


def root_freejoint_qadr(model: mujoco.MjModel, ridx: RobotIndex) -> int | None:
    """浮动基座 root body 的 free joint qpos 地址；固定基座机器人返回 None。

    关键：menagerie 四足/人形用匿名 <freejoint/>（j.name 为空），
    _child_joint_names 只收命名关节，因此 ridx.joints 里没有它——
    躯干 xyz+quat 不会进 qpos 观测，GUI 渲染端的浮动机器人就被锁在
    出生位（四足"悬空奔跑"的根因）。这里按"world 直接子 body 上的
    free 关节"单独定位。
    """
    for bid in ridx.bodies.values():
        if int(model.body_parentid[bid]) != 0:
            continue
        n = int(model.body_jntnum[bid])
        for k in range(n):
            jid = int(model.body_jntadr[bid]) + k
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[jid])
    return None


def robot_obs(ridx: RobotIndex, model: mujoco.MjModel,
              data: mujoco.MjData) -> tuple[list[float], list[float]]:
    """单机器人 qpos/qvel（按 RobotIndex.joints 声明顺序拼接）。"""
    lay = obs_layout(model, ridx)
    qpos, qvel = [], []
    for jname, jid in ridx.joints.items():
        w = lay[jname][2]
        vw = _QVEL_WIDTH.get(int(model.jnt_type[jid]), 1)
        adr = model.jnt_qposadr[jid]
        qpos += list(map(float, data.qpos[adr:adr + w]))
        vadr = model.jnt_dofadr[jid]
        qvel += list(map(float, data.qvel[vadr:vadr + vw]))
    return qpos, qvel


def task_success(task: ResolvedTask, data: mujoco.MjData) -> bool | None:
    """终态任务判定：成功 body 的 xpos 落在 zone 内。无任务返回 None。"""
    if task.success_body is None:
        return None
    p = data.xpos[task.success_body]
    lo = np.asarray(task.success_min)
    hi = np.asarray(task.success_max)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def task_reward(task: ResolvedTask, model: mujoco.MjModel,
                data: mujoco.MjData) -> float:
    """终态奖励值（分项求和）。"""
    total = 0.0
    for kind, ia, ib, w in task.rewards:
        if kind == "distance":
            total += w * float(np.linalg.norm(
                data.xpos[ia] - data.xpos[ib]))
        elif kind == "zone":
            p = data.xpos[ia]
            lo, hi = np.asarray(ib[0]), np.asarray(ib[1])
            total += w * float(np.all(p >= lo) and np.all(p <= hi))
        elif kind == "contact":
            # a/b 是 body id，contact 里是 geom id：经 geom_bodyid 对齐
            pair = {ia, ib}
            hit = any(
                {model.geom_bodyid[data.contact[i].geom1],
                 model.geom_bodyid[data.contact[i].geom2]}
                == pair
                for i in range(data.ncon))
            total += w * float(hit)
    return total


def lidar_scan(model: mujoco.MjModel, data: mujoco.MjData,
               six: "SensorIndex") -> list[float]:
    """雷达一次全束扫描：内核 mj_ray 射线投射，返回 rays 个距离（米）。

    - 原点 = 挂载 body 世界位姿 + pos 偏移；扫描平面 = body xy 平面，
      0° 沿 body +x，逆时针为正（真实 2D 雷达习惯，body 随机器人转向）
    - 实测语义（mujoco 4.0）：flg_static=True 才含静态墙体，geomgroup=None
      含全部 group（视觉机壳也在内），起点位于几何内部时命中内表面
      ——因此自体几何用「命中后前移 1mm 续投」迭代穿过（机壳外环、
      底盘/轮子遮挡都会被跳过），最多跳 16 个
    - 无回波或超出 range 返回 -1.0（MuJoCo rangefinder 惯例，真实雷达的 inf）
    - 只读派生量：不调 mj_forward、不写 data，物理轨迹零影响
      （bit 级守护见 tests/test_lidar.py）
    """
    li = six.lidar
    bid = int(li["body"])
    rot = data.xmat[bid].reshape(3, 3)
    origin = (np.asarray(data.xpos[bid], dtype=float)
              + rot @ np.asarray(li["pos"], dtype=float))
    own = li["robot_bodies"]
    rng = float(li["range"])
    gid = np.zeros(1, dtype=np.int32)
    out: list[float] = []
    for ang in li["angles"]:
        vec = rot @ np.array([math.cos(ang), math.sin(ang), 0.0])
        base, d = 0.0, -1.0
        for _ in range(16):
            dist = mujoco.mj_ray(model, data, origin + base * vec, vec,
                                 None, True, -1, gid)
            if dist < 0.0 or base + dist > rng:
                break                      # 无回波 / 超量程
            if int(model.geom_bodyid[gid[0]]) not in own:
                d = base + dist            # 命中环境
                break
            base += dist + 1e-3            # 命中自体：穿过外壳续投
        out.append(float(d))
    return out


def lidar_pose(data: mujoco.MjData,
               six: "SensorIndex") -> tuple[float, float, float]:
    """雷达挂点世界坐标 + 扫描平面航向（kernel/mapping.py 的建图输入）。

    yaw = 挂载 body 的 +x 轴在世界 xy 平面上的朝向（2D SLAM 平面假设，
    与 lidar_scan 的 0° 沿 body +x 约定一致）。只读不写 data。
    """
    li = six.lidar
    bid = int(li["body"])
    rot = data.xmat[bid].reshape(3, 3)
    origin = (np.asarray(data.xpos[bid], dtype=float)
              + rot @ np.asarray(li["pos"], dtype=float))
    return float(origin[0]), float(origin[1]), float(math.atan2(
        float(rot[1, 0]), float(rot[0, 0])))


def sensors_obs(ridx: RobotIndex, model: mujoco.MjModel,
                data: mujoco.MjData) -> dict:
    """单机器人全部传感器读数 {名: 标量 or 列表}。

    dim=1 出标量（encoder/joint_torque/tactile），多维出列表
    （imu 按 gyro+accel 两个 part 顺序拼接成 6 维）。
    lidar 按 SensorIndex.lidar 配置逐束射线计算（parts 为空、dim=rays）。
    """
    sensors = {}
    for sname, six in ridx.sensors.items():
        if six.lidar is not None:
            sensors[sname] = lidar_scan(model, data, six)
            continue
        vals = []
        for adr, dim in six.parts:
            vals += list(map(float, data.sensordata[adr:adr + dim]))
        sensors[sname] = vals[0] if six.dim == 1 else vals
    return sensors
