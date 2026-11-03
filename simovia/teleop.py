"""遥操作：键盘速度指令 -> 阻尼最小二乘 IK -> 关节目标（纯逻辑，无 Qt）。

链路（GUI 边看相机画面边操作）：

    GUI 按键状态 --teleop_cmd--> 会话（线程安全命令位）
        内核 run 线程每个控制 tick：积分 TCP 目标 -> IK -> data.ctrl

- 只支持固定基座机械臂（>= 6 个 arm 关节 + 末端 TCP site）；IK 用
  mj_jacSite 的 6D 雅可比做阻尼最小二乘，姿态目标跟随初始姿态（键位
  只做平移 + 绕 z 偏航，符合"看着相机怼上去抓"的直觉）。
- 目标位姿按 tick 积分（速度 × 控制周期），不是一步跳到位——位置伺服
  跟得平滑，也不会一脚把方块踢飞。
- 夹爪开合归一化为 0=闭合 / 1=张开，按 gripper 库的 open/close 映射；
  无夹爪时忽略 grip 指令。
"""
from __future__ import annotations

import threading

import mujoco
import numpy as np

from simovia.errors import WorldError
from simovia.grippers import GRIPPERS

# 缺省速度（按住 Shift 乘倍率）
LINEAR_SPEED = 0.30      # m/s
ANGULAR_SPEED = 1.2      # rad/s
FAST_MULTIPLIER = 2.5
GRIP_SPEED = 6.0         # 开合行程 / s

_IK_ITERS = 60
_IK_DAMPING = 0.02
_IK_MAX_STEP = 0.2       # 单次迭代关节增量上限（rad）
_REANCHOR_POS_ERR = 0.05  # 位置残差超过此值视为不可达：目标回到当前位姿


def _rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 旋转向量（轴*角）。"""
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(ang))
    return axis * ang


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ArmTeleop:
    """单台固定基座机械臂的笛卡尔遥操作控制器。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, index,
                 spec, robot_name: str, site: str | None = None):
        self.model = model
        self.robot_name = robot_name
        ridx = index.robots[robot_name]
        self._ridx = ridx
        self.ctrl_period = int(spec.ctrl_period)

        # ---- 机械臂关节（排除 gripper/ 前缀的夹爪关节）
        self._joints = [(name, jid) for name, jid in ridx.joints.items()
                        if not name.startswith("gripper/")
                        and int(model.jnt_type[jid])
                        in (int(mujoco.mjtJoint.mjJNT_HINGE),
                            int(mujoco.mjtJoint.mjJNT_SLIDE))]
        if len(self._joints) < 6:
            raise WorldError(
                f"机器人 '{robot_name}' 只有 {len(self._joints)} 个机械臂关节，"
                "遥操作 IK 至少需要 6 个（固定基座机械臂）")
        self._qadr = [int(model.jnt_qposadr[jid]) for _, jid in self._joints]
        self._dofadr = [int(model.jnt_dofadr[jid]) for _, jid in self._joints]

        # ---- 关节 -> 执行器（按 actuator_trnid 反查，不依赖命名）
        act_by_joint = {}
        for aid in ridx.actuators.values():
            if int(model.actuator_trntype[aid]) != \
                    int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            act_by_joint[int(model.actuator_trnid[aid, 0])] = aid
        missing = [name for name, jid in self._joints if jid not in act_by_joint]
        if missing:
            raise WorldError(
                f"机器人 '{robot_name}' 的关节 {missing} 没有对应执行器，"
                "无法做位置伺服遥操作")
        self._arm_acts = [int(act_by_joint[jid]) for _, jid in self._joints]

        # ---- TCP site：显式 site > gripper_tcp > attachment_site
        self.site_id = self._resolve_site(robot_name, site)
        self.site_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, self.site_id)

        # ---- 夹爪（可选）：开合执行器 + open/close 目标值
        self._grip_aid, self._grip_open, self._grip_close = \
            self._resolve_gripper(spec, robot_name, ridx)

        # ---- 状态：目标位姿从当前 TCP 位姿起步
        mujoco.mj_kinematics(model, data)
        self._base_rot = np.array(data.xmat[self._root_body(ridx)],
                                  dtype=float).reshape(3, 3)
        self._target_pos = np.array(data.site_xpos[self.site_id], dtype=float)
        self._target_mat = np.array(data.site_xmat[self.site_id],
                                    dtype=float).reshape(3, 3).copy()
        self._ik_data = mujoco.MjData(model)
        self._ik_data.qpos[:] = data.qpos
        self._ik_data.qvel[:] = 0.0
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        self._lock = threading.Lock()
        self._cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "wz": 0.0,
                     "fast": False, "grip": 1.0 if self._grip_aid else None}
        self._grip = self._cmd["grip"] if self._grip_aid is not None else None
        self.last_pos_error = 0.0

    # ------------------------------------------------------------ 查询

    def describe(self) -> dict:
        return {
            "robot": self.robot_name,
            "site": self.site_name,
            "joints": [name for name, _ in self._joints],
            "gripper_actuator": (mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._grip_aid)
                if self._grip_aid is not None else None),
            "linear_speed": LINEAR_SPEED,
            "angular_speed": ANGULAR_SPEED,
            "fast_multiplier": FAST_MULTIPLIER,
        }

    # ------------------------------------------------------------ 指令

    def set_command(self, cmd: dict) -> dict:
        """更新速度指令（GUI 线程调用；只碰锁保护的小字典）。"""
        with self._lock:
            for key in ("vx", "vy", "vz", "wz"):
                if key in cmd and cmd[key] is not None:
                    self._cmd[key] = float(np.clip(float(cmd[key]), -1.0, 1.0))
            if cmd.get("fast") is not None:
                self._cmd["fast"] = bool(cmd["fast"])
            if cmd.get("grip") is not None and self._grip_aid is not None:
                self._cmd["grip"] = float(np.clip(float(cmd["grip"]), 0.0, 1.0))
            return dict(self._cmd)

    def release(self) -> None:
        """清空速度（松开按键/窗口失焦），夹爪保持当前目标。"""
        with self._lock:
            for key in ("vx", "vy", "vz", "wz"):
                self._cmd[key] = 0.0
            self._cmd["fast"] = False

    # ------------------------------------------------------------ 每 tick

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """内核 run 线程每个控制 tick 调用：算 IK 并写 data.ctrl。"""
        with self._lock:
            cmd = dict(self._cmd)
        dt = model.opt.timestep * self.ctrl_period
        speed = LINEAR_SPEED * (FAST_MULTIPLIER if cmd["fast"] else 1.0)
        delta = self._base_rot @ np.array(
            [cmd["vx"], cmd["vy"], cmd["vz"]], dtype=float) * speed * dt
        if np.any(delta):
            self._target_pos += delta
        if abs(cmd["wz"]) > 1e-6:
            w = ANGULAR_SPEED * (FAST_MULTIPLIER if cmd["fast"] else 1.0) * dt
            self._target_mat = _rz(cmd["wz"] * w) @ self._target_mat

        q0 = np.array([data.qpos[a] for a in self._qadr], dtype=float)
        q, pos_err = self._solve(q0)
        self.last_pos_error = pos_err
        if pos_err > _REANCHOR_POS_ERR:
            # 目标不可达（撞工作空间边界/奇异）：目标回到当前位姿，避免越积越远
            self._target_pos = np.array(data.site_xpos[self.site_id],
                                        dtype=float)
            self._target_mat = np.array(data.site_xmat[self.site_id],
                                        dtype=float).reshape(3, 3).copy()
            q = q0

        data.ctrl[self._arm_acts] = q
        if self._grip_aid is not None and cmd["grip"] is not None:
            target = float(cmd["grip"])
            step = GRIP_SPEED * dt
            self._grip = float(np.clip(
                self._grip + np.clip(target - self._grip, -step, step), 0.0, 1.0))
            data.ctrl[self._grip_aid] = (self._grip_close +
                                         (self._grip_open - self._grip_close)
                                         * self._grip)

    # ------------------------------------------------------------ 内部

    def _solve(self, q0: np.ndarray):
        """从 q0 出发解 IK，返回 (q, 位置残差)。"""
        q = q0.copy()
        d = self._ik_data
        pos_err = 0.0
        for _ in range(_IK_ITERS):
            for k, adr in enumerate(self._qadr):
                d.qpos[adr] = q[k]
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos = d.site_xpos[self.site_id]
            mat = d.site_xmat[self.site_id].reshape(3, 3)
            ep = self._target_pos - pos
            er = _rotvec_from_matrix(self._target_mat @ mat.T)
            pos_err = float(np.linalg.norm(ep))
            if pos_err < 1e-4 and float(np.linalg.norm(er)) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr,
                              self.site_id)
            jac = np.vstack([self._jacp[:, self._dofadr],
                             self._jacr[:, self._dofadr]])
            err = np.concatenate([ep, 0.5 * er])
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + _IK_DAMPING ** 2 * np.eye(6), err)
            norm = float(np.linalg.norm(dq))
            if norm > _IK_MAX_STEP:
                dq *= _IK_MAX_STEP / norm
            q = q + dq
            for k, (_, jid) in enumerate(self._joints):
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    q[k] = float(np.clip(q[k], lo, hi))
        return q, pos_err

    def _resolve_site(self, robot_name: str, site: str | None) -> int:
        """site 解析：显式短名/全名 -> 机器人自己的 gripper_tcp/attachment_site。"""
        m = self.model
        candidates = []
        if site:
            candidates = [site, f"{robot_name}/{site}"]
        sid = -1
        for name in candidates:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                return sid
        # 自动：优先夹爪 TCP（指尖参考点），其次腕部挂点
        names = []
        for s in range(m.nsite):
            full = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s)
            if full and full.startswith(f"{robot_name}/"):
                names.append(full)
        for needle in ("gripper_tcp", "tcp", "attachment_site"):
            for full in names:
                if full.endswith(f"/{needle}") or full == needle:
                    return mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_SITE, full)
        raise WorldError(
            f"机器人 '{robot_name}' 没有可用于遥操作的 TCP site（可用: "
            f"{[n[len(robot_name) + 1:] for n in names]}）；"
            "请在 teleop_start 里用 site 参数指定")

    def _resolve_gripper(self, spec, robot_name, ridx):
        """夹爪开合执行器与 open/close 目标；没有夹爪返回 (None, None, None)。"""
        rd = next((r for r in spec.robots if r.name == robot_name), None)
        info = None
        if rd is not None and rd.gripper is not None:
            info = GRIPPERS.get(rd.gripper.model)
        if info is not None and "actuator" in info:
            aid = ridx.actuators.get(f"gripper/{info['actuator']}")
            if aid is not None:
                return int(aid), float(info["open"]), float(info["close"])
        # 兜底：找名字带 gripper/finger 的关节执行器，ctrlrange 高值当张开
        for aname, aid in ridx.actuators.items():
            if "gripper" in aname or "finger" in aname:
                lo, hi = self.model.actuator_ctrlrange[aid]
                return int(aid), float(hi), float(lo)
        return None, None, None

    def _root_body(self, ridx) -> int:
        """机器人根 body：parent 为世界(0) 的那个。"""
        for bid in ridx.bodies.values():
            if int(self.model.body_parentid[bid]) == 0:
                return int(bid)
        return int(next(iter(ridx.bodies.values())))


__all__ = ["ArmTeleop"]
