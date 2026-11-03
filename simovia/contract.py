"""策略接口契约：跨仿真器/训练/录制共用的观测-动作描述。

设计要点：
- 名字驱动：契约只保存关节/执行器/传感器的**名字与顺序**，不保存
  MuJoCo 裸 id——导出的数据拿到 Genesis、真机或另一个 MuJoCo 模型里
  仍可按名字对齐。
- 单一来源：关节顺序、传感器维度取自 kernel/task.py 的同一套布局语义，
  录制、训练、sim2sim 三处共用一份契约，避免各写一套导致错位。
- 可序列化：to_dict()/to_json() 可直接进数据集元信息，也便于人工比对
  两份契约（例如训练端与部署端的 joint_names 是否一致）。

观测布局（拼接顺序固定）：
    浮动基座：root_pos(3) + root_quat(4) + root_lin_vel(3) + root_ang_vel(3)
    固定基座：无 root 字段
    然后：joint_pos（按 joints 顺序）+ joint_vel（同序）+ 各传感器

动作布局：按 actuators 顺序逐个执行器一个标量（与内核 ctrl 语义一致）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import mujoco

from simovia.world.builder import RobotIndex
from simovia.world.schema import WorldSpec

# MuJoCo 枚举的整数常量（avoid enum 版本差异带来的比较开销）
_BIAS_NONE = int(mujoco.mjtBias.mjBIAS_NONE)
_BIAS_AFFINE = int(mujoco.mjtBias.mjBIAS_AFFINE)
_GAIN_FIXED = int(mujoco.mjtGain.mjGAIN_FIXED)
_JNT_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_JNT_BALL = int(mujoco.mjtJoint.mjJNT_BALL)

# free/ball 关节的 qpos 宽度；qvel 宽度单独一张表（free: 7/6, ball: 4/3）
_QPOS_WIDTH = {_JNT_FREE: 7, _JNT_BALL: 4}
_QVEL_WIDTH = {_JNT_FREE: 6, _JNT_BALL: 3}


@dataclass(frozen=True)
class JointContract:
    """一个命名关节（匿名 free joint 不计入，单独由 root_* 描述）。"""
    name: str
    type: str                 # hinge / slide / ball
    qpos_dim: int
    qvel_dim: int


@dataclass(frozen=True)
class ActuatorContract:
    """一个执行器的动作语义。"""
    name: str
    target_type: str          # joint / jointinparent / tendon / site ...
    mode: str                 # position / velocity / torque / general
    ctrl_min: float | None    # ctrllimited=false 时为 None（不做缩放）
    ctrl_max: float | None
    gear: float = 1.0
    kp: float | None = None   # position/velocity 执行器的刚度
    kd: float | None = None   # position/velocity 执行器的阻尼


@dataclass(frozen=True)
class SensorContract:
    """一个传感器的读数维度（imu 等多 part 已合并）。"""
    name: str
    kind: str
    dim: int


@dataclass(frozen=True)
class ObsField:
    """观测拼接中的一段：名字 + 起始偏移 + 宽度。"""
    name: str
    offset: int
    dim: int


@dataclass(frozen=True)
class RobotContract:
    """单机器人的完整观测/动作契约。"""
    robot: str
    timestep: float
    ctrl_period: int
    control_dt: float
    joints: tuple[JointContract, ...]
    actuators: tuple[ActuatorContract, ...]
    sensors: tuple[SensorContract, ...]
    obs_fields: tuple[ObsField, ...]
    floating_base: bool
    obs_dim: int
    action_dim: int

    # ------------------------------------------------------------ 构造

    @classmethod
    def from_world(cls, spec: WorldSpec, model: mujoco.MjModel,
                   ridx: RobotIndex) -> "RobotContract":
        """从已编译的世界模型推导契约（无手动配置，字段全部来自权威来源）。"""
        joints = []
        for jname, jid in ridx.joints.items():
            jtype = int(model.jnt_type[jid])
            joints.append(JointContract(
                name=jname,
                type=mujoco.mjtJoint(jtype).name.replace("mjJNT_", "").lower(),
                qpos_dim=_QPOS_WIDTH.get(jtype, 1),
                qvel_dim=_QVEL_WIDTH.get(jtype, 1),
            ))

        actuators = tuple(_actuator_contract(model, name, aid)
                          for name, aid in ridx.actuators.items())
        sensors = tuple(SensorContract(name=six.name, kind=six.kind,
                                       dim=six.dim)
                        for six in ridx.sensors.values())

        floating = root_joint_addresses(model, ridx) is not None
        fields: list[ObsField] = []
        offset = 0

        def _add(name: str, dim: int):
            nonlocal offset
            fields.append(ObsField(name=name, offset=offset, dim=dim))
            offset += dim

        if floating:
            _add("root_pos", 3)
            _add("root_quat", 4)
            _add("root_lin_vel", 3)
            _add("root_ang_vel", 3)
        _add("joint_pos", sum(j.qpos_dim for j in joints))
        _add("joint_vel", sum(j.qvel_dim for j in joints))
        for six in sensors:
            _add(f"sensor:{six.name}", six.dim)

        return cls(
            robot=ridx.name,
            timestep=float(spec.timestep),
            ctrl_period=int(spec.ctrl_period),
            control_dt=float(spec.timestep * spec.ctrl_period),
            joints=tuple(joints),
            actuators=actuators,
            sensors=sensors,
            obs_fields=tuple(fields),
            floating_base=floating,
            obs_dim=offset,
            action_dim=len(actuators),
        )

    # ------------------------------------------------------------ 导出

    def to_dict(self) -> dict:
        """可 JSON 序列化的字典（tuple -> list 由 asdict 处理）。"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "RobotContract":
        """从 to_dict()/meta.json 还原（用于 sim2sim 端校验）。"""
        return cls(
            robot=d["robot"],
            timestep=float(d["timestep"]),
            ctrl_period=int(d["ctrl_period"]),
            control_dt=float(d["control_dt"]),
            joints=tuple(JointContract(**j) for j in d["joints"]),
            actuators=tuple(ActuatorContract(**a) for a in d["actuators"]),
            sensors=tuple(SensorContract(**s) for s in d["sensors"]),
            obs_fields=tuple(ObsField(**f) for f in d["obs_fields"]),
            floating_base=bool(d["floating_base"]),
            obs_dim=int(d["obs_dim"]),
            action_dim=int(d["action_dim"]),
        )

    def digest(self) -> str:
        """契约指纹：两份契约不同（改名/换序/改频率）指纹就不同。"""
        canonical = json.dumps(self.to_dict(), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ 观测/动作读取

    def split_obs(self, obs: list[float] | tuple[float, ...]) -> dict[str, list[float]]:
        """把 flat 观测按 obs_fields 切回命名字段。"""
        values = list(obs)
        if len(values) != self.obs_dim:
            raise ValueError(
                f"契约要求 {self.obs_dim} 维观测，实际收到 {len(values)} 维")
        return {f.name: values[f.offset:f.offset + f.dim]
                for f in self.obs_fields}


def contracts_for_world(spec: WorldSpec, model: mujoco.MjModel,
                        index) -> dict[str, RobotContract]:
    """世界内全部机器人的契约（机器人名 -> 契约）。"""
    return {rd.name: RobotContract.from_world(spec, model, index.robots[rd.name])
            for rd in spec.robots}


# ---------------------------------------------------------------- 内部推导

def root_joint_addresses(model: mujoco.MjModel,
                         ridx: RobotIndex) -> tuple[int, int] | None:
    """浮动基座 free joint 的 (qpos 地址, qvel 地址)；固定基座返回 None。

    与 kernel/task.py:root_freejoint_qadr 同一识别规则：world 直接子 body
    上的匿名 free joint（menagerie 模型常不写名字）。
    """
    for bid in ridx.bodies.values():
        if int(model.body_parentid[bid]) != 0:
            continue
        for k in range(int(model.body_jntnum[bid])):
            jid = int(model.body_jntadr[bid]) + k
            if int(model.jnt_type[jid]) == _JNT_FREE:
                return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return None


def _actuator_contract(model: mujoco.MjModel, name: str,
                       aid: int) -> ActuatorContract:
    """按 MuJoCo 执行器参数推断动作模式。

    - biastype=affine 且 biasprm[1] != 0 -> position（kp=gainprm[0],
      kd=-biasprm[2]）
    - biastype=affine 且 biasprm[1] == 0 -> velocity
    - biastype=none 且 gainprm[0] == 1 -> torque（motor 直写）
    - 其余 -> general（自定义 gain/bias，调用方自行映射）
    """
    bias = int(model.actuator_biastype[aid])
    gain = int(model.actuator_gaintype[aid])
    gain0 = float(model.actuator_gainprm[aid][0])
    kp = kd = None
    if bias == _BIAS_AFFINE and gain == _GAIN_FIXED:
        kp = gain0
        kd = -float(model.actuator_biasprm[aid][2])
        mode = "position" if float(model.actuator_biasprm[aid][1]) != 0.0 \
            else "velocity"
    elif bias == _BIAS_NONE and gain == _GAIN_FIXED and abs(gain0 - 1.0) < 1e-12:
        mode = "torque"
    else:
        mode = "general"
    lo, hi = (float(x) for x in model.actuator_ctrlrange[aid])
    if not bool(model.actuator_ctrllimited[aid]):
        lo = hi = None
    trn = int(model.actuator_trntype[aid])
    return ActuatorContract(
        name=name,
        target_type=mujoco.mjtTrn(trn).name.replace("mjTRN_", "").lower(),
        mode=mode,
        ctrl_min=lo,
        ctrl_max=hi,
        gear=float(model.actuator_gear[aid][0]),
        kp=kp,
        kd=kd,
    )


__all__ = [
    "RobotContract", "JointContract", "ActuatorContract",
    "SensorContract", "ObsField", "contracts_for_world",
    "root_joint_addresses",
]
