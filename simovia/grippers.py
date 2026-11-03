"""夹爪库：末端执行器（gripper/end-effector）注册表与装配解析。

与 zoo.py（机器人库）同构，但夹爪不是独立世界实体——它通过 MjSpec 的
site-attach 挂到某台机器人腕部的挂点 site（如 menagerie 机械臂标配的
attachment_site），其关节/执行器/tendon 经嵌套 attach 后全名变为
"<robot>/gripper/..."，自动并入该机器人的设备索引，控制器直接
robot.set("gripper/gripper_act", v) 开合。

mount 标定（见 models/grippers/onrobot_2fg7.xml 顶部注释）：
夹爪 root 在自包含 XML 中归零，相对挂点 site 的安装变换（贴合量 +
轴向翻转）全部记录在下表 mounts 里。关键事实：base STL 经 mesh 内置
旋转后，其后法兰对接面在 gripper_base body 系 +y=0.1094 处、手指沿
body -y 伸出（开合沿 ±x）；四台 menagerie 机械臂挂点的工具轴均为
site +z（UR 的 site y 沿腕部壳体、非法兰法向）。故安装朝向统一用
Rot_x(-90°)（wxyz=[0.7071,-0.7071,0,0]）：手指转向 site +z（法兰
法向）、base 后端面转向 site -z 贴合机器人法兰；x=y=0 保证两法兰
同轴。z 分量逐臂按"法兰面在 site 系的 z + 0.1094"几何标定
（ur5e/xarm7/panda 法兰面≈0，ur10e=0.017），不可共用一值。旧值
误写 Rot_x(180°) 且 z=0.1494、y=-0.028，夹爪被转到 site ±y 方向
垂在腕部壳体下方、并与法兰留有约 9cm 间隙。

新增夹爪沿用同一约定，但模型内部已把"手指伸出方向"摆到 body +z、
把法兰对接面摆到 body 的固定 z（见各 XML 顶部注释），所以 mount 的
quat 都是单位四元数 [1,0,0,0]，只需按臂补 z：
- robotiq_2f85：后端面 body z=-0.0070（base_mount 包围盒），z=法兰+0.0070
- umi_gripper：中央后端面 body z=+0.0031（外侧支耳在 z=0，半径>9cm），
  z=法兰-0.0031
- leap_hand：腕部对接面已归一化到 body z=0，z=法兰
法兰面实测值由 2FG7 标定反推：ur5e/xarm7=0、ur10e=+0.017、panda≈0。

多执行器夹爪（灵巧手）接口：单执行器条目继续用 actuator/open/close；
若条目没有 actuator，则提供 actuators（夹爪内短名列表）与 poses
（{"open": [...], "close": [...]}，顺序同 actuators）。本模块只做数据
描述，manifest 消费者按两种形式分别展开，2FG7 行为不变。
"""
from __future__ import annotations

import pathlib

from simovia.errors import WorldError

ROOT = pathlib.Path(__file__).resolve().parents[1]   # 仓库根（含 models/）
GRIPPER_DIR = ROOT / "models" / "grippers"

# 内层 attach 的固定前缀：夹爪 body/joint/actuator 去掉机器人前缀后都以
# "gripper/" 开头，避免与机器人本体同名设备冲突
GRIPPER_PREFIX = "gripper/"

#: 2FG7 安装朝向（site 局部系，wxyz）：Rot_x(-90°)，把沿 body -y 伸出的
#: 手指转到 site +z（法兰法向），四臂挂点轴约定一致，共用此朝向
_QUAT_2FG7 = [0.7071067811865476, -0.7071067811865475, 0.0, 0.0]

#: 2FG7 逐臂安装位姿（site 局部系；pos 的 z = 该臂法兰面 z + base 后端面
#: 0.1094，几何标定，见模块 docstring；x=y=0 保持两法兰同轴）
_MOUNT_UR5E = {"pos": [0.0, 0.0, 0.1094], "quat": _QUAT_2FG7}
_MOUNT_UR10E = {"pos": [0.0, 0.0, 0.1264], "quat": _QUAT_2FG7}
_MOUNT_XARM7 = {"pos": [0.0, 0.0, 0.1094], "quat": _QUAT_2FG7}
_MOUNT_PANDA = {"pos": [0.0, 0.0, 0.1092], "quat": _QUAT_2FG7}

#: 新增夹爪已把手指摆到 body +z、法兰对接面摆到 body z=0（2F85 为 -0.007），
#: 朝向均为单位四元数；z = 法兰面（ur5e/xarm7/panda≈0，ur10e=+0.017）
_QUAT_ID = [1.0, 0.0, 0.0, 0.0]

# Robotiq 2F85：base_mount 包围盒后端面 body z=-0.0070，故 root 需前移 7mm
_MOUNT_2F85 = {
    "ur5e.xml": {"pos": [0.0, 0.0, 0.0070], "quat": _QUAT_ID},
    "ur10e.xml": {"pos": [0.0, 0.0, 0.0240], "quat": _QUAT_ID},
    "xarm7_nohand.xml": {"pos": [0.0, 0.0, 0.0070], "quat": _QUAT_ID},
    "panda_nohand.xml": {"pos": [0.0, 0.0, 0.0070], "quat": _QUAT_ID},
}
# UMI：中央后端面在 root z=+0.0031，外侧支耳（半径>9cm）在 z=0
_MOUNT_UMI = {
    "ur5e.xml": {"pos": [0.0, 0.0, -0.0031], "quat": _QUAT_ID},
    "ur10e.xml": {"pos": [0.0, 0.0, 0.0139], "quat": _QUAT_ID},
    "xarm7_nohand.xml": {"pos": [0.0, 0.0, -0.0031], "quat": _QUAT_ID},
    "panda_nohand.xml": {"pos": [0.0, 0.0, -0.0031], "quat": _QUAT_ID},
}
# LEAP：腕部对接面已归一化到 root z=0
_MOUNT_LEAP = {
    "ur5e.xml": {"pos": [0.0, 0.0, 0.0], "quat": _QUAT_ID},
    "ur10e.xml": {"pos": [0.0, 0.0, 0.0170], "quat": _QUAT_ID},
    "xarm7_nohand.xml": {"pos": [0.0, 0.0, 0.0], "quat": _QUAT_ID},
    "panda_nohand.xml": {"pos": [0.0, 0.0, 0.0], "quat": _QUAT_ID},
}

#: 未逐臂标定时的通用安装变换（builder 在机器人末端自动生成挂点后使用）。
#: 挂点约定：origin 在法兰对接面、+z 沿工具轴向外伸出（menagerie 惯例）。
#: pos.z = 夹爪后端面到 root 的距离（把后端面贴到法兰面上），朝向同逐臂值。
_MOUNT_GENERIC = {
    "onrobot_2fg7": {"pos": [0.0, 0.0, 0.1094], "quat": _QUAT_2FG7},
    "robotiq_2f85": {"pos": [0.0, 0.0, 0.0070], "quat": _QUAT_ID},
    "umi_gripper": {"pos": [0.0, 0.0, -0.0031], "quat": _QUAT_ID},
    "leap_hand": {"pos": [0.0, 0.0, 0.0], "quat": _QUAT_ID},
}

# LEAP 16 执行器顺序（与 XML actuator 段一致）：三指各 mcp/rot/pip/dip，
# 拇指 cmc/axl/mcp/ipl；open 全零伸展，close 为包握姿态（正角弯曲）
_LEAP_ACTUATORS = (
    "if_mcp_act", "if_rot_act", "if_pip_act", "if_dip_act",
    "mf_mcp_act", "mf_rot_act", "mf_pip_act", "mf_dip_act",
    "rf_mcp_act", "rf_rot_act", "rf_pip_act", "rf_dip_act",
    "th_cmc_act", "th_axl_act", "th_mcp_act", "th_ipl_act",
)
_LEAP_JOINTS = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
_LEAP_POSES = {
    "open": [0.0] * 16,
    "close": [1.2, 0.0, 1.2, 1.0,
              1.2, 0.0, 1.2, 1.0,
              1.2, 0.0, 1.2, 1.0,
              0.6, 0.0, 1.0, 1.0],
}

GRIPPERS: dict[str, dict] = {
    "onrobot_2fg7": {
        "title": "OnRobot 2FG7 平行夹爪",
        "cat": "平行夹爪",
        "xml": "onrobot_2fg7.xml",
        "site": "attachment_site",          # 缺省挂点（可在 YAML 覆盖）
        "actuator": "gripper_act",           # 开合执行器（夹爪内短名）
        "open": 0.038,                       # ctrl=0.038 全开（内间隙 64.8mm）
        "close": 0.0,                        # ctrl=0 闭合（内间隙 36mm）
        "joints": ("left_slide", "right_slide"),
        # key = 机器人模型文件 basename；未列出的机器人报清晰错误而非静默错挂
        "mounts": {
            "ur5e.xml": _MOUNT_UR5E,
            "ur10e.xml": _MOUNT_UR10E,
            "xarm7_nohand.xml": _MOUNT_XARM7,
            "panda_nohand.xml": _MOUNT_PANDA,
        },
    },
    "robotiq_2f85": {
        "title": "Robotiq 2F-85 自适应夹爪",
        "cat": "平行夹爪",
        "xml": "robotiq_2f85/robotiq_2f85.xml",
        "site": "attachment_site",
        "actuator": "fingers_actuator",      # 夹爪内短名
        "open": 0.0,                        # ctrl=0 全开（手指外张）
        "close": 255.0,                     # ctrl=255 闭合（原模型 0~255 标度）
        "joints": ("right_driver_joint", "right_coupler_joint",
                   "right_spring_link_joint", "right_follower_joint",
                   "left_driver_joint", "left_coupler_joint",
                   "left_spring_link_joint", "left_follower_joint"),
        "mounts": _MOUNT_2F85,
    },
    "umi_gripper": {
        "title": "UMI Gripper 平行夹爪",
        "cat": "平行夹爪",
        "xml": "umi_gripper/umi_gripper.xml",
        "site": "attachment_site",
        "actuator": "fingers_actuator",
        "open": 0.0,                        # ctrl=0 两指张开
        "close": 0.05,                      # ctrl=0.05 两指闭合
        "joints": ("left_finger_joint", "right_finger_joint"),
        "mounts": _MOUNT_UMI,
    },
    "leap_hand": {
        "title": "LEAP Hand 灵巧手（左手，16 自由度）",
        "cat": "灵巧手",
        "xml": "leap_hand/leap_hand.xml",
        "site": "attachment_site",
        # 16 自由度无单一开合执行器：用 actuators + poses 描述开合目标
        "actuators": _LEAP_ACTUATORS,
        "poses": _LEAP_POSES,
        "joints": _LEAP_JOINTS,
        "mounts": _MOUNT_LEAP,
    },
}


def ensure_gripper(key: str) -> pathlib.Path:
    """返回夹爪入口 MJCF 的绝对路径（夹爪资产随仓库，无需下载）。"""
    if key not in GRIPPERS:
        raise WorldError(
            f"夹爪库没有 '{key}'，可用: {sorted(GRIPPERS)}")
    path = GRIPPER_DIR / GRIPPERS[key]["xml"]
    if not path.exists():
        raise WorldError(f"夹爪 '{key}' 的模型文件缺失: {path}")
    return path


def gripper_info(key: str) -> dict:
    if key not in GRIPPERS:
        raise WorldError(f"夹爪库没有 '{key}'，可用: {sorted(GRIPPERS)}")
    return GRIPPERS[key]


def mount_for(key: str, robot_model: str,
              override_pos=None, override_quat=None,
              allow_generic: bool = False) -> dict:
    """查夹爪挂到指定机器人模型上的安装变换。

    robot_model：机器人 MJCF 路径（用 basename 匹配 mount 表）。
    override_*：YAML gripper.pos/quat 显式覆盖（高级用户/非标准臂），
    覆盖时绕过 mount 表，于是任意带挂点 site 的机器人也能手动挂。
    allow_generic：模型不在标定表时回退到通用变换（挂点约定为
    "origin 在法兰面、+z 沿工具轴"，builder 自动生成的腕部挂点满足
    该约定）。GUI/世界装配默认开启，手工调用默认关闭以保留报错。
    """
    info = gripper_info(key)
    if override_pos is not None or override_quat is not None:
        return {"pos": list(override_pos) if override_pos is not None
                else [0.0, 0.0, 0.0],
                "quat": list(override_quat) if override_quat is not None
                else [1.0, 0.0, 0.0, 0.0]}
    # basename 匹配：正/反斜杠都切——Windows 用户写的 C:\robots\x.xml 在
    # Linux 上、Linux 的 /robots/x.xml 在 Windows 上都能正确命中 mount 表
    base = str(robot_model).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if base in info["mounts"]:
        return dict(info["mounts"][base])
    if allow_generic and key in _MOUNT_GENERIC:
        return dict(_MOUNT_GENERIC[key])
    raise WorldError(
        f"夹爪 '{key}' 没有为机器人模型 '{base}' 标定安装位姿，"
        f"已标定: {sorted(info['mounts'])}。可在世界 YAML 的 gripper 下"
        f"显式写 pos/quat 手动指定，或在 grippers.py mount 表补一条。")
