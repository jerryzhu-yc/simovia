"""机器人连杆树描述（教学用）：links -> MjSpec。

这是一份"从零造机器人"的最小描述格式，目的是让初学者看清两件事：

1. **每个连杆（link）相对父连杆的位姿**：`pos` 平移 + `rpy` 欧拉角旋转。
   数学上就是齐次变换 T = [R | p]，其中 R = Rz(yaw)·Ry(pitch)·Rx(roll)。
2. **每个关节（joint）挂在哪、绕什么轴转**：关节原点就是连杆坐标系原点，
   轴在连杆坐标系里给出。

YAML 示例（角度全部用**度**，长度用米）::

    name: two_link
    links:
      - name: base                       # 根连杆（无 parent / 无 joint）
        geom: {type: cylinder, size: [0.06, 0.05], pos: [0, 0, 0.05]}
      - name: link1
        parent: base
        pos: [0, 0, 0.10]                # 相对 base 坐标系平移 10cm
        rpy: [0, 0, 0]                   # 相对 base 坐标系旋转（度）
        mass: 1.2
        joint: {type: revolute, axis: [0, 0, 1], range: [-170, 170]}
        geom: {type: box, size: [0.03, 0.03, 0.12], pos: [0, 0, 0.12]}
      - name: link2
        parent: link1
        pos: [0, 0, 0.24]
        rpy: [0, 90, 0]
        joint: {type: revolute, axis: [0, 0, 1], range: [-120, 120], kp: 80}

字段说明：

- ``links[].pos`` / ``rpy`` / ``quat``：相对父连杆的位姿；``rpy`` 与 ``quat``
  （wxyz）二选一。根连杆的位姿即它在世界（或 attach frame）里的位姿。
- ``joint.type``：``fixed``（默认）/ ``revolute``（转动）/ ``prismatic``（移动）
  / ``free``（浮动基座，只能给根连杆）/ ``ball``。
- ``joint.range``：revolute 用度，prismatic 用米；不给则不限位。
- ``joint.kp`` / ``kd``：位置执行器刚度与关节阻尼，默认 50 / 1；
  ``joint.actuator: false`` 表示不建执行器（纯被动关节）。
  **kd 落在关节阻尼（dof_damping）而不是伺服速度反馈上**：MuJoCo 的
  Euler 积分器只对关节阻尼做隐式积分，速度反馈在高刚度下会数值发散
  （实测 kp=50 时 Euler + biasprm kd 必然爆 NaN）。这样初学者用默认
  ``integrator: euler`` 的世界也不会炸。
- ``geom.type``：box（size=半边长）/ sphere（半径）/ cylinder（半径, 半长）
  / capsule（半径, 半长）/ ellipsoid（三半径）。``geom: false`` 表示无几何。

GUI 的「造机器人」面板就是在编辑这份结构，并实时显示每个连杆的 3×3
旋转矩阵 R 与世界位姿 T。
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from simovia.errors import WorldError
from simovia.world.schema import hex_to_rgba

#: 支持的几何类型 -> (MuJoCo 枚举, size 说明)
GEOM_TYPES = {
    "box": (mujoco.mjtGeom.mjGEOM_BOX, "半边长 [x, y, z]"),
    "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, "半径 [r]"),
    "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, "半径, 半长 [r, h]"),
    "capsule": (mujoco.mjtGeom.mjGEOM_CAPSULE, "半径, 半长 [r, h]"),
    "ellipsoid": (mujoco.mjtGeom.mjGEOM_ELLIPSOID, "三轴半径 [rx, ry, rz]"),
}

#: 关节类型别名 -> (MuJoCo 枚举, 是否需要执行器)
JOINT_TYPES = {
    "fixed": (None, False),
    "revolute": (mujoco.mjtJoint.mjJNT_HINGE, True),
    "hinge": (mujoco.mjtJoint.mjJNT_HINGE, True),
    "prismatic": (mujoco.mjtJoint.mjJNT_SLIDE, True),
    "slide": (mujoco.mjtJoint.mjJNT_SLIDE, True),
    "ball": (mujoco.mjtJoint.mjJNT_BALL, False),
    "free": (mujoco.mjtJoint.mjJNT_FREE, False),
}

_DEFAULT_COLORS = ["#4A90D9", "#E8833A", "#57A773", "#B45F9D", "#C9A227"]


# ---------------------------------------------------------------- 数据结构

@dataclass
class JointDoc:
    type: str = "fixed"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    range: tuple[float, float] | None = None      # revolute: 度；prismatic: 米
    kp: float = 50.0
    kd: float = 1.0
    actuator: bool = True                         # False = 不建执行器

    @property
    def mj_type(self):
        return JOINT_TYPES[self.type][0]

    @property
    def movable(self) -> bool:
        return self.type not in ("fixed",)


@dataclass
class LinkDoc:
    name: str
    parent: str | None = None
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float | None = None
    geom: dict | None = None
    joint: JointDoc = field(default_factory=JointDoc)
    color: str | None = None


@dataclass
class LinkFrame:
    """教学用途：单个连杆相对父系 / 世界系的位姿与旋转矩阵。"""
    name: str
    parent: str | None
    p_rel: np.ndarray          # 3
    R_rel: np.ndarray          # 3x3
    p_world: np.ndarray        # 3
    R_world: np.ndarray        # 3x3
    joint: str                 # 关节类型
    axis: np.ndarray           # 3（fixed 时为零向量）


@dataclass
class RobotDoc:
    name: str
    links: list[LinkDoc] = field(default_factory=list)

    def link(self, name: str) -> LinkDoc:
        for lk in self.links:
            if lk.name == name:
                return lk
        raise WorldError(f"机器人 '{self.name}' 没有连杆 '{name}'")


# ---------------------------------------------------------------- 旋转工具

def euler_to_matrix(rpy_deg) -> np.ndarray:
    """R = Rz(yaw)·Ry(pitch)·Rx(roll)（与 MuJoCo euler "xyz" 约定一致）。"""
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 旋转矩阵 -> 四元数 (w, x, y, z)。"""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    return q


def quat_to_matrix(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 旋转矩阵 -> (roll, pitch, yaw) 度，euler_to_matrix 的逆。"""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                   # 万向锁：roll 固定为 0
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return tuple(round(math.degrees(v), 6) for v in (roll, pitch, yaw))


# ---------------------------------------------------------------- 解析

def _num_list(value, n: int | None, where: str, what: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise WorldError(f"[{where}] {what} 需要数字列表，实际是: {value!r}")
    if n is not None and len(value) != n:
        raise WorldError(f"[{where}] {what} 需要 {n} 个数字，实际是: {value!r}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise WorldError(f"[{where}] {what} 含非数字: {value!r}") from None


def _parse_joint(raw, where: str) -> JointDoc:
    if raw is None:
        return JointDoc()
    if not isinstance(raw, dict):
        raise WorldError(f"[{where}] joint 需要字典，实际是: {raw!r}")
    jt = str(raw.get("type", "fixed")).lower()
    if jt not in JOINT_TYPES:
        raise WorldError(
            f"[{where}] 未知关节类型 '{jt}'，支持: {sorted(JOINT_TYPES)}")
    axis = _num_list(raw.get("axis", [0, 0, 1]), 3, where, "joint.axis")
    if jt not in ("fixed", "free") and np.linalg.norm(axis) < 1e-9:
        raise WorldError(f"[{where}] 关节轴不能是零向量: {axis}")
    rng = raw.get("range")
    if rng is not None:
        rng = _num_list(rng, 2, where, "joint.range")
        if rng[0] >= rng[1]:
            raise WorldError(f"[{where}] joint.range 下限必须小于上限: {rng}")
    act = raw.get("actuator", True)
    act_on = False if act is False else True
    return JointDoc(
        type=jt, axis=axis, range=rng,
        kp=float(raw.get("kp", 50.0)), kd=float(raw.get("kd", 1.0)),
        actuator=act_on)


def parse_robot(doc: dict, base_dir: pathlib.Path | None = None) -> RobotDoc:
    """校验并解析机器人连杆树（结构错误一律报清楚是哪个连杆哪个字段）。"""
    if not isinstance(doc, dict):
        raise WorldError(f"机器人描述需要字典，实际是: {type(doc).__name__}")
    name = str(doc.get("name") or "robot")
    raw_links = doc.get("links")
    if not isinstance(raw_links, list) or not raw_links:
        raise WorldError(f"机器人 '{name}' 缺少 links 列表（至少要有一个根连杆）")
    links: list[LinkDoc] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_links):
        where = f"{name}.links[{i}]"
        if not isinstance(raw, dict):
            raise WorldError(f"[{where}] 需要字典，实际是: {raw!r}")
        lname = str(raw.get("name") or "").strip()
        if not lname:
            raise WorldError(f"[{where}] 缺少连杆名 name")
        if lname in seen:
            raise WorldError(f"[{where}] 连杆名重复: '{lname}'")
        seen.add(lname)
        rpy = _num_list(raw.get("rpy", [0, 0, 0]), 3, where, "rpy（度）")
        quat = raw.get("quat")
        if quat is not None:
            quat = _num_list(quat, 4, where, "quat（wxyz）")
            n = math.sqrt(sum(v * v for v in quat))
            if n < 1e-9:
                raise WorldError(f"[{where}] quat 不能是零向量")
            quat = tuple(v / n for v in quat)
        else:
            quat = tuple(matrix_to_quat(euler_to_matrix(rpy)))
        geom = raw.get("geom")
        if geom is not None and geom is not False:
            if not isinstance(geom, dict):
                raise WorldError(f"[{where}] geom 需要字典或 false，实际是: {geom!r}")
            gtype = str(geom.get("type", "box")).lower()
            if gtype not in GEOM_TYPES:
                raise WorldError(
                    f"[{where}] 未知几何类型 '{gtype}'，支持: "
                    f"{sorted(GEOM_TYPES)}（{GEOM_TYPES})")
            gsize = _num_list(geom.get("size", [0.05, 0.05, 0.05]), None,
                              where, "geom.size")
            geom = dict(geom, type=gtype, size=gsize)
        else:
            geom = None
        links.append(LinkDoc(
            name=lname,
            parent=(str(raw["parent"]) if raw.get("parent") else None),
            pos=_num_list(raw.get("pos", [0, 0, 0]), 3, where, "pos"),
            quat=quat, rpy_deg=rpy,
            mass=(float(raw["mass"]) if raw.get("mass") is not None else None),
            geom=geom,
            joint=_parse_joint(raw.get("joint"), where),
            color=raw.get("color")))
    doc_obj = RobotDoc(name=name, links=links)
    _validate_tree(doc_obj)
    return doc_obj


def _validate_tree(robot: RobotDoc) -> None:
    """树结构校验：唯一根、parent 存在、无环、free 只能给根。"""
    by_name = {lk.name: lk for lk in robot.links}
    roots = [lk for lk in robot.links if lk.parent is None]
    if len(roots) != 1:
        raise WorldError(
            f"机器人 '{robot.name}' 必须恰有一个根连杆（parent 留空），"
            f"实际有 {len(roots)} 个: {[r.name for r in roots]}")
    for lk in robot.links:
        if lk.parent is not None and lk.parent not in by_name:
            raise WorldError(
                f"机器人 '{robot.name}' 的连杆 '{lk.name}' 引用了不存在的父连杆 "
                f"'{lk.parent}'，可用: {sorted(by_name)}")
        if lk.parent is None and lk.joint.type == "free" and lk is not roots[0]:
            raise WorldError("free 关节只能给根连杆")
    # 环检测：每个连杆沿 parent 上溯，步数超过总数即环
    for lk in robot.links:
        cur, steps = lk, 0
        while cur.parent is not None:
            cur = by_name[cur.parent]
            steps += 1
            if steps > len(robot.links):
                raise WorldError(
                    f"机器人 '{robot.name}' 的连杆树有环（从 '{lk.name}' 上溯不收敛）")


def load_robot(path: str | pathlib.Path) -> RobotDoc:
    """读机器人 YAML 文件（.yaml / .yml / .robot.yaml 均可）。

    Windows 适配：中文 Windows 的记事本等工具常把带中文注释的 YAML 存成
    GBK（ANSI），utf-8 解码失败时回退 GBK，避免用户一打开就报编码错误。
    """
    import yaml
    p = pathlib.Path(path)
    if not p.exists():
        raise WorldError(f"机器人描述文件不存在: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="gbk", errors="replace")
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise WorldError(f"机器人 YAML 解析失败 ({p}): {e}") from e
    return parse_robot(doc, p.parent)


def dump_robot(robot: RobotDoc) -> dict:
    """把 RobotDoc 还原成可写盘的 YAML 字典（GUI 保存用）。"""
    links = []
    for lk in robot.links:
        item = {"name": lk.name}
        if lk.parent is not None:
            item["parent"] = lk.parent
        item["pos"] = [round(v, 6) for v in lk.pos]
        item["rpy"] = [round(v, 4) for v in lk.rpy_deg]
        if lk.mass is not None:
            item["mass"] = lk.mass
        joint = {"type": lk.joint.type}
        if lk.joint.movable:
            joint["axis"] = [round(v, 6) for v in lk.joint.axis]
            if lk.joint.range is not None:
                joint["range"] = [round(v, 4) for v in lk.joint.range]
            joint["kp"] = lk.joint.kp
            joint["kd"] = lk.joint.kd
            if not lk.joint.actuator:
                joint["actuator"] = False
        if lk.joint.type != "fixed":
            item["joint"] = joint
        if lk.geom is not None:
            item["geom"] = {
                "type": lk.geom["type"],
                "size": [round(v, 6) for v in lk.geom["size"]],
                **({"pos": list(lk.geom["pos"])} if "pos" in lk.geom else {}),
                **({"color": lk.geom["color"]} if "color" in lk.geom else {}),
            }
        if lk.color:
            item["color"] = lk.color
        links.append(item)
    return {"name": robot.name, "links": links}


# ---------------------------------------------------------------- 位姿分析

def link_frames(robot: RobotDoc) -> list[LinkFrame]:
    """按 links 顺序返回每个连杆的相对/世界位姿与旋转矩阵（教学显示用）。"""
    by_name = {lk.name: lk for lk in robot.links}
    frames: dict[str, LinkFrame] = {}
    out: list[LinkFrame] = []

    def resolve(lk: LinkDoc) -> LinkFrame:
        if lk.name in frames:
            return frames[lk.name]
        R_rel = quat_to_matrix(lk.quat)
        p_rel = np.array(lk.pos, dtype=float)
        if lk.parent is None:
            R_w, p_w = R_rel.copy(), p_rel.copy()
        else:
            pf = resolve(by_name[lk.parent])
            R_w = pf.R_world @ R_rel
            p_w = pf.p_world + pf.R_world @ p_rel
        fr = LinkFrame(
            name=lk.name, parent=lk.parent, p_rel=p_rel, R_rel=R_rel,
            p_world=p_w, R_world=R_w,
            joint=lk.joint.type,
            axis=(np.array(lk.joint.axis, dtype=float)
                  if lk.joint.movable or lk.joint.type == "ball"
                  else np.zeros(3)))
        frames[lk.name] = fr
        return fr

    for lk in robot.links:
        out.append(resolve(lk))
    return out


# ---------------------------------------------------------------- 编译

def build_robot_spec(robot: RobotDoc) -> mujoco.MjSpec:
    """连杆树 -> MjSpec（尚未 compile，可被世界 builder attach）。"""
    mj = mujoco.MjSpec()
    mj.modelname = robot.name
    bodies: dict[str, object] = {}
    joint_names: set[str] = set()
    act_targets: list[tuple[str, LinkDoc]] = []

    for i, lk in enumerate(robot.links):
        parent = (mj.worldbody if lk.parent is None else bodies[lk.parent])
        if parent is None:      # _validate_tree 已保证不会发生
            raise WorldError(f"连杆 '{lk.name}' 的父连杆尚未创建")
        body = parent.add_body(name=lk.name, pos=list(lk.pos), quat=list(lk.quat))
        bodies[lk.name] = body

        jt = lk.joint.type
        if jt != "fixed":
            jname = lk.name if lk.name not in joint_names else f"{lk.name}_joint"
            joint_names.add(jname)
            kwargs = dict(name=jname, type=JOINT_TYPES[jt][0],
                          axis=list(lk.joint.axis) if jt not in ("free",)
                          else None)
            if jt in ("revolute", "hinge", "prismatic", "slide"):
                kwargs["damping"] = lk.joint.kd
            if lk.joint.range is not None and jt in ("revolute", "hinge",
                                                     "prismatic", "slide"):
                # MjSpec 的 joint range 走编译器 angle 单位（默认 degree），
                # 直接喂度；编译后 model.jnt_range 才是弧度
                kwargs.update(limited=True, range=list(lk.joint.range))
            body.add_joint(**{k: v for k, v in kwargs.items() if v is not None})
            if lk.joint.actuator and JOINT_TYPES[jt][1]:
                act_targets.append((jname, lk))

        if lk.geom is not None:
            g = lk.geom
            gtype = GEOM_TYPES[g["type"]][0]
            rgba = hex_to_rgba(g.get("color") or lk.color
                               or _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
            gkw = dict(name=f"{lk.name}_geom", type=gtype,
                       size=list(g["size"]), rgba=rgba)
            if g.get("pos") is not None:
                gkw["pos"] = list(_num_list(g["pos"], 3, lk.name, "geom.pos"))
            if g.get("rpy") is not None:
                rpy = _num_list(g["rpy"], 3, lk.name, "geom.rpy（度）")
                gkw["quat"] = list(matrix_to_quat(euler_to_matrix(rpy)))
            if lk.mass is not None:
                gkw["mass"] = lk.mass
            body.add_geom(**gkw)

    # 位置执行器：执行器名与关节同名（robot.set("<关节名>", 值)）
    for jname, lk in act_targets:
        act = mj.add_actuator(name=jname,
                              trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
                              target=jname)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0:3] = [lk.joint.kp, 0.0, 0.0]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        # kd 已经落到关节阻尼（Euler 隐式积分），伺服只保留刚度项
        act.biasprm[0:3] = [0.0, -lk.joint.kp, 0.0]
        if lk.joint.range is not None:
            # ctrlrange 不受编译器 angle 影响，按弧度给
            r = lk.joint.range
            if lk.joint.type in ("revolute", "hinge"):
                r = (math.radians(r[0]), math.radians(r[1]))
            act.ctrlrange[0:2] = list(r)
            act.ctrllimited = True
    return mj


def actuator_names(robot: RobotDoc) -> list[str]:
    """连杆树会生成的执行器名（与 build_robot_spec 中的命名一致）。"""
    out, seen = [], set()
    for lk in robot.links:
        if lk.joint.type == "fixed" or not lk.joint.actuator:
            continue
        if not JOINT_TYPES[lk.joint.type][1]:
            continue
        name = lk.name
        if name in seen:
            name = f"{lk.name}_joint"
        seen.add(name)
        out.append(name)
    return out
