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
from simovia.world.schema import hex_to_rgba, read_text_compat

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
    #: 一个连杆挂多个几何（DH 自动生成的立管/横杆/法兰）；geom 为单几何简写
    geoms: list[dict] | None = None
    joint: JointDoc = field(default_factory=JointDoc)
    color: str | None = None


@dataclass
class DHRow:
    """标准 Denavit-Hartenberg 参数表的一行。

    相邻坐标系变换（教材标准约定）::

        T_i = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)

    - ``revolute``（转动关节）：theta 是关节变量，``theta_deg`` 只是零位偏置；
    - ``prismatic``（移动关节）：d 是关节变量，``d`` 字段只是中位偏置；
    - ``fixed``：四个量全为常量，这一行不产生自由度。

    range 语义与 JointDoc 一致：revolute 用度、prismatic 用米。
    """
    theta_deg: float = 0.0
    d: float = 0.0
    a: float = 0.0
    alpha_deg: float = 0.0
    joint_type: str = "revolute"
    range: tuple[float, float] | None = None
    kp: float = 50.0
    kd: float = 1.0


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
    #: 非空表示文档由 DH 参数表驱动；links 是 dh_to_links 的派生产物
    dh: list[DHRow] | None = None
    base_height: float = 0.08         # DH 底座圆柱高度（自动几何用）

    @property
    def is_dh(self) -> bool:
        return self.dh is not None

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


def _parse_geom_dict(geom, where: str) -> dict:
    """解析单个几何字典：type/size 规范化，并接受 rpy 或 quat(wxyz)。"""
    if not isinstance(geom, dict):
        raise WorldError(f"[{where}] geom 需要字典，实际是: {geom!r}")
    gtype = str(geom.get("type", "box")).lower()
    if gtype not in GEOM_TYPES:
        raise WorldError(
            f"[{where}] 未知几何类型 '{gtype}'，支持: {sorted(GEOM_TYPES)}")
    gsize = _num_list(geom.get("size", [0.05, 0.05, 0.05]), None,
                      where, "geom.size")
    out = dict(geom, type=gtype, size=gsize)
    if geom.get("quat") is not None:
        q = _num_list(geom["quat"], 4, where, "geom.quat(w,x,y,z)")
        n = math.sqrt(sum(v * v for v in q))
        if n < 1e-9:
            raise WorldError(f"[{where}] geom.quat 不能是零向量")
        out["quat"] = tuple(v / n for v in q)
    return out


# ---------------------------------------------------------------- DH 参数表

#: DH 骨架几何的默认尺寸（米）
DH_BASE_R = 0.08          # 底座圆柱半径
DH_JOINT_R = 0.045        # 关节球半径
DH_LINK_R = 0.022         # 连杆（立管/横杆）半径
DH_FLANGE_H = 0.05        # 末端法兰圆柱半长

_DH_JOINT_OK = {"fixed", "revolute", "hinge", "prismatic", "slide"}


def _parse_dh(raw, name: str) -> list[DHRow]:
    if not isinstance(raw, list) or not raw:
        raise WorldError(
            f"机器人 '{name}' 的 dh 需要非空列表（每行一个关节）")
    rows: list[DHRow] = []
    for i, item in enumerate(raw):
        where = f"{name}.dh[{i}]"
        if not isinstance(item, dict):
            raise WorldError(f"[{where}] 需要字典，实际是: {item!r}")
        jt = str(item.get("type", "revolute")).lower()
        if jt not in _DH_JOINT_OK:
            raise WorldError(
                f"[{where}] DH 关节类型只支持 {sorted(_DH_JOINT_OK)}，"
                f"实际是 '{jt}'")
        rng = item.get("range")
        if rng is not None:
            rng = _num_list(rng, 2, where, "range")
            if rng[0] >= rng[1]:
                raise WorldError(f"[{where}] range 下限必须小于上限: {rng}")
        rows.append(DHRow(
            theta_deg=float(item.get("theta", item.get("theta_deg", 0.0))),
            d=float(item.get("d", 0.0)),
            a=float(item.get("a", 0.0)),
            alpha_deg=float(item.get("alpha", item.get("alpha_deg", 0.0))),
            joint_type=jt, range=rng,
            kp=float(item.get("kp", 50.0)),
            kd=float(item.get("kd", 1.0))))
    return rows


def _dh_row_T(row: DHRow) -> np.ndarray:
    """一行 DH 在零位（关节变量 q=0）时的完整齐次变换。

    revolute 的关节角是 ``theta_deg + q``、prismatic 的位移是 ``d + q``：
    MuJoCo 里关节运动发生在 body 原点（Rz(q)/Tz(q)），link 子 body 只承担
    这个含零位偏置的常量子变换；Rz 与 Tz 同轴可交换，两种关节拼起来都与
    教材 DH 公式等价。
    """
    th = math.radians(row.theta_deg)
    al = math.radians(row.alpha_deg)
    ct, st = math.cos(th), math.sin(th)
    ca, sa = math.cos(al), math.sin(al)
    return np.array([
        [ct, -st * ca,  st * sa, row.a * ct],
        [st,  ct * ca, -ct * sa, row.a * st],
        [0.0,       sa,       ca, row.d],
        [0.0,      0.0,      0.0, 1.0],
    ])


def _cap(name: str, p0, p1, radius: float, color: str) -> dict:
    """沿 p0->p1 的胶囊几何（中点 + 圆柱半长，端点自动补半球）。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    half = max(0.0, length - 2.0 * radius) / 2.0
    mid = (p0 + p1) / 2.0
    quat = (1.0, 0.0, 0.0, 0.0)
    if length > 1e-9:
        z = vec / length
        # MuJoCo 圆柱/胶囊默认沿 z 轴：把 z 轴转到 vec 方向
        R = np.eye(3)
        R[:, 2] = z
        R[:, 0] = np.cross([0.0, 1.0, 0.0], z)
        if np.linalg.norm(R[:, 0]) < 1e-9:
            R[:, 0] = np.cross([1.0, 0.0, 0.0], z)
        R[:, 0] /= np.linalg.norm(R[:, 0])
        R[:, 1] = np.cross(z, R[:, 0])
        quat = tuple(float(v) for v in matrix_to_quat(R))
    return {"name": name, "type": "capsule",
            "size": (radius, half),
            "pos": tuple(float(v) for v in mid), "quat": quat,
            "color": color}


def dh_to_links(rows: list[DHRow], name: str = "robot",
                base_height: float = 0.08) -> list[LinkDoc]:
    """标准 DH 参数表 -> LinkDoc 链（含极简骨架几何）。

    每行 DH 落两个 body（这是让 MuJoCo 运动学与教材 DH 完全一致的关键）：

    - ``jointN``：关节载体 body，原点在第 N 条关节轴上，绕/沿自身 z 轴
      运动。零位时它与 frame N-1 重合。
    - ``linkN``：固定子 body，相对 jointN 做 Tz(d)·Tx(a)·Rx(alpha)，
      原点即 frame N，也是下一行关节的安装位。

    转动发生在 frame N-1 原点：theta 旋转的是"d-a-alpha 这一整段"，
    所以杆长 a 的物理连杆随 theta 摆动，与教科书平面臂一致。
    """
    # 累加齐次变换，得到每个 frame 在世界中的零位位姿（用于摆骨架几何）
    T = np.eye(4)
    row_Ts: list[np.ndarray] = []
    for row in rows:
        Az = _dh_row_T(row)
        row_Ts.append(Az)
        T = T @ Az

    links: list[LinkDoc] = []

    # ---- 底座（固定）：圆柱把 frame 0 抬到 base_height 高度 -------------
    links.append(LinkDoc(
        name="base",
        pos=(0.0, 0.0, 0.0),
        joint=JointDoc(type="fixed"),
        color="#4B5563",
        geoms=[
            {"name": "base_pedestal", "type": "cylinder",
             "size": (DH_BASE_R, base_height / 2.0),
             "pos": (0.0, 0.0, base_height / 2.0), "color": "#4B5563"},
        ]))

    prev = "base"

    for i, row in enumerate(rows, start=1):
        jname = f"joint{i}"
        lname = f"link{i}"
        jt = "revolute" if row.joint_type in ("revolute", "hinge") else (
            "prismatic" if row.joint_type in ("prismatic", "slide")
            else "fixed")
        movable = jt in ("revolute", "prismatic")

        # joint body 零位与 frame i-1 重合（上一个 link body 的原点就是
        # frame i-1，所以相对位姿为零）；第一行安装在底座顶部 base_height
        jpos = (0.0, 0.0, base_height) if i == 1 else (0.0, 0.0, 0.0)
        joint = JointDoc(
            type=jt, axis=(0.0, 0.0, 1.0),
            range=row.range if movable else None,
            kp=row.kp, kd=row.kd, actuator=movable)
        col = _DEFAULT_COLORS[(i - 1) % len(_DEFAULT_COLORS)]
        # 骨架几何挂在关节载体 body 上（随关节一起运动）：
        #   关节球在 frame i-1 原点；d 立管沿 z 到 (0,0,d)；
        #   a 横杆端点经 Rz(theta0) 旋转后正好是 frame i 原点 (a·c,a·s,d)
        body_geoms = [{
            "name": f"{jname}_sphere", "type": "sphere",
            "size": (DH_JOINT_R,), "pos": (0.0, 0.0, 0.0), "color": col}]
        ct0, st0 = math.cos(math.radians(row.theta_deg)), \
            math.sin(math.radians(row.theta_deg))
        end_a = (row.a * ct0, row.a * st0, row.d)
        if abs(row.d) > 1e-9:
            body_geoms.append(_cap(
                f"{lname}_d", (0.0, 0.0, 0.0), (0.0, 0.0, row.d),
                DH_LINK_R, col))
        if abs(row.a) > 1e-9:
            body_geoms.append(_cap(
                f"{lname}_a", (0.0, 0.0, row.d), end_a,
                DH_LINK_R, col))
        links.append(LinkDoc(
            name=jname, parent=prev,
            pos=jpos,
            joint=joint,
            color=col, geoms=body_geoms))

        # 固定 link body：完成 Tz(d)·Tx(a)·Rx(alpha)，原点 = frame i，
        # 是下一行关节的安装位（alpha 决定下一轴朝向）；本身不挂几何
        Az = row_Ts[i - 1]
        lp = Az[:3, 3]
        lR = Az[:3, :3]
        lquat = tuple(float(v) for v in matrix_to_quat(lR))
        lrpy = matrix_to_euler_deg(lR)
        links.append(LinkDoc(
            name=lname, parent=jname,
            pos=(float(lp[0]), float(lp[1]), float(lp[2])),
            quat=lquat, rpy_deg=lrpy,
            joint=JointDoc(type="fixed"),
            color=col))
        prev = lname

    # ---- 末端法兰：最后一个 frame 沿自身 z 伸出一小段短圆柱 -------------
    last = f"link{len(rows)}"
    links.append(LinkDoc(
        name="flange", parent=last,
        pos=(0.0, 0.0, 0.0),
        joint=JointDoc(type="fixed"),
        color="#9CA3AF",
        geoms=[
            {"name": "flange_cyl", "type": "cylinder",
             "size": (DH_JOINT_R * 0.7, DH_FLANGE_H),
             "pos": (0.0, 0.0, DH_FLANGE_H), "color": "#9CA3AF"},
        ]))

    # 自动骨架只做可视化：相邻关节球/连杆端点位置重叠，统一关闭碰撞
    for lk in links:
        for g in (lk.geoms or []):
            g.setdefault("contype", 0)
            g.setdefault("conaffinity", 0)
    return links


def parse_robot(doc: dict, base_dir: pathlib.Path | None = None) -> RobotDoc:
    """校验并解析机器人描述：支持 ``dh:`` 参数表或 ``links:`` 连杆树。

    结构错误一律报清楚是哪个连杆/哪一行哪个字段。
    """
    if not isinstance(doc, dict):
        raise WorldError(f"机器人描述需要字典，实际是: {type(doc).__name__}")
    name = str(doc.get("name") or "robot")

    if isinstance(doc.get("dh"), list):
        rows = _parse_dh(doc["dh"], name)
        base_height = float(doc.get("base_height", 0.08))
        links = dh_to_links(rows, name, base_height=base_height)
        robot = RobotDoc(name=name, links=links, dh=rows,
                         base_height=base_height)
        _validate_tree(robot)
        return robot

    raw_links = doc.get("links")
    if not isinstance(raw_links, list) or not raw_links:
        raise WorldError(
            f"机器人 '{name}' 缺少 dh 参数表或 links 列表"
            f"（至少要有一个根连杆）")
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
            geom = _parse_geom_dict(geom, where)
        else:
            geom = None
        geoms = None
        raw_geoms = raw.get("geoms")
        if raw_geoms is not None:
            if not isinstance(raw_geoms, list) or not raw_geoms:
                raise WorldError(
                    f"[{where}] geoms 需要非空列表，实际是: {raw_geoms!r}")
            geoms = [_parse_geom_dict(gx, f"{where}.geoms[{gi}]")
                     for gi, gx in enumerate(raw_geoms)]
        links.append(LinkDoc(
            name=lname,
            parent=(str(raw["parent"]) if raw.get("parent") else None),
            pos=_num_list(raw.get("pos", [0, 0, 0]), 3, where, "pos"),
            quat=quat, rpy_deg=rpy,
            mass=(float(raw["mass"]) if raw.get("mass") is not None else None),
            geom=geom, geoms=geoms,
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
    text = read_text_compat(p)
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise WorldError(f"机器人 YAML 解析失败 ({p}): {e}") from e
    return parse_robot(doc, p.parent)


def dump_robot(robot: RobotDoc) -> dict:
    """把 RobotDoc 还原成可写盘的 YAML 字典（GUI 保存用）。"""
    if robot.is_dh:
        rows = []
        for r in robot.dh:
            item = {"theta": round(float(r.theta_deg), 4),
                    "d": round(float(r.d), 6),
                    "a": round(float(r.a), 6),
                    "alpha": round(float(r.alpha_deg), 4),
                    "type": r.joint_type,
                    "kp": r.kp, "kd": r.kd}
            if r.range is not None:
                item["range"] = [round(float(v), 4) for v in r.range]
            rows.append(item)
        out = {"name": robot.name, "dh": rows}
        if abs(robot.base_height - 0.08) > 1e-9:
            out["base_height"] = round(float(robot.base_height), 6)
        return out

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

        def _dump_geom(g: dict) -> dict:
            out = dict(g)
            out.update(type=g["type"],
                       size=[round(v, 6) for v in g["size"]])
            if "pos" in out:
                out["pos"] = [round(float(v), 6) for v in out["pos"]]
            if "quat" in out:
                out["quat"] = [round(float(v), 6) for v in out["quat"]]
            if "rpy" in out:
                out["rpy"] = [round(float(v), 4) for v in out["rpy"]]
            return out

        if lk.geoms:
            item["geoms"] = [_dump_geom(g) for g in lk.geoms]
        elif lk.geom is not None:
            item["geom"] = _dump_geom(lk.geom)
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
    """连杆树 -> MjSpec（尚未 compile，可被世界 builder attach）。

    links 列表不保证父连杆在前：按父子关系拓扑排序后再建 body，否则
    "把 link1 的父连杆改成列表后面的 link4"会 KeyError（F03）。
    """
    mj = mujoco.MjSpec()
    mj.modelname = robot.name
    bodies: dict[str, object] = {}
    joint_names: set[str] = set()
    act_targets: list[tuple[str, LinkDoc]] = []

    by_name = {lk.name: lk for lk in robot.links}
    index = {lk.name: i for i, lk in enumerate(robot.links)}
    order: list[tuple[int, LinkDoc]] = []
    done: set[str] = set()

    def visit(lk: LinkDoc) -> None:
        if lk.name in done:
            return
        done.add(lk.name)           # 先标记：环会被 _validate_tree 拦下，这里只防死递归
        if lk.parent is not None:
            parent = by_name.get(lk.parent)
            if parent is None:
                raise WorldError(
                    f"连杆 '{lk.name}' 的父连杆 '{lk.parent}' 不存在")
            visit(parent)
        order.append((index[lk.name], lk))

    for lk in robot.links:
        visit(lk)

    for i, lk in order:
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

        geom_list = list(lk.geoms) if lk.geoms else (
            [lk.geom] if lk.geom is not None else [])
        for gi, g in enumerate(geom_list):
            gtype = GEOM_TYPES[g["type"]][0]
            rgba = hex_to_rgba(g.get("color") or lk.color
                               or _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
            gname = g.get("name") or (
                f"{lk.name}_geom" if gi == 0 else f"{lk.name}_geom{gi}")
            gkw = dict(name=gname, type=gtype,
                       size=list(g["size"]), rgba=rgba)
            if g.get("pos") is not None:
                gkw["pos"] = list(_num_list(g["pos"], 3, lk.name, "geom.pos"))
            if g.get("quat") is not None:
                gkw["quat"] = [float(v) for v in g["quat"]]
            elif g.get("rpy") is not None:
                rpy = _num_list(g["rpy"], 3, lk.name, "geom.rpy（度）")
                gkw["quat"] = list(matrix_to_quat(euler_to_matrix(rpy)))
            # DH 自动骨架几何标 contype=0：只可视化不参与碰撞
            if g.get("contype") is not None:
                gkw["contype"] = int(g["contype"])
            if g.get("conaffinity") is not None:
                gkw["conaffinity"] = int(g["conaffinity"])
            # 手动质量只在显式指定时给第一个几何，其余几何用默认密度
            if lk.mass is not None and gi == 0:
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
