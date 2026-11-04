"""世界组装器：WorldSpec -> MjSpec -> MjModel + NameIndex。

流程（方案 5.3 节）：
1. 建 floor/light/objects
2. 每个机器人 attach 到独立 frame（前缀 "<name>/" 隔离命名空间）
3. compile 出 MjModel
4. NameIndex 把 YAML 名字解析成 MuJoCo id，此后内核全部用名字操作

已实测的 API 事实（mujoco 3.15.0）：
- world.attach(child, prefix=..., frame=...) 要求 frame 或 site 之一，返回 MjsFrame
- 前缀对 child 内全部名字（body/joint/actuator/geom）生效
"""
from __future__ import annotations

import math
import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco

from simovia.errors import WorldError
from simovia.grippers import (GRIPPER_PREFIX, GRIPPERS, ensure_gripper,
                               gripper_info, mount_for)
from simovia.world.schema import (
    WorldSpec, RobotSpec, SensorSpec, TaskSpec, DistanceReward, ZoneReward,
    ContactReward, hex_to_rgba,
)

_GEOM_TYPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}

# 地面贴图目录（仓库 models/textures/<floor.texture>.png）
_TEXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "textures"


# ---------------------------------------------------------------- 名字索引

@dataclass
class SensorIndex:
    """单个传感器的读数定位：obs 按顺序拼接 parts 得到该传感器全量读数。

    encoder/joint_torque/tactile 为单 part；imu 为 [gyro(3), accel(3)] 两个 part。
    """
    name: str
    kind: str                       # YAML 类型
    parts: list[tuple[int, int]] = field(default_factory=list)  # (sensor_adr, dim)

    @property
    def dim(self) -> int:
        return sum(d for _, d in self.parts)


@dataclass
class RobotIndex:
    """单个机器人的设备索引（名字均为无前缀原始名）"""
    name: str
    actuators: dict[str, int] = field(default_factory=dict)   # 执行器名 -> id
    joints: dict[str, int] = field(default_factory=dict)     # 关节名 -> id
    bodies: dict[str, int] = field(default_factory=dict)     # body 名 -> id
    sensors: dict[str, SensorIndex] = field(default_factory=dict)  # YAML 传感器名 -> 索引
    cameras: dict[str, int] = field(default_factory=dict)     # 相机名 -> camera id


@dataclass
class NameIndex:
    """编译后的全局名字索引：内核此后只用名字，不碰裸 id"""
    objects: dict[str, int] = field(default_factory=dict)     # YAML 物体名 -> body id
    robots: dict[str, RobotIndex] = field(default_factory=dict)
    bodies: dict[str, int] = field(default_factory=dict)     # 全部 body（含前缀名）-> id
    #: 每台机器人自带末端（人形灵巧手/夹爪）的检测结果，供 GUI 列出/删除。
    #: [{"key","anchor","bodies","site","pos","quat","stripped"}]
    end_effectors: dict[str, list[dict]] = field(default_factory=dict)

    def resolve_body(self, ref: str) -> int:
        """按 物体名 -> 完整前缀名 -> 任意机器人前缀 依次解析 body 引用。"""
        if ref in self.objects:
            return self.objects[ref]
        if ref in self.bodies:
            return self.bodies[ref]
        for robot in self.robots.values():
            pref = f"{robot.name}/{ref}"
            if pref in self.bodies:
                return self.bodies[pref]
        raise WorldError(
            f"引用的 body '{ref}' 不存在。"
            f"可用: 物体 {sorted(self.objects)}，"
            f"机器人内部 body {sorted(set(self.bodies) - set(self.objects))}"
        )

    def available_names(self) -> list[str]:
        return sorted(set(self.bodies))


# ---------------------------------------------------------------- 任务引用解析

@dataclass
class ResolvedTask:
    """schema 的 TaskSpec 编译后版本：所有 body 引用已解析成 id"""
    success_body: int | None = None
    success_min: tuple = ()
    success_max: tuple = ()
    rewards: list = field(default_factory=list)   # (kind, id_a, id_b/zone, weight)
    max_ticks: int | None = None


def resolve_task(task: TaskSpec, index: NameIndex) -> ResolvedTask:
    """把 TaskSpec 里的名字引用解析成 MuJoCo id，错误时列出全部可用名字。"""
    rt = ResolvedTask(max_ticks=task.max_ticks)
    if task.success is not None:
        rt.success_body = index.resolve_body(task.success.body)
        rt.success_min = tuple(task.success.zone.min)
        rt.success_max = tuple(task.success.zone.max)
    for r in task.reward:
        if isinstance(r, DistanceReward):
            rt.rewards.append(("distance", index.resolve_body(r.a),
                               index.resolve_body(r.b), r.weight))
        elif isinstance(r, ZoneReward):
            # zone reward 的 body 用 id，zone 存 (min,max) 元组
            rt.rewards.append(("zone", index.resolve_body(r.body),
                               (tuple(r.zone.min), tuple(r.zone.max)), r.weight))
        elif isinstance(r, ContactReward):
            rt.rewards.append(("contact", index.resolve_body(r.a),
                               index.resolve_body(r.b), r.weight))
    return rt


# ---------------------------------------------------------------- 组装

def _resolve_path(base: pathlib.Path, rel: str, what: str) -> pathlib.Path:
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = base / p
    if not p.exists():
        raise WorldError(f"{what} 文件不存在: {p}（世界文件目录: {base}）")
    return p


def _build_scene(spec: WorldSpec, mj: mujoco.MjSpec):
    sc = spec.scene
    # 白天背景：内置渐变 skybox（rgb1=天顶，rgb2=地平线方向）。
    # MuJoCo 3.13 classic 渲染器清屏色硬编码为黑色，只有 skybox 能给出非黑背景。
    # 天顶偏蓝、地平线偏冷灰（不是纯白）：保留白天亮度，但地平线不再一片
    # 过曝的白色，网格地面与远处地平线有层次，接近 Isaac Sim 的默认取景。
    mj.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.42, 0.62, 0.85], rgb2=[0.86, 0.90, 0.94],
        width=512, height=3072,
    )
    # 白天明亮基调：headlight 是跟随相机的匀光补光，抬高它背光面不发暗。
    # 留一点 specular 让机器人塑料壳有高光，形体边界更清楚；地面材质另外
    # 把 specular 归零，所以这片高光不会糊在地板上。
    mj.visual.headlight.ambient = [0.30, 0.31, 0.33]
    mj.visual.headlight.diffuse = [0.34, 0.34, 0.34]
    mj.visual.headlight.specular = [0.15, 0.15, 0.15]
    # 地面：floor.texture="grid" 时贴网格贴图（一瓦片 2m，内含 0.5m 细格）；
    # 贴图缺失或其它取值时退回纯色。texuniform=True 让格子按世界尺寸走，
    # 不同大小的地面格子一样大（texrepeat 实测 1.0 = 2m 瓦片）。
    floor_kwargs = dict(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[sc.floor.size[0], sc.floor.size[1], 0.1],
        friction=[sc.floor.friction, 0.005, 0.0001],
    )
    floor_tex = _TEXTURE_DIR / f"{sc.floor.texture}.png"
    if sc.floor.texture != "solid" and floor_tex.exists():
        mj.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                       file=str(floor_tex))
        # textures 列表按纹理角色下标排列，RGB 角色在下标 1。
        # specular/shininess 归零：地面是大平面，任何高光都会摊成一片白，
        # 把网格线冲淡（高光只在机器人这类曲面上才有意义）。
        mj.add_material(name="floor_mat", textures=["", "floor_tex"],
                        texrepeat=[1.0, 1.0], texuniform=True,
                        specular=0.0, shininess=0.0)
        floor_kwargs["material"] = "floor_mat"
    else:
        floor_kwargs["rgba"] = [0.45, 0.47, 0.52, 1.0]
    mj.worldbody.add_geom(**floor_kwargs)
    # 场景灯（亮度/角度/阴影由 _normalize_lights 统一）
    if not sc.lights:
        mj.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    for i, light in enumerate(sc.lights):
        mj.worldbody.add_light(name=f"light_{i}", pos=list(light.pos), dir=[0, 0, -1])
    # 物体：独立 body + freejoint（可被抓取/瞬移）；static 物体无 freejoint
    for od in sc.objects:
        body = mj.worldbody.add_body(name=od.name, pos=list(od.pos),
                                     quat=list(od.quat) if od.quat else None)
        if not od.static:
            body.add_joint(name=f"{od.name}_free", type=mujoco.mjtJoint.mjJNT_FREE)
        kwargs = dict(name=f"{od.name}_geom", type=_GEOM_TYPE_MAP[od.type],
                      size=list(od.size), rgba=hex_to_rgba(od.color),
                      friction=[od.friction, 0.005, 0.0001])
        if od.mass is not None:
            kwargs["mass"] = od.mass
        body.add_geom(**kwargs)


def _add_child_sensors(robot: RobotSpec, child: mujoco.MjSpec) -> None:
    """attach 前往 child spec 注入传感器/site/相机（前缀随后自动生效）。

    命名约定（attach 后的全名 = "<robot>/<名字>"）：
    - encoder/joint_torque/tactile：MuJoCo 传感器名 = YAML 传感器名
    - imu：拆成 "<名>_gyro"(3) + "<名>_accel"(3)，obs 拼成 6 维
    - imu/tactile 的挂点 site："<名>_site"
    - camera：相机名 = YAML 传感器名（不加 MuJoCo sensor）
    """
    if not robot.sensors:
        return
    joint_names = {j.name for j in child.joints}
    body_names = {b.name for b in child.bodies}
    site_names = {s.name for s in child.sites}
    sensor_names = {s.name for s in child.sensors}
    camera_names = {c.name for c in child.cameras}

    def _find_body(s: SensorSpec):
        if s.body not in body_names:
            raise WorldError(
                f"机器人 '{robot.name}' 的传感器 '{s.name}' 引用的 body "
                f"'{s.body}' 不在模型里，可用: {sorted(body_names)}")
        return child.body(s.body)

    for s in robot.sensors:
        if s.name in sensor_names or s.name in camera_names:
            raise WorldError(
                f"机器人 '{robot.name}' 的传感器 '{s.name}' 与模型里既有"
                f" sensor/camera 名冲突")
        if s.type in ("encoder", "joint_torque"):
            if s.joint not in joint_names:
                raise WorldError(
                    f"机器人 '{robot.name}' 的传感器 '{s.name}' 引用的关节 "
                    f"'{s.joint}' 不在模型里，可用: {sorted(joint_names)}")
            stype = (mujoco.mjtSensor.mjSENS_JOINTPOS if s.type == "encoder"
                     else mujoco.mjtSensor.mjSENS_JOINTACTFRC)
            child.add_sensor(name=s.name, type=stype,
                             objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=s.joint)
        elif s.type == "imu":
            site_name = f"{s.name}_site"
            if site_name in site_names:
                raise WorldError(
                    f"机器人 '{robot.name}' 的挂点 site '{site_name}' 与模型里"
                    f"既有 site 名冲突")
            _find_body(s).add_site(name=site_name, pos=list(s.pos))
            child.add_sensor(name=f"{s.name}_gyro",
                             type=mujoco.mjtSensor.mjSENS_GYRO,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
            child.add_sensor(name=f"{s.name}_accel",
                             type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
        elif s.type == "tactile":
            site_name = f"{s.name}_site"
            if site_name in site_names:
                raise WorldError(
                    f"机器人 '{robot.name}' 的挂点 site '{site_name}' 与模型里"
                    f"既有 site 名冲突")
            # size 即 touch 的接触过滤球半径：MuJoCo 3.x 只统计接触点落在
            # 以 site 为球心、size 为半径的球内的接触力（size=0 → 恒 0）
            _find_body(s).add_site(name=site_name, pos=list(s.pos),
                                   size=[s.radius] * 3)
            child.add_sensor(name=s.name, type=mujoco.mjtSensor.mjSENS_TOUCH,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
        elif s.type == "camera":
            body = _find_body(s)
            kwargs = dict(name=s.name, pos=list(s.pos))
            if s.quat is not None:
                kwargs["quat"] = list(s.quat)
            if s.fovy is not None:
                kwargs["fovy"] = float(s.fovy)
            body.add_camera(**kwargs)
            # 实体外观：相机机身（小方块）+ 镜头（朝自身 -z 伸出的圆柱）。
            # 纯视觉：不参与碰撞（contype/conaffinity=0）、密度 0 不加质量。
            q = list(s.quat) if s.quat is not None else [1.0, 0.0, 0.0, 0.0]
            body.add_geom(name=f"{s.name}_cam_body",
                          type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.018, 0.012, 0.010], pos=list(s.pos), quat=q,
                          rgba=[0.12, 0.55, 0.95, 1.0],
                          contype=0, conaffinity=0, density=0.0, group=2)
            lens_off = _quat_rotate(q, [0.0, 0.0, -0.014])
            body.add_geom(name=f"{s.name}_cam_lens",
                          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[0.007, 0.006],
                          pos=[s.pos[i] + lens_off[i] for i in range(3)],
                          quat=q, rgba=[0.04, 0.16, 0.28, 1.0],
                          contype=0, conaffinity=0, density=0.0, group=2)


def _ensure_actuator_names(child: mujoco.MjSpec) -> list[str]:
    """给无名执行器补名（google_robot 等模型的 <position joint=.../> 不写 name）。

    名字取传输目标（关节名），冲突按 actuatorN 兜底。attach 前执行——
    名字随前缀存活，索引与 hello 清单才引用得到。返回全部执行器名
    （与 attach 后的前缀名一一对应，去掉前缀即这里的名字）。
    """
    taken = {a.name for a in child.actuators if a.name}
    names = []
    for i, a in enumerate(child.actuators):
        if not a.name:
            base = a.target or f"actuator{i}"
            name, k = base, 1
            while name in taken:
                name = f"{base}_{k}"
                k += 1
            a.name = name
            taken.add(name)
        names.append(a.name)
    return names


def _load_gripper_xml(gmodel: str, spec: WorldSpec) -> pathlib.Path:
    """夹爪 model 字段 -> MJCF 路径：夹爪库 key 优先，否则按世界相对路径。"""
    if gmodel in GRIPPERS:
        return ensure_gripper(gmodel)
    return _resolve_path(spec.base_dir, gmodel, "夹爪模型")


# ---------------------------------------------------------------- 自带末端

#: 自带末端（灵巧手/夹爪）识别 token：按名字分隔符切词后精确匹配，
#: "handle" 不会被误判成 "hand"。
_HAND_TOKENS = {"hand", "finger", "thumb", "palm"}


def _tokens(name: str) -> set[str]:
    return {p for p in re.split(r"[^a-z0-9]+", name.lower()) if p}


def _looks_like_end_effector(name: str) -> bool:
    return bool(_tokens(name) & _HAND_TOKENS)


def _quat_from_z_to(direction) -> list[float]:
    """最短旋转：把 +z 转到 direction（wxyz），direction 单位化后使用。"""
    d = [float(x) for x in direction]
    n = math.sqrt(sum(x * x for x in d))
    if n < 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    d = [x / n for x in d]
    z = (0.0, 0.0, 1.0)
    cx = z[1] * d[2] - z[2] * d[1]
    cy = z[2] * d[0] - z[0] * d[2]
    cz = z[0] * d[1] - z[1] * d[0]
    s = math.sqrt(cx * cx + cy * cy + cz * cz)
    c = max(-1.0, min(1.0, z[0] * d[0] + z[1] * d[1] + z[2] * d[2]))
    if s < 1e-9:
        if c > 0:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]        # 180° 绕 x
    ang = math.atan2(s, c)
    k = math.sin(ang / 2.0) / s
    return [math.cos(ang / 2.0), cx * k, cy * k, cz * k]


def _quat_rotate(q, v) -> list[float]:
    """wxyz 四元数旋转向量（相机实体 lens 偏移用）。"""
    w, x, y, z = [float(t) for t in q]
    vx, vy, vz = [float(t) for t in v]
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)]


def detect_end_effectors(child: mujoco.MjSpec) -> list[dict]:
    """扫描机器人 spec，找出自带末端（手/灵巧手/指）子树。

    返回每个末端的描述：
    - key：left_hand / right_hand / hand（按根 body 侧别命名）
    - anchor：末端接在哪个 body（腕部）
    - bodies：该末端的全部 body 短名（含手指链）
    - site：删除后自动生成的腕部挂点名
    - pos/quat：挂点相对 anchor 的位姿——origin 落在手的根部（法兰面），
      +z 沿手伸出的方向，供夹爪库的通用安装变换直接使用。
    """
    nodes: dict[str, tuple[str, list[float]]] = {}
    world = child.worldbody
    for b in child.bodies:
        # worldbody 也在这份列表里，而它的 parent 在 C 侧是 NULL：读
        # b.parent.name 会段错误（mujoco 3.13 实测），必须先跳过。
        # 同时按名字兜底，避免不同平台上 wrapper 缓存策略差异
        if not b.name or b is world or b.name == world.name:
            continue
        parent = "" if b.parent is world else (b.parent.name or "")
        nodes[b.name] = (parent, [float(x) for x in b.pos])
    hand_names = [n for n in nodes if _looks_like_end_effector(n)]
    if not hand_names:
        return []
    roots = [n for n in hand_names if nodes[n][0] not in hand_names]
    by_anchor: dict[str, list[str]] = {}
    for r in roots:
        by_anchor.setdefault(nodes[r][0], []).append(r)

    groups = []
    for anchor, rts in by_anchor.items():
        members = []
        for n in hand_names:              # 沿父链归属到最近的根
            p = n
            while p and p in nodes:
                if p in rts:
                    members.append(n)
                    break
                p = nodes[p][0]
        if all(r.startswith("left") for r in rts):
            key = "left_hand"
        elif all(r.startswith("right") for r in rts):
            key = "right_hand"
        elif len(rts) == 1 and not any(r.startswith(("left", "right"))
                                      for r in rts):
            key = rts[0]
        else:
            key = rts[0]
        positions = [nodes[r][1] for r in rts]
        mean = [sum(p[i] for p in positions) / len(positions) for i in range(3)]
        norm = math.sqrt(sum(x * x for x in mean))
        if norm < 1e-6:                   # 手根在 anchor 原点：默认工具轴 +z
            outward = [0.0, 0.0, 1.0]
            proj = 0.0
        else:
            outward = [x / norm for x in mean]
            proj = sum(outward[i] * mean[i] for i in range(3))
        groups.append({
            "key": key,
            "anchor": anchor,
            "bodies": sorted(members),
            "site": ("attachment_site" if len(by_anchor) == 1
                     else f"{key}_attachment_site"),
            "pos": [outward[i] * proj for i in range(3)],
            "quat": _quat_from_z_to(outward),
            "stripped": False,
        })
    return groups


def _prune_spec_bodies(child: mujoco.MjSpec,
                       remove_bodies: set[str]) -> mujoco.MjSpec:
    """从 spec XML 里删除 body 子树及其关联 actuator/sensor/keyframe。

    MjSpec.delete 在 mujoco 3.13 会留下悬垂引用（解释器退出时段错误），
    因此走 to_xml -> 删节点 -> from_string。to_xml 已展开 include，
    modelfiledir 继承后 meshdir 相对路径照常解析。
    """
    root = ET.fromstring(child.to_xml())
    removed_bodies: set[str] = set()
    removed_joints: set[str] = set()
    removed_other: set[str] = set()      # site/camera/light/tendon 名

    def _collect(el):
        for j in el.iter("joint"):
            if j.get("name"):
                removed_joints.add(j.get("name"))
        for tag in ("site", "camera", "light", "tendon"):
            for s in el.iter(tag):
                if s.get("name"):
                    removed_other.add(s.get("name"))

    def walk(parent):
        for el in list(parent):
            if el.tag == "body" and el.get("name") in remove_bodies:
                removed_bodies.add(el.get("name"))
                _collect(el)
                parent.remove(el)
                continue
            if el.tag in ("geom", "site"):
                nm = el.get("name") or ""
                mesh = el.get("mesh") or ""
                # 留在腕部 body 上的手掌 geom（mesh 以手 body 命名）一并删除
                if (nm in remove_bodies or mesh in remove_bodies
                        or _looks_like_end_effector(mesh)):
                    if nm:
                        removed_other.add(nm)
                    parent.remove(el)
                    continue
            walk(el)

    walk(root)
    gone = removed_bodies | removed_joints | removed_other

    def _refs_gone(el) -> bool:
        return any(v in gone for v in el.attrib.values())

    for tag in ("actuator", "sensor", "tendon"):
        sec = root.find(tag)
        if sec is None:
            continue
        for el in list(sec):
            if _refs_gone(el):
                if el.get("name"):
                    removed_other.add(el.get("name"))
                sec.remove(el)
            else:
                for sub in list(el):
                    if _refs_gone(sub):
                        el.remove(sub)
    for tag in ("contact", "equality", "exclude"):
        sec = root.find(tag)
        if sec is None:
            continue
        for el in list(sec):
            if _refs_gone(el):
                sec.remove(el)
    # keyframe 的 qpos 长度与已删关节绑定，删手后必然失配
    key = root.find("keyframe")
    if key is not None:
        root.remove(key)

    new = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    new.modelfiledir = child.modelfiledir
    return new


def _strip_end_effectors(child: mujoco.MjSpec,
                         robot: RobotSpec) -> tuple[mujoco.MjSpec, list[dict]]:
    """按 robot.strip_hands 删除自带末端，并在腕部补自动挂点。

    返回 (新 spec, 被删除末端的描述列表)。未开启 strip_hands 时原样返回。
    """
    groups = detect_end_effectors(child)
    want = robot.strip_hands
    if not want:
        return child, []
    if not groups:
        raise WorldError(
            f"机器人 '{robot.name}' 设置了 strip_hands，但模型里没有检测到"
            f"自带末端（名字含 hand/finger/thumb/palm 的 body）")
    if want is True:
        selected = groups
    else:
        keys = set(want)
        known = {g["key"] for g in groups}
        if not keys <= known:
            raise WorldError(
                f"机器人 '{robot.name}' 的 strip_hands 引用了不存在的自带末端 "
                f"{sorted(keys - known)}，可用: {sorted(known)}")
        selected = [g for g in groups if g["key"] in keys]

    remove = {b for g in selected for b in g["bodies"]}
    new_child = _prune_spec_bodies(child, remove)
    existing = {b.name for b in new_child.bodies}
    taken_sites = {s.name for s in new_child.sites}
    for g in selected:
        if g["anchor"] not in existing:
            raise WorldError(
                f"机器人 '{robot.name}' 删除自带末端后找不到腕部 body "
                f"'{g['anchor']}'")
        name = g["site"]
        if name in taken_sites:        # 原模型已有同名 site：换成侧别专属名
            base = f"{g['key']}_attachment_site"
            name, k = base, 1
            while name in taken_sites:
                name, k = f"{base}_{k}", k + 1
            g["site"] = name
        taken_sites.add(name)
        new_child.body(g["anchor"]).add_site(
            name=name, pos=list(g["pos"]), quat=list(g["quat"]),
            size=[0.006, 0.006, 0.006], rgba=[0.15, 0.85, 0.45, 1])
        g["stripped"] = True
    return new_child, groups


def _load_robot_child(spec: WorldSpec, robot: RobotSpec) -> mujoco.MjSpec:
    """只加载机器人本体（不挂夹爪、不注入传感器）。"""
    if robot.spec is not None:
        # 内联连杆树（GUI「造机器人」/ 世界 YAML 里直接写 links）
        from simovia.robot_spec import build_robot_spec, parse_robot
        try:
            return build_robot_spec(parse_robot(robot.spec))
        except WorldError:
            raise
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 内联连杆树解析失败: {e}") from e
    model_path = _resolve_path(spec.base_dir, robot.model, "机器人模型")
    try:
        return _load_robot_spec(model_path)
    except WorldError:
        raise
    except Exception as e:
        raise WorldError(
            f"机器人 '{robot.name}' 模型解析失败 ({model_path}): {e}") from e


def _assemble_robot_child(spec: WorldSpec,
                          robot: RobotSpec) -> tuple[mujoco.MjSpec, list[str]]:
    """装配单台机器人的合并 MjSpec（机器人本体 + 可选夹爪 + 传感器）。

    统一入口：attach 到世界（_attach_robot）与建设备索引
    （_child_actuator_names/_child_joint_names）都调它，保证三处看到的
    是同一份装配结果，不再各自重新 load。

    顺序很关键：先挂夹爪（site-attach 到腕部），再注入传感器——这样触觉
    传感器可以引用夹爪手指 body（body 名以 "gripper/" 开头）。
    返回 (合并后的 child spec, 全部执行器短名列表)。
    """
    child = _load_robot_child(spec, robot)

    # ---- 自带末端（人形灵巧手/夹爪）：按需删除并补腕部挂点 ----
    child, _ = _strip_end_effectors(child, robot)

    # ---- 夹爪 site-attach 到机器人腕部挂点 ----
    if robot.gripper is not None:
        g = robot.gripper
        grip_path = _load_gripper_xml(g.model, spec)
        try:
            grip = mujoco.MjSpec.from_file(str(grip_path))
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 的夹爪解析失败 ({grip_path}): {e}") from e
        site_names = {s.name for s in child.sites}
        auto_sites = sorted(n for n in site_names
                            if n == "attachment_site"
                            or n.endswith("_attachment_site"))
        if g.site is not None:
            site_name = g.site          # 显式指定：缺失时直接报错
        elif "attachment_site" in site_names:
            site_name = "attachment_site"
        elif auto_sites:                # 双手都删除后自动挑第一个腕部挂点
            site_name = auto_sites[0]
        elif g.model in GRIPPERS:
            site_name = gripper_info(g.model)["site"]
        else:                           # 自定义 MJCF 路径：menagerie 惯例挂点名
            site_name = "attachment_site"
        if site_name not in site_names:
            raise WorldError(
                f"机器人 '{robot.name}' 模型里没有挂点 site '{site_name}'，"
                f"可用: {sorted(site_names)}（夹爪 {g.model}）")
        if g.model in GRIPPERS:
            # allow_generic：未逐臂标定的机器人（人形/自定义模型）用
            # 通用安装变换——挂点已由 builder 按机器人末端自动生成，
            # origin 在法兰面、+z 沿工具轴，与通用变换的约定一致。
            xf = mount_for(g.model, robot.model or "", g.pos, g.quat,
                           allow_generic=True)
        elif g.pos is None or g.quat is None:
            raise WorldError(
                f"自定义夹爪 '{g.model}' 不在夹爪库、无标定 mount，必须在 "
                f"gripper 下同时显式写 pos 与 quat 指定相对挂点 '{site_name}' "
                f"的安装变换（库内夹爪则自动标定）")
        else:
            xf = {"pos": list(g.pos), "quat": list(g.quat)}
        try:
            grip_frame = child.attach(grip, site=site_name,
                                      prefix=GRIPPER_PREFIX)
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 挂载夹爪 '{g.model}' 失败: {e}") from e
        grip_frame.pos[:] = list(xf["pos"])
        grip_frame.quat[:] = list(xf["quat"])

    _add_child_sensors(robot, child)          # 在合并 spec 上注入（触觉可挂手指）
    act_names = _ensure_actuator_names(child)  # 无名执行器补名（attach 前改名才存活）

    # root body 自带 pos/quat 清零：spec.pos/quat 必须是机器人 root 的绝对世界
    # 位姿，否则 menagerie 四足（trunk 自带 0.445 高度）会被叠加出生到 0.9m
    # 高空——GUI 拖动回写位姿也因此不幂等
    for b in child.bodies:
        if b.parent is child.worldbody:
            b.pos[:] = [0.0, 0.0, 0.0]
            b.quat[:] = [1.0, 0.0, 0.0, 0.0]
    return child, act_names


def _load_robot_spec(model_path: pathlib.Path) -> mujoco.MjSpec:
    """读机器人模型：MJCF 直接读；URDF 做导入适配；.yaml 走连杆树编译。

    URDF 与 SolidWorks 工作流（sw2urdf 插件导出）的差异由这里吸收：
    - mesh 路径：MuJoCo 的 from_string 无文件目录上下文，先把全部
      mesh filename 重写为基于 urdf 所在目录的绝对路径；package://NAME/REST
      按 REST / NAME/REST 两级就近查找（sw2urdf 的导出目录惯例）
    - 执行器：URDF 没有 <actuator>（transmission 少有人写），每个关节
      自动补一个同名 motor（力矩控制，simovia 的 ctrl 直写语义）

    注意：MuJoCo 的 URDF 解析只用 <collision>，<visual> 全部忽略——
    SolidWorks 导出时确保零件有碰撞几何（sw2urdf 默认会生成）。

    Windows 适配：
    - 编码：sw2urdf 在中文 Windows 上可能写出 GBK 编码的 URDF，
      utf-8 优先、GBK 兜底（中文用户路径/注释常见）
    - 绝对路径：/x（POSIX）、C:\\x / C:/x（盘符）、\\\\srv\\x（UNC）
      均不参与相对改写
    - 重写结果统一正斜杠（.as_posix()）：XML 属性 / YAML / 跨平台
      三处都安全，MuJoCo 在 Windows 上也接受正斜杠
    """
    if model_path.suffix.lower() in (".yaml", ".yml"):
        from simovia.robot_spec import build_robot_spec, load_robot
        return build_robot_spec(load_robot(model_path))
    if model_path.suffix.lower() != ".urdf":
        return mujoco.MjSpec.from_file(str(model_path))

    import re
    try:
        text = model_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = model_path.read_text(encoding="gbk", errors="replace")
    base = model_path.parent

    def _resolve(m):
        fn = m.group(1)
        # 跨平台绝对路径不参与改写（POSIX / 盘符 / UNC）
        if re.match(r"^([A-Za-z]:[\\/]|\\\\|/)", fn):
            return fn.replace("\\", "/")
        fn = re.sub(r"^(package|file)://", "", fn)  # sw2urdf 惯例前缀
        fn = fn.replace("\\", "/")                  # 相对路径 \ 也当分隔符
        # package://NAME/REST：先按原相对结构找，再按 NAME 目录下找
        cands = [base / fn]
        parts = fn.split("/", 1)
        if len(parts) == 2 and parts[1]:
            cands.append(base / parts[0] / parts[1])
        for c in cands:
            if c.is_file():
                return c.resolve().as_posix()
        return fn                                   # 找不到保持原样（报错可读）

    text = re.sub(r'filename="([^"]+)"',
                  lambda m: f'filename="{_resolve(m)}"', text)
    try:
        child = mujoco.MjSpec.from_string(text)
    except Exception as e:
        raise WorldError(
            f"URDF 解析失败 ({model_path}): {e}\n"
            f"（提示：MuJoCo 只读取 URDF 的 <collision>；mesh 仅支持 "
            f"STL/OBJ）") from e
    # URDF 无执行器：逐关节补 motor（URDF 解析器已把 fixed joint 合并，
    # joints 列表里全是可动关节）
    for j in child.joints:
        child.add_actuator(name=j.name,
                           trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
                           target=j.name)
    return child


def _attach_robot(spec: WorldSpec, robot: RobotSpec, mj: mujoco.MjSpec) -> None:
    child, _ = _assemble_robot_child(spec, robot)
    frame = mj.worldbody.add_frame(
        name=f"{robot.name}_frame", pos=list(robot.pos),
        quat=list(robot.quat) if robot.quat else None)
    try:
        mj.attach(child, prefix=f"{robot.name}/", frame=frame)
    except Exception as e:
        raise WorldError(f"机器人 '{robot.name}' 挂载失败: {e}") from e


def _build_index(model: mujoco.MjModel, spec: WorldSpec) -> NameIndex:
    index = NameIndex()
    # 物体
    for od in spec.scene.objects:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, od.name)
        if bid < 0:
            raise WorldError(f"编译后找不到物体 '{od.name}' 的 body（编译器吞了名字？）")
        index.objects[od.name] = bid
    # 机器人设备
    for rd in spec.robots:
        prefix = f"{rd.name}/"
        ridx = RobotIndex(name=rd.name)
        # 自带末端检测（在原始模型上做：删除后仍能列出/恢复）
        try:
            groups = detect_end_effectors(_load_robot_child(spec, rd))
        except Exception:
            groups = []
        want = rd.strip_hands
        for g in groups:
            g["stripped"] = bool(
                want is True or (isinstance(want, (list, tuple))
                                 and g["key"] in want))
        index.end_effectors[rd.name] = groups
        for act_name in _child_actuator_names(spec, rd):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     f"{prefix}{act_name}")
            if aid < 0:
                raise WorldError(
                    f"编译后找不到机器人 '{rd.name}' 的执行器 '{act_name}'"
                    f"（期望全名 '{prefix}{act_name}'）")
            ridx.actuators[act_name] = aid
        for jnt_name in _child_joint_names(spec, rd):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"{prefix}{jnt_name}")
            if jid < 0:
                raise WorldError(
                    f"编译后找不到机器人 '{rd.name}' 的关节 '{jnt_name}'"
                    f"（期望全名 '{prefix}{jnt_name}'）")
            ridx.joints[jnt_name] = jid
        # 传感器（imu 拆成 gyro/accel 两个 MuJoCo sensor，按 part 记录读数区间）
        for s in rd.sensors:
            if s.type == "camera":
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                         f"{prefix}{s.name}")
                if cid < 0:
                    raise WorldError(
                        f"编译后找不到机器人 '{rd.name}' 的相机 '{s.name}'"
                        f"（期望全名 '{prefix}{s.name}'）")
                ridx.cameras[s.name] = cid
                continue
            mj_names = ([f"{prefix}{s.name}"] if s.type != "imu"
                        else [f"{prefix}{s.name}_gyro", f"{prefix}{s.name}_accel"])
            parts = []
            for full in mj_names:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, full)
                if sid < 0:
                    raise WorldError(
                        f"编译后找不到机器人 '{rd.name}' 的传感器 '{full}'"
                        f"（编译器吞了名字？）")
                parts.append((int(model.sensor_adr[sid]), int(model.sensor_dim[sid])))
            ridx.sensors[s.name] = SensorIndex(name=s.name, kind=s.type, parts=parts)
        index.robots[rd.name] = ridx
    # 全部 body（含前缀）
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name:
            continue
        index.bodies[name] = i
        # 归入所属机器人的 bodies（去掉前缀）
        for rd in spec.robots:
            if name.startswith(f"{rd.name}/"):
                index.robots[rd.name].bodies[name[len(rd.name) + 1:]] = i
    return index


def _child_actuator_names(spec: WorldSpec, rd: RobotSpec) -> list[str]:
    """读取装配后（含夹爪）全部执行器名（机器人前缀之外的短名），供索引解析。

    与 _attach_robot 走同一个 _assemble_robot_child，补名/夹爪 prefix
    确定性一致，夹爪执行器（gripper/gripper_act）自动并入。
    """
    _, names = _assemble_robot_child(spec, rd)
    if not names:
        raise WorldError(
            f"机器人 '{rd.name}' 模型里没有任何执行器 "
            f"({rd.model or '内联连杆树'})，"
            f"无法建立设备索引")
    return names


def _child_joint_names(spec: WorldSpec, rd: RobotSpec) -> list[str]:
    """读取装配后（含夹爪）全部关节名（机器人前缀之外的短名）。"""
    child, _ = _assemble_robot_child(spec, rd)
    return [j.name for j in child.joints if j.name]


def _normalize_lights(mj: mujoco.MjSpec) -> None:
    """把所有灯统一成柔和天光：宽角、无聚光衰减、只有第 0 盏投影。

    机器人/夹爪模型自带的光源参数各不相同（menagerie 常见 cutoff=45、
    exponent=10、specular=0.5），在亮色网格地面上会形成刺眼亮斑和锯齿
    阴影；统一后场景光照一致。第 0 盏一定是世界灯（_build_scene 保底
    加了一盏），只让它投一份影子把物体"压"在地面上——多盏灯各投一份
    会叠成脏斑。specular 留 0.20（不是 0）：机器人塑料壳靠这点高光才
    看得出曲面，地面材质已单独归零不会受影响。纯渲染参数，不影响物理。
    """
    for i, light in enumerate(mj.lights):
        light.cutoff = 90
        light.exponent = 2.0
        light.diffuse = [0.28, 0.28, 0.28]
        light.specular = [0.20, 0.20, 0.20]
        light.castshadow = 1 if i == 0 else 0


def _smooth_meshes(mj: mujoco.MjSpec) -> None:
    """所有 mesh 改用平滑法线（smoothnormal=True）。

    menagerie 的机器人网格普遍是低多边形 STL，默认 flat shading 会让圆柱
    面出现一条条棱面，远看就是"糊/毛"。这是编译期选项，必须在 compile
    前设置；对 STL 只影响渲染法线，不动碰撞几何。
    """
    for mesh in mj.meshes:
        mesh.smoothnormal = True


def build_world(spec: WorldSpec) -> tuple[mujoco.MjModel, NameIndex]:
    """WorldSpec -> (MjModel, NameIndex)。schema 校验必须已完成（load_world 保证）。"""
    mj = mujoco.MjSpec()
    mj.modelname = spec.name
    mj.option.timestep = spec.timestep
    mj.option.gravity = list(spec.gravity)
    # attach 时 parent（世界）的 option 优先：大质量机器人需要高
    # impratio（如 spot/anymal 50-100），在此设置才会覆盖模型原值
    mj.option.impratio = spec.impratio
    # 积分器同理被 attach 覆盖：menagerie 高刚度模型（spot 的 solref=0.004）
    # 原生用 implicitfast，被世界默认 Euler 覆盖后"硬接触 + 显式积分"会
    # 数值震荡弹跳（躯干周期性跳起、关节跟踪却良好的根因）
    mj.option.integrator = spec.integrator
    # 离屏帧缓冲调大：GUI 渲染视口可达 ~1600px，默认 640 会抛异常
    mj.visual.global_.offwidth = 1920
    mj.visual.global_.offheight = 1440

    _build_scene(spec, mj)
    for rd in spec.robots:
        _attach_robot(spec, rd, mj)
    # 机器人模型自带的灯（cutoff=45/exponent=10/castshadow）会在亮色地面上
    # 打出刺眼亮斑和锯齿阴影，合并后统一归一化成柔和天光
    _normalize_lights(mj)
    _smooth_meshes(mj)

    try:
        model = mj.compile()
    except Exception as e:
        raise WorldError(f"世界 '{spec.name}' 编译失败: {e}") from e

    # 离屏 MSAA：默认 4 份采样，机器人轮廓/细长构件仍有明显锯齿，翻倍到 8。
    # 必须赶在 mujoco.Renderer 建帧缓冲之前设置（GUI/截图都是建 Renderer 前
    # 先拿 model），实测 960x720 每帧代价 <0.2ms
    model.vis.quality.offsamples = 8

    index = _build_index(model, spec)

    # 任务引用最终核对（此前只有无机器人场景能提前校验）
    resolve_task(spec.task, index)

    return model, index
