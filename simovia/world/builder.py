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
import os
import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco

from simovia.errors import WorldError
from simovia.grippers import (END_EFFECTOR_PREFIX, END_EFFECTORS,
                               ensure_end_effector, end_effector_info,
                               mount_for)
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
    #: 每台机器人的末端执行器挂载位（左右手各一个；单臂为默认挂点）。
    #: [{"side","site","label","builtin"}]——GUI/manifest 按此列出可装位置。
    mounts: dict[str, list[dict]] = field(default_factory=dict)

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
    """末端执行器 model 字段 -> MJCF 路径：库 key 优先，否则按世界相对路径。"""
    if gmodel in END_EFFECTORS:
        return ensure_end_effector(gmodel)
    return _resolve_path(spec.base_dir, gmodel, "末端执行器模型")


# ---------------------------------------------------------------- 自带末端

#: 自带末端（灵巧手/夹爪）识别 token：按名字分隔符切词后精确匹配，
#: "handle" 不会被误判成 "hand"。
_HAND_TOKENS = {"hand", "finger", "thumb", "palm"}


def _tokens(name: str) -> set[str]:
    return {p for p in re.split(r"[^a-z0-9]+", name.lower()) if p}


def _looks_like_end_effector(name: str) -> bool:
    return bool(_tokens(name) & _HAND_TOKENS)


def _palm_normal(*names: str) -> list[float]:
    """挂点 +x（掌心法向）应钉到的参考轴，父 body 本体系。

    只用最短旋转对齐工具轴时绕轴滚转任意，灵巧手装上会"手心朝外/手指
    横着长"。G1 自带手的约定是：手指沿本体系 +x 伸出，左手掌心朝 -y、
    右手掌心朝 +y（站立时掌面贴向大腿）。夹爪库里的 LEAP 掌心朝自身 +x，
    因此挂点 +x 必须钉到该侧掌心法向，否则整体会绕工具轴滚 180°。
    """
    for n in names:
        if n.startswith("left"):
            return [0.0, -1.0, 0.0]
        if n.startswith("right"):
            return [0.0, 1.0, 0.0]
    return [0.0, 1.0, 0.0]


def _quat_from_axes(z_dir, x_dir) -> list[float]:
    """构造旋转：局部 +z 对齐工具轴 z_dir、局部 +x 对齐 x_dir 的正交化方向。

    夹爪挂点的 +z 是工具轴（法兰法向外伸），+x 是掌心法向；两者一起钉住
    后朝向唯一，不依赖最短旋转的任意滚转。
    """
    def unit(v):
        n = math.sqrt(sum(float(t) * float(t) for t in v))
        return [float(t) / n for t in v] if n > 1e-9 else None

    z = unit(z_dir) or [0.0, 0.0, 1.0]
    x = [float(t) for t in x_dir]
    dot = sum(x[i] * z[i] for i in range(3))
    x = [x[i] - dot * z[i] for i in range(3)]
    if unit(x) is None:                    # x 与工具轴平行：换一根参考轴
        alt = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 0.0, 1.0]
        dot = sum(alt[i] * z[i] for i in range(3))
        x = [alt[i] - dot * z[i] for i in range(3)]
    x = unit(x)
    y = [z[1] * x[2] - z[2] * x[1],
         z[2] * x[0] - z[0] * x[2],
         z[0] * x[1] - z[1] * x[0]]
    # R 的列 = (x, y, z)，转为 wxyz 四元数
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(0.25 * s), (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    return q


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
            "quat": _quat_from_axes(outward, _palm_normal(key, anchor)),
            "stripped": False,
        })
    return groups


#: 腕部末端 body 识别 token：模型既没有自带手、也没有 attachment_site 时，
#: 按这些名字找安装点（如 Unitree G1 的 g1.xml：left/right_wrist_yaw_link）
_WRIST_TOKENS = {"wrist", "flange", "tcp"}


def _has_attachment_site(child: mujoco.MjSpec) -> bool:
    """模型里是否已有夹爪挂点（menagerie 惯例 attachment_site 或自动生成名）。"""
    return any(s.name == "attachment_site"
               or (s.name or "").endswith("_attachment_site")
               for s in child.sites)


def _end_effector_side(model: str) -> str | None:
    """末端执行器库声明的侧别（leap_hand 是左手），用于挑挂点/替换哪只手。"""
    return (END_EFFECTORS.get(model) or {}).get("side")


def _side_of_key(key: str) -> str | None:
    """自带末端 key（left_hand/right_hand）-> 侧别。"""
    return "left" if key.startswith("left") else \
        "right" if key.startswith("right") else None


def _mount_label(side: str | None) -> str:
    return {"left": "左手", "right": "右手"}.get(side or "", "默认挂点")


def _end_effector_prefix(index: int, total: int, side: str | None) -> str:
    """attach 前缀：单个沿用 gripper/（既有控制器兼容）；多个按侧别区分。"""
    if total <= 1:
        return END_EFFECTOR_PREFIX
    return f"gripper_{side}/" if side else f"gripper_{index + 1}/"


def detect_wrist_sites(child: mujoco.MjSpec) -> list[dict]:
    """没有自带手也没有挂点时，按腕部末端 body 生成安装点。

    典型场景：zoo 的 unitree_g1（g1.xml）只有 29 自由度本体，腕部是
    left/right_wrist_yaw_link 这样的叶子连杆，既没有手 body，也没有
    attachment_site——直接挂夹爪会报"模型里没有挂点 site"。

    - 只取名字含 wrist/flange/tcp 的叶子 body（有子 body 说明还没到末端）
    - 法兰面与工具轴：优先用 body 上残留的手形 geom（如 G1 的
      left_rubber_hand 手套网格）的位姿推断；没有则退化为 body 原点 +
      本体系 +x（人形腕部连杆的常规伸出方向），用户可在 GUI 微调 pos/rot
    """
    world = child.worldbody
    bodies: dict[str, object] = {}
    for b in child.bodies:
        if not b.name or b is world or b.name == world.name:
            continue
        bodies[b.name] = b
    sites = []
    for name in sorted(bodies):
        b = bodies[name]
        if not (_tokens(name) & _WRIST_TOKENS) or b.bodies:
            continue
        distal = None
        for g in b.geoms:
            if _looks_like_end_effector(g.name or "") or \
                    _looks_like_end_effector(g.meshname or ""):
                distal = [float(x) for x in g.pos]
                break
        if distal is None or math.sqrt(sum(x * x for x in distal)) < 1e-6:
            distal = [0.05, 0.0, 0.0]
        key = ("left_hand" if name.startswith("left")
               else "right_hand" if name.startswith("right") else name)
        sites.append({"key": key, "anchor": name, "bodies": [],
                      "pos": distal,
                      "quat": _quat_from_axes(distal, _palm_normal(name)),
                      "stripped": False})
    for s in sites:
        s["site"] = (f"{s['key']}_attachment_site" if len(sites) > 1
                     else "attachment_site")
    return sites


def _mount_slots(child: mujoco.MjSpec, groups: list[dict]) -> list[dict]:
    """末端执行器挂载位：有自带手时左右手各一位；无手按腕部/挂点推导。

    返回 [{"side","site","label","builtin"}]。GUI/manifest 用这份清单让
    用户选择“装左手还是右手”，不必自己猜 site 名。
    """
    slots: list[dict] = []
    for g in groups:
        side = _side_of_key(g["key"])
        slots.append({"side": side, "site": g["site"],
                      "label": _mount_label(side), "builtin": g["key"]})
    if slots:
        return slots
    for w in detect_wrist_sites(child):
        side = _side_of_key(w["key"])
        slots.append({"side": side, "site": w["site"],
                      "label": _mount_label(side), "builtin": None})
    if slots:
        return slots
    for s in sorted(child.sites, key=lambda x: x.name or ""):
        if s.name and s.name.endswith("attachment_site"):
            slots.append({"side": None, "site": s.name,
                          "label": "默认挂点", "builtin": None})
    return slots


def _auto_strip_keys(groups: list[dict], robot: RobotSpec) -> set[str]:
    """装末端执行器但没有挂点时，要自动替换的自带末端（按侧别优先匹配）。

    人形双手都在时可以左右各挂一个：左手末端删 left_hand、右手末端删
    right_hand，另一侧没人用就保留；side 缺省时按库声明侧别推断，仍无法
    判断则取当前剩余的第一个。
    """
    if not groups:
        return set()
    available = {g["key"] for g in groups}
    chosen: set[str] = set()
    for ee in robot.end_effectors:
        if ee.site is not None:
            continue
        side = ee.side or _end_effector_side(ee.model)
        key = f"{side}_hand" if side else None
        if key in available:
            chosen.add(key)
            available.discard(key)
            continue
        if not available:
            break
        pick = sorted(available)[0]
        chosen.add(pick)
        available.discard(pick)
    return chosen


def _drop_end_effector_geoms(child: mujoco.MjSpec) -> mujoco.MjSpec:
    """删掉残留在腕部 body 上的自带手外观 geom（如 G1 的 *_rubber_hand）。

    无手模型的"手"只剩一层装饰网格，装外挂夹爪/灵巧手时留着会和新的末端
    穿插，所以生成腕部挂点的同时一并清掉。仍走 to_xml 路线（MjSpec.delete
    在 3.13 有悬垂引用问题），但只删 geom/site，不碰 keyframe/传感器。
    """
    root = ET.fromstring(child.to_xml())

    def walk(parent):
        for el in list(parent):
            if el.tag in ("geom", "site"):
                if (_looks_like_end_effector(el.get("mesh") or "")
                        or _looks_like_end_effector(el.get("name") or "")):
                    parent.remove(el)
                    continue
            walk(el)

    walk(root)
    new = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    new.modelfiledir = child.modelfiledir
    return new


def _ensure_wrist_sites(child: mujoco.MjSpec) -> mujoco.MjSpec:
    """给无挂点机器人补腕部安装点（并清掉原来那层手套网格）。"""
    sites = detect_wrist_sites(child)
    if not sites:
        return child
    child = _drop_end_effector_geoms(child)
    taken = {s.name for s in child.sites}
    for g in sites:
        name, k = g["site"], 1
        while name in taken:
            name, k = f"{g['site']}_{k}", k + 1
        taken.add(name)
        child.body(g["anchor"]).add_site(
            name=name, pos=list(g["pos"]), quat=list(g["quat"]),
            size=[0.006, 0.006, 0.006], rgba=[0.15, 0.85, 0.45, 1])
        g["site"] = name
    return child


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
                         robot: RobotSpec,
                         keys: set[str] | None = None) -> tuple[mujoco.MjSpec, list[dict]]:
    """按 robot.strip_hands 删除自带末端，并在腕部补自动挂点。

    keys：显式指定要删的自带末端 key（"装夹爪即替换"的自动路径用），
    给了就不再参考 robot.strip_hands。
    返回 (新 spec, 被删除末端的描述列表)。未开启 strip_hands 时原样返回。
    """
    groups = detect_end_effectors(child)
    if keys is not None:
        selected = [g for g in groups if g["key"] in keys]
        if not selected:
            return child, []
    else:
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
            keys_w = set(want)
            known = {g["key"] for g in groups}
            if not keys_w <= known:
                raise WorldError(
                    f"机器人 '{robot.name}' 的 strip_hands 引用了不存在的自带末端 "
                    f"{sorted(keys_w - known)}，可用: {sorted(known)}")
            selected = [g for g in groups if g["key"] in keys_w]

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


def _resolve_ee_site(child: mujoco.MjSpec, robot: RobotSpec,
                     ee, side: str | None) -> tuple[mujoco.MjSpec, str]:
    """为一条末端执行器解析挂点，必要时先删同侧自带末端。

    返回 (可能已删手的 child, 挂点 site 名)。规则：
    - 显式 site：真 site 直接用；写的是自带末端 key（left_hand）就先删它
    - 未指定 site：优先替换同侧自带手（left/right 各一个），没有手则用
      同侧已有的自动挂点；完全没有挂点就按腕部末端 body 生成
    """
    groups = detect_end_effectors(child)
    by_key = {g["key"]: g for g in groups}

    def _strip(key: str) -> tuple[mujoco.MjSpec, str]:
        new, stripped = _strip_end_effectors(child, robot, keys={key})
        g = next((x for x in stripped if x["key"] == key), None)
        if g is None:
            raise WorldError(
                f"机器人 '{robot.name}' 删除自带末端 '{key}' 后没有生成挂点")
        return new, g["site"]

    if ee.site is not None:
        if ee.site in by_key:
            return _strip(ee.site)
        return child, ee.site

    if side and f"{side}_hand" in by_key:
        return _strip(f"{side}_hand")

    site_names = {s.name for s in child.sites}
    preferred = f"{side}_hand_attachment_site" if side else None
    if preferred and preferred in site_names:
        return child, preferred
    if "attachment_site" in site_names:
        return child, "attachment_site"
    if groups:                 # 侧别不明：占用剩余的第一个自带末端
        return _strip(sorted(by_key)[0])

    child = _ensure_wrist_sites(child)
    site_names = {s.name for s in child.sites}
    if preferred and preferred in site_names:
        return child, preferred
    side_sites = sorted(n for n in site_names
                        if n.endswith("_attachment_site"))
    if side_sites:
        return child, side_sites[0]
    if "attachment_site" in site_names:
        return child, "attachment_site"
    raise WorldError(
        f"机器人 '{robot.name}' 没有可用的末端执行器挂点，"
        f"可用 site: {sorted(site_names)}")


def _assemble_robot_child(spec: WorldSpec,
                          robot: RobotSpec) -> tuple[mujoco.MjSpec, list[str]]:
    """装配单台机器人的合并 MjSpec（机器人本体 + 可选末端执行器 + 传感器）。

    统一入口：attach 到世界（_attach_robot）与建设备索引
    （_child_actuator_names/_child_joint_names）都调它，保证三处看到的
    是同一份装配结果，不再各自重新 load。

    顺序很关键：先挂末端执行器（site-attach 到腕部），再注入传感器——这样
    触觉传感器可以引用末端执行器手指 body（body 名以 "gripper/" 开头）。
    返回 (合并后的 child spec, 全部执行器短名列表)。
    """
    child = _load_robot_child(spec, robot)

    # ---- 自带末端（人形灵巧手/夹爪）：按需删除并补腕部挂点 ----
    child, _ = _strip_end_effectors(child, robot)

    # 要装外挂末端的腕部先一次性删掉对应自带手：分两侧先后删会让第二侧
    # 只剩一个手组，挂点名退化回 attachment_site（左右两侧就混在一个挂点上）。
    groups_now = detect_end_effectors(child)
    key_set = {g["key"] for g in groups_now}
    auto_keys = _auto_strip_keys(groups_now, robot)
    auto_keys |= {ee.site for ee in robot.end_effectors
                  if ee.site is not None and ee.site in key_set}
    site_override: dict[str, str] = {}
    if auto_keys:
        child, stripped = _strip_end_effectors(child, robot, keys=auto_keys)
        site_override = {g["key"]: g["site"] for g in stripped}

    # ---- 末端执行器 site-attach 到机器人腕部挂点（可左右各一个）----
    # 分两趟：先把全部挂点解析好（这一步可能删自带手并 to_xml/from_string
    # 重建 child），再逐个 attach 末端执行器——否则第二趟改 child 会把已经
    # attach 进去的末端 mesh 相对路径（modelfiledir）弄丢，编译时报
    # "Error opening file 'assets/xxx.obj'"。
    total = len(robot.end_effectors)
    used_sites: set[str] = set()
    plans: list[tuple[int, object, str | None, str]] = []
    for i, ee in enumerate(robot.end_effectors):
        side = ee.side or _end_effector_side(ee.model)
        if ee.site in site_override:
            site_name = site_override[ee.site]
        else:
            child, site_name = _resolve_ee_site(child, robot, ee, side)
        if site_name in used_sites:
            raise WorldError(
                f"机器人 '{robot.name}' 的两条末端执行器都挂到同一挂点 "
                f"'{site_name}'；请在 end_effectors 里用 side（left/right）"
                f"或 site 区分左右")
        used_sites.add(site_name)
        plans.append((i, ee, side, site_name))

    for i, ee, side, site_name in plans:
        site_names = {s.name for s in child.sites}
        if site_name not in site_names:
            raise WorldError(
                f"机器人 '{robot.name}' 模型里没有挂点 site '{site_name}'，"
                f"可用: {sorted(site_names)}（末端执行器 {ee.model}）")
        grip_path = _load_gripper_xml(ee.model, spec)
        try:
            grip = mujoco.MjSpec.from_file(str(grip_path))
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 的末端执行器解析失败 "
                f"({grip_path}): {e}") from e
        if ee.model in END_EFFECTORS:
            # allow_generic：未逐臂标定的机器人（人形/自定义模型）用
            # 通用安装变换——挂点已由 builder 按机器人末端自动生成，
            # origin 在法兰面、+z 沿工具轴，与通用变换的约定一致。
            xf = mount_for(ee.model, robot.model or "", ee.pos, ee.quat,
                           allow_generic=True)
        elif ee.pos is None or ee.quat is None:
            raise WorldError(
                f"自定义末端执行器 '{ee.model}' 不在末端执行器库、无标定 "
                f"mount，必须在 end_effectors 下同时显式写 pos 与 quat 指定"
                f"相对挂点 '{site_name}' 的安装变换（库内条目则自动标定）")
        else:
            xf = {"pos": list(ee.pos), "quat": list(ee.quat)}
        try:
            frame = child.attach(
                grip, site=site_name,
                prefix=_end_effector_prefix(i, total, side))
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 挂载末端执行器 '{ee.model}' 失败: "
                f"{e}") from e
        frame.pos[:] = list(xf["pos"])
        frame.quat[:] = list(xf["quat"])

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
      依次查 urdf 目录、包根目录（sw2urdf 的 <包>/urdf + <包>/meshes 布局）
      以及 SIMOVIA_PACKAGE_PATH / ROS_PACKAGE_PATH 显式搜索路径
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
    # 显式包搜索路径：与 catkin/ROS 习惯一致，按平台分隔符切分
    # （Windows 上不能用 [;:]——盘符 "C:\..." 会被从冒号处切断）
    env_roots = os.environ.get("SIMOVIA_PACKAGE_PATH") or \
        os.environ.get("ROS_PACKAGE_PATH") or ""
    pkg_roots = [pathlib.Path(p) for p in env_roots.split(os.pathsep) if p.strip()]

    def _resolve(m):
        raw = m.group(1)
        # 跨平台绝对路径不参与改写（POSIX / 盘符 / UNC）
        if re.match(r"^([A-Za-z]:[\\/]|\\\\|/)", raw):
            return raw.replace("\\", "/")
        fn = raw.replace("\\", "/")                 # 相对路径 \ 也当分隔符
        pkg = None
        mm = re.match(r"^(?:package|file)://([^/]+)/(.+)$", fn)
        if mm:
            pkg, fn = mm.group(1), mm.group(2)      # sw2urdf 惯例前缀
        cands = [base / fn]                         # 相对 urdf 所在目录
        if pkg:
            # package://NAME/REST 的真实回退（原实现两个候选相同）：
            #   base/NAME/REST        包目录就在 urdf 旁边
            #   base.parent/REST      sw2urdf 布局：<包>/urdf/x.urdf + <包>/meshes/
            #   base.parent/NAME/REST 包目录在 urdf 的上一级
            #   搜索路径/NAME/REST    显式包搜索路径（ROS/simovia 环境变量）
            cands += [base / pkg / fn, base.parent / fn,
                      base.parent / pkg / fn]
            for root in pkg_roots:
                cands += [root / pkg / fn, root / fn]
        seen = set()
        for c in cands:
            key = c.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if c.is_file():
                return c.resolve().as_posix()
        return raw                                  # 找不到保持原样（报错可读）

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
            orig = _load_robot_child(spec, rd)
            groups = detect_end_effectors(orig)
            # 装外挂末端但没有挂点时会自动替换对应侧自带末端，索引要一致
            auto = (set() if not rd.end_effectors or _has_attachment_site(orig)
                    else _auto_strip_keys(groups, rd))
            mounts = _mount_slots(orig, groups)
        except Exception:
            groups, auto, mounts = [], set(), []
        want = rd.strip_hands
        for g in groups:
            g["stripped"] = bool(
                want is True or (isinstance(want, (list, tuple))
                                 and g["key"] in want)
                or g["key"] in auto)
        index.end_effectors[rd.name] = groups
        index.mounts[rd.name] = mounts
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
