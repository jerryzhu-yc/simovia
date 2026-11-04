"""M6 P1：夹爪 site-attach 到四台 menagerie 机械臂。

覆盖：
- 四臂（ur5e/ur10e/xarm7_nohand/panda_nohand）挂爪均编译成功
- 夹爪 actuator/joint/body 并入机器人设备索引（gripper/ 前缀）
- 物理：ctrl=close 后两 slide 实际闭合（qpos>0.015），open 回 0
- 触觉传感器可引用夹爪手指 body（先挂爪再注传感器的顺序保证）
- mount 未标定的机器人给清晰错误
- 新增夹爪（Robotiq 2F85 / UMI / LEAP Hand）编译、四臂挂载、贴合与驱动
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from simovia.errors import WorldError
from simovia.grippers import GRIPPERS, GRIPPER_PREFIX, ensure_gripper
from simovia.world.builder import build_world
from simovia.world.schema import parse_world
from simovia.zoo import ensure_model


def _menagerie_xml(key: str) -> str:
    """模型库取臂 XML（ensure_model 跨平台解析缓存目录，Windows 同样可用）。"""
    try:
        return str(ensure_model(key))
    except Exception as e:                       # 未下载/无网络等
        pytest.skip(f"menagerie 模型不可用: {e}")


ARMS = [
    ("ur5e", "universal_robots_ur5e"),
    ("ur10e", "universal_robots_ur10e"),
    ("xarm7", "ufactory_xarm7"),
    ("panda", "franka_emika_panda"),
]

NEW_GRIPPERS = ["robotiq_2f85", "umi_gripper", "leap_hand"]

# 新增夹爪的“朝向探针”：四臂挂载后这些 body/site 应在 site +z 方向伸出。
# 2F85 用两指 pad body；UMI 用 TCP site（两指中点前方）；LEAP 用三指远端 body。
Z_PROBES = {
    "robotiq_2f85": ("body", ["left_pad", "right_pad"]),
    "umi_gripper": ("site", ["gripper_tcp"]),
    "leap_hand": ("body", ["if_ds", "mf_ds", "rf_ds"]),
}


def _world_dict(arm_xml: str, gripper=True, sensors=(),
                gripper_key="onrobot_2fg7", pos=(0, 0, 0)) -> dict:
    robot = {"name": "arm", "model": arm_xml, "pos": list(pos)}
    if gripper:
        robot["gripper"] = {"model": gripper_key}
    if sensors:
        robot["sensors"] = list(sensors)
    return {"name": "gripper_test",
            "timestep": 0.002, "ctrl_period": 8,
            "integrator": "implicitfast", "impratio": 100,
            "scene": {"floor": {"size": [2.0, 2.0]}},
            "robots": [robot]}


def _site_frame(model, data):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "arm/attachment_site")
    return data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)


def _geom_points(model, data, gid, n=30000):
    """geom 表面点集（mesh 取真实顶点；基本体采样），世界系。"""
    rng = np.random.default_rng(0)
    mid = model.geom_dataid[gid]
    if mid >= 0:
        v = model.mesh_vert[
            model.mesh_vertadr[mid]:
            model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        return (data.geom_xmat[gid].reshape(3, 3) @ v.T).T \
            + data.geom_xpos[gid]
    ty = int(model.geom_type[gid])
    sz = model.geom_size[gid]
    a = rng.uniform(0, 2 * np.pi, n)
    if ty == int(mujoco.mjtGeom.mjGEOM_BOX):
        q = rng.uniform(-1, 1, (n, 3)) * sz
    elif ty == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        z = rng.uniform(-sz[1], sz[1], n)
        q = np.c_[sz[0] * np.cos(a), sz[0] * np.sin(a), z]
    elif ty == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, hh = sz[0], sz[1]
        z = rng.uniform(-(hh + r), hh + r, n)
        t = np.clip(np.abs(z) - hh, 0, r)
        rad = np.sqrt(np.clip(r * r - t * t, 0, None))
        q = np.c_[rad * np.cos(a), rad * np.sin(a), z]
    elif ty == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        q = u * sz[0]
    elif ty == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        q = u * sz
    else:
        return None
    return (data.geom_xmat[gid].reshape(3, 3) @ q.T).T \
        + data.geom_xpos[gid]


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_attaches_to_four_arms(label, key):
    """四臂挂 2FG7 均编译成功，且夹爪设备并入索引。"""
    xml = _menagerie_xml(key)
    spec = parse_world(_world_dict(xml))
    model, index = build_world(spec)
    ridx = index.robots["arm"]

    assert f"{GRIPPER_PREFIX}gripper_act" in ridx.actuators
    assert f"{GRIPPER_PREFIX}left_slide" in ridx.joints
    assert f"{GRIPPER_PREFIX}right_slide" in ridx.joints
    # body 全名（机器人前缀 + 夹爪前缀）
    assert f"arm/{GRIPPER_PREFIX}gripper_base" in index.bodies
    # tendon 跨嵌套 attach 存活
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,
                            "arm/gripper/gripper_tendon")
    assert tid >= 0


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_actuator_count(label, key):
    """挂爪后执行器/关节数 = 裸臂 + 夹爪各 1 个 actuator / 2 个 slide。"""
    xml = _menagerie_xml(key)
    bare_spec = parse_world(_world_dict(xml, gripper=False))
    bare_model, _ = build_world(bare_spec)
    full_spec = parse_world(_world_dict(xml, gripper=True))
    full_model, _ = build_world(full_spec)
    assert full_model.nu == bare_model.nu + 1
    assert full_model.njnt == bare_model.njnt + 2


def test_gripper_closes_and_opens():
    """ur5e + 2FG7：ctrl=0.038 张开（两 slide qpos→0.019），回 0 闭合。"""
    xml = _menagerie_xml("universal_robots_ur5e")
    spec = parse_world(_world_dict(xml))
    model, index = build_world(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ridx = index.robots["arm"]
    grip_act = ridx.actuators[f"{GRIPPER_PREFIX}gripper_act"]
    lid = ridx.joints[f"{GRIPPER_PREFIX}left_slide"]
    rid = ridx.joints[f"{GRIPPER_PREFIX}right_slide"]

    def slide_qpos():
        return (float(data.qpos[model.jnt_qposadr[lid]]),
                float(data.qpos[model.jnt_qposadr[rid]]))

    # 非夹爪执行器目标设为当前关节角（保持臂姿态，隔离夹爪运动）
    for short, aid in ridx.actuators.items():
        if not short.startswith(GRIPPER_PREFIX):
            jid = model.actuator_trnid[aid, 0]
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    l0, r0 = slide_qpos()
    assert abs(l0) < 1e-6 and abs(r0) < 1e-6     # 出生零位（闭合姿态，ctrl=0）

    data.ctrl[grip_act] = GRIPPERS["onrobot_2fg7"]["open"]
    for _ in range(400):
        mujoco.mj_step(model, data)
    lo, ro = slide_qpos()
    assert lo > 0.015 and ro > 0.015, (lo, ro)    # ctrl=0.038 两指张开到位

    data.ctrl[grip_act] = GRIPPERS["onrobot_2fg7"]["close"]
    for _ in range(400):
        mujoco.mj_step(model, data)
    lc, rc = slide_qpos()
    assert lc < 0.005 and rc < 0.005, (lc, rc)    # 回 ctrl=0 重新并拢


def test_tactile_sensor_on_gripper_finger():
    """触觉传感器可挂到夹爪手指 body（先挂爪、后注传感器）。"""
    xml = _menagerie_xml("universal_robots_ur5e")
    sensors = [{"name": "lf_touch", "type": "tactile",
                "body": f"{GRIPPER_PREFIX}left_finger", "radius": 0.03}]
    spec = parse_world(_world_dict(xml, sensors=sensors))
    model, index = build_world(spec)
    assert "lf_touch" in index.robots["arm"].sensors
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                            "arm/lf_touch")
    assert sid >= 0


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_mount_flushes_on_flange(label, key):
    """2FG7 后法兰面与机器人法兰贴合（无悬浮间隙、无明显穿插）。

    历史回归：旧 mount 误用 Rot_x(180°) 且 z=0.1494/y=-0.028，夹爪被转到
    site ±y 方向垂在腕部壳体下方，与法兰留约 9cm 间隙。这里把"贴合"固化为
    几何断言：夹爪 base 后端面与腕部法兰面在法兰半径内轴向间距≈0。
    """
    xml = _menagerie_xml(key)
    spec = parse_world(_world_dict(xml))
    model, _ = build_world(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "arm/attachment_site")
    sp = data.site_xpos[sid]
    sq = data.site_xmat[sid].reshape(3, 3)

    def to_site(points):
        return (points - sp) @ sq

    def geom_points(gid, n=30000):
        return _geom_points(model, data, gid, n)

    # 腕部：site 所在 body 及其父 body（panda 挂点是空 body，法兰在 link7）
    wrist_bodies = {model.site_bodyid[sid],
                    model.body_parentid[model.site_bodyid[sid]]}
    wrist_z, grip_z = [], []
    for gid in range(model.ngeom):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[gid]) or ""
        pts = geom_points(gid)
        if pts is None:
            continue
        s = to_site(pts)
        r = np.hypot(s[:, 0], s[:, 1])
        if model.geom_bodyid[gid] in wrist_bodies:
            wrist_z.append(s[r < 0.030, 2])
        if "/gripper/gripper_base" in bn:
            grip_z.append(s[r < 0.030, 2])
    wrist_front = np.max(np.concatenate(wrist_z))
    grip_back = np.min(np.concatenate(grip_z))
    gap = grip_back - wrist_front
    assert -0.003 <= gap <= 0.004, (
        f"{label}: 夹爪后端面与法兰面轴向间距 {gap:.4f}m，应≈0")


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_fingers_point_along_flange_axis(label, key):
    """指尖沿挂点 +z（法兰法向）伸出、与法兰同轴、开合沿 site x。

    旧 Rot_x(180°) 标定下指尖沿 site ±y（UR home 时竖直向下），是本 bug 的
    朝向特征；该断言确保不再回退。
    """
    xml = _menagerie_xml(key)
    spec = parse_world(_world_dict(xml))
    model, _ = build_world(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "arm/attachment_site")
    sp = data.site_xpos[sid]
    sq = data.site_xmat[sid].reshape(3, 3)

    def site_pos(site_name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return (data.site_xpos[i] - sp) @ sq

    lp = site_pos("arm/gripper/left_pad_site")
    rp = site_pos("arm/gripper/right_pad_site")
    mid_xy = (lp + rp) / 2
    # 指尖朝 site +z 伸出（远离法兰 >10cm），而非沿 ±y 垂下
    assert lp[2] > 0.10 and rp[2] > 0.10, (label, lp, rp)
    assert abs(mid_xy[0]) < 0.005 and abs(mid_xy[1]) < 0.005, (label, mid_xy)
    # 两指开合沿 site x（分离约 25mm），z 同高
    assert abs(lp[0] - rp[0]) > 0.020, (label, lp, rp)
    assert abs(lp[2] - rp[2]) < 0.005 and abs(lp[1] - rp[1]) < 0.005


def test_mount_table_per_arm_values():
    """mount 表逐臂标定值黄金断言（防回退到错误的共用值）。"""
    from simovia.grippers import mount_for
    assert mount_for("onrobot_2fg7", "x/ur5e.xml")["pos"] == [0, 0, 0.1094]
    assert mount_for("onrobot_2fg7", "x/ur10e.xml")["pos"] == [0, 0, 0.1264]
    assert mount_for("onrobot_2fg7",
                     "x/xarm7_nohand.xml")["pos"] == [0, 0, 0.1094]
    assert mount_for("onrobot_2fg7",
                     "x/panda_nohand.xml")["pos"] == [0, 0, 0.1092]
    # 朝向统一 Rot_x(-90°)，wxyz
    for arm in ("ur5e.xml", "ur10e.xml", "xarm7_nohand.xml",
                "panda_nohand.xml"):
        q = mount_for("onrobot_2fg7", f"x/{arm}")["quat"]
        assert abs(q[0] - 0.7071) < 1e-3 and abs(q[1] + 0.7071) < 1e-3
        assert q[2] == 0 and q[3] == 0
    # Windows 风格反斜杠路径同样按 basename 命中（跨平台 YAML 复制）
    assert mount_for("onrobot_2fg7",
                     r"C:\models\zoo\ur5e\ur5e.xml")["pos"] == [0, 0, 0.1094]


def test_unknown_mount_robot_raises():
    """夹爪挂到未标定的机器人（且未写 pos/quat）→ 清晰 WorldError。"""
    from simovia.grippers import mount_for
    with pytest.raises(WorldError, match="标定安装位姿"):
        mount_for("onrobot_2fg7", "/some/path/acme_arm.xml")
    # 显式覆盖 pos/quat 则放行（非标准臂手动指定）
    xf = mount_for("onrobot_2fg7", "/some/path/acme_arm.xml",
                   override_pos=[0, 0, 0.1],
                   override_quat=[1, 0, 0, 0])
    assert xf["pos"] == [0, 0, 0.1]


def test_no_attachment_site_raises(tmp_path):
    """模型既没有挂点、也没有可替换末端/腕部连杆 → 错误列出可用 site。"""
    xml = tmp_path / "plain_arm.xml"
    xml.write_text(
        "<mujoco model='plain_arm'>"
        "<worldbody><body name='base'>"
        "<geom type='capsule' size='0.03 0.2' rgba='0.5 0.5 0.5 1'/>"
        "<joint name='j1' type='hinge' axis='0 1 0' range='-1.57 1.57'/>"
        "<body name='link2'>"
        "<geom type='capsule' size='0.02 0.15' pos='0 0 0.15'/>"
        "<joint name='j2' type='hinge' axis='0 1 0' range='-1.57 1.57'/>"
        "</body></body></worldbody>"
        "<actuator><position name='a1' joint='j1' kp='50'/>"
        "<position name='a2' joint='j2' kp='50'/></actuator>"
        "</mujoco>", encoding="utf-8")
    spec = parse_world(_world_dict(str(xml)))
    with pytest.raises(WorldError, match="没有挂点 site"):
        build_world(spec)


def test_missing_site_raises():
    """挂点 site 不在机器人模型里 → 错误列出可用 site。"""
    xml = _menagerie_xml("universal_robots_ur5e")
    d = _world_dict(xml)
    d["robots"][0]["gripper"]["site"] = "no_such_site"
    spec = parse_world(d)
    with pytest.raises(WorldError, match="没有挂点 site"):
        build_world(spec)


def test_manifest_reports_gripper_sites_and_catalog():
    """M6 P2：get_robot_manifest 返回夹爪信息、挂点清单与夹爪库目录。"""
    import yaml as _yaml
    from simovia.mcp_server import SimoviaSession

    xml = _menagerie_xml("universal_robots_ur5e")
    wd = _world_dict(xml, gripper=True, sensors=[
        {"name": "pad_touch", "type": "tactile",
         "body": "gripper/left_finger", "radius": 0.02}])
    text = _yaml.safe_dump(wd, sort_keys=False)

    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_text": text, "base_dir": "/tmp"})
        m = s.call_tool("get_robot_manifest", {})
        arm = m["robots"][0]
        # 夹爪信息
        assert arm["gripper"] is not None
        assert arm["gripper"]["model"] == "onrobot_2fg7"
        assert arm["gripper"]["info"]["close"] == GRIPPERS["onrobot_2fg7"]["close"]
        assert arm["gripper"]["info"]["actuator"] == "gripper/gripper_act"
        # 挂点含机器人自带 attachment_site 与夹爪内 site
        assert "attachment_site" in arm["sites"]
        assert "gripper/gripper_tcp" in arm["sites"]
        # 夹爪执行器在清单里
        assert "gripper/gripper_act" in {a["name"] for a in arm["actuators"]}
        # 触觉传感器出现在清单
        assert {x["name"] for x in arm["sensors"]} == {"pad_touch"}
        # 夹爪库目录
        cat = {g["key"]: g for g in m["gripper_catalog"]}
        assert "onrobot_2fg7" in cat
        assert "ur5e.xml" in cat["onrobot_2fg7"]["supported_robots"]
    finally:
        s.close()


def test_manifest_gripper_none_when_bare():
    """裸臂 manifest 的 gripper 为 None。"""
    import yaml as _yaml
    from simovia.mcp_server import SimoviaSession

    xml = _menagerie_xml("universal_robots_ur5e")
    text = _yaml.safe_dump(_world_dict(xml, gripper=False), sort_keys=False)
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_text": text, "base_dir": "/tmp"})
        arm = s.call_tool("get_robot_manifest", {})["robots"][0]
        assert arm["gripper"] is None
        assert "attachment_site" in arm["sites"]
    finally:
        s.close()


# ---------------------------------------------------------------- 新增夹爪

@pytest.mark.parametrize("key", NEW_GRIPPERS)
def test_new_gripper_model_compiles(key):
    """三个自包含夹爪 XML 可独立编译，root body 为 gripper_base。"""
    path = ensure_gripper(key)
    model = mujoco.MjModel.from_xml_path(str(path))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    assert body > 0 and model.body_parentid[body] == 0, (
        f"{key}: 应只有一个 gripper_base 根 body，实际 id={body}")
    info = GRIPPERS[key]
    n_act = 1 if "actuator" in info else len(info["actuators"])
    assert model.nu == n_act
    for j in info["joints"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0


@pytest.mark.parametrize("key", NEW_GRIPPERS)
@pytest.mark.parametrize("label,arm_key", ARMS)
def test_new_gripper_attaches_to_four_arms(label, arm_key, key):
    """四臂挂三个新夹爪均编译成功，执行器/关节并入 gripper/ 命名空间。"""
    xml = _menagerie_xml(arm_key)
    _, index = build_world(
        parse_world(_world_dict(xml, gripper_key=key)))
    ridx = index.robots["arm"]
    info = GRIPPERS[key]

    assert f"arm/{GRIPPER_PREFIX}gripper_base" in index.bodies
    acts = ([info["actuator"]] if "actuator" in info else info["actuators"])
    for a in acts:
        assert f"{GRIPPER_PREFIX}{a}" in ridx.actuators, (label, key, a)
    for j in info["joints"]:
        assert f"{GRIPPER_PREFIX}{j}" in ridx.joints, (label, key, j)


@pytest.mark.parametrize("key", NEW_GRIPPERS)
@pytest.mark.parametrize("label,arm_key", ARMS)
def test_new_gripper_mount_flushes_and_axis_aligned(label, arm_key, key):
    """新夹爪安装后：与法兰同轴（x=y=0）、贴合（|gap|<2mm 且不穿透）、
    手指沿挂点 +z 伸出。"""
    xml = _menagerie_xml(arm_key)
    model, _ = build_world(
        parse_world(_world_dict(xml, gripper_key=key)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "arm/attachment_site")
    sp, sq = _site_frame(model, data)

    # 1) 两法兰同轴：root body 原点落在 site z 轴上
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             "arm/gripper/gripper_base")
    bp = (data.xpos[base] - sp) @ sq
    assert abs(bp[0]) < 1e-6 and abs(bp[1]) < 1e-6, (label, key, bp)

    # 2) 贴合：中央对接面（半径 3cm 内）vs 腕部法兰面
    wrist_bodies = {model.site_bodyid[sid],
                    model.body_parentid[model.site_bodyid[sid]]}
    wrist_z, wrist_plane_r, grip_s = [], [], []
    for gid in range(model.ngeom):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[gid]) or ""
        pts = _geom_points(model, data, gid)
        if pts is None:
            continue
        s = (pts - sp) @ sq
        r = np.hypot(s[:, 0], s[:, 1])
        if model.geom_bodyid[gid] in wrist_bodies:
            wrist_z.append(s[r < 0.030, 2])
            wrist_plane_r.append(r[np.abs(s[:, 2]) < 0.010])
        if "/gripper/" in bn:
            grip_s.append(s)
    wrist_front = np.max(np.concatenate(wrist_z))
    gs = np.concatenate(grip_s)
    grip_back = np.min(gs[np.hypot(gs[:, 0], gs[:, 1]) < 0.030, 2])
    gap = grip_back - wrist_front
    assert -0.002 <= gap <= 0.004, (
        f"{label}/{key}: 夹爪中央后端面与法兰面间距 {gap:.4f}m，应≈0 且不穿透")

    # 2b) 不穿透：越过法兰平面的夹爪点必须落在腕部壳体半径之外
    wrist_r_max = np.max(np.concatenate(wrist_plane_r))
    pen = gs[:, 2] < -0.0005
    if pen.any():
        pen_r = np.hypot(gs[pen, 0], gs[pen, 1])
        assert np.all(pen_r > wrist_r_max + 0.005), (
            f"{label}/{key}: 夹爪在腕部半径内穿入法兰，r={pen_r.min():.4f}")

    # 3) 手指沿 site +z 伸出：探针 z 足够大且明显大于横向偏移
    kind, names = Z_PROBES[key]
    for name in names:
        obj = (mujoco.mjtObj.mjOBJ_SITE if kind == "site"
               else mujoco.mjtObj.mjOBJ_BODY)
        oid = mujoco.mj_name2id(model, obj, f"arm/gripper/{name}")
        assert oid >= 0, (key, name)
        xyz = ((data.site_xpos[oid] - sp) @ sq if kind == "site"
               else (data.xpos[oid] - sp) @ sq)
        assert xyz[2] > 0.10, (label, key, name, xyz)
        assert xyz[2] > 2.0 * max(abs(xyz[0]), abs(xyz[1])), (
            label, key, name, xyz)


def _hold_arm(model, data, ridx):
    for short, aid in ridx.actuators.items():
        if not short.startswith(GRIPPER_PREFIX) \
                and int(model.actuator_trntype[aid]) == 0:
            jid = model.actuator_trnid[aid, 0]
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]


@pytest.mark.parametrize("key", ["robotiq_2f85", "umi_gripper"])
def test_parallel_gripper_actuator_moves_fingers(key):
    """单执行器平行夹爪：open→close 两指实际相向移动（位移>2cm）。"""
    xml = _menagerie_xml("universal_robots_ur5e")
    model, index = build_world(
        parse_world(_world_dict(xml, gripper_key=key, pos=(0, 0, 0.8))))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ridx = index.robots["arm"]
    info = GRIPPERS[key]
    aid = ridx.actuators[f"{GRIPPER_PREFIX}{info['actuator']}"]
    fingers = {
        "robotiq_2f85": ["left_pad", "right_pad"],
        "umi_gripper": ["left_finger_holder", "right_finger_holder"],
    }[key]
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              f"arm/gripper/{b}") for b in fingers]

    def settle(value, steps=900):
        _hold_arm(model, data, ridx)
        data.ctrl[aid] = value
        for _ in range(steps):
            mujoco.mj_step(model, data)
        return [data.xpos[b].copy() for b in bids]

    po = settle(info["open"])
    pc = settle(info["close"])
    d = [np.linalg.norm(a - b) for a, b in zip(po, pc)]
    assert all(x > 0.02 for x in d), (key, d)
    sep_o = np.linalg.norm(po[0] - po[1])
    sep_c = np.linalg.norm(pc[0] - pc[1])
    # 2F85 开爪时 pad 间距更大；UMI 的指 body 原点在 qpos=0 重合、闭合后分开，
    # 两者都体现为“开合两态间距变化显著”
    assert abs(sep_o - sep_c) > 0.03, (key, sep_o, sep_c)


def test_leap_all_actuators_drive_joints():
    """LEAP 16 个执行器逐个可驱动，全部关节跟随到目标附近。"""
    xml = _menagerie_xml("universal_robots_ur5e")
    model, index = build_world(
        parse_world(_world_dict(xml, gripper_key="leap_hand",
                                pos=(0, 0, 0.8))))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ridx = index.robots["arm"]
    _hold_arm(model, data, ridx)
    info = GRIPPERS["leap_hand"]
    for a in info["actuators"]:
        data.ctrl[ridx.actuators[f"{GRIPPER_PREFIX}{a}"]] = 0.5
    for _ in range(1200):
        mujoco.mj_step(model, data)
    qpos = {
        j: float(data.qpos[model.jnt_qposadr[ridx.joints[f"{GRIPPER_PREFIX}{j}"]]])
        for j in info["joints"]
    }
    assert all(q > 0.40 for q in qpos.values()), qpos


GRIPPER_CONTROLLER = """\
from simovia.controller import Robot

robot = Robot()
for name, value in {targets!r}.items():
    robot.set(name, value)
while robot.step():
    pass
robot.close()
"""


def test_g1_strip_hands_and_remount_gripper():
    """人形自带灵巧手删除 + 腕部自动挂点 + 重挂夹爪。

    回归点：MjSpec.bodies 里包含 worldbody，而 worldbody 的 parent 在 C 侧
    为 NULL，detect_end_effectors 若不跳过会直接段错误（进程崩溃）。
    """
    import pathlib

    g1 = pathlib.Path(_menagerie_xml("unitree_g1"))
    model_path = g1.parent / "g1_with_hands.xml"
    if not model_path.is_file():
        pytest.skip("G1 with hands 资产缺失")

    wd = _world_dict(str(model_path), gripper=True)
    wd["robots"][0]["strip_hands"] = True
    model, index = build_world(parse_world(wd))

    # 两手都被识别并标记删除
    ee = index.end_effectors["arm"]
    assert {g["key"] for g in ee} == {"left_hand", "right_hand"}
    assert all(g["stripped"] for g in ee)
    # 腕部生成了自动挂点 site（attach 前缀 arm/）
    for side in ("left", "right"):
        site = f"arm/{side}_hand_attachment_site"
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0
    # 手部执行器已随子树删除，夹爪执行器已挂上
    acts = set(index.robots["arm"].actuators)
    assert not any("hand" in n for n in acts), sorted(acts)
    assert GRIPPER_PREFIX + "gripper_act" in acts
    assert "left_wrist_yaw_joint" in index.robots["arm"].joints


def test_end_effectors_empty_for_arm_without_hands():
    """普通机械臂（panda_nohand）没有自带末端，检测结果为空且不崩。"""
    xml = _menagerie_xml("franka_emika_panda")
    _, index = build_world(parse_world(_world_dict(xml)))
    assert index.end_effectors["arm"] == []


def test_g1_without_hands_gets_wrist_sites_for_gripper():
    """无手人形（zoo 的 unitree_g1 / g1.xml）装夹爪：按腕部连杆自动生成挂点。

    回归点：g1.xml 只有 29 自由度本体，既没有手 body 也没有 attachment_site，
    以前会报"模型里没有挂点 site 'attachment_site'"，夹爪既装不上也卸不掉。
    """
    from simovia.world.schema import resolve_zoo_models

    wd = {"name": "g1_auto_site", "robots": [
        {"name": "arm", "model": "zoo://unitree_g1",
         "gripper": {"model": "leap_hand"}}]}
    resolve_zoo_models(wd)
    model, index = build_world(parse_world(wd))

    for side in ("left", "right"):
        site = f"arm/{side}_hand_attachment_site"
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0
    # 左手（leap_hand 声明的侧别）被自动选中挂上
    assert GRIPPER_PREFIX + "if_mcp_act" in index.robots["arm"].actuators
    # 腕部残留的手套装饰网格已清掉，避免和新手穿插
    used = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(d))
            for d in model.geom_dataid if d >= 0}
    assert not any(n and "rubber_hand" in n for n in used), used


def test_g1_with_hands_auto_replaces_only_matching_side():
    """双手人形装左手夹爪：只替换 left_hand，右手保留。"""
    import pathlib

    g1 = pathlib.Path(_menagerie_xml("unitree_g1"))
    model_path = g1.parent / "g1_with_hands.xml"
    if not model_path.is_file():
        pytest.skip("G1 with hands 资产缺失")

    wd = _world_dict(str(model_path), gripper_key="leap_hand")
    model, index = build_world(parse_world(wd))

    ee = {g["key"]: g for g in index.end_effectors["arm"]}
    assert ee["left_hand"]["stripped"] is True
    assert ee["right_hand"]["stripped"] is False
    joints = index.robots["arm"].joints
    assert any("right_hand" in n for n in joints)
    assert not any("left_hand" in n for n in joints)
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE,
        "arm/left_hand_attachment_site") >= 0



@pytest.mark.parametrize("key", NEW_GRIPPERS)
def test_controller_can_drive_gripper_actuators(key, tmp_path):
    """控制器 robot.set("gripper/...") 可直接驱动新夹爪（真实子进程）。"""
    from simovia.kernel.sim import Simulation

    info = GRIPPERS[key]
    if "actuator" in info:
        targets = {f"gripper/{info['actuator']}": info["close"]}
    else:
        targets = {f"gripper/{a}": 0.5 for a in info["actuators"]}
    ctrl = tmp_path / f"ctrl_{key}.py"
    ctrl.write_text(GRIPPER_CONTROLLER.format(targets=targets),
                    encoding="utf-8")

    xml = _menagerie_xml("universal_robots_ur5e")
    wd = _world_dict(xml, gripper_key=key, pos=(0, 0, 0.8))
    wd["gravity"] = [0, 0, 0]          # 无重力：无需保持臂姿态即可隔离夹爪
    wd["robots"][0]["controller"] = str(ctrl)
    spec = parse_world(wd)
    with Simulation(spec) as sim:
        stats = sim.run(max_ticks=400)
        qpos = {
            j: float(sim.data.qpos[
                sim.model.jnt_qposadr[
                    sim.index.robots["arm"].joints[f"{GRIPPER_PREFIX}{j}"]]])
            for j in info["joints"]
        }
    assert stats["ticks"] == 400, stats
    assert stats["crashed"] == [], stats
    if key == "robotiq_2f85":
        # 耦合机构里 coupler 是随动关节（闭合时仍近零），只看两个主动关节
        for j in ("right_driver_joint", "left_driver_joint"):
            assert abs(qpos[j]) > 0.2, qpos
    elif key == "umi_gripper":
        assert all(q > 0.03 for q in qpos.values()), qpos
    else:
        assert all(q > 0.40 for q in qpos.values()), qpos


def test_manifest_reports_multi_actuator_gripper():
    """LEAP 在 manifest 中以 actuators/poses 展开；2FG7 单执行器接口不变。"""
    import yaml as _yaml
    from simovia.mcp_server import SimoviaSession

    xml = _menagerie_xml("universal_robots_ur5e")
    text = _yaml.safe_dump(
        _world_dict(xml, gripper_key="leap_hand"), sort_keys=False)
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_text": text, "base_dir": "/tmp"})
        m = s.call_tool("get_robot_manifest", {})
        info = m["robots"][0]["gripper"]["info"]
        assert info["key"] == "leap_hand"
        assert info["actuators"] == [
            f"gripper/{a}" for a in GRIPPERS["leap_hand"]["actuators"]]
        assert info["poses"] == GRIPPERS["leap_hand"]["poses"]
        assert "actuator" not in info

        cat = {g["key"]: g for g in m["gripper_catalog"]}
        for key in NEW_GRIPPERS:
            assert key in cat
            assert len(cat[key]["supported_robots"]) == 4
        assert "actuators" in cat["leap_hand"]
        assert "poses" in cat["leap_hand"]
        # 现有 2FG7 仍是单执行器 open/close 描述，行为不变
        assert cat["onrobot_2fg7"]["open"] == GRIPPERS["onrobot_2fg7"]["open"]
        assert cat["onrobot_2fg7"]["close"] == GRIPPERS["onrobot_2fg7"]["close"]
        assert "actuators" not in cat["onrobot_2fg7"]
    finally:
        s.close()
