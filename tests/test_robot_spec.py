"""从零造机器人的连杆树格式：解析校验、旋转矩阵、编译与物理一致性。"""
import math

import mujoco
import numpy as np
import pytest

from simovia.errors import WorldError
from simovia.robot_spec import (
    build_robot_spec, dump_robot, euler_to_matrix, link_frames,
    load_robot, matrix_to_euler_deg, matrix_to_quat, parse_robot,
    quat_to_matrix,
)

TWO_LINK = {
    "name": "two_link",
    "links": [
        {"name": "base", "pos": [0, 0, 0.1],
         "geom": {"type": "cylinder", "size": [0.06, 0.05],
                  "pos": [0, 0, 0.05]}},
        {"name": "link1", "parent": "base", "pos": [0, 0, 0.1],
         "rpy": [0, 0, 0], "mass": 1.0,
         "joint": {"type": "revolute", "axis": [0, 0, 1],
                   "range": [-170, 170]},
         "geom": {"type": "box", "size": [0.03, 0.03, 0.12],
                  "pos": [0, 0, 0.12]}},
        {"name": "link2", "parent": "link1", "pos": [0, 0, 0.24],
         "rpy": [0, 90, 0], "mass": 0.8,
         "joint": {"type": "revolute", "axis": [0, 0, 1],
                   "range": [-120, 120], "kp": 80},
         "geom": {"type": "sphere", "size": [0.04],
                  "pos": [0, 0, 0.24]}},
    ],
}


# ---------------------------------------------------------------- 旋转数学

def test_euler_matrix_known_values():
    """rpy=[0,0,90] 必须得到 Rz(90)，且矩阵/欧拉角可互逆。"""
    R = euler_to_matrix([0, 0, 90])
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)
    R2 = euler_to_matrix([30, -20, 45])
    assert matrix_to_euler_deg(R2) == pytest.approx((30, -20, 45), abs=1e-4)
    assert np.allclose(quat_to_matrix(matrix_to_quat(R2)), R2, atol=1e-9)
    # 正交性：RᵀR = I，det = +1
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R2) == pytest.approx(1.0)


# ---------------------------------------------------------------- 解析校验

def test_parse_ok_and_actuator_names():
    from simovia.robot_spec import actuator_names
    r = parse_robot(TWO_LINK)
    assert [lk.name for lk in r.links] == ["base", "link1", "link2"]
    assert r.link("base").joint.type == "fixed"
    assert actuator_names(r) == ["link1", "link2"]


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["links"].append(dict(d["links"][1])), "连杆名重复"),
    (lambda d: d["links"][1].update(parent="ghost"), "不存在的父连杆"),
    (lambda d: d["links"][0].update(parent="link2"), "恰有一个根连杆"),
    (lambda d: d["links"][1]["geom"].update(type="donut"), "未知几何类型"),
    (lambda d: d["links"][1]["joint"].update(axis=[0, 0, 0]), "零向量"),
    (lambda d: d["links"][1]["joint"].update(range=[1, -1]), "下限必须小于上限"),
    (lambda d: d.update(links=[]), "缺少 dh 参数表或 links 列表"),
])
def test_parse_errors_have_context(mutate, match):
    """结构错误要报清楚（含连杆位置提示），初学者能自己改。"""
    import copy
    d = copy.deepcopy(TWO_LINK)
    mutate(d)
    with pytest.raises(WorldError, match=match):
        parse_robot(d)


def test_cycle_detection():
    d = {"name": "loop", "links": [
        {"name": "a", "parent": None},
        {"name": "b", "parent": "c"},
        {"name": "c", "parent": "b"},
    ]}
    with pytest.raises(WorldError, match="有环"):
        parse_robot(d)


# ---------------------------------------------------------------- 位姿矩阵

def test_link_frames_compose():
    """世界位姿 = 父世界位姿 ∘ 相对位姿（旋转矩阵逐级左乘）。"""
    r = parse_robot(TWO_LINK)
    fr = {f.name: f for f in link_frames(r)}
    assert np.allclose(fr["base"].p_world, [0, 0, 0.1])
    assert np.allclose(fr["link1"].p_world, [0, 0, 0.2])
    # link2 的平移在父系（link1，未旋转）里量：z 直接累加；
    # 它自己的 rpy=[0,90,0] 只影响其子连杆与几何，不改变自身原点
    assert np.allclose(fr["link2"].p_world, [0, 0, 0.44], atol=1e-9)
    assert np.allclose(fr["link2"].R_world,
                       euler_to_matrix([0, 90, 0]), atol=1e-9)


def test_frames_match_mujoco_kinematics():
    """教学矩阵必须与 MuJoCo 真实运动学一致（否则显示会骗人）。"""
    r = parse_robot(TWO_LINK)
    model = build_robot_spec(r).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for f in link_frames(r):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f.name)
        assert np.allclose(data.xpos[bid], f.p_world, atol=1e-9), f.name
        assert np.allclose(data.xmat[bid].reshape(3, 3), f.R_world,
                           atol=1e-9), f.name


# ---------------------------------------------------------------- 编译与物理

def test_build_compile_structure():
    r = parse_robot(TWO_LINK)
    model = build_robot_spec(r).compile()
    assert model.nbody == 4                       # world + 3 连杆
    assert model.njnt == 2                        # 两个 revolute
    assert model.nu == 2                          # 两个位置执行器
    assert model.jnt_range[0] == pytest.approx(
        [-math.radians(170), math.radians(170)])
    for name in ("base_geom", "link1_geom", "link2_geom"):
        assert model.geom(name).id >= 0


def test_position_actuator_drives_joint():
    """位置执行器闭环：ctrl=0.5rad 后关节应转到 0.5 附近。"""
    r = parse_robot(TWO_LINK)
    model = build_robot_spec(r).compile()
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "link1")
    qadr = model.jnt_qposadr[jid]
    data.ctrl[0] = 0.5
    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert data.qpos[qadr] == pytest.approx(0.5, abs=0.05)


def test_free_root_builds():
    """浮动基座：根连杆 free 关节 -> 7 维 qpos。"""
    d = {"name": "floater", "links": [
        {"name": "body", "joint": {"type": "free"},
         "geom": {"type": "box", "size": [0.1, 0.1, 0.1]}},
        {"name": "leg", "parent": "body", "pos": [0, 0, -0.15],
         "joint": {"type": "prismatic", "axis": [0, 0, 1],
                   "range": [-0.1, 0.1]},
         "geom": {"type": "cylinder", "size": [0.02, 0.1]}},
    ]}
    r = parse_robot(d)
    model = build_robot_spec(r).compile()
    assert model.nq == 8 and model.nv == 7        # free(7) + slide(1)
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_dump_roundtrip(tmp_path):
    r = parse_robot(TWO_LINK)
    doc = dump_robot(r)
    assert doc["links"][2]["rpy"] == [0, 90, 0]
    r2 = parse_robot(doc)
    # 逐个 quat 比较（pytest.approx 不支持嵌套序列；quat 可能是 numpy 标量）
    assert np.allclose([lk.quat for lk in r2.links],
                       [lk.quat for lk in r.links])
    # 写盘后 load_robot 可读回
    import yaml
    p = tmp_path / "two_link.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    assert load_robot(p).name == "two_link"


def test_no_actuator_flag():
    d = {"name": "passive", "links": [
        {"name": "base", "geom": {"type": "box", "size": [0.05] * 3}},
        {"name": "arm", "parent": "base", "pos": [0, 0, 0.05],
         "joint": {"type": "revolute", "axis": [0, 1, 0],
                   "actuator": False},
         "geom": {"type": "box", "size": [0.02, 0.02, 0.1]}},
    ]}
    model = build_robot_spec(parse_robot(d)).compile()
    assert model.nu == 0


def test_load_robot_gbk_fallback(tmp_path):
    """中文 Windows 记事本存的 GBK 机器人 YAML 也能读（utf-8 优先兜底）。"""
    import yaml
    doc = {"name": "中文臂", "links": [
        {"name": "底座", "geom": {"type": "box", "size": [0.05] * 3}},
    ]}
    p = tmp_path / "gbk.yaml"
    p.write_bytes(yaml.safe_dump(doc, allow_unicode=True).encode("gbk"))
    r = load_robot(p)
    assert r.name == "中文臂"
    assert r.links[0].name == "底座"


# ---------------------------------------------------------------- DH 参数表

SCARA_DH = {
    "name": "scara",
    "dh": [
        {"theta": 0, "d": 0.20, "a": 0.25, "alpha": 0,
         "type": "revolute", "range": [-160, 160]},
        {"theta": 0, "d": 0.00, "a": 0.20, "alpha": 180,
         "type": "revolute", "range": [-135, 135]},
        {"theta": 0, "d": 0.10, "a": 0.00, "alpha": 0,
         "type": "prismatic", "range": [-0.05, 0.05]},
    ],
}

TWO_R_DH = {
    "name": "two_r",
    "dh": [
        {"theta": 0, "d": 0.0, "a": 0.30, "alpha": 0,
         "type": "revolute", "range": [-170, 170]},
        {"theta": 0, "d": 0.0, "a": 0.22, "alpha": 0,
         "type": "revolute", "range": [-150, 150]},
    ],
}


def test_dh_parse_structure():
    """DH 表展开为 base/jointN/linkN/flange 合法链，且标记 is_dh。"""
    r = parse_robot(SCARA_DH)
    assert r.is_dh and len(r.dh) == 3
    names = [lk.name for lk in r.links]
    assert names[0] == "base" and names[-1] == "flange"
    assert {"joint1", "link1", "joint2", "link2", "joint3",
            "link3"} <= set(names)
    # 关节载体 jointN 挂在 link(N-1)（首关节挂底座顶），linkN 挂 jointN
    by = {lk.name: lk for lk in r.links}
    assert by["joint1"].parent == "base"
    assert by["link1"].parent == "joint1"
    assert by["joint2"].parent == "link1"
    # 关节载体带骨架几何，固定 linkN 不带
    assert by["joint1"].geoms and len(by["joint1"].geoms) == 3  # 球+d+a
    assert by["joint2"].geoms and len(by["joint2"].geoms) == 2  # 球+a
    assert by["link1"].geoms is None and by["link1"].geom is None


def test_dh_scara_zero_pose_kinematics():
    """SCARA 零位法兰 = (a1+a2, 0, base+d1-d3)：α2=180 使 d3 朝下。"""
    r = parse_robot(SCARA_DH)
    fr = {f.name: f for f in link_frames(r)}
    assert np.allclose(fr["flange"].p_world, [0.45, 0.0, 0.18], atol=1e-9)
    # MuJoCo 编译后一致
    model = build_robot_spec(r).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "flange")
    assert np.allclose(data.xpos[bid], [0.45, 0.0, 0.18], atol=1e-9)


def test_dh_two_r_endpoint_matches_analytic():
    """平面 2R：θ=(30°,60°) 时法兰位置与解析正解一致（关重力收敛）。"""
    r = parse_robot(TWO_R_DH)
    model = build_robot_spec(r).compile()
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    data.ctrl[:] = [math.radians(30), math.radians(60)]
    for _ in range(3000):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert data.qpos == pytest.approx(
        [math.radians(30), math.radians(60)], abs=1e-3)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "flange")
    # x = .30cos30 + .22cos90 = 0.2598；y = .30sin30 + .22sin90 = 0.37
    assert data.xpos[bid][0] == pytest.approx(0.2598, abs=1e-2)
    assert data.xpos[bid][1] == pytest.approx(0.37, abs=1e-2)


def test_dh_actuators_and_ranges():
    """每个非固定行一个执行器 jointN；转动范围弧度、移动范围米。"""
    from simovia.robot_spec import actuator_names
    r = parse_robot(SCARA_DH)
    assert actuator_names(r) == ["joint1", "joint2", "joint3"]
    model = build_robot_spec(r).compile()
    assert model.nu == 3 and model.nq == 3
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")
    assert model.jnt_range[j1] == pytest.approx(
        [math.radians(-160), math.radians(160)])
    assert model.jnt_range[j3] == pytest.approx([-0.05, 0.05])


def test_dh_fixed_row_has_no_actuator():
    d = {"name": "fixed_row", "dh": [
        {"theta": 0, "d": 0.1, "a": 0.2, "alpha": 0, "type": "revolute"},
        {"theta": 0, "d": 0.0, "a": 0.1, "alpha": 90, "type": "fixed"},
    ]}
    r = parse_robot(d)
    from simovia.robot_spec import actuator_names
    assert actuator_names(r) == ["joint1"]
    model = build_robot_spec(r).compile()
    assert model.nu == 1 and model.nq == 1
    assert model.jnt_type[model.njnt - 1] != mujoco.mjtJoint.mjJNT_FREE


def test_dh_skeleton_geoms_are_visual_only():
    """骨架几何关闭碰撞（相邻关节球位置重叠，参与碰撞会自怼）。"""
    model = build_robot_spec(parse_robot(SCARA_DH)).compile()
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)


def test_dh_dump_roundtrip():
    r = parse_robot(SCARA_DH)
    doc = dump_robot(r)
    assert set(doc) == {"name", "dh"}
    assert doc["dh"][0] == {"theta": 0.0, "d": 0.20, "a": 0.25,
                            "alpha": 0.0, "type": "revolute",
                            "kp": 50.0, "kd": 1.0,
                            "range": [-160.0, 160.0]}
    r2 = parse_robot(doc)
    assert [r.__dict__ for r in r2.dh] == [r.__dict__ for r in r.dh]
    assert np.allclose(
        link_frames(r2)[-1].p_world, link_frames(r)[-1].p_world)


def test_dh_base_height_roundtrip():
    d = {"name": "tall", "base_height": 0.25, "dh": [
        {"theta": 0, "d": 0.1, "a": 0.2, "alpha": 0, "type": "revolute"}]}
    r = parse_robot(d)
    assert r.base_height == pytest.approx(0.25)
    doc = dump_robot(r)
    assert doc["base_height"] == pytest.approx(0.25)
    # 默认底座高不写盘
    assert "base_height" not in dump_robot(parse_robot(
        {"name": "d", "dh": d["dh"]}))


def test_links_dump_preserves_geoms_list():
    """回归：DH 展开（多几何 geoms）后保存不能丢几何。"""
    r = parse_robot(SCARA_DH)
    r.dh = None                            # 模拟高级模式展开
    doc = dump_robot(r)
    joint1 = next(x for x in doc["links"] if x["name"] == "joint1")
    assert "geoms" in joint1 and len(joint1["geoms"]) == 3
    r2 = parse_robot(doc)
    model = build_robot_spec(r2).compile()  # 无几何会报 moving body mass 错
    assert model.ngeom >= 8


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.update(dh=[]), "dh 需要非空列表"),
    (lambda d: d["dh"][0].update(type="ball"), "DH 关节类型只支持"),
    (lambda d: d["dh"][0].update(range=[1, -1]), "下限必须小于上限"),
])
def test_dh_parse_errors(mutate, match):
    import copy
    d = copy.deepcopy(TWO_R_DH)
    mutate(d)
    with pytest.raises(WorldError, match=match):
        parse_robot(d)


def test_dh_theta_offset_rotates_a_direction():
    """θ₀ 零位偏置让 a 横杆方向旋转：θ₀=90° 时首段沿 +y。"""
    d = {"name": "offset", "dh": [
        {"theta": 90, "d": 0.0, "a": 0.30, "alpha": 0,
         "type": "revolute", "range": [-170, 170]},
        {"theta": 0, "d": 0.0, "a": 0.20, "alpha": 0,
         "type": "revolute", "range": [-170, 170]}]}
    r = parse_robot(d)
    fr = {f.name: f for f in link_frames(r)}
    # frame1 原点（关节2 安装位）= Rz(90)·(a1,0,0) = (0, 0.3, 0)
    assert np.allclose(fr["link1"].p_world[:2], [0.0, 0.30], atol=1e-9)
