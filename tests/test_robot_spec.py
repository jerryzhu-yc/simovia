"""从零造机器人的连杆树格式：解析校验、旋转矩阵、编译与物理一致性。"""
import math

import mujoco
import numpy as np
import pytest

from agentsim.errors import WorldError
from agentsim.robot_spec import (
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
    from agentsim.robot_spec import actuator_names
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
    (lambda d: d.update(links=[]), "缺少 links 列表"),
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
    assert [lk.quat for lk in r2.links] == pytest.approx(
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
