"""「造机器人」面板：离屏实例化、旋转矩阵联动、增删连杆、保存回读。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def dlg(qapp):
    from simovia.gui.robot_builder import RobotBuilderDialog
    d = RobotBuilderDialog()
    yield d
    d.close()
    d.deleteLater()


def _tree_names(dlg):
    names = []
    stack = [dlg.tree.topLevelItem(i) for i in range(dlg.tree.topLevelItemCount())]
    while stack:
        it = stack.pop(0)
        names.append(it.text(0))
        stack.extend(it.child(j) for j in range(it.childCount()))
    return names


def test_default_robot_and_matrix(dlg):
    """默认示例可见，选中 link1 后矩阵表已填值且 R_rel 为单位阵。"""
    assert _tree_names(dlg) == ["base", "link1", "link2"]
    assert dlg.tb_rel.item(0, 0) is not None
    assert float(dlg.tb_rel.item(0, 0).text()) == pytest.approx(1.0, abs=1e-3)
    assert "R = Rz(0°) · Ry(0°) · Rx(0°)" in dlg.lbl_formula.text()


def test_rpy_edit_updates_matrix_and_preview(dlg):
    """改 rpy 后 R_rel/R_world 与子连杆世界位置同步变化。"""
    dlg._select_link("link1")
    dlg.sp_rpy[2].setValue(90.0)          # yaw 90°
    assert float(dlg.tb_rel.item(0, 1).text()) == pytest.approx(-1.0, abs=1e-3)
    assert float(dlg.tb_rel.item(1, 0).text()) == pytest.approx(1.0, abs=1e-3)
    # link2 的原点在 link1 的 z 轴上：link1 绕自身 z 轴转 90° 不改变
    # 自身原点、也不改变 link2 沿 z 轴的距离（世界位置仍是 0.500）
    dlg._select_link("link2")
    assert "+0.500" in dlg.lbl_pose.text()
    # link2 的 R_world 继承了 link1 的 yaw
    assert float(dlg.tb_world.item(0, 1).text()) == pytest.approx(-1.0, abs=1e-3)


def test_add_and_delete_child(dlg):
    dlg._select_link("link2")
    dlg._add_child()
    names = _tree_names(dlg)
    assert len(names) == 4
    new = names[-1]
    assert new.startswith("link")
    dlg._select_link(new)
    dlg._del_link()                       # 叶子可直接删
    assert new not in _tree_names(dlg)


def test_delete_parent_reparents_children(dlg):
    """删中间连杆时子连杆上移，树不断裂。"""
    dlg._select_link("link1")
    dlg._del_link()
    names = _tree_names(dlg)
    assert "link1" not in names and "link2" in names
    from simovia.robot_spec import _validate_tree
    _validate_tree(dlg.robot)             # 结构仍合法


def test_matrix_matches_kinematics(dlg):
    """面板显示的矩阵必须与 MuJoCo 编译后的 xmat 一致。"""
    import mujoco
    from simovia.robot_spec import build_robot_spec, link_frames
    dlg.sp_rpy[0].setValue(15.0)
    dlg.sp_rpy[1].setValue(-30.0)
    dlg.sp_rpy[2].setValue(45.0)
    model = build_robot_spec(dlg.robot).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for f in link_frames(dlg.robot):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f.name)
        assert np.allclose(data.xmat[bid].reshape(3, 3), f.R_world, atol=1e-9)


def test_save_load_roundtrip(dlg, tmp_path):
    """保存的 YAML 能被 load_robot 原样读回，且连杆姿态一致。"""
    import yaml
    from simovia.robot_spec import dump_robot, load_robot
    dlg._select_link("link2")
    dlg.sp_rpy[1].setValue(30.0)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(dump_robot(dlg.robot), allow_unicode=True),
                 encoding="utf-8")
    r2 = load_robot(p)
    assert [lk.name for lk in r2.links] == _tree_names(dlg)
    q1 = [lk.quat for lk in dlg.robot.links]
    q2 = [lk.quat for lk in r2.links]
    assert np.allclose(q1, q2, atol=1e-6)
