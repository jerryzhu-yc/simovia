"""「造机器人」面板：DH 参数表为主、高级连杆编辑为辅。

离屏实例化、表格/树联动、旋转矩阵、保存回读。对话框内含实时预览，
创建即建 mujoco.Renderer：无 GL 环境整体跳过。
"""
import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.gl

#: 旧版两连杆 links 文档：高级模式回归用
LEGACY_ROBOT = {
    "name": "my_robot",
    "links": [
        {"name": "base", "pos": [0, 0, 0.08],
         "geom": {"type": "cylinder", "size": [0.07, 0.06],
                  "pos": [0, 0, 0.06], "color": "#4B5563"}},
        {"name": "link1", "parent": "base", "pos": [0, 0, 0.12],
         "mass": 1.0, "color": "#4A90D9",
         "joint": {"type": "revolute", "axis": [0, 0, 1],
                   "range": [-170, 170], "kp": 50, "kd": 1},
         "geom": {"type": "box", "size": [0.035, 0.035, 0.15],
                  "pos": [0, 0, 0.15]}},
        {"name": "link2", "parent": "link1", "pos": [0, 0, 0.30],
         "mass": 0.7, "color": "#E8833A",
         "joint": {"type": "revolute", "axis": [0, 0, 1],
                   "range": [-120, 120], "kp": 50, "kd": 1},
         "geom": {"type": "box", "size": [0.03, 0.03, 0.12],
                  "pos": [0, 0, 0.12]}},
    ],
}


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


@pytest.fixture()
def legacy_dlg(dlg, monkeypatch):
    """载入旧 links 文档、切到高级连杆编辑模式的对话框。"""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
    dlg._load_doc(copy.deepcopy(LEGACY_ROBOT))
    assert dlg.mode == "links"
    return dlg


def _tree_names(dlg):
    names = []
    stack = [dlg.tree.topLevelItem(i) for i in range(dlg.tree.topLevelItemCount())]
    while stack:
        it = stack.pop(0)
        names.append(it.text(0))
        stack.extend(it.child(j) for j in range(it.childCount()))
    return names


# ------------------------------------------------------------ DH 默认界面

def test_default_is_dh_scara(dlg):
    """打开面板默认是 DH 表 + SCARA 三行（2 转动 + 1 移动）。"""
    from simovia.gui.robot_builder import DEFAULT_SCARA
    assert dlg.mode == "dh" and dlg.robot.is_dh
    assert dlg.robot.name == DEFAULT_SCARA["name"]
    assert dlg.dh_table.rowCount() == 3
    assert [r.joint_type for r in dlg.robot.dh] == [
        "revolute", "revolute", "prismatic"]
    assert [r.a for r in dlg.robot.dh] == pytest.approx([0.25, 0.20, 0.0])
    # 矩阵表已填 joint1（单位阵）
    assert float(dlg.tb_rel.item(0, 0).text()) == pytest.approx(1.0, abs=1e-3)
    assert "joint1" in dlg.lbl_formula.text()


def test_dh_zero_pose_info(dlg):
    """中栏显示自由度与 SCARA 零位法兰 (0.45, 0, 0.18)。"""
    txt = dlg.lbl_ee.text()
    assert "2 转动" in txt and "1 移动" in txt
    assert "+0.450" in txt and "+0.180" in txt


def test_dh_edit_a_rebuilds_kinematics(dlg):
    """改 a1 后 links 重建，法兰 x 跟随变化并标记 dirty。"""
    dlg.dh_table.cellWidget(0, 3).setValue(0.33)
    assert dlg._dirty
    assert dlg.robot.dh[0].a == pytest.approx(0.33)
    from simovia.robot_spec import link_frames
    fr = {f.name: f for f in link_frames(dlg.robot)}
    assert fr["flange"].p_world[0] == pytest.approx(0.53, abs=1e-6)


def test_dh_row_selection_updates_matrix(dlg):
    """选第 2 行时矩阵面板切到 joint2（其原点在 frame1 = (a1,0,d1)）。"""
    dlg.dh_table.setCurrentCell(1, 1)
    assert "+0.250" in dlg.lbl_pose.text()
    assert "joint2" in dlg.lbl_formula.text()


def test_dh_add_delete_move(dlg):
    dlg.dh_table.setCurrentCell(1, 1)
    dlg._dh_add_row()                       # 插到选中行之后（第 3 行）
    assert dlg.dh_table.rowCount() == 4
    a = [r.a for r in dlg.robot.dh]
    assert a == pytest.approx([0.25, 0.20, 0.15, 0.0])
    dlg._dh_move_up()
    assert [r.a for r in dlg.robot.dh] == pytest.approx(
        [0.25, 0.15, 0.20, 0.0])
    dlg._dh_move_down()
    assert [r.a for r in dlg.robot.dh] == pytest.approx(
        [0.25, 0.20, 0.15, 0.0])
    dlg._dh_del_row()                        # 删选中的新行（index 2）
    assert dlg.dh_table.rowCount() == 3
    assert [r.a for r in dlg.robot.dh] == pytest.approx([0.25, 0.20, 0.0])


def test_dh_type_switch_range_units(dlg):
    """转动（度）切移动（米）：范围 spin 单位与默认值跟着换。"""
    dlg.dh_table.setCurrentCell(0, 1)
    cb = dlg.dh_table.cellWidget(0, 5)
    cb.setCurrentIndex(1)                  # prismatic
    assert dlg.dh_table.cellWidget(0, 6).suffix() == " m"
    assert abs(dlg.dh_table.cellWidget(0, 6).value()) == pytest.approx(
        0.05, abs=1e-6)
    assert dlg.robot.dh[0].joint_type == "prismatic"
    cb.setCurrentIndex(2)                  # fixed：范围禁用、无自由度
    assert not dlg.dh_table.cellWidget(0, 6).isEnabled()
    assert dlg.robot.dh[0].joint_type == "fixed"


def test_dh_kp_kd_apply_to_selected_row(dlg):
    dlg.dh_table.setCurrentCell(2, 1)      # prismatic 行
    dlg.sp_dh_kp.setValue(120.0)
    dlg.sp_dh_kd.setValue(2.5)
    assert dlg.robot.dh[2].kp == pytest.approx(120.0)
    assert dlg.robot.dh[2].kd == pytest.approx(2.5)
    assert dlg.robot.dh[0].kp == pytest.approx(50.0)


def test_dh_preset_switch(dlg):
    # 预设顺序：1=SCARA, 2=六轴, 3=平面 2R
    dlg.cb_preset.setCurrentIndex(3)
    assert dlg.dh_table.rowCount() == 2
    assert dlg.robot.name == "my_2r"
    assert dlg.cb_preset.currentIndex() == 0   # 载入后复位占位项
    dlg.cb_preset.setCurrentIndex(2)
    assert dlg.dh_table.rowCount() == 6


def test_dh_expand_clears_dh_table(dlg, monkeypatch):
    """确认展开后变成 links 模式且不再保留 dh；取消则留在 DH。"""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
    dlg._toggle_mode()
    assert dlg.mode == "dh" and dlg.robot.is_dh
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    dlg._toggle_mode()
    assert dlg.mode == "links" and dlg.robot.dh is None
    names = _tree_names(dlg)
    assert names[0] == "base" and "joint1" in names and "flange" in names


def test_legacy_links_open_in_advanced(legacy_dlg):
    """旧 links 文档打开后自动进高级模式，树与旧示例一致。"""
    assert _tree_names(legacy_dlg) == ["base", "link1", "link2"]
    assert "R = Rz(0°)" in legacy_dlg.lbl_formula.text()


def test_dh_save_roundtrip(tmp_path, dlg):
    import yaml
    from simovia.robot_spec import dump_robot, load_robot
    dlg.dh_table.cellWidget(1, 3).setValue(0.22)
    p = tmp_path / "scara.yaml"
    p.write_text(yaml.safe_dump(dump_robot(dlg.robot), allow_unicode=True),
                 encoding="utf-8")
    r2 = load_robot(p)
    assert r2.is_dh and r2.dh[1].a == pytest.approx(0.22)


# ------------------------------------------------------------ 高级连杆表单

def test_rpy_edit_updates_matrix_and_preview(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link1")
    dlg.sp_rpy[2].setValue(90.0)          # yaw 90°
    assert float(dlg.tb_rel.item(0, 1).text()) == pytest.approx(-1.0, abs=1e-3)
    assert float(dlg.tb_rel.item(1, 0).text()) == pytest.approx(1.0, abs=1e-3)
    dlg._select_link("link2")
    assert "+0.500" in dlg.lbl_pose.text()
    assert float(dlg.tb_world.item(0, 1).text()) == pytest.approx(-1.0, abs=1e-3)


def test_add_and_delete_child(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link2")
    dlg._add_child()
    names = _tree_names(dlg)
    assert len(names) == 4
    new = names[-1]
    assert new.startswith("link")
    dlg._select_link(new)
    dlg._del_link()
    assert new not in _tree_names(dlg)


def test_new_child_sits_on_parent_top(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link2")
    dlg._add_child()
    new = _tree_names(dlg)[-1]
    lk = dlg.robot.link(new)
    assert lk.pos[2] == pytest.approx(0.24)
    assert lk.geom["pos"][2] == pytest.approx(lk.geom["size"][2])


def test_size_change_keeps_origin_at_bottom(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link1")
    dlg.sp_gsize[2].setValue(0.20)
    assert dlg.sp_gpos[2].value() == pytest.approx(0.20)
    assert dlg.robot.link("link1").geom["pos"][2] == pytest.approx(0.20)


def test_manual_geom_offset_disables_autofollow(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link1")
    dlg.sp_gpos[2].setValue(0.05)
    dlg.sp_gsize[2].setValue(0.22)
    assert dlg.sp_gpos[2].value() == pytest.approx(0.05)


def test_geom_type_switch_gives_matching_size(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link1")
    dlg.cb_geom.setCurrentIndex(3)        # cylinder
    assert [w.value() for w in dlg.sp_gsize] == pytest.approx(
        [0.04, 0.075, 0.0])
    assert dlg.sp_gpos[2].value() == pytest.approx(0.075)


def test_joint_type_range_conversion(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link1")
    assert dlg.sp_range[0].value() == pytest.approx(-170.0)
    dlg.cb_joint.setCurrentIndex(2)       # prismatic
    assert abs(dlg.sp_range[0].value()) == pytest.approx(0.297, abs=0.002)
    dlg.cb_joint.setCurrentIndex(1)       # revolute
    assert abs(dlg.sp_range[0].value()) == pytest.approx(170.0, abs=1.0)


def test_fixed_to_prismatic_range_default(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("base")
    dlg.cb_joint.setCurrentIndex(2)
    assert dlg.sp_range[0].value() == pytest.approx(-0.5)
    assert dlg.sp_range[1].value() == pytest.approx(0.5)


def test_delete_parent_reparents_children(legacy_dlg):
    dlg = legacy_dlg
    dlg._select_link("link1")
    dlg._del_link()
    names = _tree_names(dlg)
    assert "link1" not in names and "link2" in names
    from simovia.robot_spec import _validate_tree
    _validate_tree(dlg.robot)


def test_matrix_matches_kinematics(legacy_dlg):
    import mujoco
    from simovia.robot_spec import build_robot_spec, link_frames
    dlg = legacy_dlg
    dlg._select_link("link1")
    dlg.sp_rpy[0].setValue(15.0)
    dlg.sp_rpy[1].setValue(-30.0)
    dlg.sp_rpy[2].setValue(45.0)
    model = build_robot_spec(dlg.robot).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for f in link_frames(dlg.robot):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f.name)
        assert np.allclose(data.xmat[bid].reshape(3, 3), f.R_world, atol=1e-9)


# ------------------------------------------------------------ 与模式无关

def test_save_button_always_asks_path(dlg, monkeypatch):
    """clicked 把 checked(False) 传进槽：必须 lambda 包一层，否则静默覆盖。"""
    from PySide6.QtWidgets import QFileDialog, QPushButton
    dlg._doc_path = "/tmp/不应被覆盖.yaml"
    calls = []

    def fake_save(parent, caption, directory, *args, **kwargs):
        calls.append(caption)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)
    btn = next(b for b in dlg.findChildren(QPushButton)
               if b.text().startswith("保存为机器人"))
    btn.click()
    assert calls == ["保存机器人 YAML"]


def test_esc_respects_unsaved_changes(dlg, monkeypatch):
    dlg._dirty = True
    asked = []
    monkeypatch.setattr(dlg, "_confirm_discard",
                        lambda action: asked.append(action) or False)
    dlg.show()
    dlg.reject()
    assert asked == ["关闭"]
    assert dlg.isVisible()
    dlg._dirty = False
    dlg.hide()


def test_preview_not_overlapped_by_matrix(dlg):
    from PySide6.QtWidgets import QApplication, QGroupBox
    gb = next(g for g in dlg.findChildren(QGroupBox)
              if g.title().startswith("坐标系矩阵"))
    for size in ((1180, 760), (954, 575)):
        dlg.resize(*size)
        dlg.show()
        QApplication.processEvents()
        assert not dlg.lbl_img.geometry().intersects(gb.geometry()), size
        assert dlg.lbl_img.height() >= 240, size
    dlg.hide()


def test_close_confirms_only_once(dlg, monkeypatch):
    dlg._dirty = True
    asked = []
    monkeypatch.setattr(dlg, "_confirm_discard",
                        lambda action: asked.append(action) or True)
    dlg.show()
    dlg.close()
    assert asked == ["关闭"]
    assert not dlg.isVisible()
    assert dlg._renderer is None


def test_axes_overlay_always_visible(dlg):
    import math
    from types import SimpleNamespace
    from PySide6.QtGui import QImage

    img = QImage(240, 240, QImage.Format.Format_RGB888)
    img.fill(0)
    az, el = math.radians(135.0), math.radians(-22.0)
    cam = SimpleNamespace(
        pos=np.array([0.9, -0.9, 0.5]),
        forward=np.array([-math.sin(az) * math.cos(el),
                          -math.cos(az) * math.cos(el), math.sin(el)]),
        up=np.array([math.sin(az) * math.sin(el),
                     math.cos(az) * math.sin(el), math.cos(el)]),
        frustum_near=1.0, frustum_top=math.tan(math.radians(22.5)))
    frame = SimpleNamespace(p_world=np.array([0.0, 0.0, 0.3]),
                            R_world=np.eye(3))
    dlg._draw_axes(img, cam, frame)

    dominant = {"r": 0, "g": 0, "b": 0}
    for y in range(240):
        for x in range(240):
            r, g, b = img.pixelColor(x, y).getRgb()[:3]
            if r > 160 and g < 120 and b < 120:
                dominant["r"] += 1
            elif g > 160 and r < 140 and b < 140:
                dominant["g"] += 1
            elif b > 160 and r < 140 and g < 160:
                dominant["b"] += 1
    assert all(n > 10 for n in dominant.values()), dominant


def test_legacy_links_save_roundtrip(tmp_path, legacy_dlg):
    import yaml
    from simovia.robot_spec import dump_robot, load_robot
    dlg = legacy_dlg
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
