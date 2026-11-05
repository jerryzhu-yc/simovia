"""「造机器人」面板：离屏实例化、旋转矩阵联动、增删连杆、保存回读。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# 对话框内含实时预览，创建即建 mujoco.Renderer：无 GL 环境整体跳过
pytestmark = pytest.mark.gl


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


def test_new_child_sits_on_parent_top(dlg):
    """新加的子连杆关节原点落在父几何顶面，两段不重叠。

    旧实现 pos.z 写死 0.15，父连杆（link2 高 0.24）时新连杆会插进
    父几何体一半。
    """
    dlg._select_link("link2")
    dlg._add_child()
    new = _tree_names(dlg)[-1]
    lk = dlg.robot.link(new)
    # link2: box 半高 0.12 + 几何偏置 z 0.12 -> 顶面 0.24
    assert lk.pos[2] == pytest.approx(0.24)
    # 新连杆自身：关节原点 = 几何底面中心（geom.pos.z == 半高）
    assert lk.geom["pos"][2] == pytest.approx(lk.geom["size"][2])


def test_size_change_keeps_origin_at_bottom(dlg):
    """改几何高度时几何偏置 z 自动跟随半高，关节原点保持在底面中心。"""
    dlg._select_link("link1")
    dlg.sp_gsize[2].setValue(0.20)
    assert dlg.sp_gpos[2].value() == pytest.approx(0.20)
    assert dlg.robot.link("link1").geom["pos"][2] == pytest.approx(0.20)


def test_manual_geom_offset_disables_autofollow(dlg):
    """用户手动偏移过几何 z 后，改尺寸不再强行吸回半高。"""
    dlg._select_link("link1")
    dlg.sp_gpos[2].setValue(0.05)
    dlg.sp_gsize[2].setValue(0.22)
    assert dlg.sp_gpos[2].value() == pytest.approx(0.05)


def test_geom_type_switch_gives_matching_size(dlg):
    """换几何类型时按新类型给尺寸与底面中心偏置，不保留旧语义数值。"""
    dlg._select_link("link1")
    # box=1, sphere=2, cylinder=3
    dlg.cb_geom.setCurrentIndex(3)
    assert [w.value() for w in dlg.sp_gsize] == pytest.approx(
        [0.04, 0.075, 0.0])
    assert dlg.sp_gpos[2].value() == pytest.approx(0.075)


def test_joint_type_range_conversion(dlg):
    """转动 <-> 移动切换时范围按力臂双向换算，不再出现 ±170 m。"""
    dlg._select_link("link1")
    assert dlg.sp_range[0].value() == pytest.approx(-170.0)
    dlg.cb_joint.setCurrentIndex(2)        # prismatic
    assert abs(dlg.sp_range[0].value()) == pytest.approx(0.297, abs=0.002)
    dlg.cb_joint.setCurrentIndex(1)        # 回到 revolute
    assert abs(dlg.sp_range[0].value()) == pytest.approx(170.0, abs=1.0)


def test_fixed_to_prismatic_range_default(dlg):
    """fixed 没有范围概念，切到移动关节时给米级默认而非 ±180 m。"""
    dlg._select_link("base")
    dlg.cb_joint.setCurrentIndex(2)        # prismatic
    assert dlg.sp_range[0].value() == pytest.approx(-0.5)
    assert dlg.sp_range[1].value() == pytest.approx(0.5)


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


def test_save_button_always_asks_path(dlg, monkeypatch):
    """已打开文件后点"保存为机器人 YAML…"仍要询问路径（实测 P1）。

    回归点：clicked 会把 checked(False) 当 _save 的第一个位置参数传进去，
    直接连 _save 会被当成"不用问"，静默覆盖原文件。
    """
    from PySide6.QtWidgets import QFileDialog, QPushButton
    dlg._doc_path = "/tmp/不应被覆盖.yaml"
    calls = []

    def fake_save(parent, caption, directory, *args, **kwargs):
        calls.append(caption)
        return ("", "")                   # 用户在对话框里取消

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)
    btn = next(b for b in dlg.findChildren(QPushButton)
               if b.text().startswith("保存为机器人"))
    btn.click()
    assert calls == ["保存机器人 YAML"]


def test_esc_respects_unsaved_changes(dlg, monkeypatch):
    """Esc 走 reject()：有未保存修改时必须先确认，不能直接关掉丢改动。"""
    dlg._dirty = True
    asked = []
    monkeypatch.setattr(dlg, "_confirm_discard",
                        lambda action: asked.append(action) or False)
    dlg.show()
    dlg.reject()
    assert asked == ["关闭"]
    assert dlg.isVisible()                # 拒绝后窗口仍在
    dlg._dirty = False                    # 让 fixture close() 能正常关闭
    dlg.hide()


def test_preview_not_overlapped_by_matrix(dlg):
    """旋转矩阵组不得压住预览（实测 P2 复测）。

    回归点：右栏没给预览 stretch、矩阵组也没高度上限时，布局把矩阵组
    分到 655px，几何上直接盖在预览区上。
    """
    from PySide6.QtWidgets import QApplication, QGroupBox
    gb = next(g for g in dlg.findChildren(QGroupBox)
              if g.title().startswith("旋转矩阵"))
    for size in ((1180, 760), (954, 575)):
        dlg.resize(*size)
        dlg.show()
        QApplication.processEvents()
        assert not dlg.lbl_img.geometry().intersects(gb.geometry()), size
        assert dlg.lbl_img.height() >= 240, size
    dlg.hide()


def test_close_confirms_only_once(dlg, monkeypatch):
    """选"丢弃"后关闭只确认一次；第二次取消不应发生在渲染器已释放之后。"""
    dlg._dirty = True
    asked = []
    monkeypatch.setattr(dlg, "_confirm_discard",
                        lambda action: asked.append(action) or True)
    dlg.show()
    dlg.close()
    assert asked == ["关闭"]
    assert not dlg.isVisible()
    assert dlg._renderer is None


def test_layout_fits_1024_screen(dlg):
    """默认窗口三栏 + 旋转矩阵在 1024 宽屏幕内完整可见（实测 P2）。

    回归点：预览 QLabel 会把 420px 原图当最小宽度、按钮行四个并排，
    整窗最小宽度曾达 1033px，矩阵第三列被挤出屏幕。
    """
    assert dlg.minimumSizeHint().width() <= 1000
    assert dlg.tb_rel.minimumWidth() == 150 and dlg.tb_world.minimumWidth() == 150
    # 两表并排不超过右栏可用宽度（312 = 150*2 + 间距）
    assert dlg.tb_rel.width() + dlg.tb_world.width() <= 320


def test_axes_overlay_always_visible(dlg):
    """轴标是画在图像上的 2D 叠加：黑色背景（机身遮住的像素）也能看到三轴。

    回归点：旧实现把三轴做成场景里的 3D 胶囊，参与深度测试，轴指向
    连杆内部时被几何体完全遮住，看不出坐标系方向。
    """
    import math
    from types import SimpleNamespace

    from PySide6.QtGui import QImage

    img = QImage(240, 240, QImage.Format.Format_RGB888)
    img.fill(0)                              # 黑色 = 被机身挡住的像素
    az, el = math.radians(135.0), math.radians(-22.0)
    cam = SimpleNamespace(
        pos=np.array([0.9, -0.9, 0.5]),
        # 与预览相机一致：orbit 视角推导出的 forward/up
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
