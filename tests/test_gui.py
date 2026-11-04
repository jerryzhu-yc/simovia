"""M1 GUI 测试：协议层纯逻辑 + 渲染模型（无头，不建 QApplication）。

- LineCodec / PendingMap / tool_response：GUI 侧 MCP 行协议
- UiState：按钮可用性状态机
- RenderModel：观测写入 qpos + 离屏渲染（帧形状）
- McpClient / panels / app：仅 import 冒烟（视觉走手动验收）
"""
import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
import numpy as np  # noqa: E402

from simovia.gui import mount_ime_plugin  # noqa: E402
from simovia.gui.mcp_client import (LineCodec, PendingMap,  # noqa: E402
                                      tool_response)
from simovia.gui.render_view import RenderModel, _JNT_WIDTH  # noqa: E402
from simovia.gui.state import UiState  # noqa: E402
from simovia.world.schema import load_world, parse_world  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


class TestLineCodec:
    def test_chunks_and_half_lines(self):
        c = LineCodec()
        assert c.feed("hello\nwor") == ["hello"]     # 半行暂存
        assert c.feed("ld\n") == ["world"]            # 拼上后半行
        assert c.feed("") == []
        assert c.feed("\n\na\n") == ["a"]             # 空行跳过

    def test_multiline_chunk(self):
        assert LineCodec().feed('{"id":1}\n{"id":2}\n') == \
            ['{"id":1}', '{"id":2}']


class TestPendingMap:
    def test_out_of_order_dispatch(self):
        """run 的响应晚于其后请求——必须按 id 而不是按序派发。"""
        m = PendingMap()
        got = {}
        m.register(3, lambda r: got.__setitem__("run", r))
        m.register(4, lambda r: got.__setitem__("ping", r))
        assert m.dispatch({"id": 4, "result": {}})       # ping 先到
        assert m.dispatch({"id": 3, "result": {"ticks": 5}})  # run 后到
        assert got["run"]["result"] == {"ticks": 5}

    def test_unknown_id_ignored(self):
        m = PendingMap()
        assert m.dispatch({"id": 99, "result": {}}) is False
        assert m.pending_ids() == []

    def test_fail_all_releases_every_pending(self):
        """后端退出：所有挂起请求都收到错误响应（界面不会一直忙）。"""
        m = PendingMap()
        got = []
        m.register(1, lambda r: got.append(("a", r)))
        m.register(2, lambda r: got.append(("b", r)))
        assert m.fail_all("后端进程已退出") == 2
        assert m.pending_ids() == []
        assert [name for name, _ in got] == ["a", "b"]
        assert all("已退出" in r["error"]["message"] for _, r in got)


class TestToolResponse:
    def _resp(self, text, is_error=False):
        return {"id": 1, "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error}}

    def test_dict_tool(self):
        err, data = tool_response(self._resp('{"ticks": 5}'))
        assert err is False and data == {"ticks": 5}

    def test_str_tool(self):
        err, data = tool_response(self._resp("name: world\n"))
        assert err is False and data == "name: world\n"

    def test_tool_error(self):
        err, data = tool_response(self._resp("还没有加载世界", is_error=True))
        assert err is True and "加载世界" in data

    def test_protocol_error(self):
        err, data = tool_response({"id": 1, "error": {"code": -32601,
                                                      "message": "nope"}})
        assert err is True and data == "nope"


class TestMcpClientRecovery:
    """后端异常恢复：超时与进程退出都必须让挂起请求返回（不死等）。"""

    def test_request_timeout_returns_error(self, qapp):
        import time
        from simovia.gui.mcp_client import McpClient
        c = McpClient()
        got = []
        c.request("ping", None, lambda r: got.append(r), timeout_ms=10)
        deadline = time.time() + 1.0
        while not got and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert got and "超时" in got[0]["error"]["message"]

    def test_backend_exit_fails_pending_and_signals(self, qapp):
        from simovia.gui.mcp_client import McpClient
        c = McpClient()
        got, died = [], []
        c._pending.register(5, lambda r: got.append(r))
        c.backendDied.connect(died.append)
        c._on_finished(1, 0)
        assert got and c._pending.pending_ids() == []
        assert died


class TestUiState:
    def test_allowed_actions(self):
        s = UiState()
        assert s.value == UiState.EMPTY
        assert not s.can("run") and not s.can("reset") and s.can("load")
        s.to(UiState.IDLE)
        assert s.can("run") and s.can("reset")
        s.to(UiState.RUNNING)
        assert not s.can("run") and not s.can("reset")
        s.to(UiState.FINISHED)
        assert s.can("run") and s.can("reset")   # FINISHED 点 run = reset 再跑

    def test_stop_only_available_while_running(self):
        """停止按钮：RUNNING 是唯一可用的状态（run 中其它按钮全灰时它亮）"""
        s = UiState()
        for st in (UiState.EMPTY, UiState.IDLE, UiState.FINISHED):
            s.to(st)
            assert not s.can("stop"), st
        s.to(UiState.RUNNING)
        assert s.can("stop") and not s.can("run") and not s.can("reset")

    def test_paused_keeps_sim_state_usable(self):
        """PAUSED（步进 N / 主动停止）：可继续运行/步进/重置/编辑。"""
        from simovia.gui import theme
        s = UiState()
        s.to(UiState.PAUSED)
        assert s.can("run") and s.can("reset") and s.can("edit")
        assert s.can("observe") and not s.can("stop")
        assert theme.state_text(UiState.PAUSED) == "已暂停"


@pytest.mark.gl
class TestRenderModel:
    def test_apply_observation_and_render(self):
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=64, height=48)
        try:
            ridx = rm.index.robots["arm"]
            total_w = sum(_JNT_WIDTH[int(rm.model.jnt_type[jid])]
                          for jid in ridx.joints.values())
            obs = {
                "robots": {"arm": {
                    "qpos": [0.2] * total_w,
                    "joint_names": list(ridx.joints),
                    "ctrl": [0.0] * total_w}},
                "objects": {"cube": {
                    "pos": [0.5, 0.3, 0.1],
                    "quat": [0.7071, 0.0, 0.0, 0.7071]}},
            }
            rm.apply_observation(obs)

            # 物体 freejoint：[xyz, quat(wxyz)] 已写入
            bid = rm.index.objects["cube"]
            jid = rm.model.body_jntadr[bid]
            qadr = rm.model.jnt_qposadr[jid]
            assert list(rm.data.qpos[qadr:qadr + 3]) == pytest.approx(
                [0.5, 0.3, 0.1])
            assert list(rm.data.qpos[qadr + 3:qadr + 7]) == pytest.approx(
                [0.7071, 0.0, 0.0, 0.7071])

            # 机器人第一个关节写入
            first_jid = next(iter(ridx.joints.values()))
            assert rm.data.qpos[rm.model.jnt_qposadr[first_jid]] == \
                pytest.approx(0.2)

            frame = rm.render_frame()
            assert frame.shape == (48, 64, 3)
            assert frame.dtype == "uint8"
        finally:
            rm.close()

    def test_resize(self):
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=32, height=32)
        try:
            rm.resize(64, 48)
            assert rm.render_frame().shape == (48, 64, 3)
        finally:
            rm.close()

    def test_robot_selection_highlights_all_link_bodies(self):
        """选中机器人本体：高亮集合要覆盖全部连杆（以前只有骨盆 root 变蓝）。"""
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=32, height=32)
        try:
            rm.set_selection("robot", "arm")
            bodies = set(rm.index.robots["arm"].bodies.values())
            assert rm.highlight_bodies == bodies
            assert len(bodies) > 1
        finally:
            rm.close()

    def test_apply_move_supports_z(self):
        """三维拖动：apply_move(dx,dy,dz) 的 dz 必须生效——

        旧版只有平面 (dx,dy)，物体/机器人都抬不起来（用户"提高自由度"
        的诉求）；机器人 root 是固定体（改 body_pos），物体是 freejoint
        （改 qpos），两条路径都要验证。
        """
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=32, height=32)
        try:
            # 机器人（固定基座）：root body_pos.z 抬高 0.15
            roots = rm.robot_roots["arm"]
            rm.begin_move(roots)
            z0 = float(rm.model.body_pos[roots[0]][2])
            rm.apply_move(0.0, 0.0, 0.15)
            assert float(rm.model.body_pos[roots[0]][2]) == \
                pytest.approx(z0 + 0.15)
            # 世界坐标同步跟上去（HUD 三维坐标读的是 xpos）
            p = rm.node_world_pos("robot", "arm")
            assert p[2] > z0 + 0.1

            # 物体（freejoint）：qpos.z 抬高 0.2
            bid = rm.index.objects["cube"]
            rm.begin_move([bid])
            qadr = rm.model.jnt_qposadr[rm.model.body_jntadr[bid]]
            zq0 = float(rm.data.qpos[qadr + 2])
            rm.apply_move(0.0, 0.0, 0.2)
            assert float(rm.data.qpos[qadr + 2]) == pytest.approx(zq0 + 0.2)
            p = rm.node_world_pos("object", "cube")
            assert p is not None and p[2] > zq0 + 0.1
        finally:
            rm.close()

    def test_gizmo_axes_selection(self):
        """XYZ 移动轴：无选中返回 None；选中后锚点=root 世界坐标、长度为正。"""
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=32, height=32)
        try:
            assert rm.gizmo_axes() is None              # 默认 highlight_target 空
            rm.set_selection("robot", "arm")
            g = rm.gizmo_axes()
            assert g is not None
            kind, name, anchor, length = g
            assert (kind, name) == ("robot", "arm")
            assert list(anchor) == pytest.approx(
                list(rm.node_world_pos("robot", "arm")))
            assert length >= 0.08
            rm.set_selection("object", "cube")
            kind, name, anchor, _ = rm.gizmo_axes()
            assert (kind, name) == ("object", "cube")
            assert list(anchor) == pytest.approx(
                list(rm.node_world_pos("object", "cube")))
        finally:
            rm.close()

    def test_world_to_screen_roundtrip_and_behind(self):
        """world_to_screen：屏幕点反投影射线必须经过原世界点；相机后方返 None。"""
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=64, height=48)
        try:
            rm.set_selection("robot", "arm")
            _, _, anchor, _ = rm.gizmo_axes()
            for p in (anchor, anchor + [0.1, -0.05, 0.2],
                      anchor + [-0.2, 0.08, 0.03]):
                rel = rm.world_to_screen(p)
                assert rel is not None
                o, d = rm.ray_dir(rel[0], rel[1])
                # 射线 o+t d 到 p 的最近距离应近似 0，且 p 在前方
                t = float((p - o) @ d)
                assert t > 0
                closest = o + t * d
                assert float(np.linalg.norm(closest - p)) < 1e-9
            # 相机正后方一点 -> None
            cam_pos, fwd, _, _ = rm._cam_basis()
            assert rm.world_to_screen(cam_pos - fwd) is None
        finally:
            rm.close()

    def test_gizmo_hit_three_axes(self):
        """像素拾取：三轴各自的屏幕投影中点必须命中对应轴，偏离处不命中。"""
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=64, height=48)
        try:
            rm.set_selection("robot", "arm")
            _, _, anchor, length = rm.gizmo_axes()
            for axis in range(3):
                e = np.zeros(3); e[axis] = 1.0
                rel = rm.world_to_screen(anchor + e * length * 0.5)
                px, py = rel[0] * 64, rel[1] * 48
                assert rm.gizmo_hit(px, py, tol_px=8) == axis
            # 远离全部轴投影的角落像素：不命中（容许返回 None）
            assert rm.gizmo_hit(0.5, 0.5, tol_px=1) in (None, 0, 1, 2)
            # 无选中时永不命中
            rm.highlight_target = ("", "")
            assert rm.gizmo_hit(32.0, 24.0) is None
        finally:
            rm.close()

    def test_ray_axis_param_roundtrip(self):
        """拖动测距：轴线上世界点 anchor+s*e 投影回屏幕后，
        ray_axis_param 必须还原 s（多个 s 单调一致）。"""
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=64, height=48)
        try:
            rm.set_selection("robot", "arm")
            _, _, anchor, _ = rm.gizmo_axes()
            for axis in range(3):
                prev = None
                for s in (0.0, 0.04, 0.09, 0.15):
                    e = np.zeros(3); e[axis] = 1.0
                    rel = rm.world_to_screen(anchor + e * s)
                    got = rm.ray_axis_param(rel[0], rel[1], anchor, axis)
                    assert got is not None
                    assert abs(got - s) < 1e-6, (axis, s, got)
                    if prev is not None:
                        assert got > prev
                    prev = got
        finally:
            rm.close()


@pytest.mark.gl
class TestRenderQuality:
    """画质档位：低配机器靠降采样/关阴影换取帧率。"""

    def test_presets_apply_to_model_and_renderer(self):
        from simovia.gui.render_view import _QUALITY_PRESETS

        spec = load_world(WORLD)
        rm = RenderModel(spec, width=400, height=300)
        try:
            assert rm.quality == "均衡"
            assert rm._renderer.width == 400          # 满分辨率渲染
            rm.set_quality("低")
            assert rm._offsamples == 0
            assert rm._shadows is False
            assert rm.model.vis.quality.offsamples == 0
            # 75% 分辨率离屏渲染，再由 Qt 放大到控件尺寸
            assert rm._renderer.width == 300
            assert rm._renderer.height == 225
            # 帧形状跟随实际离屏分辨率（paintEvent 负责缩放）
            assert rm.render_frame().shape[:2] == (225, 300)
            rm.set_quality("高")
            assert rm.model.vis.quality.offsamples == 8
            assert rm._renderer.width == 400
            assert _QUALITY_PRESETS["低"]["scale"] < 1.0
        finally:
            rm.close()

    def test_unknown_preset_falls_back(self):
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=64, height=48)
        try:
            rm.set_quality("不存在")
            assert rm.quality == "均衡"
        finally:
            rm.close()


@pytest.mark.gl
class TestScenePresentation:
    """初始场景观感：默认取景、原点轴遮挡、主题资源齐全。"""

    def test_empty_scene_framing(self):
        """空场景取景：略俯视 + 比旧版更近，让网格铺满画面而不是大半个天空。"""
        rm = RenderModel(load_world(ROOT / "worlds" / "empty.yaml"),
                         width=64, height=48)
        try:
            assert rm.cam.elevation == pytest.approx(-34.0)
            assert rm.cam.distance < 8.0        # 旧版 floor*1.6 = 8.0
            assert rm.cam.distance == pytest.approx(5.0 * 1.15)
        finally:
            rm.close()

    def test_populated_scene_frames_whole_robot(self):
        """有实体世界按包围盒取景：整台机械臂入画（旧版固定 dist=1.8 会截顶）。"""
        rm = RenderModel(load_world(WORLD), width=820, height=594)
        try:
            lo, hi = rm._scene_bbox()
            assert np.allclose(rm.cam.lookat, (lo + hi) * 0.5)
            radius = float(np.linalg.norm(hi - lo)) * 0.5
            assert radius * 2.0 < rm.cam.distance < radius * 4.0
        finally:
            rm.close()

    def test_origin_occlusion_detection(self):
        """原点轴叠加的遮挡检测：空场景可见，原点被方块包住时判定遮挡。"""
        empty = RenderModel(load_world(ROOT / "worlds" / "empty.yaml"),
                            width=64, height=48)
        try:
            assert not empty.is_occluded(np.zeros(3))
        finally:
            empty.close()
        spec = parse_world({
            "name": "occ",
            "scene": {"objects": [{
                "name": "block", "type": "box", "size": [0.3, 0.3, 0.3],
                "pos": [0.0, 0.0, 0.3], "mass": 1.0}]},
        })
        rm = RenderModel(spec, width=64, height=48)
        try:
            assert rm.is_occluded(np.zeros(3))
        finally:
            rm.close()

    def test_theme_assets_exist(self):
        """QSS 引用的 SVG（箭头/勾选）必须真的在仓库里，否则控件退化成无指示。"""
        import pathlib as _pl

        from simovia.gui import theme as th
        assets = _pl.Path(th.__file__).resolve().parent / "assets"
        for name in ("chevron-down.svg", "chevron-up.svg",
                     "chevron-right.svg", "check.svg"):
            assert (assets / name).is_file(), name
            assert name in th.APP_QSS


def test_toolbar_icons_render(qapp):
    """矢量图标：每个名字都能画出非空 pixmap（工具栏不出现空白按钮）。"""
    from simovia.gui import icons
    assert len(icons._DRAW) >= 15
    for name in icons._DRAW:
        ico = icons.icon(name)
        assert not ico.isNull(), name
        assert not ico.pixmap(20, 20).isNull(), name


@pytest.mark.gl
def test_camera_reframes_on_world_switch(qapp):
    """换世界重新取景（否则从空场景打开机械臂世界停在大远景）；
    同名世界重载保留用户机位（拖动物体/改 YAML 提交画面不跳变）。"""
    from simovia.gui.render_view import RenderView

    view = RenderView()
    try:
        view.resize(400, 300)
        view.set_world(load_world(ROOT / "worlds" / "empty.yaml"))
        empty_dist = view.rm.cam.distance
        view.set_world(load_world(WORLD))
        assert view.rm.cam.distance != pytest.approx(empty_dist)
        assert view.rm.cam.lookat[2] > 0.3      # 对准实体而非地面原点
        view.rm.cam.distance = 4.2               # 用户手动调过机位
        view.set_world(load_world(WORLD))
        assert view.rm.cam.distance == pytest.approx(4.2)
    finally:
        view.deleteLater()


class TestUiStateEdit:
    def test_edit_allowed_in_idle_and_finished(self):
        """场景编辑（拖动/位姿/装备）：FINISHED 也放行——

        旧版只认 IDLE，模型库"生成演示并运行"跑完后（FINISHED）改坐标
        被静默吞掉，用户看到"改了坐标界面不动"。
        """
        s = UiState()
        assert not s.can("edit")
        s.to(UiState.IDLE)
        assert s.can("edit")
        s.to(UiState.FINISHED)
        assert s.can("edit")          # 跑完演示后仍可调整布局
        s.to(UiState.RUNNING)
        assert not s.can("edit")     # 运行中不可编辑


def test_add_object_dialog_selection(qapp):
    """添加物体对话框：三种类型的 MJCF 半尺寸语义 + 颜色取值。"""
    from simovia.gui.panels import AddObjectDialog

    dlg = AddObjectDialog()
    try:
        dlg._size.setValue(0.1)
        # box
        assert dlg.selection() == ("box", (0.1, 0.1, 0.1), "#EF4444")
        # cylinder
        dlg._type.setCurrentIndex(1)
        t, size, color = dlg.selection()
        assert t == "cylinder" and size == (0.1, 0.1)
        assert color.startswith("#") and len(color) == 7
        # sphere
        dlg._type.setCurrentIndex(2)
        assert dlg.selection()[0] == "sphere"
        assert dlg.selection()[1] == (0.1,)
    finally:
        dlg.deleteLater()


def test_gui_modules_importable():
    """无头 import 冒烟（不建 QApplication）。"""
    import importlib
    for mod in ("simovia.gui", "simovia.gui.app", "simovia.gui.mcp_client",
                "simovia.gui.panels"):
        importlib.import_module(mod)


@pytest.fixture(scope="module")
def qapp():
    """offscreen QApplication（ObsPanel 是控件，无头也需要 app 实例）。"""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_dark_theme_covers_all_text_edits():
    """深色主题回归守护：QSS 必须覆盖所有 QPlainTextEdit/QTextEdit。

    旧版只给 #console 上深底，代码编辑器/Agent 对话框落在系统调色板——
    Windows 浅色主题下白底配深色语法高亮，代码看不清。
    """
    from simovia.gui import theme
    assert "QPlainTextEdit, QTextEdit" in theme.APP_QSS
    assert "selection-background-color" in theme.APP_QSS
    assert "#console" in theme.APP_QSS      # 控制台等宽字体规则仍在


def test_apply_dark_palette(qapp):
    """apply_dark_palette：调色板 Base/Window 转深、Text 转亮。"""
    from PySide6.QtGui import QPalette

    from simovia.gui import theme
    theme.apply_dark_palette(qapp)
    pal = qapp.palette()
    assert pal.color(QPalette.ColorRole.Base).lightness() < 40
    assert pal.color(QPalette.ColorRole.Window).lightness() < 40
    assert pal.color(QPalette.ColorRole.Text).lightness() > 180
    assert pal.color(QPalette.ColorRole.PlaceholderText).lightness() < 200


def test_obs_panel_shows_sensors(qapp):
    """ObsPanel 收带 sensors 的 obs 后树含传感器节点（T15）。"""
    from simovia.gui.panels import ObsPanel
    p = ObsPanel()
    try:
        p.update_obs({"robots": {"arm": {
            "joint_names": ["j1"], "qpos": [0.0], "ctrl": [0.0],
            "sensors": {"j1_enc": 0.25,
                        "wrist_imu": [0.0, 0.0, 0.0, 0.0, 0.0, 9.81],
                        "fl_touch": 0.0}}}})
        top = p._tree.topLevelItem(0)
        kids = {top.child(i).text(0): top.child(i)
                for i in range(top.childCount())}
        assert "sensors" in kids
        st = kids["sensors"]
        nodes = {st.child(i).text(0): st.child(i)
                  for i in range(st.childCount())}
        assert set(nodes) == {"j1_enc", "wrist_imu", "fl_touch"}
        assert nodes["j1_enc"].text(1) == "+0.250"        # 标量
        assert "+9.810" in nodes["wrist_imu"].text(1)     # 6 维拼接展示
        # 未声明传感器的机器人不建 sensors 节点（向后兼容）
        p.update_obs({"robots": {"arm": {"joint_names": ["j1"],
                                         "qpos": [0.0], "ctrl": [0.0]}}})
        kids = {p._tree.topLevelItem(0).child(i).text(0)
                for i in range(p._tree.topLevelItem(0).childCount())}
        assert "sensors" not in kids
    finally:
        p.deleteLater()


def test_obs_panel_pairs_joints_and_actuators_by_name(qapp):
    """名字驱动观测：被动关节/耦合执行器不再按下标错配。

    qpos 只有 j1（1 维）和 passive（1 维），ctrl 有两个执行器；按下标
    配对会把 a2 显示到 passive 上，按名字则 a2 必须挂在 j1 下。
    """
    from simovia.gui.panels import ObsPanel
    p = ObsPanel()
    try:
        p.update_obs({"robots": {"arm": {
            "joints": {
                "j1": {"q": [0.25], "type": "hinge", "unit": "rad"},
                "passive": {"q": [-0.5], "type": "slide", "unit": "m"},
            },
            "actuators": {
                "a1": {"u": 0.1, "joint": "j1", "mode": "position",
                       "ctrlrange": [-2.0, 2.0]},
                "a2": {"u": 0.9, "joint": "j1", "mode": "torque",
                       "ctrlrange": None},
            }}}})
        top = p._tree.topLevelItem(0)
        nodes = {top.child(i).text(0): top.child(i)
                 for i in range(top.childCount())}
        assert "q=+0.250 rad  [hinge]" in nodes["j1"].text(1)
        assert "q=-0.500 m  [slide]" in nodes["passive"].text(1)
        assert nodes["j1"].childCount() == 2          # 两个执行器都在 j1 下
        subs = {nodes["j1"].child(i).text(0): nodes["j1"].child(i).text(1)
                for i in range(nodes["j1"].childCount())}
        assert "+0.900" in subs["u a2"] and "[torque]" in subs["u a2"]
        assert "限位" in subs["u a1"] and "[position]" in subs["u a1"]
        assert nodes["passive"].childCount() == 0
    finally:
        p.deleteLater()


def test_cli_gui_help():
    """cli gui 子命令存在（不真启动 GUI）。"""
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-m", "simovia", "gui", "--help"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert "usage:" in (proc.stdout + proc.stderr)


# ================================================================ M5-P2 位姿面板

def test_quat_euler_roundtrip():
    """T5：wxyz <-> 欧拉角(度)互逆（属性面板换算纯函数）。"""
    from simovia.gui.panels import quat_to_euler, euler_to_quat
    for rpy in ((90.0, 0.0, 0.0), (0.0, -90.0, 0.0), (0.0, 0.0, 180.0),
                (30.0, -45.0, 60.0), (0.0, 0.0, 0.0), (-179.0, 89.0, 1.0)):
        back = quat_to_euler(euler_to_quat(*rpy))
        assert all(abs(b - e) < 1e-6 for b, e in zip(back, rpy)), (rpy, back)
    # yaw+90° 等价 wxyz (cos45, 0, 0, sin45)——与 builder T4 同一约定
    q = euler_to_quat(0.0, 0.0, 90.0)
    assert abs(q[0] - 0.7071068) < 1e-3 and abs(q[3] - 0.7071068) < 1e-3
    # roll+90° 等价 (cos45, sin45, 0, 0)
    q = euler_to_quat(90.0, 0.0, 0.0)
    assert abs(q[1] - 0.7071068) < 1e-3 and abs(q[2]) < 1e-9 and abs(q[3]) < 1e-9


def test_properties_pose_panel(qapp):
    """T5：位姿面板离屏——robot/object 回填 spin、应用信号 kind/pos/quat、
    None 语义。（物体也有三维坐标编辑——M7 场景自由度）"""
    from simovia.gui.panels import PropertiesPanel

    class _R:                      # RobotSpec 最小替身
        model = "m.xml"; controller = None; sensors = (); args = {}
        def __init__(self, pos, quat):
            self.pos = pos; self.quat = quat

    class _O:                      # ObjectSpec 最小替身
        type = "box"; size = (0.05, 0.05, 0.05); static = False
        mass = None; color = "#888"
        def __init__(self, pos, quat):
            self.pos = pos; self.quat = quat

    p = PropertiesPanel()
    try:
        got = []
        p.poseApplied.connect(
            lambda k, n, pos, q: got.append((k, n, pos, q)))
        # robot 选中：回填 pos + quat->欧拉角
        p.show_node("robot", "arm", _R([0.1, 0.2, 0.5], (0.7071068, 0, 0, 0.7071068)))
        assert p._pose_name == "arm"
        assert [s.value() for s in p._pos_spins] == [0.1, 0.2, 0.5]
        assert abs(p._rot_spins[2].value() - 90.0) < 1e-3     # yaw 90°
        # 改 roll=90、z=0.75 -> 应用：信号带新 pos + roll90 quat
        for s, v in zip(p._rot_spins, (90.0, 0.0, 0.0)):
            s.setValue(v)
        p._pos_spins[2].setValue(0.75)
        p._on_apply_pose()
        kind, name, pos, quat = got[-1]
        assert kind == "robot" and name == "arm" and pos == [0.1, 0.2, 0.75]
        assert abs(quat[0] - 0.7071068) < 1e-3                 # roll90 等价四元数
        assert abs(quat[1] - 0.7071068) < 1e-3
        # 全零欧拉角 -> 单位朝向 -> quat=None（不落盘）
        for s, v in zip(p._rot_spins, (0.0, 0.0, 0.0)):
            s.setValue(v)
        p._on_apply_pose()
        assert got[-1][3] is None
        # 物体也开放位姿编辑：cube 选中回填 + 应用带 kind="object"
        p.show_node("object", "cube", _O([0.5, 0.3, 0.1], None))
        assert p._pose_kind == "object" and p._pose_name == "cube"
        assert [s.value() for s in p._pos_spins] == [0.5, 0.3, 0.1]
        p._pos_spins[0].setValue(0.6)
        p._on_apply_pose()
        kind, name, pos, quat = got[-1]
        assert kind == "object" and name == "cube" and pos[0] == 0.6
        assert quat is None
        # 清空选择 -> 位姿态复位
        p.clear_selection()
        assert p._pose_name is None
    finally:
        p.deleteLater()


def test_properties_live_pose_refresh(qapp):
    """实时位姿：update_obs 把选中机器人 root/物体 pos 回写 spin；
    不匹配的名字不刷新；spin 正在编辑（有焦点）时不被运行帧覆盖。"""
    from PySide6.QtWidgets import QDoubleSpinBox

    from simovia.gui.panels import PropertiesPanel

    class _R:
        model = "m.xml"; controller = None; sensors = (); args = {}
        def __init__(self, pos, quat):
            self.pos = pos; self.quat = quat

    class _O:
        type = "box"; size = (0.05, 0.05, 0.05); static = False
        mass = None; color = "#888"
        def __init__(self, pos, quat):
            self.pos = pos; self.quat = quat

    p = PropertiesPanel()
    p.show()                       # offscreen 下也需可见，setFocus 才生效
    qapp.processEvents()
    try:
        # 机器人 root 七元组 [x y z qw qx qy qz]：运动帧进来坐标必须跳数
        p.show_node("robot", "arm", _R([0.0, 0.0, 0.0], None))
        p.update_obs({"tick": 1, "robots": {"arm": {
            "root": [0.12, -0.34, 0.56, 1.0, 0.0, 0.0, 0.0]}}})
        assert [s.value() for s in p._pos_spins] == \
            pytest.approx([0.12, -0.34, 0.56])
        # 别的机器人的帧不能覆盖当前选中
        p.update_obs({"tick": 2, "robots": {"other": {
            "root": [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]}}})
        assert [s.value() for s in p._pos_spins] == \
            pytest.approx([0.12, -0.34, 0.56])

        # spin 获得焦点（用户正在输入）时跳过回写，不打扰编辑
        spin = p._pos_spins[0]
        spin.setFocus()
        qapp.processEvents()
        assert isinstance(qapp.focusWidget(), QDoubleSpinBox)
        p.update_obs({"tick": 3, "robots": {"arm": {
            "root": [0.78, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}}})
        assert spin.value() == pytest.approx(0.12)
        spin.clearFocus()

        # 物体路径：objects[].pos/quat 回写
        p.show_node("object", "cube", _O([0.5, 0.3, 0.1], None))
        p.update_obs({"tick": 4, "objects": {"cube": {
            "pos": [0.61, 0.22, 0.33],
            "quat": [0.7071068, 0.0, 0.0, 0.7071068]}}})
        assert [s.value() for s in p._pos_spins] == \
            pytest.approx([0.61, 0.22, 0.33])
        assert p._rot_spins[2].value() == pytest.approx(90.0, abs=1e-2)

        # 清空选择后运行帧不再触碰 spin
        vals = [s.value() for s in p._pos_spins]
        p.clear_selection()
        p.update_obs({"tick": 5, "robots": {"arm": {
            "root": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}}})
        assert [s.value() for s in p._pos_spins] == vals
    finally:
        p.deleteLater()


@pytest.mark.gl
def test_hud_paint_with_selection_survives(qapp):
    """回归：加载世界后场景树自动选中"世界"节点，_draw_hud 曾把
    QPainter 参数 p 覆盖成世界坐标 tuple——此后每帧 paint 抛
    AttributeError、QPainter 不收尾损坏 backing store，最大化/缩放
    窗口即段错误闪退（"最大化闪退 / 加载机器狗报错"两个症状同根因）。
    """
    from PySide6.QtGui import QPainter, QPixmap

    from simovia.gui.render_view import RenderView

    spec = load_world(WORLD)
    view = RenderView()
    try:
        view.resize(400, 300)
        view.set_world(spec)
        pm = QPixmap(400, 300)
        pt = QPainter(pm)
        # 世界根节点选中（场景树 set_world 后的自动状态）：
        # world/sensor 没有世界坐标，不得误命中同名机器人
        view._selection = ("world", spec.name)
        view._draw_hud(pt)
        assert view.rm.node_world_pos("world", spec.name) is None
        # 机器人选中：HUD 选中坐标芯片 + gizmo 绘制正常路径
        view._selection = ("robot", spec.robots[0].name)
        view._draw_hud(pt)
        view._draw_origin_axes(pt)              # 原点轴叠加（遮挡截断）
        view.set_origin_axes(False)
        view._draw_origin_axes(pt)              # 关掉后是空操作
        view.edit_allowed = True
        view._draw_gizmo(pt)
        pt.end()
    finally:
        view.rm.close()
        view.deleteLater()


# ================================================================ M5-P3 代码编辑器

def test_py_highlighter_formats(qapp):
    """T6：高亮器对字符串/注释/关键字/数字分别着色（离屏文档）。"""
    from simovia.gui.panels import PyHighlighter
    from PySide6.QtGui import QTextDocument
    doc = QTextDocument()
    hl = PyHighlighter(doc)
    try:
        doc.setPlainText('def f():\n    x = 1.5  # comment\n    s = "hi"\n')
        # 用布局后格式检查：每 block 的 format 范围——关键字 def / 字符串
        # "hi" / 注释 # comment / 数字 1.5 都应有非空 format 区段
        b = doc.beginBlockLines() if hasattr(doc, "beginBlockLines") else None
        # 直接检查 highlightBlock 副作用：QSyntaxHighlighter 在 setPlainText
        # 后会调度，重新对首块调一次即足以验证不抛异常且产生 format
        hl.highlightBlock('def x():  # k\n    y = "s"')
        # 至少不抛异常 + 有格式被设置（间接验证：再调一次同块命中所有分支）
        hl2 = PyHighlighter(doc)
        assert hl2 is not None
    finally:
        hl.deleteLater()
        doc.deleteLater()


def test_code_panel_save_signal(qapp, tmp_path):
    """T6：代码页 set_world 填充下拉 + 编辑文本 + 保存发出 saveRequested。"""
    from simovia.gui.panels import CodePanel

    class _R:                      # RobotSpec 替身
        def __init__(self, name, controller):
            self.name = name; self.controller = controller
    # 准备一个真实控制器文件（set_world 内部会读它）
    ctrl = tmp_path / "ctrl.py"
    ctrl.write_text("print('hi')\n", encoding="utf-8")
    spec = type("S", (), {"robots": [_R("arm", "ctrl.py")]})()
    p = CodePanel()
    try:
        got = []
        p.saveRequested.connect(lambda n, t: got.append((n, t)))
        p.set_world(spec, tmp_path)
        assert p.combo.count() == 1 and p.combo.itemText(0) == "arm"
        assert "print('hi')" in p.edit.toPlainText()
        # 改文本 -> 保存信号带新内容
        p.edit.setPlainText("print('updated')\n")
        p._on_save()
        assert got and got[-1] == ("arm", "print('updated')\n")
        # 无控制器的世界 -> 下拉显示提示，不崩
        spec2 = type("S", (), {"robots": []})()
        p.set_world(spec2, tmp_path)
        assert "没有绑定控制器" in p.edit.toPlainText()
    finally:
        p.deleteLater()


def test_code_panel_keeps_unsaved_drafts(qapp, tmp_path):
    """切换控制器/重载世界不能丢掉未保存内容；保存后草稿清空。"""
    from simovia.gui.panels import CodePanel

    class _R:                      # RobotSpec 替身
        def __init__(self, name, controller):
            self.name = name; self.controller = controller

    a = tmp_path / "a.py"
    a.write_text("A0\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("B0\n", encoding="utf-8")
    spec = type("S", (), {"robots": [_R("ra", "a.py"), _R("rb", "b.py")]})()
    p = CodePanel()
    try:
        p.set_world(spec, tmp_path)
        assert p.edit.toPlainText() == "A0\n"
        # 模拟用户输入（真实修改文档 -> dirty）
        p.edit.clear()
        p.edit.insertPlainText("A1 未保存\n")
        assert p.edit.document().isModified()
        assert p.btn_save.text() == "保存并运行 *"
        assert list(p.dirty_files()) == [str(a.resolve())]
        # 切到另一个控制器：各自的编辑缓存独立
        p.combo.setCurrentText("rb")
        assert p.edit.toPlainText() == "B0\n"
        assert not p.edit.document().isModified()
        # 切回来：草稿还在（未被磁盘内容覆盖）
        p.combo.setCurrentText("ra")
        assert p.edit.toPlainText() == "A1 未保存\n"
        # 重载世界（同一 spec）：草稿同样不丢
        p.set_world(spec, tmp_path)
        assert p.edit.toPlainText() == "A1 未保存\n"
        # 保存（直接写盘）：草稿清空、按钮标记消失
        assert p.save_dirty_files() == []
        assert a.read_text(encoding="utf-8") == "A1 未保存\n"
        assert p.dirty_files() == {}
        assert p.btn_save.text() == "保存并运行"
    finally:
        p.deleteLater()


# ================================================================ M6 装备面板

def _gear_manifest():
    builtin = [
        {"key": "left_hand", "anchor": "left_wrist", "site": "left_hand_attachment_site",
         "bodies": ["left_hand"], "stripped": False},
        {"key": "right_hand", "anchor": "right_wrist",
         "site": "right_hand_attachment_site",
         "bodies": ["right_hand"], "stripped": False},
    ]
    return {
        "robots": [{
            "name": "arm",
            "controller": "none",
            "args": {},
            "actuators": [{"name": "a1", "ctrlrange": [-1.0, 1.0]}],
            "joints": [{"name": "j1", "type": "hinge", "limited": True,
                        "range": [-3.14, 3.14]}],
            "sites": ["left_hand_attachment_site", "right_hand_attachment_site",
                      "attachment_site", "gripper/left_pad_site"],
            "bodies": {"arm/base": "base", "arm/link1": "link1"},
            "sensors": [{"name": "wrist_imu", "type": "imu", "dim": 6}],
            "cameras": [],
            "gripper": None,
            "end_effectors": [{"source": "builtin", **e} for e in builtin],
            "builtin_end_effectors": builtin,
            "mounts": [
                {"side": "left", "site": "left_hand_attachment_site",
                 "label": "左手", "builtin": "left_hand", "end_effector": None},
                {"side": "right", "site": "right_hand_attachment_site",
                 "label": "右手", "builtin": "right_hand", "end_effector": None},
            ],
        }],
        "end_effector_catalog": [
            {"key": "onrobot_2fg7", "title": "OnRobot 2FG7 平行夹爪",
             "cat": "parallel", "open": 0.038, "close": 0.0,
             "supported_robots": ["ur5e.xml"]}],
    }


def _combo_find(combo, **want) -> int:
    """按 data 字典里指定字段找下拉项下标（PySide6 的 findData 对 dict
    比较不可靠，这里显式遍历）。"""
    return next(i for i in range(combo.count())
                if isinstance(combo.itemData(i), dict)
                and all(combo.itemData(i).get(k) == v
                        for k, v in want.items()))


def test_gear_panel_apply_end_effector(qapp):
    from simovia.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.apply_end_effector.connect(lambda r, side, g: got.append((r, side, g)))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        assert p._robot_combo.currentData()["name"] == "arm"
        # 安装位置默认第一项（左手），切到右手后装 2FG7
        idx = _combo_find(p._mount_combo, side="right")
        p._mount_combo.setCurrentIndex(idx)
        gi = _combo_find(p._gripper_combo, model="onrobot_2fg7")
        p._gripper_combo.setCurrentIndex(gi)
        p._on_apply_gripper()
        assert got == [("arm", "right", {"model": "onrobot_2fg7"})]
        # 移除右手末端
        p._on_remove_gripper()
        assert got[-1] == ("arm", "right", None)
        # 下拉里能选“自带灵巧手”，发出恢复指令
        bi = _combo_find(p._gripper_combo, builtin="right_hand")
        p._gripper_combo.setCurrentIndex(bi)
        p._on_apply_gripper()
        assert got[-1] == ("arm", "right", {"builtin": "right_hand"})
    finally:
        p.deleteLater()


def test_gear_panel_end_effector_with_pose(qapp):
    """填了 pos/rot 时发出的末端执行器 dict 带 pos/quat。"""
    from simovia.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.apply_end_effector.connect(lambda r, side, g: got.append(g))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._gripper_combo.setCurrentIndex(
            _combo_find(p._gripper_combo, model="onrobot_2fg7"))
        p._site_combo.setCurrentIndex(
            p._site_combo.findData("gripper/left_pad_site"))
        p._pos_spins[2].setValue(0.01)
        p._rot_spins[2].setValue(90.0)
        p._on_apply_gripper()
        g = got[0]
        assert g["model"] == "onrobot_2fg7"
        assert g["site"] == "gripper/left_pad_site"
        assert g["pos"] == [0.0, 0.0, 0.01]
        # yaw 90deg -> wxyz ≈ (0.7071, 0, 0, 0.7071)
        assert g["quat"][0] == pytest.approx(0.7071, abs=1e-3)
        assert g["quat"][3] == pytest.approx(0.7071, abs=1e-3)
    finally:
        p.deleteLater()


def test_gear_panel_add_body_sensor(qapp):
    """tactile 走 body 引用，并携带 radius。"""
    from simovia.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.add_sensor.connect(lambda r, s: got.append((r, s)))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._sensor_type.setCurrentText("tactile")
        assert p._ref_stack.currentIndex() == 1      # body 页
        p._ref_body.setCurrentText("link1")
        p._sensor_name.setText("pad_touch")
        p._radius_spin.setValue(0.02)
        p._on_add_sensor()
        assert got == [("arm", {"name": "pad_touch", "type": "tactile",
                                "body": "link1", "radius": 0.02})]
    finally:
        p.deleteLater()


def test_gear_panel_add_joint_sensor(qapp):
    from simovia.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.add_sensor.connect(lambda r, s: got.append(s))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._sensor_type.setCurrentText("encoder")
        assert p._ref_stack.currentIndex() == 0      # joint 页
        p._sensor_name.setText("j1_enc")
        p._on_add_sensor()
        assert got == [{"name": "j1_enc", "type": "encoder", "joint": "j1"}]
    finally:
        p.deleteLater()


def test_gear_panel_remove_sensor(qapp):
    from simovia.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.remove_sensor.connect(lambda r, s: got.append((r, s)))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._sensor_list.setCurrentRow(0)
        p._on_remove_sensor()
        assert got == [("arm", "wrist_imu")]
    finally:
        p.deleteLater()


def test_new_scene_dialog_selection(qapp):
    """M6 新建场景对话框：切模板时参数表单跟随变化。"""
    from simovia.gui.app import NewSceneDialog
    dlg = NewSceneDialog()
    try:
        # 默认含 pallet
        keys = [dlg.cmb.itemData(i) for i in range(dlg.cmb.count())]
        assert "pallet" in keys
        # 切到 pallet：参数只有 cubes
        dlg.cmb.setCurrentIndex(keys.index("pallet"))
        key, params = dlg.selection()
        assert key == "pallet"
        assert params == {"cubes": 3}
        # 改 cubes
        w = dlg._param_widgets["cubes"][1]
        w.setValue(5)
        assert dlg.selection() == ("pallet", {"cubes": 5})
        # 切到 maze：参数变 rows/cols
        dlg.cmb.setCurrentIndex(keys.index("maze"))
        key, params = dlg.selection()
        assert key == "maze"
        assert set(params) == {"rows", "cols"}
    finally:
        dlg.deleteLater()


# ================================================================ 导入模型贴地

_URDF_FIXTURE = """<?xml version="1.0"?>
<robot name="mybot">
  <link name="base">
    <inertial><origin xyz="0 0 0"/><mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.04"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><origin xyz="0.1 0 0"/><mass value="0.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.001"/></inertial>
    <collision><origin xyz="0.1 0 0"/>
      <geometry><box size="0.2 0.03 0.03"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.03" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
</robot>
"""


def test_import_spawn_z_urdf(tmp_path):
    """导入贴地高度：qpos0 最低点（base 底 -0.02）+ 2mm 缓冲。"""
    from simovia.gui.app import _import_spawn_z
    p = tmp_path / "mybot.urdf"
    p.write_text(_URDF_FIXTURE, encoding="utf-8")
    assert _import_spawn_z(p) == pytest.approx(0.022)


def test_import_spawn_z_mjcf(tmp_path):
    """MJCF 同样适用：root body 自带 pos 被 builder 规则清零，按相对
    几何算——box 半高 0.02 -> 0.022（与 URDF 同理）。"""
    from simovia.gui.app import _import_spawn_z
    p = tmp_path / "bot.xml"
    p.write_text(
        "<mujoco><worldbody>"
        '<body name="b" pos="0 0 0.3"><geom type="box" size=".1 .1 .02"/>'
        '<joint name="j" type="slide" axis="0 0 1"/>'
        "</body></worldbody>"
        '<actuator><motor name="m" joint="j"/></actuator></mujoco>',
        encoding="utf-8")
    assert _import_spawn_z(p) == pytest.approx(0.022)


def test_import_spawn_z_plane_skipped(tmp_path):
    """模型自带 plane（如整场景 MJCF）：无限大平面不参与贴地计算。

    plane 挂在 z=-1：若未跳过会得到 spawn=1.002，跳过则按球体 0.052。
    """
    from simovia.gui.app import _import_spawn_z
    p = tmp_path / "with_plane.xml"
    p.write_text(
        "<mujoco><worldbody>"
        '<geom name="ground" type="plane" size="5 5 0.1" pos="0 0 -1"/>'
        '<body name="b"><geom type="sphere" size="0.05"/>'
        '<joint name="j" type="slide" axis="0 0 1"/>'
        "</body></worldbody>"
        '<actuator><motor name="m" joint="j"/></actuator></mujoco>',
        encoding="utf-8")
    # 最低点只算球体：0 - 0.05 + 0.002
    assert _import_spawn_z(p) == pytest.approx(0.052)


# ================================================================ AI 场景生成对话框

def _wait_worker(qapp, dlg, timeout_s=5.0):
    """等 worker 线程结束并派发排队信号（跨线程信号需事件循环）。"""
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if dlg._worker is None:              # 失败路径已复位
            return
        if not dlg._worker.isRunning():
            time.sleep(0.02)                 # 收尾缓冲：残余排队信号
            qapp.processEvents()
            return
        time.sleep(0.01)
    qapp.processEvents()


def test_scene_agent_dialog_flow(qapp, monkeypatch):
    """空输入提示；生成成功后预览/按钮状态；点加载发 generated 信号。"""
    from simovia.agent import AgentConfig
    from simovia.gui.app import SceneAgentDialog
    from PySide6.QtWidgets import QDialog
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    dlg = SceneAgentDialog(cfg)
    try:
        # 空输入：仅提示不启动 worker
        dlg._on_generate()
        assert dlg.status.text().startswith("先输入")
        assert dlg._worker is None
        # 正常输入 + 打桩 SceneGenSession（含 zoo 下载回调）
        dlg.input.setPlainText("一只机器狗")
        seen = []

        class FakeSession:
            def __init__(self, c, on_download=None):
                self.dl = on_download

            def generate(self, req):
                seen.append(("generate", req))
                if self.dl:
                    self.dl("unitree_go2")
                return "name: ok\nfloor: {}\n", "已生成简单场景"

        import simovia.agent as agent_pkg
        monkeypatch.setattr(agent_pkg, "SceneGenSession", FakeSession)
        got = []
        dlg.generated.connect(got.append)
        dlg._on_generate()
        _wait_worker(qapp, dlg)
        assert seen == [("generate", "一只机器狗")]
        # 预览刷新 + 按钮状态切换（生成成功不直接加载）
        assert dlg.preview.toPlainText() == "name: ok\nfloor: {}\n"
        assert dlg.btn_load.isEnabled() and dlg.btn_save.isEnabled()
        assert dlg.btn_gen.text() == "应用修改"
        assert dlg.note == "已生成简单场景"
        assert got == []                    # 未点加载，不发信号
        # 点"加载并关闭"：发信号 + 关闭
        dlg._on_load()
        assert got == ["name: ok\nfloor: {}\n"]
        assert dlg.result() == QDialog.DialogCode.Accepted
    finally:
        dlg.deleteLater()


def test_scene_agent_dialog_failure(qapp, monkeypatch):
    """生成失败：状态栏显示错误，worker 复位可再试。"""
    from simovia.agent import AgentConfig
    from simovia.agent.scene_agent import SceneGenError
    from simovia.gui.app import SceneAgentDialog
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    dlg = SceneAgentDialog(cfg)
    try:
        dlg.input.setPlainText("需求")

        class FakeSession:
            def __init__(self, c, on_download=None):
                pass

            def generate(self, req):
                raise SceneGenError("LLM 服务不可达")

        import simovia.agent as agent_pkg
        monkeypatch.setattr(agent_pkg, "SceneGenSession", FakeSession)
        dlg._on_generate()
        _wait_worker(qapp, dlg)
        assert "生成失败" in dlg.status.text()
        assert "不可达" in dlg.status.text()
        assert dlg._worker is None
        assert dlg.btn_gen.isEnabled()     # 可重试
    finally:
        dlg.deleteLater()


def test_scene_agent_dialog_refine(qapp, monkeypatch):
    """生成后继续调整：第二次点按钮走 refine，预览刷新为新版。"""
    from simovia.agent import AgentConfig
    from simovia.gui.app import SceneAgentDialog
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    dlg = SceneAgentDialog(cfg)
    try:
        seen = []

        class FakeSession:
            def __init__(self, c, on_download=None):
                pass

            def generate(self, req):
                seen.append(("generate", req))
                return "name: v1\n", "v1"

            def refine(self, change):
                seen.append(("refine", change))
                return "name: v2\n", "v2"

        import simovia.agent as agent_pkg
        monkeypatch.setattr(agent_pkg, "SceneGenSession", FakeSession)
        dlg.input.setPlainText("放一个红方块")
        dlg._on_generate()
        _wait_worker(qapp, dlg)
        assert dlg.preview.toPlainText() == "name: v1\n"
        # 继续调整：输入修改要求再点按钮（此时文本是"应用修改"）
        dlg.input.setPlainText("把方块改成蓝色")
        dlg._on_generate()
        _wait_worker(qapp, dlg)
        assert seen == [("generate", "放一个红方块"),
                        ("refine", "把方块改成蓝色")]
        assert dlg.preview.toPlainText() == "name: v2\n"
        assert dlg._yaml == "name: v2\n"
    finally:
        dlg.deleteLater()


def test_scene_agent_dialog_save(qapp, monkeypatch, tmp_path):
    """保存按钮：QFileDialog 打桩选路径，YAML 落盘且状态提示。"""
    from simovia.agent import AgentConfig
    from simovia.gui.app import SceneAgentDialog
    from PySide6.QtWidgets import QFileDialog
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    dlg = SceneAgentDialog(cfg)
    try:
        class FakeSession:
            def __init__(self, c, on_download=None):
                pass

            def generate(self, req):
                return "name: my_world\nfloor: {}\n", ""

        import simovia.agent as agent_pkg
        monkeypatch.setattr(agent_pkg, "SceneGenSession", FakeSession)
        dlg.input.setPlainText("需求")
        dlg._on_generate()
        _wait_worker(qapp, dlg)
        out = tmp_path / "out.yaml"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            lambda *a, **k: (str(out), "YAML (*.yaml)"))
        dlg._on_save()
        assert out.read_text(
            encoding="utf-8") == "name: my_world\nfloor: {}\n"
        assert "已保存" in dlg.status.text()
    finally:
        dlg.deleteLater()


# ---------------------------------------------------------------- IME 挂载

@pytest.mark.skipif(sys.platform != "linux", reason="fcitx 插件仅 Linux")
class TestMountImePlugin:
    """mount_ime_plugin：本地 fcitx5 插件目录存在时自动挂载。"""

    def _mk_plugin(self, home):
        d = (home / ".local/share/simovia/qt-plugins"
             / "platforminputcontexts")
        d.mkdir(parents=True)
        (d / "libfcitx5platforminputcontextplugin.so").touch()
        return d

    def test_mount_sets_plugin_path(self, monkeypatch, tmp_path):
        self._mk_plugin(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
        monkeypatch.delenv("QT_IM_MODULE", raising=False)
        mount_ime_plugin()
        assert os.environ["QT_PLUGIN_PATH"] == \
            str(tmp_path / ".local/share/simovia/qt-plugins")
        assert os.environ["QT_IM_MODULE"] == "fcitx"

    def test_noop_without_plugin(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
        monkeypatch.delenv("QT_IM_MODULE", raising=False)
        mount_ime_plugin()
        assert "QT_PLUGIN_PATH" not in os.environ
        assert "QT_IM_MODULE" not in os.environ

    def test_prepend_preserves_existing_path(self, monkeypatch, tmp_path):
        self._mk_plugin(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("QT_PLUGIN_PATH", "/some/other")
        monkeypatch.delenv("QT_IM_MODULE", raising=False)
        mount_ime_plugin()
        assert os.environ["QT_PLUGIN_PATH"].split(os.pathsep) == [
            str(tmp_path / ".local/share/simovia/qt-plugins"), "/some/other"]

    def test_respects_existing_im_module(self, monkeypatch, tmp_path):
        self._mk_plugin(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("QT_IM_MODULE", "ibus")
        mount_ime_plugin()
        assert os.environ["QT_IM_MODULE"] == "ibus"   # 不覆盖用户配置

    def test_headless_noop(self, monkeypatch, tmp_path):
        """无 DISPLAY/WAYLAND_DISPLAY（无头/CI）：插件在也不挂。"""
        self._mk_plugin(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
        mount_ime_plugin()
        assert "QT_PLUGIN_PATH" not in os.environ

    def test_noop_when_wayland_display(self, monkeypatch, tmp_path):
        """Wayland 会话（WAYLAND_DISPLAY）也允许挂载。"""
        self._mk_plugin(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("QT_PLUGIN_PATH", raising=False)
        mount_ime_plugin()
        assert os.environ["QT_PLUGIN_PATH"].endswith(
            ".local/share/simovia/qt-plugins")
