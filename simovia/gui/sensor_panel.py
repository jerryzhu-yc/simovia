"""传感器工作台：平台全部传感器类型的库 + 挂载到机器人 + 参数调整 + 实时读数。

数据链路（复用装备面板已验证的 YAML 写回链，全部经主窗口走 read_world ->
改 dict -> load_world 重载，运行中被 _edit_blocked 拦截提示）：
- 挂载：add_sensor(robot, dict) -> _on_gear_add_sensor
- 卸载：remove_sensor(robot, name) -> _on_gear_remove_sensor
- 改参数：update_sensor(robot, name, patch) -> _on_gear_update_sensor
- 数值读数：get_observation（500ms 轮询，标量 / imu 向量 / lidar 条形图）
- 相机读数：screenshot 按 camera 名渲染（1s 轮询缩略画面）
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox,
                               QSplitter, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from simovia.gui import theme

# 传感器库：type -> (中文名, 说明, 挂点类型)
_SENSOR_LIB: dict[str, tuple[str, str, str]] = {
    "encoder": ("关节编码器", "读关节位置（rad/m），1 维标量。挂 joint。", "joint"),
    "imu": ("惯性测量单元", "陀螺 3 + 加速度计 3 共 6 维。挂 body。", "body"),
    "tactile": ("触觉传感器", "挂点接触力标量（N），radius 为过滤球半径。挂 body。", "body"),
    "joint_torque": ("关节力矩", "关节约束力标量。挂 joint。", "joint"),
    "camera": ("相机", "按名渲染画面/深度，不进数值观测流。挂 body。", "body"),
    "lidar": ("2D 激光雷达", "每 tick rays 个距离（米），-1 = 无回波/超量程。挂 body。", "body"),
}

# 可调参数字段：type -> [(字段, 类型, 默认, 下限, 上限)]
_FIELDS: dict[str, list[tuple]] = {
    "camera": [("fovy", "float", 45.0, 1.0, 179.0),
               ("near", "float", 0.01, 0.001, 10.0),
               ("far", "float", 50.0, 0.1, 200.0),
               ("depth", "bool", False)],
    "lidar": [("rays", "int", 36, 2, 1024),
              ("min_angle", "float", -180.0, -359.0, 0.0),
              ("max_angle", "float", 180.0, 0.0, 359.0),
              ("range", "float", 5.0, 0.1, 100.0)],
    "tactile": [("radius", "float", 0.06, 0.005, 0.5)],
    "encoder": [], "imu": [], "joint_torque": [],
}


class _ScanView(QWidget):
    """lidar 读数条形图：每束一根，绿色 = 距离/量程，灰底 = 无回波。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self._vals: list[float] | None = None
        self._rng = 5.0

    def set_values(self, vals: list[float], rng: float) -> None:
        self._vals = vals
        self._rng = max(rng, 1e-3)
        self.update()

    def clear_values(self) -> None:
        self._vals = None
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 31, 34))
        if not self._vals:
            p.setPen(QColor(theme.C_TEXT_DIM))
            p.drawText(self.rect(), 0x84, "等待读数…")
            return
        n = max(len(self._vals), 1)
        bw = self.width() / n
        for i, v in enumerate(self._vals):
            x = i * bw
            if v < 0:
                p.fillRect(x, self.height() - 5, bw - 0.5, 3,
                           QColor(70, 70, 70))    # 无回波：底部灰槽
            else:
                h = min(v / self._rng, 1.0) * (self.height() - 6)
                p.fillRect(x, self.height() - 3 - h, bw - 0.5, h,
                           QColor(0, 200, 80))
        p.end()


class SensorWindow(QDialog):
    """非模态传感器工作台（主窗口 Ctrl+Shift+N 打开）。"""

    # 与 GearPanel 同签名：直接接到主窗口的 _on_gear_* 槽复用写回链
    add_sensor = Signal(str, object)
    remove_sensor = Signal(str, str)
    update_sensor = Signal(str, str, dict)

    def __init__(self, main, parent=None):
        super().__init__(parent or main)
        self.setWindowTitle("传感器工作台")
        self.setModal(False)
        self.resize(920, 640)
        self._main = main
        self._spec = None
        self._manifest = None
        self._sel: tuple[str, str] | None = None     # (robot, sensor_name)
        self._pending_pick: tuple[str, str] | None = None
        self._field_getters: dict[str, object] = {}
        self._form_holder: QWidget | None = None

        root = QVBoxLayout(self)

        # ------------------------------------------------ 上部：库/树 | 参数/读数
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # 左：传感器库 + 当前世界传感器树
        left = QVBoxLayout()
        lib_box = QGroupBox("传感器库（选择类型挂载）")
        lv = QVBoxLayout(lib_box)
        self.lib_list = QListWidget()
        for t, (title, desc, _m) in _SENSOR_LIB.items():
            it = QListWidgetItem(f"{title}  {t}\n{desc}")
            it.setData(Qt.ItemDataRole.UserRole, t)
            self.lib_list.addItem(it)
        self.lib_list.currentRowChanged.connect(self._on_lib_row)
        lv.addWidget(self.lib_list)
        tree_box = QGroupBox("当前世界的传感器")
        tv = QVBoxLayout(tree_box)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["机器人 / 传感器", "类型"])
        self.tree.currentItemChanged.connect(self._on_tree_sel)
        tv.addWidget(self.tree)
        left.addWidget(lib_box, 1)
        left.addWidget(tree_box, 1)
        left_w = QWidget()
        left_w.setLayout(left)
        split.addWidget(left_w)

        # 右：参数表单 + 实时读数
        right = QVBoxLayout()
        self.param_box = QGroupBox("参数（选中传感器后可调整）")
        self._form_lay = QVBoxLayout(self.param_box)
        self._form_lay.setContentsMargins(4, 4, 4, 4)
        self.btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("应用参数")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_remove = QPushButton("卸载传感器")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_row.addWidget(self.btn_apply)
        self.btn_row.addWidget(self.btn_remove)
        self.btn_row.addStretch()
        self.btn_row.addWidget(QLabel("应用后世界自动重载"))
        self._form_lay.addLayout(self.btn_row)
        right.addWidget(self.param_box)

        read_box = QGroupBox("实时读数（仿真运行中有效）")
        rv = QVBoxLayout(read_box)
        self.read_label = QLabel("未选择传感器")
        self.read_label.setWordWrap(True)
        rv.addWidget(self.read_label)
        self.scan_view = _ScanView()
        rv.addWidget(self.scan_view)
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setMinimumHeight(60)
        rv.addWidget(self.cam_label)
        right.addWidget(read_box, 1)
        right_w = QWidget()
        right_w.setLayout(right)
        split.addWidget(right_w)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        # ------------------------------------------------ 下部：挂载区
        mount_box = QGroupBox("挂载新传感器")
        form = QFormLayout(mount_box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.m_type = QComboBox()
        self.m_type.addItems(list(_SENSOR_LIB))
        self.m_type.currentIndexChanged.connect(self._on_mount_type)
        form.addRow("类型：", self.m_type)
        self.m_desc = QLabel("")
        self.m_desc.setWordWrap(True)
        self.m_desc.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        form.addRow("", self.m_desc)
        self.m_name = QLineEdit()
        form.addRow("名字：", self.m_name)
        self.m_robot = QComboBox()
        self.m_robot.currentIndexChanged.connect(self._on_mount_robot)
        form.addRow("机器人：", self.m_robot)
        self.m_target = QComboBox()
        form.addRow("挂点：", self.m_target)
        pos_row = QHBoxLayout()
        self.m_pos = [QDoubleSpinBox() for _ in range(3)]
        for s in self.m_pos:
            s.setRange(-5.0, 5.0)
            s.setDecimals(3)
            s.setSingleStep(0.05)
            pos_row.addWidget(s)
        pos_w = QWidget()
        pos_w.setLayout(pos_row)
        form.addRow("挂点偏移 pos：", pos_w)
        btn_mount = QPushButton("挂载到机器人")
        btn_mount.clicked.connect(self._on_mount)
        form.addRow("", btn_mount)
        tip = QLabel("运行中不可编辑；也可以用 Ctrl+G 的 AI 生成场景让 Agent 直接加。")
        tip.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        form.addRow("", tip)
        root.addWidget(mount_box)

        # 读数轮询：仅窗口可见时取数，数值 500ms / 相机 1s
        self._poll = QTimer(self)
        self._poll.setInterval(500)
        self._poll.timeout.connect(self._poll_readout)
        self._poll.start()
        self._cam_tick = 0

        self._rebuild_form()

    # ------------------------------------------------------------ 数据入口
    def set_context(self, spec, manifest) -> None:
        """世界加载/重载后由主窗口调用：刷新树、下拉与表单。"""
        self._spec, self._manifest = spec, manifest
        self.tree.clear()
        self._field_getters = {}
        if spec is not None:
            for r in spec.robots:
                top = QTreeWidgetItem([r.name, ""])
                top.setFlags(Qt.ItemFlag.ItemIsEnabled)
                for s in r.sensors:
                    it = QTreeWidgetItem(
                        [s.name, _SENSOR_LIB.get(s.type, ("",))[0] or s.type])
                    it.setData(0, Qt.ItemDataRole.UserRole, (r.name, s.name))
                    top.addChild(it)
                self.tree.addTopLevelItem(top)
            self.tree.expandAll()
        # 机器人下拉（保留当前选择）
        names = [r.name for r in spec.robots] if spec else []
        cur = self.m_robot.currentText()
        self.m_robot.blockSignals(True)
        self.m_robot.clear()
        self.m_robot.addItems(names)
        if cur in names:
            self.m_robot.setCurrentText(cur)
        self.m_robot.blockSignals(False)
        self._on_mount_robot()
        # 世界换了，旧选择失效；挂载成功（reload 回来）则自动选中新传感器
        self._sel = None
        self._rebuild_form()
        self._reset_readout()
        if self._pending_pick is not None:
            self._select_sensor(self._pending_pick)
            if self._sel == self._pending_pick:
                self._pending_pick = None

    # ------------------------------------------------------------ 左栏交互
    def _on_lib_row(self, row: int) -> None:
        if 0 <= row < self.m_type.count():
            self.m_type.setCurrentIndex(row)

    def _selected_spec(self):
        if self._spec is None or self._sel is None:
            return None
        for r in self._spec.robots:
            if r.name == self._sel[0]:
                return next((s for s in r.sensors if s.name == self._sel[1]),
                            None)
        return None

    def _select_sensor(self, key: tuple[str, str]) -> None:
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                it = top.child(j)
                if it.data(0, Qt.ItemDataRole.UserRole) == key:
                    self.tree.setCurrentItem(it)
                    return

    def _on_tree_sel(self, cur, _old) -> None:
        if cur is None or cur.parent() is None:
            self._sel = None
        else:
            self._sel = cur.data(0, Qt.ItemDataRole.UserRole)
        self._rebuild_form()
        self._reset_readout()

    # ------------------------------------------------------------ 参数表单
    def _rebuild_form(self) -> None:
        if self._form_holder is not None:
            self._form_lay.removeWidget(self._form_holder)
            self._form_holder.deleteLater()
            self._form_holder = None
        self._field_getters = {}
        s = self._selected_spec()
        self.btn_apply.setEnabled(s is not None)
        self.btn_remove.setEnabled(s is not None)
        holder = QWidget()
        form = QFormLayout(holder)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        if s is None:
            form.addRow(QLabel("在左侧树选择一个传感器查看/调整参数。"))
        else:
            form.addRow("名字：", QLabel(f"{s.name}（{_SENSOR_LIB.get(s.type, (s.type,))[0]}）"))
            # body 系可调 pos；joint 系无挂点偏移
            if s.body is not None:
                pos_row = QHBoxLayout()
                spins = []
                for i in range(3):
                    sp = QDoubleSpinBox()
                    sp.setRange(-5.0, 5.0)
                    sp.setDecimals(3)
                    sp.setSingleStep(0.05)
                    sp.setValue(float(s.pos[i]))
                    spins.append(sp)
                    pos_row.addWidget(sp)
                pw = QWidget()
                pw.setLayout(pos_row)
                form.addRow("pos：", pw)
                self._field_getters["pos"] = (
                    lambda sp=spins: [round(x.value(), 4) for x in spins])
            for field, kind, default, lo, hi in _FIELDS.get(s.type, []):
                cur = getattr(s, field)
                if kind == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(cur))
                    form.addRow(f"{field}：", w)
                    self._field_getters[field] = w.isChecked
                elif kind == "int":
                    w = QSpinBox()
                    w.setRange(int(lo), int(hi))
                    w.setValue(int(cur) if cur is not None else int(default))
                    form.addRow(f"{field}：", w)
                    self._field_getters[field] = w.value
                else:
                    w = QDoubleSpinBox()
                    w.setRange(lo, hi)
                    w.setDecimals(3)
                    w.setValue(float(cur) if cur is not None else default)
                    form.addRow(f"{field}：", w)
                    self._field_getters[field] = w.value
        self._form_holder = holder
        self._form_lay.insertWidget(0, holder)

    def _on_apply(self) -> None:
        s = self._selected_spec()
        if s is None or not self._field_getters:
            return
        patch = {k: getter() for k, getter in self._field_getters.items()}
        self.update_sensor.emit(self._sel[0], s.name, patch)

    def _on_remove(self) -> None:
        s = self._selected_spec()
        if s is None:
            return
        self.remove_sensor.emit(self._sel[0], s.name)

    # ------------------------------------------------------------ 挂载区
    def _mount_targets(self) -> tuple[str, list[str]]:
        """当前类型 + 机器人可选的挂点名列表（joint 或 body 短名）。"""
        t = self.m_type.currentText()
        kind = _SENSOR_LIB[t][2]
        rname = self.m_robot.currentText()
        names: list[str] = []
        if self._manifest:
            for r in self._manifest.get("robots", []):
                if r.get("name") != rname:
                    continue
                if kind == "joint":
                    names = [j["name"] for j in r.get("joints", [])]
                else:
                    names = list(r.get("bodies", {}).values())
                break
        return kind, names

    def _on_mount_type(self) -> None:
        t = self.m_type.currentText()
        self.m_desc.setText(_SENSOR_LIB[t][1])
        self.lib_list.blockSignals(True)
        idx = list(_SENSOR_LIB).index(t)
        self.lib_list.setCurrentRow(idx)
        self.lib_list.blockSignals(False)
        self._on_mount_robot()

    def _on_mount_robot(self) -> None:
        _kind, names = self._mount_targets()
        self.m_target.clear()
        self.m_target.addItems(names)
        # 默认名字：类型_序号（避开现有重名）
        base = self.m_type.currentText()
        n = 1
        if self._spec:
            existing = {s.name for r in self._spec.robots
                         for s in r.sensors}
            while f"{base}_{n}" in existing:
                n += 1
        self.m_name.setText(f"{base}_{n}")

    def _on_mount(self) -> None:
        t = self.m_type.currentText()
        name = self.m_name.text().strip()
        rname = self.m_robot.currentText()
        target = self.m_target.currentText()
        if not name or not rname or not target:
            return
        kind = _SENSOR_LIB[t][2]
        d: dict = {"name": name, "type": t,
                   "joint" if kind == "joint" else "body": target}
        pos = [round(s.value(), 4) for s in self.m_pos]
        if kind == "body" and any(abs(v) > 1e-9 for v in pos):
            d["pos"] = pos
        self._pending_pick = (rname, name)
        self.add_sensor.emit(rname, d)

    # ------------------------------------------------------------ 实时读数
    def _reset_readout(self) -> None:
        self.read_label.setText("未选择传感器")
        self.scan_view.clear_values()
        self.cam_label.setPixmap(QPixmap())
        self.cam_label.setText("")

    def _poll_readout(self) -> None:
        if not self.isVisible() or self._sel is None:
            return
        s = self._selected_spec()
        if s is None:
            return
        if s.type == "camera":
            self._cam_tick += 1
            if self._cam_tick % 2:            # 1s 一帧
                return
            self._main.client.call_tool(
                "screenshot", {"camera": s.name, "width": 260, "height": 195},
                self._on_cam)
        else:
            self._main.client.call_tool(
                "get_observation", {},
                lambda e, d, r=self._sel[0], n=s.name: self._on_obs(e, d, r, n))

    def _on_obs(self, err, data, robot: str, sname: str) -> None:
        if self._sel != (robot, sname) or err or not isinstance(data, dict):
            return
        try:
            val = (data.get("robots", {}).get(robot, {})
                   .get("sensors", {}).get(sname))
        except AttributeError:
            return
        if val is None:
            self.read_label.setText("读数：暂无（世界未运行或无此传感器）")
            self.scan_view.clear_values()
            return
        s = self._selected_spec()
        if s is not None and s.type == "lidar" and isinstance(val, list):
            rng = float(s.range) if s.range else 5.0
            self.scan_view.set_values(val, rng)
            hits = [v for v in val if v >= 0]
            self.read_label.setText(
                f"{sname}：{len(val)} 束，有效回波 {len(hits)}"
                + (f"，最近 {min(hits):.2f} m" if hits else ""))
        elif isinstance(val, list):
            self.read_label.setText(
                f"{sname}：\n" + "\n".join(
                    f"  [{i}] {v:+.4f}" for i, v in enumerate(val)))
        else:
            self.read_label.setText(f"{sname}：{float(val):+.4f}")

    def _on_cam(self, err, data) -> None:
        if err or not isinstance(data, dict) or not data.get("image"):
            self.cam_label.setText("相机：暂无画面（世界未加载？）")
            return
        pm = QPixmap()
        pm.loadFromData(data["image"])
        self.cam_label.setPixmap(pm.scaledToWidth(260))

    def hideEvent(self, e) -> None:
        self._poll.stop()
        super().hideEvent(e)

    def showEvent(self, e) -> None:
        self._poll.start()
        super().showEvent(e)
