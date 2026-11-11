"""遥操作面板：实时相机画面 + 机器人/相机/录制选择 + 键位说明。

信号：
    startRequested(dict)  开始遥操作（参数直接投给 MCP teleop_start）
    stopRequested()       请求停止（走 stop 工具）
    previewToggled(bool)  只开关相机实时画面（不启动 IK 遥操作）

画面由 app 层定时轮询 camera_frame 后调用 CameraView.show_png 刷新。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from simovia.gui import theme

_FREE_CAMERA = "自由机位（跟随推进）"

# (标签, 宽, 高)
_RESOLUTIONS = [("320 x 240", 320, 240),
                ("480 x 360", 480, 360),
                ("640 x 480", 640, 480)]

DEFAULT_RECORD_DIR = "episodes/teleop"


class CameraView(QLabel):
    """等比缩放显示最新一帧（无画面时给出占位提示）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 最小尺寸压小：停靠面板在 1024×768 下不给滚动条留横向溢出
        self.setMinimumSize(120, 80)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background:{theme.C_BG}; border:1px solid {theme.C_BORDER};"
            f"color:{theme.C_TEXT_DIM};")
        self.setText("无画面")
        self._pix: QPixmap | None = None

    def show_png(self, data: bytes) -> bool:
        pix = QPixmap()
        if not pix.loadFromData(data, "PNG"):
            return False
        self._pix = pix
        self._rescale()
        return True

    def clear(self) -> None:
        self._pix = None
        self.setPixmap(QPixmap())
        self.setText("无画面")

    def _rescale(self) -> None:
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)


class TeleopPanel(QWidget):
    """遥操作控制面板。"""

    startRequested = Signal(dict)
    stopRequested = Signal()
    previewToggled = Signal(bool)
    floatRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manifest = None
        self._cam_cfg: dict[str, dict] = {}   # "robot/camera" -> manifest 相机配置
        self._loading = False
        self._teleop_running = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        self.camera = CameraView()
        # 画面留在滚动区外：dock 再矮也只会滚控件，相机不会被滚掉一截
        self.camera.setMinimumHeight(110)
        outer.addWidget(self.camera, 1)

        # 控件区自己滚动（外层 dock 不再包 QScrollArea）
        ctrl = QWidget()
        lay = QVBoxLayout(ctrl)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        form = QFormLayout()
        self.robot_combo = QComboBox()
        self.camera_combo = QComboBox()
        self.res_combo = QComboBox()
        # 长相机名不撑宽 dock：内容最短 6 字符宽，超出省略显示（悬停看全名）
        for cb in (self.robot_combo, self.camera_combo):
            cb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(6)
            cb.setMinimumWidth(90)
        for label, w, h in _RESOLUTIONS:
            self.res_combo.addItem(label, (w, h))
        self.res_combo.setCurrentIndex(1)
        form.addRow("机器人：", self.robot_combo)
        form.addRow("相机：", self.camera_combo)
        form.addRow("画面：", self.res_combo)
        lay.addLayout(form)

        # 独立相机预览：只出实时画面，不启动 IK 遥操作
        # （任意机器人/暂停/普通运行时都可用；不与运行状态互斥）
        self.bt_preview = QPushButton("相机预览")
        self.bt_preview.setCheckable(True)
        self.bt_preview.setToolTip(
            "只开启实时画面（只读，不影响仿真，也不需要 IK 遥操作）。\n"
            "2/3 自由度、移动底盘、无控制器的机器人都能看。")
        self.bt_preview.toggled.connect(self._on_preview_toggled)
        bt_float = QPushButton("独立窗口")
        bt_float.setToolTip("把相机面板变为浮动窗口，可放大观察画面")
        bt_float.clicked.connect(self.floatRequested.emit)
        preview_row = QHBoxLayout()
        preview_row.addWidget(self.bt_preview, 1)
        preview_row.addWidget(bt_float)
        lay.addLayout(preview_row)

        # 深度画面（近红远蓝伪彩）+ 裁剪面，可随时切换
        # 开关和裁剪面分两行：单行并排会在 1024 宽停靠时顶出横向滚动条
        self.depth_chk = QCheckBox("深度图")
        self.depth_chk.setToolTip("以近红远蓝伪彩显示相机深度，用于判断抓取距离")
        lay.addWidget(self.depth_chk)

        clip_row = QHBoxLayout()
        clip_row.addWidget(QLabel("near:"))
        self.near_spin = QDoubleSpinBox()
        self.near_spin.setRange(0.001, 10.0)
        self.near_spin.setDecimals(3)
        self.near_spin.setSingleStep(0.01)
        self.near_spin.setValue(0.01)
        self.near_spin.setMaximumWidth(76)
        clip_row.addWidget(self.near_spin)
        clip_row.addWidget(QLabel("far:"))
        self.far_spin = QDoubleSpinBox()
        self.far_spin.setRange(0.1, 1000.0)
        self.far_spin.setDecimals(1)
        self.far_spin.setSingleStep(1.0)
        self.far_spin.setValue(50.0)
        self.far_spin.setMaximumWidth(76)
        clip_row.addWidget(self.far_spin)
        clip_row.addStretch(1)
        lay.addLayout(clip_row)

        # 持久配置 vs 临时覆盖：切换相机时回填世界 YAML 里的相机配置，
        # 面板上的改动只影响本次预览/遥操作，不回写世界文件
        self.lbl_cam_hint = QLabel("深度参数取自相机配置；此处改动仅临时生效")
        self.lbl_cam_hint.setWordWrap(True)
        self.lbl_cam_hint.setStyleSheet(
            f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        lay.addWidget(self.lbl_cam_hint)

        self.depth_chk.toggled.connect(self._on_camera_params_changed)
        self.near_spin.valueChanged.connect(self._on_camera_params_changed)
        self.far_spin.valueChanged.connect(self._on_camera_params_changed)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        self.rec_chk = QCheckBox("同时录制 episode")
        self.rec_chk.setChecked(True)
        self.rec_chk.setToolTip("每控制 tick 采样关节/夹爪/物体位姿，停止时写盘\n"
                                "（data.npz + meta.json + frames/*.png）")
        lay.addWidget(self.rec_chk)

        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel("目录："))
        self.rec_dir = QLineEdit(DEFAULT_RECORD_DIR)
        self.rec_dir.setToolTip("episode 输出目录（相对路径按后端进程工作目录）")
        rec_row.addWidget(self.rec_dir, 1)
        bt_pick = QPushButton("…")
        bt_pick.setFixedWidth(30)
        bt_pick.clicked.connect(self._pick_dir)
        rec_row.addWidget(bt_pick)
        lay.addLayout(rec_row)

        btn_row = QHBoxLayout()
        self.bt_start = QPushButton("开始遥操作")
        self.bt_start.clicked.connect(self._on_start)
        self.bt_stop = QPushButton("停止")
        self.bt_stop.setEnabled(False)
        self.bt_stop.clicked.connect(self.stopRequested.emit)
        btn_row.addWidget(self.bt_start)
        btn_row.addWidget(self.bt_stop)
        lay.addLayout(btn_row)

        hint = QLabel("W/S 前后 · A/D 左右 · Q/E 升降 · J/L 偏航\n"
                      "Shift 加速 · 空格 夹爪开合（点视口后按键）")
        # 换行：长键位说明是面板最小宽度的主要来源，窄 dock 会横向裁掉画面
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        lay.addWidget(hint)

        self.status = QLabel("就绪：加载世界后点「开始遥操作」")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{theme.C_TEXT_DIM};")
        lay.addWidget(self.status)

        self.ctrl_scroll = QScrollArea()
        self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.ctrl_scroll.setWidget(ctrl)
        outer.addWidget(self.ctrl_scroll, 0)

    # ------------------------------------------------------------ 外部接口

    def set_manifest(self, manifest: dict) -> None:
        """按 get_robot_manifest 填充机器人/相机下拉（保持当前选择）。"""
        self._manifest = manifest
        self._loading = True
        try:
            old_robot = self.robot_combo.currentData()
            old_cam = self.camera_combo.currentData()
            self.robot_combo.clear()
            self.camera_combo.clear()
            self._cam_cfg = {}
            self.camera_combo.addItem(_FREE_CAMERA, "")
            for r in manifest.get("robots") or []:
                name = r.get("name")
                if not name:
                    continue
                self.robot_combo.addItem(name, name)
                for s in r.get("sensors") or []:
                    if s.get("type") != "camera":
                        continue
                    cam = s.get("name")
                    if not cam:
                        continue
                    self.camera_combo.addItem(f"{name}/{cam}", f"{name}/{cam}")
                    # 回填用：世界 YAML 声明的深度开关与裁剪面（米）
                    self._cam_cfg[f"{name}/{cam}"] = {
                        "depth": bool(s.get("depth", False)),
                        "near": s.get("near"),
                        "far": s.get("far"),
                    }
                # 模型 XML 自带、但 YAML 未声明的相机：也要能预览
                listed = {f"{name}/{cam}" for cam in r.get("cameras") or []}
                for key in sorted(listed - set(self._cam_cfg)):
                    self.camera_combo.addItem(key, key)
            # 世界固定机位（scene.cameras）：名字不带机器人前缀，按原名渲染
            for cam in manifest.get("scene_cameras") or []:
                if cam:
                    self.camera_combo.addItem(cam, cam)
            for combo, old in ((self.robot_combo, old_robot),
                               (self.camera_combo, old_cam)):
                idx = combo.findData(old) if old is not None else -1
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            if self.camera_combo.count() > 1 and \
                    self.camera_combo.currentIndex() == 0:
                self.camera_combo.setCurrentIndex(1)   # 有挂载相机优先用它
        finally:
            self._loading = False
        self._apply_camera_cfg()

    def set_running(self, running: bool) -> None:
        self._teleop_running = running
        self.bt_start.setEnabled(not running)
        self.bt_stop.setEnabled(running)
        # 深度开关/裁剪面保持可用：运行中也能实时切伪彩看距离
        for w in (self.robot_combo, self.camera_combo, self.res_combo,
                  self.rec_chk, self.rec_dir):
            w.setEnabled(not running)
        if running:
            self.set_preview_checked(False)

    def set_status(self, text: str, level: str = "info") -> None:
        color = {"ok": theme.C_OK, "warn": theme.C_WARN,
                 "error": theme.C_ERR}.get(level, theme.C_TEXT_DIM)
        self.status.setStyleSheet(f"color:{color};")
        self.status.setText(text)

    def show_frame(self, data: bytes) -> bool:
        return self.camera.show_png(data)

    def clear_frame(self) -> None:
        self.camera.clear()

    def set_preview_checked(self, on: bool) -> None:
        """程序性开关预览（不触发 previewToggled，调用方自己负责关流）。"""
        self._loading = True
        try:
            self.bt_preview.setChecked(on)
        finally:
            self._loading = False
        self.bt_preview.setText("停止预览" if on else "相机预览")

    # ------------------------------------------------------------ 内部

    def _pick_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 episode 输出目录",
                                                self.rec_dir.text() or ".")
        if path:
            self.rec_dir.setText(path)

    def camera_args(self) -> dict:
        """camera_frame 的取帧参数（深度开关 + 裁剪面）。"""
        args: dict = {"depth": self.depth_chk.isChecked()}
        if args["depth"]:
            args["near"] = round(self.near_spin.value(), 3)
            args["far"] = round(self.far_spin.value(), 1)
        return args

    def stream_args(self) -> dict:
        """camera_stream 的开流参数（独立预览用；不含遥操作字段）。"""
        w, h = self.res_combo.currentData()
        args = {"width": w, "height": h, "fps": 15}
        args.update(self.camera_args())
        cam = self.camera_combo.currentData()
        args["camera"] = cam if cam is not None else ""
        return args

    def _apply_camera_cfg(self) -> None:
        """把当前相机的 manifest 配置回填到深度开关/裁剪面。"""
        cfg = self._cam_cfg.get(self.camera_combo.currentData() or "")
        self._loading = True
        try:
            if cfg is None:
                # 自由机位/无 YAML 配置：退回缺省裁剪面
                self.depth_chk.setChecked(False)
                self.near_spin.setValue(0.01)
                self.far_spin.setValue(50.0)
                free = not (self.camera_combo.currentData() or "")
                self.lbl_cam_hint.setText(
                    "自由机位：深度参数为临时值（近 0.01 / 远 50 米）" if free
                    else "该相机没有深度配置：深度参数为临时值，"
                         "如需持久化请在装备页填写")
            else:
                self.depth_chk.setChecked(cfg["depth"])
                if cfg.get("near") is not None:
                    self.near_spin.setValue(float(cfg["near"]))
                if cfg.get("far") is not None:
                    self.far_spin.setValue(float(cfg["far"]))
                self.lbl_cam_hint.setText(
                    f"参数来自相机配置（near {self.near_spin.value():g} / "
                    f"far {self.far_spin.value():g} 米）；此处改动仅临时生效")
        finally:
            self._loading = False

    def _on_camera_changed(self, _index: int = 0) -> None:
        if self._loading:
            return
        self._apply_camera_cfg()
        if self.bt_preview.isChecked():
            # 预览中换相机：按新相机/分辨率重开流
            self.previewToggled.emit(True)

    def _on_camera_params_changed(self, _value=None) -> None:
        if self._loading:
            return
        if self._cam_cfg:
            self.lbl_cam_hint.setText(
                f"当前为临时覆盖（near {self.near_spin.value():g} / "
                f"far {self.far_spin.value():g} 米）；世界文件里的相机配置不变")

    def _on_preview_toggled(self, on: bool) -> None:
        if self._loading:
            return
        if on and self._teleop_running:
            # 遥操作自己会开流，避免两条流互相覆盖
            self.set_preview_checked(False)
            self.status.setText("遥操作进行中，已由遥操作接管相机画面")
            return
        self.bt_preview.setText("停止预览" if on else "相机预览")
        self.previewToggled.emit(on)

    def _on_start(self) -> None:
        w, h = self.res_combo.currentData()
        args = {"width": w, "height": h, "fps": 15}
        args.update(self.camera_args())
        robot = self.robot_combo.currentData()
        if robot:
            args["robot"] = robot
        cam = self.camera_combo.currentData()
        if cam is not None:
            args["camera"] = cam
        if self.rec_chk.isChecked():
            d = self.rec_dir.text().strip()
            if d:
                args["record_dir"] = d
            args["record_images"] = True
            args["record_width"] = w
            args["record_height"] = h
        self.startRequested.emit(args)
