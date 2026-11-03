"""遥操作面板：实时相机画面 + 机器人/相机/录制选择 + 键位说明。

信号：
    startRequested(dict)  开始遥操作（参数直接投给 MCP teleop_start）
    stopRequested()       请求停止（走 stop 工具）

画面由 app 层定时轮询 camera_frame 后调用 CameraView.show_png 刷新。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

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
        self.setMinimumSize(160, 120)
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manifest = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.camera = CameraView()
        lay.addWidget(self.camera, 1)

        form = QFormLayout()
        self.robot_combo = QComboBox()
        self.camera_combo = QComboBox()
        self.res_combo = QComboBox()
        for label, w, h in _RESOLUTIONS:
            self.res_combo.addItem(label, (w, h))
        self.res_combo.setCurrentIndex(1)
        form.addRow("机器人：", self.robot_combo)
        form.addRow("相机：", self.camera_combo)
        form.addRow("画面：", self.res_combo)
        lay.addLayout(form)

        # 深度画面（近红远蓝伪彩）+ 裁剪面，可随时切换
        depth_row = QHBoxLayout()
        self.depth_chk = QCheckBox("深度图")
        self.depth_chk.setToolTip("以近红远蓝伪彩显示相机深度，用于判断抓取距离")
        depth_row.addWidget(self.depth_chk)
        depth_row.addWidget(QLabel("near:"))
        self.near_spin = QDoubleSpinBox()
        self.near_spin.setRange(0.001, 10.0)
        self.near_spin.setDecimals(3)
        self.near_spin.setSingleStep(0.01)
        self.near_spin.setValue(0.01)
        depth_row.addWidget(self.near_spin)
        depth_row.addWidget(QLabel("far:"))
        self.far_spin = QDoubleSpinBox()
        self.far_spin.setRange(0.1, 1000.0)
        self.far_spin.setDecimals(1)
        self.far_spin.setSingleStep(1.0)
        self.far_spin.setValue(50.0)
        depth_row.addWidget(self.far_spin)
        lay.addLayout(depth_row)

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
        hint.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        lay.addWidget(hint)

        self.status = QLabel("就绪：加载世界后点「开始遥操作」")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{theme.C_TEXT_DIM};")
        lay.addWidget(self.status)

    # ------------------------------------------------------------ 外部接口

    def set_manifest(self, manifest: dict) -> None:
        """按 get_robot_manifest 填充机器人/相机下拉（保持当前选择）。"""
        self._manifest = manifest
        old_robot = self.robot_combo.currentData()
        old_cam = self.camera_combo.currentData()
        self.robot_combo.clear()
        self.camera_combo.clear()
        self.camera_combo.addItem(_FREE_CAMERA, "")
        for r in manifest.get("robots") or []:
            name = r.get("name")
            if not name:
                continue
            self.robot_combo.addItem(name, name)
            for cam in r.get("cameras") or []:
                self.camera_combo.addItem(f"{name}/{cam}", f"{name}/{cam}")
        for combo, old in ((self.robot_combo, old_robot),
                           (self.camera_combo, old_cam)):
            idx = combo.findData(old) if old is not None else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if self.camera_combo.count() > 1 and self.camera_combo.currentIndex() == 0:
            self.camera_combo.setCurrentIndex(1)   # 有挂载相机优先用它

    def set_running(self, running: bool) -> None:
        self.bt_start.setEnabled(not running)
        self.bt_stop.setEnabled(running)
        # 深度开关/裁剪面保持可用：运行中也能实时切伪彩看距离
        for w in (self.robot_combo, self.camera_combo, self.res_combo,
                  self.rec_chk, self.rec_dir):
            w.setEnabled(not running)

    def set_status(self, text: str, level: str = "info") -> None:
        color = {"ok": theme.C_OK, "warn": theme.C_WARN,
                 "error": theme.C_ERR}.get(level, theme.C_TEXT_DIM)
        self.status.setStyleSheet(f"color:{color};")
        self.status.setText(text)

    def show_frame(self, data: bytes) -> bool:
        return self.camera.show_png(data)

    def clear_frame(self) -> None:
        self.camera.clear()

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
