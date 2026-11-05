"""「造机器人」面板：DH 参数表驱动 + 实时预览 + 高级连杆编辑。

主模式是 Denavit-Hartenberg 参数表：每一行一个关节，填 θ₀/d/a/α 与
关节类型，几何体（关节球 + d 立管 + a 横杆 + 底座 + 法兰）按运动学
自动生成，杆长显式可控、不会互相插穿，编译后每个转动/移动关节自动带
位置执行器，保证能运动。

「高级连杆编辑」把 DH 展开成普通连杆树（单向：展开后不再保留 DH 表），
旧的逐连杆位姿/几何/关节表单仍可用于打开历史 links 格式机器人。

数据格式见 simovia/robot_spec.py。保存产物是机器人 YAML（dh 或 links），
可直接写进世界文件的 robots[].model。
"""
from __future__ import annotations

import copy
import math
import pathlib

import mujoco
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from simovia.errors import WorldError
from simovia.robot_spec import (
    DHRow, GEOM_TYPES, RobotDoc, build_robot_spec, dh_to_links, dump_robot,
    euler_to_matrix, link_frames, load_robot, matrix_to_quat, parse_robot,
)

#: 打开面板时的默认示例：SCARA（3 自由度：2 个水平转动 + 1 个竖直移动）
DEFAULT_SCARA = {
    "name": "my_scara",
    "dh": [
        {"theta": 0, "d": 0.20, "a": 0.25, "alpha": 0,
         "type": "revolute", "range": [-160, 160]},
        {"theta": 0, "d": 0.00, "a": 0.20, "alpha": 180,
         "type": "revolute", "range": [-135, 135]},
        # SCARA 的第三轴沿 z3（alpha2=180 使 z3 朝下）：d 正向向下
        {"theta": 0, "d": 0.10, "a": 0.00, "alpha": 0,
         "type": "prismatic", "range": [-0.05, 0.05]},
    ],
}

#: DH 表预设（"预设"下拉的选项）
DH_PRESETS: dict[str, dict] = {
    "SCARA（2 转动 + 1 移动）": DEFAULT_SCARA,
    "六轴工业臂": {
        "name": "my_arm6",
        "dh": [
            {"theta": 0, "d": 0.30, "a": 0.00, "alpha": 90,
             "type": "revolute", "range": [-170, 170]},
            {"theta": 0, "d": 0.00, "a": 0.25, "alpha": 0,
             "type": "revolute", "range": [-120, 120]},
            {"theta": 0, "d": 0.00, "a": 0.05, "alpha": 90,
             "type": "revolute", "range": [-130, 130]},
            {"theta": 0, "d": 0.22, "a": 0.00, "alpha": -90,
             "type": "revolute", "range": [-180, 180]},
            {"theta": 0, "d": 0.00, "a": 0.00, "alpha": 90,
             "type": "revolute", "range": [-120, 120]},
            {"theta": 0, "d": 0.08, "a": 0.00, "alpha": 0,
             "type": "revolute", "range": [-360, 360]},
        ],
    },
    "平面 2R 两连杆": {
        "name": "my_2r",
        "dh": [
            {"theta": 0, "d": 0.0, "a": 0.30, "alpha": 0,
             "type": "revolute", "range": [-170, 170]},
            {"theta": 0, "d": 0.0, "a": 0.22, "alpha": 0,
             "type": "revolute", "range": [-150, 150]},
        ],
    },
}

#: 关节类型下拉顺序（与 cb_joint.addItems 一一对应）
_JOINT_LABELS = ["fixed", "revolute", "prismatic", "ball", "free"]

#: DH 表类型列下拉 -> DHRow.joint_type
_DH_TYPE_LABELS = [("转动 revolute", "revolute"),
                   ("移动 prismatic", "prismatic"),
                   ("固定 fixed", "fixed")]

#: 角度范围 <-> 移动范围换算时假定的关节力臂（m）：0.1 m 半径处
_RANGE_ARM_M = 0.1


def _geom_half_height(gtype: str, size) -> float:
    """几何体沿自身 z 轴的半高（关节原点在底面中心时的几何偏置 z）。"""
    s = list(size) + [0.0, 0.0, 0.0]
    if gtype in ("box", "ellipsoid"):
        return float(s[2])
    if gtype == "sphere":
        return float(s[0])
    if gtype == "capsule":
        return float(s[1]) + float(s[0])
    return float(s[1])                    # cylinder: [半径, 半长]


def _link_top_z(lk) -> float:
    """连杆几何体在自身坐标系中的最高点 z（几何带 rpy 时不精确）。"""
    g = lk.geom
    if not g:
        return 0.0
    pz = float(list(g.get("pos", (0.0, 0.0, 0.0)))[2])
    return pz + _geom_half_height(g["type"], g["size"])


def _world_to_screen(glcam, point, w: int, h: int):
    """世界坐标 -> 离屏图像像素坐标；点在相机后方或退化时返回 None。

    mjvGLCamera.frustum_width 在 update_scene 后仍为 0，只能用
    frustum_top/frustum_near 还原垂直半视角，再乘宽高比得到水平方向。
    """
    pos = np.asarray(glcam.pos, dtype=float)
    fwd = np.asarray(glcam.forward, dtype=float)
    up = np.asarray(glcam.up, dtype=float)
    near, top = float(glcam.frustum_near), float(glcam.frustum_top)
    if near <= 0.0 or top <= 0.0:
        return None
    tan_v = top / near
    aspect = w / max(h, 1)
    right = np.cross(fwd, up)
    norm = float(np.linalg.norm(right))
    if norm < 1e-9:
        return None
    right /= norm
    cam_up = np.cross(right, fwd)
    d = np.asarray(point, dtype=float) - pos
    depth = float(np.dot(d, fwd))
    if depth <= 1e-6:
        return None
    x_ndc = float(np.dot(d, right)) / (depth * tan_v * aspect)
    y_ndc = float(np.dot(d, cam_up)) / (depth * tan_v)
    return ((x_ndc + 1.0) * 0.5 * w, (1.0 - y_ndc) * 0.5 * h)


class RobotBuilderDialog(QDialog):
    """非模态：可一边开着面板改参数，一边回主窗口看场景。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("造机器人 —— DH 参数表 / 运动学")
        self.resize(1180, 760)
        self.robot: RobotDoc = parse_robot(copy.deepcopy(DEFAULT_SCARA))
        self.mode = "dh"               # "dh" 或 "links"（高级连杆编辑）
        self._loading = False
        self._dirty = False
        self._doc_path = None
        self._renderer = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._build_ui()
        self._refresh_dh_table(select_row=0)
        self._refresh_tree(select=self.robot.links[0].name)
        self._set_mode("dh")
        self._update_ee_info()
        self._update_matrix()
        self._refresh_preview()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._release_preview)

    # ------------------------------------------------------------ UI 骨架

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # 顶栏：名称 + 模式切换 + 预设（DH）
        top = QHBoxLayout()
        top.addWidget(QLabel("机器人名"))
        self.ed_name = QLineEdit()
        self.ed_name.setMaximumWidth(160)
        self.ed_name.editingFinished.connect(self._on_name_changed)
        top.addWidget(self.ed_name)
        top.addSpacing(12)
        self.lbl_mode = QLabel("")
        self.lbl_mode.setStyleSheet("color:#2563EB; font-weight:600;")
        top.addWidget(self.lbl_mode)
        top.addStretch(1)
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("载入 DH 预设…")
        self.cb_preset.addItems(list(DH_PRESETS))
        self.cb_preset.currentIndexChanged.connect(self._on_preset)
        top.addWidget(self.cb_preset)
        self.btn_mode = QPushButton("高级：逐连杆编辑")
        self.btn_mode.clicked.connect(self._toggle_mode)
        top.addWidget(self.btn_mode)
        root.addLayout(top)

        body = QHBoxLayout()

        # 左栏：DH 表 / 连杆树 堆叠
        self.stack_left = QStackedWidget()
        self.stack_left.addWidget(self._build_dh_page())     # 0
        self.stack_left.addWidget(self._build_tree_page())   # 1
        self.stack_left.setFixedWidth(486)
        body.addWidget(self.stack_left)

        # 中栏：DH 说明 / 连杆属性表单 堆叠
        self.stack_mid = QStackedWidget()
        self.stack_mid.addWidget(self._build_dh_info_page())  # 0
        self.stack_mid.addWidget(self._build_form_page())     # 1
        body.addWidget(self.stack_mid)

        # 右栏：预览 + 矩阵（两模式共用）
        body.addLayout(self._build_preview_page(), 1)
        root.addLayout(body, 1)

    # -- 左：DH 参数表 ---------------------------------------------------

    def _build_dh_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.addWidget(QLabel(
            "DH 参数表（标准约定  T = Rz(θ)·Tz(d)·Tx(a)·Rx(α)）"))
        self.dh_table = QTableWidget(0, 8)
        self.dh_table.setHorizontalHeaderLabels(
            ["#", "θ₀°", "d m", "a m", "α°", "关节", "min", "max"])
        hdr = self.dh_table.horizontalHeader()
        hdr.setSectionResizeMode(hdr.ResizeMode.Fixed)
        for col, w in enumerate((26, 58, 62, 62, 54, 96, 58, 58)):
            self.dh_table.setColumnWidth(col, w)
        self.dh_table.verticalHeader().setVisible(False)
        self.dh_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.dh_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.dh_table.currentCellChanged.connect(self._on_dh_row_sel)
        lay.addWidget(self.dh_table)

        row = QHBoxLayout()
        for text, slot in (("＋ 添加关节", self._dh_add_row),
                           ("删除选中", self._dh_del_row),
                           ("上移", self._dh_move_up),
                           ("下移", self._dh_move_down)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)

        tune = QFormLayout()
        tune.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.sp_base = self._spin(0.0, 1.0, 0.01, suffix=" m")
        self.sp_base.valueChanged.connect(self._on_dh_edited)
        tune.addRow("底座高度", self.sp_base)
        self.sp_dh_kp = self._spin(0, 5000, 5, 1, suffix=" N·m/rad")
        self.sp_dh_kd = self._spin(0, 200, 0.1, 2, suffix=" N·s/rad")
        self.sp_dh_kp.valueChanged.connect(self._on_dh_edited)
        self.sp_dh_kd.valueChanged.connect(self._on_dh_edited)
        tune.addRow("选中行伺服 kp / 阻尼 kd",
                    self._row([self.sp_dh_kp, self.sp_dh_kd]))
        box = QGroupBox("骨架与伺服")
        box.setLayout(tune)
        lay.addWidget(box)
        return page

    # -- 左：连杆树（高级） ----------------------------------------------

    def _build_tree_page(self) -> QWidget:
        page = QWidget()
        left = QVBoxLayout(page)
        left.setContentsMargins(0, 4, 0, 0)
        left.addWidget(QLabel("连杆树（父子关系）"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["连杆", "父连杆", "关节"])
        self.tree.setColumnWidth(0, 120)
        self.tree.currentItemChanged.connect(self._on_tree_sel)
        left.addWidget(self.tree)
        row = QHBoxLayout()
        self.btn_add = QPushButton("添加子连杆")
        self.btn_add.clicked.connect(self._add_child)
        row.addWidget(self.btn_add)
        self.btn_del = QPushButton("删除连杆")
        self.btn_del.clicked.connect(self._del_link)
        row.addWidget(self.btn_del)
        left.addLayout(row)
        return page

    # -- 中：DH 说明 -----------------------------------------------------

    def _build_dh_info_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        gb = QGroupBox("这一行参数的含义")
        form = QFormLayout(gb)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.lbl_dh_hint = QLabel("")
        self.lbl_dh_hint.setWordWrap(True)
        self.lbl_dh_hint.setStyleSheet("color:#374151;")
        form.addRow(self.lbl_dh_hint)
        lay.addWidget(gb)

        gb2 = QGroupBox("零位运动学")
        f2 = QVBoxLayout(gb2)
        self.lbl_ee = QLabel("")
        self.lbl_ee.setWordWrap(True)
        self.lbl_ee.setStyleSheet("font-family:monospace;")
        f2.addWidget(self.lbl_ee)
        hint = QLabel(
            "θ₀：转动关节零位偏置（度）；关节变量 q 叠加在 θ₀ 上。\n"
            "d：沿当前 z 轴的连杆偏置（米）；移动关节的变量也沿此轴。\n"
            "a：沿（转动后的）x 轴的连杆长度（米）——这就是连杆长度。\n"
            "α：绕 x 轴的扭转角（度），决定相邻关节轴是否平行/垂直。\n"
            "几何体自动生成：底座立柱、关节球、d 立管、a 横杆、末端法兰。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6B7280; font-size:11px;")
        f2.addWidget(hint)
        lay.addWidget(gb2)
        lay.addStretch(1)
        sw = QScrollArea()
        sw.setWidgetResizable(True)
        sw.setWidget(page)
        sw.setFixedWidth(370)
        holder = QWidget()
        h = QVBoxLayout(holder)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(sw)
        return holder

    # -- 中：连杆属性表单（高级） ----------------------------------------

    def _build_form_page(self) -> QWidget:
        box = QGroupBox("选中连杆")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.ed_link_name = QLineEdit()
        self.ed_link_name.editingFinished.connect(self._on_name_changed)
        form.addRow("名称", self.ed_link_name)
        self.cb_parent = QComboBox()
        self.cb_parent.currentIndexChanged.connect(self._on_form)
        form.addRow("父连杆", self.cb_parent)
        self.sp_pos = [self._spin(-10, 10, 0.01, suffix=" m")
                       for _ in range(3)]
        form.addRow("平移 pos x/y/z", self._row(self.sp_pos))
        self.sp_rpy = [self._spin(-360, 360, 1, 1) for _ in range(3)]
        for s in self.sp_rpy:
            s.setSuffix("°")
        form.addRow("旋转 rpy（度）", self._row(self.sp_rpy))
        self.sp_mass = self._spin(0, 1000, 0.1, suffix=" kg")
        form.addRow("质量（0=按密度）", self.sp_mass)
        self.cb_geom = QComboBox()
        self.cb_geom.addItems(["（无几何）", *GEOM_TYPES.keys()])
        self.cb_geom.currentIndexChanged.connect(self._on_geom_type)
        form.addRow("几何类型", self.cb_geom)
        self.sp_gsize = [self._spin(0, 10, 0.005, suffix=" m")
                         for _ in range(3)]
        form.addRow("尺寸（半长/半径）", self._row(self.sp_gsize))
        self.sp_gpos = [self._spin(-10, 10, 0.005, suffix=" m")
                        for _ in range(3)]
        form.addRow("几何偏置", self._row(self.sp_gpos))
        self.cb_joint = QComboBox()
        self.cb_joint.addItems(["fixed（固定）", "revolute（转动）",
                                "prismatic（移动）", "ball（球关节）",
                                "free（浮动基座）"])
        self.cb_joint.currentIndexChanged.connect(self._on_joint_type)
        form.addRow("关节类型", self.cb_joint)
        self.sp_axis = [self._spin(-1, 1, 0.1, 3) for _ in range(3)]
        form.addRow("关节轴", self._row(self.sp_axis))
        self.sp_range = [self._spin(-3600, 3600, 1, 1, suffix="°")
                         for _ in range(2)]
        form.addRow("关节范围 min/max", self._row(self.sp_range))
        self.sp_kp = self._spin(0, 5000, 5, 1, suffix=" N·m/rad")
        self.sp_kd = self._spin(0, 200, 0.1, 2, suffix=" N·s/rad")
        form.addRow("伺服 kp / 阻尼 kd", self._row([self.sp_kp, self.sp_kd]))
        for w in (self.sp_pos + self.sp_rpy + [self.sp_mass] +
                  self.sp_gsize + self.sp_gpos + self.sp_axis +
                  self.sp_range + [self.sp_kp, self.sp_kd]):
            w.valueChanged.connect(self._on_form)
        for w in self.sp_gsize:
            w.valueChanged.connect(self._on_geom_size)
        self.sp_gpos[2].valueChanged.connect(self._on_geom_pos)

        mid_w = QWidget()
        mid = QVBoxLayout(mid_w)
        mid.addWidget(box)
        mid.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(mid_w)
        scroll.setFixedWidth(370)
        holder = QWidget()
        h = QVBoxLayout(holder)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(scroll)
        return holder

    # -- 右：预览 + 矩阵 -------------------------------------------------

    def _build_preview_page(self) -> QVBoxLayout:
        right = QVBoxLayout()
        self.lbl_img = QLabel("预览")
        self.lbl_img.setMinimumSize(240, 240)
        self.lbl_img.setSizePolicy(QSizePolicy.Policy.Ignored,
                                   QSizePolicy.Policy.Ignored)
        self._preview_pix: QPixmap | None = None
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet(
            "background:#111827; color:#9CA3AF; border-radius:6px;")
        right.addWidget(self.lbl_img, 1)
        self.lbl_err = QLabel("")
        self.lbl_err.setWordWrap(True)
        self.lbl_err.setStyleSheet("color:#DC2626;")
        right.addWidget(self.lbl_err)
        gb = QGroupBox("坐标系矩阵（当前选中）")
        gb_lay = QVBoxLayout(gb)
        self.lbl_formula = QLabel("")
        self.lbl_formula.setStyleSheet(
            "font-family:monospace; color:#374151;")
        gb_lay.addWidget(self.lbl_formula)
        self.tb_rel = QTableWidget(3, 3)
        self.tb_world = QTableWidget(3, 3)
        for tb in (self.tb_rel, self.tb_world):
            tb.horizontalHeader().setVisible(False)
            tb.verticalHeader().setVisible(False)
            tb.horizontalHeader().setDefaultSectionSize(50)
            tb.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            tb.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            tb.setFixedHeight(96)
            tb.setFixedWidth(150)
        row2 = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(QLabel("相对父系  R_rel"))
        col1.addWidget(self.tb_rel)
        col2 = QVBoxLayout()
        col2.addWidget(QLabel("世界系  R_world"))
        col2.addWidget(self.tb_world)
        row2.addLayout(col1)
        row2.addLayout(col2)
        gb_lay.addLayout(row2)
        self.lbl_pose = QLabel("")
        self.lbl_pose.setStyleSheet("font-family:monospace;")
        gb_lay.addWidget(self.lbl_pose)
        gb.setMaximumHeight(300)
        right.addWidget(gb, 0)

        btns = QHBoxLayout()
        b_new = QPushButton("新建")
        b_new.clicked.connect(self._new_robot)
        b_open = QPushButton("打开…")
        b_open.clicked.connect(self._open)
        b_close = QPushButton("关闭")
        b_close.clicked.connect(self.close)
        btns.addWidget(b_new)
        btns.addWidget(b_open)
        btns.addStretch(1)
        btns.addWidget(b_close)
        right.addLayout(btns)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        b_save = QPushButton("保存为机器人 YAML…")
        b_save.setObjectName("primary")
        b_save.clicked.connect(lambda: self._save(ask=True))
        save_row.addWidget(b_save)
        right.addLayout(save_row)
        return right

    @staticmethod
    def _spin(lo, hi, step, decimals=3, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setMaximumWidth(96)
        if suffix:
            s.setSuffix(suffix)
        return s

    @staticmethod
    def _row(widgets):
        lay = QHBoxLayout()
        for w in widgets:
            lay.addWidget(w)
        holder = QWidget()
        holder.setLayout(lay)
        lay.setContentsMargins(0, 0, 0, 0)
        return holder

    # ------------------------------------------------------------ 模式切换

    def _toggle_mode(self):
        if self.mode == "dh":
            # DH -> links 是单向展开：展开后 DH 表不再保留
            r = QMessageBox.question(
                self, "切换到逐连杆编辑",
                "高级模式会把 DH 表展开成普通连杆树，之后本机器人将不再\n"
                "保留 DH 参数表（几何体已自动生成，可自由增删改）。\n\n"
                "确定展开吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if r != QMessageBox.StandardButton.Yes:
                return
            self.robot.dh = None
            self._dirty = True
            self._set_mode("links")
            self._refresh_tree(select=self.robot.links[0].name)
            self._schedule_preview()
        else:
            # links -> dh 只能新建：旧 links 无法反解成唯一 DH 表
            if not self._confirm_discard("新建 DH 机器人"):
                return
            self._load_doc(copy.deepcopy(DEFAULT_SCARA))

    def _set_mode(self, mode: str):
        self.mode = mode
        is_dh = mode == "dh"
        self.stack_left.setCurrentIndex(0 if is_dh else 1)
        self.stack_mid.setCurrentIndex(0 if is_dh else 1)
        self.btn_mode.setText("高级：逐连杆编辑" if is_dh else "回到 DH（新建）")
        self.lbl_mode.setText("DH 参数表模式" if is_dh
                              else "高级连杆编辑（links）")
        self.cb_preset.setVisible(is_dh)
        self.ed_name.setText(self.robot.name)

    # ------------------------------------------------------------ DH 表格

    def _dh_make_spin(self, kind: str) -> QDoubleSpinBox:
        if kind == "angle":
            s = self._spin(-3600, 3600, 5, 1, suffix="°")
        elif kind == "length":
            s = self._spin(-10, 10, 0.01, 3, suffix=" m")
        else:  # range 单位随关节类型切换，初始按角度
            s = self._spin(-3600, 3600, 5, 1, suffix="°")
        s.setMaximumWidth(56)
        s.valueChanged.connect(self._on_dh_edited)
        return s

    def _refresh_dh_table(self, select_row: int = 0):
        """按 self.robot.dh 重建表格内容（增删行/预设/加载时调用）。"""
        self._loading = True
        try:
            rows = self.robot.dh or []
            self.dh_table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                num = QTableWidgetItem(str(i + 1))
                num.setFlags(Qt.ItemFlag.ItemIsEnabled)
                num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.dh_table.setItem(i, 0, num)

                sp_th = self._dh_make_spin("angle")
                sp_d = self._dh_make_spin("length")
                sp_a = self._dh_make_spin("length")
                sp_al = self._dh_make_spin("angle")
                sp_th.setValue(float(r.theta_deg))
                sp_d.setValue(float(r.d))
                sp_a.setValue(float(r.a))
                sp_al.setValue(float(r.alpha_deg))

                cb = QComboBox()
                for label, _ in _DH_TYPE_LABELS:
                    cb.addItem(label)
                idx = next((k for k, (_, v) in enumerate(_DH_TYPE_LABELS)
                            if v == r.joint_type), 0)
                cb.setCurrentIndex(idx)
                cb.currentIndexChanged.connect(self._on_dh_type_changed)

                sp_lo = self._dh_make_spin("range")
                sp_hi = self._dh_make_spin("range")
                if r.range is not None:
                    sp_lo.setValue(float(r.range[0]))
                    sp_hi.setValue(float(r.range[1]))
                else:
                    sp_lo.setValue(-180.0)
                    sp_hi.setValue(180.0)

                for col, w in ((1, sp_th), (2, sp_d), (3, sp_a),
                               (4, sp_al), (5, cb), (6, sp_lo), (7, sp_hi)):
                    self.dh_table.setCellWidget(i, col, w)
                self._dh_apply_range_units(i)
            self.sp_base.setValue(float(self.robot.base_height))
            if rows:
                self.dh_table.setCurrentCell(
                    min(max(select_row, 0), len(rows) - 1), 1)
            self._dh_sync_row_form()
        finally:
            self._loading = False

    def _dh_widgets(self, row: int) -> dict | None:
        if row < 0 or row >= self.dh_table.rowCount():
            return None
        get = self.dh_table.cellWidget
        return {"theta": get(row, 1), "d": get(row, 2), "a": get(row, 3),
                "alpha": get(row, 4), "type": get(row, 5),
                "lo": get(row, 6), "hi": get(row, 7)}

    def _dh_current_row(self) -> int:
        r = self.dh_table.currentRow()
        return r if r >= 0 else 0

    def _dh_apply_range_units(self, row: int):
        """范围 spin 的单位/可用性随这一行关节类型变化（度 vs 米）。"""
        ws = self._dh_widgets(row)
        if ws is None:
            return
        jt = _DH_TYPE_LABELS[ws["type"].currentIndex()][1]
        for sp in (ws["lo"], ws["hi"]):
            sp.blockSignals(True)
            if jt == "prismatic":
                sp.setSuffix(" m")
                sp.setDecimals(3)
                sp.setSingleStep(0.01)
                # 切到米：若当前是角度量级（±几十上百），给米级默认
                if abs(sp.value()) > 5.0:
                    sp.setValue(-0.05 if sp is ws["lo"] else 0.05)
            else:
                sp.setSuffix("°")
                sp.setDecimals(1)
                sp.setSingleStep(5)
                if abs(sp.value()) <= 5.0 and jt != "fixed":
                    sp.setValue(-180.0 if sp is ws["lo"] else 180.0)
            sp.blockSignals(False)
            sp.setEnabled(jt != "fixed")

    def _on_dh_type_changed(self, *_):
        if self._loading:
            return
        self._dh_apply_range_units(self._dh_current_row())
        self._on_dh_edited()

    def _on_dh_row_sel(self, *_):
        if self._loading:
            return
        self._dh_sync_row_form()
        self._update_matrix()
        self._schedule_preview()

    def _dh_sync_row_form(self):
        """选中行的 kp/kd 与含义说明同步到中栏。"""
        rows = self.robot.dh or []
        i = self._dh_current_row()
        if not rows or i >= len(rows):
            self.lbl_dh_hint.setText("")
            return
        r = rows[i]
        var = "θ（绕 z 转动）" if r.joint_type == "revolute" else (
            "d（沿 z 移动）" if r.joint_type == "prismatic" else "无（固定）")
        self.lbl_dh_hint.setText(
            f"关节 {i + 1}：{var}\n"
            f"θ₀ = {r.theta_deg:g}°，d = {r.d:g} m，"
            f"a = {r.a:g} m，α = {r.alpha_deg:g}°\n"
            f"执行器名：joint{i + 1}（robot.set(\"joint{i + 1}\", …)）"
            if r.joint_type != "fixed" else
            f"关节 {i + 1}：固定行，不产生自由度")
        self._loading = True
        try:
            self.sp_dh_kp.setValue(float(r.kp))
            self.sp_dh_kd.setValue(float(r.kd))
            jt = r.joint_type
            self.sp_dh_kp.setSuffix(" N/m" if jt == "prismatic"
                                    else " N·m/rad")
            self.sp_dh_kd.setSuffix(" N·s/m" if jt == "prismatic"
                                    else " N·s/rad")
            self.sp_dh_kp.setEnabled(jt != "fixed")
            self.sp_dh_kd.setEnabled(jt != "fixed")
        finally:
            self._loading = False

    def _read_dh_table(self) -> list[DHRow]:
        rows = []
        for i in range(self.dh_table.rowCount()):
            ws = self._dh_widgets(i)
            jt = _DH_TYPE_LABELS[ws["type"].currentIndex()][1]
            rng = None
            if jt != "fixed":
                lo, hi = ws["lo"].value(), ws["hi"].value()
                if lo < hi:
                    rng = (float(lo), float(hi))
            old = self.robot.dh[i] if i < len(self.robot.dh or []) else None
            rows.append(DHRow(
                theta_deg=float(ws["theta"].value()),
                d=float(ws["d"].value()),
                a=float(ws["a"].value()),
                alpha_deg=float(ws["alpha"].value()),
                joint_type=jt, range=rng,
                kp=float(old.kp if old is not None else 50.0),
                kd=float(old.kd if old is not None else 1.0)))
        return rows

    def _on_dh_edited(self, *_):
        """表格/底座/kp,kd -> 重建 DH 运动学与骨架几何 -> 刷新信息与预览。"""
        if self._loading or not self.robot.dh:
            return
        i = self._dh_current_row()
        rows = self._read_dh_table()
        # kp/kd 表单只写当前选中行
        if 0 <= i < len(rows):
            rows[i].kp = float(self.sp_dh_kp.value())
            rows[i].kd = float(self.sp_dh_kd.value())
        self.robot.name = self.ed_name.text().strip() or self.robot.name
        self.robot.base_height = float(self.sp_base.value())
        self.robot.dh = rows
        self.robot.links = dh_to_links(
            rows, self.robot.name, base_height=self.robot.base_height)
        self._dirty = True
        self._dh_sync_row_form()
        self._update_ee_info()
        self._update_matrix()
        self._schedule_preview()

    def _update_ee_info(self):
        try:
            frames = {f.name: f for f in link_frames(self.robot)}
        except (WorldError, KeyError):
            self.lbl_ee.setText("")
            return
        ee = frames.get("flange")
        if ee is None:
            self.lbl_ee.setText("")
            return
        self.lbl_ee.setText(
            f"自由度：{df_count(self.robot)}\n"
            f"末端法兰零位：\n"
            f"x = {ee.p_world[0]:+.3f} m\n"
            f"y = {ee.p_world[1]:+.3f} m\n"
            f"z = {ee.p_world[2]:+.3f} m")

    def _dh_add_row(self):
        rows = list(self.robot.dh or [])
        i = self._dh_current_row() + 1 if rows else 0
        rows.insert(i, DHRow(a=0.15))
        self._apply_dh_rows(rows, select_row=i)

    def _dh_del_row(self):
        rows = list(self.robot.dh or [])
        if not rows:
            return
        i = self._dh_current_row()
        del rows[i]
        if not rows:
            QMessageBox.information(self, "不能删除", "至少保留一个关节")
            return
        self._apply_dh_rows(rows, select_row=min(i, len(rows) - 1))

    def _dh_move_up(self):
        rows = list(self.robot.dh or [])
        i = self._dh_current_row()
        if i <= 0:
            return
        rows[i - 1], rows[i] = rows[i], rows[i - 1]
        self._apply_dh_rows(rows, select_row=i - 1)

    def _dh_move_down(self):
        rows = list(self.robot.dh or [])
        i = self._dh_current_row()
        if i >= len(rows) - 1:
            return
        rows[i + 1], rows[i] = rows[i], rows[i + 1]
        self._apply_dh_rows(rows, select_row=i + 1)

    def _apply_dh_rows(self, rows: list[DHRow], select_row: int = 0):
        self.robot.dh = rows
        self.robot.links = dh_to_links(
            rows, self.robot.name, base_height=self.robot.base_height)
        self._dirty = True
        self._refresh_dh_table(select_row=select_row)
        self._update_ee_info()
        self._update_matrix()
        self._schedule_preview()

    def _on_preset(self, idx: int):
        if idx <= 0:
            return
        name = self.cb_preset.itemText(idx)
        doc = copy.deepcopy(DH_PRESETS[name])
        self._load_doc(doc)
        self.cb_preset.blockSignals(True)
        self.cb_preset.setCurrentIndex(0)
        self.cb_preset.blockSignals(False)

    # ------------------------------------------------------------ 树（高级）

    def _current_link_name(self) -> str | None:
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def _select_link(self, name: str):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            stack = [it]
            while stack:
                cur = stack.pop(0)
                if cur.data(0, Qt.ItemDataRole.UserRole) == name:
                    self.tree.setCurrentItem(cur)
                    return
                stack.extend(cur.child(j) for j in range(cur.childCount()))

    def _refresh_tree(self, select: str | None = None):
        select = select or self._current_link_name()
        self.tree.blockSignals(True)
        self.tree.clear()
        by_parent: dict[str | None, list] = {}
        for lk in self.robot.links:
            by_parent.setdefault(lk.parent, []).append(lk)

        def add(parent_item, parent_name):
            for lk in by_parent.get(parent_name, []):
                it = QTreeWidgetItem([lk.name, lk.parent or "—",
                                      lk.joint.type])
                it.setData(0, Qt.ItemDataRole.UserRole, lk.name)
                if parent_item is None:
                    self.tree.addTopLevelItem(it)
                else:
                    parent_item.addChild(it)
                add(it, lk.name)

        add(None, None)
        self.tree.expandAll()
        self.tree.blockSignals(False)
        if select:
            self._select_link(select)
        elif self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._load_form()

    def _on_tree_sel(self, *_):
        self._load_form()
        self._update_matrix()
        self._schedule_preview()

    # ------------------------------------------------------------ 属性表单

    def _load_form(self):
        name = self._current_link_name()
        if not name:
            return
        self._loading = True
        lk = self.robot.link(name)
        self.ed_link_name.setText(lk.name)
        banned = {lk.name}
        changed = True
        while changed:
            changed = False
            for other in self.robot.links:
                if other.parent in banned and other.name not in banned:
                    banned.add(other.name)
                    changed = True
        self.cb_parent.clear()
        self.cb_parent.addItem("（无，根连杆）", None)
        for other in self.robot.links:
            if other.name not in banned:
                self.cb_parent.addItem(other.name, other.name)
        idx = max(0, self.cb_parent.findData(lk.parent))
        self.cb_parent.setCurrentIndex(idx)
        for w, v in zip(self.sp_pos, lk.pos):
            w.setValue(float(v))
        for w, v in zip(self.sp_rpy, lk.rpy_deg):
            w.setValue(float(v))
        self.sp_mass.setValue(float(lk.mass or 0.0))
        # DH 展开的连杆可能带多个几何（geoms）：表单显示/编辑第一个
        self._form_geom_multi = bool(lk.geoms)
        g = lk.geom if lk.geom is not None else (
            lk.geoms[0] if lk.geoms else None)
        self.cb_geom.setCurrentIndex(
            0 if g is None else 1 + list(GEOM_TYPES).index(g["type"]))
        size = list(g["size"]) if g else [0.05, 0.05, 0.05]
        for w, v in zip(self.sp_gsize, (size + [0.0, 0.0, 0.0])[:3]):
            w.setValue(float(v))
        gpos = list(g.get("pos", [0, 0, 0])) if g else [0, 0, 0]
        for w, v in zip(self.sp_gpos, gpos):
            w.setValue(float(v))
        if g is not None:
            hh = _geom_half_height(g["type"], size)
            self._origin_at_bottom = abs(float(gpos[2]) - hh) < 1e-6
        else:
            self._origin_at_bottom = True
        j = lk.joint
        self.cb_joint.setCurrentIndex(_JOINT_LABELS.index(j.type)
                                      if j.type in _JOINT_LABELS else 0)
        for w, v in zip(self.sp_axis, j.axis):
            w.setValue(float(v))
        rng = j.range or (-180.0, 180.0)
        for w, v in zip(self.sp_range, rng):
            w.setValue(float(v))
        self.sp_kp.setValue(float(j.kp))
        self.sp_kd.setValue(float(j.kd))
        self._apply_enabled()
        self._apply_joint_units()
        self._loading = False
        self._update_matrix()

    def _apply_enabled(self):
        idx = self.cb_joint.currentIndex()
        jt = _JOINT_LABELS[idx] if 0 <= idx < len(_JOINT_LABELS) else "fixed"
        movable = jt in ("revolute", "prismatic", "ball")
        for w in self.sp_axis:
            w.setEnabled(movable)
        for w in self.sp_range + [self.sp_kp, self.sp_kd]:
            w.setEnabled(jt in ("revolute", "prismatic"))
        for w in self.sp_gsize + self.sp_gpos:
            w.setEnabled(self.cb_geom.currentIndex() > 0)

    def _on_geom_type(self, *_):
        if self._loading:
            return
        if self.cb_geom.currentIndex() > 0:
            gtype = list(GEOM_TYPES)[self.cb_geom.currentIndex() - 1]
            size = {
                "box": (0.03, 0.03, 0.075),
                "sphere": (0.05, 0.0, 0.0),
                "cylinder": (0.04, 0.075, 0.0),
                "capsule": (0.03, 0.06, 0.0),
                "ellipsoid": (0.04, 0.04, 0.075),
            }[gtype]
            self._loading = True
            try:
                for w, v in zip(self.sp_gsize, size):
                    w.setValue(v)
                self.sp_gpos[0].setValue(0.0)
                self.sp_gpos[1].setValue(0.0)
                self.sp_gpos[2].setValue(_geom_half_height(gtype, size))
            finally:
                self._loading = False
            self._origin_at_bottom = True
        self._apply_enabled()
        self._on_form()

    def _on_geom_size(self, *_):
        if self._loading or self.cb_geom.currentIndex() == 0:
            return
        if not getattr(self, "_origin_at_bottom", True):
            return
        gtype = list(GEOM_TYPES)[self.cb_geom.currentIndex() - 1]
        z = _geom_half_height(gtype, [w.value() for w in self.sp_gsize])
        if abs(self.sp_gpos[2].value() - z) > 1e-9:
            self.sp_gpos[2].setValue(z)

    def _on_geom_pos(self, *_):
        if self._loading or self.cb_geom.currentIndex() == 0:
            return
        gtype = list(GEOM_TYPES)[self.cb_geom.currentIndex() - 1]
        hh = _geom_half_height(gtype, [w.value() for w in self.sp_gsize])
        if abs(self.sp_gpos[2].value() - hh) > 1e-6:
            self._origin_at_bottom = False

    def _set_range_silently(self, lo: float, hi: float):
        self._loading = True
        try:
            self.sp_range[0].setValue(lo)
            self.sp_range[1].setValue(hi)
        finally:
            self._loading = False

    def _on_joint_type(self, *_):
        if self._loading:
            return
        name = self._current_link_name()
        lk = self.robot.link(name) if name else None
        prev = lk.joint.type if lk is not None else "fixed"
        jt = _JOINT_LABELS[self.cb_joint.currentIndex()]
        lo, hi = (self.sp_range[0].value(), self.sp_range[1].value())
        self._apply_enabled()
        self._apply_joint_units()
        arm = _RANGE_ARM_M
        if prev in ("revolute", "hinge") and jt == "prismatic":
            self._set_range_silently(
                lo * math.radians(1) * arm, hi * math.radians(1) * arm)
        elif prev == "prismatic" and jt in ("revolute", "hinge"):
            self._set_range_silently(
                max(-3600.0, lo / arm * math.degrees(1)),
                min(3600.0, hi / arm * math.degrees(1)))
        elif jt == "prismatic" and prev in ("fixed", "ball", "free"):
            self._set_range_silently(-0.5, 0.5)
        elif jt in ("revolute", "hinge") and prev in ("fixed", "ball", "free"):
            self._set_range_silently(-170.0, 170.0)
        self._on_form()

    def _apply_joint_units(self):
        jt = _JOINT_LABELS[self.cb_joint.currentIndex()]
        if jt == "prismatic":
            for w in self.sp_range:
                w.setSuffix(" m")
                w.setDecimals(3)
                w.setSingleStep(0.01)
            self.sp_kp.setSuffix(" N/m")
            self.sp_kd.setSuffix(" N·s/m")
        else:
            for w in self.sp_range:
                w.setSuffix("°")
                w.setDecimals(1)
                w.setSingleStep(1.0)
            self.sp_kp.setSuffix(" N·m/rad")
            self.sp_kd.setSuffix(" N·s/rad")

    def _tree_error(self) -> str:
        names = {lk.name for lk in self.robot.links}
        roots = [lk.name for lk in self.robot.links if lk.parent is None]
        if len(roots) != 1:
            return f"必须恰有一个根连杆，当前有 {len(roots)} 个: {roots}"
        for lk in self.robot.links:
            if lk.parent is not None and lk.parent not in names:
                return f"连杆 '{lk.name}' 的父连杆 '{lk.parent}' 不存在"
            if lk.parent is not None and lk.joint.type == "free":
                return f"连杆 '{lk.name}' 的 free 关节只能给根连杆"
        return ""

    def _show_error(self, msg: str):
        self.lbl_err.setText(msg)
        self.lbl_err.setStyleSheet("color:#DC2626;")

    def _on_form(self, *_):
        if self._loading:
            return
        name = self._current_link_name()
        if not name:
            return
        lk = self.robot.link(name)
        parent = self.cb_parent.currentData()
        if parent != lk.parent:
            old = lk.parent
            if lk.joint.type == "free" and parent is not None:
                self._show_error("free 关节只能给根连杆，已还原父连杆")
                self._load_form()
                return
            lk.parent = parent
            err = self._tree_error()
            if err:
                lk.parent = old
                self._show_error(err)
                self._load_form()
                return
            self._dirty = True
            self._refresh_tree(select=name)
            self._schedule_preview()
            return
        self._dirty = True
        lk.pos = tuple(w.value() for w in self.sp_pos)
        rpy = tuple(w.value() for w in self.sp_rpy)
        lk.rpy_deg = rpy
        lk.quat = tuple(float(v) for v in
                        matrix_to_quat(euler_to_matrix(rpy)))
        lk.mass = self.sp_mass.value() or None
        jt = _JOINT_LABELS[self.cb_joint.currentIndex()]
        if jt == "free" and lk.parent is not None:
            jt = "fixed"
            self.cb_joint.setCurrentIndex(0)
        lk.joint.type = jt
        lk.joint.axis = tuple(w.value() for w in self.sp_axis)
        lk.joint.range = tuple(w.value() for w in self.sp_range)
        lk.joint.kp = self.sp_kp.value()
        lk.joint.kd = self.sp_kd.value()
        if self.cb_geom.currentIndex() == 0:
            lk.geom = None
            if getattr(self, "_form_geom_multi", False):
                lk.geoms = None           # DH 展开体：选"无几何"清空全部
        else:
            gtype = list(GEOM_TYPES)[self.cb_geom.currentIndex() - 1]
            g = dict(lk.geom or (lk.geoms[0] if lk.geoms else {}) or {})
            g.update(type=gtype,
                     size=tuple(w.value() for w in self.sp_gsize),
                     pos=tuple(w.value() for w in self.sp_gpos))
            g.setdefault("color", lk.color)
            if lk.geoms:
                lk.geoms[0] = g           # 多几何体：编辑作用于第一个
            else:
                lk.geom = g
        self._update_matrix()
        self._schedule_preview()

    def _on_name_changed(self):
        new = self.ed_name.text().strip() if self.mode == "dh" \
            else self.ed_link_name.text().strip()
        if not new or new == self.robot.name:
            return
        if self.mode == "dh":
            self.robot.name = new
            self._dirty = True
            return
        old = self._current_link_name()
        if not old or not new or new == old:
            return
        if any(lk.name == new for lk in self.robot.links):
            QMessageBox.warning(self, "重名", f"连杆名 '{new}' 已存在")
            self.ed_link_name.setText(old)
            return
        for lk in self.robot.links:
            if lk.name == old:
                lk.name = new
            if lk.parent == old:
                lk.parent = new
        self.robot.name = new
        self._dirty = True
        self._refresh_tree(select=new)

    # ------------------------------------------------------------ 增删连杆

    def _add_child(self):
        parent = self._current_link_name()
        if not parent:
            return
        base = f"link{len(self.robot.links)}"
        name, i = base, 1
        while any(lk.name == name for lk in self.robot.links):
            name = f"link{len(self.robot.links)}_{i}"
            i += 1
        from simovia.robot_spec import JointDoc, LinkDoc
        half_h = 0.075
        parent_lk = self.robot.link(parent)
        top = _link_top_z(parent_lk)
        lk = LinkDoc(
            name=name, parent=parent,
            pos=(0.0, 0.0, top if top > 1e-9 else 0.15),
            joint=JointDoc(type="revolute", range=(-170.0, 170.0)),
            geom={"type": "box", "size": (0.03, 0.03, half_h),
                  "pos": (0.0, 0.0, half_h)})
        self.robot.links.append(lk)
        self._dirty = True
        self._refresh_tree(select=name)

    def _del_link(self):
        name = self._current_link_name()
        if not name:
            return
        lk = self.robot.link(name)
        if lk.parent is None and sum(1 for x in self.robot.links
                                     if x.parent is None) == 1:
            QMessageBox.information(self, "不能删除", "根连杆必须保留")
            return
        for other in self.robot.links:
            if other.parent == name:
                other.parent = lk.parent
        self.robot.links = [x for x in self.robot.links if x.name != name]
        self._dirty = True
        self._refresh_tree(select=lk.parent)
        self._schedule_preview()

    def _confirm_discard(self, action: str) -> bool:
        if not self._dirty:
            return True
        r = QMessageBox.question(
            self, "未保存的机器人",
            f"机器人「{self.robot.name}」有未保存的修改。\n\n{action}前保存吗？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.Discard:
            return True
        return self._save(ask=not self._doc_path)

    def _load_doc(self, doc: dict | str):
        """载入新机器人文档（dict 或路径），按 is_dh 自动切模式。"""
        try:
            robot = load_robot(doc) if isinstance(doc, str) \
                else parse_robot(doc)
        except WorldError as e:
            QMessageBox.critical(self, "载入失败", str(e))
            return
        self.robot = robot
        self._doc_path = doc if isinstance(doc, str) else None
        self._dirty = False
        self._set_mode("dh" if robot.is_dh else "links")
        self.sp_base.blockSignals(True)
        self.sp_base.setValue(float(robot.base_height))
        self.sp_base.blockSignals(False)
        if robot.is_dh:
            self._refresh_dh_table(select_row=0)
            self._update_ee_info()
        else:
            self._refresh_tree(select=robot.links[0].name)
        self._update_matrix()
        self._refresh_preview()

    def _new_robot(self):
        if not self._confirm_discard("新建机器人"):
            return
        self._load_doc(copy.deepcopy(DEFAULT_SCARA))

    # ------------------------------------------------------------ 打开/保存

    def _open(self):
        if not self._confirm_discard("打开机器人"):
            return
        p, _ = QFileDialog.getOpenFileName(
            self, "打开机器人描述", str(pathlib.Path.cwd()),
            "机器人 YAML (*.yaml *.yml)")
        if not p:
            return
        self._load_doc(p)

    def _save(self, ask: bool = True) -> bool:
        """Schema + MuJoCo 编译校验通过才落盘；失败保留旧文件（F05）。"""
        p = self._doc_path
        if ask or not p:
            p, _ = QFileDialog.getSaveFileName(
                self, "保存机器人 YAML", p or f"{self.robot.name}.yaml",
                "机器人 YAML (*.yaml)")
            if not p:
                return False
        import yaml
        try:
            doc = dump_robot(self.robot)
            text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
            check = parse_robot(doc)
            build_robot_spec(check).compile()
        except (OSError, WorldError, ValueError, TypeError,
                mujoco.FatalError) as e:
            self._show_error(f"保存失败：{e}")
            QMessageBox.critical(self, "保存失败",
                                 f"机器人模型无效，未写入文件：\n{e}")
            return False
        try:
            pathlib.Path(p).write_text(text, encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return False
        self._doc_path = p
        self._dirty = False
        self.lbl_err.setText("")
        QMessageBox.information(
            self, "已保存",
            f"机器人已保存到:\n{p}\n\n在世界 YAML 里这样引用:\n"
            f"robots:\n  - name: {self.robot.name}\n"
            f"    model: {pathlib.Path(p).as_posix()}")
        return True

    # ------------------------------------------------------------ 矩阵显示

    def _selected_frame_name(self) -> str | None:
        if self.mode == "dh":
            if not (self.robot.dh or []):
                return None
            i = self._dh_current_row()
            return f"joint{i + 1}"
        return self._current_link_name()

    def _update_matrix(self):
        name = self._selected_frame_name()
        try:
            frames = {f.name: f for f in link_frames(self.robot)}
        except (WorldError, KeyError):
            self.tb_rel.clearContents()
            self.tb_world.clearContents()
            return
        f = frames.get(name)
        if f is None:
            self.tb_rel.clearContents()
            self.tb_world.clearContents()
            self.lbl_formula.setText("")
            self.lbl_pose.setText("")
            return
        for tb, R in ((self.tb_rel, f.R_rel), (self.tb_world, f.R_world)):
            for i in range(3):
                for j in range(3):
                    it = QTableWidgetItem(f"{R[i, j]:+.3f}")
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    tb.setItem(i, j, it)
        if self.mode == "dh" and self.robot.dh:
            i = self._dh_current_row()
            if i < len(self.robot.dh):
                r = self.robot.dh[i]
                self.lbl_formula.setText(
                    f"joint{i + 1}：{r.joint_type}，轴 = 本体 z\n"
                    f"θ₀={r.theta_deg:g}° d={r.d:g} a={r.a:g} "
                    f"α={r.alpha_deg:g}°")
        else:
            lk = self.robot.link(name)
            r, p, y = [round(v, 1) for v in lk.rpy_deg]
            self.lbl_formula.setText(
                f"R = Rz({y:g}°) · Ry({p:g}°) · Rx({r:g}°)")
        self.lbl_pose.setText(
            f"世界位置 p_world = "
            f"({f.p_world[0]:+.3f}, {f.p_world[1]:+.3f}, "
            f"{f.p_world[2]:+.3f}) m")

    # ------------------------------------------------------------ 预览渲染

    def _schedule_preview(self):
        self._preview_timer.start()

    def _preview_spec(self, floor_z: float):
        mj = build_robot_spec(self.robot)
        mj.worldbody.add_geom(
            name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3.0, 3.0, 0.1], rgba=[0.35, 0.37, 0.42, 1.0],
            pos=[0, 0, float(floor_z)], friction=[1.0, 0.005, 0.0001])
        mj.worldbody.add_light(name="preview_light", pos=[0.5, -0.5, 2.5],
                               dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5],
                               cutoff=90, exponent=2.0, castshadow=0)
        mj.visual.headlight.ambient = [0.35, 0.36, 0.38]
        mj.visual.headlight.diffuse = [0.4, 0.4, 0.4]
        mj.visual.quality.offsamples = 8
        return mj

    def _geom_world_points(self, lk, f, pts):
        """把一个连杆上的全部几何（geom/geoms）外接球端点加入取景点集。"""
        geom_list = list(lk.geoms) if lk.geoms else (
            [lk.geom] if lk.geom is not None else [])
        for g in geom_list:
            size = [float(v) for v in g["size"]] + [0.0, 0.0]
            local = [float(v) for v in g.get("pos", (0.0, 0.0, 0.0))]
            center = (np.asarray(f.p_world, dtype=float)
                      + np.asarray(f.R_world, dtype=float)
                      @ np.asarray(local, dtype=float))
            r = size[0] if g["type"] == "sphere" else max(size[:3])
            pts.append(center + r)
            pts.append(center - r)

    def _refresh_preview(self):
        name = self._selected_frame_name()
        try:
            frames = {f.name: f for f in link_frames(self.robot)}
            zmin = min([0.0] + [float(f.p_world[2]) for f in frames.values()])
            mj = self._preview_spec(min(0.0, zmin - 0.01))
            model = mj.compile()
        except (WorldError, ValueError, TypeError) as e:
            self.lbl_err.setText(str(e))
            return
        self.lbl_err.setText("")

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(model, 420, 420)
        cam = mujoco.MjvCamera()
        if frames:
            pts = []
            for f in frames.values():
                pts.append(np.asarray(f.p_world, dtype=float))
                self._geom_world_points(self.robot.link(f.name), f, pts)
            pts = np.array(pts)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            cam.lookat[:] = (lo + hi) / 2
            span = float(np.linalg.norm(hi - lo))
            cam.distance = max(0.6, span * 2.0)
        else:
            cam.lookat[:] = [0, 0, 0.3]
            cam.distance = 1.4
        cam.azimuth = 135
        cam.elevation = -22
        self._renderer.update_scene(data, camera=cam)
        rgb = self._renderer.render()
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        frame = frames.get(name) if name else None
        if frame is not None:
            self._draw_axes(img, self._renderer.scene.camera[0], frame)
        self._preview_pix = QPixmap.fromImage(img)
        self._fit_preview()

    #: 轴标颜色：x 红 / y 绿 / z 蓝
    _AXIS_COLORS = ((1.0, 0.25, 0.25), (0.25, 0.85, 0.35), (0.35, 0.55, 1.0))

    def _draw_axes(self, img: QImage, glcam, frame) -> None:
        """渲染结果上叠加当前关节/连杆坐标系三轴（2D，不被几何遮挡）。"""
        w, h = img.width(), img.height()
        origin = np.asarray(frame.p_world, dtype=float)
        ox, oy = _world_to_screen(glcam, origin, w, h) or (None, None)
        if ox is None:
            return
        R = np.asarray(frame.R_world, dtype=float)
        length = min(w, h) * 0.14
        head = length * 0.24
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1.5))
            painter.drawEllipse(QPointF(ox, oy), 4.0, 4.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i, rgb in enumerate(self._AXIS_COLORS):
                tip = _world_to_screen(glcam, origin + R[:, i] * 0.05, w, h)
                if tip is None:
                    continue
                dx, dy = tip[0] - ox, tip[1] - oy
                norm = math.hypot(dx, dy)
                if norm < 1e-6:
                    continue
                dx, dy = dx / norm, dy / norm
                ex, ey = ox + dx * length, oy + dy * length
                px, py = -dy, dx
                segs = (
                    ((ox, oy), (ex, ey)),
                    ((ex, ey), (ex - head * dx + head * 0.5 * px,
                                ey - head * dy + head * 0.5 * py)),
                    ((ex, ey), (ex - head * dx - head * 0.5 * px,
                                ey - head * dy - head * 0.5 * py)),
                )
                for pen in (QPen(QColor(255, 255, 255, 210), 4.5,
                                 Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap),
                            QPen(QColor.fromRgbF(*rgb), 2.2,
                                 Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)):
                    painter.setPen(pen)
                    for a, b in segs:
                        painter.drawLine(QPointF(*a), QPointF(*b))
        finally:
            painter.end()

    def _fit_preview(self):
        if getattr(self, "_preview_pix", None) is None:
            return
        self.lbl_img.setPixmap(self._preview_pix.scaled(
            self.lbl_img.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_preview()

    def _release_preview(self):
        self._preview_timer.stop()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def closeEvent(self, ev):
        if self.isVisible() and not self._confirm_discard("关闭"):
            ev.ignore()
            return
        self._release_preview()
        ev.accept()

    def reject(self):
        if self.isVisible() and not self._confirm_discard("关闭"):
            return
        self._release_preview()
        super().reject()


def df_count(robot: RobotDoc) -> str:
    """自由度描述（转动/移动各几个），供 DH 信息面板显示。"""
    if not robot.dh:
        n = sum(1 for lk in robot.links if lk.joint.type in
                ("revolute", "hinge", "prismatic", "slide"))
        return str(n)
    rev = sum(1 for r in robot.dh if r.joint_type in
              ("revolute", "hinge"))
    pris = sum(1 for r in robot.dh if r.joint_type in
               ("prismatic", "slide"))
    parts = []
    if rev:
        parts.append(f"{rev} 转动")
    if pris:
        parts.append(f"{pris} 移动")
    return f"{rev + pris}（{' + '.join(parts)}）" if parts else "0"
