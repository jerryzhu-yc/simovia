"""「造机器人」面板：连杆树编辑 + 实时预览 + 旋转矩阵教学显示。

左侧连杆树（父子关系），中间编辑选中连杆的位姿/几何/关节，右侧预览，
并在下方显示该连杆相对父系与世界系的 3×3 旋转矩阵——初学者可以直接看到
改一个 rpy 角度时矩阵怎么变、子连杆怎么跟着走。

数据格式见 simovia/robot_spec.py。保存产物是机器人 YAML，可直接写进
世界文件的 robots[].model。
"""
from __future__ import annotations

import copy
import pathlib

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from simovia.errors import WorldError
from simovia.robot_spec import (
    GEOM_TYPES, JOINT_TYPES, RobotDoc, build_robot_spec, dump_robot,
    euler_to_matrix, link_frames, load_robot, matrix_to_quat, parse_robot,
)

#: 打开面板时的默认示例：两连杆平面臂（初学者从这里改最直观）
DEFAULT_ROBOT = {
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


#: 关节类型下拉顺序（与 cb_joint.addItems 一一对应）
_JOINT_LABELS = ["fixed", "revolute", "prismatic", "ball", "free"]


class RobotBuilderDialog(QDialog):
    """非模态：可一边开着面板改参数，一边回主窗口看场景。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("造机器人 —— 连杆树 / 位姿 / 旋转矩阵")
        self.resize(1180, 760)
        self.robot: RobotDoc = parse_robot(copy.deepcopy(DEFAULT_ROBOT))
        self._loading = False
        self._dirty = False          # 文档有未保存修改（新建/打开/关闭前提示）
        self._doc_path = None        # 最近打开/保存的机器人 YAML
        self._renderer = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._build_ui()
        self._refresh_tree()
        self._select_link("link1")
        self._refresh_preview()

    # ------------------------------------------------------------ UI 骨架

    def _build_ui(self):
        root = QHBoxLayout(self)

        # 左：连杆树
        left = QVBoxLayout()
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
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(300)
        root.addWidget(left_w)

        # 中：属性表单（可滚动）
        box = QGroupBox("选中连杆")
        form = QFormLayout(box)
        self.ed_name = QLineEdit()
        self.ed_name.editingFinished.connect(self._on_name_changed)
        form.addRow("名称", self.ed_name)
        self.cb_parent = QComboBox()
        self.cb_parent.currentIndexChanged.connect(self._on_form)
        form.addRow("父连杆", self.cb_parent)
        self.sp_pos = [self._spin(-10, 10, 0.01, suffix=" m")
                       for _ in range(3)]
        form.addRow("平移 pos x/y/z", self._row(self.sp_pos))
        self.sp_rpy = [self._spin(-360, 360, 1, 1)
                       for _ in range(3)]
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
        # ball 必须在列表里：否则打开含球关节的机器人时下拉回落到 fixed，
        # 下次任何表单改动都会把关节类型静默改掉（F04）
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
        mid_w = QWidget()
        mid = QVBoxLayout(mid_w)
        mid.addWidget(box)
        mid.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(mid_w)
        scroll.setFixedWidth(330)
        root.addWidget(scroll)

        # 右：预览 + 旋转矩阵
        right = QVBoxLayout()
        self.lbl_img = QLabel("预览")
        self.lbl_img.setMinimumSize(380, 380)
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet(
            "background:#111827; color:#9CA3AF; border-radius:6px;")
        right.addWidget(self.lbl_img)
        self.lbl_err = QLabel("")
        self.lbl_err.setWordWrap(True)
        self.lbl_err.setStyleSheet("color:#DC2626;")
        right.addWidget(self.lbl_err)
        gb = QGroupBox("旋转矩阵（当前连杆）")
        gb_lay = QVBoxLayout(gb)
        self.lbl_formula = QLabel("")
        # 深色背景上用浅灰：原来的 #374151 几乎看不见（F11）
        self.lbl_formula.setStyleSheet("font-family:monospace; color:#9CA3AF;")
        gb_lay.addWidget(self.lbl_formula)
        # 3 列各占 68px：两表并排不再需要横向滚动（F11）
        self.tb_rel = QTableWidget(3, 3)
        self.tb_world = QTableWidget(3, 3)
        for tb in (self.tb_rel, self.tb_world):
            tb.horizontalHeader().setVisible(False)
            tb.verticalHeader().setVisible(False)
            tb.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            tb.setFixedHeight(96)
            tb.setFixedWidth(212)
        row2 = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(QLabel("相对父连杆  R_rel"))
        col1.addWidget(self.tb_rel)
        col2 = QVBoxLayout()
        col2.addWidget(QLabel("世界系  R_world"))
        col2.addWidget(self.tb_world)
        row2.addLayout(col1)
        row2.addLayout(col2)
        gb_lay.addLayout(row2)
        hint = QLabel("R 的第 j 列 = 本连杆第 j 根坐标轴在参考系中的方向；"
                      "对角元 = 该轴与参考轴的夹角余弦")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6B7280; font-size:11px;")
        gb_lay.addWidget(hint)
        self.lbl_pose = QLabel("")
        self.lbl_pose.setStyleSheet("font-family:monospace;")
        gb_lay.addWidget(self.lbl_pose)
        right.addWidget(gb)

        # 底部按钮
        btns = QHBoxLayout()
        b_new = QPushButton("新建")
        b_new.clicked.connect(self._new_robot)
        b_open = QPushButton("打开…")
        b_open.clicked.connect(self._open)
        b_save = QPushButton("保存为机器人 YAML…")
        b_save.setObjectName("primary")
        b_save.clicked.connect(self._save)
        b_close = QPushButton("关闭")
        b_close.clicked.connect(self.close)
        btns.addWidget(b_new)
        btns.addWidget(b_open)
        btns.addStretch(1)
        btns.addWidget(b_save)
        btns.addWidget(b_close)
        right.addLayout(btns)
        root.addLayout(right, 1)

    @staticmethod
    def _spin(lo, hi, step, decimals=3, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setMaximumWidth(96)        # 三轴参数一行放得下，不用横向滚动（F11）
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

    # ------------------------------------------------------------ 树与选中

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
        else:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._load_form()

    def _on_tree_sel(self, *_):
        self._load_form()
        self._refresh_preview()

    # ------------------------------------------------------------ 表单读写

    def _load_form(self):
        name = self._current_link_name()
        if not name:
            return
        self._loading = True
        lk = self.robot.link(name)
        self.ed_name.setText(lk.name)
        # 父连杆下拉：排除自己与全部后代（避免成环）
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
        g = lk.geom
        self.cb_geom.setCurrentIndex(
            0 if g is None else 1 + list(GEOM_TYPES).index(g["type"]))
        size = list(g["size"]) if g else [0.05, 0.05, 0.05]
        for w, v in zip(self.sp_gsize, (size + [0.0, 0.0, 0.0])[:3]):
            w.setValue(float(v))
        gpos = list(g.get("pos", [0, 0, 0])) if g else [0, 0, 0]
        for w, v in zip(self.sp_gpos, gpos):
            w.setValue(float(v))
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
        self._apply_enabled()
        self._on_form()

    def _on_joint_type(self, *_):
        if self._loading:
            return
        name = self._current_link_name()
        lk = self.robot.link(name) if name else None
        prev = lk.joint.type if lk is not None else "fixed"
        jt = _JOINT_LABELS[self.cb_joint.currentIndex()]
        # 转动 <-> 移动：角度范围换成米，量级明显是角度时给合理默认（F09）
        if jt == "prismatic" and prev in ("revolute", "hinge") \
                and max(abs(w.value()) for w in self.sp_range) > 5:
            self._loading = True
            try:
                for w, v in zip(self.sp_range, (-0.5, 0.5)):
                    w.setValue(v)
            finally:
                self._loading = False
        self._apply_enabled()
        self._apply_joint_units()
        self._on_form()

    def _apply_joint_units(self):
        """按关节类型切换范围/刚度单位（转动=度/N·m·rad，移动=米/N/m）。"""
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
        """结构校验（提交前用）：唯一根、parent 存在、free 只能给根连杆。"""
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
        """错误显示在面板里（不只写 tooltip/日志）。"""
        self.lbl_err.setText(msg)
        self.lbl_err.setStyleSheet("color:#DC2626;")

    def _on_form(self, *_):
        """表单 -> 数据模型 -> 树/矩阵/预览（任何失败只提示，不落盘）。"""
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
            jt = "fixed"          # free 只能给根连杆
            self.cb_joint.setCurrentIndex(0)
        lk.joint.type = jt
        lk.joint.axis = tuple(w.value() for w in self.sp_axis)
        lk.joint.range = tuple(w.value() for w in self.sp_range)
        lk.joint.kp = self.sp_kp.value()
        lk.joint.kd = self.sp_kd.value()
        if self.cb_geom.currentIndex() == 0:
            lk.geom = None
        else:
            gtype = list(GEOM_TYPES)[self.cb_geom.currentIndex() - 1]
            # 保留未在表单里暴露的字段（rpy 等）：重建会静默改变模型语义
            g = dict(lk.geom or {})
            g.update(type=gtype,
                     size=tuple(w.value() for w in self.sp_gsize),
                     pos=tuple(w.value() for w in self.sp_gpos))
            g.setdefault("color", lk.color)
            lk.geom = g
        self._update_matrix()
        self._schedule_preview()

    def _on_name_changed(self):
        if self._loading:
            return
        old = self._current_link_name()
        new = self.ed_name.text().strip()
        if not old or not new or new == old:
            return
        if any(lk.name == new for lk in self.robot.links):
            QMessageBox.warning(self, "重名", f"连杆名 '{new}' 已存在")
            self.ed_name.setText(old)
            return
        for lk in self.robot.links:
            if lk.name == old:
                lk.name = new
            if lk.parent == old:
                lk.parent = new
        self._dirty = True
        self._refresh_tree(select=new)

    # ------------------------------------------------------------ 增删/新建

    def _add_child(self):
        parent = self._current_link_name()
        if not parent:
            return
        base = f"link{len(self.robot.links)}"
        name, i = base, 1
        while any(lk.name == name for lk in self.robot.links):
            name = f"{base}_{i}"
            i += 1
        from simovia.robot_spec import JointDoc, LinkDoc
        lk = LinkDoc(name=name, parent=parent, pos=(0.0, 0.0, 0.15),
                     joint=JointDoc(type="revolute",
                                    range=(-170.0, 170.0)),
                     geom={"type": "box", "size": (0.03, 0.03, 0.10),
                           "pos": (0.0, 0.0, 0.10)})
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
        # 子连杆上移一级，避免断树
        for other in self.robot.links:
            if other.parent == name:
                other.parent = lk.parent
        self.robot.links = [x for x in self.robot.links if x.name != name]
        self._dirty = True
        self._refresh_tree(select=lk.parent)
        self._schedule_preview()

    def _confirm_discard(self, action: str) -> bool:
        """新建/打开/关闭前确认未保存的机器人文档（F02）。"""
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
        return self._save()

    def _new_robot(self):
        if not self._confirm_discard("新建机器人"):
            return
        self.robot = parse_robot(copy.deepcopy(DEFAULT_ROBOT))
        self._doc_path = None
        self._dirty = False
        self._refresh_tree(select="link1")
        self._refresh_preview()

    # ------------------------------------------------------------ 打开/保存

    def _open(self):
        if not self._confirm_discard("打开机器人"):
            return
        p, _ = QFileDialog.getOpenFileName(
            self, "打开机器人描述", str(pathlib.Path.cwd()),
            "机器人 YAML (*.yaml *.yml)")
        if not p:
            return
        try:
            self.robot = load_robot(p)
        except WorldError as e:
            QMessageBox.critical(self, "打开失败", str(e))
            return
        self._doc_path = p
        self._dirty = False
        # 旧选中名在新机器人里不存在：显式选根连杆，表单/矩阵才有内容
        self._refresh_tree(select=self.robot.links[0].name)
        self._refresh_preview()

    def _save(self, ask: bool = True) -> bool:
        """Schema + MuJoCo 编译校验通过才落盘；失败保留旧文件（F05）。

        返回是否保存成功（未保存确认流程用）。
        """
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
            # 保存前做与加载/编译同级别的校验：零尺寸、零轴、反向范围等
            # 不能算"已保存"的有效模型；失败时不覆盖磁盘上的上一个有效版本
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
            # 统一正斜杠：Windows 反斜杠路径跨平台重开易出解析问题
            f"    model: {pathlib.Path(p).as_posix()}")
        return True

    # ------------------------------------------------------------ 矩阵显示

    def _update_matrix(self):
        name = self._current_link_name()
        try:
            frames = {f.name: f for f in link_frames(self.robot)}
        except (WorldError, KeyError):
            self.tb_rel.clearContents()
            self.tb_world.clearContents()
            return
        f = frames.get(name)
        if f is None:
            return
        for tb, R in ((self.tb_rel, f.R_rel), (self.tb_world, f.R_world)):
            for i in range(3):
                for j in range(3):
                    it = QTableWidgetItem(f"{R[i, j]:+.3f}")
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    tb.setItem(i, j, it)
        r, p, y = [round(v, 1) for v in self.robot.link(name).rpy_deg]
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
        """编译预览模型：机器人 + 地面 + 灯（编译后由调用方加轴标）。"""
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

    def _refresh_preview(self):
        name = self._current_link_name()
        try:
            frames = {f.name: f for f in link_frames(self.robot)}
            zmin = min([0.0] + [float(f.p_world[2]) for f in frames.values()])
            mj = self._preview_spec(min(0.0, zmin - 0.01))
            # 轴三色标：R 的三列就是连杆坐标系 x/y/z 轴在世界系的方向
            if name and name in frames:
                f = frames[name]
                triad = mj.worldbody.add_body(
                    name="preview_triad",
                    pos=[float(v) for v in f.p_world],
                    quat=[float(v) for v in matrix_to_quat(f.R_world)])
                L = 0.09
                for axis, rgb, quat in (
                        ((1, 0, 0), [1, 0.25, 0.25, 1], [0.7071, 0, 0.7071, 0]),
                        ((0, 1, 0), [0.25, 0.85, 0.35, 1], [0.7071, -0.7071, 0, 0]),
                        ((0, 0, 1), [0.35, 0.55, 1, 1], [1, 0, 0, 0])):
                    triad.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[0.006, L / 2], quat=quat,
                        pos=[L * a / 2 for a in axis], rgba=rgb,
                        contype=0, conaffinity=0)
            model = mj.compile()
        except (WorldError, ValueError, TypeError) as e:
            self.lbl_err.setText(str(e))
            return
        self.lbl_err.setText("")

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Renderer 与 model 绑定，每次重编译都要换一个新的（预览低频，
        # 每次开关上下文的开销可以接受）
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(model, 420, 420)
        cam = mujoco.MjvCamera()
        if frames:
            # 包围盒要含几何尺寸：只按连杆原点取景会把机器人顶部裁掉（F11）
            pts = []
            for f in frames.values():
                pts.append(np.asarray(f.p_world, dtype=float))
                lk = self.robot.link(f.name)
                g = lk.geom
                if not g:
                    continue
                size = [float(v) for v in g["size"]] + [0.0, 0.0]
                local = [float(v) for v in g.get("pos", (0.0, 0.0, 0.0))]
                center = (np.asarray(f.p_world, dtype=float)
                          + np.asarray(f.R_world, dtype=float)
                          @ np.asarray(local, dtype=float))
                # 外接球半径：sphere 取半径，其余取半尺寸最大分量
                r = size[0] if g["type"] == "sphere" else max(size[:3])
                pts.append(center + r)
                pts.append(center - r)
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
        self.lbl_img.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, ev):
        # 只在用户可见时确认（程序性 close/测试里没有展示过就不弹模态框）
        if self.isVisible() and not self._confirm_discard("关闭"):
            ev.ignore()
            return
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().closeEvent(ev)
