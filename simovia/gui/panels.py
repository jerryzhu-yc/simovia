"""Webots 风格的停靠面板集合。

- SceneTreePanel  场景树（世界 / 机器人 / 物体层级，选中节点 -> 属性面板）
- PropertiesPanel 属性（选中节点详情 + 任务统计常驻底部）
- ObsPanel        观测树（保留 M1 接口，测试依赖；标签页方式呈现）
- ConsolePanel    控制台（GUI 事件 + 后端 stderr）
- WorldPanel      世界概要（兼容旧接口）
- TaskPanel       任务状态大字（兼容旧接口）
"""
from __future__ import annotations

import datetime as _dt
import math
import pathlib

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (QColor, QFont, QIcon, QKeySequence, QPixmap,
                           QShortcut, QSyntaxHighlighter, QTextCharFormat,
                           QTextCursor)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QStackedWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from simovia.gui import theme

# 场景树 ItemDataRole：存 ("robot"|"object"|"world", 标识名)
_NODE_KIND = Qt.ItemDataRole.UserRole + 1


def _fmt_vec(v, nd=3) -> str:
    try:
        return "(" + ", ".join(f"{float(x):+.{nd}f}" for x in v) + ")"
    except (TypeError, ValueError):
        return str(v)


def quat_to_euler(q) -> tuple[float, float, float]:
    """wxyz 四元数 -> (roll, pitch, yaw) 度（ZYX 内旋序）。属性面板用。

    pitch=±90° 是万向锁：roll/yaw 不唯一，此处分母 1-2(x²+y²) 会落到
    ±1e-17，atan2 在不同平台/浮点上返回 0 或 π——同一姿态在面板里会
    跳成 (0,-90,0) 或 (180,-90,180) 两套角。这里固定取 roll=0，yaw 由
    x（pitch=-90）或 z（pitch=+90）与 w 合成，结果确定且与输入互逆。
    """
    w, x, y, z = (float(v) for v in q)
    sinp = 2 * (w * y - z * x)
    if sinp >= 1.0 - 1e-12:
        return (0.0, 90.0, math.degrees(2 * math.atan2(z, w)))
    if sinp <= -1.0 + 1e-12:
        return (0.0, -90.0, math.degrees(2 * math.atan2(x, w)))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    """(roll, pitch, yaw) 度 -> wxyz 四元数（与 quat_to_euler 互逆）。"""
    cr, sr = math.cos(math.radians(roll) / 2), math.sin(math.radians(roll) / 2)
    cp, sp = math.cos(math.radians(pitch) / 2), math.sin(math.radians(pitch) / 2)
    cy, sy = math.cos(math.radians(yaw) / 2), math.sin(math.radians(yaw) / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


# ================================================================ 场景树

class SceneTreePanel(QWidget):
    """左侧场景树：加载世界后展示 世界/机器人(关节)/物体 层级。

    nodeSelected(kind:str, name:str, payload)：选中节点时发出，供属性面板显示。
    nodeDeleteRequested(kind, name)：右键"删除"时发出（app 侧提交 YAML）。
    """

    nodeSelected = Signal(str, str, object)
    nodeDeleteRequested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        bar = QHBoxLayout()
        self._title = QLabel("场景结构")
        self._title.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-weight:600;")
        bar.addWidget(self._title)
        bar.addStretch(1)
        lay.addLayout(bar)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["名称", "类型"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.header().setSectionResizeMode(
            0, self._tree.header().ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(
            1, self._tree.header().ResizeMode.ResizeToContents)
        lay.addWidget(self._tree, stretch=1)
        self._tree.itemSelectionChanged.connect(self._on_select)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

        self._spec = None
        self.set_world(None)

    def set_world(self, spec) -> None:
        self._spec = spec
        self._tree.clear()
        if spec is None:
            empty = QTreeWidgetItem(["（未加载世界）", ""])
            empty.setDisabled(True)
            self._tree.addTopLevelItem(empty)
            return

        root = QTreeWidgetItem([spec.name, "世界"])
        root.setData(0, _NODE_KIND, ("world", spec.name, spec))
        self._tree.addTopLevelItem(root)

        rob_root = QTreeWidgetItem(["机器人", f"{len(spec.robots)} 个"])
        root.addChild(rob_root)
        for rd in spec.robots:
            # 正/反斜杠都切：Windows 用户写的路径是 C:\robots\x.xml
            model_name = (rd.model or "").replace("\\", "/").split("/")[-1]
            rnode = QTreeWidgetItem([rd.name, model_name or "连杆树"])
            rnode.setData(0, _NODE_KIND, ("robot", rd.name, rd))
            rob_root.addChild(rnode)
            for six in rd.sensors:
                sn = QTreeWidgetItem([six.name, six.type])
                sn.setData(0, _NODE_KIND, ("sensor", f"{rd.name}/{six.name}", six))
                rnode.addChild(sn)

        obj_root = QTreeWidgetItem(["物体", f"{len(spec.scene.objects)} 个"])
        root.addChild(obj_root)
        for od in spec.scene.objects:
            onode = QTreeWidgetItem([od.name,
                                     od.type + ("（固定）" if od.static else "")])
            onode.setData(0, _NODE_KIND, ("object", od.name, od))
            obj_root.addChild(onode)

        self._tree.expandAll()
        self._tree.setCurrentItem(root)

    def select_node(self, kind: str, name: str) -> bool:
        """3D 视口点击联动：程序选中对应节点（触发 nodeSelected 联动属性面板）。"""
        def walk(item) -> bool:
            for i in range(item.childCount()):
                ch = item.child(i)
                d = ch.data(0, _NODE_KIND)
                if d is not None and d[0] == kind and d[1] == name:
                    self._tree.scrollToItem(ch)
                    self._tree.setCurrentItem(ch)
                    return True
                if walk(ch):
                    return True
            return False

        for i in range(self._tree.topLevelItemCount()):
            if walk(self._tree.topLevelItem(i)):
                return True
        return False

    def _on_select(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, _NODE_KIND)
        if data is None:
            return
        kind, name, payload = data
        self.nodeSelected.emit(kind, name, payload)

    def _on_context_menu(self, pos) -> None:
        """右键菜单：机器人/物体节点 -> 删除（发 nodeDeleteRequested）。"""
        item = self._tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, _NODE_KIND)
        if data is None or data[0] not in ("robot", "object"):
            return
        kind, name = data[0], data[1]
        menu = QMenu(self)
        act = menu.addAction(f"删除 {name}（从世界移除）")
        if menu.exec(self._tree.viewport().mapToGlobal(pos)) is act:
            self.nodeDeleteRequested.emit(kind, name)


# ================================================================ 属性面板

class PropertiesPanel(QWidget):
    """右侧属性：上半部分显示选中节点属性，下半部分常驻任务统计。"""

    poseApplied = Signal(str, str, list, object)   # kind, 名字, pos, quat

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        sel_box = QGroupBox("属性")
        self._sel_form = QFormLayout(sel_box)
        self._sel_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._sel_name = QLabel("—")
        self._sel_name.setStyleSheet("font-weight:600;")
        self._sel_form.addRow("选中", self._sel_name)
        self._sel_rows: list[tuple[QLabel, QLabel]] = []
        outer.addWidget(sel_box, stretch=1)

        # 位姿编辑（机器人/物体节点）：xyz + 欧拉角（度）-> 应用写回世界 YAML。
        # quat 单位朝向时不写（保持 YAML 干净，None 语义）
        self._pose_box = QGroupBox("位姿")
        pf = QFormLayout(self._pose_box)
        self._pos_spins = []
        for lb, lo, hi in (("x", -20.0, 20.0), ("y", -20.0, 20.0),
                           ("z", -5.0, 10.0)):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setDecimals(3); s.setSingleStep(0.01)
            s.setToolTip(f"世界坐标 {lb}（米）")
            self._pos_spins.append(s)
        row_p = QHBoxLayout()
        for s in self._pos_spins:
            row_p.addWidget(s)
        pf.addRow("位置 (m)", row_p)
        self._rot_spins = []
        for lb in ("roll", "pitch", "yaw"):
            s = QDoubleSpinBox()
            s.setRange(-360.0, 360.0); s.setDecimals(1); s.setSingleStep(5.0)
            s.setToolTip(f"{lb}（度，ZYX 序）")
            self._rot_spins.append(s)
        row_r = QHBoxLayout()
        for s in self._rot_spins:
            row_r.addWidget(s)
        pf.addRow("旋转 (°)", row_r)
        btn = QPushButton("应用位姿")
        btn.setToolTip("写入世界 YAML 并重载（World as Code：改文件即改世界）")
        btn.clicked.connect(self._on_apply_pose)
        pf.addRow("", btn)
        self._pose_box.setVisible(False)
        self._pose_kind: str | None = None
        self._pose_name: str | None = None
        outer.addWidget(self._pose_box)

        # 任务统计（常驻，Webots 风格的 HUD 数据栏）
        task_box = QGroupBox("任务状态")
        tl = QFormLayout(task_box)
        self._tick = QLabel("-")
        self._time = QLabel("-")
        self._success = QLabel("-")
        self._reward = QLabel("-")
        for lb in (self._tick, self._time, self._success, self._reward):
            lb.setStyleSheet("font-size:13pt; font-weight:600;")
        self._success.setStyleSheet(
            f"font-size:13pt; font-weight:600; color:{theme.C_TEXT_DIM};")
        tl.addRow("tick", self._tick)
        tl.addRow("仿真时间 (s)", self._time)
        tl.addRow("成功", self._success)
        tl.addRow("reward", self._reward)
        outer.addWidget(task_box)

    def _clear_rows(self) -> None:
        for k, v in self._sel_rows:
            self._sel_form.removeRow(k)
        self._sel_rows.clear()

    def _add_row(self, key: str, value: str) -> None:
        k = QLabel(key)
        v = QLabel(str(value))
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._sel_form.addRow(k, v)
        self._sel_rows.append((k, v))

    # ------------------------------------------------------------ 节点选择

    def show_node(self, kind: str, name: str, payload) -> None:
        self._clear_rows()
        self._sel_name.setText(name)
        is_robot = kind == "robot"
        has_pose = kind in ("robot", "object")   # 机器人与物体都可编辑位姿
        self._pose_box.setVisible(has_pose)
        if kind == "world":
            self._add_row("timestep", f"{payload.timestep:g} s")
            self._add_row("ctrl_period", f"{payload.ctrl_period} 步")
            self._add_row("重力", _fmt_vec(payload.gravity))
            floor = payload.scene.floor
            self._add_row("地面尺寸",
                          f"{floor.size[0]:g} × {floor.size[1]:g} m")
            self._add_row("max_ticks",
                          payload.task.max_ticks if payload.task.max_ticks
                          else "不限")
        elif is_robot:
            self._pose_kind, self._pose_name = "robot", name
            self._add_row("模型", payload.model)
            self._add_row("初始位置", _fmt_vec(payload.pos))
            self._add_row("控制器", payload.controller or "（无）")
            self._add_row("传感器数", len(payload.sensors))
            if payload.args:
                self._add_row("args", ", ".join(f"{k}={v}"
                                                for k, v in payload.args.items()))
            # 选中即回填当前值（拖动后 _on_loaded 重选也会走到这里同步）
            self._fill_pose(payload.pos, payload.quat)
        elif kind == "object":
            self._pose_kind, self._pose_name = "object", name
            self._add_row("类型", payload.type)
            self._add_row("尺寸(半)", _fmt_vec(payload.size))
            self._add_row("位置", _fmt_vec(payload.pos))
            self._add_row("固定", "是" if payload.static else "否")
            self._add_row("质量", payload.mass if payload.mass is not None else "自动")
            self._add_row("颜色", payload.color)
            self._fill_pose(payload.pos, payload.quat)
        elif kind == "sensor":
            self._add_row("类型", payload.type)
            if payload.joint:
                self._add_row("挂接关节", payload.joint)
            if payload.body:
                self._add_row("挂接 body", payload.body)
            if payload.radius is not None:
                self._add_row("半径", f"{payload.radius:g} m")

    def clear_selection(self) -> None:
        self._clear_rows()
        self._sel_name.setText("—")
        self._pose_box.setVisible(False)
        self._pose_kind = None
        self._pose_name = None

    def _fill_pose(self, pos, quat) -> None:
        """位姿 spin 回填（机器人/物体共用）。"""
        for s, v in zip(self._pos_spins, pos):
            s.setValue(float(v))
        eul = quat_to_euler(quat or (1.0, 0.0, 0.0, 0.0))
        for s, v in zip(self._rot_spins, eul):
            s.setValue(v)

    def _on_apply_pose(self) -> None:
        """收集 spin 值 -> poseApplied 信号（app 侧 read_world 写 YAML 重载）。"""
        if not self._pose_name:
            return
        pos = [round(s.value(), 4) for s in self._pos_spins]
        quat = euler_to_quat(*(s.value() for s in self._rot_spins))
        quat = [round(v, 6) for v in quat]
        ident = (1.0, 0.0, 0.0, 0.0)
        quat = None if all(abs(v - e) < 1e-6
                           for v, e in zip(quat, ident)) else quat
        self.poseApplied.emit(self._pose_kind, self._pose_name, pos, quat)

    # ------------------------------------------------------------ 任务数据

    def update_live_pose(self, kind: str, name: str, pos, quat) -> None:
        """运行中快照回写选中机器人/物体的实时位姿到 spin（不重建行）。

        只在当前选中节点匹配时更新；用户正在编辑（spin 有焦点）时不打扰。
        """
        if self._pose_kind != kind or self._pose_name != name:
            return
        w = QApplication.focusWidget()
        if isinstance(w, QDoubleSpinBox):
            return
        self._fill_pose(pos, quat)

    def update_obs(self, obs: dict) -> None:
        task = obs.get("task") or {}
        # 选中机器人/物体的实时位姿（机器人运动时右侧坐标同步跳数）
        if self._pose_kind and self._pose_name:
            if self._pose_kind == "robot":
                robs = (obs.get("robots") or {}).get(self._pose_name) or {}
                root = robs.get("root")
                if root is not None:
                    self.update_live_pose("robot", self._pose_name,
                                          root[0:3], root[3:7])
            else:
                oinfo = (obs.get("objects") or {}).get(self._pose_name)
                if oinfo is not None:
                    p = oinfo.get("pos")
                    if p is not None:
                        self.update_live_pose(
                            "object", self._pose_name, p,
                            oinfo.get("quat") or (1, 0, 0, 0))
        self._tick.setText(str(obs.get("tick", "-")))
        self._time.setText(str(obs.get("time", "-")))
        succ = task.get("success")
        if succ is None:
            self._success.setText("-")
            self._success.setStyleSheet(
                f"font-size:13pt; font-weight:600; color:{theme.C_TEXT_DIM};")
        else:
            self._success.setText("成功" if succ else "未成功")
            self._success.setStyleSheet(
                f"font-size:13pt; font-weight:600; color:"
                f"{theme.C_OK if succ else theme.C_ERR};")
        self._reward.setText(str(task.get("reward", "-")))

    def update_stats(self, stats: dict) -> None:
        self._tick.setText(f"{stats.get('ticks', '-')}  (结束)")
        succ = stats.get("success")
        if succ is None:
            self._success.setText("-")
        else:
            self._success.setText("成功" if succ else "未成功")
            self._success.setStyleSheet(
                f"font-size:13pt; font-weight:600; color:"
                f"{theme.C_OK if succ else theme.C_ERR};")
        self._reward.setText(str(stats.get("reward", "-")))

    def reset_task(self) -> None:
        for lb in (self._tick, self._time, self._reward):
            lb.setText("-")
        self._success.setText("-")
        self._success.setStyleSheet(
            f"font-size:13pt; font-weight:600; color:{theme.C_TEXT_DIM};")


# ================================================================ 观测树（保留 M1 接口）

class ObsPanel(QWidget):
    """观测树：逐机器人关节 qpos/ctrl + 传感器读数 + 物体 pos/quat。

    保留 self._tree 与 update_obs/clear 接口（tests/test_gui.py 依赖）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        tip = QLabel("关节角 q / 执行器输出 u（按名字配对，含单位与控制模式）")
        tip.setStyleSheet(f"color:{theme.C_TEXT_DIM};")
        lay.addWidget(tip)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["名称", "值"])
        self._tree.setAlternatingRowColors(True)
        self._tree.header().setSectionResizeMode(
            0, self._tree.header().ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(
            1, self._tree.header().ResizeMode.Stretch)
        lay.addWidget(self._tree)

    def update_obs(self, obs: dict) -> None:
        # 记录当前展开状态，刷新后保持（避免每帧折叠抖动）
        expanded = {self._tree.topLevelItem(i).text(0)
                    for i in range(self._tree.topLevelItemCount())
                    if self._tree.topLevelItem(i).isExpanded()}

        self._tree.clear()
        for rname, robs in (obs.get("robots") or {}).items():
            top = QTreeWidgetItem([rname])
            if robs.get("joints"):
                self._build_named(top, robs["joints"], robs.get("actuators") or {})
            else:                                  # 旧 payload：按下标展示
                names = robs.get("joint_names") or []
                qpos = robs.get("qpos") or []
                ctrl = robs.get("ctrl") or []
                for i, jn in enumerate(names):
                    v = f"q={qpos[i]:+.3f}" if i < len(qpos) else ""
                    if i < len(ctrl):
                        v += f"   u={ctrl[i]:+.3f}"
                    QTreeWidgetItem(top, [jn, v])
            sensors = robs.get("sensors") or {}
            if sensors:                        # M2：有传感器才有该键
                st = QTreeWidgetItem(top, ["sensors"])
                for sn, sv in sensors.items():
                    if isinstance(sv, list):
                        sv = "(" + ", ".join(f"{x:+.3f}" for x in sv) + ")"
                    else:
                        sv = f"{sv:+.3f}"
                    QTreeWidgetItem(st, [sn, sv])
            self._tree.addTopLevelItem(top)
            top.setExpanded(rname in expanded or not expanded)
        for oname, oinfo in (obs.get("objects") or {}).items():
            pos = oinfo.get("pos") or []
            quat = oinfo.get("quat") or []
            top = QTreeWidgetItem([
                oname,
                f"pos=({', '.join(f'{x:+.2f}' for x in pos)}) "
                f"quat=({', '.join(f'{x:+.2f}' for x in quat)})"])
            self._tree.addTopLevelItem(top)

    @staticmethod
    def _build_named(top, joints: dict, actuators: dict) -> None:
        """按名字配对关节与执行器（不依赖 qpos/ctrl 的下标顺序）。"""
        by_joint: dict[str, dict] = {}
        for aname, a in actuators.items():
            by_joint.setdefault(a.get("joint") or "", {})[aname] = a
        used = set()
        for jname, j in joints.items():
            q = j.get("q") or []
            val = "q=" + (f"{q[0]:+.3f}" if len(q) == 1
                          else "(" + ", ".join(f"{x:+.3f}" for x in q) + ")")
            if j.get("unit"):
                val += f" {j['unit']}"
            if j.get("type"):
                val += f"  [{j['type']}]"
            node = QTreeWidgetItem(top, [jname, val])
            for aname, a in by_joint.get(jname, {}).items():
                QTreeWidgetItem(node, [f"u {aname}", ObsPanel._fmt_actuator(a)])
                used.add(aname)
        rest = {n: a for n, a in actuators.items() if n not in used}
        if rest:                                  # 未绑定到具名关节的执行器
            node = QTreeWidgetItem(top, ["执行器（未绑定关节）"])
            for aname, a in rest.items():
                QTreeWidgetItem(node, [aname, ObsPanel._fmt_actuator(a)])

    @staticmethod
    def _fmt_actuator(a: dict) -> str:
        u = a.get("u")
        text = f"u={u:+.3f}" if isinstance(u, (int, float)) else "u=?"
        if a.get("mode"):
            text += f"  [{a['mode']}]"
        r = a.get("ctrlrange")
        if r:
            text += f"  限位=[{r[0]:+.3f}, {r[1]:+.3f}]"
        return text

    def clear(self) -> None:
        self._tree.clear()


# ================================================================ 控制台

class ConsolePanel(QWidget):
    """控制台：带级别着色 + 时间戳 + 清空按钮。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        hint = QLabel("控制台")
        hint.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-weight:600;")
        bar.addWidget(hint)
        bar.addStretch(1)
        btn_clear = QPushButton("清空")
        btn_clear.setFlat(True)
        btn_clear.clicked.connect(self.clear)
        bar.addWidget(btn_clear)
        lay.addLayout(bar)

        self._edit = QPlainTextEdit()
        self._edit.setObjectName("console")
        self._edit.setReadOnly(True)
        self._edit.setMaximumBlockCount(5000)
        lay.addWidget(self._edit)

    def _write(self, text: str, color: str, source: str = "") -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        cursor = self._edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        prefix_fmt = QTextCharFormat()
        prefix_fmt.setForeground(QColor(theme.C_TEXT_DIM))
        cursor.insertText(f"[{ts}] ", prefix_fmt)
        if source:
            src_fmt = QTextCharFormat()
            src_fmt.setForeground(QColor(theme.C_ACCENT))
            cursor.insertText(f"{source} ", src_fmt)
        body_fmt = QTextCharFormat()
        body_fmt.setForeground(QColor(color))
        cursor.insertText(text + "\n", body_fmt)
        self._edit.setTextCursor(cursor)

    def info(self, text: str, source: str = "") -> None:
        self._write(text, theme.C_TEXT, source)

    def ok(self, text: str) -> None:
        self._write(text, theme.C_OK)

    def warn(self, text: str) -> None:
        self._write(text, theme.C_WARN)

    def error(self, text: str) -> None:
        self._write(text, theme.C_ERR)

    def backend(self, text: str) -> None:
        self._write(text, theme.C_TEXT_DIM, source="后端")

    def clear(self) -> None:
        self._edit.clear()


# ================================================================ 兼容旧接口

class WorldPanel(QGroupBox):
    """世界概要（旧接口，新布局里不再使用，保留以免外部引用失效）。"""

    def __init__(self, parent=None):
        super().__init__("世界", parent)
        self._name = QLabel("（未加载）")
        self._objects = QLabel("-")
        self._robots = QLabel("-")
        self._max_ticks = QLabel("-")
        form = QFormLayout(self)
        form.addRow("名称", self._name)
        form.addRow("物体", self._objects)
        form.addRow("机器人", self._robots)
        form.addRow("max_ticks", self._max_ticks)

    def update_world(self, spec) -> None:
        self._name.setText(spec.name)
        self._objects.setText(", ".join(o.name for o in spec.scene.objects) or "-")
        self._robots.setText(", ".join(r.name for r in spec.robots) or "-")
        self._max_ticks.setText(str(spec.task.max_ticks))


class TaskPanel(QGroupBox):
    """任务状态（旧接口，新布局用 PropertiesPanel；保留兼容）。"""

    def __init__(self, parent=None):
        super().__init__("任务", parent)
        self._tick = QLabel("-")
        self._time = QLabel("-")
        self._success = QLabel("-")
        self._reward = QLabel("-")
        for lb in (self._tick, self._time, self._success, self._reward):
            lb.setStyleSheet("font-size:14pt; font-weight:600;")
        form = QFormLayout(self)
        form.addRow("tick", self._tick)
        form.addRow("仿真时间(s)", self._time)
        form.addRow("成功", self._success)
        form.addRow("reward", self._reward)

    def update_obs(self, obs: dict) -> None:
        task = obs.get("task") or {}
        self._tick.setText(str(obs.get("tick", "-")))
        self._time.setText(str(obs.get("time", "-")))
        succ = task.get("success")
        self._success.setText("-" if succ is None else ("是" if succ else "否"))
        self._reward.setText(str(task.get("reward", "-")))

    def update_stats(self, stats: dict) -> None:
        self._tick.setText(f"{stats.get('ticks', '-')} (结束)")
        succ = stats.get("success")
        self._success.setText("-" if succ is None else ("是" if succ else "否"))
        self._reward.setText(str(stats.get("reward", "-")))

    def clear(self) -> None:
        for lb in (self._tick, self._time, self._success, self._reward):
            lb.setText("-")


# ================================================================ 代码编辑器

_PY_KW = frozenset((
    "and as assert async await break class continue def del elif else except "
    "finally for from global if import in is lambda nonlocal not or pass raise "
    "return try while with yield True False None").split())


class PyHighlighter(QSyntaxHighlighter):
    """Python 语法高亮：字符串/注释/关键字/内置常量/类名（零依赖近似版）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fmt_str = QTextCharFormat()
        self._fmt_str.setForeground(QColor("#7EE787"))
        self._fmt_com = QTextCharFormat()
        self._fmt_com.setForeground(QColor("#8B949E"))
        self._fmt_com.setFontItalic(True)
        self._fmt_kw = QTextCharFormat()
        self._fmt_kw.setForeground(QColor("#FF7B72"))
        self._fmt_kw.setFontWeight(QFont.Weight.Bold)
        self._fmt_num = QTextCharFormat()
        self._fmt_num.setForeground(QColor("#79C0FF"))

    def highlightBlock(self, text: str) -> None:
        import re as _re
        for m in _re.finditer(r'"[^"\n]*"|\'[^\'\n]*\'', text):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_str)
        for m in _re.finditer(r'#[^\n]*', text):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_com)
        for m in _re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d[\w.]*\b', text):
            w = m.group()
            if w in _PY_KW:
                self.setFormat(m.start(), len(w), self._fmt_kw)
            elif w[0].isdigit():
                self.setFormat(m.start(), len(w), self._fmt_num)


class CodePanel(QWidget):
    """代码页：控制律即改即跑（Webots 式内置编辑器）。

    set_world 填充控制器下拉；saveRequested(name, text) 由 app 侧落盘 +
    重载世界（重启控制器进程）+ 自动运行。Ctrl+S 在编辑框聚焦时等同保存。

    未保存内容保护：每个控制器文件各有独立草稿（切换下拉/重载世界都不
    丢改动），标题栏按钮显示 "保存并运行 *"；"重读"需确认后才丢弃草稿。
    """

    saveRequested = Signal(str, str)          # robot 名, 文件全文

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setToolTip("选择要编辑的机器人控制器")
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        bar.addWidget(self.combo, stretch=1)
        btn_reload = QPushButton("重读")
        btn_reload.setToolTip("放弃修改，从磁盘重读当前控制器")
        btn_reload.clicked.connect(self._reload_from_disk)
        bar.addWidget(btn_reload)
        self.btn_save = QPushButton("保存并运行")
        self.btn_save.setToolTip("写回控制器文件 -> 重载世界（控制器进程重启，"
                                 "新代码生效）-> 自动运行")
        self.btn_save.clicked.connect(self._on_save)
        bar.addWidget(self.btn_save)
        v.addLayout(bar)
        self.edit = QPlainTextEdit()
        mono = QFont()
        mono.setFamilies(["JetBrains Mono", "Cascadia Mono", "Consolas",
                          "DejaVu Sans Mono"])
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.edit.setFont(mono)
        self.edit.setPlaceholderText("加载世界后自动载入控制器代码…")
        self._hl = PyHighlighter(self.edit.document())
        # 编辑框聚焦时 Ctrl+S = 保存并运行（优先于全局截图快捷键）
        sc = QShortcut(QKeySequence.StandardKey.Save, self.edit)
        sc.activated.connect(self._on_save)
        v.addWidget(self.edit, stretch=1)
        self._paths: dict[str, str] = {}       # robot 名 -> 控制器绝对路径
        self._drafts: dict[str, str] = {}      # 文件路径 -> 未保存内容
        self._current: str | None = None       # 当前编辑的文件绝对路径
        self.edit.document().modificationChanged.connect(self._on_modified)

    # ------------------------------------------------------------ 未保存保护

    def _on_modified(self, modified: bool) -> None:
        self.btn_save.setText("保存并运行 *" if modified else "保存并运行")

    def _stash_current(self) -> None:
        """把当前编辑框里未保存的改动收进草稿（切文件/换世界前调用）。"""
        if self._current and self.edit.document().isModified():
            self._drafts[self._current] = self.edit.toPlainText()

    def _show(self, text: str, path: str | None, modified: bool = False) -> None:
        self._current = path
        self.edit.setPlainText(text)           # 清空 undo 栈
        self.edit.document().setModified(modified)
        self._on_modified(modified)

    def dirty_files(self) -> dict[str, str]:
        """所有未保存内容（路径 -> 文本）；关闭窗口提示用。"""
        files = dict(self._drafts)
        if self._current and self.edit.document().isModified():
            files[self._current] = self.edit.toPlainText()
        return files

    def save_dirty_files(self) -> list[str]:
        """把未保存内容直接写盘（不重载世界）；返回失败描述列表。"""
        errors = []
        files = self.dirty_files()
        for path, text in files.items():
            try:
                pathlib.Path(path).write_text(text, encoding="utf-8")
            except OSError as e:
                errors.append(f"{path}: {e}")
                continue
            self._drafts.pop(path, None)
            if path == self._current:
                self.edit.document().setModified(False)
        return errors

    def clear_draft(self, robot_name: str) -> None:
        """保存并运行成功写盘后清掉该机器人的草稿。"""
        path = self._paths.get(robot_name)
        if not path:
            return
        self._drafts.pop(path, None)
        if path == self._current:
            self.edit.document().setModified(False)

    # ------------------------------------------------------------ 载入

    def set_world(self, spec, base_dir) -> None:
        """加载/重载世界后刷新控制器清单（相对路径按世界目录解析）。"""
        self._stash_current()
        self._paths = {r.name: str((pathlib.Path(base_dir) / r.controller).resolve())
                       for r in spec.robots if r.controller}
        prev = self._current
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(self._paths) or ["（当前世界没有控制器）"])
        self.combo.blockSignals(False)
        if not self._paths:
            self._show("# 当前世界的机器人没有绑定控制器。\n"
                       "# 用 模型库 → 生成并打开 可获得带控制律模板的项目。", None)
            return
        # 同一控制器跨世界重载时保持当前选择（草稿继续显示）
        names = list(self._paths)
        name = next((n for n in names if self._paths[n] == prev), names[0])
        self.combo.blockSignals(True)
        self.combo.setCurrentText(name)
        self.combo.blockSignals(False)
        self._load_path(self._paths[name])

    def _on_combo_changed(self, _name: str) -> None:
        self._stash_current()
        self._load_path(self._paths.get(self.combo.currentText()))

    def _load_path(self, path: str | None) -> None:
        if not path:
            self._show("# 当前世界的机器人没有绑定控制器。", None)
            return
        draft = self._drafts.get(path)
        if draft is not None:                  # 有草稿：显示未保存内容
            self._show(draft, path, modified=True)
            return
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8")
        except OSError as e:
            text = f"# 读取 {path} 失败: {e}"
        self._show(text, path)

    def _reload_from_disk(self) -> None:
        """重读：确认后丢弃当前文件草稿，从磁盘重新载入。"""
        path = self._paths.get(self.combo.currentText())
        has_draft = self.edit.document().isModified() or (
            path is not None and path in self._drafts)
        if has_draft:
            yes = QMessageBox.StandardButton.Yes
            if QMessageBox.question(
                    self, "放弃修改",
                    "当前控制器有未保存的修改，确定放弃并重读？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != yes:
                return
        if path:
            self._drafts.pop(path, None)
        self._load_path(path)

    def _on_save(self) -> None:
        name = self.combo.currentText()
        if name not in self._paths:
            return
        self.saveRequested.emit(name, self.edit.toPlainText())


def make_side_panel() -> tuple[QWidget, dict]:
    """旧接口：右侧竖排（新主窗口已不用，保留向后兼容）。"""
    world, task, obs = WorldPanel(), TaskPanel(), ObsPanel()
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.addWidget(world)
    lay.addWidget(task)
    lay.addWidget(obs, stretch=1)
    return box, {"world": world, "task": task, "obs": obs}


# ================================================================ Agent 控制台

class AgentPanel(QWidget):
    """Agent 控制台：聊天 + 配置（OpenAI 兼容端点）+ 工具桥接。

    与 GUI 同构工作流：AI 调 zoo_create/load_world/run + 本地 write_file/read_file，
    用户可在聊天里让 AI 生成场景、写代码、运行验证。worker QThread 不阻塞 UI。
    """

    sendRequested = Signal(str)            # 用户文本 -> AgentWorker
    stopRequested = Signal()                # 停止 -> worker
    configureRequested = Signal()           # 打开配置对话框

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        bar = QHBoxLayout()
        self.btn_config = QPushButton("配置")
        self.btn_config.setToolTip("配置 OpenAI 兼容端点（base_url/api_key/model）")
        self.btn_config.clicked.connect(self.configureRequested.emit)
        bar.addWidget(self.btn_config)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("打断当前对话（不让 LLM 继续调工具）")
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        bar.addWidget(self.btn_stop)
        bar.addStretch(1)
        self.lbl_state = QLabel("未配置")
        bar.addWidget(self.lbl_state)
        v.addLayout(bar)

        self.chat = QPlainTextEdit()
        self.chat.setReadOnly(True)
        chat_mono = QFont()
        chat_mono.setFamilies(["JetBrains Mono", "Cascadia Mono", "Consolas",
                               "DejaVu Sans Mono"])
        chat_mono.setStyleHint(QFont.StyleHint.Monospace)
        chat_mono.setPointSize(10)
        self.chat.setFont(chat_mono)
        self.chat.setPlaceholderText("配置好 API 后在此向 AI 下达任务…")
        v.addWidget(self.chat, stretch=1)

        bar2 = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setFixedHeight(80)
        self.input.setPlaceholderText("对 AI 说话（Enter 发送，Shift+Enter 换行）")
        self.input.keyPressEvent = self._input_key_press
        bar2.addWidget(self.input, stretch=1)
        btn_send = QPushButton("发送")
        btn_send.clicked.connect(self._on_send)
        bar2.addWidget(btn_send)
        v.addLayout(bar2)

    def _input_key_press(self, event):
        from PySide6.QtCore import Qt as _Qt
        # Enter 发送、Shift+Enter 换行
        if (event.key() in (_Qt.Key.Key_Return, _Qt.Key.Key_Enter)
                and not event.modifiers() & _Qt.KeyboardModifier.ShiftModifier):
            self._on_send()
            return
        QPlainTextEdit.keyPressEvent(self.input, event)

    def _on_send(self):
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.sendRequested.emit(text)

    # ------------------------------------------------------------ UI 回写

    def append_user(self, text: str) -> None:
        self._append(f"[你] {text}", theme.C_TEXT)

    def append_assistant(self, text: str) -> None:
        self._append(f"[AI] {text}", theme.C_OK)

    def append_tool(self, name: str, args: dict) -> None:
        self._append(f"  ↪ 调用 {name}({args})", theme.C_TEXT_DIM)

    def append_result(self, name: str, result) -> None:
        if isinstance(result, dict) and "error" in result:
            color = theme.C_ERR
            text = f"  ← {name}: {result}"
        else:
            color = theme.C_TEXT_DIM
            preview = str(result)
            if len(preview) > 200:
                preview = preview[:200] + "…"
            text = f"  ← {name}: {preview}"
        self._append(text, color)

    def append_error(self, text: str) -> None:
        self._append(f"[错误] {text}", theme.C_ERR)

    def set_state(self, configured: bool, busy: bool) -> None:
        if configured:
            self.lbl_state.setText("运行中…" if busy else "就绪")
            self.lbl_state.setStyleSheet(
                f"color:{theme.C_OK if not busy else theme.C_WARN};"
                "font-weight:600;")
        else:
            self.lbl_state.setText("未配置")
            self.lbl_state.setStyleSheet(
                f"color:{theme.C_TEXT_DIM};font-weight:600;")
        self.btn_stop.setEnabled(busy)
        self.input.setEnabled(not busy)

    def _append(self, text: str, color: str) -> None:
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text + "\n", fmt)
        self.chat.setTextCursor(cursor)
        self.chat.verticalScrollBar().setValue(
            self.chat.verticalScrollBar().maximum())


class AgentConfigDialog(QDialog):
    """Agent 配置对话框：base_url / api_key / model 三项（持久化）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent 配置")
        from simovia.agent import load_config
        self._cfg = load_config()
        lay = QFormLayout(self)
        self.ed_url = QLineEdit(self._cfg.base_url)
        self.ed_url.setPlaceholderText("https://api.openai.com/v1 或 "
                                      "https://api.deepseek.com/v1")
        lay.addRow("base_url", self.ed_url)
        self.ed_key = QLineEdit(self._cfg.api_key)
        self.ed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_key.setPlaceholderText("sk-…（本地 Ollama 等可留空）")
        lay.addRow("api_key", self.ed_key)
        self.ed_model = QLineEdit(self._cfg.model)
        self.ed_model.setPlaceholderText("如 gpt-4o-mini / deepseek-chat / llama3")
        lay.addRow("model", self.ed_model)
        btns = QHBoxLayout()
        ok = QPushButton("保存")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(cancel); btns.addWidget(ok)
        lay.addRow("", btns)

    def config(self) -> "AgentConfig":
        from simovia.agent import AgentConfig
        return AgentConfig(base_url=self.ed_url.text().strip(),
                           api_key=self.ed_key.text().strip(),
                           model=self.ed_model.text().strip())


def _combo_index_of(combo: QComboBox, key: str, value) -> int:
    """按 itemData 字典里某个字段找下拉下标（找不到返回 0）。"""
    for i in range(combo.count()):
        d = combo.itemData(i)
        if isinstance(d, dict) and d.get(key) == value:
            return i
    return 0


class GearPanel(QWidget):
    """装备面板：机器人下拉 + 末端执行器 + 自带末端 + 传感器 + 相机参数。

    信号：
      apply_end_effector(robot, side, dict|None) 装/卸某一侧末端执行器
      add_sensor/remove_sensor          传感器增删
      set_hand(robot, key, stripped)    删除/恢复自带灵巧手
      update_sensor(robot, name, patch) 改相机参数（fovy/depth/near/far）
    """
    # robot_name, side（left/right/None）, 末端执行器 dict | None
    apply_end_effector = Signal(str, object, object)
    add_sensor = Signal(str, object)      # robot_name, sensor_dict
    remove_sensor = Signal(str, str)      # robot_name, sensor_name
    set_hand = Signal(str, str, bool)     # robot_name, key, stripped
    update_sensor = Signal(str, str, dict)  # robot_name, sensor_name, patch

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_dir = None
        self._manifest = None
        self._sensor_types = ["encoder", "imu", "tactile", "joint_torque", "camera"]

        lay = QVBoxLayout(self)
        # 机器人选择
        self._robot_row = QHBoxLayout()
        self._robot_row.addWidget(QLabel("选择机器人："))
        self._robot_combo = QComboBox()
        self._robot_combo.currentIndexChanged.connect(self._on_robot_changed)
        self._robot_row.addWidget(self._robot_combo)
        self._robot_row.addStretch()
        lay.addLayout(self._robot_row)

        # 末端执行器组（左右手各有挂载位）
        self._gripper_box = QGroupBox("末端执行器")
        gripper_lay = QFormLayout(self._gripper_box)
        # 安装位置：左手/右手/默认挂点（由 manifest 的 mounts 给出）
        self._mount_combo = QComboBox()
        self._mount_combo.currentIndexChanged.connect(self._on_mount_changed)
        gripper_lay.addRow("安装位置：", self._mount_combo)
        # 末端执行器：无 / 自带灵巧手 / 库内条目
        self._gripper_combo = QComboBox()
        self._gripper_combo.currentIndexChanged.connect(self._on_gripper_changed)
        gripper_lay.addRow("末端执行器：", self._gripper_combo)

        self._ee_status = QLabel("")
        self._ee_status.setWordWrap(True)
        self._ee_status.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        gripper_lay.addRow("", self._ee_status)

        self._site_combo = QComboBox()
        self._site_combo.setPlaceholderText("attachment_site（自动）")
        gripper_lay.addRow("挂点（高级）：", self._site_combo)

        self._pos_spins = [QDoubleSpinBox() for _ in range(3)]
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("x:"))
        for i in range(3):
            s = self._pos_spins[i]
            s.setRange(-20.0, 20.0); s.setDecimals(3); s.setSingleStep(0.01)
            pos_row.addWidget(s)
        gripper_lay.addRow("pos:", pos_row)

        self._rot_spins = [QDoubleSpinBox() for _ in range(3)]
        rot_row = QHBoxLayout()
        for i in range(3):
            s = self._rot_spins[i]
            s.setRange(-180.0, 180.0); s.setDecimals(1); s.setSingleStep(5.0)
            rot_row.addWidget(s)
        rot_row.addWidget(QLabel("（roll/pitch/yaw deg）"))
        gripper_lay.addRow("rot:", rot_row)

        self._bt_apply_gripper = QPushButton("应用末端执行器")
        self._bt_apply_gripper.clicked.connect(self._on_apply_gripper)
        self._bt_remove_gripper = QPushButton("移除")
        self._bt_remove_gripper.clicked.connect(self._on_remove_gripper)
        gripper_btn_row = QHBoxLayout()
        gripper_btn_row.addStretch()
        gripper_btn_row.addWidget(self._bt_remove_gripper)
        gripper_btn_row.addWidget(self._bt_apply_gripper)
        gripper_lay.addRow("", gripper_btn_row)
        lay.addWidget(self._gripper_box)

        # 自带末端（人形灵巧手/夹爪）组
        self._hand_box = QGroupBox("自带末端执行器（灵巧手 / 夹爪）")
        hand_lay = QVBoxLayout(self._hand_box)
        self._hand_list = QListWidget()
        self._hand_list.setMaximumHeight(110)
        hand_lay.addWidget(self._hand_list)
        hand_btns = QHBoxLayout()
        self._bt_restore_hand = QPushButton("恢复选中")
        self._bt_restore_hand.clicked.connect(lambda: self._on_hand_toggle(False))
        self._bt_strip_hand = QPushButton("删除选中")
        self._bt_strip_hand.clicked.connect(lambda: self._on_hand_toggle(True))
        hand_btns.addWidget(self._bt_restore_hand)
        hand_btns.addStretch()
        hand_btns.addWidget(self._bt_strip_hand)
        hand_lay.addLayout(hand_btns)
        hand_hint = QLabel("删除后自动在腕部生成挂点 site，可再挂末端执行器库里的末端")
        hand_hint.setWordWrap(True)
        hand_hint.setStyleSheet(f"color:{theme.C_TEXT_DIM}; font-size:11px;")
        hand_lay.addWidget(hand_hint)
        lay.addWidget(self._hand_box)

        # 传感器组
        self._sensor_box = QGroupBox("传感器")
        sensor_lay = QVBoxLayout(self._sensor_box)
        self._sensor_list = QListWidget()
        sensor_lay.addWidget(self._sensor_list)

        self._bt_del_sensor = QPushButton("删除选中传感器")
        self._bt_del_sensor.clicked.connect(self._on_remove_sensor)
        sensor_lay.addWidget(self._bt_del_sensor)

        # 新增传感器表单
        self._add_box = QGroupBox("新增传感器")
        add_lay = QFormLayout(self._add_box)
        self._sensor_type = QComboBox()
        self._sensor_type.addItems(self._sensor_types)
        self._sensor_type.currentIndexChanged.connect(self._on_sensor_type_changed)
        add_lay.addRow("类型：", self._sensor_type)
        self._sensor_name = QLineEdit()
        self._sensor_name.setPlaceholderText("如 \"left_touch\"")
        add_lay.addRow("名称：", self._sensor_name)

        self._ref_label = QLabel("引用关节：")
        self._ref_stack = QStackedWidget()
        self._ref_joint = QComboBox()
        self._ref_body = QComboBox()
        self._ref_stack.addWidget(self._ref_joint)   # 页 0：关节
        self._ref_stack.addWidget(self._ref_body)    # 页 1：body
        add_lay.addRow(self._ref_label, self._ref_stack)

        pos_row_add = QHBoxLayout()
        self._pos_spins_add = [QDoubleSpinBox() for _ in range(3)]
        for s in self._pos_spins_add:
            s.setRange(-20.0, 20.0); s.setDecimals(3); s.setSingleStep(0.01)
            pos_row_add.addWidget(s)
        add_lay.addRow("pos:", pos_row_add)

        self._quat_label = QLabel("quat (wxyz):")
        self._quat_spins_add = [QDoubleSpinBox() for _ in range(4)]
        quat_row_add = QHBoxLayout()
        self._quat_spins_add[0].setValue(1.0)
        for s in self._quat_spins_add:
            s.setRange(-1.0, 1.0); s.setDecimals(4); s.setSingleStep(0.05)
            quat_row_add.addWidget(s)
        add_lay.addRow(self._quat_label, quat_row_add)

        self._radius_label = QLabel("radius (m):")
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.001, 1.0); self._radius_spin.setDecimals(3)
        self._radius_spin.setSingleStep(0.01); self._radius_spin.setValue(0.06)
        add_lay.addRow(self._radius_label, self._radius_spin)

        self._bt_add_sensor = QPushButton("新增")
        self._bt_add_sensor.clicked.connect(self._on_add_sensor)
        add_btn_row = QHBoxLayout()
        add_btn_row.addStretch(); add_btn_row.addWidget(self._bt_add_sensor)
        add_lay.addRow("", add_btn_row)
        sensor_lay.addWidget(self._add_box)

        lay.addWidget(self._sensor_box)

        # 相机参数（视场角 / 深度 / 裁剪面）
        self._cam_box = QGroupBox("相机参数")
        cam_lay = QFormLayout(self._cam_box)
        self._cam_combo = QComboBox()
        self._cam_combo.currentIndexChanged.connect(self._on_cam_changed)
        cam_lay.addRow("相机：", self._cam_combo)
        self._cam_fovy = QDoubleSpinBox()
        self._cam_fovy.setRange(1.0, 179.0); self._cam_fovy.setDecimals(1)
        self._cam_fovy.setSingleStep(1.0); self._cam_fovy.setValue(45.0)
        self._cam_fovy.setSuffix(" °")
        cam_lay.addRow("视场角 fovy：", self._cam_fovy)
        self._cam_depth = QCheckBox("深度相机（实时画面可切伪彩）")
        cam_lay.addRow("", self._cam_depth)
        near_far = QHBoxLayout()
        self._cam_near = QDoubleSpinBox()
        self._cam_near.setRange(0.001, 10.0); self._cam_near.setDecimals(3)
        self._cam_near.setSingleStep(0.01); self._cam_near.setValue(0.01)
        self._cam_far = QDoubleSpinBox()
        self._cam_far.setRange(0.1, 1000.0); self._cam_far.setDecimals(1)
        self._cam_far.setSingleStep(1.0); self._cam_far.setValue(50.0)
        near_far.addWidget(QLabel("near:"))
        near_far.addWidget(self._cam_near)
        near_far.addWidget(QLabel("far:"))
        near_far.addWidget(self._cam_far)
        cam_lay.addRow("深度裁剪：", near_far)
        self._bt_save_cam = QPushButton("保存相机参数")
        self._bt_save_cam.clicked.connect(self._on_save_camera)
        cam_btns = QHBoxLayout()
        cam_btns.addStretch(); cam_btns.addWidget(self._bt_save_cam)
        cam_lay.addRow("", cam_btns)
        lay.addWidget(self._cam_box)
        lay.addStretch()

        self._clear()

    def set_manifest(self, manifest: dict, base_dir: str):
        self._manifest = manifest
        self._base_dir = base_dir
        self._robot_combo.clear()
        for r in manifest.get("robots", []):
            self._robot_combo.addItem(r["name"], r)
        self._on_robot_changed()

    def _clear(self):
        self._robot_combo.clear()
        self._mount_combo.clear()
        self._gripper_combo.clear()
        self._site_combo.clear()
        self._hand_list.clear()
        self._sensor_list.clear()
        self._cam_combo.clear()
        self._ref_joint.clear()
        self._ref_body.clear()

    def _current_robot(self) -> dict | None:
        return self._robot_combo.currentData()

    def _ee_catalog(self) -> list[dict]:
        m = self._manifest or {}
        return m.get("end_effector_catalog") or m.get("gripper_catalog") or []

    def _on_robot_changed(self):
        r = self._current_robot()
        self._gripper_box.setEnabled(r is not None)
        self._hand_box.setEnabled(r is not None)
        self._sensor_box.setEnabled(r is not None)
        self._cam_box.setEnabled(r is not None)
        if r is None:
            return
        # ---- 末端执行器：安装位置（左手/右手/默认）+ 当前状态 ----
        self._manifest_ees = r.get("end_effectors", [])
        mounts = r.get("mounts") or [
            {"side": None, "site": None, "label": "默认挂点",
             "builtin": None, "end_effector": r.get("gripper")}]
        self._mount_combo.blockSignals(True)
        self._mount_combo.clear()
        for m in mounts:
            label = m.get("label") or "默认挂点"
            if m.get("builtin"):
                label += "（自带末端）"
            self._mount_combo.addItem(label, m)
        self._mount_combo.blockSignals(False)
        # 挂点下拉（高级手填）：默认按安装位置自动
        self._site_combo.clear()
        self._site_combo.addItem("自动（按安装位置）", None)
        for s in r.get("sites", []):
            self._site_combo.addItem(s, s)
        self._refresh_ee_choices()
        # 传感器列表
        self._sensor_list.clear()
        for s in r.get("sensors", []):
            self._sensor_list.addItem(f"{s['name']} ({s['type']})")
        # 自带末端列表（灵巧手/夹爪）
        self._hand_list.clear()
        for ee in r.get("builtin_end_effectors", []):
            st = "已移除" if ee.get("stripped") else "自带"
            item = QListWidgetItem(
                f"{ee['key']} · {st} · 腕部 {ee['anchor']} · "
                f"{len(ee.get('bodies', []))} body")
            item.setData(Qt.ItemDataRole.UserRole, ee["key"])
            item.setToolTip("挂点 site: " + str(ee.get("site")))
            self._hand_list.addItem(item)
        # 相机下拉（仅 camera 类型传感器）
        self._cam_combo.clear()
        for s in r.get("sensors", []):
            if s.get("type") == "camera":
                self._cam_combo.addItem(
                    f"{s['name']} @ {s.get('body', '?')}", s["name"])
        self._on_cam_changed()
        # 关节/body 下拉
        self._ref_joint.clear()
        self._ref_body.clear()
        for j in r.get("joints", []):
            self._ref_joint.addItem(j["name"])
        for short in r.get("bodies", {}).values():
            self._ref_body.addItem(short)
        self._on_sensor_type_changed()

    def _on_sensor_type_changed(self):
        t = self._sensor_type.currentText()
        is_joint_sensor = t in ("encoder", "joint_torque")
        self._ref_label.setText("引用关节：" if is_joint_sensor else "引用 body：")
        self._ref_stack.setCurrentIndex(0 if is_joint_sensor else 1)
        # 显示/隐藏 quat/radius
        self._quat_label.setVisible(t == "camera")
        for s in self._quat_spins_add:
            s.setVisible(t == "camera")
        self._radius_label.setVisible(t == "tactile")
        self._radius_spin.setVisible(t == "tactile")

    def _current_mount(self) -> dict | None:
        return self._mount_combo.currentData()

    def _refresh_ee_choices(self):
        """按当前安装位置重建末端执行器下拉：无 / 自带灵巧手 / 库内条目。"""
        mount = self._current_mount() or {}
        current = mount.get("end_effector") or {}
        self._gripper_combo.blockSignals(True)
        self._gripper_combo.clear()
        self._gripper_combo.addItem("无（该侧不装）", None)
        builtin = mount.get("builtin")
        builtin_stripped = False
        if builtin:
            stripped = any(
                e.get("key") == builtin and e.get("stripped")
                for e in getattr(self, "_manifest_ees", []))
            builtin_stripped = stripped
            label = f"自带灵巧手（{builtin}）" + ("" if stripped else "（已装配）")
            self._gripper_combo.addItem(label, {"builtin": builtin})
        for c in self._ee_catalog():
            title = c.get("title") or c["key"]
            self._gripper_combo.addItem(f"{title}（{c['key']}）",
                                        {"model": c["key"]})
        # 选中当前已装的条目：外挂条目优先，否则自带手（未删除时视为已装）
        if current.get("model"):
            for i in range(self._gripper_combo.count()):
                d = self._gripper_combo.itemData(i)
                if d and d.get("model") == current["model"]:
                    self._gripper_combo.setCurrentIndex(i)
                    break
        elif builtin and not builtin_stripped:
            self._gripper_combo.setCurrentIndex(
                _combo_index_of(self._gripper_combo, "builtin", builtin))
        self._gripper_combo.blockSignals(False)
        self._refresh_ee_status()
        self._on_gripper_changed()

    def _refresh_ee_status(self):
        """当前安装位状态（库内型号 / 自带灵巧手 / 空）。"""
        mount = self._current_mount() or {}
        current = mount.get("end_effector")
        if current and current.get("model"):
            text = f"当前：{current['model']}（挂点 {current.get('site')}）"
        elif mount.get("builtin"):
            stripped = any(
                e.get("key") == mount["builtin"] and e.get("stripped")
                for e in getattr(self, "_manifest_ees", []))
            text = (f"当前：自带灵巧手 {mount['builtin']}（已移除）"
                    if stripped else
                    f"当前：自带灵巧手 {mount['builtin']}（已装配）")
        else:
            text = "当前：未装末端执行器"
        self._ee_status.setText(text)

    def _on_mount_changed(self):
        self._refresh_ee_choices()

    def _on_gripper_changed(self):
        self._bt_apply_gripper.setEnabled(
            self._gripper_combo.currentData() is not None)
        self._bt_remove_gripper.setEnabled(self._current_mount() is not None)
        # 选了库内条目时挂点默认按安装位置自动
        if isinstance(self._gripper_combo.currentData(), dict) and \
                self._gripper_combo.currentData().get("model"):
            self._site_combo.setCurrentIndex(0)

    def _on_apply_gripper(self):
        r = self._current_robot()
        mount = self._current_mount()
        data = self._gripper_combo.currentData()
        if r is None or mount is None or data is None:
            return
        side = mount.get("side")
        spec = dict(data)
        if "model" in spec:
            site = self._site_combo.currentData()
            if site is not None:
                spec["site"] = site
            pos = tuple(round(s.value(), 4) for s in self._pos_spins)
            if pos != (0.0, 0.0, 0.0):
                spec["pos"] = list(pos)
            quat = tuple(round(s.value(), 4) for s in self._rot_spins)
            if quat != (0.0, 0.0, 0.0):
                spec["quat"] = list(euler_to_quat(*quat))
        self.apply_end_effector.emit(r["name"], side, spec)

    def _on_remove_gripper(self):
        r = self._current_robot()
        mount = self._current_mount()
        if r is not None and mount is not None:
            self.apply_end_effector.emit(r["name"], mount.get("side"), None)

    def _on_add_sensor(self):
        r = self._current_robot()
        name = self._sensor_name.text().strip()
        typ = self._sensor_type.currentText()
        if r is None or not name or not typ:
            return
        sd = {"name": name, "type": typ}
        if typ in ("encoder", "joint_torque"):
            sd["joint"] = self._ref_joint.currentText()
        else:
            sd["body"] = self._ref_body.currentText()
        pos = [round(s.value(), 4) for s in self._pos_spins_add]
        if pos != [0.0, 0.0, 0.0]:
            sd["pos"] = pos
        if typ == "camera":
            quat = [round(s.value(), 4) for s in self._quat_spins_add]
            if quat != [1.0, 0.0, 0.0, 0.0]:
                sd["quat"] = quat
        if typ == "tactile":
            sd["radius"] = round(self._radius_spin.value(), 4)
        self.add_sensor.emit(r["name"], sd)
        self._sensor_name.clear()

    def _on_remove_sensor(self):
        r = self._current_robot()
        item = self._sensor_list.currentItem()
        if r and item:
            # 解析出传感器名（"name (type)"）
            txt = item.text()
            name = txt.split()[0]
            self.remove_sensor.emit(r["name"], name)

    def _on_hand_toggle(self, stripped: bool):
        """删除/恢复选中的自带末端（灵巧手/夹爪），由 app 写回 YAML 重载。"""
        r = self._current_robot()
        item = self._hand_list.currentItem()
        if r is None or item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        # 已经是目标状态就不重复写
        for ee in r.get("end_effectors", []):
            if ee["key"] == key and bool(ee.get("stripped")) == stripped:
                return
        self.set_hand.emit(r["name"], key, bool(stripped))

    def _selected_camera(self) -> dict | None:
        r = self._current_robot()
        name = self._cam_combo.currentData()
        if r is None or not name:
            return None
        for s in r.get("sensors", []):
            if s.get("name") == name and s.get("type") == "camera":
                return s
        return None

    def _on_cam_changed(self):
        """切相机时回填该相机的 fovy / 深度 / 裁剪面参数。"""
        cam = self._selected_camera()
        blocked = [self._cam_fovy, self._cam_depth]
        for w in blocked:
            w.blockSignals(True)
        try:
            if cam is None:
                self._cam_fovy.setValue(45.0)
                self._cam_depth.setChecked(False)
                self._cam_near.setValue(0.01)
                self._cam_far.setValue(50.0)
            else:
                fovy = cam.get("fovy")
                self._cam_fovy.setValue(45.0 if fovy is None else float(fovy))
                self._cam_depth.setChecked(bool(cam.get("depth")))
                self._cam_near.setValue(float(cam.get("near") or 0.01))
                self._cam_far.setValue(float(cam.get("far") or 50.0))
        finally:
            for w in blocked:
                w.blockSignals(False)
        self._bt_save_cam.setEnabled(cam is not None)

    def _on_save_camera(self):
        r = self._current_robot()
        cam = self._selected_camera()
        if r is None or cam is None:
            return
        near = round(self._cam_near.value(), 3)
        far = round(self._cam_far.value(), 1)
        if far <= near:
            self._bt_save_cam.setToolTip("far 必须大于 near")
            return
        patch = {"fovy": round(self._cam_fovy.value(), 1),
                 "depth": self._cam_depth.isChecked(),
                 "near": near, "far": far}
        self.update_sensor.emit(r["name"], cam["name"], patch)


# ================================================================ 添加物体

_OBJECT_COLORS = [
    ("红", "#EF4444"), ("橙", "#F97316"), ("黄", "#FACC15"),
    ("绿", "#22C55E"), ("蓝", "#3B82F6"), ("紫", "#A855F7"),
]


class AddObjectDialog(QDialog):
    """往场景添加一个物体（类型/尺寸/颜色），放在相机注视点。

    selection() -> (type, size, color)：size 按 MJCF 半尺寸语义——
    box=(x,y,z 半边)，cylinder=(半径,半高)，sphere=(半径,)。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加物体")
        form = QFormLayout(self)

        self._type = QComboBox()
        self._type.addItems(["box（方块）", "cylinder（圆柱）",
                             "sphere（球）"])
        form.addRow("类型", self._type)

        self._size = QDoubleSpinBox()
        self._size.setRange(0.01, 5.0)
        self._size.setSingleStep(0.05)
        self._size.setValue(0.08)
        self._size.setSuffix(" m")
        self._size.setToolTip("box: 半边长（正方体）；cylinder: 半径；sphere: 半径")
        form.addRow("尺寸", self._size)

        self._color = QComboBox()
        for name, hexv in _OBJECT_COLORS:
            px = QPixmap(14, 14)
            px.fill(QColor(hexv))
            self._color.addItem(QIcon(px), name)
        form.addRow("颜色", self._color)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                 QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def selection(self):
        idx = self._type.currentIndex()
        s = round(float(self._size.value()), 4)
        color = _OBJECT_COLORS[self._color.currentIndex()][1]
        if idx == 1:
            return "cylinder", (s, s), color
        if idx == 2:
            return "sphere", (s,), color
        return "box", (s, s, s), color
