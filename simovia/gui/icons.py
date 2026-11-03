"""工具栏矢量图标：QPainter 现画，不依赖图片资源。

为什么手画而不是用图标字体/PNG：
- 仓库保持零额外资源依赖，任意 DPI 下都清晰（每个尺寸单独绘制）；
- 颜色跟随主题（深色主题下默认前景色，运行/停止用语义色）；
- 参考 Isaac Sim / Webots 的线性图标风格：细描边 + 圆角端点。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QGuiApplication, QIcon, QPainter,
                           QPainterPath, QPen, QPixmap)

from simovia.gui import theme

_SIZES = (16, 20, 24, 32)
_BASE = 24.0                                  # 绘制坐标系边长
_cache: dict[tuple[str, str, str], QIcon] = {}


def _pen(color: str, width: float = 2.05) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_open(p, c, a):
    """文件夹。"""
    path = QPainterPath()
    path.moveTo(3, 6)
    path.lineTo(9, 6)
    path.lineTo(11, 8.5)
    path.lineTo(21, 8.5)
    path.lineTo(21, 19)
    path.lineTo(3, 19)
    path.closeSubpath()
    p.drawPath(path)


def _draw_zoo(p, c, a):
    """机器人：圆角头 + 双眼 + 天线。"""
    p.drawRoundedRect(QRectF(5, 8, 14, 11), 2.5, 2.5)
    p.drawLine(QPointF(12, 8), QPointF(12, 4))
    p.setBrush(QColor(c))
    p.drawEllipse(QPointF(12, 3.4), 1.5, 1.5)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(9.2, 13), 1.2, 1.2)
    p.drawEllipse(QPointF(14.8, 13), 1.2, 1.2)
    p.drawLine(QPointF(9, 16), QPointF(15, 16))


def _draw_builder(p, c, a):
    """连杆树：根节点 + 两个子节点。"""
    for cx, cy, r in ((12, 5, 2.4), (5.5, 18, 2.4), (18.5, 18, 2.4)):
        p.drawEllipse(QPointF(cx, cy), r, r)
    p.drawLine(QPointF(12, 7.5), QPointF(5.5, 15.5))
    p.drawLine(QPointF(12, 7.5), QPointF(18.5, 15.5))


def _draw_import(p, c, a):
    """立方体 + 向下箭头（导入模型）。"""
    path = QPainterPath()
    path.moveTo(12, 3)
    path.lineTo(20, 7)
    path.lineTo(20, 17)
    path.lineTo(12, 21)
    path.lineTo(4, 17)
    path.lineTo(4, 7)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(12, 8), QPointF(12, 15))
    p.drawLine(QPointF(9, 12.5), QPointF(12, 15.5))
    p.drawLine(QPointF(15, 12.5), QPointF(12, 15.5))


def _draw_save(p, c, a):
    """软盘。"""
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
    p.drawRect(QRectF(9, 4, 6, 6))
    p.drawRect(QRectF(8, 14, 8, 6))


def _draw_reset(p, c, a):
    """逆时针圆弧 + 箭头（回到初态）。"""
    p.drawArc(QRectF(4.5, 4.5, 15, 15), 60 * 16, 280 * 16)
    path = QPainterPath()
    path.moveTo(4.6, 8.2)
    path.lineTo(9.4, 7.4)
    path.lineTo(6.9, 11.6)
    path.closeSubpath()
    p.setBrush(QColor(c))
    p.drawPath(path)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_run(p, c, a):
    """播放三角（绿色实心）。"""
    path = QPainterPath()
    path.moveTo(8, 5)
    path.lineTo(19, 12)
    path.lineTo(8, 19)
    path.closeSubpath()
    p.setBrush(QColor(a))
    p.setPen(QPen(QColor(a), 1.2))
    p.drawPath(path)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_stop(p, c, a):
    """停止方块（红色实心）。"""
    p.setBrush(QColor(theme.C_ERR))
    p.setPen(QPen(QColor(theme.C_ERR), 1.2))
    p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2, 2)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_step(p, c, a):
    """播放三角 + 竖线（单步）。"""
    path = QPainterPath()
    path.moveTo(7, 5)
    path.lineTo(16.5, 12)
    path.lineTo(7, 19)
    path.closeSubpath()
    p.setBrush(QColor(a))
    p.setPen(QPen(QColor(a), 1.2))
    p.drawPath(path)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(_pen(c))
    p.drawLine(QPointF(18.5, 5.5), QPointF(18.5, 18.5))


def _draw_contacts(p, c, a):
    """接触点：中心圆 + 十字刻度。"""
    p.drawEllipse(QPointF(12, 12), 3, 3)
    p.setBrush(QColor(c))
    p.drawEllipse(QPointF(12, 12), 1.2, 1.2)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        p.drawLine(QPointF(12 + dx * 6, 12 + dy * 6),
                   QPointF(12 + dx * 8.5, 12 + dy * 8.5))


def _draw_force(p, c, a):
    """向上的力箭头。"""
    p.drawLine(QPointF(12, 20), QPointF(12, 7))
    p.drawLine(QPointF(12, 7), QPointF(7.8, 11.2))
    p.drawLine(QPointF(12, 7), QPointF(16.2, 11.2))


def _draw_com(p, c, a):
    """质心十字准星。"""
    p.drawEllipse(QPointF(12, 12), 5, 5)
    p.drawLine(QPointF(12, 3), QPointF(12, 7))
    p.drawLine(QPointF(12, 17), QPointF(12, 21))
    p.drawLine(QPointF(3, 12), QPointF(7, 12))
    p.drawLine(QPointF(17, 12), QPointF(21, 12))
    p.setBrush(QColor(c))
    p.drawEllipse(QPointF(12, 12), 1.3, 1.3)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_frames(p, c, a):
    """三轴坐标系。"""
    p.drawLine(QPointF(7, 18), QPointF(18, 18))
    p.drawLine(QPointF(7, 18), QPointF(7, 6))
    p.drawLine(QPointF(7, 18), QPointF(14, 11))
    p.setBrush(QColor(c))
    for x, y in ((18.8, 18), (7, 5.2), (14.6, 10.4)):
        p.drawEllipse(QPointF(x, y), 1.2, 1.2)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_wire(p, c, a):
    """线框立方体。"""
    p.drawRect(QRectF(5.5, 7.5, 11, 11))
    p.drawLine(QPointF(5.5, 7.5), QPointF(9, 4))
    p.drawLine(QPointF(16.5, 7.5), QPointF(20, 4))
    p.drawLine(QPointF(16.5, 18.5), QPointF(20, 15))
    p.drawLine(QPointF(9, 4), QPointF(20, 4))
    p.drawLine(QPointF(20, 4), QPointF(20, 15))


def _draw_screenshot(p, c, a):
    """相机。"""
    p.drawRoundedRect(QRectF(3.5, 7, 17, 12), 2.5, 2.5)
    p.drawEllipse(QPointF(12, 13), 4, 4)
    p.drawLine(QPointF(8, 7), QPointF(9.5, 4.5))
    p.drawLine(QPointF(9.5, 4.5), QPointF(14.5, 4.5))
    p.drawLine(QPointF(14.5, 4.5), QPointF(16, 7))


def _draw_delete(p, c, a):
    """垃圾桶。"""
    p.drawLine(QPointF(5, 7), QPointF(19, 7))
    p.drawLine(QPointF(10, 4.5), QPointF(14, 4.5))
    p.drawRoundedRect(QRectF(7, 7, 10, 13), 1.5, 1.5)
    p.drawLine(QPointF(10.5, 10.5), QPointF(10.5, 16.5))
    p.drawLine(QPointF(13.5, 10.5), QPointF(13.5, 16.5))


def _draw_add(p, c, a):
    """加号。"""
    p.drawLine(QPointF(12, 5), QPointF(12, 19))
    p.drawLine(QPointF(5, 12), QPointF(19, 12))


def _draw_ai(p, c, a):
    """四角星（AI 生成）。"""
    path = QPainterPath()
    path.moveTo(12, 3)
    path.quadTo(13.5, 10.5, 21, 12)
    path.quadTo(13.5, 13.5, 12, 21)
    path.quadTo(10.5, 13.5, 3, 12)
    path.quadTo(10.5, 10.5, 12, 3)
    p.drawPath(path)


_DRAW = {
    "open": _draw_open,
    "zoo": _draw_zoo,
    "builder": _draw_builder,
    "import": _draw_import,
    "save": _draw_save,
    "reset": _draw_reset,
    "run": _draw_run,
    "stop": _draw_stop,
    "step": _draw_step,
    "contacts": _draw_contacts,
    "force": _draw_force,
    "com": _draw_com,
    "frames": _draw_frames,
    "wire": _draw_wire,
    "screenshot": _draw_screenshot,
    "delete": _draw_delete,
    "add": _draw_add,
    "ai": _draw_ai,
}


def icon(name: str, color: str = theme.C_TEXT,
         accent: str = theme.C_OK) -> QIcon:
    """取一个矢量图标；未知名字返回空 QIcon（不抛异常）。

    accent 用于"运行/步进"这类语义色图标。结果按 (名字, 颜色, 强调色)
    缓存——工具栏只构建一次，缓存避免重复绘制。
    """
    key = (name, color, accent)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    if QGuiApplication.instance() is None:       # 没有 GUI 环境（纯导入）
        return QIcon()
    draw = _DRAW.get(name)
    if draw is None:
        return QIcon()
    ico = QIcon()
    for size in _SIZES:
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(size / _BASE, size / _BASE)
        p.setPen(_pen(color))
        p.setBrush(Qt.BrushStyle.NoBrush)
        draw(p, color, accent)
        p.end()
        ico.addPixmap(pm)
    _cache[key] = ico
    return ico
