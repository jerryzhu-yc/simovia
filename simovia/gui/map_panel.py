"""雷达建图面板：占据栅格 + 机器人 + 控制器叠加层的俯视 SLAM 视图。

数据链路（平台级能力，与控制器实现无关）：
- 内核推进线程自动喂图（simovia/kernel/mapping.py，任何 lidar 传感器）
- 主窗口 10Hz 轮询 MCP lidar_map 工具拉快照（栅格 + 位姿 + 叠加层）
- 控制器经 publish_overlay 发布路径 / roadmap / 目标点等矢量图层

渲染约定（与控制器时代 publish_camera 的 SLAM 图配色一致）：
- 中灰=未知 / 灰白=自由 / 黑=占据；多机器人地图取"最确定"观测合并
  （|p-128| 最大者，避免未知覆盖已知自由）
- 叠加层色表：path 黄（半透明）/ goal 绿 / start 橙 / skeleton 蓝 / patrol 暗黄
"""
from __future__ import annotations

import base64
import zlib

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from simovia.gui import theme

# 栅格三态配色（uint8 0/1/2 -> RGB）
_C_UNKNOWN = (105, 105, 105)
_C_FREE = (235, 235, 235)
_C_OCC = (20, 20, 20)
_GRID_RGB = np.array([_C_UNKNOWN, _C_FREE, _C_OCC], dtype=np.uint8)

# 叠加层色表（label 前缀匹配，未匹配走默认紫）
_OVERLAY_COLORS = [
    ("path", QColor(255, 190, 0)),
    ("patrol", QColor(190, 140, 40)),
    ("goal", QColor(0, 200, 80)),
    ("start", QColor(255, 110, 40)),
    ("skeleton", QColor(110, 170, 255)),
    ("voronoi", QColor(110, 170, 255)),
    ("prm", QColor(90, 140, 255)),
    ("roadmap", QColor(90, 140, 255)),
]
_C_OVERLAY_DEFAULT = QColor(190, 120, 255)
_C_ROBOT = QColor(0, 230, 230)

# 需要半透明的图层：A* 路径盖在骨架图上，必须能透出底下骨架才能判断
# 它是否贴着骨架走；骨架本身作为背景参考也降透明度
_LAYER_ALPHA = {"path": 130, "skeleton": 120, "voronoi": 120,
                "prm": 120, "roadmap": 120}

_PAD = 6.0        # 画布四周留白（像素）

# 叠加层绘制顺序：数值越小越先画（被后画的压在上面）
_LAYER_ORDER = ("patrol", "skeleton", "voronoi", "prm", "roadmap",
                "path", "goal", "start")


def _ordered_layers(overlays: dict):
    """展平 {robot: {label: layer}} 并按语义排序：巡逻最底、A* 路径与目标点最上。"""
    def rank(pair) -> int:
        low = pair[0].lower()
        for i, key in enumerate(_LAYER_ORDER):
            if key in low:
                return i
        return len(_LAYER_ORDER)
    return sorted(((label, layer)
                   for layers in overlays.values()
                   for label, layer in layers.items()), key=rank)


def _layer_alpha(label: str) -> int:
    """按图层查透明度（路径/骨架半透明，能看到上下层是否重合）。"""
    low = label.lower()
    for key, alpha in _LAYER_ALPHA.items():
        if key in low:
            return alpha
    return 255


def _overlay_color(label: str) -> QColor:
    low = label.lower()
    for prefix, color in _OVERLAY_COLORS:
        if prefix in low:
            return color
    return _C_OVERLAY_DEFAULT


# 概率渐变的三个锚点色（柔和化，避免纯黑纯白的生硬观感）
_C_FREE_RGB = np.array([238, 238, 235], dtype=np.float32)
_C_UNKNOWN_RGB = np.array([106, 109, 118], dtype=np.float32)
_C_OCC_RGB = np.array([26, 30, 38], dtype=np.float32)


def _prob_to_rgb(prob: np.ndarray) -> np.ndarray:
    """uint8 概率图 (nx,ny) -> RGB (nx,ny,3)：分段线性渐变。

    0=确定自由（米白）.. 128=未知（蓝灰）.. 255=确定占据（墨黑），
    占据边缘因概率未饱和而自然过渡，比三态硬切的大块锯齿细腻得多。
    """
    t = (prob.astype(np.float32) / 255.0)[..., None]
    lo = t / 0.5
    hi = (t - 0.5) / 0.5
    rgb = np.where(t <= 0.5,
                   _C_FREE_RGB + (_C_UNKNOWN_RGB - _C_FREE_RGB) * lo,
                   _C_UNKNOWN_RGB + (_C_OCC_RGB - _C_UNKNOWN_RGB) * hi)
    return rgb.clip(0, 255).astype(np.uint8)


class RadarMapPanel(QWidget):
    """雷达建图俯视图：数据由主窗口 lidar_map 轮询喂入（update_map）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 220)
        self._maps: dict = {}         # robot -> {res,ox,oy,nx,ny,grid(bytes b64)}
        self._poses: dict = {}        # robot -> [x, y, yaw]
        self._overlays: dict = {}     # robot -> label -> {segments, points}

    # ------------------------------------------------------------ 数据入口
    def update_map(self, data: dict) -> None:
        """吃 lidar_map 工具返回；available=false 时清空画面。"""
        if not data or not data.get("available"):
            self._maps, self._poses, self._overlays = {}, {}, {}
        else:
            self._maps = data.get("maps", {})
            self._poses = data.get("poses", {})
            self._overlays = data.get("overlays", {})
        self.update()

    def has_map(self) -> bool:
        return bool(self._maps)

    # ------------------------------------------------------------ 绘制
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.C_BG))
        if not self._maps:
            p.setPen(QColor(theme.C_TEXT_DIM))
            p.drawText(self.rect(), 0x84, "等待雷达数据…")   # AlignCenter
            p.end()
            return
        try:
            self._draw_map(p)
        finally:
            p.end()

    def _draw_map(self, p: QPainter) -> None:
        # 合并所有机器人的概率图（取 max：占据 > 未知 > 自由）。
        # 优先用连续概率（prob）做灰度渐变，老数据只有三态 grid 时回退。
        first = next(iter(self._maps.values()))
        nx, ny = int(first["nx"]), int(first["ny"])
        probs, grids = [], []
        for m in self._maps.values():
            if (int(m["nx"]), int(m["ny"])) != (nx, ny):
                continue                  # 范围不一致的图跳过（理论不发生）
            blob = m.get("prob") or m.get("grid")
            raw = np.frombuffer(zlib.decompress(base64.b64decode(blob)),
                                dtype=np.uint8).reshape(nx, ny)
            (probs if m.get("prob") else grids).append(raw)
        if probs:
            # 合并取"最确定"的观测：|p-128| 最大者胜（自由与占据都算确定）。
            # 不能用 maximum——未知的 128 会把自由格（≈2）全压成未知灰
            stack = np.stack(probs).astype(np.int16)
            dev = np.abs(stack - 128)
            merged = np.take_along_axis(
                stack, dev.argmax(axis=0)[None], axis=0)[0].astype(np.uint8)
            rgb = _prob_to_rgb(merged)
        else:
            merged3 = np.maximum.reduce([np.zeros((nx, ny), dtype=np.uint8)]
                                        + grids)
            rgb = _GRID_RGB[merged3]

        # 图例先算：横向排在地图下方独立条带，不遮挡地图内容
        legend_items: list[tuple[str, QColor]] = []
        for layers in self._overlays.values():
            for label in layers:
                if label not in [x[0] for x in legend_items]:
                    legend_items.append((label, _overlay_color(label)))
        legend_h = 24.0 if legend_items else 0.0

        # 画布布局：等比缩放 + 留白（底部扣除图例条带）
        res, ox, oy = (float(first["res"]), float(first["ox"]),
                       float(first["oy"]))
        w_m, h_m = nx * res, ny * res
        avail_w = self.width() - 2 * _PAD
        avail_h = self.height() - 2 * _PAD - legend_h
        scale = min(avail_w / w_m, avail_h / h_m)
        img_w, img_h = w_m * scale, h_m * scale
        x0 = (self.width() - img_w) / 2.0
        y0 = (self.height() - legend_h - 2 * _PAD - img_h) / 2.0 + _PAD

        # 栅格 -> RGB 图像（栅格 [ix,iy]、iy 向上 -> 图像行向下）
        rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1])  # (ny,nx,3)
        image = QImage(rgb.data, nx, ny, nx * 3, QImage.Format.Format_RGB888)
        # 平滑插值：5cm 格放大到数像素时边界柔和，消除大块锯齿观感
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.drawImage(QRectF(x0, y0, img_w, img_h), image)
        # 细边框收口，地图不外溢到留白
        p.setPen(QPen(QColor(70, 74, 84), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(x0, y0, img_w, img_h))

        # 世界(米) -> 像素变换（y 轴向上翻成图像向上）。
        # scale 单位是 px/米，直接乘米即可——切勿先 /res 变格数，
        # 否则坐标放大 1/res 倍，叠加层全部飞出画布
        def wx(x):
            return x0 + (x - ox) * scale

        def wy(y):
            return y0 + img_h - (y - oy) * scale

        # 线宽随缩放自适应（固定 2px 在放大的地图上细到看不见）：
        # roadmap 参考线最细，A* 路径最粗最显眼，点/三角同步放大
        lw_road = max(1.5, scale * 0.035)
        lw_patrol = max(2.0, scale * 0.05)
        lw_path = max(3.5, scale * 0.09)
        pt_r = max(5.0, scale * 0.13)

        def _line_width(label: str) -> float:
            low = label.lower()
            if "path" in low:
                return lw_path
            if "patrol" in low:
                return lw_patrol
            return lw_road                 # prm / roadmap / 其他

        # 控制器叠加层：骨架/roadmap 半透明做背景参考，A* 路径半透明压在
        # 骨架之上——透出下层线条即可直观判断 A* 是否贴着骨架选路
        for label, layer in _ordered_layers(self._overlays):
            color = _overlay_color(label)
            draw_color = QColor(color)
            draw_color.setAlpha(_layer_alpha(label))
            p.setPen(QPen(draw_color, _line_width(label)))
            for x1, y1, x2, y2 in layer.get("segments", []):
                p.drawLine(wx(x1), wy(y1), wx(x2), wy(y2))
            p.setPen(QPen(color, max(1.0, lw_road * 0.6)))
            p.setBrush(color)
            for cx, cy in layer.get("points", []):
                p.drawEllipse(QRectF(wx(cx) - pt_r, wy(cy) - pt_r,
                                     2 * pt_r, 2 * pt_r))
            p.setBrush(Qt.BrushStyle.NoBrush)

        # 机器人：车头朝向三角（同控制器渲染的青三角，尺寸随缩放）
        tri_tip, tri_base = 0.16 * max(1.0, scale / 100.0), \
            0.075 * max(1.0, scale / 100.0)
        p.setPen(QPen(_C_ROBOT, max(2.0, scale * 0.04)))
        p.setBrush(_C_ROBOT)
        for pose in self._poses.values():
            if len(pose) < 3:
                continue
            x, y, yaw = pose
            tip = (x + tri_tip * np.cos(yaw), y + tri_tip * np.sin(yaw))
            left = (x + tri_base * np.cos(yaw + 2.6),
                    y + tri_base * np.sin(yaw + 2.6))
            right = (x + tri_base * np.cos(yaw - 2.6),
                     y + tri_base * np.sin(yaw - 2.6))
            poly = [QPointF(wx(tip[0]), wy(tip[1])),
                    QPointF(wx(left[0]), wy(left[1])),
                    QPointF(wx(right[0]), wy(right[1]))]
            p.drawPolygon(poly)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # 图例：地图下方横向条带（不遮挡地图内容）
        if legend_items:
            p.setPen(QColor(theme.C_TEXT))
            p.setFont(self.font())
            fm = p.fontMetrics()
            lx = x0
            ly = y0 + img_h + 7
            for label, color in legend_items:
                p.fillRect(QRectF(lx, ly + 2, 9, 9), color)
                p.drawText(QPointF(lx + 13, ly + 11), label)
                lx += 13 + fm.horizontalAdvance(label) + 16
