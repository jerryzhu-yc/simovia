"""3D 点云视图面板：黑底 + 彩虹高度着色的透视点云（RViz 风格）。

数据来源：主窗口 10Hz 轮询 MCP `lidar_map` 的 pc 段（内核 PointCloudMapper
的体素点，[x,y,z,r,g,b]）。本面板按体素坐标去重累积成一张逐渐变密的
3D 点云，用少量 numpy 做透视投影、直接把点画进 QImage——不引入任何
独立渲染引擎（与 RadarMapPanel 同款 Qt 光栅绘制思路）。

交互：拖拽左键旋转（方位/俯仰）、滚轮缩放、双击复位视角。
配色：按高度映射彩虹色带（低处红/黄 -> 高处蓝），与参考的点云查看器
观感一致；点阵按远近排序（远的先画、近的覆盖），每点 2x2 像素块，
大视场（80°）机位落在房间内部，呈现"站在屋里看整间房"的沉浸感。
"""
from __future__ import annotations

import base64
import collections
import math
import zlib

import numpy as np

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from simovia.gui import theme

# 体素去重量化步长（米）：与内核 PointCloudMapper.VOXEL 一致
_VOX = 0.04
_MAX_POINTS = 200000          # 累积上限（内存/绘制保护）
_NEAR = 0.10                  # 相机近裁剪（米）
_FOV = math.radians(80.0)     # 垂直视场（大视场才有"站在屋里"的沉浸感）
_ZLO, _ZHI = 0.0, 1.80        # 高度着色区间（低红 -> 高蓝，覆盖 2.4m 墙身）

# 默认机位（自动取景失败/空场景时用）：房间中心略下方 + 南侧近墙、微俯视
_HOME_TARGET = (0.0, 0.0, 1.0)
_HOME_YAW, _HOME_PITCH, _HOME_DIST = -90.0, 15.0, 2.0


def _jet(t: np.ndarray) -> np.ndarray:
    """t∈[0,1] -> RGB∈[0,1] 的彩虹色带（蓝->青->绿->黄->红）。

    支持标量与数组：标量返回 (3,)，数组 (N,) 返回 (N,3)。
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack((r, g, b), axis=-1)


class PointCloudView(QWidget):
    """累积式 3D 点云视图（黑底彩虹点阵，可旋转缩放）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 220)
        self.setMouseTracking(True)
        self._vox: "collections.OrderedDict[tuple[int,int,int], tuple[float,float,float]]" = \
            collections.OrderedDict()
        self._xyz = np.zeros((0, 3), dtype=np.float64)   # 懒惰重建的坐标缓存
        self._dirty = True
        # 视角：绕 target 的方位角/俯仰角/距离（度、度、米）
        self._target = np.array(_HOME_TARGET)
        self._yaw, self._pitch, self._dist = _HOME_YAW, _HOME_PITCH, _HOME_DIST
        self._home = (self._target.copy(), self._yaw, self._pitch, self._dist)
        self._fitted = False          # 首帧数据自动取景一次
        self._last = None

    # ------------------------------------------------------------ 数据入口
    def update_pc(self, pc: dict | None) -> None:
        """吃 lidar_map 的 pc 段（None = 清空）。"""
        if not pc:
            return
        blob = pc.get("points")
        if not blob:
            return
        try:
            arr = np.frombuffer(
                zlib.decompress(base64.b64decode(blob)),
                dtype=np.float32).reshape(-1, 6)
        except Exception:
            return                      # 撕裂帧/坏数据：丢弃等下一拍
        added = False
        for x, y, z, _r, _g, _b in arr:
            # 体素中心 = (idx+0.5)*VOX，floor 即可精确还原 idx（勿用 round，
            # 银行家舍入会让相邻体素碰撞、点数腰斩）
            k = (int(math.floor(float(x) / _VOX)),
                 int(math.floor(float(y) / _VOX)),
                 int(math.floor(float(z) / _VOX)))
            if k in self._vox:
                continue
            if len(self._vox) >= _MAX_POINTS:
                break
            self._vox[k] = (float(x), float(y), float(z))
            added = True
        if added:
            self._dirty = True
        # 首批点足够多时自动取景一次（对准房间中心、按包围盒定距离）
        if not self._fitted and len(self._vox) >= 300:
            self._autofit()
            self._fitted = True
        self.update()

    def _autofit(self) -> None:
        """按点云包围盒取一个"站在房间南侧往里看"的机位。

        orbit 相机 eye = target + dist*(yaw,pitch)，要让人有置身室内的感觉，
        eye 必须落在包围盒内部（老版本按对角线定距离，eye 会退到房间外，
        画面变成"从外面看一个浅盘"）。这里直接由期望的 eye 反算 yaw/pitch/
        dist：eye 站在南缘内侧 70% 半长处、离地约 0.7 倍点云高度（夹在
        1.2~1.7m），视线朝北、微俯视房间中心。
        """
        xyz = np.array(list(self._vox.values()), dtype=np.float64)
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        size = hi - lo
        cx, cy, cz = (lo + hi) * 0.5
        eye = np.array([
            cx,
            cy - 0.70 * float(size[1]) * 0.5,
            float(np.clip(0.70 * hi[2], 1.20, 1.70)),
        ])
        target = np.array([cx, cy, cz])
        d = eye - target
        dist = max(float(np.linalg.norm(d)), 0.3)
        self._target = target
        self._dist = float(np.clip(dist, 0.3, 12.0))
        self._pitch = float(np.degrees(np.arcsin(np.clip(d[2] / dist, -1, 1))))
        self._yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        self._home = (self._target.copy(), self._yaw, self._pitch, self._dist)

    def reset(self) -> None:
        """清空累积点云并复位视角（新世界加载时调用）。"""
        self._vox.clear()
        self._xyz = np.zeros((0, 3), dtype=np.float64)
        self._dirty = True
        self._target = np.array(_HOME_TARGET)
        self._yaw, self._pitch, self._dist = _HOME_YAW, _HOME_PITCH, _HOME_DIST
        self._home = (self._target.copy(), self._yaw, self._pitch, self._dist)
        self._fitted = False
        self.update()

    def count(self) -> int:
        return len(self._vox)

    # ------------------------------------------------------------ 投影渲染
    def _rebuild(self) -> None:
        if self._vox:
            self._xyz = np.array(list(self._vox.values()), dtype=np.float64)
        else:
            self._xyz = np.zeros((0, 3), dtype=np.float64)
        self._dirty = False

    def _camera(self):
        """返回 (eye, right, up, forward) 正交基。"""
        yaw, pitch = math.radians(self._yaw), math.radians(self._pitch)
        d = np.array([math.cos(pitch) * math.cos(yaw),
                      math.cos(pitch) * math.sin(yaw),
                      math.sin(pitch)])
        eye = self._target + self._dist * d
        fwd = -d                                   # 从 eye 看向 target
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def _project(self, w: int, h: int):
        """把全部点投到 w×h 像素坐标；返回 (px, py, depth) 供绘制。"""
        eye, right, up, fwd = self._camera()
        rel = self._xyz - eye                      # (N,3)
        cam_x = rel @ right
        cam_y = rel @ up
        cam_z = rel @ fwd
        fl = (h * 0.5) / math.tan(_FOV * 0.5)      # 焦距（像素）
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = w * 0.5 + fl * cam_x / cam_z
            sy = h * 0.5 - fl * cam_y / cam_z
        return sx, sy, cam_z

    def _render_image(self, w: int, h: int) -> QImage | None:
        """渲染到 RGB 缓冲并返回 QImage（1 点 = 1 像素，RViz 式细点阵）。"""
        buf = np.zeros((h, w, 3), dtype=np.uint8)
        if self._xyz.shape[0]:
            sx, sy, cz = self._project(w, h)
            ok = (cz > _NEAR) & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            if ok.any():
                px = sx[ok].astype(np.int32)
                py = sy[ok].astype(np.int32)
                zz = self._xyz[ok, 2]
                t = np.clip((zz - _ZLO) / (_ZHI - _ZLO), 0.0, 1.0)
                # 低处红、高处蓝：用反相 jet（与参考点云一致）
                rgb = (_jet(1.0 - t) * 255.0).astype(np.uint8)
                # 远点先画、近点覆盖：按相机深度从大到小排序
                order = np.argsort(-cz[ok])
                px, py, cols = px[order], py[order], rgb[order]
                # 2x2 像素块：RViz 式颗料感（1px 在大面板上稀到发灰）
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    buf[np.minimum(py + dy, h - 1),
                        np.minimum(px + dx, w - 1)] = cols
        img = QImage(buf.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return img.copy()                          # 拷出，脱离 buf 生命周期

    def render_qimage(self, w: int, h: int) -> QImage:
        """离屏渲染一帧完整点云画面（黑底 + 点阵 + HUD）到 QImage。

        供无头场景使用（演示视频合成、回归测试）：不依赖窗口系统的
        paintEvent/render 管线——offscreen 平台下长 episode 中 QWidget
        的 render()/grab() 会零星串成兄弟部件的画面（实测），直接把
        paintEvent 的绘制步骤画到自建 QImage 上最稳。尺寸由调用方给定
        （与部件实际尺寸一致时画面与 GUI 完全相同）。
        """
        img = QImage(w, h, QImage.Format.Format_RGB888)
        img.fill(Qt.GlobalColor.black)
        if self._dirty:
            self._rebuild()
        p = QPainter(img)
        try:
            if self._xyz.shape[0]:
                inner = self._render_image(w, h)
                if inner is not None:
                    p.drawImage(QRectF(0, 0, w, h), inner)
            self._draw_hud(p, w, h)
        finally:
            p.end()
        return img.copy()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        try:
            if self._dirty:
                self._rebuild()
            if self._xyz.shape[0] == 0:
                p.setPen(QPen(theme.C_TEXT_DIM))
                p.drawText(self.rect(), 0x84, "等待 3D 点云数据…")  # AlignCenter
                return
            w, h = self.width(), self.height()
            img = self._render_image(w, h)
            if img is not None:
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                p.drawImage(QRectF(0, 0, w, h), img)
            self._draw_hud(p, w, h)
        finally:
            p.end()

    def _draw_hud(self, p: QPainter, w: int, h: int) -> None:
        """左下操作提示 + 右侧高度色带图例。"""
        p.setPen(QPen(theme.C_TEXT_DIM))
        p.drawText(10, h - 10,
                   f"{self.count()} 点 · 拖拽旋转 · 滚轮缩放 · 双击复位")
        # 高度色带（12px 宽 x 0.5h 高），低红底 / 高蓝顶
        bar_h = min(h - 40, 160)
        bar_w = 12
        x = w - bar_w - 14
        y0 = 20
        steps = 48
        for i in range(steps):
            t = i / (steps - 1)                      # 0=底(低) 1=顶(高)
            r, g, b = (int(v * 255) for v in _jet(1.0 - t))
            p.fillRect(QRectF(x, y0 + bar_h * (1 - t) - bar_h / steps,
                              bar_w, bar_h / steps + 1),
                       QColor(r, g, b))
        p.setPen(QPen(theme.C_TEXT_DIM))
        p.drawText(x - 46, y0 + 8, f"{_ZHI:.1f}m")
        p.drawText(x - 46, y0 + bar_h, f"{_ZLO:.1f}m")

    # ------------------------------------------------------------ 交互
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._last = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._last is not None:
            d = e.position() - self._last
            self._last = e.position()
            self._yaw += d.x() * 0.4
            self._pitch = max(3.0, min(85.0, self._pitch - d.y() * 0.3))
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        self._last = None
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e) -> None:
        notch = e.angleDelta().y() / 120.0
        self._dist = max(0.4, min(30.0, self._dist * (0.9 ** notch)))
        self.update()
        super().wheelEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._target = self._home[0].copy()
            self._yaw, self._pitch, self._dist = self._home[1:]
            self.update()
        super().mouseDoubleClickEvent(e)
