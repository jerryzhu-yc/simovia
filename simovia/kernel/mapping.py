"""内核侧雷达建图：log-odds 占据栅格，GUI"雷达建图"面板按需拉取。

设计原则：
- 平台级通用能力：任何机器人声明 lidar 传感器，内核就在推进时自动
  喂图（每控制 tick 一帧），与控制器实现无关——"以后很多种雷达"
  都能直接在 GUI 里看到建图效果
- 保守建图：-1（无回波/超量程）不更新——真实 SLAM 不拿 inf 画自由，
  量程外区域保持未知灰，地图干净（控制器自己的导航建图可自行选择
  更激进的策略）
- 写入只发生在内核推进线程（_advance），读（grid()）在 MCP 协议线程
  ——numpy 整格读写撕裂最坏是某格瞬时新旧值，渲染无害
"""
from __future__ import annotations

import math

import numpy as np

RES = 0.05                      # 栅格分辨率（米/格）
L_OCC, L_FREE = 0.9, -0.7       # log-odds 更新步长：命中 / 穿越
L_CLAMP = 5.0                    # log-odds 上下限
OCC_TH, FREE_TH = 0.5, -0.5     # 占据 / 自由判读阈值
MARGIN = 0.5                     # 栅格在地缘外多铺的边（米）


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """栅格 a -> b 的穿越格序列（含两端，经典整数 Bresenham）。"""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx + dy
    out: list[tuple[int, int]] = []
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return out
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class LidarMapper:
    """单机器人的 2D 占据栅格（该机器人所有雷达融合进同一张图）。

    世界范围按 scene.floor.size 推：±(半尺寸 + MARGIN)，足够盖住外墙
    与量程内的所有静态障碍。多机器人各持一张独立图，GUI 叠加渲染。
    """

    def __init__(self, floor_size: tuple[float, float]):
        hx, hy = float(floor_size[0]), float(floor_size[1])
        self.ox, self.oy = -(hx + MARGIN), -(hy + MARGIN)
        self.nx = int((2.0 * hx + 2.0 * MARGIN) / RES)
        self.ny = int((2.0 * hy + 2.0 * MARGIN) / RES)
        self.l = np.zeros((self.nx, self.ny), dtype=np.float32)

    def _w2c(self, x: float, y: float) -> tuple[int, int] | None:
        ix, iy = int((x - self.ox) / RES), int((y - self.oy) / RES)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None

    def update(self, px: float, py: float, yaw: float,
               angles, dists, max_range: float) -> None:
        """吃雷达一帧：每束 Bresenham raycasting（穿越打自由、命中打占据）。

        px/py/yaw = 雷达挂点世界坐标与航向（task.lidar_pose 给出）；
        angles/dists = 传感器角度表与 lidar_scan 读数（-1 跳过）。
        max_range 仅用于截断超程回波（防端点画到量程外）。
        """
        for a, d in zip(angles, dists):
            if d < 0.0:
                continue                      # 无回波：不更新（保守建图）
            dist = min(float(d), max_range)
            wa = yaw + float(a)
            c0 = self._w2c(px, py)
            c1 = self._w2c(px + math.cos(wa) * dist,
                           py + math.sin(wa) * dist)
            if c0 is None or c1 is None:
                continue
            cells = _bresenham(c0, c1)
            last = len(cells) - 1
            for i, c in enumerate(cells):
                if i == last:
                    self.l[c] = min(self.l[c] + L_OCC, L_CLAMP)
                else:
                    self.l[c] = max(self.l[c] + L_FREE, -L_CLAMP)

    def grid(self) -> np.ndarray:
        """三态栅格 uint8 (nx, ny)：0 未知 1 自由 2 占据（MCP 直接下发）。"""
        out = np.zeros((self.nx, self.ny), dtype=np.uint8)
        free = self.l < FREE_TH
        occ = self.l > OCC_TH
        out[free] = 1
        out[occ] = 2          # 命中优先于自由（重叠时占据更保守）
        return out

    def prob(self) -> np.ndarray:
        """连续占据概率 uint8 (nx, ny)：0=确定自由 .. 128=未知 .. 255=确定占据。

        GUI 拿它做灰度渐变渲染（而不是三态硬阈值的大块锯齿）。量化到
        uint8 是为了传输体积不变——zlib 后与三态图同量级。
        """
        p = 1.0 / (1.0 + np.exp(-self.l))          # log-odds -> 概率
        return np.clip(p * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def meta(self) -> dict:
        return {"res": RES, "ox": self.ox, "oy": self.oy,
                "nx": self.nx, "ny": self.ny}
