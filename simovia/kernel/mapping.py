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

import collections
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


# ---------------------------------------------------------------- 3D 点云建图

class PointCloudMapper:
    """单机器人的 3D 点云建图（channels>1 多线雷达）。

    双层结构：
    - 全量真相 = 体素 dict {(ix,iy,iz): log-odds}（VOXEL 0.04m 立方格，
      只存占据——命中点世界坐标离散化；自由体素不存，内存可控）
    - 俯视 2D 占据栅格（与 LidarMapper 同网格几何/接口）：命中点的 z
      落在 BAND 高度带内才投影——带内语义 = "扫地机高度带内有无遮挡"：
      地面（z<0.02）投影成占据没意义；桌面（0.72）带外不挡扫地机；
      悬空横杆（0.105~0.195）带内必投影——2D 雷达扫不到的障碍由此进图
    - 自由段：射线在高度带内穿过的部分按俯视 Bresenham 采样打自由
      （采样步 RES*0.9 向量化，斜线允许漏格对角——粗略自由可接受）

    可视化槽位（bind_slots/sync_slots）：预置 pcl_slot geom 的全局镜像
    ——体素数 ≤ 槽位数时全显示，超出则按身体素均匀抽样（_visible_keys），
    保证整间房（含桌下地面与桌面底两层）都在视野里，而不是只剩机器人
    脚下的局部窗口；抽样后的体素写进槽位 geom_pos/size/rgba（高度色带）。
    槽位挂在静态 body 上：geom_xpos = geom_pos 恒成立，直接写 data.geom_xpos
    绕过 pipeline，不碰 warmstart（bit 级轨迹不受影响）。槽位由内核离屏
    渲染流（screenshot/camera stream）消费；GUI 的立体点云展示归独立的
    3D 点云面板（gui/pointcloud_panel），直接吃 MCP pc 段。
    """

    VOXEL = 0.04                     # 体素边长（米）
    BAND = (0.02, 0.18)               # 俯视投影高度带（扫地机可撞带）
    MAX_VOXELS = 60000                # 体素上限（演示级内存保护）
    # 槽位渲染半尺寸：体素的 0.4 倍（体素间留 20% 缝）——密集成片时观感
    # 是"点云"而不是"积木方块"，真实 3D 雷达建图（RViz 风格）的颗粒感
    RENDER_HALF = VOXEL * 0.4
    _HUE_STOPS = ((0.0, (0.25, 0.45, 1.0)), (0.4, (0.2, 0.9, 0.4)),
                  (1.0, (1.0, 0.28, 0.2)))   # z 0~0.8m 高度色带蓝->绿->红

    def __init__(self, floor_size: tuple[float, float],
                 band: tuple[float, float] = BAND, voxel: float = VOXEL):
        hx, hy = float(floor_size[0]), float(floor_size[1])
        self.ox, self.oy = -(hx + MARGIN), -(hy + MARGIN)
        self.nx = int((2.0 * hx + 2.0 * MARGIN) / RES)
        self.ny = int((2.0 * hy + 2.0 * MARGIN) / RES)
        self.l = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.band = (float(band[0]), float(band[1]))
        self.voxel = float(voxel)
        self.vox: "collections.OrderedDict[tuple[int, int, int], float]" = \
            collections.OrderedDict()
        self.seq = 0                    # 已喂帧计数（MCP pc 段带回）
        self._model = None             # bind_slots 后有：槽位 geom 镜像
        self._gids: list[int] = []

    # ------------------------------------------------------------ 建图

    def _vkey(self, p: np.ndarray) -> tuple[int, int, int]:
        return (int(math.floor(p[0] / self.voxel)),
                int(math.floor(p[1] / self.voxel)),
                int(math.floor(p[2] / self.voxel)))

    def update(self, origin: np.ndarray, rot: np.ndarray,
               dirs: np.ndarray, dists, max_range: float) -> None:
        """吃 3D 雷达一帧。

        origin/rot = 雷达挂点世界原点与旋转（task.lidar_frame 给出）；
        dirs = body 系单位方向表（builder.make_dirs，channel-major 与 dists
        平铺顺序一致）；dists = lidar_scan 读数（-1 跳过）。向量化实现：
        480 束/帧纯 Python 循环会到 10ms 级，numpy 批量 + dict 循环仅新体素。
        """
        self.seq += 1
        dists = np.asarray(dists, dtype=float)
        hit = dists > 0.0
        if not hit.any():
            return
        d = np.minimum(dists[hit], max_range)           # 截断超程回波
        wdirs = (rot @ dirs[hit].T).T                   # 世界系方向 (M,3)
        pts = origin[None, :] + wdirs * d[:, None]      # 命中点 (M,3)

        # ---- 体素占据（全部命中：带内外都进点云，展示完整结构）
        for p in pts:
            k = self._vkey(p)
            v = self.vox.get(k, 0.0)
            if v == 0.0 and len(self.vox) >= self.MAX_VOXELS:
                continue                               # 上限保护：不加新体素
            self.vox[k] = min(v + L_OCC, L_CLAMP)
            if v != 0.0:
                self.vox.move_to_end(k)                # LRU：活跃命中靠尾

        # ---- 俯视 2D 投影
        zlo, zhi = self.band
        oz, oxy = float(origin[2]), origin[:2]
        z_hit = pts[:, 2]
        in_band = (z_hit >= zlo) & (z_hit <= zhi)

        # 带内命中 -> 占据。命中格记下来，自由段更新时剔除——
        # 2D Bresenham "最后格只打 OCC" 语义的向量化等价（否则同帧
        # 自由稀释 0.9-0.7=0.2，命中格翻回未知）
        dz = wdirs[:, 2]
        hit_ix = hit_iy = None
        ib = in_band.nonzero()[0]
        if ib.size:
            hix = np.floor((pts[ib, 0] - self.ox) / RES).astype(np.int64)
            hiy = np.floor((pts[ib, 1] - self.oy) / RES).astype(np.int64)
            ok = (hix >= 0) & (hix < self.nx) & (hiy >= 0) & (hiy < self.ny)
            hit_ix, hit_iy = hix[ok], hiy[ok]
            np.add.at(self.l, (hit_ix, hit_iy), L_OCC)

        # 带内穿越 -> 自由（向量化粗采样，步长 RES*0.9，斜线漏格可接受）。
        # 自由段 = 射线 z(t) = oz + dz*t 的带内 t 区间 [t_lo, t_hi] 与
        # [0, 命中距离] 的交：挂点带内则 t_lo=0（扫地机雷达 z≈0.12 常态）；
        # 挂点带外须向上/下进入带才有区间（水平背离带 -> 无自由段）；
        # 带外命中前的带内段仍打自由，采样半开 [t_lo, t_hi) 不越过命中点。
        # 带外无回波（-1）保守不更新——与 2D 建图同款约定。
        flat = np.abs(dz) < 1e-9
        safe = np.where(flat, 1.0, dz)
        ta = (zlo - oz) / safe
        tb = (zhi - oz) / safe
        t_lo = np.minimum(ta, tb)
        t_hi = np.maximum(ta, tb)
        inband0 = (oz >= zlo) & (oz <= zhi)
        t_lo = np.where(flat, np.where(inband0, 0.0, np.inf), t_lo)
        t_hi = np.where(flat, np.where(inband0, np.inf, -np.inf), t_hi)
        t_lo = np.maximum(t_lo, 0.0)
        t_hi = np.minimum(t_hi, d)
        seg = np.where(t_hi > t_lo, t_hi - t_lo, 0.0)
        counts = np.ceil(seg / (RES * 0.9)).astype(np.int64)
        counts = np.where(seg > 0, counts, 0)
        if counts.sum() <= 0:
            self.l[:] = np.clip(self.l, -L_CLAMP, L_CLAMP)
            return
        # 展平所有束的采样点：t = t_lo[b] + 局部序号 * RES*0.9 < t_hi[b]
        total = int(counts.sum())
        bidx = np.repeat(np.arange(counts.size), counts)
        base = np.repeat(np.cumsum(counts) - counts, counts)
        offs = t_lo[bidx] + (np.arange(total) - base) * (RES * 0.9)
        sample_x = oxy[0] + wdirs[bidx, 0] * offs
        sample_y = oxy[1] + wdirs[bidx, 1] * offs
        ix = np.floor((sample_x - self.ox) / RES).astype(np.int64)
        iy = np.floor((sample_y - self.oy) / RES).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ix, iy = ix[ok], iy[ok]
        if hit_ix is not None and hit_ix.size:
            # 命中格只打 OCC：一维格 key 编码后 isin 剔除
            keep = ~np.isin(ix * self.ny + iy, hit_ix * self.ny + hit_iy)
            ix, iy = ix[keep], iy[keep]
        np.add.at(self.l, (ix, iy), L_FREE)
        self.l[:] = np.clip(self.l, -L_CLAMP, L_CLAMP)

    # ------------------------------------------------------------ 2D 兼容接口

    def grid(self) -> np.ndarray:
        """三态栅格 uint8 (nx, ny)：0 未知 1 自由 2 占据（与 LidarMapper 同语义）。"""
        out = np.zeros((self.nx, self.ny), dtype=np.uint8)
        out[self.l < FREE_TH] = 1
        out[self.l > OCC_TH] = 2
        return out

    def prob(self) -> np.ndarray:
        """连续占据概率 uint8 (nx, ny)（与 LidarMapper 同语义）。"""
        p = 1.0 / (1.0 + np.exp(-self.l))
        return np.clip(p * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def meta(self) -> dict:
        return {"res": RES, "ox": self.ox, "oy": self.oy,
                "nx": self.nx, "ny": self.ny}

    # ------------------------------------------------------------ 点云与槽位

    def _visible_keys(self, limit: int) -> list[tuple[int, int, int]]:
        """全量体素的均匀抽样（≤ limit 个），供 pointcloud / sync_slots 共用。

        体素数 ≤ limit 时全量返回；超出则按步长 stride=ceil(n/limit) 抽样。
        与"只取最近命中"不同，抽样覆盖整个房间的建图结果——桌下地面与
        桌面底两层同时在视野内，才能读出桌子净空高度（用户核心需求）。
        """
        keys = list(self.vox.keys())
        n = len(keys)
        if n <= limit:
            return keys
        step = int(math.ceil(n / float(limit)))
        return keys[::step][:limit]

    def pointcloud(self, limit: int = 1200) -> list[list[float]]:
        """全量体素均匀抽样的中心 [[x,y,z]...]（与槽位同序）。"""
        out = []
        for k in self._visible_keys(limit):
            out.append([(k[0] + 0.5) * self.voxel,
                        (k[1] + 0.5) * self.voxel,
                        (k[2] + 0.5) * self.voxel])
        return out

    def bind_slots(self, model, gids: list[int]) -> None:
        """绑定预置槽位 geom id 序（builder index.pointcloud["geoms"]）。"""
        self._model = model
        self._gids = list(gids)

    def _height_rgb(self, z: float) -> tuple[float, float, float]:
        """z -> 高度色带（蓝->绿->红，0~0.8m 分段插值）。"""
        stops = self._HUE_STOPS
        t = max(0.0, min(1.0, z / 0.8))
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return (c0[0] + (c1[0] - c0[0]) * f,
                        c0[1] + (c1[1] - c0[1]) * f,
                        c0[2] + (c1[2] - c0[2]) * f)
        return stops[-1][1]

    def sync_slots(self, data) -> None:
        """把全局抽样后的体素写进槽位 geom（pos/size/rgba + data.geom_xpos）。

        槽位挂静态 body：geom_xpos == geom_pos 恒成立，直接写不绕 pipeline、
        不碰 warmstart；多余槽位缩成零体积隐藏。槽位供内核离屏渲染流
        （screenshot/camera stream）直接出图；pointcloud() 打包的 pc 段归
        GUI 3D 点云面板消费，故两者共用 _visible_keys 保持同一抽样序。
        """
        if not self._gids:
            return
        keys = self._visible_keys(len(self._gids))
        n = len(keys)
        half = self.RENDER_HALF
        for i, gid in enumerate(self._gids):
            if i >= n:
                self._model.geom_size[gid] = [0.001] * 3
                self._model.geom_rgba[gid] = [1.0, 1.0, 1.0, 0.0]
                data.geom_xpos[gid] = [0.0, 0.0, -1.0]   # 藏到地下
                continue
            k = keys[i]                            # 全局抽样序：与 pointcloud 同序
            cx = (k[0] + 0.5) * self.voxel
            cy = (k[1] + 0.5) * self.voxel
            cz = (k[2] + 0.5) * self.voxel
            r, g, b = self._height_rgb(cz)
            self._model.geom_pos[gid] = [cx, cy, cz]
            self._model.geom_size[gid] = [half] * 3
            self._model.geom_rgba[gid] = [r, g, b, 0.85]
            data.geom_xpos[gid] = [cx, cy, cz]    # 静态 body：恒等关系
