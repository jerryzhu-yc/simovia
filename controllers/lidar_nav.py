"""lidar_nav 控制器：绕圈建图 -> 维诺骨架图 -> A* 搜路 -> 纯跟踪导航。

配合 worlds/lidar_nav.yaml（5.2m x 4.2m 房间、五处静态障碍、差速小车 +
车顶 60 束 2D 激光雷达）。三个阶段：

1. EXPLORE（建图）：沿房间外圈巡逻一圈回到起点。每个控制 tick 用雷达
   全束读数做栅格 raycasting——Bresenham 穿越格打 log-odds 自由更新，
   命中格打占据更新；-1 = 无回波，按最大量程整段记自由。
2. PLAN（规划）：对"可通行区域"（自由格减去障碍膨胀安全带）做 Zhang-Suen
   细化，得到 1px 中轴骨架——即可通行空间的广义维诺图（障碍间通道正中线）；
   骨架像素按 8 邻接成图，用经典 A* 在骨架上搜 起点->goal 最短路。
3. NAVIGATE（导航）：纯跟踪（pure pursuit）跟踪 A* 路径到目标点，
   到点停车、report_result(True) 后退出。

占据栅格的实时可视化由平台"雷达建图"面板负责（内核自动累积、GUI 轮询
渲染）；本控制器经 publish_overlay 把目标点 / 巡逻线 / 维诺骨架 / A* 路径 /
起点作为矢量叠加层发上去，与地图一起在面板上显示。
"""
import heapq
import math

import numpy as np

from simovia.controller import Robot

# ---------------------------------------------------------------- 地图参数
RES = 0.05                      # 栅格分辨率（米/格）
GX0, GX1 = -2.75, 2.75          # 栅格覆盖的世界 x 范围
GY0, GY1 = -2.25, 2.25          # 栅格覆盖的世界 y 范围
NX, NY = int((GX1 - GX0) / RES), int((GY1 - GY0) / RES)
L_OCC, L_FREE = 0.9, -0.7       # log-odds 更新步长：命中 / 穿越
L_CLAMP = 5.0                    # log-odds 上下限
OCC_TH, FREE_TH = 0.5, -0.5      # 占据 / 自由判读阈值

# ---------------------------------------------------------------- 车辆参数
WHEEL_R = 0.032                 # 驱动轮半径（m）
HALF_TRACK = 0.06               # 半轮距（m）：wheel_l/wheel_r 距中轴 0.06
V_CRUISE = 0.22                 # 巡航线速度（m/s；ctrl 上限 10rad/s ≈ 0.32m/s）
LOOKAHEAD = 0.22                # 纯跟踪前瞻距离（m）：0.3 时切角过大，会钻进
                                # 骨架安全带的边缘撞障碍
EXPLORE_TIMEOUT = 9000          # 巡逻兜底 tick：超时直接转入规划（防卡死）

# 巡逻路径：沿房间外圈一圈（西南角起，逆时针，回到起点收图）
PATROL = [(-2.1, -1.6), (-2.1, 1.6), (2.1, 1.6), (2.1, -1.6), (-2.1, -1.6)]

# ---------------------------------------------------------------- 规划参数
INFLATE = 0.26                  # 障碍膨胀半径（m）：车半长 0.09 + 纯跟踪切角
                                # 余量。0.16 时骨架贴柱太近，切角后车顶住圆柱
PLAN_INSET = 0.10               # 规划范围 = 巡逻矩形内缩 0.10m。巡逻框外那圈
                                # 墙边（约 0.5m 宽）虽被雷达扫成白（自由），但
                                # 不属于本次规划范围；不裁剪时维诺骨架会顺着它
                                # 凸出巡逻线（顶部约 0.14m），看起来"越界"


# ---------------------------------------------------------------- 基础工具

def w2c(x, y):
    """世界坐标 -> 栅格索引（越界返回 None）。"""
    ix, iy = int((x - GX0) / RES), int((y - GY0) / RES)
    return (ix, iy) if 0 <= ix < NX and 0 <= iy < NY else None


def c2w(ix, iy):
    """栅格索引 -> 格中心世界坐标。"""
    return GX0 + (ix + 0.5) * RES, GY0 + (iy + 0.5) * RES


def bresenham(a, b):
    """栅格 a -> b 的穿越格序列（含两端，经典整数 Bresenham）。"""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx + dy
    out = []
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


def wrap(a):
    """角域归一到 (-pi, pi]。"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def dilate(mask, r):
    """圆形结构元膨胀：障碍外扩一圈"安全带"（无 scipy 依赖）。"""
    nx, ny = mask.shape
    pad = np.zeros((nx + 2 * r, ny + 2 * r), dtype=bool)
    pad[r:r + nx, r:r + ny] = mask
    out = np.zeros_like(mask)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            out |= pad[r + dx:r + dx + nx, r + dy:r + dy + ny]
    return out


def erode(mask, r=1):
    """方形结构元腐蚀：r 邻域内必须全可通行才保留。

    用来剪掉贴未观测边界/墙缝的细条假自由——否则骨架会沿这些细条
    钻出房间，A* 可能选一条实际不存在的通道。
    """
    p = np.pad(mask, r)
    nx, ny = mask.shape
    out = mask.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out &= p[r + dx:r + dx + nx, r + dy:r + dy + ny]
    return out


# ---------------------------------------------------------------- 占据栅格

class GridMap:
    """log-odds 占据栅格：每 tick 吃雷达一帧，规划前一次性分类。"""

    def __init__(self):
        self.l = np.zeros((NX, NY))

    def update(self, px, py, yaw, dists, angles):
        for d, a in zip(dists, angles):
            if d < 0:
                continue                       # 无回波：保守建图不更新。
                # 若按最大量程记自由，墙外/未探索区会成为假自由，骨架会
                # 长到房间里侧去，A* 可能选一条穿墙的假路径
            dist = float(d)
            wa = yaw + a
            ex = px + math.cos(wa) * dist
            ey = py + math.sin(wa) * dist
            c0, c1 = w2c(px, py), w2c(ex, ey)
            if c0 is None or c1 is None:
                continue
            cells = bresenham(c0, c1)
            last = len(cells) - 1
            for i, c in enumerate(cells):
                if i == last:
                    self.l[c] = min(self.l[c] + L_OCC, L_CLAMP)
                else:
                    self.l[c] = max(self.l[c] + L_FREE, -L_CLAMP)

    def classify(self):
        """(occupied, free) 两个 bool 栅格。"""
        return self.l > OCC_TH, self.l < FREE_TH


# ---------------------------------------------------------------- 维诺骨架图

def skeletonize(mask):
    """Zhang-Suen 细化：可通行区域 -> 1px 中轴骨架（纯 numpy，无 scipy）。

    可通行区域的骨架 == 广义维诺图（到最近障碍等距的点集），即障碍之间
    通道的正中线。两侧交替剥离边界像素直到形状不再变化。
    """
    pad = np.pad(mask.astype(np.uint8), 1)
    sk = pad.copy()
    offs = ((-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1))
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(sk, dy, 0), dx, 1) for dy, dx in offs]
            b = sum(p)                                  # P2..P9 前景邻居数
            a = sum(((p[i] == 0) & (p[(i + 1) % 8] == 1))
                    for i in range(8))                  # 顺时针 0->1 跳变数
            if step == 0:
                kill = ((sk == 1) & (b >= 2) & (b <= 6) & (a == 1)
                        & ((p[0] * p[2] * p[4]) == 0)
                        & ((p[2] * p[4] * p[6]) == 0))
            else:
                kill = ((sk == 1) & (b >= 2) & (b <= 6) & (a == 1)
                        & ((p[0] * p[2] * p[6]) == 0)
                        & ((p[0] * p[4] * p[6]) == 0))
            if kill.any():
                sk[kill] = 0
                changed = True
        if not changed:
            return sk[1:-1, 1:-1].astype(bool)


def segment_clear(p, q, blocked):
    """p->q 线段无碰撞：沿线每 RES/2 采点查膨胀障碍图（含 INFLATE 余量）。"""
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    n = max(2, int(d / (RES / 2)))
    for i in range(n + 1):
        t = i / n
        c = w2c(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
        if c is None or blocked[c]:
            return False
    return True


def build_skeleton_graph(ok, blocked, start, goal):
    """维诺骨架图 + 起终点接入 -> (nodes, adj, si, gi, segs)。

    骨架像素按 8 邻接成图；起点/终点作为额外节点连到最近的若干"直线
    可达"（segment_clear）的骨架节点——车常常不在骨架正中线上，若直接
    吸附到最近骨架点再开过去，直线可能穿障碍把车顶死。返回
    (None, None, 0, 0, []) 表示可通行区域不足。
    """
    sk = skeletonize(ok)
    # 栅格数组形状是 (NX, NY)：nonzero 第 1 维是 ix、第 2 维是 iy，
    # 千万别把它当 (y, x) 用——交换后骨架整体镜像，路径会指向错误位置
    xs, ys = np.nonzero(sk)
    if len(xs) < 8:
        return None, None, 0, 0, []
    nodes = [c2w(int(ix), int(iy)) for ix, iy in zip(xs, ys)]
    index = {(int(ix), int(iy)): i for i, (ix, iy) in enumerate(zip(xs, ys))}
    adj = [[] for _ in nodes]
    segs = []
    # 只遍历 4 个正向邻元（+x / +x+y / +y / -x+y），无重复覆盖全部 8 邻接
    for (ix, iy), i in index.items():
        for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1)):
            j = index.get((ix + dx, iy + dy))
            if j is None:
                continue
            w = math.hypot(dx, dy) * RES
            adj[i].append((j, w))
            adj[j].append((i, w))
            segs.append([*nodes[i], *nodes[j]])

    def attach(p):
        """任意点接入骨架图：连到最近的可直线到达的骨架节点。"""
        j = len(nodes)
        nodes.append((float(p[0]), float(p[1])))
        adj.append([])
        order = sorted(range(j), key=lambda k: (nodes[k][0] - p[0]) ** 2
                       + (nodes[k][1] - p[1]) ** 2)[:12]
        for k in order:
            d = math.hypot(nodes[k][0] - p[0], nodes[k][1] - p[1])
            if d > 1.0:
                break
            if segment_clear(p, nodes[k], blocked):
                adj[j].append((k, d))
                adj[k].append((j, d))
        return j

    return nodes, adj, attach(start), attach(goal), segs


def astar(nodes, adj, si, gi):
    """经典 A*：roadmap 上搜 si -> gi 最短路（启发 = 欧氏距离，可采纳）。"""
    gx, gy = nodes[gi]
    h = [math.hypot(x - gx, y - gy) for x, y in nodes]
    g = [math.inf] * len(nodes)
    g[si] = 0.0
    parent = [-1] * len(nodes)
    pq = [(h[si], 0.0, si)]
    while pq:
        _f, gu, u = heapq.heappop(pq)
        if gu > g[u]:                          # 惰性删除过期堆项
            continue
        if u == gi:
            path = []
            while u >= 0:
                path.append(nodes[u])
                u = parent[u]
            return path[::-1]
        for v, w in adj[u]:
            ng = gu + w
            if ng < g[v]:
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h[v], ng, v))
    return None


def densify(path, step=0.10):
    """路径加密：长直段每 step 米插一个点。

    纯跟踪的 carrot 是"沿路径从最近点起第一个足够远的路径点"——稀疏路径
    （巡逻 5 拐点 / 骨架 A* 路径拐点）的相邻点可能相距较远，carrot 会在车
    前后来回翻转导致原地振荡；加密到 0.1m 间距后 carrot 始终在行进方向前方。
    """
    out = [tuple(path[0])]
    for p, q in zip(path, path[1:]):
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(1, int(d / step))
        for i in range(1, n + 1):
            t = i / n
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


# ---------------------------------------------------------------- 驾驶

def pose(robot):
    """车位姿 (x, y, yaw)：freejoint root 前 3 位 + 四元数航向。"""
    x, y, _z, qw, qx, qy, qz = robot.obs["root"]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))
    return x, y, yaw


def drive(robot, v, w):
    """差速底盘：线速度 v / 角速度 w -> 左右轮转速 ctrl（rad/s）。

    正 w = 逆时针（左转）= 左轮慢右轮快；ctrl 上限 ±10 夹到 ±9.5 留伺服余量。
    """
    vl = v - HALF_TRACK * w
    vr = v + HALF_TRACK * w
    robot.set("w_l", max(-9.5, min(9.5, vl / WHEEL_R)))
    robot.set("w_r", max(-9.5, min(9.5, vr / WHEEL_R)))


def follow(robot, path, prog, v_cruise):
    """纯跟踪一段加密路径；返回 (推进后的最近点索引, 是否走完整段)。

    最近路径点：只向后看 2 点防抖、向前看 40 点追新（防走回头路）。
    carrot = 沿路径从最近点起第一个距车 >= LOOKAHEAD 的点，加密后永远
    在行进方向前方。完成判定看 prog 是否推进到路径尾部——不能用"距末点
    距离"，巡逻是闭环、末点就是起点，车出生时距末点 0 会被误判已完成；
    减速看 prog 距尾部的剩余点数，同理。
    """
    x, y, yaw = pose(robot)
    lo, hi = max(0, prog - 2), min(len(path), prog + 40)
    prog = min(range(lo, hi),
              key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
    j = prog
    while j < len(path) - 1 and math.hypot(
            path[j][0] - x, path[j][1] - y) < LOOKAHEAD:
        j += 1
    tx, ty = path[j]
    alpha = wrap(math.atan2(ty - y, tx - x) - yaw)
    v = v_cruise if len(path) - 1 - prog > 8 else max(0.10, v_cruise * 0.5)
    if abs(alpha) > 2.2:                       # 目标在正后方：原地转向兜底
        drive(robot, 0.02, 1.5 if alpha > 0 else -1.5)
    else:
        w = 2.0 * v * math.sin(alpha) / LOOKAHEAD
        drive(robot, v, max(-2.5, min(2.5, w)))
    return prog, prog >= len(path) - 2


# ---------------------------------------------------------------- 地图发布

def to_segments(points):
    """[(x,y),...] 折线 -> [[x1,y1,x2,y2],...]（publish_overlay 的线段格式）。"""
    return [[p[0], p[1], q[0], q[1]] for p, q in zip(points, points[1:])]


# ---------------------------------------------------------------- 主循环

robot = Robot()
goal = tuple(float(v) for v in robot.args.get("goal", (2.1, 1.6)))
# 雷达角表与 worlds/lidar_nav.yaml 的声明保持一致（-180..180 半开区间）
_manifest = {s["name"]: s for s in robot.sensors}
N_RAYS = int(_manifest["lidar"]["dim"])
angles = [math.radians(a) for a in
          np.linspace(-180.0, 180.0, N_RAYS, endpoint=False)]

gmap = GridMap()
phase = "explore"          # explore -> plan -> navigate
prog = 0                   # 当前最近路径点索引（巡逻 / 导航共用）
patrol = densify(PATROL)   # 巡逻环加密（14.8m 周长 -> ~150 点）
nav_path: list = []

# 规划信息发到平台的"雷达建图"面板（矢量叠加层）：占据栅格由内核自动
# 累积，控制器只声明目标 / 巡逻线，规划后再补骨架图、A* 路径与起点
robot.publish_overlay("goal", points=[list(goal)])
robot.publish_overlay("patrol", segments=to_segments(patrol))

while robot.step():
    x, y, yaw = pose(robot)
    dists = robot.sensor("lidar")
    gmap.update(x, y, yaw, dists, angles)

    if phase == "explore":
        prog, done = follow(robot, patrol, prog, V_CRUISE)
        if done or robot.tick > EXPLORE_TIMEOUT:
            phase = "plan"
    elif phase == "plan":
        # ---- 维诺骨架图 + A* 搜路（锁步内核：小车原地暂停等规划）----
        occ, free = gmap.classify()
        blocked = dilate(occ, int(INFLATE / RES))
        ok = erode(free, 2) & ~blocked
        # 规划范围裁剪到巡逻矩形内缩 PLAN_INSET：巡逻框外那圈墙边是白（自由）
        # 但不规划，否则维诺骨架会顺墙边凸出巡逻线（顶部约 0.14m）显得越界
        ax0 = min(p[0] for p in PATROL) + PLAN_INSET
        ax1 = max(p[0] for p in PATROL) - PLAN_INSET
        ay0 = min(p[1] for p in PATROL) + PLAN_INSET
        ay1 = max(p[1] for p in PATROL) - PLAN_INSET
        ix0, iy0 = w2c(ax0, ay0)
        ix1, iy1 = w2c(ax1, ay1)
        area = np.zeros_like(ok)
        area[ix0:ix1 + 1, iy0:iy1 + 1] = True
        ok &= area
        nodes, adj, si, gi, skel_segs = build_skeleton_graph(
            ok, blocked, (x, y), goal)
        path = astar(nodes, adj, si, gi) if nodes else None
        if not path:
            drive(robot, 0.0, 0.0)
            robot.report_result(False, "维诺骨架/A* 规划失败：可通行区域不足")
            break
        # A* 选的路径直接沿骨架走（不拉直），面板上半透明显示以便检查
        # 它是否选对；实际行驶也用它，保证车轨迹与骨架重合
        nav_path = densify(path)
        robot.publish_overlay("skeleton", segments=skel_segs)
        robot.publish_overlay("path", segments=to_segments(path))
        robot.publish_overlay("start", points=[[x, y]])   # 规划起始点
        phase, prog = "navigate", 0
    else:
        prog, done = follow(robot, nav_path, prog, V_CRUISE)
        if done:
            drive(robot, 0.0, 0.0)
            robot.report_result(
                True, f"建图 {robot.tick} tick -> 维诺骨架({len(nodes)} 节点/"
                f"{len(skel_segs)} 边) -> A* {len(path)} 段路 -> 到达 {goal}")
            break

robot.close()
