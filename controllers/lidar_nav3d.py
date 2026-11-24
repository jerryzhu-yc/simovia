"""lidar_nav3d 控制器：3D 多线雷达点云建图 -> 蛇形全覆盖清扫 -> 回充。

配合 worlds/lidar_nav3d.yaml（6.4m x 4.8m 居家房间：双人床 + 衣柜 + 餐桌椅 +
沙发 + 茶几 + 电视柜 + 地毯 + 绿植 + 落地灯 + 门窗，扫地机器人 sweeper +
车顶 3D 雷达 960 束/帧、床底/桌底可钻）。三个阶段：

1. EXPLORE（建图）：沿房间开放区大环巡逻一圈（西南起点 -> 床裙南侧走廊
   -> 床东走廊北上 -> 北走廊东行 -> 东走廊南下 -> 沙发北侧西行（毯缘
   北侧绕毯、茶几南北两排腿之间穿过）-> 西南折回收图）。每个控制 tick 吃
   3D 雷达帧，复用内核 PointCloudMapper（kernel/mapping.py）——命中点离散化
   成体素（全量点云，3D 视口按高度着色渲染成片），z 落在高度带
   (0.02, 0.18) 内的命中投影成俯视占据、带内穿越打自由。带内语义 =
   "扫地机可撞带"：地面（z<0.02）不成障碍，床面/桌面（带外）不挡路，
   床裙/椅腿横撑（0.115~0.155 带内）必进占据——2D 雷达（挂点 ~0.07 平面束）
   恒低于裙底 0.115 永远扫不到它们，这正是 3D 雷达的价值。
2. PLAN（覆盖规划）：可通行区（自由 - 障碍膨胀安全带）按 COVER_STEP 间距
   切成 y 条带，每带沿 x 取连续区段做蛇形往返（相邻带首尾相接），得到
   "把整间房拖一遍"的覆盖线；相邻覆盖线之间用可通行栅格上的 8 邻接 A*
   连接（遇到家具安全带绕行），最后再接一条回充路线到任务目标点。
   维诺骨架只作为叠加层参考（不参与选路）。
3. NAVIGATE（清扫）：纯跟踪走完整条覆盖 + 回充路径，到回充点停车
   report_result(True)，报告覆盖率。

与 lidar_nav.py 的关键差异：
- 建图实现不复制：直接 import 内核 PointCloudMapper（一份语义，
  内核侧的"雷达建图"面板 GUI 视图与控制器决策用同一张栅格）
- 雷达方向表 make_dirs import 自 world/builder（channel-major 平铺，
  与内核 lidar_scan 3D 分支、传感器读数顺序严格一致）
- origin/rot 每帧从 root 四元数 + 挂点偏移现算（雷达挂 root body）
- 车辆参数换成 sweeper：半轮距 0.13、安全带 0.30（底盘 r=0.14 + 余量）

3D 点云本体（体素）由内核累积并同步进预置槽位 geom——录像流/截图直接
出图；GUI 的立体点云由独立 3D 点云面板渲染。控制器只负责决策用的俯视
占据栅格与矢量叠加层（骨架/覆盖线/路径/目标）。
"""
import heapq
import math

import numpy as np

from simovia.controller import Robot
from simovia.kernel.mapping import (PointCloudMapper, OCC_TH, FREE_TH)
from simovia.world.builder import make_dirs

# ---------------------------------------------------------------- 地图参数
RES = 0.05                      # 栅格分辨率（米/格）——与 kernel/mapping.py 同款
# 网格范围/尺寸（GX0/GY0/NX/NY）：gmap 实例化后从 mapper 对齐（mapper 的
# 原点含 MARGIN 外扩），w2c/c2w 运行时查全局名——赋值在此之后即可生效
GX0 = GY0 = GX1 = GY1 = 0.0
NX = NY = 1

# ---------------------------------------------------------------- 车辆参数
WHEEL_R = 0.032                 # 驱动轮半径（m）
HALF_TRACK = 0.13               # 半轮距（m）：sweeper 轮距 ±0.13
V_CRUISE = 0.22                 # 巡航线速度（m/s）
LOOKAHEAD = 0.22                # 纯跟踪前瞻距离（m）
EXPLORE_TIMEOUT = 7000          # 巡逻兜底 tick：超时直接转入规划（防卡死）；
                                # 正常巡逻一圈 ~3300 tick，留一倍余量，
                                # 也保证兜底后仍有足够 tick 走完全屋覆盖

# 巡逻路径（大环，避开贴墙家具的物理碰撞区）：西南起点 -> 床裙南侧走廊
# -> 床东走廊北上 -> 北走廊（餐桌椅/衣柜南侧 y1.35，留 0.13m 间隙）东行 ->
# 东走廊（电视柜西侧 x2.3）南下 -> 沙发北侧西行 y-1.1 -> 从茶几南北两排腿
# 之间（0.52m 门洞、桌面底 0.38m 净空）钻茶几底 -> 茶几西侧 -> 西南折回
# 起点收图。注：沙发北缘 y-1.51 与茶几南腿 y-1.41 之间无 0.28m 净通道，
# 南段必走沙发北侧；地毯北缘 -1.35 在巡逻线南侧，扫地机不爬毯
PATROL = [(-2.5, -1.7), (-2.7, -0.5), (-1.3, -0.5), (-1.3, 1.35),
          (2.3, 1.35), (2.3, -1.1), (0.2, -1.1), (-1.2, -1.1), (-2.5, -1.7)]

# ---------------------------------------------------------------- 规划参数
INFLATE = 0.30                  # 障碍膨胀半径：扫地机底盘 r=0.14 + 纯跟踪
                                # 切角余量（床裙/椅横撑/各家具腿的带内投影都要盖住）

# ---------------------------------------------------------------- 覆盖参数
COVER_STEP = 0.34               # 蛇形条带间距（米）：扫盘有效宽度 + 转弯余量
COVER_MIN_SEG = 0.30            # 单条覆盖线最短长度（更碎的段跳过，转弯不值）
V_COVER = 0.24                  # 覆盖巡航速度（m/s，转弯频繁，留足轮速余量）
NAV_TIMEOUT = 18000             # 覆盖+回充兜底 tick（防卡死，任务上限 20000）


# ---------------------------------------------------------------- 基础工具

def w2c(x, y):
    """世界坐标 -> 栅格索引（越界返回 None）。"""
    ix, iy = int((x - GX0) / RES), int((y - GY0) / RES)
    return (ix, iy) if 0 <= ix < NX and 0 <= iy < NY else None


def c2w(ix, iy):
    """栅格索引 -> 格中心世界坐标。"""
    return GX0 + (ix + 0.5) * RES, GY0 + (iy + 0.5) * RES


def wrap(a):
    """角域归一到 (-pi, pi]。"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def quat_rot(qw, qx, qy, qz):
    """四元数（wxyz）-> 3x3 旋转矩阵（把雷达 body 系方向投到世界系）。"""
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qw * qz)
    R[0, 2] = 2 * (qx * qz + qw * qy)
    R[1, 0] = 2 * (qx * qy + qw * qz)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qw * qx)
    R[2, 0] = 2 * (qx * qz - qw * qy)
    R[2, 1] = 2 * (qy * qz + qw * qx)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


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


def erode(mask, r=2):
    """方形结构元腐蚀：剪掉贴未观测边界/墙缝的细条假自由。"""
    p = np.pad(mask, r)
    nx, ny = mask.shape
    out = mask.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out &= p[r + dx:r + dx + nx, r + dy:r + dy + ny]
    return out


# ---------------------------------------------------------------- 维诺骨架图

def skeletonize(mask):
    """Zhang-Suen 细化：可通行区域 -> 1px 中轴骨架（纯 numpy，无 scipy）。"""
    pad = np.pad(mask.astype(np.uint8), 1)
    sk = pad.copy()
    offs = ((-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1))
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(sk, dy, 0), dx, 1) for dy, dx in offs]
            b = sum(p)
            a = sum(((p[i] == 0) & (p[(i + 1) % 8] == 1))
                    for i in range(8))
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


def skeleton_segments(sk):
    """骨架掩码 -> 叠加层线段 [[x1,y1,x2,y2],...]（8 邻接连边，仅作参考展示）。"""
    xs, ys = np.nonzero(sk)
    idx = {(int(ix), int(iy)) for ix, iy in zip(xs, ys)}
    segs = []
    for ix, iy in idx:
        for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1)):
            if (ix + dx, iy + dy) in idx:
                x1, y1 = c2w(ix, iy)
                x2, y2 = c2w(ix + dx, iy + dy)
                segs.append([x1, y1, x2, y2])
    return segs


def nearest_ok(ok, p, radius=12):
    """世界点 -> 半径内最近的可通行格中心（找不到返回 None）。

    车当前位置/任务目标点可能落在膨胀安全带里：覆盖起点与回充终点都先
    吸附到 ok 格上，栅格 A* 才有合法端点。
    """
    c = w2c(p[0], p[1])
    if c is None:
        return None
    if ok[c]:
        return c2w(*c)
    best, bd = None, None
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            ix, iy = c[0] + dx, c[1] + dy
            if not (0 <= ix < NX and 0 <= iy < NY) or not ok[ix, iy]:
                continue
            d = dx * dx + dy * dy
            if bd is None or d < bd:
                best, bd = (ix, iy), d
    return c2w(*best) if best is not None else None


def grid_astar(ok, p, q):
    """可通行栅格上的 8 邻接 A*（端点为世界坐标），返回折线或 None。"""
    s, g = w2c(p[0], p[1]), w2c(q[0], q[1])
    if s is None or g is None or not ok[s] or not ok[g]:
        return None
    nbrs = ((1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
    h = lambda v: math.hypot(v[0] - g[0], v[1] - g[1]) * RES
    gs = {s: 0.0}
    parent = {}
    pq = [(h(s), 0.0, s)]
    while pq:
        _f, gu, u = heapq.heappop(pq)
        if u == g:
            pts = []
            while u in parent:
                pts.append(c2w(*u))
                u = parent[u]
            pts.append(c2w(*s))
            return pts[::-1]
        if gu > gs.get(u, math.inf):
            continue
        for dx, dy, k in nbrs:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < NX and 0 <= v[1] < NY) or not ok[v]:
                continue
            ng = gu + k * RES
            if ng < gs.get(v, math.inf):
                gs[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h(v), ng, v))
    return None


def cover_lines(ok):
    """可通行区 -> 蛇形覆盖线 [(起点, 终点), ...]。

    按 y 每隔 COVER_STEP 取一条带中心行，取该行 ok 的连续 x 区段作为一条
    覆盖线；相邻带方向交替（左->右、右->左），上一带终点与下一带起点天然
    相邻，转弯里程最小。带内被家具安全带切断时会有多条短线，各自成段。
    """
    nx, ny = ok.shape
    step = max(1, int(round(COVER_STEP / RES)))
    lines, flip = [], False
    for iy in range(step // 2, ny, step):
        row = ok[:, iy]
        segs, start = [], None
        for ix in range(nx):
            if row[ix] and start is None:
                start = ix
            elif not row[ix] and start is not None:
                segs.append((start, ix - 1))
                start = None
        if start is not None:
            segs.append((start, nx - 1))
        segs = [s for s in segs if (s[1] - s[0] + 1) * RES >= COVER_MIN_SEG]
        if not segs:
            continue
        if flip:
            segs.reverse()
        for a, b in segs:
            p, q = c2w(a, iy), c2w(b, iy)
            lines.append((q, p) if flip else (p, q))
        flip = not flip
    return lines


def build_cover_path(ok, start, lines):
    """覆盖线 + 栅格 A* 连接 -> 一条连续折线（世界坐标）。

    每条覆盖线先接一段 A*（cur -> 线起点，绕开家具安全带），再走覆盖线
    直线本身。不可达的覆盖线直接跳过（记进未覆盖率，不阻塞整条路线）。
    """
    nok = ~ok
    path, cur = [], nearest_ok(ok, start)
    if cur is None:
        return None
    for a, b in lines:
        ca, cb = w2c(*a), w2c(*b)
        if ca is None or cb is None or not ok[ca] or not ok[cb]:
            continue
        # 纯跟踪有切角：A* 连接只负责"可达"，不追求最短
        if math.hypot(a[0] - cur[0], a[1] - cur[1]) > 1.5 * RES and \
                not segment_clear(cur, a, nok):
            seg = grid_astar(ok, cur, a)
            if seg is None:
                continue
            path.extend(seg[:-1])
        path.append(a)
        path.append(b)
        cur = b
    return path if path else None


def coverage_ratio(ok, lines):
    """覆盖线按条带宽度扫过的可通行格占 ok 的比例（0..1）。"""
    r = max(1, int(round(COVER_STEP / RES / 2)))
    covered = np.zeros_like(ok)
    ny = ok.shape[1]
    for a, b in lines:
        ca, cb = w2c(*a), w2c(*b)
        if ca is None or cb is None:
            continue
        lo, hi = min(ca[0], cb[0]), max(ca[0], cb[0])
        covered[lo:hi + 1, max(0, ca[1] - r):min(ny, ca[1] + r + 1)] = True
    return float((covered & ok).sum()) / max(1, int(ok.sum()))


def densify(path, step=0.10):
    """路径加密：长直段每 step 米插一个点（纯跟踪 carrot 稳定在行进前方）。"""
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


def lidar_frame(robot):
    """雷达挂点世界原点(3,) + 3x3 旋转（root 四元数 + 挂点偏移）。

    与内核 kernel/task.lidar_frame 同一约定（雷达挂 root body）；
    控制器只有 root 观测，故用四元数自行重建旋转矩阵。
    """
    x, y, z, qw, qx, qy, qz = robot.obs["root"]
    rot = quat_rot(qw, qx, qy, qz)
    origin = np.array([x, y, z]) + rot @ LIDAR_POS
    return origin, rot


def drive(robot, v, w):
    """差速底盘：线速度 v / 角速度 w -> 左右轮转速 ctrl（rad/s）。"""
    vl = v - HALF_TRACK * w
    vr = v + HALF_TRACK * w
    robot.set("w_l", max(-9.5, min(9.5, vl / WHEEL_R)))
    robot.set("w_r", max(-9.5, min(9.5, vr / WHEEL_R)))


def follow(robot, path, prog, v_cruise):
    """纯跟踪一段加密路径；返回 (推进后的最近点索引, 是否走完整段)。"""
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


def to_segments(points):
    """[(x,y),...] 折线 -> [[x1,y1,x2,y2],...]（publish_overlay 的线段格式）。"""
    return [[p[0], p[1], q[0], q[1]] for p, q in zip(points, points[1:])]


# ---------------------------------------------------------------- 主循环

robot = Robot()
goal = tuple(float(v) for v in robot.args.get("goal", (2.1, 1.6)))

# ---- 从握手下发的 world_spec 解析 3D 雷达配置（与 YAML 声明同一份语义）----
_rd = next(r for r in robot.world_spec["robots"] if r["name"] == robot.name)
_li = next(s for s in _rd["sensors"] if s["type"] == "lidar")
LIDAR_POS = np.asarray(_li.get("pos", [0, 0, 0]), dtype=float)
_rays = int(_li.get("rays", 36))
_channels = int(_li.get("channels", 1))
_r = float(_li.get("range", 5.0))
_a0 = float(_li.get("min_angle", -180))
_a1 = float(_li.get("max_angle", 180))
angles = np.radians(np.linspace(_a0, _a1, _rays, endpoint=False))
vangles = np.radians(np.linspace(float(_li.get("vmin", -10)),
                                 float(_li.get("vmax", 14)), _channels))
dirs = make_dirs(angles, vangles)

_floor = robot.world_spec["scene"]["floor"]["size"]
gmap = PointCloudMapper((_floor[0], _floor[1]))   # 复用内核建图（同款语义）
# 控制器栅格工具（w2c/c2w/规划裁剪）与 mapper 网格逐参数对齐：
# 两套原点会让骨架与占据栅格整体错位 0.35m，路径直接穿障碍
GX0, GY0 = gmap.ox, gmap.oy
GX1, GY1 = gmap.ox + gmap.nx * RES, gmap.oy + gmap.ny * RES
NX, NY = gmap.nx, gmap.ny

phase = "explore"          # explore -> plan -> navigate（覆盖 + 回充）
prog = 0                   # 当前最近路径点索引（巡逻 / 覆盖共用）
nav_t0 = 0                 # 进入 navigate 的 tick（超时按覆盖阶段起算）
patrol = densify(PATROL)   # 巡逻环加密
nav_path: list = []
cover_stats = ""           # 规划结果摘要（结束时回报）

# 规划信息发到平台的"雷达建图"面板（矢量叠加层）：3D 点云与俯视占据栅格
# 由内核自动累积渲染，控制器只声明目标 / 巡逻线，规划后再补骨架图、
# 覆盖线、路径与起点
robot.publish_overlay("goal", points=[list(goal)])
robot.publish_overlay("patrol", segments=to_segments(patrol))

while robot.step():
    dists = robot.sensor("lidar")            # 960 维（channel-major 平铺）
    origin, rot = lidar_frame(robot)
    gmap.update(origin, rot, dirs, dists, _r)

    if phase == "explore":
        prog, done = follow(robot, patrol, prog, V_CRUISE)
        if done or robot.tick > EXPLORE_TIMEOUT:
            phase = "plan"
    elif phase == "plan":
        # ---- 蛇形全覆盖规划（锁步内核：扫地机原地暂停等规划）----
        occ, free = gmap.l > OCC_TH, gmap.l < FREE_TH
        blocked = dilate(occ, int(INFLATE / RES))
        ok = erode(free, 2) & ~blocked
        # 覆盖整间房（不再裁剪到巡逻矩形）：用户要"把所有场地拖一遍"
        if int(ok.sum()) < 200:
            drive(robot, 0.0, 0.0)
            robot.report_result(
                False, "覆盖规划失败：可通行区域不足"
                       f"（占据 {int(occ.sum())} 格 / 自由 {int(free.sum())} 格 / "
                       f"可通行 {int(ok.sum())} 格 / 点云体素 {len(gmap.vox)}）")
            break
        skel_segs = skeleton_segments(skeletonize(ok))   # 仅叠加层参考
        lines = cover_lines(ok)
        px, py, _yaw = pose(robot)
        path = build_cover_path(ok, (px, py), lines)
        if not path:
            drive(robot, 0.0, 0.0)
            robot.report_result(False, "覆盖路径生成失败：没有可用覆盖线")
            break
        # 覆盖收尾接一段回充路线到任务目标点（世界成功判定按目标 zone）
        home = grid_astar(ok, path[-1], nearest_ok(ok, goal) or goal)
        if home:
            path = path + home[1:]
        nav_path = densify(path)
        ratio = coverage_ratio(ok, lines)
        cover_stats = (f"可通行 {int(ok.sum())} 格 / {len(lines)} 条覆盖线"
                       f"（间距 {COVER_STEP:.2f}m）预估覆盖 {ratio * 100:.0f}%"
                       f"；路线 {len(path)} 段"
                       + ("，回充段已接" if home else "，回充段未规划"))
        robot.publish_overlay("skeleton", segments=skel_segs)
        # 覆盖线逐条独立成段（不连成折线，避免跨家具的假连线）
        robot.publish_overlay("cover", segments=[[a[0], a[1], b[0], b[1]]
                                                 for a, b in lines])
        robot.publish_overlay("path", segments=to_segments(path))
        robot.publish_overlay("start", points=[[px, py]])
        phase, prog = "navigate", 0
        nav_t0 = robot.tick
    else:
        prog, done = follow(robot, nav_path, prog, V_COVER)
        if done or robot.tick - nav_t0 > NAV_TIMEOUT:
            drive(robot, 0.0, 0.0)
            state = ("完成" if done else
                     f"超时（{robot.tick - nav_t0} tick，"
                     f"路径进度 {prog}/{len(nav_path)}）")
            robot.report_result(done, f"全屋蛇形清扫{state}：{cover_stats}")
            break

robot.close()
