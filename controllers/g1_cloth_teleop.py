"""宇树 G1 双臂叠布遥操作 + π0.5（openpi）数据集采集。

本文件以 shared_world_controller 身份跑在内核进程里（世界 YAML 里
`args.shared_world_controller: true`）。只有这样才能：
  - 锁住 G1 浮动基座（否则人形一跑就倒）；
  - 逐物理步给布料顶点施加合规抓持力（控制器子进程只能写 ctrl）；
  - 逐控制 tick 渲染三路相机并把帧写进 LeRobot 数据集。

链路（PICO 头显自带浏览器打开本页，无需装 APP）：

    PICO 浏览器 --WebSocket(~60Hz)--> 本控制器（内嵌 HTTP+WS 服务器）
        左手柄位移 -> 左臂 TCP 目标（增量映射）
        右手柄位移 -> 右臂 TCP 目标
        扳机       -> 2F85 夹爪开合（0 = 张开，1 = 闭合）
        A          -> 开始 / 停止录制
        B          -> 重新对中（双手当前位置成为新的位移原点）
        X          -> 换一套随机场景（布料颜色/摆放、桌面颜色、灯光）
        Y          -> 丢弃当前这一集

坐标映射（VR 右手系 y-up -> MuJoCo z-up）：v_mj = [-v_z, -v_x, v_y]。
工具姿态固定为「手指朝下、夹爪开合轴沿世界 x」：毛毯整块平铺在桌面上，
双手从上方压到布边、合指捏住近侧布面再对折。

数据集：LeRobot v2.1（openpi π0.5 的 LeRobot loader 直接可读，不升级
v3.0）。16 维 state/action = 左 7 关节 + 左夹爪 + 右 7 关节 + 右夹爪；
相机 = cam_high（世界固定机位）+ 每个活动臂一路腕相机（双臂 3 路 /
单臂 2 路）；fps = 1 / (timestep * ctrl_period)。

控制：默认 OSC（操作空间控制，位置+姿态速度任务 + 零空间姿态保持），
目标误差 -> 任务速度 -> 关节速度，天然限速、抗手抖；`control: ik` 可退回
原来的绝对 IK 方案。`arms: left|right` 只启用一侧手臂（另一侧竖直下垂
锁在体侧，避开桌面），只采该侧腕相机。

抓取：2F85 合拢时把指尖盒内的布顶点锁住，用弹簧-阻尼让它们跟随手走
（与 demo/cloth_folding 相同的合规抓持力模型）。布料本身仍是完整 flexcomp
柔体：重力、桌面接触、自碰撞都是真实仿真。

世界 YAML 可通过 robots[].args 覆盖：
    dataset:  数据集根目录（默认 ~/桌面/g1_cloth_teleop_dataset）
    task:     语言条件文本（默认 put the blanket in the box）
    port:     HTTP/WS 端口（默认 8091）
    cam_size: 相机分辨率（默认 256，正方形）
    scale:    手柄位移 -> TCP 位移缩放（默认 0.45）
    arms:     both / left / right（默认 both）
    control:  osc / ik（默认 osc）
    home:     双手预备 TCP 位置 [x, y, z]（y 取正值，左右镜像）
    grasp:    是否启用合规抓持力（默认 true；false 时纯物理夹持）
"""
from __future__ import annotations

import asyncio
import json
import math
import pathlib
import socket
import sys
import threading
import time

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simovia.lerobot_writer import LerobotWriter      # noqa: E402
from simovia.render import render_offscreen          # noqa: E402

G1_XML = REPO / "models" / "zoo" / "unitree_g1" / "g1_with_hands.xml"

SIDES = ("left", "right")
ARM_JOINTS = {
    side: [
        f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint", f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint", f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    ]
    for side in SIDES
}
#: 右臂相对左臂的镜像符号（对称关节取反，pitch/elbow 保持同号）。
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

WAIST_PITCH = 0.45          # 躯干前倾（关节限位 ±0.52）：夹爪按真实法兰位置
                            # 装到腕端后，臂展比之前「悬空 8cm」时短了，前倾
                            # 到 0.45rad 才能让指尖够到布面远边 x≈0.43

#: 工具姿态（左右手各一份）：site z（工具轴）-> 世界 -z（手指朝下），
#: site y（两指开合轴）-> 世界 ±y。左右腕的滚转可达区间是镜像的，必须用
#: 镜像朝向——两侧用同一朝向时总有一侧的姿态够不到，表现为 IK 残差 10cm+
#: （右臂能到、左臂不能，或反之）。
R_TOOL = {
    "left": np.array([[-1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, -1.0]]),
    "right": np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]]),
}

#: VR 右手系 -> MuJoCo。VR -z(前)->mj x(前)，VR x(右)->mj -y(右)，VR y(上)->mj z。
_T = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

DEFAULT_PORT = 8091
DEFAULT_TASK = "put the blanket in the box"
DEFAULT_DATASET = pathlib.Path.home() / "桌面" / "g1_cloth_teleop_dataset"
DEFAULT_HOME = (0.40, 0.02, 0.86)   # 手悬在布料中心上方（布心 x=0.40、
                                    # y=0.02，右臂可达上限约 x=0.44）

#: 自然预备姿态种子（左臂 7 关节角，右臂按 MIRROR_SIGN 镜像）。手臂外展、
#: 手指朝下，离所有关节限位都远（最近限位余量 0.72rad）。7 自由度冗余臂
#: 到达同一个 TCP 位姿有大量分支，其中「肩 pitch/yaw 顶到限位、手臂后翻到
#: 头顶」的分支运动学上同样收敛，但 kp=500 的伺服顶不动——实测一跑手臂就
#: 下垂 12cm。`_solve_home` 用它当主种子并加姿态距离惩罚，避免选中畸形解。
HOME_SEED = np.array([-0.91, 1.40, -1.62, -0.32, -0.48, 0.58, -0.62])

CTRL_TIMEOUT = 0.8          # 手柄数据超过此秒数视为失联（冻结该臂目标）
MAX_JOINT_SPEED = 3.0       # 手臂关节速度上限（rad/s）
GRIP_OPEN = 0.0
#: 闭合上限没取执行器的 255：毯子只有 4mm 厚，指令压到 255 时两片指垫被挤到
#: 16mm 间距，把布压进指垫 1.9mm（约布厚一半）。松爪后指垫虽然张开到 98mm，
#: 布料仍卡在指垫/指体之间，跟着手一起被带走——实测上抬 12cm 再侧移 25cm，
#: 布还停在 z=0.99（桌面 0.752），这就是「松爪后缠在夹爪上」。
#: 上限取 220：指垫停在 26mm、穿入 1.1mm，松爪 3.8s 内正常落回 0.76。
#: 扫过 160/180/200/220/240 都能正常脱落，只有 255 会卡住，故留 35 的余量。
GRIP_CLOSE = 220.0
# 夹爪张合速率（1/s，0~255 量纲）。放开扳机后弹簧锁定在 grip_cmd 跌破
# GRASP_ON 时解除，但半开的手指仍会靠摩擦挂着布——速率低了布料会
# 「黏」在夹爪上零点几秒才掉。1200/s 时从全合到跌破 140 只要 0.09s。
GRIP_SPEED = 1200.0
GRASP_ON = 140.0            # 合到此值以上开始锁定布顶点（跌破即解除）
GRASP_OFF = 40.0            # （保留阈值：如需滞后窗口再启用）

# ---------------- OSC（操作空间控制）参数 ----------------
# 把「TCP 目标误差」换算成任务空间速度再映射到关节速度，而不是每 tick
# 硬解一个绝对 IK 解。好处：PICO 的毫米级抖动不会直接变成关节跳变，
# 目标超出可达域时速度饱和而不是顶限位，7 自由度冗余臂不会因分支切换
# 让末端瞬跳。
OSC_KP_POS = 12.0           # 位置误差 -> 任务线速度（1/s）
OSC_VMAX = 0.35             # 任务空间线速度上限（m/s），限制最大手速
OSC_KP_ROT = 4.0            # 姿态误差 -> 任务角速度（1/s）
OSC_WMAX = 1.2              # 任务空间角速度上限（rad/s）
OSC_LAMBDA = 0.05           # 阻尼最小二乘的 λ（抗奇异位形抖动）
OSC_KP_POSTURE = 1.0        # 零空间姿态保持：把冗余自由度拉回预备构型
OSC_KD_POSTURE = 0.25       # 零空间速度阻尼
OSC_KI = 1.5                # 末端稳态差慢速积分（1/s）：补重力/伺服下垂
# 积分门槛与限幅：空手稳态差约 2cm；抓住布料（含夹爪负载）会到 5cm，
# 门槛必须高于这个量，否则积分永远不启动、手就「够不到目标高度」。
OSC_BIAS_GATE = 0.07        # 误差小于此值才积分（米），再大视为在移动
OSC_BIAS_MAX = 0.07         # 积分偏置限幅（米）
HAND_DEADBAND = 0.0008      # 手柄位移死区（米）：滤掉 PICO 毫米级自抖动

#: 闲置臂停放姿态（单臂模式）：肩 pitch 反向补偿躯干前倾，让手臂在世界系里
#: 竖直下垂。全零位是「沿躯干方向下垂」，躯干前倾 0.45rad 后指尖落到
#: (0.35, 0.16, 0.72)——正好在桌面台面（x [0.22, 0.72]、z [0.71, 0.75]）里，
#: 开局就被桌板顶住，位置伺服硬顶还会把整条臂挤进躯干（实测关节漂到 4rad）。
#: 0.55rad 时指尖 (0.128, 0.196, 0.645)，离桌沿 9cm，且与竖直方向只差 3°；
#: 200 tick 物理推进实测与外部 geom 零接触。
PARK_SHOULDER_PITCH = 0.55

# 布料颜色随机池（rgba）
_CLOTH_COLORS = np.array([
    [0.851, 0.467, 0.341, 1.0], [0.216, 0.475, 0.851, 1.0],
    [0.250, 0.663, 0.427, 1.0], [0.855, 0.267, 0.286, 1.0],
    [0.545, 0.400, 0.800, 1.0], [0.900, 0.700, 0.250, 1.0],
    [0.180, 0.620, 0.680, 1.0], [0.850, 0.520, 0.700, 1.0],
])
# 桌面颜色随机池（rgba）
_TABLE_COLORS = np.array([
    [0.420, 0.447, 0.502, 1.0], [0.545, 0.427, 0.310, 1.0],
    [0.360, 0.380, 0.410, 1.0], [0.600, 0.580, 0.540, 1.0],
    [0.330, 0.400, 0.380, 1.0],
])


def _quat_err(target_r: np.ndarray, cur_r: np.ndarray) -> np.ndarray:
    """两个旋转矩阵之差的旋转向量（世界系）。"""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (target_r @ cur_r.T).ravel())
    if q[0] < 0:
        q = -q
    v, w = q[1:], q[0]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3)
    return v / n * (2.0 * math.atan2(n, w))


def _quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in a]
    w2, x2, y2, z2 = [float(v) for v in b]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _local_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except socket.gaierror:
        pass
    return sorted(ips)


# ---------------------------------------------------------------- 手柄输入

class _DualTracker:
    """双手柄最新状态（WS 线程写，控制循环读，GIL 下小字典安全）。

    多条连接时只认「最新连上且在传数据」的那一路；owner 变更时 generation
    自增，控制循环据此重置增量基准，避免一次大跳。
    """

    STALE = 1.0

    def __init__(self):
        self.latest = {"left": None, "right": None}
        self.recv = {"left": 0.0, "right": 0.0}
        self.owner = None
        self.generation = 0
        self._last: dict = {}
        # 按键按下沿锁存：控制循环 64ms 才采样一次，短促按键不能漏
        self.press = {"a": 0, "b": 0, "x": 0, "y": 0}
        self._prev = {"a": False, "b": False, "x": False, "y": False}

    def claim(self, cid) -> None:
        if self.owner != cid:
            self.owner = cid
            self._last[cid] = time.monotonic()
            self.generation += 1

    def release(self, cid) -> None:
        self._last.pop(cid, None)
        if self.owner != cid:
            return
        self.owner = None
        if self._last:
            self.owner = max(self._last, key=self._last.get)
            self.generation += 1
        else:
            self.latest = {"left": None, "right": None}

    def feed(self, msg: dict, cid) -> None:
        now = time.monotonic()
        self._last[cid] = now
        if self.owner is not None and cid != self.owner:
            if self._last.get(self.owner, 0.0) > now - self.STALE:
                return
            self.owner = cid
            self.generation += 1
        for side in SIDES:
            d = msg.get(side)
            if isinstance(d, dict) and "p" in d:
                self.latest[side] = d
                self.recv[side] = now
        right = msg.get("right") or {}
        left = msg.get("left") or {}
        for key, val in (("a", right.get("a")), ("b", right.get("b")),
                         ("x", left.get("x")), ("y", left.get("y"))):
            v = bool(val)
            if v and not self._prev[key]:
                self.press[key] += 1
            self._prev[key] = v


class _PicoServer:
    """HTTP(页面) + WebSocket(双手柄数据) 单端口服务器（后台线程）。"""

    def __init__(self, port: int, tracker: _DualTracker,
                 html_path: pathlib.Path):
        self.port = port
        self.tracker = tracker
        self.html = html_path.read_bytes()
        self._clients: set = set()
        self._seq = 0
        self._loop = None
        self._stop = None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        loop, ev = self._loop, self._stop
        if loop is None or ev is None:
            return
        try:
            loop.call_soon_threadsafe(ev.set)
        except RuntimeError:
            pass

    def _run(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            print("[g1_teleop] 缺 websockets 库：pip install websockets",
                  flush=True)
            return

        async def handler(ws):
            self._seq += 1
            cid = self._seq
            self._clients.add(ws)
            self.tracker.claim(cid)
            peer = getattr(ws, "remote_address", None)
            print(f"[g1_teleop] 手柄已连接 {peer}"
                  f"（共 {len(self._clients)} 路）", flush=True)
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(msg, dict) and (
                            "left" in msg or "right" in msg):
                        self.tracker.feed(msg, cid)
            finally:
                self._clients.discard(ws)
                self.tracker.release(cid)
                print(f"[g1_teleop] 手柄断开（剩 {len(self._clients)} 路）",
                      flush=True)

        async def process_request(connection, request):
            # 非 WebSocket 升级请求（PICO 浏览器打开页面）-> 伺服 HTML
            if request.headers.get("Upgrade", "").lower() != "websocket":
                from websockets.datastructures import Headers
                from websockets.http11 import Response
                headers = Headers([
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(self.html))),
                    ("Cache-Control", "no-store"),
                    ("Connection", "close"),
                ])
                return Response(200, "OK", headers, self.html)
            return None

        async def main():
            self._loop = asyncio.get_running_loop()
            self._stop = asyncio.Event()
            async with serve(handler, "0.0.0.0", self.port,
                             process_request=process_request,
                             max_size=65536, ping_interval=10):
                await self._stop.wait()

        try:
            asyncio.run(main())
        except Exception as e:                      # noqa: BLE001
            print(f"[g1_teleop] PICO 服务器退出: {e}", flush=True)

    def broadcast(self, payload: dict) -> None:
        loop = self._loop
        if loop is None or not self._clients:
            return

        async def push():
            data = json.dumps(payload)
            for ws in list(self._clients):
                try:
                    await ws.send(data)
                except Exception:                   # noqa: BLE001
                    self._clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(push(), loop)
        except Exception:                           # noqa: BLE001
            pass


# ---------------------------------------------------------------- 控制器

class G1ClothTeleop:
    """内核协调器：双臂 IK + 2F85 夹爪 + 布料抓持力 + 三路相机录制。"""

    def __init__(self, simulation):
        self.sim = simulation
        self.model = simulation.model
        self.data = simulation.data
        self.index = simulation.index
        self.spec = simulation.spec

        robot = self.spec.robots[0]
        self.robot_name = robot.name
        self.args = dict(robot.args or {})
        m = self.model
        ridx = self.index.robots[robot.name]
        self.actuators = dict(ridx.actuators)
        self.joints = dict(ridx.joints)

        self.ctrl_period = max(1, int(self.spec.ctrl_period))
        self.dt = float(self.spec.timestep)
        self.tick_dt = self.dt * self.ctrl_period
        self.fps = 1.0 / self.tick_dt
        self.rng = np.random.default_rng()

        # ---- 手臂 / 夹爪设备
        self.arm_dofs, self.arm_qadrs = {}, {}
        self.arm_lo, self.arm_hi = {}, {}
        for side in SIDES:
            jids = [self.joints[n] for n in ARM_JOINTS[side]]
            self.arm_dofs[side] = np.array([int(m.jnt_dofadr[j]) for j in jids])
            self.arm_qadrs[side] = np.array([int(m.jnt_qposadr[j]) for j in jids])
            rng = np.array([m.jnt_range[j] for j in jids], dtype=float)
            lo, hi = rng[:, 0].copy(), rng[:, 1].copy()
            # 肩 roll 只允许「外展」方向：stand 姿势把双臂内旋贴到躯干上，
            # 内旋侧肩壳直接插进躯干，物理上顶不到位（见叠毯 demo 实测）。
            if side == "left":
                lo[1] = max(lo[1], 0.0)
            else:
                hi[1] = min(hi[1], 0.0)
            self.arm_lo[side], self.arm_hi[side] = lo, hi

        self.tcp_site, self.tcp_body = {}, {}
        self.grip_act, self.grip_drv_qadr = {}, {}
        for side in SIDES:
            sname = f"{robot.name}/gripper_{side}/gripper_tcp"
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                raise RuntimeError(
                    f"找不到 2F85 的 TCP site '{sname}'。世界 YAML 的 "
                    f"end_effectors 是否左右各挂了一条 robotiq_2f85？")
            self.tcp_site[side] = sid
            self.tcp_body[side] = int(m.site_bodyid[sid])
            aname = f"gripper_{side}/fingers_actuator"
            dname = f"gripper_{side}/right_driver_joint"
            if aname not in self.actuators or dname not in self.joints:
                raise RuntimeError(
                    f"找不到夹爪执行器/关节 '{aname}' / '{dname}'")
            self.grip_act[side] = self.actuators[aname]
            self.grip_drv_qadr[side] = int(
                m.jnt_qposadr[self.joints[dname]])

        # ---- 保持类执行器（腿/腰/头）：每步把 ctrl 钉在当前关节角
        arm_acts = {n for s in SIDES for n in ARM_JOINTS[s]}
        grip_acts = {f"gripper_{s}/fingers_actuator" for s in SIDES}
        self.hold: list[tuple[int, int]] = []
        for name, aid in self.actuators.items():
            if name in arm_acts or name in grip_acts:
                continue
            if int(m.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            jid = int(m.actuator_trnid[aid, 0])
            self.hold.append((int(aid), int(m.jnt_qposadr[jid])))

        # ---- 浮动基座（G1 是 free joint，必须每步锁住）
        free_jid = next(j for j in range(m.njnt)
                        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE))
        self.base_qpos = int(m.jnt_qposadr[free_jid])
        self.base_dof = int(m.jnt_dofadr[free_jid])

        # ---- 布料（唯一 flex）与可随机化的静态位姿
        self.flex_id = -1
        for i in range(int(m.nflex)):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i) or ""
            if "cloth" in nm or nm.startswith("blanket"):
                self.flex_id = i
                break
        if self.flex_id < 0 and m.nflex == 1:
            self.flex_id = 0
        self.vert_adr = self.vert_num = 0
        self.vert_bodies: list[int] = []
        self.vert_dof: list[int] = []
        self.cloth_body = -1
        self.cloth_body0 = np.zeros(3)
        self.cloth_quat0 = np.array([1.0, 0.0, 0.0, 0.0])
        if self.flex_id >= 0:
            self.vert_adr = int(m.flex_vertadr[self.flex_id])
            self.vert_num = int(m.flex_vertnum[self.flex_id])
            vids = [int(v) for v in
                    m.flex_vertbodyid[self.vert_adr:self.vert_adr + self.vert_num]]
            self.vert_bodies = vids
            self.vert_dof = [int(m.body_dofadr[b]) for b in vids]
        # 合规抓持弹簧按「单顶点质量」定标：薄毯网格加密后单顶点只有
        # 0.08kg/960 ≈ 0.08g，固定 kp=400 的显式弹簧在 dt=2ms 下会炸
        # （稳定域约 kp < 2m/dt²）。按 0.3m/dt² 取值，再夹到旧量级以内。
        mv = float(m.body_mass[self.vert_bodies[0]]) if self.vert_bodies else 1e-3
        self.grasp_kp = float(np.clip(0.3 * mv / self.dt ** 2, 1.0, 400.0))
        self.grasp_kd = float(np.clip(0.6 * mv / self.dt, 0.002, 5.0))
        self.cloth_body = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "blanket")
        if self.cloth_body >= 0:
            self.cloth_body0 = m.body_pos[self.cloth_body].copy()
            self.cloth_quat0 = m.body_quat[self.cloth_body].copy()
        self.tint_geoms = [gid for gid in (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_geom"),
            *(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                f"table_leg_{i}_geom") for i in range(4)))]
        self.tint_geoms = [g for g in self.tint_geoms if g >= 0]

        # ---- 站姿参考（G1 自带 keyframe "stand"）
        self.stand_model = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.stand_key = mujoco.mj_name2id(
            self.stand_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key < 0:
            raise RuntimeError(f"{G1_XML.name} 里没有 keyframe 'stand'")
        self.stand_qpos = self.stand_model.key_qpos[self.stand_key]

        # ---- 遥操作参数
        home = self.args.get("home") or DEFAULT_HOME
        hx, hy, hz = (float(v) for v in home)
        self.home_tcp = {"left": np.array([hx, hy, hz]),
                         "right": np.array([hx, -hy, hz])}
        # 工作域（世界系，左右镜像）：布料中心一侧、盒子一侧。臂展实测可达
        # 约 x≤0.44、|y|≤0.34，工作域按此收紧，避免手柄把目标推到够不到的
        # 地方；近端（身体中线侧）放宽 6cm，让 TCP 能压到贴中线的布面。
        self.workspace = {
            "left":  (np.array([0.26, -0.06, 0.74]),
                     np.array([0.46, 0.34, 1.04])),
            "right": (np.array([0.26, -0.34, 0.74]),
                      np.array([0.46, 0.02, 1.04])),
        }
        # 手柄位移 -> TCP 位移：0.45 比 1:1 更好控制（细活时手柄 1cm 只走
        # 4.5mm），再叠加 HAND_DEADBAND 死区滤掉 PICO 自抖动。
        self.scale = float(self.args.get("scale", 0.45))
        self.grasp_enabled = bool(self.args.get("grasp", True))
        # arms: both | left | right —— 单臂模式只让一侧跟随手柄，另一侧
        # 锁在预备姿态；单臂时只录第三人称 + 该侧腕相机（少一路渲染）。
        arms = str(self.args.get("arms", "both")).lower()
        if arms not in ("both", "left", "right"):
            raise RuntimeError(f"args.arms 只能是 both/left/right，实际是 {arms!r}")
        self.arms = arms
        self.active = list(SIDES) if arms == "both" else [arms]
        self.control = str(self.args.get("control", "osc")).lower()
        if self.control not in ("osc", "ik"):
            raise RuntimeError(
                f"args.control 只能是 osc/ik，实际是 {self.control!r}")
        print(f"[g1_teleop] 控制: {self.control}  活动臂: {arms}"
              f"  scale={self.scale:.2f}", flush=True)

        self.target = {s: self.home_tcp[s].copy() for s in SIDES}
        self.tcp_bias = {s: np.zeros(3) for s in SIDES}   # 实测稳态差积分
        self.prev_hand = {s: None for s in SIDES}
        self.q_cmd = {}
        self.q_goal = {}
        self.grip_cmd = {s: GRIP_OPEN for s in SIDES}
        self.grip_goal = {s: GRIP_OPEN for s in SIDES}
        self.gen = 0
        self.tick = 0
        self.grasp = {s: None for s in SIDES}
        self.grasp_ramp = 0.0
        self._closed = False
        # 内核主循环每 tick 读 done（Simulation.run: `not controller.done`），
        # 收尾时读 success/detail 写进 stats——与 demo 控制器同一约定
        self.done = False
        self.success = False
        self.detail = ""

        # ---- 录制
        self.recording = False
        self.rec_frames = 0
        #: 采集状态：观测 payload 原样下发给 GUI 状态栏（recording/episode/
        #: frames 为实时状态，msg 为最近一次事件，如「已保存 episode 2」）。
        self.status = {"recording": False, "episode": -1, "frames": 0,
                       "msg": "待机：按 PICO A 键开始录制"}
        self._renderers: dict = {}
        self._render_failed = False
        size = int(self.args.get("cam_size", 256))
        cam_names = {"observation.images.cam_high": "showcase"}
        for side in self.active:
            cam_names[f"observation.images.cam_{side}_wrist"] = \
                f"{robot.name}/cam_{side}_wrist"
        self.cam_specs = {k: (v, size, size) for k, v in cam_names.items()}
        state_names = [n.replace("_joint", "") for n in ARM_JOINTS["left"]] \
            + ["left_gripper"] \
            + [n.replace("_joint", "") for n in ARM_JOINTS["right"]] \
            + ["right_gripper"]
        self.dataset_dir = pathlib.Path(
            self.args.get("dataset", DEFAULT_DATASET)).expanduser()
        self.task_text = str(self.args.get("task", DEFAULT_TASK))
        self.writer = LerobotWriter(
            self.dataset_dir, fps=self.fps, dim=16,
            cameras={k: (h, w) for k, (_c, h, w) in self.cam_specs.items()},
            task=self.task_text, robot_type="g1",
            state_names=state_names, action_names=state_names)

        # ---- PICO 服务器
        self.tracker = _DualTracker()
        self.server = _PicoServer(
            int(self.args.get("port", DEFAULT_PORT)), self.tracker,
            pathlib.Path(__file__).with_name("g1_cloth_webxr.html"))
        self.server.start()
        for ip in _local_ips():
            print(f"[g1_teleop] PICO 浏览器打开: http://{ip}:{self.server.port}",
                  flush=True)
        print(f"[g1_teleop] USB 直连（adb reverse tcp:{self.server.port} "
              f"tcp:{self.server.port}）时用: "
              f"http://localhost:{self.server.port}", flush=True)
        print(f"[g1_teleop] 数据集: {self.dataset_dir}  fps={self.fps:.3f}",
              flush=True)

        # ---- 站姿 + 预备姿态
        self.ik_data = mujoco.MjData(m)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._setup_pose()
        self.home_q = self._solve_home()
        if self.arms != "both":
            # 单臂模式：闲置臂竖直下垂收到体侧。不能留在镜像 home（y=±0.06）——
            # 两个 2F85 夹爪中心相距只有 12cm，比夹爪本体还窄，会直接顶在
            # 一起，活动臂的伺服指令与实测关节角差到 0.4rad，根本动不了。
            park = "left" if self.arms == "right" else "right"
            self.home_q[park] = np.zeros(7)
            self.home_q[park][0] = PARK_SHOULDER_PITCH
            print(f"[g1_teleop] {park} 臂竖直下垂禁用（单臂模式）", flush=True)
        self._reset_episode(randomize=False)
        if self.arms != "both":
            self.home_tcp[park] = self.data.site_xpos[
                self.tcp_site[park]].copy()
        print("[g1_teleop] 就绪：A 开始/停止录制 | B 重新对中 | "
              "X 换随机场景 | Y 丢弃本集", flush=True)

    # ------------------------------------------------------------ 姿态

    def _apply_stand_pose(self) -> None:
        """按关节名把 g1_with_hands.xml 的 stand 姿势搬进世界模型。"""
        m, d, sm = self.model, self.data, self.stand_model
        d.qpos[self.base_qpos:self.base_qpos + 7] = \
            self.stand_qpos[0:7]
        for jname, jid in self.joints.items():
            gj = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if gj >= 0:
                d.qpos[m.jnt_qposadr[jid]] = \
                    self.stand_qpos[sm.jnt_qposadr[gj]]
        wj = self.joints.get("waist_pitch_joint")
        if wj is not None:
            d.qpos[m.jnt_qposadr[wj]] = WAIST_PITCH
        # 保持类执行器钉在「初始化时的关节角」，不要每步跟随当前 qpos——
        # 跟随会把重力下垂一路「追认」，腰部会从 0.45rad 缓慢爬到 0.52
        # 限位（实测 50 tick 内就顶死），躯干越趴越低。
        self.hold_target = {aid: float(d.qpos[qadr])
                            for aid, qadr in self.hold}
        self.base_pose = d.qpos[self.base_qpos:self.base_qpos + 7].copy()

    def _setup_pose(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._apply_stand_pose()
        # 手臂静息取零位（自然下垂略前伸），IK 从这里出发不会撞躯干
        for side in SIDES:
            d.qpos[self.arm_qadrs[side]] = 0.0
        mujoco.mj_forward(m, d)
        self._rest = {s: d.qpos[self.arm_qadrs[s]].copy() for s in SIDES}

    def _solve_ik(self, side: str, target: np.ndarray, seed: np.ndarray,
                  iters: int = 48) -> tuple[np.ndarray, float]:
        """阻尼最小二乘 IK（只动该侧 7 个关节），返回 (关节角, 位置残差)。

        在独立的 MjData 上迭代，绝不污染真实物理状态（否则每次解 IK 都会
        把 qpos 拽到解上，产生跳变）。
        """
        m, ik = self.model, self.ik_data
        ik.qpos[:] = self.data.qpos
        qadrs, dofs = self.arm_qadrs[side], self.arm_dofs[side]
        lo, hi = self.arm_lo[side], self.arm_hi[side]
        ik.qpos[qadrs] = seed
        sid, bid = self.tcp_site[side], self.tcp_body[side]
        lam = 0.02 ** 2
        for _ in range(iters):
            mujoco.mj_kinematics(m, ik)
            mujoco.mj_comPos(m, ik)
            p = ik.site_xpos[sid].copy()
            ep = target - p
            R = ik.site_xmat[sid].reshape(3, 3)
            er = _quat_err(R_TOOL[side], R)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            mujoco.mj_jac(m, ik, self._jacp, self._jacr, p, bid)
            J = np.vstack([self._jacp[:, dofs], self._jacr[:, dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6),
                                       np.concatenate([ep, er]))
            ik.qpos[qadrs] = np.clip(ik.qpos[qadrs] + 0.5 * dq, lo, hi)
        mujoco.mj_kinematics(m, ik)
        mujoco.mj_comPos(m, ik)
        err = float(np.linalg.norm(target - ik.site_xpos[sid]))
        return ik.qpos[qadrs].copy(), err

    def _osc_step(self, side: str, target: np.ndarray
                  ) -> tuple[np.ndarray, float]:
        """OSC：目标误差 -> 任务空间速度 -> 关节速度增量 -> 关节目标。

        只解一次雅可比（绝对 IK 要迭代 48 次），且天然带速度上限：PICO
        毫米级抖动只会变成平滑的小速度，不会像绝对 IK 那样每 tick 把关节
        拽到新解上——7 自由度冗余臂跨分支时末端会瞬跳，这正是之前「太灵敏、
        不好控制」的主因。零空间项把多余自由度拉回预备构型，避免肘腕漂到
        限位。
        """
        m, ik = self.model, self.ik_data
        qadrs, dofs = self.arm_qadrs[side], self.arm_dofs[side]
        lo, hi = self.arm_lo[side], self.arm_hi[side]
        q = self.q_cmd[side].copy()
        ik.qpos[:] = self.data.qpos
        ik.qpos[qadrs] = q
        mujoco.mj_kinematics(m, ik)
        mujoco.mj_comPos(m, ik)
        sid, bid = self.tcp_site[side], self.tcp_body[side]
        p = ik.site_xpos[sid].copy()
        R = ik.site_xmat[sid].reshape(3, 3)
        # 内环用「指令角」做运动学：伺服有滞后，直接用实测角做高增益反馈会
        # 正反馈放大成 80mm/tick 的抖动。重力/伺服下垂（实测 ~2cm）交给下面
        # 的慢速积分偏置消除——时间常数 ~0.7s，不会引起振荡。
        ep = target + self.tcp_bias[side] - p
        er = _quat_err(R_TOOL[side], R)
        vp = OSC_KP_POS * ep
        n = float(np.linalg.norm(vp))
        if n > OSC_VMAX:
            vp *= OSC_VMAX / n
        wr = OSC_KP_ROT * er
        n = float(np.linalg.norm(wr))
        if n > OSC_WMAX:
            wr *= OSC_WMAX / n
        mujoco.mj_jac(m, ik, self._jacp, self._jacr, p, bid)
        J = np.vstack([self._jacp[:, dofs], self._jacr[:, dofs]])
        lam = OSC_LAMBDA ** 2
        Jpinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        dq = Jpinv @ np.concatenate([vp, wr])
        N = np.eye(len(dofs)) - Jpinv @ J        # 零空间投影
        qd = self.data.qvel[dofs]
        dq += N @ (OSC_KP_POSTURE * (self.home_q[side] - q)
                   - OSC_KD_POSTURE * qd)
        # 增量作用在指令角上（实测角只用于慢速积分偏置，不参与内环）
        q_new = np.clip(self.q_cmd[side] + dq * self.tick_dt, lo, hi)
        # 慢速积分：把「实测末端 - 目标」的稳态差补进偏置（限幅 OSC_BIAS_MAX）。
        # 误差大时（大步跳变/够不到）不积分并缓慢衰减，避免积分饱和把
        # 目标一路推到域外。
        p_meas = self.data.site_xpos[sid]
        err_meas = target - p_meas
        if np.linalg.norm(err_meas) < OSC_BIAS_GATE:
            self.tcp_bias[side] = np.clip(
                self.tcp_bias[side] + OSC_KI * err_meas * self.tick_dt,
                -OSC_BIAS_MAX, OSC_BIAS_MAX)
        else:
            self.tcp_bias[side] *= 0.97
        return q_new, float(np.linalg.norm(target - p_meas))

    def _solve_home(self) -> dict:
        """解一次预备姿态：以自然姿态为种子，多随机种子取最优。

        评分 = 位置残差 + 朝向残差 + 关节余量 + 姿态距离。只看位置残差会
        选出「肩 pitch/yaw 顶到限位」的畸形解：运动学上完全收敛（<2mm），
        但 kp=500 的伺服顶不动，实测一跑手臂就下垂 12cm、朝向偏 36°。
        姿态距离项让解留在 HOME_SEED 所在的自然分支上。
        """
        rng = np.random.default_rng(11)
        solved: dict[str, np.ndarray] = {}
        m, ik = self.model, self.ik_data
        for side in SIDES:
            seed0 = HOME_SEED * (MIRROR_SIGN if side == "right" else 1.0)
            seeds = [seed0]
            for scale in (0.3, 0.3, 0.6, 0.6):
                seeds.append(seed0 + rng.uniform(-scale, scale, 7))
            seeds.append(self._rest[side])
            if side == "right" and "left" in solved:
                seeds.append(solved["left"] * MIRROR_SIGN)
            best = None
            for seed in seeds:
                q, err = self._solve_ik(side, self.home_tcp[side], seed,
                                        iters=150)
                ik.qpos[:] = self.data.qpos
                ik.qpos[self.arm_qadrs[side]] = q
                mujoco.mj_kinematics(m, ik)
                mujoco.mj_comPos(m, ik)
                R = ik.site_xmat[self.tcp_site[side]].reshape(3, 3)
                eori = float(np.linalg.norm(_quat_err(R_TOOL[side], R)))
                margin = float(np.minimum(q - self.arm_lo[side],
                                          self.arm_hi[side] - q).min())
                score = (err + 0.02 * eori
                         + 0.05 * max(0.0, 0.30 - margin)
                         + 0.02 * float(np.linalg.norm(q - seed0)))
                if best is None or score < best[0]:
                    best = (score, err, eori, margin, q)
            _, err, eori, margin, q = best
            solved[side] = q
            flag = "" if err < 0.02 else "  ← 不可达，检查 args.home"
            print(f"[g1_teleop] 预备姿态 {side:5s} 位置残差 {err * 1000:5.1f}mm"
                  f" 朝向 {math.degrees(eori):4.1f}° 关节余量 {margin:.2f}rad"
                  f"{flag}", flush=True)
        return solved

    # ------------------------------------------------------------ 复位 / 随机化

    def _randomize_scene(self) -> None:
        """换一套外观与摆放：布料颜色/朝向/位置、桌面颜色、灯光。"""
        m, rng = self.model, self.rng
        if self.flex_id >= 0 and hasattr(m, "flex_rgba"):
            c = _CLOTH_COLORS[rng.integers(len(_CLOTH_COLORS))].copy()
            c[:3] = np.clip(c[:3] + rng.uniform(-0.06, 0.06, 3), 0.05, 0.95)
            m.flex_rgba[self.flex_id] = c
        if self.cloth_body >= 0:
            # 平铺布要始终躺在桌面上：平移 ±1cm 级、偏航 ±5°，避免布角探出桌沿。
            # y 向只取 ±1cm：布近边在 y=0.09，盒壁外表面在 y=0.10，随机 +1cm
            # 再叠加偏航的角点外移刚好贴面；再大就会开局就插进盒壁（接触力
            # 会把布弹开，看起来像「布料自己跳了一下」）。
            yaw = float(rng.uniform(-0.08, 0.08))
            dpos = np.array([rng.uniform(-0.006, 0.006),
                             rng.uniform(-0.010, 0.010),
                             rng.uniform(-0.002, 0.002)])
            m.body_pos[self.cloth_body] = self.cloth_body0 + dpos
            m.body_quat[self.cloth_body] = _quat_mul(
                [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                self.cloth_quat0)
        if self.tint_geoms:
            base = _TABLE_COLORS[rng.integers(len(_TABLE_COLORS))]
            for gid in self.tint_geoms:
                c = base.copy()
                c[:3] = np.clip(c[:3] + rng.uniform(-0.05, 0.05, 3), 0.05, 0.95)
                m.geom_rgba[gid] = c
        if int(m.nlight) > 0:
            # 灯光随机（两盏全随机）：位置在桌面上空半球内、光轴始终指向
            # 工作区中心，色温在暖黄~冷白间插值（各通道独立随机会出现
            # 绿紫怪光）。0 号主灯保证照度，1 号辅灯从远方位打轮廓光。
            for i in range(int(m.nlight)):
                main = (i == 0)
                spread = 0.7 if main else 1.5
                pos = np.array([
                    0.4 + rng.uniform(-spread, spread),
                    rng.uniform(-spread, spread),
                    rng.uniform(2.2, 3.2) if main
                    else rng.uniform(1.8, 3.0)])
                m.light_pos[i] = pos
                d = np.array([0.4, 0.0, 0.8]) - pos
                m.light_dir[i] = d / np.linalg.norm(d)
                if hasattr(m, "light_diffuse"):
                    t = float(rng.uniform(0.0, 1.0))    # 0=冷白 1=暖黄
                    base = rng.uniform(0.32, 0.55) if main \
                        else rng.uniform(0.12, 0.30)
                    m.light_diffuse[i] = [
                        base * (1.0 + 0.18 * t), base, base * (1.0 - 0.18 * t)]
                if hasattr(m, "light_ambient"):
                    a = float(rng.uniform(0.0, 0.08))
                    m.light_ambient[i] = [a, a, a]

    def _reset_episode(self, randomize: bool = True) -> None:
        """布料/桌面/灯光随机化 + 机器人回预备姿态 + 清空抓持状态。"""
        m, d = self.model, self.data
        if randomize:
            self._randomize_scene()
        t = float(d.time)
        mujoco.mj_resetData(m, d)
        d.time = t
        self._apply_stand_pose()
        for side in SIDES:
            d.qpos[self.arm_qadrs[side]] = self.home_q[side]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.q_cmd = {s: self.home_q[s].copy() for s in SIDES}
        self.q_goal = {s: self.home_q[s].copy() for s in SIDES}
        self.target = {s: self.home_tcp[s].copy() for s in SIDES}
        self.tcp_bias = {s: np.zeros(3) for s in SIDES}
        self.grip_cmd = {s: GRIP_OPEN for s in SIDES}
        self.grip_goal = {s: GRIP_OPEN for s in SIDES}
        self.grasp = {s: None for s in SIDES}
        self.grasp_ramp = 0.0
        self.prev_hand = {s: None for s in SIDES}
        self._write_ctrl()

    # ------------------------------------------------------------ 每 tick

    def _update_targets(self) -> None:
        """手柄位移增量 -> TCP 目标（每 tick 一次，夹回工作域盒内）。"""
        now = time.monotonic()
        if self.tracker.generation != self.gen:
            self.gen = self.tracker.generation
            self.prev_hand = {s: None for s in SIDES}
        for side in self.active:
            msg = self.tracker.latest[side]
            if msg is None or now - self.tracker.recv[side] > CTRL_TIMEOUT:
                continue
            p = _T @ np.asarray(msg["p"], dtype=float)
            prev = self.prev_hand[side]
            if prev is None:
                prev = p.copy()
            delta = p - prev
            self.prev_hand[side] = p
            trig = float(np.clip(msg.get("trig", 0.0), 0.0, 1.0))
            self.grip_goal[side] = GRIP_OPEN + (GRIP_CLOSE - GRIP_OPEN) * trig
            if np.linalg.norm(delta) < HAND_DEADBAND:
                continue                        # 静置时的毫米级抖动
            self.target[side] = self.target[side] + delta * self.scale
            lo, hi = self.workspace[side]
            self.target[side] = np.clip(self.target[side], lo, hi)

    def _consume_buttons(self) -> None:
        tr = self.tracker
        while tr.press["a"] > 0:
            tr.press["a"] -= 1
            self._toggle_recording()
        while tr.press["b"] > 0:
            tr.press["b"] -= 1
            self.prev_hand = {s: None for s in SIDES}
            tr.generation += 1
            self.server.broadcast({"buzz": 60})
            print("[g1_teleop] 已重新对中（手柄当前位置为新基准）", flush=True)
        while tr.press["x"] > 0:
            tr.press["x"] -= 1
            self._reset_episode()
            self.server.broadcast({"buzz": 60})
            if not self.recording:
                self._set_status(False, "已换随机场景，按 A 开始录制")
            print("[g1_teleop] 已换一套随机场景并回预备姿态", flush=True)
        while tr.press["y"] > 0:
            tr.press["y"] -= 1
            if self.recording:
                try:
                    self.writer.abort_episode()
                except Exception as e:              # noqa: BLE001
                    print(f"[g1_teleop] 丢弃失败: {e}", flush=True)
                self.recording = False
                self._set_status(False, f"已丢弃 episode {self.status['episode']}")
                self.server.broadcast({"rec": False, "buzz": 120})
                print("[g1_teleop] 已丢弃当前这一集", flush=True)

    def _set_status(self, recording: bool | None = None,
                    msg: str | None = None) -> None:
        """更新采集状态（观测 payload 会把它带给 GUI 状态栏）。"""
        if recording is not None:
            self.status["recording"] = bool(recording)
        if msg is not None:
            self.status["msg"] = msg

    def _toggle_recording(self) -> None:
        if not self.recording:
            try:
                self._reset_episode()
                ep = self.writer.start_episode()
                self.recording = True
                self.rec_frames = 0
                self.status.update({"recording": True, "episode": ep,
                                    "frames": 0, "msg": ""})
                self.server.broadcast({"rec": True, "buzz": 150})
                print(f"[g1_teleop] ▶ 开始录制 episode {ep}", flush=True)
            except Exception as e:                  # noqa: BLE001
                self.recording = False
                self._set_status(False, f"开始录制失败：{e}")
                print(f"[g1_teleop] 开始录制失败: {e}", flush=True)
        else:
            self.recording = False
            try:
                info = self.writer.end_episode()
                n = self.rec_frames
                self._set_status(False,
                                 f"已保存 episode {self.status['episode']}"
                                 f"（{n} 帧）")
                print(f"[g1_teleop] ■ 停止录制: {info}", flush=True)
            except Exception as e:                  # noqa: BLE001
                self._set_status(False, f"收尾失败：{e}")
                print(f"[g1_teleop] 收尾失败: {e}", flush=True)
            self.server.broadcast({"rec": False, "buzz": 150})

    def advance(self) -> bool:
        if self._closed:
            return False
        self._consume_buttons()
        self._update_targets()
        for side in SIDES:
            if side not in self.active:
                # 单臂模式：非活动臂锁在预备姿态、夹爪保持张开
                self.q_goal[side] = self.home_q[side].copy()
                self.grip_goal[side] = GRIP_OPEN
                continue
            if self.control == "osc":
                q, err = self._osc_step(side, self.target[side])
            else:
                q, err = self._solve_ik(side, self.target[side],
                                        self.q_cmd[side])
            self.q_goal[side] = q
            if err > 0.035 and self.control == "ik":
                # 绝对 IK 超出可达域时把目标朝当前 TCP 回吸；OSC 本身速度
                # 饱和，不能回吸（否则目标被一路拖回，永远到不了位）
                tcp = self.data.site_xpos[self.tcp_site[side]]
                self.target[side] += (tcp - self.target[side]) * \
                    min(1.0, 2.5 * self.tick_dt)
        for _ in range(self.ctrl_period):
            self._substep()
        self.tick += 1
        if self.recording:
            self._record()
        return True

    def _substep(self) -> None:
        m, d = self.model, self.data
        grip_step = GRIP_SPEED * self.dt
        for side in SIDES:
            self.grip_cmd[side] += float(np.clip(
                self.grip_goal[side] - self.grip_cmd[side],
                -grip_step, grip_step))
        jstep = MAX_JOINT_SPEED * self.dt
        for side in SIDES:
            self.q_cmd[side] = self.q_cmd[side] + np.clip(
                self.q_goal[side] - self.q_cmd[side], -jstep, jstep)

        if self.grasp_enabled:
            for side in SIDES:
                closed = self.grip_cmd[side] > GRASP_ON
                if closed and self.grasp[side] is None:
                    self._lock_cloth(side)
                elif not closed and self.grasp[side] is not None:
                    self._release_grasp(side)

        self._write_ctrl()
        # 浮动基座锁死（qpos 与 qvel 都钉住，否则人形会倒）
        d.qpos[self.base_qpos:self.base_qpos + 7] = self.base_pose
        d.qvel[self.base_dof:self.base_dof + 6] = 0.0
        self._apply_grasp()
        mujoco.mj_step(m, d)

    def _write_ctrl(self) -> None:
        d = self.data
        for aid, _qadr in self.hold:
            d.ctrl[aid] = self.hold_target[aid]
        for side in SIDES:
            for k, name in enumerate(ARM_JOINTS[side]):
                d.ctrl[self.actuators[name]] = self.q_cmd[side][k]
            d.ctrl[self.grip_act[side]] = self.grip_cmd[side]

    # ------------------------------------------------------------ 布料抓持

    def _lock_cloth(self, side: str) -> None:
        """把两指之间正在被夹住的布顶点锁到手指局部坐标上。

        必须先真的碰到布：TCP 到最近布顶点超过 2.5cm 时合指不建立抓持。
        旧实现只按「TCP 下方 8cm、半径 6.5cm 的一摊」选点，夹爪离布面还有
        5cm 时合指也会凭空吸附整块布——手一抬布就跟着飞（看起来像布黏在
        夹爪上），拖动时还会把布硬拽进盒壁/桌面造成穿模。
        选点窗口同时收紧到指尖夹持区：两指之间（TCP 上方 5cm）到 TCP 下方
        2cm、横向 4cm 内，只锁真正被夹住的那一小片。
        """
        if self.flex_id < 0:
            return
        d = self.data
        p = d.site_xpos[self.tcp_site[side]].copy()
        R = d.site_xmat[self.tcp_site[side]].reshape(3, 3).copy()
        verts = d.flexvert_xpos[self.vert_adr:self.vert_adr + self.vert_num]
        local = (verts - p) @ R            # = R^T (v - p)；local +z 朝下
        near = np.linalg.norm(local, axis=1)
        if near.min() > 0.025:             # 手指还没碰到布
            self.grasp[side] = None
            return
        sel = ((np.abs(local[:, 0]) < 0.04)
               & (np.abs(local[:, 1]) < 0.04)
               & (local[:, 2] > -0.05)     # 两指之间的夹持区（TCP 上方）
               & (local[:, 2] < 0.02)      # TCP 下方 2cm 内
               & (near < 0.05))
        idx = np.nonzero(sel)[0]
        if idx.size == 0:
            self.grasp[side] = None
            return
        order = idx[np.argsort(near[idx])][:18]
        self.grasp[side] = (order, local[order].copy())
        self.grasp_ramp = 0.0
        print(f"[g1_teleop] {side} 夹住 {len(order)} 个布顶点", flush=True)

    def _release_grasp(self, side: str) -> None:
        g = self.grasp[side]
        if g is not None:
            for i in g[0]:
                self.data.xfrc_applied[self.vert_bodies[int(i)], :] = 0.0
        self.grasp[side] = None

    def _apply_grasp(self) -> None:
        """弹簧-阻尼把锁定顶点拖到手指局部偏移上（力施加在世界系）。"""
        if not any(self.grasp.values()):
            return
        self.grasp_ramp = min(1.0, self.grasp_ramp + self.dt / 0.12)
        d = self.data
        kp, kd = self.grasp_kp, self.grasp_kd
        for side in SIDES:
            g = self.grasp[side]
            if g is None:
                continue
            p = d.site_xpos[self.tcp_site[side]]
            R = d.site_xmat[self.tcp_site[side]].reshape(3, 3)
            idx, local = g
            for i, off in zip(idx, local):
                b = self.vert_bodies[int(i)]
                dof = self.vert_dof[int(i)]
                target = p + R @ off
                f = kp * (target - d.xpos[b]) - kd * d.qvel[dof:dof + 3]
                d.xfrc_applied[b, :3] = self.grasp_ramp * f
                d.xfrc_applied[b, 3:] = 0.0

    # ------------------------------------------------------------ 录制

    def _render(self, key: str) -> np.ndarray:
        cam, h, w = self.cam_specs[key]
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, h, w)
            self._renderers[key] = r
        return render_offscreen(self.model, self.data, w, h,
                                camera=cam, renderer=r)

    def _grip_state(self, side: str) -> float:
        q = float(self.data.qpos[self.grip_drv_qadr[side]])
        return float(np.clip(q / 0.8, 0.0, 1.0))   # 驱动关节行程 0~0.8

    def _record(self) -> None:
        d = self.data
        state = np.concatenate([
            d.qpos[self.arm_qadrs["left"]], [self._grip_state("left")],
            d.qpos[self.arm_qadrs["right"]], [self._grip_state("right")],
        ]).astype(np.float32)
        action = np.concatenate([
            self.q_cmd["left"], [self.grip_cmd["left"] / 255.0],
            self.q_cmd["right"], [self.grip_cmd["right"] / 255.0],
        ]).astype(np.float32)
        try:
            images = {k: self._render(k) for k in self.cam_specs}
        except Exception as e:                      # noqa: BLE001
            if not self._render_failed:
                self._render_failed = True
                print(f"[g1_teleop] 相机渲染失败，停止录制: {e}", flush=True)
            self.recording = False
            try:
                self.writer.abort_episode()
            except Exception:                       # noqa: BLE001
                pass
            return
        try:
            self.writer.add_frame(state, action, images)
            self.rec_frames += 1
            self.status["frames"] = self.rec_frames
        except Exception as e:                      # noqa: BLE001
            print(f"[g1_teleop] 写帧失败，停止录制: {e}", flush=True)
            self.recording = False

    # ------------------------------------------------------------ 收尾

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.done = True
        if self.recording:
            self.recording = False
            try:
                print(f"[g1_teleop] 收尾录制: {self.writer.end_episode()}",
                      flush=True)
            except Exception as e:                  # noqa: BLE001
                print(f"[g1_teleop] 收尾录制失败: {e}", flush=True)
        self.server.stop()
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:                       # noqa: BLE001
                pass
        self._renderers.clear()


def create_controller(simulation):
    return G1ClothTeleop(simulation)
