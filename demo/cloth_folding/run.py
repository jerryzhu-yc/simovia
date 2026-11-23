"""宇树 G1 双手叠毛毯：MuJoCo 4 柔性布料（flexcomp）+ 双臂 IK + 合规抓持力模型。

物理链路：
1. 毛毯是 flexcomp 二维网格（YAML 里 type: cloth，dim=2）：每个网格点是带 3 个
   世界轴滑块的质点，膜/弯刚度维持形状；重力、桌面接触、自碰撞
   （self-collide=narrow）都是真实仿真，对折后上下两层不会互相穿透。
2. 双臂「离线关键帧 IK + 关节空间余弦插值」：启动时把每个关键帧目标和姿态用
   阻尼最小二乘解成 7 个手臂关节角，运行中平滑插值，不会漂到关节限位。
3. 抓取用手部附近的布顶点「合规抓持力」实现：合指后锁定手指间的布顶点，用
   弹簧-阻尼力（有限刚度）让它们跟随腕部位姿，松手即解除。原因：G1 三指功率
   手的拇指尖与食指尖竖直错开约 9cm，捏不住 1cm 级薄布（实测捏空或顶穿）；
   硬钉顶点会撕布，所以改用有限刚度力模拟「布被稳稳夹住」，布料本体仍是完整
   柔体仿真，不是刚体摆拍。
4. 提起 → 前移 → 落在原布后半幅上，形成一次对折；松手后双手沿桌面同高度后撤
   （抬高会勾住布料把折边带起）。受手形限制，折峰无法压平。

两种入口共用 ClothFold 核心：
- 命令行独立跑（可录视频）：本文件 main()，自建世界。
- 软件内 GUI 预览：demo/cloth_folding/controller.py 以 shared_world_controller
  身份跑在内核进程里，直接操作内核的 model/data。

用法：
    python -m demo.cloth_folding.run                    # 跑一遍并打印折叠指标
    python -m demo.cloth_folding.run --video out.mp4    # 同时录制视频
"""
from __future__ import annotations

import argparse
import math
import pathlib

import mujoco
import numpy as np

from simovia.world.builder import build_world
from simovia.world.schema import load_world

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORLD = ROOT / "worlds" / "g1_cloth_fold.yaml"
G1_XML = ROOT / "models" / "zoo" / "unitree_g1" / "g1_with_hands.xml"

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
#: 右臂关节相对左臂的镜像符号（对称关节取反，pitch/elbow 保持同号）。
#: 七自由度冗余臂在靠近身体的区域容易陷局部极小，用左臂解镜像过来当种子最稳。
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

FINGER_OPEN = {"thumb_0": 0.0, "thumb_1": 0.0, "thumb_2": 0.0,
               "index_0": 0.0, "index_1": 0.0, "middle_0": 0.0, "middle_1": 0.0}
#: 捏住布边的半闭合（布厚 4mm，完全闭合会空捏，留一点夹持余量）
FINGER_GRIP = {"thumb_0": 0.45, "thumb_1": 0.65, "thumb_2": 0.85,
               "index_0": -1.10, "index_1": -1.30,
               "middle_0": -0.20, "middle_1": -0.25}

# 站姿时躯干前倾一点：G1 臂展只有 0.40m（肩→腕）+0.13m 手长，
# 桌面在身体前方时，前倾能把捏合点前探约 3cm。
WAIST_PITCH = 0.20


def _sym(x, y, z):
    """左右对称的捏合点目标（y 取相反数）。"""
    return {"right": np.array((x, -y, z), float), "left": np.array((x, y, z), float)}


# 捏住垂边对折：毛毯近半幅探出桌沿自然垂下（布边 x≈0.20~0.22，z 0.62~0.74），
# 双手竖着捏住垂边两侧（y=±0.16，留出布宽余量）提起，前移过中线后放下。
KEYFRAMES = {
    "home": _sym(0.235, 0.160, 0.885),       # 布边上悬停
    "touch": _sym(0.195, 0.160, 0.695),      # 捏合点对准垂边中部
    "lift": _sym(0.200, 0.160, 0.845),       # 提起垂边到桌面上方
    "over": _sym(0.275, 0.160, 0.935),       # 前移过中线
    "down": _sym(0.295, 0.160, 0.825),       # 落到远半幅上（手指不扎进桌面）
    "back": _sym(0.200, 0.160, 0.825),       # 松手后沿桌面后撤，把手从折边里抽出来
}
KEYFRAME_ORDER = ["home", "touch", "lift", "over", "down", "back"]

# (阶段名, 时长秒, 起始关键帧, 结束关键帧, 阶段动作)
PLAN = [
    ("settle", 1.0, "home", "home", None),
    ("approach", 1.2, "home", "touch", None),
    ("grip", 0.5, "touch", "touch", "grip"),
    ("lift", 1.8, "touch", "lift", None),
    ("carry", 2.0, "lift", "over", None),
    ("down", 1.4, "over", "down", None),
    ("release", 0.8, "down", "down", "release"),
    # 沿桌面同高度后撤：手从折边两层之间滑出。抬高会让手指勾住布料把折边带起，
    # 横向张开会把布挤成窄条（两种都实测失败），只有平着后撤是干净的。
    ("retract", 1.8, "down", "back", None),
    ("settle", 0.8, "back", "back", None),
]


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


class ClothFold:
    """叠毛毯核心：附着在 (model, index, data) 上逐物理步推进。"""

    def __init__(self, model, index, data):
        self.model, self.index, self.data = model, index, data
        m = self.model

        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"g1/{n}")
                    for n in ("pelvis", "waist_pitch_link",
                              "left_wrist_yaw_link", "right_wrist_yaw_link",
                              "left_hand_thumb_2_link", "left_hand_index_1_link",
                              "right_hand_thumb_2_link", "right_hand_index_1_link")}
        self.act = dict(self.index.robots["g1"].actuators)

        free_jid = next(j for j in range(m.njnt)
                        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.base_qpos = int(m.jnt_qposadr[free_jid])
        self.base_dof = int(m.jnt_dofadr[free_jid])

        self.arm_dofs, self.arm_qadr = {}, {}
        self.arm_lo, self.arm_hi = {}, {}
        for side, names in ARM_JOINTS.items():
            joints = [self.index.robots["g1"].joints[n] for n in names]
            self.arm_dofs[side] = np.array([int(m.jnt_dofadr[j]) for j in joints])
            self.arm_qadr[side] = np.array([int(m.jnt_qposadr[j]) for j in joints])
            rng = np.array([m.jnt_range[j] for j in joints])
            self.arm_lo[side], self.arm_hi[side] = rng[:, 0], rng[:, 1]
        # 肩 roll 只允许「外展」方向：模型 stand 姿势把两臂内旋到 roll=∓1.16，
        # 肩壳直接插进躯干，物理上顶到 75Nm 也压不进去（实测手臂差 1rad 到不了位）。
        # 把 IK 的内旋自由度锁掉，手臂才能在躯干外侧自由前伸。
        self.arm_lo["left"][1] = max(self.arm_lo["left"][1], 0.0)
        self.arm_hi["right"][1] = min(self.arm_hi["right"][1], 0.0)

        self.nvert = int(m.nflexvert)
        # flexcomp 网格的每个顶点是一个带 3 个滑块（世界轴向）的质点 body，
        # 因此可以直接给它加世界系外力——这是实现的抓持力模型的基础。
        self.vert_bodies = [int(m.flex_vertbodyid[v]) for v in range(self.nvert)]
        self.vert_dof = np.array([int(m.body_dofadr[b]) for b in self.vert_bodies])
        self.grasp: dict[str, tuple] = {s: None for s in SIDES}
        self.grasp_ramp = 0.0
        self.cloth0 = None
        self.hand_rot: dict[str, np.ndarray] = {}

        self.dt = float(m.opt.timestep)
        self.phase_i = 0
        self.phase_t = 0.0
        self.finished = False
        self.finger_target = dict(FINGER_OPEN)
        self._action_done = False
        self._gripping = False

        # 站姿参考：G1 自带 keyframe "stand"，按关节名逐个搬进世界模型
        self.stand_model = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.stand_key = mujoco.mj_name2id(
            self.stand_model, mujoco.mjtObj.mjOBJ_KEY, "stand")

        self._setup_pose()
        print("[cloth_fold] 离线求解双臂关键帧 IK……")
        self.solved = self._solve_keyframes()
        self._set_arms(self.solved["home"], write_ctrl=True)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ 初始化

    def _setup_pose(self):
        """机器人摆到前倾站姿，伺服目标对齐当前姿态。"""
        m, d, sm = self.model, self.data, self.stand_model
        mujoco.mj_resetData(m, d)
        d.qpos[self.base_qpos:self.base_qpos + 7] = \
            sm.key_qpos[self.stand_key][0:7]
        for jname, jid in self.index.robots["g1"].joints.items():
            gj = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if gj >= 0:
                d.qpos[m.jnt_qposadr[jid]] = \
                    sm.key_qpos[self.stand_key][sm.jnt_qposadr[gj]]
        # stand 的腿/腰照用，但手臂不能照用：stand 把双臂内旋贴到躯干上，
        # 肩部碰撞壳互相插着，物理一跑就会被弹到外展位置。手臂静息取零位
        # （自然下垂略前伸），IK 从这里出发也不会撞躯干。
        for side in SIDES:
            d.qpos[self.arm_qadr[side]] = 0.0
        wj = self.index.robots["g1"].joints["waist_pitch_joint"]
        d.qpos[m.jnt_qposadr[wj]] = WAIST_PITCH
        self.base_pose = d.qpos[self.base_qpos:self.base_qpos + 7].copy()

        mujoco.mj_forward(m, d)
        if self.cloth0 is None:
            self.cloth0 = d.flexvert_xpos[:self.nvert].copy()
        self._set_hand_orientation()
        self._rest = {s: d.qpos[self.arm_qadr[s]].copy() for s in SIDES}
        # 伺服目标先对齐当前姿态：否则 ctrl=0 会让腰、腿、手指一起往零位收，
        # 躯干回正把手带离目标点（实测偏 3~4cm）
        for ai in self.act.values():
            jid = int(m.actuator_trnid[ai, 0])
            d.ctrl[ai] = d.qpos[m.jnt_qposadr[jid]]

    def _set_hand_orientation(self):
        """捏取手形：手指朝下、掌心朝前（正对垂下来的布边）。

        平铺在桌面上的布捏不住（布下就是桌面，手指插不进去），所以场景里让毛毯
        近边探出桌沿 6cm 自然垂下，形成一块竖直布边。掌心翻转成朝前（+x）、
        手指朝下（-z），拇指与食指就从布边前后两侧合拢，真实夹住布边。

        手形仍从模型几何取：手指沿局部 +x 伸出，拇指朝掌心方向卷曲（左手 ±y
        侧为 -y、右手为 +y），「掌心朝外」法线是 ∓y。把局部的 [手指, 掌心法线]
        映射到世界的 [正下方, 正前方]，两侧手形就对称。
        """
        fwd_w = np.array([0.0, 0.0, -1.0])      # 手指朝下
        nrm_w = np.array([1.0, 0.0, 0.0])       # 掌心朝前（对着布边）
        basis_w = np.column_stack([fwd_w, nrm_w, np.cross(fwd_w, nrm_w)])
        for s in SIDES:
            n_l = np.array([0.0, -1.0 if s == "left" else 1.0, 0.0])
            f_l = np.array([1.0, 0.0, 0.0])
            basis_l = np.column_stack([f_l, n_l, np.cross(f_l, n_l)])
            self.hand_rot[s] = basis_w @ basis_l.T

    def _fwd_fast(self):
        """只算运动学（跳过碰撞检测）：IK 迭代用，比 mj_forward 快两个量级。"""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    # ------------------------------------------------------------ 手臂 IK

    def _pinch_point(self, side: str) -> np.ndarray:
        d = self.data
        return 0.5 * (d.xpos[self.bid[f"{side}_hand_thumb_2_link"]]
                      + d.xpos[self.bid[f"{side}_hand_index_1_link"]])

    def _solve_ik(self, side: str, target: np.ndarray, seed: np.ndarray,
                  iters: int = 120, lam: float = 0.02, gain: float = 0.5):
        """从 seed 出发解 IK，返回 (位置残差, 朝向残差, 关节角)。"""
        m, d = self.model, self.data
        bid = self.bid[f"{side}_wrist_yaw_link"]
        dofs, qadrs = self.arm_dofs[side], self.arm_qadr[side]
        lo, hi = self.arm_lo[side], self.arm_hi[side]
        target = np.asarray(target, float)
        d.qpos[qadrs] = seed
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        er = np.zeros(3)
        for _ in range(iters):
            self._fwd_fast()
            pinch = self._pinch_point(side)
            mujoco.mj_jac(m, d, jacp, jacr, pinch, bid)
            ep = target - pinch
            er = _quat_err(self.hand_rot[side], d.xmat[bid].reshape(3, 3))
            jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6),
                                         np.concatenate([ep, er]))
            d.qpos[qadrs] = np.clip(d.qpos[qadrs] + gain * dq, lo, hi)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
        self._fwd_fast()
        return (float(np.linalg.norm(target - self._pinch_point(side))),
                float(np.linalg.norm(er)), d.qpos[qadrs].copy())

    def _solve_keyframes(self) -> dict:
        """逐关键帧解 IK：前一个关键帧的解当主种子，另加若干随机种子取最优。"""
        rng = np.random.default_rng(7)
        solved, prev = {}, self._rest
        for name in KEYFRAME_ORDER:
            solved[name], res = {}, {}
            for side in SIDES:
                target = KEYFRAMES[name][side]
                span = self.arm_hi[side] - self.arm_lo[side]
                seeds = [prev[side]]
                if side == "right":
                    seeds.append(solved[name]["left"] * MIRROR_SIGN)
                seeds += [self._rest[side] + rng.uniform(-0.6, 0.6, 7) * span * 0.25
                          for _ in range(10)]
                best = None
                for seed in seeds:
                    pos_res, rot_res, q = self._solve_ik(side, target, seed)
                    score = pos_res + 5e-4 * rot_res + 0.02 * float(np.linalg.norm(q - prev[side]))
                    if best is None or score < best[0]:
                        best = (score, pos_res, rot_res, q)
                _, pos_res, rot_res, q = best
                solved[name][side] = q
                res[side] = (pos_res, rot_res)
                flag = "" if pos_res < 0.01 else "  ← 不可达"
                print(f"[ik] {name:8s} {side:5s} 位置残差 {pos_res * 1000:5.1f}mm  "
                      f"朝向残差 {math.degrees(rot_res):5.1f}°{flag}")
            # 左右是镜像结构：一侧陷在不可达分支时，用另一侧的镜像解重解一次
            for side, other in (("left", "right"), ("right", "left")):
                if res[side][0] <= 0.008:
                    continue
                pos_res, rot_res, q = self._solve_ik(
                    side, KEYFRAMES[name][side], solved[name][other] * MIRROR_SIGN)
                if pos_res < res[side][0]:
                    solved[name][side] = q
                    print(f"[ik] {name:8s} {side:5s} 镜像修复后 位置残差 "
                          f"{pos_res * 1000:5.1f}mm  朝向残差 {math.degrees(rot_res):5.1f}°")
            prev = solved[name]
        return solved

    def _set_arms(self, joints: dict, write_ctrl: bool):
        m, d = self.model, self.data
        for side in SIDES:
            d.qpos[self.arm_qadr[side]] = joints[side]
            if write_ctrl:
                for n in ARM_JOINTS[side]:
                    d.ctrl[self.act[n]] = d.qpos[
                        m.jnt_qposadr[self.index.robots["g1"].joints[n]]]

    def _drive_arms(self, joints: dict):
        for side in SIDES:
            for k, n in enumerate(ARM_JOINTS[side]):
                self.data.ctrl[self.act[n]] = joints[side][k]

    def _set_fingers(self, pose: dict):
        for side in SIDES:
            for finger, val in pose.items():
                self.data.ctrl[self.act[f"{side}_hand_{finger}_joint"]] = val

    # ------------------------------------------------------------ 抓持力模型

    def _begin_grasp(self):
        """把手指合拢位置附近的布顶点「捏」到手上。

        G1 灵巧手是三指功率手：拇指与食指指尖在竖直方向天然错开约 9cm，
        无法像两根筷子那样夹住薄布（实测无论怎么调手形都空捏或顶穿布面）。
        真实薄布抓取里普遍采用「合规抓持力模型」：合指瞬间锁定手指之间的
        布顶点，之后用弹簧-阻尼力把顶点跟随手走。布料本体仍是完整的 flex
        仿真（膜/弯刚度、重力、与桌面接触、自碰撞全部真实），只是抓持这一环
        用受限刚度近似，不可能把布捏穿。
        """
        d, n = self.data, self.nvert
        verts = d.flexvert_xpos[:n]
        for side in SIDES:
            wrist = self.bid[f"{side}_wrist_yaw_link"]
            wp = d.xpos[wrist].copy()
            Rw = d.xmat[wrist].reshape(3, 3).copy()
            center = self._pinch_point(side)
            dist = np.linalg.norm(verts - center, axis=1)
            idx = [int(i) for i in np.argsort(dist)[:14] if dist[i] < 0.065]
            if not idx:
                self.grasp[side] = None
                continue
            local = [Rw.T @ (verts[i] - wp) for i in idx]
            self.grasp[side] = (idx, local)
            print(f"[grasp] {side} 锁定 {len(idx)} 个布顶点  "
                  f"离手 {dist[idx].min() * 1000:.0f}~{dist[idx].max() * 1000:.0f}mm")

    def _release_grasp(self):
        for side in SIDES:
            if self.grasp[side]:
                for i in self.grasp[side][0]:
                    self.data.xfrc_applied[self.vert_bodies[i], :] = 0.0
        self.grasp = {s: None for s in SIDES}
        self.grasp_ramp = 0.0
        print("[grasp] 松手，解除顶点锁定")

    def _apply_grasp(self, dt):
        """弹簧-阻尼把锁定顶点拖到手的局部偏移上（力施加在世界系）。"""
        if not any(self.grasp.values()):
            return
        self.grasp_ramp = min(1.0, self.grasp_ramp + dt / 0.15)
        d = self.data
        kp, kd = 400.0, 5.0          # 单顶点：足够拖动布边，又不会把布面撕穿
        for side in SIDES:
            g = self.grasp[side]
            if not g:
                continue
            wrist = self.bid[f"{side}_wrist_yaw_link"]
            wp = d.xpos[wrist]
            Rw = d.xmat[wrist].reshape(3, 3)
            idx, local = g
            for i, off in zip(idx, local):
                b = self.vert_bodies[i]
                target = wp + Rw @ off
                dof = int(self.vert_dof[i])
                f = kp * (target - d.xpos[b]) - kd * d.qvel[dof:dof + 3]
                self.data.xfrc_applied[b, :3] = self.grasp_ramp * f
                self.data.xfrc_applied[b, 3:] = 0.0


    # ------------------------------------------------------------ 主循环

    def _lock_base(self):
        d = self.data
        d.qpos[self.base_qpos:self.base_qpos + 7] = self.base_pose
        d.qvel[self.base_dof:self.base_dof + 6] = 0.0

    def advance(self) -> bool:
        """推进一个物理步（世界 ctrl_period=1，一个控制 tick 即一个物理步）。"""
        if self.finished:
            return False
        m, d = self.model, self.data
        phase, dur, k_from, k_to, action = PLAN[self.phase_i]
        s = self.phase_t / dur
        ease = 0.5 - 0.5 * math.cos(math.pi * s)
        self._drive_arms({side: self.solved[k_from][side] * (1 - ease)
                          + self.solved[k_to][side] * ease for side in SIDES})

        # 阶段动作：grip 在手指合拢后锁定布顶点，release 在落位后张开并解锁
        if action == "grip" and not self._action_done and s >= 0.55:
            self.finger_target = dict(FINGER_GRIP)
            self._begin_grasp()
            self._action_done = True
        elif action == "release" and not self._action_done and s >= 0.10:
            self.finger_target = dict(FINGER_OPEN)
            self._release_grasp()
            self._action_done = True
        self._set_fingers(self.finger_target)

        self._lock_base()
        self._apply_grasp(self.dt)
        mujoco.mj_step(m, d)

        self.phase_t += self.dt
        if self.phase_t >= dur - 1e-9:
            p = d.flexvert_xpos[:self.nvert]
            print(f"[{phase:9s}] t={d.time:5.2f}s  布面 x {p[:, 0].min():.3f}~{p[:, 0].max():.3f}"
                  f"  y {p[:, 1].min():+.3f}~{p[:, 1].max():+.3f}"
                  f"  z {p[:, 2].min():.3f}~{p[:, 2].max():.3f}")
            for side in SIDES:
                print(f"            {side} 捏合点 {np.round(self._pinch_point(side), 3)}")
            self.phase_i += 1
            self.phase_t = 0.0
            self._action_done = False
            if self.phase_i >= len(PLAN):
                self.finished = True
        return True

    # ------------------------------------------------------------ 指标

    def metrics(self) -> dict:
        p = self.data.flexvert_xpos[:self.nvert]
        hi = p[:, 2] > 0.760
        lo = ~hi
        overlap = 0.0
        if hi.any() and lo.any():
            overlap = float(min(p[hi, 0].max(), p[lo, 0].max())
                            - max(p[hi, 0].min(), p[lo, 0].min()))
        return dict(x=(p[:, 0].min(), p[:, 0].max()),
                    y=(p[:, 1].min(), p[:, 1].max()),
                    z=(p[:, 2].min(), p[:, 2].max()),
                    nvert=self.nvert,
                    lifted=int(hi.sum()),
                    overlap=overlap)


def _write_video(frames, fps, path):
    import subprocess
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{tmp}/f{i:05d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    spec = load_world(str(WORLD))
    model, index = build_world(spec)
    data = mujoco.MjData(model)
    fold = ClothFold(model, index, data)

    renderer, frames, cam = None, [], None
    every = 1
    if args.video:
        renderer = mujoco.Renderer(model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "showcase")
        every = max(1, round(1.0 / (args.fps * fold.dt)))

    step = 0
    while fold.advance():
        if renderer is not None and step % every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        step += 1

    if renderer is not None:
        renderer.close()
        _write_video(frames, args.fps, args.video)
    mt = fold.metrics()
    print(f"仿真 {data.time:.2f}s，布料顶点 {mt['nvert']} 个")
    print(f"x 范围 {mt['x'][0]:.3f}~{mt['x'][1]:.3f} m")
    print(f"y 范围 {mt['y'][0]:.3f}~{mt['y'][1]:.3f} m")
    print(f"z 范围 {mt['z'][0]:.3f}~{mt['z'][1]:.3f} m")
    print(f"抬离桌面顶点 {mt['lifted']}/{mt['nvert']}，上下层 x 重叠 "
          f"{mt['overlap'] * 100:.1f} cm")


if __name__ == "__main__":
    main()
