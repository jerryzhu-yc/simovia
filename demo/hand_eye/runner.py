"""Robot-independent controller for joint-actuated eye-in-hand calibration."""
from pathlib import Path
import json, math
import cv2
import mujoco
import numpy as np
from simovia.world.schema import parse_world
from simovia.world.builder import build_world
from simovia.world.initial import apply_initial_state
from .calibration import (WIDTH, HEIGHT, transform, inverse, intrinsics, rig_identity,
                          detect_target, solve, save_result)
HERE = Path(__file__).resolve().parent

class State:
    def __init__(self, robot):
        self.robot = robot
        self.spec = parse_world(robot.world_spec)
        self.spec.base_dir = Path(robot.world_spec['base_dir'])
        self.model, self.index = build_world(self.spec)
        self.data = mujoco.MjData(self.model)
        apply_initial_state(self.spec, self.model, self.data, self.index)
        self.dt = self.spec.timestep * self.spec.ctrl_period
        self.sync(robot.observe())
    def sync(self, obs):
        m, d = self.model, self.data
        ri = self.index.robots[self.robot.name]
        for name, q in self.robot.joint_state().items():
            j = ri.joints[name]
            if m.jnt_type[j] in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                d.qpos[m.jnt_qposadr[j]] = q
        for name, value in obs.get('actuators', {}).items():
            d.ctrl[ri.actuators[name]] = value['u']
        d.time = obs['time']; self.tick = obs['tick']
        mujoco.mj_forward(m, d)
    def step(self, ctrl):
        ri = self.index.robots[self.robot.name]
        self.robot.set_all([float(ctrl[ri.actuators[n]]) for n in self.robot.actuators])
        obs = self.robot.step()
        if obs is False: raise SystemExit(0)
        self.sync(obs)

class Motion:
    def __init__(self, state, profile, robot_name='arm'):
        self.state, self.profile = state, profile
        self.m, self.d = state.model, state.data
        self.jids = [self.m.joint(robot_name+'/'+n).id for n in profile['joints']]
        self.aids = [self.m.actuator(robot_name+'/'+n).id for n in profile['actuators']]
        if len(self.jids) != len(self.aids) or len(self.jids) < 6:
            raise ValueError('配置至少 6 个单自由度关节及一一对应的位置执行器')
        self.qadr = self.m.jnt_qposadr[self.jids]; self.dof = self.m.jnt_dofadr[self.jids]
        for j, a in zip(self.jids, self.aids):
            if self.m.jnt_type[j] != int(mujoco.mjtJoint.mjJNT_HINGE) or self.m.actuator_trnid[a,0] != j or self.m.actuator_gainprm[a,0] <= 0 or not np.isclose(self.m.actuator_biasprm[a,1], -self.m.actuator_gainprm[a,0]) or self.m.actuator_gear[a,0] != 1:
                raise ValueError('当前运动适配器要求直接驱动 hinge 的位置执行器；其他驱动需另写运动适配器')
        self.frame = profile['tool_frame']
        self.frame_id = getattr(self.m, self.frame['type'])(robot_name+'/'+self.frame['name']).id
        self.ik = mujoco.MjData(self.m)
        self.jp = np.zeros((3,self.m.nv)); self.jr = self.jp.copy()
        self.q = self.d.qpos[self.qadr].copy()
    def pose(self, data=None):
        d = self.d if data is None else data
        if self.frame['type'] == 'site':
            return transform(d.site_xmat[self.frame_id].reshape(3,3), d.site_xpos[self.frame_id])
        if self.frame['type'] == 'body':
            return transform(d.xmat[self.frame_id].reshape(3,3), d.xpos[self.frame_id])
        raise ValueError('tool_frame.type 仅支持 site 或 body')
    def solve(self, p, R, seed=None):
        q = self.q.copy() if seed is None else np.array(seed, dtype=float)
        self.ik.qpos[:] = self.d.qpos
        for _ in range(250):
            self.ik.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m,self.ik); mujoco.mj_comPos(self.m,self.ik)
            T = self.pose(self.ik); ep = np.asarray(p)-T[:3,3]
            er = cv2.Rodrigues(R @ T[:3,:3].T)[0].ravel()
            if np.linalg.norm(ep) < .00015 and np.linalg.norm(er) < .001: return q
            jac = mujoco.mj_jacSite if self.frame['type'] == 'site' else mujoco.mj_jacBody
            jac(self.m,self.ik,self.jp,self.jr,self.frame_id)
            J = np.vstack((self.jp[:,self.dof], .4*self.jr[:,self.dof]))
            dq = J.T @ np.linalg.solve(J @ J.T+.00006*np.eye(6), np.r_[ep,.4*er])
            q += dq * min(1.,.16/max(np.linalg.norm(dq),1e-9))
            q = np.clip(q,self.m.jnt_range[self.jids,0]+.015,self.m.jnt_range[self.jids,1]-.015)
        raise ValueError(f'采样位姿不可达：{p}，请按机械臂工作范围修改 profile')
    def tick(self, q):
        self.q = q.copy()
        ctrl = self.d.ctrl.copy()
        ctrl[self.aids] = q+self.d.qfrc_bias[self.dof]/self.m.actuator_gainprm[self.aids,0]
        self.state.step(ctrl)
    def wait(self, seconds):
        for _ in range(math.ceil(seconds/self.state.dt)):
            self.tick(self.q); yield None
    def move(self, p, R):
        T = self.pose(); p0 = T[:3,3].copy(); R0 = T[:3,:3].copy()
        rv = cv2.Rodrigues(R @ R0.T)[0].ravel()
        duration = max(.8,1.875*np.linalg.norm(np.asarray(p)-p0)/.10,1.875*np.linalg.norm(rv)/.5)
        steps = math.ceil(duration/self.state.dt)
        for i in range(1,steps+1):
            u=i/steps; s=u*u*u*(10+u*(-15+6*u))
            self.tick(self.solve(p0+s*(np.asarray(p)-p0),cv2.Rodrigues(s*rv)[0] @ R0))
            yield None
        yield from self.wait(.5)

class CalibrationDemo:
    def __init__(self, robot):
        self.robot = robot; self.state = State(robot)
        self.profile_path = (self.state.spec.base_dir / robot.args['calibration_profile']).resolve()
        self.profile = json.loads(self.profile_path.read_text())
        self.camera = self.profile['camera']; self.motion = Motion(self.state,self.profile,robot.name)
        sensor = next(s for r in self.state.spec.robots if r.name==robot.name for s in r.sensors if s.name==self.camera)
        self.K = intrinsics(sensor.fovy or 45)
        self.identity = rig_identity(self.state.spec,robot.name,self.camera,self.profile['tool_frame'])
        self.output = HERE/'runs'/self.profile['name']; self.output.mkdir(parents=True,exist_ok=True)
        self.result_path = (self.profile_path.parent/self.profile['result']).resolve()
        self.samples=[]; self._last=-100; self.count=0
        self.board = json.loads((HERE/'board_corners.json').read_text())
    def capture(self): return self.robot.camera(self.camera,WIDTH,HEIGHT)
    def preview(self, force=False):
        if not force and self.state.tick-self._last<10: return
        self._last=self.state.tick
        rgb=self.capture()
        try: _,_,rgb,_=detect_target(rgb,self.board,self.K)
        except ValueError: pass
        cv2.rectangle(rgb,(0,0),(WIDTH,30),(14,22,35),-1)
        cv2.putText(rgb,f'HAND-EYE | {self.profile["name"]} | sample {self.count}/12',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
        self.robot.publish_camera(self.camera,rgb)
    def run(self):
        print('独立手眼标定：'+self.profile['name'],flush=True)
        for i,pose in enumerate(self.profile['poses']):
            self.count=i+1
            for _ in self.motion.move(pose['position'],np.array(pose['rotation'])):
                self.preview(); yield None
            rgb=self.capture(); C,error,overlay,n=detect_target(rgb,self.board,self.K)
            self.samples.append(dict(gripper_to_world=self.motion.pose().tolist(),target_to_camera=C.tolist(),reprojection_px=error,marker_count=n))
            cv2.imwrite(str(self.output/f'sample_{i:02d}.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
            self.robot.publish_camera(self.camera,overlay)
            print(f'采样 {i+1}/{len(self.profile["poses"])}，标记 {n}，重投影 {error:.3f}px',flush=True)
        result=solve(self.samples,self.K,self.identity,self.profile['name'])
        # Truth is used exclusively for an audit; estimation uses RGB PnP and FK.
        m,d=self.state.model,self.state.data; mujoco.mj_camlight(m,d)
        cid=m.camera(self.robot.name+'/'+self.camera).id
        actual=inverse(self.motion.pose()) @ transform(d.cam_xmat[cid].reshape(3,3) @ np.diag([1.,-1.,-1.]),d.cam_xpos[cid])
        X=np.array(result['gripper_T_camera'])
        result['ground_truth_validation_only']=dict(translation_m=float(np.linalg.norm(X[:3,3]-actual[:3,3])),rotation_deg=float(np.linalg.norm(cv2.Rodrigues(X[:3,:3] @ actual[:3,:3].T)[0])*180/np.pi))
        save_result(self.result_path,result); save_result(self.output/'hand_eye.json',result)
        self.preview(force=True)
        print('标定通过，已保存：'+str(self.result_path),flush=True)
        self.robot.report_result(True,'标定通过留出验证，文件已保存')
