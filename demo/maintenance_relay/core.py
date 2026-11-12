"""Physical relay orchestration. All three robots share one MuJoCo clock.

The scheduler uses simulator poses; no learned perception is claimed. After
initialization, only actuator targets change: there are no payload welds,
teleports, root-position overrides or external stabilizing forces.
"""
from pathlib import Path
import json, math
import numpy as np
import mujoco
from simovia.world.schema import load_world
from simovia.world.builder import build_world
from simovia.world.initial import apply_initial_state
from .scene import HERE
from .control import Arm, Dog, Balance, DOWN

RIGHT_DOWN=np.array([[0.,-1,0],[1,0,0],[0,0,1]])@DOWN
RIGHT_SEED=[-.6,-.3,1,1.1,-.5,.2,.9]
ARM_JOINTS=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
LIMBS=('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')

class Demo:
    def __init__(self, output=None, station_delay=2., simulation=None, log_stream=None):
        self.log_stream=log_stream
        if simulation is None:
            self.spec=load_world(HERE/'scene.yaml')
            self.model,self.index=build_world(self.spec)
            self.data=mujoco.MjData(self.model)
            apply_initial_state(self.spec,self.model,self.data,self.index)
        else:
            self.spec=simulation.spec
            self.model,self.index,self.data=simulation.model,simulation.index,simulation.data
        m,d=self.model,self.data
        self.dt=self.spec.timestep*self.spec.ctrl_period
        self.loader=Arm(m,d,'loader',ARM_JOINTS)
        self.right=Arm(m,d,'technician',[f'right_{n}_joint' for n in LIMBS],site='technician/gripper_right/gripper_tcp')
        self.left=Arm(m,d,'technician',[f'left_{n}_joint' for n in LIMBS],site='technician/gripper_left/button_tcp')
        # Retract the added left gripper clear of the panel before settling.
        left_R=d.site_xmat[self.left.sid].reshape(3,3).copy()
        self.left.q[3]=.8
        d.qpos[self.left.qa]=self.left.q
        mujoco.mj_forward(m,d)
        self.loader.q=self.loader.solve([-.15,.05,.85],seed=[-1.57,-1.57,1.57,-1.57,-1.57,0])
        d.qpos[self.loader.qa]=self.loader.q
        self.loader_rest=self.loader.q.copy()
        self.right_rest=self.right.q.copy()
        self.left_rest=self.left.q.copy()
        self.left_R=left_R.copy()
        self.grippers={'loader':'loader/gripper','technician':'technician/gripper_right'}
        for name in ('loader','technician'):
            for side in ('left','right'):
                d.qpos[m.jnt_qposadr[m.joint(self.grippers[name]+'/'+side+'_slide').id]]=.019
        mujoco.mj_forward(m,d)
        self.dog=Dog(m,d);self.balance=Balance(m,d)
        self.ctrl=d.ctrl.copy()
        self.grip={'loader':.038,'technician':.038}
        self.crouching=False
        self.drive=0.;self.drive_target=0.;self.held_by=None;self.hold_misses=0
        self.stage='READY';self.detail='等待开始';self.events=[];self.done=False;self.success=False
        self.owner='备料架';self.station_delay=station_delay
        self.output=Path(output) if output else None
        self.payload=m.body('module_blue').id
        self.cargo=m.site('courier/cargo_center').id
        self.button=m.body('reset_button').id
        self.button_probe=m.geom('technician/gripper_left/button_probe').id
        self.pressed=False;self.collision_peak=0.;self.steps=0
        self.renderer=None
        self.initial_payload=d.xpos[self.payload].copy()

    def transition(self,stage,detail):
        self.stage,self.detail=stage,detail
        self.events.append(dict(time=round(self.data.time,3),stage=stage,detail=detail,owner=self.owner))
        print(f'[{self.data.time:6.2f}] {stage}: {detail}',flush=True,file=self.log_stream)

    def module_grasp(self):
        return self.data.xpos[self.payload]+self.data.xmat[self.payload].reshape(3,3)@np.array([0,0,.045])

    def tick(self):
        if self.done:return
        m,d=self.model,self.data
        self.drive+=float(np.clip(self.drive_target-self.drive,-.008,.008))
        self.dog.control(self.ctrl,self.drive,self.dt)
        self.balance.control(self.ctrl,self.dt)
        for arm in (self.loader,self.right,self.left):arm.hold(self.ctrl)
        for name,value in self.grip.items():self.ctrl[m.actuator(self.grippers[name]+'/gripper_act').id]=value
        self.ctrl[m.actuator('technician/gripper_left/gripper_act').id]=0.
        self.ctrl=np.clip(self.ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        d.ctrl[:]=self.ctrl
        mujoco.mj_step(m,d,nstep=self.spec.ctrl_period)
        self.steps+=1
        if self.held_by is not None:
            error=np.linalg.norm(self.module_grasp()-d.site_xpos[self.held_by.sid])
            self.hold_misses=self.hold_misses+1 if error>.035 else 0
            if self.hold_misses*self.dt>.1:raise RuntimeError('搬运过程中夹爪丢失模块')
        if not np.isfinite(d.qpos).all():raise RuntimeError('仿真状态出现非有限值')
        for bid,height in ((self.dog.base,.17),(self.balance.pelvis,.57 if self.crouching else .62)):
            if d.xpos[bid,2]<height or d.xmat[bid].reshape(3,3)[2,2]<.85:
                raise RuntimeError('机器人失稳，任务停止')
        for i in range(d.ncon):
            c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            if (g1==self.button_probe and b2==self.button) or (g2==self.button_probe and b1==self.button):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                if force[0]>.2:self.pressed=True
            n1=m.body(b1).name.split('/')[0];n2=m.body(b2).name.split('/')[0]
            if n1 in ('loader','courier','technician') and n2 in ('loader','courier','technician') and n1!=n2:
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                self.collision_peak=max(self.collision_peak,float(force[0]))
                if force[0]>5:raise RuntimeError(f'机器人间碰撞: {m.body(b1).name}/{m.body(b2).name} {force[0]:.1f}N')

    def wait(self,seconds):
        for _ in range(math.ceil(seconds/self.dt)):
            self.tick();yield

    def stance(self,knee,seconds=2.):
        self.crouching=knee>.5
        start=self.balance.home.copy();target=start.copy()
        for side in ('left','right'):
            for kind,value in [('hip_pitch',-knee/2),('knee',knee),('ankle_pitch',-knee/2)]:
                aid=self.model.actuator(f'technician/{side}_{kind}_joint').id
                target[np.flatnonzero(self.balance.ids==aid)[0]]=value
        waist=self.model.actuator('technician/waist_pitch_joint').id
        target[np.flatnonzero(self.balance.ids==waist)[0]]= .65 if knee>.5 else 0.
        for i in range(1,math.ceil(seconds/self.dt)+1):
            u=min(1.,i*self.dt/seconds);s=u*u*u*(10+u*(-15+6*u))
            self.balance.home=start+(target-start)*s
            self.tick();yield
        yield from self.wait(.5)

    def joint_move(self,arm,q,seconds=2.):
        q0=arm.q.copy();n=math.ceil(seconds/self.dt)
        for i in range(1,n+1):
            u=i/n;s=u*u*u*(10+u*(-15+6*u))
            arm.q=q0+(q-q0)*s
            self.tick();yield
        yield from self.wait(.15)

    def move(self,arm,p,R=DOWN,seconds=2.,seed=None,linear=False):
        if linear:
            p0=self.data.site_xpos[arm.sid].copy();n=math.ceil(seconds/self.dt)
            for i in range(1,n+1):
                u=i/n;s=u*u*u*(10+u*(-15+6*u))
                arm.q=arm.solve(p0+(np.asarray(p)-p0)*s,R=R)
                self.tick();yield
            yield from self.wait(.15)
        else:
            q=arm.solve(p,R=R,seed=seed)
            yield from self.joint_move(arm,q,seconds)
        # Compensate the measured floating-base displacement at the final pose.
        for _ in range(2):
            if np.linalg.norm(self.data.site_xpos[arm.sid]-p)<.004:break
            q=arm.solve(p,R=R,seed=arm.q)
            yield from self.joint_move(arm,q,.35)
        if np.linalg.norm(self.data.site_xpos[arm.sid]-p)>.012:
            raise RuntimeError(f'{arm.prefix} 末端跟踪误差过大')

    def grasp(self,arm):
        self.grip[arm.prefix]=0.
        yield from self.wait(.6)
        # Contact must occur on both physical fingertips.
        names=set()
        for c in self.data.contact[:self.data.ncon]:
            b=[int(self.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            if self.payload in b:
                names.add(self.model.body(b[1] if b[0]==self.payload else b[0]).name)
        if not all(self.grippers[arm.prefix]+'/'+s+'_finger' in names for s in ('left','right')):
            raise RuntimeError(arm.prefix+' 未建立双指抓取接触')
        self.held_by=arm;self.hold_misses=0

    def placed(self):
        p=self.data.xpos[self.payload]
        j=self.model.body_jntadr[self.payload];v=self.data.qvel[self.model.jnt_dofadr[j]:self.model.jnt_dofadr[j]+6]
        return bool(abs(p[0]-1.60)<.025 and abs(p[1]-.36)<.025 and .66<p[2]<.695 and np.linalg.norm(v)<.04 and self.data.xmat[self.payload].reshape(3,3)[2,2]>.95)

    def run(self):
        if self.done:return
        try:
            self.transition('STABILIZE','Go2 站稳，G1 保持平衡')
            yield from self.wait(2.)
            self.transition('PICK','UR5e 选择蓝色维修模块')
            p=self.module_grasp().copy()
            yield from self.move(self.loader,p+[0,0,.15],seconds=2.5)
            yield from self.move(self.loader,p,seconds=1.5,linear=True)
            yield from self.grasp(self.loader);self.owner='UR5e'
            yield from self.move(self.loader,p+[0,0,.20],seconds=2.)
            if self.data.xpos[self.payload,2]<self.initial_payload[2]+.12:raise RuntimeError('模块未被抬起')
            self.transition('LOAD','Go2 保持不动，UR5e 装载')
            target=self.data.site_xpos[self.cargo]+[0,0,.071]
            yield from self.move(self.loader,target+[0,0,.16],seconds=2.5)
            yield from self.move(self.loader,target,seconds=1.5,linear=True)
            self.held_by=None;self.grip['loader']=.038
            yield from self.wait(.6);self.owner='Go2 托盘'
            yield from self.move(self.loader,target+[0,0,.20],seconds=1.7)
            yield from self.joint_move(self.loader,self.loader_rest,2.)
            delta=self.data.xpos[self.payload]-self.data.site_xpos[self.cargo]
            if np.linalg.norm(delta[:2])>.04 or not .01<delta[2]<.05:raise RuntimeError('装载检查失败：模块未落入托盘')
            self.transition('DELIVER','机械臂已退出，Go2 配送到维修工位')
            start=self.data.time
            while self.data.xpos[self.dog.base,0]<1.70:
                distance=1.72-self.data.xpos[self.dog.base,0]
                self.drive_target=.4 if distance>.3 else .27
                self.tick();yield
                delta=self.data.xpos[self.payload]-self.data.site_xpos[self.cargo]
                if abs(delta[0])>.12 or abs(delta[1])>.10 or not -.01<delta[2]<.10:raise RuntimeError('配送途中模块离开托盘')
                if self.data.time-start>35:raise RuntimeError('配送超时')
            self.drive_target=0.
            yield from self.wait(1.5)
            self.transition('WAIT_STATION','已到站，等待 G1 完成工位准备')
            yield from self.wait(self.station_delay)
            if abs(self.data.xpos[self.dog.base,1])>.06:raise RuntimeError('Go2 停靠横向偏差过大')
            root_j=self.model.body_jntadr[self.dog.base];vadr=self.model.jnt_dofadr[root_j]
            if np.linalg.norm(self.data.qvel[vadr:vadr+6])>.08:raise RuntimeError('Go2 尚未停稳')
            self.transition('UNLOAD','Go2 已停稳，G1 接收维修模块')
            yield from self.stance(.8)
            p=self.module_grasp().copy()
            yield from self.move(self.right,p+[0,0,.10],RIGHT_DOWN,2.5,RIGHT_SEED)
            yield from self.move(self.right,p,RIGHT_DOWN,1.5,linear=True)
            yield from self.grasp(self.right);self.owner='G1'
            yield from self.move(self.right,p+[0,0,.13],RIGHT_DOWN,1.8,linear=True)
            yield from self.stance(.4)
            self.transition('INSTALL','G1 将模块放入设备导向槽')
            target=np.array([1.60,.36,.72])
            yield from self.move(self.right,target+[0,0,.06],RIGHT_DOWN,2.)
            yield from self.move(self.right,target,RIGHT_DOWN,1.2,linear=True)
            self.held_by=None;self.grip['technician']=.038
            yield from self.wait(.6);self.owner='设备槽位'
            yield from self.move(self.right,target+[0,0,.10],RIGHT_DOWN,1.5)
            yield from self.joint_move(self.right,self.right_rest,2.)
            stable=0
            for _ in range(math.ceil(2/self.dt)):
                self.tick();yield
                stable=stable+1 if self.placed() else 0
                if stable*self.dt>=.6:break
            if stable*self.dt<.6:raise RuntimeError('模块未在设备槽位稳定落座')
            self.transition('RESET_DEVICE','模块到位，G1 按下设备复位按钮')
            self.pressed=False
            yield from self.move(self.left,[2.17,.035,.84],self.left_R,1.5)
            start=self.data.site_xpos[self.left.sid].copy()
            target=np.array([2.17,-.008,.84])
            for i in range(1,161):
                # Replan against the measured floating base, stopping on contact.
                try:
                    self.left.q=self.left.solve(start+(target-start)*i/160,R=self.left_R)
                except RuntimeError as error:
                    raise RuntimeError('复位按钮没有实际接触：末端已到可达边界') from error
                self.tick();yield
                if self.pressed:break
            if not self.pressed:raise RuntimeError('复位按钮没有实际接触')
            yield from self.joint_move(self.left,self.left_rest,1.5)
            yield from self.wait(1.)
            if not self.placed():raise RuntimeError('复位后模块位置发生变化')
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g]==self.model.body('alarm').id:self.model.geom_rgba[g]=[.1,.9,.55,1]
            self.success=True
            self.transition('COMPLETE','设备恢复运行 · 三机器人任务完成')
        except Exception as error:
            self.transition('FAILED',str(error))
            raise
        finally:
            self.drive_target=0.;self.done=True;self.save()

    def save(self):
        if self.output is None:return
        self.output.mkdir(parents=True,exist_ok=True)
        result=dict(success=self.success,stage=self.stage,detail=self.detail,sim_time=self.data.time,
                    module_position=self.data.xpos[self.payload].tolist(),button_contact=self.pressed,
                    dog_position=self.data.xpos[self.dog.base].tolist(),dog_rotation=self.data.xmat[self.dog.base].tolist(),
                    inter_robot_contact_peak_N=self.collision_peak,events=self.events)
        (self.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
