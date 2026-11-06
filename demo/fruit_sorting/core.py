"""Vision-guided fruit sorting on Simovia. No fruit poses are used by perception/planning."""
from __future__ import annotations
import json,math,time
from pathlib import Path
import cv2
import mujoco
import numpy as np
from simovia.kernel.sim import Simulation
from simovia.world.schema import load_world
from simovia.render import render_offscreen
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text())
PHYSICS=json.loads((HERE/'physics.json').read_text())
W,H=640,480
DOWN=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
CV_TO_GL=np.diag([1.,-1.,-1.])
CLASSES={'banana':((18,75,65),(38,255,255)), 'apple':((0,85,45),(12,255,255)), 'grape':((125,65,35),(165,255,255))}
TOP_HEIGHT={'banana':.025,'apple':.030,'grape':.031}  # RGB silhouette-centroid plane, not top surface
GRASP_HEIGHT={'banana':.025,'apple':.032,'grape':.020}
COLORS={'banana':(255,206,56),'apple':(255,76,85),'grape':(195,109,255)}
def transform(R,t):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=np.asarray(t).reshape(3);return T
def inv(T):
    return transform(T[:3,:3].T,-T[:3,:3].T@T[:3,3])
def intrinsic(fovy=65,width=W,height=H):
    f=height/(2*np.tan(np.radians(fovy)/2))
    return np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
def detect(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);detections=[]
    for kind,(lo,hi) in CLASSES.items():
        mask=cv2.inRange(hsv,np.array(lo),np.array(hi))
        if kind=='apple':mask|=cv2.inRange(hsv,np.array([170,85,45]),np.array([179,255,255]))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        for c in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            area=cv2.contourArea(c)
            if area<80:continue
            x,y,w,h=cv2.boundingRect(c);mom=cv2.moments(c)
            if w<8 or h<8 or area/(w*h)<.25:continue
            detections.append(dict(kind=kind,uv=[mom['m10']/mom['m00'],mom['m01']/mom['m00']],bbox=[x,y,w,h],area=area))
    return detections
def annotate(rgb,detections,status=''):
    out=rgb.copy()
    for d in detections:
        x,y,w,h=d['bbox'];color=COLORS[d['kind']]
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
        cv2.circle(out,tuple(np.round(d['uv']).astype(int)),3,color,-1)
        label=d['kind'].upper()
        cv2.putText(out,label,(x,max(17,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.47,color,1,cv2.LINE_AA)
    if status:
        cv2.rectangle(out,(0,0),(W,30),(14,22,35),-1)
        cv2.putText(out,status[:85],(10,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,237,248),1,cv2.LINE_AA)
    return out
def unproject(uv,T_world_camera,z,K=None):
    K=intrinsic() if K is None else K
    ray=T_world_camera[:3,:3]@np.linalg.solve(K,np.r_[uv,1.])
    origin=T_world_camera[:3,3]
    if abs(ray[2])<1e-6:raise ValueError('Camera ray parallel to grasp plane')
    lam=(z-origin[2])/ray[2]
    if lam<=0:raise ValueError('Grasp plane behind camera')
    return origin+lam*ray
class Arm:
    def __init__(self,sim):
        self.sim=sim;self.m=sim.model;self.d=sim.data
        self.jids=[self.m.joint('arm/joint'+str(i)).id for i in range(1,8)]
        self.qadr=self.m.jnt_qposadr[self.jids];self.dof=self.m.jnt_dofadr[self.jids]
        self.aids=[self.m.actuator('arm/actuator'+str(i)).id for i in range(1,8)]
        self.grip_id=self.m.actuator('arm/gripper/gripper_act').id
        self.sid=self.m.site('arm/gripper/gripper_tcp').id
        self.ik=mujoco.MjData(self.m);self.jp=np.zeros((3,self.m.nv));self.jr=self.jp.copy()
        self.grip=.038;self.q=self.d.qpos[self.qadr].copy()
        self.peak_speed=np.zeros(7);self.peak_torque=np.zeros(7)
        self.hold_check=None;self._hold_ticks=0
    def pose(self):
        return transform(self.d.site_xmat[self.sid].reshape(3,3).copy(),self.d.site_xpos[self.sid].copy())
    def solve(self,p,R=DOWN,seed=None):
        q=self.q.copy() if seed is None else np.asarray(seed).copy()
        self.ik.qpos[:]=self.d.qpos
        for _ in range(160):
            self.ik.qpos[self.qadr]=q
            mujoco.mj_kinematics(self.m,self.ik);mujoco.mj_comPos(self.m,self.ik)
            ep=np.asarray(p)-self.ik.site_xpos[self.sid]
            er=cv2.Rodrigues(R@self.ik.site_xmat[self.sid].reshape(3,3).T)[0].ravel()
            if np.linalg.norm(ep)<.00015 and np.linalg.norm(er)<.001:return q
            mujoco.mj_jacSite(self.m,self.ik,self.jp,self.jr,self.sid)
            J=np.vstack((self.jp[:,self.dof],.4*self.jr[:,self.dof]))
            dq=J.T@np.linalg.solve(J@J.T+.00006*np.eye(6),np.r_[ep,.4*er])
            dq*=min(1,.16/max(np.linalg.norm(dq),1e-9));q+=dq
            q=np.clip(q,self.m.jnt_range[self.jids,0]+.015,self.m.jnt_range[self.jids,1]-.015)
        raise RuntimeError(f'IK unreachable {np.round(p,3)}: position={np.linalg.norm(ep):.4f}, rotation={np.linalg.norm(er):.3f}')
    def initialize(self):
        seed=np.array([-1.3176,.8125,1.4687,-2.1172,-.9286,2.0114,-1.7435])
        self.q=self.solve([.43,0,1.13],seed=seed)
        self.d.qpos[self.qadr]=self.q
        for n in ('left_slide','right_slide'):self.d.qpos[self.m.jnt_qposadr[self.m.joint('arm/gripper/'+n).id]]=.019
        self.d.ctrl[self.aids]=self.q;self.d.ctrl[self.grip_id]=.038
        mujoco.mj_forward(self.m,self.d)
    def tick(self,q=None,grip=None):
        if q is not None:self.q=np.asarray(q).copy()
        if grip is not None:self.grip+=float(np.clip(grip-self.grip,-PHYSICS['grip_speed_m_s']*.02,PHYSICS['grip_speed_m_s']*.02))
        if hasattr(self.sim,'remote_step'):
            bias=self.d.qfrc_bias[self.dof]/self.m.actuator_gainprm[self.aids,0]
            self.d.ctrl[self.aids]=self.q+bias
            self.d.ctrl[self.grip_id]=self.grip
            self.sim.remote_step(self.d.ctrl.copy())
            self.check_dynamics()
            self.check_hold()
            return
        for _ in range(self.sim.spec.ctrl_period):
            bias=self.d.qfrc_bias[self.dof]/self.m.actuator_gainprm[self.aids,0]
            self.d.ctrl[self.aids]=self.q+bias
            self.d.ctrl[self.grip_id]=self.grip
            mujoco.mj_step(self.m,self.d)
            self.check_dynamics()
        self.check_hold()
        self.sim.tick+=1
        if not np.isfinite(self.d.qpos).all():raise RuntimeError('Non-finite simulation state')
    def check_dynamics(self):
        speed=np.abs(self.d.qvel[self.dof]);torque=np.abs(self.d.actuator_force[self.aids])
        self.peak_speed=np.maximum(self.peak_speed,speed);self.peak_torque=np.maximum(self.peak_torque,torque)
        if np.any(speed>np.array(PHYSICS['joint_velocity_limit_rad_s'])):
            raise RuntimeError('Measured Panda joint speed exceeds manufacturer limit')
        if np.any(torque>np.array(PHYSICS['joint_torque_limit_Nm'])+1e-5):
            raise RuntimeError('Panda torque limit exceeded')
    def check_hold(self):
        self._hold_ticks+=1
        if self.hold_check is not None and self._hold_ticks%5==0:
            forces=self.hold_check()
            if min(forces.values())<.1:raise RuntimeError(f'Grasp lost during transfer: {forces}')
    def wait(self,seconds,grip=None):
        for _ in range(round(seconds/.02)):
            self.tick(grip=grip);yield None
    def move(self,p,R=DOWN,speed=.12):
        T=self.pose();p0=T[:3,3];R0=T[:3,:3]
        rv=cv2.Rodrigues(R@R0.T)[0].ravel()
        duration=max(.8,1.875*np.linalg.norm(np.asarray(p)-p0)/speed,1.875*np.linalg.norm(rv)/.5)
        steps=math.ceil(duration/.02)
        for i in range(1,steps+1):
            u=i/steps;s=u*u*u*(10+u*(-15+6*u))
            target=p0+s*(np.asarray(p)-p0);rot=cv2.Rodrigues(s*rv)[0]@R0
            q=self.solve(target,rot)
            self.tick(q);yield None
        yield from self.wait(.25)
class Demo:
    def __init__(self,output,seed=0):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.seed=seed;self.sim=None;self.renderer=None;self.view_renderer=None
        self.stage='Ready';self.detail='';self.rgb=None;self.detections=[];self.calibration=None;self.results=[];self.audit=[];self.frame=0
        self.overview_cam=mujoco.MjvCamera()
        self.overview_cam.lookat[:]=[.38,0,.82];self.overview_cam.distance=1.65
        self.overview_cam.azimuth=135;self.overview_cam.elevation=-35
        self.load('world.yaml')
    def load(self,filename):
        self.close_scene()
        self.sim=Simulation(load_world(HERE/filename),autostart_controllers=False)
        self.arm=Arm(self.sim);self.arm.initialize()
        if filename == 'world.yaml' and self.seed:
            rng=np.random.default_rng(self.seed)
            for f in CONFIG['fruits']:
                bid=self.sim.model.body(f['name']).id
                j=int(self.sim.model.body_jntadr[bid]);adr=int(self.sim.model.jnt_qposadr[j])
                self.sim.data.qpos[adr:adr+2]+=rng.uniform(-.003,.003,2)
            mujoco.mj_forward(self.sim.model,self.sim.data)
        self.renderer=mujoco.Renderer(self.sim.model,H,W)
        self.view_renderer=mujoco.Renderer(self.sim.model,600,900)
        self.camera_id=self.sim.model.camera('arm/wrist_rgb').id
        self.rgb=self.capture()
    def close_scene(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
        if self.view_renderer is not None:self.view_renderer.close();self.view_renderer=None
        if self.sim is not None:self.sim.close();self.sim=None
    def capture(self):
        return render_offscreen(self.sim.model,self.sim.data,W,H,'arm/wrist_rgb',renderer=self.renderer)
    def overview(self):
        rgb=render_offscreen(self.sim.model,self.sim.data,900,600,renderer=self.view_renderer,free_cam=self.overview_cam)
        if self.sim.spec.name == 'fruit_sorting':
            cams=self.view_renderer.scene.camera
            origin=(np.array(cams[0].pos)+np.array(cams[1].pos))/2
            forward=np.array(cams[0].forward);up=np.array(cams[0].up);right=np.cross(forward,up)
            focal=300/np.tan(np.radians(self.sim.model.vis.global_.fovy)/2)
            for k,(x,y) in CONFIG['bins'].items():
                delta=np.array([x,y,.77])-origin;z=delta@forward
                u=int(450+focal*(delta@right)/z);v=int(300-focal*(delta@up)/z)
                label=k.upper();tw=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.45,1)[0][0]
                cv2.rectangle(rgb,(u-tw//2-5,v-16),(u+tw//2+5,v+3),(20,29,41),-1)
                cv2.putText(rgb,label,(u-tw//2,v-3),cv2.FONT_HERSHEY_SIMPLEX,.45,COLORS[k],1,cv2.LINE_AA)
        return rgb
    def save_rgb(self,name,rgb):
        cv2.imwrite(str(self.output/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    def event(self,stage,detail=''):
        self.stage=stage;self.detail=detail
        print(json.dumps(dict(stage=stage,detail=detail,time=round(self.sim.data.time,2)),ensure_ascii=False),flush=True)
    def load_calibration(self):
        from demo.hand_eye.calibration import load_result
        path=getattr(self,'calibration_path',HERE.parent/'hand_eye/results/panda_wrist_rgb.json')
        self.calibration=load_result(path,self.sim.spec)
        (self.output/'hand_eye_used.json').write_text(json.dumps(self.calibration,indent=2))
        self.event('CALIBRATION_LOADED','Using saved hand-eye calibration: '+Path(path).name)
    def perceive(self):
        rgb=self.capture();X=np.array(self.calibration['gripper_T_camera'])
        T=self.arm.pose()@X;ds=detect(rgb);valid=[];display=[]
        for d in ds:
            try:p=unproject(d['uv'],T,.68+TOP_HEIGHT[d['kind']],np.array(self.calibration['intrinsics']))
            except ValueError:continue
            in_work=.415<p[0]<.71 and -.23<p[1]<.24
            in_bin=any(abs(p[0]-x)<.083 and abs(p[1]-y)<.105 for x,y in CONFIG['bins'].values())
            if in_work or in_bin:display.append(d)
            if in_work:
                d['world']=p.tolist();valid.append(d)
        self.detections=valid;self.rgb=annotate(rgb,display,f'RGB HSV | tick {self.sim.tick} | '+self.stage)
        return valid
    def grasp_contacts(self):
        """Read bilateral contact force, analogous to fingertip tactile feedback."""
        m,d=self.sim.model,self.sim.data
        finger_ids={m.body('arm/gripper/'+side+'_finger').id:side for side in ('left','right')}
        robot_bodies=set(self.sim.index.robots['arm'].bodies.values())
        forces={'left':0.,'right':0.}
        for ci,contact in enumerate(d.contact):
            b1=int(m.geom_bodyid[contact.geom1]);b2=int(m.geom_bodyid[contact.geom2])
            for finger,other in ((b1,b2),(b2,b1)):
                if finger in finger_ids and other not in robot_bodies:
                    force=np.zeros(6)
                    mujoco.mj_contactForce(m,d,ci,force)
                    forces[finger_ids[finger]]+=float(abs(force[0]))
        return forces
    def run(self):
        self.load_calibration()
        self.event('OBSERVE','Production scene: 2 bananas, 2 apples, 2 grape bunches')
        yield from self.arm.wait(.6)
        slots={k:0 for k in CONFIG['bins']}
        for attempt in range(8):
            yield from self.arm.move([.43,0,1.13])
            self.event('DETECT','RGB -> HSV -> connected components -> calibrated world coordinates')
            ds=self.perceive()
            self.save_rgb(f'detections_{attempt:02d}.png',self.rgb)
            if attempt==0:
                (self.output/'initial_detections.json').write_text(json.dumps(ds,indent=2))
                self.save_rgb('scene_initial.png',self.overview())
                if len(ds)!=6:raise RuntimeError(f'Expected 6 visible fruits, detected {len(ds)}')
            yield None
            if not ds:break
            target=sorted(ds,key=lambda d:(d['world'][0],d['world'][1]))[0]
            kind=target['kind'];xy=np.array(target['world'][:2])
            if slots[kind]>=2:raise RuntimeError(f'{kind} still visible after two placements; refusing blind retry')
            self.event('PICK',f'{kind.upper()} at ({xy[0]:.3f}, {xy[1]:.3f}) m')
            yield from self.arm.wait(2.8,grip=.038)
            yield from self.arm.move([*xy,.94])
            yield from self.arm.move([*xy,.68+GRASP_HEIGHT[kind]],speed=.055)
            yield from self.arm.wait(2.8,grip=0.)
            yield from self.arm.move([*xy,.99],speed=.075)
            self.audit.append(dict(stage='lift',kind=kind,positions={f['name']:self.sim.data.body(f['name']).xpos.copy().tolist() for f in CONFIG['fruits']}))
            forces=self.grasp_contacts()
            if min(forces.values())<.01:raise RuntimeError(f'Lost grasp: {kind}, bilateral contact forces={forces}')
            # Final bin scoring independently uses ground truth, outside planning.
            self.arm.hold_check=self.grasp_contacts
            slot=slots[kind];bx,by=CONFIG['bins'][kind]
            destination=([bx+(-.036 if slot==0 else .036),by] if kind=='banana' else [bx,by+(-.047 if slot==0 else .047)])
            self.event('PLACE',f'{kind.upper()} -> bin {kind.upper()}, slot {slot+1}/2')
            yield from self.arm.move([*destination,.99],speed=.10)
            # At the destination, support contact can unload a fingertip during settling.
            # Bilateral hold monitoring covers the lift and free-space transfer.
            self.arm.hold_check=None
            yield from self.arm.move([*destination,.755],speed=.035)
            yield from self.arm.wait(2.8,grip=.038)
            yield from self.arm.move([*destination,.99],speed=.09)
            yield from self.arm.wait(.4)
            slots[kind]+=1;self.results.append(dict(kind=kind,visual_pick_xy=xy.tolist(),destination=destination,grasp_forces_N=forces))
            self.save_rgb(f'placed_{len(self.results):02d}.png',self.overview())
            yield None
        yield from self.arm.move([.43,0,1.13])
        yield from self.arm.wait(1.)
        self.event('VERIFY','Independent ground-truth scoring of bin occupancy')
        final=[]
        for f in CONFIG['fruits']:
            p=self.sim.data.body(f['name']).xpos.copy()
            bx,by=CONFIG['bins'][f['kind']]
            correct=abs(p[0]-bx)<.083 and abs(p[1]-by)<.105 and .69<p[2]<.81
            final.append(dict(**f,position=p.tolist(),correct_bin=bool(correct)))
        success=all(f['correct_bin'] for f in final) and len(self.results)==6
        self.perceive();self.save_rgb('scene_final.png',self.overview());self.save_rgb('vision_final.png',self.rgb)
        (self.output/'physics_audit.json').write_text(json.dumps(self.audit,indent=2))
        result=dict(success=success,sorted=sum(f['correct_bin'] for f in final),total=6,placements=self.results,verification=final,simulation_seconds=self.sim.data.time)
        result['dynamics']={'peak_joint_speed_rad_s':self.arm.peak_speed.tolist(),'peak_joint_torque_Nm':self.arm.peak_torque.tolist()}
        (self.output/'result.json').write_text(json.dumps(result,indent=2))
        self.event('COMPLETE' if success else 'FAILED',f"{result['sorted']}/6 in correct bins")
        yield None
        if not success:raise RuntimeError('Sorting verification failed; see result.json')
    def close(self):
        self.close_scene()
