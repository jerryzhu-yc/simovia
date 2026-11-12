"""Joint-space control and damped least-squares IK for the relay."""
import math
import numpy as np
import mujoco

DOWN = np.array([[1.,0,0],[0,0,-1],[0,1,0]])

class Arm:
    def __init__(self, model, data, prefix, joints, site=None):
        self.m,self.d,self.prefix=model,data,prefix
        self.jids=np.array([model.joint(prefix+'/'+n).id for n in joints])
        self.qa=model.jnt_qposadr[self.jids]
        self.va=model.jnt_dofadr[self.jids]
        self.aids=np.array([next(i for i in range(model.nu) if model.actuator_trnid[i,0]==j and model.actuator_trntype[i]==0) for j in self.jids])
        self.sid=model.site(site or prefix+'/gripper/gripper_tcp').id
        self.q=data.qpos[self.qa].copy()
        self.ik=mujoco.MjData(model)
        self.jp=np.zeros((3,model.nv));self.jr=self.jp.copy()
    def solve(self,p,R=DOWN,seed=None,tol=.001):
        q=self.q.copy() if seed is None else np.array(seed,dtype=float)
        self.ik.qpos[:]=self.d.qpos
        for _ in range(200):
            self.ik.qpos[self.qa]=q
            mujoco.mj_kinematics(self.m,self.ik);mujoco.mj_comPos(self.m,self.ik)
            ep=np.asarray(p)-self.ik.site_xpos[self.sid]
            rot=R@self.ik.site_xmat[self.sid].reshape(3,3).T
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,rot.ravel())
            er=np.empty(3);mujoco.mju_quat2Vel(er,quat,1.)
            if np.linalg.norm(ep)<tol and np.linalg.norm(er)<.01:return q
            mujoco.mj_jacSite(self.m,self.ik,self.jp,self.jr,self.sid)
            J=np.vstack([self.jp[:,self.va],.3*self.jr[:,self.va]])
            dq=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(6),np.r_[ep,.3*er])
            q+=dq*min(1.,.15/max(np.linalg.norm(dq),1e-9))
            q=np.clip(q,self.m.jnt_range[self.jids,0]+.005,self.m.jnt_range[self.jids,1]-.005)
        raise RuntimeError(f'{self.prefix} IK {np.round(p,3)}: pos={np.linalg.norm(ep):.3f} rot={np.linalg.norm(er):.3f}')
    def hold(self,ctrl):
        bias=self.d.qfrc_bias[self.va]/self.m.actuator_gainprm[self.aids,0]
        ctrl[self.aids]=self.q+bias

class Dog:
    def __init__(self,m,d):
        self.m,self.d=m,d
        self.aids=np.array([m.actuator('courier/'+leg+'_'+kind).id for leg in ('FL','FR','RL','RR') for kind in ('hip','thigh','calf')])
        jids=m.actuator_trnid[self.aids,0]
        self.qa=m.jnt_qposadr[jids];self.va=m.jnt_dofadr[jids]
        self.home=np.tile([0.,.9,-1.8],4)
        self.base=m.body('courier/base').id
        self.phase=0.
    def control(self,ctrl,drive=0.,dt=.016):
        self.phase+=2*math.pi*2.4*dt
        target=self.home.copy()
        R=self.d.xmat[self.base].reshape(3,3)
        yaw=math.atan2(R[1,0],R[0,0])
        wanted=float(np.clip(-2*self.d.xpos[self.base,1],-.4,.4))
        turn=float(np.clip(.5*(wanted-yaw),-.12,.12))*min(1.,drive/.25)
        for i,phase in enumerate((0,math.pi,math.pi,0)):
            amplitude=drive+(-turn if i in (0,2) else turn)
            target[3*i+1]+=.35*amplitude*math.sin(self.phase+phase)
            target[3*i+2]+=.55*amplitude*math.sin(self.phase+phase+.5)
        ctrl[self.aids]=np.clip(np.tile([25,60,60],4)*(target-self.d.qpos[self.qa])-np.tile([.5,1,1],4)*self.d.qvel[self.va],self.m.actuator_ctrlrange[self.aids,0],self.m.actuator_ctrlrange[self.aids,1])

class Balance:
    def __init__(self,m,d):
        self.m,self.d=m,d
        self.ids=np.array([i for i in range(m.nu) if m.actuator(i).name.startswith('technician/') and '/gripper' not in m.actuator(i).name])
        self.jids=m.actuator_trnid[self.ids,0]
        self.qa=m.jnt_qposadr[self.jids];self.va=m.jnt_dofadr[self.jids]
        self.home=d.qpos[self.qa].copy()
        self.pelvis=m.body('technician/pelvis').id
        self.feet=[m.body('technician/'+s+'_ankle_roll_link').id for s in ('left','right')]
        self.R0=d.xmat[self.pelvis].reshape(3,3).copy()
        self.origin=self.offset()
        self.prev=np.zeros(2)
    def offset(self):
        return (self.R0.T@(self.d.xpos[self.pelvis]-self.d.xpos[self.feet].mean(axis=0)))[:2]
    def control(self,ctrl,dt=.016):
        xy=self.offset()-self.origin
        vel=(xy-self.prev)/dt;self.prev=xy
        ctrl[self.ids]=self.home
        for s in ('left','right'):
            ctrl[self.m.actuator('technician/'+s+'_ankle_pitch_joint').id]+=2*xy[0]+.4*vel[0]
            ctrl[self.m.actuator('technician/'+s+'_ankle_roll_joint').id]-=2.*xy[1]+.4*vel[1]
