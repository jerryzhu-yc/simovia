"""Standard Simovia controller adapter: physics runs only in the platform kernel."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import mujoco
from simovia.world.schema import load_world,parse_world
from simovia.world.builder import build_world
from simovia.world.initial import apply_initial_state
from .core import Arm,Demo,HERE,CONFIG,W,H,annotate

class RemoteState:
    def __init__(self,robot):
        self.robot=robot
        self.spec=parse_world(robot.world_spec) if robot.world_spec else load_world(HERE/'scene.yaml')
        if robot.world_spec:self.spec.base_dir=Path(robot.world_spec['base_dir'])
        self.model,self.index=build_world(self.spec)
        self.data=mujoco.MjData(self.model)
        apply_initial_state(self.spec,self.model,self.data,self.index)
        self.tick=0
        self.sync(robot.observe())
    def sync(self,obs):
        m,d=self.model,self.data
        for name,q in self.robot.joint_state().items():
            j=m.joint(self.robot.name+'/'+name).id;adr=m.jnt_qposadr[j]
            d.qpos[adr]=q
        for (name,jid),v in zip(self.index.robots[self.robot.name].joints.items(),obs.get('qvel',[])):
            d.qvel[m.jnt_dofadr[jid]]=v
        # Object truth is retained ONLY for the final independent audit/scoring.
        for name,pos in obs.get('objects',{}).items():
            if not name.startswith('fruit_'):continue
            bid=m.body(name).id;j=int(m.body_jntadr[bid])
            d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]=pos
        # FK/gravitation are computed from measured joint state, never local stepping.
        ri=self.index.robots[self.robot.name]
        for name,value in obs.get('actuators',{}).items():d.ctrl[ri.actuators[name]]=value['u']
        self.tick=obs['tick'];d.time=obs['time']
        mujoco.mj_forward(m,d)
        for name,force in obs.get("actuator_forces",{}).items():d.actuator_force[ri.actuators[name]]=force
    def remote_step(self,ctrl):
        ri=self.index.robots[self.robot.name]
        self.robot.set_all([float(ctrl[ri.actuators[n]]) for n in self.robot.actuators])
        obs=self.robot.step()
        if obs is False:raise SystemExit(0)
        self.sync(obs)
    def close(self):pass

class NativeDemo(Demo):
    def __init__(self,robot,output):
        self.robot=robot
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.seed=0;self.sim=RemoteState(robot);self.arm=Arm(self.sim)
        configured=robot.args.get('calibration_file')
        self.calibration_path=(self.sim.spec.base_dir/configured).resolve() if configured else HERE.parent/'hand_eye/results/panda_wrist_rgb.json'
        self.arm.q=self.sim.data.qpos[self.arm.qadr].copy()
        self.arm.grip=.038
        self.stage='Ready';self.detail='';self.rgb=None;self.detections=[]
        self.calibration=None;self.results=[];self.audit=[];self.frame=0
        self.camera_id=self.sim.model.camera('arm/wrist_rgb').id
        self.renderer=None;self.view_renderer=None
        self.rgb=self.capture()
        self._last_preview=-100
        original=self.sim.remote_step
        def step(ctrl):
            original(ctrl)
            if self.sim.tick-self._last_preview>=5:
                self._last_preview=self.sim.tick
                if self.calibration is not None:self.perceive()
                else:self.rgb=annotate(self.capture(),[],self.stage+' '+self.detail)
                self.robot.publish_camera('wrist_rgb',self.rgb)
        self.sim.remote_step=step
    def load(self,filename):
        # Sorting stays in this scene; calibration is read from a validated file.
        pass
    def capture(self):return self.robot.camera('wrist_rgb',W,H)
    def overview(self):return self.capture()
    def grasp_contacts(self):
        forces=self.robot.contact_forces()
        return {s:forces.get('gripper/'+s+'_finger',0.) for s in ('left','right')}
    def close(self):pass
