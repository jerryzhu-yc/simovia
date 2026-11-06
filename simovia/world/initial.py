"""Apply named initial joint positions without changing kinematic references."""
import numpy as np
import mujoco
from simovia.errors import WorldError

def apply_initial_state(spec, model, data, index):
    for robot in spec.robots:
        if not robot.initial_qpos:continue
        ri=index.robots[robot.name]
        for name,value in robot.initial_qpos.items():
            if name not in ri.joints:
                raise WorldError(f"机器人 {robot.name} 初始关节不存在: {name}")
            jid=ri.joints[name]
            if int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise WorldError(f"初始关节 {name} 只支持 hinge/slide")
            if not np.isfinite(value) or (model.jnt_limited[jid] and not model.jnt_range[jid,0]<=value<=model.jnt_range[jid,1]):
                raise WorldError(f"初始关节 {name} 超过范围或不是有限数")
            data.qpos[model.jnt_qposadr[jid]]=value
        mujoco.mj_forward(model,data)
        for aid in ri.actuators.values():
            if int(model.actuator_biastype[aid])==int(mujoco.mjtBias.mjBIAS_AFFINE) and model.actuator_biasprm[aid,1]<0:
                data.ctrl[aid]=data.actuator_length[aid]
    mujoco.mj_forward(model,data)
