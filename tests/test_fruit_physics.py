"""Regression checks for physical parameters and native scene devices."""
from pathlib import Path
import base64,zlib
import numpy as np
import pytest
import mujoco
from simovia.world.schema import load_world
from simovia.kernel.sim import Simulation
from simovia.errors import WorldError
D=Path(__file__).resolve().parents[1]/'demo/fruit_sorting'

def scene():return load_world(D/'scene.yaml')

def test_mass_force_and_friction_contract():
    with Simulation(scene(),autostart_controllers=False) as sim:
        m=sim.model
        assert m.body('fruit_01').mass[0]==pytest.approx(.120)
        assert m.body('fruit_02').mass[0]==pytest.approx(.100)
        assert m.body('fruit_03').mass[0]==pytest.approx(.100)
        assert sum(m.body('arm/gripper/'+n).mass[0] for n in ('gripper_base','left_finger','right_finger'))==pytest.approx(1.1)
        assert m.body('arm/gripper/camera_payload').mass[0]==pytest.approx(.12)
        assert m.actuator('arm/gripper/gripper_act').forcerange.tolist()==[-20.,20.]
        assert np.allclose(m.geom('arm/gripper/left_finger_pad_coll').friction,[.6,.001,.0001])
        assert np.allclose(m.actuator('arm/actuator1').forcerange,[-87,87])
        assert np.allclose(m.actuator('arm/actuator7').forcerange,[-12,12])
        # Initial pose does not modify the model's kinematic joint reference.
        j=m.joint('arm/joint1').id;a=m.jnt_qposadr[j]
        assert sim.data.qpos[a]==pytest.approx(scene().robots[0].initial_qpos['joint1'])
        assert m.qpos0[a]==pytest.approx(0.)

def test_initial_pose_rejects_invalid_joint_values():
    for name,value in [('missing',0.),('joint1',9.),('joint1',float('nan'))]:
        s=scene();s.robots[0].initial_qpos={name:value}
        with pytest.raises(WorldError):Simulation(s,autostart_controllers=False)

@pytest.mark.gl   # camera 设备要建离屏渲染上下文，无 GL 环境（如 Windows CI）跳过
def test_native_camera_and_overlay_are_current_and_bounded():
    with Simulation(scene(),autostart_controllers=False) as sim:
        sim._ident[b'test']='arm';replies=[];sim._send=lambda identity,msg:replies.append(msg)
        for request in [{'op':'camera','name':'missing'}, {'op':'camera','name':'wrist_rgb','width':9999}]:
            sim._device_request(b'test',request);assert replies.pop()['op']=='error'
        sim._device_request(b'test',{'op':'camera','name':'wrist_rgb','width':64,'height':48})
        reply=replies.pop()
        assert reply['op']=='camera'
        raw=zlib.decompress(base64.b64decode(reply['rgb_zlib']))
        assert len(raw)==64*48*3 and sim.tick==0
        rgb=np.zeros((48,64,3),np.uint8);rgb[:,:,0]=255
        sim._device_request(b'test',{'op':'camera_overlay','name':'wrist_rgb','width':64,'height':48,'rgb':base64.b64encode(rgb.tobytes()).decode()})
        assert replies.pop()['op']=='ok'
        assert sim.controller_frames['arm/wrist_rgb'][1].startswith(b'\x89PNG')
        sim.close_device_renderers();assert not sim._vision_renderers

def test_controller_result_not_confused_with_implicit_success():
    with Simulation(scene(),autostart_controllers=False) as sim:
        assert sim._task_success() is None
        sim._st['arm']={'crashed':False}
        sim.controller_results['arm']={'success':True,'detail':'checked'}
        assert sim._task_success() is True
        sim._st['arm']['crashed']=True
        assert sim._task_success() is False

def test_surface_contact_uses_real_friction_not_old_value_two():
    with Simulation(scene(),autostart_controllers=False) as sim:
        for _ in range(120):mujoco.mj_step(sim.model,sim.data)
        fruit=sim.model.body('fruit_02').id;mus=[]
        for c in sim.data.contact:
            if fruit in [sim.model.geom_bodyid[c.geom1],sim.model.geom_bodyid[c.geom2]]:
                mus.append(c.friction[0])
        assert mus and max(mus)==pytest.approx(.30)


@pytest.mark.parametrize('integrator', range(4))
def test_controller_world_roundtrip_preserves_scene(integrator):
    import json
    from simovia.world.schema import (world_to_dict, parse_world, DistanceReward,
                                     ZoneReward, ContactReward, ZoneSpec)
    original = scene()
    original.integrator = integrator
    original.task.reward = [DistanceReward('fruit_01', 'fruit_02'),
                            ZoneReward('fruit_01', ZoneSpec((0, 0, 0), (1, 1, 1))),
                            ContactReward('fruit_01', 'fruit_02')]
    payload = json.loads(json.dumps(world_to_dict(original)))
    restored = parse_world(payload)
    restored.base_dir = Path(payload['base_dir'])
    assert restored == original
    with Simulation(restored, autostart_controllers=False) as sim:
        assert sim.model.ncam > 0
