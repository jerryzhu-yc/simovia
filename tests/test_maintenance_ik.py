"""Regression for GUI observation changing relay feedback and boundary IK."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from demo.maintenance_relay.core import Demo
from demo.maintenance_relay.scene import HERE
from simovia.mcp_server import SimoviaSession


def test_reported_ik_boundary_keeps_original_tolerances():
    # Captured from the unmodified controller at t=39.568 with GUI observations.
    state=json.loads((Path(__file__).parent/'fixtures/maintenance_ik_boundary.json').read_text())
    demo=Demo()
    try:
        arm=demo.right
        demo.data.qpos[:]=state['qpos']
        arm.q[:]=state['q']
        before=demo.data.qpos.copy()
        q=arm.solve(state['p'],R=np.array(state['R']))
        arm.ik.qpos[arm.qa]=q
        mujoco.mj_kinematics(demo.model,arm.ik)
        assert np.linalg.norm(arm.ik.site_xpos[arm.sid]-state['p'])<.001
        rotation=np.array(state['R'])@arm.ik.site_xmat[arm.sid].reshape(3,3).T
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,rotation.ravel())
        error=np.empty(3);mujoco.mju_quat2Vel(error,quat,1.)
        assert np.linalg.norm(error)<.01
        limits=demo.model.jnt_range[arm.jids]
        assert np.all(q>=limits[:,0]+.005) and np.all(q<=limits[:,1]-.005)
        np.testing.assert_array_equal(demo.data.qpos,before)
        with pytest.raises(RuntimeError,match=r'pos=.*mm.*rot=.*rad'):
            arm.solve([10.,10.,10.])
    finally:
        demo.close()


def test_observation_cadence_does_not_change_relay_trajectory():
    # GUI snapshots call mj_kinematics. Their cadence must not alter feedback.
    trajectories=[]
    for cadence in (0,1,7):
        session=SimoviaSession()
        try:
            session.call_tool('load_world',{'yaml_path':str(HERE/'scene.yaml')})
            session.call_tool('get_observation',{})
            sim=session.sim
            poses=[]
            def sample():
                if cadence and sim.tick%cadence==0:
                    session._obs_payload()
                poses.append(sim.data.qpos.copy())
            result=sim.run(on_tick=sample)
            assert result['success'] and len(result['done'])==3
            assert sim._coordinator.demo.pressed and sim._coordinator.demo.placed()
            assert sim._coordinator.demo.collision_peak==0
            trajectories.append(np.array(poses))
        finally:
            session.close()
    for trajectory in trajectories[1:]:
        assert trajectory.shape==trajectories[0].shape
        assert np.array_equal(trajectory,trajectories[0]), np.max(np.abs(trajectory-trajectories[0]))


def test_relay_runs_through_gui_backend():
    session=SimoviaSession()
    try:
        session.call_tool('load_world',{'yaml_path':str(HERE/'scene.yaml')})
        result=session.call_tool('run',{})
        assert result['success'] and len(result['done'])==3
        assert session.call_tool('get_observation',{})['task']['success']
    finally:
        session.close()
