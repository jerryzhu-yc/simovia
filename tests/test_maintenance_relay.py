"""Physical acceptance tests: no policy mocks and no learned model required."""
import numpy as np
import mujoco
import pytest
from demo.maintenance_relay.core import Demo

@pytest.mark.parametrize('delay',[2.]*10)
def test_complete_relay_repeated(delay,tmp_path):
    demo=Demo(tmp_path,station_delay=delay)
    for _ in demo.run():pass
    assert demo.success and demo.done and demo.pressed and demo.placed()
    assert demo.collision_peak==0
    stages=[e['stage'] for e in demo.events]
    assert stages==['STABILIZE','PICK','LOAD','DELIVER','WAIT_STATION','UNLOAD','INSTALL','RESET_DEVICE','COMPLETE']
    wait=next(e['time'] for e in demo.events if e['stage']=='WAIT_STATION')
    unload=next(e['time'] for e in demo.events if e['stage']=='UNLOAD')
    assert unload-wait>=delay-.001
    # Completed jobs cannot restart or continue emitting actions.
    time=demo.data.time;q=demo.data.qpos.copy();ctrl=demo.data.ctrl.copy()
    demo.tick()
    assert list(demo.run())==[]
    assert demo.data.time==time
    np.testing.assert_array_equal(demo.data.qpos,q)
    np.testing.assert_array_equal(demo.data.ctrl,ctrl)
    demo.close()

@pytest.mark.parametrize('delay',[0.,5.,10.])
def test_station_delay_keeps_robots_stable(delay,tmp_path):
    demo=Demo(tmp_path,station_delay=delay)
    for _ in demo.run():pass
    assert demo.success and demo.collision_peak==0
    demo.close()

def test_dropped_load_stops_before_handover(tmp_path):
    demo=Demo(tmp_path);run=demo.run()
    while demo.stage!='DELIVER':next(run)
    # Fault injection only: move the free payload off the tray.
    j=demo.model.body_jntadr[demo.payload];qa=demo.model.jnt_qposadr[j]
    demo.data.qpos[qa+1]+=.4;mujoco.mj_forward(demo.model,demo.data)
    with pytest.raises(RuntimeError,match='离开托盘'):next(run)
    assert demo.done and not demo.success and demo.stage=='FAILED'
    assert 'UNLOAD' not in [e['stage'] for e in demo.events]
    demo.close()

def test_button_requires_physical_contact(tmp_path):
    demo=Demo(tmp_path)
    gid=demo.model.geom('reset_button_geom').id
    demo.model.geom_contype[gid]=0;demo.model.geom_conaffinity[gid]=0
    with pytest.raises(RuntimeError,match='没有实际接触'):
        for _ in demo.run():pass
    assert demo.done and not demo.success and not demo.pressed
    assert demo.placed()
    demo.close()
