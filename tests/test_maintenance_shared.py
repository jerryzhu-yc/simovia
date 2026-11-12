import numpy as np
import mujoco
from simovia.kernel.sim import Simulation
from simovia.world.schema import load_world
from demo.maintenance_relay.scene import HERE

def test_shared_controller_pause_finish_reset_and_table_clearance():
    spec=load_world(HERE/'scene.yaml')
    with Simulation(spec) as sim:
        initial=sim.data.qpos.copy()
        assert not sim._procs and sim._coordinator.demo.data is sim.data
        m=sim.model
        dogs=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith('courier/') and (m.geom_contype[g] or m.geom_conaffinity[g])]
        tables=[g for g in range(m.ngeom) if any(t in m.geom(g).name for t in ('loading_bench','bench_upright','bench_foot','bench_shelf','bench_front_trim','receiver_','fixture_spacer','service_tray'))]
        for g in tables:
            if any(t in m.geom(g).name for t in ('bench_upright','bench_foot','bench_shelf')):
                assert m.geom_contype[g] and m.geom_conaffinity[g]
        pairs=np.array([(a,b) for a in dogs for b in tables])
        radii=m.geom_rbound[pairs].sum(axis=1)
        gaps=[]
        def check():
            delta=sim.data.geom_xpos[pairs[:,0]]-sim.data.geom_xpos[pairs[:,1]]
            for k in np.flatnonzero(np.linalg.norm(delta,axis=1)-radii<.08):
                a,b=pairs[k]
                gaps.append(mujoco.mj_geomDistance(m,sim.data,int(a),int(b),.08,None))
        sim.run(max_ticks=400,on_tick=check)
        assert sim.tick==400
        paused=sim.data.qpos.copy()
        sim.run(stop_check=lambda:True)
        np.testing.assert_array_equal(sim.data.qpos,paused)
        result=sim.run(on_tick=check)
        assert result['success'] and len(result['done'])==3
        assert min(gaps)>.01, min(gaps)
        print('minimum table clearance',min(gaps))
        final=sim.data.qpos.copy();tick=sim.tick
        sim.run()
        assert sim.tick==tick
        np.testing.assert_array_equal(sim.data.qpos,final)
        sim.reset()
        np.testing.assert_allclose(sim.data.qpos,initial,atol=1e-12)
        assert sim.tick==0 and not sim._coordinator.done
        assert sim.run()['success']
