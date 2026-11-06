"""Calibration reuse must never silently accept a different camera assembly."""
from pathlib import Path
import copy,json
import numpy as np
import pytest
cv2=pytest.importorskip('cv2')
from simovia.world.schema import load_world,absolutize_world_paths
from simovia.world.builder import build_world
from demo.hand_eye.calibration import (load_result,solve,save_result,transform,inverse,
                                       intrinsics,rig_identity,TOOL_FRAME)
ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/'demo/hand_eye';FRUIT=ROOT/'demo/fruit_sorting'

def test_fruit_has_no_board_and_reuses_separate_calibration():
    spec=load_world(FRUIT/'scene.yaml')
    assert all(o.name!='target' for o in spec.scene.objects)
    path=spec.base_dir/spec.robots[0].args['calibration_file']
    result=load_result(path,spec)
    assert result['profile']=='panda' and result['held_out_samples']==2
    from demo.fruit_sorting.core import Demo
    assert not hasattr(Demo,'calibrate')

@pytest.mark.parametrize('profile',['panda','ur5e'])
def test_independent_calibration_scenes_compile(profile):
    spec=load_world(CAL/f'scene_{profile}.yaml')
    m,index=build_world(spec)
    assert 'target' in index.objects
    assert not any(n.startswith('fruit_') for n in index.objects)
    assert len(spec.robots[0].initial_qpos) in (8,9)
    load_result(CAL/f'results/{profile}_wrist_rgb.json',spec)

def test_wrong_mount_and_missing_file_are_rejected(tmp_path):
    spec=load_world(FRUIT/'scene.yaml')
    with pytest.raises(ValueError,match='缺少'):load_result(tmp_path/'missing.json',spec)
    spec.robots[0].sensors[0].pos=(.08,.01,0)
    with pytest.raises(ValueError,match='不匹配'):load_result(CAL/'results/panda_wrist_rgb.json',spec)
    ur=load_world(CAL/'scene_ur5e.yaml')
    with pytest.raises(ValueError,match='不匹配'):load_result(CAL/'results/panda_wrist_rgb.json',ur)

def test_invalid_result_never_replaces_working_file(tmp_path):
    good=json.loads((CAL/'results/panda_wrist_rgb.json').read_text())
    path=tmp_path/'result.json';save_result(path,good);old=path.read_bytes()
    bad=copy.deepcopy(good);bad['quality']['validated']=False
    with pytest.raises(ValueError):save_result(path,bad)
    assert path.read_bytes()==old
    bad=copy.deepcopy(good);bad['gripper_T_camera'][0][0]=2
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='刚体变换'):load_result(path,load_world(FRUIT/'scene.yaml'))

def test_save_as_keeps_calibration_references(tmp_path):
    for source in [FRUIT/'scene.yaml',CAL/'scene_panda.yaml']:
        path=tmp_path/source.name
        path.write_text(absolutize_world_paths(source.read_text(),source.parent))
        spec=load_world(path)
        key='calibration_file' if source.parent==FRUIT else 'calibration_profile'
        assert Path(spec.robots[0].args[key]).is_file()
        build_world(spec)

def synthetic_samples():
    X=transform(cv2.Rodrigues(np.array([.2,-.1,.3]))[0],[.05,.01,.03])
    B=transform(np.eye(3),[.5,.1,.6]);samples=[]
    rng=np.random.default_rng(4)
    for i in range(12):
        G=transform(cv2.Rodrigues(rng.normal(0,.3,3))[0],rng.uniform(-.2,.2,3))
        C=inverse(X)@inverse(G)@B
        samples.append(dict(gripper_to_world=G.tolist(),target_to_camera=C.tolist(),reprojection_px=.1))
    return X,samples

def test_general_solver_recovers_known_transform_and_rejects_degenerate_motion():
    X,samples=synthetic_samples()
    identity={'assembly':{'resolution':[640,480]}}
    result=solve(samples,intrinsics(65),identity,'synthetic')
    assert np.allclose(result['gripper_T_camera'],X,atol=1e-8)
    for sample in samples:sample['gripper_to_world']=np.eye(4).tolist()
    with pytest.raises(ValueError,match='退化'):solve(samples,intrinsics(65),identity,'bad')

def test_holdout_failure_is_rejected():
    X,samples=synthetic_samples();samples[-1]['target_to_camera'][0][3]+=.05
    with pytest.raises(ValueError,match='留出'):solve(samples,intrinsics(65),{'assembly':{'resolution':[640,480]}},'bad')
