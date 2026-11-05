from pathlib import Path
import numpy as np
import pytest
import yaml
from simovia.world.schema import parse_world,load_world,absolutize_world_paths
from simovia.world.builder import build_world
from simovia.errors import WorldError
ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'demo'/'fruit_sorting'
def test_passive_scene_assets_compile():
    m,index=build_world(load_world(DEMO/'world.yaml'))
    assert len(index.robots)==1 and len([n for n in index.objects if n.startswith('fruit_')])==6
    for n in [f'fruit_{i:02d}' for i in range(1,7)]:
        body=index.objects[n]
        assert m.body_jntnum[body]==1
        assert m.body_mass[body]>0
    assert m.nu==8
def test_static_calibration_asset():
    m,index=build_world(load_world(DEMO/'calibration.yaml'))
    assert m.body_jntnum[index.objects['target']]==0
def test_asset_save_as_rewrites_model_paths(tmp_path):
    text=absolutize_world_paths((DEMO/'world.yaml').read_text(),DEMO)
    dest=tmp_path/'world.yaml';dest.write_text(text)
    spec=load_world(dest)
    assert all(Path(o.model).is_absolute() for o in spec.scene.objects if o.model)
    m,_=build_world(spec)
    assert m.nu==8
def test_reject_active_scene_asset(tmp_path):
    p=tmp_path/'active.xml';p.write_text('<mujoco><worldbody><body name="b"><joint name="j"/><geom size=".05"/></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>')
    spec=parse_world({'name':'asset_test','scene':{'objects':[{'name':'asset','type':'model','model':str(p),'static':True}]}})
    with pytest.raises(WorldError,match='没有执行器'):build_world(spec)
def test_reject_free_joint_static_asset():
    spec=parse_world({'name':'asset_test','scene':{'objects':[{'name':'asset','type':'model','model':str(DEMO/'assets/apple.xml'),'static':True}]}})
    with pytest.raises(WorldError,match='不能包含关节'):build_world(spec)
def test_hsv_three_classes_and_red_hue_wrap():
    cv2=pytest.importorskip('cv2')
    from demo.fruit_sorting.core import detect
    hsv=np.zeros((100,240,3),np.uint8)
    for x,h in [(10,0),(70,178),(130,28),(190,145)]:hsv[20:60,x:x+30]=[h,220,210]
    ds=detect(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB))
    assert sorted(d['kind'] for d in ds)==['apple','apple','banana','grape']
def test_calibrated_projection_roundtrip():
    pytest.importorskip('cv2')
    from demo.fruit_sorting.core import unproject,intrinsic,transform
    T=transform(np.diag([1,-1,-1]),[.5,0,1.2]);p=np.array([.57,.13,.707])
    camera=T[:3,:3].T@(p-T[:3,3]);uv=(intrinsic()@camera)[:2]/camera[2]
    assert np.allclose(unproject(uv,T,p[2]),p,atol=1e-10)
def test_ray_behind_camera_rejected():
    pytest.importorskip('cv2')
    from demo.fruit_sorting.core import unproject
    with pytest.raises(ValueError,match='behind'):unproject([320,240],np.eye(4),-1)
