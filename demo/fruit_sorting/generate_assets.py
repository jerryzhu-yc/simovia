from pathlib import Path
import json,math
import xml.etree.ElementTree as ET
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ASSET=HERE/'assets';ASSET.mkdir(exist_ok=True)
def save(name,body):
    (ASSET/name).write_text(body)
def mesh_obj(name,verts,faces):
    save(name,'\n'.join(['v '+' '.join(f'{x:.7f}' for x in p) for p in verts]+['f '+' '.join(str(i+1) for i in f) for f in faces]))
# A curved banana mesh; its convex collision hull is a deliberate rigid approximation.
verts=[];faces=[];N=24;M=16
for i in range(N+1):
    t=-1+2*i/N
    center=np.array([.014*(t*t-0.5),.070*t,0])
    radius=.024*(.30+.70*max(0,1-t*t)**.35)
    for j in range(M):
        a=2*np.pi*j/M;verts.append(center+[radius*np.cos(a),0,radius*np.sin(a)])
for i in range(N):
    for j in range(M):
        a=i*M+j;b=i*M+(j+1)%M;c=(i+1)*M+(j+1)%M;d=(i+1)*M+j
        faces.extend([(a,b,c),(a,c,d)])
faces.extend([(0,j+1,j) for j in range(1,M-1)])
off=N*M;faces.extend([(off,off+j,off+j+1) for j in range(1,M-1)])
mesh_obj('banana.obj',verts,faces)
banana_parts=[]
for i,t in enumerate(np.linspace(-.8,.8,5)):
    x=.014*(t*t-.5); y=.070*t; r=.024*(.30+.70*max(0,1-t*t)**.35)
    banana_parts.append(f'<geom name="banana_contact_{i}" type="ellipsoid" size="{r} .022 {r}" pos="{x} {y} 0" rgba="0 0 0 0"/>')
save('banana.xml','<mujoco><default><geom friction=".35 .0005 .0001" condim="6" solref=".008 1" solimp=".90 .95 .002"/></default><asset><mesh name="banana_mesh" file="banana.obj"/></asset><worldbody><body name="fruit"><freejoint name="free"/><inertial mass=".120" pos="-.002 0 0" diaginertia=".000130 .000028 .000130"/><geom name="banana_visual" type="mesh" mesh="banana_mesh" contype="0" conaffinity="0" rgba="1 .79 .035 1"/>'+''.join(banana_parts)+'<geom type="sphere" size=".007" pos=".014 -.073 0" rgba=".25 .12 .025 1" contype="0" conaffinity="0"/></body></worldbody></mujoco>')
save('apple.xml','<mujoco><default><geom friction=".30 .0005 .0001" condim="6" solref=".008 1" solimp=".90 .95 .002"/></default><worldbody><body name="fruit"><freejoint name="free"/><inertial mass=".100" pos="0 0 0" diaginertia=".00003722 .00003722 .00003844"/><geom name="apple" type="ellipsoid" size=".031 .031 .030" rgba=".86 .035 .045 1"/><geom type="capsule" size=".0025" fromto="0 0 .028 .003 0 .041" rgba=".20 .09 .025 1" contype="0" conaffinity="0"/><geom type="ellipsoid" size=".011 .004 .001" pos=".010 0 .034" rgba=".12 .38 .045 1" contype="0" conaffinity="0"/></body></worldbody></mujoco>')
grapes=[]
for z,y,rs in [(0,-.016,[-.014,.014]),(0,.011,[-.014,.014]),(.020,-.006,[-.010,.010]),(-.012,.031,[0])]:
    for x in rs:grapes.append(f'<geom name="berry_{len(grapes)}" type="sphere" size=".015" pos="{x} {y} {z}" mass="{.100/7}" rgba=".44 .06 .66 1"/>')
# Rigid cluster collision follows the convex outer envelope of the visible berries.
# This approximates a bunch held by broad soft pads; berry deformation is not modeled.
grape_vertices=[];grape_faces=[]
for part in grapes:
    node=ET.fromstring(part);center=np.fromstring(node.get('pos'),sep=' ');off=len(grape_vertices)
    for lat in range(9):
        phi=np.pi*lat/8
        for lon in range(16):
            theta=2*np.pi*lon/16
            grape_vertices.append(center+.015*np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]))
    for lat in range(8):
        for lon in range(16):
            a=off+lat*16+lon;b=off+lat*16+(lon+1)%16;c=b+16;d=a+16
            grape_faces.extend([(a,b,c),(a,c,d)])
mesh_obj('grape_hull.obj',grape_vertices,grape_faces)
visual=''.join(grapes).replace('mass="0.014285714285714287"','mass="0" contype="0" conaffinity="0"')
save('grape.xml','<mujoco><default><geom friction=".25 .0005 .0001" condim="6" solref=".012 1" solimp=".85 .95 .002"/></default><asset><mesh name="cluster_hull" file="grape_hull.obj"/></asset><worldbody><body name="fruit"><freejoint name="free"/>'+visual+'<geom name="cluster_contact" type="mesh" mesh="cluster_hull" mass=".100" rgba="0 0 0 0"/><geom type="capsule" size=".002" fromto="0 -.027 .018 0 -.040 .035" rgba=".20 .27 .04 1" contype="0" conaffinity="0" mass="0"/></body></worldbody></mujoco>')
# Demo-local derivative: preserve upstream meshes, add real camera/mount load,
# force limiting, softer pads. Panda link inertias remain upstream values.
g=ET.parse(ROOT/'models/grippers/onrobot_2fg7.xml')
g.getroot().find('compiler').set('meshdir','../../../models/grippers/assets')
for geom in g.findall('.//geom'):
    if geom.get('mesh','').endswith('_coll'):
        geom.set('name',geom.get('mesh'));geom.set('friction','.6 .001 .0001');geom.set('condim','6')
        geom.set('solref','.008 1');geom.set('solimp','.90 .95 .002')
basebody=g.find('.//body[@name="gripper_base"]')
# Current datasheet mass 1.1 kg = 0.95 + 2*0.075; preserve measured total.
cam=ET.SubElement(basebody,'body',name='camera_payload',pos='.075 .018 0')
ET.SubElement(cam,'geom',name='camera_housing',type='box',size='.018 .008 .012',mass='.12',rgba='.08 .10 .13 1',contype='0',conaffinity='0')
act=g.find('.//actuator/position');act.set('kp','2500');act.set('kv','15');act.set('forcerange','-20 20');act.set('forcelimited','true')
# One tendon unit produces the same force at each finger; 20 N is not divided by two.
ET.indent(g);g.write(ASSET/'gripper.xml',encoding='unicode')
objects=[dict(name='table',type='box',size=[.45,.48,.025],pos=[.37,0,.655],static=True,color='#D3D8DE',friction=.20)]
for x in [-.045,.785]:
    for y in [-.44,.44]:objects.append(dict(name=f'leg_{x}_{y}',type='box',size=[.025,.025,.315],pos=[x,y,.315],static=True,color='#475569'))
# Neutral bins avoid class-colored false detections. Labels are drawn in the viewer.
bins={'banana':[.30,-.29],'apple':[.30,0],'grape':[.30,.29]}
for label,(x,y) in bins.items():
    objects.append(dict(name=label+'_bin_floor',type='box',size=[.083,.105,.006],pos=[x,y,.686],static=True,color='#B3BEC9',friction=.20))
    for k,(dx,dy,sx,sy) in enumerate([(-.087,0,.004,.113),(.087,0,.004,.113),(0,-.109,.083,.004),(0,.109,.083,.004)]):
        objects.append(dict(name=f'{label}_bin_wall{k}',type='box',size=[sx,sy,.032],pos=[x+dx,y+dy,.718],static=True,color='#607587',friction=.20))
arm=dict(name='arm',model='zoo://franka_emika_panda',pos=[0,0,.68],end_effectors=[dict(model='assets/gripper.xml',pos=[0,0,.1092],quat=[2**-.5,-2**-.5,0,0])],sensors=[dict(name='wrist_rgb',type='camera',body='gripper/gripper_base',pos=[.075,.010,0],quat=[2**-.5,-2**-.5,0,0],fovy=65)])
base=dict(name='fruit_sorting',timestep=.002,ctrl_period=10,integrator='implicitfast',impratio=10,scene=dict(floor=dict(size=[2,2],texture='solid'),lights=[dict(pos=[.3,0,3])],objects=objects),robots=[arm],runtime=dict(realtime=False))
fruits=[]
for k,(kind,x,y) in enumerate([('banana',.50,-.15),('apple',.62,-.14),('grape',.50,0),('banana',.62,.01),('apple',.50,.16),('grape',.62,.16)]):
    z=.68+dict(banana=.024,apple=.030,grape=.027)[kind]+.002
    name=f'fruit_{k+1:02d}';fruits.append(dict(name=name,kind=kind))
    base['scene']['objects'].append(dict(name=name,type='model',model=f'assets/{kind}.xml',pos=[x,y,z]))
(HERE/'world.yaml').write_text(yaml.safe_dump(base,sort_keys=False,allow_unicode=True))
(HERE/'config.json').write_text(json.dumps(dict(bins=bins,fruits=fruits,table_z=.68),indent=2))
physics=json.loads((HERE/'physics.json').read_text())
for kind,mass in physics['fruit_mass_kg'].items():
    path=ASSET/(kind+'.xml');tree=ET.parse(path)
    body=tree.find('.//body');inertial=body.find('inertial')
    if inertial is not None:
        old=float(inertial.get('mass'));inertial.set('mass',str(mass))
        inertial.set('diaginertia',' '.join(str(float(x)*mass/old) for x in inertial.get('diaginertia').split()))
    else:body.find("geom[@name='cluster_contact']").set('mass',str(mass))
    tree.find('.//default/geom').set('friction',str(physics['fruit_surface_friction'][kind])+' .0005 .0001')
    ET.indent(tree);tree.write(path,encoding='unicode')
g=ET.parse(ASSET/'gripper.xml')
for geom in g.findall('.//geom'):
    if geom.get('mesh','').endswith('_coll'):geom.set('friction',str(physics['pad_sliding_friction'])+' .001 .0001')
g.find(".//geom[@name='camera_housing']").set('mass',str(physics['camera_mount_mass_kg']))
force=physics['grip_force_limit_N_per_finger']
if not 20<=force<=140:raise ValueError('2FG7 force setting must be 20..140 N')
g.find('.//actuator/position').set('forcerange',f'{-force} {force}')
ET.indent(g);g.write(ASSET/'gripper.xml',encoding='unicode')
# Derive center of mass/inertia from the actual meshes at the specified part masses.
# Freeze them before the platform adds decorative sensor geometry.
import mujoco
for body in g.findall('.//body'):
    if body.get('name') not in ('gripper_base','left_finger','right_finger'):continue
    old=body.find('inertial')
    if old is not None:body.remove(old)
    for geom in body.findall('geom'):
        mesh=geom.get('mesh','')
        mass=.95 if mesh=='base_mesh' else (.065 if mesh.endswith('finger_body') else (.010 if mesh.endswith('finger_pad') else 0.))
        geom.set('mass',str(mass))
g.write(ASSET/'gripper.xml',encoding='unicode')
model=mujoco.MjModel.from_xml_path(str(ASSET/'gripper.xml'))
for body in g.findall('.//body'):
    if body.get('name') not in ('gripper_base','left_finger','right_finger'):continue
    b=model.body(body.get('name')).id
    ET.SubElement(body,'inertial',mass=str(model.body_mass[b]),pos=' '.join(map(str,model.body_ipos[b])),quat=' '.join(map(str,model.body_iquat[b])),diaginertia=' '.join(map(str,model.body_inertia[b])))
ET.indent(g);g.write(ASSET/'gripper.xml',encoding='unicode')
native=json.loads(json.dumps(base))
native['task']={'max_ticks':30000}
native['runtime'].update(controller_timeout=5000,timeout_policy='pause')
native['robots'][0]['controller']='controller.py'
q=[-1.81377252,.83052309,2.25959338,-2.044603,-.61069286,1.46076571,-1.79144536]
native['robots'][0]['initial_qpos']={f'joint{i+1}':v for i,v in enumerate(q)}
native['robots'][0]['initial_qpos'].update({'gripper/left_slide':.019,'gripper/right_slide':.019})
native['robots'][0]['args']={'calibration_file':'../hand_eye/results/panda_wrist_rgb.json','preview_camera':'wrist_rgb'}
(HERE/'scene.yaml').write_text(yaml.safe_dump(native,sort_keys=False,allow_unicode=True))
print('Generated physical assets, native scene/controller binding, and standalone worlds.')
