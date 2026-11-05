from pathlib import Path
import json,math
import numpy as np
import cv2,yaml
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
    center=np.array([.014*(t*t-0.5),.050*t,0])
    radius=.025*(.30+.70*max(0,1-t*t)**.35)
    for j in range(M):
        a=2*np.pi*j/M;verts.append(center+[radius*np.cos(a),0,radius*np.sin(a)])
for i in range(N):
    for j in range(M):
        a=i*M+j;b=i*M+(j+1)%M;c=(i+1)*M+(j+1)%M;d=(i+1)*M+j
        faces.extend([(a,b,c),(a,c,d)])
faces.extend([(0,j+1,j) for j in range(1,M-1)])
off=N*M;faces.extend([(off,off+j,off+j+1) for j in range(1,M-1)])
mesh_obj('banana.obj',verts,faces)
save('banana.xml','''<mujoco><asset><mesh name="banana_mesh" file="banana.obj"/></asset><worldbody><body name="fruit"><freejoint name="free"/><geom name="banana" type="mesh" mesh="banana_mesh" mass=".00001" contype="0" conaffinity="0" rgba="1 .79 .035 1" friction="2 .02 .002"/><geom type="ellipsoid" size=".026 .050 .025" rgba="0 0 0 0" mass=".045" friction="2 .02 .002"/><geom type="sphere" size=".008" pos=".014 -.052 0" rgba=".25 .12 .025 1" contype="0" conaffinity="0" mass=".0001"/></body></worldbody></mujoco>''')
save('apple.xml','''<mujoco><worldbody><body name="fruit"><freejoint name="free"/><geom name="apple" type="ellipsoid" size=".028 .028 .027" mass=".060" rgba=".86 .035 .045 1" friction="2 .02 .002"/><geom type="capsule" size=".0025" fromto="0 0 .024 .003 0 .038" rgba=".20 .09 .025 1" contype="0" conaffinity="0" mass=".0001"/><geom type="ellipsoid" size=".011 .004 .001" pos=".010 0 .031" rgba=".12 .38 .045 1" contype="0" conaffinity="0" mass=".0001"/></body></worldbody></mujoco>''')
grapes=[]
for z,y,rs in [(0,-.016,[-.014,.014]),(0,.011,[-.014,.014]),(.020,-.006,[-.010,.010]),(-.012,.031,[0])]:
    for x in rs:grapes.append(f'<geom type="sphere" size=".015" pos="{x} {y} {z}" mass=".00001" contype="0" conaffinity="0" rgba=".44 .06 .66 1" friction="2 .02 .002"/>')
save('grape.xml','<mujoco><worldbody><body name="fruit"><freejoint name="free"/>'+''.join(grapes)+'<geom type="ellipsoid" size=".028 .038 .027" rgba="0 0 0 0" mass=".055" friction="2 .02 .002"/><geom type="capsule" size=".002" fromto="0 -.027 .018 0 -.040 .035" rgba=".20 .27 .04 1" contype="0" conaffinity="0" mass=".0001"/></body></worldbody></mujoco>')
# ArUco calibration plate: target coordinates x right, y down; world y is negative.
dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board=['<mujoco><worldbody><body name="plate"><geom type="box" size=".09 .075 .001" pos=".074 -.058 -.001" rgba="1 1 1 1" contype="0" conaffinity="0"/>']
corners={}
for row in range(4):
    for col in range(5):
        idx=row*5+col;s=.026;pitch=.032;x=col*pitch;y=row*pitch
        bits=cv2.aruco.generateImageMarker(dictionary,idx,6)
        corners[idx]=[[x,y,0],[x+s,y,0],[x+s,y+s,0],[x,y+s,0]]
        for iy in range(6):
            for ix in range(6):
                if bits[iy,ix]==0:
                    unit=s/6;board.append(f'<geom type="box" size="{unit/2} {unit/2} .00006" pos="{x+(ix+.5)*unit} {-y-(iy+.5)*unit} .00006" rgba=".005 .005 .005 1" contype="0" conaffinity="0"/>')
board.append('</body></worldbody></mujoco>');save('calibration_board.xml',''.join(board))
(HERE/'board_corners.json').write_text(json.dumps(corners))
objects=[dict(name='table',type='box',size=[.45,.48,.025],pos=[.37,0,.655],static=True,color='#D3D8DE',friction=1.2)]
for x in [-.045,.785]:
    for y in [-.44,.44]:objects.append(dict(name=f'leg_{x}_{y}',type='box',size=[.025,.025,.315],pos=[x,y,.315],static=True,color='#475569'))
# Neutral bins avoid class-colored false detections. Labels are drawn in the viewer.
bins={'banana':[.30,-.29],'apple':[.30,0],'grape':[.30,.29]}
for label,(x,y) in bins.items():
    objects.append(dict(name=label+'_bin_floor',type='box',size=[.083,.105,.006],pos=[x,y,.686],static=True,color='#B3BEC9'))
    for k,(dx,dy,sx,sy) in enumerate([(-.087,0,.004,.113),(.087,0,.004,.113),(0,-.109,.083,.004),(0,.109,.083,.004)]):
        objects.append(dict(name=f'{label}_bin_wall{k}',type='box',size=[sx,sy,.032],pos=[x+dx,y+dy,.718],static=True,color='#607587'))
arm=dict(name='arm',model='zoo://franka_emika_panda',pos=[0,0,.68],end_effectors=[dict(model='onrobot_2fg7')],sensors=[dict(name='wrist_rgb',type='camera',body='gripper/gripper_base',pos=[.075,.010,0],quat=[2**-.5,-2**-.5,0,0],fovy=65)])
base=dict(name='fruit_sorting',timestep=.002,ctrl_period=10,integrator='implicitfast',impratio=10,scene=dict(floor=dict(size=[2,2],texture='solid'),lights=[dict(pos=[.3,0,3])],objects=objects),robots=[arm],runtime=dict(realtime=False))
fruits=[]
for k,(kind,x,y) in enumerate([('banana',.50,-.15),('apple',.62,-.14),('grape',.50,0),('banana',.62,.01),('apple',.50,.16),('grape',.62,.16)]):
    z=.68+dict(banana=.025,apple=.027,grape=.027)[kind]+.002
    name=f'fruit_{k+1:02d}';fruits.append(dict(name=name,kind=kind))
    base['scene']['objects'].append(dict(name=name,type='model',model=f'assets/{kind}.xml',pos=[x,y,z]))
(HERE/'world.yaml').write_text(yaml.safe_dump(base,sort_keys=False,allow_unicode=True))
calib=dict(base);calib['name']='fruit_hand_eye_calibration'
calib['robots']=[arm]
calib['scene']=dict(base['scene']);calib['scene']['objects']=objects[:5]+[dict(name='target',type='model',model='assets/calibration_board.xml',pos=[.435,.06,.684],static=True)]
(HERE/'calibration.yaml').write_text(yaml.safe_dump(calib,sort_keys=False))
(HERE/'config.json').write_text(json.dumps(dict(bins=bins,fruits=fruits,table_z=.68),indent=2))
print('Generated fruit models, bins, wrist camera, calibration plate and two Simovia worlds.')
