"""Build the maintenance relay from the bundled robot library."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from .factory import generate_factory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def write_xml(tree,path):
    text=ET.tostring(tree.getroot(),encoding='unicode')
    path.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')

def assets():
    dest = HERE / 'assets'
    dest.mkdir(exist_ok=True)
    tree = ET.parse(ROOT / 'models/zoo/unitree_go2/go2.xml')
    root = tree.getroot()
    root.find('compiler').set('meshdir', '../../../models/zoo/unitree_go2/assets')
    trunk = root.find("worldbody/body")
    frame=ET.SubElement(trunk,'body',name='cargo_frame')
    # Low-profile load carrier on four feet above the robot back.
    def cargo_geom(parent,name,pos,size,color,mass=0,contact=False):
        return ET.SubElement(parent,'geom',name=name,type='box',pos=pos,size=size,
            mass=str(mass),rgba=color,contype='1' if contact else '0',
            conaffinity='1' if contact else '0',friction='1.5 .02 .002')
    for x in (-.10,.10):
        for y in (-.06,.06):
            cargo_geom(frame,f'mount_{x}_{y}',f'{x} {y} .132','.023 .022 .054','.25 .28 .30 1',.02,True)
    cargo_geom(frame,'carrier_plate','0 .04 .193','.13 .16 .008','.58 .64 .68 1',.12,True)
    for x in (-.11,.11):
        cargo_geom(frame,f'deck_rail_{x}',f'{x} .04 .214','.012 .15 .013','.22 .27 .3 1',.015,True)
    cargo_geom(frame,'nest_mount','0 .12 .2065','.075 .06 .0055','.42 .47 .51 1',.02,True)
    tray = ET.SubElement(frame,'body',name='cargo_tray',pos='0 .12 .22')
    cargo_geom(tray,'tray_floor','0 0 0','.075 .06 .008','.14 .18 .21 1',.08,True)
    # Replaceable shallow pocket; the raised handle remains accessible from above.
    for name,pos,size in [('front','.07 0 .012','.005 .06 .012'),('back','-.07 0 .012','.005 .06 .012'),('left','0 .055 .016','.065 .005 .008'),('right','0 -.055 .016','.065 .005 .008')]:
        cargo_geom(tray,'tray_'+name,pos,size,'.26 .30 .33 1',.01,True)
    ET.SubElement(tray,'site',name='cargo_center',pos='0 0 .008',size='.005',rgba='0 1 1 1')
    write_xml(tree,dest/'go2_cargo.xml')
    tree = ET.parse(ROOT / 'models/zoo/unitree_g1/g1.xml')
    root = tree.getroot()
    root.find('compiler').set('meshdir','../../../models/zoo/unitree_g1/assets')
    right_wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    ET.SubElement(right_wrist,'geom',name='tool_flange',type='cylinder',
        fromto='.024 -.001 0 .049 -.0035 0',size='.025',mass='0',
        contype='0',conaffinity='0',rgba='.35 .40 .43 1')
    for z in (-.018,.018):
        ET.SubElement(right_wrist,'geom',name='tool_bolt_'+str(z),type='sphere',
            pos=f'.038 -.024 {z}',size='.004',mass='0',contype='0',
            conaffinity='0',rgba='.75 .78 .8 1')
    left_wrist = root.find(".//body[@name='left_wrist_yaw_link']")
    ET.SubElement(left_wrist,'geom',name='left_tool_flange',type='cylinder',
        fromto='.024 .001 0 .049 .0035 0',size='.025',mass='0',
        contype='0',conaffinity='0',rgba='.35 .40 .43 1')
    key=root.find('keyframe')
    if key is not None:root.remove(key)
    write_xml(tree,dest/'g1_service.xml')
    tree=ET.parse(ROOT/'models/grippers/onrobot_2fg7.xml')
    tree.getroot().find('compiler').set('meshdir','../../../models/grippers/assets')
    tree.getroot().find(".//material[@name='polished_aluminium']").set('rgba','.66 .70 .73 1')
    actuator=tree.getroot().find('actuator/position')
    actuator.set('kp','500');actuator.set('forcerange','-25 25')
    write_xml(tree,dest/'relay_gripper.xml')
    # The left gripper presses with a compliant cap on its right fingertip.
    finger=tree.getroot().find(".//body[@name='right_finger']")
    ET.SubElement(finger,'geom',name='button_probe',type='sphere',pos='.029 -.055 0',
        size='.009',mass='.005',rgba='.15 .18 .2 1')
    ET.SubElement(finger,'site',name='button_tcp',pos='.029 -.055 0',size='.003')
    write_xml(tree,dest/'left_gripper.xml')
    (dest/'module.xml').write_text('''<mujoco model="service_module"><worldbody><body name="module"><freejoint/>
      <geom name="case" type="box" size=".045 .035 .02" mass=".08" rgba=".12 .45 .95 1" friction="1.2 .02 .002"/>
      <geom name="handle" type="box" pos="0 0 .045" size=".024 .024 .025" mass=".02" rgba=".35 .72 1 1" friction="1.2 .02 .002"/>
      <geom type="box" pos="0 -.0355 0" size=".026 .001 .01" mass=".001" rgba=".85 .94 1 1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>''')


def objects():
    def box(name,size,pos,color,**extra):return dict(name=name,type='box',size=size,pos=pos,color=color,static=True,**extra)
    items=[
        dict(name='factory',type='model',model='assets/factory.xml',pos=[0,0,0],static=True),
        box('loading_bench',[.25,.2,.025],[-.65,-.16,.425],'#253D52'),
        box('arm_pedestal',[.12,.12,.14],[-.30,.50,.14],'#253D52'),
        box('service_tray',[.08,.075,.012],[1.60,.36,.638],'#6C757D'),
        box('receiver_worktop',[.30,.25,.012],[1.42,.50,.616],'#8C959D'),
        box('fixture_spacer',[.07,.04,.012],[1.60,.36,.626],'#38434B'),
        box('panel',[.19,.025,.17],[2.22,-.11,.88],'#253D52'),
        box('panel_post',[.035,.035,.35],[2.38,-.11,.35],'#253D52'),
        box('reset_button',[.03,.015,.03],[2.17,-.007,.84],'#F2BB45'),
        dict(name='alarm',type='sphere',size=[.035],pos=[2.22,-.11,1.10],color='#F65060',static=True),
        dict(name='module_blue',type='model',model='assets/module.xml',pos=[-.65,-.16,.472],color='#2879E7'),
        box('spare_red',[.045,.035,.04],[-.79,-.16,.49],'#EB526C'),
    ]
    # Load-bearing bench frame; open front leaves room for the approaching robot.
    for x in (1.16,1.67):
        for y in (.35,.71):
            items.append(box(f'receiver_leg_{x}_{y}',[.02,.02,.29],[x,y,.314],'#71808A'))
            items.append(box(f'receiver_foot_{x}_{y}',[.04,.04,.012],[x,y,.012],'#30373D'))
        items.append(box(f'receiver_sidebeam_{x}',[.02,.20,.025],[x,.53,.16],'#71808A'))
    for z in (.16,.58):
        items.append(box('receiver_backbeam_'+str(z),[.275,.02,.02],[1.415,.71,z],'#71808A'))
    for x in (1.54,1.66):items.append(box('guide_x_'+str(x),[.005,.05,.015],[x,.36,.665],'#303A43'))
    for y in (.31,.41):items.append(box('guide_y_'+str(y),[.055,.005,.015],[1.60,y,.665],'#303A43'))
    for i in range(10):items.append(box('lane_'+str(i),[.06,.008,.001],[i*.25-.15,-.28,.001],'#F2BB45'))
    return items

def document():
    legs={f'{leg}_{kind}_joint':q for leg in ('FL','FR','RL','RR') for kind,q in [('hip',0),('thigh',.9),('calf',-1.8)]}
    stance={f'{side}_{kind}_joint':q for side in ('left','right') for kind,q in [('hip_pitch',-.2),('knee',.4),('ankle_pitch',-.2)]}
    return dict(name='异构协作 · 无人厂区维修',timestep=.002,ctrl_period=8,integrator='implicitfast',impratio=100,
        scene=dict(floor=dict(size=[5,3],texture='solid',friction=1),lights=[dict(pos=[0,-2,4]),dict(pos=[2,2,3])],objects=objects()),
        robots=[
            dict(name='loader',model='zoo://universal_robots_ur5e',pos=[-.30,.50,.28],gripper=dict(model='assets/relay_gripper.xml',pos=[0,0,.1094],quat=[.7071067812,-.7071067812,0,0])),
            dict(name='courier',model='assets/go2_cargo.xml',pos=[0,0,.32],initial_qpos=legs),
            dict(name='technician',model='assets/g1_service.xml',pos=[2.0,.41,.76],quat=[.7071067812,0,0,-.7071067812],initial_qpos=stance,end_effectors=[dict(model='assets/relay_gripper.xml',pos=[0,0,.1094],quat=[.7071067812,-.7071067812,0,0],side='right'),dict(model='assets/left_gripper.xml',pos=[0,0,.1094],quat=[.7071067812,-.7071067812,0,0],side='left')])
        ],runtime=dict(controller_timeout=2000,timeout_policy='pause'),task=dict(max_ticks=18000))

def generate():
    assets()
    generate_factory(HERE/'assets/factory.xml')
    path=HERE/'scene.yaml'
    doc=document()
    for robot in doc['robots']:
        robot['controller']='controller.py'
        robot['args']={'shared_world_controller':True}
    path.write_text(yaml.safe_dump(doc,allow_unicode=True,sort_keys=False))
    return path

if __name__=='__main__':
    print(generate())
