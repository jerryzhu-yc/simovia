"""Visual fit-out of a compact material handling cell.

Fixtures augment existing work surfaces. Decorative geometry has no contact or
mass so the calibrated pickup, walking and handover physics remain unchanged.
"""
import xml.etree.ElementTree as ET

STEEL = '.22 .29 .34 1'
ALUMINIUM = '.58 .65 .69 1'
YELLOW = '.96 .65 .12 1'
BLUE = '.12 .31 .46 1'

def generate_factory(path):
    root = ET.Element('mujoco', model='material_cell')
    ET.SubElement(root, 'compiler', angle='radian')
    body = ET.SubElement(ET.SubElement(root, 'worldbody'), 'body', name='cell')
    def box(name, pos, size, color=STEEL, **kw):
        return ET.SubElement(body, 'geom', name=name, type='box',
            pos=' '.join(map(str, pos)), size=' '.join(map(str, size)),
            rgba=color, contype='1' if name.startswith('bench_') else '0',
            conaffinity='1' if name.startswith('bench_') else '0', mass='0', **kw)
    # Epoxy floor, painted traffic corridor and clearly bounded work areas.
    box('floor_finish', [.85,.25,-.004], [2.45,1.6,.005], '.48 .53 .55 1')
    box('traffic_lane', [.85,0,.0015], [1.55,.30,.001], '.26 .34 .37 1')
    for y in (-.32,.32):
        box('lane_edge_'+str(y), [.85,y,.003], [1.55,.012,.001], YELLOW)
    for x in (.55,1.0,1.45):
        box('arrow_stem_'+str(x), [x,0,.004], [.075,.012,.001], '.8 .85 .85 1')
        for side in (-1,1):
            box('arrow_tip_'+str(x)+'_'+str(side), [x+.055,side*.025,.004],
                [.04,.01,.001], '.8 .85 .85 1', euler=f'0 0 {side*-.65}')
    for x,y,sx,sy in ((-.35,.25,.65,.75),(2.05,.35,.55,.65)):
        for dy in (-sy,sy):
            box(f'bay_h_{x}_{dy}', [x,y+dy,.004], [sx,.008,.001], '.74 .79 .79 1')
        for dx in (-sx,sx):
            box(f'bay_v_{x}_{dx}', [x+dx,y,.004], [.008,sy,.001], '.74 .79 .79 1')
    # Four-post aluminium feed stand with adjustable feet and lower bracing.
    for x in (-.86,-.44):
        for y in (-.32,0):
            box(f'bench_upright_{x}_{y}', [x,y,.20], [.016,.016,.20], ALUMINIUM)
            box(f'bench_foot_{x}_{y}', [x,y,.015], [.029,.029,.015])
    box('bench_shelf', [-.65,-.16,.10], [.235,.18,.012], ALUMINIUM)
    box('bench_front_trim', [-.65,-.361,.418], [.25,.008,.026], ALUMINIUM)
    # Dedicated nests: high contrast stops, plate fasteners and part-ID plates.
    for x in (-.65,-.79):
        box('feed_nest_'+str(x), [x,-.16,.448], [.055,.045,.002], '.16 .20 .23 1')
        for dx in (-.053,.053):
            box(f'feed_locator_{x}_{dx}', [x+dx,-.16,.46], [.003,.035,.01], YELLOW)
        box('feed_id_'+str(x), [x,-.225,.452], [.04,.012,.001], '.86 .88 .84 1')
    # Receive station: replaceable fixture plate with mounting grid and sensor.
    for x in (1.18,1.30,1.42,1.54):
        for y in (.32,.44,.56,.68):
            box(f'fixture_grid_{x}_{y}',[x,y,.629],[.004,.004,.001],'.24 .28 .31 1')
    box('receiver_front_trim',[1.42,.248,.606],[.30,.004,.02],ALUMINIUM)
    box('receiver_backstop',[1.42,.74,.66],[.30,.01,.033],BLUE)
    box('fixture_sensor',[1.60,.437,.668],[.016,.012,.014],'.12 .16 .18 1')
    box('sensor_lens',[1.60,.424,.668],[.006,.001,.006],'.1 .65 .85 1')
    for x in (1.535,1.665):
        for y in (.297,.423):
            box(f'fixture_bolt_{x}_{y}',[x,y,.651],[.004,.004,.001],'.12 .15 .17 1')
    box('receiver_id',[1.28,.242,.607],[.08,.001,.013],'.83 .86 .86 1')
    # Loader base plate with four anchor bolt heads.
    box('loader_foot', [-.3,.5,.015], [.17,.17,.015])
    for x in (-.43,-.17):
        for y in (.37,.63):box(f'anchor_{x}_{y}',[x,y,.034],[.012,.012,.006],ALUMINIUM)
    # Equipment cabinet behind the existing panel; button housing reaches face.
    box('equipment_cabinet', [2.22,-.285,.43], [.205,.15,.43], '.68 .72 .73 1')
    box('cabinet_plinth', [2.22,-.285,.035], [.22,.165,.035])
    box('panel_bracket', [2.38,-.11,.708], [.04,.04,.02], ALUMINIUM)
    box('button_housing', [2.17,-.054,.84], [.039,.035,.039], '.15 .18 .21 1')
    box('screen_frame', [2.27,-.082,.965], [.095,.004,.057], '.08 .12 .15 1')
    box('screen_glass', [2.27,-.076,.965], [.082,.002,.045], '.08 .28 .32 1')
    for i in range(3):
        box('screen_readout_'+str(i), [2.25,-.073,.943+i*.02], [.05-i*.008,.001,.003], '.4 .78 .75 1')
    box('alarm_base', [2.22,-.11,1.054], [.028,.028,.018], ALUMINIUM)
    for i in range(7):
        box('cabinet_vent_'+str(i), [2.426,-.28,.22+i*.025], [.001,.085,.004])
    # Stock shelving with lipped reusable totes and palletized packing cartons.
    for rack_x in (-.55,.65):
        y=1.28
        for dx in (-.44,.44):
            for dy in (-.18,.18):
                box(f'rack_post_{rack_x}_{dx}_{dy}', [rack_x+dx,y+dy,.64], [.025,.025,.64], BLUE)
                box(f'rack_base_{rack_x}_{dx}_{dy}', [rack_x+dx,y+dy,.012], [.05,.05,.012], YELLOW)
        for level,z in enumerate((.12,.55,.98)):
            box(f'shelf_{rack_x}_{level}', [rack_x,y,z], [.46,.21,.018], ALUMINIUM)
            box(f'beam_{rack_x}_{level}', [rack_x,y-.215,z], [.46,.012,.027], YELLOW)
            for slot,dx in enumerate((-.29,0,.29)):
                x=rack_x+dx
                color=('.12 .35 .45 1' if level!=1 else '.40 .48 .47 1')
                box(f'tote_base_{rack_x}_{level}_{slot}',[x,y,z+.04],[.12,.16,.02],color)
                for sy in (-1,1):
                    box(f'tote_side_{rack_x}_{level}_{slot}_{sy}',[x,y+sy*.15,z+.11],[.12,.01,.07],color)
                for sx in (-1,1):
                    box(f'tote_end_{rack_x}_{level}_{slot}_{sx}',[x+sx*.11,y,z+.11],[.01,.14,.07],color)
                box(f'tote_label_{rack_x}_{level}_{slot}',[x,y-.161,z+.105],[.05,.001,.022],'.85 .86 .78 1')
    # Packing pallet in the rear corner, with clear stacking seams.
    for i in range(5):
        box('pallet_slat_'+str(i), [2.7,.78+i*.10,.11], [.30,.04,.015], '.53 .37 .20 1')
    for x in (2.48,2.7,2.92):
        box('pallet_runner_'+str(x),[x,.98,.05],[.035,.27,.05],'.42 .29 .17 1')
    for layer in range(2):
        for x in (2.55,2.85):
            box(f'carton_{layer}_{x}',[x,.98,.25+layer*.25],[.14,.23,.12],'.65 .49 .31 1')
            box(f'tape_{layer}_{x}',[x,.98,.371+layer*.25],[.024,.23,.001],'.77 .65 .44 1')
    ET.indent(root)
    path.write_text(ET.tostring(root,encoding='unicode')+'\n')
