"""软件内的场景控制器。打开 scene.yaml，选 arm，在代码页保存并运行。

参数与详细实现：physics.json、core.py、native.py。
本脚本读取后端真实 RGB 和编码器，不使用物体真值规划抓取。
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
if (HERE/'.deps').exists():sys.path.insert(0,str(HERE/'.deps'))
sys.path.insert(0,str(HERE.parents[1]))
from simovia.controller import Robot
from demo.fruit_sorting.native import NativeDemo

robot=Robot()
try:
    output=HERE/'runs'/'in_app'
    demo=NativeDemo(robot,output)
    for _ in demo.run():
        pass
    robot.report_result(True,'6/6 水果经过独立落框检查')
    print('水果分拣完成：结果保存到 '+str(output),flush=True)
except SystemExit:
    robot.close()
    raise
except Exception as error:
    robot.report_result(False,str(error))
    print("分拣失败："+str(error),flush=True)
    raise
else:
    robot.close()
