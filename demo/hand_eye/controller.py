"""独立手眼标定控制器。机器人、关节、TCP、采样位姿由 profile 配置。"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
for deps in (HERE/'.deps',HERE.parent/'fruit_sorting'/'.deps'):
    if deps.is_dir(): sys.path.insert(0,str(deps))
from simovia.controller import Robot
from demo.hand_eye.runner import CalibrationDemo
robot=Robot()
try:
    demo=CalibrationDemo(robot)
    for _ in demo.run(): pass
except SystemExit:
    robot.close(); raise
except Exception as error:
    robot.report_result(False,str(error)); raise
else:
    robot.close()
