# 模型来源

- go2_cargo.xml：由 models/zoo/unitree_go2/go2.xml 派生，增加运输架；原模型许可证见同目录 LICENSE.go2。
- g1_service.xml：由 models/zoo/unitree_g1/g1.xml 派生，增加左腕按钮接触探头并移除未使用的 keyframe；原模型许可证见 LICENSE.g1。
- relay_gripper.xml：基于项目 models/grippers/onrobot_2fg7.xml，使用原有 meshes，明确设定本演示的伺服刚度和力限。
- module.xml：本演示新增的简化维修模块。

所有网格通过相对路径复用项目内的原始资产，不复制二进制模型。附件可由上一级 scene.py 重建。

- left_gripper.xml：与 relay_gripper.xml 相同的驱动夹爪，右指尖增加按钮软触头。
- factory.xml：factory.py 生成的车间视觉布置，不增加质量或碰撞。

工业版载具：cargo_frame 为固定于 Go2 的承载刚体，几何质量参与动力学；cargo_tray 为其刚性子体。接收台的承载结构定义在 scene.py，factory.py 仅补充孔阵、标签和传感器外观。
