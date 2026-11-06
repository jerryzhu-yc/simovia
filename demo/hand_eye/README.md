# 独立腕部相机手眼标定 demo

标定场景只含桌面、标定板和机械臂；不包含水果和分拣框。水果分拣读取本 demo 已保存的结果，不再执行标定动作。

## 在软件中运行

1. 重启 Simovia，打开“文件 → 手眼标定 Demo → Panda 手眼标定…”或“UR5e 手眼标定…”。
2. 也可用“打开世界”加载仓库内 demo/hand_eye/scene_panda.yaml 或 scene_ur5e.yaml。
3. 代码页选择 arm，点击“保存并运行”。运行会自动展开相机预览，显示检测到的 ArUco 标记。
4. 控制器按配置采集 12 组姿态：前 10 组求解，后 2 组留出验证。通过后自动保存 results/panda_wrist_rgb.json 或 results/ur5e_wrist_rgb.json。
5. 打开水果场景运行即可。Panda 默认配置已有通过验证的结果，第一次使用水果 demo 也无需先重复标定。

点击实时画面面板的“独立窗口”可放大。相机选择 arm/wrist_rgb，关闭“深度图”查看 RGB 图像。

## 适配其他机械臂

已实际验证 Panda（7 轴）和 UR5e（6 轴）。这两份场景是配置示例，不代表全部机器人可以直接使用相同的姿态。

1. 复制 scene_panda.yaml 或 scene_ur5e.yaml，替换 robots[0].model 为模型库引用或自己的 MJCF/机器人 YAML；挂好相机与末端工具。
2. 复制对应 profiles/*.json，填写 joints、actuators、camera 和 tool_frame。tool_frame 支持 site 或 body，相机必须刚性安装在该末端所属运动链上。
3. 为新机械臂设置可达且不碰撞的 initial_qpos，调整 poses 中的世界坐标位置和旋转矩阵，让固定标定板在每个姿态中可见。至少 12 组，包含两个不同旋转轴；每张图至少识别 6 个标记。
4. 场景 robots[0].args.calibration_profile 指向新的配置文件。profile 的 result 指向该机械臂专属 JSON；相对路径以 profile 文件所在目录为基准。
5. 运行并查看质量指标。如果不可达、视野遮挡、运动退化或留出验证失败，控制器报错，已有合格结果不会被覆盖。

通用运动适配器支持至少 6 个直接驱动 hinge 关节的位置执行器，关节与执行器一一对应，gear=1。欠驱动、浮动基座、力矩控制或非串联机器人需要另写运动适配器。采样轨迹不含通用避障规划，新机械臂需要自行选择适合其工作空间的安全采样路径。本 demo 当前支持 eye-in-hand；固定外部相机的 eye-to-hand 需使用对应算法与坐标约定。

## 文件与坐标约定

- controller.py：标准 Simovia 控制器入口，物理由平台后端推进。
- runner.py：通用关节映射、FK/IK、运动与图像采样。
- calibration.py：ArUco/PnP、PARK 手眼求解、质量验证、结果原子写入及装配匹配校验。
- profiles/：机器人专属关节、TCP、采样姿态与输出路径。
- assets/calibration_board.xml、board_corners.json：同一块 5×4 ArUco 板，DICT_4X4_50，单标记边长 26 mm、间距 6 mm；坐标单位米。
- results/：经过验证的可复用结果；runs/：最近一次采样图片与审计。

输出 gripper_T_camera 将 OpenCV 光学坐标（x 向右、y 向下、z 向前）的点变换到配置的 tool_frame。内参来自仿真相机 fovy 与 640×480 分辨率，不是对真实镜头的内参标定；仿真畸变为零。

求解仅使用渲染 RGB 的 PnP 和编码器 FK。ground_truth_validation_only 只用于误差报告，不参与求解。合格条件：至少 12 张图，最后 2 张的板位姿位置残差 ≤8 mm、旋转残差 ≤2°，平均重投影误差 ≤1 px。

结果记录机械臂模型、工具装配、相机安装与分辨率的指纹。机械臂、夹爪、相机挂载或内参改变后应重新标定；不同机械臂的外参不能混用。移动整个工作台/机械臂在世界中的位置不改变眼在手上的相对外参，但抓取任务的工作空间配置仍需相应更新。

## 依赖

使用 64 位 Python 3.12 虚拟环境，在项目根目录执行：

```bash
python -m pip install -r requirements.txt -r demo/hand_eye/requirements.txt
python -m pip check
python -m simovia doctor
python -m simovia gui
```

环境创建见[主 README](../../README.md#安装)，硬件与渲染限制见[配置建议](../../README.md#配置建议)。本模块与水果 demo 使用相同的 OpenCV 依赖，安装一次即可；实现不依赖水果识别或分拣。采样需要离屏渲染 RGB 图像，不能用“无 GUI”替代图形后端配置。
