# 机械臂 RGB 视觉水果分拣 Demo

模型库：**Franka Panda + OnRobot 2FG7**。腕部相机随夹爪运动；香蕉、红苹果、紫葡萄各 2 个，分别放入三个框。

## 启动

安装后重启 Simovia，在 **文件 → 水果视觉分拣 Demo…** 打开独立演示窗口。点击“开始：手眼标定 → 识别 → 分拣”；支持暂停、继续、重置和 1/2/4/8 倍速。

也可在项目根目录运行：

~~~bash
python -m demo.fruit_sorting
python -m demo.fruit_sorting --headless --video --output demo/fruit_sorting/runs/latest
python -m demo.fruit_sorting --headless --seed 1 --output demo/fruit_sorting/runs/seed1
~~~

新增依赖为 OpenCV，其他依赖沿用 Simovia：

~~~bash
python -m pip install -r demo/fruit_sorting/requirements.txt
~~~

也支持安装到示例内的 .deps 目录，入口会自动加载，无需修改全局 Python：

~~~bash
python -m pip install --target demo/fruit_sorting/.deps --no-deps -r demo/fruit_sorting/requirements.txt
~~~

原平台可直接打开 world.yaml 查看、编辑水果场景。自动标定、识别、分拣由专用演示入口驱动；仅打开 YAML 再点原平台“运行”不会启动视觉流水线。专用窗口使用同一套 Simovia 世界组装与 Simulation，在同一线程顺序处理物理和图像，避开当前进程控制器不传相机图像的限制。

## 实际流程

1. 校准场景中，机械臂实际移动至 12 个不同位置/姿态。
2. 从腕部 RGB 渲染图检测 ArUco 标记，亚像素角点细化，solvePnP 得到标定板到相机的位姿。
3. 前 10 组编码器正运动学与图像位姿用于 OpenCV PARK 手眼标定（AX=XB）；后 2 组用于标定板位置一致性验证。保存所有样本、重投影误差和变换矩阵。
4. 加载生产场景，六个水果在台面自然落稳。RGB 转 HSV，阈值分割、形态学清理、轮廓筛选，画出边框、类别和估计 XY。
5. 用 world_T_TCP × TCP_T_camera 将像素射线转换到世界坐标，再与各类水果的轮廓中心高度平面相交。没有使用深度图或水果真值位置来决定抓取坐标。
6. 笛卡尔插值、阻尼最小二乘 IK、位置伺服及重力补偿驱动机械臂。张开、下降、闭合、抬升；抬升后检查双侧指尖接触力，空抓会停止报错。
7. 高位搬运至对应框、下降、张开夹爪、回到扫描位，再次识别剩余水果。
8. 最后独立读取物体位置检查落框；真值仅用于验证与报告，不用于抓取规划。

没有把水果焊接到夹爪，也没有瞬移水果来制造成功。初始位姿初始化及 --seed 的 ±3 mm 初始 XY 扰动只在场景启动时发生。

## 坐标与标定

- 内参来自已知仿真 fovy=65°、640×480 分辨率，畸变系数为 0。本例做手眼外参标定，不把已知内参称作重新标定。
- TCP 为 arm/gripper/gripper_tcp。
- hand_eye.json 的 gripper_T_camera 使用 OpenCV 光学坐标：x 右、y 下、z 前。
- MuJoCo 相机到 OpenCV 光学坐标转换为 diag(1,-1,-1)。
- 模型中的真实相机安装变换仅用于报告标定误差；抓取使用图像计算出的手眼矩阵。
- 世界位置单位为米，齐次变换为 4×4 矩阵。

## HSV 阈值

OpenCV H 为 0–179，S/V 为 0–255：

| 水果 | H | 最小 S | 最小 V |
|---|---|---:|---:|
| 香蕉 | 18–38 | 75 | 65 |
| 苹果 | 0–12 或 170–179 | 85 | 45 |
| 葡萄 | 125–165 | 65 | 35 |

修改 core.py 的 CLASSES 调整阈值。针对黄香蕉、红苹果、紫葡萄；绿色苹果、青葡萄、强烈变色照明、严重遮挡不在当前验收范围。

## 模型与工作空间

- 台面高 0.68 m，Panda 基座装在台面。
- 初始水果 X≈0.50/0.62 m，Y≈−0.15/0/0.16 m。
- 三框中心：(0.30,−0.29)、(0.30,0)、(0.30,0.29) m，每框 2 个。
- 水果为程序生成刚体：椭球苹果带果梗、弯曲香蕉网格、多球葡萄外观。香蕉/葡萄使用简化椭球碰撞包络，不模拟果肉变形或损伤。
- RGB 定位使用已知桌面和类别尺寸先验，是受控单层桌面的 2.5D 抓取，并非通用未知物体 6D 位姿估计。
- 路径采用本工作空间验证的高位搬运方式，没有通用障碍规划；改动箱体、机械臂或物体尺寸后应重新验证。

## 输出

默认保存到 runs/latest/：

- demo.mp4：加 --headless --video 录制，约 2 倍仿真时间播放。
- hand_eye.json：12 组样本、矩阵、训练/留出验证、重投影误差。
- calibration_*.png：标定采集图。
- detections_*.png：检测框、类别与 XY。
- physics_audit.json：抬升后物理位置审计，控制器不读取该文件。
- result.json：放置目标、接触力、最终落框核验。
- scene_initial.png、scene_final.png、dashboard_final.png：截图。

## 平台的最小扩展

复合水果作为场景物体，而不是机器人：

~~~yaml
scene:
  objects:
    - name: fruit_01
      type: model
      model: assets/banana.xml
      pos: [0.50, -0.15, 0.707]
~~~

MJCF 必须单根 body、无执行器。动态模型只含根 body 上一个 freejoint；静态模型需 static: true 且无关节。几何、质量、材质、摩擦由 MJCF 定义。另存世界会处理资产路径。

generate_assets.py 可重建资产及世界文件，**会覆盖本例已有资产与世界**；修改配置后请先备份。

## 本轮验证

- 默认布局及两组 ±3 mm 初始位置扰动：3 轮均 6/6，共 18/18 正确落框。
- 默认手眼误差约 2.23 mm、0.109°，相对于仿真真值，仅用于验证。
- 真实 Qt 离屏控件测试：开始、暂停、继续、完整分拣、重置通过。
- 新增 8 个场景模型、保存路径、HSV、像素反投影测试通过。
- 发布版本（当前 main 基础 + 本 Demo，不含其他未提交改动）全量测试：465 通过、35 项警告、0 失败。
- 迁移至 demo/fruit_sorting 后，重新执行完整标定与分拣，6/6 正确落框。
- 安装后再次从项目根目录直接启动，重新完成 12 姿态手眼标定和 6/6 分拣；示例内隔离安装了 OpenCV，无需修改全局环境。
- 上述为仿真验收，不代表实际机器人部署已完成。

方法参考：[OpenCV 手眼标定/位姿估计](https://docs.opencv.org/4.5.1/d9/d0c/group__calib3d.html)、[OpenCV HSV inRange](https://docs.opencv.org/4.13.0/da/d97/tutorial_threshold_inRange.html)。
