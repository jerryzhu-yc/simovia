# Simovia 演示

包含 G1 双臂收拾毛毯、三机联合维修、视觉分拣和手眼标定。G1 叠毯支持命令行脚本演示与 PICO 遥操作采集；联合维修在独立任务控制台中运行；视觉分拣与手眼标定沿用主窗口场景工作流。

每个 demo 另有独立的 playground 分支（[simovia-playground](https://github.com/jerryzhu-yc/simovia-playground)）：分支根目录就是该 demo 的世界文件、控制器与完整复现说明，`git checkout demo/<名字>` 即跑。

| 场景 | 软件内入口 | 场景文件 |
|---|---|---|
| G1 双臂收拾毛毯（脚本叠毯） | 命令行 `python -m demo.cloth_folding.run` 或加载世界 | [cloth_folding/README.md](cloth_folding/README.md) |
| G1 收拾毛毯（遥操作采集） | 文件 → 打开世界 → `g1_cloth_teleop.yaml` | [worlds/g1_cloth_teleop.yaml](../worlds/g1_cloth_teleop.yaml) |
| G1 收拾毛毯（数据回放增强） | 文件 → 打开世界 → `g1_cloth_augment.yaml` | [worlds/g1_cloth_augment.yaml](../worlds/g1_cloth_augment.yaml) |
| G1 收拾毛毯（π0.5 策略闭环推理） | 文件 → 打开世界 → `g1_cloth_policy_eval.yaml` | [worlds/g1_cloth_policy_eval.yaml](../worlds/g1_cloth_policy_eval.yaml) |
| Go2 × G1 × UR5e 联合维修 | 文件 → Go2 × G1 × UR5e 联合维修 Demo… | [使用说明](maintenance_relay/README.md) |
| 香蕉、苹果、葡萄分拣 | 文件 → 水果视觉分拣 Demo… | [fruit_sorting/scene.yaml](fruit_sorting/scene.yaml) |
| Panda 腕部相机标定 | 文件 → 手眼标定 Demo → Panda 手眼标定… | [hand_eye/scene_panda.yaml](hand_eye/scene_panda.yaml) |
| UR5e 腕部相机标定 | 文件 → 手眼标定 Demo → UR5e 手眼标定… | [hand_eye/scene_ur5e.yaml](hand_eye/scene_ur5e.yaml) |

联合维修可直接运行 `python -m demo.maintenance_relay`，使用平台现有依赖；具体范围、无头验证与录像方式见[联合维修说明](maintenance_relay/README.md)。

## 视觉示例快速运行

先按[主 README 安装说明](../README.md#安装)创建并激活 64 位 Python 3.12 虚拟环境。在仓库根目录统一安装平台和视觉依赖，然后启动软件：

~~~bash
python -m pip install -r requirements.txt -r demo/fruit_sorting/requirements.txt
python -m pip check
python -m simovia doctor
python -m simovia gui
~~~

两个视觉演示的 OpenCV 依赖相同，安装一次即可；NumPy 统一使用核心依赖中的 2.2.x。硬件建议、平台限制与画质说明以[主 README](../README.md#配置建议)为准，目前没有经低配机器认证的最低配置。两个 demo 即使无 GUI 运行也需要可用的离屏渲染；腕部相机为 640×480，降低 GUI 主视口画质不改变标定数据。

通过上表菜单打开场景，代码页选择 arm，点击“保存并运行”或 F5。也可用“打开世界”加载对应 YAML。运行时自动展开 arm/wrist_rgb 相机预览，点击“独立窗口”可放大，查看 RGB 识别图时关闭“深度图”。

默认携带实际标定并通过验证的 Panda、UR5e 数据，首次分拣可直接运行。水果场景没有标定板；重新标定请运行对应标定场景，通过后再加载水果场景。更换机械臂、夹爪、相机安装或内参时应重新标定，配置不匹配会拒绝使用旧数据。

## G1 收拾毛毯

四套世界共用 G1 + 2F85 夹爪，分别覆盖脚本演示、PICO 采集、数据增强回放和 π0.5 策略闭环；详细命令与物理实现见 [cloth_folding/README.md](cloth_folding/README.md)。

- **脚本叠毯**：`python -m demo.cloth_folding.run`，加 `--video out.mp4` 同时录视频。毛毯是 flexcomp 二维网格柔体，双臂用离线关键帧 IK 对折，抓取靠有限刚度的合规抓持力。
- **遥操作采集**：加载 `worlds/g1_cloth_teleop.yaml`，PICO 头显浏览器打开 `http://<PC 的 IP>:8090`，双手柄驱动双臂、扳机控夹爪、A 键开关录制，落成 LeRobot v2.1 数据集。
- **数据回放增强**：加载 `worlds/g1_cloth_augment.yaml`，读取已有 episode 的 action 在随机化（外观 / 灯光 / 机位 / 后处理）场景重放并追加回数据集。**会写回源数据集**，先复制一份再跑。
- **π0.5 策略闭环推理**：加载 `worlds/g1_cloth_policy_eval.yaml`（或 `python -m simovia run worlds/g1_cloth_policy_eval.yaml`）。控制器把 cam_high / cam_left_wrist 两路 256×256 图像与 16 维关节状态送进子进程里的 π0.5（pi05 + LoRA）推理 worker，按 10 帧动作 chunk、31.25 Hz 回放；抓起毛毯、松手落入盒内并保持 0.5s 判成功，超过 30s 仿真时间判失败。前置条件：`~/openpi/.venv` 可用、checkpoint 存在，路径与 `device` / `num_steps` 在 YAML 的 `robots[].args` 里配置。展示站上的「遥操作 vs π0.5」对比视频就是本世界与 `g1_cloth.mp4` 同机位逐帧对齐拼出来的。

## 目录

- [cloth_folding](cloth_folding/README.md)：G1 双臂柔性毛毯对折（脚本演示 + GUI 预览），并说明遥操作与回放世界。
- [fruit_sorting](fruit_sorting/README.md)：六个水果、三个框、RGB/HSV 检测、物理抓取及实时识别预览。
- [hand_eye](hand_eye/README.md)：独立的眼在手上标定，包含两种机械臂示例及其他机械臂的配置说明。
- [maintenance_relay](maintenance_relay/README.md)：Go2 × G1 × UR5e 三机联合维修。
- [物理参数与依据](fruit_sorting/物理参数.md)：水果质量、摩擦、夹爪力限制和机器人载荷。

标定板及板角点数据统一放在 hand_eye 中。运行图片、日志、视频位于各模块 runs/，依赖缓存位于 .deps/，均不纳入 Git；hand_eye/results/ 内的已验证标定文件随示例提供。
