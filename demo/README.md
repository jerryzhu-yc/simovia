# Simovia 视觉 demo

当前示例分为两个独立模块：水果分拣直接加载标定结果；手眼标定单独采集并保存结果。

| 场景 | 软件内入口 | 场景文件 |
|---|---|---|
| 香蕉、苹果、葡萄分拣 | 文件 → 水果视觉分拣 Demo… | [fruit_sorting/scene.yaml](fruit_sorting/scene.yaml) |
| Panda 腕部相机标定 | 文件 → 手眼标定 Demo → Panda 手眼标定… | [hand_eye/scene_panda.yaml](hand_eye/scene_panda.yaml) |
| UR5e 腕部相机标定 | 文件 → 手眼标定 Demo → UR5e 手眼标定… | [hand_eye/scene_ur5e.yaml](hand_eye/scene_ur5e.yaml) |

## 快速运行

先按[主 README 安装说明](../README.md#安装)创建并激活 64 位 Python 3.12 虚拟环境。在仓库根目录统一安装平台和视觉依赖，然后启动软件：

~~~bash
python -m pip install -r requirements.txt -r demo/fruit_sorting/requirements.txt
python -m pip check
python -m simovia doctor
python -m simovia gui
~~~

两个 demo 的 OpenCV 依赖相同，安装一次即可；NumPy 统一使用核心依赖中的 2.2.x。硬件建议、平台限制与画质说明以[主 README](../README.md#配置建议)为准，目前没有经低配机器认证的最低配置。两个 demo 即使无 GUI 运行也需要可用的离屏渲染；腕部相机为 640×480，降低 GUI 主视口画质不改变标定数据。

通过上表菜单打开场景，代码页选择 arm，点击“保存并运行”或 F5。也可用“打开世界”加载对应 YAML。运行时自动展开 arm/wrist_rgb 相机预览，点击“独立窗口”可放大，查看 RGB 识别图时关闭“深度图”。

默认携带实际标定并通过验证的 Panda、UR5e 数据，首次分拣可直接运行。水果场景没有标定板；重新标定请运行对应标定场景，通过后再加载水果场景。更换机械臂、夹爪、相机安装或内参时应重新标定，配置不匹配会拒绝使用旧数据。

## 目录

- [fruit_sorting](fruit_sorting/README.md)：六个水果、三个框、RGB/HSV 检测、物理抓取及实时识别预览。
- [hand_eye](hand_eye/README.md)：独立的眼在手上标定，包含两种机械臂示例及其他机械臂的配置说明。
- [物理参数与依据](fruit_sorting/物理参数.md)：水果质量、摩擦、夹爪力限制和机器人载荷。

标定板及板角点数据统一放在 hand_eye 中。运行图片、日志、视频位于各模块 runs/，依赖缓存位于 .deps/，均不纳入 Git；hand_eye/results/ 内的已验证标定文件随示例提供。
