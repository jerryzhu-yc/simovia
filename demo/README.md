# Simovia Demo

## 机械臂视觉水果分拣

[完整说明](fruit_sorting/README.md)

Franka Panda + OnRobot 2FG7 + 腕部 RGB 相机，使用 HSV 阈值识别香蕉、苹果、葡萄，自动手眼标定并分拣到三个框。

在项目根目录运行：

~~~bash
python -m pip install -r demo/fruit_sorting/requirements.txt
python -m demo.fruit_sorting
~~~

也可通过平台菜单“文件 → 水果视觉分拣 Demo…”启动。
