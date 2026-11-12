# Go2 × G1 × UR5e 联合维修 Demo

一个约 55 秒仿真时长的物理协作任务：UR5e 抓取蓝色维修模块并装载到 Go2 的运输架；Go2 沿直线通道配送、停稳；G1 在维修工位取件、放入导向槽，并实际接触复位按钮。模块稳定落座、按钮触发、机械臂退回待机后，设备指示灯变绿并停止仿真。

## 在 Simovia 中运行

启动平台后选择 **文件 → Go2 × G1 × UR5e 联合维修 Demo…**，点击“开始联合任务”。演示在独立任务控制台中运行，保留主窗口当前编辑的世界。

控制台支持暂停/继续、单步、重置、1×/2×/4×、总览/装载/跟随 Go2/维修/俯视，以及拖动旋转、滚轮缩放。重置会重建整个物理世界和控制器；每次运行保存在独立 episode 目录。

也可在仓库根目录运行：

~~~bash
python -m demo.maintenance_relay
python -m demo.maintenance_relay --autostart
python -m demo.maintenance_relay --headless --output /tmp/relay-run
python -m demo.maintenance_relay --headless --station-delay 10 --output /tmp/relay-delay
~~~

桌面版使用平台现有 GUI 依赖。纯物理无头验证只需要 requirements-core.txt，不需要 GPU、渲染环境、视觉模型或 OpenCV。

可选录像：

~~~bash
python -m pip install -r demo/maintenance_relay/requirements-video.txt
python -m demo.maintenance_relay --headless --video --output /tmp/relay-video
~~~

录像需要可用的 MuJoCo 离屏渲染，Linux 默认 EGL。输出 relay.mp4、final.png 和 result.json。普通无头运行只输出 result.json；失败返回非零退出码。

## 协调与物理

- 三台机器人来自项目内置库。Go2 增加轻量运输架，G1 双腕安装可驱动夹爪，左夹爪指尖使用软触头按按钮，UR5e 使用同型号夹爪。
- 核心是统一时钟下的集中调度与三个机器人的控制器。它直接使用 Simovia 世界加载器、模型装配器与 MuJoCo，未修改平台的多进程内核。
- 位置来源是仿真状态。UR5e/G1 使用数值 IK 和关节伺服；Go2 使用低速步态与航向反馈；G1 使用踝部平衡反馈。
- 初始化后只改变执行器控制量。模块靠实际接触、摩擦与托盘挡边运输，不使用运行时瞬移、载荷焊接或外加基座稳定力。
- UR5e 退出后 Go2 才能发车；Go2 停稳后 G1 才能取件。装载位置、双指抓取接触、运输掉件、末端跟踪、机器人失稳及机器人间碰撞均有检查。
- 最终同时检查模块位置、姿态、速度和实际按钮接触。完成后不会再次进入抓取循环；出现异常时停在失败状态并保存原因。
- 首版场景是单件蓝色模块、固定工位和直线路径。红色备用件用于区分备料位置。G1 驻站作业，未包含双足行走、任意地图导航、学习型视觉或真实机器人控制。

scene.yaml 是供检查/编辑的几何场景。完整任务应通过上述菜单或模块入口启动；直接把 YAML 放进通用运行器不会自动执行集中调度。

## 文件

- scene.py：基于内置模型生成的场景与物理附件；需要重新生成时运行 python -m demo.maintenance_relay.scene。
- scene.yaml / assets/：随仓库携带的场景；普通运行不修改这些文件。
- control.py：IK、Go2 步态和 G1 平衡控制。
- core.py：交接状态、物理动作、异常检测与结果报告。
- gui.py / view.py：任务控制台与相机。
- runs/：默认运行报告，已忽略 Git。

验证入口：

~~~bash
python -m pytest -q tests/test_maintenance_relay.py
~~~

测试覆盖默认场景连续 10 次完成、不同工位等待时间、完成后无追加动作、配送掉件中止，以及禁用按钮碰撞后必须失败。测试中的故障注入仅用于验证中止条件。

装配与场景更新：Go2 托盘采用连续底座、立柱和横梁；G1 两侧增加连接法兰，左夹爪待机时收臂避开面板。料架、周转箱、托盘纸箱、地面标线和工位外壳为无碰撞视觉布置；任务中的托盘、物料、夹爪和按钮仍使用真实接触。

工业工装更新：参考 Bosch Rexroth TS 1 的工件载板与定位工装思路（https://www.boschrexroth.com/en/gb/c/transfer-system-ts-1/），本演示采用自定义低位四点承载架、浅限位料巢和四脚型材接收台，不是原厂产品复刻。运输载板基准高度相对原版降低 90 mm；G1 前倾配合浅屈膝取料，起身后放入台面内定位治具。承载框架独立刚体计入质量；接收台台面、腿、横梁与定位挡块参与碰撞。货架、光电传感器外观、台面孔阵等只作视觉展示；到位判定使用仿真位姿与物理接触，不宣称真实光电传感或工业认证。
