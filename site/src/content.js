export const CODE = `git clone https://github.com/jerryzhu-yc/simovia.git
cd simovia
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python -m simovia doctor                        # 环境自检：缺什么会提示怎么装
python -m simovia run worlds/pick_cube.yaml     # 无窗口跑一个内置抓取任务
python -m simovia gui                           # 打开工作台，选世界按 F5 运行`;

export const FEATURES = [
  {
    icon: "code",
    title: "World as Code",
    en: "YAML → MjSpec → MuJoCo",
    body: "一份 YAML 描述物理、机器人、任务与奖励。字段语义与 MJCF 对齐（size 是半尺寸、pos 是中心），有 MuJoCo 经验就能直接上手；校验错误带字段路径、期望值与实际值。",
  },
  {
    icon: "process",
    title: "进程级控制器隔离",
    en: "ZMQ fence sync",
    body: "每个机器人对应一个独立 OS 进程，通过 ZMQ REQ / ROUTER 与内核栅栏同步。控制器崩溃或超时只会让该机器人掉线，不会死锁整个仿真内核。",
  },
  {
    icon: "agent",
    title: "MCP 原生 Agent 接入",
    en: "17 tools · screenshot",
    body: "零依赖自实现的 stdio JSON-RPC MCP 服务器，任何支持 MCP 的 Agent 都能读世界、改 YAML、重跑、截图观察结果，形成感知—决策闭环。",
  },
  {
    icon: "gui",
    title: "GUI 全流程工作台",
    en: "Build · Run · Diagnose",
    body: "场景 / 机器人 / 装备三个构建器，任务状态与失败原因解释、环境诊断、导入模型自动生成可运行控制器、一键导出可分享的项目包。",
  },
  {
    icon: "team",
    title: "多机器人协同",
    en: "3 robots · one clock",
    body: "三台机器人各自独立控制、统一时钟调度：UR5e 装载、Go2 配送、G1 工位作业。交接靠真实接触与摩擦，不做瞬移或载荷焊死。",
  },
  {
    icon: "vr",
    title: "VR 遥操作采集",
    en: "PICO → LeRobot v3.0",
    body: "PICO 头显自带浏览器直连控制器，手柄位移经操作空间控制驱动机械臂，A 键录制多路相机 + 关节状态与动作，退出时自动升级为 LeRobot v3.0 数据集。",
  },
  {
    icon: "eye",
    title: "视觉分拣与手眼标定",
    en: "RGB → HSV → 2.5D",
    body: "独立标定场景生成结果文件，水果分拣场景直接复用；换机械臂、夹爪或相机后旧标定会被拒绝，避免静默错抓。",
  },
  {
    icon: "nav",
    title: "雷达建图与自主导航",
    en: "LiDAR → 栅格 → 骨架 → A*",
    body: "2D 激光雷达实时 log-odds 占据栅格建图；对自由区做障碍膨胀与 Zhang-Suen 细化得到维诺骨架（广义维诺图），在骨架上 A* 搜路、纯跟踪导航到目标。建图、骨架、路径全部经 MCP 叠加到 GUI 建图面板实时呈现。",
  },
  {
    icon: "vla",
    title: "VLA 闭环部署",
    en: "SmolVLA @ 31.25 Hz",
    body: "SmolVLA 推理控制器按 chunk 输出关节目标，两路相机输入、语言指令条件，31.25Hz 闭环；附同一策略的 sim2sim 变体世界，用来看跨域表现。",
  },
];

export const STEPS = [
  {
    n: "01",
    title: "遥操作采集",
    body: "PICO 手柄驱动机械臂，A 键开录，每集自动复位并随机摆放物体",
  },
  {
    n: "02",
    title: "LeRobot v3.0",
    body: "多路相机 + 关节状态/动作增量落盘，退出时自动升级数据集格式",
  },
  {
    n: "03",
    title: "SmolVLA 微调",
    body: "预训练策略 + rename_map 对齐相机槽位，单卡即可完成微调",
  },
  {
    n: "04",
    title: "闭环评估",
    body: "同一世界回灌策略，或换物理与渲染条件的 sim2sim 世界做对照",
  },
];

export const MCP_TOOLS = [
  ["load_world", "载入 YAML 世界"],
  ["read_world", "读回 YAML 全文"],
  ["save_world", "写回 YAML"],
  ["run", "按 tick 推进仿真"],
  ["stop", "停止运行"],
  ["reset", "复位世界"],
  ["get_observation", "读关节与物体位姿"],
  ["get_robot_manifest", "查机器人能力清单"],
  ["camera_frame", "抓取相机 RGB"],
  ["camera_stream", "实时画面推流"],
  ["lidar_map", "读雷达建图与叠加层"],
  ["screenshot", "整场景截图"],
  ["save_state", "保存仿真状态"],
  ["load_state", "恢复仿真状态"],
  ["zoo_create", "生成机器人 / 夹爪"],
  ["teleop_start", "启动遥操作"],
  ["teleop_cmd", "下发末端速度"],
];

export const AGENT_LOG = [
  { role: "user", text: "在 so101_policy_eval 里加一个演示机位，跑一遍截图看看取景。" },
  { role: "tool", name: "read_world", text: "返回 scene.cameras：只有训练用的 top 机位" },
  { role: "tool", name: "save_world", text: "插入 showcase 机位（pos / quat / fovy），top 机位不动" },
  { role: "tool", name: "run", text: "ticks=200 · success=false · 无崩溃" },
  { role: "tool", name: "screenshot", text: "返回 640×480 渲染帧，取景过远" },
  { role: "tool", name: "save_world", text: "把机位拉近到 0.62 / -0.42 / 0.40" },
  { role: "tool", name: "run", text: "ticks=200 · 取景通过，策略输入机位未改动" },
];

export const AGENT_DIFF = ` scene:
   cameras:
     - name: top              # 训练输入，位置角度不能改
       pos: [0.419, 0.0, 0.430]
       euler: [0, 40, 90]
       fovy: 60
+    - name: showcase         # 演示机位：不影响策略输入
+      pos: [0.62, -0.42, 0.40]
+      quat: [0.753146, 0.539035, 0.219481, 0.306661]
+      fovy: 45`;

export const ONBOARD = [
  {
    img: "./assets/gui/robot_builder.png",
    tag: "造机器人",
    title: "填 DH 参数，旋转矩阵自动算",
    body: "不写 MJCF：按行填 θ₀ / d / a / α 与关节类型，右侧同步给出相对父系 R_rel 与世界系 R_world 两个旋转矩阵，3D 预览跟着变。初学者按 DH 表逐行填就能拼出串联臂。",
    meta: "标准 DH · T = Rz(θ)·Tz(d)·Tx(a)·Rx(α)",
  },
  {
    img: "./assets/gui/end_effector.png",
    video: "./assets/demos/end_effector_switch.mp4",
    tag: "换夹爪",
    title: "选一个末端执行器，一键换装",
    body: "录的是真实操作：装备页把末端执行器从 OnRobot 2FG7 换成 Robotiq 2F-85，点“应用末端执行器”后世界重建，3D 里的夹爪和挂点一起换掉；换臂或换夹爪后旧的手眼标定会被判失效。",
    meta: "2FG7 → Robotiq 2F-85 · 真实录屏",
  },
];

export const DEMOS = [
  {
    kind: "video",
    src: "./assets/demos/agent_scene_gen.mp4",
    tag: "AI 生成",
    title: "一句话需求，Agent 生成世界并跑通抓取",
    body: "右侧是 Agent 会话实录：本地 Ollama qwen2.5-coder 微调模型接到一句自然语言需求，直接产出平台原生世界 YAML——工作台、三色方块、Panda+2FG7、空中成功区一应俱全；生成的方块半尺寸写错时自动回喂错误让模型修正，parse+编译校验通过后 save_world / load_world / run / screenshot 逐个工具调用，返回值全部真实。左侧 MuJoCo 视口就是被生成出来的世界本体：Panda 从高位接近、直线下探、闭合夹起红色方块，抬进空中成功区并保持。",
    meta: "Ollama qwen2.5-coder:7b · 生成-校验-回喂闭环",
    featured: true,
  },
  {
    kind: "video",
    src: "./assets/demos/lidar_nav3d.mp4",
    tag: "3D 雷达 · 全屋清扫",
    title: "3D 点云建图 × 2D 覆盖导航 × 实车行驶（三视图同步）",
    body: "车顶 16 线 960 束/帧的 3D 雷达边巡逻边建图：右上是按高度彩虹着色的 RViz 风格立体点云（墙身、桌面底、桌下地面分层可见），右下是俯视占据栅格——床裙、椅腿横撑这类悬空低矮障碍只有 3D 雷达低仰角通道能扫进图。巡逻收图后把整间房切成 0.34m 间距的蛇形条带，A* 连接成全屋覆盖路线（青色覆盖线、黄色路线），扫地机把所有场地拖一遍再回充到目标点。左侧 MuJoCo 3D 实车跟随镜头取同一 control tick 逐帧合成，约 5 倍速播放。",
    meta: "三视图同步 · 全屋覆盖 · ~5x 加速",
    featured: true,
  },
  {
    kind: "video",
    src: "./assets/demos/lidar_nav.mp4",
    tag: "建图导航",
    title: "3D 实车行驶 × 2D 雷达建图导航（同步）",
    body: "左半是 MuJoCo 3D 实时行驶（相机跟随小车），右半是雷达建图面板，两边取同一 control tick 逐帧合成。车顶 60 束 2D 雷达边巡逻边建图：log-odds 占据栅格逐帧填充，白为自由、灰为未知、黑为障碍。回到起点后对自由区做障碍膨胀与 Zhang-Suen 细化，得到障碍通道正中的 1px 维诺骨架，在骨架上 A* 搜路，再由纯跟踪控制器沿线行驶到目标。蓝为骨架、黄为 A* 路径、棕为巡逻框、橙/绿为起终点。",
    meta: "3D 实跑 + 建图同步 · 31s · 1280×960",
  },
  {
    kind: "video",
    src: "./assets/demos/g1_cloth.mp4",
    tag: "遥操作 · 数据回放",
    title: "宇树 G1 收拾毛毯：遥操作采集 + LeRobot 回放",
    body: "PICO 手柄遥操作 G1 的 2F85 夹爪，把桌上的毛毯抓起放进盒子；第三人称与腕部两路相机同步采集，逐帧记录关节状态与动作，形成 LeRobot v2.1 数据集（148 集 / 51,728 帧 / 31.25 Hz）。视频按采集到的动作序列逐帧回放，左为 cam_high、右为 cam_left_wrist，底部进度条对应数据集内的时间轴。",
    meta: "G1 + 2F85 · 148 集 · 31.25 Hz · 双相机",
  },
  {
    kind: "video",
    src: "./assets/demos/g1_cloth_policy_compare.mp4",
    tag: "策略闭环 · 对比",
    title: "同一任务：遥操作示教 vs π0.5 自主推理（逐帧对齐）",
    body: "上排是 PICO 遥操作采集的原始 episode，下排是 π0.5（pi05 + LoRA，30k step）在同一世界、同一任务、同一机位下的自主闭环：两路相机 256×256 输入、语言指令条件，每 0.32s 输出一个 10 帧动作 chunk，31.25 Hz 回放，抓起毛毯松手落入盒内由控制器判定成功。两排压在同一条时间轴上：人类示教 10.9s，策略 7.6s 完成（之后定格）。",
    meta: "π0.5 + LoRA 30k · 31.25 Hz · 10.9s vs 7.6s",
  },
  {
    kind: "video",
    src: "./assets/demos/go2_sim2sim.mp4",
    tag: "sim2sim · 跨引擎",
    title: "同一个 Go2，两套仿真软件跑同一份步态",
    body: "左半是直接用 MuJoCo 原生接口步进的 Go2，右半是同一个 go2.xml、同一份对角小跑控制律跑在 simovia 里：地面、背景、界面都来自各自软件，一眼就能看出不是一个仿真器。目标域只改物理——控制周期 62.5→50 Hz、积分器 implicitfast→implicit、impratio 100→20、地面摩擦 0.8→0.6、重力 -9.81→-9.6；两端共用同一相机、同一起始位姿、同一条时间轴，4 秒里步频与位移的差别只来自仿真条件本身（源端 2.80 m，目标端 3.35 m）。",
    meta: "Go2 对角小跑 · 62.5Hz vs 50Hz · 4s 对齐",
    featured: true,
    row: true,
  },
  {
    kind: "video",
    src: "./assets/demos/maintenance_relay.mp4",
    tag: "多机器人协同",
    title: "Go2 × G1 × UR5e 联合维修",
    body: "UR5e 抓取维修模块装到 Go2 运输架，Go2 配送到维修工位，G1 取件放入导向槽并按下复位按钮。三机各自独立控制器、统一时钟调度，全程靠接触与摩擦完成交接。",
    meta: "三台机器人 · 65s · 1100×720",
    featured: true,
  },
  {
    kind: "video",
    src: "./assets/demos/fruit_sorting.mp4",
    tag: "视觉分拣",
    title: "腕部 RGB 识别，分类入框",
    body: "画面右侧是腕部 RGB 相机的实时视觉：HSV 分割 + 连通域给出识别框，2.5D 反投影转成世界坐标；香蕉、苹果、葡萄分别放入三个框，落框结果独立复核，6/6 全部通过。",
    meta: "手眼标定 · 6 个物体 · 6/6 通过",
  },
  {
    kind: "video",
    src: "./assets/demos/pick_cube.mp4",
    tag: "抓取任务",
    title: "相位机 + 6D 位姿 IK",
    body: "Franka Panda 配 2FG7 夹爪，从高位接近、五段直线下探、闭合夹起，再抬升到空中成功区并保持。",
    meta: "Panda + 2FG7 · 1280×960 重录",
  },
  {
    kind: "video",
    src: "./assets/demos/policy_eval.mp4",
    tag: "策略闭环",
    title: "SmolVLA 驱动 SO-101（成功片段）",
    body: "这段是策略真正跑通的完整片段：两路相机 + 语言指令输入，按 chunk 输出关节目标，31.25Hz 闭环；红、蓝方块依次入盒，夹爪释放后方块稳定保持，控制器判定 success=true。",
    meta: "SmolVLA · 31.25 Hz · 成功片段",
  },
  {
    kind: "link",
    href: "https://github.com/jerryzhu-yc/simovia-playground",
    tag: "Playground",
    title: "14 个 demo 分支，逐个可复现",
    body: "每个 demo 独立分支 + README 复现流程：环境、命令、世界文件齐备，clone 即跑。",
    meta: "simovia-playground",
    featured: true,
  },
];
