# Simovia

**AI 第一用户、人类第二用户的轻量机器人仿真平台。**

Simovia 把"世界"做成代码（World as Code：一份 YAML 即一个完整仿真世界），把"控制器"做成独立进程（Webots 风格的 `robot.step()` 循环），并通过标准 MCP（Model Context Protocol）把整个仿真内核暴露给 AI Agent——Agent 可以读世界、改世界、重跑、观察结果、再调整，形成完整的感知-决策闭环。

```
        ┌──────────────────────────────────────────────┐
        │                AI Agent (MCP)                 │
        │   read_world → 改 YAML → load_world → run     │
        └───────────────┬──────────────────────────────┘
                        │ stdio JSON-RPC (MCP)
        ┌───────────────▼──────────────────────────────┐
        │           MCP 工具层（12 个工具）             │
        │  load_world / run / get_observation / ...    │
        └───────────────┬──────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────┐
        │        仿真内核（ZMQ ROUTER + 栅栏同步）       │
        │   World as Code: YAML → MjSpec → MuJoCo       │
        └───┬──────────────────────────────┬───────────┘
            │ spawn + REQ/REP 锁步          │
   ┌────────▼────────┐            ┌─────────▼────────┐
   │  控制器进程 #1   │            │   控制器进程 #N   │
   │ while robot.step│            │  （独立 OS 进程） │
   └─────────────────┘            └──────────────────┘
```

## 特性

- **World as Code**：一份 YAML 描述物理、机器人、任务、奖励，字段语义与 MJCF 对齐（`size`=半尺寸、`pos`=中心），有 MuJoCo 经验即可直接上手；错误信息带字段路径、期望值、实际值，人和 AI 都能直接看懂
- **进程级控制器隔离**：每个机器人的控制器是独立 OS 进程，通过 ZMQ REQ/ROUTER 与内核栅栏同步；控制器崩溃、超时都不会死锁内核
- **MCP 原生接入**：零依赖自实现的 stdio JSON-RPC MCP 服务器，任何支持 MCP 的 AI Agent（Claude、Trae 等）都能直接驱动仿真，包括让 Agent 用 `screenshot` 截取仿真画面"亲眼"看世界
- **传感器系统**：YAML 里给机器人声明 encoder / IMU / 触觉 / 关节力矩 / 相机五类传感器，编译进 MuJoCo 模型、读数进观测流，GUI 面板实时显示；传感器是只读派生量，不改变物理轨迹（bit 级守护测试保证）
- **机器人模型库（资产自带）**：MuJoCo Menagerie 官方模型一键接入——预配置 34 个著名机器人（四足 Go1/Go2/A1/ANYmal C/Spot/Barkour、人形 H1/G1/OP3/TALOS/T1/Berkeley、机械臂 Panda/FR3/iiwa/Gen3/UR5e/UR10e/xArm7/Lite6/SO-ARM100 等、无人机 X2/Crazyflie、移动操作 TIAGo/Stretch 3/TidyBot），**模型资产随仓库自带（`models/zoo/`），clone 完即离线可用、无需下载**；GUI 点"模型库"选模型即生成世界 + 控制器模板并自动打开编辑器，改完控制律点 Run 即跑；AI Agent 走同一条 `zoo_create` 工具
- **夹爪装配工作台**：内置 4 款末端执行器（OnRobot 2FG7、Robotiq 2F-85、UMI Gripper 三款平行夹爪 + LEAP Hand 16 自由度灵巧手），GUI"装备"页点选即可给 UR5e / UR10e / xArm7 / Panda 四臂挂爪/拆爪，并可视化挂点、关节、body 清单；灵巧手用 `actuators` + `poses` 描述多关节开合姿态。同一页表单式添加五类传感器（encoder/IMU/触觉/关节力矩/相机），全部改 YAML 自动重载，不手写配置。夹爪安装位姿（mount）按各臂腕部几何逐臂标定，夹爪后法兰与机器人法兰同轴贴合、指尖沿法兰法向伸出；自带"码垛分区"场景模板（UR5e+2FG7，A 区取料→B 区码放）
- **从零造机器人**：连杆树 YAML（parent / pos / rpy / 几何 / 关节）直接编译成 MJCF 挂进世界；GUI 文件 → 造机器人（Ctrl+B）提供连杆树编辑 + 实时 3D 预览 + 每个连杆的 R_rel / R_world 旋转矩阵与 RGB 坐标轴，按 `R = Rz·Ry·Rx` 直观展示角度怎么变成矩阵，适合机器人学入门
- **RL 训练接口**：一个 YAML 世界 = 一个确定性 gymnasium.Env——动作空间 = 执行器拼接、观测 = qpos/qvel/传感器、reset 逐位一致（同 YAML 跑两遍轨迹逐 bit 相同，eval 可复现），单进程直驱不起 ZMQ，训练吞吐优先
- **明亮场景观感**：每个世界自动带白天渐变天空 + 网格地面（2m 主格 / 0.5m 细格，贴图按世界尺寸走，地面多大格子都一样大，`floor.texture: solid` 可退回纯色），场景灯统一归一化成柔和天光、只有一盏投影，机器人模型自带的刺眼聚光灯不会再打斑；机器人网格统一开平滑法线 + 保留高光 + 离屏抗锯齿（默认 4× MSAA，高画质 8×），低多边形模型看起来也是干净的塑料壳而不是一片糊；打开世界按实体包围盒自动取景（机械臂/人形/四足整台入画），空场景给大远景铺满网格
- **Supervisor 能力**：save/load 内存快照（物理状态逐位恢复）、teleport、reset——快照包含 `mjSTATE_FULLPHYSICS | mjSTATE_WARMSTART`，保证轨迹级可复现
- **场景生成器**：6 个确定性模板（tabletop / push / maze / race / pallet / empty），同一命令 + 同一参数 = 逐字节相同的输出
- **AI 场景生成**：自然语言需求 → LLM 直接生成世界 YAML → 本地 schema+编译双重校验，失败自动回喂重试（Ctrl+G）；机器人经 `zoo://key` 引用模型库，无需知道任何文件路径
- **URDF / SolidWorks 导入**：`robots[].model` 直接填 `.urdf` 路径——mesh 路径重写、`package://` 前缀剥除、逐关节自动补 motor 全部自动；SolidWorks 装 sw2urdf 插件导出即可用
- **零重量依赖**：核心只依赖 mujoco / numpy / pyyaml / pyzmq，MCP 层不依赖任何 SDK

## 安装

统一使用 **64 位 Python 3.12** 作为新环境的推荐安装基线。其他 Python 版本是否可用取决于全部依赖的 wheel 和兼容性；当前没有覆盖所有 Python/操作系统组合的验收矩阵。MuJoCo 4.0 已取消 Python 3.10 支持，**最低要求 Python 3.11**。

Linux 安装命令如下；Windows 见 [Windows 部署](#windows-部署)。macOS 的图形后端尚未完成专项验收，见 [平台与渲染要求](#平台与渲染要求)。

```bash
git clone https://github.com/jerryzhu-yc/simovia.git
cd simovia
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m simovia doctor
python -m simovia gui
```

> 仓库自带 34 个机器人模型资产（`models/zoo/`，约 600MB）。依赖安装完成且资产齐全后，模型库和本地 demo 可离线运行；AI 功能按所配置的服务连接。

`requirements.txt` 包含核心、GUI、RL 和 pytest，**不包含视觉 demo 的 OpenCV**。按需安装时，拆分文件是各自能力的依赖清单，需要显式组合：

```bash
# 核心：仿真 / MCP / 控制器
python -m pip install -r requirements-core.txt
# 核心 + 桌面 GUI
python -m pip install -r requirements-core.txt -r requirements-gui.txt
# 核心 + RL 接口
python -m pip install -r requirements-core.txt -r requirements-rl.txt
# 完整开发环境 + 水果分拣 / 手眼标定的视觉依赖
python -m pip install -r requirements.txt -r demo/fruit_sorting/requirements.txt
# 可选：模型资产缺失时的 menagerie 下载回退
python -m pip install -r requirements-models.txt
```

两个视觉 demo 使用相同的 OpenCV 依赖，装一次即可。2026-11-06 已通过 Python 3.12 的 Linux x86_64 / Windows x64 wheel 依赖解析检查；这不等于两种平台的完整运行验收。本机渲染与 HSV 冒烟检查使用 MuJoCo 4.0.0、NumPy 2.2.6、OpenCV 4.12.0。兼容版本范围可能解析到更新版本，仍需在实际环境验证。新增依赖后执行 `python -m pip check`；全量回归按[测试](#测试)章节运行。

**从 MuJoCo 3.x 升级到 4.0**：4.0 把 Python 发行包拆成 `mujoco-core`（物理 + 绑定）和 `mujoco-studio`（Studio / Filament 渲染），`mujoco` 变成同时装两者的元包。直接用 `pip install -U` 会让 pip 去删除已迁移到 `mujoco-core` 的文件，因此必须先卸再装：

```bash
python -m pip uninstall -y mujoco
python -m pip install -r requirements-core.txt
```

4.0 还取消了 Python 3.10 支持（需 3.11+），并改变了执行器 `forcerange` 饱和时对 `implicit`/`implicitfast` 隐式导数的贡献——使用饱和力限的执行器时，积分结果会与 3.x 不同，必要时补关节 armature 或减小 `timestep`。

依赖使用兼容版本范围，`~=` 不等于固定到同一 minor：例如 `~=4.0` 表示 `>=4.0,<5`，`~=2.2.6` 表示 `>=2.2.6,<2.3`。核心 NumPy 统一为 2.2.x，以满足视觉依赖 `opencv-python-headless==4.12.0.88` 的 `numpy>=2,<2.3` 要求（[OpenCV 包元数据](https://pypi.org/pypi/opencv-python-headless/4.12.0.88/json)）。请在独立虚拟环境中安装，避免混装多个提供 `cv2` 的 OpenCV 包。

### 配置建议

**目前没有经低配机器实测认证的最低配置。** 下表是单机械臂、小场景和单路 640×480 相机的规划建议，不保证特定帧率或实时仿真速度；多机器人、复杂接触、高面数网格及多路相机会增加负担。

| 项目 | 入门试运行基线（估计） | 日常使用建议（估计） |
|---|---|---|
| CPU | 近年的 4 核 64 位处理器 | 单核性能较好的 6 核或更多核心 |
| 内存 | 8 GB | 16 GB 或更多 |
| 图形 | 具备完整 OpenGL 驱动、可创建离屏上下文的现代核显 | 较强核显或 4 GB 显存级独显 |
| 可用磁盘 | 预留 5 GB | 预留 10 GB；录屏和采样图另计 |
| 主视口 | 较小窗口、低画质 | 1080p 窗口、均衡画质 |

不强制要求 NVIDIA 显卡或 CUDA。仅做物理计算且不创建相机/截图时可不使用图形渲染；水果识别和手眼标定即使无 GUI，仍需渲染 RGB 图像。是否可用应以目标机器的 `doctor`、场景加载和实际运行结果判断。

### 平台与渲染要求

- **Linux**：GUI/MCP 默认使用 EGL；需要可用的显卡驱动和 EGL/OpenGL 运行库。无显示器不等于不需要图形上下文。OSMesa 是可选的软件渲染路径，需要系统库，性能需单独评估。
- **Windows**：GUI/MCP 默认使用 WGL，需显卡厂商提供的 OpenGL 驱动。部署步骤以 Windows 11、Python 3.12 x64 为建议基线；当前没有完整 Windows 硬件兼容性验收矩阵。
- **macOS**：当前 GUI/MCP 的非 Windows 分支默认设置 EGL，不能据此宣称 macOS 开箱即用；仍需适配并验证原生图形上下文、Qt 和启动流程。

当前已验证的 MuJoCo 4.0.0 环境使用经典 OpenGL 渲染路径，通过离屏图像显示到 PySide6 GUI。画面效果同时取决于模型网格、材质贴图、灯光、阴影和渲染器；Python 负责组织应用，物理与渲染的主要计算由原生库执行。

| GUI 主视口画质 | MSAA | 阴影 | 渲染宽高倍率 |
|---|---|---|---|
| 高 | 8× | 开 | 100% |
| 均衡（默认） | 4× | 开 | 100% |
| 低 | 关闭 | 关 | 75% |

切换高画质主要改善边缘锯齿，不会自动增加纹理细节或启用 PBR。MuJoCo 官方另提供 Filament/PBR 渲染路径，但 **Simovia 当前尚未接入**，需要开发集成（[MuJoCo 渲染文档](https://mujoco.readthedocs.io/en/latest/programming/rendering.html)）。

低配机器可先降低主视口画质、缩小窗口，并减少额外相机预览。主视口画质与腕部传感器相机分辨率分别配置：两个视觉 demo 使用 640×480 相机，降低主视口画质不影响其标定；改变传感器相机分辨率、内参或挂载后需要重新标定。

### Windows 部署

内核使用本机 TCP 通信，控制器通过 `subprocess.Popen` 启动；GUI/MCP 在 Windows 上默认选择 WGL。请按以下步骤安装，并在目标机器验证图形驱动与场景运行。

**1. 装 64 位 Python 3.12**（python.org 安装包勾选 "Add python.exe to PATH"，或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)）

**2. 克隆 + 装依赖**（PowerShell 或 CMD）：

```powershell
git clone https://github.com/jerryzhu-yc/simovia.git
cd simovia
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

以上命令不需要激活环境。后续命令中的 `python` 请使用 `.\.venv\Scripts\python.exe`，或先激活虚拟环境；CMD 使用 `.venv\Scripts\activate.bat`，PowerShell 使用 `.\.venv\Scripts\Activate.ps1`。

建议优先使用依赖的预编译 wheel；若安装尝试源码编译，先检查 Python 版本、架构及包版本。机器人模型资产随仓库自带。视觉 demo 还需按 [demo 安装说明](demo/README.md#快速运行) 安装 OpenCV。

**3. 验证安装**：

```powershell
python -m simovia doctor        # 环境诊断：依赖版本 / 模型资产 / 离屏渲染 / 样例世界
python -m pytest tests/ -q      # 全绿即环境就绪（含 GUI/传感器/RL 全量回归）
python -m simovia run worlds/pick_cube.yaml     # 跑一个内置任务
python -m simovia gui                            # 桌面工作台
```

**4. 桌面快捷方式（可选，推荐）**：

仓库根目录带两个批处理（UTF-8 中文输出，自动定位 Python，无需手动激活 venv）：

```bat
create_desktop_shortcut.bat    # 双击一次：在桌面生成 Simovia 图标（simovia.ico）
start.bat                      # 双击即启动 GUI（快捷方式即指向它）
```

`start.bat` 按 `SIMOVIA_PYTHON` 环境变量 → 仓库内 `.venv` → PATH 里的 `python` → `py -3` 的顺序定位解释器；GUI 异常退出时窗口保留并提示装依赖，正常关闭时自动收窗。之后桌面双击 Simovia 图标即可进入工作台，不用再开终端。

**Windows 常见问题（FAQ）**：

| 现象 | 处理 |
|---|---|
| PowerShell 激活 venv 报 "禁止运行脚本" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 后重试，或改用 CMD |
| 首次运行弹 Windows 防火墙提示 | 点"允许"——内核 ZMQ 只绑 `127.0.0.1` 本机回环，不对外通信 |
| 远程桌面（RDP）中渲染失败或断开后异常 | 在本机桌面会话验证，并检查远程会话的图形驱动支持；无 GUI 的 `run` 若使用相机仍需可用的离屏渲染上下文 |
| 渲染报 `wgl` 相关错误 | 安装或更新显卡厂商的 OpenGL 驱动，检查远程会话的图形支持，再运行 `doctor`；不要将 Linux 的 `MUJOCO_GL=osmesa` 设置直接套用到 Windows |
| `mujoco-menagerie` 模型下载慢/失败 | 正常不需要联网：34 个模型资产已随仓库自带（`models/zoo/`，约 600MB）。仅当该目录被删时才回退 `mujoco-menagerie` 下载——重新 clone 仓库即可恢复，无需任何网络处理 |
| 中文/空格路径 | 完全支持（pathlib 全程处理），但建议路径不含空格避免个别工具链问题 |
| sw2urdf 导出的 URDF 导入后 mesh 报错 | mesh 路径按 urdf 所在目录解析（`package://` 前缀自动剥除）；编码兼容 UTF-8 与 GBK（中文 Windows 导出常见），确认 mesh 是 STL/OBJ 且 `<collision>` 里有几何 |

**可选依赖**（都是 `pip install`，Windows 同样直接装）：

```powershell
pip install stable-baselines3   # RL 训练示例（examples/train_ppo_go2.py）
```

### Linux 部署

按上面"安装"章节建好环境后，先验证再上路：

```bash
python -m simovia doctor                # 检查依赖、模型资产和离屏渲染
python -m pytest tests/ -q               # 开发验收：数量以当前输出为准
./start.sh run worlds/pick_cube.yaml     # 跑通一个内置任务（输出见"快速开始"）
```

仓库根自带 `start.sh` 启动脚本：自动按 `.venv` → `venv` → PATH 里的 `python3` 顺序定位解释器，所有 CLI 子命令原样透传，无需手动激活 venv：

```bash
./start.sh gui                            # 桌面工作台
./start.sh run worlds/pick_cube.yaml      # 跑一个内置任务（加 --json 出机器可读结果）
./start.sh mcp                            # MCP 服务器（stdio，供 AI Agent 接入）
```

首次使用若提示无执行权限：`chmod +x start.sh`。Linux 的 GUI/MCP 默认 EGL；无显示器运行仍需驱动和 EGL 系统库。若改用 OSMesa，先安装对应系统库，并在启动 Python 前设置 `MUJOCO_GL=osmesa`。软件渲染速度需在实际场景中验证。

### Linux 中文输入（fcitx5 GUI 集成）

**现象**：fcitx5 桌面下 GUI 文本框（尤其 Agent 对话框）打不了中文（拼音候选词不弹出）。

**原因**：pip 装的 PySide6 只自带 compose/ibus 输入法插件；系统 Qt 的 fcitx 插件又因 Qt 私有 ABI 必须与运行时 Qt 严格同版本而无法借用——需要针对 PySide6 捆绑的 Qt 版本单独编译一个 fcitx5 插件。

**方案**：把编译好的插件装到约定目录 `~/.local/share/simovia/qt-plugins/platforminputcontexts/`，GUI 启动时自动挂载（`simovia/gui/__init__.py` 的 `mount_ime_plugin`，在 QApplication 之前生效）。构建全程免 root，一次约 3 分钟：

```bash
# 0) 依赖开发包免 root 解包到 /tmp/fcitxbuild/root
mkdir -p /tmp/fcitxbuild/root && cd /tmp/fcitxbuild
apt-get download extra-cmake-modules libfcitx5utils-dev libfcitx5utils2 \
    libxkbcommon-dev libxkbcommon-x11-dev gettext
for d in ./*.deb; do dpkg -x "$d" root/; done
export PATH=/tmp/fcitxbuild/root/usr/bin:$PATH

# 1) 与 PySide6 捆绑 Qt 完全同版本的开发包（aqt 官方在线安装器）
python -m pip install -q aqtinstall
QT_VER=$(python -c "import PySide6;print(PySide6.__version__)")   # 如 6.12.0
python -m aqt install-qt linux desktop "$QT_VER" linux_gcc_64 -O /tmp/qt612

# 2) 编译 fcitx5-qt 输入法插件（BUILD_ONLY_PLUGIN 只编译这一个插件）
git clone --depth 1 https://github.com/fcitx/fcitx5-qt.git
cd fcitx5-qt
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/tmp/qt612/$QT_VER/gcc_64;/tmp/fcitxbuild/root/usr" \
    -DBUILD_ONLY_PLUGIN=ON -DENABLE_QT4=OFF -DENABLE_QT5=OFF -DENABLE_QT6=ON \
    -DENABLE_QT6_WAYLAND_WORKAROUND=OFF
cmake --build build -j$(nproc)

# 3) 去掉构建机 rpath、装进约定目录（GUI 启动自动挂载）
python -m pip install -q patchelf
patchelf --remove-rpath build/qt6/platforminputcontext/libfcitx5platforminputcontextplugin.so
mkdir -p ~/.local/share/simovia/qt-plugins/platforminputcontexts
cp build/qt6/platforminputcontext/libfcitx5platforminputcontextplugin.so \
   ~/.local/share/simovia/qt-plugins/platforminputcontexts/
```

装好后重启 GUI 即可在 Agent 对话框用 fcitx5 打中文。注意：**升级 PySide6 后需按同版本号重跑第 1~3 步**（Qt 插件 ABI 必须与运行时 Qt 一致，版本不匹配会静默回退到英文输入）。

## 快速开始

### 1. 跑一个内置任务（人类用户）

> 以下命令都在**仓库根目录**执行（包未打包安装，`python -m simovia` 依赖当前目录，换目录会报 `No module named simovia`）；已装好依赖时也可以统一用 `./start.sh <子命令>`。

```bash
python -m simovia run worlds/pick_cube.yaml
```

输出示例（panda+2FG7 抓起桌面 cube、抬入空中成功区的端到端任务）：

```
世界 'pick_cube'：2 物体，1 机器人
  控制器 arm: ../controllers/pick_place.py
运行 297 ticks（墙钟 0.212s，仿真 19.008s）
物理状态: 正常
任务: 成功（reward 0.0074）
  table 最终位置: (+0.325, +0.000, +0.650)
  cube 最终位置: (+0.553, -0.002, +0.841)
```

加 `--json` 得到机器可读输出，加 `--ticks N` 限定最大步数。场景没配控制器时，`run` 会明确提示"机器人保持初始位姿不动"而不是只报"任务未完成"。

### 1.5 录制 episode（数据采集）

```bash
# 只录数值：data.npz + meta.json
python -m simovia run worlds/pick_cube.yaml --record episodes/pick1

# 同时录相机帧：相机需先在世界 YAML 的 robots[].sensors 里声明
python -m simovia run worlds/pick_cube.yaml --record episodes/pick1_img \
  --record-images --record-camera wrist_cam \
  --record-width 640 --record-height 480 --record-image-stride 2
```

- 相机挂在机器人 body 上，随 body 刚性运动（第一视角/腕部视角）；`pos`/`quat` 是 body 局部系的挂载位姿。
- `--record-camera` 支持 YAML 传感器短名（`wrist_cam`）或全名（`arm/wrist_cam`），`inspect` 可查看世界里已声明的相机；名字错误会直接报错并列出可用相机。
- 产物：`data.npz`（观测/动作/root 位姿/物体位姿）+ `meta.json`（契约、统计、`images` 里的相机全名、挂载 body/pos/quat、分辨率、步长、帧数）+ `frames/frame_000000.png`。
- 当前保存 PNG 帧序列，需要视频时可用 `ffmpeg -framerate 30 -i frames/frame_%06d.png -c:v libx264 out.mp4` 合成。

### 1.6 写控制器前先 `inspect`

```bash
python -m simovia inspect worlds/pick_cube.yaml
```

打印这个世界的全部设备清单：每个机器人的执行器名 + `ctrlrange`、关节名 + 类型 + 限位、传感器名 + 维度、相机、夹爪，以及场景物体名（= 控制器里 `obs["objects"]` 的键）和任务成功条件——不用去翻 XML 就知道 `robot.set(...)` 该写什么名字、范围多少。

### 2. 让 AI Agent 驱动仿真（MCP）

```bash
python -m simovia mcp
```

在支持 MCP 的客户端里把它注册为 stdio 服务器，Agent 即可使用 12 个工具完成闭环：

| 工具 | 说明 |
|---|---|
| `load_world` | 加载世界（`yaml_path` 或 `yaml_text` 二选一），整体替换当前世界 |
| `run` | 推进仿真并返回摘要（ticks / success / reward / 物体终位）；`ticks=N` 只推进 N 步 |
| `stop` | run 进行中提前收尾：控制器正常退出、已跑轨迹保留，之后照常 reset / 继续跑 |
| `get_observation` | 当前完整观察量：qpos / qvel / ctrl / body 位置 + 传感器读数 + 任务状态 |
| `get_robot_manifest` | 机器人设备清单：执行器、关节限位、挂点 site、body 名、传感器、相机、当前夹爪（含开合范围）+ 可用夹爪库目录 |
| `screenshot` | 把当前状态渲染成 PNG：缺省自由机位返回 image content，`camera` 指定已声明相机，`path` 写盘 |
| `save_state` / `load_state` | 内存快照，物理状态逐位恢复 |
| `reset` | 世界回到 YAML 初态（重编译 + 控制器进程重启） |
| `read_world` | 回读当前世界的 YAML 原文 |
| `save_world` | 当前世界（含 GUI/Agent 全部修改）保存为 YAML 文件；跨目录保存时相对资源路径自动改绝对，任何位置重开都有效 |
| `zoo_create` | 从模型库（MuJoCo Menagerie）生成可编辑项目：世界 YAML + 控制器模板（含执行器清单与 API 注释）；`key='list'` 只列清单 |

Agent 的标准闭环：`read_world` → 修改 YAML → `load_world(yaml_text)` → `run` → `get_observation` → 观察成败 → 调整 → 重试。

> 语义提示：`run` 是一次性 episode——返回时本轮控制器已收尾退出；继续推进请 `reset` 后重跑，或改世界后重新 `load_world`。

### 3. 桌面 GUI 实时可视化

```bash
python -m simovia gui
```

（GUI 依赖 PySide6 已随 `requirements.txt` 装好，直接启动即可。）

> Windows 用户可免终端：双击仓库根的 `start.bat`，或先跑一次 `create_desktop_shortcut.bat` 生成桌面图标（见 [Windows 部署](#windows-部署)）。

GUI 本身就是一个 MCP 客户端（与 AI Agent 同构）：左键加载世界 YAML，后端经 stdio JSON-RPC 驱动同一个 `simovia.mcp` 服务器——Run 时仿真跑在服务器后台线程，GUI 以 60Hz 拉取观测快照并渲染 MuJoCo 3D 画面 + 关节/物体面板。窗口里可以：

- **模型库**：选 Menagerie 官方模型（资产随仓库自带，离线即用）→ 自动生成世界 + 控制器模板 → 自动加载并打开编辑器——点击-打开-编程闭环
- **装备页**：选中机器人后点选即可装配/移除夹爪（OnRobot 2FG7 / Robotiq 2F-85 / UMI Gripper / LEAP Hand，适配 UR5e/UR10e/xArm7/Panda），并按类型表单添加/删除五类传感器——所有改动写回世界 YAML 并自动重载，无需手写
- **新建场景**（文件 → 新建场景，Ctrl+N）：选参数化模板（含码垛分区 pallet）→ 设参数 → 保存即生成并加载
- **AI 生成场景**（文件 → AI 生成场景，Ctrl+G）：一句话描述需求（"一只 Go2 站在 3x3 场地中央，面前放个红方块"）→ LLM 直接生成世界 YAML → 本地 schema+编译双重校验，失败自动把错误喂回重试 → 通过即加载；机器人引用模型库写 `zoo://key` 自动解析成本地模型路径（资产随仓库自带，离线即用）。需先在 Agent 页配置任意 OpenAI 兼容端点（DeepSeek/GLM/Ollama 均可）
- **导入模型**（文件 → 导入模型，Ctrl+I）：选本地 `.urdf` / `.xml` 机器人模型导入当前场景或新世界——URDF 自动适配（mesh 相对路径按 urdf 目录解析、`package://` 前缀剥除、逐关节自动补 motor）、贴地出生高度自动计算，SolidWorks 工作流见 [导入 URDF（含 SolidWorks）](#导入-urdf含-solidworks)
- **保存世界**（Ctrl+S）：拖动/位姿/装备/添加机器人等全部修改写回 YAML 文件——改完的世界可以留到下次（内存世界会提示另存位置；跨目录保存时相对资源路径自动改绝对）
- **最近打开**（文件菜单）：最近 8 个世界一键重开（跨会话持久化）
- **添加物体 / 删除节点**（场景菜单）：往场景加方块/圆柱/球（类型/尺寸/颜色，放在相机注视点）；场景树右键或 Del 键删除选中的机器人/物体——搭场景不用手写 YAML
- **Load**：选择世界 YAML（解析在本地，坏文件直接报错）
- **Run / Run N**：跑一整个 episode（结束后可再 Run，自动 reset）
- **停止**：run 进行中随时可点（F6）——提前收尾当前 run，控制器正常退出、已跑轨迹保留，之后照常 Reset / 继续 Run
- **实时勾选框**：勾选=按墙钟实时播放动画（1 仿真秒≈1 墙钟秒，MCP `run` 的 `realtime` 参数）；取消=全速推进瞬间出结果（跑批验证）
- **画质下拉框**：高 = 8× MSAA；均衡（默认）= 4× MSAA；低 = 关 MSAA/阴影 + 75% 分辨率渲染（可减轻渲染负担，收益取决于场景和硬件）；选择跨会话记忆
- **原点坐标轴**（视图菜单，默认开）：视口原点叠加 XYZ 轴（X 红 / Y 绿 / Z 蓝），摆场景时一眼看清方位；空场景还会在视口上方给出 Ctrl+M（模型库）/ Ctrl+B（造机器人）/ Ctrl+N（模板场景）三条入口
- **Reset**：世界回初态；渲染视角支持鼠标拖拽旋转、滚轮缩放、**F 键聚焦选中对象**；属性/观测/装备面板带滚动条，右停靠栏可自由拖窄放大视口

> run 期间 `get_observation` 返回约 16ms 节流的运动学快照（含物体位置 + 四元数），不影响物理轨迹。实时模式下内核把每个控制 tick 的物理步按约 1/60 秒切片出快照——pick_cube 这类 64ms 控制周期的世界也能以 60Hz 播放，而不是只在控制 tick 边界跳 15.6 帧/秒。

### 4. 从模板生成场景

```bash
python -m simovia scenegen maze --rows 4 --cols 4 --out worlds/my_maze.yaml
python -m simovia scenegen race --lanes 3 --out worlds/my_race.yaml
python -m simovia scenegen tabletop --objects 2 --out worlds/my_table.yaml
python -m simovia scenegen pallet --cubes 3 --out worlds/pallet.yaml
python -m simovia run worlds/my_maze.yaml
```

| 模板 | 说明 | 主要参数 |
|---|---|---|
| `tabletop` | 桌面抓取/搬运：抬升区 + 若干 cube | `--objects`（1~3） |
| `push` | 地面推物：panda+2FG7 把低摩擦 cube 推进成功区（控制器留给 Agent 写） | - |
| `maze` | 差速车走蛇形迷宫 | `--rows` `--cols`（2~8） |
| `race` | 多车竞速 | `--lanes`（2~3） |
| `pallet` | 码垛分区：UR5e+2FG7 把 A 区 N 个 cube 逐个码到 B 区目标格 | `--cubes`（1~6） |
| `empty` | 空世界（配 arm / diff_car / 无机器人） | `--robot` |

## 世界文件格式

一个完整的世界长这样（`worlds/pick_cube.yaml`，有删节）：

```yaml
name: pick_cube
timestep: 0.002        # 物理步长（秒）
ctrl_period: 32        # 一个控制 tick 的物理步数
gravity: [0, 0, -9.81]
integrator: implicitfast  # 隐式积分：panda 高刚度伺服稳态才准（见下）
impratio: 10

scene:
  floor:
    size: [2.0, 2.0]      # 半尺寸
    texture: grid         # 网格贴图（2m 主格/0.5m 细格）；solid = 纯色
    friction: 1.0         # 滑动摩擦（MuJoCo 语义：接触对取 max，需双方都低才能滑）
  lights:
    - {pos: [0, 0, 3]}
  objects:
    - name: table
      type: box            # box | sphere | cylinder
      size: [0.45, 0.35, 0.01]   # MJCF 半尺寸
      pos: [0.35, 0, 0.67]
      static: true         # 静态物体（无 freejoint）
      color: "#6B7280"
    - name: cube
      type: box
      size: [0.03, 0.03, 0.03]
      pos: [0.55, 0, 0.71]
      mass: 0.1
      friction: 1.0
      color: "#F59E0B"

robots:
  - name: arm
    model: zoo://franka_emika_panda   # 模型库引用；也可写 MJCF 相对路径
    pos: [0, 0, 0.68]                 # root 绝对位姿（模型自带 pos 被吸收）
    controller: ../controllers/pick_place.py   # 控制器脚本（相对本文件）
    gripper: {model: onrobot_2fg7}    # 末端夹爪：库 key（适配四臂，见下）
    sensors:                         # 可选：五类传感器，编译进 MuJoCo
      - {name: j1_enc, type: encoder, joint: joint1}       # 关节位置
      - {name: j1_tau, type: joint_torque, joint: joint1}  # 关节约束力
      - {name: wrist_imu, type: imu, body: link7, pos: [0, 0, 0.02]}   # 6 维
      - {name: pad_touch, type: tactile, body: gripper/left_finger, radius: 0.03}
      - {name: gripper_cam, type: camera, body: link7,
         pos: [0, 0.06, 0.10], quat: [0.7071, 0.7071, 0, 0]}

task:
  success:                  # 成功判定：body 位置落入 zone
    body: cube
    zone: {min: [0.49, -0.06, 0.80], max: [0.61, 0.06, 0.92]}
  reward:                   # 距离/zone/接触 三种奖励项
    - distance: {a: cube, b: arm/link7, weight: -0.01}
    - zone: {body: cube, zone: {min: [...], max: [...]}, weight: 0.01}
  max_ticks: 5000

runtime:
  realtime: false           # 实时模式（true = 按墙钟限速）
  controller_timeout: 100   # 控制器响应超时（ms）
  timeout_policy: last_ctrl # last_ctrl=用上次控制量补位 | pause=原地等待
```

命名引用规则：物体名全局唯一；机器人 body 自动带机器人前缀（`arm/wrist`）。当机器人名与其 body 名不同时，任务里引用必须写全名（如 `car_a/car`）；同名时可用短名。

传感器说明：

- **类型与挂点**：`encoder` / `joint_torque` 挂 `joint`；`imu` / `tactile` / `camera` 挂 `body`（`pos` 为 body 局部系挂点偏移，`quat` 仅 camera 用于朝向）
- **读数进观测流**：obs 里 `sensors` 键为 `{名字: 读数}`，标量（encoder/joint_torque/tactile）是 float，IMU 是 6 维列表 `[陀螺 xyz, 加速度 xyz]`；相机不产生数值读数，只作 `screenshot` 的 viewpoint
- **tactile 的 `radius`**：MuJoCo 的 touch 只统计接触点落在以挂点为球心、`radius` 为半径的球内的接触力——`radius` 要罩住挂点 body 的预期接触区，默认 0.06（约指尖/指板整面大小）
- **实现映射**：encoder→JOINTPOS、joint_torque→JOINTACTFRC、imu→GYRO+ACCELEROMETER、tactile→TOUCH 标量（不是需要 plugin 的 mjSENS_TACTILE 网格阵列）；传感器是只读派生量，加不加传感器物理轨迹逐位一致（有 bit 级守护测试）

## 末端夹爪装配（4 款：2FG7 / 2F-85 / UMI / LEAP）

给机器人挂夹爪有两条等价路径：

1. **GUI（推荐）**：加载世界 → 右侧"装备"页 → 选机器人 → 夹爪下拉选型号 → "应用夹爪"；传感器也在同页表单里点选添加。改动写回世界 YAML 并自动重载。
2. **YAML**：机器人下加 `gripper` 字段（见上例）。

```yaml
gripper:
  model: onrobot_2fg7      # 库 key；也可写自定义 MJCF（相对世界文件的路径）
  site: attachment_site    # 可选：挂点 site，缺省用该夹爪标定的 attachment_site
  pos: [0, 0, 0]           # 可选：覆盖标定安装平移（米，挂点局部系）
  quat: [1, 0, 0, 0]       # 可选：覆盖标定安装朝向（wxyz）
```

**夹爪库**（`simovia/grippers.py`，资产随仓库自带 `models/grippers/`）：

| key | 型号 | 类型 | 执行器（短名） | 开合 ctrl 范围 |
|---|---|---|---|---|
| `onrobot_2fg7` | OnRobot 2FG7 | 平行夹爪 | `gripper_act` | 0 ~ 0.038（0.038 全开 / 0 闭合） |
| `robotiq_2f85` | Robotiq 2F-85 | 自适应夹爪 | `fingers_actuator` | 0 ~ 255（0 全开 / 255 闭合，原模型标度） |
| `umi_gripper` | UMI Gripper | 平行夹爪 | `fingers_actuator` | 0 ~ 0.05（0 张开 / 0.05 闭合） |
| `leap_hand` | LEAP Hand（左手） | 灵巧手 16 DoF | 16 个（`actuators` + `poses`） | 见下方多执行器说明 |

要点：

- **适配机型**：`ur5e.xml` / `ur10e.xml` / `xarm7_nohand.xml` / `panda_nohand.xml` 四臂，四款夹爪逐臂标定安装位姿（mount）——夹爪后法兰与机器人法兰同轴贴合、指尖沿法兰法向伸出，中央贴合间隙 ≤2mm；未标定的机器人会报清晰错误，也可在 YAML 里显式写 `pos`/`quat` 手动指定。
- **设备命名**：挂爪后夹爪设备带 `gripper/` 前缀并入该机器人设备索引——执行器 `gripper/gripper_act`（2FG7）、`gripper/fingers_actuator`（2F-85 / UMI），关节 `gripper/left_slide` 等，body `gripper/...`。传感器可引用夹爪手指 body（如 `body: gripper/left_finger`），2FG7 指尖预置了 `gripper/left_pad_site` / `gripper/right_pad_site`，TCP 为 `gripper/gripper_tcp`。
- **开合控制（单执行器）**：2FG7 / 2F-85 / UMI 都是一个 position 执行器，按上表范围直接写 ctrl：

```python
robot.set("gripper/gripper_act", 0.038)      # 2FG7 张开（内间隙 64.8mm）
robot.set("gripper/gripper_act", 0.0)        # 2FG7 闭合抓取（内间隙 36mm）
robot.set("gripper/fingers_actuator", 0.0)   # 2F-85 全开
robot.set("gripper/fingers_actuator", 255.0) # 2F-85 闭合
```

- **多执行器灵巧手（LEAP Hand）**：16 自由度没有单一开合执行器，注册表里用 `actuators`（夹爪内短名列表）+ `poses`（`{"open": [...], "close": [...]}`，顺序同 `actuators`）描述。控制器逐个写执行器即按姿态弯曲；`get_robot_manifest` 把 16 个执行器全部展开（全名带 `gripper/` 前缀）并附开合姿态：

```python
info = robot_manifest["robots"][0]["gripper"]["info"]   # LEAP 条目
for name, target in zip(info["actuators"], info["poses"]["close"]):
    robot.set(name, target)      # poses["open"] 即伸展姿态
```


## 控制器 API（Webots 风格）

控制器是普通 Python 脚本，唯一要 import 的东西是 `Robot`：

```python
from simovia.controller import Robot

robot = Robot()                     # 自动从 --sock/--robot 参数连接内核
while robot.step():                 # 推进一 tick，返回观测；内核叫停时返回 False
    lo, hi = robot.range("a1")      # 执行器控制范围
    robot.set("a1", 0.5)            # 按名字写控制量
    obs = robot.obs                 # {"tick","qpos","qvel","bodies","objects",...}
    cube = obs["objects"]["cube"]    # 场景物体全局可见
    touch = robot.sensor("fl_touch")           # 传感器读数（世界 YAML 里声明过）
    imu = robot.sensor("wrist_imu")            # 6 维 IMU 返回 [gx,gy,gz,ax,ay,az]
    snap = robot.supervise("save", tag="mid")   # 快照/读档/teleport
```

控制器与内核是独立进程锁步（每个 step 请求有且只有一个回复）。控制器可以无限等待内核开跑（Agent 的"思考时间"不会杀死控制器）；内核会在 run 结束时通知控制器收尾。

控制器里的 `print` 会原样出现在终端/后端日志里并带机器人名前缀（`[arm] ...`）——多机器人时能分清是谁打的；调试控制律直接 print 即可。内核 stdout 是 MCP 协议通道，所以这些输出统一走 stderr，不会污染协议。

## RL 训练接口（gymnasium）

一个 YAML 世界 = 一个确定性 gym 环境。`reset` 重建 `MjData`，同 YAML 跑两遍轨迹逐 bit 相同——eval 可复现是 Simovia 的差异化卖点：

```python
from simovia.rl_env import SimoviaEnv

env = SimoviaEnv("worlds/pick_cube.yaml")     # YAML 路径 / dict / WorldSpec
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
# action  = 所有机器人执行器按声明顺序拼接（Box，界 = 各自 ctrlrange）
# obs     = 各机器人 qpos + qvel + 传感器读数拼接（float32）
# reward  = 世界 task.reward 的当步加权值；terminated = task.success；
#           truncated = tick 到 max_ticks
# info    = {"tick", "time", "raw": {机器人: {qpos/qvel/ctrl/sensors}}}（float64）
```

设计要点：

- **单进程直驱**：不走控制器子进程、不起 ZMQ，训练吞吐优先；MCP/内核零改动
- **语义一处实现**：任务评估与观测组装和内核共用 `kernel/task.py`，两路不会漂移
- **官方合规**：过 `gymnasium.utils.env_checker.check_env`；`render_mode="rgb_array"` 返回 (600,800,3) 截图（自由机位），SB3/CleanRL 等标准算法库可直接接
- **确定性守护测试**：同动作序列两个 episode 逐 tick qpos `array_equal`（禁 allclose）

## 机器人模型库（MuJoCo Menagerie）

预配置 34 个著名机器人（`simovia/zoo.py`），覆盖八大类：

- **四足（6）**：Unitree Go1 / Go2 / A1、ANYmal C、Boston Dynamics Spot、Google Barkour VB
- **人形（6）**：Unitree H1 / G1、ROBOTIS OP3、PAL TALOS、Booster T1、Berkeley Humanoid
- **双足（1）**：Agility Cassie
- **机械臂（13）**：Franka Emika Panda / FR3、KUKA iiwa 14、Kinova Gen3、UR5e / UR10e、xArm7、Lite 6、WidowX 250s、Dynamixel 2R、SO-ARM100、ARX L5、AgileX Piper
- **灵巧手（1）**：Shadow Hand E3M5
- **双臂（1）**：ALOHA
- **无人机（2）**：Skydio X2、Bitcraze Crazyflie 2
- **移动操作（4）**：PAL TIAGo、Hello Robot Stretch 3、Stanford TidyBot、Google Robot

全部来自 DeepMind 官方精选模型库，经 MjSpec attach 编译进 Simovia 世界（浮动基座四足/人形/无人机实测落地稳定，出生高度按 qpos0 最低点自动贴地）。

**GUI 闭环**（`python -m simovia gui`）：点"模型库"按钮 → 选模型（资产随仓库自带，离线即用）→ 自动生成世界 + 控制器模板 → 自动加载并打开编辑器 → 改完控制律点 Run 重跑。

**Agent 闭环**（走 MCP 工具，与 GUI 按钮同一条）——注意这是 MCP/`tools.*` 接口，CLI 没有对应的 `zoo` 子命令，人类用户请用 GUI 的"模型库"：

```python
r = tools.zoo_create(key="unitree_go2")   # 或 key="list" 先列清单
# -> {"world": "~/.simovia/zoo/unitree_go2/world.yaml",
#     "controller": ".../controller.py", "actuators": [...12 个...]}
tools.load_world(yaml_path=r["world"])
tools.run(ticks=200)
```

生成的 `controller.py` 模板带完整执行器清单与 API 注释——AI Agent 可以直接读清单后改写控制律，人类用户用任意编辑器改写。**模型资产随仓库自带**：34 个模型完整目录（XML + mesh + LICENSE，约 600MB）在 `models/zoo/<key>/`，clone 完即可离线使用，首次点模型库不联网；仅当该目录缺失时才回退可选包 `mujoco-menagerie` 的下载能力。

> 已知语义：attach 时子模型的 MuJoCo option（impratio / cone 等）被父世界值覆盖——接触求解参数以世界 YAML 为准（`worlds/zoo` 生成模板已按各模型类目调好 floor 摩擦）。

## 导入 URDF（含 SolidWorks）

世界 YAML 的 `robots[].model` 直接支持 `.urdf`（也支持 `.xml` MJCF）——GUI 走 文件 → 导入模型（Ctrl+I），YAML 手写直接填路径即可：

```yaml
robots:
  - name: my_arm
    model: /path/to/my_robot.urdf   # 或相对世界文件目录的路径
    pos: [0, 0, 0]
```

URDF 与 MJCF 的差异由导入层（`simovia/world/builder.py` 的 `_load_robot_spec`）统一吸收：

- **mesh 路径**：相对路径按 urdf 文件所在目录解析；`package://NAME/REST` 前缀自动剥除并两级就近查找（ROS / sw2urdf 导出目录惯例）；找不到时报错带原文件名
- **执行器**：URDF 没有 `<actuator>`，每个可动关节自动补一个同名 motor（力矩控制，`ctrl` 直写）；`fixed joint` 在解析时已合并进父 body
- **贴地出生**：GUI 导入时按 qpos0 姿态最低几何点 + 2mm 缓冲自动算出生高度（box 精确到角点、cylinder/mesh 用包围球保守估计）

**SolidWorks 工作流**：装 [sw2urdf](https://github.com/ros/solidworks_urdf_exporter) 插件 → 装配体导出 URDF → GUI 导入选导出的 `.urdf` 即可。注意两点：

1. MuJoCo 的 URDF 解析**只读 `<collision>` 几何，忽略 `<visual>`**——SolidWorks 零件导出时确保勾选碰撞几何（sw2urdf 默认生成）
2. mesh 仅支持 STL / OBJ 格式（sw2urdf 默认导出 STL，无需转换）

导入后机器人未绑定控制器（纯场景摆放），需要控制律时用"代码页"写或让 AI 生成；MJCF（`.xml`）导入行为不变，与旧版完全兼容。

## 从零造机器人（连杆树 + 旋转矩阵）

不想建模、也不想找现成模型时，可以直接用连杆树描述一个机器人：每个连杆写 parent / 平移 / 旋转 / 几何 / 关节，平台编译成 MJCF 挂进世界。

**GUI（推荐，适合初学者）**：文件 → 造机器人（Ctrl+B）打开非模态面板——

- 左侧连杆树：显示 parent 关系与每个连杆的关节类型；"添加子连杆"/"删除连杆"即时改结构
- 中间属性表单：名称、父连杆、`pos`（相对父系的平移）、`rpy`（相对父系的欧拉角，**单位：度**）、质量、几何（box/sphere/cylinder/capsule/ellipsoid + 半尺寸 + 偏置）、关节（fixed/revolute/prismatic/free + 轴 + 范围 + kp/kd）
- 右侧实时预览 + 旋转矩阵表：同时给出该连杆的 **R_rel（相对父系）** 与 **R_world（世界系）**，下方 `R = Rz(yaw)·Ry(pitch)·Rx(roll)` 公式行随 rpy 实时刷新，正好对应 MuJoCo 的 `euler`（xyz）约定；选中的连杆在预览里画出 RGB 三色轴（红=x、绿=y、蓝=z，即 R 的三列）
- "保存为机器人 YAML…" 写出的文件可直接被世界引用

**YAML 格式**（`simovia/robot_spec.py`，字段校验失败会给出 `[links[i]]` 路径）：

```yaml
name: two_link
links:
  - name: base
    pos: [0, 0, 0.08]                 # 相对父系平移（根连杆即世界系）
    geom: {type: cylinder, size: [0.07, 0.06], pos: [0, 0, 0.06]}
  - name: link1
    parent: base
    pos: [0, 0, 0.12]
    rpy: [0, 0, 0]                    # 相对父系欧拉角（度，Rz·Ry·Rx）
    mass: 1.0
    joint: {type: revolute, axis: [0, 0, 1], range: [-170, 170], kp: 50, kd: 1}
    geom: {type: box, size: [0.035, 0.035, 0.15], pos: [0, 0, 0.15]}
```

约定与要点：

- **平移/旋转只作用于自身坐标系**：`pos`/`rpy` 描述"本连杆原点在父系里的位姿"，不影响父连杆；子连杆的 `pos` 再相对本连杆。世界位姿 = 沿链路逐级相乘（面板里的 `p_world` / `R_world` 就是这个结果）
- **旋转矩阵**：`R = Rz(yaw)·Ry(pitch)·Rx(roll)`，与 MuJoCo `<compiler angle="degree" eulerseq="xyz">` 一致；面板同时显示 `R_rel` 与 `R_world`，改任意一个 rpy 都能立刻看到矩阵与子连杆位置怎么变
- **`size` 是半尺寸**：box 为半边长，sphere/cylinder 为半径（cylinder 的 size 为 `[半径, 半高]`），与 MJCF 语义一致
- **关节**：`fixed`（焊死）/ `revolute`（转动，`range` 单位度）/ `prismatic`（移动，`range` 单位米）/ `free`（浮动基座，只能出现在根连杆）；每个可动关节自动生成一个同名位置执行器，`kp` 为刚度、`kd` 落到关节阻尼（Euler 积分器下稳定）
- **命名**：关节名/执行器名默认与连杆同名，控制器里直接 `robot.set("<连杆名>", 目标值)`

**世界引用**：机器人 YAML 可以存成文件，也可以直接内联在 `robots[].spec`：

```yaml
robots:
  - name: my_arm
    model: my_robot.yaml      # 连杆树文件（相对世界文件目录）
    pos: [0, 0, 0]            # 机器人根在世界里的出生位姿
    quat: [1, 0, 0, 0]
  - name: inline_arm
    spec:                     # 也可以直接内联整棵连杆树
      name: two_link
      links: [...]
```

`model` 与 `spec` 二选一（都写或都不写会报错）。连杆树编译后与 Menagerie 模型走完全相同的装配管线：前缀隔离、夹爪挂载、传感器注入、设备索引、`inspect` 清单全部适用。

## AI 场景生成 Agent

文件 → AI 生成场景（Ctrl+G）：自然语言需求 → LLM 生成世界 YAML → 本地校验 → 加载，与参数化模板（`scenegen`）、手写 YAML 三条路径并存。

- **生成闭环**（`simovia/agent/scene_agent.py`）：LLM 输出 → 剥围栏提取 YAML → `parse_world` schema 校验 + `build_world` 编译校验 → 失败把可读错误喂回 LLM 重试（默认 3 次）→ 通过即加载
- **模型引用**：prompt 注入完整模型库清单，LLM 写 `zoo://unitree_go2` 引用 → 本地自动解析成模型绝对路径（资产随仓库自带，离线即用）
- **配置复用**：与 Agent 控制台共用 `~/.simovia/agent.json`（任意 OpenAI 兼容端点：OpenAI / DeepSeek / GLM / Ollama / vLLM）
- **抗"MJCF 人设"模型**：给本地模型（如 Ollama）预设过 MJCF/XML 生成器提示词也能用——请求级 system prompt 带人设覆盖指令 + 生成规则明令禁 MJCF/mesh；模型仍输出 XML 时回喂定向纠错（"平台不加载外部 mesh，改用基本体+zoo://"），实测收敛
- **保留手动路径**：AI 生成不取代手写——模板（`scenegen`）、手写 YAML、导入模型全部保留，AI 只是多一条"说一句话就出场景"的捷径

### 本地模型接入：Ollama 从安装到一句话出场景

全程本地免费、无需任何 API Key。三步：装 Ollama → 拉模型 → GUI 填两项配置。

**第一步 安装 Ollama**

- Windows：官网 https://ollama.com/download 下载 `OllamaSetup.exe` 双击安装（装完自动运行，系统托盘出现羊驼图标）；或 PowerShell 执行 `winget install -e --id Ollama.Ollama`
- Linux：`curl -fsSL https://ollama.com/install.sh | sh`
- macOS：`brew install ollama`，或官网下载 dmg

验证服务在跑（任一系统）：

```bash
ollama list                       # 列出已有模型，不报错即正常
```

浏览器打开 http://localhost:11434 显示 "Ollama is running" 同样说明正常。

**第二步 拉一个模型**

推荐 `qwen2.5-coder:7b`（约 4.7GB，YAML 遵从度高，消费级显卡/纯 CPU 可跑）：

```bash
ollama pull qwen2.5-coder:7b
ollama run qwen2.5-coder:7b 你好    # 出话即就绪
```

- **不要**给模型预设 MJCF/XML 生成提示词——平台每次请求都注入自己的场景生成规范（schema + 模型库清单 + few-shot 示例），并主动覆盖模型内置人设；已经烘焙过"MJCF 生成器"提示词的模型也能用（定向纠错回喂兜底），但纯白模更干净
- 机器跑不动 7B 换 `qwen2.5-coder:3b`；想提高一次通过率上 `qwen2.5-coder:14b`。7B 生成世界 YAML 已足够

**第三步 接入平台**

两种方式等价，任选其一：

- **GUI 配置（推荐，全平台一致）**：Agent 页 → "配置"，填三项——`base_url` 填 `http://localhost:11434/v1`（注意带 `/v1`）、`api_key` 留空、`model` 填 `qwen2.5-coder:7b`
- **手写配置文件**（Linux/macOS 在 `~/.simovia/agent.json`，Windows 在 `C:\Users\<用户名>\.simovia\agent.json`）：

```json
{
  "base_url": "http://localhost:11434/v1",
  "model": "qwen2.5-coder:7b"
}
```

配置好后两条路立即生效：文件 → AI 生成场景（Ctrl+G）一句话出场景（不满意继续说，"应用修改"增量微调）；Agent 页聊天让 AI 直接调平台工具干活。

**常见问题**

- 生成报"重试 3 次仍失败"——多半模型太小，换 7B 起步；7B 仍频繁失败上 14B
- 模型输出 MJCF/XML 或引用 `.obj/.dae` mesh——平台自动定向纠错回喂（"平台不加载外部 mesh，改用基本体 + zoo://"），属正常闭环不是错误；**永远不需要**去网上下载任何 mesh 文件，物体用基本体、机器人走 `zoo://` 模型库
- 报"LLM 服务不可达"——Ollama 没在跑（Windows 看托盘图标），或 `base_url` 忘了带 `/v1`
- 远程机器上的 Ollama——服务端设 `OLLAMA_HOST=0.0.0.0` 重启服务（Windows 用户环境变量同名），客户端 `base_url` 填 `http://<机器IP>:11434/v1`，防火墙放行 11434 端口
- Windows 说明——配置文件权限保护（`chmod 0600`）是 Linux 行为，Windows 自动忽略；本地 Ollama 无 api_key，本就不存在泄密面


## 桌面 GUI 智能工作台（M5）

GUI（`python -m simovia gui`）从可视化扩展为完整的机器人编程工作台——左右四块面板（场景树/属性/代码/Agent），配合模型库的运动演示与位姿编辑，覆盖"打开/写代码/调位姿/跑/让 AI 来"全闭环。

### 1. 运动演示一键播放

模型库窗口选中任意机器人 → "生成演示并运行"按钮，自动生成内置运动演示并加载世界 → 自动 Run。34 个模型全部预配置运动（见 `simovia/zoo.py`）：

- **四足小跑（trot）**：Unitree Go1 / Go2 / A1、ANYmal C、Boston Dynamics Spot——按对角腿相位（FL-RR / FR-RL）交替摆动；CPG 控制律 `freq`/`amp`/`lag` 在 `zoo.py` 顶部 `_trot` 工厂生成，落地即稳定行走
- **机械臂扫掠**：Franka Panda / FR3、KUKA iiwa 14、Kinova Gen3、UR5e / UR10e、xArm7、Lite 6、WidowX 250s、SO-ARM100、ARX L5、AgileX Piper 等——多关节正弦相位机，base 绕 z 旋转 + 末端周期摆动
- **人形摆臂**：Unitree H1 / G1、OP3、TALOS、Booster T1、Berkeley Humanoid、Cassie——髋膝肩肘相位差摆动模拟踏步
- **无人机**：Skydio X2 / Crazyflie 2——姿态振荡展示
- **扫掠/静止**：Shadow Hand / ALOHA / Google Robot / Stretch 3 / TIAGo / TidyBot——多指/双臂/移动底盘姿态展示

> 四足稳定行走的根因：MuJoCo `attach` 时父世界（默认 Euler 积分器）的 option 会覆盖子模型原值——Spot 原生 `solref=0.004 1` 超硬接触用 `implicitfast`，被 Euler 覆盖后"硬接触+显式积分"会数值震荡弹跳。`WorldSpec` 已加 `integrator` 字段，四足 demo 世界统一设 `implicitfast`。详见 `zoo.py` 中各模型的根因注释。

### 2. 机器人/物体位姿编辑（属性面板 + 3D 拖动）

在场景树选中机器人**或物体**节点 → 右侧"属性"面板出现位姿组：3 个 xyz 双精度 SpinBox（步进 0.01m，范围 ±20m）+ 3 个 roll/pitch/yaw 角度框（步进 5°，范围 ±360°）→ "应用位姿"按钮：

- 欧拉角与四元数双向转换（`panels.quat_to_euler` / `euler_to_quat`，ZYX 内旋序）
- 单位朝向（1,0,0,0）时 YAML 里 `quat` 字段省略（不写就是默认）
- 应用流程：UI → `read_world` MCP 工具拿当前 YAML → 改 `pos`/`quat` 字段 → `load_world` 重载 → 自动重选该节点
- **跑完演示（FINISHED）后也能直接编辑**——提交走 `load_world` 全量重载，世界自动回到 YAML 初态；仅运行中（RUNNING）禁止编辑并给出提示

3D 视口内的拖动（编辑态）：

- **左键拖动**：水平面（XY）内移动选中对象
- **Ctrl+左键拖动**：沿 Z 轴调高度（向上拖=升）
- 拖动时 HUD 左上实时显示选中对象的**三维世界坐标**，松手写入世界 YAML 并重载

机器人模型自带 root body 的 pos/quat 会被 `build_world._attach_robot` 清零——`spec.pos`/`spec.quat` 就是机器人 root 的**绝对世界位姿**。例如 `demo_arm.xml` 的 base 自带 0.05m 高度，世界 YAML 里要写 `pos: [0, 0, 0.05]`（吸收内部偏移）。

### 3. GUI 内置代码编辑器（代码页）

GUI 右侧第四个 tab"代码"：一个 Python 语法高亮编辑器（`PyHighlighter`——QSyntaxHighlighter 零依赖近似实现，字符串/注释/关键字/数字四类着色），下拉选当前世界的机器人 → 加载其控制器文件全文 → 编辑 → "保存并运行"按钮或 `Ctrl+S` 快捷键：

- 保存路径：世界 YAML `base_dir` + 机器人 `controller` 字段
- 保存后流程：写文件 → `read_world` 重载世界 → 自动 Run（`_pending_run` 状态标记 + 重载完成后 `if self.state.can("run"): self._do_run(...)`）
- 没绑控制器的机器人显示"没有绑定控制器"提示

可以同时打开任意编辑器改 `.py` 后回到 GUI 点 Run——GUI 内编辑只是更顺手。

### 4. Agent 控制台（Agent 页）

GUI 右侧第五个 tab"Agent"：聊天式下达任务给 AI（OpenAI 兼容 chat/completions API，支持本地 Ollama / OpenAI / DeepSeek / 智谱等任何 OpenAI 兼容服务）。点"配置"填三项：

- `base_url`：例如 `https://api.openai.com/v1` 或 `http://localhost:11434/v1`
- `api_key`：本地服务可空（Ollama），公开服务必填；密码模式输入
- `model`：例如 `gpt-4o-mini`、`deepseek-chat`、`glm-4`

配置保存到 `~/.simovia/agent.json`，`chmod 0600` 保护 api_key。配好后聊天下达任务（Enter 发送，Shift+Enter 换行）：

- Agent 拿到的是 OpenAI function calling 格式工具清单（`bridge.to_openai_tools`）——MCP 的 12 个工具（含 `stop` 停止仿真）+ 3 个本地工具（`write_file`/`read_file` 限定在世界目录内防越界；`read_mujoco_doc` 按需读 MuJoCo 4.0 随包自带的官方手册与 agent skills，避免凭记忆写模型参数）
- 工具循环：Agent 返回文本 → 显示在聊天框；返回 tool_calls → MCP 同步等回复（QEventLoop + 30s QTimer 兜底）→ 显示工具调用与结果 → 继续下一轮直到 Agent 不再调工具
- 系统提示词包含：身份（机器人仿真助手）、当前世界名、所有工具清单（MCP + 本地）、建议工作流（read_world → load_world → run → get_observation）

让 Agent 干的事例：生成一个抓取场景、写一个控制器让臂推 cube、跑起来看结果——Agent 直接读 MCP 工具结果自循环完成。

### 5. 造机器人面板（Ctrl+B）

文件 → 造机器人：左侧连杆树增删改父子关系，中间编辑该连杆的 pos / rpy / 质量 / 几何 / 关节，右侧 420×420 实时 3D 预览 + 旋转矩阵表：

- 选中任意连杆，表格同时显示 **R_rel（相对父连杆）** 与 **R_world（世界系）**，下方 `R = Rz(yaw)·Ry(pitch)·Rx(roll)` 随 rpy 实时刷新
- 预览里在该连杆世界位置画出 RGB 三色坐标轴（红=x、绿=y、蓝=z），直接对应 R_world 的三列——改一个角度就能看到轴怎么转、子连杆怎么跟着走
- 结构修改（添加子连杆 / 删除连杆）实时反映在树上；删除中间连杆时子连杆自动上移一级，不会断树
- "保存为机器人 YAML…" 得到连杆树文件，按"从零造机器人"一节的写法在 `robots[].model` 引用（也可把 `links` 直接内联进 `robots[].spec`）

### 6. PPO 训练示例

`examples/train_ppo_go2.py`：用 stable-baselines3 PPO 在 Go2 上训练行走策略（demo CPG 已经能让 Go2 走 10m，这里做 RL 微调）。sb3 是可选依赖——不装时脚本会给出 `pip install stable-baselines3` 提示后退出，不在回归路径里。

```bash
pip install stable-baselines3
python examples/train_ppo_go2.py
# 200k 步训练，输出 models/ppo_go2.zip
```

奖励设计：`forward（5× dz_root）+ 高度罚（-3 × |z - 0.30|）+ 能耗正则（-0.01 × Σ a²）+ 跌倒终止（rootz < 0.15）`。世界用 `zoo.generate("unitree_go2")` 生成 + 删 controller（env 直驱）+ 保留 `integrator=implicitfast` / `impratio=100` / `ctrl_period=8`——四足稳定的关键参数。

## 视觉水果分拣与独立手眼标定

[demo/](demo/README.md) 提供可直接在软件里加载、编辑控制器并运行的场景：

- 水果分拣：Panda + 2FG7 + 腕部 RGB 相机，香蕉、苹果、葡萄各两个，HSV 识别后分拣到三个框，运行时自动显示识别画面。
- 独立标定：Panda 与 UR5e 的眼在手上标定场景，采集 ArUco 图像、执行 PARK 手眼求解并保存经过留出验证的结果。

在已激活的虚拟环境中统一安装平台与视觉依赖：`python -m pip install -r requirements.txt -r demo/fruit_sorting/requirements.txt`，然后启动 GUI，从“文件”菜单进入对应 demo。水果场景直接读取预先保存的标定结果；重新标定在独立场景中完成。

详细操作、场景路径和其他机械臂的适配说明见 [demo 使用说明](demo/README.md)。

## 测试

```bash
python -m pytest tests/ -q
# 测试数量以当前输出为准；覆盖 schema/builder/内核/多机器人/Supervisor 逐位重现/
#              MCP（含 screenshot/zoo_create/stop/save_world 往返）/
#              传感器（含 bit 级轨迹守护）/
#              RL 环境（含逐位一致性守护）/控制器心跳（含跨平台死亡探测）/
#              握手鲁棒性（超时终止/握手期停止/stderr 直通）/模型库/场景生成/
#              GUI（含深色主题守护、浮动基座 root 位姿同步、三维拖动/
#              FINISHED 可编辑/添加物体对话框/AI 场景对话框/导入贴地高度）/
#              M5 运动演示 / 位姿 quat 链路 / 代码编辑器 / Agent 控制台后端 /
#              M6 夹爪（2FG7/2F-85/UMI/LEAP 四臂装配/索引/物理开合/
#              16 执行器展开/触觉挂手指）/ 装备面板 /
#              码垛分区 pallet / ur10e 入库 /
#              URDF 导入（motor 自动补/mesh 相对与 package:// 解析/
#              坏文件可读报错/编译冒烟/盘符路径保留/GBK 编码兜底）/
#              造机器人（连杆树编译/矩阵与 MuJoCo 运动学逐连杆一致/
#              面板增删改/保存回读/robots[].spec 与 .yaml 引用）/
#              AI 场景生成（YAML 提取/zoo:// 替换/校验回喂重试/耗尽报错）
```

其中包含真实 Agent 压测回归：控制器在 LLM 思考时间下存活、崩溃控制器在统计中可见、每轮 run 零进程泄漏。M5 新增 13 项测试覆盖 demo 生成（全覆盖 + go2 600 tick 行走 >1m + panda 关节活动度）、quat 链路（attach + 欧拉↔四元数 roundtrip + 属性面板应用）、PyHighlighter 四类着色 + CodePanel 保存信号、Agent 后端（config 0600 + chat_url 兼容 + MCP↔OpenAI 工具转换 + 本地文件越界防护）。停止功能另加 5 项：内核 `stop_check` 三态（带控制器提前收尾不泄漏 / 无控制器空跑可停 / 恒 False 行为不变）、MCP stop 工具（run 中停止 + 无 run 无操作）、GUI 状态机（停止仅 RUNNING 可用）。

## 项目结构

```
simovia/
├── simovia/
│   ├── kernel/sim.py      # 仿真内核：ZMQ ROUTER + 栅栏同步推进
│   ├── kernel/task.py    # 任务评估 + 观测组装纯函数（内核与 RL env 共用）
│   ├── rl_env.py          # gymnasium.Env 包装（RL 训练接口）
│   ├── zoo.py              # 机器人模型库（Menagerie 34 个预配置模型 + 运动演示生成器）
│   ├── robot_spec.py       # 连杆树描述 → MjSpec（造机器人面板的数据层/矩阵教学）
│   ├── grippers.py         # 夹爪库（4 款：2FG7/2F-85/UMI/LEAP 灵巧手，四臂 mount 标定）
│   ├── controller.py      # 控制器侧 Robot 客户端（用户唯一 API）
│   ├── world/schema.py    # YAML → dataclass，前置校验（含 gripper/integrator/quat）
│   ├── world/builder.py   # spec → MjSpec → MuJoCo 模型编译（机器人+夹爪统一装配）
│   ├── supervisor.py      # 快照/teleport（逐位可复现）
│   ├── mcp_server.py      # MCP stdio 服务器（零依赖 JSON-RPC）
│   ├── render.py          # 离屏渲染 + 手写 PNG 编码（screenshot 实现层）
│   ├── agent/             # GUI Agent 控制台后端（config/bridge/local_tools/
│   │                        # worker）+ scene_agent（AI 场景生成闭环）
│   ├── gui/               # PySide6 桌面工作台（app/render_view/panels[含装备页]/
│   │                        # robot_builder[造机器人面板]/mcp_client）
│   ├── scene_gen.py       # 6 个确定性场景模板（含码垛分区 pallet）
│   └── cli.py             # run / inspect / mcp / gui / scenegen 命令行
├── models/                # 机器人 MJCF（demo_arm 三指臂、diff_car）
│                          # + grippers/{onrobot_2fg7,robotiq_2f85,umi_gripper,leap_hand}
│                          # + zoo/<key>/：34 个 Menagerie 模型资产随仓库自带（离线即用）
├── controllers/           # 示例控制器（pick_place 桌面抓取相位机等）
├── worlds/                # 示例世界（pick_cube、twin_arms、empty——GUI 启动默认加载）
├── examples/             # PPO 训练示例（train_ppo_go2.py，sb3 可选依赖）
├── start.bat / create_desktop_shortcut.bat / simovia.ico
│                            # Windows 双击启动 + 桌面快捷方式（项目图标）
└── tests/                 # 自动化回归测试（含 URDF 导入/AI 场景生成/夹爪/码垛/造机器人）
```

## Roadmap

- [x] M0 仿真内核（物理 / 进程隔离 / Supervisor / MCP / 场景生成）
- [x] M1 GUI（PySide6 可视化 + 实时观测面板）—— [实施方案](docs/M1_GUI实施计划.md)：GUI 作为 MCP 客户端（与 AI Agent 同构）+ MuJoCo 离屏渲染；run 后台线程化，60Hz 观测快照（`python -m simovia gui`）
- [x] M2 传感器系统 + MCP screenshot —— 五类传感器（encoder / IMU / 触觉 / 关节力矩 / 相机）YAML 声明进观测流，bit 级守护保证不改变物理轨迹；`screenshot` 工具让 Agent 截取仿真画面（自由机位或声明的相机机位）
- [x] M3 RL 训练接口 —— `SimoviaEnv`（gymnasium.Env）：YAML 世界即确定性环境，动作/观测/奖励语义与内核一处实现（`kernel/task.py`），同 YAML 两遍轨迹逐位一致
- [x] M1 心跳机制 —— 控制器 `_recv` 100ms poller 轮询 + `getppid` 孤儿检测：内核进程一死控制器必被 init/subreaper 收养（父 pid 变化），随即 `RuntimeError("内核已退出")` 退出；思考时间无限等待语义不变
- [x] M4 机器人模型库 —— MuJoCo Menagerie 官方模型一键接入（34 个预配置著名机器人，覆盖四足/人形/机械臂/灵巧手/双臂/无人机/移动操作），`zoo_create` 生成世界 + 控制器模板，GUI 点击-打开-编程闭环
- [x] M5 运动演示与智能工作台 —— 34 个模型一键运动演示（四足 trot / 机械臂扫掠 / 人形摆臂 / 无人机振荡）+ GUI 位姿编辑面板（xyz+欧拉角）+ 内置 Python 代码编辑器（保存并运行）+ Agent 控制台（OpenAI 兼容 chat API 调用，AI 直接生成场景/写代码/运行验证）+ PPO 训练示例（`examples/train_ppo_go2.py`）
- [x] M6 夹具/传感器装配工作台 —— OnRobot 2FG7 夹爪入库（site-attach 到 UR5e/UR10e/xArm7/Panda，mount 按四臂腕部几何逐臂标定，夹爪与法兰同轴贴合）+ GUI"装备"页一键挂爪/拆爪与五类传感器表单式增删（自动改 YAML 重载）+ 码垛分区场景模板 pallet（UR5e+2FG7，A 区取料→B 区码放，IK 验证全部取放点可达）+ GUI"新建场景"对话框 + UR10e 入库
- [x] M7 URDF/SolidWorks 导入 + AI 场景生成 —— `robots[].model` 直读 URDF（mesh 路径重写 / package:// 剥除 / 逐关节补 motor，SolidWorks sw2urdf 导出即用）+ GUI 导入模型入口（贴地出生自动计算）+ AI 场景生成 Agent（自然语言 → 世界 YAML，本地双重校验失败回喂重试，`zoo://key` 模型引用）+ GUI 分栏自由拖动（面板滚动条化，视口可放大）
- [x] M8 模型库扩容 + 从零造机器人 —— 模型库 14 → 34 个（新增 9 机械臂 / 2 四足 / 4 人形 / 1 双足 / 2 无人机 / 2 移动操作，全部资产自带离线可用），夹爪库 1 → 4 款（新增 Robotiq 2F-85 / UMI Gripper / LEAP Hand 16 自由度灵巧手，四臂 mount 逐臂标定，灵巧手 `actuators`+`poses` 多执行器开合）；连杆树 YAML + `robots[].spec` 内联定义 + GUI「造机器人」面板（连杆树编辑 / 实时预览 / R_rel + R_world 旋转矩阵 / RGB 轴标 / 保存 YAML）

## 已知取舍

- ~~内核进程被强杀（kill -9）时控制器可能成为孤儿阻塞~~ **已由 M1 心跳机制补全**（`tests/test_heartbeat.py`：强杀内核后控制器 ≤2s 退出）
- 已支持桌面 GUI 和离屏相机；当前使用经典 OpenGL 渲染，尚未接入 Filament/PBR。最低硬件配置及跨平台兼容性仍需扩大实机验收。
- MCP 实现为协议子集（initialize / ping / tools），覆盖主流客户端所需

---

如果你在做一个用 AI 操控机器人仿真的项目，欢迎 issue / PR。
