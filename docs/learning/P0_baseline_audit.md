# P0 Baseline Audit —— AI Learning Engine 动工前的只读诊断

日期：2026-11-16 ｜ 依据：《Simovia AI Learning Engine V1.0》§16 首轮指令 ｜ 性质：**只读，未修改任何现有代码**

## 1. 现状

### 1.1 代码版本

- HEAD commit：`e23a3d9`（`ci: 自动部署展示站到 GitHub Pages`）
- **工作区不干净**：13 个已跟踪文件有未提交修改（含 `simovia/world/schema.py`、`simovia/mcp_server.py`、`worlds/so101_policy_eval.yaml`、`demo/hand_eye/results/panda_wrist_rgb.json` 等），另有 10 个未跟踪文件/目录（`demo/cloth_folding/`、`worlds/g1_cloth_*.yaml`、`tests/test_maintenance_ik.py`、`episodes/` 等）。
- **影响**：当前物理基线绑定的是"HEAD + 未提交改动"的混合状态，任何复现实验必须先处理这些改动（提交或暂存），否则场景哈希与代码版本无法一一对应。这是 P1 动工前的第一项工程风险。

### 1.2 环境

| 项 | 值 | 状态 |
|---|---|---|
| Python（base） | 3.14.7 x86_64 | OK |
| MuJoCo | 4.0.0 | OK |
| NumPy / PyYAML / pyzmq | 2.2.6 / 6.0.3 / 27.2.0 | OK |
| Gymnasium | 1.4.0 | OK |
| OpenCV | 4.12.0 | OK |
| torch / lerobot（base） | **未安装** | 见 1.3 推理环境 |
| GPU | NVIDIA RTX 5070，12,227 MiB 总显存（诊断时空闲 7,814 MiB） | 验证 §9.3"约 12GB"假设成立 |
| `python -m simovia doctor` | 全部通过（35/35 模型资产、离屏渲染、pick_cube 编译） | OK |

### 1.3 推理环境（SmolVLA 链路）

- 带 lerobot 的解释器为 conda 环境 `smolvla`：torch 2.15.0+cu132 / lerobot 0.7.0 / `torch.cuda.is_available()=True`。
- `controllers/so101_policy_eval.py` 内置环境自举：检测不到 lerobot 时按 `SIMOVIA_POLICY_PY`（默认 `smolvla` 环境）`os.execv` 切换解释器——推理链路**不阻塞**。
- 其他环境（`orca_repro_040`：torch 2.7.1+cu126/lerobot 0.3.3、`orcalab_lerobot`：torch 2.7.1 cpu）与本 demo 无关，注意别混用。

### 1.4 冻结基线资产（哈希已存档，见 §5 manifest）

- **Checkpoint** `so101_sort2_20k` 双份存放，除 `model.safetensors` 仅存在于 `simovia_workspace/models/` 外，其余 9 个文件两处逐位一致（config/tokenizer/preprocessor 等 sha256 已记录）。`worlds/so101_policy_eval.yaml` 的 `args.model` 指向 `simovia_workspace/models/so101_sort2_20k`。
- **数据集** `simovia_workspace/datasets/so101_smolvla_sort2`：LeRobot v2.1，50 集 / 20,247 帧，fps 31.25，100 路视频（2 相机 × 50 集），`observation.state` 与 `action` 各 6 维（shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper）——与《so101-smolvla-demo》README 及方案文档 §1.1 的记录一致。
- **测试集/评测世界**：`worlds/so101_policy_eval.yaml`（原始固定布局）与 `worlds/so101_policy_eval_sim2sim.yaml`（物理/渲染扰动版，几何与相机严格一致）均已就位。

### 1.5 已核实的能力清单（方案文档 §1.1 复核结论）

核对结果：文档对仓库现状的描述**总体准确**，16 个 MCP 工具、`SimoviaEnv`（gym.Env）、`recorder.py`（NPZ+meta+可选 PNG）、`lerobot_writer.py` v2.1→v3.0 升级链、PICO/WebXR/OSC 遥操作、状态快照、35 个 zoo 模型（含 so101_lerobot）全部属实。但文档**低估了已有能力**，以下资产在 P1/P2 中必须复用而不是重造：

1. `simovia/scene_gen.py`：`scenegen` 命令已有 6 个确定性参数化模板（tabletop/push/maze/race/empty/pallet）——P1 的 `scene_sampler.py` 应扩展它，而非新建。
2. `simovia/agent/scene_agent.py`：自然语言 → YAML → `parse_world`+`build_world` 本地校验 → 错误回喂重试的闭环**已经存在**（文档 §0 描述的"LLM 提出 + 验证器生成"雏形）。
3. `simovia/contract.py`：名字驱动的观测-动作契约（可序列化、录制/训练/sim2sim 三处共用）——§5.2 的 `PolicyAdapter` 必须建立在其上，避免第二套名字语义。
4. `worlds/so101_policy_eval_sim2sim.yaml`：已是一个手动"域随机化"样本，可直接作为 P1 随机化空间的参数对照。

### 1.6 复现实验（文档现象核验）

按 §16 指令应"对固定布局和方块交换场景各执行多次"。本审计判定：**推理运行属于 P0 可选项，但建议延后到工作区干净后进行**——当前 13 个未提交修改中包含 `worlds/so101_policy_eval.yaml` 本身，此刻跑出的成功率无法归因到任何确定版本。README 记录的现象（固定布局可完成、交换红蓝初始位置后蓝块反复在原红块位置重试）按 §17 要求标记为**项目记录，尚未本地复现**，不作为已验证事实。

## 2. 自动化测试基线

全量 581 个用例（base 环境，HEAD `e23a3d9` + 未提交改动）：**581 passed / 0 failed**（分 6 批串行，合计约 2.5 分钟；明细见 §5 manifest）。

- `test_hand_eye_demo.py` 8 例全过——前一会话诊断的 4 个 CRLF 标定指纹失败**已在当前工作区的未提交修改中修复**（`demo/hand_eye/results/*.json` 有未提交更新），但修复未提交；R2 相应降级为"提交前保持绿灯"。
- **遗留异常**：本审计启动的全量单跑 `pytest tests/`（不分批）在 300～480 秒超时窗口内未完成，且诊断时机器上另有 3 个更早遗留的全量 pytest 进程各持续 100% CPU 80+ 分钟并各持有一个 ZMQ 监听端口——说明**"全量一跑"存在卡点**（复测：fruit/heartbeat/teleop 五文件组合两次卡死在第 32 个用例附近，而 5 文件单独跑全部秒过；两两分组亦全过）。分批跑不影响结论正确性，但该组合卡点（疑似测试间 ZMQ/WebSocket 资源交互）应在 P2 `batch_runner` 动工前排查——批量评测器的测试隔离若踩同一个坑会持续踩雷。

## 3. 缺口确认（相对方案文档 §1.2）

以下在仓库中确认**不存在**，需要 P1/P2 新建：`simovia/learning/` 包及其 14 个模块、`learning` CLI 子命令、`configs/learning/`、`benchmarks/`、`experiments/`、`tests/learning/`。文档标"拟新增"的定位准确。

需要**澄清后修正文档**的 4 点（已在上轮核对汇报，此处存档）：

1. §1.1/§1.2 未提 `scene_gen.py` 与 `agent/scene_agent.py`（见 1.5）。
2. §5.2 未提 `contract.py`。
3. §5.1 示例 `checkpoint: models/so101_sort2_20k` 的仓库归属应写明 `simovia_workspace/models/`。
4. "README 写 12 个工具"实为两处（架构图 + 工具表漏列 teleop_start/teleop_cmd/camera_stream/camera_frame 共 4 个）。

## 4. 风险与优先级

| # | 风险 | 建议处置 | 优先级 |
|---|---|---|---|
| R1 | 工作区 13 个未提交修改 + 10 个未跟踪文件污染基线 | P1 第一步：与用户确认归属后提交或清理；冻结 `baseline_worlds` 的场景哈希 | 高 |
| R2 | 全量单跑 pytest 存在组合卡点：fruit/heartbeat/teleop 五文件合跑两次卡死在第 32 例附近，单文件与两两分组全过 | P1 期间 CI 用分批跑兜底；P2 前排查测试间 ZMQ/WS 资源交互（另：3 个遗留全量 pytest 进程 100% CPU 各 80+ 分钟持 ZMQ 端口，需人工清理） | 高 |
| R3 | base 环境无 torch/lerobot，测试/训练环境分裂 | P2 的 `batch_runner` 必须显式声明解释器（复用 `SIMOVIA_POLICY_PY` 自举机制），manifest 记录双环境版本 | 中 |
| R4 | `worlds/so101_policy_eval.yaml` 相机位姿是策略输入分布的一部分 | P1 随机化白名单**不得**触碰 cameras 段（文档 §6.2 第 4 条已有，落实到 `scene_validator` 的硬编码禁改清单） | 中 |
| R5 | 两份 checkpoint 内容不一致疑点（demo 仓库一份缺 `model.safetensors`） | P1 的 registry 以 `simovia_workspace/models/so101_sort2_20k` 为权威并锁定其哈希 | 低 |

## 5. 结论与下一 PR 范围

P0 判定：**基础设施可支撑 AI Learning Engine 动工**，无阻塞性缺陷；两个高优先级工程风险（R1 工作区、R2 全量跑卡点）应在 P1 前或期间处理。

下一 PR（P1，仅安全场景生成）范围：
1. 复用 `scene_gen.py` 机制新增 `simovia/learning/scene_sampler.py`（SO-101 分拣变体：红蓝方块 XY + yaw 随机化，白名单字段）。
2. `simovia/learning/scene_validator.py`：WorldSpec 校验 + `build_world` 编译 + 初始穿透检查 + 场景指纹（sha256，对文本输入先归一化换行再哈希——吸取 hand_eye `rig_identity` 的 CRLF/LF 指纹漂移教训）。
3. train/dev/test 分区种子固定 + 100 个冻结测试世界定义。
4. `tests/learning/`：同种子确定性、非法拒绝、无穿透、指纹稳定四类测试。
5. 不启用自动微调，不触碰 VLA 推理控制器。

哈希与环境明细见 `experiments/learning/p0/manifest.json`。
