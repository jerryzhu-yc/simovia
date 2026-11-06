"""MCP 工具层：把内核能力经 stdio JSON-RPC 暴露给 AI Agent（方案 5.6 节）。

零依赖实现：环境里没有 mcp 包，而协议子集很小，自己写更稳——
- stdio 传输，换行分隔的 JSON-RPC 2.0
- 只实现 MCP 必需方法：initialize / notifications/initialized / ping /
  tools/list / tools/call

工具即内核薄封装（AI Agent 的标准闭环）：
    read_world -> 改 YAML -> load_world -> run -> get_observation -> 循环

语义说明：
- load_world 会整体替换当前世界（旧仿真的控制器进程先收尾再退出）
- reset 按方案 5.4 语义：重编译 + 控制器进程重启，世界回到 YAML 初态
- save_state/load_state 用内核内存快照（tag），与 M0.4 逐位重现测试同一路径
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import threading
import time
import traceback

# 离屏渲染后端：Linux 默认 EGL（无头/远程友好），Windows 默认 wgl
# （桌面 Windows 无 EGL 用户态库，wgl 是官方支持路径）。必须在 import
# mujoco 之前设置；用户可用环境变量覆盖（与 gui/__init__.py 同款）。
os.environ.setdefault("MUJOCO_GL",
                      "wgl" if sys.platform == "win32" else "egl")

import mujoco
import yaml

from simovia import supervisor
from simovia.errors import WorldError
from simovia.grippers import END_EFFECTORS as _END_EFFECTORS
from simovia.kernel.sim import Simulation
from simovia.render import depth_to_rgb, encode_png, render_offscreen
from simovia.teleop import ArmTeleop
from simovia.world.schema import load_world as _load_world_file
from simovia.world.schema import read_text_compat
from simovia.world.schema import (absolutize_world_paths, parse_world,
                                  resolve_zoo_models)

_PROTOCOL = "2024-11-05"

# 工具结果直传 MCP content（image 等非文本块）：dispatch 识别此类型并透传，
# 其余结果仍按惯例 JSON 化后包成 text content。
class _RawContent(dict):
    def __init__(self, content: list):
        super().__init__(content=content, isError=False)


# run 的响应被推迟（后台线程跑完才写回）时 dispatch 返回此哨兵，
# serve 主循环见到它不写 stdout（响应由 run 线程经 Writer 写）。
_DEFERRED = object()

# 观测快照节流：GUI 60Hz 拉观测，快照按 60Hz 生成。内核实时模式已按物理
# 子步约 16ms 出一次采样，这里的阈值必须显著小于 16ms（否则抖动会隔帧
# 丢采样，画面又变回 30Hz）；全速跑批时 on_sample 不启用，只有每控制
# tick 一次 on_tick，节流不影响结果。
_SNAP_INTERVAL = 0.010

_JTYPE_NAME = {int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
               int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
               int(mujoco.mjtJoint.mjJNT_FREE): "free",
               int(mujoco.mjtJoint.mjJNT_BALL): "ball"}

# ---------------------------------------------------------------- 工具清单

_TOOLS = [
    {
        "name": "load_world",
        "description": "加载世界（yaml_path 与 yaml_text 二选一），整体替换当前世界。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_path": {"type": "string",
                              "description": "世界 YAML 文件路径"},
                "yaml_text": {"type": "string",
                              "description": "世界 YAML 全文（Agent 改完直接投喂）"},
                "base_dir": {"type": "string",
                             "description": "相对路径（model/controller）的解析基准目录；"
                                            "yaml_path 缺省取文件所在目录，"
                                            "yaml_text 缺省取当前目录"},
            },
        },
    },
    {
        "name": "run",
        "description": "推进仿真并返回摘要（ticks/success/reward/物体终位）。"
                       "缺省跑到终态（控制器退出/世界 max_ticks）；ticks=N 只推进 N 步。"
                       "run 在后台线程执行：期间可以继续 get_observation（返回实时快照，"
                       "GUI 实时可视化的基础）；本条响应会等 run 结束才返回。"
                       "注意：run 返回时本轮控制器已收尾退出；下一段 run 会自动重新"
                       "拉起控制器并从当前 tick/qpos 继续（ticks 累积），reset 才回到"
                       "世界初态。同一时刻只允许一个 run。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticks": {"type": "integer", "minimum": 1,
                          "description": "本段推进的 tick 数上限"},
                "realtime": {"type": "boolean",
                             "description": "按墙钟实时节流（看动画用，1 仿真秒≈1 墙钟秒）；"
                                            "缺省 false 全速推进"},
            },
        },
    },
    {
        "name": "stop",
        "description": "请求停止当前进行中的 run（GUI 停止按钮的同一路径）。"
                       "立即返回；run 线程在下个控制 tick 提前收尾——控制器收 stop"
                       " 正常退出（不算崩溃），已跑轨迹保持，run 的响应照常"
                       " 写回且 stats 带 stopped=true。没有 run 在跑时无操作。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "teleop_start",
        "description": "开启遥操作（固定基座机械臂，>=6 关节）：重建一个不启动控制器"
                       "进程的干净世界，后台按实时推进；用 teleop_cmd 下发速度指令，"
                       "camera_stream/camera_frame 边看相机画面边操作。"
                       "传 record_dir 时同时录制 episode（data.npz + meta.json；"
                       "record_images=true 另存 frames/*.png），停止时自动写盘并"
                       "在 stats.recording 里返回路径/帧数。"
                       "响应与 run 一样延迟到停止后返回；用 stop 结束。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "robot": {"type": "string",
                          "description": "遥操作哪台机器人（缺省第一台）"},
                "site": {"type": "string",
                         "description": "TCP site（短名或 <robot>/<name>；缺省自动找"
                                        " gripper_tcp / attachment_site）"},
                "camera": {"type": "string",
                           "description": "实时画面相机名（短名或 <robot>/<name>）；"
                                          "缺省自由机位"},
                "width": {"type": "integer", "minimum": 64, "maximum": 1280,
                          "description": "实时画面宽度（缺省 320）"},
                "height": {"type": "integer", "minimum": 48, "maximum": 960,
                           "description": "实时画面高度（缺省 240）"},
                "fps": {"type": "integer", "minimum": 1, "maximum": 60,
                        "description": "实时画面帧率（缺省 15）"},
                "depth": {"type": "boolean",
                          "description": "实时画面输出深度伪彩（近红远蓝）"},
                "near": {"type": "number",
                         "description": "深度渲染近裁剪面（米，缺省 0.01）"},
                "far": {"type": "number",
                        "description": "深度渲染远裁剪面（米，缺省 50）"},
                "record_dir": {"type": "string",
                               "description": "episode 输出目录；缺省不录制"},
                "record_images": {"type": "boolean",
                                  "description": "是否保存相机帧 PNG（缺省 true）"},
                "record_width": {"type": "integer",
                                 "description": "录制帧宽（缺省同 width）"},
                "record_height": {"type": "integer",
                                  "description": "录制帧高（缺省同 height）"},
                "image_stride": {"type": "integer", "minimum": 1,
                                 "description": "每 N 个控制 tick 存一帧（缺省 1）"},
                "max_ticks": {"type": "integer", "minimum": 1,
                              "description": "遥操作 tick 上限（缺省 100000，通常用 stop）"},
            },
        },
    },
    {
        "name": "teleop_cmd",
        "description": "遥操作速度指令（持续状态）：vx/vy/vz 平移 -1..1（按机器人基座"
                       "坐标系，m/s 量级）、wz 绕 z 偏航 -1..1、grip 0=闭合/1=张开、"
                       "fast=true 加速。松开按键必须显式发 0 才停；release=true 清空"
                       "平移/旋转速度（夹爪保持）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vx": {"type": "number", "minimum": -1, "maximum": 1},
                "vy": {"type": "number", "minimum": -1, "maximum": 1},
                "vz": {"type": "number", "minimum": -1, "maximum": 1},
                "wz": {"type": "number", "minimum": -1, "maximum": 1},
                "grip": {"type": "number", "minimum": 0, "maximum": 1},
                "fast": {"type": "boolean"},
                "release": {"type": "boolean"},
            },
        },
    },
    {
        "name": "camera_stream",
        "description": "开启/关闭实时相机流（遥操作时边看边操作）：按 fps 在推进线程"
                       "渲染到缓存，camera_frame 取最新一帧。camera 传名字=开启，"
                       "缺省 null=关闭；自由机位传 camera=''。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": ["string", "null"],
                           "description": "相机名；null 关闭，'' 自由机位"},
                "width": {"type": "integer", "minimum": 64, "maximum": 1280,
                          "description": "宽度（缺省 320）"},
                "height": {"type": "integer", "minimum": 48, "maximum": 960,
                           "description": "高度（缺省 240）"},
                "fps": {"type": "integer", "minimum": 1, "maximum": 60,
                        "description": "帧率（缺省 15）"},
                "depth": {"type": "boolean",
                          "description": "深度伪彩模式（近红远蓝）"},
                "near": {"type": "number", "description": "深度近裁剪面（米）"},
                "far": {"type": "number", "description": "深度远裁剪面（米）"},
            },
        },
    },
    {
        "name": "camera_frame",
        "description": "取相机流最新一帧（image content，base64 PNG）。还没有帧时"
                       "返回 pending 提示（先 camera_stream 开启或等一帧）。"
                       "depth/near/far 可在取帧时切换深度伪彩与裁剪面。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "depth": {"type": "boolean",
                          "description": "切换深度伪彩模式（缺省沿用流设置）"},
                "near": {"type": "number", "description": "深度近裁剪面（米）"},
                "far": {"type": "number", "description": "深度远裁剪面（米）"},
            },
        },
    },
    {
        "name": "get_observation",
        "description": "当前完整观察量：各机器人 qpos/qvel/ctrl/body 位置 + "
                       "场景物体位置 + 任务状态（success/reward）。"
                       "带传感器的机器人另含 sensors 读数。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "screenshot",
        "description": "把当前仿真状态渲染成 PNG 截图。缺省自由机位（与 GUI 默认"
                       "视角一致）；传 camera 用声明过的相机视角（名字见 "
                       "get_robot_manifest 的 cameras）。缺省返回 image content"
                       "（base64 PNG）；传 path 则写盘并返回文件信息。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "minimum": 16, "maximum": 1920,
                           "description": "像宽，缺省 800"},
                "height": {"type": "integer", "minimum": 16, "maximum": 1920,
                            "description": "像高，缺省 600"},
                "camera": {"type": "string",
                            "description": "相机名（manifest 里的名字）；缺省自由机位"},
                "path": {"type": "string",
                          "description": "PNG 写盘路径；缺省内联 image content"},
            },
        },
    },
    {
        "name": "get_robot_manifest",
        "description": "机器人设备清单：执行器名字与控制范围、关节类型与限位、"
                       "body 名（带前缀的全名与短名都在）、传感器清单"
                       "（名字/类型/维度）与相机名（screenshot 的 camera 参数用）。"
                       "写控制器前先看这个。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_state",
        "description": "存内存快照（tag），与逐位重现测试同一路径。",
        "inputSchema": {
            "type": "object",
            "required": ["tag"],
            "properties": {"tag": {"type": "string",
                                   "description": "快照名（不可用 '__' 前缀）"}},
        },
    },
    {
        "name": "load_state",
        "description": "读内存快照，物理状态逐位恢复。注意：已退出的控制器进程"
                       "不会复活，需要重跑请用 reset。",
        "inputSchema": {
            "type": "object",
            "required": ["tag"],
            "properties": {"tag": {"type": "string"}},
        },
    },
    {
        "name": "reset",
        "description": "世界回到 YAML 初态：重编译 + 控制器进程重启。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_world",
        "description": "回读当前世界的 YAML 原文（Agent 修改世界前先看）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_world",
        "description": "把当前世界（含 GUI 拖动/位姿/装备等全部修改）保存为 YAML "
                       "文件。缺省保存回最近一次 load_world(yaml_path) 的文件；"
                       "内存世界无来源文件时必须显式给 path。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "目标 YAML 路径（缺省 = 最近一次"
                                        "load_world 的 yaml_path）"},
            },
        },
    },
    {
        "name": "zoo_create",
        "description": "从模型库（MuJoCo Menagerie 官方模型）生成可编辑项目："
                       "世界 YAML + 控制器模板（含执行器清单与 API 注释），"
                       "生成后 load_world 打开、改控制律即编程。"
                       "模型资产随仓库自带（models/zoo/），离线即用、无需下载。"
                       "key 列表见返回的 models 字段，或传 key='list' 只列清单。"
                       "demo=true 时另生成 demo.py 运动演示（四足小跑/臂扫掠/"
                       "人形摆臂）且世界指向它——load_world + run 即可看动画。",
        "inputSchema": {
            "type": "object",
            "required": ["key"],
            "properties": {
                "key": {"type": "string",
                        "description": "模型 key（如 unitree_go2）或 'list'"},
                "out_dir": {"type": "string",
                             "description": "输出目录，缺省 ~/.simovia/zoo/<key>"},
                "ensure_only": {"type": "boolean",
                                 "description": "不生成项目，只确保模型已缓存，"
                                                "返回 xml 路径与建议初始 pos "
                                                "（GUI 拖放添加机器人用）"},
                "demo": {"type": "boolean",
                          "description": "生成运动演示（demo.py）：四足对角小跑、"
                                         "机械臂/灵巧手关节扫掠、人形站立摆臂；"
                                         "世界指向 demo.py，load_world+run 即看动画"},
            },
        },
    },
]


# ---------------------------------------------------------------- 工具实现

class SimoviaSession:
    """MCP 工具的内核侧状态：当前世界 + 当前仿真。"""

    def __init__(self):
        self.spec = None
        self.sim: Simulation | None = None
        self.yaml_text: str | None = None
        self.world_path: str | None = None     # 最近一次 load_world 的来源文件
        # screenshot 渲染器按 (w, h) 缓存；Renderer 绑定 model，
        # load_world/reset 经 close() 一并销毁重建
        self._renderer: "mujoco.Renderer | None" = None
        self._renderer_wh: tuple[int, int] | None = None
        # run 后台线程化（GUI 实时观测的基础）
        self._run_thread: threading.Thread | None = None
        self._snap = None                 # run 期间最新观测快照（run 线程写）
        self._snap_t = 0.0
        self._snap_lock = threading.Lock()
        self._run_req = None              # (req_id, writer)：run 结束后写延迟响应
        self._run_error: BaseException | None = None
        self._last_stats: dict | None = None
        self._stop_event = threading.Event()   # stop 工具 -> run 线程提前收尾
        # 遥操作：无控制器进程，run 线程每 tick 调 ArmTeleop.step 写 ctrl
        self._teleop: ArmTeleop | None = None
        self._manual_sim = False          # 当前 sim 是不起控制器的遥操作实例
        # 录制（遥操作采集）：每个控制 tick 采样，run 结束统一写盘
        self._recorder = None
        self._rec_tick = -1
        self._record_error: str | None = None
        self._record_failed = False       # 采样出错：停止再采样，收尾仍写盘
        # 实时相机流：推进线程按 fps 渲染到 _cam_frame，camera_frame 取最新帧
        self._cam_stream: dict | None = None
        self._cam_frame: tuple[int, bytes] | None = None
        self._cam_t = 0.0
        self._cam_lock = threading.Lock()
        self._cam_error: str | None = None    # 渲染失败原因（无 EGL 等）
        self._cam_renderer: "mujoco.Renderer | None" = None
        self._cam_renderer_wh: tuple[int, int] | None = None
        self._cam_renderer_lock = threading.Lock()

    # ------------------------------------------------------------ 工具入口

    def call_tool(self, name: str, args: dict):
        """执行工具，返回可直接 JSON 化的结果。WorldError 原样抛给协议层。"""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise WorldError(f"未知工具 '{name}'，可用: "
                             f"{sorted(t['name'] for t in _TOOLS)}")
        return handler(args or {})

    @property
    def is_running(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    def close(self):
        # run 还在跑（如 realtime 世界）：先请求停止（下个 tick 收尾），
        # 给它一点时间等不住就随进程退出
        if self.is_running and threading.current_thread() is not self._run_thread:
            self._stop_event.set()
            self._run_thread.join(timeout=5)
        self._finish_recording(None)          # 未收尾的录制先落盘
        self._teleop = None
        self._manual_sim = False
        self._cam_stream = None
        self._cam_error = None
        self._close_camera_renderer()
        with self._cam_lock:
            self._cam_frame = None
        if self._renderer is not None:   # Renderer 绑定的 model 随 sim 一起失效
            self._renderer.close()
            self._renderer, self._renderer_wh = None, None
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    # ------------------------------------------------------------ 各工具

    def _tool_load_world(self, args: dict):
        self._require_idle()
        path, text = args.get("yaml_path"), args.get("yaml_text")
        if (path is None) == (text is None):
            raise WorldError("load_world 需要 yaml_path 或 yaml_text 恰好一个")
        if path is not None:
            spec = _load_world_file(path)          # base_dir = 文件所在目录
            # 与 load_world 同一套编码兜底：GBK 世界文件在上一行能过，
            # 这里若裸读 utf-8 会炸 UnicodeDecodeError（中文 Windows）
            yaml_text = read_text_compat(path)
            new_path = str(pathlib.Path(path).resolve())
        else:
            try:
                d = yaml.safe_load(text)
            except yaml.YAMLError as e:
                raise WorldError(f"YAML 语法错误: {e}") from e
            resolve_zoo_models(d)            # zoo:// -> menagerie 绝对路径
            spec = parse_world(d)
            base = args.get("base_dir") or "."
            spec.base_dir = pathlib.Path(base).resolve()
            yaml_text = text
            new_path = None                      # 内存世界（无来源文件）

        # 先创建新仿真、成功后再替换旧世界：新模型缺 mesh/关节引用错误/
        # 控制器路径不存在等编译失败时，旧世界仍然完整可用（GUI 误点一个
        # 坏文件不至于把当前场景一起关掉）。
        new_sim = Simulation(spec)
        self.close()
        self.spec, self.yaml_text = spec, yaml_text
        # 来源文件跟踪：yaml_text 重载（GUI 拖动/位姿提交）不动它——
        # 那些修改的目标仍是原文件；只有换世界才更新
        if new_path is not None:
            self.world_path = new_path
        self.sim = new_sim
        return {
            "world": spec.name,
            "objects": [o.name for o in spec.scene.objects],
            "robots": [r.name for r in spec.robots],
            "task_max_ticks": spec.task.max_ticks,
        }

    def _tool_run(self, args: dict):
        """会话层同步语义：起后台线程并等它跑完（旧 API 行为不变）。

        stdio 协议层（dispatch）走 start_run(block=False)：响应推迟到
        run 结束，期间 serve 循环继续应答 get_observation 等请求。
        """
        return self.start_run(args, block=True)

    def _tool_stop(self, args: dict):
        """请求停止进行中的 run。立即返回，收尾在下个控制 tick 完成。"""
        if not self.is_running:
            return {"stopped": False, "note": "没有进行中的 run，无需停止"}
        self._stop_event.set()
        return {"stopped": True,
                "note": "停止请求已发出：run 线程将在下个控制 tick 收尾，"
                        "其响应照常写回且带 stopped=true"}

    # ------------------------------------------------------------ 遥操作

    def _rebuild_sim(self, autostart_controllers: bool = True) -> Simulation:
        """按当前 spec 重建仿真（控制器进程起/不起由参数决定）。"""
        if self.spec is None:
            raise WorldError("还没有加载世界（先 load_world）")
        self.close()                 # 顺带收尾未落盘的录制、关渲染器
        self.sim = Simulation(self.spec,
                              autostart_controllers=autostart_controllers)
        self._manual_sim = not autostart_controllers
        return self.sim

    def start_teleop(self, args: dict, req_id=None, writer=None):
        """开启遥操作：不起控制器进程，run 线程每 tick 调 ArmTeleop.step。

        响应与 run 一样延迟到停止后返回（stats 带 recording/teleop）。
        参数校验失败（机器人/相机/关节数不足）在同步阶段抛 WorldError。
        """
        if self.is_running:
            raise WorldError("已有 run/遥操作在进行中，先 stop 再开新的")
        self._require_sim()
        sim = self._rebuild_sim(autostart_controllers=False)
        robot = args.get("robot")
        if robot:
            if robot not in sim.index.robots:
                raise WorldError(
                    f"机器人 '{robot}' 不存在，可用: {sorted(sim.index.robots)}")
        else:
            robot = next(iter(sim.index.robots), None)
            if robot is None:
                raise WorldError("世界没有机器人，无法遥操作")
        try:
            self._teleop = ArmTeleop(sim.model, sim.data, sim.index, self.spec,
                                     robot, site=args.get("site"))
            # 实时画面：不传 camera 也给自由机位（边看边操作是默认诉求）
            self._start_camera_stream(
                sim, args.get("camera"),
                int(args.get("width") or 320),
                int(args.get("height") or 240),
                int(args.get("fps") or 15),
                depth=bool(args.get("depth") or False),
                near=args.get("near"), far=args.get("far"))
            rec = self._start_recording(args)
        except BaseException:
            self._rebuild_sim(autostart_controllers=True)   # 失败回正常世界
            raise

        max_ticks = int(args.get("max_ticks") or 100_000)
        self._run_error = None
        self._run_req = (req_id, writer) if req_id is not None else None
        self._snap = None
        self._snap_t = 0.0
        self._stop_event.clear()

        def job():
            try:
                stats = self._teleop_job(max_ticks)
            except BaseException as e:
                self._run_error = e
                # 后台线程异常不能只塞进响应：stderr 才是 GUI 控制台/日志
                # 能看到的地方（CI 上也靠它定位）
                traceback.print_exc()
                stats = {"error": f"{type(e).__name__}: {e}"}
            finally:
                self._close_camera_renderer()
            self._write_run_response(stats)

        self._run_thread = threading.Thread(target=job, daemon=True,
                                            name="simovia-teleop")
        self._run_thread.start()
        info = self._teleop.describe()
        info.update({
            "realtime": True,
            "camera": (self._cam_stream or {}).get("camera"),
            "recording": str(rec) if rec is not None else None,
        })
        return _DEFERRED if req_id is not None else info

    def _teleop_job(self, max_ticks: int):
        sim = self.sim
        teleop = self._teleop
        stats = dict(sim.run(max_ticks=max_ticks, on_tick=self._on_tick,
                             on_sample=self._on_sample, realtime=True,
                             stop_check=self._stop_event.is_set,
                             ctrl_fn=teleop.step))
        if self._stop_event.is_set():
            stats["stopped"] = True
        stats["objects"] = self._objects_pos(sim)
        extra = {"source": "teleop", "teleop": teleop.describe(),
                 "pos_error": round(float(teleop.last_pos_error), 4)}
        rec = self._finish_recording(stats, extra=extra)
        if rec is not None:
            stats["recording"] = rec
        self._last_stats = stats
        return stats

    def _tool_teleop_cmd(self, args: dict):
        """遥操作速度指令（持续状态；release=true 清空速度、夹爪保持）。"""
        teleop = self._teleop
        if teleop is None:
            raise WorldError("没有进行中的遥操作（先 teleop_start）")
        if args.get("release"):
            teleop.release()
            return {"cmd": teleop.set_command({}), "released": True}
        return {"cmd": teleop.set_command(args),
                "pos_error": round(float(teleop.last_pos_error), 4)}

    # ------------------------------------------------------------ 实时相机

    def _tool_camera_stream(self, args: dict):
        """开启/关闭实时相机流（camera=null 关闭，'' 自由机位）。"""
        camera = args.get("camera")
        if camera is None:
            self._cam_stream = None
            self._cam_error = None
            self._close_camera_renderer()
            with self._cam_lock:
                self._cam_frame = None
            return {"streaming": False}
        sim = self._require_sim()
        self._start_camera_stream(
            sim, camera,
            int(args.get("width") or 320),
            int(args.get("height") or 240),
            int(args.get("fps") or 15),
            depth=bool(args.get("depth") or False),
            near=args.get("near"), far=args.get("far"))
        return {"streaming": True,
                "camera": self._cam_stream["camera"] or "",
                "width": self._cam_stream["width"],
                "height": self._cam_stream["height"],
                "fps": self._cam_stream["fps"],
                "depth": self._cam_stream["depth"],
                "near": self._cam_stream["near"],
                "far": self._cam_stream["far"]}

    def _tool_camera_frame(self, args: dict):
        """取相机流最新一帧（image content）。可临时切换深度伪彩/裁剪面。"""
        cfg = self._cam_stream
        if cfg is not None:
            if args.get("depth") is not None:
                cfg["depth"] = bool(args["depth"])
            if args.get("near") is not None:
                cfg["near"] = float(args["near"])
            if args.get("far") is not None:
                cfg["far"] = float(args["far"])
        with self._cam_lock:
            frame = self._cam_frame
            err = self._cam_error
        if cfg is not None and not cfg.get("depth") and self.sim is not None:
            annotated=self.sim.controller_frames.get(cfg.get("camera"))
            if annotated is not None and self.sim.tick-annotated[0]<=25:
                frame=annotated
        if frame is None:
            return {"pending": True, "error": err,
                    "note": "还没有帧：先 camera_stream 开启相机，"
                            "或等推进线程渲染出一帧"}
        _, png = frame
        return _RawContent([{
            "type": "image",
            "data": base64.b64encode(png).decode("ascii"),
            "mimeType": "image/png",
        }])

    def _start_camera_stream(self, sim: Simulation, camera, width: int,
                             height: int, fps: int, depth: bool = False,
                             near=None, far=None) -> None:
        if not (64 <= width <= 1280 and 48 <= height <= 960):
            raise WorldError(f"画面尺寸需在 64x48..1280x960 内，"
                             f"实际: {width}x{height}")
        if not (1 <= fps <= 60):
            raise WorldError(f"fps 需在 1..60 内，实际: {fps}")
        near = float(near) if near is not None else 0.01
        far = float(far) if far is not None else 50.0
        if near <= 0 or far <= near:
            raise WorldError(
                f"深度裁剪面需要 0 < near < far，实际 near={near}, far={far}")
        name = None
        if camera not in (None, ""):
            name = self._resolve_camera(sim, str(camera))
        self._close_camera_renderer()
        with self._cam_lock:
            self._cam_frame = None
            self._cam_error = None
        self._cam_stream = {"camera": name, "width": int(width),
                            "height": int(height), "fps": int(fps), "seq": 0,
                            "depth": bool(depth), "near": near, "far": far}
        self._cam_t = 0.0
        if not self.is_running:              # 空闲：立刻出一帧预览
            try:
                self._capture_camera(sim, self._cam_stream)
            except Exception as e:
                with self._cam_lock:
                    self._cam_error = f"{type(e).__name__}: {e}"

    def _capture_camera(self, sim: Simulation, cfg: dict) -> None:
        """推进线程内渲染一帧到缓存（调用方保证此刻 data 一致）。"""
        w, h = cfg["width"], cfg["height"]
        free_cam = None
        if cfg["camera"] is None and self._teleop is not None:
            # 自由机位但在遥操作：跟着末端 TCP，边看边操作才有意义
            free_cam = mujoco.MjvCamera()
            free_cam.lookat[:] = sim.data.site_xpos[self._teleop.site_id]
            free_cam.distance = 0.9
            free_cam.azimuth = 135.0
            free_cam.elevation = -25.0
        with self._cam_renderer_lock:
            if self._cam_renderer is None or self._cam_renderer_wh != (w, h):
                if self._cam_renderer is not None:
                    self._cam_renderer.close()
                self._cam_renderer = mujoco.Renderer(sim.model, h, w)
                self._cam_renderer_wh = (w, h)
            if cfg.get("depth"):
                depth = render_offscreen(
                    sim.model, sim.data, w, h, cfg["camera"],
                    renderer=self._cam_renderer, free_cam=free_cam,
                    depth=True, near=cfg.get("near"), far=cfg.get("far"))
                rgb = depth_to_rgb(depth, near=cfg.get("near") or 0.01,
                                   far=cfg.get("far") or 50.0)
            else:
                rgb = render_offscreen(sim.model, sim.data, w, h,
                                       cfg["camera"],
                                       renderer=self._cam_renderer,
                                       free_cam=free_cam)
        png = encode_png(rgb)
        cfg["seq"] = cfg.get("seq", 0) + 1
        with self._cam_lock:
            self._cam_frame = (cfg["seq"], png)
            self._cam_error = None

    def _close_camera_renderer(self) -> None:
        with self._cam_renderer_lock:
            if self._cam_renderer is not None:
                self._cam_renderer.close()
                self._cam_renderer, self._cam_renderer_wh = None, None

    # ------------------------------------------------------------ 录制

    def _start_recording(self, args: dict):
        """record_dir 存在则起 EpisodeRecorder；返回输出目录（未开录制返回 None）。"""
        out = args.get("record_dir")
        if not out:
            return None
        from simovia.recorder import EpisodeRecorder
        sim = self._require_sim()
        camera = args.get("camera")
        if camera not in (None, ""):
            camera = self._resolve_camera(sim, str(camera))
        else:
            camera = None
        out_dir = _unique_dir(pathlib.Path(out).expanduser())
        self._record_error = None
        self._record_failed = False
        self._rec_tick = -1
        self._recorder = EpisodeRecorder(
            self.spec, sim.model, sim.index, out_dir,
            images=bool(args.get("record_images", True)), camera=camera,
            width=int(args.get("record_width") or args.get("width") or 320),
            height=int(args.get("record_height") or args.get("height") or 240),
            image_stride=int(args.get("image_stride") or 1))
        return out_dir

    def _finish_recording(self, stats, extra: dict | None = None):
        """收尾录制并写盘；返回摘要（没录制返回 None）。异常不抛出。"""
        rec = self._recorder
        if rec is None:
            return None
        self._recorder = None
        err = self._record_error
        self._record_error = None
        self._record_failed = False
        extra = dict(extra or {})
        if err:
            # 采样中途出错也照常收尾：已有轨迹写盘并在 meta 标记不完整
            extra.update({"incomplete": True, "record_error": err})
        try:
            out = rec.finish(stats=stats, extra=extra)
            return {"dir": str(out), "samples": rec.sample_count,
                    "images": rec.frame_count, "camera": rec.camera,
                    "incomplete": bool(err or rec.errors),
                    "error": err}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def start_run(self, args: dict, req_id=None, writer=None, block: bool = False):
        """启动后台 run。返回 _DEFERRED（异步）或 stats dict（block=True）。

        WorldError（未加载世界/并发 run）同步抛出。
        """
        if self.is_running:
            raise WorldError("已有 run 在进行中，等它结束（或完成后 reset）")
        self._require_sim()
        if self._manual_sim:
            # 遥操作结束后是"不起控制器"的实例：普通 run 恢复多进程控制器
            self._rebuild_sim(autostart_controllers=True)
        sim = self._require_sim()
        # 上一轮 run 收尾时控制器已正常退出：继续推进（GUI 步进累积/暂停
        # 续跑）需要重新拉起控制器，qpos/tick 与场景状态保持不变。
        sim.ensure_controllers()
        ticks = args.get("ticks")
        max_ticks = (sim.tick + int(ticks)) if ticks is not None else None
        realtime = bool(args.get("realtime", False))
        self._run_error = None
        self._run_req = (req_id, writer) if req_id is not None else None
        self._snap = None
        self._snap_t = 0.0
        self._stop_event.clear()          # 新一轮 run：清掉上一轮的停止请求

        def job():
            try:
                stats = self._run_job(max_ticks, realtime)
            except BaseException as e:      # 线程里异常不能无声吞掉
                self._run_error = e
                traceback.print_exc()
                stats = {"error": f"{type(e).__name__}: {e}"}
            self._write_run_response(stats)

        self._run_thread = threading.Thread(target=job, daemon=True,
                                            name="simovia-run")
        self._run_thread.start()
        if block:
            self._run_thread.join()
            if self._run_error is not None:
                raise self._run_error
            return self._last_stats
        return _DEFERRED

    def _run_job(self, max_ticks, realtime=False):
        sim = self.sim               # 线程里重取（load_world 会换实例）
        stats = dict(sim.run(max_ticks=max_ticks, on_tick=self._on_tick,
                             on_sample=self._on_sample if realtime else None,
                             realtime=realtime,
                             stop_check=self._stop_event.is_set))
        if self._stop_event.is_set():
            stats["stopped"] = True       # 提前收尾（GUI 停止按钮/stop 工具）
        stats["objects"] = self._objects_pos(sim)
        rec = self._finish_recording(stats)
        if rec is not None:
            stats["recording"] = rec
        self._last_stats = stats
        return stats

    def _on_tick(self):
        """每个控制 tick 推进完成后的回调（录制 + 快照 + 相机帧）。"""
        self._tick_body(record=True)

    def _on_sample(self):
        """实时模式的物理子步中间帧：只出画面/快照，不重复录制。"""
        self._tick_body(record=False)

    def _tick_body(self, record: bool):
        """run 线程内回调（data 一致，可安全读）。

        录制与相机流必须每 tick 判断（不能走 10ms 快照节流）：录制按控制
        tick 采样，相机流按自己的 fps 节流。
        """
        sim = self.sim
        if sim is None:
            return
        if record and self._recorder is not None \
                and not self._record_failed and sim.tick != self._rec_tick:
            self._rec_tick = sim.tick
            try:
                self._recorder.add(sim.tick, sim.data)
            except Exception as e:        # 录坏一帧不能拖垮整个 run
                # 不丢弃 recorder：已缓存的轨迹在收尾时照常写盘并标记不完整
                self._record_error = f"{type(e).__name__}: {e}"
                self._record_failed = True
        cfg = self._cam_stream
        if cfg is not None:
            now_c = time.perf_counter()
            if now_c - self._cam_t >= 1.0 / cfg["fps"]:
                self._cam_t = now_c
                try:
                    self._capture_camera(sim, cfg)
                except Exception as e:     # 渲染失败（无 EGL 等）不影响推进
                    with self._cam_lock:
                        self._cam_error = f"{type(e).__name__}: {e}"
        now = time.perf_counter()
        if now - self._snap_t < _SNAP_INTERVAL:
            return
        self._snap_t = now
        with self._snap_lock:
            self._snap = self._obs_payload()

    def _write_run_response(self, stats):
        if self._run_req is not None:
            req_id, writer = self._run_req
            self._run_req = None
            if writer is not None:
                is_err = "error" in stats
                writer.write(_rpc_result(req_id, {
                    "content": [{"type": "text",
                                 "text": json.dumps(stats, ensure_ascii=False)}],
                    "isError": is_err}))

    def _objects_pos(self, sim) -> dict:
        return {name: [round(float(x), 4) for x in sim.data.xpos[bid]]
                for name, bid in sim.index.objects.items()}

    def _obs_payload(self) -> dict:
        """完整观测（含物体 pos+quat）。调用方保证此刻没有并发 mj_step。"""
        sim = self.sim
        # xpos/xquat 是派生量：刚 load（还没跑过 step）或 load_state 恢复后
        # 都是过期/零值。mj_kinematics 只重算运动学，不碰 warmstart，
        # 对 bit 级轨迹重现无影响。
        mujoco.mj_kinematics(sim.model, sim.data)
        robots = {}
        for rname in sim.index.robots:
            obs = sim._obs_msg(rname)
            obs.pop("op", None)
            robots[rname] = obs
        objects = {}
        for name, bid in sim.index.objects.items():
            objects[name] = {
                "pos": [round(float(x), 4) for x in sim.data.xpos[bid]],
                "quat": [round(float(x), 4) for x in sim.data.xquat[bid]],
            }
        return {
            "tick": sim.tick,
            "time": round(float(sim.data.time), 4),
            "running": self.is_running,
            "robots": robots,
            "objects": objects,
            "task": {"success": sim._task_success(),
                     "reward": round(sim._task_reward(), 4)},
        }

    def _tool_get_observation(self, args: dict):
        sim = self._require_sim()
        if self.is_running:
            # run 期间 data 正被 run 线程 mj_step，直接读有竞态——返回快照
            with self._snap_lock:
                snap = self._snap
            if snap is not None:
                return snap
            return {"running": True, "tick": sim.tick,
                    "note": "首帧快照未生成（run 刚启动）"}
        return self._obs_payload()

    def _tool_get_robot_manifest(self, args: dict):
        sim = self._require_sim()
        m = sim.model
        robots = []
        for rd in self.spec.robots:
            ridx = sim.index.robots[rd.name]
            actuators = []
            for a_name, aid in ridx.actuators.items():
                lo, hi = m.actuator_ctrlrange[aid]
                actuators.append({"name": a_name,
                                  "ctrlrange": [float(lo), float(hi)]})
            joints = []
            for j_name, jid in ridx.joints.items():
                joints.append({
                    "name": j_name,
                    "type": _JTYPE_NAME.get(int(m.jnt_type[jid]), "?"),
                    "limited": bool(m.jnt_limited[jid]),
                    "range": [float(x) for x in m.jnt_range[jid]],
                })
            # sites（挂点）：扫描编译模型，按机器人前缀过滤；夹爪内 site
            # 全名 "<robot>/gripper/..."，短名保留 gripper/ 前缀以便引用
            prefix = f"{rd.name}/"
            sites = []
            for sid in range(m.nsite):
                full = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid)
                if full and full.startswith(prefix):
                    sites.append(full[len(prefix):])
            sites.sort()

            # ---- 末端执行器：外挂（库内/自定义）+ 自带（灵巧手）统一列表 ----
            # source=library：YAML end_effectors 里显式装的；source=builtin：
            # 机器人模型自带的手（未装外挂时也在列，stripped 标记是否被删）。
            ees: list[dict] = []
            for idx, ee in enumerate(rd.end_effectors):
                side = ee.side or (_END_EFFECTORS.get(ee.model) or {}).get("side")
                info = None
                if ee.model in _END_EFFECTORS:
                    gi = _END_EFFECTORS[ee.model]
                    info = {"key": ee.model, "title": gi["title"]}
                    if len(rd.end_effectors) == 1:
                        prefix = "gripper/"
                    elif side:
                        prefix = f"gripper_{side}/"
                    else:
                        prefix = f"gripper_{idx + 1}/"
                    if "actuator" in gi:       # 单执行器（平行夹爪）
                        info.update({
                            "actuator": f"{prefix}{gi['actuator']}",
                            "open": gi["open"], "close": gi["close"]})
                    else:                       # 多执行器（灵巧手）
                        info.update({
                            "actuators": [f"{prefix}{a}"
                                          for a in gi["actuators"]],
                            "poses": gi["poses"]})
                ees.append({"source": "library", "key": ee.model,
                            "model": ee.model, "side": side,
                            "site": ee.site, "info": info})
            for g in self.sim.index.end_effectors.get(rd.name, []):
                side = ("left" if g["key"].startswith("left")
                        else "right" if g["key"].startswith("right") else None)
                ees.append({"source": "builtin", "key": g["key"], "side": side,
                            "site": g["site"], "anchor": g["anchor"],
                            "bodies": g["bodies"], "stripped": g["stripped"]})
            # 挂载位：左右手各一个（单臂为默认挂点），附当前已装的外挂条目
            mounts = []
            for slot in self.sim.index.mounts.get(rd.name, []):
                cur = next(
                    (e for e in ees if e["source"] == "library"
                     and (e["side"] == slot["side"]
                          or (slot["side"] is None
                              and len(rd.end_effectors) == 1))), None)
                mounts.append({**slot, "end_effector": cur})
            # 旧字段兼容：gripper = 第一个外挂末端执行器
            first = next((e for e in ees if e["source"] == "library"), None)
            gripper = (None if first is None else
                       {"model": first["model"], "site": first["site"],
                        "info": first["info"]})

            sensors = []
            by_name = {s.name: s for s in rd.sensors}
            for six in ridx.sensors.values():
                item = {"name": six.name, "type": six.kind, "dim": six.dim}
                s = by_name.get(six.name)
                if s is not None and s.type == "camera":
                    item.update({
                        "body": s.body, "pos": list(s.pos),
                        "quat": list(s.quat) if s.quat is not None else None,
                        "fovy": s.fovy, "depth": s.depth,
                        "near": s.near, "far": s.far,
                    })
                sensors.append(item)
            # 相机在 MuJoCo 里不是 <sensor>（不进 ridx.sensors），但装备页的
            # 「相机参数」需要它们：YAML 声明的 camera 单独补进 sensors 列表
            for s in rd.sensors:
                if s.type != "camera":
                    continue
                sensors.append({
                    "name": s.name, "type": "camera", "dim": 0,
                    "body": s.body, "pos": list(s.pos),
                    "quat": list(s.quat) if s.quat is not None else None,
                    "fovy": s.fovy, "depth": s.depth,
                    "near": s.near, "far": s.far,
                })
            robots.append({
                "name": rd.name,
                "controller": rd.controller,
                "args": rd.args,
                "actuators": actuators,
                "joints": joints,
                "sites": sites,
                "bodies": {f"{rd.name}/{short}": short for short in ridx.bodies},
                "sensors": sensors,
                "cameras": list(ridx.cameras),
                "gripper": gripper,          # 旧字段兼容（第一个外挂条目）
                "end_effectors": ees,        # 统一列表：library + builtin
                "builtin_end_effectors": self.sim.index.end_effectors.get(
                    rd.name, []),
                "mounts": mounts,            # 左右手挂载位
                "strip_hands": rd.strip_hands,
            })
        catalog = [
            {"key": k, "title": v["title"], "cat": v["cat"],
             # 单执行器条目照旧给 open/close；多执行器给 actuators/poses
             **({"open": v["open"], "close": v["close"]}
                if "actuator" in v else
                {"actuators": list(v["actuators"]),
                 "poses": v["poses"]}),
             "supported_robots": sorted(v["mounts"])}
            for k, v in _END_EFFECTORS.items()]
        return {
            "world": self.spec.name,
            "timestep": self.spec.timestep,
            "ctrl_period": self.spec.ctrl_period,
            "robots": robots,
            "objects": [{"name": o.name, "type": o.type, "pos": list(o.pos)}
                        for o in self.spec.scene.objects],
            "end_effector_catalog": catalog,
            "gripper_catalog": catalog,      # 旧字段兼容
        }

    def _tool_save_state(self, args: dict):
        self._require_idle()
        sim = self._require_sim()
        tag = args.get("tag")
        if not tag:
            raise WorldError("save_state 缺少必填参数 tag")
        sim.save_state(tag)
        return {"tag": tag, "tick": sim.tick,
                "tags": supervisor.list_tags(sim._snaps)}

    def _tool_load_state(self, args: dict):
        self._require_idle()
        sim = self._require_sim()
        tag = args.get("tag")
        if not tag:
            raise WorldError("load_state 缺少必填参数 tag")
        sim.load_state(tag)
        return {"tag": tag, "tick": sim.tick,
                "tags": supervisor.list_tags(sim._snaps)}

    def _tool_reset(self, args: dict):
        self._require_idle()
        if self.spec is None:
            raise WorldError("还没有加载世界（先 load_world）")
        self.close()
        self.sim = Simulation(self.spec)     # 重编译 + 控制器进程重启
        return {"world": self.spec.name, "tick": 0}

    def _tool_read_world(self, args: dict):
        if self.yaml_text is None:
            raise WorldError("还没有加载世界（先 load_world）")
        return self.yaml_text

    def _tool_save_world(self, args: dict):
        """当前世界 YAML 落盘（GUI"保存世界"与 Agent 复用世界都靠它）。

        保存到与来源不同的目录时，robots[].model/controller 的相对路径
        会跟着新 base_dir 走而失效——统一改写为绝对路径，任何目录重开都行。
        """
        if self.yaml_text is None:
            raise WorldError("还没有加载世界（先 load_world）")
        path = args.get("path") or self.world_path
        if not path:
            raise WorldError("当前世界来自 yaml_text（无来源文件），"
                              "请显式传 path")
        p = pathlib.Path(path)
        if p.suffix.lower() not in (".yaml", ".yml"):
            p = p.with_suffix(".yaml")
        p.parent.mkdir(parents=True, exist_ok=True)

        text = self.yaml_text
        base = self.spec.base_dir if self.spec else None
        # 相对资源改绝对（model/controller）；另存到别的目录后重开也有效
        text = absolutize_world_paths(text, base)

        p.write_text(text, encoding="utf-8")
        return {"saved": str(p), "bytes": len(text.encode("utf-8")),
                "note": f"世界 '{self.spec.name}' 已保存"}

    def _tool_zoo_create(self, args: dict):
        """模型库生成：key='list' 列清单；ensure_only 只取模型路径；否则生成项目。"""
        from simovia.zoo import MODELS, ensure_model, generate
        key = str(args.get("key", ""))
        if key == "list":
            return {"models": [
                {"key": k, "title": m["title"], "category": m["cat"],
                 "desc": m["desc"]}
                for k, m in MODELS.items()]}
        if args.get("ensure_only"):
            # GUI 拖放添加机器人用：取模型 xml 路径与建议初始高度
            xml = ensure_model(key)             # 自带资产，离线即取
            return {"key": key, "title": MODELS[key]["title"],
                    "xml": str(xml), "pos": list(MODELS[key]["pos"])}
        out_dir = args.get("out_dir")
        r = generate(key, out_dir=out_dir, demo=bool(args.get("demo")))
        if "demo" in r:                        # 运动演示项目
            r["hint"] = ("已生成运动演示。load_world 打开 " + r["world"] +
                         "，run 即可看 " + MODELS[key]["title"] + " 的演示动作；"
                         "写自己的控制律用 " + r["controller"])
        else:
            r["hint"] = ("已生成。load_world 打开 " + r["world"] +
                         "，编辑 " + r["controller"] + " 写控制律后重新 run")
        return r

    def _tool_screenshot(self, args: dict):
        """渲染当前状态为 PNG：内联 image content 或写盘。"""
        self._require_idle()
        sim = self._require_sim()
        w = int(args.get("width") or 800)
        h = int(args.get("height") or 600)
        if not (16 <= w <= 1920 and 16 <= h <= 1920):
            raise WorldError(f"width/height 需在 16..1920 内，实际: {w}x{h}")
        camera = args.get("camera")
        if camera is not None:
            camera = self._resolve_camera(sim, str(camera))
        rend = self._renderer
        if rend is None or self._renderer_wh != (w, h):
            if rend is not None:
                rend.close()
            rend = mujoco.Renderer(sim.model, h, w)
            self._renderer, self._renderer_wh = rend, (w, h)
        rgb = render_offscreen(sim.model, sim.data, w, h, camera, renderer=rend)
        png = encode_png(rgb)
        path = args.get("path")
        if path:
            path = pathlib.Path(path).expanduser()
            path.write_bytes(png)
            return {"path": str(path), "width": w, "height": h, "bytes": len(png)}
        return _RawContent([{
            "type": "image",
            "data": base64.b64encode(png).decode("ascii"),
            "mimeType": "image/png",
        }])

    def _resolve_camera(self, sim: Simulation, name: str) -> str:
        """相机名解析：manifest 短名 -> 带机器人前缀全名；模型自带相机名直接用。"""
        m = sim.model
        shortnames, allnames = set(), set()
        for rname, ridx in sim.index.robots.items():
            for short in ridx.cameras:
                shortnames.add(short)
                allnames.add(short)
                allnames.add(f"{rname}/{short}")
        for cid in range(m.ncam):
            long = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            allnames.add(long)
        if name in shortnames:                      # 短名补前缀
            for rname, ridx in sim.index.robots.items():
                if name in ridx.cameras:
                    return f"{rname}/{name}"
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
            return name                             # 全名或模型自带名
        raise WorldError(f"相机 '{name}' 不存在，可用: {sorted(allnames)}")

    # ------------------------------------------------------------ 辅助

    def _require_idle(self):
        """run 进行中禁止换世界/动状态（否则 run 线程会用着已 close 的 sim）。"""
        if self.is_running:
            raise WorldError("run 正在进行中，此操作需要等它结束")

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise WorldError("还没有加载世界（先 load_world）")
        return self.sim


# ---------------------------------------------------------------- JSON-RPC / MCP 协议

def _unique_dir(p: pathlib.Path) -> pathlib.Path:
    """录制目录已存在且非空时加后缀，避免覆盖上一次采集的 episode。"""
    if not p.exists() or not any(p.iterdir()):
        return p
    for i in range(1, 1000):
        cand = p.with_name(f"{p.name}_{i:03d}")
        if not cand.exists():
            return cand
    raise WorldError(f"录制目录 {p} 已存在且找不到可用后缀目录")


class _Writer:
    """线程安全写出器：serve 线程与 run 线程共用 stdout。"""

    def __init__(self, out):
        self._out = out
        self._lock = threading.Lock()

    def write(self, obj) -> None:
        with self._lock:
            self._out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._out.flush()


def _rpc_result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def dispatch(session: SimoviaSession, method: str, params: dict, req_id,
             writer: _Writer | None = None):
    """单个 JSON-RPC 请求 -> 响应 dict（通知返回 None，延迟响应返回 _DEFERRED）。"""
    if req_id is None:            # 通知：静默接受
        return None
    if method == "initialize":
        version = (params or {}).get("protocolVersion") or _PROTOCOL
        return _rpc_result(req_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "simovia", "version": "0.1.0"},
        })
    if method == "ping":
        return _rpc_result(req_id, {})
    if method == "tools/list":
        return _rpc_result(req_id, {"tools": _TOOLS})
    if method == "tools/call":
        params = params or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        if name == "run":
            # run 后台线程化：响应推迟到跑完（GUI 实时观测的基础）。
            # 校验类错误（未加载世界/并发 run）仍同步返回。
            try:
                return session.start_run(args, req_id, writer, block=False)
            except WorldError as e:
                return _rpc_result(req_id, {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True})
        if name == "teleop_start":
            # 遥操作同样是延迟响应：停止后返回 stats（含录制摘要）
            try:
                return session.start_teleop(args, req_id, writer)
            except WorldError as e:
                return _rpc_result(req_id, {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True})
        try:
            result = session.call_tool(name, args)
        except WorldError as e:
            return _rpc_result(req_id, {
                "content": [{"type": "text", "text": str(e)}],
                "isError": True})
        if isinstance(result, _RawContent):      # screenshot：image content 透传
            return _rpc_result(req_id, dict(result))
        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False)
        return _rpc_result(req_id, {
            "content": [{"type": "text", "text": text}], "isError": False})
    return _rpc_error(req_id, -32601, f"未知方法: {method}")


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    """MCP stdio 主循环：每行一个 JSON-RPC 消息。

    stdio 固定 UTF-8：Windows 下管道默认走本地代码页（中文 cp936），而
    GUI/Agent 客户端按 UTF-8 收发，不重配会导致中文错误信息乱码或整行
    解析失败；Linux/macOS 本来就是 UTF-8，重配无副作用。
    """
    try:
        stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):     # 测试传入 StringIO 等：跳过
        pass
    try:
        stdout.reconfigure(encoding="utf-8", newline="\n")
    except (AttributeError, ValueError):
        pass
    session = SimoviaSession()
    writer = _Writer(stdout)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            resp = _rpc_error(None, -32700, f"解析错误: {e}")
        else:
            method = req.get("method", "") if isinstance(req, dict) else ""
            if not method:
                resp = _rpc_error(req.get("id") if isinstance(req, dict) else None,
                                  -32600, "非法请求：缺少 method")
            else:
                try:
                    resp = dispatch(session, method, req.get("params"),
                                    req.get("id"), writer)
                except Exception as e:      # 内部错误兜底：进程绝不能死
                    resp = _rpc_error(req.get("id"), -32603,
                                      f"内部错误: {type(e).__name__}: {e}")
        if resp is not None and resp is not _DEFERRED:
            writer.write(resp)
    session.close()


if __name__ == "__main__":
    serve()
