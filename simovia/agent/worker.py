"""AgentWorker：HTTP + 工具循环 QThread。

UI 非阻塞：
- send(user_text)：把消息塞队列，循环驱动 OpenAI 调用
- 工具调用：本地工具直发 dispatch_local，MCP 工具发 McpClient.call_tool
- 完成或异常：发 finished/completed 信号回 UI
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from PySide6.QtCore import QThread, Signal, Slot

from simovia.agent.bridge import (content_text, extract_tool_calls,
                                    local_tools_manifest, system_prompt,
                                    to_openai_tools, tool_payload)
from simovia.agent.config import AgentConfig
from simovia.agent.local_tools import LocalFiles, dispatch_local


class AgentWorker(QThread):
    """OpenAI 兼容 chat/completions + 工具桥接（非阻塞后台线程）。"""

    # 信号：用户消息 / 工具调用日志 / 出错 / 一轮对话结束
    assistantText = Signal(str)             # LLM 文本回复（无 tool_calls 时）
    toolCall = Signal(str, dict)            # (工具名, args)
    toolResult = Signal(str, object)        # (工具名, 结果 dict 或 错误字串)
    error = Signal(str)
    finishedRound = Signal()                # 一轮对话结束（无更多 tool_calls）

    #: 长任务工具：耗时由仿真本身决定，等待不设 30s 上限（与 GUI 的 run
    #: 一致，报告 F7）。取消时先下发 stop，再等这次调用的收尾响应。
    _LONG_TOOLS = frozenset({"run", "teleop_start"})

    def __init__(self, client, parent=None):
        """client：McpClient（call_tool 异步发 MCP）。本地文件工具内置。"""
        super().__init__(parent)
        self._client = client                # McpClient
        self._config: AgentConfig = AgentConfig()
        self._files = LocalFiles()
        self._tools_manifest: list[dict] = []
        self._messages: list[dict] = []
        self._pending: list[tuple[str, dict]] = []  # 待发 McpClient 调用 [(name, args, cb)]
        # 简易消息队列：UI -> send() 入队，run() 主循环拉取
        self._queue: list[str] = []
        self._running = True          # 线程是否继续存活（shutdown 置 False）
        self._cancel = False          # 仅取消当前一轮（stop 置 True）

    def configure(self, cfg: AgentConfig, mcp_tools: list[dict]) -> None:
        """更新配置与工具清单（运行中也可调，下一轮生效）。"""
        self._config = cfg
        self._tools_manifest = list(mcp_tools)

    def set_world_dir(self, path, extra_roots=None) -> None:
        """加载/切换世界时调（LocalFiles 根目录）。

        extra_roots：世界显式引用的控制器所在目录（worlds/ 与 controllers/
        同级布局下，Agent 需要能读写 ../controllers/xxx.py，见报告 F4）。
        """
        import pathlib
        roots = [pathlib.Path(path)] if path else []
        self._files.set_roots(roots, [pathlib.Path(p) for p in (extra_roots or [])])

    def set_history(self, world_name: str | None) -> None:
        """加载新世界后重置对话历史（system prompt 重注）。"""
        self._messages = [{
            "role": "system",
            "content": system_prompt(world_name, self._tools_manifest),
        }]

    @Slot(str)
    def send(self, text: str) -> None:
        """UI 发用户文本（队列驱动，避免主线程直接调 OpenAI）。"""
        self._queue.append(text)

    def stop(self) -> None:
        """取消当前这一轮对话（不退出线程）。

        置取消标志即可：worker 在 HTTP 返回后、每个工具调用前、以及同步
        等 MCP 回复的轮询定时器里都会检查它，本轮随即收尾。run 线程继续
        待命——用户停止后还能直接再发消息。
        """
        self._cancel = True

    def shutdown(self) -> None:
        """退出 worker 线程（窗口关闭时用；调用方随后 wait）。"""
        self._running = False
        self._cancel = True
        self._queue.append("__shutdown__")   # 唤醒可能在 wait 的循环

    # ------------------------------------------------------------ 主循环

    def run(self) -> None:
        import time
        while self._running:
            if not self._queue:
                time.sleep(0.05)
                continue
            text = self._queue.pop(0)
            if text == "__shutdown__":
                break
            if not self._config.usable:
                self.error.emit("未配置（点 配置 填 base_url/model/api_key）")
                self.finishedRound.emit()      # 界面恢复可输入
                continue
            self._cancel = False               # 每轮开始清掉上一轮的取消标志
            try:
                self._messages.append({"role": "user", "content": text})
                self._run_round()
            except Exception as e:
                self.error.emit(f"对话异常: {e}")
            finally:
                # 所有结束路径（正常/取消/HTTP 失败/异常）统一恢复界面
                self.finishedRound.emit()
        self._running = False

    def _run_round(self) -> None:
        """一轮对话：HTTP -> tool_calls? -> 分发 -> 回灌 -> 再 HTTP，直到无。

        结束信号由 run() 的 finally 统一发，这里只管提前返回（取消/出错）。
        """
        max_iters = 8          # 防死循环
        for _ in range(max_iters):
            if self._cancel:
                return
            resp = self._post_chat()
            if resp is None:
                return        # 出错已发 error 信号
            text = content_text(resp)
            if text:
                self.assistantText.emit(text)
                self._messages.append({"role": "assistant", "content": text})
            calls = extract_tool_calls(resp)
            if not calls:
                return
            # 把 tool_calls 也记进 assistant 消息（OpenAI 要求回灌 tool 角色）
            self._messages.append(self._assistant_with_tool_calls(resp, calls))
            image_msgs: list[dict] = []
            answered = 0
            try:
                for c in calls:
                    if self._cancel:
                        break
                    self.toolCall.emit(c["name"], c["arguments"])
                    result = self._dispatch_tool(c["name"], c["arguments"])
                    self.toolResult.emit(c["name"], result)
                    # tool 消息：role=tool，content 为文本，tool_call_id 关联。
                    # 截图/相机帧的 bytes 在此转 base64 并另发图片消息（F3）。
                    content, imgs = tool_payload(result)
                    image_msgs.extend(imgs)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": c.get("id") or c["name"],
                        "content": content,
                    })
                    answered += 1
            except Exception:
                # 工具抛异常也不能留下悬空 tool_call_id（F6）
                self._fill_skipped(calls[answered:], "工具执行异常，未执行")
                raise
            # 取消/异常都会让部分 tool_call 没有回复：必须补齐占位 tool
            # 消息，否则 assistant.tool_calls 与 tool 消息不配对（F6）
            self._fill_skipped(calls[answered:])
            # 图片消息必须排在同一轮的 tool 消息之后（视觉模型按此顺序读）
            if image_msgs:
                self._messages.extend(image_msgs)
            if self._cancel:
                return

    def _fill_skipped(self, calls: list[dict],
                      reason: str = "已取消（用户停止，未执行该工具调用）") -> None:
        """给没有执行的 tool_call 补占位 tool 消息，避免悬空 tool_call_id。

        OpenAI 消息协议要求 assistant 的每个 tool_calls 都有对应 role=tool
        回复；中途取消时直接返回会留下缺口，严格校验的服务端会拒收（F6）。
        """
        for c in calls:
            self._messages.append({
                "role": "tool",
                "tool_call_id": c.get("id") or c["name"],
                "content": reason,
            })

    def _assistant_with_tool_calls(self, resp: dict, calls: list[dict]) -> dict:
        """构造 OpenAI 期望的 assistant 消息（带 tool_calls 字段）。"""
        try:
            msg = dict(resp["choices"][0]["message"])
        except (TypeError, KeyError, IndexError):
            msg = {"role": "assistant", "content": ""}
        msg["role"] = "assistant"
        return msg

    def _dispatch_tool(self, name: str, args: dict):
        """本地工具直发；MCP 工具同步阻塞 McpClient.call_tool（在 worker 线程）。

        McpClient.call_tool 本是异步（发 JSON-RPC 给子进程），但本 worker 在
        QThread，可同步等回复（QEventLoop）：复用 GUI 工具调用链。
        """
        local = dispatch_local(self._files, name, args)
        if local is not None:
            return local
        # MCP 工具：同步阻塞调
        return self._call_mcp_sync(name, args)

    def _call_mcp_sync(self, name: str, args: dict):
        """同步等 McpClient.call_tool 回复（在 worker 线程用 QEventLoop）。

        McpClient.call_tool 的回调签名是 on_done(is_error, data)（与 GUI
        工具调用链同一套语义）；错误统一包成 {"error": ...}，让 LLM 能
        从工具结果里看到失败原因而不是拿到异常。

        长任务工具（run/teleop_start）不设等待上限，与 GUI 的 run 一致
        （报告 F7）：耗时由仿真本身决定，被 30s 提前判 timeout 而后台还
        在跑会造成状态不一致。取消时先下发 stop，等这次调用的收尾响应
        正常回灌，再用兜底定时器防后端卡死。
        """
        from PySide6.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        result_box: list = []
        long_task = name in self._LONG_TOOLS
        timeout = QTimer()
        timeout.setSingleShot(True)
        # 取消轮询：stop()/shutdown() 从 GUI 线程置标志，这里在本线程的
        # 事件循环里轮询退出（跨线程直接 quit 其它线程的 loop 不可靠）
        cancel_poll = QTimer()
        cancel_poll.setInterval(50)
        stop_sent = {"done": False}

        def finish(item):
            if not result_box:
                result_box.append(item)
            loop.quit()

        def on_cancel_tick():
            if not self._cancel or result_box:
                return
            if not long_task:
                loop.quit()
                return
            if stop_sent["done"]:
                return
            # 长任务不能一走了之：先让后端停下（stop 立即返回），run 的
            # 收尾响应照常写回；15s 兜底防后端卡死
            stop_sent["done"] = True
            self._client.call_tool("stop", {}, lambda e, d: None)
            timeout.start(15000)

        timeout.timeout.connect(lambda: finish({"error": "timeout"}))
        cancel_poll.timeout.connect(on_cancel_tick)

        def cb(is_error: bool, data):
            finish({"error": str(data)} if is_error else data)

        if not long_task:
            timeout.start(30000)         # 普通工具 30s 等待上限
        cancel_poll.start()
        try:
            if long_task:
                self._client.call_tool(name, args, cb, timeout_ms=None)
            else:
                self._client.call_tool(name, args, cb)
            if not result_box:       # 回调可能同步到达（测试/本地后端）
                loop.exec()
        finally:
            cancel_poll.stop()
            timeout.stop()
        if result_box:
            return result_box[0]
        return {"error": "已取消" if self._cancel else "no response"}

    def _post_chat(self) -> dict | None:
        """POST chat/completions；返回响应 JSON（出错发 error 返回 None）。"""
        tools = to_openai_tools(self._tools_manifest)
        # 本地工具也注入 OpenAI tools（让 LLM 知道参数 schema）
        for t in local_tools_manifest():
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["inputSchema"]}})
        body = {
            "model": self._config.model,
            "messages": self._messages,
            "tools": tools,
            "stream": False,
        }
        body.update(self._config.extra)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._config.chat_url(), data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._config.api_key:
            req.add_header("Authorization", f"Bearer {self._config.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            self.error.emit(f"HTTP {e.code}: {body or e.reason}")
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            self.error.emit(f"网络/解析错误: {e}")
            return None
