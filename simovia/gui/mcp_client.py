"""GUI 侧 MCP 客户端：用 QProcess 托管 `python -m simovia mcp` 后端。

分层（下层不依赖 Qt，可无头单测）：
- LineCodec：字节流 -> 完整行（防半行，FIFO/管道分块到达都能拼）
- PendingMap：id -> 回调登记，响应按 id 派发（run 的响应会晚于其后
  请求的响应——服务器端 run 是延迟响应，绝不能按序配对）
- McpClient(QObject)：QProcess 生命周期 + 行协议编解码 + 回调派发

GUI 与 AI Agent 同构：都只是 MCP 客户端，走同一套工具。
"""
from __future__ import annotations

import base64
import codecs
import json
import sys

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

# 通用请求超时（毫秒）。run 等长任务的响应本来就晚到（服务器延迟应答），
# 调用方显式传 timeout_ms=None 关闭。
DEFAULT_TIMEOUT_MS = 60_000


# ---------------------------------------------------------------- 纯逻辑层

class LineCodec:
    """把陆续到达的字节块拼成完整行。"""

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        parts = self._buf.split("\n")
        self._buf = parts.pop()          # 尾段：不完整行（或空串）
        return [ln for ln in parts if ln.strip()]


class PendingMap:
    """请求-响应按 id 关联：响应乱序到达也能派发到正确的回调。"""

    def __init__(self):
        self._cbs: dict = {}

    def register(self, rid: int, cb) -> None:
        self._cbs[rid] = cb

    def dispatch(self, resp: dict) -> bool:
        """派发一个响应。返回 False 表示没有登记的 id（如服务器自发消息）。"""
        rid = resp.get("id")
        cb = self._cbs.pop(rid, None)
        if cb is None:
            return False
        cb(resp)
        return True

    def pending_ids(self) -> list[int]:
        return list(self._cbs)

    def fail_all(self, message: str, code: int = -32000) -> int:
        """后端退出/断开时让所有挂起请求以错误返回（界面状态得以释放）。"""
        cbs = list(self._cbs.values())
        self._cbs.clear()
        for cb in cbs:
            cb({"error": {"code": code, "message": message}})
        return len(cbs)


def tool_response(resp: dict):
    """把 JSON-RPC 响应解析成 (is_error, data)。

    data：run/get_observation 等 dict 工具解析为 dict；read_world 等
    str 工具原样返回；image content（screenshot/camera_frame）解析为
    {"image": bytes, "mime": str}；协议层错误返回 message 字符串。
    """
    if "error" in resp:
        return True, str(resp["error"].get("message", resp["error"]))
    result = resp.get("result", {})
    content = result.get("content") or [{}]
    if result.get("isError"):
        return True, content[0].get("text", "")
    first = content[0]
    if first.get("type") == "image":
        try:
            data = base64.b64decode(first.get("data", ""))
        except (ValueError, TypeError):
            data = b""
        return False, {"image": data,
                       "mime": first.get("mimeType", "image/png")}
    text = first.get("text", "")
    try:
        return False, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, text


# ---------------------------------------------------------------- Qt 客户端

class McpClient(QObject):
    """QProcess 托管的 MCP 后端客户端。

    用法：
        client = McpClient(self)
        client.start()
        client.call_tool("load_world", {"yaml_path": "..."},
                         on_done=lambda err, data: ...)
    """

    backendDied = Signal(str)            # 后端异常退出（msg）
    backendLog = Signal(str)             # 后端 stderr 一行日志（控制台显示）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._codec = LineCodec()
        # 增量 UTF-8 解码器：管道分块可能从多字节汉字中间切断（Windows
        # QProcess 尤其常见），逐块 decode 会产生 U+FFFD，必须跨块续接
        self._utf8_out = codecs.getincrementaldecoder("utf-8")("replace")
        self._utf8_err = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = PendingMap()
        self._next_id = 0
        self._timers: dict[int, QTimer] = {}
        self._closing = False
        self._status = ""

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        self._closing = False
        self._codec = LineCodec()
        self._utf8_out = codecs.getincrementaldecoder("utf-8")("replace")
        self._utf8_err = codecs.getincrementaldecoder("utf-8")("replace")
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._on_ready_read)
        self._proc.readyReadStandardError.connect(self._on_ready_err)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_proc_error)
        # cwd=项目根：`python -m simovia mcp` 能找到 simovia 包
        root = self._package_root()
        self._proc.setWorkingDirectory(str(root))
        self._proc.start(sys.executable, ["-m", "simovia", "mcp"])

    @staticmethod
    def _package_root():
        import pathlib
        return pathlib.Path(__file__).resolve().parents[2]

    def shutdown(self) -> None:
        """优雅退出：关 stdin（EOF）让服务器收尾，等不到再杀。"""
        self._closing = True
        if self._proc is None:
            self._fail_pending("后端已关闭")
            return
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.closeWriteChannel()
            if not self._proc.waitForFinished(5000):
                self._proc.kill()
                self._proc.waitForFinished(2000)
        self._proc = None
        self._fail_pending("后端已关闭")

    def restart(self) -> None:
        """重启后端进程（崩溃后恢复用；调用方随后重新 load_world）。"""
        self.shutdown()
        self.start()

    def is_alive(self) -> bool:
        return (self._proc is not None
                and self._proc.state() != QProcess.ProcessState.NotRunning)

    # ------------------------------------------------------------ 请求

    def request(self, method: str, params: dict | None, cb,
                timeout_ms: int | None = DEFAULT_TIMEOUT_MS) -> int:
        """发一个 JSON-RPC 请求，响应到达时 cb(resp_dict)。返回请求 id。

        timeout_ms 缺省 60s：超时后 cb 收到错误响应（界面状态可恢复）；
        None 表示不设超时（run 这类延迟应答的长任务）。
        """
        self._next_id += 1
        rid = self._next_id
        self._pending.register(rid, cb)
        if timeout_ms:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(
                lambda r=rid, m=method: self._on_timeout(r, m))
            timer.start(int(timeout_ms))
            self._timers[rid] = timer
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        self._write(req)
        return rid

    def call_tool(self, name: str, args: dict | None, on_done,
                  timeout_ms: int | None = DEFAULT_TIMEOUT_MS) -> int:
        """tools/call 封装。on_done(is_error, data)——见 tool_response。"""
        def cb(resp):
            err, data = tool_response(resp)
            on_done(err, data)
        return self.request("tools/call", {"name": name, "arguments": args or {}},
                            cb, timeout_ms=timeout_ms)

    def _on_timeout(self, rid: int, method: str) -> None:
        self._timers.pop(rid, None)
        self._pending.dispatch({"id": rid, "error": {
            "code": -32001, "message": f"请求超时：{method}"}})

    def _fail_pending(self, message: str) -> None:
        for timer in self._timers.values():
            timer.stop()
            timer.deleteLater()
        self._timers.clear()
        self._pending.fail_all(message)

    # ------------------------------------------------------------ 内部

    def _write(self, obj: dict) -> None:
        if self._proc is None:
            return
        data = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.write(data.encode("utf-8"))

    def _on_ready_read(self) -> None:
        raw = bytes(self._proc.readAllStandardOutput())
        chunk = self._utf8_out.decode(raw)
        for line in self._codec.feed(chunk):
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue                  # 服务器输出的坏行直接忽略
            if isinstance(resp, dict):
                timer = self._timers.pop(resp.get("id"), None)
                if timer is not None:
                    timer.stop()
                    timer.deleteLater()
                self._pending.dispatch(resp)

    def _on_ready_err(self) -> None:
        raw = bytes(self._proc.readAllStandardError())
        chunk = self._utf8_err.decode(raw)
        for line in chunk.splitlines():
            if line.strip():
                self.backendLog.emit(line)

    def _on_proc_error(self, err) -> None:
        if self._closing:
            return
        self._fail_pending(f"后端进程错误: {err}")
        self.backendDied.emit(f"后端进程错误: {err}")

    def _on_finished(self, code, status) -> None:
        # 无论正常/异常退出，挂起请求都不可能再收到响应：统一以错误返回，
        # 否则界面会一直停在忙碌状态（spinner 不消失、按钮不恢复）。
        self._fail_pending("后端进程已退出")
        if self._closing:
            return
        self.backendDied.emit(f"后端进程退出 code={code} status={status}")
