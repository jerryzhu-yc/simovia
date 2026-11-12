"""M5-P4 验收：Agent 控制台桥接（纯函数 + 配置持久化，不打真 API）。"""
import os
import stat

import pytest


@pytest.fixture(scope="module")
def qapp():
    """offscreen QApplication（worker 的 QEventLoop/QTimer 需要 app 实例）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ================================================================ T7-a 配置持久化

def test_config_roundtrip(tmp_path, monkeypatch):
    """配置写读往返；目录与文件权限 0600 护 api_key。"""
    from simovia.agent import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", tmp_path / ".simovia")
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / ".simovia" / "agent.json")

    cfg = cfg_mod.AgentConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test-123456",
        model="deepseek-chat")
    assert cfg.usable and cfg.chat_url().endswith("/chat/completions")
    cfg_mod.save_config(cfg)
    p = tmp_path / ".simovia" / "agent.json"
    assert p.exists()
    # Linux 上权限必须 0600（护 api_key；Windows chmod 静默忽略）
    if os.name == "posix":
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600
    loaded = cfg_mod.load_config()
    assert loaded.base_url == cfg.base_url
    assert loaded.api_key == cfg.api_key
    assert loaded.model == cfg.model
    # 空 api_key 允许（本地 Ollama 等）
    cfg2 = cfg_mod.AgentConfig(base_url="http://localhost:11434/v1", model="llama3")
    assert cfg2.usable and not cfg2.api_key
    # 缺 base_url 不可用
    assert not cfg_mod.AgentConfig(model="x").usable
    # 已有 /chat/completions 后缀不重复
    cfg3 = cfg_mod.AgentConfig(base_url="https://x/v1/chat/completions", model="m")
    assert cfg3.chat_url() == "https://x/v1/chat/completions"


# ================================================================ T7-b 工具 manifest 转换

def test_to_openai_tools_shape():
    """MCP manifest -> OpenAI tools 数组（function calling 格式）。"""
    from simovia.agent.bridge import to_openai_tools
    mcp = [{"name": "run", "description": "运行仿真",
            "inputSchema": {"type": "object",
                            "properties": {"ticks": {"type": "integer"}}}}]
    tools = to_openai_tools(mcp)
    assert tools == [{"type": "function",
                      "function": {"name": "run", "description": "运行仿真",
                                   "parameters": mcp[0]["inputSchema"]}}]


def test_local_tools_manifest_fields():
    """本地工具 write_file/read_file/read_mujoco_doc 的参数 schema 完整。"""
    from simovia.agent.bridge import local_tools_manifest
    names = {t["name"] for t in local_tools_manifest()}
    assert names == {"write_file", "read_file", "read_mujoco_doc"}
    wf = next(t for t in local_tools_manifest() if t["name"] == "write_file")
    assert set(wf["inputSchema"]["properties"]) == {"path", "content"}
    assert set(wf["inputSchema"]["required"]) == {"path", "content"}
    doc = next(t for t in local_tools_manifest() if t["name"] == "read_mujoco_doc")
    assert set(doc["inputSchema"]["properties"]) == {
        "skill", "path", "anchor", "max_chars"}
    assert doc["inputSchema"]["required"] == []      # 不传参 = 看清单


def test_system_prompt_lists_all_tools():
    """系统 prompt 列出 MCP 10 + 本地 3，且含当前世界名 + 工作流建议。"""
    from simovia.agent.bridge import system_prompt, local_tools_manifest
    mcp_tools = [{"name": "zoo_create", "description": "生成世界"},
                 {"name": "load_world", "description": "加载世界"},
                 {"name": "run", "description": "运行"}]
    s = system_prompt("my_world", mcp_tools)
    assert "my_world" in s
    for t in mcp_tools:
        assert t["name"] in s
    for t in local_tools_manifest():
        assert t["name"] in s
    assert "zoo_create" in s and "write_file" in s
    assert "工作流" in s


# ================================================================ T7-c tool_calls 解析

def test_extract_tool_calls_and_content():
    """OpenAI 响应解析：tool_calls 与 content 文本双向覆盖。"""
    from simovia.agent.bridge import extract_tool_calls, content_text
    # 有 tool_calls
    resp = {"choices": [{"message": {
        "content": "好的，先加载世界",
        "tool_calls": [{"id": "c1", "function": {
            "name": "load_world",
            "arguments": '{"yaml_path": "/tmp/w.yaml"}'}}]}}]}
    calls = extract_tool_calls(resp)
    assert len(calls) == 1
    assert calls[0]["name"] == "load_world"
    assert calls[0]["arguments"] == {"yaml_path": "/tmp/w.yaml"}
    assert calls[0]["id"] == "c1"          # 原始调用 id 必须保留
    assert content_text(resp) == "好的，先加载世界"
    # 无 tool_calls：仅 content
    resp2 = {"choices": [{"message": {"content": "已处理完"}}]}
    assert extract_tool_calls(resp2) == []
    assert content_text(resp2) == "已处理完"
    # 异常结构不崩
    assert extract_tool_calls({}) == []
    assert content_text({}) == ""
    # arguments 不是 JSON 字符串（dict 直接）也兼容
    resp3 = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "run", "arguments": {"ticks": 10}}}]}},]}
    assert extract_tool_calls(resp3)[0]["arguments"] == {"ticks": 10}


# ================================================================ T7-d 本地文件工具

def test_local_files_in_world_dir(tmp_path):
    """write_file/read_file 限定世界目录：相对/绝对路径都解析到根内。"""
    from simovia.agent.local_tools import LocalFiles, dispatch_local
    from simovia.errors import WorldError
    root = tmp_path / "world_dir"
    root.mkdir()
    f = LocalFiles(root)
    # 相对路径写文件
    r = f.write_file("scene.yaml", "name: test\n")
    assert (root / "scene.yaml").read_text() == "name: test\n"
    assert r["bytes"] == len("name: test\n")
    # 读文件
    r = f.read_file("scene.yaml")
    assert r["content"] == "name: test\n"
    # 绝对路径在根内允许
    abs_p = root / "sub" / "ctrl.py"
    r = f.write_file(str(abs_p), "print('x')\n")
    assert abs_p.exists()
    # 越界写：抛 WorldError
    with pytest.raises(WorldError):
        f.write_file(str(tmp_path / "outside.yaml"), "x")
    # 未设置 root：抛错
    f2 = LocalFiles()
    with pytest.raises(WorldError):
        f2.write_file("a", "b")
    # dispatch_local 分发：本地工具名走本地，否则返回 None
    assert dispatch_local(f, "write_file", {"path": "a.txt", "content": "x"}).get("bytes") == 1
    assert dispatch_local(f, "read_file", {"path": "scene.yaml"}).get("content") == "name: test\n"
    assert dispatch_local(f, "run", {}) is None      # MCP 工具走 MCP


def test_local_files_read_nonexistent(tmp_path):
    """读不存在文件：抛 WorldError 列出根目录。"""
    from simovia.agent.local_tools import LocalFiles
    from simovia.errors import WorldError
    f = LocalFiles(tmp_path)
    with pytest.raises(WorldError) as e:
        f.read_file("ghost.yaml")
    assert "ghost.yaml" in str(e.value)


# ================================================================ T7-d2 MuJoCo 官方文档工具

def test_read_mujoco_doc_catalog():
    """不传参：列出随包自带的 skills 与用法。"""
    from simovia.agent.local_tools import SKILLS, read_mujoco_doc
    d = read_mujoco_doc({})
    assert set(SKILLS) <= set(d["skills"])     # 以 doc/skills 目录为准，允许多
    assert "spec_editing" in d["skills"] and d["usage"]


def test_read_mujoco_doc_skill_clipped():
    """skill 模式：整份 SKILL.md，按 max_chars 截断并标记。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"skill": "spec_editing", "max_chars": 200})
    assert d["source"] == "skills/spec_editing/SKILL.md"
    assert d["truncated"] is True and len(d["content"]) == 200
    assert d["chars"] > 200
    # 全量读时不应误截断
    full = read_mujoco_doc({"skill": "spec_editing"})
    assert "MjSpec" in full["content"]


def test_read_mujoco_doc_anchor_section():
    """rst + anchor：从 '.. _CTuning:' 切到同级/更高级标题为止。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"path": "modeling.rst", "anchor": "CTuning"})
    assert d["source"] == "doc/modeling.rst#CTuning"
    body = d["content"]
    assert body.lstrip().startswith(".. _CTuning:")
    assert len(body) < d["chars"] or not d["truncated"]   # 只取小节


def test_read_mujoco_doc_errors():
    """未知 skill / 路径越界 / 锚点不存在：都给可读错误。"""
    from simovia.agent.local_tools import read_mujoco_doc
    from simovia.errors import WorldError
    with pytest.raises(WorldError) as e:
        read_mujoco_doc({"skill": "nope"})
    assert "未知 skill" in str(e.value)
    with pytest.raises(WorldError) as e:
        read_mujoco_doc({"path": "../../../../etc/passwd"})
    assert "越界" in str(e.value)
    with pytest.raises(WorldError) as e:
        read_mujoco_doc({"path": "modeling.rst", "anchor": "NoSuchAnchor"})
    assert "没有锚点" in str(e.value)
    assert "相近可用" in str(e.value)      # 报错带真实候选，模型不用瞎猜


def test_read_mujoco_doc_max_chars_edge():
    """max_chars 为负不能变成「从尾部砍」：夹到 1 个字符。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"skill": "python", "max_chars": -100})
    assert len(d["content"]) == 1 and d["truncated"] is True


def test_read_mujoco_doc_windows_path():
    """LLM 从 Windows 习惯写反斜杠路径也能命中。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"path": "programming\\modeledit.rst"})
    assert d["source"] == "doc/programming/modeledit.rst"
    assert "mjSpec" in d["content"]


def test_read_mujoco_doc_ambiguous_args():
    """skill+path 同时给、或只给 anchor：直接报错而非静默忽略。"""
    from simovia.agent.local_tools import read_mujoco_doc
    from simovia.errors import WorldError
    with pytest.raises(WorldError) as e:
        read_mujoco_doc({"skill": "python", "path": "modeling.rst"})
    assert "二选一" in str(e.value)
    with pytest.raises(WorldError) as e:
        read_mujoco_doc({"anchor": "CTuning"})
    assert "必须和 path" in str(e.value)


def test_read_mujoco_doc_section_not_include_next_anchor():
    """切出的小节不能带下一节的 `.. _label:` 锚点行。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"path": "XMLreference.rst", "anchor": "actuator"})
    lines = [ln for ln in d["content"].splitlines() if ln.strip()]
    assert lines[0].strip() == ".. _actuator:"
    assert not lines[-1].strip().startswith(".. _")


def test_read_mujoco_doc_inline_anchor_block():
    """行内锚点（属性说明）只切自己那一段，不能甩到下一个标题。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"path": "XMLreference.rst", "anchor": "replicate-count"})
    assert "The number of replicas" in d["content"]
    assert len(d["content"].splitlines()) <= 10       # 一段而非整章
    assert ".. _replicate-sep:" not in d["content"]


def test_read_mujoco_doc_overline_heading():
    """overline 风格（上划线 + 标题 + 下划线）的锚点仍按小节切。"""
    from simovia.agent.local_tools import read_mujoco_doc
    d = read_mujoco_doc({"path": "mjx.rst", "anchor": "Mjx"})
    assert "MuJoCo XLA (MJX)" in d["content"]
    assert len(d["content"].splitlines()) > 10


def test_read_mujoco_doc_dispatch():
    """dispatch_local 认这个名字，且不依赖世界目录（LocalFiles 无 root）。"""
    from simovia.agent.local_tools import LocalFiles, dispatch_local
    r = dispatch_local(LocalFiles(), "read_mujoco_doc", {})
    assert r is not None and "skills" in r


# ================================================================ T7-e 工具调用闭环

def test_extract_tool_calls_keeps_ids_for_multiple_calls():
    """同一轮多个 tool_calls 各自保留 id（回传 tool_call_id 需一一对应）。"""
    from simovia.agent.bridge import extract_tool_calls
    resp = {"choices": [{"message": {"tool_calls": [
        {"id": "call_a", "function": {"name": "load_world", "arguments": "{}"}},
        {"id": "call_b", "function": {"name": "run", "arguments": '{"ticks": 3}'}},
    ]}}]}
    calls = extract_tool_calls(resp)
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert [c["name"] for c in calls] == ["load_world", "run"]
    # 端点没给 id：保留 None（worker 回退工具名，而不是假装有 id）
    resp2 = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "run", "arguments": "{}"}}]}}]}
    assert extract_tool_calls(resp2)[0]["id"] is None


def test_worker_mcp_sync_callback_is_two_args(qapp):
    """McpClient.call_tool 回调签名是 (is_error, data)：worker 必须按双参接收。"""
    from simovia.agent.worker import AgentWorker

    class _Client:
        def __init__(self, err, data):
            self.err, self.data, self.seen = err, data, None
            self.timeout_ms = "unset"

        def call_tool(self, name, args, on_done, timeout_ms="unset"):
            self.seen = (name, args)
            self.timeout_ms = timeout_ms
            on_done(self.err, self.data)      # 同步回调（本地后端/测试）
            return 1

    ok_client = _Client(False, {"ticks": 5})
    w = AgentWorker(ok_client)
    assert w._call_mcp_sync("run", {"ticks": 5}) == {"ticks": 5}
    assert ok_client.seen == ("run", {"ticks": 5})
    assert ok_client.timeout_ms is None   # run 是长任务，不设等待上限（F7）

    err_client = _Client(True, "后端未加载世界")
    w2 = AgentWorker(err_client)
    assert w2._call_mcp_sync("run", {}) == {"error": "后端未加载世界"}


def test_worker_tool_messages_reuse_model_call_ids(qapp):
    """role=tool 消息必须带模型下发的原始 tool_call_id（不能换成工具名）。"""
    from simovia.agent.worker import AgentWorker

    class _W(AgentWorker):
        def __init__(self):
            super().__init__(client=None)
            self._rounds = [
                {"choices": [{"message": {"tool_calls": [
                    {"id": "call_1", "function": {
                        "name": "run", "arguments": '{"ticks": 1}'}}]}}]},
                {"choices": [{"message": {"content": "完成"}}]},
            ]

        def _post_chat(self):
            return self._rounds.pop(0)

        def _dispatch_tool(self, name, args):
            return {"ok": name}

    w = _W()
    w._run_round()
    tool_msgs = [m for m in w._messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"


def test_worker_cancel_fills_skipped_tool_calls(qapp):
    """取消时未执行的 tool_call 必须补齐占位回复（报告 F6）：
    assistant.tool_calls 与 role=tool 必须一一对应，否则下一轮请求
    会被严格校验的消息服务拒绝。"""
    from simovia.agent.worker import AgentWorker

    class _W(AgentWorker):
        def __init__(self):
            super().__init__(client=None)
            self._cancel = False

        def _post_chat(self):
            return {"choices": [{"message": {"tool_calls": [
                {"id": "call_1", "function": {
                    "name": "run", "arguments": '{"ticks": 1}'}},
                {"id": "call_2", "function": {
                    "name": "run", "arguments": '{"ticks": 1}'}},
            ]}}]}

        def _dispatch_tool(self, name, args):
            self.stop()               # 第一个工具执行期间用户取消
            return {"ok": name}

    w = _W()
    w._run_round()
    asked = [c["id"] for m in w._messages if m.get("role") == "assistant"
             for c in (m.get("tool_calls") or [])]
    answered = [m["tool_call_id"] for m in w._messages
                if m.get("role") == "tool"]
    assert asked == ["call_1", "call_2"]
    assert answered == ["call_1", "call_2"]     # 无悬空 tool_call_id
    assert "已取消" in w._messages[-1]["content"]


def test_worker_long_tools_skip_wait_timeout(qapp):
    """run/teleop_start 不设等待上限（报告 F7）：长任务耗时由仿真决定，
    不能像普通工具那样 30s 提前判 timeout；普通工具仍用默认超时。"""
    from simovia.agent.worker import AgentWorker

    seen = {}

    class _Client:
        def call_tool(self, name, args, on_done, timeout_ms="unset"):
            seen[name] = timeout_ms
            on_done(False, {"ok": name})
            return 1

    w = AgentWorker(_Client())
    assert w._call_mcp_sync("run", {}) == {"ok": "run"}
    assert seen["run"] is None
    assert w._call_mcp_sync("teleop_start", {}) == {"ok": "teleop_start"}
    assert seen["teleop_start"] is None
    assert w._call_mcp_sync("read_world", {}) == {"ok": "read_world"}
    assert seen["read_world"] == "unset"


def test_worker_stop_only_cancels_current_round(qapp):
    """stop 只取消当前轮：worker 线程继续待命，下一条消息照常处理。"""
    from simovia.agent.config import AgentConfig
    from simovia.agent.worker import AgentWorker

    rounds = []

    class _W(AgentWorker):
        def __init__(self):
            super().__init__(client=None)
            self._config = AgentConfig(base_url="http://localhost/v1", model="m")

        def _run_round(self):
            rounds.append(len(self._messages))
            if len(rounds) == 1:
                self._cancel = True        # 第一轮被 stop 打断（走到检查点）
                return
            self._running = False          # 第二轮正常结束 -> 线程收工

    w = _W()
    ended = []
    w.finishedRound.connect(lambda: ended.append(1))
    w.send("first")
    w.send("second")
    w.run()
    assert rounds == [1, 2]                # 两轮都执行了（第一轮取消不影响下一轮）
    assert len(ended) == 2                 # 每轮结束都发一次（界面恢复可输入）
    assert w._cancel is False              # 新一轮清掉了上一轮的取消标志
    assert w._running is False


