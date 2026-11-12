"""本地文件工具：write_file / read_file，范围限定当前世界目录。

AI Agent 自己写场景 YAML / 控制器代码后，再调 load_world + run 验证——
与 GUI 用户的工作流同构。无 ZMQ 中转：直接读写磁盘，回复给 LLM。
"""
from __future__ import annotations

import pathlib

from simovia.errors import WorldError


class LocalFiles:
    """管理本地文件工具的根目录集合。

    世界 YAML 所在目录是主根（相对路径的解析基准）；世界显式引用的
    控制器所在目录作为附加根——项目把 worlds/ 与 controllers/ 设计成
    同级，若只允许 worlds/，Agent 连示例世界的控制器都读不了（报告 F4）。

    路径解析：
    - 绝对路径：必须落在任一允许根内（防止 AI 越界写 ~/.bashrc 等）
    - 相对路径：按主根拼接
    """

    def __init__(self, root: pathlib.Path | None = None,
                 extra_roots=None):
        self._roots: list[pathlib.Path] = []
        self.set_roots([root] if root else [], extra_roots)

    @property
    def root(self) -> pathlib.Path | None:
        """主根（世界目录）；无根时为 None。"""
        return self._roots[0] if self._roots else None

    @property
    def roots(self) -> list[pathlib.Path]:
        return list(self._roots)

    def set_root(self, root: pathlib.Path | None) -> None:
        self.set_roots([root] if root else [])

    def set_roots(self, roots, extra_roots=None) -> None:
        """设置允许根：第一个是主根，其余只作越界校验。去重保序。"""
        out: list[pathlib.Path] = []
        for r in list(roots) + list(extra_roots or []):
            if not r:
                continue
            p = pathlib.Path(r).resolve()
            if p not in out:
                out.append(p)
        self._roots = out

    def _in_roots(self, full: pathlib.Path) -> bool:
        for r in self._roots:
            try:
                full.relative_to(r)
                return True
            except ValueError:
                continue
        return False

    def _resolve(self, path: str) -> pathlib.Path:
        if not self._roots:
            raise WorldError("LocalFiles", "未设置世界目录，无法读写文件")
        p = pathlib.Path(path)
        full = p if p.is_absolute() else (self._roots[0] / p)
        full = full.resolve()
        if not self._in_roots(full):
            allowed = "、".join(str(r) for r in self._roots)
            raise WorldError(
                "LocalFiles",
                f"路径 '{path}' 越界：只能在 {allowed} 内读写") from None
        return full

    def write_file(self, path: str, content: str) -> dict:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        rel = full
        for r in self._roots:
            try:
                rel = full.relative_to(r)
                break
            except ValueError:
                continue
        return {"path": str(rel), "bytes": len(content.encode("utf-8")),
                "abs": str(full)}

    def read_file(self, path: str) -> dict:
        full = self._resolve(path)
        try:
            content = full.read_text(encoding="utf-8")
        except FileNotFoundError:
            allowed = "、".join(str(r) for r in self._roots)
            raise WorldError(
                "LocalFiles",
                f"文件不存在: {path}（在 {allowed} 下）") from None
        return {"path": path, "content": content,
                "bytes": len(content.encode("utf-8"))}


_LOCAL_DISPATCH = {"write_file", "read_file", "read_mujoco_doc"}


def dispatch_local(files: LocalFiles, name: str, args: dict) -> dict | None:
    """分发本地工具调用。非本地工具名返回 None（让 worker 走 MCP）。"""
    if name not in _LOCAL_DISPATCH:
        return None
    if name == "write_file":
        return files.write_file(args.get("path", ""), args.get("content", ""))
    if name == "read_file":
        return files.read_file(args.get("path", ""))
    if name == "read_mujoco_doc":
        return read_mujoco_doc(args)
    return None


# ---------------------------------------------------------------- MuJoCo 官方文档

#: mujoco 4.0 起随 pip 包自带的 7 份 agent skills（doc/skills/<name>/SKILL.md）
SKILLS = ("accelerated", "gui", "parity", "python", "rendering",
          "spec_editing", "studio")

#: 单次返回的字符上限：整份 SKILL.md 最大 28KB，全量回灌会挤占上下文
DOC_MAX_CHARS = 12000

#: rst 标题下划线字符（顺序即层级，用于切分锚点所在小节）
_RST_UNDERLINE = set("=-~^\"'`#*+:.!")


def _doc_root() -> pathlib.Path:
    """mujoco 包内置的 doc/ 目录；未安装或旧版本没有时给出可读错误。"""
    try:
        import mujoco
    except ImportError:
        raise WorldError("read_mujoco_doc：当前环境没装 mujoco，读不到官方文档") from None
    root = pathlib.Path(mujoco.__file__).resolve().parent / "doc"
    if not root.is_dir():
        raise WorldError(
            f"read_mujoco_doc：当前 mujoco 包没有内置 doc/（{root} 不存在）。"
            "需要 mujoco>=4.0；旧版本请改用 https://mujoco.readthedocs.io/")
    return root


def _clip(source: str, text: str, max_chars: int) -> dict:
    """截断到 max_chars。max_chars 为负时切片会从尾部反着砍，必须先夹到 >=1。"""
    max_chars = max(1, max_chars)
    return {"source": source, "chars": len(text),
            "truncated": len(text) > max_chars,
            "content": text[:max_chars]}


def _arg_str(args: dict, key: str) -> str:
    """宽容取字符串参数：LLM 可能把数字塞进 path，不能直接 .strip()。"""
    v = args.get(key)
    return "" if v is None else str(v).strip()


def _available_skills(root: pathlib.Path) -> list[str]:
    """以 doc/skills/*/SKILL.md 为准动态枚举：4.1 新增 skill 不必改代码。"""
    found = sorted(p.parent.name for p in (root / "skills").glob("*/SKILL.md"))
    return found or list(SKILLS)


def _is_anchor_line(line: str) -> bool:
    s = line.strip()
    return s.startswith(".. _") and s.endswith(":") and len(s) > 5


def _anchor_names(lines: list[str]) -> list[str]:
    return [ln.strip()[4:-1] for ln in lines if _is_anchor_line(ln)]


def _rst_headings(lines: list[str]) -> list[tuple[int, int]]:
    """返回 [(标题行号, 层级)]。层级 = 下划线字符在文档中首次出现的顺序。"""
    order: list[str] = []
    out = []
    for i in range(len(lines) - 1):
        cur, nxt = lines[i].rstrip(), lines[i + 1].rstrip()
        if not cur or len(nxt) < len(cur) or len(set(nxt)) != 1:
            continue
        ch = nxt[0]
        if ch not in _RST_UNDERLINE:
            continue
        if ch not in order:
            order.append(ch)
        out.append((i, order.index(ch) + 1))
    return out


def _next_nonblank(lines: list[str], i: int) -> int:
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def _next_content(lines: list[str], i: int) -> int:
    """下一个非空且非锚点行：锚点可能连续出现（别名），跳过它们再判断标题。"""
    i += 1
    while i < len(lines) and (not lines[i].strip() or _is_anchor_line(lines[i])):
        i += 1
    return i


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _extract_block(lines: list[str], start: int, max_lines: int = 40) -> str:
    """行内锚点（属性/列表项/图）没有标题：切出锚点所属的那一段。

    XMLreference 的 `.. _replicate-count:` 后面直接是属性说明，changelog 的
    `.. _midphase:` 后面是一条条目——按「首个非空行缩进 + 空行后同级起新块」
    切，比「切到下一个标题」精确得多（后者可能一次回灌上百 KB）。
    """
    j = _next_content(lines, start)
    if j >= len(lines):
        return "\n".join(lines[start:])
    base = _indent(lines[j])
    k = j + 1
    while k < len(lines) and k - j <= max_lines:
        if not lines[k].strip():
            n = _next_nonblank(lines, k)
            if n >= len(lines) or _indent(lines[n]) <= base:
                break
        k += 1
    body = lines[start:k]
    while len(body) > 1 and not body[-1].strip():
        body.pop()
    return "\n".join(body)


def _extract_section(text: str, anchor: str) -> str:
    """按 rst 锚点 ``.. _Anchor:`` 切出小节；行内锚点则切出所属段落。"""
    lines = text.splitlines()
    target = f".. _{anchor}:"
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == target)
    except StopIteration:
        names = _anchor_names(lines)
        near = [n for n in names if anchor.lower() in n.lower()][:8]
        raise WorldError(
            f"read_mujoco_doc：文档里没有锚点 '{anchor}'。"
            f"相近可用: {near or names[:8]}") from None
    heads = _rst_headings(lines)
    first = next((h for h in heads if h[0] > start), None)
    if first is None:
        return "\n".join(lines[start:])
    # 标题紧跟锚点（含 overline 风格的上划线占一行）才算小节锚点
    if first[0] - _next_content(lines, start) > 1:
        return _extract_block(lines, start)
    end = next((i for i, lvl in heads
                if i > first[0] and lvl <= first[1]), len(lines))
    body = lines[start:end]
    # 下一节的 `.. _label:` 紧贴在它的标题前，会被切进来；去掉尾部空行与锚点
    while len(body) > 1 and (not body[-1].strip() or _is_anchor_line(body[-1])):
        body.pop()
    return "\n".join(body)


def read_mujoco_doc(args: dict) -> dict:
    """读 MuJoCo 4.0 自带的官方手册 / agent skills（离线，按需取片段）。

    - 不传参：返回 skills 清单与用法
    - skill=<name>：整份 doc/skills/<name>/SKILL.md（截断到 max_chars）
    - path=<rst 相对路径> + 可选 anchor=<锚点名>：手册章节或其中一小节
    """
    skill = _arg_str(args, "skill")
    path = _arg_str(args, "path").replace("\\", "/")   # Windows 路径也能用
    anchor = _arg_str(args, "anchor")
    if skill and path:
        raise WorldError("read_mujoco_doc：skill 与 path 只能二选一")
    if anchor and not path:
        raise WorldError(
            "read_mujoco_doc：anchor 必须和 path 一起用（skill 返回整份文档）")
    try:
        max_chars = int(args.get("max_chars") or DOC_MAX_CHARS)
    except (TypeError, ValueError):
        raise WorldError("read_mujoco_doc：max_chars 需要正整数") from None
    root = _doc_root()

    if skill:
        skills = _available_skills(root)
        if skill not in skills:
            raise WorldError(
                f"read_mujoco_doc：未知 skill '{skill}'；可用: {skills}")
        full = root / "skills" / skill / "SKILL.md"
        if not full.is_file():          # 目录里有同名目录但没 SKILL.md
            raise WorldError(f"read_mujoco_doc：{full} 不存在")
        return _clip(f"skills/{skill}/SKILL.md",
                     full.read_text(encoding="utf-8"), max_chars)

    if path:
        full = (root / path).resolve()
        try:
            full.relative_to(root)
        except ValueError:
            raise WorldError(
                f"read_mujoco_doc：path 越界（只能读 mujoco 包内 doc/），"
                f"实际: {path}") from None
        if not full.is_file():
            raise WorldError(
                f"read_mujoco_doc：文档不存在 {path}；"
                "常用: modeling.rst / XMLreference.rst / programming/modeledit.rst")
        text = full.read_text(encoding="utf-8")
        if anchor:
            text = _extract_section(text, anchor)
        return _clip(f"doc/{path}" + (f"#{anchor}" if anchor else ""),
                     text, max_chars)

    return {"doc_root": str(root), "skills": _available_skills(root),
            "usage": "read_mujoco_doc(skill='spec_editing') 读 agent skills；"
                     "read_mujoco_doc(path='modeling.rst', anchor='CTuning') "
                     "读手册小节"}
