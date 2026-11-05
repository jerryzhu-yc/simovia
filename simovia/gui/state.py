"""UI 状态机（纯逻辑，无 Qt 依赖，可单测）。

状态与按钮可用性：
  EMPTY    刚启动（仅 Load 可用）
  IDLE     世界已加载、没在跑（全部可用）
  RUNNING  run 进行中（仅停止可用——run 提前收尾，控制器正常退出）
  PAUSED   一轮 run 提前收尾（步进 N tick 跑完 / 用户停止）：仿真状态
           保留（qpos/tick 不变），可继续运行/继续步进/重置；控制器进程
           在下一次 run 时由后端自动重新拉起
  FINISHED 一轮 run 跑到终态（控制器退出/世界 max_ticks）
           （Load/Reset 可用；点 Run 由 app 层自动 reset 再 run；
            场景编辑可用——提交走 load_world 全量重载，天然回初态）
"""
from __future__ import annotations

import difflib


class UiState:
    EMPTY = "empty"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"

    # action -> 允许发起的状态集合
    _ALLOWED = {
        "load": {EMPTY, IDLE, RUNNING, FINISHED, PAUSED},  # run 中换世界由服务器拒绝
        "reset": {IDLE, FINISHED, PAUSED},
        "run": {IDLE, FINISHED, PAUSED},            # FINISHED 点击=自动 reset 再 run；
                                                     # PAUSED 直接从当前状态继续
        "stop": {RUNNING},                          # 唯一在 RUNNING 可用的动作
        "observe": {IDLE, FINISHED, RUNNING, PAUSED},
        # 场景编辑（拖动/位姿/装备/添加机器人）：FINISHED/PAUSED 也允许——
        # 用户跑完演示后调整布局再跑是高频路径；提交走 load_world 全量重载
        "edit": {IDLE, FINISHED, PAUSED},
    }

    def __init__(self):
        self._s = self.EMPTY

    @property
    def value(self) -> str:
        return self._s

    def can(self, action: str) -> bool:
        return self._s in self._ALLOWED.get(action, set())

    def to(self, state: str) -> None:
        self._s = state


class EditHistory:
    """场景编辑历史：YAML 全文快照（撤销/重做/操作历史共用，纯逻辑）。

    文本级快照简单可靠：所有场景编辑最终都以 load_world(yaml_text)
    提交，撤销只需把上一份全文再投喂一次。不记录与当前文本相同的快照，
    避免拖动无位移等空操作把历史挤爆。
    """

    def __init__(self, limit: int = 50):
        self._limit = max(2, int(limit))
        self._items: list[dict] = []
        self._idx = -1

    def __len__(self) -> int:
        return len(self._items)

    @property
    def index(self) -> int:
        return self._idx

    def reset(self, text: str, label: str = "初始状态") -> None:
        """新世界载入：历史从这份全文重新开始。"""
        self._items = [{"label": label, "text": text}]
        self._idx = 0

    def push(self, text: str, label: str = "编辑场景") -> bool:
        """记录一次编辑。与当前快照相同返回 False（不入栈）。"""
        cur = self.current()
        if cur is not None and cur["text"] == text:
            return False
        if self._idx < len(self._items) - 1:
            del self._items[self._idx + 1:]      # 新编辑截断重做分支
        self._items.append({"label": label, "text": text})
        if len(self._items) > self._limit:
            del self._items[:len(self._items) - self._limit]
        self._idx = len(self._items) - 1
        return True

    def current(self) -> dict | None:
        if 0 <= self._idx < len(self._items):
            return self._items[self._idx]
        return None

    def can_undo(self) -> bool:
        return self._idx > 0

    def can_redo(self) -> bool:
        return 0 <= self._idx < len(self._items) - 1

    def undo(self) -> dict | None:
        if not self.can_undo():
            return None
        self._idx -= 1
        return self.current()

    def redo(self) -> dict | None:
        if not self.can_redo():
            return None
        self._idx += 1
        return self.current()

    def jump(self, index: int) -> dict | None:
        if not (0 <= index < len(self._items)):
            return None
        self._idx = index
        return self.current()

    def entries(self) -> list[dict]:
        """[(label, text)] 只读视图（操作历史对话框用）。"""
        return [dict(it) for it in self._items]


def text_diff(old: str, new: str, context: int = 3,
              max_lines: int = 400) -> str:
    """两段 YAML 全文的 unified diff（空串 = 无改动）。"""
    if old == new:
        return ""
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile="已保存", tofile="当前（未保存）",
        lineterm="", n=context))
    if len(lines) > max_lines:
        rest = len(lines) - max_lines
        lines = lines[:max_lines] + [f"…（还有 {rest} 行差异未显示）"]
    return "\n".join(lines)
