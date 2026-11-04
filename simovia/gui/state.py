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
