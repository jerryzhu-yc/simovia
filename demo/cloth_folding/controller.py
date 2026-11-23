"""GUI 场景控制器：在软件里运行 G1 叠毛毯 demo。

世界 YAML（worlds/g1_cloth_fold.yaml）给 g1 挂了
args.shared_world_controller: true，内核加载世界时会在内核进程内执行本文件，
用 create_controller(simulation) 创建协调器。之后每个控制 tick 调一次
advance()，协调器直接在内核自己的 model/data 上推进物理——布料顶点钉扎、
基座锁定这些「逐物理步写状态」的操作只有这样才能生效（普通控制器子进程
只能写 ctrl）。

GUI 用法：文件 → 打开世界 → worlds/g1_cloth_fold.yaml → 工具栏 Run。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.cloth_folding.run import ClothFold


class FoldController:
    """内核协调器：包一层 ClothFold，暴露 advance/done/success/detail。"""

    def __init__(self, simulation):
        if simulation.spec.ctrl_period != 1:
            raise RuntimeError(
                "叠毛毯 demo 要求世界 ctrl_period=1：布料钉扎必须逐物理步生效，"
                "一个控制 tick 跨多个物理步会漏掉钉扎窗口")
        self.fold = ClothFold(simulation.model, simulation.index, simulation.data)
        self.done = False
        self.success = False
        self.detail = ""

    def advance(self) -> bool:
        if self.fold.finished:
            self._finish()
            return False
        more = self.fold.advance()
        if not more:
            self._finish()
        return more

    def _finish(self):
        if self.done:
            return
        self.done = True
        mt = self.fold.metrics()
        # 成功判据：对折的本质是「有超过半数的布料离开桌面，且上层在 x 方向
        # 压回下层上方」。不能用总宽度/总厚度判断——折峰立起会让 z 跨度很大。
        self.success = bool(mt["overlap"] > 0.08
                            and mt["lifted"] > 0.5 * mt["nvert"])
        self.detail = (f"抬离桌面 {mt['lifted']}/{mt['nvert']} 个顶点、"
                       f"上下层 x 重叠 {mt['overlap'] * 100:.1f} cm、"
                       f"布面 x {mt['x'][0]:.3f}~{mt['x'][1]:.3f} m")
        print(f"[cloth_fold] {self.detail} -> "
              f"{'对折成功' if self.success else '未达标'}")

    def close(self):
        pass


def create_controller(simulation):
    return FoldController(simulation)
