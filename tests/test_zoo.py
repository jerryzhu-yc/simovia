"""M4 验收：机器人模型库（Menagerie 接入，资产随仓库自带）。

资产在仓库 models/zoo/，装好即离线可用；缺失则跳过相应用例。
"""
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from simovia.errors import WorldError  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _cached(key: str, xml: str) -> bool:
    """自带资产或 menagerie 缓存命中即可跑（离线优先）。"""
    from simovia.zoo import bundled_model
    if bundled_model(key) is not None:
        return True
    return bool(list(pathlib.Path.home().glob(
        f".cache/mujoco_menagerie/models/{key}-*/{xml}")))


pytestmark = pytest.mark.skipif(
    not _cached("unitree_go2", "go2.xml"),
    reason="go2 资产缺失（仓库 models/zoo/ 不完整）")


def test_ensure_model_cached_and_unknown():
    """T1：缓存命中返回存在的 xml；未知 key 报错列可用名。"""
    from simovia.zoo import ensure_model, MODELS
    p = ensure_model("unitree_go2")
    assert p.name == "go2.xml" and p.exists()
    assert "unitree_go2" in MODELS
    with pytest.raises(WorldError, match="可用"):
        ensure_model("no_such_robot")


def test_all_models_bundled_offline():
    """T1b：34 个模型资产随仓库自带（离线即用，不依赖 mujoco-menagerie）。"""
    from simovia.zoo import BUNDLED_DIR, MODELS, ensure_model
    missing = [k for k in MODELS
               if not (BUNDLED_DIR / k / MODELS[k]["xml"]).exists()]
    assert not missing, f"仓库缺模型资产: {missing}"
    for key, m in MODELS.items():
        d = BUNDLED_DIR / key
        assert (d / "LICENSE").exists(), f"{key} 缺 LICENSE（模型许可要求）"
        assert (d / "assets").is_dir(), f"{key} 缺 assets 目录"
        p = ensure_model(key)              # 命中自带路径，不触发下载
        assert p == d / m["xml"], f"{key} 未优先使用自带资产: {p}"


def test_generate_products(tmp_path):
    """T2：产物完整——world.yaml 可编译、控制器模板含执行器与 API 示例。"""
    from simovia.zoo import generate
    from simovia.world.schema import parse_world
    from simovia.world.builder import build_world
    r = generate("unitree_go2", out_dir=tmp_path / "go2")
    world_p = pathlib.Path(r["world"])
    ctrl_p = pathlib.Path(r["controller"])
    assert world_p.exists() and ctrl_p.exists()
    # 世界 YAML 落盘后独立可编译（model 绝对路径）
    import yaml
    with open(world_p, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    spec = parse_world(doc)
    spec.base_dir = world_p.parent
    _model, index = build_world(spec)
    assert len(index.robots["unitree_go2"].actuators) == 12
    # 控制器模板：执行器清单 + set/range/obs 示例 + 修改引导
    text = ctrl_p.read_text(encoding="utf-8")
    for name in ("FL_hip", "RR_calf"):
        assert name in text
    for api in ("robot.set(", "robot.range(", "robot.step()", "robot.obs"):
        assert api in text
    assert "改成你的" in text


def test_generated_world_runs_end_to_end(tmp_path):
    """T3：生成的世界经 MCP 会话层 load_world + run，零 crash、清单正确。"""
    from simovia.zoo import generate
    from simovia.mcp_server import SimoviaSession
    r = generate("unitree_go2", out_dir=tmp_path / "go2")
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": r["world"]})
        stats = s.call_tool("run", {"ticks": 40})
        assert stats["ticks"] == 40
        assert stats["crashed"] == []          # 模板控制器不得 crash
    finally:
        s.close()


def test_zoo_create_list_via_session():
    """T4a：MCP zoo_create(key='list') 下发预配置清单（GUI/Agent 同构入口）。"""
    from simovia.mcp_server import SimoviaSession
    s = SimoviaSession()
    try:
        r = s.call_tool("zoo_create", {"key": "list"})
        keys = [m["key"] for m in r["models"]]
        assert "unitree_go2" in keys and "franka_emika_panda" in keys
        assert all("title" in m and "category" in m for m in r["models"])
        with pytest.raises(WorldError):
            s.call_tool("zoo_create", {"key": "no_such"})
    finally:
        s.close()


def test_zoo_dialog_offscreen(qapp):
    """T4b：ZooDialog 离屏实例化、清单非空、selected_key 可取（GUI 闭环）。"""
    from simovia.gui.app import ZooDialog
    dlg = ZooDialog()
    try:
        assert dlg.listw.count() >= 34
        dlg.listw.setCurrentRow(1)
        assert dlg.selected_key        # 任一 key 可取即 UI 链路通
    finally:
        dlg.deleteLater()


# ================================================================ M5-P1 运动演示

def test_demo_config_full_coverage():
    """T1：模型库全部带 demo 配置，kind/字段完整（纯离线，零缓存依赖）。"""
    from simovia.zoo import MODELS
    assert len(MODELS) >= 34              # 34 基线（含 M7 扩至 34 个模型）
    assert all(m.get("demo") for m in MODELS.values())
    kinds = {m["demo"]["kind"] for m in MODELS.values()}
    assert kinds <= {"trot", "osc", "balance"}
    # 四足优先 trot（关节命名匹配 <LEG>_hip/_thigh/_calf），命名不匹配的
    # 退回 osc——两者都要控制要素齐备；其余走 osc/balance
    quads = [m for m in MODELS.values() if m["cat"] == "四足"]
    assert len(quads) == 6
    for m in quads:
        d = m["demo"]
        if d["kind"] == "trot":
            assert len(d["legs"]) == 4 and len(d["kinds"]) == 3
            for f in ("home", "amp", "freq", "settle", "ramp", "lag"):
                assert d[f] or d[f] == 0.0, (m["title"], f)
        else:
            assert d["kind"] == "osc", (m["title"], d["kind"])


def test_demo_generate_products(tmp_path):
    """T2 前置：demo=True 产物完整——demo.py/世界隐含 demo 引擎参数。"""
    import yaml
    from simovia.zoo import generate
    r = generate("unitree_go2", out_dir=tmp_path / "g", demo=True)
    assert "demo" in r and pathlib.Path(r["demo"]).exists()
    world_p = pathlib.Path(r["world"])
    with open(world_p, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    # 高摩擦刚度 + implicitfast：硬接触模型（spot solref=0.004）被 Euler
    # 覆盖会数值震荡弹跳——demo 世界的隐含引擎参数（attach 时优先）
    assert doc["impratio"] == 100
    assert doc["integrator"] == "implicitfast"
    assert doc["ctrl_period"] == 8          # demo 控制周期加密
    text = pathlib.Path(r["demo"]).read_text(encoding="utf-8")
    assert "robot.set(" in text and "robot.step()" in text
    assert "FL_hip" in text                 # 执行器清单已注入


def test_demo_go2_walks_end_to_end(tmp_path):
    """T2：go2 demo 端到端 600 tick 零 crash 且位移 > 1m（真实 ZMQ 全链路）。"""
    import math
    from simovia.zoo import generate
    from simovia.mcp_server import SimoviaSession
    r = generate("unitree_go2", out_dir=tmp_path / "g", demo=True)
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": r["world"]})
        obs0 = s.call_tool("get_observation", {})
        stats = s.call_tool("run", {"ticks": 600})
        assert stats["ticks"] == 600 and stats["crashed"] == []
        obs = s.call_tool("get_observation", {})
        rob0, rob1 = obs0["robots"]["unitree_go2"], obs["robots"]["unitree_go2"]
        root = next(iter(rob1["bodies"]))
        p0 = rob0["bodies"][next(iter(rob0["bodies"]))]
        p1 = rob1["bodies"][root]
        dxy = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        assert dxy > 1.0, f"位移 {dxy:.2f}m 未过 1m 门槛"
        assert p1[2] > 0.15, f"rootz {p1[2]:.2f} 趴地"
    finally:
        s.close()


@pytest.mark.skipif(not _cached("franka_emika_panda", "panda.xml"),
                    reason="panda 未缓存（首次使用需网络下载）")
def test_demo_panda_moves(tmp_path):
    """T3：panda demo 零 crash 且关节动（观测变化 > 0.1 rad）。"""
    from simovia.zoo import generate
    from simovia.mcp_server import SimoviaSession
    r = generate("franka_emika_panda", out_dir=tmp_path / "p", demo=True)
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": r["world"]})
        obs0 = s.call_tool("get_observation", {})
        stats = s.call_tool("run", {"ticks": 200})
        assert stats["ticks"] == 200 and stats["crashed"] == []
        obs = s.call_tool("get_observation", {})
        q0 = obs0["robots"]["franka_emika_panda"]["qpos"]
        q1 = obs["robots"]["franka_emika_panda"]["qpos"]
        dq = max(abs(a - b) for a, b in zip(q0, q1))
        assert dq > 0.1, f"关节最大变化 {dq:.3f} rad，未在运动"
    finally:
        s.close()


def _cached_path(key: str, xml: str) -> bool:
    """自带资产或 menagerie 缓存（files() 清单可能滞后，查目录兜底）。"""
    return _cached(key, xml)


@pytest.mark.skipif(not _cached_path("universal_robots_ur10e", "ur10e.xml"),
                    reason="ur10e 未缓存（首次使用需网络下载）")
def test_demo_ur10e_registered_and_moves(tmp_path):
    """M6：ur10e 入库——可生成 demo 且 6 关节扫掠运动，零 crash。"""
    from simovia.zoo import MODELS, generate
    from simovia.mcp_server import SimoviaSession
    assert "universal_robots_ur10e" in MODELS
    assert MODELS["universal_robots_ur10e"]["xml"] == "ur10e.xml"

    r = generate("universal_robots_ur10e", out_dir=tmp_path / "u", demo=True)
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": r["world"]})
        obs0 = s.call_tool("get_observation", {})
        stats = s.call_tool("run", {"ticks": 200})
        assert stats["ticks"] == 200 and stats["crashed"] == []
        obs = s.call_tool("get_observation", {})
        q0 = obs0["robots"]["universal_robots_ur10e"]["qpos"]
        q1 = obs["robots"]["universal_robots_ur10e"]["qpos"]
        assert len(q0) == 6                      # UR10e 6 轴
        dq = max(abs(a - b) for a, b in zip(q0, q1))
        assert dq > 0.1, f"关节最大变化 {dq:.3f} rad，未在运动"
    finally:
        s.close()


# ------------------------------------------------ 浮动基座 root 位姿同步

def test_floating_base_root_in_observation(tmp_path):
    """四足观测带浮动基座 root [x,y,z,qw,qx,qy,qz]，且 run 后躯干 z 随
    物理压低（不再锁在出生位）。匿名 free joint 不在 qpos 里，靠 root 键。"""
    from simovia.zoo import generate
    from simovia.mcp_server import SimoviaSession
    r = generate("unitree_go2", out_dir=tmp_path / "root", demo=True)
    s = SimoviaSession()
    try:
        s.call_tool("load_world", {"yaml_path": r["world"]})
        obs0 = s.call_tool("get_observation", {})
        rb0 = obs0["robots"]["unitree_go2"]
        assert "root" in rb0 and len(rb0["root"]) == 7
        assert len(rb0["qpos"]) == 12           # root 不混进 qpos
        z0 = rb0["root"][2]
        s.call_tool("run", {"ticks": 100})
        z1 = s.call_tool("get_observation", {}) \
            ["robots"]["unitree_go2"]["root"][2]
        assert z1 < z0 - 0.05, (
            f"躯干未随物理下落（出生 {z0:.3f} -> {z1:.3f}），root 可能没更新")
    finally:
        s.close()


def test_render_model_follows_floating_root(tmp_path):
    """GUI 渲染模型 apply_observation 后浮动基座躯干跟随 root（四足
    "悬空奔跑"的回归守护：以前匿名 free joint 不同步，只有腿动）。"""
    import mujoco
    from simovia.zoo import generate
    from simovia.world.schema import load_world
    from simovia.gui.render_view import RenderModel
    r = generate("unitree_go2", out_dir=tmp_path / "render", demo=True)
    spec = load_world(r["world"])
    rm = RenderModel(spec, width=64, height=48)
    try:
        ridx = rm.index.robots["unitree_go2"]

        def root_qadr():
            for bid in ridx.bodies.values():
                if int(rm.model.body_parentid[bid]) != 0:
                    continue
                for k in range(int(rm.model.body_jntnum[bid])):
                    jid = int(rm.model.body_jntadr[bid]) + k
                    if int(rm.model.jnt_type[jid]) == \
                            int(mujoco.mjtJoint.mjJNT_FREE):
                        return int(rm.model.jnt_qposadr[jid])
            return None
        qadr = root_qadr()
        assert qadr is not None                # go2 必须有匿名 free joint
        z_init = float(rm.data.qpos[qadr + 2])
        fake_root = [0.0, 0.0, 0.276, 1.0, 0.0, 0.0, 0.0]
        rm.apply_observation(
            {"robots": {"unitree_go2": {
                "root": fake_root,
                "qpos": [0.0] * 12,
                "joint_names": list(ridx.joints)}}})
        assert float(rm.data.qpos[qadr + 2]) == pytest.approx(0.276)
        assert abs(float(rm.data.qpos[qadr + 2]) - z_init) > 0.05
    finally:
        rm.close()
