"""M0.7 验收：场景模板生成器（方案 5.7 A 档）。

验收标准：每个模板产出的 YAML 全部通过 schema 校验 + 编译 + 冒烟跑
100 tick；同一命令永远产出同一世界。
"""
import glob
import pathlib
import subprocess
import sys

import pytest

from simovia.kernel.sim import Simulation
from simovia.scene_gen import TEMPLATES, generate
from simovia.world.schema import load_world

ROOT = pathlib.Path(__file__).parent.parent

# pallet 依赖 UR5e 资产（随仓库 models/zoo/ 自带，离线可用）
def _has_ur5e() -> bool:
    from simovia.zoo import bundled_model
    if bundled_model("universal_robots_ur5e") is not None:
        return True
    return bool(glob.glob(str(pathlib.Path.home() / (
        ".cache/mujoco_menagerie/models/universal_robots_ur5e-*/ur5e.xml"))))


_UR5E = _has_ur5e()

# (模板, 参数变体)：默认 + 每个参数至少一个变体
CASES = [
    ("tabletop", {}),
    ("tabletop", {"objects": 1}),
    ("tabletop", {"objects": 3}),
    ("push", {}),
    ("maze", {}),
    ("maze", {"rows": 5, "cols": 6}),
    ("race", {}),
    ("race", {"lanes": 3}),
    ("empty", {}),
    ("empty", {"robot": "diff_car"}),
    ("empty", {"robot": "none"}),
]
PALLET_CASES = [("pallet", {}), ("pallet", {"cubes": 1}),
                ("pallet", {"cubes": 6})]


@pytest.mark.parametrize("name,params", CASES)
def test_template_smoke(tmp_path, name, params):
    out = tmp_path / f"{name}.yaml"
    out.write_text(generate(name, tmp_path, params), encoding="utf-8")
    spec = load_world(out)                 # schema 校验
    with Simulation(spec) as sim:          # 编译 + 100 tick 冒烟
        stats = sim.run(max_ticks=100)
    assert stats["ticks"] == 100
    assert not stats["nan"]
    assert stats["crashed"] == []


@pytest.mark.parametrize("name,params", PALLET_CASES)
@pytest.mark.skipif(not _UR5E, reason="menagerie UR5e 未下载，pallet 无法构建")
def test_pallet_smoke(tmp_path, name, params):
    """M6 码垛分区：UR5e+2FG7 + N cube + N 目标区，编译并冒烟 100 tick。"""
    import yaml
    from simovia.world.builder import build_world

    out = tmp_path / "pallet.yaml"
    text = generate(name, tmp_path, params)
    out.write_text(text, encoding="utf-8")
    spec = load_world(out)
    n = params.get("cubes", 3)
    assert spec.name == f"pallet_{n}"
    assert len(spec.robots) == 1
    assert spec.robots[0].gripper.model == "onrobot_2fg7"
    # A/B 两区物体齐全（N cube + N pad + 1 table）
    names = {o.name for o in spec.scene.objects}
    assert names >= ({f"cube_{i}" for i in range(n)}
                     | {f"pad_{i}" for i in range(n)})
    # 每个 cube 一条 zone reward
    assert len(spec.task.reward) == n

    # 编译 + 夹爪设备并入
    model, index = build_world(spec)
    ridx = index.robots["arm"]
    assert "gripper/gripper_act" in ridx.actuators

    with Simulation(spec) as sim:
        stats = sim.run(max_ticks=100)
    assert stats["ticks"] == 100
    assert not stats["nan"]
    assert stats["crashed"] == []
    # 产出是合法 YAML 且 model 是 zoo:// 模型库引用（load 时解析成缓存路径）
    assert str(yaml.safe_load(text)["robots"][0]["model"]) \
        .startswith("zoo://")


@pytest.mark.parametrize("name,params", [("tabletop", {}),
                                         ("tabletop", {"objects": 3}),
                                         ("pallet", {"cubes": 3}),
                                         ("pallet", {"cubes": 6})])
def test_table_is_supported_and_covers_work_area(tmp_path, name, params):
    """桌面不能悬空：四条腿落地并顶住桌面；cube 与机器人基座都在桌面内。"""
    import yaml

    d = yaml.safe_load(generate(name, tmp_path, params))
    objs = {o["name"]: o for o in d["scene"]["objects"]}
    top = objs["table"]
    top_z = top["pos"][2] + top["size"][2]
    legs = [o for n, o in objs.items() if n.startswith("table_leg_")]
    assert len(legs) == 4
    for leg in legs:
        # 腿底贴地、腿顶顶住桌面底面
        assert leg["pos"][2] - leg["size"][2] == pytest.approx(0.0, abs=1e-9)
        assert (leg["pos"][2] + leg["size"][2]
                == pytest.approx(top["pos"][2] - top["size"][2], abs=1e-9))
    x0 = top["pos"][0] - top["size"][0]
    x1 = top["pos"][0] + top["size"][0]
    y0 = top["pos"][1] - top["size"][1]
    y1 = top["pos"][1] + top["size"][1]
    cubes = [o for n, o in objs.items() if n.startswith("cube_")]
    for c in cubes:
        assert c["pos"][2] - c["size"][2] == pytest.approx(top_z, abs=1e-9)
        assert x0 < c["pos"][0] < x1 and y0 < c["pos"][1] < y1
    # 码垛场景的机械臂基座必须落在桌面上，不能悬在桌沿外
    if name == "pallet":
        base = d["robots"][0]["pos"]
        assert base[2] == pytest.approx(top_z, abs=1e-9)
        assert x0 < base[0] < x1 and y0 < base[1] < y1


def test_deterministic(tmp_path):
    for name in TEMPLATES:
        if name == "pallet" and not _UR5E:
            continue               # 无缓存时不触发下载
        assert generate(name, tmp_path) == generate(name, tmp_path)


def test_bad_template_and_params(tmp_path):
    with pytest.raises(Exception, match="未知模板"):
        generate("nope", tmp_path)
    with pytest.raises(Exception, match="超出范围"):
        generate("maze", tmp_path, {"rows": 99})
    with pytest.raises(Exception, match="没有参数"):
        generate("maze", tmp_path, {"bogus": 1})


def test_cli_scenegen(tmp_path):
    out = tmp_path / "maze.yaml"
    r = subprocess.run([sys.executable, "-m", "simovia", "scenegen", "maze",
                        "--rows", "3", "--cols", "3", "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "已生成" in r.stdout
    spec = load_world(out)
    assert spec.name == "maze_3x3"
