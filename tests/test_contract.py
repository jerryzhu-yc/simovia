"""P0 验收：策略接口契约（contract.py）。

守护点：
- 契约从编译后的世界推导，关节/执行器顺序与内核观测一致
- 固定基座与浮动基座（go2）的 obs 布局正确
- JSON 往返后逐字段相等，指纹对变更敏感
"""
import json
import pathlib

import pytest

from simovia.contract import RobotContract, contracts_for_world
from simovia.world.builder import build_world
from simovia.world.schema import load_world

ROOT = pathlib.Path(__file__).resolve().parents[1]
PICK = ROOT / "worlds" / "pick_cube.yaml"


def _pick_contract() -> RobotContract:
    spec = load_world(str(PICK))
    model, index = build_world(spec)
    return RobotContract.from_world(spec, model, index.robots["arm"])


def _go2_world(tmp_path) -> tuple:
    from simovia.zoo import generate
    r = generate("unitree_go2", out_dir=tmp_path / "go2", demo=True)
    spec = load_world(r["world"])
    model, index = build_world(spec)
    return spec, model, index


def test_pick_cube_contract():
    c = _pick_contract()
    assert c.robot == "arm"
    assert c.floating_base is False
    assert c.timestep == pytest.approx(0.002)
    assert c.ctrl_period == 32
    assert c.control_dt == pytest.approx(0.064)
    assert c.action_dim == 8 and len(c.actuators) == 8
    assert len(c.joints) == 9
    assert c.obs_dim == sum(f.dim for f in c.obs_fields) == 18

    # 观测字段连续且关节顺序与 joints 一致
    assert [f.name for f in c.obs_fields] == ["joint_pos", "joint_vel"]
    assert c.obs_fields[0].offset == 0
    assert c.obs_fields[1].offset == c.obs_fields[0].dim

    # pick_cube 的执行器是位置控制（内部 PD），契约要带上 kp/kd
    first = c.actuators[0]
    assert first.mode == "position"
    assert first.target_type == "joint"
    assert first.kp and first.kp > 0 and first.kd is not None
    assert first.ctrl_min < first.ctrl_max

    parts = c.split_obs([0.0] * c.obs_dim)
    assert set(parts) == {"joint_pos", "joint_vel"}
    assert len(parts["joint_pos"]) == 9 and len(parts["joint_vel"]) == 9


def test_go2_floating_base_contract(tmp_path):
    spec, model, index = _go2_world(tmp_path)
    c = RobotContract.from_world(spec, model, index.robots["unitree_go2"])
    assert c.robot == "unitree_go2"
    assert c.floating_base is True
    assert c.action_dim == 12 and len(c.joints) == 12
    # 13 维 root（pos3+quat4+lin3+ang3）+ 12 关节位置 + 12 关节速度
    assert c.obs_dim == 37
    assert [f.name for f in c.obs_fields] == [
        "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
        "joint_pos", "joint_vel",
    ]
    assert c.obs_fields[1].offset == 3 and c.obs_fields[1].dim == 4
    # menagerie go2 默认是 motor（力矩直写）
    assert {a.mode for a in c.actuators} == {"torque"}
    assert all(a.name for a in c.actuators)


def test_contract_json_roundtrip_and_digest(tmp_path):
    c = _pick_contract()
    d = json.loads(c.to_json())
    c2 = RobotContract.from_dict(d)
    assert c2 == c
    assert c2.digest() == c.digest()

    # 改控制频率 -> 指纹必须变化（训练端/部署端据此发现不一致）
    c3 = RobotContract.from_dict({**d, "ctrl_period": 16,
                                  "control_dt": d["timestep"] * 16})
    assert c3.digest() != c.digest()


def test_contracts_for_world():
    spec = load_world(str(PICK))
    model, index = build_world(spec)
    contracts = contracts_for_world(spec, model, index)
    assert list(contracts) == ["arm"]
    assert contracts["arm"] == RobotContract.from_world(spec, model,
                                                        index.robots["arm"])
