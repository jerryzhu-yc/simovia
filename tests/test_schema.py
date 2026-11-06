"""schema 校验测试：合法世界全字段解析 + 各种坏输入报错可读。"""
import pytest

from simovia.errors import WorldError
from simovia.world.schema import (load_world, parse_world, hex_to_rgba,
                                   resolve_zoo_models)

import pathlib

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


def _base() -> dict:
    """最小合法世界"""
    return {
        "name": "t",
        "scene": {"objects": [
            {"name": "ball", "type": "sphere", "size": [0.1], "pos": [0, 0, 0.1]}
        ]},
    }


def test_load_world_gbk_crlf_fallback(tmp_path):
    """中文 Windows 记事本存的 GBK + CRLF 世界 YAML 也能读（utf-8 优先兜底）。"""
    import yaml
    text = yaml.safe_dump({"name": "测试世界", "scene": {"objects": []}},
                          allow_unicode=True, sort_keys=False)
    p = tmp_path / "world_gbk.yaml"
    p.write_bytes(text.replace("\n", "\r\n").encode("gbk"))
    spec = load_world(p)
    assert spec.name == "测试世界"


def test_read_text_compat_utf8_first_gbk_fallback(tmp_path):
    """统一读文本入口：utf-8 原样返回，GBK 文件自动兜底，不抛解码错误。
    MCP load_world 的二次读取与 GUI 打开世界/编辑器都依赖这一行为。"""
    from simovia.world.schema import read_text_compat
    a = tmp_path / "a_utf8.yaml"
    a.write_text("name: 中文世界", encoding="utf-8")
    assert read_text_compat(a) == "name: 中文世界"
    b = tmp_path / "b_gbk.yaml"
    b.write_bytes("name: 中文世界\r\n".encode("gbk"))
    # read_text 走 universal newlines：CRLF 统一读成 \n（两条路径一致）
    assert read_text_compat(b) == "name: 中文世界\n"
    # 纯 ASCII（任意编码可解）永远走 utf-8 路径
    c = tmp_path / "c_ascii.txt"
    c.write_bytes(b"hello")
    assert read_text_compat(c) == "hello"


class TestValidWorld:
    def test_pick_cube_loads(self):
        spec = load_world(WORLD)
        assert spec.name == "pick_cube"
        assert spec.timestep == 0.002
        assert spec.ctrl_period == 32
        assert spec.gravity == (0.0, 0.0, -9.81)

        table = spec.scene.objects[0]
        assert table.name == "table"
        assert table.static is True

        cube = next(o for o in spec.scene.objects if o.name == "cube")
        assert cube.type == "box"
        assert cube.size == (0.03, 0.03, 0.03)
        assert cube.mass == 0.1
        assert cube.friction == 1.0
        assert spec.scene.floor.friction == 1.0

        arm = spec.robots[0]
        assert arm.name == "arm"
        # zoo://franka_emika_panda -> menagerie 缓存 panda_nohand.xml
        assert arm.model.endswith("panda_nohand.xml")
        assert arm.controller.endswith("pick_place.py")
        assert arm.args == {}

        assert spec.task.success.body == "cube"
        assert spec.task.max_ticks == 5000
        assert spec.runtime.timeout_policy == "last_ctrl"
        assert spec.base_dir == WORLD.resolve().parent

    def test_minimal_world(self):
        spec = parse_world(_base())
        assert spec.scene.floor.size == (2.0, 2.0)
        assert spec.robots == []
        assert spec.runtime.realtime is False


class TestSchemaErrors:
    def _err(self, d):
        with pytest.raises(WorldError) as e:
            parse_world(d)
        return str(e.value)

    def test_missing_name(self):
        msg = self._err({})
        assert "缺少必填字段" in msg and "name" in msg

    def test_duplicate_names(self):
        d = _base()
        d["robots"] = [{"name": "ball", "model": "x.xml"}]
        assert "重复" in self._err(d)

    def test_bad_geom_type(self):
        d = _base()
        d["scene"]["objects"][0]["type"] = "torus"
        assert "torus" in self._err(d)

    def test_bad_size_arity(self):
        d = _base()
        d["scene"]["objects"][0]["size"] = [0.1, 0.2]
        assert "1 个数字" in self._err(d)

    def test_negative_size(self):
        d = _base()
        d["scene"]["objects"][0]["size"] = [-0.1]
        assert "必须为正" in self._err(d)

    def test_bad_color(self):
        d = _base()
        d["scene"]["objects"][0]["color"] = "red"
        assert "#RRGGBB" in self._err(d)

    def test_bad_zone(self):
        d = _base()
        d["task"] = {"success": {"body": "ball",
                                 "zone": {"min": [0, 0, 0], "max": [0, 0, 1]}}}
        assert "min 必须 < max" in self._err(d)

    def test_unknown_task_body_without_robots(self):
        d = _base()
        d["task"] = {"success": {"body": "ghost",
                                 "zone": {"min": [0, 0, 0], "max": [1, 1, 1]}}}
        msg = self._err(d)
        assert "ghost" in msg and "ball" in msg  # 错误里要列出可用名字

    def test_unknown_reward_kind(self):
        d = _base()
        d["task"] = {"reward": [{"velocity": {"a": "ball"}}]}
        assert "velocity" in self._err(d)

    def test_bad_timeout_policy(self):
        d = _base()
        d["runtime"] = {"timeout_policy": "explode"}
        assert "explode" in self._err(d)

    def test_bad_ctrl_period(self):
        d = _base()
        d["ctrl_period"] = 0
        assert "ctrl_period" in self._err(d)

    def test_nan_and_infinity_rejected(self):
        """YAML 的 .nan/.inf 能解析成功但会让物理静默失真，必须在 schema 拦。"""
        for bad in (float("nan"), float("inf"), float("-inf")):
            d = _base()
            d["timestep"] = bad
            msg = self._err(d)
            assert "有限数字" in msg and "timestep" in msg

            d = _base()
            d["gravity"] = [0, 0, bad]
            msg = self._err(d)
            assert "有限数字" in msg and "gravity[2]" in msg

    def test_ctrl_period_rejects_fraction(self):
        """1.9 不能被静默截断成 1（控制频率会变）。"""
        d = _base()
        d["ctrl_period"] = 1.9
        msg = self._err(d)
        assert "整数" in msg and "ctrl_period" in msg and "1.9" in msg

    def test_ctrl_period_integral_float_ok(self):
        d = _base()
        d["ctrl_period"] = 32.0
        assert parse_world(d).ctrl_period == 32

    def test_max_ticks_rejects_fraction(self):
        d = _base()
        d["task"] = {"max_ticks": 2.5}
        msg = self._err(d)
        assert "整数" in msg and "max_ticks" in msg

    def test_missing_world_file(self):
        with pytest.raises(WorldError) as e:
            load_world("不存在.yaml")
        assert "不存在" in str(e.value)


class TestHexColor:
    def test_hex_to_rgba(self):
        assert hex_to_rgba("#FF8000") == [1.0, 0.502, 0.0, 1.0]


class TestGripperParsing:
    """M6：RobotSpec.gripper 的 YAML 解析（不含物理，物理见 test_grippers）。"""

    def _robot(self, gripper=None):
        d = _base()
        r = {"name": "arm", "model": "x.xml"}
        if gripper is not None:
            r["gripper"] = gripper
        d["robots"] = [r]
        return parse_world(d).robots[0]

    def test_no_gripper_defaults_none(self):
        assert self._robot().gripper is None

    def test_gripper_full_fields(self):
        g = self._robot({
            "model": "onrobot_2fg7", "site": "attachment_site",
            "pos": [0, 0, 0.1], "quat": [0, 1, 0, 0]}).gripper
        assert g is not None
        assert g.model == "onrobot_2fg7"
        assert g.site == "attachment_site"
        assert g.pos == (0.0, 0.0, 0.1)
        assert g.quat == (0.0, 1.0, 0.0, 0.0)

    def test_gripper_model_only(self):
        g = self._robot({"model": "onrobot_2fg7"}).gripper
        assert g.model == "onrobot_2fg7" and g.site is None
        assert g.pos is None and g.quat is None

    def test_gripper_must_be_dict(self):
        with pytest.raises(WorldError) as e:
            self._robot("onrobot_2fg7")
        assert "gripper" in str(e.value)

    def test_gripper_missing_model(self):
        with pytest.raises(WorldError) as e:
            self._robot({"site": "attachment_site"})
        assert "model" in str(e.value)

    def test_gripper_bad_vec(self):
        with pytest.raises(WorldError) as e:
            self._robot({"model": "g", "pos": [0, 0]})
        assert "pos" in str(e.value)
        with pytest.raises(WorldError) as e:
            self._robot({"model": "g", "quat": [1, 0, 0]})
        assert "quat" in str(e.value)


class TestEndEffectorsParsing:
    """末端执行器列表解析：左右手各一个 + 旧 gripper 字段兼容。"""

    def _robot(self, **extra):
        d = _base()
        d["robots"] = [{"name": "g1", "model": "x.xml", **extra}]
        return parse_world(d).robots[0]

    def test_single_dict_accepted(self):
        rd = self._robot(end_effectors={"model": "onrobot_2fg7"})
        assert len(rd.end_effectors) == 1
        assert rd.end_effectors[0].model == "onrobot_2fg7"
        assert rd.gripper.model == "onrobot_2fg7"   # 旧属性取第一个

    def test_left_and_right(self):
        rd = self._robot(end_effectors=[
            {"model": "leap_hand", "side": "left"},
            {"model": "onrobot_2fg7", "side": "right"}])
        assert [e.side for e in rd.end_effectors] == ["left", "right"]
        assert rd.gripper.model == "leap_hand"

    def test_side_case_insensitive(self):
        rd = self._robot(end_effectors=[{"model": "m", "side": "RIGHT"}])
        assert rd.end_effectors[0].side == "right"

    def test_bad_side(self):
        with pytest.raises(WorldError) as e:
            self._robot(end_effectors=[{"model": "m", "side": "middle"}])
        assert "side" in str(e.value)

    def test_must_be_dict_or_list(self):
        with pytest.raises(WorldError) as e:
            self._robot(end_effectors="onrobot_2fg7")
        assert "end_effectors" in str(e.value)

    def test_old_gripper_migrated(self):
        rd = self._robot(gripper={"model": "onrobot_2fg7"})
        assert len(rd.end_effectors) == 1
        assert rd.end_effectors[0].model == "onrobot_2fg7"


class TestCameraAndHandsParsing:
    """相机深度参数与自带末端删除开关（schema 层，不含物理）。"""

    def _robot(self, **extra):
        d = _base()
        d["robots"] = [{"name": "arm", "model": "x.xml", **extra}]
        return parse_world(d).robots[0]

    def test_camera_depth_defaults(self):
        r = self._robot(sensors=[{"name": "cam", "type": "camera",
                                  "body": "link1"}])
        c = r.sensors[0]
        assert c.fovy is None and c.depth is False
        assert c.near == 0.01 and c.far == 50.0

    def test_camera_depth_fields(self):
        c = self._robot(sensors=[{
            "name": "cam", "type": "camera", "body": "link1",
            "fovy": 60, "depth": True, "near": 0.05, "far": 3.0}]).sensors[0]
        assert c.fovy == 60.0 and c.depth is True
        assert c.near == 0.05 and c.far == 3.0

    def test_camera_fovy_out_of_range(self):
        with pytest.raises(WorldError) as e:
            self._robot(sensors=[{"name": "cam", "type": "camera",
                                  "body": "link1", "fovy": 200}])
        assert "fovy" in str(e.value)

    def test_camera_depth_not_bool(self):
        with pytest.raises(WorldError) as e:
            self._robot(sensors=[{"name": "cam", "type": "camera",
                                  "body": "link1", "depth": "yes"}])
        assert "depth" in str(e.value)

    def test_camera_bad_clip_planes(self):
        with pytest.raises(WorldError) as e:
            self._robot(sensors=[{"name": "cam", "type": "camera",
                                  "body": "link1", "near": 5, "far": 2}])
        assert "near" in str(e.value)

    def test_strip_hands_bool_and_list(self):
        assert self._robot().strip_hands is False
        assert self._robot(strip_hands=True).strip_hands is True
        assert self._robot(
            strip_hands=["left_hand"]).strip_hands == ["left_hand"]

    def test_strip_hands_bad_type(self):
        with pytest.raises(WorldError) as e:
            self._robot(strip_hands=1)
        assert "strip_hands" in str(e.value)


class TestZooBirthPose:
    """zoo:// 机器人生出生态兜底：LLM 常写 pos z=0.1 导致穿地弹飞。"""

    def _resolve(self, robot: dict) -> dict:
        d = {"name": "t", "robots": [robot]}
        return resolve_zoo_models(d)["robots"][0]

    def test_fills_missing_pos_with_recommended(self):
        r = self._resolve({"name": "dog", "model": "zoo://unitree_go2"})
        assert r["pos"] == [0, 0, 0.45]

    def test_clamps_too_low_z_and_keeps_xy(self):
        r = self._resolve({"name": "dog", "model": "zoo://unitree_go2",
                           "pos": [1.5, -2.0, 0.1]})
        assert r["pos"] == [1.5, -2.0, 0.45]

    def test_keeps_higher_z(self):
        r = self._resolve({"name": "dog", "model": "zoo://unitree_go2",
                           "pos": [0, 0, 0.8]})
        assert r["pos"] == [0, 0, 0.8]

    def test_fixed_base_arm_stays_zero(self):
        r = self._resolve({"name": "arm", "model": "zoo://franka_emika_panda"})
        assert r["pos"] == [0, 0, 0]
