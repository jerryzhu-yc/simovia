"""CLI 测试：inspect 设备清单、run 的控制器状态提示（新手首次使用路径）。"""
import json
import pathlib

import pytest

from simovia.cli import main

ROOT = pathlib.Path(__file__).parent.parent


class TestInspect:
    def test_lists_actuators_joints_sensors(self, capsys):
        main(["inspect", str(ROOT / "worlds" / "pick_cube.yaml")])
        out = capsys.readouterr().out
        # 写控制器要先知道执行器叫什么、范围多少
        assert "actuator1" in out and "ctrlrange" in out
        assert "gripper/gripper_act" in out
        assert "joint1" in out and "hinge" in out
        assert "onrobot_2fg7" in out
        assert "成功条件 cube" in out
        # 场景物体名 = obs["objects"] 的键
        assert "cube" in out and "table" in out

    def test_bad_world_reports_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\nscene:\n  objects:\n  - {name: a, type: boxx}\n",
                     encoding="utf-8")
        try:
            main(["inspect", str(p)])
        except SystemExit as e:
            assert "世界加载失败" in str(e)
        else:
            raise AssertionError("坏世界应当 SystemExit")


class TestRunSummary:
    def test_reports_missing_controller(self, tmp_path, capsys):
        """没有控制器的世界：run 必须明说"机器人不会动"，否则新手以为物理坏了"""
        p = tmp_path / "no_ctrl.yaml"
        p.write_text(
            "name: no_ctrl\n"
            "robots:\n"
            f"  - {{name: arm, model: {ROOT / 'models' / 'demo_arm.xml'},"
            " pos: [0, 0, 0.05]}\n", encoding="utf-8")
        main(["run", str(p), "--ticks", "1"])
        out = capsys.readouterr().out
        assert "控制器 arm: 未配置" in out
        assert "没有控制器" in out

    def test_reports_controller_path(self, capsys):
        main(["run", str(ROOT / "worlds" / "pick_cube.yaml"), "--ticks", "1"])
        out = capsys.readouterr().out
        assert "控制器 arm: ../controllers/pick_place.py" in out

    def test_reports_robot_base_pose(self, capsys):
        """跑完要能看到机器人 root 位置，否则无法判断四足站没站住"""
        main(["run", str(ROOT / "worlds" / "pick_cube.yaml"), "--ticks", "1"])
        out = capsys.readouterr().out
        assert "arm/link0 最终位置" in out

    def test_json_includes_robot_base_pose(self, capsys):
        main(["run", str(ROOT / "worlds" / "pick_cube.yaml"),
              "--ticks", "1", "--json"])
        d = json.loads(capsys.readouterr().out)
        assert d["robots"]["arm"]["base"] == "arm/link0"
        assert len(d["robots"]["arm"]["pos"]) == 3
        assert "cube" in d["objects"]          # 原有物体输出保持兼容


class TestDoctor:
    """环境诊断：缺依赖/资产时给出可执行的提示，且退出码反映必需项。"""

    def test_json_report_covers_required_items(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["doctor", "--json"])
        d = json.loads(capsys.readouterr().out)
        names = {c["name"] for c in d["checks"]}
        # 必需依赖 + 资产 + 样例世界编译都要在报告里
        assert {"Python", "mujoco", "numpy", "PyYAML", "pyzmq",
                "样例世界"} <= names
        assert d["ok"] is (e.value.code == 0)
        assert all(c["level"] in ("OK", "WARN", "FAIL") for c in d["checks"])

    def test_human_report_has_conclusion(self, capsys):
        with pytest.raises(SystemExit):
            main(["doctor"])
        out = capsys.readouterr().out
        assert "环境诊断" in out and "结论" in out


class TestStartScripts:
    def test_windows_launcher_quotes_interpreter(self):
        """解释器可能在 C:\\Program Files\\... 或中文路径下，必须加引号调用。"""
        text = (ROOT / "start.bat").read_text(encoding="utf-8")
        assert '"%PYEXE%" %PYARGS% -m simovia gui' in text
        assert '%PYEXE% -m simovia gui' not in text
