"""lidar 传感器测试：schema 校验 -> builder 编译（外观+索引）-> 读数正确性
-> 控制器协议端到端 -> bit 级只读守护。

读数测试直接用 worlds/lidar_nav.yaml 的真实几何（房间四墙 + 五处障碍 +
差速小车出厂位姿西南角朝北），beam 角度 -> 期望命中全部手算可验证。
"""
import math
import pathlib

import mujoco
import numpy as np
import pytest
import yaml

from simovia.errors import WorldError
from simovia.kernel.sim import Simulation
from simovia.kernel.task import lidar_scan
from simovia.world.builder import build_world
from simovia.world.schema import parse_world

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "lidar_nav.yaml"


def _nav_dict(lidar=True, controller=None):
    """载入 demo 世界 dict；可选摘掉雷达 / 换控制器（默认无控制器）。"""
    d = yaml.safe_load(WORLD.read_text(encoding="utf-8"))
    robot = d["robots"][0]
    robot["controller"] = controller
    if not lidar:
        robot["sensors"] = [s for s in robot["sensors"]
                            if s["type"] != "lidar"]
    return d


class TestLidarSchema:                            # ------------------------------ T1

    def test_explicit_values(self):
        s = parse_world(_nav_dict()).robots[0].sensors[0]
        assert s.type == "lidar" and s.body == "car"
        assert (s.rays, s.min_angle, s.max_angle, s.range) \
            == (60, -180.0, 180.0, 3.0)

    def test_defaults(self):
        """省略 rays/min_angle/max_angle/range：默认 36 束 / ±180° / 5m。"""
        d = _nav_dict()
        for k in ("rays", "min_angle", "max_angle", "range"):
            d["robots"][0]["sensors"][0].pop(k)
        s = parse_world(d).robots[0].sensors[0]
        assert (s.rays, s.min_angle, s.max_angle, s.range) \
            == (36, -180.0, 180.0, 5.0)

    @pytest.mark.parametrize("rays", [1, 2000])
    def test_rays_out_of_range(self, rays):
        d = _nav_dict()
        d["robots"][0]["sensors"][0]["rays"] = rays
        with pytest.raises(WorldError, match="光束数需在 2..1024"):
            parse_world(d)

    def test_bad_angle_window(self):
        """min >= max（角度窗非法）：-360 < min < max < 360。"""
        d = _nav_dict()
        d["robots"][0]["sensors"][0]["min_angle"] = 200
        with pytest.raises(WorldError,
                           match=r"-360 < min_angle < max_angle < 360"):
            parse_world(d)

    def test_angle_out_of_bounds(self):
        d = _nav_dict()
        d["robots"][0]["sensors"][0]["max_angle"] = 400
        with pytest.raises(WorldError,
                           match=r"-360 < min_angle < max_angle < 360"):
            parse_world(d)

    def test_range_must_be_positive(self):
        d = _nav_dict()
        d["robots"][0]["sensors"][0]["range"] = 0
        with pytest.raises(WorldError, match="量程必须为正数"):
            parse_world(d)

    def test_rejects_joint(self):
        d = _nav_dict()
        d["robots"][0]["sensors"][0]["joint"] = "car_free"
        with pytest.raises(WorldError, match="不使用 joint"):
            parse_world(d)

    def test_rejects_quat(self):
        d = _nav_dict()
        d["robots"][0]["sensors"][0]["quat"] = [1, 0, 0, 0]
        with pytest.raises(WorldError, match="只有 camera 用 quat"):
            parse_world(d)


@pytest.fixture(scope="module")
def built():
    spec = parse_world(_nav_dict())               # 无控制器：纯编译
    spec.base_dir = ROOT / "worlds"
    return spec, *build_world(spec)


class TestLidarBuild:                             # ------------------------------ T2

    def test_index_config(self, built):
        _, model, index = built
        six = index.robots["car"].sensors["lidar"]
        assert six.kind == "lidar" and six.dim == 60
        li = six.lidar
        # 60 束、-180..180 半开区间（步进 6°）、量程 3m
        assert len(li["angles"]) == 60
        assert li["angles"][0] == pytest.approx(math.radians(-180.0))
        assert li["angles"][1] == pytest.approx(math.radians(-174.0))
        assert li["angles"][-1] == pytest.approx(math.radians(174.0))
        assert li["range"] == 3.0
        # 挂载 body 解析成前缀全名 id；自体排除集合 = 本机器人全部 body
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car/car")
        assert li["body"] == bid
        assert li["robot_bodies"] == frozenset(
            index.robots["car"].bodies.values())
        assert len(li["robot_bodies"]) >= 5

    def test_visual_geoms_purely_visual(self, built):
        """雷达实体外观 mast/puck/lens/led：group2、无碰撞。"""
        _, model, _ = built
        for suffix in ("mast", "puck", "lens", "led"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"car/lidar_{suffix}")
            assert gid >= 0, suffix
            assert model.geom_group[gid] == 2
            assert model.geom_contype[gid] == 0
            assert model.geom_conaffinity[gid] == 0

    def test_visual_geoms_zero_mass(self, built):
        """雷达外观零质量：带雷达与不带雷达的 car body 质量逐位一致。"""
        spec0 = parse_world(_nav_dict(lidar=False))
        spec0.base_dir = ROOT / "worlds"
        model0, _ = build_world(spec0)
        _, model, index = built
        bid = index.robots["car"].sensors["lidar"].lidar["body"]
        bid0 = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_BODY, "car/car")
        assert model.body_mass[bid] == model0.body_mass[bid0]
        assert np.array_equal(model.body_inertia[bid],
                             model0.body_inertia[bid0])

    def test_no_mujoco_sensor_compiled(self, built):
        """lidar 不编译 MuJoCo sensor（读数走内核 mj_ray）。"""
        _, model, _ = built
        assert model.nsensor == 0


class TestLidarReadings:                          # ------------------------------ T3
    """读数正确性：车在西南角 (-2.1,-1.6) 朝北（+y），雷达原点在车中心。

    beam 角度 a_i = -180 + 6i（度），世界方向 = yaw(90°) + a_i：
    - i=0   (a=-180°，车尾/南)：南墙内面 y=-2.08 -> 0.48
    - i=15  (a= -90°，车右/东)：东墙 4.68 超量程 -> -1
    - i=19  (a= -66°)：pillar_d 圆柱（垂距 0.03 < r 0.25）-> ~1.41
    - i=30  (a=   0°，车头/北)：北墙 3.68 超量程 -> -1
    - i=45  (a= +90°，车左/西)：西墙内面 x=-2.58 -> 0.48
    """

    def test_beam_geometry(self, built):
        _, model, index = built
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scan = lidar_scan(model, data, index.robots["car"].sensors["lidar"])
        assert len(scan) == 60
        assert scan[0] == pytest.approx(0.48, abs=0.02)
        assert scan[15] == -1.0
        assert scan[19] == pytest.approx(1.41, abs=0.02)
        assert scan[30] == -1.0
        assert scan[45] == pytest.approx(0.48, abs=0.02)

    def test_self_geometry_excluded(self, built):
        """起点在机壳内部：读数绝不能是自身几何的近距离回波。"""
        _, model, index = built
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scan = lidar_scan(model, data, index.robots["car"].sensors["lidar"])
        assert all(d == -1.0 or d > 0.1 for d in scan)

    def test_obs_msg_lidar(self):
        """内核观测流：sensors 键带 lidar 60 维读数（-1 = 无回波惯例）。"""
        spec = parse_world(_nav_dict())
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            sim.run(max_ticks=1)
            scan = sim._obs_msg("car")["sensors"]["lidar"]
        assert len(scan) == 60
        assert scan[45] == pytest.approx(0.48, abs=0.03)   # 西墙
        assert scan[30] == -1.0                            # 北墙超量程


class TestLidarControllerProtocol:                # ------------------------------ T4

    def test_manifest_readings_publish(self, tmp_path):
        """真实 ZMQ 协议端到端：hello 清单 dim + sensor() + publish_camera
        + publish_overlay（雷达建图面板的矢量叠加层通道）。"""
        ctrl = tmp_path / "assert_lidar.py"
        ctrl.write_text(
            "from simovia.controller import Robot\n"
            "import numpy as np\n"
            "r = Robot()\n"
            "dims = {s['name']: s['dim'] for s in r.sensors}\n"
            "assert dims['lidar'] == 60, dims\n"
            "obs = r.step()\n"
            "scan = r.sensor('lidar')\n"
            "assert len(scan) == 60, len(scan)\n"
            "assert abs(scan[45] - 0.48) < 0.03, scan[45]   # 西墙\n"
            "assert scan[30] == -1.0                         # 北墙超量程\n"
            "r.publish_camera('overview',\n"
            "                np.zeros((48, 64, 3), np.uint8))\n"
            "r.publish_overlay('path', segments=[[0, 0, 1, 1]],\n"
            "                 points=[[2.1, 1.6]])\n"
            "r.report_result(True, 'lidar ok')\n",
            encoding="utf-8")
        spec = parse_world(_nav_dict(controller=str(ctrl)))
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=6)
        assert stats["crashed"] == []
        assert stats["ticks"] >= 1
        assert sim.controller_results["car"]["success"] is True
        assert sim.controller_overlays["car"]["path"] == {
            "segments": [[0.0, 0.0, 1.0, 1.0]], "points": [[2.1, 1.6]]}


class TestLidarMapping:                             # ------------------------------ T6
    """平台级雷达建图：内核 mapper 自动累积（GUI"雷达建图"面板数据源）。"""

    def test_mapper_created_with_lidar(self):
        """声明 lidar 的机器人有 mapper，无 lidar 世界的 mapper 表为空。"""
        spec = parse_world(_nav_dict())
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            assert set(sim.lidar_mappers) == {"car"}
        spec = parse_world(_nav_dict(lidar=False))
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            assert sim.lidar_mappers == {}

    def test_mapper_grid_fills_from_scan(self):
        """静止 3 tick：出生点周围西/南墙在量程内——栅格应同时出现
        占据（墙）与自由（射线穿越）格，其余保持未知。"""
        import base64
        import zlib
        spec = parse_world(_nav_dict())
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            sim.run(max_ticks=3)
            mapper = sim.lidar_mappers["car"]
            g = mapper.grid()
            assert g.dtype == np.uint8 and g.shape == (mapper.nx, mapper.ny)
            assert np.count_nonzero(g == 2) > 0      # 占据（西南墙可见）
            assert np.count_nonzero(g == 1) > 0      # 自由（穿越路径）
            assert np.count_nonzero(g == 0) > 0      # 未知（量程外的北/东）
            # meta 与 MCP 下发格式（_tool_lidar_map 通道）：zlib+base64 可逆
            m = mapper.meta()
            assert m["res"] == 0.05
            assert abs(m["ox"] - -(2.6 + 0.5)) < 1e-9
            wire = np.frombuffer(zlib.decompress(base64.b64decode(
                base64.b64encode(zlib.compress(g.tobytes(), 1)))),
                dtype=np.uint8).reshape(m["nx"], m["ny"])
            assert np.array_equal(wire, g)
            # 连续概率通道（GUI 平滑渐变渲染用）：128=未知，自由 <128 < 占据
            pr = mapper.prob()
            assert pr.dtype == np.uint8 and pr.shape == g.shape
            assert pr.max() > 192 and pr.min() < 64
            assert 100 < pr[g == 0].mean() < 156       # 未知区量化后≈128

    def test_lidar_pose_origin(self):
        """lidar_pose 给出挂点世界坐标（出生位姿 = (-2.1,-1.6) + 偏移）。"""
        from simovia.kernel.task import lidar_pose
        spec = parse_world(_nav_dict())
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            sim.run(max_ticks=1)
            ridx = sim.index.robots["car"]
            six = next(s for s in ridx.sensors.values() if s.lidar)
            px, py, yaw = lidar_pose(sim.data, six)
            assert abs(px - (-2.1 + 0.0)) < 1e-6      # pos 偏移 (0,0,0.075)
            assert abs(py - (-1.6 + 0.0)) < 1e-6
            assert abs(yaw - math.pi / 2) < 1e-6      # 出生朝北（+y）

    def test_mcp_lidar_map_snapshot(self):
        """MCP lidar_map 工具：GUI 面板数据源——栅格/位姿/叠加打包下发。"""
        import base64
        import zlib
        from simovia.mcp_server import SimoviaSession
        srv = SimoviaSession.__new__(SimoviaSession)  # 跳过 __init__，只测打包
        srv.sim = None
        assert srv._tool_lidar_map({}) == {"available": False}
        spec = parse_world(_nav_dict())
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            srv.sim = sim
            sim.run(max_ticks=1)
            data = srv._tool_lidar_map({})
            g = sim.lidar_mappers["car"].grid()
            sim_prob = sim.lidar_mappers["car"].prob()
        assert data["available"] is True
        m = data["maps"]["car"]
        assert (m["res"], m["nx"], m["ny"]) == (0.05, 124, 104)
        wire = np.frombuffer(zlib.decompress(base64.b64decode(m["grid"])),
                             dtype=np.uint8).reshape(m["nx"], m["ny"])
        assert np.array_equal(wire, g)            # zlib+base64 传输可逆
        pr = np.frombuffer(zlib.decompress(base64.b64decode(m["prob"])),
                           dtype=np.uint8).reshape(m["nx"], m["ny"])
        assert np.array_equal(pr, sim_prob)       # 概率通道同样可逆
        assert data["poses"]["car"] == pytest.approx(
            [-2.1, -1.6, math.pi / 2], abs=1e-3)  # 出生西南角朝北
        assert data["overlays"] == {}


class TestMapOverlayLimits:                          # ------------------------------ T7
    """publish_overlay 服务端校验：坏数据必须报错，不能静默画错图。"""

    def test_bad_overlay_rejected(self, tmp_path):
        ctrl = tmp_path / "bad_overlay.py"
        ctrl.write_text(
            "from simovia.controller import Robot\n"
            "r = Robot()\n"
            "r.step()\n"
            "try:\n"
            "    r.publish_overlay('bad', segments=[[1, 2, 3]])\n"   # 3 元非 4
            "except RuntimeError:\n"
            "    r.report_result(True, 'rejected as expected')\n"
            "else:\n"
            "    r.report_result(False, 'bad overlay was accepted')\n"
            "r.close()\n",
            encoding="utf-8")
        spec = parse_world(_nav_dict(controller=str(ctrl)))
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            sim.run(max_ticks=4)
        assert sim.controller_results["car"]["success"] is True


class TestLidarReadOnlyGuard:                     # ------------------------------ T5

    def test_scan_does_not_change_trajectory(self):
        """bit 级守护：声明雷达 + 逐 tick 全束扫描，轨迹与无雷达逐位一致。

        原地差速转圈（w_l=+4, w_r=-4）：轮地接触持续变化，扫描射线还逐 tick
        穿过自身机壳（自体排除路径高频激活）——扫描必须是纯只读派生量。
        """
        results = []
        for lidar in (False, True):
            spec = parse_world(_nav_dict(lidar=lidar))
            spec.base_dir = ROOT / "worlds"
            with Simulation(spec) as sim:
                def drive(_model, data):
                    data.ctrl[:] = [4.0, -4.0]

                probe = (lambda: sim._obs_msg("car")) if lidar else None
                sim.run(max_ticks=60, ctrl_fn=drive, on_tick=probe)
                results.append(sim.data.qpos.copy())
        q0, q1 = results
        # 严格逐位比较（allclose 会掩盖 warmstart 级漂移——项目硬约束）
        assert np.array_equal(q0, q1), \
            "雷达扫描改变了物理轨迹（扫描必须是只读派生量）"
        assert abs(q0[-1]) > 0.01                    # 轮子确实转了（守护非平凡）
