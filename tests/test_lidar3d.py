"""3D 多线雷达（lidar_nav3d）测试：schema 校验 -> builder 编译（dirs/
槽位/排除体）-> 读数与床裙叙事 -> PointCloudMapper 单元 -> 协议端到端。

世界用 worlds/lidar_nav3d.yaml 的真实几何（6.4m x 4.8m 居家房间 +
扫地机器人 + 8 线 3D 雷达 + 床裙 z 0.115~0.155：扫地机顶盖 0.14 必撞、
2D 雷达挂点 ~0.07 平面束恒低于裙底扫不到、只有 3D 雷达 +0.29° 仰角
通道能建出——本 demo 的核心价值叙事）。
"""
import math
import pathlib

import mujoco
import numpy as np
import pytest
import yaml

from simovia.errors import WorldError
from simovia.kernel.mapping import PointCloudMapper
from simovia.kernel.sim import Simulation
from simovia.kernel.task import lidar_frame, lidar_scan
from simovia.world.builder import build_world, make_dirs
from simovia.world.schema import parse_world

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "lidar_nav3d.yaml"


def _world_dict(controller=None):
    """载入 3D demo 世界 dict（默认无控制器）。"""
    d = yaml.safe_load(WORLD.read_text(encoding="utf-8"))
    d["robots"][0]["controller"] = controller
    return d


def _load(controller=None):
    spec = parse_world(_world_dict(controller))
    spec.base_dir = ROOT / "worlds"
    return spec


class TestLidar3DSchema:                            # ---------------------- T1

    def test_channels_fields(self):
        s = parse_world(_world_dict()).robots[0].sensors[0]
        assert s.type == "lidar" and s.channels == 16
        assert (s.vmin, s.vmax) == (-25.0, 30.0)
        assert s.rays == 60 and s.range == 3.5

    def test_channels_out_of_range(self):
        d = _world_dict()
        d["robots"][0]["sensors"][0]["channels"] = 100
        with pytest.raises(WorldError, match="垂直线数需在 1..64"):
            parse_world(d)

    def test_vspan_inverted(self):
        d = _world_dict()
        d["robots"][0]["sensors"][0]["vmin"] = 14.0   # vmin == vmax 边界外
        d["robots"][0]["sensors"][0]["vmax"] = -10.0
        with pytest.raises(WorldError, match="垂直角需满足"):
            parse_world(d)

    def test_pointcloud_slots(self):
        spec = parse_world(_world_dict())
        assert spec.scene.pointcloud_slots == 8192

    def test_make_dirs_layout(self):
        """channel-major 平铺：第 c 条线的 60 束在 rows[c*60:(c+1)*60]。"""
        angles = np.radians(np.linspace(-180, 180, 4, endpoint=False))
        vangles = np.radians(np.linspace(-10, 14, 3))
        dirs = make_dirs(angles, vangles)
        assert dirs.shape == (12, 3)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
        # ch0 全部 -10° 仰角；ch2 全部 +14°
        assert np.allclose(np.degrees(np.arcsin(dirs[:4, 2])), -10.0)
        assert np.allclose(np.degrees(np.arcsin(dirs[8:, 2])), 14.0)
        # 同一 channel 内方位角从 -180° 起步
        a0 = math.atan2(dirs[0, 1], dirs[0, 0])
        assert a0 == pytest.approx(math.radians(-180))


class TestLidar3DBuilder:                           # ---------------------- T2

    def test_index_and_slots(self):
        """编译后索引：dim=960、dirs 表、exclude_body、8192 个槽位 geom。"""
        spec = _load()
        model, index = build_world(spec)
        six = index.robots["sweeper"].sensors["lidar"]
        li = six.lidar
        assert six.dim == 960                       # 60 rays x 16 channels
        assert li["dirs"].shape == (960, 3)
        assert li["exclude_body"] == index.pointcloud["body"]
        assert len(index.pointcloud["geoms"]) == 8192
        # 槽位是预置隐藏 box：零碰撞零质量 + alpha=0（内核运行时填充）
        gid = index.pointcloud["geoms"][0]
        assert int(model.geom_contype[gid]) == 0
        assert float(model.geom_rgba[gid][3]) == 0.0

    def test_2d_world_untouched(self):
        """老 2D 世界（lidar_nav.yaml）不带 dirs/exclude_body：逐字段确认
        3D 扩展没有污染 2D 老语义。"""
        spec = parse_world(yaml.safe_load(
            (ROOT / "worlds" / "lidar_nav.yaml").read_text(encoding="utf-8")))
        spec.base_dir = ROOT / "worlds"
        index = build_world(spec)[1]
        six = index.robots["car"].sensors["lidar"]
        assert six.dim == 60
        assert "dirs" not in six.lidar
        assert "exclude_body" not in six.lidar
        assert index.pointcloud is None


class TestLidar3DScan:                              # ---------------------- T3

    def _world(self):
        spec = _load()
        model, index = build_world(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, index

    def test_frame_channels_and_bed_skirt(self):
        """朝床扫一帧（扫地机 (-1.9,-0.9) 朝北）：960 束 + 床裙叙事分层。

        床裙（红）贴床垫南缘 y=0.04、从床垫底 0.28 往下垂到 0.10；床垫南面
        y=0.05。雷达挂点世界 z=0.165（root 0.045 + body 系偏移 0.12）。16 个
        仰角通道（-25°..+30°，3.67°/线）在床前形成多层级：
        - -25°..-14°（ch0..ch3）：近处触地，z 快速下行，D≈0.39..0.68
        - -10.3°（ch4）：远触地 D≈0.92（床前地面）
        - -6.7°（ch5）：掠裙底穿到床下地面（D≈1.42）
        - -3.0°/0.7°/4.3°（ch6/ch7/ch8）：z 落进裙带 0.10~0.28 -> 命中床裙
          D≈0.93
        - 8.0°/11.7°/15.3°（ch9/ch10/ch11）：越裙顶打床垫南面 D≈0.96~0.99
        - 19°（ch12）：越过床顶打到北墙（2.4m 层高）D≈3.47
        - 22.7°..30°（ch13..ch15）：北墙已在 3.5m 量程外，无回波（-1）
        - 2D 平面雷达（挂 ~0.07 水平束）恒低于裙底 0.10：几何盲区（3D 独有）
        分层证据：近地/远地/穿裙底/裙面/垫面——多线 3D 的完整结构信息
        """
        model, data, index = self._world()
        six = index.robots["sweeper"].sensors["lidar"]
        q = data.qpos.copy()
        q[0], q[1] = -1.9, -0.9           # 床裙正南方（避开西墙距离带）
        yaw = math.pi / 2                  # 朝北（+y）：body +x -> 世界 +y
        q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out = np.asarray(lidar_scan(model, data, six))
        assert out.shape == (960,)
        b = 30                             # a=0° 束（正北朝床）
        ch = lambda c: out[c * 60 + b]
        assert 0.30 < ch(0) < 0.48        # -25°：近处触地（D≈0.39）
        assert 0.60 < ch(3) < 0.76        # -14°：地面（D≈0.68）
        assert 0.82 < ch(4) < 1.00        # -10.3°：远触地（D≈0.92）
        assert 1.30 < ch(5) < 1.52        # -6.7°：掠裙底穿到床下地面（D≈1.42）
        assert 0.90 < ch(6) < 0.96        # -3.0°：命中床裙（D≈0.93）
        assert 0.90 < ch(7) < 0.96        #  0.7°：命中床裙
        assert 0.90 < ch(8) < 0.96        #  4.3°：命中床裙
        assert 0.93 < ch(9) < 1.02        #  8.0°：越裙顶打床垫南面（D≈0.96）
        assert 0.94 < ch(11) < 1.02       # 15.3°：床垫南面（D≈0.99）
        assert 3.30 < ch(12) < 3.50      # 19°：越床顶打北墙（2.4m 层高，D≈3.47）
        assert ch(13) < 0.0               # 22.7°：北墙已超 3.5m 量程（无回波）
        assert ch(15) < 0.0               # 30°：打空

    def test_coffee_table_underside(self):
        """茶几桌下净空重建：扫地机 (-0.75,-1.15) 朝东一帧。

        茶几桌面底 0.38、圆腿在桌面正中，东、西两侧无腿（南北两排腿间
        0.52m 门洞）。雷达从西侧俯仰扫：低仰角通道打到桌下地面（z<0.10），
        高仰角通道打到桌面底（0.32<z<0.46），中间 z 出现双峰 -> 点云能
        读出"桌下净空 0.38m、可钻入"这一判断依据（用户核心需求）。
        """
        model, data, index = self._world()
        six = index.robots["sweeper"].sensors["lidar"]
        q = data.qpos.copy()
        q[0], q[1] = -0.75, -1.15
        q[3:7] = [1.0, 0.0, 0.0, 0.0]     # body +x -> 世界 +x（朝东）
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        six_lidar = six
        fr = lidar_frame(data, six_lidar)
        origin, rot = np.asarray(fr[0]), np.asarray(fr[1])
        dists = lidar_scan(model, data, six_lidar)
        m = PointCloudMapper((2.6, 2.1))
        m.update(origin, rot, np.asarray(six_lidar.lidar["dirs"]), dists, 3.5)
        # 从体素键还原世界坐标（体素中心）
        keys = np.array(list(m.vox.keys()), dtype=float)
        centers = (keys + 0.5) * m.voxel
        cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]
        under = (cx > -0.35) & (cx < 0.75) & (cy > -1.45) & (cy < -0.85)
        floor_pts = int(np.sum(under & (cz < 0.10)))
        top_pts = int(np.sum(under & (cz > 0.32) & (cz < 0.46)))
        assert floor_pts >= 10, f"桌下地面点不足: {floor_pts}"
        assert top_pts >= 10, f"桌面底点不足: {top_pts}"

    def test_readonly_bit_guard(self):
        """bit 级只读守护：扫描前后 qpos/qvel 逐位不变（3D 分支同 2D 约定）。"""
        model, data, index = self._world()
        six = index.robots["sweeper"].sensors["lidar"]
        q0 = data.qpos.copy().tobytes()
        v0 = data.qvel.copy().tobytes()
        lidar_scan(model, data, six)
        lidar_scan(model, data, six)                 # 连扫两帧更严
        assert data.qpos.tobytes() == q0
        assert data.qvel.copy().tobytes() == v0


class TestPointCloudMapper:                         # ---------------------- T4

    def _frame(self, origin, rot, dirs, dists, rng):
        m = PointCloudMapper((2.6, 2.1))
        m.update(np.asarray(origin), np.asarray(rot), dirs, dists, rng)
        return m

    def test_band_hit_projects_occupied(self):
        """带内命中（z 0.02~0.18）投影占据：横杆类障碍进 2D 栅格。"""
        dirs = np.array([[1.0, 0.0, 0.0]])           # 正 +x 水平束
        # 原点 (0,0,0.12)，命中 2m 处 z=0.12（带内）：占据格应在 x≈2
        m = self._frame((0, 0, 0.12), np.eye(3), dirs, [2.0], 3.5)
        g = m.grid()
        ix, iy = int((2.0 - m.ox) / 0.05), int((0.0 - m.oy) / 0.05)
        assert g[ix, iy] == 2
        # 自由：0..2m 穿越路径打自由
        jx, jy = int((1.0 - m.ox) / 0.05), int((0.0 - m.oy) / 0.05)
        assert g[jx, jy] == 1

    def test_out_of_band_hit_no_occupancy(self):
        """带外命中（如桌面 z 0.72）不投影占据——桌面不挡扫地机，
        但体素点云仍收下它（3D 结构展示）。"""
        dirs = np.array([[1.0, 0.0, 0.0]])
        m = self._frame((0, 0, 0.60), np.eye(3), dirs, [2.0], 3.5)
        g = m.grid()
        ix, iy = int((2.0 - m.ox) / 0.05), int((0.0 - m.oy) / 0.05)
        assert g[ix, iy] != 2                        # 带外不进占据
        assert len(m.vox) == 1                        # 体素仍收下
        # 挂点 0.60 带外向下：束 z 恒 0.60 带外 -> 也无自由段
        assert not (g == 1).any()

    def test_slots_sync(self):
        """槽位同步：体素写进 geom_pos/size/rgba（高度色带）+ geom_xpos。"""
        m = self._frame((0, 0, 0.12), np.eye(3),
                        np.array([[1.0, 0.0, 0.0]]), [2.0], 3.5)
        model = _FakeModel()
        m.bind_slots(model, [7])
        data = _FakeData()
        m.sync_slots(data)
        assert np.allclose(model.geom_pos[7], [2.0, 0.02, 0.12], atol=0.05)
        # 渲染半尺寸 = 体素 0.04 的 0.4 倍（留缝 -> 点云颗粒感）
        assert np.allclose(model.geom_size[7], [0.016] * 3)
        assert model.geom_rgba[7][3] == 0.85         # 可见
        assert np.allclose(data.geom_xpos[7], model.geom_pos[7])  # 恒等关系

    def test_global_sample_pointcloud(self):
        """pointcloud() 全量均匀抽样：超量时仍覆盖整个建图范围，而非局部窗口。"""
        m = PointCloudMapper((2.6, 2.1))
        # 沿 +x 方向铺 5 个互不相同的体素（1..5m）
        for x in (1.0, 2.0, 3.0, 4.0, 5.0):
            m.update(np.array([0, 0, 0.12]), np.eye(3),
                     np.array([[1.0, 0.0, 0.0]]), [x], 5.5)
        assert len(m.vox) == 5
        pts = m.pointcloud(2)                         # 限 2：按 stride=3 抽样
        assert len(pts) == 2
        assert pts[0][0] < pts[1][0]                  # 前向序（非尾部倒序）
        assert pts[1][0] > 2.0                        # 抽样点跨越广度而非聚在一端
        # 体素数未超限：全量返回
        assert len(m.pointcloud(10)) == 5


class _FakeModel:
    """sync_slots 的最小 model 仿件：numpy 数组即可索引赋值。"""

    def __init__(self, n=16):
        self.geom_pos = np.zeros((n, 3))
        self.geom_size = np.zeros((n, 3))
        self.geom_rgba = np.zeros((n, 4))


class _FakeData:
    def __init__(self, n=16):
        self.geom_xpos = np.zeros((n, 3))


class TestLidar3DProtocol:                          # ---------------------- T5

    def test_manifest_and_readings(self, tmp_path):
        """真实 ZMQ 协议端到端：dim=960 + 3D 读数 + publish_overlay。"""
        ctrl = tmp_path / "assert_lidar3d.py"
        ctrl.write_text(
            "from simovia.controller import Robot\n"
            "import numpy as np\n"
            "r = Robot()\n"
            "dims = {s['name']: s['dim'] for s in r.sensors}\n"
            "assert dims['lidar'] == 960, dims\n"
            "obs = r.step()\n"
            "scan = r.sensor('lidar')\n"
            "assert len(scan) == 960, len(scan)\n"
            "hit = [d for d in scan if d > 0]\n"
            "assert len(hit) > 50, len(hit)          # 静止一帧也有大量回波\n"
            "r.publish_overlay('goal', points=[[2.1, 1.6]])\n"
            "r.report_result(True, 'lidar3d ok')\n",
            encoding="utf-8")
        spec = _load(str(ctrl))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=6)
        assert stats["crashed"] == []
        assert sim.controller_results["sweeper"]["success"] is True

    def test_kernel_mapper_and_pc_snapshot(self):
        """内核自动建图：3D 世界建 PointCloudMapper + 槽位镜像 + pointcloud 序。"""
        spec = _load()
        with Simulation(spec) as sim:
            sim.run(max_ticks=30)
            m = sim.lidar_mappers["sweeper"]
            assert isinstance(m, PointCloudMapper)
            assert len(m.vox) > 100                  # 体素在累积
            g = m.grid()
            assert np.count_nonzero(g == 2) > 0      # 占据（墙/障碍）
            assert np.count_nonzero(g == 1) > 0      # 自由（穿越）
            # 槽位镜像：model.geom_pos 持久有效；data.geom_xpos 会被后续
            # mj_step 按编译期缓存覆盖回 0，渲染前重刷一拍（mcp_server
            # ._refresh_pc_slots 同款约定）写完即读才有效
            gid = sim.index.pointcloud["geoms"][0]
            m.sync_slots(sim.data)
            assert np.allclose(sim.data.geom_xpos[gid],
                               sim.model.geom_pos[gid], atol=1e-9)
            # pointcloud() 是 [x,y,z] 列表（MCP pc 段的打包源）
            pts = m.pointcloud(4)
            assert len(pts) == 4 and len(pts[0]) == 3
