"""render 层测试：encode_png 纯 stdlib 逐字节结构校验（不依赖 EGL）。"""
import struct
import zlib

import numpy as np
import pytest

from simovia.render import encode_png


def _chunks(png: bytes):
    """逐块解 PNG：yield (tag, body)，同时校验每块长度与 CRC32。"""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"          # 签名
    off = 8
    while off < len(png):
        (ln,) = struct.unpack(">I", png[off:off + 4])
        tag, body = png[off + 4:off + 8], png[off + 8:off + 8 + ln]
        (crc,) = struct.unpack(">I", png[off + 8 + ln:off + 12 + ln])
        assert crc == zlib.crc32(tag + body), f"chunk {tag} CRC 错"
        yield tag, body
        off += 12 + ln
    assert off == len(png)                          # 无尾随垃圾


def test_encode_png_known_image():
    """2x2 已知像素：IHDR/IDAT/IEND 与像素数据全部逐字节可复算（T14）。"""
    rgb = np.array([[[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    png = encode_png(rgb)
    chunks = dict(_chunks(png))

    # IHDR：宽 2 高 2、8bit、truecolor(2)、压缩 0、滤波 0、交织 0
    assert chunks[b"IHDR"] == struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    # IDAT：解压 = 每行 filter byte 0 + 像素字节
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(2))
    assert zlib.decompress(chunks[b"IDAT"]) == raw
    assert chunks[b"IEND"] == b""                    # 收尾块为空


def test_encode_png_rejects_wrong_shape():
    import pytest
    with pytest.raises(ValueError, match="RGB"):
        encode_png(np.zeros((4, 4), dtype=np.uint8))


def test_sync_camlights_positions_scene_lights_and_cameras():
    """刚 load 就截图：mj_kinematics 不刷相机/灯位姿，必须单独补齐，
    否则相机停在原点（机位错误）、灯全停在原点（画面发暗）。"""
    import pathlib

    import mujoco

    from simovia.render import _sync_camlights
    from simovia.world.builder import build_world
    from simovia.world.schema import load_world

    root = pathlib.Path(__file__).parent.parent
    model, _ = build_world(load_world(root / "worlds" / "pick_cube.yaml"))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    _sync_camlights(model, data)
    # 世界灯在 (0,0,3) 朝下；与 mj_forward 结果一致
    assert np.allclose(data.light_xpos[0], [0, 0, 3])
    assert np.allclose(data.light_xdir[0], [0, 0, -1])
    # 相机也已就位（不再全零）
    if model.ncam:
        assert not np.allclose(data.cam_xpos[0], [0, 0, 0])


_CAM_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <geom type="box" size=".2 .2 .1" pos="0 0 .1"/>
    <body name="b" pos=".208 0 .933">
      <camera name="c" pos="0 0 0" quat="1 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.gl
def test_render_offscreen_updates_mounted_camera_pose():
    """未步进就直接截图：挂载相机位姿必须已刷新（C01 回归）。

    只做 mj_kinematics 时 cam_xpos 全零，相机在原点出图；补 mj_camlight
    后画面才对准目标。这里逐像素要求两种顺序结果一致。
    """
    import mujoco

    from simovia.render import render_offscreen

    model = mujoco.MjModel.from_xml_string(_CAM_XML)
    data = mujoco.MjData(model)
    first = render_offscreen(model, data, 160, 120, "c")
    assert np.allclose(data.cam_xpos[0], [0.208, 0, 0.933], atol=1e-6)
    data2 = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data2)
    mujoco.mj_camlight(model, data2)
    second = render_offscreen(model, data2, 160, 120, "c")
    assert np.array_equal(first, second)


@pytest.mark.gl
@pytest.mark.parametrize("near,far", [(None, None), (0.01, 2.0),
                                      (0.05, 5.0), (0.1, 10.0)])
def test_depth_is_metric_regardless_of_clip(near, far):
    """相机 z=1 m、箱顶 z=0.2 m：中心像素深度必须恒为 0.8 m（C02 回归）。"""
    import mujoco

    from simovia.render import render_offscreen

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/>'
        '<geom type="box" size=".2 .2 .1" pos="0 0 .1"/>'
        '<camera name="c" pos="0 0 1" quat="1 0 0 0"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = render_offscreen(model, data, 320, 240, "c", depth=True,
                             near=near, far=far)
    assert depth[120, 160] == pytest.approx(0.8, abs=2e-3)
    # 渲染结束后 model 的裁剪配置必须复原（不污染物理/其他渲染）
    assert model.vis.map.znear == pytest.approx(0.01)
    assert model.vis.map.zfar == pytest.approx(50.0)


@pytest.mark.parametrize("near,far", [(0.0, 5.0), (-1.0, 5.0),
                                      (0.1, 0.05), (float("nan"), 5.0)])
def test_depth_rejects_invalid_clip(near, far):
    """非法裁剪范围必须在渲染前报错（F13）：near>0 且 far>near。"""
    import mujoco

    from simovia.render import render_offscreen

    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)
    with pytest.raises(ValueError):
        render_offscreen(model, data, 32, 32, depth=True, near=near, far=far)
