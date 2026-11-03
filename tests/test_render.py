"""render 层测试：encode_png 纯 stdlib 逐字节结构校验（不依赖 EGL）。"""
import struct
import zlib

import numpy as np

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


def test_sync_lights_positions_scene_lights():
    """刚 load 就截图：mj_kinematics 不刷灯位姿，必须单独补齐，
    否则灯全停在原点，画面只剩 headlight（整片发暗、无投影）。"""
    import pathlib

    import mujoco

    from simovia.render import _sync_lights
    from simovia.world.builder import build_world
    from simovia.world.schema import load_world

    root = pathlib.Path(__file__).parent.parent
    model, _ = build_world(load_world(root / "worlds" / "pick_cube.yaml"))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    _sync_lights(model, data)
    # 世界灯在 (0,0,3) 朝下；与 mj_forward 结果一致
    assert np.allclose(data.light_xpos[0], [0, 0, 3])
    assert np.allclose(data.light_xdir[0], [0, 0, -1])
