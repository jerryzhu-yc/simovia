"""测试全局配置：无离屏 OpenGL 环境自动跳过渲染类用例。

GitHub Actions 的 Windows 运行器没有 OpenGL 3.3 驱动，Linux 无显卡时
EGL 也可能直接 abort（C 层，无法 try/except）。凡是真正创建
mujoco.Renderer / GUI RenderView 的用例都标 ``gl``，由这里统一跳过；
探测本身隔子进程做，崩溃不会带走整个测试会话。
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gl: 需要离屏 OpenGL 上下文（无 GL 环境自动跳过）")


def pytest_collection_modifyitems(config, items):
    if not any("gl" in item.keywords for item in items):
        return
    from simovia.render import probe_offscreen_gl
    ok, _detail = probe_offscreen_gl()
    if ok:
        return
    skip = pytest.mark.skip(
        reason="无可用离屏 OpenGL 上下文（EGL/OSMesa/WGL），跳过渲染类用例")
    for item in items:
        if "gl" in item.keywords:
            item.add_marker(skip)
