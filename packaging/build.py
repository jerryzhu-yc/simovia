#!/usr/bin/env python3
"""把 Simovia 打成可直接分发的桌面软件包（Linux / Windows）。

用法：
    python packaging/build.py             # 两个平台都打
    python packaging/build.py linux       # 只打 Linux
    python packaging/build.py windows     # 只打 Windows

产物在 dist/ 下：
    Simovia-<版本>-linux-x64.zip
    Simovia-<版本>-windows-x64.zip

包内已自带机器人模型、示例世界和安装/启动脚本；解压后按 INSTALL.txt
配置一次环境，之后双击桌面图标即可启动。
"""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

# 随包内容：运行时真正需要的目录与文件（不含站点、测试、录制产物）
COMMON_DIRS = ["simovia", "models", "worlds", "controllers", "demo",
               "examples", "docs"]
COMMON_FILES = ["README.md", "requirements.txt", "requirements-core.txt",
                "requirements-gui.txt", "requirements-models.txt",
                "requirements-rl.txt"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".idea", ".vscode"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

NOTES_LINUX = """Simovia 机器人仿真平台 v{version}（Linux x64）
==================================================

一、准备
  需要 64 位 Python 3.11 或更高版本（推荐 3.12）。
  检查：终端执行  python3 --version
  没有的话（Ubuntu / Debian）：sudo apt install python3 python3-venv python3-pip

二、配置环境（只需要一次）
  在本文件夹打开终端，执行：
      ./install.sh
  脚本会自动创建 .venv、安装依赖、做环境自检，并在桌面和应用菜单
  创建 Simovia 图标。安装依赖需要联网（约几分钟）；模型资产已随包
  自带，之后运行和模型库都可离线使用。

三、启动
  双击桌面上的 Simovia 图标，或从应用菜单选择 Simovia。
  也可以在本文件夹执行：
      ./run.sh

四、常见问题
  - 提示缺少 Python：按第一条安装后重新执行 ./install.sh。
  - 双击图标没反应：在终端执行 ./run.sh 查看具体报错。
  - 3D 画面黑屏 / OpenGL 报错：先更新显卡驱动；没有可用的 GPU 时
    可改用软件渲染：
        sudo apt install libosmesa6
        MUJOCO_GL=osmesa ./run.sh
  - 启动日志：~/.local/state/simovia/launcher.log
  - 移动或改名本文件夹后桌面图标会失效：在新位置重新执行 ./install.sh。
  - 卸载：删除本文件夹，再删除桌面图标与
    ~/.local/share/applications/simovia.desktop。

自带的 35 个机器人模型在 models/zoo/，示例世界在 worlds/。
更多用法见 README.md。
"""

NOTES_WINDOWS = """Simovia 机器人仿真平台 v{version}（Windows x64）
===================================================

一、准备
  需要 64 位 Python 3.11 或更高版本（推荐 3.12）。
  下载地址：https://www.python.org/downloads/windows/
  安装时务必勾选 "Add python.exe to PATH"。

二、配置环境（只需要一次）
  双击本文件夹里的  install-windows.bat
  脚本会自动创建 .venv、安装依赖、做环境自检，并在桌面和开始菜单
  创建 Simovia 快捷方式。安装依赖需要联网（约几分钟）；模型资产已
  随包自带，之后运行和模型库都可离线使用。

三、启动
  双击桌面上的 Simovia 图标，或从开始菜单选择 Simovia。
  也可以双击本文件夹里的 Simovia.bat（会保留控制台输出，便于排错）。

四、常见问题
  - 提示找不到 Python：安装好 Python 后重新双击 install-windows.bat。
  - 双击图标没有反应：双击 Simovia.bat 查看报错信息。
  - 3D 画面黑屏：更新显卡驱动（需要 OpenGL 3.3 或更高）。
  - 移动或改名本文件夹后快捷方式会失效：
    在新位置重新双击 install-windows.bat。
  - 卸载：删除本文件夹，再删除桌面和开始菜单里的 Simovia 快捷方式。

自带的 35 个机器人模型在 models\\zoo\\，示例世界在 worlds\\。
更多用法见 README.md。
"""


def read_version() -> str:
    text = (ROOT / "simovia" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("找不到 simovia/__init__.py 里的 __version__")
    return match.group(1)


def _iter_common():
    for name in COMMON_DIRS:
        base = ROOT / name
        if not base.is_dir():
            raise SystemExit(f"缺少目录：{base}")
        print(f"  收集 {name}/ ...", flush=True)
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if path.suffix in SKIP_SUFFIXES or path.name.startswith("."):
                continue
            yield path, rel.as_posix()
    for name in COMMON_FILES:
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"缺少文件：{path}")
        yield path, name


def _crlf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def _write(zf: zipfile.ZipFile, arc: str, data: bytes, mode: int,
           create_system: int) -> None:
    info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = create_system
    info.external_attr = (mode & 0xFFFF) << 16
    zf.writestr(info, data)


def build(platform: str, version: str) -> Path:
    if platform == "linux":
        tag = "linux-x64"
        create_system = 3
        crlf_bats = False
        extras = [
            (ROOT / "packaging/linux/install.sh", "install.sh", 0o755),
            (ROOT / "packaging/linux/run.sh", "run.sh", 0o755),
            (ROOT / "packaging/assets/simovia.png", "assets/simovia.png", 0o644),
        ]
        notes = NOTES_LINUX.format(version=version)
    elif platform == "windows":
        tag = "windows-x64"
        create_system = 0
        crlf_bats = True
        extras = [
            (ROOT / "packaging/windows/install.bat", "install-windows.bat", 0o644),
            (ROOT / "packaging/windows/Simovia.bat", "Simovia.bat", 0o644),
            (ROOT / "packaging/assets/simovia.ico", "simovia.ico", 0o644),
        ]
        notes = NOTES_WINDOWS.format(version=version)
    else:
        raise SystemExit(f"未知平台：{platform}")

    top = f"Simovia-{version}-{tag}"
    out = DIST / f"{top}.zip"
    DIST.mkdir(exist_ok=True)
    if out.exists():
        out.unlink()

    print(f"\n打包 {out.relative_to(ROOT)} ...", flush=True)
    total = 0
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for src, rel in _iter_common():
            data = src.read_bytes()
            if crlf_bats and src.suffix.lower() in {".bat", ".cmd"}:
                data = _crlf(data)
            _write(zf, f"{top}/{rel}", data, 0o644, create_system)
            total += len(data)
            count += 1
            if count % 250 == 0:
                print(f"    {count} 个文件 ...", flush=True)

        for src, arc, mode in extras:
            if not src.is_file():
                raise SystemExit(f"缺少打包资源：{src}")
            data = src.read_bytes()
            if crlf_bats and src.suffix.lower() in {".bat", ".cmd"}:
                data = _crlf(data)
            _write(zf, f"{top}/{arc}", data, mode, create_system)

        notes = notes.replace("\r\n", "\n")
        if crlf_bats:
            notes = notes.replace("\n", "\r\n")
        _write(zf, f"{top}/INSTALL.txt", notes.encode("utf-8"),
               0o644, create_system)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"完成：{out}（{count + len(extras) + 1} 个文件，"
          f"原始 {total / 1024 / 1024:.0f} MB，压缩后 {size_mb:.0f} MB）")
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if any(a in {"-h", "--help"} for a in argv):
        print(__doc__)
        return 0
    targets = argv or ["linux", "windows"]
    bad = [t for t in targets if t not in {"linux", "windows"}]
    if bad:
        raise SystemExit(f"未知平台：{', '.join(bad)}（可用：linux / windows）")

    version = read_version()
    print(f"Simovia 桌面安装包 v{version}")
    built = [build(t, version) for t in targets]

    sums = DIST / "SHA256SUMS.txt"
    sums.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in built),
                    encoding="utf-8")
    print(f"\n校验值：{sums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
