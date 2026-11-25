#!/usr/bin/env bash
# Simovia Linux 一键配置：创建 .venv、安装依赖、自检、注册桌面图标。
#
# 用法：./install.sh [--no-shortcut]
#   --no-shortcut  只配置运行环境，不写桌面/应用菜单图标（无桌面环境时用）
set -Eeuo pipefail

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$DIR"

SHORTCUT=1
for arg in "$@"; do
    case "$arg" in
        --no-shortcut) SHORTCUT=0 ;;
        -h|--help)
            echo "用法: ./install.sh [--no-shortcut]"
            exit 0 ;;
        *)
            echo "未知参数: $arg（可用: --no-shortcut）" >&2
            exit 2 ;;
    esac
done

die() {
    echo >&2
    echo "[错误] $*" >&2
    exit 1
}

echo "============================================"
echo " Simovia 一键配置（Linux）"
echo " 安装目录：$DIR"
echo "============================================"
echo

# ---------------------------------------------------------------- 1. Python
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
            >/dev/null 2>&1; then
        PY=$cand
        break
    fi
done
if [[ -z "$PY" ]]; then
    die "没有找到 Python 3.11 或更高版本。
  Ubuntu / Debian:  sudo apt install python3 python3-venv python3-pip
  Fedora:           sudo dnf install python3 python3-pip
  其他发行版请安装 Python 3.12（推荐）后重新运行本脚本。"
fi
echo "[1/4] Python：$("$PY" -c 'import platform,sys; print(platform.python_implementation(), sys.version.split()[0])')（$PY）"

# ------------------------------------------------------------ 2. 虚拟环境
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
    echo "[2/4] 复用已有虚拟环境 .venv"
else
    rm -rf .venv
    echo "[2/4] 创建虚拟环境 .venv ..."
    "$PY" -m venv .venv || die "创建虚拟环境失败。
  Ubuntu / Debian 需要先安装：sudo apt install python3-venv"
fi
VPY="$DIR/.venv/bin/python"

# ---------------------------------------------------------------- 3. 依赖
echo "[3/4] 安装依赖（首次需要联网，约几分钟）..."
"$VPY" -m pip install --upgrade pip --disable-pip-version-check
"$VPY" -m pip install --disable-pip-version-check \
    -r requirements-core.txt \
    -r requirements-gui.txt \
    -r demo/fruit_sorting/requirements.txt
"$VPY" -m pip check || echo "[提示] pip check 有告警，通常不影响启动。"

# ------------------------------------------------------- 4. 自检 + 图标
echo "[4/4] 环境自检 ..."
if ! "$VPY" -m simovia doctor; then
    echo
    echo "[提示] 诊断有未通过项。若只是显卡/离屏渲染告警，可先启动试试；"
    echo "       缺少必需依赖时请把上面的输出反馈给维护者。"
fi

# Qt 的 xcb 平台插件依赖一批系统库，缺失时双击图标会静默失败，
# 这里提前检查一次并给出发行版安装命令。
QXCB=$("$VPY" - <<'PY' 2>/dev/null || true
import pathlib
import PySide6
print(pathlib.Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms" / "libqxcb.so")
PY
)
if [[ -n "$QXCB" && -f "$QXCB" ]] && command -v ldd >/dev/null 2>&1; then
    MISSING=$(ldd "$QXCB" 2>/dev/null | grep 'not found' | awk '{print $1}' | paste -sd' ' - || true)
    if [[ -n "$MISSING" ]]; then
        echo
        echo "[提示] Qt 缺少系统图形库：$MISSING"
        echo "  Ubuntu / Debian 可尝试："
        echo "    sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \\"
        echo "                     libxkbcommon-x11-0 libgl1 libegl1 libosmesa6"
        echo "  缺少这些库时桌面图标可能点不开，先在终端运行 ./run.sh 查看报错。"
    fi
fi

if [[ $SHORTCUT -eq 1 ]]; then
    DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
    APPS="$DATA/applications"
    ICON_DIR="$DATA/icons/hicolor/256x256/apps"
    mkdir -p "$APPS" "$ICON_DIR"
    cp -f assets/simovia.png "$ICON_DIR/simovia.png"
    ICON="$ICON_DIR/simovia.png"

    ENTRY="$APPS/simovia.desktop"
    cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Simovia
Name[zh_CN]=Simovia 机器人仿真
Comment=轻量机器人仿真工作台（MuJoCo + 多进程控制器）
Exec="$DIR/run.sh"
Path=$DIR
Icon=$ICON
Terminal=false
Categories=Development;Science;Education;
StartupNotify=true
EOF
    chmod 755 "$ENTRY"

    DESK=""
    if command -v xdg-user-dir >/dev/null 2>&1; then
        DESK=$(xdg-user-dir DESKTOP 2>/dev/null || true)
    fi
    if [[ -z "$DESK" ]]; then
        for cand in "$HOME/桌面" "$HOME/Desktop"; do
            if [[ -d "$cand" ]]; then DESK=$cand; break; fi
        done
    fi
    if [[ -n "$DESK" && -d "$DESK" ]]; then
        cp -f "$ENTRY" "$DESK/Simovia.desktop"
        chmod 755 "$DESK/Simovia.desktop"
        # GNOME 需要 trusted 标记才允许双击启动
        gio set "$DESK/Simovia.desktop" metadata::trusted true >/dev/null 2>&1 || true
        echo "桌面图标：$DESK/Simovia.desktop"
    fi
    echo "应用菜单：$ENTRY"
fi

echo
echo "配置完成。双击桌面或应用菜单里的 Simovia 图标即可启动（也可以运行 ./run.sh）。"
