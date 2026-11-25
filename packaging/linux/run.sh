#!/usr/bin/env bash
# Simovia 桌面启动器：由桌面图标调用，也可以直接运行 ./run.sh。
set -uo pipefail

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VPY="$DIR/.venv/bin/python"

STATE="${XDG_STATE_HOME:-$HOME/.local/state}/simovia"
mkdir -p "$STATE" 2>/dev/null || STATE="$DIR"
LOG="$STATE/launcher.log"

notify_error() {
    local msg=$1
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title=Simovia --text="$msg" >/dev/null 2>&1 || true
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --title Simovia --error "$msg" >/dev/null 2>&1 || true
    elif command -v xmessage >/dev/null 2>&1; then
        xmessage -center "$msg" >/dev/null 2>&1 || true
    fi
}

if [[ ! -x "$VPY" ]]; then
    notify_error "Simovia 还没有配置运行环境。

请先在安装目录运行：./install.sh

目录：$DIR"
    echo "未配置运行环境，请先运行 $DIR/install.sh" >&2
    exit 1
fi

cd -- "$DIR" || exit 1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$VPY" -m simovia gui >>"$LOG" 2>&1
code=$?
if (( code != 0 )); then
    detail=$(tail -n 12 "$LOG" 2>/dev/null || true)
    notify_error "Simovia 启动失败（退出码 $code）。

日志：$LOG

$detail"
    echo "Simovia 启动失败（退出码 $code），日志：$LOG" >&2
fi
exit $code
