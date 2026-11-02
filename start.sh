#!/usr/bin/env bash
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
elif [[ -x venv/bin/python ]]; then
    PYTHON=venv/bin/python
else
    PYTHON=python3
fi

if ! "$PYTHON" -c "import mujoco, zmq" >/dev/null 2>&1; then
    echo "缺少依赖（mujoco / pyzmq 等）：当前解释器 $PYTHON 里没找到。" >&2
    echo "请先建虚拟环境并装依赖：" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

exec "$PYTHON" -m agentsim "$@"
