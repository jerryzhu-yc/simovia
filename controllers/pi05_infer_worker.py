"""π0.5 (openpi) 推理 worker：由 g1_cloth_policy_eval 控制器拉起。

为什么是独立进程：共享控制器（shared_world_controller）跑在内核进程里，
用的是装了 mujoco 的解释器；π0.5 需要 openpi 的 .venv（jax/flax），两套
依赖互斥，只能子进程隔离。这里用 stdin/stdout 走定长二进制协议，避免
JSON+base64 编码 256×256 图像的开销。

协议（全部小端）：
    请求  b"A" + <I n> + payload
          payload = <I state_dim><I h><I w><I n_cam>
                    + state(f4 × state_dim)
                    + n_cam 张 HWC uint8 图像（按请求顺序）
    退出  b"Q"
    响应  <I n> + actions(f4 × T × state_dim)

stdout 只走协议，所有日志写 stderr。

用法（由控制器按世界 YAML 的 args 拼参数）：
    python pi05_infer_worker.py --checkpoint DIR [--config NAME]
        [--prompt TEXT] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

# 与控制器约定：图像按这个顺序发过来
CAM_KEYS = ("cam_high", "cam_left_wrist")


def read_exact(stream, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_g1_cloth_lora")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="put the blanket in the box")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--mem-fraction", type=float, default=0.55,
                    help="cuda 下预分配显存比例；5070 只有 12G，要和桌面其他进程共存")
    args = ap.parse_args()

    # 必须在 import jax 之前定平台；5070 显存常被其他进程占着，默认 CPU
    os.environ.setdefault("JAX_PLATFORMS", args.device)
    if args.device == "cuda":
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",
                              str(args.mem_fraction))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[pi05] 加载 {args.config} <- {args.checkpoint} ({args.device})",
          file=sys.stderr, flush=True)
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint)
    print("[pi05] 模型就绪", file=sys.stderr, flush=True)

    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        cmd = inp.read(1)
        if not cmd or cmd == b"Q":
            break
        if cmd != b"A":
            continue
        head = read_exact(inp, 4)
        if head is None:
            break
        (size,) = struct.unpack("<I", head)
        payload = read_exact(inp, size)
        if payload is None:
            break

        state_dim, h, w, n_cam = struct.unpack("<IIII", payload[:16])
        off = 16
        state = np.frombuffer(payload[off:off + 4 * state_dim],
                              dtype="<f4").astype(np.float32)
        off += 4 * state_dim
        images = {}
        for key in CAM_KEYS[:n_cam]:
            frame = np.frombuffer(payload[off:off + h * w * 3],
                                  dtype=np.uint8).reshape(h, w, 3)
            off += h * w * 3
            # openpi 的 _DecodeG1Images 期望 CHW float32 [0,1]（lerobot 解码格式）
            images[key] = (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)

        obs = {"image": images, "state": state, "prompt": args.prompt}
        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)[:, :state_dim]
        actions = np.ascontiguousarray(actions, dtype="<f4")
        out.write(struct.pack("<I", actions.nbytes) + actions.tobytes())
        out.flush()


if __name__ == "__main__":
    main()
