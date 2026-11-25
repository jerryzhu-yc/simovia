"""sim2sim 源端：用原生 MuJoCo 独立步进 Go2 对角小跑（不经过 simovia）。

同一份 go2.xml、同一份控制律（simovia/sim2sim/go2_trot.py），只有引擎配置
由本脚本自己设置——目标端 controllers/go2_trot_sim2sim.py 跑在 simovia 里。
两边视频可以并排对比"同一控制律在不同仿真条件"下的步态差异。

跑：
    # 交互观看（MuJoCo 自带 viewer，实时 62.5Hz）
    ~/miniconda3/bin/python examples/sim2sim/mujoco_go2_trot.py --viewer

    # 录视频（离屏渲染）
    MUJOCO_GL=egl ~/miniconda3/bin/python examples/sim2sim/mujoco_go2_trot.py \
        --ticks 250 --video /tmp/go2_mujoco.mp4 --stride 1

引擎参数与 worlds/go2_trot_demo.yaml 对齐：timestep 0.002、ctrl_period 8、
integrator implicitfast、impratio 100、地面摩擦 0.8。改这些参数就等于在
源端引入物理差异（对照 worlds/go2_trot_sim2sim.yaml）。外观保留 MuJoCo
引擎自带的地面与背景（灰地面 / 黑背景），与 simovia 画面一眼可分。
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from simovia.sim2sim.go2_trot import ACTUATORS, Go2Trot  # noqa: E402

XML = ROOT / "models" / "zoo" / "unitree_go2" / "go2.xml"

# 与 worlds/go2_trot_demo.yaml 完全一致的机位（对比视频要用）；地面/背景/
# 光照刻意保留 MuJoCo 默认外观，好让两边一眼看出是不同的仿真软件
CAM_POS = [3.418, 2.428, 1.964]
CAM_QUAT = [0.200893, 0.130462, 0.528783, 0.814254]
CAM_FOVY = 50.0
FLOOR_FRICTION = 0.8


def build_model() -> mujoco.MjModel:
    """go2.xml + 地面 + 展示相机，求解器参数与目标世界一致。

    构建顺序刻意与 simovia 的 builder 一致：先建空 spec + 场景，再以 frame
    平移把机器人 attach 进来。geom 顺序会影响接触求解顺序（进而影响混沌
    轨迹），反过来先 load 机器人再补地面会让同一个控制律跑出明显不同的
    结果——那会把"仿真器差异"和"构造顺序差异"混在一起。

    外观则刻意全部保留 MuJoCo 默认：默认灰地面、无天空盒（黑背景）、
    默认 headlight、默认法线。这样对比视频里哪半是原生 MuJoCo 一眼可分。
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody

    floor = body.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15.0, 15.0, 0.1]           # 半尺寸 15m，与 YAML 一致
    floor.friction = [FLOOR_FRICTION, 0.005, 0.0001]

    # 一盏参数全默认的平行光（不改 diffuse/ambient，也无 skybox）：地面/背景
    # 仍是 MuJoCo 默认的灰面黑底，只保证机器狗不至于暗到看不清
    light = body.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    cam = body.add_camera()
    cam.name = "showcase"
    cam.pos = CAM_POS
    cam.quat = CAM_QUAT
    cam.fovy = CAM_FOVY

    # 与 worlds/go2_trot_demo.yaml 的 robots[].pos 一致（出生贴地）
    frame = body.add_frame(name="go2_frame", pos=[0.0, 0.0, 0.45])
    spec.attach(mujoco.MjSpec.from_file(str(XML)), prefix="go2/", frame=frame)

    spec.option.timestep = 0.002             # 0.002 x 8 = 0.016s -> 62.5Hz
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.impratio = 100.0
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser(description="Go2 CPG 行走（原生 MuJoCo 侧）")
    ap.add_argument("--ticks", type=int, default=None,
                    help="控制 tick 数（默认 250=4s；加 --viewer 时默认一直跑）")
    ap.add_argument("--nsub", type=int, default=8, help="每个控制 tick 的物理步数")
    ap.add_argument("--video", default=None, help="输出 mp4（需要 ffmpeg）")
    ap.add_argument("--viewer", action="store_true",
                    help="打开 MuJoCo 自带交互窗口，实时观看（Ctrl+C 或关窗退出）")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--stride", type=int, default=2, help="每 N 个控制 tick 存一帧")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    model = build_model()
    # go2.xml 的 <visual><global> 默认离屏尺寸 640x480，渲染 960x720 要抬上去
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    model.vis.quality.offsamples = 8   # 与 builder 的 MSAA 档位一致
    data = mujoco.MjData(model)
    # 出生位姿与 YAML 的 robots[].pos [0,0,0.45] 一致：直腿贴地
    data.qpos[:] = model.qpos0
    data.qpos[:3] = [0.0, 0.0, 0.45]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[mujoco] birth z={data.qpos[2]:.3f}", flush=True)

    if model.nu != len(ACTUATORS):
        raise RuntimeError(f"执行器数量不对：nu={model.nu}，期望 {len(ACTUATORS)}")

    dt = model.opt.timestep * args.nsub
    gait = Go2Trot(dt)
    print(f"[mujoco] dt={dt:.4f}s ({1.0 / dt:.1f}Hz) nu={model.nu} "
          f"integrator=implicitfast impratio={model.opt.impratio:.0f}", flush=True)

    writer = None
    if args.video:
        if shutil.which("ffmpeg") is None:
            raise SystemExit("找不到 ffmpeg，无法写视频")
        fps = 1.0 / (dt * args.stride)
        writer = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.width}x{args.height}", "-r", f"{fps:.4f}", "-i", "-",
            "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", args.video,
        ], stdin=subprocess.PIPE)
    renderer = mujoco.Renderer(model, args.height, args.width) if writer else None

    viewer = None
    if args.viewer:
        import mujoco.viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "showcase")
        viewer.sync()

    # 与控制器同序：先提交上一 tick 的 ctrl，再按新观测算下一 tick 力矩
    ctrl = np.zeros(len(ACTUATORS))
    q_birth = None
    start_x = 0.0
    wall0 = time.time()
    n_ticks = args.ticks if args.ticks is not None else (
        10 ** 9 if args.viewer else 250)
    try:
        for tick in range(n_ticks):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data, nstep=args.nsub)
            q = data.qpos[7:].copy()
            qd = data.qvel[6:].copy()
            if q_birth is None:
                q_birth = q.copy()
                start_x = float(data.qpos[0])
            ctrl = gait.torques(q, qd, tick, q_birth)

            if renderer is not None and tick % args.stride == 0:
                renderer.update_scene(data, camera="showcase")
                writer.stdin.write(renderer.render().astype(np.uint8).tobytes())

            if viewer is not None:
                if not viewer.is_running():
                    print("[mujoco] 查看器已关闭，结束", flush=True)
                    break
                # 实时节奏：按墙钟等，避免 62.5Hz 在这台机器上跑成几倍速
                remain = (tick + 1) * dt - (time.time() - wall0)
                if remain > 0:
                    time.sleep(remain)
                viewer.sync()

            if args.log_every and tick % args.log_every == 0:
                t = data.time
                dist = float(data.qpos[0]) - start_x
                speed = dist / t if t > 0 else 0.0
                print(f"[mujoco] t={t:5.2f}s x={data.qpos[0]:5.2f} "
                      f"y={data.qpos[1]:6.2f} z={data.qpos[2]:.3f} v={speed:4.2f}m/s",
                      flush=True)
    except KeyboardInterrupt:
        print("[mujoco] Ctrl+C，结束", flush=True)

    if viewer is not None:
        viewer.close()
    if renderer is not None:
        renderer.close()
        writer.stdin.close()
        writer.wait()
        print(f"[mujoco] 视频已保存: {args.video}", flush=True)

    t = data.time
    dist = float(data.qpos[0]) - start_x
    speed = dist / t if t > 0 else 0.0
    print(f"[mujoco] done t={t:.2f}s x={data.qpos[0]:.2f} y={data.qpos[1]:.2f} "
          f"distance={dist:.2f}m avg_v={speed:.2f}m/s z={data.qpos[2]:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
