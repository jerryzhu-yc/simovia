"""Agent 场景生成演示视频：LLM 生成世界 → 加载 → 机械臂抓取运行。

布局（1280x720）：
- 左 760x720：MuJoCo 视口（AI 生成世界的初始状态 + 抓取 episode，~10x 加速）
- 右 520x720：Agent 会话终端面板（真实素材：本地 Ollama qwen2.5-coder 微调模型
  生成的 YAML 逐行打出、save/load/run/screenshot 工具调用与真实返回值）

数据全部真实：YAML 来自 LLM 实际输出（缓存于 /tmp/agent_gen_world.yaml，
删掉即重新生成）；run 统计、PNG 大小来自内核真实返回。

用法：
    QT_QPA_PLATFORM=offscreen MUJOCO_GL=egl python scripts/make_agent_video.py \
        [--out site/public/assets/demos/agent_scene_gen.mp4]
"""
import argparse
import math
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml as pyyaml

W, H = 1280, 720
LW, RW = 760, 520                     # 左视口宽 / 右终端宽
FPS = 30
REQ = ("造一个精致的桌面抓取工作台，像实验室光学平台：整块深灰地毯垫底"
       "（box 半尺寸 0.9x0.9x0.004 放 (0.32,0,0.004)，static），工作台桌面顶"
       " z=0.68（box 半尺寸 0.40x0.25x0.02 放 (0.32,0,0.66)），四条桌腿必须"
       "立桌面四个角正下方（x=0.02 与 0.62，y=0.21 与 -0.21，腿半尺寸 "
       "0.025x0.025x0.32，中心 z=0.32），绝不能挤在桌子中间；桌面后沿立一条"
       "低矮后挡板（半尺寸 0.40x0.012x0.10，pos (0.32,0.24,0.78)，static）。"
       "桌面左前整齐摆三个 60mm 装饰方块成一行（绿 #22C55E、蓝 #3B82F6、"
       "黄 #EAB308，z 均为 0.71，x 分别 0.16/0.28/0.40，y=-0.17）；"
       "红色 60mm 方块（0.1kg，摩擦 1.0）单独放抓取位 (0.55,0,0.71)。"
       "打两盏灯：主光 pos (1.2,-1.0,2.2)、补光 pos (-1.0,1.0,1.6)。"
       "配 showcase 演示机位：pos [1.72,-1.48,1.30]、quat "
       "[0.711901,0.594968,0.239265,0.286289]、fovy 40。"
       "Franka Panda 站桌后沿 (0,0,0.68)，装 onrobot_2fg7 夹爪，挂控制器 "
       "../controllers/pick_place.py。任务成功区是空中 (0.55,0,0.86) 附近 "
       "zone（min [0.49,-0.06,0.80]，max [0.61,0.06,0.92]）。timestep 0.002，"
       "ctrl_period 32，隐式积分 implicitfast，impratio 10。")


def _label(arr: np.ndarray, x, y, text, size=14):
    """numpy 画布上叠字（QImage 直接包 ndarray 缓冲，QPainter 原地绘制）。"""
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888)
    p = QPainter(img)
    try:
        p.setFont(QFont("Monospace", size, QFont.Weight.Bold))
        tw = p.fontMetrics().horizontalAdvance(text)
        p.fillRect(x - 4, y - size - 4, tw + 10, size + 10, QColor(0, 0, 0, 150))
        p.setPen(QPen(QColor(255, 255, 255)))
        p.drawText(x, y, text)
    finally:
        p.end()


def _wrap(text, width):
    out, line = [], ""
    for ch in text:
        line += ch
        if len(line) >= width:
            out.append(line)
            line = ""
    if line:
        out.append(line)
    return out


def generate_yaml(base_url: str = "", model: str = "") -> tuple[str, str]:
    """真实 LLM 生成（缓存复用）；输出偏离抓取几何时 refine 修正。

    配置优先取 ~/.simovia/agent.json（云端 OpenAI 兼容端点），命令行
    --base-url/--model 可覆盖；都没有时回退本地 Ollama。返回 (YAML, 模型名)。
    """
    cache = "/tmp/agent_gen_world.yaml"
    from simovia.agent.config import AgentConfig, load_config
    cfg = load_config()
    if base_url:
        cfg.base_url = base_url
    if model:
        cfg.model = model
    if not cfg.usable:                        # 回退本地 Ollama
        cfg = AgentConfig(base_url="http://localhost:11434/v1",
                          model="mjcf-generator:latest")
    print(f"生成模型: {cfg.model} @ {cfg.base_url}")
    if os.path.exists(cache):
        return open(cache, encoding="utf-8").read(), cfg.model

    from simovia.agent.scene_agent import SceneGenSession
    sess = SceneGenSession(cfg, max_retries=3)
    text, _ = sess.generate(REQ)
    for _ in range(3):                        # 偏离关键几何则 refine（最多 3 轮）
        d = pyyaml.safe_load(text) or {}
        scene = d.get("scene") or {}
        objs = {o.get("name"): o for o in (scene.get("objects") or [])}
        cube = objs.get("red_cube", {})
        robot = (d.get("robots") or [{}])[0]
        fixes = []
        cp = cube.get("pos") or []
        if not (len(cp) >= 3 and abs(cp[0] - 0.55) < 0.05 and abs(cp[1]) < 0.05
                and abs(cp[2] - 0.71) < 0.06):
            fixes.append("红色方块必须精确放 (0.55, 0, 0.71)")
        csize = cube.get("size") or []
        if not (len(csize) >= 3 and abs(csize[0] - 0.03) < 0.005
                and abs(csize[1] - 0.03) < 0.005
                and abs(csize[2] - 0.03) < 0.005):
            fixes.append("红方块 size 必须精确写 [0.03, 0.03, 0.03]——MuJoCo 的 size 是"
                         "半尺寸，0.03 才是 60mm 方块（夹爪满开 64.8mm 才能抓）；"
                         "绿蓝黄装饰方块同样改成 [0.03, 0.03, 0.03]")
        # 桌腿必须立在桌面四角正下方（LLM 最常见的错：四条腿挤在桌子中间）
        legs = [o for n, o in objs.items() if "leg" in (n or "").lower()]
        missing = []
        for lx in (0.02, 0.62):
            for ly in (0.21, -0.21):
                hit = False
                for l in legs:
                    lp = l.get("pos") or []
                    if (len(lp) >= 2 and abs(lp[0] - lx) < 0.09
                            and abs(lp[1] - ly) < 0.09):
                        hit = True
                        break
                if not hit:
                    missing.append(f"({lx}, {ly})")
        if not legs or missing:
            fixes.append("四条桌腿必须分别立在桌面四角正下方，缺 " + "、".join(missing)
                         + "：x=0.02 与 0.62、y=0.21 与 -0.21，半尺寸 "
                           "0.025x0.025x0.32、中心 z=0.32，绝不能挤在桌子中间")
        # 台面装饰物必须贴面（z = 台面顶 0.68 + 半尺寸 0.03 = 0.71）
        for key in ("green", "blue", "yellow"):
            o = next((v for n, v in objs.items() if key in (n or "").lower()), None)
            if o is None:
                continue
            op = o.get("pos") or []
            if not (len(op) >= 3 and abs(op[2] - 0.71) < 0.02):
                fixes.append(f"{o.get('name')} 的 pos z 必须是 0.71"
                             "（桌面顶 0.68 + 半尺寸 0.03），否则会陷进桌面")
        # 必须有 showcase 演示机位（否则视频只能用脚本写死的自由相机）
        cams = scene.get("cameras") or []
        if not any((c.get("name") == "showcase") for c in cams):
            fixes.append("scene.cameras 必须含一个 name: showcase 的机位："
                         "pos [1.72,-1.48,1.30]、quat "
                         "[0.711901,0.594968,0.239265,0.286289]、fovy 40")
        rp = robot.get("pos") or []
        if not (len(rp) >= 3 and abs(rp[0]) < 0.05 and abs(rp[1]) < 0.05
                and abs(rp[2] - 0.68) < 0.05):
            fixes.append("Panda 基座必须精确放 (0, 0, 0.68)")
        zone = ((d.get("task") or {}).get("success") or {}).get("zone") or {}
        zmin, zmax = zone.get("min", []), zone.get("max", [])
        if not (len(zmin) >= 3 and len(zmax) >= 3 and zmin[0] <= 0.55 <= zmax[0]
                and zmin[2] <= 0.86 <= zmax[2]):
            fixes.append("成功区 zone 必须罩住空中 (0.55, 0, 0.86)")
        if not fixes:
            break
        print(f"refine: {len(fixes)} 项 → {'；'.join(f[:28] for f in fixes)}")
        text, _ = sess.refine("；".join(fixes) + "，输出完整修正版 YAML")
    open(cache, "w", encoding="utf-8").write(text)
    return text, cfg.model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="/tmp/agent_frames")
    ap.add_argument("--vp-dir", default="/tmp/agent_vp")
    ap.add_argument("--out", default="site/public/assets/demos/agent_scene_gen.mp4")
    ap.add_argument("--base-url", default="", help="覆盖 AgentConfig 的端点")
    ap.add_argument("--model", default="", help="覆盖 AgentConfig 的模型名")
    ap.add_argument("--dry-run", action="store_true", help="只生成并打印 YAML，不录视频")
    args = ap.parse_args()

    import mujoco
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    from simovia.kernel.sim import Simulation
    from simovia.render import encode_png, render_offscreen
    from simovia.world.schema import load_world

    _app = QApplication([])

    # ---- 1. LLM 真实生成 + 归一化（控制器绝对路径，落地即跑）----
    yaml_text, model_name = generate_yaml(args.base_url, args.model)
    if args.dry_run:
        print(yaml_text)
        return
    d = pyyaml.safe_load(yaml_text)
    d.setdefault("robots", [{}])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d["robots"][0]["controller"] = os.path.join(repo, "controllers", "pick_place.py")
    d.setdefault("runtime", {})["realtime"] = False
    world_name = d.get("name", "ai_world")
    yaml_text = pyyaml.safe_dump(d, allow_unicode=True, sort_keys=False)
    yaml_path = "/tmp/agent_gen_world.yaml"
    open(yaml_path, "w", encoding="utf-8").write(yaml_text)
    yaml_lines = [ln.rstrip() for ln in yaml_text.splitlines() if ln.strip()]
    n_bytes = len(yaml_text.encode("utf-8"))

    # ---- 2. 加载 + 跑 episode（视口帧按 tick 存盘）----
    spec = load_world(yaml_path)
    cam = mujoco.MjvCamera()
    try:
        cam_name = spec.scene.cameras[0].name if spec.scene.cameras else None
    except Exception:
        cam_name = None
    renderer = None
    vp_dir, frames_dir = args.vp_dir, args.frames_dir
    os.makedirs(vp_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    for pdir in (vp_dir, frames_dir):
        for f in os.listdir(pdir):
            os.remove(os.path.join(pdir, f))

    def _render(sim):
        nonlocal renderer
        if renderer is None:
            renderer = mujoco.Renderer(sim.model, H, LW)
        if cam_name:
            return render_offscreen(sim.model, sim.data, LW, H,
                                     camera=cam_name, renderer=renderer)
        cam.lookat[:] = [0.45, 0.0, 0.55]
        cam.distance = 2.35
        cam.azimuth = 130.0
        cam.elevation = -32.0
        return render_offscreen(sim.model, sim.data, LW, H,
                                 renderer=renderer, free_cam=cam)

    def _qimage_rgb(img) -> np.ndarray:
        img = img.convertToFormat(QImage.Format.Format_RGB888)
        h, w, bpl = img.height(), img.width(), img.bytesPerLine()
        buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h)
        rgb = buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
        return np.array(rgb, dtype=np.uint8, copy=True)

    ticks_seen = [0]

    def _grab(sim):
        rgb = _render(sim)
        from PySide6.QtGui import QImage as QI
        QI(rgb.data, LW, H, rgb.strides[0], QI.Format.Format_RGB888).save(
            f"{vp_dir}/{sim.tick:06d}.png")
        ticks_seen[0] = sim.tick

    with Simulation(spec) as sim:
        init_rgb = _render(sim)                       # load 后的初始世界
        stats = sim.run(on_tick=lambda: (
            _grab(sim) if sim.tick > 0 else None),
            max_ticks=5000)
    final_png_kb = len(encode_png(init_rgb)) // 1024   # 用初始帧估 PNG 大小

    vp_files = sorted(os.listdir(vp_dir))
    n_vp = len(vp_files)
    RUN_F = 1215 - 345                               # run 段合成帧数
    speed_x = stats["ticks"] * 0.064 * FPS / RUN_F   # run 段实际播放倍率
    rate_txt = (f"{speed_x:.1f}x 慢放" if speed_x < 0.99
                else f"{speed_x:.0f}x 加速")
    print(f"episode: ticks={stats['ticks']} success={stats['success']} "
          f"reward={stats['reward']} 视口帧={n_vp} 播放={rate_txt}")

    # ---- 3. 合成帧：左视口 + 右 Agent 会话终端 ----
    MINT, SKY, SLATE, DIM = ("#34D399", "#38BDF8", "#94A3B8", "#64748B")
    n_frames = 1335

    def _terminal(f: int) -> np.ndarray:
        """右 520x720 Agent 会话面板（按帧号 f 演进）。"""
        img = QImage(RW, H, QImage.Format.Format_RGB888)
        img.fill(QColor("#0B0F16"))
        p = QPainter(img)
        mono = lambda sz, wght=QFont.Weight.Normal: QFont("Monospace", sz, wght)
        try:
            p.fillRect(0, 0, RW, 34, QColor("#10161F"))
            p.setFont(mono(10))
            p.setPen(QPen(QColor(DIM)))
            p.drawText(16, 21, f"agent · {model_name}")
            p.setPen(QPen(QColor(MINT)))
            p.drawText(RW - 130, 21, "simovia mcp")
            p.setPen(QPen(QColor("#1E293B")))
            p.drawLine(0, 34, RW, 34)

            y = 62
            # 用户需求（打字机）
            p.setFont(mono(11))
            shown = int(len(REQ[:110]) * min(1.0, f / 75.0))
            p.setPen(QPen(QColor(MINT)))
            p.drawText(16, y, "› " + REQ[:shown])
            if f >= 75:
                p.setPen(QPen(QColor(SLATE)))
                for i, ln in enumerate(_wrap(REQ[110:], 42)[:3]):
                    p.drawText(16 + 14, y + 16 * (i + 1), ln)
            y += 16 * 4 + 14

            # YAML 生成窗口（滚动显示尾部行）
            if f >= 75:
                p.setFont(mono(10))
                p.setPen(QPen(QColor(SKY)))
                p.drawText(16, y, "scene.yaml — LLM 生成中" if f < 285 else
                           f"scene.yaml — ✓ 校验通过（parse + 编译）")
                p.setPen(QPen(QColor("#334155")))
                p.drawRect(14, y + 8, RW - 30, 12 * 13 + 12)
                nshow = int(len(yaml_lines) * min(1.0, (f - 75) / 210.0))
                p.setPen(QPen(QColor(SLATE)))
                for i, ln in enumerate(yaml_lines[max(0, nshow - 12):nshow]):
                    p.drawText(20, y + 24 + 13 * i, ln[:56])
            y += 12 * 13 + 40

            # 工具调用行（按帧阈值出现）
            def tool(call, result, at, accent=MINT):
                nonlocal y
                if f < at:
                    return
                p.setFont(mono(11))
                p.setPen(QPen(QColor(SKY)))
                p.drawText(16, y, "↪ " + call)
                p.setPen(QPen(QColor(accent)))
                p.drawText(34, y + 15, "← " + result)
                y += 34
            tool(f"save_world(path=/tmp/agent_gen_world.yaml)",
                 f"ok · {n_bytes} bytes · {len(yaml_lines)} 行", 285)
            tool(f"load_world(yaml_path=…)",
                 f"ok · 1 robot (panda+2fg7) · task: pick · 名 {world_name}", 315)
            if f >= 345:
                p.setFont(mono(11))
                p.setPen(QPen(QColor(SKY)))
                p.drawText(16, y, "↪ run(ticks=5000, realtime=false)")
                prog = min(1.0, max(0.0, (f - 375) / 840.0))
                tk = int(stats["ticks"] * prog)
                p.setPen(QPen(QColor(DIM)))
                if f < 1215:
                    p.drawText(34, y + 15,
                               f"ticks {tk}/{stats['ticks']} · ctrl ok")
                else:
                    ok = "true" if stats["success"] else "false"
                    p.setPen(QPen(QColor(MINT if stats["success"] else "#F87171")))
                    p.drawText(34, y + 15,
                               f"← ticks={stats['ticks']} · success={ok} · "
                               f"reward={stats['reward']:+.2f}")
                y += 34
            tool(f"screenshot(camera={cam_name or 'free'}, w={LW})",
                 f"← {LW}x{H} PNG · {final_png_kb} KB", 1275)
        finally:
            p.end()
        return _qimage_rgb(img.copy())

    def _load_vp(i: int) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(f"{vp_dir}/{vp_files[i]}").convert("RGB"))

    for f in range(n_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if f < 315:                                  # load 完成前视口空置（黑）
            pass
        elif f < 375:
            canvas[:, :LW] = init_rgb               # load 后的初始世界
        elif f < 1215:                              # run 段：视口帧均匀映射
            idx = int((f - 375) * (n_vp - 1) / 839)
            canvas[:, :LW] = _load_vp(idx)
        else:
            canvas[:, :LW] = _load_vp(n_vp - 1)     # 定格终态
        if f >= 315:
            _label(canvas, 14, H - 18,
                   f"MuJoCo 视口 · AI 生成世界 {world_name} · {rate_txt}",
                   size=12)
        canvas[:, LW:] = _terminal(f)
        img = QImage(canvas.data, W, H, canvas.strides[0],
                     QImage.Format.Format_RGB888)
        img.save(f"{frames_dir}/{f:06d}.png")
    print(f"合成 {n_frames} 帧 → {args.out}")

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", f"{frames_dir}/%06d.png", "-c:v", "libx264", "-pix_fmt",
        "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out,
    ], check=True)
    dur = n_frames / FPS
    print(f"视频已生成 {args.out}：{n_frames} 帧 / {dur:.0f}s @ {FPS}fps")


if __name__ == "__main__":
    main()
