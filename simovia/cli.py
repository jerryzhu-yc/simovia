"""simovia 命令行入口（人类用户；AI 用户走 mcp_server）。

用法：
    python -m simovia run worlds/pick_cube.yaml [--ticks N]
    python -m simovia inspect worlds/pick_cube.yaml   # 看执行器/关节/传感器清单
    python -m simovia mcp                     # MCP stdio 服务器（AI Agent 接入）
    python -m simovia gui                     # 桌面 GUI（PySide6）
"""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simovia")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="运行一个世界")
    run_p.add_argument("world", help="世界 YAML 文件路径")
    run_p.add_argument("--ticks", type=int, default=None,
                       help="最大 tick 数（缺省用世界 task.max_ticks）")
    run_p.add_argument("--json", action="store_true",
                       help="以 JSON 输出统计（机器可读）")
    run_p.add_argument("--record", default=None, metavar="DIR",
                       help="把本次 episode 录制到目录 DIR（data.npz + meta.json）")
    run_p.add_argument("--record-images", action="store_true",
                       help="录制时同时保存相机帧 PNG（需配合 --record）")
    run_p.add_argument("--record-camera", default=None, metavar="NAME",
                       help="录制用的相机名：YAML 传感器短名或 <robot>/<name>"
                            "（缺省自由机位；用 inspect 查看可用相机）")
    run_p.add_argument("--record-width", type=int, default=640, metavar="PX",
                       help="录制图像宽度（缺省 640）")
    run_p.add_argument("--record-height", type=int, default=480, metavar="PX",
                       help="录制图像高度（缺省 480）")
    run_p.add_argument("--record-image-stride", type=int, default=1,
                       metavar="N", help="每 N 个控制 tick 存一帧（缺省 1）")
    in_p = sub.add_parser("inspect", help="打印世界的机器人/物体清单（写控制器前先看这里）")
    in_p.add_argument("world", help="世界 YAML 文件路径")
    sub.add_parser("mcp", help="启动 MCP stdio 服务器")
    sub.add_parser("gui", help="启动桌面 GUI（可视化 + 观测面板）")
    doc_p = sub.add_parser("doctor", help="环境诊断：依赖版本、模型资产、离屏渲染")
    doc_p.add_argument("--json", action="store_true",
                       help="以 JSON 输出诊断结果（机器可读，便于报错附上）")
    gen_p = sub.add_parser("scenegen", help="从模板生成世界 YAML")
    gen_p.add_argument("template", help="模板名")
    gen_p.add_argument("--out", default=None,
                       help="输出文件路径（缺省打印到 stdout）")
    # parse_known_args：scenegen 的模板参数是动态的，外层放行、内层消化
    args, rest = ap.parse_known_args(argv)
    if rest and args.cmd != "scenegen":
        ap.error(f"无法识别的参数: {' '.join(rest)}")

    if args.cmd == "run":
        _run(args)
    elif args.cmd == "inspect":
        _inspect(args)
    elif args.cmd == "mcp":
        from simovia.mcp_server import serve
        serve()
    elif args.cmd == "gui":
        try:
            from simovia.gui import main as gui_main
        except ImportError as e:
            raise SystemExit(
                f"GUI 需要 PySide6：pip install PySide6-Essentials（{e}）")
        raise SystemExit(gui_main())
    elif args.cmd == "scenegen":
        _scenegen(args, argv)
    elif args.cmd == "doctor":
        raise SystemExit(_doctor(args))


def _doctor(args):
    """环境诊断：依赖版本、模型资产、离屏渲染、样例世界能否编译。

    返回进程退出码（0 = 必需项全通过；1 = 有 FAIL，可直接贴给维护者）。
    """
    import importlib.metadata as md
    import platform
    import sys

    checks = []          # [(level, name, detail)]

    def add(level, name, detail):
        checks.append((level, name, detail))

    # ---- Python ----
    v = sys.version_info
    add("OK" if v >= (3, 11) else "FAIL", "Python",
        f"{v.major}.{v.minor}.{v.micro}（需要 >= 3.11，"
        f"{platform.system()} {platform.machine()}）")

    # ---- 依赖：必需 / 可选 ----
    for mod, pkg, optional, hint in (
        ("mujoco", "mujoco", False, ""),
        ("numpy", "numpy", False, ""),
        ("yaml", "PyYAML", False, ""),
        ("zmq", "pyzmq", False, ""),
        ("PySide6", "PySide6-Essentials", True, "GUI"),
        ("gymnasium", "gymnasium", True, "RL 训练"),
        ("mujoco_menagerie", "mujoco-menagerie", True, "模型下载回退"),
    ):
        try:
            __import__(mod)
        except Exception as e:                       # noqa: BLE001 诊断要吞掉一切
            add("WARN" if optional else "FAIL", pkg,
                f"未安装{f'（可选：{hint}）' if optional else ''}"
                f"：{type(e).__name__}")
            continue
        try:
            ver = md.version(pkg)
        except Exception:                            # noqa: BLE001
            ver = "已导入（版本未知）"
        add("OK", pkg, ver + (f"（可选：{hint}）" if optional else ""))

    # ---- 模型资产 ----
    try:
        from simovia.zoo import BUNDLED_DIR, MODELS, bundled_model
        have = sum(1 for k in MODELS if bundled_model(k) is not None)
        level = "OK" if have == len(MODELS) else ("WARN" if have else "FAIL")
        detail = f"{have}/{len(MODELS)} 个模型就绪（{BUNDLED_DIR}）"
        if have < len(MODELS):
            detail += "；缺失资产需重新 clone 仓库或 pip install -r requirements-models.txt"
        add(level, "模型资产", detail)
    except Exception as e:                           # noqa: BLE001
        add("WARN", "模型资产", f"检查失败：{type(e).__name__}: {e}")

    # ---- 离屏渲染（录制图像 / GUI 实时画面依赖）----
    # 隔子进程探测：无 GL 环境下 MuJoCo 可能直接 abort，不能在本进程里试
    try:
        from simovia.render import probe_offscreen_gl
        ok, detail = probe_offscreen_gl()
        if ok:
            add("OK", "离屏渲染", "可用（录制图像 / GUI 实时画面）")
        else:
            add("WARN", "离屏渲染",
                f"不可用（{detail}）；录制图像与 GUI 实时画面会降级。"
                "Linux 下可装 libosmesa6 或设 MUJOCO_GL=osmesa；"
                "Windows 需显卡驱动支持 OpenGL 3.3")
    except Exception as e:                           # noqa: BLE001
        add("WARN", "离屏渲染", f"探测失败：{type(e).__name__}: {e}")

    # ---- 样例世界能编译（验证路径解析 + 资源完整性）----
    try:
        import pathlib

        from simovia.world.builder import build_world
        from simovia.world.schema import load_world
        sample = pathlib.Path(__file__).resolve().parents[1] / "worlds" / "pick_cube.yaml"
        spec = load_world(sample)
        build_world(spec)
        add("OK", "样例世界", f"编译通过（{sample.name}）")
    except Exception as e:                           # noqa: BLE001
        add("FAIL", "样例世界", f"编译失败：{type(e).__name__}: {e}")

    if getattr(args, "json", False):
        print(json.dumps(
            {"ok": not any(lv == "FAIL" for lv, _, _ in checks),
             "checks": [{"level": lv, "name": n, "detail": d}
                        for lv, n, d in checks]},
            ensure_ascii=False, indent=2))
    else:
        print("Simovia 环境诊断：")
        for lv, name, detail in checks:
            print(f"  [{lv:<4}] {name:<24} {detail}")
        fails = [n for lv, n, _ in checks if lv == "FAIL"]
        warns = [n for lv, n, _ in checks if lv == "WARN"]
        print()
        if fails:
            print(f"结论：{len(fails)} 项必需检查未通过（{', '.join(fails)}）。"
                  "先按上面的提示安装依赖再启动。")
        elif warns:
            print(f"结论：必需项全部通过；{len(warns)} 项可选/降级"
                  f"（{', '.join(warns)}）。仿真与 MCP 可正常使用。")
        else:
            print("结论：全部通过。")
    return 1 if any(lv == "FAIL" for lv, _, _ in checks) else 0


def _scenegen(args, argv=None):
    import pathlib

    from simovia import scene_gen

    # 动态模板参数（各模板自己声明；scenegen/--out 等其余参数被跳过）
    ap2 = argparse.ArgumentParser(add_help=False)
    for pname, (ptype, default, help_) in \
            scene_gen.TEMPLATES.get(args.template, (None, {}, None))[1].items():
        ap2.add_argument(f"--{pname}", type=ptype, default=default, help=help_)
    p = vars(ap2.parse_known_args(argv)[0])

    base = pathlib.Path(args.out).resolve().parent if args.out \
        else pathlib.Path.cwd()
    try:
        text = scene_gen.generate(args.template, base, p)
    except Exception as e:
        raise SystemExit(f"生成失败: {e}")
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"已生成 {args.out}")
    else:
        print(text, end="")


def _inspect(args):
    """打印世界的设备清单：写控制器前先知道执行器/关节叫什么、范围多少。"""
    import mujoco

    from simovia.world.builder import build_world
    from simovia.world.schema import load_world

    try:
        spec = load_world(args.world)
        model, index = build_world(spec)
    except Exception as e:
        raise SystemExit(f"世界加载失败: {e}")

    print(f"世界 '{spec.name}'：{len(spec.scene.objects)} 物体，"
          f"{len(spec.robots)} 机器人"
          f"（timestep {spec.timestep}s，每 tick 执行 {spec.ctrl_period} 步）")
    if spec.scene.objects:
        print("场景物体（obs['objects'] 的键）:")
        for od in spec.scene.objects:
            tag = "static" if od.static else (
                f"mass {od.mass}" if od.mass is not None else "mass 按密度")
            print(f"  {od.name:<12} {od.type:<9} size={list(od.size)} "
                  f"pos={list(od.pos)} {tag}")
    for rd in spec.robots:
        ridx = index.robots[rd.name]
        print(f"\n机器人 '{rd.name}'  模型 "
              f"{rd.model or '内联连杆树（spec）'}")
        if rd.controller:
            print(f"  控制器: {rd.controller}")
        else:
            print("  控制器: 未配置（不会动）—— 参考 controllers/random_arm.py，"
                  "把脚本路径填进 YAML 的 robots[].controller")
        print(f"  执行器（{len(ridx.actuators)}）——robot.set(名字, 值):")
        for name, aid in ridx.actuators.items():
            lo, hi = model.actuator_ctrlrange[aid]
            print(f"    {name:<24} ctrlrange [{lo:+.3f}, {hi:+.3f}]")
        if ridx.joints:
            print(f"  关节（{len(ridx.joints)}）:")
            for name, jid in ridx.joints.items():
                jtype = mujoco.mjtJoint(model.jnt_type[jid]).name
                jtype = jtype.replace("mjJNT_", "").lower()
                rng = (f"[{model.jnt_range[jid][0]:+.3f}, "
                       f"{model.jnt_range[jid][1]:+.3f}]"
                       if model.jnt_limited[jid] else "无限位")
                print(f"    {name:<24} {jtype:<10} {rng}")
        if ridx.sensors:
            print(f"  传感器（{len(ridx.sensors)}）——robot.sensor(名字):")
            for six in ridx.sensors.values():
                print(f"    {six.name:<24} {six.kind:<14} {six.dim} 维")
        else:
            print("  传感器: 无")
        if ridx.cameras:
            print(f"  相机: {', '.join(ridx.cameras)}")
        for ee in rd.end_effectors:
            side = ee.side or ""
            print(f"  末端执行器: {ee.model}" + (f"（{side}）" if side else ""))
    task = spec.task
    if task.success is not None:
        limit = task.max_ticks if task.max_ticks is not None else "不限"
        print(f"\n任务: 成功条件 {task.success.body} ∈ zone "
              f"min={list(task.success.zone.min)} max={list(task.success.zone.max)}，"
              f"奖励 {len(task.reward)} 项，最多 {limit} ticks")


def _run(args):
    from simovia.kernel.sim import Simulation
    from simovia.world.schema import load_world

    if args.record_images and not args.record:
        raise SystemExit("--record-images 需要配合 --record DIR 使用")
    if args.record_camera and not args.record_images:
        raise SystemExit("--record-camera 需要配合 --record-images 使用")
    if args.record_width <= 0 or args.record_height <= 0:
        raise SystemExit("--record-width/--record-height 必须为正数")
    if args.record_image_stride < 1:
        raise SystemExit("--record-image-stride 必须 >= 1")

    try:
        spec = load_world(args.world)
    except Exception as e:
        raise SystemExit(f"世界加载失败: {e}")

    recorder = None
    with Simulation(spec) as sim:
        if args.record:
            from simovia.recorder import EpisodeRecorder
            try:
                recorder = EpisodeRecorder(
                    spec, sim.model, sim.index, args.record,
                    images=args.record_images, camera=args.record_camera,
                    width=args.record_width, height=args.record_height,
                    image_stride=args.record_image_stride)
            except ValueError as e:
                raise SystemExit(f"录制配置错误: {e}")
        stats = sim.run(
            max_ticks=args.ticks,
            on_tick=(lambda: recorder.add(sim.tick, sim.data))
            if recorder is not None else None)
        objects = {name: [round(float(x), 4) for x in sim.data.xpos[bid]]
                   for name, bid in sim.index.objects.items()}
        # 机器人 root body（每台机器人唯一一个 parent=world 的 body）：
        # 跑完看不到机器人位置时，新手无法判断"四足站住没有/机械臂倒没倒"
        full_names = {bid: name for name, bid in sim.index.bodies.items()}
        robots = {}
        for rname, ridx in sim.index.robots.items():
            for bshort, bid in ridx.bodies.items():
                if sim.model.body_parentid[bid] == 0:
                    robots[rname] = {
                        "base": full_names.get(bid, bshort),
                        "pos": [round(float(x), 4) for x in sim.data.xpos[bid]],
                    }
                    break

    episode_dir = None
    if recorder is not None:
        episode_dir = recorder.finish(
            stats=stats, extra={"world_path": str(args.world)})

    if args.json:
        payload = {**stats, "objects": objects, "robots": robots}
        if episode_dir is not None:
            payload["record"] = str(episode_dir)
            if recorder.images:
                payload["record_images"] = {
                    "count": recorder.frame_count,
                    "camera": recorder.camera,
                    "width": recorder.width,
                    "height": recorder.height,
                    "stride": recorder.image_stride,
                }
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(f"世界 '{spec.name}'：{len(spec.scene.objects)} 物体，"
          f"{len(spec.robots)} 机器人")
    for rd in spec.robots:
        print(f"  控制器 {rd.name}: {rd.controller or '未配置'}")
    if spec.robots and not any(rd.controller for rd in spec.robots):
        print("  提示: 本场景没有控制器，机器人保持初始位姿不动。"
              "参考 controllers/random_arm.py 写一个，"
              "填到 YAML 的 robots[].controller 后重跑")
    print(f"运行 {stats['ticks']} ticks（墙钟 {stats['wall_time']}s，"
          f"仿真 {stats['sim_time']}s）")
    if stats["lag"]:
        print(f"控制器超时补位 {stats['lag']} 次（策略 {spec.runtime.timeout_policy}）")
    if stats["crashed"]:
        print(f"崩溃的控制器: {stats['crashed']}")
    print(f"物理状态: {'NaN!' if stats['nan'] else '正常'}")
    if stats["success"] is not None:
        print(f"任务: {'成功' if stats['success'] else '未完成'}"
              f"（reward {stats['reward']}）")
    for name, pos in objects.items():
        print(f"  {name} 最终位置: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    for info in robots.values():
        pos = info["pos"]
        print(f"  {info['base']} 最终位置: "
              f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    if episode_dir is not None:
        detail = "data.npz + meta.json"
        if recorder.images:
            cam = recorder.camera or "自由机位"
            detail += (f"，{recorder.frame_count} 帧 {recorder.width}x"
                       f"{recorder.height}（相机 {cam}，每 "
                       f"{recorder.image_stride} tick 一帧）")
        print(f"录制已保存: {episode_dir}（{detail}）")


if __name__ == "__main__":
    main()
