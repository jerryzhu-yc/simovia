import { useState } from "react";
import {
  AGENT_DIFF,
  AGENT_LOG,
  CODE,
  DEMOS,
  FEATURES,
  MCP_TOOLS,
  ONBOARD,
  STEPS,
} from "./content.js";

const REPO = "https://github.com/jerryzhu-yc/simovia";

const ICON_PATHS = {
  code: "M8 9l-3 3 3 3M16 9l3 3-3 3M13 6l-2 12",
  process: "M4 6h6v6H4zM14 6h6v6h-6zM9 18h6M12 12v3",
  agent: "M12 3v3M5 8h14v10H5zM9 13h.01M15 13h.01M9 21h6",
  gui: "M3 5h18v12H3zM7 21h10M12 17v4M7 9h4M7 12h7",
  team: "M8 11a3 3 0 100-6 3 3 0 000 6zM17 12a2.5 2.5 0 100-5 2.5 2.5 0 000 5zM2 20c0-3 2.7-5 6-5s6 2 6 5M15 20c0-2.2 1.6-3.6 3.5-3.6S22 17.8 22 20",
  vr: "M4 8h16v8h-5l-3-3-3 3H4zM8 12h.01M16 12h.01",
  eye: "M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6zM12 9.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z",
  vla: "M4 17V7l5 4 5-6 6 8",
};

function Icon({ name, className = "h-5 w-5" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}

function Mark({ className = "h-8 w-8" }) {
  return (
    <svg className={className} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="#111823" stroke="#1E2A38" />
      <circle cx="16" cy="16" r="7" fill="none" stroke="#34D399" strokeWidth="2.5" />
      <circle cx="16" cy="16" r="2" fill="#22D3EE" />
    </svg>
  );
}

function SectionHead({ kicker, title, desc }) {
  return (
    <div className="max-w-2xl">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-mint">{kicker}</p>
      <h2 className="mt-3 text-2xl font-semibold text-slate-50 md:text-3xl">{title}</h2>
      {desc && <p className="mt-3 text-sm leading-6 text-slate-400">{desc}</p>}
    </div>
  );
}

function Nav() {
  const links = [
    ["能力", "#features"],
    ["Agent", "#agent"],
    ["演示", "#demos"],
    ["上手", "#onboarding"],
    ["闭环", "#pipeline"],
    ["快速开始", "#start"],
    ["架构", "#arch"],
  ];
  return (
    <header className="sticky top-0 z-40 border-b border-line/80 bg-ink/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-5">
        <a href="#top" className="flex shrink-0 items-center gap-3">
          <Mark />
          <span className="text-[15px] font-semibold tracking-wide text-slate-100">
            Simovia
          </span>
        </a>
        <nav className="flex flex-1 items-center gap-6 overflow-x-auto whitespace-nowrap text-sm text-slate-400 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {links.map(([label, href]) => (
            <a key={href} href={href} className="py-1 transition hover:text-slate-100">
              {label}
            </a>
          ))}
        </nav>
        <a
          href={REPO}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 rounded-lg border border-line bg-panel px-3.5 py-1.5 text-sm text-slate-200 transition hover:border-mint/50 hover:text-mint"
        >
          GitHub
        </a>
      </div>
    </header>
  );
}

function Hero() {
  const caps = [
    ["场景树 + 3D 视口", "拖拽摆放、选中改位姿，编辑即校验"],
    ["任务状态解释", "成功判定、失败原因直接写在界面上"],
    ["装备页", "换夹爪、加相机、调 fovy，一键重建"],
    ["Agent 页", "同一个后端，工具调用与结果可视化"],
  ];
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="bg-grid pointer-events-none absolute inset-0" />
      <div className="glow pointer-events-none absolute inset-0" />
      <div className="relative mx-auto max-w-6xl px-5 pb-16 pt-16 md:pt-24">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {["MuJoCo", "MCP", "LeRobot", "SmolVLA"].map((t) => (
            <span
              key={t}
              className="rounded-full border border-line bg-panel/70 px-3 py-1 font-mono text-slate-400"
            >
              {t}
            </span>
          ))}
        </div>

        <div className="mt-7 grid items-end gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <h1 className="max-w-3xl text-4xl font-semibold leading-tight tracking-tight text-slate-50 md:text-6xl">
              把机器人世界
              <span className="bg-gradient-to-r from-mint to-sky bg-clip-text text-transparent">
                写成代码
              </span>
            </h1>
            <p className="mt-6 max-w-2xl text-base leading-7 text-slate-400 md:text-lg">
              一份 YAML 就是一个可复现的仿真世界。AI Agent 通过 MCP 读世界、改世界、跑世界；
              人类在同一个能力的 GUI 里搭场景、换夹爪、遥操作采数据，再训练 VLA 闭环部署。
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <a
                href="#start"
                className="rounded-xl bg-mint px-5 py-2.5 text-sm font-semibold text-ink transition hover:bg-emerald-300"
              >
                快速开始
              </a>
              <a
                href="#agent"
                className="rounded-xl border border-line bg-panel px-5 py-2.5 text-sm text-slate-200 transition hover:border-slate-500"
              >
                看 Agent 改世界
              </a>
              <span className="font-mono text-xs text-slate-500">
                pip install -r requirements.txt
              </span>
            </div>
          </div>

          <ul className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-1">
            {caps.map(([t, d]) => (
              <li
                key={t}
                className="rounded-xl border border-line bg-panel/60 px-4 py-3"
              >
                <p className="text-[13px] font-medium text-slate-200">{t}</p>
                <p className="mt-0.5 text-[11px] leading-5 text-slate-500">{d}</p>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-12 overflow-hidden rounded-2xl border border-line bg-panel shadow-2xl shadow-black/40">
          <div className="flex items-center justify-between gap-4 border-b border-line px-4 py-2.5">
            <span className="truncate font-mono text-xs text-slate-400">
              Simovia — 机器人仿真平台 · worlds/pick_cube.yaml
            </span>
            <span className="flex shrink-0 items-center gap-1.5 rounded-md bg-mint/10 px-2 py-0.5 font-mono text-[11px] text-mint">
              <span className="h-1.5 w-1.5 rounded-full bg-mint" />
              1920×1200 实机截图
            </span>
          </div>
          <img
            src="./assets/gui/workbench.png"
            alt="Simovia GUI 工作台：场景树、3D 视口、属性面板与任务状态"
            className="aspect-[16/10] w-full bg-ink object-cover"
          />
        </div>

        <p className="mt-5 font-mono text-[11px] text-slate-500">
          35 个机器人模型 · 13 个示例世界 · 16 个 MCP 工具 · 31.25 Hz 策略闭环
        </p>
      </div>
    </section>
  );
}

function Features() {
  return (
    <section id="features" className="mx-auto max-w-6xl px-5 py-20">
      <SectionHead
        kicker="核心能力"
        title="从写世界到训练策略，一条链路"
        desc="不依赖商业仿真器，核心只要求 mujoco / numpy / pyyaml / pyzmq；模型资产随仓库自带，clone 完离线可用。"
      />

      <div className="mt-10 grid auto-rows-fr gap-4 md:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map((f) => (
          <article
            key={f.title}
            className="group flex flex-col rounded-2xl border border-line bg-panel/70 p-5 transition hover:-translate-y-0.5 hover:border-mint/40 hover:shadow-lg hover:shadow-mint/5"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-line bg-ink text-mint">
              <Icon name={f.icon} />
            </div>
            <h3 className="mt-4 text-[15px] font-semibold text-slate-100">{f.title}</h3>
            <p className="mt-1 font-mono text-[11px] uppercase tracking-wide text-slate-500">
              {f.en}
            </p>
            <p className="mt-3 text-sm leading-6 text-slate-400">{f.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function DiffPanel() {
  return (
    <div className="overflow-hidden rounded-2xl border border-line bg-ink">
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span className="font-mono text-xs text-slate-500">
          worlds/so101_policy_eval.yaml
        </span>
        <span className="font-mono text-[11px] text-slate-600">+4 −0</span>
      </div>
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[11.5px] leading-6">
        {AGENT_DIFF.split("\n").map((line, i) => (
          <div
            key={i}
            className={
              line.startsWith("+")
                ? "bg-mint/10 text-mint"
                : "text-slate-400"
            }
          >
            {line || " "}
          </div>
        ))}
      </pre>
    </div>
  );
}

function AgentCoding() {
  return (
    <section id="agent" className="border-y border-line bg-panel/30">
      <div className="mx-auto max-w-6xl px-5 py-20">
        <SectionHead
          kicker="Agent Coding"
          title="一句话改世界，Agent 和 GUI 走同一条链路"
          desc="零依赖 stdio JSON-RPC MCP 服务器，16 个工具覆盖读世界、改 YAML、跑仿真、截图观察。下面是一次真实的改机位过程：Agent 不碰内部 id，只改 YAML，改完立刻重跑验证。"
        />

        <div className="mt-10 grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="overflow-hidden rounded-2xl border border-line bg-ink">
            <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
              <span className="font-mono text-xs text-slate-500">
                agent · mcp stdio
              </span>
              <span className="font-mono text-[11px] text-mint">16 tools</span>
            </div>
            <div className="space-y-2.5 px-4 py-4 font-mono text-[12px] leading-6">
              {AGENT_LOG.map((l, i) =>
                l.role === "user" ? (
                  <div key={i} className="flex gap-2 text-slate-100">
                    <span className="shrink-0 text-mint">›</span>
                    <span>{l.text}</span>
                  </div>
                ) : (
                  <div
                    key={i}
                    className="rounded-lg border border-line/70 bg-panel/60 px-3 py-2"
                  >
                    <p className="text-sky">tool_call {l.name}</p>
                    <p className="mt-0.5 text-[11px] text-slate-500">{l.text}</p>
                  </div>
                )
              )}
            </div>
          </div>

          <div className="flex flex-col gap-5">
            <DiffPanel />
            <div className="rounded-2xl border border-line bg-panel/70 px-4 py-4">
              <p className="font-mono text-[11px] uppercase tracking-wide text-slate-500">
                MCP 工具面
              </p>
              <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5">
                {MCP_TOOLS.map(([name, desc]) => (
                  <div key={name} className="flex items-baseline gap-2">
                    <span className="font-mono text-[11px] text-mint/90">
                      {name}
                    </span>
                    <span className="truncate text-[10.5px] text-slate-600">
                      {desc}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function DemoMedia({ demo }) {
  const ratio = demo.featured ? "aspect-[16/9]" : "aspect-[4/3]";
  if (demo.kind === "image") {
    return (
      <img
        src={demo.src}
        alt={demo.title}
        className={`${ratio} w-full bg-ink object-cover`}
        loading="lazy"
      />
    );
  }
  return (
    <video
      src={demo.src}
      className={`${ratio} w-full bg-ink object-cover`}
      autoPlay
      loop
      muted
      playsInline
      preload="metadata"
    />
  );
}

function DemoCard({ demo }) {
  return (
    <figure
      className={`group overflow-hidden rounded-2xl border border-line bg-panel transition hover:border-mint/40 ${
        demo.featured ? "md:col-span-2" : ""
      }`}
    >
      <DemoMedia demo={demo} />
      <figcaption className="px-5 py-4">
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-mint/10 px-2 py-0.5 font-mono text-[11px] text-mint">
            {demo.tag}
          </span>
          <span className="font-mono text-[11px] text-slate-600">{demo.meta}</span>
        </div>
        <p className="mt-2 text-sm font-semibold text-slate-100">{demo.title}</p>
        <p className="mt-1 text-xs leading-5 text-slate-400">{demo.body}</p>
      </figcaption>
    </figure>
  );
}

function Demos() {
  return (
    <section id="demos" className="border-y border-line bg-panel/30">
      <div className="mx-auto max-w-6xl px-5 py-20">
        <SectionHead
          kicker="演示"
          title="真实仿真画面，实时录制"
          desc="下面每一段都来自仓库里的世界与控制器：多机协同、VR 采集、视觉分拣、抓取与 VLA 闭环。"
        />

        <div className="mt-10 grid gap-5 md:grid-cols-2">
          {DEMOS.map((demo) => (
            <DemoCard key={demo.src} demo={demo} />
          ))}
        </div>

        <div className="mt-6 rounded-2xl border border-line bg-panel/40 px-5 py-4 font-mono text-xs leading-6 text-slate-500">
          <span className="text-slate-300">复现入口：</span>
          python -m demo.maintenance_relay · python -m demo.fruit_sorting ·
          python -m simovia run worlds/pick_cube.yaml · worlds/so101_policy_eval.yaml
        </div>
      </div>
    </section>
  );
}

function Onboarding() {
  return (
    <section id="onboarding" className="mx-auto max-w-6xl px-5 py-20">
      <SectionHead
        kicker="上手"
        title="不写 MJCF，也能从零拼出可跑的机器人"
        desc="两个最容易劝退新人的环节，GUI 都做成了表格与下拉框：机器人用 DH 参数和旋转矩阵定义，末端执行器点一下就换。"
      />

      <div className="mt-10 grid items-start gap-5 md:grid-cols-2">
        {ONBOARD.map((o) => (
          <figure
            key={o.title}
            className="group overflow-hidden rounded-2xl border border-line bg-panel transition hover:border-mint/40"
          >
            {o.video ? (
              <video
                src={o.video}
                poster={o.img}
                className="w-full bg-ink object-cover"
                autoPlay
                loop
                muted
                playsInline
                preload="metadata"
              />
            ) : (
              <img
                src={o.img}
                alt={o.title}
                className="w-full bg-ink object-cover"
                loading="lazy"
              />
            )}
            <figcaption className="px-5 py-4">
              <div className="flex items-center gap-2">
                <span className="rounded-md bg-mint/10 px-2 py-0.5 font-mono text-[11px] text-mint">
                  {o.tag}
                </span>
                <span className="font-mono text-[11px] text-slate-600">
                  {o.meta}
                </span>
              </div>
              <p className="mt-2 text-sm font-semibold text-slate-100">
                {o.title}
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-400">{o.body}</p>
            </figcaption>
          </figure>
        ))}
      </div>

      <p className="mt-6 text-sm leading-6 text-slate-400">
        <span className="font-medium text-slate-200">全程可解释：</span>
        DH 表里每一行都对应一个真实几何体，矩阵、关节轴和 3D 预览同步变化；
        换夹爪改的是 YAML 里的末端执行器声明，不会动到物理与控制器。
      </p>
    </section>
  );
}

function Pipeline() {
  return (
    <section id="pipeline" className="mx-auto max-w-6xl px-5 py-20">
      <SectionHead
        kicker="数据闭环"
        title="采集 → 训练 → 部署，全在同一仓库"
        desc="人工示教的数据直接落到标准数据集格式，训练与回灌都在同一套世界定义上，避免采集与部署两套场景对不上。"
      />

      <ol className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((s, i) => (
          <li
            key={s.n}
            className="relative rounded-2xl border border-line bg-panel/70 p-5"
          >
            <span className="font-mono text-xs text-mint">{s.n}</span>
            <h3 className="mt-3 text-sm font-semibold text-slate-100">{s.title}</h3>
            <p className="mt-2 text-xs leading-5 text-slate-400">{s.body}</p>
            {i < STEPS.length - 1 && (
              <span className="absolute -right-3 top-1/2 hidden -translate-y-1/2 text-line lg:block">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M5 12h14M13 6l6 6-6 6" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </span>
            )}
          </li>
        ))}
      </ol>

      <div className="mt-6 rounded-2xl border border-line bg-panel/50 p-5 text-sm leading-6 text-slate-400">
        <span className="font-medium text-slate-200">可复现性：</span>
        同一份 YAML 跑两遍轨迹逐 bit 相同，reset 结果一致；数据集字段全部名字驱动，
        不依赖任何 MuJoCo 内部 id，跨仿真器与真机可复用。
      </div>
    </section>
  );
}

function CopyButton({ text }) {
  const [done, setDone] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1600);
    } catch {
      setDone(false);
    }
  }
  return (
    <button
      onClick={copy}
      className="rounded-md border border-line px-2.5 py-1 font-mono text-[11px] text-slate-400 transition hover:border-mint/50 hover:text-mint"
    >
      {done ? "已复制" : "复制"}
    </button>
  );
}

function QuickStart() {
  return (
    <section id="start" className="border-y border-line bg-panel/30">
      <div className="mx-auto max-w-6xl px-5 py-20">
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <SectionHead
              kicker="快速开始"
              title="五条命令，看到第一个世界"
              desc="需要 64 位 Python 3.11+。核心依赖只有 mujoco、numpy、pyyaml、pyzmq，模型资产随仓库自带，无需联网下载。"
            />
            <div className="mt-6 space-y-3 text-sm text-slate-400">
              {[
                ["python -m simovia run worlds/pick_cube.yaml", "跑一个世界，看统计结果"],
                ["python -m simovia inspect worlds/pick_cube.yaml", "列出执行器 / 关节 / 传感器"],
                ["python -m demo.maintenance_relay", "多机器人协同维修完整演示"],
                ["python -m simovia doctor", "环境诊断，缺什么直接给安装提示"],
              ].map(([cmd, desc]) => (
                <div key={cmd}>
                  <code className="rounded-md border border-line bg-ink px-2 py-1 font-mono text-xs text-slate-200">
                    {cmd}
                  </code>
                  <p className="mt-1 text-xs text-slate-500">{desc}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="overflow-hidden rounded-2xl border border-line bg-ink">
            <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
              <span className="font-mono text-xs text-slate-500">bash</span>
              <CopyButton text={CODE} />
            </div>
            <pre className="overflow-x-auto px-4 py-4 font-mono text-[12.5px] leading-7 text-slate-300">
              {CODE}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

const LAYERS = [
  {
    title: "AI Agent",
    tag: "MCP 客户端",
    body: "读世界、改 YAML、重跑、截图观察",
  },
  {
    title: "MCP 工具层",
    tag: "stdio JSON-RPC · 16 工具",
    body: "load_world / run / get_observation / screenshot ...",
  },
  {
    title: "仿真内核",
    tag: "YAML → MjSpec → MuJoCo",
    body: "ZMQ ROUTER + 栅栏同步，按 tick 推进物理",
  },
  {
    title: "控制器进程",
    tag: "独立 OS 进程 × N",
    body: "while robot.step()：读观测、写控制量、再同步",
  },
];

function Architecture() {
  return (
    <section id="arch" className="mx-auto max-w-6xl px-5 py-20">
      <SectionHead kicker="架构" title="四层，每层都能单独替换" />

      <div className="mt-10 space-y-3">
        {LAYERS.map((l, i) => (
          <div key={l.title}>
            <div className="flex flex-col gap-2 rounded-xl border border-line bg-panel/70 px-5 py-4 md:flex-row md:items-center md:justify-between">
              <div className="flex items-baseline gap-4">
                <span className="font-mono text-[11px] text-slate-600">L{i + 1}</span>
                <span className="text-sm font-semibold text-slate-100">{l.title}</span>
              </div>
              <span className="font-mono text-[11px] text-mint/80">{l.tag}</span>
            </div>
            <p className="px-5 pt-2 text-xs text-slate-500">{l.body}</p>
            {i < LAYERS.length - 1 && (
              <div className="ml-10 h-4 border-l border-dashed border-line" />
            )}
          </div>
        ))}
      </div>

      <p className="mt-8 max-w-3xl text-sm leading-6 text-slate-400">
        人类与 Agent 走同一套能力：GUI 底层的每个动作都落在 MCP 工具上，
        因此任何手工操作都能被 Agent 复现；反过来，Agent 的每一步也都能在 GUI 里回看。
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-line bg-panel/40">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-10 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <Mark className="h-7 w-7" />
          <div>
            <p className="text-sm text-slate-200">Simovia</p>
            <p className="text-xs text-slate-500">AI 第一用户，人类第二用户</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-5 text-xs text-slate-500">
          <a href={REPO} target="_blank" rel="noreferrer" className="hover:text-mint">
            仓库
          </a>
          <a href="#start" className="hover:text-mint">
            快速开始
          </a>
          <span className="font-mono">MuJoCo · MCP · LeRobot</span>
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <Nav />
      <main>
        <Hero />
        <Features />
        <AgentCoding />
        <Demos />
        <Onboarding />
        <Pipeline />
        <QuickStart />
        <Architecture />
      </main>
      <Footer />
    </div>
  );
}
