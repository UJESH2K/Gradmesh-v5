"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import CopyLine from "./CopyLine";
import Logo from "./Logo";
import SpaceBackdrop from "./SpaceBackdrop";
import { CREDIT_URL } from "./three/station";

type Props = {
  origin: string;
  signedIn: boolean;
  needsFirstAccount: boolean;
  meshName: string;
};

type Health = { coordinator?: string; lan?: string; setup?: { torch?: string; backend?: string } };

const VENDORS = [
  {
    key: "nvidia",
    name: "NVIDIA",
    api: "CUDA",
    spec: "GTX 900 → RTX 50",
    detail: "Build chosen from compute capability and driver: CUDA 13.0 from Turing on, 12.6 for older cards.",
    code: "sm_50 … sm_120",
  },
  {
    key: "intel",
    name: "Intel",
    api: "XPU",
    spec: "Arc A · Arc B · Core Ultra",
    detail: "PyTorch XPU with the validated Arc trainer. AMP off, foreach off, verified with a kernel launch.",
    code: "level zero",
  },
  {
    key: "apple",
    name: "Apple",
    api: "Metal",
    spec: "M1 and later · macOS 14+",
    detail: "MPS on unified memory, with batch sizes budgeted against what Metal will actually grant.",
    code: "mps · unified",
  },
];

/** Two identical GPUs, one with heavier per-round overhead (seconds). */
const SCHEDULE = {
  linear: [
    { name: "node a", fixed: 19, work: 21, total: 40 },
    { name: "node b", fixed: 30, work: 21, total: 51 },
  ],
  affine: [
    { name: "node a", fixed: 19, work: 26, total: 45 },
    { name: "node b", fixed: 30, work: 15, total: 45 },
  ],
};

const PHASES = [
  { name: "fetch", value: "0.07 s", note: "images come from the worker's cache; only weights move", width: 6 },
  { name: "load", value: "0.30 s", note: "checkpoint plus the global fp32 weights", width: 9 },
  { name: "setup", value: "0.26 s", note: "trainer and loaders; no AMP download", width: 8 },
  { name: "train", value: "1.04 s", note: "the only part that scales with the shard", width: 52 },
  { name: "post", value: "0.35 s", note: "no per-worker validation", width: 10 },
  { name: "aggregate", value: "0.15 s", note: "streamed FedAvg on the host", width: 15 },
];

function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch("/api/health", { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((payload) => alive && setHealth(payload))
        .catch(() => alive && setHealth(null));
    void load();
    const timer = setInterval(load, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  return health;
}

export default function Landing({ origin, signedIn, needsFirstAccount, meshName }: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const health = useHealth();
  const [platform, setPlatform] = useState<"windows" | "unix">("windows");

  useEffect(() => {
    if (/Mac|Linux/i.test(navigator.platform || navigator.userAgent)) setPlatform("unix");
  }, []);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let cleanup = () => {};
    let cancelled = false;

    (async () => {
      const [{ gsap }, { ScrollTrigger }] = await Promise.all([import("gsap"), import("gsap/ScrollTrigger")]);
      if (cancelled) return;
      gsap.registerPlugin(ScrollTrigger);
      const context = gsap.context(() => {
        gsap.from("[data-hero]", { y: 28, opacity: 0, duration: 1, ease: "power3.out", stagger: 0.08, delay: 0.15 });
        gsap.from(".ld-annotation", { opacity: 0, x: 18, duration: 0.9, ease: "power2.out", stagger: 0.18, delay: 0.9 });
        gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((element) => {
          gsap.from(element, {
            y: 34,
            opacity: 0,
            duration: 0.85,
            ease: "power3.out",
            scrollTrigger: { trigger: element, start: "top 85%", once: true },
          });
        });
        gsap.utils.toArray<HTMLElement>("[data-cascade]").forEach((group) => {
          gsap.from(group.children, {
            y: 26,
            opacity: 0,
            duration: 0.7,
            ease: "power3.out",
            stagger: 0.09,
            scrollTrigger: { trigger: group, start: "top 82%", once: true },
          });
        });
        gsap.utils.toArray<HTMLElement>("[data-grow]").forEach((bar) => {
          gsap.from(bar, {
            scaleX: 0,
            transformOrigin: "left center",
            duration: 1.1,
            ease: "power2.inOut",
            scrollTrigger: { trigger: bar.closest("[data-grow-root]") ?? bar, start: "top 75%", once: true },
          });
        });
        ScrollTrigger.create({
          start: 40,
          onUpdate: (self) => document.querySelector(".ld-nav")?.classList.toggle("is-stuck", self.scroll() > 40),
        });
      }, rootRef);
      cleanup = () => context.revert();
    })();

    return () => {
      cancelled = true;
      cleanup();
    };
  }, []);

  const dashboardHref = signedIn ? "/dashboard" : needsFirstAccount ? "/login?first=1" : "/login";
  const primaryLabel = signedIn ? "Open the dashboard" : needsFirstAccount ? "Claim this mesh" : "Sign in";
  const windowsCommand = `irm ${origin}/join.ps1 | iex`;
  const unixCommand = `curl -fsSL ${origin}/join.sh | sh`;
  const online = health?.coordinator === "up";
  let host = origin;
  try {
    host = new URL(origin).host;
  } catch {
    host = origin;
  }

  return (
    <div ref={rootRef} className="ld">
      <SpaceBackdrop />

      <header className="ld-nav">
        <div className="ld-container ld-nav-inner">
          <Link href="/" className="ld-brand">
            <Logo />
            <span>{meshName}</span>
            <span className="ld-version">v5</span>
          </Link>
          <nav className="ld-links">
            <a href="#vendors">Hardware</a>
            <a href="#scheduler">Scheduler</a>
            <a href="#round">A round</a>
            <a href="#join">Join</a>
          </nav>
          <div className="ld-nav-actions">
            <Link className="ld-btn ld-btn-ghost" href="/join">
              Lend a GPU
            </Link>
            <Link className="ld-btn ld-btn-solid" href={dashboardHref}>
              {signedIn ? "Dashboard" : primaryLabel}
            </Link>
          </div>
        </div>
      </header>

      <main>
        {/* hero ------------------------------------------------------------ */}
        <section className="ld-hero" data-scene-x="0.6" data-scene-y="-0.05" data-scene-zoom="1" data-scene-dim="0">
          <div className="ld-container ld-hero-grid">
            <div className="ld-hero-copy">
              <div className="ld-hud" data-hero>
                <span className={`ld-pulse${online ? " is-on" : ""}`} />
                <span>{online ? "mesh online" : "mesh offline"}</span>
                <span className="ld-hud-sep">/</span>
                <span>{host}</span>
                <span className="ld-hud-sep">/</span>
                <span>torch 2.13.0</span>
              </div>

              <h1 className="ld-title" data-hero>
                Every GPU in the room.
                <span className="ld-title-glow">One model.</span>
              </h1>

              <p className="ld-lede" data-hero>
                GradMesh turns the NVIDIA, Intel and Apple machines already on your network into a single training
                cluster. It measures what each one really delivers, sizes the work so they all finish together, and
                averages their updates into one model.
              </p>

              <div className="ld-actions" data-hero>
                <Link className="ld-btn ld-btn-solid ld-btn-lg" href={dashboardHref}>
                  {primaryLabel}
                  <span aria-hidden="true">→</span>
                </Link>
                <Link className="ld-btn ld-btn-ghost ld-btn-lg" href="/join">
                  Contribute this machine
                </Link>
              </div>

              <div className="ld-command" data-hero>
                <div className="ld-tabs" role="tablist">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={platform === "windows"}
                    className={platform === "windows" ? "is-on" : ""}
                    onClick={() => setPlatform("windows")}
                  >
                    Windows
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={platform === "unix"}
                    className={platform === "unix" ? "is-on" : ""}
                    onClick={() => setPlatform("unix")}
                  >
                    macOS · Linux
                  </button>
                </div>
                <CopyLine value={platform === "windows" ? windowsCommand : unixCommand} />
              </div>
            </div>

            <div className="ld-hero-stage" aria-hidden="true">
              <div className="ld-annotation" style={{ top: "16%", left: "4%" }}>
                <span className="vendor-dot vendor-nvidia" />
                <span>
                  <b>orbit 1</b> nvidia · cuda
                </span>
              </div>
              <div className="ld-annotation" style={{ top: "38%", right: "0%" }}>
                <span className="vendor-dot vendor-intel" />
                <span>
                  <b>orbit 2</b> intel · xpu
                </span>
              </div>
              <div className="ld-annotation" style={{ bottom: "22%", left: "0%" }}>
                <span className="vendor-dot vendor-apple" />
                <span>
                  <b>orbit 3</b> apple · metal
                </span>
              </div>
              <span className="ld-corner ld-corner-tl" />
              <span className="ld-corner ld-corner-br" />
            </div>
          </div>

          <div className="ld-container">
            <dl className="ld-telemetry" data-hero>
              <div>
                <dt>gpu vendors</dt>
                <dd>3</dd>
              </div>
              <div>
                <dt>software stack</dt>
                <dd>1</dd>
              </div>
              <div>
                <dt>overhead per round</dt>
                <dd>&lt; 1 s</dd>
              </div>
              <div>
                <dt>leg 1 ceiling, predicted</dt>
                <dd>1.53×</dd>
              </div>
            </dl>
          </div>
        </section>

        {/* vendors ---------------------------------------------------------- */}
        <section
          id="vendors"
          className="ld-section"
          data-scene-x="-0.62"
          data-scene-y="0"
          data-scene-zoom="1.08"
          data-scene-dim="0.1"
        >
          <div className="ld-container ld-split ld-split-right">
            <div className="ld-copy" data-reveal>
              <span className="ld-index">01 / 05</span>
              <h2>Three families. One mesh.</h2>
              <p>
                CUDA, XPU and Metal machines join the same round on the same pinned stack, so a model trained across a
                GeForce, an Arc and a MacBook averages exactly like one trained on three of the same card. Every round
                is reported per vendor.
              </p>
              <div className="ld-vendor-grid" data-cascade>
                {VENDORS.map((vendor) => (
                  <article key={vendor.key} className={`ld-vendor vendor-${vendor.key}`}>
                    <header>
                      <span className="ld-vendor-name">
                        <span className="vendor-dot" />
                        {vendor.name}
                      </span>
                      <span className="ld-vendor-api">{vendor.api}</span>
                    </header>
                    <strong>{vendor.spec}</strong>
                    <p>{vendor.detail}</p>
                    <code>{vendor.code}</code>
                  </article>
                ))}
              </div>
              <div className="ld-stack">
                <span>reference stack</span>
                <code>torch 2.13.0</code>
                <code>torchvision 0.28.0</code>
                <code>ultralytics 8.4.46</code>
              </div>
            </div>
          </div>
        </section>

        {/* scheduler --------------------------------------------------------- */}
        <section
          id="scheduler"
          className="ld-section"
          data-scene-x="0.66"
          data-scene-y="0.05"
          data-scene-zoom="0.95"
          data-scene-dim="0.25"
        >
          <div className="ld-container ld-split">
            <div className="ld-copy" data-reveal>
              <span className="ld-index">02 / 05</span>
              <h2>Overhead is part of the plan.</h2>
              <p>
                A round costs what the slowest machine costs. v5 models every machine as a fixed overhead plus a
                per-image rate, learns both from its own rounds, and sizes shards so everyone finishes at once. A
                machine whose overhead alone outlasts the round sits it out.
              </p>
              <div className="ld-equation">
                <span>T = (N + Σ rᵢ·fᵢ) / Σ rᵢ</span>
                <span className="ld-equation-sub">nᵢ = rᵢ · (T − fᵢ)</span>
              </div>

              <div className="ld-schedule" data-grow-root>
                {(["linear", "affine"] as const).map((arm) => (
                  <div key={arm} className={`ld-schedule-arm${arm === "affine" ? " is-good" : ""}`}>
                    <div className="ld-schedule-head">
                      <span>{arm === "affine" ? "v5 · affine" : "v4 · rate only"}</span>
                      <b>{Math.max(...SCHEDULE[arm].map((row) => row.total))} s round</b>
                    </div>
                    {SCHEDULE[arm].map((row) => (
                      <div key={row.name} className="ld-schedule-row">
                        <span>{row.name}</span>
                        <div className="ld-schedule-track">
                          <i className="is-fixed" data-grow style={{ width: `${(row.fixed / 55) * 100}%` }} />
                          <i className="is-work" data-grow style={{ width: `${(row.work / 55) * 100}%` }} />
                        </div>
                        <em>{row.total} s</em>
                      </div>
                    ))}
                  </div>
                ))}
                <p className="ld-footnote">
                  Two identical GPUs, one with heavier per-round overhead. Amber is overhead, blue is training. Leg 1
                  measured 18.77 s of fixed cost per round on an RTX 5070.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* round ------------------------------------------------------------- */}
        <section
          id="round"
          className="ld-section ld-section-wide"
          data-scene-x="0"
          data-scene-y="0.4"
          data-scene-zoom="0.62"
          data-scene-dim="0.6"
        >
          <div className="ld-container" data-reveal>
            <span className="ld-index ld-center">03 / 05</span>
            <h2 className="ld-center">Anatomy of a round.</h2>
            <p className="ld-center ld-narrow">
              Measured on one RTX 3050 with 56 images. Leg 1 paid nineteen seconds of overhead per round; v5 pays under
              one, because images stay cached, weights move as raw fp32 bytes, and nothing is validated twice.
            </p>
            <div className="ld-phases" data-cascade>
              {PHASES.map((phase) => (
                <div
                  key={phase.name}
                  className={`ld-phase${phase.name === "train" ? " is-train" : ""}`}
                  style={{ flexGrow: phase.width }}
                >
                  <span className="ld-phase-name">{phase.name}</span>
                  <span className="ld-phase-value">{phase.value}</span>
                  <span className="ld-phase-note">{phase.note}</span>
                </div>
              ))}
            </div>
            <div className="ld-facts" data-cascade>
              <div>
                <b>every gradient counts</b>
                <span>Pending gradients flush at the end of each round, so small shards still learn.</span>
              </div>
              <div>
                <b>fp32 end to end</b>
                <span>Workers send full-precision EMA weights, not a half-precision checkpoint.</span>
              </div>
              <div>
                <b>warmup once</b>
                <span>Learning-rate warmup in round 1 only, which fixed the accuracy falling after round 1.</span>
              </div>
            </div>
          </div>
        </section>

        {/* join -------------------------------------------------------------- */}
        <section
          id="join"
          className="ld-section"
          data-scene-x="-0.62"
          data-scene-y="0"
          data-scene-zoom="1.05"
          data-scene-dim="0.15"
        >
          <div className="ld-container ld-split ld-split-right">
            <div className="ld-copy" data-reveal>
              <span className="ld-index">04 / 05</span>
              <h2>One line to join.</h2>
              <p>
                It finds Python, detects the GPU, installs the matching build under the home folder, proves the GPU runs
                a kernel, and joins. Nothing system-wide. Ctrl+C leaves.
              </p>
              <div className="ld-terminal">
                <div className="ld-terminal-bar">
                  <i />
                  <i />
                  <i />
                  <span>contributor</span>
                </div>
                <div className="ld-terminal-body">
                  <CopyLine value={windowsCommand} label="Windows" />
                  <CopyLine value={unixCommand} label="macOS and Linux" />
                </div>
              </div>
              <ul className="ld-checks">
                <li>Python 3.10 to 3.13, offered on Windows if missing</li>
                <li>Stays awake while contributing, and finds the host again if it moves</li>
                <li>One worker per GPU; a restart replaces the old process cleanly</li>
              </ul>
            </div>
          </div>
        </section>

        {/* trust ------------------------------------------------------------- */}
        <section className="ld-section" data-scene-x="0.62" data-scene-y="0" data-scene-zoom="1" data-scene-dim="0.3">
          <div className="ld-container ld-split">
            <div className="ld-copy" data-reveal>
              <span className="ld-index">05 / 05</span>
              <h2>Nothing leaves the network.</h2>
              <div className="ld-trust" data-cascade>
                <div>
                  <b>LAN only</b>
                  <span>Shards, checkpoints and updates move between machines you can see. Telemetry off.</span>
                </div>
                <div>
                  <b>Token to join</b>
                  <span>Workers present a mesh token before any data. Rotate it and every machine drops out.</span>
                </div>
                <div>
                  <b>Every decision explained</b>
                  <span>Admission, shard size, drops and slowdowns each carry a written reason.</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="ld-cta" data-scene-x="0" data-scene-y="0.7" data-scene-zoom="0.8" data-scene-dim="0">
          <div className="ld-container" data-reveal>
            <h2>The cluster you were going to rent is already on your Wi-Fi.</h2>
            <div className="ld-actions ld-actions-center">
              <Link className="ld-btn ld-btn-solid ld-btn-lg" href={dashboardHref}>
                {primaryLabel}
                <span aria-hidden="true">→</span>
              </Link>
              <Link className="ld-btn ld-btn-ghost ld-btn-lg" href="/join">
                Lend a GPU instead
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="ld-footer">
        <div className="ld-container ld-footer-inner">
          <span className="ld-brand">
            <Logo size={18} />
            <span>GradMesh 5</span>
          </span>
          <span>Collaborative GPU training over an ordinary network. Research prototype.</span>
          <a href={CREDIT_URL} target="_blank" rel="noreferrer">
            Scene: “space boi” by silvercrow101, CC BY-NC 4.0
          </a>
        </div>
      </footer>
    </div>
  );
}
