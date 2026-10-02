import { readFileSync } from "node:fs";
import path from "node:path";
import { headers } from "next/headers";
import Link from "next/link";

import CopyLine from "@/components/CopyLine";
import VendorBadge from "@/components/dashboard/VendorBadge";
import { coordinator } from "@/lib/coordinator";
import { COORDINATOR_PORT, MACHINE, STATE_DIR, meshOrigin, meshToken, setupState } from "@/lib/config";
import { VENDORS } from "@/lib/format";
import type { MeshState } from "@/lib/types";

export const metadata = { title: "Setup and health" };
export const dynamic = "force-dynamic";

type Level = "ok" | "warn" | "fail";
type Check = { name: string; level: Level; detail: string; fix?: string };

function referenceStack(): Record<string, string> {
  try {
    const text = readFileSync(path.join(process.cwd(), "engine", "version.py"), "utf8");
    const pick = (key: string) => text.match(new RegExp(`"${key}":\\s*"([^"]+)"`))?.[1] ?? "?";
    return { torch: pick("torch"), torchvision: pick("torchvision"), ultralytics: pick("ultralytics") };
  } catch {
    return { torch: "?", torchvision: "?", ultralytics: "?" };
  }
}

const VENDOR_STEPS: Record<string, { before: string[]; drivers?: { label: string; href: string } }> = {
  cuda: {
    before: [
      "Install or update the NVIDIA driver. Driver 580 or newer gets the CUDA 13.0 build; RTX 50 series cards need at least 570.",
      "Python 3.10 to 3.13 (3.12 recommended). The command offers to install it on Windows.",
    ],
    drivers: { label: "NVIDIA drivers", href: "https://www.nvidia.com/Download/index.aspx" },
  },
  xpu: {
    before: [
      "Install the latest Intel Arc and Iris Xe graphics driver (Windows), or Intel's compute runtime packages (Linux).",
      "Python 3.10 to 3.13. No oneAPI toolkit is needed; the PyTorch XPU wheels carry their runtime.",
    ],
    drivers: {
      label: "Intel graphics drivers",
      href: "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html",
    },
  },
  mps: {
    before: [
      "An Apple Silicon Mac (M1 or later) on macOS 14 Sonoma or newer. Intel Macs cannot run current PyTorch.",
      "An Apple Silicon Python 3.10 to 3.13: brew install python@3.12 (Homebrew in /opt/homebrew) or python.org. The built-in python3 is too old, and an Intel Python under Rosetta has no PyTorch; the command picks the native one.",
      "On the day: plugged in, lid open on a hard surface, Low Power Mode off, browsers closed. An 8 GB Mac shares that memory with the GPU. Docs: docs/MAC-M1.md.",
    ],
  },
};

function Mark({ level }: { level: Level }) {
  return <span className={`check-mark is-${level}`}>{level === "ok" ? "✓" : level === "warn" ? "!" : "×"}</span>;
}

export default async function SetupPage() {
  const headerList = await headers();
  const origin = meshOrigin(headerList.get("host"));
  const hostname = new URL(origin).hostname;
  const token = meshToken();
  const setup = setupState();
  const reference = referenceStack();

  let mesh: MeshState | null = null;
  let coordinatorError: string | null = null;
  try {
    mesh = await coordinator<MeshState>("/mesh", { timeoutMs: 6000 });
  } catch (error) {
    coordinatorError = (error as Error).message;
  }

  const torchBase = String(setup.torch || "").split("+")[0];
  const checks: Check[] = [
    {
      name: "Coordinator",
      level: mesh ? "ok" : "fail",
      detail: mesh ? `version ${mesh.version || "?"}, protocol ${mesh.protocol ?? "?"}` : coordinatorError || "not reachable",
      fix: "Start the host with npm run dev.",
    },
    {
      name: "Python environment",
      level: setup.controlPlane === "ready" ? "ok" : "fail",
      detail: `${setup.controlPlane} · ${MACHINE.venv}`,
      fix: "Run npm run setup in the project folder.",
    },
    {
      name: "PyTorch on this host",
      level: setup.trainingPlane === "ready" ? "ok" : setup.trainingPlane === "installing" ? "warn" : "fail",
      detail:
        setup.trainingPlane === "ready"
          ? `${setup.profileLabel || setup.profile || "installed"} · torch ${setup.torch || "?"}`
          : setup.trainingPlane,
      fix:
        setup.trainingPlane === "installing"
          ? "Still downloading; runs can start when it finishes."
          : `Run npm run setup. The background install log is ${MACHINE.setupLog}.`,
    },
    {
      name: "This host's GPU",
      level: setup.accelerator === "ok" ? "ok" : setup.backend === "cpu" ? "warn" : setup.accelerator ? "fail" : "warn",
      detail:
        setup.accelerator === "ok"
          ? `${setup.gpu || "GPU"} ran a test kernel on ${setup.backend}`
          : setup.acceleratorProblem || setup.profileReason || "not checked yet",
      fix:
        setup.backend === "cpu"
          ? "The host can coordinate and aggregate without a GPU; it just cannot contribute one."
          : setup.fix || "Update the GPU driver, then run npm run setup.",
    },
    {
      name: "Reference stack",
      level: !setup.torch ? "warn" : torchBase === reference.torch && setup.ultralytics === reference.ultralytics ? "ok" : "warn",
      detail: `torch ${reference.torch}, torchvision ${reference.torchvision}, ultralytics ${reference.ultralytics}`,
      fix: `This host runs torch ${setup.torch || "?"} and ultralytics ${setup.ultralytics || "?"}. Run npm run setup -- --force.`,
    },
    {
      name: "Checkout location",
      level: MACHINE.synced ? "warn" : "ok",
      detail: MACHINE.synced ? "in a synced folder (OneDrive, Dropbox or iCloud)" : process.cwd(),
      fix: `Supported: machine-specific files live in ${MACHINE.machineDir}, so opening this folder on another device does not break it. Cloning outside the synced folder still saves sync traffic.`,
    },
  ];
  for (const warning of setup.warnings || []) {
    checks.push({ name: "Build note", level: "warn", detail: warning });
  }

  // The coordinator compares torch, torchvision and Ultralytics per machine;
  // fall back to the torch version alone for a host that predates that.
  const offStack = (mesh?.nodes || []).filter((node) => {
    if (node.software) return node.software.on_reference === false;
    const version = String(node.capability?.torch_version || "").split("+")[0];
    return version && version !== reference.torch;
  });

  const windows = `irm ${origin}/join.ps1 | iex`;
  const unix = `curl -fsSL ${origin}/join.sh | sh`;
  const manual = `python setup_env.py agent --server http://${hostname}:${COORDINATOR_PORT} --token ${token || "<token>"}`;

  return (
    <>
      <div>
        <h1 className="page-title">Setup and health</h1>
        <p className="small faint" style={{ marginTop: 4 }}>
          Whether this host is ready, and exactly what a machine of each kind needs to join. Everything here is also
          in <span className="mono">SETUP.md</span>, and <span className="mono">npm run doctor</span> runs the same
          checks in a terminal.
        </p>
      </div>

      <section className="panel">
        <div className="check-list">
          {checks.map((check) => (
            <div className="check-row" key={check.name + check.detail}>
              <Mark level={check.level} />
              <div style={{ minWidth: 0 }}>
                <div className="row-between" style={{ gap: 12 }}>
                  <strong className="small">{check.name}</strong>
                  <span className="small faint truncate" style={{ maxWidth: "70%" }} title={check.detail}>
                    {check.detail}
                  </span>
                </div>
                {check.level !== "ok" && check.fix ? (
                  <p className="small" style={{ color: check.level === "fail" ? "var(--danger)" : "var(--warn)", marginTop: 4 }}>
                    {check.fix}
                  </p>
                ) : null}
              </div>
            </div>
          ))}
        </div>
        <p className="small faint" style={{ marginTop: 14 }}>
          Shared mesh state (accounts, token, datasets): <span className="mono">{STATE_DIR}</span>
        </p>
      </section>

      {offStack.length > 0 ? (
        <div className="notice notice-warn">
          <strong>
            {offStack.length} machine{offStack.length === 1 ? " is" : "s are"} not on the reference stack.
          </strong>{" "}
          {offStack
            .map((node) =>
              node.software?.drift.length
                ? `${node.display_name} (${node.software.drift.join(", ")})`
                : `${node.display_name} (torch ${node.capability?.torch_version})`
            )
            .join("; ")}
          . Their
          results are not directly comparable with the rest; rerunning the join command on them updates them.
        </div>
      ) : null}

      <section className="panel panel-flush">
        <header className="panel-header">
          <span className="panel-title">Add a machine</span>
          <Link className="btn btn-sm" href="/dashboard/invite">
            Invite page
          </Link>
        </header>
        <div className="panel-body stack">
          <div className="grid grid-2">
            <CopyLine value={windows} label="Windows: PowerShell" />
            <CopyLine value={unix} label="macOS and Linux: Terminal" />
          </div>
          <p className="small faint">
            The command detects the GPU, installs the matching PyTorch build into its own folder under the home
            directory, proves the GPU runs a kernel, and joins. Run it again at any time; it only installs what
            changed.
          </p>
          <div className="grid grid-3">
            {VENDORS.map((vendor) => {
              const steps = VENDOR_STEPS[vendor.backend];
              const online = mesh?.backends?.find((item) => item.backend === vendor.backend)?.online ?? 0;
              return (
                <div key={vendor.backend} className={`mix-card vendor-${vendor.vendor}`}>
                  <div className="row-between">
                    <VendorBadge backend={vendor.backend} />
                    <span className="small faint">{online} online</span>
                  </div>
                  <ul className="small muted" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.65 }}>
                    {steps.before.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ul>
                  {steps.drivers ? (
                    <a className="small accent" href={steps.drivers.href} target="_blank" rel="noreferrer">
                      {steps.drivers.label}
                    </a>
                  ) : null}
                </div>
              );
            })}
          </div>
          <details>
            <summary className="small faint" style={{ cursor: "pointer" }}>
              If the one-line command does not work
            </summary>
            <div className="stack-sm" style={{ marginTop: 12 }}>
              <p className="small muted">
                Download the agent files from <span className="mono">{origin}/api/agent/&lt;file&gt;</span> (the list
                is in SETUP.md), or copy the repository&apos;s <span className="mono">engine</span> folder, then from
                that folder run:
              </p>
              <CopyLine tone="muted" value={manual} />
              <p className="small faint">
                Or, with the repository cloned: <span className="mono">npm run worker -- --server http://{hostname}:
                {COORDINATOR_PORT} --token {token ? "<token>" : "<token>"}</span>. SETUP.md lists every requirement
                and a fully manual install, step by step, for each operating system and vendor.
              </p>
            </div>
          </details>
        </div>
      </section>
    </>
  );
}
