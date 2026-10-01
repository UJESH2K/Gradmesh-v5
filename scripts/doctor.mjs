/**
 * `npm run doctor` — tell the user exactly why the mesh will not start, and how
 * to fix it, on whichever machine this is.
 *
 * Every check prints a fix, not just a status, because the person running this
 * is usually a contributor who did not set the project up. The checks that
 * matter most are the ones that only fail on the *second* machine: a checkout
 * synced from another computer, dependencies built for another operating
 * system, a Python environment that points at an interpreter that is not here,
 * and a PyTorch build that does not match this GPU.
 *
 *   npm run doctor            human-readable report
 *   npm run doctor -- --json  machine-readable, for bug reports
 */

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { hostname } from "node:os";
import path from "node:path";

import { readSetupState } from "./bootstrap.mjs";
import {
  COORDINATOR_PORT,
  ENGINE_DIR,
  IS_WINDOWS,
  PATHS,
  PYTHON_MAX,
  PYTHON_MIN,
  REPO_ROOT,
  STATE_DIR,
  WEB_PORT,
  detectGpuProfile,
  findSystemPython,
  lanAddresses,
  paint,
  pythonInstallHint,
  readCoordinatorState,
  venvPython,
  venvReady,
} from "./lib/env.mjs";
import { isSyncedPath } from "./lib/paths.mjs";

const results = [];
const asJson = process.argv.includes("--json");

/** level: ok, warn or fail. Warnings do not fail the run. */
function check(name, level, detail, fix = "") {
  results.push({ name, level: level === true ? "ok" : level === false ? "fail" : level, detail, fix });
}

async function reachable(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2500) });
    return response.ok ? await response.json().catch(() => ({})) : null;
  } catch {
    return null;
  }
}

function readJson(file) {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function referenceStack() {
  const text = readFileSync(path.join(ENGINE_DIR, "version.py"), "utf8");
  const pick = (key) => text.match(new RegExp(`"${key}":\\s*"([^"]+)"`))?.[1];
  return { torch: pick("torch"), torchvision: pick("torchvision"), ultralytics: pick("ultralytics") };
}

function nextSwcPresent() {
  // Next ships its compiler as a per-platform package. node_modules copied or
  // synced from another OS has the wrong one, and `next dev` fails to start.
  const scope = path.join(REPO_ROOT, "node_modules", "@next");
  if (!existsSync(scope)) return { ok: false, found: [] };
  const found = readdirSync(scope).filter((name) => name.startsWith("swc-"));
  const wanted = `swc-${process.platform}-${process.arch}`;
  return { ok: found.some((name) => name.startsWith(wanted)), found, wanted };
}

function longPathsEnabled() {
  if (!IS_WINDOWS) return true;
  const probe = spawnSync(
    "reg",
    ["query", "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem", "/v", "LongPathsEnabled"],
    { encoding: "utf8", windowsHide: true }
  );
  return /0x1\b/.test(probe.stdout || "");
}

function firewallRulePresent() {
  if (!IS_WINDOWS) return null;
  const probe = spawnSync(
    "powershell",
    [
      "-NoProfile",
      "-Command",
      "(Get-NetFirewallRule -DisplayName 'GradMesh' -ErrorAction SilentlyContinue | Measure-Object).Count",
    ],
    { encoding: "utf8", windowsHide: true, timeout: 15000 }
  );
  return Number((probe.stdout || "0").trim()) > 0;
}

async function main() {
  // --- This checkout --------------------------------------------------------
  const nodeMajor = Number(process.versions.node.split(".")[0]);
  check("Node.js", nodeMajor >= 20, `v${process.versions.node}`, "Install Node.js 20 or newer from nodejs.org");

  const hasNext = existsSync(path.join(REPO_ROOT, "node_modules", "next"));
  check("npm dependencies", hasNext, hasNext ? "installed" : "missing", "Run: npm install");
  if (hasNext) {
    const swc = nextSwcPresent();
    check(
      "npm dependencies match this OS",
      swc.ok ? "ok" : "fail",
      swc.ok ? `${process.platform}-${process.arch}` : `found ${swc.found.join(", ") || "none"}, need ${swc.wanted}`,
      "node_modules came from another computer. Delete the node_modules folder and run: npm install"
    );
  }

  check(
    "Checkout location",
    PATHS.synced ? "warn" : "ok",
    PATHS.synced ? `${REPO_ROOT} is in a synced folder` : REPO_ROOT,
    PATHS.synced
      ? `Supported: the Python environment is kept machine-locally in ${PATHS.machineDir}. For less sync traffic, ` +
          "clone the repository outside OneDrive or Dropbox, for example C:\\dev\\gradmesh."
      : ""
  );
  if (isSyncedPath(STATE_DIR)) {
    check(
      "Runtime state location",
      "warn",
      `${STATE_DIR} syncs to your other devices`,
      "Datasets and run artifacts sync too. For large datasets set GRADMESH_STATE_DIR to a local folder."
    );
  }
  if (IS_WINDOWS) {
    check(
      "Windows long paths",
      longPathsEnabled() ? "ok" : "warn",
      longPathsEnabled() ? "enabled" : "disabled",
      "PyTorch installs can exceed 260 characters. From an Administrator PowerShell: " +
        "New-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
    );
  }

  // --- Python -----------------------------------------------------------------
  const python = findSystemPython();
  const pythonOk = Boolean(python && !python.missing);
  check(
    "Python",
    pythonOk,
    pythonOk
      ? `${python.command} ${python.args.join(" ")} (${python.version})`.replace("  ", " ")
      : python?.rejected?.length
        ? `found ${python.rejected.join(", ")}, need ${PYTHON_MIN.join(".")} to ${PYTHON_MAX.join(".")}`
        : "not found",
    `Install Python ${PYTHON_MIN.join(".")}-${PYTHON_MAX.join(".")}: ${pythonInstallHint()}`
  );

  const marker = readJson(path.join(PATHS.venv, "gradmesh-env.json"));
  let envOk = venvReady();
  let envDetail = envOk ? PATHS.venv : `missing (${PATHS.venv})`;
  if (envOk) {
    const probe = spawnSync(venvPython(), ["-c", "import sys;print(sys.version.split()[0])"], {
      encoding: "utf8",
      windowsHide: true,
      timeout: 60000,
    });
    if (probe.status !== 0) {
      envOk = false;
      envDetail = "its interpreter will not start (copied from another machine?)";
    } else if (marker?.hostname && marker.hostname.toLowerCase() !== hostname().toLowerCase()) {
      envOk = false;
      envDetail = `built on ${marker.hostname}`;
    } else {
      envDetail = `${PATHS.venv} (Python ${probe.stdout.trim()})`;
    }
  }
  check("Python environment", envOk, envDetail, "Run: npm run setup   (it rebuilds an environment from another machine)");

  // --- Setup record and the training stack ----------------------------------
  const setup = readSetupState();
  check(
    "Coordinator runtime",
    setup.controlPlane === "ready",
    setup.controlPlane,
    "Run: npm run setup"
  );

  const detected = detectGpuProfile({ host: true });
  const profile = detected.profile || {};
  const gpus = detected.facts?.gpus || [];
  check(
    "GPUs",
    gpus.length ? "ok" : "warn",
    gpus.length
      ? gpus
          .map(
            (gpu) =>
              `${gpu.name}${gpu.compute_capability ? ` (compute ${gpu.compute_capability})` : ""}` +
              `${gpu.driver ? `, driver ${gpu.driver}` : ""}`
          )
          .join("; ")
      : "none detected",
    "This machine can host and aggregate, but needs an NVIDIA, Intel Arc or Apple Silicon GPU to train."
  );
  check(
    "PyTorch build for this machine",
    profile.blocked ? "warn" : "ok",
    `${profile.label || "unknown"} (${profile.requirements || "?"})${profile.reason ? ` — ${profile.reason}` : ""}`,
    profile.blocked ? `${profile.blocked} ${profile.fix || ""}` : ""
  );
  for (const warning of profile.warnings || []) check("Build note", "warn", warning, "");

  check(
    "Training runtime",
    setup.trainingPlane === "ready" ? "ok" : setup.trainingPlane === "installing" ? "warn" : "fail",
    setup.trainingPlane === "ready"
      ? `torch ${setup.torch || "?"}, ultralytics ${setup.ultralytics || "?"}`
      : setup.trainingPlane,
    setup.trainingPlane === "installing"
      ? "Still installing; this downloads PyTorch and takes a few minutes."
      : `Run: npm run setup. Background install log: ${PATHS.setupLog}`
  );
  check(
    "Installed build matches this GPU",
    !setup.profile || !profile.name || setup.profile === profile.name ? "ok" : "fail",
    setup.profile ? `installed ${setup.profile}, this machine wants ${profile.name}` : "nothing installed yet",
    "Run: npm run setup. The GPU or driver changed since the last install, or the record came from another machine."
  );
  if (setup.trainingPlane === "ready") {
    check(
      "GPU runs a kernel",
      setup.accelerator === "ok" ? "ok" : profile.backend === "cpu" ? "warn" : "fail",
      setup.accelerator === "ok"
        ? `${setup.verify?.device || "ok"} on ${setup.backend}`
        : setup.acceleratorProblem || "not verified",
      setup.fix || "Update the GPU driver, then run: npm run setup"
    );
    const reference = referenceStack();
    const torchBase = String(setup.torch || "").split("+")[0];
    check(
      "Reference stack",
      torchBase === reference.torch && setup.ultralytics === reference.ultralytics ? "ok" : "warn",
      `torch ${torchBase || "?"} (reference ${reference.torch}), ultralytics ${setup.ultralytics || "?"} (reference ${reference.ultralytics})`,
      "Results from machines on different stacks are not directly comparable. Run: npm run setup -- --force"
    );
  }

  const models = setup.models || [];
  check("Base checkpoints", models.length > 0, models.join(", ") || "none", "Run: npm run setup while online");

  // --- Mesh state -----------------------------------------------------------
  const state = readCoordinatorState();
  check(
    "Mesh token",
    Boolean(state?.mesh_token),
    state?.mesh_token ? "present" : "not minted yet",
    "Start the host once with: npm run dev"
  );

  const datasets = Object.values(state?.datasets || {}).filter((record) => !record.is_subset);
  const missing = datasets.filter((record) => {
    const stored = record.extracted_path || "";
    const local = stored.startsWith("state:") ? path.join(STATE_DIR, stored.slice(6)) : stored;
    return local && !existsSync(local);
  });
  check(
    "Datasets",
    datasets.length === 0 ? "warn" : missing.length ? "warn" : "ok",
    datasets.length === 0
      ? "none uploaded"
      : missing.length
        ? `${missing.length} of ${datasets.length} are not on this machine: ${missing.map((d) => d.name).join(", ")}`
        : `${datasets.length} registered`,
    datasets.length === 0
      ? "Upload a YOLO dataset zip on the Datasets page, or import a standard one in Testing parameters"
      : "Upload them again here, or set GRADMESH_STATE_DIR to where they live"
  );

  // --- Network ----------------------------------------------------------------
  const addresses = lanAddresses();
  const real = addresses.filter((entry) => !entry.virtual);
  check(
    "Network",
    real.length > 0 ? "ok" : addresses.length ? "warn" : "fail",
    addresses.map((entry) => `${entry.address} (${entry.name}${entry.virtual ? ", virtual" : ""})`).join(", ") ||
      "no LAN address",
    "Connect to Wi-Fi or Ethernet so peers can reach this host"
  );
  const firewall = firewallRulePresent();
  if (firewall !== null) {
    check(
      "Firewall rule",
      firewall ? "ok" : "warn",
      firewall ? "GradMesh rule present" : "no GradMesh rule",
      "Other devices cannot reach ports 3000 and 8000 until Windows allows them. From an Administrator PowerShell: " +
        'New-NetFirewallRule -DisplayName "GradMesh" -Direction Inbound -Protocol TCP -LocalPort 3000,8000 -Action Allow'
    );
  }

  const coordinator = await reachable(`http://127.0.0.1:${COORDINATOR_PORT}/health`);
  check(
    "Coordinator running",
    coordinator ? "ok" : "warn",
    coordinator ? `port ${COORDINATOR_PORT}, version ${coordinator.version}, ${coordinator.nodes_active} machines online` : `nothing on port ${COORDINATOR_PORT}`,
    "Run: npm run dev"
  );
  if (coordinator) {
    check(
      "Coordinator has PyTorch",
      coordinator.torch_ready ? "ok" : "fail",
      coordinator.torch_ready ? "aggregation available" : "torch not importable",
      "Restart npm run dev once the training runtime finishes installing"
    );
    check(
      "gradmesh.local",
      coordinator.mdns?.active ? "ok" : "warn",
      coordinator.mdns?.active ? `advertised at ${coordinator.mdns.address}` : coordinator.mdns?.error || "not advertised",
      "This network blocks multicast. Other devices use the IP address the launcher prints instead."
    );
  }
  const dashboard = await reachable(`http://127.0.0.1:${WEB_PORT}/api/health`);
  check("Dashboard running", dashboard ? "ok" : "warn", dashboard ? `port ${WEB_PORT}` : `nothing on port ${WEB_PORT}`, "Run: npm run dev");

  // --- Report -----------------------------------------------------------------
  if (asJson) {
    console.log(JSON.stringify({ results, paths: PATHS, stateDir: STATE_DIR, profile: detected }, null, 2));
    process.exit(results.some((r) => r.level === "fail") ? 1 : 0);
  }

  console.log("");
  for (const result of results) {
    const mark =
      result.level === "ok" ? paint("green", "  ok  ") : result.level === "warn" ? paint("yellow", " warn ") : paint("red", " fail ");
    console.log(`${mark} ${result.name.padEnd(32)} ${paint("gray", result.detail || "")}`);
    if (result.level !== "ok" && result.fix) {
      console.log(`       ${paint("yellow", result.fix)}`);
    }
  }

  const failures = results.filter((result) => result.level === "fail").length;
  const warnings = results.filter((result) => result.level === "warn").length;
  console.log("");
  console.log(
    failures === 0
      ? paint("green", `Everything required checks out${warnings ? `, with ${warnings} note${warnings === 1 ? "" : "s"}` : ""}.`)
      : paint("yellow", `${failures} item${failures === 1 ? "" : "s"} need attention.`)
  );
  console.log(paint("gray", `Machine-local files: ${PATHS.machineDir}`));
  console.log(paint("gray", `Shared mesh state:   ${STATE_DIR}`));
  console.log(paint("gray", "Manual setup and every requirement: SETUP.md"));
  process.exit(failures === 0 ? 0 : 1);
}

main();
