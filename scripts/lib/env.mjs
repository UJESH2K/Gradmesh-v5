import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import net from "node:net";
import { networkInterfaces } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { machinePaths, stateDir } from "./paths.mjs";

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export const ENGINE_DIR = path.join(REPO_ROOT, "engine");
export const STATE_DIR = stateDir(REPO_ROOT);
export const PATHS = machinePaths(REPO_ROOT);
export const VENV_DIR = PATHS.venv;
export const IS_WINDOWS = process.platform === "win32";

/** Interpreters PyTorch 2.13 publishes wheels for on every supported backend. */
export const PYTHON_MIN = [3, 10];
export const PYTHON_MAX = [3, 13];

export const COORDINATOR_PORT = Number(process.env.GRADMESH_COORDINATOR_PORT || 8000);
export const WEB_PORT = Number(process.env.PORT || 3000);

const RESET = "\u001b[0m";
const COLORS = {
  gray: "\u001b[90m",
  cyan: "\u001b[36m",
  green: "\u001b[32m",
  yellow: "\u001b[33m",
  red: "\u001b[31m",
  magenta: "\u001b[35m",
  bold: "\u001b[1m",
};

export function paint(color, text) {
  if (!process.stdout.isTTY) return text;
  return `${COLORS[color] || ""}${text}${RESET}`;
}

export function log(scope, message, color = "cyan") {
  console.log(`${paint(color, scope.padEnd(11))} ${message}`);
}

export function banner(lines) {
  const width = Math.max(...lines.map((line) => stripAnsi(line).length)) + 2;
  const bar = "─".repeat(width);
  console.log(paint("gray", `╭${bar}╮`));
  for (const line of lines) {
    const pad = " ".repeat(width - stripAnsi(line).length - 1);
    console.log(`${paint("gray", "│")} ${line}${pad}${paint("gray", "│")}`);
  }
  console.log(paint("gray", `╰${bar}╯`));
}

function stripAnsi(value) {
  // eslint-disable-next-line no-control-regex
  return value.replace(new RegExp(String.fromCharCode(27) + "\\[[0-9;]*m", "g"), "");
}

export function venvPython() {
  return PATHS.venvPython;
}

/** Present on disk. Whether it actually runs here is setup_env.py's job to check. */
export function venvReady() {
  return existsSync(venvPython());
}

function versionSupported(major, minor) {
  const value = major * 100 + minor;
  return value >= PYTHON_MIN[0] * 100 + PYTHON_MIN[1] && value <= PYTHON_MAX[0] * 100 + PYTHON_MAX[1];
}

/**
 * Find a system Python that PyTorch 2.13 has wheels for.
 *
 * Accepting "3.9 or newer", as v4 did, is how a machine with only Python 3.14
 * got as far as a 2 GB download before pip announced there was no matching
 * wheel. The range is checked up front instead, preferring 3.12.
 *
 * Windows ships a `python` shim that opens the Microsoft Store instead of
 * running anything, so a version check is the only reliable probe.
 */
export function findSystemPython() {
  const preferred = ["3.12", "3.11", "3.13", "3.10"];
  const candidates = IS_WINDOWS
    ? [...preferred.map((version) => `py -${version}`), "python", "python3"]
    : [
        ...preferred.map((version) => `python${version}`),
        ...(process.platform === "darwin"
          ? preferred.flatMap((version) => [
              `/opt/homebrew/bin/python${version}`,
              `/usr/local/bin/python${version}`,
              `/Library/Frameworks/Python.framework/Versions/${version}/bin/python3`,
            ])
          : []),
        "python3",
        "python",
      ];

  const rejected = [];
  for (const candidate of candidates) {
    const [command, ...args] = candidate.startsWith("/") ? [candidate] : candidate.split(" ");
    // No shell here: cmd.exe mangles the quoting, and PATH lookup for .exe
    // files works without one anyway.
    const probe = spawnSync(
      command,
      [...args, "-c", "import sys;print(sys.version_info.major);print(sys.version_info.minor);print(sys.executable)"],
      { encoding: "utf8", windowsHide: true }
    );
    if (probe.status !== 0) continue;
    const [majorText, minorText, executable] = (probe.stdout || "").trim().split(/\r?\n/);
    const major = Number(majorText);
    const minor = Number(minorText);
    if (major === 3 && versionSupported(major, minor)) {
      return { command, args, version: `${major}.${minor}`, executable: executable?.trim() };
    }
    if (Number.isFinite(major) && Number.isFinite(minor)) rejected.push(`${major}.${minor}`);
  }
  return rejected.length ? { missing: true, rejected: [...new Set(rejected)] } : null;
}

export function pythonInstallHint() {
  if (IS_WINDOWS) return "winget install Python.Python.3.12";
  if (process.platform === "darwin") return "brew install python@3.12   (macOS's built-in python3 is 3.9, too old)";
  return "sudo apt install python3.12 python3.12-venv   (or your distribution's equivalent)";
}

/** findSystemPython, or exit with an instruction a person can act on. */
export function requireSystemPython() {
  const found = findSystemPython();
  if (found && !found.missing) return found;
  const rejected = found?.rejected?.length
    ? ` Found ${found.rejected.join(", ")}, which the pinned PyTorch has no wheels for.`
    : "";
  console.error(
    paint(
      "red",
      `\nGradMesh needs Python ${PYTHON_MIN.join(".")} to ${PYTHON_MAX.join(".")} on PATH.${rejected}\n` +
        `  Install it:  ${pythonInstallHint()}\n` +
        "  then open a new terminal and run the command again.\n"
    )
  );
  process.exit(1);
}

let cachedProfile = null;

/**
 * This machine's GPU and the PyTorch build it needs, from engine/hardware.py.
 *
 * The rules live in Python, in one place, because the join flow runs them on
 * machines that have no Node. This only shells out to them.
 */
export function detectGpuProfile({ host = true } = {}) {
  if (cachedProfile) return cachedProfile;
  const system = findSystemPython();
  const python = system && !system.missing ? system : venvReady() ? { command: venvPython(), args: [] } : null;
  if (!python) {
    return {
      facts: null,
      profile: { name: "unknown", backend: "unknown", label: "unknown", reason: "no supported Python was found" },
    };
  }
  const probe = spawnSync(
    python.command,
    [...python.args, path.join(ENGINE_DIR, "hardware.py"), "--json", ...(host ? ["--host"] : [])],
    { encoding: "utf8", windowsHide: true, timeout: 60000 }
  );
  try {
    cachedProfile = JSON.parse(probe.stdout);
  } catch {
    cachedProfile = {
      facts: null,
      profile: {
        name: "unknown",
        backend: "unknown",
        label: "unknown",
        reason: (probe.stderr || "hardware detection failed").trim().slice(-300),
      },
    };
  }
  return cachedProfile;
}

export function ensureStateDir() {
  if (!existsSync(STATE_DIR)) mkdirSync(STATE_DIR, { recursive: true });
  return STATE_DIR;
}

export function readCoordinatorState() {
  const file = path.join(STATE_DIR, "coordinator.json");
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

// Adapters that exist on the host but that no other device on the Wi-Fi can
// reach: hypervisor host-only networks, WSL, Docker, VPN overlays. v4 ranked by
// address range alone, so VirtualBox's 192.168.56.1 beat the real Wi-Fi address
// and every printed join command pointed at a network nobody else was on.
const VIRTUAL_ADAPTER =
  /vethernet|virtualbox|vbox|vmware|vmnet|hyper-v|wsl|docker|br-|veth|tailscale|zerotier|wireguard|utun|tun\d|tap\d|vpn|npcap|loopback|bluetooth/i;

/** Every routable IPv4 address this host answers on, best guess first. */
export function lanAddresses() {
  const found = [];
  for (const [name, entries] of Object.entries(networkInterfaces())) {
    for (const entry of entries || []) {
      if (entry.family !== "IPv4" || entry.internal) continue;
      if (entry.address.startsWith("169.254.")) continue; // link-local: no DHCP answer
      found.push({ name, address: entry.address, virtual: VIRTUAL_ADAPTER.test(name) });
    }
  }
  const range = (address) =>
    address.startsWith("192.168.") ? 0 : address.startsWith("10.") ? 1 : address.startsWith("172.") ? 2 : 3;
  const wireless = (name) => (/wi-?fi|wlan|wireless|en0|eth|ethernet/i.test(name) ? 0 : 1);
  found.sort(
    (a, b) =>
      Number(a.virtual) - Number(b.virtual) ||
      wireless(a.name) - wireless(b.name) ||
      range(a.address) - range(b.address)
  );
  return found;
}

export function primaryLanAddress() {
  return lanAddresses()[0]?.address || "127.0.0.1";
}

/**
 * The address the operating system actually routes LAN traffic from.
 *
 * Connecting a UDP socket sends nothing; it only asks the routing table which
 * interface would be used. That is the address a peer can reach, whatever the
 * adapters are called.
 */
export async function routedLanAddress() {
  const dgram = await import("node:dgram");
  return new Promise((resolve) => {
    const socket = dgram.createSocket("udp4");
    const finish = (value) => {
      try {
        socket.close();
      } catch {
        // already closed
      }
      resolve(value);
    };
    socket.on("error", () => finish(primaryLanAddress()));
    try {
      socket.connect(9, "10.255.255.255", () => {
        try {
          const { address } = socket.address();
          finish(address && address !== "0.0.0.0" ? address : primaryLanAddress());
        } catch {
          finish(primaryLanAddress());
        }
      });
    } catch {
      finish(primaryLanAddress());
    }
  });
}

export function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd || REPO_ROOT,
      env: { ...process.env, ...(options.env || {}) },
      stdio: options.stdio || "inherit",
      shell: options.shell ?? false,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0 || options.allowFailure) resolve(code);
      else reject(new Error(`${command} exited with code ${code}`));
    });
  });
}

export function spawnBackground(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || REPO_ROOT,
    env: { ...process.env, ...(options.env || {}) },
    stdio: options.stdio || ["ignore", "pipe", "pipe"],
    shell: options.shell ?? false,
  });
  if (options.label && child.stdout) {
    pipeLines(child.stdout, options.label, options.color || "gray", options.filter);
  }
  if (options.label && child.stderr) {
    pipeLines(child.stderr, options.label, options.color || "gray", options.filter);
  }
  return child;
}

function pipeLines(stream, label, color, filter) {
  let buffer = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => {
    buffer += chunk;
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      if (filter && !filter(line)) continue;
      console.log(`${paint(color, label.padEnd(11))} ${line}`);
    }
  });
}

/**
 * Is something already listening here?
 *
 * Worth checking before spawning anything: a stale dev server holding port 3000
 * otherwise surfaces as a raw EADDRINUSE stack trace from deep inside Next,
 * which tells the user nothing about what to do next.
 */
export function portInUse(port, host = "127.0.0.1") {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    const done = (result) => {
      socket.destroy();
      resolve(result);
    };
    socket.setTimeout(700);
    socket.once("connect", () => done(true));
    socket.once("timeout", () => done(false));
    socket.once("error", () => done(false));
    socket.connect(port, host);
  });
}

export async function waitForHttp(url, { timeoutMs = 120000, intervalMs = 400 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2500) });
      if (response.ok) return await response.json().catch(() => ({}));
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

