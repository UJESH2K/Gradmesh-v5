import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { randomBytes } from "node:crypto";
import { networkInterfaces } from "node:os";
import path from "node:path";

import { machinePaths, stateDir } from "../scripts/lib/paths.mjs";

export const REPO_ROOT = process.cwd();
/** Shared mesh state: accounts, token, datasets. Same rule as the launcher. */
export const STATE_DIR = stateDir(REPO_ROOT);
/**
 * Machine-specific files: the Python environment, the setup record, the local
 * worker's log and pid. Outside the checkout when it is in a synced folder, so
 * a repository opened from OneDrive on a second laptop does not inherit the
 * first laptop's environment or think its worker is running.
 */
export const MACHINE = machinePaths(REPO_ROOT);
export const COORDINATOR_URL = (
  process.env.GRADMESH_COORDINATOR_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");
export const WEB_PORT = Number(process.env.PORT || 3000);
export const COORDINATOR_PORT = Number(process.env.GRADMESH_COORDINATOR_PORT || 8000);

export type CoordinatorState = {
  mesh_token?: string;
  mesh_name?: string;
  mesh_id?: string;
  datasets?: Record<string, unknown>;
  default_dataset_id?: string | null;
};

export function ensureStateDir(): string {
  if (!existsSync(STATE_DIR)) mkdirSync(STATE_DIR, { recursive: true });
  return STATE_DIR;
}

/**
 * The mesh token is minted by the Python coordinator, so the dashboard reads it
 * from the same file rather than keeping a second copy that could drift.
 */
export function readCoordinatorState(): CoordinatorState | null {
  const file = path.join(STATE_DIR, "coordinator.json");
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, "utf8")) as CoordinatorState;
  } catch {
    return null;
  }
}

export function meshToken(): string {
  return readCoordinatorState()?.mesh_token || "";
}

export function meshName(): string {
  return readCoordinatorState()?.mesh_name || "GradMesh";
}

/** Session signing key. Persisted so logins survive a dev-server restart. */
export function sessionSecret(): string {
  if (process.env.GRADMESH_SESSION_SECRET) return process.env.GRADMESH_SESSION_SECRET;
  ensureStateDir();
  const file = path.join(STATE_DIR, "session.key");
  if (existsSync(file)) return readFileSync(file, "utf8").trim();
  const generated = randomBytes(32).toString("hex");
  writeFileSync(file, generated, { mode: 0o600 });
  return generated;
}

export type LanAddress = { name: string; address: string; virtual: boolean };

// Adapters no other device on the Wi-Fi can reach. See scripts/lib/env.mjs.
const VIRTUAL_ADAPTER =
  /vethernet|virtualbox|vbox|vmware|vmnet|hyper-v|wsl|docker|br-|veth|tailscale|zerotier|wireguard|utun|tun\d|tap\d|vpn|npcap|loopback|bluetooth/i;

export function lanAddresses(): LanAddress[] {
  const found: LanAddress[] = [];
  for (const [name, entries] of Object.entries(networkInterfaces())) {
    for (const entry of entries || []) {
      if (entry.family !== "IPv4" || entry.internal) continue;
      if (entry.address.startsWith("169.254.")) continue;
      found.push({ name, address: entry.address, virtual: VIRTUAL_ADAPTER.test(name) });
    }
  }
  const range = (address: string) =>
    address.startsWith("192.168.") ? 0 : address.startsWith("10.") ? 1 : address.startsWith("172.") ? 2 : 3;
  const wireless = (name: string) => (/wi-?fi|wlan|wireless|en0|eth|ethernet/i.test(name) ? 0 : 1);
  return found.sort(
    (a, b) =>
      Number(a.virtual) - Number(b.virtual) ||
      wireless(a.name) - wireless(b.name) ||
      range(a.address) - range(b.address)
  );
}

export function primaryLanAddress(): string {
  return lanAddresses()[0]?.address || "127.0.0.1";
}

/**
 * The origin a peer on the same network should use. A request's Host header is
 * the most reliable source, because it is literally the address that worked for
 * whoever is looking at the page.
 */
export function meshOrigin(requestHost?: string | null): string {
  if (requestHost && !requestHost.startsWith("localhost") && !requestHost.startsWith("127.")) {
    return `http://${requestHost}`;
  }
  return `http://${primaryLanAddress()}:${WEB_PORT}`;
}

export type SetupState = {
  controlPlane: string;
  trainingPlane: string;
  models: string[];
  backend: string | null;
  profile?: string;
  profileLabel?: string;
  profileReason?: string;
  gpu?: string | null;
  torch?: string;
  ultralytics?: string;
  accelerator?: "ok" | "unavailable" | null;
  acceleratorProblem?: string | null;
  blocked?: string | null;
  fix?: string | null;
  warnings?: string[];
  reference?: boolean;
  venv?: string;
  venvPython?: string;
  messages: { at: number; message: string }[];
};

export function setupState(): SetupState {
  const fallback: SetupState = {
    controlPlane: "pending",
    trainingPlane: "pending",
    models: [],
    backend: null,
    messages: [],
  };
  if (!existsSync(MACHINE.setupState)) return fallback;
  try {
    return { ...fallback, ...JSON.parse(readFileSync(MACHINE.setupState, "utf8")) };
  } catch {
    return fallback;
  }
}

/** The Python that runs the engine on this machine. */
export function venvPython(): string {
  return MACHINE.venvPython;
}
