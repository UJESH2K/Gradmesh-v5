/**
 * Where machine-specific files live. Shared by the launcher scripts and the
 * Next.js server, so both always agree on which Python environment to run.
 *
 * The rule exists because of how GradMesh actually gets moved between devices:
 * the repository sits in a OneDrive (or Dropbox, or iCloud) folder and is simply
 * opened on the next laptop. Everything in it syncs, including a 5 GB virtual
 * environment whose interpreter path points at the first machine, a setup record
 * claiming PyTorch is installed, and a worker pid that belongs to a process on
 * a different computer. On the second machine all three are lies.
 *
 * So anything that is only true for one machine is kept outside a synced
 * folder, under the user's local application data, keyed by the checkout path.
 * A repository that is not in a synced folder keeps its environment in `.venv`
 * as usual. `GRADMESH_VENV` and `GRADMESH_HOME` override both.
 *
 * Pure functions of their arguments and the environment: nothing here is
 * resolved relative to this file, because Next bundles it somewhere else.
 */

import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const SYNCED_SEGMENT =
  /(^|[\\/])(onedrive( - [^\\/]+)?|dropbox( \([^\\/]+\))?|google ?drive|my drive|icloud ?drive|mobile documents|box sync|pcloud ?drive|nextcloud|seafile|syncthing)([\\/]|$)/i;

/** True when a path sits inside a folder a sync client mirrors to other devices. */
export function isSyncedPath(target) {
  return SYNCED_SEGMENT.test(String(target || ""));
}

/** Per-user, per-machine data directory. Never synced by any mainstream client. */
export function machineHome() {
  if (process.env.GRADMESH_HOME) return path.resolve(process.env.GRADMESH_HOME);
  if (process.platform === "win32") {
    return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "GradMesh");
  }
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support", "GradMesh");
  }
  return path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "gradmesh");
}

/** Short stable key for one checkout, so two clones on one machine do not collide. */
export function checkoutKey(repoRoot) {
  let resolved = repoRoot;
  try {
    resolved = realpathSync(repoRoot);
  } catch {
    // A path that does not exist yet still hashes consistently.
  }
  const normalised = process.platform === "win32" ? resolved.toLowerCase() : resolved;
  return createHash("sha1").update(normalised).digest("hex").slice(0, 8);
}

/**
 * Every machine-local path for one checkout.
 *
 * Kept short on purpose: PyTorch's wheel contains paths over 150 characters
 * deep, and Windows without long-path support stops at 260.
 */
export function machinePaths(repoRoot) {
  const synced = isSyncedPath(repoRoot);
  const machineDir = path.join(machineHome(), checkoutKey(repoRoot));
  const venv = process.env.GRADMESH_VENV
    ? path.resolve(process.env.GRADMESH_VENV)
    : synced
      ? path.join(machineDir, "venv")
      : path.join(repoRoot, ".venv");
  const venvPython =
    process.platform === "win32" ? path.join(venv, "Scripts", "python.exe") : path.join(venv, "bin", "python");

  return {
    synced,
    machineDir,
    venv,
    venvPython,
    setupState: path.join(machineDir, "setup.json"),
    setupLog: path.join(machineDir, "setup.log"),
    workerLog: path.join(machineDir, "worker.log"),
    workerPid: path.join(machineDir, "worker.pid"),
  };
}

/** Shared runtime state: accounts, token, datasets. Overridable for big datasets. */
export function stateDir(repoRoot) {
  return process.env.GRADMESH_STATE_DIR
    ? path.resolve(process.env.GRADMESH_STATE_DIR)
    : path.join(repoRoot, ".gradmesh");
}
