/**
 * Environment setup, run automatically by `npm run dev` and by `npm run setup`.
 *
 * The work happens in engine/setup_env.py, which the one-line join flow also
 * uses, so a host and a contributor are set up by the same code: it repairs or
 * rebuilds an environment that came from another machine, picks the PyTorch
 * build from the GPU and driver, retries downloads, and proves the accelerator
 * runs a kernel before calling anything ready. This file finds a Python to run
 * it with, splits the fast control plane from the slow training plane so the
 * dashboard is usable in seconds, and fetches the base checkpoints.
 *
 *   npm run setup                       everything, verbose
 *   npm run setup -- --control-only     just the coordinator runtime
 *   npm run setup -- --force            reinstall even if nothing changed
 *   npm run setup -- --backend xpu      override the detected backend
 *   npm run setup -- --wheelhouse DIR   install from pre-downloaded wheels
 */

import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { rename, rm, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { spawn } from "node:child_process";

import {
  ENGINE_DIR,
  PATHS,
  REPO_ROOT,
  STATE_DIR,
  VENV_DIR,
  ensureStateDir,
  log,
  paint,
  requireSystemPython,
} from "./lib/env.mjs";

const SETUP_FILE = PATHS.setupState;
const MODELS_DIR = path.join(STATE_DIR, "models");

const BASE_MODELS = [
  {
    name: "yolov8n.pt",
    url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
  },
  {
    name: "yolo11n.pt",
    url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
  },
];

const EMPTY_STATE = { controlPlane: "pending", trainingPlane: "pending", models: [], backend: null, messages: [] };

export function readSetupState() {
  if (!existsSync(SETUP_FILE)) return { ...EMPTY_STATE };
  try {
    return { ...EMPTY_STATE, ...JSON.parse(readFileSync(SETUP_FILE, "utf8")) };
  } catch {
    return { ...EMPTY_STATE };
  }
}

export function writeSetupState(patch) {
  mkdirSync(path.dirname(SETUP_FILE), { recursive: true });
  const next = { ...readSetupState(), ...patch, updatedAt: Date.now() };
  writeFileSync(SETUP_FILE, JSON.stringify(next, null, 2));
  return next;
}

function cliOptions(argv = process.argv.slice(2)) {
  const value = (flag) => {
    const index = argv.indexOf(flag);
    return index !== -1 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : null;
  };
  return {
    controlOnly: argv.includes("--control-only"),
    force: argv.includes("--force"),
    backend: value("--backend") || process.env.GRADMESH_BACKEND || "auto",
    wheelhouse: value("--wheelhouse") || process.env.GRADMESH_WHEELHOUSE || null,
  };
}

/** Run engine/setup_env.py with the system Python. Resolves to its exit code. */
function setupEnv(args, { quiet = false } = {}) {
  const python = requireSystemPython();
  mkdirSync(PATHS.machineDir, { recursive: true });
  return new Promise((resolve, reject) => {
    const child = spawn(
      python.command,
      [...python.args, path.join(ENGINE_DIR, "setup_env.py"), ...args],
      {
        cwd: ENGINE_DIR,
        stdio: quiet ? ["ignore", "pipe", "pipe"] : "inherit",
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        windowsHide: true,
      }
    );
    if (quiet) {
      // Background installs keep a full log next to the setup state, so a
      // failure the dashboard reports can actually be read.
      const logFile = createWriteStream(PATHS.setupLog, { flags: "a" });
      logFile.write(`\n# ${new Date().toISOString()} setup_env.py ${args.join(" ")}\n`);
      child.stdout.pipe(logFile, { end: false });
      child.stderr.pipe(logFile, { end: false });
      child.on("close", () => logFile.end());
    }
    child.on("error", reject);
    child.on("close", (code) => resolve(code ?? 1));
  });
}

function sharedArgs(options) {
  const args = ["--venv", VENV_DIR, "--state", SETUP_FILE];
  if (options.backend && options.backend !== "auto") args.push("--backend", options.backend);
  if (options.wheelhouse) args.push("--wheelhouse", options.wheelhouse);
  if (options.force) args.push("--force");
  return args;
}

export async function setupControlPlane(options = cliOptions()) {
  ensureStateDir();
  if (PATHS.synced) {
    log("setup", `this folder syncs to other devices, so its Python environment lives at ${VENV_DIR}`, "gray");
  }
  const code = await setupEnv(["install", "--plane", "control", ...sharedArgs(options)]);
  if (code !== 0) {
    throw new Error("the coordinator runtime could not be installed; the output above has the cause");
  }
}

export async function setupTrainingPlane({ quiet = false, ...rest } = {}) {
  const options = { ...cliOptions(), ...rest };
  if (!quiet) log("setup", "installing the training plane, PyTorch is a large download the first time");
  const code = await setupEnv(
    ["install", "--plane", "training", "--host", ...sharedArgs(options), ...(quiet ? ["--quiet"] : [])],
    { quiet }
  );
  const state = readSetupState();
  if (code !== 0 && state.trainingPlane !== "failed") {
    writeSetupState({ trainingPlane: "failed", trainingError: `setup exited with code ${code}` });
  }
  if (!quiet) {
    if (code === 0 && state.accelerator === "ok") log("setup", "training plane ready", "green");
    else if (code === 0) {
      log("setup", `training plane ready, but: ${state.acceleratorProblem || "the GPU check failed"}`, "yellow");
    } else log("setup", `training plane install failed, see ${PATHS.setupLog}`, "red");
  }
  return code;
}

async function downloadModels() {
  mkdirSync(MODELS_DIR, { recursive: true });
  const present = [];

  for (const model of BASE_MODELS) {
    const target = path.join(MODELS_DIR, model.name);
    if (existsSync(target)) {
      const info = await stat(target);
      if (info.size > 1_000_000) {
        present.push(model.name);
        continue;
      }
      await rm(target, { force: true });
    }

    log("setup", `downloading ${model.name}`);
    const partial = `${target}.part`;
    try {
      const response = await fetch(model.url, { redirect: "follow", signal: AbortSignal.timeout(180000) });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      // Written beside the target and renamed, so an interrupted download can
      // never leave a truncated checkpoint that later looks present.
      await pipeline(Readable.fromWeb(response.body), createWriteStream(partial));
      await rename(partial, target);
      present.push(model.name);
    } catch (error) {
      // A checkpoint is only needed when a run starts, so a flaky network at
      // setup time must not stop the dashboard from coming up.
      log("setup", `could not fetch ${model.name}: ${error.message}`, "yellow");
      await rm(partial, { force: true });
    }
  }

  writeSetupState({ models: present });
  return present;
}

async function main() {
  const options = cliOptions();
  await setupControlPlane(options);
  await downloadModels();

  let code = 0;
  if (!options.controlOnly) code = await setupTrainingPlane({ ...options, quiet: false });

  const state = readSetupState();
  log("setup", code === 0 ? "done" : "finished with problems", code === 0 ? "green" : "yellow");
  console.log(
    `${paint("gray", "           ")} control plane: ${state.controlPlane}, training plane: ${state.trainingPlane}` +
      `${state.torch ? ` (torch ${state.torch})` : ""}, models: ${(state.models || []).join(", ") || "none"}`
  );
  console.log(`${paint("gray", "           ")} environment: ${VENV_DIR}`);
  process.exit(code);
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("bootstrap.mjs")) {
  main().catch((error) => {
    console.error(paint("red", `setup failed: ${error.message}`));
    process.exit(1);
  });
}

export { MODELS_DIR, REPO_ROOT, downloadModels };
