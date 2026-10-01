/**
 * The exact set of files a joining machine downloads to become a worker.
 *
 * An explicit allowlist rather than a directory read: the route that serves
 * these is reachable by anyone who can open the join page, and it must not
 * become a way to read the host's filesystem.
 *
 * Everything a worker needs on any backend is here: the agent, the per-vendor
 * training adapters, the hardware rules and installer shared with the host,
 * and every requirements file, because which one applies is decided on the
 * contributor's machine.
 */
export const AGENT_FILES = [
  "version.py",
  "hardware.py",
  "setup_env.py",
  "accelerator.py",
  "probe.py",
  "trainers.py",
  "federated_training.py",
  "ultralytics_xpu.py",
  "worker.py",
  "requirements-control.txt",
  "requirements-common.txt",
  "requirements-train-cu130.txt",
  "requirements-train-cu126.txt",
  "requirements-train-cu128.txt",
  "requirements-train-cpu.txt",
  "requirements-xpu.txt",
  "requirements-mps.txt",
] as const;

export const AGENT_FILE_SET = new Set<string>(AGENT_FILES);
