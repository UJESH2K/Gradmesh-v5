/**
 * `npm run worker` — contribute this machine's GPU to a mesh.
 *
 * With no arguments it joins the coordinator running on this same machine,
 * reading the token straight off disk. Point it at another host with
 *
 *   npm run worker -- --server http://192.168.1.20:8000 --token <token>
 *
 * Every other flag is passed to the agent unchanged, for example
 * `--gpu-index 1` on a machine with two GPUs, `--backend xpu` to contribute an
 * Intel GPU on a machine that also has an NVIDIA one, or `--name lab-pc-3`.
 * `python engine/worker.py --help` lists them all.
 *
 * A contributor who has not cloned the repository uses the one-line join
 * command from the host's Invite a GPU page instead; it runs the same agent.
 */

import {
  COORDINATOR_PORT,
  ENGINE_DIR,
  log,
  paint,
  readCoordinatorState,
  run,
  venvPython,
} from "./lib/env.mjs";
import { readSetupState, setupControlPlane, setupTrainingPlane } from "./bootstrap.mjs";

const OWN_FLAGS = new Set(["--server", "--token"]);

function argValue(flag, fallback) {
  const index = process.argv.indexOf(flag);
  return index !== -1 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

/** Everything after `--` that is not ours, passed through to worker.py. */
function passthrough() {
  const args = process.argv.slice(2);
  const forwarded = [];
  for (let index = 0; index < args.length; index += 1) {
    if (OWN_FLAGS.has(args[index])) {
      index += 1;
      continue;
    }
    forwarded.push(args[index]);
  }
  return forwarded;
}

async function main() {
  await setupControlPlane();

  const backend = argValue("--backend", "auto");
  // A worker installs the build its own GPU needs; unlike a host it gains
  // nothing from a CPU fallback, so a blocked GPU stops here with the fix.
  await setupTrainingPlane({ quiet: false, host: false, backend });
  const setup = readSetupState();
  if (setup.trainingPlane === "blocked") {
    console.error(paint("red", `\n${setup.blocked}\n`));
    if (setup.fix) console.error(paint("yellow", `${setup.fix}\n`));
    process.exit(2);
  }
  if (setup.trainingPlane !== "ready") {
    console.error(paint("red", "\nThe training runtime is not installed. Run `npm run setup` to see why.\n"));
    process.exit(1);
  }
  if (setup.accelerator === "unavailable") {
    log("worker", `this machine's GPU cannot train yet: ${setup.acceleratorProblem}`, "yellow");
    if (setup.fix) log("worker", setup.fix, "yellow");
  }

  const server = argValue("--server", `http://127.0.0.1:${COORDINATOR_PORT}`);
  const token = argValue("--token", readCoordinatorState()?.mesh_token || "");

  if (!token) {
    console.error(
      paint("red", "\nNo mesh token. Start the host with `npm run dev`, or pass --token from its Invite a GPU page.\n")
    );
    process.exit(1);
  }

  const forwarded = passthrough();
  const args = ["worker.py", "--server-url", server, "--token", token, "--discover", ...forwarded];
  if (!forwarded.includes("--backend") && setup.backend && setup.backend !== "cpu") {
    args.push("--backend", setup.backend);
  }

  log("worker", `joining ${server}`);
  const code = await run(venvPython(), args, { cwd: ENGINE_DIR, env: { PYTHONUNBUFFERED: "1" }, allowFailure: true });
  process.exit(code ?? 0);
}

main().catch((error) => {
  console.error(paint("red", error.message));
  process.exit(1);
});
