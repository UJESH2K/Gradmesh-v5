/**
 * `npm test` — every check that runs without a GPU or a network.
 *
 *   1. Machine-local path rules (Node, below).
 *   2. Engine tests: hardware and build selection for every vendor, benchmark
 *      design, portable state, file-list sharding (engine/tests/run_tests.py).
 *   3. The scheduler's claims (scripts/test-scheduler.mjs).
 *   4. With --full, a TypeScript check of the dashboard as well.
 *
 * Needs only Node and a system Python, so it runs on a fresh clone before
 * `npm run setup`, and in CI.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";

import { ENGINE_DIR, REPO_ROOT, findSystemPython, paint, venvPython, venvReady } from "./lib/env.mjs";
import { isSyncedPath, machinePaths } from "./lib/paths.mjs";

let failed = 0;

function section(title) {
  console.log(paint("bold", `\n${title}\n`));
}

function check(name, fn) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (error) {
    failed += 1;
    console.log(`  FAIL ${name}: ${error.message}`);
  }
}

section("Machine-local paths");
check("OneDrive folders are recognised as synced", () => {
  assert.equal(isSyncedPath("C:\\Users\\a\\OneDrive\\Documents\\GitHub\\repo"), true);
  assert.equal(isSyncedPath("C:\\Users\\a\\OneDrive - Contoso\\repo"), true);
});
check("Dropbox and iCloud are recognised as synced", () => {
  assert.equal(isSyncedPath("/Users/a/Dropbox/repo"), true);
  assert.equal(isSyncedPath("/Users/a/Library/Mobile Documents/com~apple~CloudDocs/repo"), true);
});
check("an ordinary folder is not synced", () => {
  assert.equal(isSyncedPath("C:\\dev\\gradmesh"), false);
  assert.equal(isSyncedPath("/home/a/src/gradmesh"), false);
  assert.equal(isSyncedPath("/home/a/onedrivers-club/gradmesh"), false);
});
check("a synced checkout keeps its environment outside the checkout", () => {
  const synced = machinePaths("C:\\Users\\a\\OneDrive\\repo");
  assert.equal(synced.synced, true);
  assert.ok(!synced.venv.toLowerCase().includes("onedrive"), synced.venv);
});
check("an ordinary checkout keeps .venv", () => {
  const local = machinePaths(path.join(path.sep, "srv", "gradmesh"));
  assert.ok(local.venv.endsWith(".venv"), local.venv);
});
check("two checkouts on one machine do not share state", () => {
  assert.notEqual(machinePaths("/srv/a").machineDir, machinePaths("/srv/b").machineDir);
});

const system = findSystemPython();
const python = system && !system.missing ? system : venvReady() ? { command: venvPython(), args: [] } : null;
if (!python) {
  console.error(paint("red", "\nNo Python 3.10-3.13 found; the engine tests need one."));
  process.exit(1);
}

section("Engine");
const engine = spawnSync(python.command, [...python.args, path.join(ENGINE_DIR, "tests", "run_tests.py")], {
  cwd: ENGINE_DIR,
  stdio: "inherit",
});
if (engine.status !== 0) failed += 1;

const scheduler = spawnSync(process.execPath, [path.join(REPO_ROOT, "scripts", "test-scheduler.mjs")], {
  cwd: REPO_ROOT,
  stdio: "inherit",
});
if (scheduler.status !== 0) failed += 1;

if (process.argv.includes("--full")) {
  section("TypeScript");
  const tsc = spawnSync(process.execPath, [path.join(REPO_ROOT, "node_modules", "typescript", "bin", "tsc"), "--noEmit", "-p", "."], {
    cwd: REPO_ROOT,
    stdio: "inherit",
  });
  if (tsc.status !== 0) failed += 1;
  else console.log("  ok   the dashboard typechecks");
}

console.log("");
if (failed) {
  console.log(paint("red", `${failed} suite${failed === 1 ? "" : "s"} failed`));
  process.exit(1);
}
console.log(paint("green", "All tests passed."));
