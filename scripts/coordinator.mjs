/**
 * `npm run coordinator` - run only the Python control plane.
 *
 * Useful when the dashboard is served from somewhere else, or when debugging
 * the scheduler against the FastAPI docs at /docs.
 */

import { setupControlPlane, downloadModels } from "./bootstrap.mjs";
import { COORDINATOR_PORT, ENGINE_DIR, STATE_DIR, WEB_PORT, log, paint, run, venvPython } from "./lib/env.mjs";

async function main() {
  await setupControlPlane();
  await downloadModels();
  log("coordinator", `listening on 0.0.0.0:${COORDINATOR_PORT}`);
  await run(
    venvPython(),
    [
      "-m",
      "uvicorn",
      "coordinator.app:app",
      "--host",
      "0.0.0.0",
      "--port",
      String(COORDINATOR_PORT),
      "--timeout-keep-alive",
      "75",
    ],
    {
      cwd: ENGINE_DIR,
      env: {
        PYTHONUNBUFFERED: "1",
        GRADMESH_STATE_DIR: STATE_DIR,
        GRADMESH_WEB_PORT: String(WEB_PORT),
        GRADMESH_COORDINATOR_PORT: String(COORDINATOR_PORT),
      },
      allowFailure: true,
    }
  );
}

main().catch((error) => {
  console.error(paint("red", error.message));
  process.exit(1);
});
