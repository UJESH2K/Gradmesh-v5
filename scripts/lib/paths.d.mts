export type MachinePaths = {
  synced: boolean;
  machineDir: string;
  venv: string;
  venvPython: string;
  setupState: string;
  setupLog: string;
  workerLog: string;
  workerPid: string;
};

export function isSyncedPath(target: string): boolean;
export function machineHome(): string;
export function checkoutKey(repoRoot: string): string;
export function machinePaths(repoRoot: string): MachinePaths;
export function stateDir(repoRoot: string): string;
