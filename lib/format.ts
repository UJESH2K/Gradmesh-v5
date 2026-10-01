export function bytes(value: number): string {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  const scaled = value / 1024 ** index;
  return `${scaled >= 100 || index === 0 ? Math.round(scaled) : scaled.toFixed(1)} ${units[index]}`;
}

export function seconds(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  if (value < 60) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function compact(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(Math.round(value));
}

export function gflops(value: number | undefined | null): string {
  if (!value) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(1)} TFLOP/s`;
  return `${Math.round(value)} GFLOP/s`;
}

export function memory(mb: number | undefined | null): string {
  if (!mb) return "—";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

export function percent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function ago(timestampSeconds: number): string {
  const delta = Date.now() / 1000 - timestampSeconds;
  if (delta < 5) return "just now";
  if (delta < 60) return `${Math.round(delta)}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return `${Math.round(delta / 86400)}d ago`;
}

export function clock(timestampSeconds: number | null | undefined): string {
  if (!timestampSeconds) return "—";
  return new Date(timestampSeconds * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export type VendorInfo = {
  backend: "cuda" | "xpu" | "mps" | "cpu";
  vendor: "nvidia" | "intel" | "apple" | "cpu";
  /** The company, as a person would say it. */
  name: string;
  /** The PyTorch backend that drives it. */
  api: string;
  /** What to install so a machine of this kind can join. */
  needs: string;
};

/** The three GPU families GradMesh 5 trains on, in the order they are listed everywhere. */
export const VENDORS: VendorInfo[] = [
  {
    backend: "cuda",
    vendor: "nvidia",
    name: "NVIDIA",
    api: "CUDA",
    needs: "GeForce GTX 900 or newer on Windows or Linux, with the NVIDIA driver installed.",
  },
  {
    backend: "xpu",
    vendor: "intel",
    name: "Intel",
    api: "XPU",
    needs: "Arc A or B series, or a Core Ultra with Arc Graphics, on Windows or Linux with Intel's graphics driver.",
  },
  {
    backend: "mps",
    vendor: "apple",
    name: "Apple",
    api: "Metal",
    needs: "Any Apple Silicon Mac (M1 or later) on macOS 14 Sonoma or newer.",
  },
];

const CPU_INFO: VendorInfo = {
  backend: "cpu",
  vendor: "cpu",
  name: "CPU",
  api: "CPU",
  needs: "Joins and is measured, but receives no training work.",
};

export function vendorInfo(backend: string | undefined | null): VendorInfo {
  return VENDORS.find((item) => item.backend === backend) ?? CPU_INFO;
}

export function backendLabel(backend: string | undefined): string {
  if (!backend) return "Unknown";
  const info = vendorInfo(backend);
  return info.backend === "cpu" && backend !== "cpu" ? backend.toUpperCase() : `${info.name} ${info.api}`;
}

/** "NVIDIA+Intel+Apple" for a list of backends. */
export function mixLabel(backends: string[] | null | undefined): string {
  if (!backends || backends.length === 0) return "Any GPU";
  const order = ["cuda", "xpu", "mps", "cpu"];
  return [...new Set(backends)]
    .sort((a, b) => order.indexOf(a) - order.indexOf(b))
    .map((backend) => vendorInfo(backend).name)
    .join(" + ");
}

/**
 * Wall clock for a run: time since it started while it is live, the sum of its
 * rounds once it has finished. Summing rounds alone reads "0 ms" for the whole
 * first round of a live run.
 */
export function runElapsed(run: {
  status: string;
  wall_clock_seconds: number;
  started_at?: number;
  created_at: number;
}): number {
  const live = ["running", "planning", "waiting"].includes(run.status);
  if (!live) return run.wall_clock_seconds;
  return Math.max(run.wall_clock_seconds, Date.now() / 1000 - (run.started_at ?? run.created_at));
}
