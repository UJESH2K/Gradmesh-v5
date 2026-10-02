export type Tier = "full" | "probation" | "rejected";

/** cuda is NVIDIA, xpu is Intel, mps is Apple Silicon. */
export type Backend = "cuda" | "xpu" | "mps" | "cpu";
export type Vendor = "nvidia" | "intel" | "apple" | "cpu";

export type Capability = {
  backend: string;
  vendor?: Vendor;
  device_name: string;
  total_memory_mb: number;
  unified_memory?: boolean;
  gflops: number;
  mem_bandwidth_gbps: number;
  host: string;
  platform: string;
  torch_version: string;
  supports_training: boolean;
  supports_amp?: boolean;
  compute_capability?: string | null;
  // v5.1: the software each machine trains with.
  python_version?: string;
  torchvision_version?: string;
  ultralytics_version?: string;
  /** "CUDA 13.0", "Intel XPU", "Metal". */
  runtime?: string;
  /** "macOS 14.5", "Windows 11 (build 26200)". */
  os_version?: string;
  gpu_cores?: number | null;
};

/** A machine's training stack as the coordinator sees it, against the pinned reference. */
export type NodeSoftware = {
  python: string;
  torch: string;
  /** The wheel flavour: cu130, cu126, xpu, macOS, cpu. */
  torch_build: string;
  torchvision: string;
  ultralytics: string;
  runtime: string;
  os: string;
  driver: string;
  agent: string;
  agent_behind: boolean;
  /** null when an older agent did not report enough to tell. */
  on_reference: boolean | null;
  drift: string[];
};

/** What the worker reports about its host, refreshed every half minute. */
export type HostDiagnostics = {
  cpu?: string;
  cpu_logical?: number;
  cpu_physical?: number;
  os?: string;
  python?: string;
  ram_mb?: number;
  ram_available_mb?: number;
  cpu_load?: number;
  on_battery?: boolean;
  battery_percent?: number;
  power_plan?: string;
  swap_used_mb?: number;
  disk_free_mb?: number;
  /** macOS: nominal, fair, serious, critical. */
  thermal_state?: string;
  low_power_mode?: boolean;
  gpu_utilization?: number | null;
  gpu_temperature_c?: number | null;
  sm_clock_mhz?: number | null;
  sm_clock_max_mhz?: number | null;
  power_draw_w?: number | null;
  power_limit_w?: number | null;
  pcie_gen?: number | null;
  pcie_gen_max?: number | null;
  pcie_width?: number | null;
  pcie_width_max?: number | null;
  throttle_reasons?: string[];
  other_gpu_processes?: number;
  driver?: string | null;
};

/** The learned cost model for one checkpoint at one image size. */
export type Workload = { rate: number; fixed: number; rounds: number; updated_at?: number; fixed_cold?: boolean };

export type MeshNode = {
  node_id: string;
  display_name: string;
  gpu: string;
  backend: string;
  gpu_memory_mb: number;
  max_batch_size: number;
  supports_training: boolean;
  capability: Partial<Capability>;
  owner: string | null;
  agent_version: string;
  joined_at: number;
  last_seen: number;
  active: boolean;
  load: number;
  latency_ms: number;
  allocated_memory_mb: number;
  active_batches: number;
  completed_rounds: number;
  failed_rounds: number;
  samples_trained: number;
  seconds_trained: number;
  reliability: number;
  throughput_sps: number;
  training_epoch: number;
  training_total_epochs: number;
  fitness: number;
  tier?: Tier;
  admission_reason?: string;
  // v5
  vendor?: Vendor;
  instance_id?: string | null;
  protocol?: number;
  liveness?: "online" | "suspect" | "offline";
  phase?: "idle" | "downloading" | "loading" | "training" | "uploading" | null;
  progress?: { batch: number; batches: number } | null;
  diagnostics?: HostDiagnostics;
  warnings?: string[];
  co_located?: boolean;
  dataloader_workers?: number | null;
  fixed_seconds?: number;
  workloads?: Record<string, Workload>;
  workload?: string;
  batch_cap?: number | null;
  address?: string | null;
  software?: NodeSoftware;
};

export type ShardAssignment = {
  node_id: string;
  shard_index: number;
  samples: number;
  fitness: number;
  tier: Tier;
  throughput_sps: number;
  predicted_seconds: number;
  /** The fixed per-round part of predicted_seconds. */
  fixed_seconds?: number;
  backend?: string;
};

export type RoundPlan = {
  total_samples: number;
  assignments: ShardAssignment[];
  rejected: { node_id: string; reason: string; fitness: number }[];
  predicted_makespan_seconds: number;
  predicted_serial_seconds: number;
  predicted_speedup: number;
  predicted_imbalance?: number;
  strategy?: PartitionStrategy;
};

export type BackendShare = {
  workers: number;
  samples: number;
  seconds: number;
  compute_seconds: number;
  weight: number;
};

export type RoundRecord = {
  round: number;
  workers: number;
  samples: number;
  dropped_shards: number;
  makespan_seconds: number;
  fastest_seconds: number;
  straggler_gap_seconds: number;
  aggregation_seconds: number;
  wall_clock_seconds: number;
  serial_estimate_seconds: number;
  speedup: number;
  efficiency: number;
  imbalance?: number;
  predicted_imbalance?: number;
  comm_bytes?: number;
  comm_seconds?: number;
  mean_overhead_seconds?: number | null;
  speculated_shards?: number;
  by_backend?: Record<string, BackendShare>;
  accuracy?: { ok: boolean; map50?: number; map50_95?: number; error?: string };
  shards: {
    node_id: string;
    node_name: string | null;
    backend?: string | null;
    samples: number;
    seconds: number;
    compute_seconds?: number;
    overhead_seconds?: number;
    predicted_seconds: number;
    predicted_fixed_seconds?: number | null;
    tier: Tier;
    weight: number;
    metrics: Record<string, unknown> | null;
  }[];
};

export type RunStatus = "planning" | "waiting" | "running" | "done" | "failed" | "stopped";

export type RunSummary = {
  id: string;
  name: string;
  status: RunStatus;
  mode: "mesh" | "solo";
  created_at: number;
  finished_at: number | null;
  dataset_id: string;
  dataset_name: string;
  base_model: string;
  rounds: number;
  current_round: number;
  imgsz: number;
  batch_size: number;
  total_samples: number;
  wall_clock_seconds: number;
  serial_estimate_seconds: number;
  speedup: number;
  efficiency: number;
  peak_workers: number;
  has_artifact: boolean;
  // v5
  started_at?: number;
  /** Machines holding a shard of the current round right now. */
  live_workers?: number;
  partition_strategy?: PartitionStrategy;
  backends?: Backend[] | null;
  warmup_mode?: WarmupMode;
  warmup_epochs?: number | null;
  optimizer?: string;
  lr0?: number | null;
  worker_validation?: boolean;
  evaluate?: boolean;
  seed?: number;
  map50?: number | null;
  best_map50?: number | null;
  accuracy_history?: { round: number; map50: number | null; map50_95: number | null; train_seconds: number }[];
  comm_bytes_total?: number;
  mean_imbalance?: number | null;
  reference_stack?: Record<string, string> | null;
  backend_totals?: Record<
    string,
    { samples: number; seconds: number; compute_seconds: number; rounds: number; throughput_sps: number | null }
  >;
  error?: string | null;
};

export type WarmupMode = "first-round" | "none" | "every-round";

export type LiveShard = {
  batch_id: string;
  node_id: string;
  node_name: string | null;
  status: "queued" | "assigned" | "done" | "failed" | "dropped" | "superseded";
  backend?: string | null;
  phase?: string | null;
  progress?: { batch: number; batches: number } | null;
  speculative?: boolean;
  predicted_fixed_seconds?: number | null;
  samples: number;
  tier: Tier;
  round: number;
  predicted_seconds: number;
  batch_size: number;
  elapsed_seconds: number | null;
  soft_deadline_seconds: number;
  hard_deadline_seconds: number;
  error: string | null;
};

export type RunDetail = RunSummary & {
  round_history: RoundRecord[];
  plan: RoundPlan | null;
  error: string | null;
  notes: string | null;
  class_names: string[];
  live_shards: LiveShard[];
};

export type Dataset = {
  id: string;
  name: string;
  filename: string;
  created_at: number;
  bytes: number;
  train_count: number;
  val_count: number;
  class_names: string[];
  is_default?: boolean;
  available?: boolean;
  source?: string;
  is_subset?: boolean;
  parent_id?: string;
};

export type MeshPolicy = Record<string, number>;

export type BackendSummary = {
  backend: Backend;
  vendor: Vendor;
  nodes: number;
  online: number;
  eligible: number;
  gflops: number;
  memory_mb: number;
  throughput_sps: number;
};

export type MeshState = {
  mesh_name: string;
  mesh_id?: string;
  version?: string;
  protocol?: number;
  reference_stack?: Record<string, string>;
  workload?: string;
  backends?: BackendSummary[];
  nodes: MeshNode[];
  metrics: {
    nodes_total: number;
    nodes_active: number;
    nodes_training: number;
    nodes_admitted: number;
    total_gflops: number;
    total_memory_mb: number;
    active_shards: number;
    stream_subscribers: number;
    nodes_warned?: number;
  };
  active_runs: RunSummary[];
  plan_preview: RoundPlan;
  policy: MeshPolicy;
  torch_ready: boolean;
  default_dataset: Dataset | null;
};

export type MeshEvent = {
  id: number;
  kind: string;
  at: number;
  data: Record<string, unknown>;
};

export type NetworkDevice = {
  ip: string;
  mac: string | null;
  hostname: string | null;
  open_ports: number[];
  is_coordinator: boolean;
  is_this_host: boolean;
  is_member: boolean;
  is_visitor: boolean;
  source: "arp" | "scan";
  /** Measured TCP round trip. Null when the device answers on no port. */
  rtt_ms: number | null;
  proximity: "close" | "nearby" | "far" | "unknown";
};

export type Visitor = {
  visitor_id: string;
  name: string;
  platform: string | null;
  user_agent: string | null;
  cores: number | null;
  memory_gb: number | null;
  gpu: string | null;
  webgpu: boolean;
  screen: string | null;
  ip: string | null;
  first_seen: number;
  last_seen: number;
  has_agent: boolean;
};

export type DiscoverState = {
  mdns: {
    active: boolean;
    hostname: string;
    address: string | null;
    web_port: number;
    api_port: number;
    error: string | null;
  };
  addresses: { lan_ip: string; web_port: number; api_port: number };
  scan: {
    subnet: string | null;
    at: number;
    running: boolean;
    duration_seconds: number;
    scanned: number;
  };
  devices: NetworkDevice[];
  visitors: Visitor[];
  nodes: {
    node_id: string;
    display_name: string | null;
    gpu: string | null;
    backend: string | null;
    active: boolean;
    host: string | null;
  }[];
  idle_count: number;
};

// ---------------------------------------------------------------------------
// Benchmark suites
// ---------------------------------------------------------------------------

export type PartitionStrategy = "proportional" | "proportional-linear" | "equal";

export type SuiteConfig = {
  name: string;
  base_model: string;
  parent_dataset_id: string | null;
  dataset_sizes: number[];
  node_counts: number[];
  strategies: PartitionStrategy[];
  repeats: number;
  rounds: number;
  imgsz: number;
  batch_size: number;
  node_selection: "strongest" | "random";
  evaluate: boolean;
  network_label: string;
  notes: string;
  trial_timeout_seconds: number;
  settle_seconds: number;
  /** Legs of one campaign share this and differ only in network_label. */
  campaign_id: string | null;
  leg: number;
  baseline_repeats?: number;
  warmup_mode?: WarmupMode;
  warmup_epochs?: number | null;
  optimizer?: string;
  worker_validation?: boolean;
  /** Hardware combinations to compare, each a list of backends. */
  vendor_mixes?: Backend[][];
};

export type TrialSpec = {
  trial_id: string;
  index: number;
  node_count: number;
  sample_count: number;
  strategy: PartitionStrategy;
  repeat: number;
  node_ids: string[];
  dataset_id: string | null;
  label: string;
  backends?: Backend[] | null;
  mix?: string;
};

export type TrialResult = {
  trial_id: string;
  index: number;
  label: string;
  status: "done" | "failed" | "skipped" | "aborted";
  error: string | null;
  run_id: string | null;
  mix?: string;
  node_count: number;
  sample_count: number;
  strategy: PartitionStrategy;
  repeat: number;
  rounds_completed: number;
  train_seconds: number;
  serial_estimate_seconds: number;
  eval_seconds: number;
  /** Against the one-machine baseline at the same dataset size. */
  speedup: number | null;
  efficiency: number | null;
  /** Shard-time overlap inside a round. Diagnostic, not the headline. */
  parallel_speedup?: number | null;
  baseline_train_seconds?: number | null;
  map50: number | null;
  map50_95: number | null;
  mean_imbalance: number | null;
  mean_straggler_gap_seconds: number | null;
  comm_bytes: number;
  comm_seconds: number;
  comm_fraction: number | null;
};

export type SuiteCell = {
  mix?: string;
  node_count: number;
  sample_count: number;
  strategy: PartitionStrategy;
  runs: number;
  train_seconds_mean: number | null;
  train_seconds_std: number | null;
  speedup_mean: number | null;
  speedup_std: number | null;
  parallel_speedup_mean?: number | null;
  efficiency_mean: number | null;
  efficiency_std: number | null;
  map50_mean: number | null;
  map50_std: number | null;
  mean_imbalance_mean: number | null;
  baseline_map50: number | null;
  delta_map50: number | null;
};

export type Suite = {
  id: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  status: "pending" | "running" | "done" | "aborted" | "failed" | "interrupted";
  config: SuiteConfig;
  trials: TrialSpec[];
  results: TrialResult[];
  current_trial: TrialSpec | null;
  cells: SuiteCell[];
  estimated_seconds: number;
  is_active?: boolean;
  error?: string | null;
  environment: {
    coordinator: Record<string, unknown>;
    dataset: { id: string; name: string; train_count: number; val_count: number; class_names: string[] };
    nodes: {
      node_id: string;
      name: string | null;
      gpu: string | null;
      backend: string | null;
      memory_mb: number | null;
      capability: Record<string, unknown> | null;
    }[];
  };
};

export type SuiteIndexEntry = {
  id: string;
  name: string;
  status: Suite["status"];
  created_at: number;
  finished_at: number | null;
  total_trials: number;
  completed_trials: number;
  failed_trials: number;
  network_label: string;
  campaign_id: string | null;
  leg: number;
};

export type BenchmarkIndex = {
  suites: SuiteIndexEntry[];
  campaigns: Campaign[];
  active_suite_id: string | null;
  available_nodes: number;
  nodes: {
    node_id: string;
    name: string | null;
    gpu: string | null;
    backend: string | null;
    memory_mb: number | null;
    gflops: number | null;
    throughput_sps: number | null;
  }[];
  torch_ready: boolean;
};

export type SuitePreview = {
  trials: number;
  estimated_seconds: number;
  available_nodes: number;
  /** Requested machine counts above what is online, so excluded from the design. */
  dropped_counts: number[];
  planned_counts: number[];
  breakdown: TrialSpec[];
};

export type NetworkComparisonRow = {
  suite_id: string;
  leg: number;
  network_label: string;
  trials: number;
  mean_latency_ms: number | null;
  train_seconds: number | null;
  speedup: number | null;
  efficiency: number | null;
  map50: number | null;
  comm_fraction: number | null;
  comm_bytes: number | null;
  imbalance: number | null;
};

export type Campaign = {
  campaign_id: string;
  name: string;
  legs: (SuiteIndexEntry & { leg: number })[];
  networks: string[];
  created_at: number;
  complete_legs: number;
  comparison?: NetworkComparisonRow[];
};

export type StandardDatasetEntry = {
  key: string;
  name: string;
  images: number;
  classes: number;
  download_mb: number;
  blurb: string;
  good_for: string;
  already_downloaded: boolean;
};

export type StandardCatalogue = {
  catalogue: StandardDatasetEntry[];
  import: { running: boolean; key: string | null; message: string | null; error: string | null };
};
