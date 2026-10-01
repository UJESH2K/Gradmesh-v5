# GradMesh 5

**NVIDIA, Intel and Apple GPUs on one network, one training cluster.**

GradMesh turns the idle consumer GPUs already sitting on a local network into a
single coordinated training cluster. One machine hosts. Everyone else pastes one
line. The mesh measures what each device can actually do, sizes the work to
match, and trains a shared YOLO model across all of them.

Version 5 makes three GPU families first-class in one mesh: **NVIDIA through
CUDA, Intel Arc through XPU, and Apple Silicon through Metal**. They run one
pinned PyTorch stack, their weights average into one model, and every result
is broken down by vendor. It is the baseline for the cross-vendor paper.

```
npm install
npm run dev
```

That is the whole host setup. Every requirement, and a fully manual install
for each operating system and vendor, is in **[SETUP.md](SETUP.md)**. What
changed and why is in **[CHANGELOG.md](CHANGELOG.md)**.

---

## What v5 changes

| | v4 | v5 |
|---|---|---|
| GPU families | NVIDIA, Intel (XPU path bolted on) | NVIDIA, Intel and Apple behind one accelerator interface |
| Software stack | torch 2.5 to 2.13 depending on the machine | **one reference stack** on every vendor: torch 2.13.0, torchvision 0.28.0, ultralytics 8.4.46 |
| Build selection | three copies of the rules (Node, PowerShell, sh), chosen by compute capability | one Python module, chosen by compute capability **and driver**, unit tested on fabricated machines |
| Proof the GPU works | `is_available()` | a real kernel launch at install and at worker start |
| Opening the repo on another device | synced `.venv` and setup record broke it | machine-specific files kept outside synced folders; environments from another machine rebuilt; dataset paths portable |
| Shard sizing | proportional to rate | **affine**: fixed overhead per round plus a per-image rate, learned per workload, persisted across restarts |
| Per-round overhead | ~19 s (two validation passes, AMP check download, shard zip) | under 1 s in testing |
| Shard transfer | a fresh zip every round | file lists; workers cache images and fetch only what is new |
| Weight transfer | base64 inside JSON | raw bytes |
| Training correctness | small shards never stepped the optimiser; every round rounded weights to fp16; warmup every round | gradients flushed, fp32 weights, warmup in round 1 only (all run options) |
| Staying connected | heartbeat from the poll loop, no retries | dedicated heartbeat, retries with backoff, re-registration, rediscovery by mesh id, sleep blocked, one worker per GPU |
| Why a twin is slower | not visible | host diagnostics: shared coordinator, battery, power plan, throttling, other GPU processes, PCIe, CPU load |
| Cross-vendor experiments | not possible | runs restricted to vendors; sweeps over vendor mixes |
| Training screen | a spinning GPU fan | a space scene whose planets orbit faster as the mesh works harder |

---

## Quick start

### Host, the machine with the dataset

```bash
npm install
npm run dev
```

The launcher creates the Python environment, installs the PyTorch build this
machine needs, downloads the base checkpoints, starts the coordinator and the
dashboard, and prints:

```
● Mesh is live

  Dashboard       http://localhost:3000
  Other devices   http://gradmesh.local:3000  no IP needed
  On this Wi-Fi   http://192.168.1.85:3000
  Join page       http://192.168.1.85:3000/join

  Anyone on this network pastes one line to lend their GPU:
  irm http://192.168.1.85:3000/join.ps1 | iex          Windows
  curl -fsSL http://192.168.1.85:3000/join.sh | sh     macOS, Linux
```

Open the dashboard, create the first account (it becomes the mesh owner), and
upload a dataset on **Datasets** or import a standard one in **Testing
parameters**. **Use this machine** in the top bar contributes the host's own GPU.

### Contributors

Open `http://gradmesh.local:3000` on the machine with the GPU and paste the
line for its operating system. It finds Python 3.10-3.13 (and on Windows offers
to install it), detects the GPU, installs the matching PyTorch build under
`~/.gradmesh/agent`, launches a test kernel, measures the device and joins.
Ctrl+C leaves the mesh. Nothing is installed system-wide.

What each kind of machine needs first:

| | Needs |
|---|---|
| NVIDIA | GTX 900 or newer, Windows or Linux, NVIDIA driver 580+ (RTX 50 series: at least 570) |
| Intel | Arc A/B series or Core Ultra with Arc Graphics, Windows or Linux, current Intel graphics driver |
| Apple | Apple Silicon (M1 or later), macOS 14 Sonoma or newer |

The **Setup and health** page in the dashboard shows the same, live.

---

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Everything: setup, coordinator, dashboard, LAN addresses |
| `npm run setup` | Just the environment. Safe to re-run. `-- --force`, `-- --backend xpu`, `-- --wheelhouse DIR` |
| `npm run build` then `npm start` | Production mode |
| `npm run worker` | Contribute this machine. `-- --server http://host:8000 --token <token>` for a remote host; other flags pass through to the agent |
| `npm run coordinator` | Control plane only, with FastAPI docs at `/docs` |
| `npm run doctor` | Diagnose this machine, with a fix per failure. `-- --json` for bug reports |
| `npm test` | Path rules, 30 engine tests and the scheduler's claims; `-- --full` adds a typecheck |
| `npm run report [suite-id]` | Figures and tables from a benchmark sweep |
| `python engine/hardware.py` | What GPU this machine has and which PyTorch build it needs |

---

## Architecture

```
                        Browser
                           |
                   Next.js dashboard          <- accounts, live view, uploads
                           |  server-side, holds the mesh token
                           v
                  FastAPI coordinator          <- registry, planning, aggregation
                           |
      +--------------------+--------------------+
      |                    |                    |
   Worker               Worker               Worker
   RTX 4070 (CUDA)      Arc A770 (XPU)       M3 Pro (Metal)
      |                    |                    |
   612 images           251 images           193 images    <- sized by overhead + rate
      |                    |                    |
      +--------------------+--------------------+
                           |
              sample-weighted FedAvg on CPU
                           |
                    next round replans
```

**`engine/`** is Python:

| File | Role |
|---|---|
| `hardware.py` | Detects GPUs before PyTorch exists and chooses the build. The single source of truth for setup, the join flow and doctor. |
| `setup_env.py` | Standard-library installer used by the host and the join flow: builds or repairs environments, installs, verifies with a kernel launch. |
| `accelerator.py` | One interface over `cuda`, `xpu`, `mps` and `cpu`. |
| `trainers.py` | One training round per backend, with phase timings, gradient flushing, fp32 weights and optional worker validation. |
| `ultralytics_xpu.py` | The validated Intel XPU trainer, unchanged from v3. |
| `probe.py` | Throughput probe, kernel diagnosis, host diagnostics. |
| `worker.py` | The agent: transport, heartbeat, identity, image cache, rediscovery. |
| `coordinator/` | FastAPI app, scheduler, aggregation, sharding, store, benchmark harness, evaluation, discovery. |

**The Next.js app** is the dashboard and the join surface. The browser never
holds the mesh token; Next attaches it server-side.

### Where things live

Mesh state (accounts, token, datasets, runs, learned machine statistics) is in
`.gradmesh/` or `GRADMESH_STATE_DIR`. Machine-specific files (the Python
environment, setup record, local worker log) are kept outside synced folders,
so the repository can be opened from OneDrive on another laptop. Details in
[SETUP.md](SETUP.md#4-moving-between-devices).

---

## The scheduler

Documented in full in [RESEARCH.md](RESEARCH.md). A round costs the slowest
machine's time, so the scheduler's job is to make every machine finish
together.

**1. Admission.** Every worker probes FP32 throughput and memory bandwidth at
startup and proves the GPU runs a kernel. A device that cannot fit the model,
or whose probe measured nothing, is rejected with a written reason; one far
below the mesh median is admitted on probation with a capped shard.

**2. Affine sharding (new in v5).** Each machine's round time is modelled as
`fixed_i + n_i / rate_i`: a fixed overhead (transfer, model load, trainer
construction, upload) plus a per-image rate, both learned from its own rounds,
per checkpoint and image size, and kept across coordinator restarts. Shards are
sized so every predicted finish time is equal:

```
T = (N + Σ rate_i · fixed_i) / Σ rate_i        n_i = rate_i · (T − fixed_i)
```

A machine whose overhead alone exceeds `T` sits the round out, because adding
it could only make the round longer. Fitted to leg 1's single-machine data,
this model reproduces the measured two-machine ceilings (1.53x at 1000 images,
about 1.1x at 100) with no free parameters. v4's rate-only split stays
available as `proportional-linear` for ablation, beside `equal`.

**3. Stragglers.** Each shard has a soft and hard deadline from its own
prediction. Past the soft deadline the fastest idle machine *allowed in this
run* starts a backup copy; past the hard one the round continues without it.
A worker busy with a shard gets three heartbeat timeouts of silence before it
is declared lost.

**4. Aggregation.** Sample-weighted FedAvg, damped by reliability, accumulated
one update at a time on CPU, so the host's memory does not grow with the mesh
and weights from Metal, XPU and CUDA average identically.

---

## Benchmarking for the paper

**Testing parameters** runs a factorial sweep and writes JSON, CSV and figures
that survive a crash; `npm run report` turns a sweep into figures and tables.

v5 adds the axes the cross-vendor paper needs:

- **Vendor mixes**: NVIDIA, Intel, Apple, any pair, or all three. Each mix runs
  on every online machine of those vendors against an automatic one-machine
  baseline. A mix with a vendor missing is recorded as skipped, never run
  smaller under the same label.
- **Three partitioning arms**: affine (v5), rate-proportional (v4), equal.
- **Per-vendor breakdowns** in every round record: images, machine time, pure
  training time and aggregation weight per backend.
- **The training recipe as a parameter**: warmup policy, optimiser, worker
  validation, determinism, the same on every vendor.

[TESTING.md](TESTING.md) has the methodology; [RESULTS-LEG-1.md](RESULTS-LEG-1.md)
analyses the first sweep and lists what v5 changed in response.

---

## Trust model

GradMesh is designed for a network you can see.

- **Workers authenticate with a mesh token**, minted by the coordinator.
  Rotating it on the Invite page disconnects every machine.
- **The join page shows the token** to anyone who can reach the host: on a LAN,
  being on the network is the trust boundary.
- **Dashboard accounts are separate.** The first is the owner; later accounts
  are read-only members. Passwords are scrypt-hashed; sessions are HMAC-signed.
- **Nothing leaves the network.** Workers run Ultralytics with telemetry,
  update checks and automatic installs switched off, and the AMP check no
  longer downloads a model from GitHub.
- **Not solved:** a malicious worker can return poisoned weights. Do not run
  this across a network you do not control.

---

## Limitations

1. Round-synchronised weight averaging, not step-level gradient sync. For one
   machine with several GPUs, PyTorch DDP remains the right tool.
2. Communication scales with model size; tuned for small detection models on a LAN.
3. No worker result verification.
4. In-flight round state is in memory; a coordinator restart loses the current
   round (learned machine statistics survive).
5. AMD GPUs and Intel Macs cannot contribute in v5.
6. Intel XPU and Apple MPS paths are implemented against PyTorch's documented
   APIs and unit tested; the end-to-end training path in this release was
   exercised on NVIDIA hardware. Run the vendor-mix sweep's smallest design on
   each new machine type before a long experiment.

---

## Credits

The training screen's scene is "space boi" by silvercrow101, licensed
CC BY-NC 4.0; see [public/models/CREDITS.md](public/models/CREDITS.md). It is
non-commercial: replace it before any commercial use.
