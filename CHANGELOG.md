# Changelog

## 5.1.0

Ready for the Apple test machine, an 8 GB MacBook Air M1. The software each
machine runs is shown on the dashboard. The landing page is rebuilt around the
3D scene.

### Apple Silicon, and an 8 GB M1 in particular

- An Intel Python under Rosetta no longer makes an M1 look like an Intel Mac.
  `hardware.detect()` tells the two apart (`translated`). The join script tries
  `/opt/homebrew` first and runs every candidate as `arch -arm64`. If setup
  starts under Rosetta anyway, it restarts itself natively, and if no native
  Python exists it stops with the exact fix.
- On Macs with 8 GB or less, the worker caps Metal allocations at the
  recommended working set (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`), so an
  oversized batch fails cleanly and is halved instead of swapping to the SSD.
- Apple GPUs report their core count (`Apple M1, 8-core GPU`), and a Metal
  out-of-memory error says that the memory is shared with macOS and open apps.
- New diagnostics: macOS thermal state and Low Power Mode, read through
  Foundation without extra packages, plus swap in use and free disk, each with
  a dashboard warning.
- `docs/MAC-M1.md`: the machine, what GradMesh does for it, the checklist for
  the day before and the day itself, expected batch sizes, troubleshooting,
  and a session log.

### Software on the dashboard

- Workers report Python, torchvision, Ultralytics, the runtime (CUDA version,
  Intel XPU, Metal) and a readable OS name next to the torch build.
- The coordinator compares each machine with the reference stack
  (`coordinator/software.py`).
- Each machine card has a Software block with a *reference stack* / *off
  reference* badge. The Machines page has a table of every machine's stack,
  the sidebar shows the GradMesh version and the pinned stack, and the Setup
  page names exactly what drifted.

### Landing page

- Rebuilt around the space scene: a fixed WebGL backdrop that turns on its own
  while the model moves to whichever side each section leaves free. Planets
  orbit, ripples spread, and the rings take the vendor colours.
- Sections cover the three vendors, the scheduler's equation with an animated
  v4-vs-v5 timeline, the measured anatomy of a round, the join commands and
  the trust model. Live mesh status comes from `/api/health`.
- Mobile layout: the scene sits smaller and dimmer behind the text. Rendering
  is capped at 1.5x pixels, pauses in background tabs and respects reduced
  motion.
- The copy-command field gets real styles; it had none, so the copy button
  could overlap long commands.

## 5.0.0

The baseline for the cross-vendor paper: NVIDIA, Intel and Apple GPUs in one
mesh, on one software stack, with the setup problems that made v4 hard to move
between machines removed.

### Three GPU families

- `engine/accelerator.py` covers CUDA, Intel XPU, Apple MPS and CPU behind one
  interface: device, memory (including Apple's unified memory), synchronise,
  cache, AMP support.
- `engine/trainers.py` runs one round on any backend with the same optimiser,
  learning rate, warmup and seed from the run. v4 trained XPU with Adam and
  CUDA with Ultralytics' automatic choice.
- Workers serialise weights on CPU, so Metal, XPU and CUDA updates average
  identically.
- Runs can be restricted to vendors; every round records images, machine time,
  training time and aggregation weight per backend; sweeps take vendor mixes as
  a factor with an automatic one-machine baseline.
- Dashboard: vendor badges everywhere, a hardware-mix panel, vendor filters,
  per-vendor run breakdowns, a Setup and health page.

### One reference stack, chosen correctly

- torch 2.13.0, torchvision 0.28.0, ultralytics 8.4.46 on every vendor
  (`engine/version.py`). CUDA 13.0 for Turing onward on driver R580+, CUDA 12.6
  for older cards and drivers, CUDA 12.8 (torch 2.11, flagged) for Blackwell on
  R570-R579, XPU for Intel Arc and Core Ultra, the stock macOS wheel for Apple
  Silicon on macOS 14+.
- `engine/hardware.py` is the only place these rules live; the host setup, the
  join scripts and `npm run doctor` all call it. v4 had three diverging copies.
- `engine/setup_env.py` installs with the standard library only, retries,
  supports a local wheelhouse, and proves the GPU with a kernel launch before
  calling anything ready.

### Moving between devices

- When the checkout is in OneDrive, Dropbox or iCloud, the Python environment,
  setup record and local-worker files live in a machine-local folder; a synced
  `.venv` from another machine is detected and rebuilt.
- Dataset paths are stored relative to the state directory.
- Python is required to be 3.10-3.13 up front instead of failing after a 2 GB
  download; the PowerShell join script no longer misses Microsoft Store Python.
- The printed LAN address prefers the routed interface over VirtualBox,
  Hyper-V, WSL and VPN adapters.
- `npm run doctor` checks node_modules from another OS, synced folders, long
  paths, the environment's origin, build-versus-GPU match, kernel launch,
  reference stack, datasets present on this machine and the firewall rule.
- `.gitignore` no longer hides `app/dashboard/runs/` from every clone.

### Staying connected

- A dedicated heartbeat thread for the worker's lifetime; requests retry with
  backoff; result uploads are idempotent.
- Workers re-register after a coordinator restart and find a moved coordinator
  by mesh id, via `gradmesh.local` then the subnet.
- Node ids are derived from the machine, backend and GPU index; an OS lock
  allows one worker per GPU; a restarted worker supersedes its old process.
- A busy worker gets three heartbeat timeouts of silence before its shard is
  dropped. Speculative backups only use machines the run is allowed to use.
- Workers block system sleep while contributing.
- Learned machine statistics survive coordinator restarts.

### Scheduling

- Affine cost model: fixed overhead per round plus per-image rate, learned per
  checkpoint and image size, robust to single noisy rounds, cold first rounds
  handled separately. Machines whose overhead exceeds the round sit it out.
- v4's split kept as `proportional-linear` for ablations.
- Out-of-memory failures halve that machine's batch for the next round instead
  of counting towards quarantine.
- Host diagnostics explain why identical GPUs train at different speeds.

### Faster rounds

- Worker-side validation is off by default (two passes per worker per round).
- The AMP check no longer downloads `yolo26n.pt` from GitHub every round.
- Shards are file lists; workers cache images and download only what is new;
  legacy zips are built on demand and stored, not deflated.
- Weights move as raw bytes.
- In testing on an RTX 3050 the per-round overhead fell from about 9 s (legacy
  path, same machine) to under 1 s.

### Training correctness

- **Optimiser steps in small rounds.** Ultralytics steps only after a nominal
  64-image batch accumulates, so a shard of fewer batches never updated the
  model, and every round discarded leftover gradients. Pending gradients are
  now flushed at the end of each round; optimiser steps are reported.
- **fp16 rounding every round.** Workers returned the reloaded `best.pt`, saved
  in half precision. They now send the trainer's fp32 EMA weights.
- **Warmup.** Round 1 only by default (`warmup_mode`), fixing leg 1's falling
  accuracy; `every-round` reproduces v4.
- Verified: mAP50 0.046, 0.055, 0.057, 0.074 over four rounds where v4's path
  stayed flat at 0.046.

### Training screen

- The spinning fan is replaced by a 3D scene ("space boi" by silvercrow101,
  CC BY-NC 4.0): planets orbit a figure in still water, faster as more of the
  mesh trains, coloured by run state.

### Tooling

- `npm test` (path rules, 30 engine tests, scheduler suite; `--full` adds a
  typecheck) and a CI workflow on Windows, macOS and Linux.
