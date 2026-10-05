# Test device: MacBook Air M1 (8 GB)

The Apple machine for the GradMesh 5 cross-vendor runs. This file is kept up
to date with this one machine: what it is, what GradMesh does differently for
it, what to do before and on the test day, what to expect, and a log of every
session on it.

**Status (2 October 2026):** prepared, not yet tested. No Mac has been
connected to this build yet. Everything below comes from the code paths and
from how PyTorch's Metal backend behaves on an 8 GB M1. Correct it after the
first session.

---

## 1. The machine

| | |
|---|---|
| Model | MacBook Air (M1, 2020) |
| CPU | 8 cores: 4 performance, 4 efficiency |
| GPU | Apple M1, 7 or 8 cores (the Air shipped with both; record which below) |
| Memory | 8 GB unified, shared by macOS, every app and the GPU |
| Storage | 256 GB SSD |
| Cooling | none (fanless), so it slows itself down when hot |
| GradMesh backend | `mps` (Metal) through the stock macOS PyTorch wheel |
| Build | `engine/requirements-mps.txt`: torch 2.13.0, torchvision 0.28.0, ultralytics 8.4.46 (the reference stack) |
| OS needed | macOS 14 Sonoma or newer |

Record which one this is:

```sh
system_profiler SPDisplaysDataType | grep -i "cores"     # "Total Number of Cores: 7" or 8
sw_vers -productVersion                                   # macOS version
```

The dashboard shows the same information. The machine card reads
`Apple M1, 8-core GPU` (or 7-core), and its Software block lists the macOS
version.

---

## 2. What GradMesh does for this Mac

| Concern | What happens | Where |
|---|---|---|
| **Python under Rosetta.** An Intel (x86_64) Python on an M1 reports an Intel Mac, and PyTorch has no Intel-Mac builds. | The join script tries `/opt/homebrew` Pythons first and runs every candidate as `arch -arm64`, so an Intel-only Python fails the check instead of being picked. If setup still starts under Rosetta, it restarts itself as arm64. If no native Python exists, it stops and names the fix. | `app/api/join/[platform]/route.ts`, `engine/setup_env.py` (`relaunch_natively_if_translated`), `engine/hardware.py` |
| **8 GB of shared memory.** PyTorch normally lets Metal use 1.7 times its recommended working set. On 8 GB that is more memory than the machine has, so an oversized batch would swap to the SSD and stall the Mac instead of failing. | On a Mac with 8 GB or less, the worker caps Metal at the recommended working set (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, `LOW=0.8`). An oversized batch then fails cleanly with "MPS backend out of memory", and the coordinator halves that machine's batch for the next round. A value you set yourself is left alone. | `engine/worker.py` (`cap_metal_memory`) |
| **Batch size.** | The coordinator sizes each machine's batch from the memory it reports. For Apple that is Metal's recommended working set, not the 8 GB total, and only 55% of it is spent (75% on a discrete GPU). | `engine/coordinator/scheduler.py` (`safe_batch_size`) |
| **Heat.** A fanless Air slows its GPU down once it is hot. | The worker reads macOS's own thermal state (nominal, fair, serious, critical) and whether Low Power Mode is on. The machine card warns at "serious". The scheduler learns the slower rate from the rounds themselves either way. | `engine/probe.py`, `engine/coordinator/health.py` |
| **Memory pressure and disk.** | The worker reports swap in use and free disk. The card warns about 2 GB or more of swap on an 8 GB machine, and about less than 5 GB of free disk. | same |
| **Sleep.** | `caffeinate -i` keeps the Mac awake while it contributes. Closing the lid still puts it to sleep, so keep the lid open. | `engine/worker.py` |
| **Missing Metal operators.** | `PYTORCH_ENABLE_MPS_FALLBACK=1` runs the few missing operators on the CPU instead of failing the round. | `engine/accelerator.py`, `engine/worker.py` |
| **AMP and data loading.** | Ultralytics trains in FP32 on MPS and loads data in the training process (`workers=0`). That is its default and fine here: the M1 GPU, not data loading, limits YOLOv8n. | Ultralytics |
| **Software shown on the dashboard.** | The card's Software block shows Python, PyTorch with its build, torchvision, Ultralytics, the Metal runtime, the macOS version and the agent version. A badge says whether that matches the reference stack. | `engine/probe.py`, `engine/coordinator/software.py`, `components/dashboard/SoftwareStack.tsx` |

### Expected batch sizes

Metal's recommended working set on an 8 GB M1 is about 5.3 GB. The worker
prints the real figure when it starts (`Apple M1, 8-core GPU via mps, NNNN MB`),
and it appears as *Memory* on the card. With 5.3 GB the coordinator allows:

| Image size | Largest batch it will assign |
|---|---|
| 640 | 6 |
| 512 | 9 |
| 416 | 14 |
| 320 | 24 |

A run's batch size is a ceiling for every machine. The Mac gets the smaller of
the run's batch and the value above. After an out-of-memory round its cap is
halved.

---

## 3. Before the day (once, about 20 minutes, needs internet)

1. **macOS 14 or newer.** Apple menu > About This Mac. If it is older:
   System Settings > General > Software Update.
2. **Terminal must not run under Rosetta.** In Terminal, `arch` must print
   `arm64`. If it prints `i386`: Finder > Applications > Utilities > Terminal >
   Get Info > untick *Open using Rosetta*, then reopen Terminal.
3. **Homebrew** (skip if `brew --version` works):
   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Then run the two `echo ... >> ~/.zprofile` / `eval ...` lines it prints
   under *Next steps*, so `/opt/homebrew/bin` is on PATH.
4. **Python 3.12, Apple Silicon build:**
   ```sh
   brew install python@3.12
   /opt/homebrew/bin/python3.12 -c "import platform; print(platform.machine())"   # must print arm64
   ```
   The python.org macOS installer works too. Do not use `/usr/bin/python3`;
   it is 3.9.
5. **Optional, saves the download on the day.** With the host running anywhere
   on the same network, run the join command once (section 4) and stop it with
   Ctrl+C once it says *Contributing this GPU*. PyTorch (about 1 GB) and the
   YOLO weights stay cached under `~/.gradmesh`, and the next join takes
   seconds.
6. **Free disk:** keep at least 10 GB free. The environment is about 1.5 GB,
   and the image cache grows with the dataset.

---

## 4. On the day

Checklist, in order:

- [ ] Charger connected. On battery, macOS lowers clocks and the card warns.
- [ ] Low Power Mode off: System Settings > Battery > Low Power Mode > *Never*.
- [ ] Lid open, on a hard flat surface (not a bed or a lap), out of direct sun.
      A laptop stand helps a fanless Mac stay cool.
- [ ] Quit browsers, Slack, Teams, Docker and anything else heavy. Activity
      Monitor > Memory > *Memory Pressure* should be green. Restarting the Mac
      first clears leftover swap.
- [ ] On the same Wi-Fi as the host, and not a guest network that isolates
      devices. Check: `nc -vz HOST_IP 8000` must say *succeeded*.
- [ ] Join, with the command from the host's dashboard (Invite a GPU):
      ```sh
      curl -fsSL http://HOST_IP:3000/join.sh | sh
      ```
- [ ] The terminal should show, in order:
      - `found Python: /opt/homebrew/bin/python3.12`
      - `detected Apple M1, 8-core GPU`
      - a note about 8 GB of unified memory (expected, informational)
      - `PyTorch macOS build (Metal) ready: torch 2.13.0 on Apple GPU (Metal)`
      - `Apple M1, 8-core GPU via mps, NNNN MB, NNN GFLOP/s`
      - `caffeinate is keeping this Mac awake`
      - `joined ... (full: admitted)` or `(probation: ...)`
- [ ] On the host, open Dashboard > Machines. The Mac's card shows the Apple
      badge, and its Software block reads Python 3.12.x, PyTorch 2.13.0 ·
      macOS, torchvision 0.28.0, Ultralytics 8.4.46, Runtime Metal, OS macOS
      1x.x, with the green **reference stack** badge.
- [ ] If macOS asks whether Python may accept incoming connections, either
      answer works: the worker only makes outgoing connections.
- [ ] If macOS asks whether **Terminal** may find and connect to devices on
      your local network, choose **Allow**. Without it (macOS 15 Sequoia and
      later), the Mac cannot reach the host at all. If you already denied it:
      System Settings > Privacy & Security > Local Network > Terminal on.

### Datasets and the Mac

The Mac does not need a dataset to contribute. The host owns the datasets,
and each round the worker downloads only the images in its slice and caches
them under `~/.gradmesh/cache`.

To move a dataset onto or off the Mac anyway, use the dashboard from Safari on
the Mac:

- **Download:** Datasets > Download on any row gives a zip in the layout the
  page accepts. Safari unzips it into a folder by default.
- **Upload:** drop the dataset folder onto the Datasets page, or use *Choose a
  folder instead*. It is zipped in the browser and uploaded. A zip works as
  well, including one made with Finder's *Compress*: its `__MACOSX` folder is
  ignored.

### Run settings that suit this Mac

- **Warm up first.** Run one short round before measuring. The first Metal
  round compiles shaders, and its overhead is larger than in later rounds.
- **Image size.** 640 works at batch 6 or less. 416 or 512 give more headroom
  and shorter rounds if the run allows it.
- **Let it learn.** Round 1 is planned from the probe. By rounds 2 and 3 the
  scheduler has the Mac's measured rate and overhead, and its shard size
  settles.
- **Heat.** For the paper, note the thermal state the card shows. Comparing
  the rate in rounds 1 to 3 with later rounds shows how much passive cooling
  costs.

---

## 5. Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| `An Apple Silicon (arm64) Python 3.10 to 3.13 is needed and was not found.` | Only Intel or too-old Pythons are installed | Section 3, steps 3 and 4 |
| `this Python was running under Rosetta; restarting as Apple Silicon` | Setup started under Rosetta | Nothing to do; it fixed itself |
| `This Mac has Apple Silicon, but the Python that ran this is an Intel (x86_64) build` | No native Python to switch to | `brew install python@3.12`, then join again |
| `PyTorch 2.13.0 needs macOS 14 Sonoma or newer` | Older macOS | Update macOS |
| `the Metal backend is not available` | Older macOS, or a damaged environment | Update macOS, then rejoin with `--force`: `/opt/homebrew/bin/python3.12 ~/.gradmesh/agent/setup_env.py agent --server http://HOST_IP:8000 --token TOKEN --force` |
| A dialog asks to install the command line developer tools | Something ran `/usr/bin/python3` | Cancel it. The join script never runs that path. |
| `curl: (7) Failed to connect ... No route to host` while the host is up | Terminal was denied Local Network access (macOS 15+) | System Settings > Privacy & Security > Local Network > Terminal on, then rerun |
| The dashboard only accepts a zip, but the download became a folder | Safari unzips downloads | Drop the folder, or use *Choose a folder instead* (5.1.1) |
| `out of memory (Metal, unified memory shared with macOS and open apps)` | The batch did not fit beside the other apps | Automatic: the next round's batch is halved. Quit other apps. |
| Card: `macOS reports a serious thermal state` | The fanless Mac is hot and slowing down | Airflow: lid open, hard surface, stand, cooler room. A smaller image size also helps. |
| Card: `Low Power Mode is on` | | System Settings > Battery |
| Card: `N GB of swap in use on a 8 GB machine` | Memory is short | Quit apps, or restart the Mac before the session |
| `NotImplementedError ... MPS` | An operator Metal lacks | Should not happen, because the CPU fallback is on. If it does, keep the worker log for the issue. |
| The Mac drops out of the mesh | The lid was closed (sleep), or the Wi-Fi changed | Keep the lid open. The worker finds the host again once the network is back. |

Logs: the worker prints to the terminal that ran the join command. The
installer log is `~/.gradmesh/agent/setup.json`.

---

## 6. Session log

Add one row per session. These numbers go into the paper's hardware table.

| Date | macOS | GPU cores | Python | torch | Run id | imgsz | Batch given | Rounds | Rate (img/s) | Overhead (s/round) | Share of images | Thermal seen | Warnings | OOM rounds | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |

Where the numbers come from: *Rate* and *Overhead* are on the machine card
(Training rate, Overhead per round). *Share of images* and per-round timings
are in the run's page under the per-vendor breakdown. *Batch given* is in the
worker's terminal (`round N shard M: K images, batch B`).

---

## 7. Changes made for this machine

- **5.1.1 (5 October 2026):**
  - Datasets can be downloaded from the dashboard on any machine.
  - Folders can be uploaded, and are zipped in the browser.
  - Zips made by Finder (with `__MACOSX`) register the real images.
  - The host now runs v5, with v4's state carried over.

- **5.1.0 (2 October 2026):**
  - Rosetta detection, with a native relaunch in setup and arm64-only Python
    selection in the join script and the host scripts.
  - The Metal memory cap on Macs with 8 GB or less.
  - The GPU core count in the device name.
  - New diagnostics: macOS thermal state, Low Power Mode, swap and free disk,
    each with a dashboard warning.
  - A clearer out-of-memory message for Metal.
  - The per-machine software stack on the dashboard.
