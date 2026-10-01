# Setting up GradMesh 5

Everything a machine needs before it can host a mesh or contribute a GPU,
and how to install it by hand when the automatic path does not work.

The automatic paths are:

| Role | Command | What it does |
|---|---|---|
| Host | `npm install` then `npm run dev` | Python environment, PyTorch for this machine's GPU, base checkpoints, coordinator, dashboard |
| Contributor (no repository) | the one line on the host's **Invite a GPU** page | Finds Python, installs the agent and the right PyTorch build under `~/.gradmesh/agent`, joins |
| Contributor (repository cloned) | `npm run worker -- --server http://HOST:8000 --token TOKEN` | Same, using the repository |
| Any machine | `npm run doctor` | Checks every item below and prints a fix for each failure |

If those work, you do not need the rest of this file.

---

## 1. Requirements at a glance

### Every machine

| Requirement | Version | Why |
|---|---|---|
| Python | **3.10, 3.11, 3.12 or 3.13** (3.12 recommended) | PyTorch 2.13 publishes wheels for exactly these. 3.9 and 3.14 do not work. |
| Network | same LAN as the host; TCP **3000** and **8000** reachable on the host | dashboard and coordinator |
| Disk | about **6 GB** free (PyTorch with CUDA is about 5 GB installed) | |
| Internet, once | `pypi.org`, `download.pytorch.org`, `github.com` | to download wheels and the base checkpoints; training itself needs only the LAN |

### The host additionally

| Requirement | Version |
|---|---|
| Node.js | **20 or newer** (22 LTS recommended) |
| npm | comes with Node |

The host needs PyTorch even when it has no GPU, because it averages the
weights every round. Setup installs the CPU build in that case.

### Per GPU vendor

| Vendor | Hardware | Operating system | Driver | PyTorch build installed |
|---|---|---|---|---|
| **NVIDIA** | GeForce GTX 900 series (compute 5.0) up to RTX 50 series (compute 12.0) | Windows 10/11, Linux | **R580 or newer** for the CUDA 13.0 build; R525+ (Linux) / R528+ (Windows) for CUDA 12.6; RTX 50 series needs at least **R570** | `torch==2.13.0+cu130`, `+cu126`, or `2.11.0+cu128` (see below) |
| **Intel** | Arc A-series, Arc B-series, Core Ultra with **Arc Graphics**; Iris Xe is attempted but not officially supported | Windows 10/11, Linux | latest Intel graphics driver (Windows); Intel compute runtime, `intel-opencl-icd` and `level-zero` (Linux) | `torch==2.13.0+xpu` |
| **Apple** | any Apple Silicon Mac (M1 or later) | **macOS 14 Sonoma or newer** | built in | `torch==2.13.0` from PyPI (Metal / MPS) |
| none | anything | any | - | `torch==2.13.0+cpu`; the machine joins, is measured, and receives no shards |

Not supported: Intel Macs (PyTorch stopped publishing builds after 2.2), AMD
GPUs (no ROCm path in GradMesh 5 yet), macOS before 14 on Apple Silicon.

### The reference stack

Every machine in a mesh runs the same software, so a result from an NVIDIA, an
Intel and an Apple machine can be compared:

| Package | Version |
|---|---|
| torch | **2.13.0** |
| torchvision | **0.28.0** |
| ultralytics | **8.4.46** |

The pins live in `engine/version.py` and the `engine/requirements-*.txt`
files. The one exception is an RTX 50 series card on an R570-R579 driver,
which gets `torch 2.11.0+cu128` because PyTorch published no newer cu128
wheels; it is flagged in the dashboard and in `npm run doctor`, and updating
the driver to R580+ moves it onto the reference stack.

### How the NVIDIA build is chosen

From the **compute capability and the driver**, which `nvidia-smi` reports
before PyTorch exists, never from the card's name. A CUDA wheel only contains
kernels for the GPU architectures it was compiled for: an RTX 50 series card
on a CUDA 12.6 build installs cleanly, reports CUDA available, and then fails
every launch with `no kernel image is available for execution on the device`.

| Compute capability | Driver | Build |
|---|---|---|
| 12.x (Blackwell: RTX 50, RTX PRO 6000) | R580+ | `cu130` |
| 12.x | R570-R579 | `cu128` (torch 2.11, not reference) |
| 12.x | older | blocked: update the driver |
| 7.5 to 9.0 (Turing to Hopper: RTX 20/30/40) | R580+ | `cu130` |
| 7.5 to 9.0 | R525-R579 | `cu126` |
| 5.0 to 7.0 (Maxwell, Pascal, Volta: GTX 900/10) | R525+ | `cu126` (CUDA 13 dropped these) |
| below 5.0 | - | CPU only |

`python engine/hardware.py` prints what this machine has and which build it
needs, without installing anything.

---

## 2. Installing the prerequisites

### Windows

1. **Python 3.12**: `winget install Python.Python.3.12`, or the installer from
   python.org with *Add python.exe to PATH* ticked. Open a new terminal after.
2. **Node.js** (host only): `winget install OpenJS.NodeJS.LTS`
3. **GPU driver**:
   - NVIDIA: <https://www.nvidia.com/Download/index.aspx>, version 580 or newer.
     Check with `nvidia-smi`.
   - Intel: the *Intel Arc & Iris Xe Graphics* driver from intel.com.
4. **Long paths** (recommended): PyTorch contains paths over 200 characters.
   From an Administrator PowerShell:
   ```powershell
   New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
   ```
5. **Firewall** (host only): allow the two ports once, from an Administrator PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "GradMesh" -Direction Inbound -Protocol TCP -LocalPort 3000,8000 -Action Allow
   ```
6. **Power** (recommended for benchmarks): Settings > System > Power > *Best
   performance*, and plug laptops in. The dashboard warns about Balanced plans
   and battery power because both cut clocks.

### macOS (Apple Silicon)

1. **Python**: `brew install python@3.12`, or the macOS installer from
   python.org. The built-in `/usr/bin/python3` is 3.9 and is too old.
2. **Node.js** (host only): `brew install node`
3. macOS **14 Sonoma or newer**: System Settings > General > Software Update.

### Linux (Ubuntu / Debian shown)

1. **Python**: `sudo apt install python3.12 python3.12-venv` (the `-venv`
   package matters: without it `python -m venv` fails).
2. **Node.js** (host only): from nodejs.org or `nvm`; distribution packages
   are often older than 20.
3. **NVIDIA**: `sudo ubuntu-drivers install` (or your distribution's
   `nvidia-driver-580`), reboot, check `nvidia-smi`.
4. **Intel**: Intel's GPU compute runtime packages
   (`intel-opencl-icd`, `libze1` / `level-zero`) from Intel's repository for
   your distribution; add your user to the `render` group.
5. **Firewall** (host only): `sudo ufw allow 3000,8000/tcp` if ufw is active.

---

## 3. Fully manual install

Use this when the one-line command or `npm run setup` fails and you want to
see each step. The automatic paths do exactly this.

### 3.1 A contributor, without the repository

```bash
# 1. Get the agent: these files from the host, or the engine/ folder of the repository
mkdir -p ~/.gradmesh/agent && cd ~/.gradmesh/agent
for f in version.py hardware.py setup_env.py accelerator.py probe.py trainers.py \
         federated_training.py ultralytics_xpu.py worker.py \
         requirements-control.txt requirements-common.txt \
         requirements-train-cu130.txt requirements-train-cu126.txt requirements-train-cu128.txt \
         requirements-train-cpu.txt requirements-xpu.txt requirements-mps.txt; do
  curl -fsSL "http://HOST:3000/api/agent/$f" -o "$f"
done

# 2. See what this machine needs
python3.12 hardware.py

# 3. Create the environment and install the build it printed, for example CUDA 13.0
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-train-cu130.txt

# 4. Prove the GPU runs
python3.12 setup_env.py verify --venv .venv

# 5. Join
.venv/bin/python worker.py --server-url http://HOST:8000 --token TOKEN --discover
```

On Windows the same, with `py -3.12` for `python3.12`, `.venv\Scripts\python.exe`
for `.venv/bin/python`, and `Invoke-WebRequest -Uri ... -OutFile ...` for curl.
Steps 3 to 5 are what `python setup_env.py agent --server http://HOST:8000 --token TOKEN`
does in one go.

Pick the requirements file from the table in section 1: `requirements-train-cu130.txt`,
`requirements-train-cu126.txt`, `requirements-train-cu128.txt`,
`requirements-xpu.txt`, `requirements-mps.txt` or `requirements-train-cpu.txt`.

### 3.2 The host

```bash
git clone <repository> gradmesh && cd gradmesh
npm install

# Environment and coordinator runtime
python3.12 engine/setup_env.py install --venv .venv --plane control
# PyTorch for this machine (host mode falls back to CPU if the GPU is blocked)
python3.12 engine/setup_env.py install --venv .venv --plane training --host

npm run dev
```

`npm run setup` runs the same two installs, plus the checkpoint downloads, and
writes the record the dashboard reads.

### 3.3 Offline or slow internet: a wheelhouse

Download the wheels once on a good connection, then install from the folder:

```bash
# on the connected machine, for a Windows target on Python 3.12 with CUDA 13.0
python -m pip download -d wheels --only-binary=:all: --python-version 3.12 --platform win_amd64 \
  --extra-index-url https://download.pytorch.org/whl/cu130 -r engine/requirements-train-cu130.txt

# on the target
npm run setup -- --wheelhouse D:\wheels
# or a contributor:  python setup_env.py agent --server ... --token ... --wheelhouse D:\wheels
```

`GRADMESH_WHEELHOUSE` does the same for every command.

---

## 4. Moving between devices

GradMesh 5 is built to be opened from a synced folder (OneDrive, Dropbox,
iCloud Drive) on several machines. What lives where:

| What | Where | Synced? |
|---|---|---|
| Source code | the repository | yes, that is the point |
| `node_modules` | the repository | yes, but platform-specific; run `npm install` on each machine (`npm run doctor` detects a copy from another OS) |
| Python environment | repository `.venv`, **or** `%LOCALAPPDATA%\GradMesh\<key>\venv` / `~/Library/Application Support/GradMesh/<key>/venv` / `~/.local/share/gradmesh/<key>/venv` when the repository is in a synced folder | never |
| Setup record, local worker log and pid | same machine-local folder | never |
| Mesh state: accounts, join token, datasets, run history, learned machine statistics | `.gradmesh/` in the repository, or `GRADMESH_STATE_DIR` | yes, unless you move it |
| A contributor's agent | `~/.gradmesh/agent` | never |

Why it matters: a virtual environment records the absolute path of the Python
that created it, so a synced `.venv` looks present on the second machine and
does not run. v4 kept it in the repository. v5 keeps it outside any synced
folder, marks every environment with the machine that built it, and rebuilds
one that came from somewhere else. Dataset paths are stored relative to the
state directory, so they also survive a different user name.

For large datasets, point `GRADMESH_STATE_DIR` at a local folder so gigabytes
of images do not sync.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PORT` | 3000 | dashboard port |
| `GRADMESH_COORDINATOR_PORT` | 8000 | coordinator port |
| `GRADMESH_STATE_DIR` | `<repo>/.gradmesh` | shared mesh state |
| `GRADMESH_VENV` | automatic | Python environment location |
| `GRADMESH_HOME` | per-OS app data | machine-local folder |
| `GRADMESH_WHEELHOUSE` | none | install wheels from this folder |
| `GRADMESH_BACKEND` | auto | force `cuda`, `xpu`, `mps` or `cpu` |
| `GRADMESH_EVAL_DEVICE` | auto | pin the device the coordinator scores models on |
| `GRADMESH_HEARTBEAT_TIMEOUT` | 20 | seconds of silence before a worker counts as offline (three times that while it holds a shard) |
| `GRADMESH_AGENT_HOME` | `~/.gradmesh/agent` | where the join flow installs |
| `GRADMESH_WORKER_HOME` | `~/.gradmesh` | worker cache, models and locks |

---

## 5. Contributor options

`python worker.py --help` lists them all. The ones that matter:

| Flag | Use |
|---|---|
| `--gpu-index N` | a machine with several GPUs; run one worker per GPU |
| `--backend xpu` | a laptop with an Intel Arc iGPU **and** an NVIDIA card, contributing the Intel one (needs the XPU build, so a separate agent folder: `GRADMESH_AGENT_HOME=~/.gradmesh/agent-xpu`) |
| `--workers N` | dataloader processes; Windows defaults to 0 because each process loads the CUDA libraries, which exhausted the page file in testing |
| `--no-cache` | download whole shard archives instead of caching images |
| `--allow-sleep` | do not block system sleep |
| `--allow-cpu` | let a machine without a GPU take shards; slow, for testing the pipeline only |

---

## 6. Troubleshooting

Run `npm run doctor` first. The usual causes, by symptom:

| Symptom | Cause | Fix |
|---|---|---|
| `Python 3.10 to 3.13 is needed` | none installed, or only 3.9 / 3.14 | install 3.12 (section 2) |
| `No matching distribution found for torch==2.13.0...` | unsupported Python, or a 32-bit Python | 64-bit Python 3.10-3.13 |
| `OSError: [Errno 2] No such file or directory` during the PyTorch install on Windows | 260-character path limit | enable long paths (section 2) |
| `no kernel image is available for execution on the device` | a CUDA build without kernels for this GPU, almost always an RTX 50 on an older build | `npm run setup` (or rerun the join command); both choose by compute capability |
| `torch.cuda.is_available() is False` after install | NVIDIA driver missing or too old for the build | update the driver (R580+), rerun setup |
| `torch.xpu.is_available() is False` | Intel driver or compute runtime missing | install them (section 2), rerun |
| MPS unavailable on a Mac | macOS older than 14 | update macOS |
| the environment "will not start" / rebuilt on every run | it was synced from another machine | nothing; v5 rebuilds it machine-locally once |
| `next dev` fails with a missing `@next/swc-...` | `node_modules` from another OS | delete `node_modules`, `npm install` |
| a contributor cannot reach the host | firewall, or a different network | `Test-NetConnection HOST -Port 8000` (Windows) or `nc -vz HOST 8000`; add the firewall rule |
| `gradmesh.local` does not resolve | network blocks multicast | use the IP address the launcher prints |
| a worker joins, then disappears | machine slept, Wi-Fi dropped, or two terminals for one GPU | v5 blocks sleep, retries, re-registers and refuses a second worker per GPU; check the worker's terminal |
| `the paging file is too small` / `DataLoader worker exited unexpectedly` | Windows memory with dataloader processes | v5 retries with `workers=0` automatically; close other applications |
| `out of memory (GPU)` | batch too large for the card | v5 halves that machine's batch automatically for the next round |
| `out of memory (system RAM, not GPU)` | the machine is short of RAM | close other applications |
| identical GPUs train at very different speeds | coordinator on the same machine, battery, power plan, throttling, other GPU processes, slower CPU for data loading | the machine card lists each of these it detects; the scheduler adapts either way |
