"""What is in this machine, and which PyTorch build it needs.

This is the single source of truth for build selection. The host's `npm run
setup`, the one-line join scripts and `npm run doctor` all ask this module,
rather than each carrying its own copy of the rules. v4 had three copies, in
JavaScript, PowerShell and sh, and they drifted: the PowerShell one knew about
Blackwell before the sh one did.

Two halves, kept apart on purpose.

* `detect()` gathers facts. It shells out to nvidia-smi, the Windows video
  controller list, lspci, sysfs and macOS system tools. It needs nothing but the
  standard library, because it runs before PyTorch exists.
* `select_profile()` is a pure function from those facts to a profile. It has no
  side effects, so every rule in it is unit tested against fabricated machines.

The rules, and why each one exists:

NVIDIA. A CUDA wheel only contains kernels for the architectures it was
compiled for, and the toolkit it was built with sets a minimum driver. So the
build is chosen from the compute capability *and* the driver, never from the
card's name. Blackwell (compute 12.x, the RTX 50 series) has no kernels in any
CUDA 12.6 wheel and fails every launch with "no kernel image is available".
CUDA 13.0 covers Turing onward and needs driver 580 or newer; CUDA 12.6 covers
Maxwell through Hopper on any driver from 525.

Intel. Arc discrete cards and Core Ultra "Arc Graphics" integrated GPUs run the
PyTorch XPU build. Iris Xe is not on Intel's supported list; it is attempted
and verified, and falls back to CPU if the runtime will not initialise.

Apple. Apple Silicon runs the stock macOS wheel, which carries the Metal (MPS)
backend. PyTorch 2.13 wheels require macOS 14. Intel Macs lost PyTorch support
after 2.2, and nothing current installs there.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

try:  # Imported by path from the join flow, where the engine is not a package.
    from version import REFERENCE_STACK
except Exception:  # pragma: no cover - only when run from an odd working directory
    REFERENCE_STACK = {"torch": "2.13.0", "torchvision": "0.28.0", "ultralytics": "8.4.46"}


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@dataclass
class Gpu:
    vendor: str  # nvidia, intel, amd, apple, unknown
    name: str
    index: int = 0
    compute_capability: Optional[float] = None
    driver: Optional[str] = None
    memory_mb: Optional[int] = None


@dataclass
class HostFacts:
    os: str  # windows, linux, macos
    os_version: str
    arch: str  # x86_64, arm64
    python: str
    hostname: str
    cpu_count: int
    ram_mb: Optional[int] = None
    gpus: List[Gpu] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HostFacts":
        gpus = [Gpu(**gpu) for gpu in data.get("gpus", [])]
        payload = {key: value for key, value in data.items() if key != "gpus"}
        return cls(gpus=gpus, **payload)


def _run(command: List[str], timeout: float = 8.0) -> Optional[str]:
    """stdout of a command, or None if it is missing, fails or hangs."""
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, **kwargs
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout or ""


def _os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return machine or "unknown"


def _ram_mb() -> Optional[int]:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys // (1024 * 1024))
        if sys.platform == "darwin":
            raw = _run(["sysctl", "-n", "hw.memsize"])
            return int(raw.strip()) // (1024 * 1024) if raw else None
        with open("/proc/meminfo", "r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def _macos_version() -> str:
    raw = _run(["sw_vers", "-productVersion"])
    return (raw or platform.mac_ver()[0] or "").strip()


def _nvidia_gpus() -> List[Gpu]:
    """Every NVIDIA GPU nvidia-smi can see, with capability and driver."""
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform == "win32":
        # The driver installs it here even when it is not on PATH.
        candidate = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
        )
        smi = candidate if os.path.isfile(candidate) else None
    if not smi:
        return []

    fields = "index,name,compute_cap,driver_version,memory.total"
    raw = _run([smi, "--query-gpu=%s" % fields, "--format=csv,noheader,nounits"])
    with_capability = raw is not None
    if raw is None:
        # nvidia-smi from before the compute_cap field existed.
        raw = _run([smi, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if not raw:
        return []

    gpus: List[Gpu] = []
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            if with_capability and len(parts) >= 5:
                index, name, capability, driver, memory = parts[:5]
            elif len(parts) >= 4:
                index, name, driver, memory = parts[:4]
                capability = ""
            else:
                continue
            gpus.append(
                Gpu(
                    vendor="nvidia",
                    name=name,
                    index=int(index) if index.isdigit() else len(gpus),
                    compute_capability=_float(capability),
                    driver=driver or None,
                    memory_mb=int(float(memory)) if _float(memory) else None,
                )
            )
        except Exception:
            continue
    return gpus


def _float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("nvidia", "geforce", "quadro", "tesla", "rtx ")):
        return "nvidia"
    if "intel" in lowered or " arc" in lowered:
        return "intel"
    if any(token in lowered for token in ("amd", "radeon", "advanced micro devices")):
        return "amd"
    if "apple" in lowered:
        return "apple"
    return "unknown"


def _windows_video_controllers() -> List[Gpu]:
    raw = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress",
        ],
        timeout=15,
    )
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    gpus = []
    for index, entry in enumerate(data or []):
        name = str(entry.get("Name") or "").strip()
        if not name:
            continue
        gpus.append(
            Gpu(vendor=_vendor_from_name(name), name=name, index=index, driver=entry.get("DriverVersion"))
        )
    return gpus


_PCI_VENDORS = {"0x10de": "nvidia", "0x8086": "intel", "0x1002": "amd"}


def _linux_display_devices() -> List[Gpu]:
    gpus: List[Gpu] = []
    raw = _run(["lspci", "-mm"])
    if raw:
        for line in raw.splitlines():
            if not re.search(r'"(VGA compatible controller|3D controller|Display controller)"', line):
                continue
            # -mm quotes every field: slot "class" "vendor" "device" ...
            fields = re.findall(r'"([^"]*)"', line)
            name = " ".join(fields[1:3]) if len(fields) >= 3 else line
            gpus.append(Gpu(vendor=_vendor_from_name(name), name=name.strip(), index=len(gpus)))
        if gpus:
            return gpus

    # No lspci, which is common in containers. sysfs still names the vendor.
    drm = "/sys/class/drm"
    try:
        for card in sorted(os.listdir(drm)):
            if not re.fullmatch(r"card\d+", card):
                continue
            try:
                with open(os.path.join(drm, card, "device", "vendor"), "r", encoding="utf-8") as stream:
                    vendor_id = stream.read().strip().lower()
            except Exception:
                continue
            vendor = _PCI_VENDORS.get(vendor_id, "unknown")
            gpus.append(Gpu(vendor=vendor, name="%s GPU (%s)" % (vendor, card), index=len(gpus)))
    except Exception:
        pass
    return gpus


def _apple_gpus() -> List[Gpu]:
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20)
    if raw:
        try:
            entries = json.loads(raw).get("SPDisplaysDataType", [])
            gpus = []
            for index, entry in enumerate(entries):
                name = entry.get("sppci_model") or entry.get("_name") or "Apple GPU"
                gpus.append(Gpu(vendor=_vendor_from_name(name) if "apple" not in name.lower() else "apple", name=name, index=index))
            if gpus:
                return gpus
        except Exception:
            pass
    if _arch() == "arm64":
        chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon").strip()
        return [Gpu(vendor="apple", name=chip, index=0)]
    return []


def detect() -> HostFacts:
    """Gather everything select_profile() needs. Safe to call before torch exists."""
    os_name = _os_name()
    facts = HostFacts(
        os=os_name,
        os_version=_macos_version() if os_name == "macos" else platform.release(),
        arch=_arch(),
        python="%d.%d.%d" % sys.version_info[:3],
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count() or 1,
        ram_mb=_ram_mb(),
    )

    nvidia = _nvidia_gpus()
    if os_name == "windows":
        listed = _windows_video_controllers()
    elif os_name == "linux":
        listed = _linux_display_devices()
    else:
        listed = _apple_gpus()

    if not nvidia and any(gpu.vendor == "nvidia" for gpu in listed):
        facts.notes.append("an NVIDIA GPU is present but nvidia-smi is not, so the driver is not installed")
        nvidia_listed = [gpu for gpu in listed if gpu.vendor == "nvidia"]
        for gpu in nvidia_listed:
            gpu.driver = None
    else:
        nvidia_listed = []

    # Virtual adapters (remote-desktop and screen-sharing displays, the basic
    # display driver) are not GPUs anyone can train on, so they never reach
    # the profile rules or the dashboard.
    others = [gpu for gpu in listed if gpu.vendor in {"intel", "amd", "apple"}]
    facts.gpus = nvidia + nvidia_listed + others
    return facts


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    name: str  # cuda-cu130, cuda-cu126, cuda-cu128, xpu, mps, cpu
    backend: str  # cuda, xpu, mps, cpu
    requirements: str  # file under engine/
    label: str
    reason: str
    torch: str
    reference: bool = True  # on the reference stack every backend shares
    can_train: bool = True
    gpu_index: int = 0
    gpu_name: Optional[str] = None
    blocked: Optional[str] = None  # set when this machine cannot get a working build
    fix: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# Minimum driver majors. CUDA 13.0 ships with the R580 driver branch; 12.8 with
# R570; 12.x minor-version compatibility goes back to R525 on Linux and R528 on
# Windows.
DRIVER_CU130 = 580
DRIVER_CU128 = 570
DRIVER_CU126_LINUX = 525
DRIVER_CU126_WINDOWS = 528

NVIDIA_DRIVERS_URL = "https://www.nvidia.com/Download/index.aspx"
INTEL_DRIVERS_URL = "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html"

# Integrated and discrete parts on Intel's supported list for the XPU build.
_INTEL_XPU = re.compile(
    r"arc|dg2|battlemage|\bbmg\b|alchemist|meteor ?lake|lunar ?lake|arrow ?lake|panther ?lake|data center gpu",
    re.IGNORECASE,
)
_INTEL_EXPERIMENTAL = re.compile(r"iris\s*\(?r?\)?\s*xe|iris xe", re.IGNORECASE)


def _driver_major(driver: Optional[str]) -> Optional[int]:
    if not driver:
        return None
    match = re.match(r"(\d+)", driver)
    return int(match.group(1)) if match else None


def _cpu(reason: str, *, gpu: Optional[Gpu] = None, blocked: Optional[str] = None, fix: Optional[str] = None,
         warnings: Optional[List[str]] = None) -> Profile:
    return Profile(
        name="cpu",
        backend="cpu",
        requirements="requirements-train-cpu.txt",
        label="PyTorch CPU build",
        reason=reason,
        torch=REFERENCE_STACK["torch"],
        can_train=False,
        gpu_name=gpu.name if gpu else None,
        blocked=blocked,
        fix=fix,
        warnings=list(warnings or []),
    )


def _nvidia_profile(gpus: List[Gpu], os_name: str) -> Profile:
    # Train on the most capable card. A machine mixing a Pascal and a Blackwell
    # card has no single build that covers both, so the stronger one wins.
    best = max(gpus, key=lambda gpu: (gpu.compute_capability or 0.0, gpu.memory_mb or 0))
    capability = best.compute_capability
    driver = _driver_major(best.driver)
    cc_text = "compute %.1f" % capability if capability else "unknown compute capability"
    warnings: List[str] = []
    if len({gpu.compute_capability for gpu in gpus}) > 1:
        warnings.append(
            "this machine has GPUs of different generations; GradMesh trains on GPU %d (%s)"
            % (best.index, best.name)
        )

    def cuda(name: str, requirements: str, label: str, reason: str, torch: str, reference: bool = True) -> Profile:
        return Profile(
            name=name,
            backend="cuda",
            requirements=requirements,
            label=label,
            reason=reason,
            torch=torch,
            reference=reference,
            gpu_index=best.index,
            gpu_name=best.name,
            warnings=warnings,
        )

    if capability is not None and capability < 5.0:
        return _cpu(
            "%s at %s is older than any current PyTorch CUDA build supports" % (best.name, cc_text),
            gpu=best,
            warnings=warnings,
        )

    minimum_cu126 = DRIVER_CU126_WINDOWS if os_name == "windows" else DRIVER_CU126_LINUX
    if driver is not None and driver < minimum_cu126:
        return _cpu(
            "NVIDIA driver %s is too old for CUDA 12" % best.driver,
            gpu=best,
            blocked="The NVIDIA driver on this machine (%s) is too old for any current PyTorch build." % best.driver,
            fix="Install NVIDIA driver 580 or newer from %s, reboot, then run this again." % NVIDIA_DRIVERS_URL,
            warnings=warnings,
        )

    blackwell = capability is not None and capability >= 12.0
    if blackwell:
        if driver is None or driver >= DRIVER_CU130:
            return cuda(
                "cuda-cu130",
                "requirements-train-cu130.txt",
                "PyTorch CUDA 13.0 build",
                "%s is %s (Blackwell), which needs CUDA 12.8 or newer" % (best.name, cc_text),
                REFERENCE_STACK["torch"],
            )
        if driver >= DRIVER_CU128:
            profile = cuda(
                "cuda-cu128",
                "requirements-train-cu128.txt",
                "PyTorch CUDA 12.8 build (fallback)",
                "%s is Blackwell and driver %s predates CUDA 13.0" % (best.name, best.driver),
                "2.11.0",
                reference=False,
            )
            profile.warnings.append(
                "this machine is on torch 2.11 rather than the reference %s, because its driver is "
                "older than R580. Update the NVIDIA driver to keep cross-vendor results comparable."
                % REFERENCE_STACK["torch"]
            )
            return profile
        return _cpu(
            "Blackwell needs driver %d or newer, this machine has %s" % (DRIVER_CU128, best.driver),
            gpu=best,
            blocked="%s needs NVIDIA driver 570 or newer; this machine has %s." % (best.name, best.driver),
            fix="Install NVIDIA driver 580 or newer from %s, reboot, then run this again." % NVIDIA_DRIVERS_URL,
            warnings=warnings,
        )

    # Turing (7.5) onward runs CUDA 13.0 when the driver allows it. Older
    # architectures were dropped from CUDA 13, so they stay on 12.6.
    if capability is not None and capability >= 7.5 and driver is not None and driver >= DRIVER_CU130:
        return cuda(
            "cuda-cu130",
            "requirements-train-cu130.txt",
            "PyTorch CUDA 13.0 build",
            "%s is %s with driver %s" % (best.name, cc_text, best.driver),
            REFERENCE_STACK["torch"],
        )
    return cuda(
        "cuda-cu126",
        "requirements-train-cu126.txt",
        "PyTorch CUDA 12.6 build",
        "%s is %s%s"
        % (best.name, cc_text, " with driver %s" % best.driver if best.driver else ""),
        REFERENCE_STACK["torch"],
    )


def _version_tuple(text: str) -> tuple:
    parts = []
    for piece in re.split(r"[.\-]", text or ""):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            break
    return tuple(parts) or (0,)


def select_profile(facts: HostFacts, prefer: str = "auto") -> Profile:
    """The PyTorch build this machine should run. Pure function of `facts`.

    `prefer` forces a backend family, for a machine that has two kinds of GPU
    and should contribute the one that is not picked by default, for example a
    laptop with an Intel Arc iGPU next to an NVIDIA card.
    """
    prefer = (prefer or "auto").lower()
    nvidia = [gpu for gpu in facts.gpus if gpu.vendor == "nvidia" and gpu.driver is not None]
    nvidia_without_driver = [gpu for gpu in facts.gpus if gpu.vendor == "nvidia" and gpu.driver is None]
    intel = [gpu for gpu in facts.gpus if gpu.vendor == "intel"]
    apple = [gpu for gpu in facts.gpus if gpu.vendor == "apple"]

    if prefer == "cpu":
        return _cpu("CPU was requested explicitly")

    # --- Apple -------------------------------------------------------------
    if facts.os == "macos":
        if facts.arch != "arm64":
            return _cpu(
                "Intel Macs have no current PyTorch builds",
                blocked="PyTorch stopped publishing builds for Intel Macs after version 2.2, so this Mac cannot "
                "run the GradMesh training stack.",
                fix="Contribute an Apple Silicon Mac, or a Windows or Linux machine with an NVIDIA or Intel GPU.",
            )
        if _version_tuple(facts.os_version) < (14,):
            return Profile(
                name="mps",
                backend="mps",
                requirements="requirements-mps.txt",
                label="PyTorch macOS build (Metal)",
                reason="Apple Silicon on macOS %s" % facts.os_version,
                torch=REFERENCE_STACK["torch"],
                gpu_name=(apple[0].name if apple else "Apple Silicon"),
                blocked="PyTorch %s needs macOS 14 Sonoma or newer; this Mac runs %s."
                % (REFERENCE_STACK["torch"], facts.os_version),
                fix="Update macOS from System Settings > General > Software Update, then run this again.",
            )
        return Profile(
            name="mps",
            backend="mps",
            requirements="requirements-mps.txt",
            label="PyTorch macOS build (Metal)",
            reason="Apple Silicon (%s) trains on the Metal backend" % (apple[0].name if apple else facts.arch),
            torch=REFERENCE_STACK["torch"],
            gpu_name=(apple[0].name if apple else "Apple Silicon"),
        )

    # --- Explicit Intel request -------------------------------------------
    if prefer == "xpu" and intel:
        return _intel_profile(intel)

    # --- NVIDIA --------------------------------------------------------------
    if nvidia and prefer in {"auto", "cuda"}:
        return _nvidia_profile(nvidia, facts.os)

    if nvidia_without_driver and prefer in {"auto", "cuda"} and not intel:
        return _cpu(
            "an NVIDIA GPU is present but its driver is not installed",
            gpu=nvidia_without_driver[0],
            blocked="This machine has an NVIDIA GPU, but nvidia-smi is missing, so the driver is not installed.",
            fix="Install the NVIDIA driver from %s (on Linux: your distribution's nvidia-driver package), "
            "reboot, then run this again." % NVIDIA_DRIVERS_URL,
        )

    # --- Intel ---------------------------------------------------------------
    if intel and prefer in {"auto", "xpu"}:
        supported = [gpu for gpu in intel if _INTEL_XPU.search(gpu.name)]
        experimental = [gpu for gpu in intel if _INTEL_EXPERIMENTAL.search(gpu.name)]
        if supported or prefer == "xpu":
            return _intel_profile(supported or intel)
        if experimental:
            profile = _intel_profile(experimental)
            profile.warnings.append(
                "%s is not on Intel's supported list for PyTorch XPU. It is installed and verified; if the "
                "runtime does not start, the machine joins without a GPU." % experimental[0].name
            )
            return profile

    if prefer == "cuda":
        return _cpu("CUDA was requested but no NVIDIA GPU with a driver was found")
    if prefer == "xpu":
        return _cpu("XPU was requested but no Intel GPU was found")
    if prefer == "mps":
        return _cpu("MPS is only available on Apple Silicon Macs")

    amd = [gpu for gpu in facts.gpus if gpu.vendor == "amd"]
    if amd:
        return _cpu(
            "%s is an AMD GPU, which GradMesh 5 does not train on yet" % amd[0].name,
            gpu=amd[0],
        )
    return _cpu("no supported GPU was detected")


def _intel_profile(gpus: List[Gpu]) -> Profile:
    # Prefer a discrete card over an integrated one when both are present.
    def rank(gpu: Gpu) -> int:
        name = gpu.name.lower()
        if re.search(r"arc.*\b[ab]\d{3}", name) or "dg2" in name or "battlemage" in name:
            return 2
        return 1 if _INTEL_XPU.search(name) else 0

    best = max(gpus, key=rank)
    return Profile(
        name="xpu",
        backend="xpu",
        requirements="requirements-xpu.txt",
        label="PyTorch Intel XPU build",
        reason="%s trains on the Intel XPU backend" % best.name,
        torch=REFERENCE_STACK["torch"],
        gpu_index=0,
        gpu_name=best.name,
        fix="If the XPU runtime does not start, install the latest Intel graphics driver from %s "
        "(on Linux: Intel's compute runtime packages, intel-opencl-icd and level-zero)." % INTEL_DRIVERS_URL,
    )


def _fit_os(profile: Profile, facts: HostFacts) -> Profile:
    """PyTorch's CPU index has no +cpu wheels for macOS; the stock wheel is the CPU build there."""
    if facts.os == "macos" and profile.requirements == "requirements-train-cpu.txt":
        profile.requirements = "requirements-mps.txt"
        profile.label = "PyTorch macOS build"
    return profile


def select_profile_for(facts: HostFacts, prefer: str = "auto") -> Profile:
    return _fit_os(select_profile(facts, prefer), facts)


def host_profile(facts: HostFacts, prefer: str = "auto") -> Profile:
    """The build for a coordinator host.

    The host needs PyTorch even when its own GPU cannot train, because it
    aggregates every round. So a blocked GPU profile becomes the CPU build with
    the reason kept as a warning, rather than stopping the whole mesh from
    starting over a driver.
    """
    profile = select_profile(facts, prefer)
    if profile.blocked:
        if facts.os == "macos" and facts.arch != "arm64":
            # Nothing installs on an Intel Mac, not even for aggregation.
            return profile
        fallback = _cpu(
            "the GPU cannot be used yet, so the host installs the CPU build for aggregation",
            warnings=profile.warnings + [profile.blocked + (" " + profile.fix if profile.fix else "")],
        )
        return _fit_os(fallback, facts)
    return _fit_os(profile, facts)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Detect this machine's GPU and the PyTorch build it needs")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "xpu", "mps", "cpu"])
    parser.add_argument("--host", action="store_true", help="Select for a coordinator host")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    facts = detect()
    profile = host_profile(facts, args.backend) if args.host else select_profile_for(facts, args.backend)
    if args.json:
        print(json.dumps({"facts": facts.as_dict(), "profile": profile.as_dict()}, indent=2))
        return 0

    print("Machine   %s %s (%s), Python %s" % (facts.os, facts.os_version, facts.arch, facts.python))
    for gpu in facts.gpus:
        extra = []
        if gpu.compute_capability:
            extra.append("compute %.1f" % gpu.compute_capability)
        if gpu.driver:
            extra.append("driver %s" % gpu.driver)
        print("GPU %d     %s [%s]%s" % (gpu.index, gpu.name, gpu.vendor, " " + ", ".join(extra) if extra else ""))
    if not facts.gpus:
        print("GPU       none detected")
    print("Build     %s (%s), torch %s" % (profile.label, profile.requirements, profile.torch))
    print("Why       %s" % profile.reason)
    for warning in profile.warnings:
        print("Note      %s" % warning)
    if profile.blocked:
        print("Blocked   %s" % profile.blocked)
        if profile.fix:
            print("Fix       %s" % profile.fix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
