"""Device capability probe and host diagnostics.

Every worker runs `probe()` once at startup and reports it. The coordinator's
admission gate needs a number that is comparable across NVIDIA, Intel and Apple
devices, and vendor-reported specs are not comparable, so the probe measures
sustained FP32 matmul throughput and device memory bandwidth on the device
itself, with each backend's own synchronisation.

The probe is a ranking signal for a machine that has never trained. Leg 1
showed its limit: two RTX 5070s probed within 4% of each other and then trained
at 29.9 and 14.7 images per second. Nothing about the GPU explains a factor of
two, so `diagnostics()` reports the things that do: whether this machine is
also running the coordinator, whether it is on battery or a power-saving plan,
whether the GPU is throttling, whether other processes hold the GPU, and the
PCIe link it negotiated. The scheduler learns real throughput regardless, but
the dashboard can now say *why* a twin is slow instead of only that it is.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from accelerator import Accelerator


@dataclass(frozen=True)
class Capability:
    backend: str
    vendor: str
    device_name: str
    total_memory_mb: int
    unified_memory: bool
    gflops: float
    mem_bandwidth_gbps: float
    host: str
    platform: str
    torch_version: str
    supports_training: bool
    supports_amp: bool
    compute_capability: Optional[str] = None
    # The software this machine trains with, shown per machine on the
    # dashboard: a cross-vendor result is only comparable if every machine ran
    # the same stack, so drift has to be visible, not just possible.
    python_version: str = ""
    torchvision_version: str = ""
    ultralytics_version: str = ""
    runtime: str = ""  # "CUDA 13.0", "Intel XPU", "Metal"
    os_version: str = ""  # "macOS 14.5", "Windows 11 (build 26200)", "Ubuntu 24.04 LTS"
    gpu_cores: Optional[int] = None  # Apple reports these; NVIDIA and Intel do not here

    def as_dict(self) -> dict:
        return asdict(self)


def _package_version(name: str) -> str:
    """Installed version without importing the package (Ultralytics takes seconds)."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


def friendly_os() -> str:
    """The operating system as a person would name it, not the kernel release."""
    try:
        if sys.platform == "darwin":
            release = platform.mac_ver()[0]
            return "macOS %s" % release if release else "macOS"
        if sys.platform == "win32":
            build = platform.version().split(".")[-1]
            name = "Windows 11" if build.isdigit() and int(build) >= 22000 else "Windows %s" % platform.release()
            return "%s (build %s)" % (name, build) if build.isdigit() else name
        release = getattr(platform, "freedesktop_os_release", None)
        if release:
            pretty = release().get("PRETTY_NAME")
            if pretty:
                return pretty
    except Exception:
        pass
    return "%s %s" % (platform.system(), platform.release())


def _runtime(torch, backend: str) -> str:
    try:
        if backend == "cuda" and getattr(torch.version, "cuda", None):
            return "CUDA %s" % torch.version.cuda
        if backend == "xpu":
            return "Intel XPU"
        if backend == "mps":
            return "Metal"
    except Exception:
        pass
    return "CPU" if backend == "cpu" else backend


def _timed_matmul(torch, accelerator: Accelerator, size: int, iters: int) -> float:
    """Achieved GFLOP/s for a size x size FP32 matmul."""
    device = accelerator.torch_device
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warm up: the first calls pay kernel selection and allocator growth, and
    # on Metal the first dispatch also compiles the shader.
    for _ in range(3):
        a @ b
    accelerator.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    accelerator.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    return 2.0 * (size ** 3) * iters / elapsed / 1e9


def _timed_copy(torch, accelerator: Accelerator, mb: int, iters: int) -> float:
    """Achieved device memory bandwidth in GB/s."""
    device = accelerator.torch_device
    elements = (mb * 1024 * 1024) // 4
    src = torch.empty(elements, device=device, dtype=torch.float32).fill_(1.0)
    dst = torch.empty_like(src)

    dst.copy_(src)
    accelerator.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    accelerator.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    return 2.0 * elements * 4 * iters / elapsed / 1e9


def probe(accelerator: Accelerator) -> Capability:
    import torch

    # CPU gets a smaller problem so the probe stays sub-second there too.
    if accelerator.backend == "cpu":
        size, iters, mb, copies = 512, 6, 64, 6
    else:
        size, iters, mb, copies = 2048, 12, 256, 12

    try:
        gflops = _timed_matmul(torch, accelerator, size, iters)
    except Exception:
        gflops = 0.0
    try:
        bandwidth = _timed_copy(torch, accelerator, mb, copies)
    except Exception:
        bandwidth = 0.0
    accelerator.empty_cache()

    capability = None
    if accelerator.backend == "cuda":
        try:
            capability = "%d.%d" % torch.cuda.get_device_capability(0)
        except Exception:
            capability = None

    return Capability(
        backend=accelerator.backend,
        vendor=accelerator.vendor,
        device_name=accelerator.device_name,
        total_memory_mb=accelerator.total_memory_mb,
        unified_memory=accelerator.unified_memory,
        gflops=round(gflops, 2),
        mem_bandwidth_gbps=round(bandwidth, 2),
        host=socket.gethostname(),
        platform="%s %s" % (platform.system(), platform.release()),
        torch_version=getattr(torch, "__version__", "unknown"),
        supports_training=accelerator.supports_training,
        supports_amp=accelerator.supports_amp,
        compute_capability=capability,
        python_version=platform.python_version(),
        torchvision_version=_package_version("torchvision"),
        ultralytics_version=_package_version("ultralytics"),
        runtime=_runtime(torch, accelerator.backend),
        os_version=friendly_os(),
        gpu_cores=accelerator.gpu_cores,
    )


def diagnose(accelerator: Accelerator) -> Optional[str]:
    """Explain why this device cannot run kernels, if it cannot. None when fine.

    `is_available()` is not enough on any backend. A Blackwell card on a CUDA
    12.6 wheel, an Arc GPU whose Level Zero runtime is missing, and an MPS
    device on an old macOS all report themselves present and then fail the
    first real launch. So this launches one.
    """
    import torch

    if accelerator.backend == "cpu":
        return None

    if accelerator.backend == "cuda":
        try:
            major, minor = torch.cuda.get_device_capability(0)
            target = "sm_%d%d" % (major, minor)
            arches = list(torch.cuda.get_arch_list())
            has_sass = any(arch == target for arch in arches)
            # A build carrying PTX at or below this device can JIT-compile for
            # it. The test launch below is the real arbiter either way.
            ptx = [
                int(a.split("_", 1)[1])
                for a in arches
                if a.startswith("compute_") and a.split("_", 1)[1].isdigit()
            ]
            has_ptx = any(value <= major * 10 + minor for value in ptx)
            if not has_sass and not has_ptx:
                return (
                    "This PyTorch build has no kernels for %s (%s). It ships %s. Reinstall with the matching "
                    "build: run `npm run setup` on this machine, or rerun the join command from the host's "
                    "Invite a GPU page; both pick the build from the compute capability and driver."
                    % (torch.cuda.get_device_name(0), target, ", ".join(arches) or "no architectures")
                )
        except Exception:
            pass

    try:
        x = torch.arange(256, device=accelerator.torch_device, dtype=torch.float32)
        value = float((x * 2.0).sum().item())
        accelerator.synchronize()
        if abs(value - 65280.0) > 1e-2:
            return "A test kernel on %s returned a wrong result (%s)." % (accelerator.device_name, value)
    except Exception as exc:
        hints = {
            "cuda": "Update the NVIDIA driver, then rerun setup so the matching PyTorch build is installed.",
            "xpu": "Install or update the Intel graphics driver (on Linux: Intel's compute runtime packages).",
            "mps": "Update macOS to 14 or newer.",
        }
        return "A test kernel failed on %s: %s: %s. %s" % (
            accelerator.device_name,
            type(exc).__name__,
            exc,
            hints.get(accelerator.backend, ""),
        )
    return None


def probe_or_none(accelerator: Optional[Accelerator]) -> Optional[dict]:
    if accelerator is None:
        return None
    try:
        return probe(accelerator).as_dict()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Host diagnostics
# ---------------------------------------------------------------------------


def _run(command: List[str], timeout: float = 4.0) -> Optional[str]:
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


_cpu_name_cache: Optional[str] = None


def cpu_name() -> str:
    global _cpu_name_cache
    if _cpu_name_cache:
        return _cpu_name_cache
    name = ""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
        elif sys.platform == "darwin":
            name = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
        else:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as stream:
                for line in stream:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
    except Exception:
        name = ""
    _cpu_name_cache = re.sub(r"\s+", " ", name).strip() or platform.processor() or "unknown CPU"
    return _cpu_name_cache


def _power_plan() -> Optional[str]:
    if sys.platform != "win32":
        return None
    raw = _run(["powercfg", "/getactivescheme"])
    if not raw:
        return None
    match = re.search(r"\(([^)]+)\)", raw)
    return match.group(1).strip() if match else None


# nvidia-smi clocks_event_reasons bits that mean the GPU is being held back.
_THROTTLE_BITS = {
    0x4: "software power cap",
    0x8: "hardware slowdown",
    0x20: "software thermal slowdown",
    0x40: "hardware thermal slowdown",
    0x80: "power brake",
}


def _nvidia_diagnostics(index: int = 0) -> Dict[str, object]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    fields = [
        "utilization.gpu",
        "temperature.gpu",
        "clocks.sm",
        "clocks.max.sm",
        "power.draw",
        "power.limit",
        "pcie.link.gen.current",
        "pcie.link.gen.max",
        "pcie.link.width.current",
        "pcie.link.width.max",
        "clocks_event_reasons.active",
        "driver_version",
    ]
    raw = _run([smi, "-i", str(index), "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    if not raw:
        return {}
    values = [value.strip() for value in raw.strip().splitlines()[0].split(",")]
    info: Dict[str, object] = {}
    for field, value in zip(fields, values):
        if value in {"[N/A]", "N/A", "[Not Supported]", ""}:
            continue
        info[field] = value
    reasons: List[str] = []
    try:
        mask = int(str(info.get("clocks_event_reasons.active", "0")), 16)
        reasons = [label for bit, label in _THROTTLE_BITS.items() if mask & bit]
    except Exception:
        pass

    others = 0
    apps = _run([smi, "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader"])
    if apps:
        own = os.getpid()
        others = sum(1 for line in apps.splitlines() if line.strip().isdigit() and int(line.strip()) != own)

    def number(key):
        try:
            return float(str(info.get(key)))
        except Exception:
            return None

    return {
        "gpu_utilization": number("utilization.gpu"),
        "gpu_temperature_c": number("temperature.gpu"),
        "sm_clock_mhz": number("clocks.sm"),
        "sm_clock_max_mhz": number("clocks.max.sm"),
        "power_draw_w": number("power.draw"),
        "power_limit_w": number("power.limit"),
        "pcie_gen": number("pcie.link.gen.current"),
        "pcie_gen_max": number("pcie.link.gen.max"),
        "pcie_width": number("pcie.link.width.current"),
        "pcie_width_max": number("pcie.link.width.max"),
        "throttle_reasons": reasons,
        "other_gpu_processes": others,
        "driver": info.get("driver_version"),
    }


# NSProcessInfo.thermalState: what macOS itself says about heat. A fanless
# MacBook Air reaches "serious" after a few minutes of sustained GPU load and
# slows the GPU to cool down, which is exactly the slowdown a person should see
# explained on the dashboard.
_THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
_process_info = None


def _macos_process_info():
    """(thermal state, Low Power Mode) from Foundation, through the Objective-C runtime.

    ctypes rather than PyObjC, which the agent does not install. Each message
    gets its own typed function pointer: objc_msgSend must be called with the
    exact signature of the method on arm64.
    """
    global _process_info
    if _process_info is None:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "/usr/lib/libobjc.dylib")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        send_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(address)
        send_bool = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(address)
        info = send_id(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        if not info:
            raise RuntimeError("NSProcessInfo unavailable")
        _process_info = (info, objc, send_long, send_bool)
    info, objc, send_long, send_bool = _process_info
    thermal = _THERMAL_STATES.get(int(send_long(info, objc.sel_registerName(b"thermalState"))))
    low_power = bool(send_bool(info, objc.sel_registerName(b"isLowPowerModeEnabled")))
    return thermal, low_power


def _macos_diagnostics() -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        thermal, low_power = _macos_process_info()
        if thermal:
            info["thermal_state"] = thermal
        info["low_power_mode"] = low_power
    except Exception:
        # pmset reports Low Power Mode on macOS 12 and newer.
        raw = _run(["pmset", "-g"]) or ""
        match = re.search(r"lowpowermode\s+(\d)", raw)
        if match:
            info["low_power_mode"] = match.group(1) == "1"
    return info


def diagnostics(accelerator: Optional[Accelerator] = None, gpu_index: int = 0, sample_cpu: bool = True) -> dict:
    """Everything about this host that can make an identical GPU train slower.

    Cheap enough to send with a heartbeat every half minute: one nvidia-smi
    call, one psutil sample, and on Windows one powercfg call.
    """
    info: Dict[str, object] = {
        "cpu": cpu_name(),
        "cpu_logical": os.cpu_count(),
        "os": "%s %s" % (platform.system(), platform.release()),
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["ram_mb"] = int(psutil.virtual_memory().total / 2**20)
        info["ram_available_mb"] = int(psutil.virtual_memory().available / 2**20)
        info["swap_used_mb"] = int(psutil.swap_memory().used / 2**20)
        if sample_cpu:
            info["cpu_load"] = psutil.cpu_percent(interval=0.2)
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if battery is not None:
            info["on_battery"] = not bool(battery.power_plugged)
            info["battery_percent"] = round(float(battery.percent), 1)
    except Exception:
        pass

    plan = _power_plan()
    if plan:
        info["power_plan"] = plan
    if sys.platform == "darwin":
        info.update(_macos_diagnostics())
    try:
        # The image cache and the PyTorch install live under the home folder;
        # a 256 GB laptop can run short.
        info["disk_free_mb"] = int(shutil.disk_usage(os.path.expanduser("~")).free / 2**20)
    except Exception:
        pass

    if accelerator is not None and accelerator.backend == "cuda":
        info.update(_nvidia_diagnostics(gpu_index))
    return info


if __name__ == "__main__":
    from accelerator import detect_accelerator

    accelerator = detect_accelerator("auto")
    print("diagnosis           %s" % (diagnose(accelerator) or "ok"))
    for key, value in probe(accelerator).as_dict().items():
        print("%-20s%s" % (key, value))
    for key, value in diagnostics(accelerator).items():
        print("%-20s%s" % (key, value))
