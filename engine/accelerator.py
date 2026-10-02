"""One interface over every accelerator GradMesh trains on.

GradMesh 5 combines three GPU families in one mesh, and each exposes its device
through a different corner of PyTorch:

    backend  vendor  torch device  memory                         notes
    cuda     NVIDIA  cuda:0        dedicated, torch.cuda          AMP on
    xpu      Intel   xpu:0         dedicated, torch.xpu           AMP off, foreach off
    mps      Apple   mps           unified with system RAM        AMP off, CPU fallback for missing ops
    cpu      any     cpu           system RAM                     measured, never given shards

Everything that differs between them, from synchronising to clearing the cache
to how much memory a batch may use, is answered here. The rest of the worker
asks an `Accelerator` and never branches on the vendor itself.

Which physical GPU a backend sees is decided before torch is imported, through
CUDA_VISIBLE_DEVICES or ZE_AFFINITY_MASK (see `pin_device`), so the device
index used inside the process is always 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Metal still lacks a handful of operators. Falling back to CPU for those is
# slower but correct; without this the first one raises NotImplementedError
# mid-round. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

BACKENDS = ("cuda", "xpu", "mps", "cpu")
TRAINING_BACKENDS = frozenset({"cuda", "xpu", "mps"})
VENDOR = {"cuda": "nvidia", "xpu": "intel", "mps": "apple", "cpu": "cpu"}


def pin_device(backend: str, index: int) -> None:
    """Expose only GPU `index` to this process. Call before importing torch.

    Ultralytics rewrites CUDA_VISIBLE_DEVICES from its device argument, which
    is too late once CUDA has initialised, so a worker told to use the second
    card would silently train on the first. Pinning up front makes the chosen
    card device 0 for everything that follows.
    """
    if index is None or index < 0:
        return
    if backend in {"cuda", "auto"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    if backend in {"xpu", "auto"}:
        os.environ["ZE_AFFINITY_MASK"] = str(index)


@dataclass(frozen=True)
class Accelerator:
    backend: str
    torch_device: torch.device
    device_name: str
    total_memory_mb: int
    ultralytics_device: object
    index: int = 0
    # Apple GPUs share system memory, so "device memory" is a budget the OS
    # grants rather than a physical card, and batch sizing has to treat it so.
    unified_memory: bool = False
    allow_cpu_training: bool = False
    gpu_cores: Optional[int] = None

    @property
    def vendor(self) -> str:
        return VENDOR.get(self.backend, "cpu")

    @property
    def supports_training(self) -> bool:
        if self.backend == "cpu":
            return self.allow_cpu_training
        return self.backend in TRAINING_BACKENDS

    @property
    def supports_amp(self) -> bool:
        # Ultralytics enables AMP on CUDA only. The XPU trainer turns it off
        # explicitly and Ultralytics' own check refuses MPS.
        return self.backend == "cuda"

    def synchronize(self) -> None:
        if self.backend == "cuda":
            torch.cuda.synchronize(self.torch_device)
        elif self.backend == "xpu":
            torch.xpu.synchronize(self.torch_device)
        elif self.backend == "mps":
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        try:
            if self.backend == "cuda":
                torch.cuda.empty_cache()
            elif self.backend == "xpu":
                torch.xpu.empty_cache()
            elif self.backend == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    def memory_in_use_mb(self) -> int:
        try:
            if self.backend == "cuda":
                return int(torch.cuda.memory_reserved(self.torch_device) / 2**20)
            if self.backend == "xpu":
                return int(torch.xpu.memory_reserved(self.torch_device) / 2**20)
            if self.backend == "mps":
                return int(torch.mps.driver_allocated_memory() / 2**20)
        except Exception:
            return 0
        return 0

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "vendor": self.vendor,
            "device": str(self.torch_device),
            "device_name": self.device_name,
            "total_memory_mb": self.total_memory_mb,
            "unified_memory": self.unified_memory,
            "supports_training": self.supports_training,
            "supports_amp": self.supports_amp,
        }


def _system_memory_mb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total / 2**20)
    except Exception:
        return 8192


def _cuda_accelerator() -> Accelerator:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Either this PyTorch build is not a "
            "CUDA build, or the NVIDIA driver is missing or older than the build needs."
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    return Accelerator(
        backend="cuda",
        torch_device=device,
        device_name=torch.cuda.get_device_name(0),
        total_memory_mb=int(properties.total_memory / 2**20),
        # The v3 and v4 Ultralytics argument, unchanged.
        ultralytics_device="cuda",
    )


def _xpu_accelerator() -> Accelerator:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "XPU was requested but torch.xpu.is_available() is False. Install or update the Intel graphics "
            "driver, and check that this is the +xpu PyTorch build."
        )
    if torch.xpu.device_count() < 1:
        raise RuntimeError("XPU was requested but no XPU devices were found")
    device = torch.device("xpu:0")
    properties = torch.xpu.get_device_properties(0)
    return Accelerator(
        backend="xpu",
        torch_device=device,
        device_name=torch.xpu.get_device_name(0),
        total_memory_mb=int(properties.total_memory / 2**20),
        # Ultralytics 8.4.46 rejects XPU strings but accepts torch.device.
        ultralytics_device=device,
    )


def _mps_accelerator() -> Accelerator:
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        built = bool(mps is not None and mps.is_built())
        raise RuntimeError(
            "MPS was requested but torch.backends.mps.is_available() is False. "
            + (
                "This PyTorch has Metal support, so the Mac is likely on macOS older than 14."
                if built
                else "This PyTorch build has no Metal support."
            )
        )
    # The Metal driver reports how much of unified memory it will let one
    # process use. That, not total RAM, is the batch-size budget.
    budget = 0
    try:
        budget = int(torch.mps.recommended_max_memory() / 2**20)
    except Exception:
        budget = 0
    if budget <= 0:
        budget = int(_system_memory_mb() * 0.65)
    chip, cores = _apple_chip()
    return Accelerator(
        backend="mps",
        torch_device=torch.device("mps"),
        # "Apple M1, 8-core GPU": the M1 Air came with 7 or 8 GPU cores, and
        # the paper's hardware table needs to say which.
        device_name="%s, %d-core GPU" % (chip, cores) if cores else chip,
        total_memory_mb=budget,
        ultralytics_device="mps",
        unified_memory=True,
        gpu_cores=cores,
    )


def _apple_chip() -> tuple:
    """('Apple M1', 8): the chip name and its GPU core count, where macOS says."""
    import json
    import subprocess

    chip, cores = "Apple GPU", None
    try:
        chip = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3
            ).stdout.strip()
            or chip
        )
    except Exception:
        pass
    try:
        raw = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, timeout=15
        ).stdout
        for entry in json.loads(raw).get("SPDisplaysDataType", []):
            if "apple" in str(entry.get("sppci_model", "")).lower():
                chip = entry.get("sppci_model") or chip
                cores = int(str(entry.get("sppci_cores") or "0")) or None
                break
    except Exception:
        pass
    return chip, cores


def _cpu_accelerator(advertised_memory_mb: Optional[int], allow_training: bool) -> Accelerator:
    return Accelerator(
        backend="cpu",
        torch_device=torch.device("cpu"),
        device_name="CPU",
        total_memory_mb=int(advertised_memory_mb or _system_memory_mb()),
        ultralytics_device="cpu",
        unified_memory=True,
        allow_cpu_training=allow_training,
    )


def available_backends() -> list:
    found = []
    try:
        if torch.cuda.is_available():
            found.append("cuda")
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            found.append("xpu")
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            found.append("mps")
    except Exception:
        pass
    found.append("cpu")
    return found


def detect_accelerator(
    requested: str = "auto",
    advertised_memory_mb: Optional[int] = None,
    allow_cpu_training: bool = False,
) -> Accelerator:
    """Pick the accelerator, failing rather than falling back for explicit requests.

    `auto` prefers CUDA, then XPU, then MPS. Only one is ever present in a
    given PyTorch build in practice, because the wheels are vendor-specific,
    so the order matters only for unusual hand-built environments.
    """
    backend = (requested or "auto").lower().strip()
    if backend not in {"auto", *BACKENDS}:
        raise ValueError("Unsupported backend %r; choose auto, cuda, xpu, mps or cpu" % requested)
    if backend == "cuda":
        return _cuda_accelerator()
    if backend == "xpu":
        return _xpu_accelerator()
    if backend == "mps":
        return _mps_accelerator()
    if backend == "cpu":
        return _cpu_accelerator(advertised_memory_mb, allow_cpu_training)

    for name, factory in (("cuda", _cuda_accelerator), ("xpu", _xpu_accelerator), ("mps", _mps_accelerator)):
        if name in available_backends():
            return factory()
    return _cpu_accelerator(advertised_memory_mb, allow_cpu_training)
