"""What software each machine trains with, and whether it matches the reference.

A GradMesh 5 result compares NVIDIA, Intel and Apple GPUs. That comparison is
only fair if every machine ran the same PyTorch, torchvision and Ultralytics,
so the dashboard shows each machine's stack next to its GPU and flags any
machine that drifted from the pinned reference. Pure functions, no torch.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _base(version: Optional[str]) -> str:
    """'2.13.0+cu130' -> '2.13.0'. The local suffix names the build, not the release."""
    return str(version or "").split("+")[0].strip()


def _build(version: Optional[str], backend: str) -> str:
    """The wheel flavour: the local suffix where there is one, else what the backend implies."""
    text = str(version or "")
    if "+" in text:
        return text.split("+", 1)[1]
    return {"mps": "macOS", "cpu": "cpu"}.get(backend, "")


def software_view(
    capability: Optional[dict],
    diagnostics: Optional[dict],
    agent_version: Optional[str],
    reference: Dict[str, str],
    host_version: str,
) -> dict:
    capability = capability or {}
    diagnostics = diagnostics or {}
    backend = str(capability.get("backend") or "cpu")
    torch_full = capability.get("torch_version") or ""
    installed = {
        "torch": _base(torch_full),
        "torchvision": _base(capability.get("torchvision_version")),
        "ultralytics": _base(capability.get("ultralytics_version")),
    }

    drift: List[str] = []
    known = 0
    for package, version in installed.items():
        wanted = reference.get(package)
        if not version or not wanted:
            continue
        known += 1
        if version != wanted:
            drift.append("%s %s (reference %s)" % (package, version, wanted))

    return {
        "python": capability.get("python_version") or diagnostics.get("python") or "",
        "torch": installed["torch"],
        "torch_build": _build(torch_full, backend),
        "torchvision": installed["torchvision"],
        "ultralytics": installed["ultralytics"],
        "runtime": capability.get("runtime") or "",
        "os": capability.get("os_version") or diagnostics.get("os") or capability.get("platform") or "",
        "driver": diagnostics.get("driver") or "",
        "agent": agent_version or "",
        # A worker older than the host still trains, but misses whatever the
        # host's newer protocol added; worth a re-run of the join command.
        "agent_behind": bool(agent_version) and _base(agent_version) != _base(host_version),
        # None when an older agent did not report enough to tell.
        "on_reference": (not drift) if known else None,
        "drift": drift,
    }
