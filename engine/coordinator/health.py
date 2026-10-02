"""Why a machine may train slower than its GPU suggests, in plain language.

Leg 1's two RTX 5070s probed within 4% of each other and trained at 29.9 and
14.7 images per second. The scheduler learns that difference whatever causes
it, but a person looking at the dashboard should be told the likely cause, not
just the number. Workers report host diagnostics (see probe.diagnostics); this
turns them into warnings. Pure functions, no torch, so the coordinator can run
them before the training plane is installed.
"""

from __future__ import annotations

from typing import List, Optional


def health_warnings(diag: Optional[dict], co_located: bool = False) -> List[str]:
    diag = diag or {}
    warnings: List[str] = []
    if co_located:
        warnings.append(
            "also runs the coordinator, so serving data, aggregating and evaluating compete with its training"
        )
    if diag.get("on_battery"):
        warnings.append("on battery power, which caps CPU and GPU clocks on most laptops")
    plan = str(diag.get("power_plan") or "")
    if plan and any(word in plan.lower() for word in ("saver", "balanced", "eco")):
        warnings.append("Windows power plan is %s; High performance avoids clock down-shifts" % plan)
    reasons = diag.get("throttle_reasons") or []
    if reasons:
        warnings.append("GPU is throttling: %s" % ", ".join(str(reason) for reason in reasons))
    temperature = diag.get("gpu_temperature_c")
    if isinstance(temperature, (int, float)) and temperature >= 87:
        warnings.append("GPU is at %d C" % int(temperature))
    others = diag.get("other_gpu_processes") or 0
    if others:
        warnings.append("%d other process%s using this GPU" % (others, "" if others == 1 else "es"))
    width, width_max = diag.get("pcie_width"), diag.get("pcie_width_max")
    if width and width_max and width < width_max and (diag.get("gpu_utilization") or 0) > 20:
        warnings.append("PCIe link running at x%d of x%d" % (int(width), int(width_max)))
    load = diag.get("cpu_load")
    if isinstance(load, (int, float)) and load >= 85:
        warnings.append("CPU is %d%% busy, and YOLO data loading runs on the CPU" % int(load))
    available = diag.get("ram_available_mb")
    if isinstance(available, (int, float)) and available < 1500:
        warnings.append("only %d MB of system memory free" % int(available))

    # macOS. A fanless MacBook Air has no fan to spin up: when it gets hot it
    # lowers the GPU clock, and macOS says so through its thermal state.
    thermal = str(diag.get("thermal_state") or "")
    if thermal in {"serious", "critical"}:
        warnings.append(
            "macOS reports a %s thermal state, so it is slowing the chip to cool down; give the Mac airflow "
            "(lid open, on a hard surface, out of the sun)" % thermal
        )
    if diag.get("low_power_mode"):
        warnings.append("Low Power Mode is on, which caps CPU and GPU clocks; turn it off in System Settings > Battery")
    swap = diag.get("swap_used_mb")
    ram = diag.get("ram_mb")
    if isinstance(swap, (int, float)) and isinstance(ram, (int, float)) and ram <= 8704 and swap >= 2048:
        warnings.append(
            "%.1f GB of swap in use on a %d GB machine: memory is short and training may be paging to the SSD"
            % (swap / 1024.0, round(ram / 1024.0))
        )
    disk = diag.get("disk_free_mb")
    if isinstance(disk, (int, float)) and disk < 5120:
        warnings.append("only %.1f GB free on disk; the image cache and PyTorch need room" % (disk / 1024.0))
    return warnings
