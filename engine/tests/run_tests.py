"""GradMesh engine tests. Standard library only, so they run before PyTorch exists.

    python engine/tests/run_tests.py          (or: npm test)

Each test is a plain function; a failed assertion prints its name and the
values involved. The hardware tests fabricate machines rather than reading
this one, so every rule is checked on every machine.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE))

# The store resolves its state directory at import, so point it somewhere
# disposable before anything imports it.
_STATE = tempfile.mkdtemp(prefix="gradmesh-test-state-")
os.environ["GRADMESH_STATE_DIR"] = _STATE

import hardware  # noqa: E402
from hardware import Gpu, HostFacts  # noqa: E402
from version import REFERENCE_STACK  # noqa: E402

failures = []
passed = 0


def test(function):
    global passed
    try:
        function()
        passed += 1
        print("  ok   %s" % function.__name__.replace("_", " "))
    except Exception as exc:  # noqa: BLE001
        failures.append(function.__name__)
        print("  FAIL %s: %s" % (function.__name__.replace("_", " "), exc))
        if not isinstance(exc, AssertionError):
            traceback.print_exc()
    return function


def windows(*gpus):
    return HostFacts(os="windows", os_version="11", arch="x86_64", python="3.12.10", hostname="pc", cpu_count=8, gpus=list(gpus))


def linux(*gpus):
    return HostFacts(os="linux", os_version="6.8", arch="x86_64", python="3.12.3", hostname="box", cpu_count=8, gpus=list(gpus))


def mac(version="14.5", arch="arm64"):
    return HostFacts(
        os="macos", os_version=version, arch=arch, python="3.12.4", hostname="mac", cpu_count=10,
        gpus=[Gpu(vendor="apple" if arch == "arm64" else "intel", name="Apple M3 Pro" if arch == "arm64" else "Intel Iris Plus")],
    )


def nvidia(name, cc, driver, index=0):
    return Gpu(vendor="nvidia", name=name, index=index, compute_capability=cc, driver=driver, memory_mb=8192)


# ---------------------------------------------------------------------------
# Build selection
# ---------------------------------------------------------------------------


@test
def an_rtx_5090_on_a_current_driver_gets_cuda_13():
    profile = hardware.select_profile(windows(nvidia("NVIDIA GeForce RTX 5090", 12.0, "581.80")))
    assert profile.name == "cuda-cu130" and profile.reference, profile


@test
def blackwell_on_an_r570_driver_falls_back_to_cuda_12_8_and_says_so():
    profile = hardware.select_profile(windows(nvidia("NVIDIA GeForce RTX 5070", 12.0, "572.16")))
    assert profile.name == "cuda-cu128" and not profile.reference, profile
    assert any("reference" in warning for warning in profile.warnings), profile.warnings


@test
def blackwell_on_an_old_driver_is_blocked_with_a_fix():
    profile = hardware.select_profile(windows(nvidia("NVIDIA GeForce RTX 5070", 12.0, "560.94")))
    assert profile.blocked and profile.fix and "580" in profile.fix, profile


@test
def blackwell_never_gets_the_cuda_12_6_build():
    for driver in ("590.10", "581.80", "575.00", "570.86"):
        profile = hardware.select_profile(linux(nvidia("NVIDIA RTX PRO 6000", 12.0, driver)))
        assert profile.name != "cuda-cu126", (driver, profile.name)


@test
def ampere_follows_the_driver_between_cuda_13_and_12_6():
    assert hardware.select_profile(windows(nvidia("RTX 3050", 8.6, "581.80"))).name == "cuda-cu130"
    assert hardware.select_profile(windows(nvidia("RTX 3050", 8.6, "552.22"))).name == "cuda-cu126"


@test
def pascal_stays_on_cuda_12_6_even_on_a_new_driver():
    profile = hardware.select_profile(linux(nvidia("GeForce GTX 1080", 6.1, "580.65.06")))
    assert profile.name == "cuda-cu126", profile.name


@test
def a_card_older_than_maxwell_gets_cpu():
    profile = hardware.select_profile(linux(nvidia("Tesla K80", 3.7, "470.10")))
    assert profile.backend == "cpu" and not profile.can_train, profile


@test
def an_nvidia_card_without_a_driver_is_blocked():
    facts = windows(Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 4060", driver=None))
    profile = hardware.select_profile(facts)
    assert profile.blocked and "driver" in profile.blocked.lower(), profile


@test
def the_strongest_of_several_nvidia_cards_decides():
    facts = windows(nvidia("GTX 1060", 6.1, "581.80", 0), nvidia("RTX 4090", 8.9, "581.80", 1))
    profile = hardware.select_profile(facts)
    assert profile.gpu_index == 1 and profile.name == "cuda-cu130" and profile.warnings, profile


@test
def intel_arc_discrete_gets_xpu():
    profile = hardware.select_profile(windows(Gpu(vendor="intel", name="Intel(R) Arc(TM) A770 Graphics")))
    assert profile.backend == "xpu" and profile.can_train, profile


@test
def core_ultra_arc_graphics_gets_xpu():
    profile = hardware.select_profile(windows(Gpu(vendor="intel", name="Intel(R) Arc(TM) Graphics")))
    assert profile.backend == "xpu", profile


@test
def linux_battlemage_gets_xpu():
    profile = hardware.select_profile(linux(Gpu(vendor="intel", name="Intel Corporation Battlemage G21 [Arc B580]")))
    assert profile.backend == "xpu", profile


@test
def iris_xe_is_attempted_with_a_warning():
    profile = hardware.select_profile(windows(Gpu(vendor="intel", name="Intel(R) Iris(R) Xe Graphics")))
    assert profile.backend == "xpu" and profile.warnings, profile


@test
def a_plain_intel_uhd_igpu_gets_cpu():
    profile = hardware.select_profile(windows(Gpu(vendor="intel", name="Intel(R) UHD Graphics 630")))
    assert profile.backend == "cpu", profile


@test
def nvidia_beats_an_intel_igpu_unless_xpu_is_asked_for():
    facts = windows(nvidia("RTX 4070 Laptop GPU", 8.9, "581.80"), Gpu(vendor="intel", name="Intel(R) Arc(TM) Graphics"))
    assert hardware.select_profile(facts).backend == "cuda"
    assert hardware.select_profile(facts, "xpu").backend == "xpu"


@test
def apple_silicon_on_sonoma_gets_mps():
    profile = hardware.select_profile(mac("14.5"))
    assert profile.backend == "mps" and not profile.blocked and profile.requirements == "requirements-mps.txt", profile


@test
def apple_silicon_on_ventura_is_blocked():
    profile = hardware.select_profile(mac("13.6.7"))
    assert profile.blocked and "14" in profile.blocked, profile


@test
def an_intel_mac_is_blocked():
    profile = hardware.select_profile(mac("14.5", arch="x86_64"))
    assert profile.blocked and profile.backend == "cpu", profile


@test
def an_intel_python_under_rosetta_on_an_m1_is_told_to_use_a_native_python():
    facts = mac("14.5", arch="x86_64")
    facts.translated = True
    facts.gpus = [Gpu(vendor="apple", name="Apple M1, 8-core GPU")]
    profile = hardware.select_profile(facts)
    assert profile.blocked and "Rosetta" in profile.blocked, profile
    assert "/opt/homebrew" in (profile.fix or ""), profile.fix
    assert "Intel Macs" not in profile.reason, profile.reason


@test
def an_8gb_m1_air_gets_the_mps_build_with_a_unified_memory_note():
    facts = mac("15.2")
    facts.ram_mb = 8192
    facts.gpus = [Gpu(vendor="apple", name="Apple M1, 8-core GPU", memory_mb=8192, cores=8)]
    profile = hardware.select_profile(facts)
    assert profile.name == "mps" and not profile.blocked and profile.reference, profile
    assert profile.gpu_name == "Apple M1, 8-core GPU", profile.gpu_name
    assert any("unified memory" in warning for warning in profile.warnings), profile.warnings


@test
def a_16gb_mac_gets_no_memory_note():
    facts = mac("14.5")
    facts.ram_mb = 16384
    assert not hardware.select_profile(facts).warnings


@test
def apple_gpu_names_carry_the_core_count_when_macos_reports_it():
    assert hardware.apple_gpu_name("Apple M1", 7) == "Apple M1, 7-core GPU"
    assert hardware.apple_gpu_name("Apple M1", None) == "Apple M1"


@test
def system_profiler_output_for_an_m1_becomes_a_named_unified_memory_gpu():
    sample = json.dumps({"SPDisplaysDataType": [{
        "_name": "Apple M1", "sppci_model": "Apple M1", "sppci_cores": "7",
        "spdisplays_vendor": "sppci_vendor_Apple",
    }]})
    original = hardware._run
    hardware._run = lambda command, timeout=8.0: sample if command[0] == "system_profiler" else None
    try:
        gpus = hardware._apple_gpus(8192)
    finally:
        hardware._run = original
    assert len(gpus) == 1 and gpus[0].vendor == "apple", gpus
    assert gpus[0].name == "Apple M1, 7-core GPU" and gpus[0].cores == 7 and gpus[0].memory_mb == 8192, gpus[0]


@test
def an_8gb_mac_caps_metal_memory_and_a_16gb_mac_does_not():
    import types

    import worker

    keys = ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")
    saved_env = {key: os.environ.pop(key, None) for key in keys}
    saved_platform, saved_run = worker.sys.platform, worker.subprocess.run
    try:
        worker.sys.platform = "darwin"
        worker.subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=str(8 * 2**30), returncode=0)
        assert worker.cap_metal_memory(), "no note for an 8 GB Mac"
        assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.0"
        assert float(os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]) <= 1.0
        for key in keys:
            os.environ.pop(key, None)
        worker.subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=str(16 * 2**30), returncode=0)
        assert worker.cap_metal_memory() is None and keys[0] not in os.environ
        # A value the person set is never overridden.
        os.environ[keys[0]] = "0.0"
        worker.subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=str(8 * 2**30), returncode=0)
        assert worker.cap_metal_memory() is None and os.environ[keys[0]] == "0.0"
    finally:
        worker.sys.platform, worker.subprocess.run = saved_platform, saved_run
        for key, value in saved_env.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


@test
def setup_restarts_itself_as_arm64_when_it_starts_under_rosetta():
    import setup_env

    calls = []
    saved = (setup_env.sys.platform, setup_env.hardware.under_rosetta, setup_env.os.path.exists,
             setup_env._runs_natively, setup_env.os.execv, os.environ.pop("GRADMESH_NATIVE_RELAUNCH", None))
    try:
        setup_env.sys.platform = "darwin"
        setup_env.hardware.under_rosetta = lambda: True
        setup_env.os.path.exists = lambda path: True
        setup_env._runs_natively = lambda python: python == "/opt/homebrew/bin/python3.12"
        setup_env.os.execv = lambda path, argv: calls.append((path, argv))
        setup_env.relaunch_natively_if_translated(["agent", "--server", "http://h:8000", "--token", "t"])
    finally:
        (setup_env.sys.platform, setup_env.hardware.under_rosetta, setup_env.os.path.exists,
         setup_env._runs_natively, setup_env.os.execv) = saved[:5]
        os.environ.pop("GRADMESH_NATIVE_RELAUNCH", None)
        if saved[5] is not None:
            os.environ["GRADMESH_NATIVE_RELAUNCH"] = saved[5]
    assert calls, "did not relaunch"
    path, argv = calls[0]
    assert path == "/usr/bin/arch" and argv[:3] == ["/usr/bin/arch", "-arm64", "/opt/homebrew/bin/python3.12"], argv
    assert argv[3].endswith("setup_env.py") and argv[4:] == ["agent", "--server", "http://h:8000", "--token", "t"], argv


@test
def host_facts_round_trip_with_the_new_fields():
    facts = mac("14.5")
    facts.translated = True
    facts.gpus[0].cores = 8
    again = HostFacts.from_dict(json.loads(json.dumps(facts.as_dict())))
    assert again.translated and again.gpus[0].cores == 8, again


@test
def an_amd_only_machine_gets_cpu_with_a_reason():
    profile = hardware.select_profile(linux(Gpu(vendor="amd", name="AMD Radeon RX 7900 XTX")))
    assert profile.backend == "cpu" and "AMD" in profile.reason, profile


@test
def a_host_with_a_blocked_gpu_still_gets_pytorch_for_aggregation():
    profile = hardware.host_profile(windows(nvidia("RTX 5070", 12.0, "560.94")))
    assert profile.backend == "cpu" and not profile.blocked and profile.warnings, profile


@test
def a_mac_host_without_a_usable_gpu_uses_the_stock_macos_wheel():
    profile = hardware.host_profile(mac("14.5"), prefer="cpu")
    assert profile.requirements == "requirements-mps.txt", profile.requirements


@test
def every_profile_points_at_a_real_requirements_file_on_the_reference_stack():
    files = {
        "requirements-train-cu130.txt": True,
        "requirements-train-cu126.txt": True,
        "requirements-train-cu128.txt": False,
        "requirements-xpu.txt": True,
        "requirements-mps.txt": True,
        "requirements-train-cpu.txt": True,
    }
    for name, reference in files.items():
        text = (ENGINE / name).read_text(encoding="utf-8")
        assert "-r requirements-common.txt" in text, name
        torch = re.search(r"^torch==([0-9.]+)", text, re.M)
        assert torch, name
        if reference:
            assert torch.group(1) == REFERENCE_STACK["torch"], (name, torch.group(1))
    common = (ENGINE / "requirements-common.txt").read_text(encoding="utf-8")
    assert "ultralytics==%s" % REFERENCE_STACK["ultralytics"] in common


@test
def the_join_flow_ships_every_file_the_agent_imports():
    agent_ts = (ENGINE.parent / "lib" / "agent.ts").read_text(encoding="utf-8")
    listed = set(re.findall(r'"([\w.-]+\.(?:py|txt))"', agent_ts))
    needed = {"worker.py", "setup_env.py", "hardware.py", "version.py", "accelerator.py", "probe.py",
              "trainers.py", "federated_training.py", "ultralytics_xpu.py", "requirements-common.txt",
              "requirements-control.txt"}
    for path in ENGINE.glob("requirements-*.txt"):
        needed.add(path.name)
    missing = needed - listed
    assert not missing, missing


# ---------------------------------------------------------------------------
# Machine health and software
# ---------------------------------------------------------------------------


@test
def a_hot_mac_in_low_power_mode_is_explained():
    from coordinator.health import health_warnings

    warnings = health_warnings({"thermal_state": "serious", "low_power_mode": True})
    assert any("thermal" in warning for warning in warnings), warnings
    assert any("Low Power Mode" in warning for warning in warnings), warnings
    assert not health_warnings({"thermal_state": "fair", "low_power_mode": False})


@test
def swap_on_an_8gb_machine_and_a_full_disk_are_flagged():
    from coordinator.health import health_warnings

    warnings = health_warnings({"swap_used_mb": 3072, "ram_mb": 8192, "disk_free_mb": 2048})
    assert any("swap" in warning for warning in warnings), warnings
    assert any("free on disk" in warning for warning in warnings), warnings
    # The same swap on a 32 GB workstation is not worth a warning.
    assert not health_warnings({"swap_used_mb": 3072, "ram_mb": 32768, "disk_free_mb": 200000})


@test
def software_view_marks_machines_on_and_off_the_reference_stack():
    from coordinator.software import software_view

    on = software_view(
        {"backend": "cuda", "torch_version": "2.13.0+cu130", "torchvision_version": "0.28.0+cu130",
         "ultralytics_version": REFERENCE_STACK["ultralytics"], "python_version": "3.12.10", "runtime": "CUDA 13.0",
         "os_version": "Windows 11 (build 26200)"},
        {"driver": "581.80"}, "5.0.0", REFERENCE_STACK, "5.0.0",
    )
    assert on["on_reference"] is True and on["torch_build"] == "cu130" and on["driver"] == "581.80", on
    assert not on["agent_behind"], on

    off = software_view(
        {"backend": "cuda", "torch_version": "2.11.0+cu128", "ultralytics_version": REFERENCE_STACK["ultralytics"]},
        {}, "4.9.0", REFERENCE_STACK, "5.0.0",
    )
    assert off["on_reference"] is False and any("torch 2.11.0" in item for item in off["drift"]), off
    assert off["agent_behind"], off

    mac_view = software_view({"backend": "mps", "torch_version": "2.13.0"}, {}, "5.0.0", REFERENCE_STACK, "5.0.0")
    assert mac_view["torch_build"] == "macOS" and mac_view["on_reference"] is True, mac_view

    unknown = software_view({}, {}, "", REFERENCE_STACK, "5.0.0")
    assert unknown["on_reference"] is None, unknown


# ---------------------------------------------------------------------------
# Benchmark design
# ---------------------------------------------------------------------------


@test
def vendor_mix_sweeps_add_a_single_machine_baseline():
    from coordinator import benchmark

    config = benchmark.SuiteConfig.from_dict(
        {
            "vendor_mixes": [["cuda"], ["cuda", "xpu"], ["xpu", "cuda"], ["cuda", "xpu", "mps"]],
            "dataset_sizes": [1000],
            "repeats": 2,
            "baseline_repeats": 3,
            "strategies": ["proportional", "proportional-linear"],
        }
    )
    trials = benchmark.expand(config, available_nodes=4)
    baselines = [trial for trial in trials if trial.backends is None]
    mixes = {tuple(trial.backends) for trial in trials if trial.backends}
    assert len(baselines) == 3 and all(trial.node_count == 1 for trial in baselines), len(baselines)
    assert mixes == {("cuda",), ("cuda", "xpu"), ("cuda", "mps", "xpu")}, mixes
    assert len(trials) == 3 + 3 * 2 * 2, len(trials)
    assert any("NVIDIA+Intel+Apple" in trial.label() for trial in trials)


@test
def node_count_sweeps_still_alternate_strategy_arms():
    from coordinator import benchmark

    config = benchmark.SuiteConfig.from_dict(
        {"node_counts": [1, 2], "dataset_sizes": [100], "repeats": 2, "baseline_repeats": 2,
         "strategies": ["proportional", "equal"]}
    )
    trials = benchmark.expand(config, available_nodes=2)
    two = [trial.strategy for trial in trials if trial.node_count == 2]
    assert two == ["proportional", "equal", "proportional", "equal"], two


# ---------------------------------------------------------------------------
# Portable state
# ---------------------------------------------------------------------------


@test
def dataset_paths_are_stored_relative_to_the_state_directory():
    from coordinator import store

    folder = store.dataset_dir("abc123") / "data"
    folder.mkdir(parents=True, exist_ok=True)
    store.put_dataset({"id": "abc123", "name": "x", "extracted_path": str(folder), "created_at": 1})
    raw = json.loads(store.STATE_FILE.read_text(encoding="utf-8"))["datasets"]["abc123"]
    assert raw["extracted_path"] == "state:datasets/abc123/data", raw["extracted_path"]
    loaded = store.get_dataset("abc123")
    assert Path(loaded["extracted_path"]) == folder.resolve() and loaded["available"], loaded


@test
def a_v4_absolute_path_from_another_machine_is_remapped():
    from coordinator import store

    folder = store.dataset_dir("old456") / "data"
    folder.mkdir(parents=True, exist_ok=True)

    def mutate(state):
        state["datasets"]["old456"] = {
            "id": "old456",
            "name": "legacy",
            "extracted_path": "C:\\\\Users\\\\someone-else\\\\OneDrive\\\\repo\\\\.gradmesh\\\\datasets\\\\old456\\\\data",
        }

    store.update(mutate)
    loaded = store.get_dataset("old456")
    assert Path(loaded["extracted_path"]) == folder, loaded["extracted_path"]


@test
def learned_machine_statistics_survive_a_restart():
    from coordinator import store

    store.put_node_stats({"n1": {"workloads": {"yolov8n.pt@640": {"rate": 12.5, "fixed": 6.0, "rounds": 3}},
                                 "reliability": 0.9, "last_seen": 9e12}})
    restored = store.node_stats("n1")
    assert restored["workloads"]["yolov8n.pt@640"]["rate"] == 12.5 and restored["reliability"] == 0.9, restored


# ---------------------------------------------------------------------------
# Shards as file lists
# ---------------------------------------------------------------------------


def _toy_dataset(root: Path, count: int = 5) -> None:
    for index in range(count):
        image = root / "images" / "train" / ("img%d.jpg" % index)
        label = root / "labels" / "train" / ("img%d.txt" % index)
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\xff\xd8" + bytes([index]) * (100 + index))
        if index != 2:
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


@test
def shards_are_planned_as_lists_without_copying_images():
    from coordinator import sharding

    root = Path(tempfile.mkdtemp(prefix="gradmesh-ds-"))
    _toy_dataset(root)
    planned = sharding.plan_shard_lists(root, [3, 2], seed=1)
    groups = planned["groups"]
    assert sorted(len(group) for group in groups) == [2, 3], groups
    assert sorted(sum(groups, [])) == ["img%d.jpg" % index for index in range(5)]
    manifest = sharding.shard_manifest(planned["dirs"], groups[0])
    assert all(entry["size"] > 0 for entry in manifest)


@test
def bundles_carry_labels_and_refuse_paths_outside_the_dataset():
    import zipfile
    from coordinator import sharding

    root = Path(tempfile.mkdtemp(prefix="gradmesh-ds-"))
    _toy_dataset(root)
    dirs = sharding.list_split_images(root)["dirs"]
    target = root / "bundle.zip"
    count = sharding.write_bundle(dirs, ["img0.jpg", "img2.jpg"], target)
    names = set(zipfile.ZipFile(target).namelist())
    assert count == 2 and "labels/train/img0.txt" in names and "images/train/img2.jpg" in names, names
    assert "labels/train/img2.txt" not in names  # a background image has no label file
    try:
        sharding.write_bundle(dirs, ["../../../secret.txt"], root / "evil.zip")
    except ValueError:
        return
    raise AssertionError("a path outside the dataset was accepted")


print("")
if failures:
    print("%d of %d engine tests failed" % (len(failures), len(failures) + passed))
    sys.exit(1)
print("engine suite passed (%d tests)" % passed)
