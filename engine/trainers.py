"""One training round on any backend, with the time split into phases.

The Ultralytics call itself is the v3 call, and the Intel XPU path is still the
validated `ultralytics_xpu` trainer. What this module adds is around it.

**Per-backend adapters.** CUDA and Apple MPS go through Ultralytics' stock
trainer with the device string it expects (`cuda`, `mps`); Intel XPU goes
through the XPU trainer, which needs a `torch.device` and AMP off. Every
backend receives the same optimiser, learning rate, warmup and seed from the
run, so an NVIDIA, an Intel and an Apple machine in one mesh are running the
same recipe. v4 trained XPU with Adam and CUDA with Ultralytics' automatic
choice, which made any cross-vendor comparison a comparison of optimisers.

**Phase timing.** A round is reported as setup (trainer construction, data
loaders, optimiser), epoch (the training loop proper) and post (checkpoint
saving and reload). The scheduler models a machine as a fixed cost per round
plus a per-image rate, and these are the measurements behind both terms.

**Worker-side validation is optional.** With one epoch per round, Ultralytics
validates on the final epoch even with `val=False`, then validates the saved
checkpoint again in `final_eval`: two passes over the validation split, per
worker, per round, whose results nobody uses because the coordinator scores
the aggregated model itself on a fixed held-out split. Skipping them is the
largest single cut to the per-round fixed cost. A run can turn them back on.

**The AMP check no longer downloads a model.** Ultralytics' `check_amp` loads
`yolo26n.pt` from GitHub into the working directory and runs inference twice,
on every round, which is both a fixed cost and an internet dependency on a
machine that is supposed to need only the LAN. It is replaced by the same GPU
blocklist plus a numeric autocast check, cached per process. Set
GRADMESH_ULTRALYTICS_AMP_CHECK=1 to use the original.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, Optional

import torch

from accelerator import Accelerator

# GPUs Ultralytics already knows have broken FP16. Same list, same behaviour.
_AMP_BLOCKLIST = re.compile(
    r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
)
_amp_cache: Dict[str, bool] = {}


def _gradmesh_check_amp(model) -> bool:
    device = next(model.parameters()).device
    if device.type != "cuda":
        return False
    key = str(device)
    if key in _amp_cache:
        return _amp_cache[key]
    name = torch.cuda.get_device_name(device)
    if _AMP_BLOCKLIST.search(name):
        result = False
    else:
        try:
            generator = torch.Generator(device=device).manual_seed(0)
            a = torch.randn(256, 256, device=device, generator=generator)
            b = torch.randn(256, 256, device=device, generator=generator)
            reference = a @ b
            with torch.autocast("cuda", dtype=torch.float16):
                half = a @ b
            result = bool(torch.isfinite(half).all()) and bool(
                torch.allclose(reference, half.float(), atol=0.5, rtol=0.05)
            )
        except Exception:
            result = False
    _amp_cache[key] = result
    return result


_patched = False


def install_patches() -> None:
    """Swap in the offline AMP check. Idempotent."""
    global _patched
    if _patched or os.environ.get("GRADMESH_ULTRALYTICS_AMP_CHECK") == "1":
        return
    try:
        import ultralytics.engine.trainer as trainer_module

        trainer_module.check_amp = _gradmesh_check_amp
        _patched = True
    except Exception:
        pass


class _LeanMixin:
    """Skip per-round validation on the worker when the run asks for that."""

    gradmesh_validate = False

    def validate(self):  # type: ignore[override]
        if self.gradmesh_validate:
            return super().validate()
        # Ultralytics uses negative loss as fitness when validation is
        # unavailable; doing the same keeps best.pt bookkeeping consistent.
        loss = getattr(self, "loss", None)
        try:
            fitness = -float(loss.detach().float().sum().cpu()) if loss is not None else 0.0
        except Exception:
            fitness = 0.0
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return {}, fitness

    def final_eval(self):  # type: ignore[override]
        if self.gradmesh_validate:
            return super().final_eval()
        return None


_class_cache: Dict[tuple, type] = {}


def trainer_class(backend: str, task: str, validate: bool):
    """The Ultralytics trainer class for this backend, task and validation mode."""
    key = (backend, task, validate)
    if key in _class_cache:
        return _class_cache[key]

    if backend == "xpu":
        from ultralytics_xpu import trainer_for_task

        base = trainer_for_task(task)
    else:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.models.yolo.obb import OBBTrainer

        bases = {"detect": DetectionTrainer, "obb": OBBTrainer}
        if task not in bases:
            raise NotImplementedError("GradMesh trains detect and obb tasks; this model is %r" % task)
        base = bases[task]

    cls = type("GradMesh%s%sTrainer" % (backend.upper(), task.title()), (_LeanMixin, base), {"gradmesh_validate": validate})
    _class_cache[key] = cls
    return cls


def default_dataloader_workers(accelerator: Accelerator) -> int:
    """How many dataloader processes this machine should use.

    v3 and v4 used zero everywhere, which puts JPEG decoding and mosaic
    augmentation on the training process's own thread. On a fast GPU that, not
    the GPU, sets the pace, and it is one reason two identical cards in
    different machines train at different speeds: the CPUs are not identical.
    Ultralytics forces zero on MPS and CPU itself.
    """
    if accelerator.backend in {"mps", "cpu"}:
        return 0
    cpus = os.cpu_count() or 2
    # Windows spawns workers rather than forking, which costs a second or two
    # each per round, so it gets fewer.
    ceiling = 4 if os.name == "nt" else 8
    return max(0, min(ceiling, cpus // 2 - 1))


def train_round(
    model,
    accelerator: Accelerator,
    options: Dict[str, Any],
    *,
    validate: bool = False,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> Dict[str, float]:
    """Train `model` in place for one round. Returns phase timings in seconds.

    `options` are Ultralytics train arguments shared by every backend:
    data, epochs, imgsz, batch, project, name, workers, seed, optimizer, lr0,
    warmup_epochs, deterministic. Device-specific arguments are added here.
    """
    install_patches()
    if accelerator.backend == "xpu":
        # Ultralytics' final validation reloads the checkpoint through a path
        # that rejects XPU devices (see ultralytics_xpu.final_eval), so XPU
        # workers never validate locally. The coordinator's scoring is
        # unaffected; it is the number every run reports anyway.
        validate = False
    marks: Dict[str, float] = {"call": time.perf_counter()}
    state = {"batch": 0, "batches": 0}

    def mark(name: str):
        def callback(trainer) -> None:
            marks[name] = time.perf_counter()
            if name == "epoch_start":
                state["batches"] = len(getattr(trainer, "train_loader", []) or [])
        return callback

    def batch_end(trainer) -> None:
        state["batch"] += 1
        if on_progress and state["batches"]:
            # Throttled by the caller; cheap to call per batch.
            on_progress({"batch": state["batch"], "batches": state["batches"]})

    model.add_callback("on_pretrain_routine_end", mark("setup_end"))
    model.add_callback("on_train_epoch_start", mark("epoch_start"))
    model.add_callback("on_train_epoch_end", mark("epoch_end"))
    model.add_callback("on_train_batch_end", batch_end)

    arguments = dict(options)
    arguments.setdefault("verbose", False)
    arguments.setdefault("plots", False)
    arguments["val"] = bool(validate)
    arguments["trainer"] = trainer_class(accelerator.backend, model.task, validate)

    if accelerator.backend == "xpu":
        # The same policy as ultralytics_xpu.xpu_train, which v3 validated: a
        # real torch.device, AMP off, and the XPU trainer, here wrapped in the
        # lean subclass so validation follows the run's setting.
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU was requested but torch.xpu.is_available() is False")
        arguments["device"] = torch.device("xpu:0")
        arguments["amp"] = False
        model.train(**arguments)
    else:
        model.train(device=accelerator.ultralytics_device, **arguments)

    finished = time.perf_counter()
    setup_end = marks.get("setup_end", marks["call"])
    epoch_start = marks.get("epoch_start", setup_end)
    epoch_end = marks.get("epoch_end", finished)
    return {
        "train_seconds": round(finished - marks["call"], 3),
        "setup_seconds": round(max(0.0, epoch_start - marks["call"]), 3),
        "epoch_seconds": round(max(0.0, epoch_end - epoch_start), 3),
        "post_seconds": round(max(0.0, finished - epoch_end), 3),
        "batches": state["batches"],
    }
