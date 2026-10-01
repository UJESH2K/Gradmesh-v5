"""Durable coordinator state.

Everything that has to survive a coordinator restart lives under .gradmesh/ at
the repository root, or wherever GRADMESH_STATE_DIR points: the dataset
registry, finished run records, the mesh join token, the scheduling policy and,
new in v5, what the mesh has learned about each machine. In-flight round state
stays in memory on purpose, because a half-finished barrier is not resumable
and pretending otherwise would produce silently wrong aggregation.

Two v5 changes, both about the state directory moving between machines:

* **Dataset paths are stored relative to the state directory.** v4 stored
  absolute paths, so a repository opened from OneDrive on a second laptop, or
  under another user name, had a registry full of datasets that pointed at a
  folder that did not exist there. Paths outside the state directory, such as
  a standard dataset Ultralytics downloaded, stay absolute and are reported as
  missing on a machine that does not have them.
* **Learned machine statistics persist.** Throughput, overhead and reliability
  used to reset whenever the coordinator restarted, so every restart cost two
  rounds of bad plans while the estimator relearned machines it already knew.

A JSON file rather than a database is deliberate: a native SQLite build is the
most common way "works on a laptop with nothing preinstalled" breaks. Reads are
cached by modification time, because the dashboard asks several times a second.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ENGINE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ENGINE_DIR.parent
STATE_DIR = Path(os.getenv("GRADMESH_STATE_DIR", str(REPO_ROOT / ".gradmesh"))).resolve()
DATASETS_DIR = STATE_DIR / "datasets"
RUNS_DIR = STATE_DIR / "runs"
MODELS_DIR = STATE_DIR / "models"
STATE_FILE = STATE_DIR / "coordinator.json"

_lock = threading.RLock()
_cache: Dict[str, Any] = {"mtime": None, "state": None}

_DEFAULT_STATE: Dict[str, Any] = {
    "version": 5,
    "created_at": None,
    "mesh_token": None,
    "mesh_id": None,
    "mesh_name": "GradMesh",
    "policy": {},
    "datasets": {},
    "default_dataset_id": None,
    "runs": {},
    "nodes": {},
}

# Keys of a dataset record that hold filesystem paths.
_PATH_KEYS = ("extracted_path", "archive_path", "manifest_path")
_SPLIT_KEYS = ("root", "train_images", "train_labels", "val_images", "val_labels")
_RELATIVE_PREFIX = "state:"


def _ensure_dirs() -> None:
    for directory in (STATE_DIR, DATASETS_DIR, RUNS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, payload: str) -> None:
    """Write through a temp file so a crash mid-write cannot truncate state."""
    handle, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Portable paths
# ---------------------------------------------------------------------------


def _to_stored(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    try:
        relative = Path(value).resolve().relative_to(STATE_DIR)
    except Exception:
        return value
    return _RELATIVE_PREFIX + relative.as_posix()


def _to_local(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return value
    if value.startswith(_RELATIVE_PREFIX):
        return str(STATE_DIR / value[len(_RELATIVE_PREFIX):])
    # A v4 record with an absolute path from another machine or user. If the
    # same relative location exists under this state directory, use that.
    marker = "/.gradmesh/"
    normalised = re.sub(r"/+", "/", value.replace("\\", "/"))
    if marker in normalised and not Path(value).exists():
        candidate = STATE_DIR / normalised.split(marker, 1)[1]
        if candidate.exists():
            return str(candidate)
    return value


def _dataset_out(record: dict) -> dict:
    stored = copy.deepcopy(record)
    for key in _PATH_KEYS:
        if key in stored:
            stored[key] = _to_stored(stored[key])
    splits = stored.get("splits")
    if isinstance(splits, dict):
        for key in _SPLIT_KEYS:
            if key in splits:
                splits[key] = _to_stored(splits[key])
    stored.pop("available", None)
    return stored


def _dataset_in(record: Optional[dict]) -> Optional[dict]:
    if record is None:
        return None
    local = copy.deepcopy(record)
    for key in _PATH_KEYS:
        if key in local:
            local[key] = _to_local(local[key])
    splits = local.get("splits")
    if isinstance(splits, dict):
        for key in _SPLIT_KEYS:
            if key in splits:
                splits[key] = _to_local(splits[key])
    extracted = local.get("extracted_path")
    local["available"] = bool(extracted and Path(extracted).exists())
    return local


# ---------------------------------------------------------------------------
# Load and save
# ---------------------------------------------------------------------------


def _fresh_state() -> Dict[str, Any]:
    state = copy.deepcopy(_DEFAULT_STATE)
    state["created_at"] = time.time()
    state["mesh_token"] = secrets.token_urlsafe(24)
    state["mesh_id"] = secrets.token_hex(6)
    return state


def load() -> Dict[str, Any]:
    """The whole state, cached until the file changes. Treat as read-only."""
    with _lock:
        _ensure_dirs()
        if not STATE_FILE.exists():
            state = _fresh_state()
            save(state)
            return state
        mtime = STATE_FILE.stat().st_mtime_ns
        if _cache["state"] is not None and _cache["mtime"] == mtime:
            return _cache["state"]
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt file must not take the mesh down; keep it for inspection.
            backup = STATE_FILE.with_suffix(".corrupt-%d.json" % int(time.time()))
            try:
                shutil.copy2(STATE_FILE, backup)
            except Exception:
                pass
            state = _fresh_state()
        changed = False
        for key, value in _DEFAULT_STATE.items():
            if key not in state:
                state[key] = copy.deepcopy(value)
                changed = True
        if not state.get("mesh_token"):
            state["mesh_token"] = secrets.token_urlsafe(24)
            changed = True
        if not state.get("mesh_id"):
            state["mesh_id"] = secrets.token_hex(6)
            changed = True
        if changed:
            save(state)
            return state
        _cache.update({"mtime": mtime, "state": state})
        return state


def save(state: Dict[str, Any]) -> None:
    with _lock:
        _ensure_dirs()
        _atomic_write(STATE_FILE, json.dumps(state, indent=2))
        _cache.update({"mtime": STATE_FILE.stat().st_mtime_ns, "state": state})


def update(mutator) -> Dict[str, Any]:
    """Read, mutate and persist under one lock."""
    with _lock:
        state = copy.deepcopy(load())
        mutator(state)
        save(state)
        return state


# ---------------------------------------------------------------------------
# Mesh identity
# ---------------------------------------------------------------------------


def mesh_token() -> str:
    return load()["mesh_token"]


def mesh_id() -> str:
    return load()["mesh_id"]


def rotate_mesh_token() -> str:
    new_token = secrets.token_urlsafe(24)

    def mutate(state: Dict[str, Any]) -> None:
        state["mesh_token"] = new_token

    update(mutate)
    return new_token


def policy() -> Dict[str, Any]:
    return dict(load().get("policy") or {})


def set_policy(values: Dict[str, Any]) -> Dict[str, Any]:
    def mutate(state: Dict[str, Any]) -> None:
        merged = dict(state.get("policy") or {})
        merged.update(values)
        state["policy"] = merged

    return update(mutate).get("policy") or {}


def reset_policy() -> None:
    def mutate(state: Dict[str, Any]) -> None:
        state["policy"] = {}

    update(mutate)


# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------

NODE_STAT_KEYS = (
    "display_name",
    "gpu",
    "backend",
    "vendor",
    "workloads",
    "reliability",
    "completed_rounds",
    "failed_rounds",
    "samples_trained",
    "seconds_trained",
    "joined_at",
    "last_seen",
    "batch_cap",
)


def node_stats(node_id: str) -> Optional[dict]:
    record = (load().get("nodes") or {}).get(node_id)
    return copy.deepcopy(record) if record else None


def all_node_stats() -> Dict[str, dict]:
    return copy.deepcopy(load().get("nodes") or {})


def put_node_stats(records: Dict[str, dict]) -> None:
    """Persist what the mesh has learned about these machines. One write per round."""
    if not records:
        return

    def mutate(state: Dict[str, Any]) -> None:
        nodes = state.setdefault("nodes", {})
        for node_id, node in records.items():
            nodes[node_id] = {key: node.get(key) for key in NODE_STAT_KEYS if node.get(key) is not None}
        # Bound the history: machines unseen for 90 days are forgotten.
        cutoff = time.time() - 90 * 86400
        for node_id in [key for key, value in nodes.items() if float(value.get("last_seen") or 0) < cutoff]:
            nodes.pop(node_id, None)

    update(mutate)


def forget_node(node_id: str) -> None:
    def mutate(state: Dict[str, Any]) -> None:
        (state.get("nodes") or {}).pop(node_id, None)

    update(mutate)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def list_datasets() -> List[dict]:
    state = load()
    default_id = state.get("default_dataset_id")
    datasets = []
    for record in state.get("datasets", {}).values():
        item = _dataset_in(record)
        item["is_default"] = record.get("id") == default_id
        datasets.append(item)
    datasets.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return datasets


def get_dataset(dataset_id: str) -> Optional[dict]:
    return _dataset_in((load().get("datasets") or {}).get(dataset_id))


def default_dataset() -> Optional[dict]:
    state = load()
    default_id = state.get("default_dataset_id")
    if default_id and default_id in (state.get("datasets") or {}):
        return _dataset_in(state["datasets"][default_id])
    datasets = [item for item in list_datasets() if not item.get("is_subset")]
    return datasets[0] if datasets else None


def put_dataset(record: dict, make_default: bool = False) -> dict:
    def mutate(state: Dict[str, Any]) -> None:
        state.setdefault("datasets", {})[record["id"]] = _dataset_out(record)
        if not record.get("is_subset") and (make_default or not state.get("default_dataset_id")):
            state["default_dataset_id"] = record["id"]

    update(mutate)
    return record


def set_default_dataset(dataset_id: str) -> None:
    def mutate(state: Dict[str, Any]) -> None:
        if dataset_id in (state.get("datasets") or {}):
            state["default_dataset_id"] = dataset_id

    update(mutate)


def delete_dataset(dataset_id: str) -> bool:
    removed = {"value": False}

    def mutate(state: Dict[str, Any]) -> None:
        datasets = state.get("datasets") or {}
        if dataset_id in datasets:
            datasets.pop(dataset_id)
            removed["value"] = True
        # Subsets are manifests over their parent; they go with it.
        for child in [key for key, value in datasets.items() if value.get("parent_id") == dataset_id]:
            datasets.pop(child, None)
        if state.get("default_dataset_id") not in datasets:
            state["default_dataset_id"] = next(
                (key for key, value in datasets.items() if not value.get("is_subset")), None
            )

    update(mutate)
    shutil.rmtree(dataset_dir(dataset_id), ignore_errors=True)
    return removed["value"]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_runs(limit: int = 100) -> List[dict]:
    runs = list((load().get("runs") or {}).values())
    runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return copy.deepcopy(runs[:limit])


def get_run(run_id: str) -> Optional[dict]:
    record = (load().get("runs") or {}).get(run_id)
    return copy.deepcopy(record) if record else None


def put_run(record: dict) -> dict:
    def mutate(state: Dict[str, Any]) -> None:
        state.setdefault("runs", {})[record["id"]] = record

    update(mutate)
    return record


def delete_run(run_id: str) -> bool:
    removed = {"value": False}

    def mutate(state: Dict[str, Any]) -> None:
        runs = state.get("runs") or {}
        if runs.pop(run_id, None) is not None:
            removed["value"] = True

    update(mutate)
    shutil.rmtree(RUNS_DIR / run_id, ignore_errors=True)
    return removed["value"]
