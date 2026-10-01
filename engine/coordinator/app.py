"""GradMesh 5 coordinator.

One FastAPI process owns the whole control plane: node registry, admission,
round planning, shard planning, the round barrier, aggregation, run history and
the live event stream.

What v4 established, and v5 keeps:

* Shards are sized per node from measured performance instead of split evenly.
* A background supervisor enforces per-shard deadlines and speculates on
  stragglers.
* Aggregation is sample-weighted, which is what unequal shards require.
* Datasets are uploaded through the dashboard and stored in a registry.
* Every transition is published to an event bus that the dashboard streams.

What v5 changes:

* **Three GPU families in one mesh.** Workers report a backend (cuda, xpu, mps)
  and vendor; runs can be restricted to a vendor mix, and every round records a
  per-backend breakdown so cross-vendor results can be reported directly.
* **An affine cost model.** Each machine's fixed per-round overhead and its
  per-image rate are learned separately, per workload, and persisted across
  coordinator restarts (see scheduler.affine_split).
* **Shards are file lists.** Workers cache images and fetch only what they lack,
  so the host no longer copies and zips the dataset every round.
* **Weights move as raw bytes.** The v4 base64 JSON endpoints remain for older
  agents.
* **Connections are forgiving.** A busy worker gets a longer liveness window
  than an idle one, a restarted worker supersedes its old process cleanly, and
  machine statistics survive a restart.
* **The training recipe is a run setting.** Warmup policy, optimiser and
  worker-side validation are chosen per run and applied identically on every
  backend, which fixes the leg 1 accuracy defect by default.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

# The engine directory holds the v3 modules that must stay importable by name.
ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from coordinator import (
    aggregation,
    benchmark,
    datasets_std,
    discovery,
    evaluation,
    sharding,
    store,
)
from coordinator.events import bus
from coordinator.health import health_warnings
from version import PROTOCOL, REFERENCE_STACK, __version__

from coordinator.scheduler import (
    DEFAULT_POLICY,
    Deadline,
    MeshPolicy,
    admit,
    aggregation_weights,
    deadline_for,
    efficiency,
    fitness,
    mesh_reference,
    PARTITION_EQUAL,
    PARTITION_LINEAR,
    PARTITION_PROPORTIONAL,
    PARTITION_STRATEGIES,
    imbalance,
    plan_round,
    safe_batch_size,
    should_abort_round,
    straggler_action,
    update_fixed,
    update_reliability,
    update_throughput,
)

# Rounds close on the aggregator thread and deadlines fire on an executor thread,
# so every publish goes through the thread-safe path.
emit = bus.publish_threadsafe

# What this coordinator offers a worker. A v5 worker uses each one only when the
# coordinator lists it, so either side can be older than the other.
FEATURES = ["binary-weights", "image-cache", "phase-timing", "diagnostics", "instance-id"]
BACKENDS = ("cuda", "xpu", "mps", "cpu")
VENDOR_OF = {"cuda": "nvidia", "xpu": "intel", "mps": "apple", "cpu": "cpu"}
WARMUP_MODES = ("first-round", "none", "every-round")

HEARTBEAT_TIMEOUT_SECONDS = float(os.getenv("GRADMESH_HEARTBEAT_TIMEOUT", "20"))
# A machine in the middle of a shard is given this many timeouts of silence
# before its work is declared lost. Training saturates a laptop, a Wi-Fi link
# drops for a few seconds, and dropping a nearly finished shard over that costs
# far more than waiting a little longer.
BUSY_TIMEOUT_MULTIPLE = 3.0
SUPERVISOR_INTERVAL_SECONDS = 2.0
# Multiples of the heartbeat timeout after which a silent node is dropped from
# the registry entirely rather than shown as an offline member forever.
NODE_EVICTION_MULTIPLE = 15
# A run that is nominally running but has no shard in flight is between rounds.
# That gap should last milliseconds. If it lasts this long, something went wrong
# that nobody reported, and the watchdog steps in.
STALL_GRACE_SECONDS = 90.0
MAX_STALL_RECOVERIES = 2
MAX_ROUND_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# In-memory mesh state
# ---------------------------------------------------------------------------

state_lock = RLock()
nodes: Dict[str, dict] = {}
runs: Dict[str, dict] = {}
batches: Dict[str, dict] = {}
aggregator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradmesh-agg")
scanner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradmesh-scan")
# One thread, one sweep. Two benchmark sweeps at once would share GPUs and
# neither set of timings would mean anything.
sweeper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradmesh-sweep")
suite_lock = RLock()
active_suite: Dict[str, Any] = {"id": None, "abort": False}

# Devices that opened the join page but have not installed the agent. This is
# how the host sees "my other laptop is here, looking at the join screen right
# now" without anything being installed on that laptop.
visitors: Dict[str, dict] = {}
VISITOR_TTL_SECONDS = 75.0
_visitor_rate: Dict[str, List[float]] = {}
VISITOR_RATE_WINDOW = 10.0
VISITOR_RATE_LIMIT = 8

advertiser = discovery.MulticastAdvertiser(
    hostname=os.getenv("GRADMESH_MDNS_NAME", "gradmesh"),
    web_port=int(os.getenv("GRADMESH_WEB_PORT", "3000")),
    api_port=int(os.getenv("GRADMESH_COORDINATOR_PORT", "8000")),
)

_scan_cache: Dict[str, Any] = {"result": None, "at": 0.0, "running": False}
SCAN_CACHE_SECONDS = 25.0


def supervise(future, label: str, on_error=None):
    """Surface exceptions raised on a background thread.

    ThreadPoolExecutor stores an exception on the Future and never raises it, so
    a submit whose result nobody inspects fails in total silence. That is
    exactly what happened in the field: aggregation finished a round, the call
    that planned the next one raised, and the dashboard sat on "planning the
    next round" forever with no error anywhere, in the log or on screen.

    Every submit now routes through here, so a failure is printed with its
    traceback, published to the event stream, and handed to a callback that can
    mark the affected run or sweep failed rather than leaving it hanging.
    """

    def done(settled):
        try:
            settled.result()
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch everything
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print("[gradmesh] %s failed:\n%s" % (label, detail), flush=True)
            emit(
                "internal.error",
                {"where": label, "error": "%s: %s" % (type(exc).__name__, exc)},
            )
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception:
                    print("[gradmesh] %s error handler failed" % label, flush=True)

    future.add_done_callback(done)
    return future


def current_policy() -> MeshPolicy:
    return MeshPolicy.from_dict(store.policy())


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
#
# A machine's speed is a property of the machine *and* the job: the same GPU
# trains yolov8n at 320 px several times faster than at 640 px. v4 kept one
# throughput number per machine, so a run at a new image size started from a
# number measured on a different workload. v5 keys the learned cost model by
# checkpoint and image size.


def workload_key(base_model: str, imgsz: int) -> str:
    return "%s@%d" % (Path(base_model or "yolov8n.pt").name, int(imgsz or 640))


def _with_workload(node: dict, key: Optional[str]) -> dict:
    """A copy of `node` whose rate and overhead describe workload `key`."""
    view = dict(node)
    workloads = node.get("workloads") or {}
    entry = workloads.get(key) if key else None
    if entry is None and key and workloads:
        # Same checkpoint at another image size: activation cost scales with
        # pixel count, so scale the rate by the area ratio. Better than
        # falling back to the probe, which knows nothing about this machine's
        # data loading.
        model, _, size = key.rpartition("@")
        nearest = None
        for other_key, other in workloads.items():
            other_model, _, other_size = other_key.rpartition("@")
            if other_model == model and other.get("rate"):
                if nearest is None or abs(int(other_size) - int(size)) < abs(int(nearest[0]) - int(size)):
                    nearest = (other_size, other)
        if nearest is not None:
            scale = (int(nearest[0]) / max(1, int(size))) ** 2
            entry = {"rate": float(nearest[1]["rate"]) * scale, "fixed": nearest[1].get("fixed")}
    view["throughput_sps"] = float((entry or {}).get("rate") or 0.0)
    view["fixed_seconds"] = float((entry or {}).get("fixed") or 0.0)
    return view


def _preview_workload() -> str:
    """The workload the dashboard's plan preview should describe."""
    with state_lock:
        live = sorted(runs.values(), key=lambda run: run.get("created_at") or 0, reverse=True)
    if live:
        return workload_key(live[0]["base_model"], live[0]["imgsz"])
    archived = store.list_runs(limit=1)
    if archived:
        return workload_key(archived[0].get("base_model", "yolov8n.pt"), archived[0].get("imgsz", 640))
    return workload_key("yolov8n.pt", 640)


def _node_view(node: dict, mesh: dict, policy: MeshPolicy, key: Optional[str] = None) -> dict:
    view = {k: v for k, v in _with_workload(node, key).items() if k != "secret"}
    view["fitness"] = round(fitness(node, mesh, policy), 4)
    view["vendor"] = node.get("vendor") or VENDOR_OF.get(node.get("backend") or "", "cpu")
    view["warnings"] = health_warnings(node.get("diagnostics"), bool(node.get("co_located")))
    view["workload"] = key
    return view


def snapshot_nodes(key: Optional[str] = None) -> List[dict]:
    policy = current_policy()
    now = time.time()
    key = key or _preview_workload()
    with state_lock:
        _refresh_liveness(now)
        pool = list(nodes.values())
        mesh = mesh_reference(pool, now)
        return [_node_view(node, mesh, policy, key) for node in pool]


def _refresh_liveness(now: float) -> None:
    for node in nodes.values():
        was_active = node.get("active", False)
        silent = now - float(node.get("last_seen") or 0)
        busy = int(node.get("active_batches") or 0) > 0
        limit = HEARTBEAT_TIMEOUT_SECONDS * (BUSY_TIMEOUT_MULTIPLE if busy else 1.0)
        node["active"] = silent <= limit
        node["liveness"] = (
            "online" if silent <= HEARTBEAT_TIMEOUT_SECONDS else "suspect" if node["active"] else "offline"
        )
        if was_active and not node["active"]:
            emit(
                "node.offline", {"node_id": node["node_id"], "name": node.get("display_name")}
            )


def _persist_nodes(node_ids: Sequence[str]) -> None:
    """Write what the mesh has learned about these machines. Called once per round."""
    with state_lock:
        records = {node_id: dict(nodes[node_id]) for node_id in node_ids if node_id in nodes}
    try:
        store.put_node_stats(records)
    except Exception as exc:  # never fail a round over bookkeeping
        print("[gradmesh] could not persist machine statistics: %s" % exc, flush=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def require_mesh_token(x_mesh_token: Optional[str] = Header(default=None)) -> str:
    """Workers and the dashboard proxy must present the mesh join token.

    This is a shared secret for a LAN, not an identity system. It stops a random
    device on a coffee-shop network from registering as a worker and receiving
    dataset shards, which is the actual threat when you hand out a URL.
    """
    expected = store.mesh_token()
    if not x_mesh_token or not _constant_time_equals(x_mesh_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing mesh token")
    return x_mesh_token


def _constant_time_equals(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _is_local_client(host: Optional[str]) -> bool:
    """True when the caller is on this machine or this /24."""
    if not host:
        return False
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    own = discovery.local_ipv4()
    if own.startswith("127."):
        return False
    return host.rsplit(".", 1)[0] == own.rsplit(".", 1)[0]


def allow_local_or_token(
    request: Request, x_mesh_token: Optional[str] = Header(default=None)
) -> str:
    """Auth for the presence beacon only.

    The beacon has to be callable by a browser that has not installed anything
    and does not hold the token, and it must see the visitor's real address,
    which a server-side proxy would replace with its own. So a request from this
    subnet is accepted without a token. It carries no dataset access and writes
    only to a list of who is looking at the join page, which anyone able to reach
    this host could observe anyway.
    """
    expected = store.mesh_token()
    if x_mesh_token and _constant_time_equals(x_mesh_token, expected):
        return x_mesh_token
    client = request.client.host if request.client else None
    if _is_local_client(client):
        return "local"
    raise HTTPException(status_code=401, detail="Invalid or missing mesh token")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    node_id: str
    display_name: Optional[str] = None
    gpu: str = "unknown"
    gpu_memory_mb: int = Field(default=8000, ge=1)
    max_batch_size: int = Field(default=4, ge=1, le=256)
    backend: str = "cpu"
    supports_training: bool = True
    capability: Optional[dict] = None
    owner: Optional[str] = None
    agent_version: str = "4.0.0"
    # v5 workers. All optional, so a v4 agent still registers.
    instance_id: Optional[str] = None
    protocol: int = 4
    features: List[str] = Field(default_factory=list)
    vendor: Optional[str] = None
    diagnostics: Optional[dict] = None
    co_located: Optional[bool] = None
    dataloader_workers: Optional[int] = None


class HeartbeatRequest(BaseModel):
    node_id: str
    instance_id: Optional[str] = None
    load: Optional[float] = None
    active_batches: Optional[int] = None
    allocated_memory_mb: Optional[int] = None
    training_epoch: Optional[int] = None
    training_total_epochs: Optional[int] = None
    latency_ms: Optional[float] = None
    phase: Optional[str] = None
    progress: Optional[dict] = None
    diagnostics: Optional[dict] = None


class RoundResultRequest(BaseModel):
    node_id: str
    batch_id: str
    round_index: int = Field(ge=0)
    weights_b64: str = Field(..., min_length=1)
    metrics: Optional[dict] = None


class BatchFailureRequest(BaseModel):
    node_id: str
    batch_id: str
    error: str = Field(..., min_length=1)


class CreateRunRequest(BaseModel):
    name: str = Field(default="mesh-run", min_length=1, max_length=80)
    dataset_id: Optional[str] = None
    base_model: str = Field(default="yolov8n.pt", min_length=1)
    rounds: int = Field(default=4, ge=1, le=200)
    imgsz: int = Field(default=640, ge=32, le=2048)
    batch_size: int = Field(default=8, ge=1, le=128)
    mode: str = Field(default="mesh")  # mesh or solo
    notes: Optional[str] = None
    # Restrict the run to specific machines. The benchmark harness uses this to
    # hold hardware constant while varying node count, which is the only way a
    # scaling curve means anything.
    node_ids: Optional[List[str]] = None
    partition_strategy: str = Field(default=PARTITION_PROPORTIONAL)
    # Score the aggregated model after every round. Off by default because it
    # costs real time; the benchmark harness turns it on.
    evaluate: bool = False
    suite_id: Optional[str] = None
    trial_id: Optional[str] = None
    # Training seed. Ultralytics defaults to 0 and sets deterministic mode, so
    # without varying this every repeat of a cell returns bit-identical
    # accuracy and the standard deviation a reviewer asked for is always zero.
    # Timing still varies; accuracy does not.
    seed: int = 0
    # Learning-rate warmup. This is the leg 1 accuracy defect: Ultralytics
    # warms up for 3 epochs by default, a federated round is one epoch, and
    # the optimiser is rebuilt every round, so with the default every round of
    # every run sat inside warmup and mAP fell after round 1.
    #
    #   first-round  warm up for `warmup_epochs` in round 1 only (default)
    #   none         never warm up
    #   every-round  the Ultralytics default every round, i.e. v4 behaviour,
    #                kept so the defect can be reproduced and measured
    warmup_mode: str = Field(default="first-round")
    warmup_epochs: Optional[float] = Field(default=1.0, ge=0.0, le=10.0)
    # The optimiser, identical on every backend. "auto" is Ultralytics' choice,
    # which for round-sized jobs is AdamW with a learning rate fitted to the
    # class count. v4 forced Adam on Intel XPU only, so a mixed-vendor run was
    # quietly comparing two optimisers.
    optimizer: str = Field(default="auto")
    lr0: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    deterministic: bool = True
    # Validate on every worker after every round. The coordinator scores the
    # aggregated model on a fixed split regardless, so this is off by default:
    # it is two validation passes per worker per round, which nobody reads.
    worker_validation: bool = False
    # Dataloader processes per worker. None lets each machine choose from its
    # own CPU count and backend.
    dataloader_workers: Optional[int] = Field(default=None, ge=0, le=16)
    # Only machines with these backends take part: any of cuda, xpu, mps.
    # This is how a cross-vendor experiment selects NVIDIA-only, Intel plus
    # Apple, all three, and so on, without unplugging anything.
    backends: Optional[List[str]] = None


class PolicyRequest(BaseModel):
    values: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.bind_loop(asyncio.get_running_loop())
    store.load()
    supervisor = asyncio.create_task(_supervisor_loop())

    # Claiming gradmesh.local is what lets a peer open the dashboard without
    # anybody reading an IP address off a screen. Best effort: a network that
    # blocks multicast falls back to the address, it does not fail startup.
    #
    # This runs in a worker thread because zeroconf's synchronous API blocks on
    # its own event loop, which it refuses to do from inside another one.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, advertiser.start)

    emit("coordinator.ready", {"version": __version__, "mdns": advertiser.as_dict()})
    try:
        yield
    finally:
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass
        await loop.run_in_executor(None, advertiser.stop)
        aggregator.shutdown(wait=False)
        scanner.shutdown(wait=False)
        sweeper.shutdown(wait=False)


app = FastAPI(title="GradMesh Coordinator", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Node lifecycle
# ---------------------------------------------------------------------------


@app.post("/register_node")
def register_node(req: RegisterRequest, request: Request, _: str = Depends(require_mesh_token)):
    now = time.time()
    client_host = request.client.host if request.client else None
    co_located = bool(req.co_located) or client_host in {"127.0.0.1", "::1"} or (
        client_host is not None and client_host == discovery.local_ipv4()
    )
    backend = req.backend if req.backend in BACKENDS else "cpu"
    superseded = None

    with state_lock:
        existing = nodes.get(req.node_id)
        if existing is None:
            # The coordinator restarted, or this machine has been here before:
            # pick up what the mesh learned about it rather than starting cold.
            existing = store.node_stats(req.node_id) or {}
        elif (
            req.instance_id
            and existing.get("instance_id")
            and existing.get("instance_id") != req.instance_id
        ):
            # A new process for the same GPU. It wins: the old one has usually
            # crashed or been restarted, and if it is still alive it is told so
            # on its next heartbeat and exits. Any shard it held is lost, so
            # the round replans rather than waiting out a deadline.
            superseded = existing.get("instance_id")
            for batch in batches.values():
                if batch.get("node_id") == req.node_id and batch["status"] in {"queued", "assigned"}:
                    batch["status"] = "dropped"
                    batch["error"] = "the worker restarted mid-round"

        nodes[req.node_id] = {
            "node_id": req.node_id,
            "instance_id": req.instance_id,
            "protocol": req.protocol,
            "features": list(req.features or []),
            "display_name": req.display_name or req.gpu,
            "gpu": req.gpu,
            "backend": backend,
            "vendor": req.vendor or VENDOR_OF.get(backend, "cpu"),
            "gpu_memory_mb": req.gpu_memory_mb,
            "max_batch_size": req.max_batch_size,
            # Lowered after an out-of-memory failure, and kept across restarts.
            "batch_cap": existing.get("batch_cap"),
            "supports_training": req.supports_training,
            "capability": req.capability or existing.get("capability") or {},
            "diagnostics": req.diagnostics or existing.get("diagnostics") or {},
            "co_located": co_located,
            "dataloader_workers": req.dataloader_workers,
            "owner": req.owner,
            "agent_version": req.agent_version,
            "address": client_host,
            "joined_at": existing.get("joined_at", now),
            "last_seen": now,
            "active": True,
            "liveness": "online",
            "load": 0.0,
            # None, not 0.0. A node that has not been timed yet is not a
            # node on a zero-latency link, and the scheduler's latency term
            # should not reward it for a measurement that never happened.
            "latency_ms": existing.get("latency_ms"),
            "allocated_memory_mb": 0,
            "active_batches": 0,
            "phase": "idle",
            "progress": None,
            "completed_rounds": existing.get("completed_rounds", 0),
            "failed_rounds": existing.get("failed_rounds", 0),
            "samples_trained": existing.get("samples_trained", 0),
            "seconds_trained": existing.get("seconds_trained", 0.0),
            # A fresh node starts trusted enough to receive work but not enough
            # to outweigh a node that has actually finished rounds.
            "reliability": existing.get("reliability", 0.7),
            # Reset on registration: a reconnecting worker has usually been
            # fixed, so quarantine should not outlive the process that earned it.
            "consecutive_failures": 0,
            "workloads": existing.get("workloads") or {},
            "throughput_sps": 0.0,
            "fixed_seconds": 0.0,
            "training_epoch": 0,
            "training_total_epochs": 0,
        }
    policy = current_policy()
    # Admission is relative to the rest of the mesh, so judge the new node
    # against the whole pool rather than against itself.
    with state_lock:
        pool = [dict(node) for node in nodes.values()]
    decision = admit(pool, policy)[req.node_id]
    emit(
        "node.joined",
        {
            "node_id": req.node_id,
            "name": req.display_name,
            "gpu": req.gpu,
            "backend": backend,
            "vendor": VENDOR_OF.get(backend, "cpu"),
            "tier": decision.tier,
            "reason": decision.reason,
            "gflops": (req.capability or {}).get("gflops"),
            "restarted": bool(superseded),
        },
    )
    if req.protocol > PROTOCOL:
        emit(
            "node.version",
            {
                "node_id": req.node_id,
                "name": req.display_name,
                "detail": "this worker is newer than the host (protocol %d against %d); update the host"
                % (req.protocol, PROTOCOL),
            },
        )
    return {
        "status": "registered",
        "node_id": req.node_id,
        "mesh_id": store.mesh_id(),
        "version": __version__,
        "protocol": PROTOCOL,
        "features": FEATURES,
        "admission": decision.as_dict(),
        "heartbeat_seconds": max(3.0, HEARTBEAT_TIMEOUT_SECONDS / 4),
    }


@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.get(req.node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Unknown node. Register first.")
        if req.instance_id and node.get("instance_id") and node["instance_id"] != req.instance_id:
            raise HTTPException(
                status_code=409,
                detail="Superseded: a newer GradMesh worker for this GPU registered, so this one is leaving.",
            )
        node["last_seen"] = time.time()
        node["active"] = True
        node["liveness"] = "online"
        if req.load is not None:
            node["load"] = req.load
        if req.active_batches is not None:
            node["active_batches"] = req.active_batches
        if req.allocated_memory_mb is not None:
            node["allocated_memory_mb"] = req.allocated_memory_mb
        if req.training_epoch is not None:
            node["training_epoch"] = req.training_epoch
        if req.training_total_epochs is not None:
            node["training_total_epochs"] = req.training_total_epochs
        if req.latency_ms is not None:
            node["latency_ms"] = req.latency_ms
        if req.phase is not None:
            node["phase"] = req.phase
        node["progress"] = req.progress
        if req.diagnostics:
            node["diagnostics"] = req.diagnostics
    return {"status": "ok"}


@app.post("/leave")
def leave(req: HeartbeatRequest, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.get(req.node_id)
        # A superseded process saying goodbye must not unregister its successor.
        if node is not None and req.instance_id and node.get("instance_id") not in (None, req.instance_id):
            return {"status": "ignored"}
        node = nodes.pop(req.node_id, None)
        if node is not None:
            for batch in batches.values():
                if batch.get("node_id") == req.node_id and batch["status"] in {"queued", "assigned"}:
                    batch["status"] = "dropped"
                    batch["error"] = "the worker left the mesh"
    if node:
        store.put_node_stats({req.node_id: node})
        emit("node.left", {"node_id": req.node_id, "name": node.get("display_name")})
    return {"status": "ok"}


@app.delete("/nodes/{node_id}")
def evict_node(node_id: str, forget: bool = Query(default=False), _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.pop(node_id, None)
        for batch in batches.values():
            if batch.get("node_id") == node_id and batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = "node was evicted by the mesh owner"
    if forget:
        store.forget_node(node_id)
    if node is None and not forget:
        raise HTTPException(status_code=404, detail="Unknown node")
    emit("node.evicted", {"node_id": node_id, "name": (node or {}).get("display_name")})
    return {"status": "evicted"}


@app.post("/nodes/{node_id}/reset")
def reset_node_estimates(node_id: str, _: str = Depends(require_mesh_token)):
    """Forget what the mesh learned about a machine's speed and overhead.

    For after a hardware change, a driver update or moving the coordinator off
    a machine: its old measurements no longer describe it.
    """
    with state_lock:
        node = nodes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Unknown node")
        node["workloads"] = {}
        node["batch_cap"] = None
        node["reliability"] = 0.7
        node["consecutive_failures"] = 0
    _persist_nodes([node_id])
    emit("node.reset", {"node_id": node_id, "name": node.get("display_name")})
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Work dispatch
# ---------------------------------------------------------------------------


@app.get("/get_batch/{node_id}")
def get_batch(node_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Unknown node. Register first.")
        _refresh_liveness(time.time())

        for batch in batches.values():
            if batch["status"] != "queued" or batch.get("node_id") != node_id:
                continue
            run = runs.get(batch["run_id"])
            if run is None or run["status"] not in {"running", "planning"}:
                continue

            batch["status"] = "assigned"
            batch["assigned_at"] = time.time()
            node["active_batches"] = 1
            node["allocated_memory_mb"] = batch["memory_mb"]
            payload = _batch_payload(batch, run, node)
            batch["resolved_batch_size"] = payload["batch_size"]

            emit(
                "shard.assigned",
                {
                    "run_id": run["id"],
                    "batch_id": batch["batch_id"],
                    "node_id": node_id,
                    "name": node.get("display_name"),
                    "backend": node.get("backend"),
                    "round": batch["round_index"],
                    "samples": batch["samples"],
                    "predicted_seconds": batch["predicted_seconds"],
                },
            )
            return {"batch": payload}

    return {"batch": None}


def _warmup_for_round(run: dict, round_index: int) -> Optional[float]:
    """The warmup_epochs value one round trains with, or None for Ultralytics' default."""
    mode = run.get("warmup_mode") or "first-round"
    if mode == "every-round":
        return None
    if mode == "none":
        return 0.0
    return float(run.get("warmup_epochs") or 0.0) if round_index == 0 else 0.0


def _node_batch_ceiling(node: Optional[dict], fallback: Optional[int]) -> Optional[int]:
    if node is None:
        return fallback
    ceilings = [value for value in (node.get("max_batch_size"), node.get("batch_cap")) if value]
    return min(ceilings) if ceilings else fallback


def _batch_payload(batch: dict, run: dict, node: Optional[dict] = None) -> dict:
    """The wire shape every worker since v3 understands, plus v5 fields."""
    # Batch size is decided per device, not per run. The requested value is a
    # ceiling; a smaller card gets whatever it can actually hold, and a card
    # that ran out of memory once is capped below that.
    resolved_batch = safe_batch_size(
        device_memory_mb=batch.get("device_memory_mb") or 0,
        imgsz=run["imgsz"],
        requested=run["batch_size"],
        node_max=_node_batch_ceiling(node, batch.get("max_batch_size")),
        unified_memory=bool(batch.get("unified_memory")),
    )
    warmup = _warmup_for_round(run, batch["round_index"])
    payload = {
        "batch_id": batch["batch_id"],
        "job_id": run["id"],
        "kind": "training",
        "round_index": batch["round_index"],
        "shard_index": batch["shard_index"],
        "shard_url": "/runs/%s/shards/%d.zip" % (run["id"], batch["shard_index"]),
        "manifest_url": "/runs/%s/shards/%d/manifest" % (run["id"], batch["shard_index"]),
        "weights_url": "/runs/%s/weights" % run["id"],
        "weights_bin_url": "/runs/%s/weights.bin" % run["id"],
        "dataset_key": run.get("dataset_key"),
        "base_model": run["base_model"],
        "imgsz": run["imgsz"],
        "batch_size": resolved_batch,
        "estimated_memory_mb": batch["memory_mb"],
        "epochs": 1,
        "seed": int(run.get("seed") or 0),
        "optimizer": run.get("optimizer") or "auto",
        "lr0": run.get("lr0"),
        "deterministic": bool(run.get("deterministic", True)),
        "worker_validation": bool(run.get("worker_validation")),
        "dataloader_workers": run.get("dataloader_workers"),
        "job_name": run["name"],
        "class_names": run["class_names"],
        "current_round": run["current_round"],
        "total_rounds": run["rounds"],
        "samples": batch["samples"],
        "deadline_seconds": batch["hard_deadline_seconds"],
    }
    # Omitted rather than sent as None, so a v4 worker keeps Ultralytics'
    # default exactly as it did.
    if warmup is not None:
        payload["warmup_epochs"] = warmup
    if payload["lr0"] is None:
        payload.pop("lr0")
    return payload


def _learn_from_result(node: dict, batch: dict, run: dict, elapsed: float, metrics: dict) -> None:
    """Update the machine's cost model, per workload, from one finished shard.

    The slope is the training loop alone: samples over the epoch time the
    worker measured. The intercept is everything else in the round, from the
    moment the shard was handed out to the moment its result arrived, which
    covers transfer, model load, trainer construction, saving and upload.

    A worker's first shard after it starts pays one-off costs, CUDA context
    creation and cuDNN autotuning among them, that later rounds do not. Its
    overhead is recorded but marked cold, and the first warm measurement
    replaces it outright rather than being averaged with it, so one cold start
    does not leave a machine looking slow for five rounds.
    """
    policy = current_policy()
    key = workload_key(run["base_model"], run["imgsz"])
    workloads = node.setdefault("workloads", {})
    entry = workloads.setdefault(key, {"rate": 0.0, "fixed": 0.0, "rounds": 0})
    samples = int(batch["samples"])

    epoch_seconds = float(metrics.get("epoch_seconds") or 0.0)
    train_seconds = float(metrics.get("train_seconds") or 0.0)
    compute = epoch_seconds if epoch_seconds > 0 else (train_seconds if train_seconds > 0 else elapsed)
    overhead = max(0.0, elapsed - compute)

    entry["rate"] = update_throughput({"throughput_sps": entry.get("rate") or 0.0}, samples, compute, policy)
    cold = bool(metrics.get("first_batch"))
    if overhead > 0:
        if entry.get("fixed_cold") and not cold:
            entry["fixed"] = overhead
            entry["fixed_cold"] = False
        elif cold and float(entry.get("fixed") or 0.0) > 0:
            pass  # a restart's cold round says nothing about steady overhead
        else:
            entry["fixed"] = update_fixed({"fixed_seconds": entry.get("fixed") or 0.0}, overhead, policy)
            entry["fixed_cold"] = cold
    entry["rounds"] = int(entry.get("rounds") or 0) + 1
    entry["updated_at"] = time.time()

    node["throughput_sps"] = entry["rate"]
    node["fixed_seconds"] = entry.get("fixed") or 0.0
    node["reliability"] = update_reliability(node, True, policy)
    node["completed_rounds"] = node.get("completed_rounds", 0) + 1
    node["consecutive_failures"] = 0
    node["samples_trained"] = node.get("samples_trained", 0) + samples
    node["seconds_trained"] = node.get("seconds_trained", 0.0) + elapsed
    node["active_batches"] = 0
    node["allocated_memory_mb"] = 0


def _accept_result(batch_id: str, node_id: str, round_index: int, weights: bytes, metrics: dict) -> dict:
    """Record one finished shard. Idempotent: a retried upload is acknowledged."""
    finished_at = time.time()
    with state_lock:
        batch = batches.get(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Unknown batch; its round has already closed")
        if batch.get("node_id") != node_id:
            raise HTTPException(status_code=409, detail="Batch is not assigned to this node")
        if batch["round_index"] != round_index:
            raise HTTPException(status_code=400, detail="round_index does not match batch")

        if batch["status"] in {"done", "superseded"}:
            # This shard already has a winner, or this is a retried upload of
            # a result that already landed. Acknowledge and discard.
            return {"status": "superseded", "batch_id": batch_id}

        elapsed = max(0.001, finished_at - float(batch.get("assigned_at") or finished_at))
        batch["status"] = "done"
        batch["finished_at"] = finished_at
        batch["elapsed_seconds"] = elapsed
        batch["weights"] = bytes(weights)
        batch["metrics"] = metrics or {}

        run = runs.get(batch["run_id"])
        node = nodes.get(node_id)
        if node is not None and run is not None:
            _learn_from_result(node, batch, run, elapsed, metrics or {})

        # A speculative pair races for the same shard. The loser must leave the
        # barrier immediately, otherwise the round waits on work it no longer
        # needs.
        _supersede_twins_locked(batch)
        run_id = run["id"] if run else None

    emit(
        "shard.completed",
        {
            "run_id": run_id,
            "batch_id": batch_id,
            "node_id": node_id,
            "round": round_index,
            "seconds": round(elapsed, 2),
            "samples": batch["samples"],
            "throughput_sps": round(batch["samples"] / elapsed, 3),
            "epoch_seconds": (metrics or {}).get("epoch_seconds"),
            "backend": (metrics or {}).get("backend"),
        },
    )

    if run_id:
        _maybe_close_round(run_id)
    return {"status": "received", "batch_id": batch_id}


@app.post("/batches/{batch_id}/result")
async def submit_result_binary(
    batch_id: str,
    request: Request,
    x_node_id: str = Header(...),
    x_round_index: int = Header(...),
    x_metrics: Optional[str] = Header(default=None),
    _: str = Depends(require_mesh_token),
):
    """A finished shard as raw serialised weights. Metrics travel in a header."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="The result carried no weights")
    metrics: dict = {}
    if x_metrics:
        try:
            metrics = json.loads(base64.urlsafe_b64decode(x_metrics.encode("ascii")).decode("utf-8"))
        except Exception:
            metrics = {}
    return await run_in_threadpool(_accept_result, batch_id, x_node_id, x_round_index, body, metrics)


@app.post("/submit_training_round_result")
def submit_round_result(req: RoundResultRequest, _: str = Depends(require_mesh_token)):
    """The v4 wire format: base64 weights inside JSON."""
    try:
        weights = base64.b64decode(req.weights_b64.encode("ascii"))
    except Exception:
        raise HTTPException(status_code=400, detail="weights_b64 is not valid base64")
    return _accept_result(req.batch_id, req.node_id, req.round_index, weights, req.metrics or {})


@app.post("/submit_training_batch_failure")
def submit_batch_failure(req: BatchFailureRequest, _: str = Depends(require_mesh_token)):
    capped = None
    with state_lock:
        batch = batches.get(req.batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Unknown batch")
        if batch["status"] in {"done", "dropped", "superseded"}:
            return {"status": "ignored"}
        batch["status"] = "failed"
        batch["error"] = req.error[:4000]
        batch["finished_at"] = time.time()

        node = nodes.get(req.node_id)
        if node is not None:
            node["reliability"] = update_reliability(node, False, current_policy())
            node["failed_rounds"] = node.get("failed_rounds", 0) + 1
            node["consecutive_failures"] = int(node.get("consecutive_failures", 0) or 0) + 1
            node["active_batches"] = 0
            node["allocated_memory_mb"] = 0
            lowered = req.error.lower()
            if "out of memory" in lowered or "outofmemory" in lowered:
                # The memory model guessed wrong for this machine. Halve its
                # ceiling so the next round fits, rather than failing it again
                # and quarantining a machine that only needed a smaller batch.
                used = int(batch.get("resolved_batch_size") or batch.get("max_batch_size") or 2)
                capped = max(1, used // 2)
                node["batch_cap"] = capped
                node["consecutive_failures"] = max(0, node["consecutive_failures"] - 1)
        run_id = batch["run_id"]

    emit(
        "shard.failed",
        {"run_id": run_id, "batch_id": req.batch_id, "node_id": req.node_id, "error": req.error[:400]},
    )
    if capped is not None:
        emit("node.batch_capped", {"node_id": req.node_id, "batch_size": capped})
        _persist_nodes([req.node_id])
    _maybe_close_round(run_id)
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# Shards, weights and dataset files
# ---------------------------------------------------------------------------


class BundleRequest(BaseModel):
    files: List[str] = Field(..., min_length=1, max_length=1000)


_dirs_cache: Dict[str, Dict[str, Any]] = {}
_shard_zip_lock = RLock()


def _dataset_dirs(dataset: dict) -> Dict[str, Any]:
    """Split directories for a dataset, cached because bundles ask repeatedly."""
    cache_key = "%s|%s" % (dataset["id"], dataset.get("extracted_path"))
    if cache_key not in _dirs_cache:
        listing = sharding.list_split_images(Path(dataset["extracted_path"]), splits=dataset.get("splits"))
        _dirs_cache[cache_key] = {key: (str(value) if value is not None else None) for key, value in listing["dirs"].items()}
    return _dirs_cache[cache_key]


@app.get("/runs/{run_id}/shards/{shard_index}/manifest")
def get_shard_manifest(run_id: str, shard_index: int, _: str = Depends(require_mesh_token)):
    """The files in one shard, with sizes, so a worker can check its cache."""
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        shards = run.get("shards") or []
        if shard_index < 0 or shard_index >= len(shards):
            raise HTTPException(status_code=404, detail="Unknown shard")
        files = list(shards[shard_index]["files"])
        dirs = dict(run["shard_dirs"])
        dataset_key = run.get("dataset_key")
        class_names = list(run["class_names"])
    entries = sharding.shard_manifest(dirs, files)
    return {"dataset_key": dataset_key, "files": entries, "class_names": class_names, "train_count": len(entries)}


@app.post("/datasets/{dataset_id}/bundle")
def get_dataset_bundle(dataset_id: str, req: BundleRequest, _: str = Depends(require_mesh_token)):
    """A zip of just the image and label pairs a worker's cache is missing."""
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    if not dataset.get("available"):
        raise HTTPException(status_code=410, detail="This dataset's files are not on the host any more")
    dirs = _dataset_dirs(dataset)
    import tempfile

    handle, temp_path = tempfile.mkstemp(suffix=".zip", prefix="gradmesh-bundle-")
    os.close(handle)
    try:
        sharding.write_bundle(dirs, req.files, Path(temp_path))
    except ValueError as exc:
        Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    return FileResponse(
        temp_path,
        media_type="application/zip",
        filename="bundle.zip",
        background=BackgroundTask(lambda: Path(temp_path).unlink(missing_ok=True)),
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.post("/runs")
def create_run(req: CreateRunRequest, _: str = Depends(require_mesh_token)):
    dataset = store.get_dataset(req.dataset_id) if req.dataset_id else store.default_dataset()
    if dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset is available. Upload one from the Datasets page first.",
        )
    if not dataset.get("available", True):
        raise HTTPException(
            status_code=400,
            detail="%s is registered but its files are not on this machine. Upload it again, or re-import it."
            % dataset.get("name", "That dataset"),
        )

    model_path = store.MODELS_DIR / Path(req.base_model).name
    if not model_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Base checkpoint %s is not available yet. It downloads during setup."
            % Path(req.base_model).name,
        )

    if not aggregation.torch_ready():
        raise HTTPException(
            status_code=503,
            detail="The training plane is still installing. Aggregation needs torch on the host.",
        )

    if req.partition_strategy not in PARTITION_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail="partition_strategy must be one of %s" % ", ".join(PARTITION_STRATEGIES),
        )
    if req.warmup_mode not in WARMUP_MODES:
        raise HTTPException(status_code=400, detail="warmup_mode must be one of %s" % ", ".join(WARMUP_MODES))
    backends = sorted({backend for backend in (req.backends or []) if backend})
    unknown = [backend for backend in backends if backend not in BACKENDS]
    if unknown:
        raise HTTPException(status_code=400, detail="Unknown backend %s" % ", ".join(unknown))

    manifest = dataset.get("manifest_path")
    dataset_root = Path(dataset["extracted_path"])
    dataset_splits = dataset.get("splits")
    try:
        total_samples = sharding.count_training_samples(
            dataset_root, Path(manifest) if manifest else None, splits=dataset_splits
        )
    except Exception:
        total_samples = int(dataset.get("train_count") or 0)

    run_id = uuid.uuid4().hex[:12]
    now = time.time()
    run = {
        "id": run_id,
        "name": req.name,
        "status": "planning",
        "mode": req.mode,
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "dataset_id": dataset["id"],
        # Workers cache images per dataset key. Subsets share their parent's
        # files, so they share its cache too.
        "dataset_key": dataset.get("parent_id") or dataset["id"],
        "dataset_name": dataset["name"],
        "class_names": dataset.get("class_names") or ["object"],
        "total_samples": total_samples,
        "dataset_manifest": manifest,
        "dataset_root": str(dataset_root),
        "dataset_splits": dataset_splits,
        "node_ids": req.node_ids,
        "backends": backends or None,
        "partition_strategy": req.partition_strategy,
        "seed": int(req.seed),
        "evaluate": bool(req.evaluate),
        "suite_id": req.suite_id,
        "trial_id": req.trial_id,
        "accuracy_history": [],
        "eval_seconds_total": 0.0,
        "comm_bytes_total": 0,
        "base_model": Path(req.base_model).name,
        "workload": workload_key(req.base_model, req.imgsz),
        "rounds": req.rounds,
        "current_round": 0,
        "imgsz": req.imgsz,
        "batch_size": req.batch_size,
        "warmup_mode": req.warmup_mode,
        "warmup_epochs": req.warmup_epochs,
        "optimizer": req.optimizer,
        "lr0": req.lr0,
        "deterministic": req.deterministic,
        "worker_validation": req.worker_validation,
        "dataloader_workers": req.dataloader_workers,
        "notes": req.notes,
        "weights": None,
        "round_history": [],
        "plan": None,
        "error": None,
        "shards": [],
        "reference_stack": dict(REFERENCE_STACK),
    }
    with state_lock:
        runs[run_id] = run

    emit("run.created", {"run_id": run_id, "name": req.name, "rounds": req.rounds})

    try:
        _start_round(run_id)
    except HTTPException:
        with state_lock:
            runs.pop(run_id, None)
        raise
    except Exception as exc:
        with state_lock:
            run["status"] = "failed"
            run["error"] = "%s: %s" % (type(exc).__name__, exc)
        emit("run.failed", {"run_id": run_id, "error": run["error"]})
        raise HTTPException(status_code=500, detail=run["error"])

    return _run_view(run_id)


@app.get("/runs")
def list_runs(_: str = Depends(require_mesh_token)):
    with state_lock:
        live = [_run_summary(run) for run in runs.values()]
    live_ids = {r["id"] for r in live}
    archived = [run for run in store.list_runs() if run["id"] not in live_ids]
    combined = live + archived
    combined.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"runs": combined}


@app.get("/runs/{run_id}")
def get_run(run_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        if run_id in runs:
            return _run_view(run_id)
    archived = store.get_run(run_id)
    if archived is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    archived.setdefault("live_shards", [])
    return archived


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        run["status"] = "stopped"
        run["finished_at"] = time.time()
        for batch in batches.values():
            if batch["run_id"] == run_id and batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = "run stopped by the mesh owner"
    emit("run.stopped", {"run_id": run_id})
    _archive_run(run_id)
    return {"status": "stopped"}


@app.delete("/runs/{run_id}")
def delete_run_record(run_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        if run_id in runs and runs[run_id]["status"] in {"running", "planning", "waiting"}:
            raise HTTPException(status_code=409, detail="Stop the run before deleting it")
        runs.pop(run_id, None)
    if not store.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Unknown run")
    emit("run.deleted", {"run_id": run_id})
    return {"status": "deleted"}


@app.get("/runs/{run_id}/shards/{shard_index}.zip")
def get_shard(run_id: str, shard_index: int, _: str = Depends(require_mesh_token)):
    """The v4 shard archive, for agents that do not cache images.

    Built on first request and kept for the round, so a v5-only mesh never
    pays for it, and a speculative clone of the same shard reuses it.
    """
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        shards = run.get("shards") or []
        if shard_index < 0 or shard_index >= len(shards):
            raise HTTPException(status_code=404, detail="Unknown shard")
        shard = shards[shard_index]
        round_dir = store.run_dir(run_id) / ("round_%d" % run["current_round"])
        dirs = dict(run["shard_dirs"])
        val_images = [Path(path) for path in run.get("shard_val") or []]
        class_names = list(run["class_names"])
    target = round_dir / ("shard_%d.zip" % shard_index)
    with _shard_zip_lock:
        if not target.is_file():
            workdir = round_dir / ("build_%d" % shard_index)
            workdir.mkdir(parents=True, exist_ok=True)
            built = sharding.materialize_shard_zip(dirs, shard["files"], val_images, class_names, workdir)
            os.replace(built, target)
            shutil.rmtree(workdir, ignore_errors=True)
    with state_lock:
        for batch in batches.values():
            if batch["run_id"] == run_id and batch["shard_index"] == shard_index and batch["status"] == "assigned":
                batch["shard_bytes"] = target.stat().st_size
    return FileResponse(target, media_type="application/zip", filename=target.name)


def _charge_weights_download(run_id: str, size: int) -> None:
    """Attribute a global-weights download to the shards in flight on this run."""
    if not size:
        return
    for batch in batches.values():
        if batch["run_id"] == run_id and batch["status"] == "assigned" and not batch.get("weights_in_bytes"):
            batch["weights_in_bytes"] = size


@app.get("/runs/{run_id}/weights.bin")
def get_run_weights_binary(run_id: str, _: str = Depends(require_mesh_token)):
    """Current global weights as raw bytes. 204 before the first aggregation."""
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        weights = run.get("weights")
        round_index = run["current_round"]
        _charge_weights_download(run_id, len(weights or b""))
    headers = {"X-Round": str(round_index)}
    if not weights:
        return Response(status_code=204, headers=headers)
    return Response(content=weights, media_type="application/octet-stream", headers=headers)


@app.get("/runs/{run_id}/weights")
def get_run_weights(run_id: str, _: str = Depends(require_mesh_token)):
    """The v4 wire format: current global weights as base64 inside JSON."""
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        weights = run.get("weights")
        _charge_weights_download(run_id, len(weights or b""))
        return {
            "job_id": run_id,
            "base_model": run["base_model"],
            "weights_b64": base64.b64encode(weights).decode("ascii") if weights else None,
            "round": run["current_round"],
        }


@app.get("/runs/{run_id}/artifact")
def download_artifact(run_id: str, _: str = Depends(require_mesh_token)):
    """Download the aggregated global weights as a .pt file."""
    path = store.run_dir(run_id) / "global.pt"
    if not path.is_file():
        record = store.get_run(run_id)
        if record is None and run_id not in runs:
            raise HTTPException(status_code=404, detail="No artifact for this run yet")
        raise HTTPException(status_code=404, detail="This run has not produced weights yet")
    return FileResponse(path, media_type="application/octet-stream", filename="%s.pt" % run_id)


# ---------------------------------------------------------------------------
# Round machinery
# ---------------------------------------------------------------------------


def _start_round(run_id: str) -> None:
    """Plan one round and queue its shards. No files are copied."""
    policy = current_policy()

    with state_lock:
        run = runs.get(run_id)
        if run is None or run["status"] in {"stopped", "failed", "done"}:
            return
        _refresh_liveness(time.time())
        key = run.get("workload") or workload_key(run["base_model"], run["imgsz"])
        # Plan against this run's workload, so a machine measured at 640 px is
        # not planned with its 320 px speed.
        pool = [_with_workload(node, key) for node in nodes.values()]
        round_index = run["current_round"]
        total_samples = run["total_samples"]
        mode = run["mode"]
        allowlist = run.get("node_ids")
        wanted_backends = set(run.get("backends") or [])
        strategy = run.get("partition_strategy", PARTITION_PROPORTIONAL)

    candidates = [node for node in pool if node.get("active") and node.get("supports_training", True)]
    # Machines mid-way through someone else's shard are not idle.
    candidates = [node for node in candidates if not _busy_elsewhere(node["node_id"], run_id)]

    if allowlist:
        wanted = set(allowlist)
        candidates = [node for node in candidates if node["node_id"] in wanted]
    if wanted_backends:
        candidates = [node for node in candidates if node.get("backend") in wanted_backends]

    if mode == "solo" and candidates:
        mesh = mesh_reference(candidates)
        candidates = [max(candidates, key=lambda node: fitness(node, mesh, policy))]

    plan = plan_round(candidates, total_samples, policy, strategy=strategy)

    if not plan.assignments:
        with state_lock:
            run["status"] = "waiting"
            run["plan"] = plan.as_dict()
        reason = "no eligible worker is online"
        if allowlist:
            reason = "none of the %d requested machines are eligible" % len(allowlist)
        elif wanted_backends:
            reason = "no eligible %s machine is online" % " or ".join(sorted(wanted_backends))
        emit("run.waiting", {"run_id": run_id, "reason": reason, "rejected": plan.rejected})
        return

    dataset = store.get_dataset(run["dataset_id"])
    if dataset is None:
        raise RuntimeError("The dataset for this run was deleted")

    sizes = [assignment.samples for assignment in plan.assignments]

    # Shards are re-drawn every round, even when the plan did not move. The seed
    # is the round index, so each round reshuffles the pool before splitting it.
    # Reusing last round's split would hand every worker the same images every
    # time, which is exactly the class-correlated split that biases local
    # gradients before aggregation sees them.
    run_manifest = run.get("dataset_manifest")
    planned = sharding.plan_shard_lists(
        Path(dataset["extracted_path"]),
        sizes,
        seed=round_index + int(run.get("seed") or 0) * 1009,
        manifest=Path(run_manifest) if run_manifest else None,
        splits=run.get("dataset_splits"),
    )
    dirs = {name: (str(value) if value is not None else None) for name, value in planned["dirs"].items()}
    shards = [
        {"shard_index": index, "files": files, "train_count": len(files)}
        for index, files in enumerate(planned["groups"])
    ]
    _prune_old_rounds(run_id, round_index)

    now = time.time()
    created: List[dict] = []
    by_id = {node["node_id"]: node for node in pool}

    for assignment, shard in zip(plan.assignments, shards):
        node = by_id.get(assignment.node_id, {})
        # A node that has never finished a round of this workload is still
        # paying setup costs, so it gets the cold-start grace period rather
        # than a deadline derived from an estimate it has not earned.
        cold = float(node.get("throughput_sps") or 0.0) <= 0.0
        deadline = deadline_for(assignment.predicted_seconds, policy, cold=cold)
        batch_id = uuid.uuid4().hex[:12]
        created.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "node_id": assignment.node_id,
                "backend": node.get("backend"),
                "round_index": round_index,
                "shard_index": shard["shard_index"],
                "samples": shard["train_count"],
                "status": "queued",
                "tier": assignment.tier,
                "fitness": assignment.fitness,
                "predicted_seconds": assignment.predicted_seconds,
                "predicted_fixed_seconds": assignment.fixed_seconds,
                "soft_deadline_seconds": deadline.soft_seconds,
                "hard_deadline_seconds": deadline.hard_seconds,
                "queued_at": now,
                "assigned_at": None,
                "finished_at": None,
                "elapsed_seconds": None,
                "memory_mb": max(1024, run["batch_size"] * 160),
                "max_batch_size": _node_batch_ceiling(node, run["batch_size"]),
                "device_memory_mb": node.get("gpu_memory_mb", 0),
                "unified_memory": bool((node.get("capability") or {}).get("unified_memory")),
                "weights": None,
                "metrics": None,
                "error": None,
                "speculative_for": None,
                "shard_bytes": 0,
                "weights_in_bytes": 0,
            }
        )

    with state_lock:
        run["status"] = "running"
        run["plan"] = plan.as_dict()
        run["shards"] = shards
        run["shard_dirs"] = dirs
        run["shard_val"] = [str(path) for path in planned["val"]]
        run["shard_sizes"] = sizes
        run["round_started_at"] = now
        run["last_progress_at"] = now
        for batch in created:
            batches[batch["batch_id"]] = batch

    emit(
        "round.started",
        {
            "run_id": run_id,
            "round": round_index,
            "total_rounds": run["rounds"],
            "plan": plan.as_dict(),
        },
    )


def _busy_elsewhere(node_id: str, run_id: str) -> bool:
    """Is this machine holding a shard for a different run right now?"""
    with state_lock:
        return any(
            batch.get("node_id") == node_id
            and batch["run_id"] != run_id
            and batch["status"] in {"queued", "assigned"}
            for batch in batches.values()
        )


def _prune_old_rounds(run_id: str, keep_round: int) -> None:
    """Shard copies are the largest thing on disk. Keep only the live round."""
    root = store.run_dir(run_id)
    for child in root.glob("round_*"):
        if child.name != ("round_%d" % keep_round):
            shutil.rmtree(child, ignore_errors=True)


def _supersede_twins_locked(winner: dict) -> None:
    """Retire the other half of a speculative pair once one side finishes."""
    for other in batches.values():
        if other is winner:
            continue
        if other["run_id"] != winner["run_id"]:
            continue
        if other["round_index"] != winner["round_index"]:
            continue
        if other["shard_index"] != winner["shard_index"]:
            continue
        if other["status"] not in {"queued", "assigned"}:
            continue
        other["status"] = "superseded"
        other["finished_at"] = time.time()
        node = nodes.get(other["node_id"])
        if node is not None:
            node["active_batches"] = 0
            node["allocated_memory_mb"] = 0
        emit(
            "shard.superseded",
            {
                "run_id": other["run_id"],
                "batch_id": other["batch_id"],
                "node_id": other["node_id"],
                "winner": winner["batch_id"],
            },
        )


def _round_batches(run_id: str, round_index: int) -> List[dict]:
    return [
        batch
        for batch in batches.values()
        if batch["run_id"] == run_id and batch["round_index"] == round_index
    ]


def _maybe_close_round(run_id: str) -> None:
    """Close the barrier once every shard has resolved one way or another."""
    with state_lock:
        run = runs.get(run_id)
        if run is None or run["status"] not in {"running", "waiting"}:
            return
        round_index = run["current_round"]
        round_batches = _round_batches(run_id, round_index)
        if not round_batches:
            return
        if any(batch["status"] in {"queued", "assigned"} for batch in round_batches):
            return
        if run.get("closing"):
            return
        run["closing"] = True

        # Count each shard index once. A speculative pair is two batches for one
        # slice of the dataset, so counting both would double the round's
        # sample total and understate how much data was actually lost.
        done = [
            batch
            for batch in round_batches
            if batch["status"] == "done" and batch.get("weights")
        ]
        finished_shards = {batch["shard_index"] for batch in done}
        lost = [
            batch
            for batch in round_batches
            if batch["status"] in {"failed", "dropped"}
            and batch["shard_index"] not in finished_shards
        ]
        dropped_samples = sum(batch["samples"] for batch in _unique_by_shard(lost))
        total = dropped_samples + sum(batch["samples"] for batch in done)
        policy = current_policy()

    if not done:
        _fail_round(run_id, "every shard in this round failed or was dropped")
        return
    if should_abort_round(dropped_samples, total, policy):
        _fail_round(
            run_id,
            "%d of %d samples were lost to offline workers, which is too much for a valid aggregation"
            % (dropped_samples, total),
        )
        return

    supervise(
        aggregator.submit(_aggregate_round, run_id, round_index),
        "aggregation of run %s round %d" % (run_id, round_index),
        lambda exc: _fail_run(run_id, "aggregation failed: %s: %s" % (type(exc).__name__, exc)),
    )


def _unique_by_shard(items: List[dict]) -> List[dict]:
    seen: Dict[int, dict] = {}
    for item in items:
        seen.setdefault(item["shard_index"], item)
    return list(seen.values())


def _fail_round(run_id: str, reason: str) -> None:
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        attempts = run.get("round_attempts", 0) + 1
        run["round_attempts"] = attempts
        run["closing"] = False
        should_retry = attempts < MAX_ROUND_ATTEMPTS

    emit("round.failed", {"run_id": run_id, "reason": reason, "retrying": should_retry})

    if should_retry:
        _clear_round(run_id)
        _start_round(run_id)
        return

    with state_lock:
        run = runs.get(run_id)
        if run is not None:
            run["status"] = "failed"
            run["error"] = reason
            run["finished_at"] = time.time()
    emit("run.failed", {"run_id": run_id, "error": reason})
    _archive_run(run_id)


def _clear_round(run_id: str, round_index: Optional[int] = None) -> None:
    """Drop the batches belonging to one finished round.

    The round must be named explicitly. This used to read run["current_round"],
    but aggregation increments that *before* clearing, so it deleted the round
    about to start, which was empty, and left the finished round's batches in
    memory forever. Every later round then re-scanned them, and a long sweep
    accumulated batches without bound.
    """
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        target = run["current_round"] if round_index is None else round_index
        for batch_id in [b["batch_id"] for b in _round_batches(run_id, target)]:
            batches.pop(batch_id, None)


def _shard_breakdown(batch: dict) -> Dict[str, float]:
    """Where one shard's time went, from the worker's phase timings."""
    metrics = batch.get("metrics") or {}
    elapsed = float(batch.get("elapsed_seconds") or 0.0)
    epoch = float(metrics.get("epoch_seconds") or 0.0)
    train = float(metrics.get("train_seconds") or 0.0)
    compute = epoch if epoch > 0 else train
    total = float(metrics.get("total_seconds") or 0.0)
    if "download_seconds" in metrics:
        transfer = float(metrics.get("download_seconds") or 0.0) + max(0.0, elapsed - total)
    else:
        # A v4 worker reported only its training call; the rest was transfer
        # and setup together.
        transfer = max(0.0, elapsed - train)
    return {
        "elapsed": elapsed,
        "compute": compute,
        "overhead": max(0.0, elapsed - compute) if compute > 0 else 0.0,
        "transfer": transfer,
    }


def _aggregate_round(run_id: str, round_index: int) -> None:
    """Sample-weighted FedAvg. Runs off the event loop because torch blocks."""
    started = time.time()
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        round_batches = _round_batches(run_id, round_index)
        done = _unique_by_shard(
            [batch for batch in round_batches if batch["status"] == "done" and batch.get("weights")]
        )
        finished_shards = {batch["shard_index"] for batch in done}
        lost_shards = {
            batch["shard_index"]
            for batch in round_batches
            if batch["status"] in {"failed", "dropped"} and batch["shard_index"] not in finished_shards
        }
        speculated = sum(1 for batch in round_batches if batch.get("speculative_for"))
        results = [
            {
                "batch_id": batch["batch_id"],
                "samples": batch["samples"],
                "reliability": (nodes.get(batch["node_id"]) or {}).get("reliability", 1.0),
                "weights": batch["weights"],
            }
            for batch in done
        ]
        round_started_at = run.get("round_started_at", started)
        plan = run.get("plan") or {}

    try:
        weights = aggregation_weights(results)
        aggregated = aggregation.aggregate(results, weights)
    except Exception as exc:
        _fail_round(run_id, "aggregation failed: %s: %s" % (type(exc).__name__, exc))
        return

    aggregation_seconds = time.time() - started
    breakdown = {batch["batch_id"]: _shard_breakdown(batch) for batch in done}
    shard_times = [breakdown[batch["batch_id"]]["elapsed"] for batch in done]
    makespan = max(shard_times, default=0.0)
    fastest = min(shard_times, default=0.0)
    serial_estimate = sum(shard_times)
    wall_clock = time.time() - round_started_at

    # Communication accounting. Every worker pulls its images and the current
    # global weights, then pushes its updated weights back. v5 workers report
    # what they actually downloaded, which after the first round is mostly
    # just the weights, because their image cache already holds the shard.
    bytes_down = 0
    for batch in done:
        metrics = batch.get("metrics") or {}
        if metrics.get("bytes_in") is not None:
            bytes_down += int(metrics.get("bytes_in") or 0)
        else:
            bytes_down += int(batch.get("shard_bytes") or 0) + int(batch.get("weights_in_bytes") or 0)
    bytes_up = sum(len(batch.get("weights") or b"") for batch in done)
    comm_bytes = bytes_down + bytes_up
    comm_seconds = sum(item["transfer"] for item in breakdown.values())
    overhead_seconds = [item["overhead"] for item in breakdown.values() if item["overhead"] > 0]

    by_backend: Dict[str, dict] = {}
    for batch in done:
        backend = batch.get("backend") or (batch.get("metrics") or {}).get("backend") or "unknown"
        entry = by_backend.setdefault(
            backend, {"workers": 0, "samples": 0, "seconds": 0.0, "compute_seconds": 0.0, "weight": 0.0}
        )
        entry["workers"] += 1
        entry["samples"] += batch["samples"]
        entry["seconds"] = round(entry["seconds"] + breakdown[batch["batch_id"]]["elapsed"], 2)
        entry["compute_seconds"] = round(entry["compute_seconds"] + breakdown[batch["batch_id"]]["compute"], 2)
        entry["weight"] = round(entry["weight"] + weights.get(batch["batch_id"], 0.0), 4)

    record = {
        "round": round_index,
        "workers": len(done),
        "samples": sum(batch["samples"] for batch in done),
        # Distinct slices of the dataset that were lost, not batch records: a
        # speculative clone that lost its race is not a dropped shard. Leg 1
        # reported a dropped shard in a single-worker round because of this.
        "dropped_shards": len(lost_shards),
        "speculated_shards": speculated,
        "makespan_seconds": round(makespan, 2),
        "fastest_seconds": round(fastest, 2),
        "straggler_gap_seconds": round(makespan - fastest, 2),
        "aggregation_seconds": round(aggregation_seconds, 2),
        "wall_clock_seconds": round(wall_clock, 2),
        "imbalance": round(imbalance(shard_times), 4),
        "predicted_imbalance": float(plan.get("predicted_imbalance") or 0.0),
        "predicted_makespan_seconds": float(plan.get("predicted_makespan_seconds") or 0.0),
        "comm_bytes": comm_bytes,
        "comm_seconds": round(comm_seconds, 2),
        "comm_fraction": round(comm_seconds / (serial_estimate or 1.0), 4),
        "mean_overhead_seconds": round(sum(overhead_seconds) / len(overhead_seconds), 2) if overhead_seconds else None,
        "strategy": run.get("partition_strategy", PARTITION_PROPORTIONAL),
        "by_backend": by_backend,
        # Serial time is the sum of the shard times actually observed, which is
        # what one machine would have spent doing all of this work at the
        # measured per-shard rates.
        "serial_estimate_seconds": round(serial_estimate, 2),
        "speedup": round(serial_estimate / wall_clock, 3) if wall_clock > 0 else 0.0,
        "efficiency": round(efficiency(serial_estimate / wall_clock if wall_clock else 0.0, max(1, len(done))), 3),
        "shards": [
            {
                "node_id": batch["node_id"],
                "node_name": (nodes.get(batch["node_id"]) or {}).get("display_name"),
                "backend": batch.get("backend"),
                "samples": batch["samples"],
                "seconds": round(breakdown[batch["batch_id"]]["elapsed"], 2),
                "compute_seconds": round(breakdown[batch["batch_id"]]["compute"], 2),
                "overhead_seconds": round(breakdown[batch["batch_id"]]["overhead"], 2),
                "predicted_seconds": batch["predicted_seconds"],
                "predicted_fixed_seconds": batch.get("predicted_fixed_seconds"),
                "tier": batch["tier"],
                "weight": round(weights.get(batch["batch_id"], 0.0), 4),
                "metrics": batch.get("metrics"),
            }
            for batch in done
        ],
    }

    accuracy = _evaluate_round(run_id, aggregated, round_index)
    if accuracy:
        record["accuracy"] = accuracy

    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        run["weights"] = aggregated
        run["comm_bytes_total"] = int(run.get("comm_bytes_total", 0)) + comm_bytes
        if accuracy and accuracy.get("ok"):
            run["eval_seconds_total"] = float(run.get("eval_seconds_total", 0.0)) + float(
                accuracy.get("seconds") or 0.0
            )
            run["accuracy_history"].append(
                {
                    "round": round_index,
                    "map50": accuracy.get("map50"),
                    "map50_95": accuracy.get("map50_95"),
                    # Cumulative *training* seconds, excluding evaluation. This
                    # is the x-axis for time-to-accuracy.
                    "train_seconds": round(
                        sum(item.get("wall_clock_seconds", 0.0) for item in run["round_history"])
                        + record["wall_clock_seconds"],
                        2,
                    ),
                }
            )
        run["round_history"].append(record)
        run["last_progress_at"] = time.time()
        run["current_round"] = round_index + 1
        run["round_attempts"] = 0
        run["closing"] = False
        finished = run["current_round"] >= run["rounds"]
        if finished:
            run["status"] = "done"
            run["finished_at"] = time.time()

    _clear_round(run_id, round_index)
    _write_artifact(run_id, aggregated)
    _persist_nodes([batch["node_id"] for batch in done])
    emit("round.completed", {"run_id": run_id, **{k: v for k, v in record.items() if k != "shards"}})

    if finished:
        emit("run.completed", {"run_id": run_id, "summary": _run_summary(runs[run_id])})
        _archive_run(run_id)
    else:
        _start_round(run_id)


def _evaluate_round(run_id: str, weights: bytes, round_index: int) -> Optional[dict]:
    """Score the aggregated model, if this run asked for it.

    Runs on the aggregator thread, after the round's wall clock has been
    recorded, so evaluation never inflates a training measurement.
    """
    with state_lock:
        run = runs.get(run_id)
        if run is None or not run.get("evaluate"):
            return None
        dataset_root = Path(run["dataset_root"])
        dataset_splits = run.get("dataset_splits")
        class_names = run["class_names"]
        imgsz = run["imgsz"]
        base_model = run["base_model"]

    try:
        workdir = store.run_dir(run_id) / "eval"
        eval_yaml = evaluation.write_eval_yaml(
            dataset_root, workdir / "eval.yaml", class_names, splits=dataset_splits
        )
        result = evaluation.evaluate_weights(
            weights_b64=weights,
            base_model_path=store.MODELS_DIR / base_model,
            eval_yaml=eval_yaml,
            imgsz=imgsz,
            workdir=workdir,
        )
    except Exception as exc:
        result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "seconds": 0.0}

    emit(
        "round.evaluated",
        {
            "run_id": run_id,
            "round": round_index,
            "ok": result.get("ok"),
            "map50": result.get("map50"),
            "seconds": result.get("seconds"),
            "error": result.get("error"),
        },
    )
    return result


def _fail_run(run_id: str, reason: str) -> None:
    """Mark a run failed from outside the round machinery."""
    with state_lock:
        run = runs.get(run_id)
        if run is None or run["status"] in {"done", "failed", "stopped"}:
            return
        run["status"] = "failed"
        run["error"] = reason
        run["finished_at"] = time.time()
        run["closing"] = False
        for batch in batches.values():
            if batch["run_id"] == run_id and batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = reason
    emit("run.failed", {"run_id": run_id, "error": reason})
    _archive_run(run_id)


def _fail_suite(suite_id: str, reason: str) -> None:
    """Mark a sweep failed and release the slot, so the next one can start."""
    try:
        suite = benchmark.read_suite(_suite_dir(suite_id))
        if suite is not None:
            suite["status"] = "failed"
            suite["error"] = reason
            suite["finished_at"] = time.time()
            suite["current_trial"] = None
            benchmark.write_suite(_suite_dir(suite_id), suite)
    finally:
        with suite_lock:
            if active_suite["id"] == suite_id:
                active_suite["id"] = None
                active_suite["abort"] = False
    emit("suite.finished", {"suite_id": suite_id, "status": "failed", "error": reason})


def _write_artifact(run_id: str, weights: bytes) -> None:
    try:
        path = store.run_dir(run_id) / "global.pt"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(weights)
        os.replace(temporary, path)
    except Exception:
        # An unwritable artifact must not fail a run that otherwise succeeded.
        pass


def _archive_run(run_id: str) -> None:
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        record = _run_summary(run)
        record["round_history"] = run["round_history"]
        record["plan"] = run.get("plan")
        record["error"] = run.get("error")
        record["notes"] = run.get("notes")
        record["class_names"] = run.get("class_names")
    store.put_run(record)
    try:
        (store.run_dir(run_id) / "summary.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _backend_totals(history: Sequence[dict]) -> Dict[str, dict]:
    totals: Dict[str, dict] = {}
    for item in history:
        for backend, entry in (item.get("by_backend") or {}).items():
            total = totals.setdefault(backend, {"samples": 0, "seconds": 0.0, "compute_seconds": 0.0, "rounds": 0})
            total["samples"] += int(entry.get("samples") or 0)
            total["seconds"] = round(total["seconds"] + float(entry.get("seconds") or 0.0), 2)
            total["compute_seconds"] = round(total["compute_seconds"] + float(entry.get("compute_seconds") or 0.0), 2)
            total["rounds"] += 1
    for total in totals.values():
        total["throughput_sps"] = round(total["samples"] / total["compute_seconds"], 3) if total["compute_seconds"] else None
    return totals


def _run_summary(run: dict) -> dict:
    history = run.get("round_history") or []
    total_wall = sum(item.get("wall_clock_seconds", 0.0) for item in history)
    total_serial = sum(item.get("serial_estimate_seconds", 0.0) for item in history)
    peak_workers = max((item.get("workers", 0) for item in history), default=0)
    accuracy = run.get("accuracy_history") or []
    latest = accuracy[-1] if accuracy else None
    best = max((item.get("map50") or 0.0 for item in accuracy), default=0.0)
    imbalances = [item.get("imbalance", 0.0) for item in history if item.get("imbalance") is not None]
    return {
        "id": run["id"],
        "name": run["name"],
        "status": run["status"],
        "mode": run["mode"],
        "created_at": run["created_at"],
        "finished_at": run.get("finished_at"),
        "dataset_id": run["dataset_id"],
        "dataset_name": run["dataset_name"],
        "base_model": run["base_model"],
        "rounds": run["rounds"],
        "current_round": run["current_round"],
        "imgsz": run["imgsz"],
        "batch_size": run["batch_size"],
        "total_samples": run["total_samples"],
        "wall_clock_seconds": round(total_wall, 2),
        "serial_estimate_seconds": round(total_serial, 2),
        "speedup": round(total_serial / total_wall, 3) if total_wall > 0 else 0.0,
        "efficiency": round(efficiency(total_serial / total_wall if total_wall else 0.0, max(1, peak_workers)), 3),
        "peak_workers": peak_workers,
        "has_artifact": (store.run_dir(run["id"]) / "global.pt").is_file(),
        "partition_strategy": run.get("partition_strategy", PARTITION_PROPORTIONAL),
        "seed": run.get("seed", 0),
        "node_ids": run.get("node_ids"),
        "backends": run.get("backends"),
        "warmup_mode": run.get("warmup_mode", "every-round"),
        "warmup_epochs": run.get("warmup_epochs"),
        "optimizer": run.get("optimizer", "auto"),
        "lr0": run.get("lr0"),
        "worker_validation": bool(run.get("worker_validation")),
        "evaluate": bool(run.get("evaluate")),
        "reference_stack": run.get("reference_stack"),
        "backend_totals": _backend_totals(history),
        "suite_id": run.get("suite_id"),
        "trial_id": run.get("trial_id"),
        "map50": latest.get("map50") if latest else None,
        "map50_95": latest.get("map50_95") if latest else None,
        "best_map50": round(best, 5) if accuracy else None,
        "accuracy_history": accuracy,
        "eval_seconds_total": round(float(run.get("eval_seconds_total", 0.0)), 2),
        "comm_bytes_total": int(run.get("comm_bytes_total", 0)),
        "mean_imbalance": round(sum(imbalances) / len(imbalances), 4) if imbalances else None,
        "error": run.get("error"),
    }


def _run_view(run_id: str) -> dict:
    with state_lock:
        run = runs[run_id]
        view = _run_summary(run)
        view["round_history"] = run["round_history"]
        view["plan"] = run.get("plan")
        view["error"] = run.get("error")
        view["notes"] = run.get("notes")
        view["class_names"] = run["class_names"]
        now = time.time()
        live = []
        for batch in _round_batches(run_id, run["current_round"]):
            node = nodes.get(batch["node_id"]) or {}
            live.append(
                {
                    "batch_id": batch["batch_id"],
                    "node_id": batch["node_id"],
                    "node_name": node.get("display_name"),
                    "backend": batch.get("backend") or node.get("backend"),
                    "status": batch["status"],
                    "samples": batch["samples"],
                    "tier": batch["tier"],
                    "round": batch["round_index"],
                    "predicted_seconds": batch["predicted_seconds"],
                    "predicted_fixed_seconds": batch.get("predicted_fixed_seconds"),
                    "batch_size": batch.get("resolved_batch_size")
                    or safe_batch_size(
                        device_memory_mb=batch.get("device_memory_mb") or 0,
                        imgsz=run["imgsz"],
                        requested=run["batch_size"],
                        node_max=batch.get("max_batch_size"),
                        unified_memory=bool(batch.get("unified_memory")),
                    ),
                    "elapsed_seconds": round(now - batch["assigned_at"], 1)
                    if batch.get("assigned_at") and batch["status"] == "assigned"
                    else batch.get("elapsed_seconds"),
                    "soft_deadline_seconds": batch["soft_deadline_seconds"],
                    "hard_deadline_seconds": batch["hard_deadline_seconds"],
                    "phase": node.get("phase") if batch["status"] == "assigned" else None,
                    "progress": node.get("progress") if batch["status"] == "assigned" else None,
                    "speculative": bool(batch.get("speculative_for")),
                    "error": batch.get("error"),
                }
            )
        view["live_shards"] = live
        return view


# ---------------------------------------------------------------------------
# Straggler supervisor
# ---------------------------------------------------------------------------


async def _supervisor_loop() -> None:
    """Enforce deadlines and keep waiting runs moving when workers arrive."""
    while True:
        try:
            await asyncio.sleep(SUPERVISOR_INTERVAL_SECONDS)
            await asyncio.get_running_loop().run_in_executor(None, _supervise_once)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep the supervisor alive through any bug
            emit("supervisor.error", {"error": "%s: %s" % (type(exc).__name__, exc)})


def _supervise_once() -> None:
    now = time.time()
    policy = current_policy()
    to_close: List[str] = []
    to_start: List[str] = []
    to_fail: List[tuple] = []

    with state_lock:
        _refresh_liveness(now)

        idle_nodes = [
            node
            for node in nodes.values()
            if node.get("active") and node.get("active_batches", 0) == 0 and node.get("supports_training", True)
        ]

        for batch in list(batches.values()):
            if batch["status"] == "queued":
                node = nodes.get(batch["node_id"])
                if node is None or not node.get("active"):
                    stale = now - float(batch.get("queued_at") or now)
                    if stale > HEARTBEAT_TIMEOUT_SECONDS:
                        batch["status"] = "dropped"
                        batch["error"] = "assigned worker never came online"
                        emit(
                            "shard.dropped",
                            {"run_id": batch["run_id"], "batch_id": batch["batch_id"], "reason": batch["error"]},
                        )
                        to_close.append(batch["run_id"])
                continue

            if batch["status"] != "assigned":
                continue

            elapsed = now - float(batch.get("assigned_at") or now)
            node = nodes.get(batch["node_id"])
            # Use the deadline the round was planned with, so a policy change
            # mid-round cannot retroactively make a running shard late.
            deadline = Deadline(
                soft_seconds=batch["soft_deadline_seconds"],
                hard_seconds=batch["hard_deadline_seconds"],
            )

            if node is None or not node.get("active"):
                batch["status"] = "dropped"
                batch["error"] = "worker went offline mid-round"
                if node is not None:
                    node["reliability"] = update_reliability(node, False, policy)
                    # Release the slot too, otherwise the node keeps reporting a
                    # batch in flight and is never considered idle again.
                    node["active_batches"] = 0
                    node["allocated_memory_mb"] = 0
                emit(
                    "shard.dropped",
                    {"run_id": batch["run_id"], "batch_id": batch["batch_id"], "reason": batch["error"]},
                )
                to_close.append(batch["run_id"])
                continue

            # A backup must be a machine this run is allowed to use. v4 picked
            # any idle machine, so a benchmark trial restricted to two nodes
            # could quietly finish on a third, which is a different experiment.
            run_for_batch = runs.get(batch["run_id"]) or {}
            allowed_ids = set(run_for_batch.get("node_ids") or [])
            allowed_backends = set(run_for_batch.get("backends") or [])
            eligible_backups = [
                candidate
                for candidate in idle_nodes
                if candidate["node_id"] != batch["node_id"]
                and not batch.get("speculated")
                and not batch.get("speculative_for")
                and (not allowed_ids or candidate["node_id"] in allowed_ids)
                and (not allowed_backends or candidate.get("backend") in allowed_backends)
                and float((candidate.get("capability") or {}).get("gflops") or 0.0) > 0
                and run_for_batch.get("mode") != "solo"
            ]
            # The fastest idle machine for this workload makes the best backup.
            workload = run_for_batch.get("workload")
            backup = max(
                eligible_backups,
                key=lambda candidate: float(
                    ((candidate.get("workloads") or {}).get(workload) or {}).get("rate")
                    or (candidate.get("capability") or {}).get("gflops")
                    or 0.0
                ),
                default=None,
            )
            action = straggler_action(elapsed, deadline, backup is not None)

            if action == "drop":
                batch["status"] = "dropped"
                batch["error"] = "missed the hard deadline of %.0fs" % deadline.hard_seconds
                node["reliability"] = update_reliability(node, False, policy)
                node["active_batches"] = 0
                node["allocated_memory_mb"] = 0
                emit(
                    "shard.dropped",
                    {
                        "run_id": batch["run_id"],
                        "batch_id": batch["batch_id"],
                        "node_id": batch["node_id"],
                        "reason": batch["error"],
                    },
                )
                to_close.append(batch["run_id"])
            elif action == "speculate" and backup is not None:
                batch["speculated"] = True
                clone_id = uuid.uuid4().hex[:12]
                clone = dict(batch)
                clone.update(
                    {
                        "batch_id": clone_id,
                        "node_id": backup["node_id"],
                        "backend": backup.get("backend"),
                        "status": "queued",
                        "queued_at": now,
                        "assigned_at": None,
                        "speculative_for": batch["batch_id"],
                        "speculated": False,
                        # Sized for the backup machine, not the straggler.
                        "device_memory_mb": backup.get("gpu_memory_mb", 0),
                        "max_batch_size": _node_batch_ceiling(backup, run_for_batch.get("batch_size")),
                        "unified_memory": bool((backup.get("capability") or {}).get("unified_memory")),
                        "resolved_batch_size": None,
                        "weights": None,
                        "metrics": None,
                        "error": None,
                        "shard_bytes": 0,
                        "weights_in_bytes": 0,
                    }
                )
                batches[clone_id] = clone
                idle_nodes = [n for n in idle_nodes if n["node_id"] != backup["node_id"]]
                emit(
                    "shard.speculated",
                    {
                        "run_id": batch["run_id"],
                        "original": batch["batch_id"],
                        "clone": clone_id,
                        "from_node": batch["node_id"],
                        "to_node": backup["node_id"],
                        "elapsed_seconds": round(elapsed, 1),
                    },
                )

        # A node that has been silent for many timeouts is gone, not slow. Keeping
        # it would leave a phantom in every device list and in the plan preview.
        for node_id, node in list(nodes.items()):
            silent_for = now - float(node.get("last_seen") or now)
            if silent_for > HEARTBEAT_TIMEOUT_SECONDS * NODE_EVICTION_MULTIPLE:
                if any(
                    batch.get("node_id") == node_id and batch["status"] in {"queued", "assigned"}
                    for batch in batches.values()
                ):
                    continue
                nodes.pop(node_id, None)
                try:
                    store.put_node_stats({node_id: node})
                except Exception:
                    pass
                emit(
                    "node.left",
                    {"node_id": node_id, "name": node.get("display_name"), "reason": "stopped responding"},
                )

        for run in runs.values():
            if run["status"] == "waiting" and idle_nodes:
                to_start.append(run["id"])

        # Watchdog for the gap between rounds.
        #
        # A run reported "running" with nothing in flight is supposed to be
        # mid-handover, which takes milliseconds. One was observed sitting there
        # indefinitely: a round finished, the call that plans the next one
        # failed, and because nothing inspected that thread's Future the error
        # was never reported. The dashboard showed "planning the next round"
        # forever. Exceptions are surfaced now, but a stuck run should recover
        # rather than rely on somebody reading a log, so this re-plans it and
        # gives up loudly if re-planning does not take.
        for run in runs.values():
            if run["status"] != "running" or run.get("closing"):
                continue
            if any(
                batch["run_id"] == run["id"] and batch["status"] in {"queued", "assigned"}
                for batch in batches.values()
            ):
                continue
            idle_for = now - float(run.get("last_progress_at") or now)
            if idle_for < STALL_GRACE_SECONDS:
                continue

            recoveries = int(run.get("stall_recoveries", 0))
            if recoveries >= MAX_STALL_RECOVERIES:
                to_fail.append(
                    (
                        run["id"],
                        "stalled between rounds for %.0fs and did not recover after %d attempts"
                        % (idle_for, recoveries),
                    )
                )
                continue

            run["stall_recoveries"] = recoveries + 1
            run["last_progress_at"] = now
            emit(
                "run.stalled",
                {
                    "run_id": run["id"],
                    "round": run["current_round"],
                    "idle_seconds": round(idle_for, 1),
                    "attempt": recoveries + 1,
                },
            )
            to_start.append(run["id"])

    for run_id in dict.fromkeys(to_close):
        _maybe_close_round(run_id)
    for run_id in dict.fromkeys(to_start):
        # A failure here must not kill the supervisor loop, which everything
        # else depends on.
        try:
            _start_round(run_id)
        except Exception as exc:
            _fail_run(run_id, "could not plan the next round: %s: %s" % (type(exc).__name__, exc))
    for run_id, reason in to_fail:
        _fail_run(run_id, reason)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ImportDatasetRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=60)
    make_default: bool = True


_import_state: Dict[str, Any] = {"running": False, "key": None, "message": None, "error": None}


def _import_standard(key: str, make_default: bool) -> None:
    """Fetch a catalogue dataset and register it. Runs on the sweep thread."""

    def progress(message: str) -> None:
        _import_state["message"] = message
        emit("dataset.importing", {"key": key, "message": message})

    try:
        resolved = datasets_std.download(key, progress=progress)
        splits = {
            "root": resolved["root"],
            "train_images": resolved["train_images"],
            "train_labels": resolved["train_labels"],
            "val_images": resolved["val_images"],
            "val_labels": resolved["val_labels"],
        }
        listing = sharding.list_split_images(Path(resolved["root"]), splits=splits)

        record = {
            "id": "std-%s" % key.lower().replace(".", "-"),
            "name": resolved["name"],
            "filename": "%s.yaml" % key,
            "created_at": time.time(),
            "bytes": 0,
            "train_count": len(listing["train"]),
            "val_count": len(listing["val"]),
            "class_names": resolved["class_names"],
            "extracted_path": resolved["root"],
            "archive_path": None,
            "splits": splits,
            "source": "standard",
            "standard_key": key,
        }
        store.put_dataset(record, make_default=make_default)
        _import_state.update({"message": "Imported %s" % resolved["name"], "error": None})
        emit(
            "dataset.added",
            {"id": record["id"], "name": record["name"], "images": record["train_count"]},
        )
    except Exception as exc:
        _import_state["error"] = "%s: %s" % (type(exc).__name__, exc)
        emit("dataset.import_failed", {"key": key, "error": _import_state["error"]})
    finally:
        _import_state["running"] = False
        _import_state["key"] = None


@app.get("/datasets/standard")
def list_standard_datasets(_: str = Depends(require_mesh_token)):
    return {"catalogue": datasets_std.catalogue(), "import": dict(_import_state)}


@app.post("/datasets/standard")
def import_standard_dataset(req: ImportDatasetRequest, _: str = Depends(require_mesh_token)):
    """Download a standard dataset and register it.

    A 20 GB download cannot block an HTTP request, so this returns immediately
    and reports progress through the event stream.
    """
    if req.key not in datasets_std.CATALOGUE_BY_KEY:
        raise HTTPException(status_code=404, detail="Unknown dataset %r" % req.key)
    if not aggregation.torch_ready():
        raise HTTPException(status_code=503, detail="The training plane is still installing.")
    if _import_state["running"]:
        raise HTTPException(status_code=409, detail="Another import is already running.")
    if active_suite["id"]:
        raise HTTPException(status_code=409, detail="Finish or stop the running sweep first.")

    _import_state.update({"running": True, "key": req.key, "message": "Starting", "error": None})
    supervise(
        sweeper.submit(_import_standard, req.key, req.make_default),
        "import of %s" % req.key,
        lambda exc: _import_state.update({"running": False, "error": str(exc)}),
    )
    return {"status": "importing", "key": req.key}


@app.get("/datasets")
def get_datasets(_: str = Depends(require_mesh_token)):
    return {"datasets": store.list_datasets()}


@app.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    make_default: bool = Form(default=False),
    _: str = Depends(require_mesh_token),
):
    """Accept a YOLO dataset ZIP, validate it, and register it for the mesh."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip archive")

    dataset_id = uuid.uuid4().hex[:12]
    target = store.dataset_dir(dataset_id)
    extracted = target / "data"
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / "dataset.zip"

    size = 0
    with archive_path.open("wb") as stream:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            stream.write(chunk)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)
        listing = sharding.list_split_images(extracted)
        train_count = len(listing["train"])
        val_count = len(listing["val"])
        if train_count == 0:
            raise ValueError("The archive contains no training images")
        class_names = sharding.infer_class_names(extracted) or ["object"]
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Could not read this dataset: %s. Expected images/train and labels/train inside the zip."
            % exc,
        )

    record = {
        "id": dataset_id,
        "name": name.strip() or file.filename,
        "filename": file.filename,
        "created_at": time.time(),
        "bytes": size,
        "train_count": train_count,
        "val_count": val_count,
        "class_names": class_names,
        "extracted_path": str(extracted),
        "archive_path": str(archive_path),
    }
    store.put_dataset(record, make_default=make_default)
    emit("dataset.added", {"id": dataset_id, "name": record["name"], "images": train_count})
    return record


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Reject archives that try to escape the extraction root."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError("archive contains a path outside the extraction root")
    archive.extractall(root)


@app.post("/datasets/{dataset_id}/default")
def make_default_dataset(dataset_id: str, _: str = Depends(require_mesh_token)):
    if store.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    store.set_default_dataset(dataset_id)
    emit("dataset.default", {"id": dataset_id})
    return {"status": "ok"}


@app.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        in_use = any(
            run["dataset_id"] == dataset_id and run["status"] in {"running", "planning", "waiting"}
            for run in runs.values()
        )
    if in_use:
        raise HTTPException(status_code=409, detail="This dataset is in use by a running job")
    _dirs_cache.clear()
    if not store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Unknown dataset")
    emit("dataset.removed", {"id": dataset_id})
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Models, policy, mesh state, events
# ---------------------------------------------------------------------------


@app.get("/models")
def list_models(_: str = Depends(require_mesh_token)):
    models = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(store.MODELS_DIR.glob("*.pt"))
    ]
    return {"models": models}


@app.get("/models/{model_name}")
def get_model(model_name: str, _: str = Depends(require_mesh_token)):
    path = (store.MODELS_DIR / Path(model_name).name).resolve()
    if path.parent != store.MODELS_DIR.resolve() or path.suffix != ".pt" or not path.is_file():
        raise HTTPException(status_code=404, detail="Model checkpoint not found")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/policy")
def get_policy(_: str = Depends(require_mesh_token)):
    return {"policy": current_policy().as_dict(), "defaults": DEFAULT_POLICY.as_dict()}


@app.put("/policy")
def put_policy(req: PolicyRequest, _: str = Depends(require_mesh_token)):
    allowed = set(vars(DEFAULT_POLICY))
    unknown = set(req.values) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail="Unknown policy keys: %s" % ", ".join(sorted(unknown)))
    store.set_policy(req.values)
    emit("policy.updated", {"values": req.values})
    return {"policy": current_policy().as_dict()}


@app.post("/token/rotate")
def rotate_token(_: str = Depends(require_mesh_token)):
    """Mint a new join token and drop every machine that used the old one.

    This is the revocation story for a LAN mesh: the token is what a worker
    presents to receive dataset shards, so rotating it and clearing the registry
    means an uninvited machine cannot keep pulling data.
    """
    new_token = store.rotate_mesh_token()
    with state_lock:
        removed = list(nodes.keys())
        nodes.clear()
        for batch in batches.values():
            if batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = "the mesh token was rotated"
    emit("mesh.token_rotated", {"removed": len(removed)})
    return {"token": new_token, "removed_nodes": len(removed)}


@app.get("/mesh")
def mesh_state(_: str = Depends(require_mesh_token)):
    """Everything the dashboard needs for a cold render, in one request."""
    policy = current_policy()
    key = _preview_workload()
    node_views = snapshot_nodes(key)
    decisions = admit([n for n in node_views], policy)

    for view in node_views:
        decision = decisions.get(view["node_id"])
        if decision:
            view["tier"] = decision.tier
            view["admission_reason"] = decision.reason

    with state_lock:
        active_runs = [_run_summary(run) for run in runs.values() if run["status"] in {"running", "planning", "waiting"}]
        live_batches = [
            batch for batch in batches.values() if batch["status"] in {"queued", "assigned"}
        ]

    dataset = store.default_dataset()
    preview = plan_round(node_views, int((dataset or {}).get("train_count") or 0), policy)

    online = [n for n in node_views if n.get("active")]
    total_gflops = sum(float((n.get("capability") or {}).get("gflops") or 0.0) for n in online)
    total_memory = sum(int(n.get("gpu_memory_mb") or 0) for n in online)

    # The hardware mix, by vendor. The cross-vendor paper asks this first.
    backends: Dict[str, dict] = {}
    for view in node_views:
        backend = view.get("backend") or "cpu"
        entry = backends.setdefault(
            backend,
            {
                "backend": backend,
                "vendor": VENDOR_OF.get(backend, "cpu"),
                "nodes": 0,
                "online": 0,
                "eligible": 0,
                "gflops": 0.0,
                "memory_mb": 0,
                "throughput_sps": 0.0,
            },
        )
        entry["nodes"] += 1
        if view.get("active"):
            entry["online"] += 1
            entry["gflops"] = round(entry["gflops"] + float((view.get("capability") or {}).get("gflops") or 0.0), 1)
            entry["memory_mb"] += int(view.get("gpu_memory_mb") or 0)
            entry["throughput_sps"] = round(entry["throughput_sps"] + float(view.get("throughput_sps") or 0.0), 2)
        if view.get("tier") in {"full", "probation"}:
            entry["eligible"] += 1

    return {
        "mesh_name": store.load().get("mesh_name", "GradMesh"),
        "mesh_id": store.mesh_id(),
        "version": __version__,
        "protocol": PROTOCOL,
        "reference_stack": REFERENCE_STACK,
        "workload": key,
        "nodes": node_views,
        "backends": sorted(backends.values(), key=lambda item: ("cuda", "xpu", "mps", "cpu").index(item["backend"])
                           if item["backend"] in ("cuda", "xpu", "mps", "cpu") else 9),
        "metrics": {
            "nodes_total": len(node_views),
            "nodes_active": len(online),
            "nodes_training": sum(1 for n in node_views if n.get("active_batches", 0) > 0),
            "nodes_admitted": sum(1 for d in decisions.values() if d.tier != "rejected"),
            "nodes_warned": sum(1 for n in online if n.get("warnings")),
            "total_gflops": round(total_gflops, 1),
            "total_memory_mb": total_memory,
            "active_shards": len(live_batches),
            "stream_subscribers": bus.subscriber_count,
        },
        "active_runs": active_runs,
        "plan_preview": preview.as_dict(),
        "policy": policy.as_dict(),
        "torch_ready": aggregation.torch_ready(),
        "default_dataset": dataset,
    }


@app.get("/events")
async def events(replay: int = Query(default=40, ge=0, le=200), _: str = Depends(require_mesh_token)):
    return StreamingResponse(
        bus.subscribe(replay=replay),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class VisitorRequest(BaseModel):
    visitor_id: str = Field(..., min_length=6, max_length=64)
    name: Optional[str] = None
    platform: Optional[str] = None
    user_agent: Optional[str] = None
    cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu: Optional[str] = None
    webgpu: Optional[bool] = None
    screen: Optional[str] = None


def _prune_visitors_locked(now: float) -> None:
    for visitor_id in [
        key for key, value in visitors.items() if now - value["last_seen"] > VISITOR_TTL_SECONDS
    ]:
        visitors.pop(visitor_id, None)


@app.post("/visitors")
def announce_visitor(req: VisitorRequest, request: Request, _: str = Depends(allow_local_or_token)):
    """A browser on the network says hello.

    Opening the join page is the earliest moment the host can know a device
    exists, and it costs the visitor nothing. The record is deliberately thin:
    what the browser already exposes to any page it loads, plus whether WebGPU
    is available, which is the honest upper bound on what a tab could ever
    contribute without installing the agent.
    """
    now = time.time()
    client_ip = request.client.host if request.client else None

    # The endpoint is reachable without a token, so cap how often one address
    # can write to the list regardless of how many visitor ids it invents.
    bucket = _visitor_rate.setdefault(client_ip or "unknown", [])
    bucket[:] = [stamp for stamp in bucket if now - stamp < VISITOR_RATE_WINDOW]
    if len(bucket) >= VISITOR_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many announcements.")
    bucket.append(now)

    with state_lock:
        _prune_visitors_locked(now)
        existing = visitors.get(req.visitor_id) or {}
        is_new = not existing
        has_agent = any(
            (node.get("capability") or {}).get("host") == req.name for node in nodes.values()
        )
        visitors[req.visitor_id] = {
            "visitor_id": req.visitor_id,
            "name": req.name or "Unnamed device",
            "platform": req.platform,
            "user_agent": req.user_agent,
            "cores": req.cores,
            "memory_gb": req.memory_gb,
            "gpu": req.gpu,
            "webgpu": bool(req.webgpu),
            "screen": req.screen,
            "ip": client_ip,
            "first_seen": existing.get("first_seen", now),
            "last_seen": now,
            "has_agent": has_agent,
        }

    if is_new:
        emit(
            "visitor.arrived",
            {"visitor_id": req.visitor_id, "name": req.name, "ip": client_ip, "gpu": req.gpu},
        )
    return {"status": "ok", "mdns": advertiser.as_dict()}


def _visitor_list_locked(now: float) -> List[dict]:
    _prune_visitors_locked(now)
    return sorted(visitors.values(), key=lambda item: item["last_seen"], reverse=True)


def _run_scan() -> dict:
    try:
        result = discovery.scan_network()
    finally:
        _scan_cache["running"] = False
    _scan_cache["result"] = result
    _scan_cache["at"] = time.time()
    emit("network.scanned", {"devices": len(result.get("devices", []))})
    return result


@app.get("/discover")
def discover(refresh: bool = Query(default=False), _: str = Depends(require_mesh_token)):
    """Everything the Discover page needs: how to reach this host, who is on the
    network, who is already contributing, and who is merely looking."""
    now = time.time()
    cached = _scan_cache["result"]
    stale = cached is None or (now - float(_scan_cache["at"])) > SCAN_CACHE_SECONDS

    if (refresh or stale) and not _scan_cache["running"]:
        _scan_cache["running"] = True
        if cached is None:
            # Nothing to show yet, so the first caller waits for a real answer
            # rather than being handed an empty page.
            cached = _run_scan()
        else:
            supervise(scanner.submit(_run_scan), "network scan")

    with state_lock:
        _refresh_liveness(now)
        node_summaries = [
            {
                "node_id": node["node_id"],
                "display_name": node.get("display_name"),
                "gpu": node.get("gpu"),
                "backend": node.get("backend"),
                "active": node.get("active"),
                "host": (node.get("capability") or {}).get("host"),
            }
            for node in nodes.values()
        ]
        visitor_records = _visitor_list_locked(now)

    scan = cached or {"devices": [], "subnet": None, "self_ip": discovery.local_ipv4()}
    visitor_ips = {visitor["ip"] for visitor in visitor_records if visitor.get("ip")}
    agent_ips = {
        visitor["ip"] for visitor in visitor_records if visitor.get("ip") and visitor.get("has_agent")
    }

    devices = []
    for device in scan.get("devices", []):
        entry = dict(device)
        entry["is_visitor"] = device["ip"] in visitor_ips
        entry["is_member"] = bool(device.get("is_coordinator")) or device["ip"] in agent_ips
        devices.append(entry)

    return {
        "mdns": advertiser.as_dict(),
        "addresses": {
            "lan_ip": scan.get("self_ip"),
            "web_port": advertiser.web_port,
            "api_port": advertiser.api_port,
        },
        "scan": {
            "subnet": scan.get("subnet"),
            "at": _scan_cache["at"],
            "running": bool(_scan_cache["running"]),
            "duration_seconds": scan.get("duration_seconds", 0.0),
            "scanned": scan.get("scanned", 0),
        },
        "devices": devices,
        "visitors": visitor_records,
        "nodes": node_summaries,
        "idle_count": sum(
            1
            for device in devices
            if not device["is_this_host"] and not device["is_member"] and not device["is_visitor"]
        ),
    }


@app.post("/discover/scan")
def rescan(_: str = Depends(require_mesh_token)):
    """Force a fresh sweep. Blocks for a couple of seconds by design: someone
    pressed a button and expects the list to be current when it returns."""
    _scan_cache["running"] = True
    return _run_scan()


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

BENCHMARK_DIR = store.STATE_DIR / "benchmarks"


class SuiteRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


def _suite_dir(suite_id: str) -> Path:
    return BENCHMARK_DIR / suite_id


def _ranked_nodes() -> List[dict]:
    """Active, training-capable machines, strongest first."""
    policy = current_policy()
    with state_lock:
        _refresh_liveness(time.time())
        pool = [
            dict(node)
            for node in nodes.values()
            if node.get("active") and node.get("supports_training", True)
        ]
    if not pool:
        return []
    mesh = mesh_reference(pool)
    return sorted(pool, key=lambda node: fitness(node, mesh, policy), reverse=True)


def _ensure_subset(parent: dict, sample_count: int) -> dict:
    """Find or create the dataset subset of this size.

    Subsets are cached by parent and size, so a sweep that visits 1000 images in
    thirty different cells builds the manifest once. Reusing it also means every
    one of those cells trains on exactly the same images, which is required for
    the comparison to be about node count rather than about data.
    """
    if sample_count >= int(parent.get("train_count") or 0):
        return parent

    subset_id = "%s-n%d" % (parent["id"], sample_count)
    existing = store.get_dataset(subset_id)
    if existing and Path(existing.get("manifest_path") or "").is_file():
        return existing

    root = Path(parent["extracted_path"])
    manifest_path = store.dataset_dir(parent["id"]) / ("subset_%d.txt" % sample_count)
    info = sharding.write_subset_manifest(
        root, manifest_path, sample_count, seed=1234, splits=parent.get("splits")
    )

    record = {
        "id": subset_id,
        "name": "%s (%d images)" % (parent["name"], info["train_count"]),
        "filename": parent.get("filename"),
        "created_at": time.time(),
        "bytes": 0,
        "train_count": info["train_count"],
        "val_count": info["val_count"],
        "class_names": parent.get("class_names") or ["object"],
        "extracted_path": parent["extracted_path"],
        "archive_path": parent.get("archive_path"),
        "manifest_path": info["manifest_path"],
        "splits": parent.get("splits"),
        "parent_id": parent["id"],
        "subset_seed": info["seed"],
        "is_subset": True,
    }
    store.put_dataset(record)
    emit("dataset.subset", {"id": subset_id, "images": info["train_count"]})
    return record


def _wait_for_run(run_id: str, timeout_seconds: float) -> str:
    """Block until a run reaches a terminal state. Returns the final status."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if active_suite["abort"]:
            try:
                stop_run(run_id)
            except Exception:
                pass
            return "aborted"
        with state_lock:
            run = runs.get(run_id)
            status = run["status"] if run else "missing"
        if status in {"done", "failed", "stopped", "missing"}:
            return status
        time.sleep(2.0)

    try:
        stop_run(run_id)
    except Exception:
        pass
    return "timeout"


def _execute_suite(suite_id: str) -> None:
    """Run every trial in order. Owns the sweep thread for its whole duration."""
    directory = _suite_dir(suite_id)
    suite = benchmark.read_suite(directory)
    if suite is None:
        return

    config = benchmark.SuiteConfig.from_dict(suite.get("config"))
    specs = [benchmark.TrialSpec(**{k: v for k, v in t.items() if k != "label"}) for t in suite["trials"]]
    done_ids = {r["trial_id"] for r in suite.get("results", [])}

    suite["status"] = "running"
    suite["started_at"] = suite.get("started_at") or time.time()
    benchmark.write_suite(directory, suite)
    emit("suite.started", {"suite_id": suite_id, "trials": len(specs)})

    for spec in specs:
        if active_suite["abort"]:
            break
        if spec.trial_id in done_ids:
            continue

        suite["current_trial"] = spec.as_dict()
        benchmark.write_suite(directory, suite)

        ranked = _ranked_nodes()
        if spec.backends:
            # A vendor-mix trial uses every online machine of that mix. Its
            # size is whatever is present, recorded in node_count, and a mix
            # with a vendor missing is skipped rather than run as a smaller,
            # differently composed mesh under the same label.
            ranked = [node for node in ranked if node.get("backend") in set(spec.backends)]
            present = {node.get("backend") for node in ranked}
            missing = [backend for backend in spec.backends if backend not in present]
            spec.node_count = len(ranked) if not missing else 0
            if missing:
                result = benchmark.trial_result(
                    spec,
                    {},
                    benchmark.STATUS_SKIPPED,
                    "no %s machine is online" % " or ".join(benchmark.VENDOR_NAMES.get(m, m) for m in missing),
                )
                suite.setdefault("results", []).append(result)
                benchmark.write_suite(directory, suite)
                emit("suite.trial", {"suite_id": suite_id, "trial": spec.as_dict(), "status": "skipped"})
                continue
        if len(ranked) < max(1, spec.node_count):
            result = benchmark.trial_result(
                spec,
                {},
                benchmark.STATUS_SKIPPED,
                "needs %d machines, only %d are online" % (spec.node_count, len(ranked)),
            )
            suite.setdefault("results", []).append(result)
            benchmark.write_suite(directory, suite)
            emit("suite.trial", {"suite_id": suite_id, "trial": spec.as_dict(), "status": "skipped"})
            continue

        spec.node_ids = benchmark.select_nodes(
            ranked, spec.node_count, config.node_selection, seed=spec.repeat * 97 + spec.node_count
        )

        parent = store.get_dataset(config.parent_dataset_id) if config.parent_dataset_id else store.default_dataset()
        if parent is None:
            suite["status"] = "failed"
            suite["error"] = "the parent dataset is gone"
            benchmark.write_suite(directory, suite)
            break

        try:
            dataset = _ensure_subset(parent, spec.sample_count)
            spec.dataset_id = dataset["id"]
            # A request for more images than the dataset holds is served the
            # whole dataset. Record what trained, so a row that says 1000 can
            # never again mean 4.
            spec.effective_sample_count = int(dataset.get("train_count") or 0)
        except Exception as exc:
            suite.setdefault("results", []).append(
                benchmark.trial_result(spec, {}, benchmark.STATUS_FAILED, "subset failed: %s" % exc)
            )
            benchmark.write_suite(directory, suite)
            continue

        emit(
            "suite.trial",
            {
                "suite_id": suite_id,
                "trial": spec.as_dict(),
                "status": "starting",
                "completed": len(suite.get("results", [])),
                "total": len(specs),
            },
        )

        try:
            created = create_run(
                CreateRunRequest(
                    name="%s · %s" % (config.name, spec.label()),
                    dataset_id=spec.dataset_id,
                    base_model=config.base_model,
                    rounds=config.rounds,
                    imgsz=config.imgsz,
                    batch_size=config.batch_size,
                    mode="mesh",
                    warmup_mode=config.warmup_mode,
                    warmup_epochs=config.warmup_epochs,
                    optimizer=config.optimizer,
                    worker_validation=config.worker_validation,
                    backends=spec.backends,
                    node_ids=spec.node_ids,
                    partition_strategy=spec.strategy,
                    evaluate=config.evaluate,
                    seed=spec.seed,
                    suite_id=suite_id,
                    trial_id=spec.trial_id,
                    notes=config.notes or None,
                )
            )
            run_id = created["id"]
        except HTTPException as exc:
            suite.setdefault("results", []).append(
                benchmark.trial_result(spec, {}, benchmark.STATUS_FAILED, str(exc.detail))
            )
            benchmark.write_suite(directory, suite)
            continue
        except Exception as exc:
            suite.setdefault("results", []).append(
                benchmark.trial_result(spec, {}, benchmark.STATUS_FAILED, "%s: %s" % (type(exc).__name__, exc))
            )
            benchmark.write_suite(directory, suite)
            continue

        final_status = _wait_for_run(run_id, config.trial_timeout_seconds)

        with state_lock:
            run_snapshot = dict(runs.get(run_id) or {})
        if not run_snapshot:
            archived = store.get_run(run_id)
            run_snapshot = dict(archived) if archived else {"id": run_id}

        status = (
            benchmark.STATUS_DONE
            if final_status == "done"
            else benchmark.STATUS_ABORTED
            if final_status == "aborted"
            else benchmark.STATUS_FAILED
        )
        result = benchmark.trial_result(
            spec,
            run_snapshot,
            status,
            None if status == benchmark.STATUS_DONE else run_snapshot.get("error") or final_status,
        )
        result["network"] = _network_snapshot(spec.node_ids)
        suite.setdefault("results", []).append(result)
        suite["current_trial"] = None
        benchmark.write_suite(directory, suite)

        emit(
            "suite.trial",
            {
                "suite_id": suite_id,
                "trial": spec.as_dict(),
                "status": status,
                "speedup": result.get("speedup"),
                "map50": result.get("map50"),
                "completed": len(suite.get("results", [])),
                "total": len(specs),
            },
        )

        # Let GPUs settle and memory free before the next timing measurement.
        time.sleep(config.settle_seconds)

    suite["status"] = "aborted" if active_suite["abort"] else "done"
    suite["finished_at"] = time.time()
    suite["current_trial"] = None
    suite["cells"] = benchmark.accuracy_delta(benchmark.aggregate_cells(suite.get("results", [])))
    benchmark.write_suite(directory, suite)

    with suite_lock:
        active_suite["id"] = None
        active_suite["abort"] = False

    results = suite.get("results", [])
    emit(
        "suite.finished",
        {
            "suite_id": suite_id,
            "status": suite["status"],
            "campaign_id": config.campaign_id,
            "leg": config.leg,
            "network_label": config.network_label,
            "completed": sum(1 for r in results if r.get("status") == benchmark.STATUS_DONE),
            "total": len(specs),
            # The dashboard raises the "change the network now" prompt on this.
            "awaiting_network_change": suite["status"] == "done" and bool(config.campaign_id),
        },
    )


def _network_snapshot(node_ids: Sequence[str]) -> dict:
    """Observed network conditions during a trial.

    Recorded rather than imposed. Shaping traffic needs administrator rights and
    a different tool on every platform, so the harness measures what the network
    actually did and labels the run, which is what lets results from lab
    Ethernet and from a phone hotspot be compared afterwards.
    """
    with state_lock:
        rows = [
            {
                "node_id": node_id,
                "name": (nodes.get(node_id) or {}).get("display_name"),
                "latency_ms": (nodes.get(node_id) or {}).get("latency_ms"),
                "throughput_sps": (nodes.get(node_id) or {}).get("throughput_sps"),
            }
            for node_id in node_ids
        ]
    latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    return {
        "nodes": rows,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
    }


@app.post("/benchmarks")
def create_suite(req: SuiteRequest, _: str = Depends(require_mesh_token)):
    """Plan a sweep and start it."""
    with suite_lock:
        if active_suite["id"]:
            raise HTTPException(status_code=409, detail="A sweep is already running.")

        config = benchmark.SuiteConfig.from_dict(req.config)
        parent = (
            store.get_dataset(config.parent_dataset_id)
            if config.parent_dataset_id
            else store.default_dataset()
        )
        if parent is None:
            raise HTTPException(status_code=400, detail="Upload a dataset before running a sweep.")
        if parent.get("is_subset"):
            raise HTTPException(
                status_code=400,
                detail="Pick the full dataset, not a subset. The sweep derives its own subsets.",
            )
        if not aggregation.torch_ready():
            raise HTTPException(status_code=503, detail="The training plane is still installing.")

        ranked = _ranked_nodes()
        if not ranked:
            raise HTTPException(status_code=409, detail="No eligible machine is online.")

        config.parent_dataset_id = parent["id"]

        # A dataset size larger than the dataset is silently served the whole
        # dataset, so two different rungs of the ladder can train on exactly
        # the same images and be reported as different conditions. Leg 1 ran
        # eighteen trials against COCO8, which holds four training images, and
        # published a size axis in which 100 and 1000 were both 4.
        train_count = int(parent.get("train_count") or 0)
        requested = sorted({int(size) for size in config.dataset_sizes if size and size > 0})
        if train_count and requested:
            effective = sorted({min(size, train_count) for size in requested})
            if len(effective) < len(requested):
                collapsed = [size for size in requested if size > train_count]
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "%s holds %d training images, so %s would all train on the same %d "
                        "images and the dataset-size axis would not exist. Pick sizes at or "
                        "below %d, or choose a larger dataset."
                        % (
                            parent.get("name") or "That dataset",
                            train_count,
                            " and ".join(str(size) for size in collapsed),
                            train_count,
                            train_count,
                        )
                    ),
                )

        if not config.campaign_id:
            config.campaign_id = uuid.uuid4().hex[:10]
        trials = benchmark.expand(config, len(ranked))
        if not trials:
            raise HTTPException(status_code=400, detail="That configuration produces no trials.")

        best_throughput = max(
            (float(node.get("throughput_sps") or 0.0) for node in ranked), default=0.0
        )
        suite_id = uuid.uuid4().hex[:12]
        suite = {
            "id": suite_id,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "status": "pending",
            "config": config.as_dict(),
            "trials": [spec.as_dict() for spec in trials],
            "results": [],
            "current_trial": None,
            "cells": [],
            "estimated_seconds": round(
                benchmark.estimate_seconds(trials, config, best_throughput), 1
            ),
            # The disclosure block reviewers check before they check the numbers.
            "environment": {
                "coordinator": evaluation.host_snapshot(),
                "dataset": {
                    "id": parent["id"],
                    "name": parent["name"],
                    "train_count": parent.get("train_count"),
                    "val_count": parent.get("val_count"),
                    "class_names": parent.get("class_names"),
                },
                "policy": current_policy().as_dict(),
                "nodes": [
                    {
                        "node_id": node["node_id"],
                        "name": node.get("display_name"),
                        "gpu": node.get("gpu"),
                        "backend": node.get("backend"),
                        "memory_mb": node.get("gpu_memory_mb"),
                        "capability": node.get("capability"),
                        "agent_version": node.get("agent_version"),
                    }
                    for node in ranked
                ],
            },
        }
        benchmark.write_suite(_suite_dir(suite_id), suite)
        active_suite["id"] = suite_id
        active_suite["abort"] = False

    supervise(
        sweeper.submit(_execute_suite, suite_id),
        "sweep %s" % suite_id,
        lambda exc: _fail_suite(suite_id, "%s: %s" % (type(exc).__name__, exc)),
    )
    emit("suite.created", {"suite_id": suite_id, "trials": len(trials)})
    return suite


@app.post("/benchmarks/preview")
def preview_suite(req: SuiteRequest, _: str = Depends(require_mesh_token)):
    """Trial count and rough duration for a configuration, before committing."""
    config = benchmark.SuiteConfig.from_dict(req.config)
    ranked = _ranked_nodes()
    available = max(1, len(ranked))
    trials = benchmark.expand(config, available)
    best = max((float(n.get("throughput_sps") or 0.0) for n in ranked), default=0.0)

    # A machine count above what is online is dropped from the design. Saying so
    # matters: somebody planning a six-machine sweep before the machines arrive
    # would otherwise see a small trial count with no explanation and assume the
    # design did not take.
    requested = sorted(set(config.node_counts or []))
    dropped = [count for count in requested if count > available]

    return {
        "trials": len(trials),
        "estimated_seconds": round(benchmark.estimate_seconds(trials, config, best), 1),
        "available_nodes": len(ranked),
        "dropped_counts": dropped,
        "planned_counts": sorted({spec.node_count for spec in trials}),
        "breakdown": [spec.as_dict() for spec in trials[:60]],
    }


class NextLegRequest(BaseModel):
    network_label: str = Field(..., min_length=1, max_length=60)
    notes: Optional[str] = None


@app.get("/benchmarks/campaigns")
def get_campaigns(_: str = Depends(require_mesh_token)):
    """Sweeps grouped into campaigns, with a cross-network comparison per campaign."""
    index = benchmark.list_suites(BENCHMARK_DIR)
    campaigns = benchmark.campaign_summary(index)

    for campaign in campaigns:
        loaded = []
        for leg in campaign["legs"]:
            suite = benchmark.read_suite(_suite_dir(leg["id"]))
            if suite:
                loaded.append(suite)
        campaign["comparison"] = benchmark.compare_networks(loaded)
    return {"campaigns": campaigns, "active_suite_id": active_suite["id"]}


@app.post("/benchmarks/{suite_id}/next-leg")
def start_next_leg(suite_id: str, req: NextLegRequest, _: str = Depends(require_mesh_token)):
    """Repeat a finished sweep's exact design on a different network.

    The whole design is copied and only the network label changes, which is what
    makes the two legs comparable. Machine count is re-checked at start, because
    switching networks is exactly when a machine tends to fall off.
    """
    with suite_lock:
        if active_suite["id"]:
            raise HTTPException(status_code=409, detail="A sweep is already running.")

        previous = benchmark.read_suite(_suite_dir(suite_id))
        if previous is None:
            raise HTTPException(status_code=404, detail="Unknown sweep")

        config = benchmark.SuiteConfig.from_dict(previous.get("config"))
        config.campaign_id = config.campaign_id or uuid.uuid4().hex[:10]
        config.leg = int(config.leg or 1) + 1
        config.network_label = req.network_label.strip()
        if req.notes:
            config.notes = req.notes

        parent = store.get_dataset(config.parent_dataset_id)
        if parent is None:
            raise HTTPException(status_code=400, detail="The dataset from the previous leg is gone.")

        ranked = _ranked_nodes()
        if not ranked:
            raise HTTPException(
                status_code=409,
                detail="No machine is online. Reconnect the workers to the new network first.",
            )

        # The design is fixed by the previous leg, so an explicit machine-count
        # list carries over. If fewer machines came back after the network
        # change, the missing cells are recorded as skipped rather than quietly
        # run smaller, which would make the legs incomparable.
        trials = benchmark.expand(config, max(len(ranked), max(config.node_counts or [1])))
        if not trials:
            raise HTTPException(status_code=400, detail="That design produces no trials.")

        new_id = uuid.uuid4().hex[:12]
        suite = {
            "id": new_id,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "status": "pending",
            "config": config.as_dict(),
            "trials": [spec.as_dict() for spec in trials],
            "results": [],
            "current_trial": None,
            "cells": [],
            "estimated_seconds": previous.get("estimated_seconds"),
            "previous_leg_id": suite_id,
            "environment": {
                "coordinator": evaluation.host_snapshot(),
                "dataset": previous.get("environment", {}).get("dataset"),
                "policy": current_policy().as_dict(),
                "nodes": [
                    {
                        "node_id": node["node_id"],
                        "name": node.get("display_name"),
                        "gpu": node.get("gpu"),
                        "backend": node.get("backend"),
                        "memory_mb": node.get("gpu_memory_mb"),
                        "capability": node.get("capability"),
                        "agent_version": node.get("agent_version"),
                    }
                    for node in ranked
                ],
            },
        }
        benchmark.write_suite(_suite_dir(new_id), suite)
        active_suite["id"] = new_id
        active_suite["abort"] = False

    supervise(
        sweeper.submit(_execute_suite, new_id),
        "sweep %s" % new_id,
        lambda exc: _fail_suite(new_id, "%s: %s" % (type(exc).__name__, exc)),
    )
    emit(
        "suite.created",
        {"suite_id": new_id, "campaign_id": config.campaign_id, "leg": config.leg, "trials": len(trials)},
    )
    return suite


@app.get("/benchmarks")
def get_suites(_: str = Depends(require_mesh_token)):
    ranked = _ranked_nodes()
    index = benchmark.list_suites(BENCHMARK_DIR)
    return {
        "suites": index,
        "campaigns": benchmark.campaign_summary(index),
        "active_suite_id": active_suite["id"],
        "available_nodes": len(ranked),
        "nodes": [
            {
                "node_id": node["node_id"],
                "name": node.get("display_name"),
                "gpu": node.get("gpu"),
                "backend": node.get("backend"),
                "memory_mb": node.get("gpu_memory_mb"),
                "gflops": (node.get("capability") or {}).get("gflops"),
                "throughput_sps": node.get("throughput_sps"),
            }
            for node in ranked
        ],
        "torch_ready": aggregation.torch_ready(),
    }


@app.get("/benchmarks/{suite_id}")
def get_suite(suite_id: str, _: str = Depends(require_mesh_token)):
    suite = benchmark.read_suite(_suite_dir(suite_id))
    if suite is None:
        raise HTTPException(status_code=404, detail="Unknown sweep")
    suite["is_active"] = active_suite["id"] == suite_id
    if suite.get("status") == "running" and not suite["is_active"]:
        # The coordinator restarted while this sweep was mid-flight.
        suite["status"] = "interrupted"
    return suite


@app.post("/benchmarks/{suite_id}/abort")
def abort_suite(suite_id: str, _: str = Depends(require_mesh_token)):
    with suite_lock:
        if active_suite["id"] != suite_id:
            raise HTTPException(status_code=409, detail="That sweep is not running.")
        active_suite["abort"] = True
    emit("suite.aborting", {"suite_id": suite_id})
    return {"status": "aborting"}


@app.post("/benchmarks/{suite_id}/resume")
def resume_suite(suite_id: str, _: str = Depends(require_mesh_token)):
    """Continue an interrupted sweep from the first trial without a result."""
    with suite_lock:
        if active_suite["id"]:
            raise HTTPException(status_code=409, detail="A sweep is already running.")
        suite = benchmark.read_suite(_suite_dir(suite_id))
        if suite is None:
            raise HTTPException(status_code=404, detail="Unknown sweep")
        if suite.get("status") == "done":
            raise HTTPException(status_code=409, detail="That sweep already finished.")
        active_suite["id"] = suite_id
        active_suite["abort"] = False
    supervise(
        sweeper.submit(_execute_suite, suite_id),
        "sweep %s" % suite_id,
        lambda exc: _fail_suite(suite_id, "%s: %s" % (type(exc).__name__, exc)),
    )
    return {"status": "resumed"}


@app.delete("/benchmarks/{suite_id}")
def delete_suite(suite_id: str, _: str = Depends(require_mesh_token)):
    if active_suite["id"] == suite_id:
        raise HTTPException(status_code=409, detail="Stop the sweep before deleting it.")
    directory = _suite_dir(suite_id)
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Unknown sweep")
    shutil.rmtree(directory, ignore_errors=True)
    return {"status": "deleted"}


@app.get("/benchmarks/{suite_id}/download/{artifact}")
def download_suite_artifact(suite_id: str, artifact: str, _: str = Depends(require_mesh_token)):
    """suite.json, results.csv or summary.csv for one sweep."""
    allowed = {"suite.json", "results.csv", "summary.csv"}
    if artifact not in allowed:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = _suite_dir(suite_id) / artifact
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not generated yet")
    media = "application/json" if artifact.endswith(".json") else "text/csv"
    return FileResponse(path, media_type=media, filename="%s-%s" % (suite_id, artifact))


@app.get("/health")
def health():
    """Unauthenticated: the launcher waits on it, and workers use it to find this mesh.

    `mesh_id` is not a secret. It lets a worker that is looking for its
    coordinator after an address change recognise the right one, rather than
    joining whichever GradMesh it finds first.
    """
    return {
        "status": "ok",
        "version": __version__,
        "protocol": PROTOCOL,
        "mesh_id": store.mesh_id(),
        "torch_ready": aggregation.torch_ready(),
        "nodes_active": sum(1 for node in nodes.values() if node.get("active")),
        "mdns": advertiser.as_dict(),
    }
