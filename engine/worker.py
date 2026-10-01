"""GradMesh worker agent, version 5.

Runs on any machine that lends its GPU to the mesh: NVIDIA through CUDA, Intel
through XPU, Apple Silicon through Metal. It registers with the coordinator,
reports a measured capability profile and the state of its host, then pulls
shards and trains them.

The Ultralytics call at the centre is the one v3 validated, including the
Intel XPU trainer; `trainers.train_round` wraps it per backend. Everything this
file changes is about staying connected and wasting less time per round,
because those were the two ways v4 lost machines and minutes:

* **One heartbeat thread for the whole life of the process.** v4 beat from the
  polling loop, so a slow poll or a long upload could starve it and the
  coordinator would drop a machine that was working fine.
* **Retries with backoff on every request.** A Wi-Fi hiccup during the result
  upload used to throw away a finished round. Uploads now retry, and the
  coordinator accepts a duplicate result idempotently.
* **Stable identity, one process per GPU.** The node id is derived from the
  machine, the backend and the GPU index rather than a random file, and a lock
  stops two terminals on one machine from fighting over one GPU under one id.
  A newer process for the same GPU supersedes the old one cleanly.
* **The host is found again if it moves.** After the coordinator has been
  unreachable for a while the agent tries gradmesh.local and then the local
  subnet, accepting only a coordinator that reports the same mesh id.
* **The machine stays awake** while it is contributing.
* **Images are cached.** Shards are a list of files rather than a fresh zip each
  round, and the agent downloads only the images it does not already hold, so
  after a round or two a shard costs almost nothing to fetch.
* **Weights move as raw bytes,** a third smaller than v4's base64 JSON.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from version import PROTOCOL, __version__  # noqa: E402

# Ultralytics checks for updates, downloads fonts and sends analytics unless
# told not to. A worker must need nothing beyond the LAN, and must never pip
# install anything into a contributor's environment on its own.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

AGENT_VERSION = __version__
FEATURES = ["binary-weights", "image-cache", "phase-timing", "diagnostics", "instance-id"]
WORKER_HOME = Path(os.getenv("GRADMESH_WORKER_HOME", str(Path.home() / ".gradmesh")))
DIAGNOSTICS_EVERY_SECONDS = 30.0
REDISCOVER_AFTER_SECONDS = 25.0


def log(message: str) -> None:
    print("[worker %s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


# ---------------------------------------------------------------------------
# Errors the main loop distinguishes
# ---------------------------------------------------------------------------


class TransientError(Exception):
    """The coordinator could not be reached, or answered 502 to 504."""


class TokenRejected(Exception):
    pass


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class Superseded(Exception):
    """A newer process registered for this GPU; this one should leave."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def machine_id() -> str:
    """The operating system's own machine identifier, if it has one."""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0),
            )
            return str(winreg.QueryValueEx(key, "MachineGuid")[0])
        if sys.platform == "darwin":
            out = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True, timeout=5
            ).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split("=")[-1].strip().strip('"')
        for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            if os.path.isfile(candidate):
                with open(candidate, "r", encoding="utf-8") as stream:
                    return stream.read().strip()
    except Exception:
        pass
    return ""


def derive_node_id(backend: str, gpu_index: int) -> str:
    """Same machine, same GPU, same id, across restarts and reinstalls.

    The machine identifier is combined with the network adapter address and
    hostname, because lab PCs imaged from one master can share a MachineGuid,
    and two machines with one id overwrite each other in the registry. That
    was a quiet cause of workers that joined and then vanished.
    """
    basis = "|".join(
        [machine_id(), "%012x" % uuid.getnode(), socket.gethostname().lower(), backend, str(gpu_index)]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


class InstanceLock:
    """One worker process per GPU per machine, enforced by the OS.

    An OS lock rather than a pid file, so a crashed worker never leaves a lock
    behind: the kernel releases it with the process.
    """

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            return False
        try:
            self.path.with_suffix(".pid").write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass
        return True

    def holder(self) -> str:
        try:
            return self.path.with_suffix(".pid").read_text(encoding="utf-8").strip()
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Staying awake
# ---------------------------------------------------------------------------


_inhibitor: Optional[subprocess.Popen] = None


def keep_awake() -> str:
    """Stop the machine sleeping while it contributes. Returns how, for the log."""
    global _inhibitor
    try:
        if sys.platform == "win32":
            import ctypes

            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            return "system sleep blocked while contributing"
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            _inhibitor = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            return "caffeinate is keeping this Mac awake"
        if shutil.which("systemd-inhibit"):
            _inhibitor = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=sleep:idle",
                    "--who=GradMesh",
                    "--why=Contributing a GPU to a training mesh",
                    "--mode=block",
                    "sh",
                    "-c",
                    "while kill -0 %d 2>/dev/null; do sleep 20; done" % os.getpid(),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "systemd-inhibit is blocking sleep"
    except Exception:
        pass
    return "could not block sleep; keep this machine awake while it trains"


def release_awake() -> None:
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if _inhibitor is not None:
            _inhibitor.terminate()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _detail(response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str):
            return detail
        return json.dumps(payload)[:400]
    except Exception:
        return (response.text or "")[:400] or "HTTP %d" % response.status_code


class MeshClient:
    """HTTP to the coordinator with retries, backoff and a movable address."""

    def __init__(self, server: str, token: str, instance_id: str):
        import requests

        self.requests = requests
        self.server = server.rstrip("/")
        self.token = token
        self.instance_id = instance_id
        self.last_ok = time.time()
        self._local = threading.local()

    @property
    def session(self):
        # One session per thread: the heartbeat thread and the training loop
        # must never wait on each other's connection.
        session = getattr(self._local, "session", None)
        if session is None:
            session = self.requests.Session()
            session.headers.update(
                {
                    "X-Mesh-Token": self.token,
                    "X-Instance-Id": self.instance_id,
                    "User-Agent": "gradmesh-worker/%s" % AGENT_VERSION,
                }
            )
            self._local.session = session
        return session

    def call(self, method: str, path: str, *, retries: int = 3, timeout: float = 30.0, **kwargs):
        delay = 0.6
        last: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                response = self.session.request(method, self.server + path, timeout=timeout, **kwargs)
            except (self.requests.ConnectionError, self.requests.Timeout) as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
            else:
                if response.status_code in (502, 503, 504):
                    last = "HTTP %d: %s" % (response.status_code, _detail(response))
                else:
                    self.last_ok = time.time()
                    return response
            if attempt < retries:
                time.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 12.0)
        raise TransientError(last or "unreachable")

    @staticmethod
    def check(response) -> None:
        if response.status_code < 400:
            return
        detail = _detail(response)
        if response.status_code == 401:
            raise TokenRejected(detail)
        if response.status_code == 404:
            raise NotFound(detail)
        if response.status_code == 409:
            if "supersede" in detail.lower():
                raise Superseded(detail)
            raise Conflict(detail)
        raise RuntimeError("HTTP %d: %s" % (response.status_code, detail))

    def json(self, method: str, path: str, **kwargs) -> dict:
        response = self.call(method, path, **kwargs)
        self.check(response)
        return response.json() if response.content else {}

    def download(self, path: str, destination: Path, *, method: str = "GET", json_body=None,
                 timeout: float = 600.0, retries: int = 4) -> int:
        """Stream a response body to disk. Returns bytes written."""
        delay = 1.0
        last = None
        for attempt in range(retries + 1):
            try:
                with self.session.request(
                    method, self.server + path, json=json_body, stream=True, timeout=(10, timeout)
                ) as response:
                    if response.status_code in (502, 503, 504):
                        raise TransientError("HTTP %d" % response.status_code)
                    self.check(response)
                    written = 0
                    with open(destination, "wb") as stream:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if chunk:
                                stream.write(chunk)
                                written += len(chunk)
                    self.last_ok = time.time()
                    return written
            except (self.requests.ConnectionError, self.requests.Timeout, TransientError,
                    self.requests.exceptions.ChunkedEncodingError) as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
                if attempt < retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 15.0)
        raise TransientError(last or "download failed")


def probe_health(server: str, timeout: float = 2.5) -> Optional[dict]:
    try:
        with urlrequest.urlopen(server.rstrip("/") + "/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload if payload.get("status") == "ok" else None
    except Exception:
        return None


def local_ipv4() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        probe.close()


def local_addresses() -> set:
    found = {"127.0.0.1", "::1", "localhost", local_ipv4()}
    try:
        import psutil

        for entries in psutil.net_if_addrs().values():
            for entry in entries:
                if getattr(entry, "address", None):
                    found.add(entry.address.split("%")[0])
    except Exception:
        pass
    return found


def is_local_server(server: str) -> bool:
    """Is the coordinator on this very machine? Then the two compete for it."""
    try:
        from urllib.parse import urlparse

        host = urlparse(server).hostname or ""
        resolved = socket.gethostbyname(host)
    except Exception:
        return False
    return resolved.startswith("127.") or resolved in local_addresses()


def find_mesh(mesh_id: Optional[str], port: int, current: str) -> Optional[str]:
    """Locate this mesh's coordinator after its address changed.

    Only a coordinator reporting the same mesh id is accepted, so a second
    mesh on the same network can never capture a worker by accident.
    """

    def matches(server: str) -> bool:
        health = probe_health(server)
        return bool(health) and (mesh_id is None or health.get("mesh_id") in (None, mesh_id))

    if matches(current):
        return current

    resolved: List[str] = []

    def resolve_mdns() -> None:
        try:
            resolved.append(socket.gethostbyname("gradmesh.local"))
        except Exception:
            pass

    thread = threading.Thread(target=resolve_mdns, daemon=True)
    thread.start()
    thread.join(4.0)
    for address in resolved:
        candidate = "http://%s:%d" % (address, port)
        if matches(candidate):
            return candidate

    base = local_ipv4()
    if base.startswith("127."):
        return None
    prefix = base.rsplit(".", 1)[0]
    candidates = ["http://%s.%d:%d" % (prefix, octet, port) for octet in range(1, 255)]

    def check(server: str) -> Optional[str]:
        health = probe_health(server, timeout=0.6)
        if health and (mesh_id is None or health.get("mesh_id") in (None, mesh_id)):
            return server
        return None

    with ThreadPoolExecutor(max_workers=64) as pool:
        for result in pool.map(check, candidates):
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# Image cache
# ---------------------------------------------------------------------------


def _safe_member(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise ValueError("bundle member escapes the cache: %s" % name)
    return target


class DatasetCache:
    """Training images kept between rounds, so each round fetches only what is new.

    Layout mirrors Ultralytics' expectations: images/train/<rel> beside
    labels/train/<rel>.txt, which is how it finds a label from an image path.
    A shard is then just a text file listing image paths. Datasets untouched
    for thirty days are removed on start.
    """

    MAX_AGE_SECONDS = 30 * 86400

    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self._prune()

    def _prune(self) -> None:
        now = time.time()
        for child in self.base.iterdir():
            try:
                stamp = child / ".last-used"
                last = stamp.stat().st_mtime if stamp.exists() else child.stat().st_mtime
                if child.is_dir() and now - last > self.MAX_AGE_SECONDS:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                continue

    def prepare(self, client: MeshClient, manifest: dict, workdir: Path) -> Dict[str, Any]:
        key = "".join(ch for ch in str(manifest["dataset_key"]) if ch.isalnum() or ch in "-_")[:64] or "dataset"
        root = self.base / key
        images_dir = root / "images" / "train"
        labels_dir = root / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        (root / ".last-used").touch()

        def label_path(image_rel: str) -> Path:
            return labels_dir / Path(image_rel).with_suffix(".txt")

        missing: List[str] = []
        for entry in manifest["files"]:
            image = images_dir / entry["image"]
            stale = not image.is_file() or image.stat().st_size != int(entry.get("size") or -1)
            label_size = entry.get("label_size")
            if not stale and label_size is not None:
                label = label_path(entry["image"])
                stale = not label.is_file() or label.stat().st_size != int(label_size)
            if stale:
                missing.append(entry["image"])

        downloaded = 0
        for start in range(0, len(missing), 400):
            chunk = missing[start : start + 400]
            bundle = workdir / ("bundle-%d.zip" % start)
            downloaded += client.download(
                "/datasets/%s/bundle" % manifest["dataset_key"],
                bundle,
                method="POST",
                json_body={"files": chunk},
            )
            with zipfile.ZipFile(bundle) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    target = _safe_member(root, member.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
            bundle.unlink(missing_ok=True)

        listing = []
        for entry in manifest["files"]:
            image = images_dir / entry["image"]
            label = label_path(entry["image"])
            if not label.exists():
                # An image without labels is a background image; Ultralytics
                # wants an empty file for it, as the v3 shard builder wrote.
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("", encoding="utf-8")
            listing.append(str(image))

        list_file = workdir / "train.txt"
        list_file.write_text("\n".join(listing) + "\n", encoding="utf-8")
        return {
            "root": root,
            "list_file": list_file,
            "bytes": downloaded,
            "hits": len(manifest["files"]) - len(missing),
            "misses": len(missing),
        }


def write_data_yaml(path: Path, root: Path, train: str, val: str, class_names: List[str]) -> None:
    import yaml

    names = {index: name for index, name in enumerate(class_names or ["object"])}
    path.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": train, "val": val, "nc": len(names), "names": names}, sort_keys=False
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class Agent:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.instance_id = uuid.uuid4().hex[:12]
        self.client = MeshClient(args.server_url, args.token, self.instance_id)
        self.node_id = args.node_id or ""
        self.mesh_id: Optional[str] = None
        self.server_features: List[str] = []
        self.heartbeat_seconds = args.heartbeat_seconds
        self.capability: Dict[str, Any] = {}
        self.accelerator = None
        self.co_located = False
        self.stop = threading.Event()
        self.need_register = threading.Event()
        self.unreachable_since: Optional[float] = None
        self.exit_message: Optional[str] = None
        self.status: Dict[str, Any] = {
            "active_batches": 0,
            "allocated_memory_mb": 0,
            "training_epoch": None,
            "training_total_epochs": None,
            "phase": "idle",
            "progress": None,
        }
        self.latency_ms: Optional[float] = None
        self.diagnostics: Dict[str, Any] = {}
        self.diagnostics_at = 0.0
        self.cache: Optional[DatasetCache] = None
        self.dataloader_workers = 0

    # -- setup ---------------------------------------------------------------

    def prepare_device(self) -> None:
        from accelerator import detect_accelerator, pin_device

        if self.args.gpu_index is not None and self.args.gpu_index >= 0:
            pin_device(self.args.backend, self.args.gpu_index)

        import importlib

        importlib.import_module("torch")
        self.accelerator = detect_accelerator(
            self.args.backend, advertised_memory_mb=self.args.gpu_memory_mb, allow_cpu_training=self.args.allow_cpu
        )

        from probe import diagnose, diagnostics, probe

        log("checking %s" % self.accelerator.device_name)
        problem = diagnose(self.accelerator)
        if problem:
            raise SystemExit("\n[worker] %s\n" % problem)

        log("measuring device capability")
        self.capability = probe(self.accelerator).as_dict()
        log(
            "%s via %s, %d MB, %.0f GFLOP/s"
            % (
                self.capability["device_name"],
                self.capability["backend"],
                self.capability["total_memory_mb"],
                self.capability["gflops"],
            )
        )
        self.diagnostics = diagnostics(self.accelerator, sample_cpu=False)
        self.diagnostics_at = time.time()

        from trainers import default_dataloader_workers

        self.dataloader_workers = (
            self.args.workers if self.args.workers is not None and self.args.workers >= 0
            else default_dataloader_workers(self.accelerator)
        )

        if not self.node_id:
            index = self.args.gpu_index if self.args.gpu_index is not None and self.args.gpu_index >= 0 else 0
            self.node_id = derive_node_id(self.accelerator.backend, index)

        if not self.args.no_cache:
            self.cache = DatasetCache(WORKER_HOME / "cache" / self.node_id)

    def register(self) -> None:
        self.co_located = is_local_server(self.client.server)
        payload = {
            "node_id": self.node_id,
            "instance_id": self.instance_id,
            "protocol": PROTOCOL,
            "features": FEATURES,
            "display_name": self.args.name or socket.gethostname(),
            "gpu": self.capability.get("device_name", "unknown"),
            "gpu_memory_mb": int(self.capability.get("total_memory_mb") or self.args.gpu_memory_mb or 8000),
            "max_batch_size": self.args.max_batch_size,
            "backend": self.capability.get("backend", "cpu"),
            "vendor": self.capability.get("vendor", "cpu"),
            "supports_training": bool(self.capability.get("supports_training")),
            "capability": self.capability,
            "diagnostics": self.diagnostics,
            "co_located": self.co_located,
            "dataloader_workers": self.dataloader_workers,
            "owner": self.args.owner or None,
            "agent_version": AGENT_VERSION,
        }
        response = self.client.json("POST", "/register_node", json=payload, retries=6, timeout=20)
        self.mesh_id = response.get("mesh_id") or self.mesh_id
        self.server_features = list(response.get("features") or [])
        self.heartbeat_seconds = float(response.get("heartbeat_seconds") or self.heartbeat_seconds)
        self.need_register.clear()
        admission = response.get("admission") or {}
        log(
            "joined %s as %s (%s: %s)"
            % (self.client.server, self.node_id, admission.get("tier", "unknown"), admission.get("reason", ""))
        )
        server_protocol = int(response.get("protocol") or 4)
        if server_protocol < PROTOCOL:
            log(
                "the host runs an older GradMesh (protocol %d); falling back to its transfers. "
                "Update the host for image caching and binary weights." % server_protocol
            )
        if not self.capability.get("supports_training"):
            log("no supported accelerator here, so this machine is measured but not given shards")
        elif not self.capability.get("gflops"):
            log("the capability probe measured nothing on this device; run `npm run doctor` on it")

    # -- heartbeat -------------------------------------------------------------

    def heartbeat_payload(self) -> dict:
        payload = {
            "node_id": self.node_id,
            "instance_id": self.instance_id,
            "load": 0.9 if self.status["active_batches"] else 0.1,
            "active_batches": self.status["active_batches"],
            "allocated_memory_mb": self.status["allocated_memory_mb"],
            "training_epoch": self.status["training_epoch"],
            "training_total_epochs": self.status["training_total_epochs"],
            "phase": self.status["phase"],
            "progress": self.status["progress"],
            # One beat late by construction: the round trip of a request can
            # only be known after it finished. v4 timed the building of a dict
            # and reported ten microseconds for a Wi-Fi link.
            "latency_ms": self.latency_ms,
        }
        if time.time() - self.diagnostics_at >= DIAGNOSTICS_EVERY_SECONDS:
            try:
                from probe import diagnostics

                self.diagnostics = diagnostics(self.accelerator)
            except Exception:
                pass
            self.diagnostics_at = time.time()
            payload["diagnostics"] = self.diagnostics
        return payload

    def heartbeat_loop(self) -> None:
        while not self.stop.wait(max(1.0, self.heartbeat_seconds)):
            started = time.perf_counter()
            try:
                response = self.client.call("POST", "/heartbeat", json=self.heartbeat_payload(), retries=1, timeout=10)
            except TransientError:
                if self.unreachable_since is None:
                    self.unreachable_since = time.time()
                    log("the coordinator stopped answering; will keep trying")
                continue
            finally:
                self.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if self.unreachable_since is not None:
                log("the coordinator is reachable again")
                self.unreachable_since = None
            if response.status_code == 404:
                self.need_register.set()
            elif response.status_code == 409:
                self.exit_message = _detail(response)
                self.stop.set()
            elif response.status_code == 401:
                self.exit_message = "The mesh token was rejected. Copy a fresh join command from the host."
                self.stop.set()

    def maybe_relocate(self) -> None:
        if self.unreachable_since is None or not self.args.discover:
            return
        if time.time() - self.unreachable_since < REDISCOVER_AFTER_SECONDS:
            return
        from urllib.parse import urlparse

        port = urlparse(self.client.server).port or 8000
        log("looking for the coordinator on this network")
        found = find_mesh(self.mesh_id, port, self.client.server)
        if found and found != self.client.server:
            log("found the coordinator at %s" % found)
            self.client.server = found
            self.need_register.set()
        self.unreachable_since = time.time() if not found else None

    # -- work ------------------------------------------------------------------

    def run(self) -> None:
        threading.Thread(target=self.heartbeat_loop, name="gradmesh-heartbeat", daemon=True).start()
        idle_delay = self.args.poll_seconds
        while not self.stop.is_set():
            try:
                if self.need_register.is_set():
                    log("the coordinator does not know this node, registering again")
                    self.register()
                self.maybe_relocate()
                payload = self.client.json("GET", "/get_batch/%s" % self.node_id, retries=1, timeout=30)
                batch = payload.get("batch")
                if not batch or not batch.get("batch_id"):
                    self.stop.wait(idle_delay)
                    continue
                self.run_batch(batch)
            except NotFound:
                self.need_register.set()
                self.stop.wait(1.0)
            except TransientError:
                if self.unreachable_since is None:
                    self.unreachable_since = time.time()
                self.stop.wait(min(10.0, idle_delay * 2))
            except TokenRejected:
                self.exit_message = "The mesh token was rejected. Copy a fresh join command from the host."
                break
            except Superseded as exc:
                self.exit_message = str(exc)
                break
            except Exception as exc:
                log("%s: %s" % (type(exc).__name__, exc))
                self.stop.wait(idle_delay)

    def set_status(self, **patch) -> None:
        self.status.update(patch)

    def ensure_model(self, model_name: str, workdir: Path) -> Path:
        name = Path(model_name).name
        if Path(model_name).is_file():
            return Path(model_name)
        if not name.endswith(".pt"):
            raise FileNotFoundError("Model must be a .pt checkpoint: %s" % model_name)
        cache = WORKER_HOME / "models"
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / name
        if cached.is_file() and cached.stat().st_size > 0:
            return cached
        partial = workdir / (name + ".part")
        self.client.download("/models/" + name, partial, timeout=600)
        shutil.move(str(partial), str(cached))
        return cached

    def fetch_weights(self, batch: dict) -> Optional[bytes]:
        if "binary-weights" in self.server_features and batch.get("weights_bin_url"):
            response = self.client.call("GET", batch["weights_bin_url"], retries=4, timeout=120)
            if response.status_code == 204:
                return None
            self.client.check(response)
            return response.content
        payload = self.client.json("GET", batch["weights_url"], retries=4, timeout=120)
        encoded = payload.get("weights_b64")
        return base64.b64decode(encoded) if encoded else None

    def prepare_data(self, batch: dict, workdir: Path) -> Dict[str, Any]:
        class_names = batch.get("class_names") or ["object"]
        use_cache = (
            self.cache is not None
            and "image-cache" in self.server_features
            and batch.get("manifest_url")
            and not batch.get("worker_validation")
        )
        if use_cache:
            manifest = self.client.json("GET", batch["manifest_url"], retries=4, timeout=60)
            prepared = self.cache.prepare(self.client, manifest, workdir)
            data_yaml = workdir / "data.yaml"
            listing = str(prepared["list_file"])
            write_data_yaml(data_yaml, prepared["root"], listing, listing, manifest.get("class_names") or class_names)
            return {"data_yaml": data_yaml, "bytes": prepared["bytes"], "hits": prepared["hits"], "misses": prepared["misses"]}

        archive = workdir / "shard.zip"
        size = self.client.download(batch["shard_url"], archive, timeout=900)
        extract = workdir / "shard"
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_member(extract, member.filename)
            bundle.extractall(extract)
        archive.unlink(missing_ok=True)
        data_yaml = extract / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError("training shard is missing data.yaml")
        # The shard's YAML carries the coordinator's absolute path; point it at
        # this machine's copy.
        import yaml

        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        config["path"] = str(extract.resolve())
        data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return {"data_yaml": data_yaml, "bytes": size, "hits": 0, "misses": int(batch.get("samples") or 0)}

    def run_batch(self, batch: dict) -> None:
        from federated_training import state_dict_from_bytes, state_dict_to_bytes
        from trainers import train_round
        from ultralytics import YOLO

        round_index = int(batch.get("round_index", 0))
        total_epochs = int(batch.get("epochs", 1))
        log(
            "round %d shard %s: %s images, batch %s"
            % (round_index + 1, batch.get("shard_index"), batch.get("samples"), batch.get("batch_size"))
        )
        self.set_status(
            active_batches=1,
            allocated_memory_mb=int(batch.get("estimated_memory_mb") or 0),
            training_epoch=0,
            training_total_epochs=total_epochs,
            phase="downloading",
            progress=None,
        )
        assigned = time.time()
        try:
            with TemporaryDirectory(prefix="gradmesh-", ignore_cleanup_errors=True) as temp:
                workdir = Path(temp)
                data = self.prepare_data(batch, workdir)
                model_path = self.ensure_model(batch.get("base_model", "yolov8n.pt"), workdir)
                weights = self.fetch_weights(batch)
                downloaded = time.time()

                self.set_status(phase="loading")
                model = YOLO(str(model_path))
                if weights:
                    state = state_dict_from_bytes(weights)
                    current = model.model.state_dict()
                    # Filter incompatible tensors, such as a class head whose
                    # shape changed, exactly as v3 did.
                    compatible = {
                        key: value
                        for key, value in state.items()
                        if key in current
                        and hasattr(value, "shape")
                        and tuple(value.shape) == tuple(current[key].shape)
                    }
                    model.model.load_state_dict(compatible, strict=False)
                loaded = time.time()

                options: Dict[str, Any] = {
                    "data": str(data["data_yaml"]),
                    "epochs": total_epochs,
                    "imgsz": int(batch.get("imgsz", 640)),
                    "batch": int(batch.get("batch_size", 8)),
                    "project": str(workdir / "runs"),
                    "name": "round_%d" % round_index,
                    "exist_ok": True,
                    "workers": self.dataloader_workers,
                }
                if batch.get("seed") is not None:
                    options["seed"] = int(batch["seed"])
                for key in ("warmup_epochs", "optimizer", "lr0", "deterministic", "cos_lr", "momentum", "weight_decay"):
                    if batch.get(key) is not None:
                        options[key] = batch[key]

                last_progress = {"at": 0.0}

                def progress(update: dict) -> None:
                    now = time.time()
                    if now - last_progress["at"] >= 1.0:
                        last_progress["at"] = now
                        self.set_status(progress=update)

                self.set_status(phase="training")
                timings = train_round(
                    model,
                    self.accelerator,
                    options,
                    validate=bool(batch.get("worker_validation")),
                    on_progress=progress,
                )
                self.set_status(training_epoch=total_epochs, phase="uploading", progress=None)
                result = state_dict_to_bytes(model.model.state_dict())
                finished = time.time()

                metrics = {
                    "trained": True,
                    "backend": self.accelerator.backend,
                    "vendor": self.accelerator.vendor,
                    "device": str(self.accelerator.torch_device),
                    "device_name": self.accelerator.device_name,
                    "base_model": batch.get("base_model", "yolov8n.pt"),
                    "samples": int(batch.get("samples") or 0),
                    "imgsz": batch.get("imgsz"),
                    "batch_size": batch.get("batch_size"),
                    "dataloader_workers": self.dataloader_workers,
                    "download_seconds": round(downloaded - assigned, 3),
                    "load_seconds": round(loaded - downloaded, 3),
                    "total_seconds": round(finished - assigned, 3),
                    "bytes_in": int(data["bytes"]) + len(weights or b""),
                    "bytes_out": len(result),
                    "cache_hits": data["hits"],
                    "cache_misses": data["misses"],
                    "torch_version": self.capability.get("torch_version"),
                    **timings,
                }
                self.submit(batch, result, metrics)
                log(
                    "finished shard %s: epoch %.1fs, setup %.1fs, fetched %s"
                    % (
                        batch.get("shard_index"),
                        timings["epoch_seconds"],
                        timings["setup_seconds"],
                        _human_bytes(metrics["bytes_in"]),
                    )
                )
                del model
        except (TokenRejected, Superseded):
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.report_failure(batch, exc)
        finally:
            self.set_status(
                active_batches=0, allocated_memory_mb=0, training_epoch=None, training_total_epochs=None,
                phase="idle", progress=None,
            )
            gc.collect()
            if self.accelerator is not None:
                self.accelerator.empty_cache()

    def submit(self, batch: dict, weights: bytes, metrics: dict) -> None:
        if "binary-weights" in self.server_features:
            response = self.client.call(
                "POST",
                "/batches/%s/result" % batch["batch_id"],
                data=weights,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Node-Id": self.node_id,
                    "X-Round-Index": str(int(batch.get("round_index", 0))),
                    "X-Metrics": base64.urlsafe_b64encode(json.dumps(metrics).encode("utf-8")).decode("ascii"),
                },
                retries=6,
                timeout=300,
            )
        else:
            response = self.client.call(
                "POST",
                "/submit_training_round_result",
                json={
                    "node_id": self.node_id,
                    "batch_id": batch["batch_id"],
                    "round_index": int(batch.get("round_index", 0)),
                    "weights_b64": base64.b64encode(weights).decode("ascii"),
                    "metrics": metrics,
                },
                retries=6,
                timeout=300,
            )
        if response.status_code in (404, 409):
            log("the coordinator no longer needed this shard: %s" % _detail(response))
            return
        self.client.check(response)

    def report_failure(self, batch: dict, exc: Exception) -> None:
        message = "%s: %s" % (type(exc).__name__, exc)
        lowered = message.lower()
        if "out of memory" in lowered or "outofmemory" in lowered:
            message = "out of memory at batch %s: %s" % (batch.get("batch_size"), message)
        log("shard %s failed: %s" % (batch.get("shard_index"), message[:400]))
        try:
            self.client.call(
                "POST",
                "/submit_training_batch_failure",
                json={"node_id": self.node_id, "batch_id": batch["batch_id"], "error": message[:4000]},
                retries=4,
                timeout=30,
            )
        except Exception:
            pass

    def leave(self) -> None:
        try:
            self.client.call("POST", "/leave", json={"node_id": self.node_id, "instance_id": self.instance_id},
                             retries=0, timeout=5)
        except Exception:
            pass


def _human_bytes(value: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return ("%d %s" % (value, unit)) if unit == "B" else ("%.1f %s" % (value, unit))
        value /= 1024.0
    return "%d B" % value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    env = os.getenv
    parser = argparse.ArgumentParser(description="GradMesh worker agent")
    parser.add_argument("--server-url", default=env("GRADMESH_SERVER", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=env("GRADMESH_TOKEN", ""), help="Mesh join token")
    parser.add_argument("--name", default=env("GRADMESH_NAME", ""), help="Display name in the dashboard")
    parser.add_argument("--owner", default=env("GRADMESH_OWNER", ""), help="Who is contributing this machine")
    parser.add_argument("--node-id", default=env("GRADMESH_NODE_ID", ""), help="Override the derived node id")
    parser.add_argument("--backend", choices=["auto", "cuda", "xpu", "mps", "cpu"], default=env("GRADMESH_BACKEND", "auto"))
    parser.add_argument("--gpu-index", type=int, default=int(env("GRADMESH_GPU_INDEX", "-1")),
                        help="Which GPU to use on a machine with several (default: the first)")
    parser.add_argument("--max-batch-size", type=int, default=int(env("GRADMESH_MAX_BATCH_SIZE", "64")),
                        help="Never train with a larger batch than this on this machine")
    parser.add_argument("--gpu-memory-mb", type=int, default=int(env("GRADMESH_GPU_MEMORY_MB", "0")) or None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Dataloader processes (default: chosen from the CPU and backend)")
    parser.add_argument("--poll-seconds", type=float, default=float(env("GRADMESH_POLL_SECONDS", "1.5")))
    parser.add_argument("--heartbeat-seconds", type=float, default=float(env("GRADMESH_HEARTBEAT_SECONDS", "5")))
    parser.add_argument("--discover", action="store_true",
                        help="Find the coordinator on the local network if its address changes")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Let a machine with no GPU take shards (slow; for testing the pipeline)")
    parser.add_argument("--allow-sleep", action="store_true", help="Do not block system sleep while contributing")
    parser.add_argument("--no-cache", action="store_true", help="Download whole shards instead of caching images")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.token:
        print("A mesh token is required. Copy the join command from the host's Invite a GPU page.")
        return 2

    if not probe_health(args.server_url, timeout=4):
        if args.discover:
            log("%s is unreachable, looking for the coordinator on this network" % args.server_url)
            from urllib.parse import urlparse

            found = find_mesh(None, urlparse(args.server_url).port or 8000, args.server_url)
            if not found:
                print("No GradMesh coordinator was found on this network. Is the host running `npm run dev`?")
                return 1
            log("found the coordinator at %s" % found)
            args.server_url = found
        else:
            print(
                "Cannot reach %s. Check that the host is on this network and that ports 3000 and 8000 are "
                "allowed through its firewall." % args.server_url
            )
            return 1

    agent = Agent(args)
    try:
        agent.prepare_device()
    except SystemExit as exc:
        print(exc)
        return 2

    lock = InstanceLock(WORKER_HOME / "locks" / ("%s.lock" % agent.node_id))
    if not lock.acquire():
        print(
            "This GPU is already being contributed by another GradMesh worker on this machine (pid %s). "
            "Stop that one first, or pass --gpu-index to contribute a different GPU." % lock.holder()
        )
        return 3

    if not args.allow_sleep:
        log(keep_awake())

    def terminate(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, terminate)
    except Exception:
        pass

    try:
        agent.register()
        agent.run()
    except KeyboardInterrupt:
        print("")
        log("leaving the mesh")
    except TokenRejected:
        agent.exit_message = "The mesh token was rejected. Copy a fresh join command from the host."
    except Superseded as exc:
        agent.exit_message = str(exc)
    finally:
        agent.stop.set()
        agent.leave()
        release_awake()
    if agent.exit_message:
        print("[worker] %s" % agent.exit_message)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
