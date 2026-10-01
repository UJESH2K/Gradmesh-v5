"""Build and verify a GradMesh Python environment. Standard library only.

One installer for every way GradMesh gets onto a machine:

* the host's `npm run setup` and `npm run dev` call `install`,
* the one-line join command downloads this file and calls `agent`,
* anyone setting up by hand can run it directly (see SETUP.md).

v4 had the install logic three times, in Node, PowerShell and sh, and the
failure that motivated this file was the environment itself rather than the
choice of wheel. A virtual environment records the absolute path of the
interpreter that made it. Copy one to another machine, or let OneDrive sync it
there, and it looks present, `python.exe` exists, and nothing runs. So this
installer never trusts an environment it did not verify on this machine:

1. The environment carries a marker naming the machine and interpreter that
   built it. A marker from somewhere else, or an interpreter that will not
   start, means the environment is rebuilt rather than patched.
2. Installs retry, show progress, and can come from a local wheelhouse for a
   lab whose internet is slow.
3. After installing, the accelerator is exercised: a real kernel launch on the
   GPU, not just `is_available()`, which reports True for a Blackwell card on a
   build that has no kernels for it.

Usage:

    python setup_env.py detect [--json] [--host] [--backend auto|cuda|xpu|mps|cpu]
    python setup_env.py install --venv DIR --plane control|training|all [--host]
                                [--state FILE] [--wheelhouse DIR] [--backend ...] [--force]
    python setup_env.py verify --venv DIR [--backend ...]
    python setup_env.py agent --server URL --token TOKEN [--name NAME] [--backend ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hardware  # noqa: E402
from version import PREFERRED_PYTHON, SUPPORTED_PYTHON, __version__  # noqa: E402

IS_WINDOWS = sys.platform == "win32"
MARKER = "gradmesh-env.json"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLOURS = {"gray": "90", "green": "32", "yellow": "33", "red": "31", "cyan": "36", "bold": "1"}


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def say(message: str, colour: str = "gray") -> None:
    if _tty():
        print("\033[%sm  %s\033[0m" % (_COLOURS.get(colour, "0"), message), flush=True)
    else:
        print("  %s" % message, flush=True)


# ---------------------------------------------------------------------------
# State file shared with the dashboard
# ---------------------------------------------------------------------------


class State:
    """The machine-local setup record the dashboard banner and doctor read."""

    def __init__(self, path: Optional[Path]):
        self.path = path

    def read(self) -> dict:
        if not self.path or not self.path.is_file():
            return {"controlPlane": "pending", "trainingPlane": "pending", "models": [], "messages": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"controlPlane": "pending", "trainingPlane": "pending", "models": [], "messages": []}

    def update(self, **patch) -> None:
        if not self.path:
            return
        state = self.read()
        state.update(patch)
        state["updatedAt"] = int(time.time() * 1000)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def note(self, message: str) -> None:
        state = self.read()
        messages = (state.get("messages") or [])[-39:]
        messages.append({"at": int(time.time() * 1000), "message": message})
        self.update(messages=messages)


# ---------------------------------------------------------------------------
# Interpreter and environment
# ---------------------------------------------------------------------------


def python_supported(version=None) -> bool:
    version = tuple(version or sys.version_info[:2])
    return SUPPORTED_PYTHON[0] <= version[:2] <= SUPPORTED_PYTHON[1]


def python_requirement_text() -> str:
    low, high = SUPPORTED_PYTHON
    return "Python %d.%d to %d.%d" % (low[0], low[1], high[0], high[1])


def install_python_hint() -> str:
    preferred = "%d.%d" % PREFERRED_PYTHON
    if IS_WINDOWS:
        return "winget install Python.Python.%s   (or download it from python.org)" % preferred
    if sys.platform == "darwin":
        return "brew install python@%s   (or the installer from python.org)" % preferred
    return "sudo apt install python%s python%s-venv   (or your distribution's equivalent)" % (preferred, preferred)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def machine_fingerprint() -> str:
    """Identifies this machine well enough to notice an environment copied from another."""
    parts = [socket.gethostname().lower(), sys.platform, platform.machine().lower()]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _probe_interpreter(python: Path) -> Optional[dict]:
    try:
        result = subprocess.run(
            [str(python), "-c", "import sys,json;print(json.dumps({'version':list(sys.version_info[:3]),'prefix':sys.prefix}))"],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def environment_problem(venv: Path) -> Optional[str]:
    """Why this environment cannot be used as it stands, or None if it can."""
    python = venv_python(venv)
    if not python.is_file():
        return "it does not exist yet"
    marker = _read_marker(venv)
    if marker.get("machine") and marker.get("machine") != machine_fingerprint():
        return "it was built on another machine (%s)" % marker.get("hostname", "unknown")
    probe = _probe_interpreter(python)
    if probe is None:
        return "its Python interpreter will not start, which usually means it was copied or synced from another machine"
    if not python_supported(probe["version"][:2]):
        return "it runs Python %d.%d, outside the supported range" % tuple(probe["version"][:2])
    return None


def _stamp_machine(venv: Path, base_python: Optional[str] = None) -> None:
    _write_marker(
        venv,
        machine=machine_fingerprint(),
        hostname=socket.gethostname(),
        base_python=base_python or _read_marker(venv).get("base_python"),
        gradmesh=__version__,
    )


def ensure_venv(venv: Path, state: State) -> Path:
    problem = environment_problem(venv)
    if problem is None:
        if not _read_marker(venv).get("machine"):
            # An environment that works but predates the marker, such as one
            # built by hand. Adopt it rather than throwing away a good install.
            _stamp_machine(venv)
        return venv_python(venv)

    if venv_python(venv).exists() or venv.exists():
        say("rebuilding the Python environment: %s" % problem, "yellow")
        state.note("Rebuilding the Python environment: %s." % problem)
        try:
            shutil.rmtree(venv)
        except Exception as exc:
            raise SystemExit(
                "Could not remove the old environment at %s (%s). Close any running GradMesh worker, "
                "coordinator or editor using it, then run setup again." % (venv, exc)
            )
    else:
        say("creating a Python environment at %s" % venv)

    venv.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, "-m", "venv", str(venv)])
    if result.returncode != 0 or not venv_python(venv).is_file():
        hint = ""
        if sys.platform.startswith("linux"):
            hint = " On Debian and Ubuntu the venv module is a separate package: sudo apt install python3-venv"
        raise SystemExit("Could not create a virtual environment at %s.%s" % (venv, hint))

    _write_marker(venv, created_at=time.time(), python="%d.%d.%d" % sys.version_info[:3])
    _stamp_machine(venv, base_python=sys.executable)
    return venv_python(venv)


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------


def requirements_closure(path: Path) -> List[Path]:
    """The file plus every -r include, so a change anywhere invalidates the stamp."""
    seen: List[Path] = []

    def visit(current: Path) -> None:
        if current in seen or not current.is_file():
            return
        seen.append(current)
        for line in current.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("-r "):
                visit(current.parent / line[3:].strip())

    visit(path)
    return seen


def requirements_stamp(path: Path) -> str:
    digest = hashlib.sha256()
    for item in requirements_closure(path):
        digest.update(item.name.encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()[:16]


def _read_marker(venv: Path) -> dict:
    try:
        return json.loads((venv / MARKER).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_marker(venv: Path, **patch) -> None:
    marker = _read_marker(venv)
    marker.update(patch)
    (venv / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")


_DOWNLOAD = re.compile(r"Downloading .*?/([A-Za-z0-9_.\-]+?)(?:-\d[^/\s]*)?\.whl.*?\(([\d.]+\s*[kMG]?B)\)")


def pip_install(python: Path, requirements: Path, state: State, wheelhouse: Optional[str], quiet: bool) -> None:
    say("installing %s" % requirements.name)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check", "-q"],
        capture_output=True,
    )
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--prefer-binary",
        "--retries",
        "6",
        "--timeout",
        "60",
        "-r",
        str(requirements),
    ]
    if wheelhouse:
        command += ["--find-links", wheelhouse]

    env = dict(os.environ)
    env.setdefault("PIP_NO_INPUT", "1")
    env["PYTHONIOENCODING"] = "utf-8"

    if not quiet:
        # Attached to the terminal so pip draws its own progress bar. A 2 GB
        # PyTorch wheel with no visible progress looks exactly like a hang,
        # which is how a contributor ends up pressing Ctrl+C at minute four.
        code = subprocess.call(command, env=env)
        tail_text = ""
    else:
        command.append("--progress-bar=off")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        tail: List[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            tail.append(line.rstrip())
            tail = tail[-60:]
            match = _DOWNLOAD.search(line)
            if match:
                state.note("Downloading %s (%s)." % (match.group(1), match.group(2)))
            print(line.rstrip(), flush=True)
        code = process.wait()
        tail_text = "\n".join(tail)
    if code != 0:
        text = tail_text
        hint = ""
        if IS_WINDOWS and ("No such file or directory" in text or "path too long" in text.lower()):
            hint = (
                "\nThis looks like Windows' 260 character path limit. Enable long paths from an "
                "Administrator PowerShell, then run setup again:\n"
                "  New-ItemProperty -Path HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
                "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
            )
        elif "No matching distribution" in text or "Could not find a version" in text:
            hint = (
                "\nNo wheel matches this Python (%s). GradMesh needs %s; Python %d.%d is the safest choice."
                % (platform.python_version(), python_requirement_text(), *PREFERRED_PYTHON)
            )
        elif "Read timed out" in text or "ConnectionError" in text or "Connection reset" in text:
            hint = (
                "\nThe download kept failing. Run setup again to resume, or fetch the wheels on a better "
                "connection and pass --wheelhouse (see SETUP.md)."
            )
        elif not text:
            hint = (
                "\nThe pip output above has the cause. Common ones: no internet access to "
                "download.pytorch.org, Windows' path length limit, or an unsupported Python. "
                "SETUP.md covers each."
            )
        raise RuntimeError("pip could not install %s.%s" % (requirements.name, hint))


VERIFY_SNIPPET = r"""
import json, sys, os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_build"] = getattr(torch.version, "cuda", None)
    def launch(device):
        x = torch.arange(1024, device=device, dtype=torch.float32)
        y = (x * 2.0 + 1.0).sum()
        return abs(float(y.item()) - (1023 * 1024 + 1024)) < 1e-2
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        out["device"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        out["capability"] = "%d.%d" % (major, minor)
        out["arch_list"] = list(torch.cuda.get_arch_list())
        try:
            out["cuda_kernel"] = launch("cuda:0")
        except Exception as exc:
            out["cuda_kernel"] = False
            out["cuda_error"] = "%s: %s" % (type(exc).__name__, exc)
    out["xpu"] = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    if out["xpu"]:
        out["device"] = torch.xpu.get_device_name(0)
        try:
            out["xpu_kernel"] = launch("xpu:0")
        except Exception as exc:
            out["xpu_kernel"] = False
            out["xpu_error"] = "%s: %s" % (type(exc).__name__, exc)
    mps = getattr(torch.backends, "mps", None)
    out["mps"] = bool(mps is not None and mps.is_available())
    if out["mps"]:
        out["device"] = "Apple GPU (Metal)"
        try:
            out["mps_kernel"] = launch("mps")
        except Exception as exc:
            out["mps_kernel"] = False
            out["mps_error"] = "%s: %s" % (type(exc).__name__, exc)
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
try:
    import ultralytics
    out["ultralytics"] = ultralytics.__version__
except Exception as exc:
    out["ultralytics_error"] = "%s: %s" % (type(exc).__name__, exc)
print("GRADMESH_VERIFY " + json.dumps(out))
"""


def verify(python: Path, backend: str) -> dict:
    """Exercise the installed build. Returns the report plus `ok` and `problem`."""
    try:
        result = subprocess.run(
            [str(python), "-c", VERIFY_SNIPPET], capture_output=True, text=True, timeout=300
        )
        line = next(
            (row for row in reversed(result.stdout.splitlines()) if row.startswith("GRADMESH_VERIFY ")), None
        )
        report = json.loads(line[len("GRADMESH_VERIFY "):]) if line else {"error": (result.stderr or "")[-800:]}
    except Exception as exc:
        report = {"error": "%s: %s" % (type(exc).__name__, exc)}

    problem = None
    if report.get("error"):
        problem = "PyTorch does not import: %s" % report["error"]
    elif backend == "cuda":
        if not report.get("cuda"):
            problem = (
                "this PyTorch build cannot see the NVIDIA GPU (torch.cuda.is_available() is False). "
                "The driver is usually missing or older than this build needs."
            )
        elif not report.get("cuda_kernel"):
            problem = (
                "the GPU is visible but a kernel launch failed (%s). This is the 'no kernel image' failure: "
                "the build has no kernels for compute %s." % (report.get("cuda_error", "unknown"), report.get("capability"))
            )
    elif backend == "xpu":
        if not report.get("xpu"):
            problem = "the Intel XPU runtime did not start (torch.xpu.is_available() is False). Install or update the Intel graphics driver."
        elif not report.get("xpu_kernel"):
            problem = "the Intel GPU is visible but a kernel launch failed (%s)." % report.get("xpu_error", "unknown")
    elif backend == "mps":
        if not report.get("mps"):
            problem = "the Metal backend is not available (torch.backends.mps.is_available() is False)."
        elif not report.get("mps_kernel"):
            problem = "Metal is visible but a kernel launch failed (%s)." % report.get("mps_error", "unknown")

    if not problem and report.get("ultralytics_error"):
        problem = "Ultralytics does not import: %s" % report["ultralytics_error"]

    report["ok"] = problem is None
    report["problem"] = problem
    return report


def install(
    venv: Path,
    plane: str,
    state: State,
    *,
    host: bool,
    prefer: str,
    wheelhouse: Optional[str],
    force: bool,
    quiet: bool,
    requirements_dir: Path = HERE,
) -> dict:
    if not python_supported():
        raise SystemExit(
            "GradMesh needs %s; this is Python %s.\n  Install it with: %s"
            % (python_requirement_text(), platform.python_version(), install_python_hint())
        )

    python = ensure_venv(venv, state)
    marker = _read_marker(venv)
    result: Dict[str, object] = {"venv": str(venv), "python": str(python)}

    if plane in {"control", "all"}:
        requirements = requirements_dir / "requirements-control.txt"
        stamp = requirements_stamp(requirements)
        state.update(controlPlane="installing", venv=str(venv), venvPython=str(python))
        if force or marker.get("control_stamp") != stamp:
            try:
                pip_install(python, requirements, state, wheelhouse, quiet)
            except Exception as exc:
                state.update(controlPlane="failed", controlError=str(exc))
                raise
            _write_marker(venv, control_stamp=stamp)
        state.update(controlPlane="ready")
        state.note("Control plane ready.")

    if plane in {"training", "all"}:
        facts = hardware.detect()
        profile = hardware.host_profile(facts, prefer) if host else hardware.select_profile_for(facts, prefer)
        result["profile"] = profile.as_dict()
        best = next((gpu for gpu in facts.gpus if gpu.name == profile.gpu_name), None)
        state.update(
            trainingPlane="installing",
            backend=profile.backend,
            profile=profile.name,
            profileLabel=profile.label,
            profileReason=profile.reason,
            reference=profile.reference,
            gpu=profile.gpu_name,
            computeCapability=best.compute_capability if best else None,
            driver=best.driver if best else None,
            warnings=profile.warnings,
            blocked=profile.blocked,
            fix=profile.fix,
            trainingError=None,
        )
        if profile.blocked and not host:
            state.update(trainingPlane="blocked")
            result["blocked"] = profile.blocked
            return result

        say("%s: %s" % (profile.gpu_name or "this machine", profile.reason))
        for warning in profile.warnings:
            say(warning, "yellow")
        state.note("Installing %s." % profile.label)

        requirements = requirements_dir / profile.requirements
        stamp = requirements_stamp(requirements)
        installed_profile = marker.get("training_profile")
        if force or marker.get("training_stamp") != stamp or installed_profile != profile.name:
            if installed_profile and installed_profile != profile.name:
                say("switching from %s to %s" % (installed_profile, profile.name), "yellow")
                # Different PyTorch builds share a version number, so pip would
                # consider the old one satisfied. Remove it first.
                subprocess.run(
                    [str(python), "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
                    capture_output=True,
                )
            try:
                pip_install(python, requirements, state, wheelhouse, quiet)
            except Exception as exc:
                state.update(trainingPlane="failed", trainingError=str(exc))
                state.note("Training plane install failed.")
                raise
            _write_marker(venv, training_stamp=stamp, training_profile=profile.name)

        say("checking that %s actually runs" % profile.backend)
        report = verify(python, profile.backend)
        result["verify"] = report
        accelerator = "ok" if report["ok"] else "unavailable"
        state.update(
            trainingPlane="ready" if not report.get("error") else "failed",
            torch=report.get("torch"),
            ultralytics=report.get("ultralytics"),
            verify=report,
            accelerator=accelerator,
            acceleratorProblem=report.get("problem"),
        )
        if report["ok"]:
            say("%s ready: torch %s on %s" % (profile.label, report.get("torch"), report.get("device", "CPU")), "green")
            state.note("Training plane ready. Runs can start now.")
        else:
            say(report["problem"], "red")
            if profile.fix:
                say(profile.fix, "yellow")
            state.note("Training plane installed, but the accelerator check failed.")
    return result


# ---------------------------------------------------------------------------
# The join flow
# ---------------------------------------------------------------------------


def agent_home() -> Path:
    override = os.environ.get("GRADMESH_AGENT_HOME")
    return Path(override) if override else Path.home() / ".gradmesh" / "agent"


def run_agent(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else agent_home()
    venv = home / ".venv"
    state = State(home / "setup.json")

    print("")
    say("GradMesh worker %s" % __version__, "green")
    say("joining %s" % args.server)
    print("")

    if not python_supported():
        say(
            "GradMesh needs %s; this is %s. Install it with:" % (python_requirement_text(), platform.python_version()),
            "red",
        )
        say(install_python_hint(), "yellow")
        return 2

    facts = hardware.detect()
    profile = hardware.select_profile_for(facts, args.backend)
    say("detected %s" % (profile.gpu_name or "no supported GPU"), "green")
    if profile.blocked:
        print("")
        say(profile.blocked, "red")
        if profile.fix:
            say(profile.fix, "yellow")
        return 2
    if not profile.can_train:
        say("%s. This machine will join and be measured, but it will not receive training work." % profile.reason, "yellow")

    try:
        result = install(
            venv,
            "all",
            state,
            host=False,
            prefer=args.backend,
            wheelhouse=args.wheelhouse,
            force=args.force,
            quiet=False,
        )
    except (RuntimeError, SystemExit) as exc:
        print("")
        say(str(exc), "red")
        say("Nothing was changed outside %s." % home, "gray")
        return 1

    report = result.get("verify") or {}
    if profile.can_train and not report.get("ok"):
        print("")
        say("The GPU could not run PyTorch: %s" % report.get("problem"), "red")
        if profile.fix:
            say(profile.fix, "yellow")
        say("Fix that and run the join command again. Everything is cached, so it is quick.", "gray")
        return 2

    if args.no_start:
        say("environment ready; not starting the worker (--no-start)", "green")
        return 0

    command = [
        str(venv_python(venv)),
        str(home / "worker.py"),
        "--server-url",
        args.server,
        "--token",
        args.token,
        "--backend",
        profile.backend if profile.can_train else "auto",
        "--gpu-index",
        str(profile.gpu_index),
        "--discover",
    ]
    if args.name:
        command += ["--name", args.name]
    print("")
    say("Contributing this GPU. Press Ctrl+C to leave the mesh.", "green")
    print("")
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify a GradMesh Python environment")
    sub = parser.add_subparsers(dest="command", required=True)

    detect_cmd = sub.add_parser("detect", help="Show this machine's GPU and the build it needs")
    detect_cmd.add_argument("--json", action="store_true")
    detect_cmd.add_argument("--host", action="store_true")
    detect_cmd.add_argument("--backend", default="auto", choices=["auto", "cuda", "xpu", "mps", "cpu"])

    install_cmd = sub.add_parser("install", help="Create or repair an environment and install a plane")
    install_cmd.add_argument("--venv", required=True)
    install_cmd.add_argument("--plane", default="all", choices=["control", "training", "all"])
    install_cmd.add_argument("--state", default=None)
    install_cmd.add_argument("--host", action="store_true")
    install_cmd.add_argument("--backend", default="auto", choices=["auto", "cuda", "xpu", "mps", "cpu"])
    install_cmd.add_argument("--wheelhouse", default=os.environ.get("GRADMESH_WHEELHOUSE"))
    install_cmd.add_argument("--force", action="store_true")
    install_cmd.add_argument("--quiet", action="store_true")
    install_cmd.add_argument("--json", action="store_true")

    verify_cmd = sub.add_parser("verify", help="Check that an environment's accelerator really runs")
    verify_cmd.add_argument("--venv", required=True)
    verify_cmd.add_argument("--backend", default="auto", choices=["auto", "cuda", "xpu", "mps", "cpu"])

    agent_cmd = sub.add_parser("agent", help="Set up this machine as a worker and join a mesh")
    agent_cmd.add_argument("--server", required=True)
    agent_cmd.add_argument("--token", required=True)
    agent_cmd.add_argument("--name", default="")
    agent_cmd.add_argument("--backend", default="auto", choices=["auto", "cuda", "xpu", "mps", "cpu"])
    agent_cmd.add_argument("--home", default=None)
    agent_cmd.add_argument("--wheelhouse", default=os.environ.get("GRADMESH_WHEELHOUSE"))
    agent_cmd.add_argument("--force", action="store_true")
    agent_cmd.add_argument("--no-start", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "detect":
        forwarded = ["--backend", args.backend] + (["--json"] if args.json else []) + (["--host"] if args.host else [])
        return hardware.main(forwarded)

    if args.command == "verify":
        backend = args.backend
        if backend == "auto":
            backend = hardware.select_profile_for(hardware.detect()).backend
        report = verify(venv_python(Path(args.venv)), backend)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if args.command == "agent":
        return run_agent(args)

    state = State(Path(args.state) if args.state else None)
    try:
        result = install(
            Path(args.venv),
            args.plane,
            state,
            host=args.host,
            prefer=args.backend,
            wheelhouse=args.wheelhouse,
            force=args.force,
            quiet=args.quiet,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    if result.get("blocked"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
