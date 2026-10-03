#!/usr/bin/env python3
"""Measure the Oct 20 demo's resource profile on the Jetson (U17 option a, D28).

Runs the three demo model components together, as the demo will run them, and
records whole-device memory, per-process memory and tegrastats while they work:

- llama-server (LFM2-VL) with v1's launcher flags and L4T's libcuda preloaded
  (decision D27); the run stops unless every layer and the vision encoder are
  on the GPU;
- the detector: the legacy yolov8n.engine through Ultralytics track() and
  ByteTrack at 15 fps;
- face: DeepFace Facenet512 with YuNet, TensorFlow on the CPU, at 1 Hz (decision
  D34: check 8 measured p50 896 ms per run on the CPU, so 2 Hz was unreachable).

The detector and face path run in one ~/onvif_env process (decision D24);
see demo_workload.py. Components load one after another with settle periods,
so each cold-load delta is visible, then run together for a warm-up and a
steady period, then unload.

The result is a provisional demo profile from one cold load (label
"provisional-demo"). It is not a benchmark, a Gate B record or a beta-gate
result. Memory follows guide ch. 12: whole-device use is MemTotal - MemAvailable
in decimal bytes; per-process RSS/PSS and tegrastats are separate views that
are never added together.

Run from a plain SSH session on the Jetson, headless (decision D29), with
VS Code and Claude Code closed; see docs/IMPLEMENTATION_STATUS.md, check 8.
Standard library only; run with the system Python 3. Nothing here reads camera
credentials, and model output text is never recorded. Output goes to
~/sentinel-runs/<run id>/ (outside the repository). Rebuild the summary from
the raw files with --summarize RUN_DIR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKLOAD = HERE / "demo_workload.py"
HOME = Path.home()
L4T_LIBCUDA = "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"
L4T_LIBCUDA_DIR = "/usr/lib/aarch64-linux-gnu/nvidia/"
LLAMA_FLAGS = ["--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1"]  # start_sentinel.sh
TARGET_STEADY_BYTES = 5_000_000_000  # guide ch. 12: decimal target
TARGET_PEAK_BYTES = 5_400_000_000  # guide ch. 12: decimal ceiling
MEMORY_FLOOR_BYTES = 300_000_000  # stop the run if MemAvailable stays below this
SAMPLE_INTERVAL_S = 0.2
PSS_INTERVAL_S = 1.0
LABEL = "provisional-demo"
EVENT_PREFIX = "@@EVENT "
MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree")
DESKTOP_COMMS = frozenset({"Xorg", "Xwayland", "gnome-shell", "Xtigervnc", "Xvnc", "xfwm4", "xfce4-session"})
V1_SCRIPTS = frozenset({"surveillance4_1.py", "dashboard.py", "dashboard_1.py"})
DEEPFACE_WEIGHTS = ("facenet512_weights.h5", "face_detection_yunet_2023mar.onnx")
PHASES_IN_ORDER = (
    "baseline", "llama_load", "llama_settle", "detector_load", "detector_settle",
    "face_load", "face_settle", "warmup", "steady", "unload_workload", "unload_llama",
)
COMPONENTS = (  # (key, load phase, settle phase, event carrying the load time)
    ("scene", "llama_load", "llama_settle", "llama_ready"),
    ("detector", "detector_load", "detector_settle", "detector_loaded"),
    ("face", "face_load", "face_settle", "face_loaded"),
)


class RunAborted(Exception):
    """The run cannot produce a valid profile; the message says why."""


class Interrupted(Exception):
    """SIGTERM or SIGHUP; cleanup still runs."""


# ---------------------------------------------------------------- /proc readers


def read_meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in MEMINFO_KEYS:
                values[key] = int(rest.split()[0]) * 1024
    return values


def read_process_memory(pid: int) -> tuple[int | None, int | None]:
    """VmRSS and VmHWM in bytes, or None once the process is gone."""
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None, None
    rss = re.search(r"^VmRSS:\s+(\d+) kB", text, re.M)
    hwm = re.search(r"^VmHWM:\s+(\d+) kB", text, re.M)
    return (int(rss.group(1)) * 1024 if rss else None, int(hwm.group(1)) * 1024 if hwm else None)


def read_pss(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return None
    match = re.search(r"^Pss:\s+(\d+) kB", text, re.M)
    return int(match.group(1)) * 1024 if match else None


def read_swap_counters() -> dict[str, int]:
    counters = {}
    with open("/proc/vmstat") as handle:
        for line in handle:
            key, value = line.split()
            if key in ("pswpin", "pswpout"):
                counters[key] = int(value)
    return counters


def scan_processes() -> list[tuple[int, str, str, list[str]]]:
    """(pid, comm, exe, argv) of visible processes. argv is matched, never printed."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            comm = Path(f"/proc/{entry}/comm").read_text().strip()
            argv = [part.decode(errors="replace") for part in Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        try:
            exe = os.readlink(f"/proc/{entry}/exe")
        except OSError:
            exe = ""
        found.append((int(entry), comm, exe, argv))
    return found


def top_memory_holders(count: int = 8) -> list[dict[str, object]]:
    rows = []
    for pid, comm, _exe, _argv in scan_processes():
        rss, _ = read_process_memory(pid)
        if rss:
            rows.append((rss, comm))
    rows.sort(reverse=True)
    return [{"comm": comm, "rss_bytes": rss} for rss, comm in rows[:count]]


def port_in_use(port: int) -> bool:
    """Something is listening on the loopback port (TIME_WAIT leftovers do not count)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def command_output(argv: list[str], env: dict[str, str] | None = None) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({type(exc).__name__})"
    return (done.stdout + done.stderr).strip()


def sha256_of(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------- preconditions


def deepface_weights() -> list[Path]:
    base = Path(os.environ.get("DEEPFACE_HOME", str(HOME))) / ".deepface" / "weights"
    return [base / name for name in DEEPFACE_WEIGHTS]


def model_files(args: argparse.Namespace) -> dict[str, Path]:
    files = {"engine": args.engine, "llm": args.model, "mmproj": args.mmproj}
    for path in deepface_weights():
        files[path.name] = path
    return files


def preconditions(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    problems = []
    if os.geteuid() == 0:
        problems.append("run this as your user, not as root")
    required = {"workload python": args.python, "llama-server": args.llama, "L4T libcuda": Path(L4T_LIBCUDA), **model_files(args)}
    if args.clip:
        required["clip"] = args.clip
    for label, path in required.items():
        if not Path(path).exists():
            problems.append(f"missing {label}: {path}")
    me = os.getpid()
    processes = [p for p in scan_processes() if p[0] != me]
    if any(comm == "llama-server" for _, comm, _, _ in processes):
        problems.append("a llama-server is already running; stop it first")
    if any(Path(arg).name in V1_SCRIPTS for _, _, _, argv in processes for arg in argv[:3]):
        problems.append("v1 (surveillance4_1.py or a dashboard) is running; stop it first")
    desktop = sorted({comm for _, comm, _, _ in processes if comm in DESKTOP_COMMS})
    display_manager = command_output(["systemctl", "is-active", "display-manager"])
    if (desktop or display_manager == "active") and not args.allow_desktop:
        problems.append(
            "desktop processes are running (" + ", ".join(desktop or ["display-manager"]) + "); the demo runs headless "
            "(D29): stop the display manager and any remote-desktop session, or pass --allow-desktop"
        )
    dev_tools = sorted(
        {"Claude Code" if comm == "claude" else "VS Code server" for _, comm, exe, _ in processes if comm == "claude" or "/.vscode-server/" in exe}
    )
    if dev_tools and not args.allow_dev_tools:
        problems.append(", ".join(dev_tools) + " running; close them for this measurement, or pass --allow-dev-tools")
    if port_in_use(args.port):
        problems.append(f"port {args.port} on 127.0.0.1 is in use")
    context = {"display_manager": display_manager, "desktop_processes": desktop, "dev_tools_running": dev_tools}
    return problems, context


def provenance(args: argparse.Namespace, context: dict[str, object], run_id: str) -> dict[str, object]:
    def file_facts(path: Path) -> dict[str, object] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
        return {"name": Path(path).name, "bytes": stat.st_size, "mtime_utc": mtime}

    preload_env = dict(os.environ, LD_PRELOAD=L4T_LIBCUDA)
    llama_version = [line for line in command_output([str(args.llama), "--version"], preload_env).splitlines() if line.startswith(("version", "built"))]
    git_commit = command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    git_dirty = bool(command_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]))
    return {
        "label": LABEL,
        "note": "One cold load and one combined run; not a benchmark, Gate B record or beta-gate result (U17 option a, D28).",
        "run_id": run_id,
        "started_utc": utc_now(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "kernel": os.uname().release,
        "l4t": command_output(["head", "-1", "/etc/nv_tegra_release"]).lstrip("# "),
        "power_mode": " ".join(command_output(["nvpmodel", "-q"]).split()),
        **context,
        "top_memory_holders": top_memory_holders(),
        "meminfo_at_start": read_meminfo(),
        "repository": {"commit": git_commit, "tracked_changes": git_dirty},
        "scripts_sha256": {path.name: sha256_of(path) for path in (Path(__file__).resolve(), WORKLOAD)},
        "llama_server": {"version": llama_version, "flags": LLAMA_FLAGS, "unified_memory": True, "preload": L4T_LIBCUDA},
        "files": {label: file_facts(path) for label, path in model_files(args).items()},
        "input": {"clip": file_facts(args.clip) if args.clip else None, "synthetic": not args.clip, "fps": args.fps},
        "parameters": {
            "face_hz": args.face_hz, "scene_interval_s": args.scene_interval_s, "baseline_s": args.baseline_s,
            "settle_s": args.settle_s, "warmup_s": args.warmup_s, "steady_s": args.steady_s, "evict_model_cache": args.evict,
            "min_free_gb": args.min_free_gb,
            "port": args.port, "sample_interval_s": SAMPLE_INTERVAL_S, "pss_interval_s": PSS_INTERVAL_S,
        },
    }


def evict_page_cache(paths: list[Path]) -> int:
    """Drop cached pages of these files so the load is cold. Needs no root; files are unchanged."""
    count = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        finally:
            os.close(fd)
    return count


# ------------------------------------------------------------------ recording


class Events:
    def __init__(self, path: Path, started: float) -> None:
        self._handle = open(path, "a")
        self._lock = threading.Lock()
        self._started = started

    def add(self, source: str, name: str, /, **fields: object) -> None:
        record = {"t_mono": round(time.monotonic(), 3), "utc": utc_now(), "source": source, "event": name, **fields}
        with self._lock:
            self._handle.write(json.dumps(record) + "\n")
            self._handle.flush()
        detail = " ".join(f"{k}={v}" for k, v in fields.items() if k not in ("libcuda", "buffers"))
        print(f"[{time.monotonic() - self._started:7.1f} s] {source}: {name} {detail}"[:220], flush=True)

    def close(self) -> None:
        self._handle.close()


class Sampler(threading.Thread):
    """Whole-device and per-process memory every SAMPLE_INTERVAL_S seconds."""

    COLUMNS = [
        "t_mono", "phase", "mem_total", "mem_free", "mem_available", "cached", "swap_total", "swap_free",
        "llama_rss", "llama_hwm", "llama_pss", "work_rss", "work_hwm", "work_pss", "pswpin", "pswpout",
    ]

    def __init__(self, path: Path, on_floor) -> None:
        super().__init__(name="sampler", daemon=True)
        self.path = path
        self.on_floor = on_floor
        self.phase = "start"
        self.pids: dict[str, int] = {}
        self.floor_hit = False
        self._low = 0
        self._halt = threading.Event()

    def run(self) -> None:
        next_slow = 0.0
        with open(self.path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            while not self._halt.is_set():
                now = time.monotonic()
                mem = read_meminfo()
                slow = now >= next_slow
                row: list[object] = [f"{now:.3f}", self.phase] + [mem.get(key, "") for key in MEMINFO_KEYS]
                for key in ("llama", "work"):
                    pid = self.pids.get(key)
                    rss, hwm = read_process_memory(pid) if pid else (None, None)
                    pss = read_pss(pid) if pid and slow else None
                    row += ["" if value is None else value for value in (rss, hwm, pss)]
                swap = read_swap_counters() if slow else {}
                row += [swap.get("pswpin", ""), swap.get("pswpout", "")]
                if slow:
                    next_slow = now + PSS_INTERVAL_S
                writer.writerow(row)
                handle.flush()
                self._check_floor(mem.get("MemAvailable"))
                self._halt.wait(max(0.0, SAMPLE_INTERVAL_S - (time.monotonic() - now)))

    def _check_floor(self, available: int | None) -> None:
        if available is None:
            return
        self._low = self._low + 1 if available < MEMORY_FLOOR_BYTES else 0
        if self._low >= 3 and not self.floor_hit:
            self.floor_hit = True
            self.on_floor()

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)


class Tegrastats(threading.Thread):
    """tegrastats at 1 s, each line prefixed with its receipt time and phase."""

    def __init__(self, path: Path, sampler: Sampler) -> None:
        super().__init__(name="tegrastats", daemon=True)
        self.path = path
        self.sampler = sampler
        self.proc: subprocess.Popen | None = None

    def run(self) -> None:
        with open(self.path, "w") as handle:
            try:
                self.proc = subprocess.Popen(
                    ["stdbuf", "-oL", "tegrastats", "--interval", "1000"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
            except OSError as exc:
                handle.write(f"unavailable: {type(exc).__name__}\n")
                return
            for line in self.proc.stdout:
                handle.write(f"{time.monotonic():.3f}\t{self.sampler.phase}\t{line}")
                handle.flush()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.join(timeout=5)


# ------------------------------------------------------------------ components


def start_llama(args: argparse.Namespace, run_dir: Path):
    env = dict(os.environ, LD_PRELOAD=L4T_LIBCUDA, GGML_CUDA_ENABLE_UNIFIED_MEMORY="1")
    log = open(run_dir / "llama-server.log", "wb")
    argv = [str(args.llama), "--model", str(args.model), "--mmproj", str(args.mmproj),
            "--host", "127.0.0.1", "--port", str(args.port), *LLAMA_FLAGS]
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env), log


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    """HTTP 200 from /health; llama-server answers 503 while it is still loading."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    return False


def llama_gpu_check(proc: subprocess.Popen, log_path: Path) -> dict[str, object]:
    """Decision D27: every layer and the vision encoder on the GPU, with L4T's libcuda only."""
    text = log_path.read_text(errors="replace")
    offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    try:
        maps = Path(f"/proc/{proc.pid}/maps").read_text()
    except OSError:
        maps = ""
    libcuda = sorted({line.split()[-1] for line in maps.splitlines() if "libcuda.so" in line})
    result: dict[str, object] = {
        "layers": offload.group(0) if offload else None,
        "all_layers_on_gpu": bool(offload) and offload.group(1) == offload.group(2),
        "vision_encoder_on_gpu": "CLIP using CUDA0" in text,
        "libcuda": libcuda,
        "l4t_libcuda_only": bool(libcuda) and all(path.startswith(L4T_LIBCUDA_DIR) for path in libcuda),
        "buffers": [line.split(":", 1)[1].strip() for line in text.splitlines()
                    if re.search(r"buffer size =\s+[\d.]+ MiB|^load_hparams: model size:", line)],
    }
    result["ok"] = bool(result["all_layers_on_gpu"] and result["vision_encoder_on_gpu"] and result["l4t_libcuda_only"])
    return result


def start_workload(args: argparse.Namespace, run_dir: Path) -> subprocess.Popen:
    env = dict(
        os.environ, LD_PRELOAD=L4T_LIBCUDA, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        PYTHONPATH=str(REPO / "src"), TF_CPP_MIN_LOG_LEVEL="3",
    )
    argv = [
        str(args.python), str(WORKLOAD), "--engine", str(args.engine), "--port", str(args.port),
        "--fps", str(args.fps), "--face-hz", str(args.face_hz), "--scene-interval-s", str(args.scene_interval_s),
        "--settle-s", str(args.settle_s), "--warmup-s", str(args.warmup_s), "--steady-s", str(args.steady_s),
    ]
    if args.clip:
        argv += ["--clip", str(args.clip)]
    # cwd is the run directory so no library can leave files in the repository.
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=run_dir)


def pump_workload(proc: subprocess.Popen, run_dir: Path, events: Events, set_phase) -> None:
    with open(run_dir / "workload.log", "w") as log:
        for line in proc.stdout:
            if line.startswith(EVENT_PREFIX):
                try:
                    record = json.loads(line[len(EVENT_PREFIX):])
                except json.JSONDecodeError:
                    log.write(line)
                    continue
                name = record.pop("event", "unknown")
                if name == "phase":
                    set_phase(str(record.get("name")), source="workload")
                else:
                    events.add("workload", name, **record)
            else:
                log.write(line)
                log.flush()


def stop_process(proc: subprocess.Popen | None, name: str, events: Events) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    events.add("orchestrator", "stopped", component=name, returncode=proc.returncode)


# ------------------------------------------------------------------------ run


def run(args: argparse.Namespace) -> int:
    problems, context = preconditions(args)
    if problems:
        print("Not starting:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    run_id = "demo-profile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out).expanduser() / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    manifest = provenance(args, context, run_id)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run {run_id}: writing to {run_dir}")

    events = Events(run_dir / "events.jsonl", time.monotonic())
    procs: dict[str, subprocess.Popen | None] = {"llama": None, "work": None}

    def on_floor() -> None:
        events.add("orchestrator", "memory_floor", floor_bytes=MEMORY_FLOOR_BYTES)
        for proc in procs.values():
            if proc is not None and proc.poll() is None:
                proc.terminate()

    sampler = Sampler(run_dir / "memory.csv", on_floor)
    tegrastats = Tegrastats(run_dir / "tegrastats.log", sampler)

    def set_phase(name: str, source: str = "orchestrator") -> None:
        sampler.phase = name
        events.add(source, "phase", name=name)

    def interrupted(signum, _frame) -> None:
        raise Interrupted(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    status = "incomplete"
    llama_log = None
    sampler.start()
    tegrastats.start()
    try:
        set_phase("baseline")
        time.sleep(args.baseline_s)
        if args.evict:
            before = read_meminfo()
            count = evict_page_cache(list(model_files(args).values()))
            after = read_meminfo()
            events.add("orchestrator", "evict_model_cache", files=count,
                       mem_free_before=before["MemFree"], mem_free_after=after["MemFree"])
        free = read_meminfo()["MemFree"]
        if free < args.min_free_gb * 1e9:
            # V2-01: GPU (NvMap) allocations here failed beyond MemFree even with GBs
            # of reclaimable cache; llama-server's own free-memory check misses it.
            raise RunAborted(
                f"MemFree is {free / 1e9:.2f} GB, below --min-free-gb {args.min_free_gb}: GPU allocations on "
                "this device need free, not merely reclaimable, memory. Reboot and run again, or drop caches first"
            )

        set_phase("llama_load")
        started = time.monotonic()
        procs["llama"], llama_log = start_llama(args, run_dir)
        sampler.pids["llama"] = procs["llama"].pid
        ready = wait_ready(args.port, procs["llama"], args.llama_timeout_s)
        events.add("orchestrator", "llama_ready", ready=ready, seconds=round(time.monotonic() - started, 2))
        if not ready:
            raise RunAborted("llama-server did not become ready; see llama-server.log")
        check = llama_gpu_check(procs["llama"], run_dir / "llama-server.log")
        events.add("orchestrator", "llama_gpu_check", **check)
        if not check["ok"]:
            raise RunAborted("llama-server is not fully on the GPU with L4T's libcuda (decision D27)")
        set_phase("llama_settle")
        time.sleep(args.settle_s)

        procs["work"] = start_workload(args, run_dir)
        sampler.pids["work"] = procs["work"].pid
        pump = threading.Thread(target=pump_workload, args=(procs["work"], run_dir, events, set_phase), daemon=True)
        pump.start()
        budget = args.load_timeout_s + 2 * args.settle_s + args.warmup_s + args.steady_s + 60
        try:
            returncode = procs["work"].wait(timeout=budget)
        except subprocess.TimeoutExpired:
            raise RunAborted("the workload exceeded its time budget") from None
        pump.join(timeout=10)
        events.add("orchestrator", "workload_exit", returncode=returncode)
        if sampler.floor_hit:
            raise RunAborted("MemAvailable stayed below the safety floor")
        if returncode != 0:
            raise RunAborted(f"the workload exited with status {returncode}; see workload.log")

        set_phase("unload_workload")
        time.sleep(args.settle_s)
        stop_process(procs["llama"], "llama-server", events)
        set_phase("unload_llama")
        time.sleep(args.settle_s)
        status = "complete"
    except RunAborted as exc:
        status = f"aborted: {exc}"
    except (Interrupted, KeyboardInterrupt) as exc:
        status = f"interrupted ({exc or 'SIGINT'})"
    finally:
        stop_process(procs["work"], "workload", events)
        stop_process(procs["llama"], "llama-server", events)
        if llama_log is not None:
            llama_log.close()
        set_phase("end")
        tegrastats.stop()
        sampler.stop()
        leftovers = [pid for pid, comm, _, _ in scan_processes() if comm == "llama-server"]
        events.add("orchestrator", "cleanup", llama_server_running=bool(leftovers), port_in_use=port_in_use(args.port))
        events.add("orchestrator", "run_end", status=status)
        events.close()
    manifest["finished_utc"] = utc_now()
    manifest["status"] = status
    manifest["sha256"] = {label: sha256_of(path) for label, path in model_files(args).items()}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summarize(run_dir))
    return 0 if status == "complete" else 1


# ------------------------------------------------------------------ summary


def _int(value: str) -> int | None:
    return int(value) if value not in ("", None) else None


def load_samples(path: Path) -> list[dict[str, object]]:
    samples = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            sample: dict[str, object] = {"t": float(row["t_mono"]), "phase": row["phase"]}
            for key in Sampler.COLUMNS[2:]:
                sample[key] = _int(row[key])
            sample["used"] = sample["mem_total"] - sample["mem_available"]
            sample["swap_used"] = sample["swap_total"] - sample["swap_free"]
            samples.append(sample)
    return samples


TEGRA_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
TEGRA_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
TEGRA_CPU = re.compile(r"CPU \[([^\]]*)\]")
TEGRA_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
TEGRA_TEMP = re.compile(r"(\b[a-z][a-z0-9]*)@(-?\d+(?:\.\d+)?)C\b")
TEGRA_POWER = re.compile(r"(VDD_\w+) (\d+)mW/(\d+)mW")


def parse_tegrastats_line(line: str) -> dict[str, object]:
    """One tegrastats line; its "MB" are MiB (the total matches MemTotal in MiB)."""
    parsed: dict[str, object] = {}
    if match := TEGRA_RAM.search(line):
        parsed["ram_used_bytes"] = int(match.group(1)) * 1_048_576
    if match := TEGRA_SWAP.search(line):
        parsed["swap_used_bytes"] = int(match.group(1)) * 1_048_576
    if match := TEGRA_CPU.search(line):
        cores = [int(item.split("%")[0]) for item in match.group(1).split(",") if "%" in item]
        if cores:
            parsed["cpu_cores"] = cores
    if match := TEGRA_GR3D.search(line):
        parsed["gr3d_percent"] = int(match.group(1))
    parsed["temps"] = {name: float(value) for name, value in TEGRA_TEMP.findall(line)}
    parsed["power_mw"] = {name: int(now) for name, now, _average in TEGRA_POWER.findall(line)}
    return parsed


def load_tegrastats(path: Path) -> list[dict[str, object]]:
    rows = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        parts = line.split("\t", 2)
        if len(parts) == 3:
            row = parse_tegrastats_line(parts[2])
            row["phase"] = parts[1]
            rows.append(row)
    return rows


def nearest_rank(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p * len(ordered))) - 1]


def gb(value: int | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value:,} B ({sign}{value / 1e9:.3f} GB)"


def summarize(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    samples = load_samples(run_dir / "memory.csv")
    tegra = load_tegrastats(run_dir / "tegrastats.log")

    def in_phases(*names: str) -> list[dict[str, object]]:
        return [s for s in samples if s["phase"] in names]

    def before_phase(name: str) -> dict[str, object] | None:
        previous = None
        for sample in samples:
            if sample["phase"] == name:
                return previous
            previous = sample
        return None

    def last_event(name: str) -> dict[str, object] | None:
        found = [e for e in events if e["event"] == name]
        return found[-1] if found else None

    def column_max(rows: list[dict[str, object]], key: str) -> int | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return max(values) if values else None

    end = last_event("run_end")
    status = end["status"] if end else "unknown (no run_end event)"
    lines = [
        f"{LABEL.upper()} RESOURCE PROFILE: {manifest['run_id']}",
        "One cold load and one combined run. Not a benchmark, Gate B record or beta-gate result.",
        f"Status: {status}",
        f"Device: {manifest.get('l4t')} | kernel {manifest.get('kernel')} | power mode {manifest.get('power_mode')}",
        f"Host state: display-manager {manifest.get('display_manager')}; desktop processes {manifest.get('desktop_processes') or 'none'}; "
        f"dev tools {manifest.get('dev_tools_running') or 'none'}",
        f"Input: {'synthetic noise frames' if manifest['input']['synthetic'] else 'replay clip'} at {manifest['input']['fps']} fps; "
        f"face {manifest['parameters']['face_hz']} Hz; scene every {manifest['parameters']['scene_interval_s']} s",
        f"Repository: {manifest['repository']['commit']} (tracked changes: {manifest['repository']['tracked_changes']})",
        "",
        "Whole-device memory = MemTotal - MemAvailable (decimal bytes).",
        "",
        f"{'phase':<16}{'seconds':>8}  {'used at start':>14}  {'used max':>14}  {'used at end':>14}  {'MemFree min':>14}  {'swap max':>12}",
    ]
    phases = [p for p in PHASES_IN_ORDER if in_phases(p)]
    for phase in phases:
        rows = in_phases(phase)
        lines.append(
            f"{phase:<16}{rows[-1]['t'] - rows[0]['t']:>8.1f}  {rows[0]['used']:>14,}  {column_max(rows, 'used'):>14,}  "
            f"{rows[-1]['used']:>14,}  {min(r['mem_free'] for r in rows):>14,}  {column_max(rows, 'swap_used'):>12,}"
        )

    baseline_rows = in_phases("baseline")
    baseline_used = int(statistics.median(r["used"] for r in baseline_rows)) if baseline_rows else None
    lines += ["", f"Baseline (median of baseline phase): {gb(baseline_used)}", "", "Cold loads, one after another:"]
    profile_components: dict[str, dict[str, object]] = {}
    for key, load_phase, settle_phase, load_event in COMPONENTS:
        reference = before_phase(load_phase)
        window = in_phases(load_phase, settle_phase)
        loaded = last_event(load_event)
        if reference is None or not window:
            lines.append(f"  {key}: not reached")
            continue
        peak = column_max(window, "used") - reference["used"]
        settled = window[-1]["used"] - reference["used"]
        seconds = loaded.get("seconds") if loaded else None
        profile_components[key] = {
            "cold_load_seconds": seconds,
            "baseline_delta_bytes": settled,
            "cold_load_peak_delta_bytes": peak,
            "mem_free_min_during_load_bytes": min(r["mem_free"] for r in window),
        }
        lines.append(f"  {key:<9} load {seconds if seconds is not None else 'n/a'} s; peak delta {gb(peak, True)}; settled delta {gb(settled, True)}")

    warm = in_phases("warmup", "steady")
    steady = in_phases("steady")
    combined: dict[str, object] = {}
    if warm:
        steady_used = [r["used"] for r in steady]
        run_rows = [r for r in samples if r["phase"] in PHASES_IN_ORDER]
        combined = {
            "baseline_bytes": baseline_used,
            "run_peak_bytes": column_max(run_rows, "used"),
            "warm_peak_bytes": column_max(warm, "used"),
            "steady_median_bytes": int(statistics.median(steady_used)) if steady_used else None,
            "steady_p95_bytes": nearest_rank(steady_used, 0.95),
            "swap_used_max_bytes": column_max(warm, "swap_used"),
            "llama_server_peak_rss_bytes": column_max(samples, "llama_hwm"),
            "llama_server_peak_pss_bytes": column_max(samples, "llama_pss"),
            "workload_peak_rss_bytes": column_max(samples, "work_hwm"),
            "workload_peak_pss_bytes": column_max(samples, "work_pss"),
        }
        swapin = [r["pswpin"] for r in steady if r.get("pswpin") is not None]
        swapout = [r["pswpout"] for r in steady if r.get("pswpout") is not None]
        if len(swapin) > 1:
            combined["steady_pages_swapped_in"] = swapin[-1] - swapin[0]
            combined["steady_pages_swapped_out"] = swapout[-1] - swapout[0]
        if baseline_used is not None:
            combined["warm_peak_minus_baseline_bytes"] = combined["warm_peak_bytes"] - baseline_used
        verdict_steady = combined["steady_p95_bytes"] is not None and combined["steady_p95_bytes"] <= TARGET_STEADY_BYTES
        verdict_peak = combined["run_peak_bytes"] is not None and combined["run_peak_bytes"] <= TARGET_PEAK_BYTES
        lines += [
            "",
            "All components together:",
            f"  run peak (includes cold loads) {gb(combined['run_peak_bytes'])}: "
            f"{'within' if verdict_peak else 'ABOVE'} the 5,400,000,000 B ceiling",
            f"  warm peak {gb(combined['warm_peak_bytes'])}; minus baseline {gb(combined.get('warm_peak_minus_baseline_bytes'), True)}",
            f"  steady median {gb(combined['steady_median_bytes'])}; steady p95 {gb(combined['steady_p95_bytes'])}: "
            f"{'within' if verdict_steady else 'ABOVE'} the 5,000,000,000 B target",
            f"  swap used max {gb(combined['swap_used_max_bytes'])}; steady pages swapped in/out "
            f"{combined.get('steady_pages_swapped_in', 'n/a')}/{combined.get('steady_pages_swapped_out', 'n/a')}",
            f"  llama-server peak RSS {gb(combined['llama_server_peak_rss_bytes'])}, peak PSS {gb(combined['llama_server_peak_pss_bytes'])}",
            f"  workload process peak RSS {gb(combined['workload_peak_rss_bytes'])}, peak PSS {gb(combined['workload_peak_pss_bytes'])}",
            "  (process RSS/PSS and whole-device use have different accounting; never add them)",
        ]

    unload: dict[str, object] = {}
    after_work = in_phases("unload_workload")
    before_work = in_phases("llama_settle")
    after_llama = in_phases("unload_llama")
    if after_work and before_work:
        unload["workload_residual_bytes"] = after_work[-1]["used"] - before_work[-1]["used"]
    if after_llama and baseline_used is not None:
        unload["residual_vs_baseline_bytes"] = after_llama[-1]["used"] - baseline_used
    if unload:
        lines += [
            "",
            "Unload:",
            f"  after the workload exits, vs before it started: {gb(unload.get('workload_residual_bytes'), True)}",
            f"  after llama-server stops, vs baseline: {gb(unload.get('residual_vs_baseline_bytes'), True)}",
        ]

    steady_tegra = [row for row in tegra if row["phase"] == "steady"]
    tegra_summary: dict[str, object] = {}
    if steady_tegra:
        def mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 1) if values else None

        cores = [row["cpu_cores"] for row in steady_tegra if "cpu_cores" in row]
        per_core = [mean([sample[i] for sample in cores if len(sample) > i]) for i in range(max(map(len, cores), default=0))]
        gr3d = [row["gr3d_percent"] for row in steady_tegra if "gr3d_percent" in row]
        temps: dict[str, float] = {}
        power: dict[str, list[int]] = {}
        for row in steady_tegra:
            for name, value in row["temps"].items():
                temps[name] = max(temps.get(name, value), value)
            for name, value in row["power_mw"].items():
                power.setdefault(name, []).append(value)
        tegra_summary = {
            "samples": len(steady_tegra),
            "ram_used_max_bytes": column_max(steady_tegra, "ram_used_bytes"),
            "cpu_mean_percent_all_cores": mean([sum(c) / len(c) for c in cores]),
            "cpu_mean_percent_busiest_core": max(per_core, default=None),
            "gr3d_mean_percent": mean(gr3d),
            "gr3d_max_percent": max(gr3d) if gr3d else None,
            "temps_max_c": temps,
            "power_mean_mw": {name: mean(values) for name, values in power.items()},
        }
        lines += [
            "",
            f"tegrastats, steady phase ({len(steady_tegra)} samples):",
            f"  RAM max {gb(tegra_summary['ram_used_max_bytes'])} (tegrastats accounting)",
            f"  CPU mean {tegra_summary['cpu_mean_percent_all_cores']} % over all cores; busiest core {tegra_summary['cpu_mean_percent_busiest_core']} %",
            f"  GR3D mean {tegra_summary['gr3d_mean_percent']} %, max {tegra_summary['gr3d_max_percent']} %",
            "  max temperature " + ", ".join(f"{k} {v:.1f} C" for k, v in sorted(temps.items())),
            "  mean power " + ", ".join(f"{k} {v} mW" for k, v in sorted(tegra_summary["power_mean_mw"].items())),
        ]

    stats = last_event("workload_stats")
    if stats:
        detector, face, scene = stats.get("detector", {}), stats.get("face", {}), stats.get("scene", {})
        lines += [
            "",
            f"Workload, steady phase ({stats.get('seconds')} s, {stats.get('input')}):",
            f"  detector: {detector.get('processed_frames')} of {detector.get('source_frames')} source frames, "
            f"{detector.get('processed_fps')} fps; latency ms {detector.get('latency_ms')}; frames with a person {detector.get('frames_with_person')}",
            f"  face: {face.get('runs')} runs ({face.get('achieved_hz')} Hz achieved), latency ms {face.get('latency_ms')}; "
            f"runs with a face {face.get('runs_with_face')}; errors {face.get('errors')}",
            f"  scene: {scene.get('completed')} completed, latency ms {scene.get('latency_ms')}; over the 8 s D16 timeout {scene.get('over_d16_timeout')}; "
            f"errors {scene.get('errors')}; valid/invalid/unchecked reports {scene.get('valid_reports')}/{scene.get('invalid_reports')}/{scene.get('unchecked_reports')}; "
            f"mean prompt/completion tokens {scene.get('prompt_tokens_mean')}/{scene.get('completion_tokens_mean')}",
        ]
        latency = {"detector": detector.get("latency_ms"), "face": face.get("latency_ms"), "scene": scene.get("latency_ms")}
        for key, values in latency.items():
            if key in profile_components:
                profile_components[key]["observed_latency_ms"] = values

    llama_check = last_event("llama_gpu_check")
    inputs = {
        "scene": {"image": "480x360 JPEG quality 60", "ctx_size": 2048, "max_tokens": 200, "parallel": 1,
                  "configured_timeout_ms": 8000, "gpu_check": llama_check},
        "detector": {"engine_input": "1x3x640x640 FP32", "source": "640x480", "rate_fps": manifest["input"]["fps"]},
        "face": {"input": "whole 640x480 frame; YuNet detector; Facenet512 embedding; TensorFlow on CPU",
                 "rate_hz": manifest["parameters"]["face_hz"]},
    }
    for key, extra in inputs.items():
        if key in profile_components:
            profile_components[key]["input_limit"] = extra
    profile = {
        "label": LABEL,
        "note": manifest.get("note"),
        "run_id": manifest["run_id"],
        "status": status,
        "provenance": {k: manifest.get(k) for k in ("started_utc", "finished_utc", "boot_id", "l4t", "kernel", "power_mode",
                                                   "display_manager", "repository", "llama_server", "files", "sha256", "input")},
        "components": profile_components,
        "combined": combined,
        "unload": unload,
        "tegrastats_steady": tegra_summary,
        "workload_steady": stats,
        "targets": {"steady_bytes": TARGET_STEADY_BYTES, "peak_bytes": TARGET_PEAK_BYTES},
    }
    (run_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(text)
    return text


# ------------------------------------------------------------------------ main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the Oct 20 demo resource profile (U17 option a).")
    parser.add_argument("--clip", type=Path, help="replay clip to decode as the camera stand-in (recommended)")
    parser.add_argument("--out", default=str(HOME / "sentinel-runs"), help="parent directory for run output")
    parser.add_argument("--port", type=int, default=18081, help="loopback port for llama-server")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--face-hz", type=float, default=1.0)  # D34
    parser.add_argument("--scene-interval-s", type=float, default=4.0)
    parser.add_argument("--baseline-s", type=float, default=30.0)
    parser.add_argument("--settle-s", type=float, default=15.0)
    parser.add_argument("--warmup-s", type=float, default=120.0)
    parser.add_argument("--steady-s", type=float, default=600.0)
    parser.add_argument("--llama-timeout-s", type=float, default=120.0)
    parser.add_argument("--load-timeout-s", type=float, default=300.0)
    parser.add_argument("--min-free-gb", type=float, default=3.0,
                        help="stop unless MemFree is at least this before llama-server starts (V2-01: it loaded "
                             "fully at 3.08 GB free and failed at 1.70 GB)")
    parser.add_argument("--no-evict", dest="evict", action="store_false",
                        help="keep the model files' page cache (default: evict it so the loads are cold)")
    parser.add_argument("--allow-desktop", action="store_true", help="measure with a desktop session running")
    parser.add_argument("--allow-dev-tools", action="store_true", help="measure with VS Code or Claude Code running")
    parser.add_argument("--python", type=Path, default=HOME / "onvif_env/bin/python")
    parser.add_argument("--llama", type=Path, default=HOME / "llama.cpp/build/bin/llama-server")
    parser.add_argument("--model", type=Path, default=HOME / "models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf")
    parser.add_argument("--mmproj", type=Path, default=HOME / "models/lfm2-vl/mmproj-LFM2-VL-1.6B-Q8_0.gguf")
    parser.add_argument("--engine", type=Path, default=HOME / "yolov8n.engine")
    parser.add_argument("--summarize", type=Path, metavar="RUN_DIR", help="only rebuild summary.txt and profile.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        print(summarize(args.summarize))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
