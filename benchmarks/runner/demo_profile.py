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

Memory CSV timestamps are sample-start time.monotonic() seconds within the
recorded boot. Linux meminfo kB values mean 1024 bytes; absent fields are blank
in CSV and None when loaded. Samples stream to disk every 0.2 seconds without
an in-memory history. Torch allocator events are a separate process-only view,
not total GPU/device usage or proof that process exit reclaimed memory.

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
import errno
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
if str(HERE) not in sys.path:  # loaded by path in tests; step4_criteria lives beside this file
    sys.path.insert(0, str(HERE))
import step4_criteria  # noqa: E402 - standard library only
WORKLOAD = HERE / "demo_workload.py"
HOME = Path.home()
L4T_LIBCUDA = "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"
L4T_LIBCUDA_DIR = "/usr/lib/aarch64-linux-gnu/nvidia/"
LLAMA_FLAGS = ["--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1"]  # start_sentinel.sh
# llama-server's host-RAM prompt cache (b8932 default 8192 MiB); the numbers in these lines are saved.
PROMPT_CACHE_STARTUP = re.compile(r"prompt cache is (enabled, size limit: (?:(\d+) MiB|no limit)|disabled)")
PROMPT_CACHE_STATE = re.compile(
    r"cache state: (\d+) prompts, ([\d.]+) MiB \(limits: ([\d.]+) MiB, (\d+) tokens, (\d+) est\)"
)
PROMPT_CACHE_MARKERS = {  # fixed llama-server texts -> saved label
    "prompt is already in the cache, skipping": "duplicate_skipped",
    "removing oldest entry": "evicted",
    "failed to allocate memory for prompt cache state": "allocation_failed",
}
TARGET_STEADY_BYTES = 5_000_000_000  # guide ch. 12: decimal target
TARGET_PEAK_BYTES = 5_400_000_000  # guide ch. 12: decimal ceiling
MEMORY_FLOOR_BYTES = 300_000_000  # stop the run if MemAvailable stays below this
SAMPLE_INTERVAL_S = 0.2
PSS_INTERVAL_S = 1.0
LABEL = "provisional-demo"
EVENT_PREFIX = "@@EVENT "
BASE_MEMINFO_KEYS = (
    "MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree",
    "Shmem", "Unevictable", "Mlocked", "SUnreclaim", "KReclaimable", "CmaFree",
)
# Attribution only (session 39): anonymous vs mapped memory and the file LRU split. No guard or
# criterion reads them; their memory.csv columns come last so the existing column positions stay.
ATTRIBUTION_MEMINFO_KEYS = ("AnonPages", "Mapped", "Active(file)", "Inactive(file)")
MEMINFO_KEYS = (*BASE_MEMINFO_KEYS, *ATTRIBUTION_MEMINFO_KEYS)
DESKTOP_COMMS = frozenset({"Xorg", "Xwayland", "gnome-shell", "Xtigervnc", "Xvnc", "xfwm4", "xfce4-session"})
V1_SCRIPTS = frozenset({"surveillance4_1.py", "dashboard.py", "dashboard_1.py"})
DEEPFACE_WEIGHTS = ("facenet512_weights.h5", "face_detection_yunet_2023mar.onnx")
PHASES_IN_ORDER = (  # *_release: --mr1-release-check or --post-load-release; smoke: --mr1-release-check only
    "baseline", "llama_load", "llama_settle", "scene_release", "detector_load", "detector_settle", "detector_release",
    "face_load", "face_settle", "face_release", "smoke", "warmup", "steady", "stopping", "unload_workload",
    "unload_llama",
)
# Events that carry monotonic boundaries and run-side evidence (kept in full, few per run).
BOUNDARY_EVENTS = frozenset({"steady_boundary", "stop_boundary", "llama_cmdline", "cuda_driver"})
LLAMA_BUILD_PREFIXES = ("libllama", "libggml", "libmtmd")
# MR1 (opt-in): each component's model files, released after its load and settle.
MR1_RELEASE_FILES = {"scene": ("llm", "mmproj"), "detector": ("engine",), "face": DEEPFACE_WEIGHTS}
MR1_RELEASE_SETTLE_S = 5.0  # /proc/meminfo counters fold in about every second
MR1_SMOKE_BUDGET_S = 60.0  # detector frames, one face analysis and one scene request (30 s client timeout)
MR1_FADVISE_CALL = "posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)"
MR1_BASIS = ("device-wide /proc/meminfo and per-process status before and after the release window; it includes any "
             "other activity in the window and shows neither per-file residency nor which pages were released")
MR1_SMOKE_KEYS = {
    "detector": ("frames_requested", "processed", "errors", "latency_ms"),
    "face": ("runs", "completed", "runs_with_face", "errors", "latency_ms"),
    "scene": ("attempts", "completed", "finish_reason", "valid", "rejection", "error", "errors", "latency_ms"),
}
MR1_SMOKE_LABELS = frozenset({"stop", "length", "other", "missing", "http", "timeout", "truncated", "invalid_report",
                              "incomplete_completion", "malformed_response", "server_error", "unsupported_completion",
                              "response_too_large"})
# Step-4 PLR (opt-in, D54): MR1's per-model release after each load and settle, then step 4's full warm-up and
# steady phases, judged by D47's criteria under step4_criteria.PLR_CRITERIA_ID. The outcomes stay descriptive.
PLR_LABEL = "step4plr-post-load-release"
PLR_CANNOT_ESTABLISH = (
    "per-file page-cache residency, or which pages or files were released: the counters are device-wide",
    "that a returned 0 dropped any page: mapped, dirty or locked pages stay",
    "how much of any headroom difference the releases caused: the run has no same-boot arm without them",
    "that sentinel run would behave the same: it performs no post-load release",
    "GPU (NvMap) allocation behaviour at low MemFree beyond this run",
    "model accuracy or scene accuracy",
    "other boots, other inputs or other flags",
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


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    """Convert Linux kB (1024 bytes) to bytes; omit missing/invalid fields."""
    values = {}
    with open(path) as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in MEMINFO_KEYS:
                fields = rest.split()
                if len(fields) == 2 and fields[0].isdigit() and fields[1] == "kB":
                    values[key] = int(fields[0]) * 1024
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
    if getattr(args, "scene_only", False):
        return {"llm": args.model, "mmproj": args.mmproj}
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
        if args.scene_only:
            problems.append("--scene-only uses its own synthetic images; do not pass --clip")
    if getattr(args, "mr1_release_check", False) and args.scene_only:
        problems.append("--mr1-release-check needs the detector and face models; do not pass --scene-only")
    if getattr(args, "post_load_release", False) and args.scene_only:
        problems.append("--post-load-release needs the detector and face models; do not pass --scene-only")
    if getattr(args, "post_load_release", False) and getattr(args, "mr1_release_check", False):
        problems.append("--mr1-release-check and --post-load-release are separate procedures; pass one")
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


def llama_flags(args: argparse.Namespace) -> list[str]:
    """The tracked flags, plus --cache-ram only when it was given (None keeps the server default)."""
    cache_ram = getattr(args, "llama_cache_ram", None)
    return LLAMA_FLAGS + ([] if cache_ram is None else ["--cache-ram", str(cache_ram)])


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
        "top_memory_holders": None if args.sanitized_logs else top_memory_holders(),
        "meminfo_at_start": read_meminfo(),
        "repository": {"commit": git_commit, "tracked_changes": git_dirty},
        "scripts_sha256": {path.name: sha256_of(path) for path in (Path(__file__).resolve(), WORKLOAD)},
        "llama_server": {"version": llama_version, "flags": llama_flags(args), "unified_memory": True,
                         "preload": L4T_LIBCUDA, "cache_ram_mib": args.llama_cache_ram,
                         "build_files": {label: file_facts(path) for label, path in llama_build_files(args.llama).items()},
                         "build_has_cache_ram_option": build_has_option(llama_build_files(args.llama))},
        "files": {label: file_facts(path) for label, path in model_files(args).items()},
        "input": {"clip": file_facts(args.clip) if args.clip else None, "synthetic": not args.clip, "fps": args.fps},
        "parameters": {
            "face_hz": args.face_hz, "scene_interval_s": args.scene_interval_s, "baseline_s": args.baseline_s,
            "settle_s": args.settle_s, "warmup_s": args.warmup_s, "steady_s": args.steady_s, "evict_model_cache": args.evict,
            "min_free_gb": args.min_free_gb, "scene_only": args.scene_only,
            "port": args.port, "sample_interval_s": SAMPLE_INTERVAL_S, "pss_interval_s": PSS_INTERVAL_S,
            **({"mr1_release_check": True, "mr1_release_settle_s": args.mr1_release_settle_s}
               if getattr(args, "mr1_release_check", False) else {}),
            **({"post_load_release": True, "release_settle_s": args.mr1_release_settle_s}
               if getattr(args, "post_load_release", False) else {}),
        },
        **({"mr1_note": "MR1 release check: loads and settles, each model's file cache released after its settle, "
                        "bounded smoke checks and unload. Not a resource profile, step-4 evidence or a sustained-memory test."}
           if getattr(args, "mr1_release_check", False) else {}),
        **({"plr_note": "Step-4 PLR (D54): step 4's procedure with each model's file cache released after its load and "
                        "settle, then the full warm-up and steady phases. Judged by D47's criteria under "
                        f"{step4_criteria.PLR_CRITERIA_ID}; never D47's step-4 result, accepted or admissible."}
           if getattr(args, "post_load_release", False) else {}),
    }


def llama_build_files(binary: Path) -> dict[str, Path]:
    """The llama-server binary and the llama/ggml/mtmd shared libraries beside it (real files, not links)."""
    files = {"llama-server": Path(binary)}
    try:
        for path in sorted(Path(binary).parent.iterdir()):
            if (path.name.startswith(LLAMA_BUILD_PREFIXES) and ".so" in path.name
                    and path.is_file() and not path.is_symlink()):
                files[path.name] = path
    except OSError:
        pass
    return files


def build_has_option(files: dict[str, Path], option: bytes = b"--cache-ram") -> bool | None:
    """Whether the installed build's files contain the option's text; None if they could not be read."""
    readable = False
    for name, path in files.items():
        if not (name == "llama-server" or name.startswith("libllama-common")):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        readable = True
        if option in data:
            return True
    return False if readable else None


def cmdline_evidence(pid: int, read=lambda pid: Path(f"/proc/{pid}/cmdline").read_bytes()) -> dict[str, object]:
    """The running server's own flags (no paths) and its --cache-ram value, from /proc/<pid>/cmdline."""
    try:
        argv = read(pid).decode(errors="replace").split("\0")
    except OSError:
        return {"seen": False, "cache_ram": None, "flags": None}
    argv = [arg for arg in argv[1:] if arg]
    flags, skip = [], False
    for index, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg in ("--model", "--mmproj"):
            skip = True
            continue
        flags.append(arg)
    values = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--cache-ram"]
    return {"seen": True, "cache_ram": values[-1] if values else None, "flags": flags}


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


def release_file_cache(path: Path, *, advise=None, clock=time.monotonic) -> dict[str, object]:
    """MR1: the evict_page_cache call on one whole file, with its exact outcome.

    ``returncode`` is posix_fadvise's own return value (0, or the error number it reported), or
    open()'s errno when the file could not be opened. A 0 return means the kernel accepted the advice,
    not that pages were dropped: mapped, dirty or locked pages stay. Needs no root; nothing is unmapped.
    """
    advise = advise or getattr(os, "posix_fadvise", None)
    record: dict[str, object] = {"name": Path(path).name, "bytes": None, "call": MR1_FADVISE_CALL,
                                 "result": None, "returncode": None, "error": None}
    started = clock()
    if advise is None:
        record["result"] = "unsupported"
    else:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as exc:
            record.update(result="open_failed", returncode=exc.errno, error=errno.errorcode.get(exc.errno, "unknown"))
        else:
            try:
                record["bytes"] = os.fstat(fd).st_size
                advise(fd, 0, 0, getattr(os, "POSIX_FADV_DONTNEED", 4))
                record.update(result="returned_0", returncode=0)
            except OSError as exc:
                record.update(result="returned_error", returncode=exc.errno,
                              error=errno.errorcode.get(exc.errno, "unknown"))
            finally:
                os.close(fd)
    record["elapsed_s"] = round(clock() - started, 6)
    return record


def mr1_snapshot(pids: dict[str, int | None], *, meminfo=None, process_memory=None, pss=None,
                 clock=time.monotonic) -> dict[str, object]:
    """Every MEMINFO_KEYS field (None when missing) and each process's RSS, HWM and PSS (None when unavailable)."""
    meminfo, process_memory, pss = meminfo or read_meminfo, process_memory or read_process_memory, pss or read_pss
    try:
        mem = meminfo()
    except (OSError, ValueError):
        mem = {}
    values = {key: mem.get(key) if type(mem.get(key)) is int and mem[key] >= 0 else None for key in MEMINFO_KEYS}
    total, available = values["MemTotal"], values["MemAvailable"]
    processes = {}
    for name, pid in pids.items():
        rss, hwm = process_memory(pid) if pid else (None, None)
        processes[name] = {"rss_bytes": rss, "hwm_bytes": hwm, "pss_bytes": pss(pid) if pid else None}
    return {"t_mono": round(clock(), 3), "meminfo": values,
            "pressure_bytes": total - available if total is not None and available is not None else None,
            "processes": processes}


def snapshot_deltas(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    """after - before for each field present in both; None otherwise (never zero)."""
    def delta(a, b):
        return b - a if type(a) is int and type(b) is int else None

    deltas: dict[str, object] = {key: delta(before["meminfo"].get(key), after["meminfo"].get(key)) for key in MEMINFO_KEYS}
    deltas["pressure_bytes"] = delta(before.get("pressure_bytes"), after.get("pressure_bytes"))
    deltas["processes"] = {
        name: {key: delta((before["processes"].get(name) or {}).get(key), value) for key, value in fields.items()}
        for name, fields in after["processes"].items()
    }
    return deltas


def release_outcome(files: dict[str, dict[str, object]], deltas: dict[str, object]) -> str:
    """A descriptive label from the calls' results and the directions of MemFree and Cached; no magnitude threshold.

    call_failed: a call did not return 0 (or the file could not be opened); telemetry_unavailable: MemFree or
    Cached is missing before or after; ineffective: every call returned 0 but MemFree did not rise and Cached
    did not fall; partial: only one of the two moved that way; memfree_rose_cached_fell: both did.
    """
    if not files or any(item.get("result") != "returned_0" for item in files.values()):
        return "call_failed"
    free, cached = deltas.get("MemFree"), deltas.get("Cached")
    if free is None or cached is None:
        return "telemetry_unavailable"
    rose, fell = free > 0, cached < 0
    return "memfree_rose_cached_fell" if rose and fell else "partial" if rose or fell else "ineffective"


def release_component(component: str, files: dict[str, Path], pids: dict[str, int | None], settle_s: float, *,
                      snapshot=None, release=None, sleep=time.sleep, clock=time.monotonic) -> dict[str, object]:
    """MR1: sample, release each of this component's model files, wait settle_s, sample again."""
    snapshot, release = snapshot or (lambda: mr1_snapshot(pids)), release or release_file_cache
    before = snapshot()
    started = clock()
    results = {label: release(path) for label, path in files.items()}
    calls_elapsed = round(clock() - started, 6)
    sleep(settle_s)
    after = snapshot()
    deltas = snapshot_deltas(before, after)
    sizes = [item.get("bytes") for item in results.values()]
    return {
        "component": component, "files": results,
        "files_total_bytes": sum(sizes) if sizes and all(type(size) is int for size in sizes) else None,
        "files_total_bytes_note": "the most these files' cache can occupy; not how much of it was cached or released",
        "calls_elapsed_s": calls_elapsed, "settle_s": settle_s, "before": before, "after": after, "deltas": deltas,
        "outcome": release_outcome(results, deltas), "basis": MR1_BASIS,
    }


def mr1_files(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    files = model_files(args)
    return {component: {label: files[label] for label in labels} for component, labels in MR1_RELEASE_FILES.items()}


def mr1_checkpoint_handler(procs: dict, events, set_phase, files: dict[str, dict[str, Path]], settle_s: float, *,
                           release=release_component, snapshot=mr1_snapshot):
    """MR1: the orchestrator's side of each checkpoint, run while the workload waits.

    "scene", "detector" and "face" release that component's files with samples around them; "after_smoke"
    samples only. Then ``ack <stage>`` goes to the workload. An unexpected error is recorded by class and
    sends no acknowledgement, so the workload stops itself after its bounded wait.
    """
    def pids() -> dict[str, int | None]:
        return {name: proc.pid if proc is not None else None for name, proc in procs.items()}

    def handle(stage: str) -> None:
        try:
            if stage == "after_smoke":
                events.add("orchestrator", "mr1_snapshot", stage=stage, **snapshot(pids()))
            else:
                set_phase(f"{stage}_release")
                events.add("orchestrator", "mr1_release", **release(stage, files[stage], pids(), settle_s))
        except Exception as exc:  # noqa: BLE001 - by class only
            events.add("orchestrator", "mr1_error", stage=stage, error=type(exc).__name__)
            return
        if stage == "scene":
            return  # before the workload starts: nothing waits for it
        try:
            procs["work"].stdin.write(f"ack {stage}\n")
            procs["work"].stdin.flush()
        except (AttributeError, OSError, ValueError):
            pass  # the workload is gone; it stops itself without the acknowledgement

    return handle


def sanitize_mr1_smoke(record: object) -> dict[str, object]:
    """Fixed keys, numbers, booleans, fixed labels and exception class names only."""
    def label(value):
        if value is None or type(value) is bool:
            return value
        if isinstance(value, str) and (value in MR1_SMOKE_LABELS or re.fullmatch(r"[A-Za-z]{1,40}(?:Error|Exception)", value)):
            return value
        return None

    def number(value):
        if type(value) is int:
            return value if value.bit_length() <= 63 else None
        return value if type(value) is float and math.isfinite(value) else None

    out: dict[str, object] = {}
    record = record if isinstance(record, dict) else {}
    for part, keys in MR1_SMOKE_KEYS.items():
        source = record.get(part) if isinstance(record.get(part), dict) else {}
        clean: dict[str, object] = {}
        for key in keys:
            value = source.get(key)
            if key == "errors":
                clean[key] = {name: count for name, count in list(value.items())[:16]
                              if (label(name) is not None or re.fullmatch(r"HTTP [1-5]\d\d", str(name)))
                              and type(count) is int} if isinstance(value, dict) else None
            elif key == "latency_ms" and isinstance(value, dict):
                clean[key] = {k: number(v) for k, v in value.items() if k in ("n", "p50", "p95", "p99", "max")}
            elif key in ("finish_reason", "rejection", "error"):
                clean[key] = label(value)
            elif key == "valid":
                clean[key] = value if type(value) is bool else None
            else:
                clean[key] = number(value)
        out[part] = clean
    return out


def functional_smoke(smoke: dict[str, object] | None) -> str:
    """pass: every detector frame processed without error, one face analysis completed without error and one
    scene request completed with finish reason stop and a strict-valid report; fail: any of those not met;
    unavailable: no smoke record."""
    if not smoke:
        return "unavailable"
    detector, face, scene = (smoke.get(key) or {} for key in ("detector", "face", "scene"))
    ok = (type(detector.get("processed")) is int and detector.get("processed") == detector.get("frames_requested")
          and not detector.get("errors")
          and face.get("completed") == 1 and not face.get("errors")
          and scene.get("completed") == 1 and scene.get("finish_reason") == "stop" and scene.get("valid") is True
          and scene.get("error") is None)
    return "pass" if ok else "fail"


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
        "shmem", "unevictable", "mlocked", "s_unreclaim", "k_reclaimable", "cma_free",
        "llama_rss", "llama_hwm", "llama_pss", "work_rss", "work_hwm", "work_pss", "pswpin", "pswpout",
        "anon_pages", "mapped", "active_file", "inactive_file",  # ATTRIBUTION_MEMINFO_KEYS, in order
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
                row: list[object] = [f"{now:.3f}", self.phase] + [mem.get(key, "") for key in BASE_MEMINFO_KEYS]
                for key in ("llama", "work"):
                    pid = self.pids.get(key)
                    rss, hwm = read_process_memory(pid) if pid else (None, None)
                    pss = read_pss(pid) if pid and slow else None
                    row += ["" if value is None else value for value in (rss, hwm, pss)]
                swap = read_swap_counters() if slow else {}
                row += [swap.get("pswpin", ""), swap.get("pswpout", "")]
                row += [mem.get(key, "") for key in ATTRIBUTION_MEMINFO_KEYS]
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
            "--host", "127.0.0.1", "--port", str(args.port), *llama_flags(args)]
    if not getattr(args, "sanitized_logs", False):
        return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env), log
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    proc.log_pump = threading.Thread(target=pump_llama_diagnostics, args=(proc, log), daemon=True)
    proc.log_pump.start()
    return proc, log


def pump_llama_diagnostics(proc, log, clock=time.monotonic) -> None:
    """Save only fixed GPU-placement markers, numeric buffer sizes and numeric prompt-cache lines.

    Prompt-cache lines get the receipt time.monotonic() so they can be aligned with memory.csv.
    """
    try:
        while line := proc.stdout.readline(65_536):
            text = line.decode(errors="replace")
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
            buffer = re.search(r"buffer size =\s*([\d.]+) MiB", text)
            startup = PROMPT_CACHE_STARTUP.search(text)
            state = PROMPT_CACHE_STATE.search(text)
            marker = next((label for key, label in PROMPT_CACHE_MARKERS.items() if key in text), None)
            if offload:
                log.write(f"offloaded {offload[1]}/{offload[2]} layers to GPU\n".encode())
            if "CLIP using CUDA0" in text:
                log.write(b"CLIP using CUDA0\n")
            if buffer:
                log.write(f"diagnostic: buffer size = {buffer[1]} MiB\n".encode())
            stamp = f"t_mono={clock():.3f}"
            if startup:
                limit = "disabled" if startup[1] == "disabled" else (
                    f"enabled, size limit: {startup[2]} MiB" if startup[2] else "enabled, size limit: no limit")
                log.write(f"{stamp} prompt cache is {limit}\n".encode())
            if state:
                log.write((f"{stamp} prompt cache state: {state[1]} prompts, {state[2]} MiB "
                           f"(limits: {state[3]} MiB, {state[4]} tokens, {state[5]} est)\n").encode())
            if marker:
                log.write(f"{stamp} prompt cache event: {marker}\n".encode())
            log.flush()
    except (OSError, ValueError):
        return


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
    if args.scene_only:
        argv.append("--scene-only")
    releasing = [flag for flag, on in (("--mr1-release-check", getattr(args, "mr1_release_check", False)),
                                       ("--post-load-release", getattr(args, "post_load_release", False))) if on]
    if releasing:  # MR1 or PLR: the workload waits on stdin at each checkpoint
        argv += releasing
        return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env, cwd=run_dir)
    # cwd is the run directory so no library can leave files in the repository.
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=run_dir)


def sanitize_diagnostic(value, depth: int = 0):
    """Bound trusted workload telemetry; never save arbitrary text/unknown fields."""
    keys = {
        "name", "event", "t_mono", "phase", "cuinit", "libcuda", "ok", "seconds",
        "torch", "tensorflow", "tf_visible_gpus", "clock", "units", "device", "peak_scope",
        "status", "error", "allocated_bytes", "allocated_peak_bytes", "reserved_bytes",
        "reserved_peak_bytes", "detector", "face", "scene", "source_frames", "processed_frames",
        "processed_fps", "latency_ms", "frames_with_person", "max_persons", "runs", "achieved_hz",
        "runs_with_face", "errors", "completed", "over_d16_timeout", "valid_reports",
        "invalid_reports", "unchecked_reports", "rejected_reports_by_reason", "finish_reasons",
        "valid_summaries_at_limit", "valid_observations_at_limit", "accuracy",
        "prompt_tokens_mean", "completion_tokens_mean", "input", "clip_loops",
        "n", "p50", "p95", "p99", "max", "stop", "length", "other", "missing",
        "truncated", "invalid_report", "incomplete_completion", "malformed_response", "timeout",
        "server_error", "unsupported_completion", "requests", "http", "prompt_tokens", "completion_tokens",
        "synthetic_images_issued", "edge", "boundary_t_mono", "unique_fps", "windows", "window_s", "count",
        "min_fps", "max_fps", "schedule_age_ms", "decode_to_result_age_ms", "attempts", "client_timeouts",
        "http_errors", "transport_errors", "request_sha256", "error_count",
    }
    strings = {
        *PHASES_IN_ORDER, "phase", "cuda_driver", "detector_loaded", "face_loaded",
        "torch_allocator", "workload_stats", "scene_progress", "time.monotonic", "bytes", "allocator_lifetime",
        "observed", "unavailable", "synthetic noise", "synthetic noise, distinct per request",
        "not evaluated; structural validity is not scene accuracy", "steady_boundary", "start", "end",
    }
    if depth > 6:
        return None
    if isinstance(value, dict):
        return {key: sanitize_diagnostic(item, depth + 1) for key, item in list(value.items())[:64]
                if key in keys or re.fullmatch(r"[A-Za-z]{1,40}(?:Error|Exception)|HTTP [1-5]\d\d", key)}
    if isinstance(value, list):
        return [sanitize_diagnostic(item, depth + 1) for item in value[:16]]
    if type(value) is int:
        return value if value.bit_length() <= 63 else None
    if type(value) is float:
        return value if math.isfinite(value) else None
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        if value in strings or re.fullmatch(r"\d+(?:\.\d+){1,3}(?:[A-Za-z0-9.+-]{0,24})", value):
            return value
        if re.fullmatch(r"[0-9a-f]{64}", value):  # a SHA-256 fingerprint
            return value
        if re.fullmatch(r"[A-Za-z]{1,40}(?:Error|Exception)", value):
            return value
        if value.startswith(L4T_LIBCUDA_DIR) and re.fullmatch(r"libcuda\.so(?:\.\d+)*", Path(value).name):
            return value
    return None


def pump_workload(proc: subprocess.Popen, run_dir: Path, events: Events, set_phase, *, sanitized: bool = False,
                  on_checkpoint=None) -> None:
    """Relay the workload's events. ``on_checkpoint`` (MR1 only) handles ``mr1_checkpoint`` and ``mr1_smoke``."""
    mr1 = on_checkpoint is not None
    with open(run_dir / "workload.log", "w") as log:
        while line := proc.stdout.readline(65_536):
            if line.startswith(EVENT_PREFIX):
                try:
                    record = json.loads(line[len(EVENT_PREFIX):])
                except json.JSONDecodeError:
                    if not sanitized:
                        log.write(line)
                    continue
                if mr1 and isinstance(record, dict) and record.get("event") == "mr1_checkpoint":
                    stage = record.get("stage")
                    if isinstance(stage, str) and stage in ("detector", "face", "after_smoke"):
                        on_checkpoint(stage)
                    continue
                if mr1 and isinstance(record, dict) and record.get("event") == "mr1_smoke":
                    events.add("workload", "mr1_smoke", **sanitize_mr1_smoke(record))
                    continue
                if sanitized:
                    if not isinstance(record, dict) or record.get("event") not in {
                        "phase", "cuda_driver", "detector_loaded", "face_loaded", "torch_allocator", "workload_stats",
                        "scene_progress", "steady_boundary",
                    }:
                        continue
                    record = sanitize_diagnostic(record)
                name = record.pop("event", "unknown")
                if name == "phase":
                    if sanitized and record.get("name") not in PHASES_IN_ORDER:
                        continue
                    set_phase(str(record.get("name")), source="workload")
                else:
                    events.add("workload", name, **record)
            else:
                if not sanitized:
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


def workload_budget(args: argparse.Namespace) -> float:
    """Seconds the workload may take from its start to its exit before the run aborts."""
    if args.mr1_release_check:  # loads, two releases with their checkpoints, the smoke checks, no warm-up/steady
        return args.load_timeout_s + 2 * args.settle_s + 3 * (args.mr1_release_settle_s + 10) + MR1_SMOKE_BUDGET_S + 60
    if args.post_load_release:  # step 4's budget plus the two releases the workload waits for
        return (args.load_timeout_s + 2 * args.settle_s + 2 * (args.mr1_release_settle_s + 10)
                + args.warmup_s + args.steady_s + 60)
    return args.load_timeout_s + 2 * args.settle_s + args.warmup_s + args.steady_s + 60


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
        events.add("orchestrator", "stop_boundary", reason="memory_floor", boundary_t_mono=round(time.monotonic(), 3))
        for proc in procs.values():
            if proc is not None and proc.poll() is None:
                proc.terminate()

    sampler = Sampler(run_dir / "memory.csv", on_floor)
    tegrastats = Tegrastats(run_dir / "tegrastats.log", sampler)

    def set_phase(name: str, source: str = "orchestrator") -> None:
        sampler.phase = name
        events.add(source, "phase", name=name)

    stop_mark: dict[str, float] = {}

    def interrupted(signum, _frame) -> None:
        stop_mark.setdefault("t", time.monotonic())  # no lock here; the event is written in the handler below
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
        events.add("orchestrator", "llama_cmdline", **cmdline_evidence(procs["llama"].pid))
        if not check["ok"]:
            raise RunAborted("llama-server is not fully on the GPU with L4T's libcuda (decision D27)")
        set_phase("llama_settle")
        time.sleep(args.settle_s)
        on_checkpoint = None
        if args.mr1_release_check or args.post_load_release:
            on_checkpoint = mr1_checkpoint_handler(procs, events, set_phase, mr1_files(args), args.mr1_release_settle_s)
            on_checkpoint("scene")  # llama-server has settled; the workload has not started

        procs["work"] = start_workload(args, run_dir)
        sampler.pids["work"] = procs["work"].pid
        pump = threading.Thread(
            target=pump_workload, args=(procs["work"], run_dir, events, set_phase),
            kwargs={"sanitized": args.sanitized_logs, **({"on_checkpoint": on_checkpoint} if on_checkpoint else {})},
            daemon=True,
        )
        pump.start()
        try:
            returncode = procs["work"].wait(timeout=workload_budget(args))
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
        events.add("orchestrator", "stop_boundary", reason="aborted", boundary_t_mono=round(time.monotonic(), 3))
    except (Interrupted, KeyboardInterrupt) as exc:
        status = f"interrupted ({exc or 'SIGINT'})"
        events.add("orchestrator", "stop_boundary", reason="interrupted",
                   boundary_t_mono=round(stop_mark.get("t", time.monotonic()), 3))
    finally:
        stop_process(procs["work"], "workload", events)
        stop_process(procs["llama"], "llama-server", events)
        if llama_log is not None:
            log_pump = getattr(procs["llama"], "log_pump", None)
            if log_pump is not None:
                log_pump.join(timeout=1)
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
    manifest["sha256_build"] = {label: sha256_of(path) for label, path in llama_build_files(args.llama).items()}
    manifest["sha256_clip"] = sha256_of(args.clip) if args.clip else None  # local run record only
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summarize(run_dir))
    return 0 if status == "complete" else 1


# ------------------------------------------------------------------ summary


def _int(value: str | None) -> int | None:
    return int(value) if value not in ("", None) else None


def load_samples(path: Path) -> list[dict[str, object]]:
    samples = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            sample: dict[str, object] = {"t": float(row["t_mono"]), "phase": row["phase"]}
            for key in Sampler.COLUMNS[2:]:
                sample[key] = _int(row.get(key))
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
            try:
                row["t"] = float(parts[0])
            except ValueError:
                pass
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


def load_boundary_events(path: Path) -> list[dict[str, object]]:
    """Every boundary/evidence event (BOUNDARY_EVENTS), in order; a few per run."""
    found = []
    try:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("event") in BOUNDARY_EVENTS:
                        found.append(record)
    except OSError:
        pass
    return found


def load_latest_events(path: Path) -> dict[str, dict[str, object]]:
    """Retain at most seven summary events; allocator and per-request history stays on disk."""
    names = {"run_end", "llama_ready", "detector_loaded", "face_loaded", "workload_stats", "llama_gpu_check",
             "scene_progress", "cuda_driver", "llama_cmdline"}
    latest = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["event"] in names:
                    latest[record["event"]] = record
    return latest


MR1_CANNOT_ESTABLISH = (
    "per-file page-cache residency, or which pages or files were released: the counters are device-wide",
    "that a returned 0 dropped any page: mapped, dirty or locked pages stay",
    "behaviour under sustained load: the smoke checks are single, bounded runs",
    "whether the released headroom is enough for step 4, or any step-4 criterion",
    "GPU (NvMap) allocation behaviour at low MemFree beyond this run",
    "model accuracy or scene accuracy",
    "other boots, other inputs or other flags",
)


def load_mr1_events(path: Path) -> list[dict[str, object]]:
    found = []
    try:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("event") in ("mr1_release", "mr1_snapshot", "mr1_smoke"):
                        found.append(record)
    except OSError:
        pass
    return found


def release_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Each component's release record (MR1 or PLR), keyed by component."""
    return {r["component"]: {k: r.get(k) for k in ("files", "files_total_bytes", "calls_elapsed_s", "settle_s",
                                                      "before", "after", "deltas", "outcome")}
            for r in records if r["event"] == "mr1_release" and r.get("component") in MR1_RELEASE_FILES}


def _status_label(status: str) -> str:
    return next((name for name in ("complete", "aborted", "interrupted", "incomplete") if status.startswith(name)), "unknown")


def mr1_report(run_dir: Path, status: str) -> dict[str, object]:
    """MR1's record: each release, the after-smoke sample, the smoke checks; descriptive, never acceptance."""
    records = load_mr1_events(Path(run_dir) / "events.jsonl")
    releases = release_records(records)
    after_smoke = next((r for r in records if r["event"] == "mr1_snapshot"), None)
    after_smoke = {k: after_smoke.get(k) for k in ("t_mono", "meminfo", "pressure_bytes", "processes")} if after_smoke else None
    smoke = next(({k: r.get(k) for k in MR1_SMOKE_KEYS} for r in records if r["event"] == "mr1_smoke"), None)
    face = releases.get("face")
    label = _status_label(status)
    return {
        "label": "mr1-release-check",
        "scope": "loads and settles, each model's file cache released after its settle, bounded smoke checks; "
                 "not step-4 acceptance, a resource profile or a sustained-memory test",
        "profile_status": label,
        "releases": releases,
        "after_smoke": after_smoke,
        "after_smoke_vs_face_release": snapshot_deltas(face["after"], after_smoke) if face and after_smoke else None,
        "smoke": smoke,
        "functional_smoke": functional_smoke(smoke),
        "basis": MR1_BASIS,
        "cannot_establish": list(MR1_CANNOT_ESTABLISH),
    }


def release_lines(releases: dict[str, dict[str, object]]) -> list[str]:
    lines = []
    for component in MR1_RELEASE_FILES:
        item = releases.get(component)
        if item is None:
            lines.append(f"  {component}: not reached")
            continue
        d = item["deltas"]
        calls = ", ".join(f"{label} {f.get('result')}" + (f" ({f.get('error')})" if f.get("error") else "")
                          for label, f in item["files"].items())
        lines.append(f"  {component}: {item['outcome']}; calls: {calls}; files total {gb(item['files_total_bytes'])}")
        lines.append(f"    deltas over {item['settle_s']} s: MemFree {gb(d.get('MemFree'), True)}, Cached {gb(d.get('Cached'), True)}, "
                     f"Mapped {gb(d.get('Mapped'), True)}, AnonPages {gb(d.get('AnonPages'), True)}, "
                     f"Inactive(file) {gb(d.get('Inactive(file)'), True)}, pressure {gb(d.get('pressure_bytes'), True)}")
    return lines


def mr1_lines(report: dict[str, object]) -> list[str]:
    lines = ["", "MR1 release check (descriptive; not step-4 evidence, never acceptance):", f"  basis: {MR1_BASIS}"]
    lines += release_lines(report["releases"])
    after = report.get("after_smoke_vs_face_release")
    if after:
        lines.append(f"  after the smoke checks vs after the face release: MemFree {gb(after.get('MemFree'), True)}, "
                     f"Cached {gb(after.get('Cached'), True)}, pressure {gb(after.get('pressure_bytes'), True)}")
    lines.append(f"  smoke checks: {report['functional_smoke']} {report['smoke']}")
    lines.append("  Cannot establish: " + "; ".join(report["cannot_establish"]) + ".")
    return lines


def _memory_window(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """MemFree's minimum and Cached at the first and last sample of a window; None without rows."""
    if not rows:
        return None
    free = [r["mem_free"] for r in rows if r.get("mem_free") is not None]
    cached = [r["cached"] for r in rows if r.get("cached") is not None]
    return {"samples": len(rows), "mem_free_min_bytes": min(free) if free else None,
            "cached_first_bytes": cached[0] if cached else None, "cached_last_bytes": cached[-1] if cached else None}


def plr_report(run_dir: Path, status: str, warmup: list[dict[str, object]],
               steady: list[dict[str, object]]) -> dict[str, object]:
    """Step-4 PLR's release record: each release, then MemFree and Cached over warm-up and the steady interval.

    Descriptive only; the step-4 criteria judge the run. ``steady`` holds the samples inside the monotonic steady
    interval (empty when it is unavailable)."""
    releases = release_records(load_mr1_events(Path(run_dir) / "events.jsonl"))
    face = releases.get("face")
    last = ((face or {}).get("after") or {}).get("meminfo") or {}
    return {
        "label": PLR_LABEL,
        "criteria_id": step4_criteria.PLR_CRITERIA_ID,
        "scope": "each model's file cache released after its load and settle, then step 4's warm-up and steady "
                 "phases; release outcomes are descriptive, never a criterion",
        "profile_status": _status_label(status),
        "releases": releases,
        "release_outcomes": {c: (releases.get(c) or {}).get("outcome", "not_reached") for c in MR1_RELEASE_FILES},
        "after_last_release": {"MemFree": last.get("MemFree"), "Cached": last.get("Cached")} if face else None,
        "warmup": _memory_window(warmup),
        "steady": _memory_window(steady),
        "basis": MR1_BASIS,
        "cannot_establish": list(PLR_CANNOT_ESTABLISH),
    }


def plr_lines(report: dict[str, object]) -> list[str]:
    lines = ["", "Post-load release (step-4 PLR, D54; release outcomes descriptive, never a criterion):",
             f"  basis: {MR1_BASIS}"]
    lines += release_lines(report["releases"])
    after = report.get("after_last_release")
    if after:
        lines.append(f"  after the last release: MemFree {gb(after.get('MemFree'))}, Cached {gb(after.get('Cached'))}")
    for key, name in (("warmup", "warm-up"), ("steady", "steady interval")):
        window = report.get(key)
        lines.append(f"  {name}: not reached" if not window else
                     f"  {name}: MemFree min {gb(window['mem_free_min_bytes'])}; Cached first "
                     f"{gb(window['cached_first_bytes'])}, last {gb(window['cached_last_bytes'])}")
    lines.append("  Cannot establish: " + "; ".join(report["cannot_establish"]) + ".")
    return lines


def prompt_cache_summary(log_path: Path, steady: tuple[float, float] | None) -> dict[str, object] | None:
    """Numbers from llama-server's prompt-cache lines; None when the log has none (older runs)."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    result: dict[str, object] = {"startup": None, "state_updates": 0, "steady_state_updates": None,
                                 "first": None, "last": None, "steady_first": None, "steady_last": None,
                                 "max_size_mib": None, **dict.fromkeys(PROMPT_CACHE_MARKERS.values(), 0)}
    for line in lines:
        stamp = re.match(r"t_mono=([\d.]+) ", line)
        t = float(stamp[1]) if stamp else None
        if startup := PROMPT_CACHE_STARTUP.search(line):
            result["startup"] = {"enabled": startup[1] != "disabled",
                                 "limit_mib": int(startup[2]) if startup[2] else None, "t_mono": t}
        elif state := PROMPT_CACHE_STATE.search(line):
            entry = {"t_mono": t, "prompts": int(state[1]), "size_mib": float(state[2]),
                     "limit_mib": float(state[3]), "limit_tokens": int(state[4]), "estimated_tokens": int(state[5])}
            result["state_updates"] += 1
            result["first"] = result["first"] or entry
            result["last"] = entry
            result["max_size_mib"] = max(result["max_size_mib"] or 0.0, entry["size_mib"])
            if steady and t is not None and steady[0] <= t <= steady[1]:
                result["steady_state_updates"] = (result["steady_state_updates"] or 0) + 1
                result["steady_first"] = result["steady_first"] or entry
                result["steady_last"] = entry
        else:
            for key, label in PROMPT_CACHE_MARKERS.items():
                result[label] += int(key in line or line.endswith("prompt cache event: " + label))
    if result["startup"] is None and not result["state_updates"]:
        return None
    return result


def steady_trend(rows: list[dict[str, object]]) -> dict[str, object]:
    """First/last values and least-squares slopes (bytes per minute) over the steady rows, per column.

    The steady label persists until the orchestrator changes it, so rows taken
    while the workload exits are included: compare arms under equal conditions only.
    """
    def fit(key: str) -> dict[str, object]:
        points = [(r["t"], r[key]) for r in rows if r.get(key) is not None]
        if len(points) < 2:
            return {"first": points[0][1] if points else None, "last": points[-1][1] if points else None,
                    "slope_bytes_per_min": None, "n": len(points)}
        mean_t = sum(t for t, _ in points) / len(points)
        mean_v = sum(v for _, v in points) / len(points)
        spread = sum((t - mean_t) ** 2 for t, _ in points)
        slope = sum((t - mean_t) * (v - mean_v) for t, v in points) / spread * 60 if spread else None
        return {"first": points[0][1], "last": points[-1][1],
                "slope_bytes_per_min": int(slope) if slope is not None else None, "n": len(points)}

    if not rows:
        return {}
    return {"seconds": round(rows[-1]["t"] - rows[0]["t"], 1),
            **{key: fit(key) for key in ("used", "mem_free", "cached", "llama_pss", "work_pss")}}


def summarize(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = load_latest_events(run_dir / "events.jsonl")
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
        return events.get(name)

    def column_max(rows: list[dict[str, object]], key: str) -> int | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return max(values) if values else None

    end = last_event("run_end")
    status = end["status"] if end else "unknown (no run_end event)"
    cache_ram = (manifest.get("llama_server") or {}).get("cache_ram_mib")
    cache_flag = "not passed (server default)" if cache_ram is None else f"{cache_ram} MiB"
    lines = [
        f"{LABEL.upper()} RESOURCE PROFILE: {manifest['run_id']}",
        "One cold load and one combined run. Not a benchmark, Gate B record or beta-gate result.",
        f"Status: {status}",
        f"Device: {manifest.get('l4t')} | kernel {manifest.get('kernel')} | power mode {manifest.get('power_mode')}",
        f"Host state: display-manager {manifest.get('display_manager')}; desktop processes {manifest.get('desktop_processes') or 'none'}; "
        f"dev tools {manifest.get('dev_tools_running') or 'none'}",
        (f"Input: scene-only, one distinct synthetic noise image per request; scene every "
         f"{manifest['parameters']['scene_interval_s']} s; llama-server --cache-ram {cache_flag}"
         if manifest["parameters"].get("scene_only") else
         f"Input: {'synthetic noise frames' if manifest['input']['synthetic'] else 'replay clip'} at {manifest['input']['fps']} fps; "
         f"face {manifest['parameters']['face_hz']} Hz; scene every {manifest['parameters']['scene_interval_s']} s"),
        f"Repository: {manifest['repository']['commit']} (tracked changes: {manifest['repository']['tracked_changes']})",
        "",
        "Whole-device memory = MemTotal - MemAvailable (decimal bytes).",
        "Torch allocator events, when present, are process-only bytes; they do not measure total GPU/device usage or prove unload.",
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
    scene_only = bool(manifest.get("parameters", {}).get("scene_only"))
    for key, load_phase, settle_phase, load_event in COMPONENTS:
        if scene_only and key != "scene":
            lines.append(f"  {key}: not run (scene-only)")
            continue
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

    boundary_events = load_boundary_events(run_dir / "events.jsonl")
    steady_s = float(manifest.get("parameters", {}).get("steady_s") or step4_criteria.STEADY_S)
    interval, interval_rows = step4_criteria.steady_interval(boundary_events, samples, steady_s)
    if interval["status"] != step4_criteria.UNAVAILABLE:
        steady = list(interval_rows)  # monotonic boundaries: teardown excluded from every steady statistic
        steady_basis = "monotonic boundaries"
    else:
        steady = in_phases("steady")  # legacy run: phase labels, includes rows while the workload exits
        steady_basis = "phase labels (legacy; includes rows while the workload exits; acceptance unavailable)"
    warm = in_phases("warmup") + steady
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
        verdict_peak = combined["run_peak_bytes"] is not None and combined["run_peak_bytes"] <= TARGET_PEAK_BYTES
        lines += [
            "",
            "All components together:",
            f"  run peak (includes cold loads) {gb(combined['run_peak_bytes'])}: "
            f"{'within' if verdict_peak else 'ABOVE'} the 5,400,000,000 B ceiling",
            f"  warm peak {gb(combined['warm_peak_bytes'])}; minus baseline {gb(combined.get('warm_peak_minus_baseline_bytes'), True)}",
            f"  steady basis: {steady_basis}",
            *([f"  steady interval {interval['status']}, {interval['duration_s']} s of {interval['expected_s']:g} s, "
               f"coverage {interval['coverage']}, largest gap {interval['max_gap_s']} s, ended by {interval['ended_by']}",
               f"  steady max {gb(interval.get('max_bytes'))}; {interval.get('seconds_above_target')} s above the "
               f"5,000,000,000 B target: {'within' if interval.get('seconds_above_target') == 0 else 'ABOVE'}"]
              if interval["status"] != step4_criteria.UNAVAILABLE else
              [f"  steady interval unavailable ({interval.get('reason')}): no steady verdict"]),
            f"  steady median {gb(combined['steady_median_bytes'])}; steady p95 {gb(combined['steady_p95_bytes'])} (reported only)",
            f"  swap used max {gb(combined['swap_used_max_bytes'])}; steady pages swapped in/out "
            f"{combined.get('steady_pages_swapped_in', 'n/a')}/{combined.get('steady_pages_swapped_out', 'n/a')}",
            f"  llama-server peak RSS {gb(combined['llama_server_peak_rss_bytes'])}, peak PSS {gb(combined['llama_server_peak_pss_bytes'])}",
            f"  workload process peak RSS {gb(combined['workload_peak_rss_bytes'])}, peak PSS {gb(combined['workload_peak_pss_bytes'])}",
            "  (process RSS/PSS and whole-device use have different accounting; never add them)",
        ]

    trend = steady_trend(steady)
    if trend:
        lines += [
            "",
            f"Steady trend over {trend['seconds']} s (least squares; {steady_basis}):",
            *(f"  {key}: first {gb(trend[key]['first'])}, last {gb(trend[key]['last'])}, "
              f"slope {trend[key]['slope_bytes_per_min'] if trend[key]['slope_bytes_per_min'] is not None else 'n/a'} B/min"
              for key in ("used", "llama_pss", "work_pss") if trend[key]["n"]),
        ]
    if interval["status"] != step4_criteria.UNAVAILABLE:
        steady_window = (interval["start_t_mono"], interval["end_t_mono"])
    else:
        steady_window = (steady[0]["t"], steady[-1]["t"]) if steady else None
    cache = prompt_cache_summary(run_dir / "llama-server.log", steady_window)
    if cache:
        startup = cache["startup"] or {}
        lines += [
            "",
            "llama-server prompt cache (its own log lines; numbers only):",
            f"  startup: {'enabled' if startup.get('enabled') else 'disabled' if startup else 'not logged'}"
            + (f", limit {startup['limit_mib']} MiB" if startup.get("limit_mib") is not None else ""),
            f"  state updates {cache['state_updates']} (steady {cache['steady_state_updates']}); "
            f"last {cache['last']['prompts'] if cache['last'] else 'n/a'} prompts, "
            f"{cache['last']['size_mib'] if cache['last'] else 'n/a'} MiB; max {cache['max_size_mib']} MiB",
            f"  duplicates skipped {cache['duplicate_skipped']}; evicted {cache['evicted']}; "
            f"allocation failures {cache['allocation_failed']}",
        ]
    progress = last_event("scene_progress")
    if progress:
        lines += [
            "",
            f"Scene progress at the last request (cumulative since the workload started; phase {progress.get('phase')}):",
            f"  requests {progress.get('requests')}, completed {progress.get('completed')}, "
            f"valid/invalid {progress.get('valid_reports')}/{progress.get('invalid_reports')}, errors {progress.get('errors')}, "
            f"finish reasons {progress.get('finish_reasons')}, synthetic images issued {progress.get('synthetic_images_issued')}",
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

    if interval["status"] != step4_criteria.UNAVAILABLE:
        steady_tegra = [row for row in tegra if "t" in row and steady_window[0] <= row["t"] <= steady_window[1]]
    else:
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
        lines += ["", f"Workload, steady phase ({stats.get('seconds')} s, {stats.get('input')}):"]
        if scene_only:
            lines.append("  detector and face: not run (scene-only)")
        else:
            lines += [
                f"  detector: {detector.get('processed_frames')} of {detector.get('source_frames')} source frames, "
                f"{detector.get('processed_fps')} fps; latency ms {detector.get('latency_ms')}; frames with a person {detector.get('frames_with_person')}",
                f"  face: {face.get('runs')} runs ({face.get('achieved_hz')} Hz achieved), latency ms {face.get('latency_ms')}; "
                f"runs with a face {face.get('runs_with_face')}; errors {face.get('errors')}",
            ]
        lines += [
            f"  scene: {scene.get('completed')} completed, latency ms {scene.get('latency_ms')}; over the 8 s D16 timeout {scene.get('over_d16_timeout')}; "
            f"errors {scene.get('errors')}; valid/invalid/unchecked reports {scene.get('valid_reports')}/{scene.get('invalid_reports')}/{scene.get('unchecked_reports')}; "
            f"mean prompt/completion tokens {scene.get('prompt_tokens_mean')}/{scene.get('completion_tokens_mean')}",
            f"  scene completion finish reasons {scene.get('finish_reasons', 'unavailable')}; "
            f"rejections {scene.get('rejected_reports_by_reason', 'unavailable')}; "
            f"valid summaries/observations at character limits "
            f"{scene.get('valid_summaries_at_limit', 'unavailable')}/{scene.get('valid_observations_at_limit', 'unavailable')}",
            "  Scene structural validity and exact-limit counts do not establish scene accuracy or semantic completeness.",
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
                                                   "display_manager", "repository", "llama_server", "files", "sha256", "input",
                                                   "parameters")},
        "components": profile_components,
        "combined": combined,
        "unload": unload,
        "tegrastats_steady": tegra_summary,
        "workload_steady": stats,
        "steady_trend": trend,
        "prompt_cache": cache,
        "scene_progress_last": progress,
        "targets": {"steady_bytes": TARGET_STEADY_BYTES, "peak_bytes": TARGET_PEAK_BYTES},
        "steady_interval": interval,
        "gpu_evidence": {"llama": last_event("llama_gpu_check"), "workload_cuda": last_event("cuda_driver")},
        "cache_evidence": step4_criteria.cache_evidence(
            manifest, last_event("llama_cmdline"), cache,
            telemetry_timestamped=bool(cache and (cache.get("startup") or {}).get("t_mono") is not None)),
    }
    if (manifest.get("parameters") or {}).get("mr1_release_check"):
        report = mr1_report(run_dir, status)
        (run_dir / "mr1.json").write_text(json.dumps(report, indent=2) + "\n")
        profile["step4_profile_criteria"] = None  # an MR1 run is never step-4 evidence
        profile["mr1"] = "mr1.json"
        lines += mr1_lines(report)
    else:
        profile["step4_profile_criteria"] = step4_criteria.evaluate_profile(profile)
        if (manifest.get("parameters") or {}).get("post_load_release"):
            report = plr_report(run_dir, status, in_phases("warmup"),
                                list(interval_rows) if interval["status"] != step4_criteria.UNAVAILABLE else [])
            (run_dir / "plr.json").write_text(json.dumps(report, indent=2) + "\n")
            profile["criteria_id"] = step4_criteria.PLR_CRITERIA_ID
            profile["post_load_release"] = "plr.json"
            lines += plr_lines(report)
            lines += ["", f"Step-4 criteria decidable from this run ({step4_criteria.PLR_CRITERIA_ID}: D47's rules and "
                      "thresholds under the PLR identity; demo profile, not the 1080p beta gates; never acceptance "
                      "or admission):"]
        else:
            lines += ["", f"Step-4 criteria decidable from this run ({step4_criteria.CRITERIA_ID}; demo profile, "
                      "not the 1080p beta gates; never acceptance):"]
        lines += [f"  {name}: {item['status']}" for name, item in profile["step4_profile_criteria"].items()]
    (run_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(text)
    return text


# ------------------------------------------------------------------------ main


def cache_ram_mib(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("use 0 (disabled) or a positive MiB limit; 'no limit' is not offered")
    return value


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
    parser.add_argument("--scene-only", action="store_true",
                        help="S1: llama-server and scene requests only, one distinct synthetic image per request")
    parser.add_argument("--llama-cache-ram", type=cache_ram_mib, metavar="MIB",
                        help="pass --cache-ram MIB to llama-server (0 disables its prompt cache); default: not passed")
    parser.add_argument("--mr1-release-check", action="store_true",
                        help="MR1 (opt-in): after each model's load and settle, release that model's files from the page "
                             "cache (posix_fadvise DONTNEED, no root) with memory samples around it, then bounded smoke "
                             "checks and unload; no warm-up or steady phase. Not a resource profile or step-4 run")
    parser.add_argument("--post-load-release", action="store_true",
                        help="step-4 PLR (opt-in, D54): MR1's release of each model's files after its load and settle, "
                             "then the full warm-up and steady phases; judged by D47's criteria under its own identity")
    parser.add_argument("--mr1-release-settle-s", "--release-settle-s", dest="mr1_release_settle_s", type=float,
                        default=MR1_RELEASE_SETTLE_S,
                        help="MR1 and PLR: seconds between the release calls and the after sample")
    parser.add_argument("--sanitized-logs", action="store_true",
                        help="discard raw server/workload output; retain fixed numeric/placement diagnostics only")
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
