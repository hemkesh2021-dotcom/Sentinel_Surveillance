"""Operator-only inspection and opt-in bounded diagnostics; no hardware acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import demo_profile as profile
import step4_criteria

SAMPLE_S = 0.2
PRESSURE_STOP = 4_800_000_000
FREE_FLOOR = 1 << 30
AVAILABLE_FLOOR = 2 << 30
OUTPUT_LIMIT = 1_048_576
MEMORY_KEYS = (*profile.MEMINFO_KEYS, "pswpin", "pswpout")
REQUIRED = ("MemTotal", "MemAvailable", "MemFree", "pswpin", "pswpout")
SERVICES = ("ollama.service", "display-manager.service")
PVA_UNIT = "nvidia-pva-allowd.service"
PVA_CGROUP = "/system.slice/" + PVA_UNIT
PVA_FRAGMENT = "/etc/systemd/system/" + PVA_UNIT
PVA_DAEMON = "/opt/nvidia/pva-allow-2/bin/nvidiaPvaAllowd.py"
PVA_CHECKSUMS = "/var/lib/dpkg/info/pva-allow-2.md5sums"
OUTPUT_ROOT = Path("/tmp")
STEP4_DEADLINE_S = 1200.0  # expected about 1,040 s: 30 s baseline, loads, 3 settles, 120 s warm-up, 600 s steady, unload
STEP4_JOURNAL_TIMEOUT_S = 10.0
STEP4_MARKER_TAG = "sentinel-step4"
JOURNAL_LOSS = re.compile(r"missed|suppress|rate.?limit|is full|truncat|corrupt", re.IGNORECASE)
S1_ARMS = {"a": None, "b": 0}  # llama-server --cache-ram MiB; None keeps b8932's default (8192 MiB)
_DROPPED = object()
DROP_CACHES_PROCEDURE = (
    "D37: operator-run `sync && sudo sysctl -w vm.drop_caches=1` before measurement runs only; "
    "declared by the operator, never run or verified by this workflow"
)
SERVICE_VALUES = {
    "active", "inactive", "failed", "activating", "deactivating", "reloading",
    "enabled", "enabled-runtime", "disabled", "static", "indirect", "masked",
    "masked-runtime", "generated", "transient", "alias", "loaded", "not-found",
}


def memory_problem(sample: dict, baseline: dict | None = None, projected: int = 0) -> str | None:
    if any(type(sample.get(key)) is not int or sample[key] < 0 for key in REQUIRED):
        return "telemetry_unavailable"
    total = sample["MemTotal"]
    if total <= 0 or sample["MemAvailable"] > total or sample["MemFree"] > total:
        return "telemetry_invalid"
    if total - sample["MemAvailable"] + projected >= PRESSURE_STOP:
        return "sampled_pressure_stop"
    if sample["MemFree"] - projected < FREE_FLOOR or sample["MemAvailable"] - projected < AVAILABLE_FLOOR:
        return "free_or_available_stop"
    if baseline and any(sample[key] != baseline[key] for key in ("pswpin", "pswpout")):
        return "swap_counter_change"
    return None


def baseline_problem(sample: dict) -> str | None:
    problem = memory_problem(sample)
    if problem:
        return problem
    if (sample["MemTotal"] - sample["MemAvailable"] >= 2_000_000_000
            or sample["MemFree"] < 3_500_000_000 or sample["MemAvailable"] < 4_000_000_000):
        return "initial_headroom_refused"
    return None


def safe_sample(sample: dict) -> dict:
    values = {key: value if type(value) is int and value >= 0 else None
              for key in MEMORY_KEYS for value in (sample.get(key),)}
    stamp = sample.get("t_mono")
    values["t_mono"] = stamp if type(stamp) in (int, float) and math.isfinite(stamp) else None
    total, available = values["MemTotal"], values["MemAvailable"]
    values["pressure_bytes"] = total - available if total is not None and available is not None else None
    return values


class PressureGuard:
    def __init__(self, baseline: dict, emit=lambda reading: None) -> None:
        self.baseline = baseline
        self.emit = emit
        self.previous = None
        self.peak = None
        self.minimum_free = None
        self.samples = 0

    def observe(self, sample: dict) -> str | None:
        reading = safe_sample(sample)
        self.emit(reading)
        self.samples += 1
        problem = memory_problem(sample, self.baseline)
        stamp = reading["t_mono"]
        if stamp is None:
            return "clock_unavailable"
        if self.previous is not None and (stamp < self.previous or stamp - self.previous > 0.5):
            return "sampling_gap"
        self.previous = stamp
        pressure = reading["pressure_bytes"]
        if pressure is not None:
            self.peak = max(self.peak if self.peak is not None else pressure, pressure)
        free = reading["MemFree"]
        if free is not None:
            self.minimum_free = min(self.minimum_free if self.minimum_free is not None else free, free)
        return problem


class NativeChild:
    def __init__(self, argv: list[str], env: dict | None) -> None:
        self.process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True,
            cwd=profile.REPO, env=env,
        )
        self.nonblocking = False
        self.gone = False

    def poll(self):
        return self.process.poll()

    def read(self) -> bytes:
        if not self.nonblocking:
            os.set_blocking(self.process.stdout.fileno(), False)
            self.nonblocking = True
        try:
            return os.read(self.process.stdout.fileno(), 32_768)
        except BlockingIOError:
            return b""

    def exists(self) -> bool:
        if self.gone:
            return False
        self.poll()
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            self.gone = True
            return False

    def signal(self, signum: int) -> None:
        if self.exists():
            try:
                os.killpg(self.process.pid, signum)
            except ProcessLookupError:
                self.gone = True

    def close(self) -> None:
        self.process.stdout.close()


class SystemBackend:
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    spawn = staticmethod(NativeChild)

    @staticmethod
    def boot_id() -> str:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def sample(self) -> dict:
        stamp = self.clock()
        return {"t_mono": stamp, **profile.read_meminfo(), **profile.read_swap_counters()}

    def context(self) -> dict:
        categories = {key: {"count": 0, "pids": []} for key in
                      ("model_servers", "desktop", "dev_tools", "python_unclassified", "media_or_gpu_tools")}
        unavailable = False
        known_system_services = {}
        pva_pid = None
        pva_checked = False
        try:
            entries = Path("/proc").iterdir()
            for path in entries:
                if not path.name.isdigit() or int(path.name) == os.getpid():
                    continue
                try:
                    comm = (path / "comm").read_text().strip()
                except FileNotFoundError:
                    continue
                except OSError:
                    unavailable = True
                    continue
                category = None
                if comm in ("llama-server", "ollama", "ollama_llama_se"):
                    category = "model_servers"
                elif comm in profile.DESKTOP_COMMS:
                    category = "desktop"
                elif comm in ("claude", "codex", "node", "code", "code-insiders"):
                    category = "dev_tools"
                elif comm.startswith("python"):
                    if pva_cgroup_member(path):
                        if not pva_checked:
                            pva_pid = verified_pva_main_pid(ProcessRunner(self))
                            pva_checked = True
                        if int(path.name) == pva_pid:
                            known_system_services[PVA_UNIT] = {
                                "count": 1, "pids": [pva_pid],
                                "identity_basis": "systemd MainPID, root cgroup, package-owned checksummed launcher/unit",
                            }
                            continue
                    category = "python_unclassified"
                elif comm in ("ffmpeg", "gst-launch-1.0", "trtexec"):
                    category = "media_or_gpu_tools"
                if category:
                    item = categories[category]
                    item["count"] += 1
                    if len(item["pids"]) < 16:
                        item["pids"].append(int(path.name))
        except OSError:
            unavailable = True
        args = profile.parse_args([])
        assets = [args.python, args.llama, Path(profile.L4T_LIBCUDA), *profile.model_files(args).values()]
        return {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "root": os.geteuid() == 0, "port_18081_in_use": profile.port_in_use(18081),
            "cached_assets_available": all(path.is_file() for path in assets),
            "process_inspection_unavailable": unavailable, "workloads": categories,
            "known_system_services": known_system_services,
        }


@dataclass
class ChildResult:
    status: str
    returncode: int | None
    output: bytes
    cleanup_clear: bool

    def diagnostic(self) -> dict:
        return {"status": self.status, "returncode": self.returncode, "cleanup_clear": self.cleanup_clear}


class ProcessRunner:
    def __init__(self, backend, interrupted=lambda: False) -> None:
        self.backend = backend
        self.interrupted = interrupted

    def cleanup(self, child, grace: float) -> bool:
        for signum in (signal.SIGTERM, signal.SIGKILL):
            child.signal(signum)
            deadline = self.backend.clock() + grace
            while child.exists() and self.backend.clock() < deadline:
                self.backend.sleep(min(0.1, grace))
        clear = not child.exists()
        child.close()
        return clear

    def run(self, argv: list[str], timeout: float, *, guard=None, env=None) -> ChildResult:
        if self.interrupted():
            return ChildResult("interrupted", None, b"", True)
        deadline = self.backend.clock() + timeout
        output = bytearray()
        child = None
        status = "process_error"
        returncode = None
        clear = True
        try:
            child = self.backend.spawn(argv, env)
            while True:
                if self.interrupted():
                    status = "interrupted"
                    break
                if self.backend.clock() >= deadline:
                    status = "timeout"
                    break
                if guard:
                    problem = guard.observe(self.backend.sample())
                    if problem:
                        status = problem
                        break
                block = child.read()
                if len(output) + len(block) > OUTPUT_LIMIT:
                    status = "output_limit"
                    break
                output.extend(block)
                returncode = child.poll()
                if returncode is not None:
                    while block:
                        block = child.read()
                        if len(output) + len(block) > OUTPUT_LIMIT:
                            break
                        output.extend(block)
                    status = "output_limit" if len(output) + len(block) > OUTPUT_LIMIT else (
                        "completed" if returncode == 0 else "child_failed"
                    )
                    break
                self.backend.sleep(SAMPLE_S if guard else 0.02)
        except Exception:
            status = "process_error"
        finally:
            if child is not None:
                try:
                    clear = self.cleanup(child, 5.0 if guard else 0.2)
                    returncode = child.poll()
                except Exception:
                    clear = False
        return ChildResult(status if clear else "cleanup_failed", returncode, bytes(output), clear)


def pva_cgroup_member(path: Path) -> bool:
    try:
        return path.stat().st_uid == 0 and (path / "cgroup").read_text().strip() == "0::" + PVA_CGROUP
    except OSError:
        return False


def verified_pva_main_pid(runner: ProcessRunner) -> int | None:
    fields = ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup",
              "FragmentPath", "DropInPaths", "User", "ExecStart")
    result = runner.run([
        "/usr/bin/systemctl", "show", PVA_UNIT, "--no-pager",
        *("--property=" + field for field in fields),
    ], 3.0)
    if result.status != "completed":
        return None
    values = dict(line.partition("=")[::2] for line in result.output.decode(errors="replace").splitlines() if "=" in line)
    expected = {"Id": PVA_UNIT, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                "ControlGroup": PVA_CGROUP, "FragmentPath": PVA_FRAGMENT, "DropInPaths": ""}
    if (any(values.get(key) != value for key, value in expected.items())
            or values.get("User") not in ("", "root")
            or re.findall(r"\{ path=([^ ;]+) ;", values.get("ExecStart", "")) != [PVA_DAEMON]):
        return None
    pid = values.get("MainPID", "")
    if not re.fullmatch(r"[1-9][0-9]{0,9}", pid):
        return None
    installed = runner.run([
        "/usr/bin/dpkg-query", "--show", "--showformat=${Status}\n", "pva-allow-2",
    ], 3.0)
    owned = runner.run(["/usr/bin/dpkg-query", "--search", PVA_FRAGMENT, PVA_DAEMON], 3.0)
    if (installed.status != "completed" or installed.output.strip() != b"install ok installed"
            or owned.status != "completed" or set(owned.output.decode(errors="replace").splitlines()) != {
                "pva-allow-2: " + PVA_FRAGMENT, "pva-allow-2: " + PVA_DAEMON,
            }):
        return None
    try:
        contents = {}
        for filename in (PVA_FRAGMENT, PVA_DAEMON, PVA_CHECKSUMS):
            path = Path(filename)
            info = path.stat()
            if info.st_uid != 0 or info.st_mode & 0o022 or not stat.S_ISREG(info.st_mode) or info.st_size > OUTPUT_LIMIT:
                return None
            with path.open("rb") as handle:
                contents[filename] = handle.read(OUTPUT_LIMIT + 1)
            if len(contents[filename]) > OUTPUT_LIMIT:
                return None
        checksums = dict(line.split(None, 1)[::-1] for line in contents[PVA_CHECKSUMS].decode().splitlines())
        if any(checksums.get(filename.lstrip("/")) != hashlib.md5(contents[filename]).hexdigest()
               for filename in (PVA_FRAGMENT, PVA_DAEMON)):
            return None
    except (OSError, ValueError):
        return None
    return int(pid)


def service_states(result: ChildResult) -> dict:
    states = {unit: {"ActiveState": "unknown", "UnitFileState": "unknown", "LoadState": "unknown"}
              for unit in SERVICES}
    if result.status != "completed":
        return states
    for block in result.output.decode(errors="replace").split("\n\n"):
        fields = dict(line.partition("=")[::2] for line in block.splitlines() if "=" in line)
        current = fields.get("Id")
        if current not in states:
            current = next((name for name in fields.get("Names", "").split() if name in states), None)
        if current in states:
            for key in states[current]:
                value = fields.get(key)
                states[current][key] = value if value in SERVICE_VALUES else "unknown"
    return states


def kernel_counts(result: ChildResult) -> dict:
    if result.status != "completed":
        return {"status": "unavailable", "oom_candidates": None, "nvmap_candidates": None}
    oom = nvmap = records = 0
    for line in result.output.decode(errors="replace").splitlines():
        try:
            message = json.loads(line).get("MESSAGE")
            if not isinstance(message, str):
                raise ValueError
        except (ValueError, AttributeError):
            return {"status": "unavailable", "oom_candidates": None, "nvmap_candidates": None}
        records += 1
        lowered = message.lower()
        oom += int(any(marker in lowered for marker in ("out of memory", "oom-kill", "killed process")))
        nvmap += int("nvmapmemalloc" in lowered)
    return {"status": "observed" if records else "unavailable", "records": records,
            "oom_candidates": oom if records else None, "nvmap_candidates": nvmap if records else None,
            "window": "current boot, latest 1000 kernel records; candidate lines, not unique events"}


def inspection(backend, runner: ProcessRunner) -> dict:
    sample = safe_sample(backend.sample())
    context = backend.context()
    services = service_states(runner.run([
        "/usr/bin/systemctl", "show", *SERVICES, "--no-pager",
        "--property=Id", "--property=Names", "--property=ActiveState", "--property=UnitFileState", "--property=LoadState",
    ], 3.0))
    kernel = kernel_counts(runner.run([
        "/usr/bin/journalctl", "-k", "-b", "-n", "1000", "-o", "json", "--quiet", "--no-pager",
    ], 3.0))
    revision = runner.run(["git", "-C", str(profile.REPO), "rev-parse", "HEAD"], 3.0)
    commit = revision.output.decode(errors="replace").strip()
    commit = commit if revision.status == "completed" and re.fullmatch(r"[0-9a-f]{40}", commit) else None
    refusals = []
    problem = baseline_problem(sample)
    if problem:
        refusals.append(problem)
    for key in ("root", "port_18081_in_use", "process_inspection_unavailable"):
        if context[key]:
            refusals.append(key)
    if not context["cached_assets_available"]:
        refusals.append("cached_assets_missing")
    for category, item in context["workloads"].items():
        if item["count"]:
            refusals.append(category)
    for unit, fields in services.items():
        if fields["ActiveState"] != "inactive" and fields["LoadState"] != "not-found":
            refusals.append("service_not_known_idle:" + unit)
    if not commit:
        refusals.append("repository_revision_unavailable")
    if kernel["status"] != "observed":
        refusals.append("kernel_inspection_unavailable")
    if any(not result.cleanup_clear for result in (revision,)):
        refusals.append("inspection_cleanup_failed")
    return {"memory": sample, **context, "services": services, "repository_commit": commit,
            "kernel": kernel, "workload_refusals": refusals,
            "limitations": "comm-based classification with verified PVA exception; no process argv/environment displayed; not proof of no GPU users"}


def private_result(path: Path) -> dict | None:
    """A result.json this user wrote in a private /tmp/sentinel-operator-* directory, or None."""
    resolved = path.resolve(strict=True)
    info = path.stat()
    if (path.is_symlink() or resolved.name != "result.json"
            or not resolved.parent.name.startswith("sentinel-operator-")
            or not resolved.is_relative_to(OUTPUT_ROOT.resolve())
            or info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode)
            or info.st_size > OUTPUT_LIMIT):
        return None
    with path.open("rb") as handle:
        data = handle.read(OUTPUT_LIMIT + 1)
    if len(data) > OUTPUT_LIMIT:
        return None
    prior = json.loads(data)
    return prior if isinstance(prior, dict) else None


def check9_prerequisite(path: Path | None, current: dict) -> str | None:
    if path is None:
        return "check9_report_required"
    try:
        prior = private_result(path)
        if prior is None:
            return "check9_report_refused"
        previous = prior["inspection"]
        if (prior["schema_version"] != 1 or prior["mode"] != "check9"
                or prior["check9"]["status"] != "bounded_smoke_complete"
                or previous["boot_id"] != current["boot_id"]
                or previous["repository_commit"] != current["repository_commit"]):
            return "check9_report_mismatch"
        for api in ("device", "managed"):
            phase = prior["check9"][api]
            if (phase["status"] != "completed" or phase["returncode"] != 0
                    or phase["cleanup_clear"] is not True or phase["allocated_bytes"] != 256 * (1 << 20)
                    or phase["cap_bytes"] != 256 * (1 << 20) or phase["chunk_bytes"] != 32 * (1 << 20)):
                return "check9_report_mismatch"
    except (OSError, ValueError, KeyError, TypeError):
        return "check9_report_unavailable"
    return None


def latest_check9_report(root: Path | None = None) -> Path | None:
    """The newest private Check 9 result; it must still pass check9_prerequisite, so a failed newest run refuses."""
    candidates = []
    for path in (root or OUTPUT_ROOT).glob("sentinel-operator-*/result.json"):
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > OUTPUT_LIMIT:
                continue
            with path.open("rb") as handle:
                if json.loads(handle.read(OUTPUT_LIMIT + 1)).get("mode") == "check9":
                    candidates.append((info.st_mtime_ns, str(path)))
        except (OSError, ValueError, AttributeError):
            continue
    return Path(max(candidates)[1]) if candidates else None


def numeric_excerpt(value, depth: int = 0):
    """Keep only numbers, booleans, None and fixed phase labels under short snake_case keys; drop the rest."""
    if isinstance(value, dict) and depth < 4:
        kept = {key: numeric_excerpt(item, depth + 1) for key, item in list(value.items())[:32]
                if isinstance(key, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", key)}
        return {key: item for key, item in kept.items() if item is not _DROPPED}
    if type(value) in (int, float, bool) or value is None:
        return value if type(value) is not float or math.isfinite(value) else None
    if isinstance(value, str) and value in profile.PHASES_IN_ORDER:
        return value
    return _DROPPED


def s1_profile_excerpt(output: Path) -> dict:
    """Numeric S1 comparison fields from the single profiler run under this result's directory."""
    paths = list(output.glob("demo-profile-*/profile.json"))
    if len(paths) != 1:
        return {"status": "unavailable"}
    try:
        with paths[0].open("rb") as handle:
            data = handle.read(OUTPUT_LIMIT + 1)
        if len(data) > OUTPUT_LIMIT:
            return {"status": "unavailable"}
        prof = json.loads(data)
        status = prof.get("status")
        fields = {key: prof.get(key) for key in ("steady_trend", "prompt_cache", "scene_progress_last", "unload")}
        fields["scene_component"] = (prof.get("components") or {}).get("scene")
        fields["workload_steady_scene"] = (prof.get("workload_steady") or {}).get("scene")
        fields["llama_cache_ram_mib"] = ((prof.get("provenance") or {}).get("llama_server") or {}).get("cache_ram_mib")
    except (OSError, ValueError, AttributeError):
        return {"status": "unavailable"}
    label = next((name for name in ("complete", "aborted", "interrupted", "incomplete")
                  if isinstance(status, str) and status.startswith(name)), "unknown")
    return {"status": label, **numeric_excerpt(fields)}


# ---------------------------------------------------------------- step 4 (D47)


def file_facts(path: Path) -> dict | None:
    """Name, size and modification time (as demo_profile.py records them); metadata only."""
    try:
        info = Path(path).stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    mtime = profile.datetime.fromtimestamp(info.st_mtime, profile.timezone.utc).isoformat(timespec="seconds")
    return {"name": Path(path).name, "bytes": info.st_size, "mtime_utc": mtime}


def identity_files(clip: Path) -> dict[str, Path]:
    """Everything step 4 measures with: the replay clip, the llama.cpp build, the models and the engine."""
    args = profile.parse_args([])
    files = {"clip": Path(clip)}
    files.update({f"build:{label}": path for label, path in profile.llama_build_files(args.llama).items()})
    files.update({f"model:{label}": path for label, path in profile.model_files(args).items()})
    return files


def sha256_file(path: Path, interrupted=lambda: False) -> str | None:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                if interrupted():
                    return None
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def step4_identity(backend, runner, clip: Path, clip_sha256: str, *, files=None, interrupted=lambda: False) -> dict:
    """Hash every file step 4 depends on. Run it BEFORE the operator's cache drop: hashing fills the page cache."""
    started_utc, started = profile.utc_now(), backend.clock()
    entries: dict[str, dict | None] = {}
    problems = []
    for label, path in (files or identity_files(clip)).items():
        if interrupted():
            problems.append("interrupted")
            break
        facts = file_facts(path)
        if facts is None:
            entries[label] = None
            problems.append(f"missing:{label}")
            continue
        before = backend.clock()
        digest = sha256_file(path, interrupted)
        if digest is None:
            problems.append(f"unreadable:{label}")
        entries[label] = {**facts, "sha256": digest, "hash_seconds": round(backend.clock() - before, 3)}
    clip_entry = entries.get("clip")
    clip_ok = bool(clip_entry and clip_entry["sha256"] == clip_sha256)
    revision = runner.run(["git", "-C", str(profile.REPO), "rev-parse", "HEAD"], 3.0)
    commit = revision.output.decode(errors="replace").strip()
    status = "complete" if not problems and clip_ok else ("clip_mismatch" if not problems else "incomplete")
    return {
        "status": status, "problems": problems, "clip_matches_recorded": clip_ok,
        "started_utc": started_utc, "finished_utc": profile.utc_now(),
        "hash_seconds_total": round(backend.clock() - started, 3),
        "boot_id": backend.boot_id(),
        "commit": commit if revision.status == "completed" and re.fullmatch(r"[0-9a-f]{40}", commit) else None,
        "files": entries,
        "note": "hashing reads every file into the page cache; take this snapshot before the D37 cache drop",
    }


def _utc(text: object):
    try:
        return profile.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def identity_prerequisite(path: Path | None, current: dict, check9_path: Path | None, files: dict) -> tuple[str | None, dict | None]:
    """The identity snapshot must be complete, this boot and commit, taken before the Check 9, and unchanged since."""
    if path is None:
        return "step4_identity_report_required", None
    try:
        prior = private_result(path)
        if prior is None:
            return "step4_identity_report_refused", None
        identity = prior["step4_identity"]
        if prior.get("mode") != "step4_identity" or identity["status"] != "complete" or not identity["clip_matches_recorded"]:
            return "step4_identity_incomplete", identity
        if identity["boot_id"] != current["boot_id"] or identity["commit"] != current["repository_commit"]:
            return "step4_identity_other_boot_or_commit", identity
        check9 = private_result(check9_path) if check9_path is not None else None
        taken, check9_done = _utc(identity["finished_utc"]), _utc((check9 or {}).get("finished_utc"))
        if taken is None or check9_done is None or not taken < check9_done:
            return "step4_identity_not_before_check9", identity
        for label, path_now in files.items():
            recorded, now = identity["files"].get(label), file_facts(path_now)
            if recorded is None or now is None or {k: recorded[k] for k in ("name", "bytes", "mtime_utc")} != now:
                return f"step4_identity_changed:{label}", identity
        if set(identity["files"]) != set(files):
            return "step4_identity_file_set_changed", identity
    except (OSError, ValueError, KeyError, TypeError):
        return "step4_identity_report_unavailable", None
    return None, identity


def identity_end_check(identity: dict | None, manifest: dict | None) -> dict:
    """Do the profiler's end-of-run hashes equal the snapshot? (``verified`` only if every file matches.)"""
    if identity is None or manifest is None:
        return {"status": "unavailable", "reason": "identity snapshot or profiler manifest unavailable"}
    ends: dict[str, object] = {"clip": manifest.get("sha256_clip")}
    ends.update({f"build:{k}": v for k, v in (manifest.get("sha256_build") or {}).items()})
    ends.update({f"model:{k}": v for k, v in (manifest.get("sha256") or {}).items()})
    missing = sorted(label for label in identity["files"] if not ends.get(label))
    changed = sorted(label for label, entry in identity["files"].items()
                     if ends.get(label) and (entry or {}).get("sha256") != ends[label])
    status = "verified" if not missing and not changed else ("changed" if changed else "unavailable")
    return {"status": status, "missing_end_hashes": missing, "changed": changed,
            "clip_matches_recorded": identity.get("clip_matches_recorded"),
            "snapshot_finished_utc": identity.get("finished_utc"), "hash_seconds_total": identity.get("hash_seconds_total"),
            "files": len(identity["files"])}


def _journal(result: ChildResult) -> tuple[str, list[dict]]:
    if result.status == "output_limit":
        return "truncated", []
    if result.status != "completed":
        return "unavailable", []
    records = []
    for line in result.output.decode(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            return "unavailable", []
        if not isinstance(record, dict):
            return "unavailable", []
        message = record.get("MESSAGE")
        stamp = record.get("__REALTIME_TIMESTAMP")
        records.append({"message": message if isinstance(message, str) else "",
                        "t": int(stamp) / 1e6 if isinstance(stamp, str) and stamp.isdigit() else None})
    return "observed", records


def kernel_evidence(runner, backend, manifest: dict | None, tag: str, wait_complete: bool) -> dict:
    """Kernel candidate lines for the recorded boot over the run plus the post-run wait, with coverage proof.

    Coverage is ``observed`` only when every bounded query completed untruncated and parsed, the
    run's own journal markers (written before the run and after the wait) are readable and bracket
    the interval, and journald logged no loss in it. Otherwise it is ``truncated``, ``uncertain``
    or ``unavailable``, and the candidate counts are not zero but absent (lower bounds at most).
    """
    def result(status, reasons, **extra):
        return {"status": status, "reasons": reasons, "oom_candidates": None, "nvmap_candidates": None, **extra}

    if manifest is None:
        return result("unavailable", ["profiler manifest unavailable"])
    boot = manifest.get("boot_id")
    if not isinstance(boot, str) or not boot or boot != backend.boot_id():
        return result("unavailable", ["the recorded boot is not the current boot; the journal is volatile"])
    start, end = _utc(manifest.get("started_utc")), _utc(manifest.get("finished_utc"))
    if start is None or end is None:
        return result("unavailable", ["run interval not recorded"])
    if not wait_complete:
        return result("unavailable", ["post-run wait incomplete"])
    since, until = int(start.timestamp()) - 1, int(end.timestamp() + step4_criteria.POST_RUN_WAIT_S) + 1
    window = {"boot_id": boot, "since_utc": start.isoformat(), "until_epoch_s": until, "since_epoch_s": since}
    common = ["-o", "json", "--quiet", "--no-pager"]
    kernel_q = runner.run(["/usr/bin/journalctl", "-k", "-b", boot, "--since", f"@{since}", "--until", f"@{until}",
                           *common], STEP4_JOURNAL_TIMEOUT_S)
    marker_q = runner.run(["/usr/bin/journalctl", "-b", boot, "-t", STEP4_MARKER_TAG, "--since", f"@{since - 600}",
                           *common], STEP4_JOURNAL_TIMEOUT_S)
    loss_q = runner.run(["/usr/bin/journalctl", "-b", boot, "_COMM=systemd-journal", "--since", f"@{since}",
                         "--until", f"@{until}", *common], STEP4_JOURNAL_TIMEOUT_S)
    states = {name: _journal(q) for name, q in (("kernel", kernel_q), ("markers", marker_q), ("journald", loss_q))}
    if any(state == "unavailable" for state, _ in states.values()):
        return result("unavailable", [f"{name} query unavailable" for name, (state, _) in states.items()
                                      if state == "unavailable"], **window)
    records = states["kernel"][1]
    oom = sum(any(m in r["message"].lower() for m in ("out of memory", "oom-kill", "killed process")) for r in records)
    nvmap = sum("nvmapmemalloc" in r["message"].lower() for r in records)
    reasons = [f"{name} query output truncated" for name, (state, _) in states.items() if state == "truncated"]
    marks = {r["message"]: r["t"] for r in states["markers"][1]}
    started_at, ended_at = marks.get(f"step4 start {tag}"), marks.get(f"step4 end {tag}")
    if started_at is None or ended_at is None:
        reasons.append("run markers not readable: journal access or coverage not proven")
    elif not (started_at <= since + 1 and ended_at >= until - 1):
        reasons.append("run markers do not bracket the interval")
    loss = sum(bool(JOURNAL_LOSS.search(r["message"])) for r in states["journald"][1])
    if loss:
        reasons.append(f"journald reported possible loss ({loss} lines)")
    status = "observed" if not reasons else ("truncated" if any("truncated" in r for r in reasons) else "uncertain")
    counts = {"oom_candidates": oom, "nvmap_candidates": nvmap} if status == "observed" else {
        "oom_candidates_lower_bound": oom, "nvmap_candidates_lower_bound": nvmap}
    return {**result(status, reasons, **window), **counts, "records": len(records), "journald_loss_lines": loss,
            "note": "candidate lines, not unique events"}


def _profiler_files(output: Path) -> tuple[dict | None, dict | None]:
    runs = list(output.glob("demo-profile-*"))
    if len(runs) != 1:
        return None, None
    loaded = []
    for name in ("manifest.json", "profile.json"):
        try:
            with (runs[0] / name).open("rb") as handle:
                data = handle.read(OUTPUT_LIMIT * 4 + 1)
            loaded.append(json.loads(data) if len(data) <= OUTPUT_LIMIT * 4 else None)
        except (OSError, ValueError):
            loaded.append(None)
    return loaded[0], loaded[1]


def execute(mode: str | None, backend, output: Path, *, check9_report=None, confirm_u21=False,
            s1_arm=None, confirm_s1=False, dropped_caches=False, interrupted=lambda: False,
            identity_report=None, clip=None, confirm_step4=False, explicit_check9=True, identity_files_fn=None) -> dict:
    runner = ProcessRunner(backend, interrupted)
    report = {"schema_version": 1, "mode": mode or "inspection",
              "hardware_acceptance": "PENDING", "check9": {"status": "PENDING", "u18_acceptance": "PENDING",
              "server_comparison": "PENDING"}, "u21": {"status": "PENDING"}, "s1": {"status": "PENDING"},
              "step4": {"status": "PENDING", "acceptance": "PENDING: only a maintainer-approved registry commit (D46)"},
              "preparation": {"drop_caches": "operator_declared" if dropped_caches else "not_declared",
                              "procedure": DROP_CACHES_PROCEDURE}}
    try:
        current = inspection(backend, runner)
        report["inspection"] = current
        report["check9"]["kernel_inspection"] = current["kernel"]
        if mode is None:
            return report
        refusal = current["workload_refusals"]
        if mode in ("u21", "s1", "step4"):
            problem = check9_prerequisite(check9_report, current)
            if problem:
                refusal.append(problem)
            report[mode]["check9_report_dir"] = check9_report.parent.name if check9_report else None
        if mode == "u21" and not confirm_u21:
            refusal.append("u21_operator_prerequisites_unconfirmed")
        identity = None
        if mode == "step4":
            if not explicit_check9:
                refusal.append("step4_requires_explicit_check9_report")
            if not confirm_step4:
                refusal.append("step4_operator_prerequisites_unconfirmed")
            if not dropped_caches:
                refusal.append("step4_cache_drop_not_declared")
            if clip is None:
                refusal.append("step4_clip_required")
            else:
                files = (identity_files_fn or identity_files)(clip)
                problem, identity = identity_prerequisite(identity_report, current, check9_report, files)
                if problem:
                    refusal.append(problem)
            report["step4"]["identity_report_dir"] = identity_report.parent.name if identity_report else None
        if mode == "s1":
            if s1_arm not in S1_ARMS:
                refusal.append("s1_arm_required")
            if not confirm_s1:
                refusal.append("s1_operator_prerequisites_unconfirmed")
        if refusal:
            report[mode]["status"] = "refused"
            report[mode]["refusals"] = refusal
            return report
        with (output / "guard.jsonl").open("w") as handle:
            def emit(reading):
                handle.write(json.dumps(reading) + "\n")
                handle.flush()

            baseline = backend.sample()
            problem = baseline_problem(baseline)
            if problem:
                report[mode].update(status="refused", refusals=[problem])
                return report
            guard = PressureGuard(baseline, emit)
            report[mode]["baseline"] = safe_sample(baseline)
            report[mode]["status"] = "running"
            if mode == "check9":
                for api in ("device", "managed"):
                    before = backend.sample()
                    problem = baseline_problem(before) or memory_problem(before, baseline)
                    if problem or interrupted():
                        report["check9"].update(status="refused", refusals=[problem or "interrupted"])
                        break
                    env = dict(os.environ, LD_PRELOAD=profile.L4T_LIBCUDA)
                    child = runner.run([
                        sys.executable, str(profile.HERE / "gpu_alloc_probe.py"),
                        "--api", api, "--chunk-mib", "32", "--cap-mib", "256",
                    ], 20.0, guard=guard, env=env)
                    result = child.diagnostic()
                    report["check9"][api] = result
                    try:
                        payload = json.loads(child.output)
                        valid = (payload["status"] == "bounded_smoke_complete"
                                 and payload["api"] == api and payload["allocated_bytes"] == 256 * (1 << 20)
                                 and payload["cap_bytes"] == 256 * (1 << 20)
                                 and payload["chunk_bytes"] == 32 * (1 << 20)
                                 and payload["cleanup_clear"] is True)
                        for key in ("allocated_bytes", "cap_bytes", "chunk_bytes"):
                            value = payload.get(key)
                            result[key] = value if type(value) is int and value >= 0 else None
                    except (ValueError, KeyError, TypeError):
                        valid = False
                    if child.status != "completed" or not valid:
                        report["check9"]["status"] = "inconclusive"
                        break
                else:
                    report["check9"]["status"] = "bounded_smoke_complete"
            elif mode == "u21":
                child = runner.run([
                    "/usr/bin/python3", str(profile.HERE / "demo_profile.py"), "--out", str(output),
                    "--no-evict", "--sanitized-logs", "--baseline-s", "15", "--settle-s", "15",
                    "--warmup-s", "30", "--steady-s", "120", "--face-hz", "1", "--scene-interval-s", "4",
                    "--llama-timeout-s", "60", "--load-timeout-s", "90", "--min-free-gb", "3.5",
                ], 360.0, guard=guard)
                report["u21"].update(child.diagnostic())
                report["u21"]["operator_prerequisites_confirmed"] = True
            elif mode == "step4":
                run_step4(report["step4"], runner, backend, guard, output, clip, identity, dropped_caches, interrupted)
            else:
                cache_ram = S1_ARMS[s1_arm]
                child = runner.run([
                    "/usr/bin/python3", str(profile.HERE / "demo_profile.py"), "--out", str(output),
                    "--no-evict", "--sanitized-logs", "--scene-only", "--baseline-s", "15", "--settle-s", "15",
                    "--warmup-s", "30", "--steady-s", "180", "--scene-interval-s", "4",
                    "--llama-timeout-s", "60", "--load-timeout-s", "90", "--min-free-gb", "3.5",
                    *([] if cache_ram is None else ["--llama-cache-ram", str(cache_ram)]),
                ], 360.0, guard=guard)
                report["s1"].update(child.diagnostic(), arm=s1_arm, llama_cache_ram_mib=cache_ram,
                                    operator_prerequisites_confirmed=True, profile=s1_profile_excerpt(output))
            report[mode].update(samples=guard.samples, sampled_peak_pressure_bytes=guard.peak,
                                sampled_min_free_bytes=guard.minimum_free)
        if not interrupted():
            backend.sleep(5.0)
            report[mode]["post_exit"] = safe_sample(backend.sample())
    except Exception:
        report[mode or "check9"]["status"] = "diagnostic_unavailable"
    return report


def run_step4(section: dict, runner, backend, guard, output: Path, clip: Path, identity: dict,
              dropped_caches: bool, interrupted) -> None:
    """The guarded combined profile: same guard and cleanup as U21/S1, full phases, then the post-run evidence."""
    tag = output.name
    start_marker = runner.run(["/usr/bin/logger", "-t", STEP4_MARKER_TAG, "--", f"step4 start {tag}"], 3.0)
    child = runner.run([
        "/usr/bin/python3", str(profile.HERE / "demo_profile.py"), "--out", str(output), "--clip", str(clip),
        "--no-evict", "--sanitized-logs", "--llama-cache-ram", "0", "--face-hz", "1", "--scene-interval-s", "4",
        "--baseline-s", "30", "--settle-s", "15", "--warmup-s", "120", "--steady-s", str(int(step4_criteria.STEADY_S)),
        "--llama-timeout-s", "60", "--load-timeout-s", "90", "--min-free-gb", "3.5",
    ], STEP4_DEADLINE_S, guard=guard)
    section.update(child.diagnostic(), deadline_s=STEP4_DEADLINE_S, guard_stopped_at_s=None)
    if child.status not in ("completed", "child_failed"):
        section["guard_stopped_at_s"] = round(backend.clock(), 3)  # the earliest authoritative stop boundary
    waited = 0.0
    while waited < step4_criteria.POST_RUN_WAIT_S and not interrupted():  # after the owned group's cleanup
        backend.sleep(1.0)
        waited += 1.0
    wait_complete = waited >= step4_criteria.POST_RUN_WAIT_S
    end_marker = runner.run(["/usr/bin/logger", "-t", STEP4_MARKER_TAG, "--", f"step4 end {tag}"], 3.0)
    manifest, prof = _profiler_files(output)
    kernel = kernel_evidence(runner, backend, manifest, tag, wait_complete)  # unwritten markers -> not observed
    kernel["markers_written"] = {"start": start_marker.status == "completed", "end": end_marker.status == "completed"}
    identity_check = identity_end_check(identity, manifest)
    if prof is not None and guard.peak is not None:
        combined = dict(prof.get("combined") or {})
        combined["run_peak_bytes"] = max(combined.get("run_peak_bytes") or 0, guard.peak)  # either sampler's peak
        prof = {**prof, "combined": combined}
    repo = (manifest or {}).get("repository") or {}
    run = {
        "guard_completed": child.status == "completed", "cleanup_clear": child.cleanup_clear, "check9_ok": True,
        "drop_declared": dropped_caches, "profile_complete": (prof or {}).get("status") == "complete",
        "headless": (manifest or {}).get("display_manager") == "inactive" and not (manifest or {}).get("desktop_processes"),
        "no_dev_tools": manifest is not None and not manifest.get("dev_tools_running"),
        "no_tracked_changes": repo.get("tracked_changes") is False and repo.get("commit") == (identity or {}).get("commit"),
    }
    evaluation = step4_criteria.evaluate(prof, run=run, kernel=kernel, identity=identity_check)
    section.update(post_run_wait_s=waited, kernel=kernel, identity=identity_check, criteria=evaluation,
                   status_detail="eligible for maintainer review" if evaluation["eligible_for_maintainer_review"]
                   else "not eligible: " + ", ".join(evaluation["blocking"]))


def _sha256_arg(text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise argparse.ArgumentTypeError("expected 64 lowercase hex characters")
    return text


def main(argv: list[str] | None = None, *, backend=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-workload", choices=("check9", "u21", "s1", "step4"),
                        help="explicitly execute only this guarded diagnostic; default is read-only")
    reports = parser.add_mutually_exclusive_group()
    reports.add_argument("--check9-report", type=Path, help="successful same-boot/revision bounded Check 9 report")
    reports.add_argument("--latest-check9-report", action="store_true",
                         help="use the newest private Check 9 result in /tmp; it must pass the same checks")
    parser.add_argument("--confirm-u21-prerequisites", action="store_true",
                        help="operator confirms reviewed U18 policy, model exception and benchmark conditions")
    parser.add_argument("--s1-arm", choices=tuple(S1_ARMS),
                        help="S1 prompt-cache A/B: a = llama-server default cache, b = --cache-ram 0")
    parser.add_argument("--confirm-s1-prerequisites", action="store_true",
                        help="operator confirms S1 approval, headless preparation and the same-boot Check 9")
    parser.add_argument("--operator-dropped-caches", action="store_true",
                        help="record that the operator ran the D37 drop_caches step before this invocation")
    parser.add_argument("--step4-identity", action="store_true",
                        help="read-only: hash the clip, llama.cpp build, models and engine for step 4 (before the cache drop)")
    parser.add_argument("--step4-clip", type=Path, help="step 4 replay clip (check 8's input)")
    parser.add_argument("--step4-clip-sha256", type=_sha256_arg,
                        help="the clip's recorded SHA-256 (from the local notes; never committed)")
    parser.add_argument("--identity-report", type=Path, help="the result.json of this boot's --step4-identity")
    parser.add_argument("--confirm-step4-prerequisites", action="store_true",
                        help="operator confirms step-4 approval, headless preparation, the identity snapshot, "
                             "the cache drop and the same-boot Check 9")
    args = parser.parse_args(argv)
    if args.step4_identity and (args.execute_workload or not args.step4_clip or not args.step4_clip_sha256):
        parser.error("--step4-identity takes only --step4-clip and --step4-clip-sha256")
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)
    os.umask(0o077)
    output = Path(tempfile.mkdtemp(prefix="sentinel-operator-", dir=OUTPUT_ROOT))
    backend = backend or SystemBackend()
    if args.step4_identity:
        report = {"schema_version": 1, "mode": "step4_identity",
                  "step4_identity": step4_identity(backend, ProcessRunner(backend, lambda: interrupted), args.step4_clip,
                                                   args.step4_clip_sha256, interrupted=lambda: interrupted)}
    else:
        check9_report = latest_check9_report() if args.latest_check9_report else args.check9_report
        report = execute(args.execute_workload, backend, output,
                         check9_report=check9_report, confirm_u21=args.confirm_u21_prerequisites,
                         s1_arm=args.s1_arm, confirm_s1=args.confirm_s1_prerequisites,
                         dropped_caches=args.operator_dropped_caches, interrupted=lambda: interrupted,
                         identity_report=args.identity_report, clip=args.step4_clip,
                         confirm_step4=args.confirm_step4_prerequisites,
                         explicit_check9=not args.latest_check9_report)
    report["result_file"] = str(output / "result.json")
    report["finished_utc"] = profile.utc_now()
    report["metric"] = "MemTotal - MemAvailable, integer bytes; kB x1024; time.monotonic within boot"
    report["sampled_stop_bytes"] = PRESSURE_STOP
    report["sampled_stop_is_guaranteed_cap"] = False
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.step4_identity:  # the full report (with the clip's hash) stays in the private result.json
        identity = report["step4_identity"]
        print(json.dumps({"mode": "step4_identity", "result_file": report["result_file"],
                          **{k: identity[k] for k in ("status", "problems", "clip_matches_recorded", "started_utc",
                                                      "finished_utc", "hash_seconds_total", "boot_id", "commit")},
                          "files": len(identity["files"])}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    if interrupted:
        return 130
    if args.step4_identity:
        return 0 if report["step4_identity"]["status"] == "complete" else 1
    if args.execute_workload == "step4":
        criteria = report["step4"].get("criteria") or {}
        return 0 if criteria.get("eligible_for_maintainer_review") else 1
    if args.execute_workload:
        return 0 if report[args.execute_workload]["status"] in ("completed", "bounded_smoke_complete") else 1
    return 0 if "inspection" in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
