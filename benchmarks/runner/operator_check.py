"""Operator-only inspection and opt-in bounded diagnostics; no hardware acceptance."""

from __future__ import annotations

import argparse
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

SAMPLE_S = 0.2
PRESSURE_STOP = 4_800_000_000
FREE_FLOOR = 1 << 30
AVAILABLE_FLOOR = 2 << 30
OUTPUT_LIMIT = 1_048_576
MEMORY_KEYS = (*profile.MEMINFO_KEYS, "pswpin", "pswpout")
REQUIRED = ("MemTotal", "MemAvailable", "MemFree", "pswpin", "pswpout")
SERVICES = ("ollama.service", "display-manager.service")
OUTPUT_ROOT = Path("/tmp")
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

    def sample(self) -> dict:
        stamp = self.clock()
        return {"t_mono": stamp, **profile.read_meminfo(), **profile.read_swap_counters()}

    def context(self) -> dict:
        categories = {key: {"count": 0, "pids": []} for key in
                      ("model_servers", "desktop", "dev_tools", "python_unclassified", "media_or_gpu_tools")}
        unavailable = False
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
            "limitations": "comm-only workload classification; no argv/environment; not proof of no GPU users"}


def check9_prerequisite(path: Path | None, current: dict) -> str | None:
    if path is None:
        return "check9_report_required"
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
        if (path.is_symlink() or resolved.name != "result.json"
                or not resolved.parent.name.startswith("sentinel-operator-")
                or not resolved.is_relative_to(OUTPUT_ROOT.resolve())
                or info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode)
                or info.st_size > OUTPUT_LIMIT):
            return "check9_report_refused"
        with path.open("rb") as handle:
            data = handle.read(OUTPUT_LIMIT + 1)
        if len(data) > OUTPUT_LIMIT:
            return "check9_report_refused"
        prior = json.loads(data)
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


def execute(mode: str | None, backend, output: Path, *, check9_report=None, confirm_u21=False,
            interrupted=lambda: False) -> dict:
    runner = ProcessRunner(backend, interrupted)
    report = {"schema_version": 1, "mode": mode or "inspection",
              "hardware_acceptance": "PENDING", "check9": {"status": "PENDING", "u18_acceptance": "PENDING",
              "server_comparison": "PENDING"}, "u21": {"status": "PENDING"}}
    try:
        current = inspection(backend, runner)
        report["inspection"] = current
        report["check9"]["kernel_inspection"] = current["kernel"]
        if mode is None:
            return report
        refusal = current["workload_refusals"]
        if mode == "u21":
            problem = check9_prerequisite(check9_report, current)
            if problem:
                refusal.append(problem)
            if not confirm_u21:
                refusal.append("u21_operator_prerequisites_unconfirmed")
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
            else:
                child = runner.run([
                    "/usr/bin/python3", str(profile.HERE / "demo_profile.py"), "--out", str(output),
                    "--no-evict", "--sanitized-logs", "--baseline-s", "15", "--settle-s", "15",
                    "--warmup-s", "30", "--steady-s", "120", "--face-hz", "1", "--scene-interval-s", "4",
                    "--llama-timeout-s", "60", "--load-timeout-s", "90", "--min-free-gb", "3.5",
                ], 360.0, guard=guard)
                report["u21"].update(child.diagnostic())
                report["u21"]["operator_prerequisites_confirmed"] = True
            report[mode].update(samples=guard.samples, sampled_peak_pressure_bytes=guard.peak,
                                sampled_min_free_bytes=guard.minimum_free)
        if not interrupted():
            backend.sleep(5.0)
            report[mode]["post_exit"] = safe_sample(backend.sample())
    except Exception:
        report[mode or "check9"]["status"] = "diagnostic_unavailable"
    return report


def main(argv: list[str] | None = None, *, backend=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-workload", choices=("check9", "u21"),
                        help="explicitly execute only this guarded diagnostic; default is read-only")
    parser.add_argument("--check9-report", type=Path, help="successful same-boot/revision bounded Check 9 report")
    parser.add_argument("--confirm-u21-prerequisites", action="store_true",
                        help="operator confirms reviewed U18 policy, model exception and benchmark conditions")
    args = parser.parse_args(argv)
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)
    os.umask(0o077)
    output = Path(tempfile.mkdtemp(prefix="sentinel-operator-", dir=OUTPUT_ROOT))
    report = execute(args.execute_workload, backend or SystemBackend(), output,
                     check9_report=args.check9_report, confirm_u21=args.confirm_u21_prerequisites,
                     interrupted=lambda: interrupted)
    report["result_file"] = str(output / "result.json")
    report["finished_utc"] = profile.utc_now()
    report["metric"] = "MemTotal - MemAvailable, integer bytes; kB x1024; time.monotonic within boot"
    report["sampled_stop_bytes"] = PRESSURE_STOP
    report["sampled_stop_is_guaranteed_cap"] = False
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if interrupted:
        return 130
    if args.execute_workload:
        return 0 if report[args.execute_workload]["status"] in ("completed", "bounded_smoke_complete") else 1
    return 0 if "inspection" in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
