#!/usr/bin/env python3
"""Checklist step 5 (D59): run ``sentinel run`` for a bounded time under the existing memory guard, keeping evidence.

Operator tooling, not part of the runtime. The command after ``--`` starts in its own session, with stdout written to
``run.jsonl`` and stderr to ``run.err`` in ``--out``: a new directory, normally under ~/sentinel-runs so that a
reboot keeps it. operator_check's PressureGuard samples every 0.2 s with its limits unchanged: pressure 4.8 GB,
MemFree 1 GiB, MemAvailable 2 GiB, any swap-counter change, or a sampling gap over 0.5 s.

The runtime is stopped the way Ctrl-C stops it: SIGINT to the runtime process alone. The runtime then stops its own
llama-server, which runs in its own session. That happens when:
- ``--duration-s`` has passed since the runtime's ``"run": "starting"`` line (or since launch, with
  ``--from-launch``);
- the guard meets a condition;
- the runtime is not ready within ``--ready-timeout-s``; or
- this process gets SIGINT or SIGTERM.

If the runtime has not exited ``--stop-grace-s`` after that, its process group gets SIGTERM and then SIGKILL. Any
llama-server left afterwards is stopped and reported; none may run before launch.

With ``--step4-headroom`` it also refuses before launch unless operator_check's step-4 admission headroom holds
(pressure below 2.0 GB, MemFree at least 3.5 GB, MemAvailable at least 4.0 GB; ``baseline_problem``, unchanged).

At each ``--cue-at SECONDS:NAME:TEXT`` offset (counted like the duration) it prints TEXT with a bell to its own
stderr, the operator's terminal, and records when: the guard's monotonic clock is CLOCK_MONOTONIC, the runtime's
too, so ``child.ready_mono_s`` and each cue's ``printed_mono_s`` place the runtime's records relative to the cue
the operator's stopwatch started from. Cue texts are fixed protocol text, never a secret.

At each ``--status-at SECONDS:NAME`` offset it saves ``status-NAME.json`` from the loopback status page, and it
classifies the listeners on 18090 and 18081 as loopback, any or other. ``result.json`` holds numbers and fixed labels
only. No stream URL, token, chat ID, address or environment value is read or written; the command's own output goes
to ``run.jsonl`` and ``run.err`` unchanged.

Exit status 0 means the supervision itself was clean: the runtime exited on its own, or exited 0 after the duration's
stop, with no guard stop, no forced stop, no leftover process and no listener left. Whether the runtime did what a
step-5 part expects is judged from ``result.json`` by the operator block.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import operator_check as oc

SAMPLE_S = oc.SAMPLE_S
WATCHED_PORTS = (18090, 18081)  # the status page and the scene server, both loopback-only by design
STATUS_URL = "http://127.0.0.1:18090/status.json"
MAX_STATUS_BYTES = 1_048_576
STATUS_TIMEOUT_S = 2.0
MAX_DURATION_S = 900.0
# sentinel run's shutdown is bounded at about 66 s with scene analysis (capture 16 s, scene server 10 s plus 5 s to
# kill, scene worker 21 s, outbox 12 s, status page 2 s); the grace leaves room above that.
STOP_GRACE_S = 90.0
KILL_GRACE_S = 5.0
READY_TIMEOUT_S = 300.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- read-only host facts


def _scope_v4(hex_address: str) -> str:
    octets = bytes.fromhex(hex_address)[::-1]
    if octets == b"\0\0\0\0":
        return "any"
    return "loopback" if octets[0] == 127 else "other"


def _scope_v6(hex_address: str) -> str:
    raw = bytes.fromhex(hex_address)
    words = b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4))  # four host-order 32-bit words
    if words == bytes(16):
        return "any"
    if words == bytes(15) + b"\1":
        return "loopback"
    if words[:12] == bytes(10) + b"\xff\xff":  # an IPv4-mapped address
        return "any" if words[12:] == b"\0\0\0\0" else ("loopback" if words[12] == 127 else "other")
    return "other"


def listeners(root: Path = Path("/proc/net"), ports: tuple[int, ...] = WATCHED_PORTS) -> list[dict[str, Any]]:
    """Listening TCP sockets on ``ports``: port and scope (loopback, any, other) only, never the address."""
    found = []
    for name, scope in (("tcp", _scope_v4), ("tcp6", _scope_v6)):
        try:
            lines = (root / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":  # 0A: LISTEN
                continue
            address, _, port_hex = fields[1].partition(":")
            try:
                port = int(port_hex, 16)
                if port in ports:
                    found.append({"port": port, "family": name, "scope": scope(address)})
            except ValueError:
                continue
    return sorted(found, key=lambda item: (item["port"], item["family"], item["scope"]))


def process_ids(comm: str, proc: Path = Path("/proc"), uid: int | None = None) -> list[int]:
    """This user's processes whose name is ``comm``."""
    uid = os.getuid() if uid is None else uid
    found = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid == uid and (path / "comm").read_text().strip() == comm:
                found.append(int(path.name))
        except OSError:
            continue
    return sorted(found)


def thp_enabled_self() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("THP_enabled:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def fetch_status(url: str = STATUS_URL) -> tuple[str, bytes]:
    """(label, body): ``saved`` with at most MAX_STATUS_BYTES, or an error class; never raises."""
    try:
        with urllib.request.urlopen(url, timeout=STATUS_TIMEOUT_S) as response:  # loopback only
            body = response.read(MAX_STATUS_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"unavailable:{type(exc).__name__}", b""
    if len(body) > MAX_STATUS_BYTES:
        return "too_large", b""
    try:
        json.loads(body)
    except ValueError:
        return "not_json", b""
    return "saved", body


class Child:
    def __init__(self, argv: list[str], stdout: Path, stderr: Path) -> None:
        with open(stdout, "wb") as out, open(stderr, "wb") as err:
            self.process = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                            start_new_session=True)
        self.pid = self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def signal(self, signum: int) -> None:
        if self.process.poll() is None:
            try:
                os.kill(self.pid, signum)
            except ProcessLookupError:
                pass

    def signal_group(self, signum: int) -> bool:
        """Signal the runtime's whole group; False once nothing in it is left."""
        try:
            os.killpg(self.pid, signum)
            return True
        except ProcessLookupError:
            return False

    def group_alive(self) -> bool:
        self.process.poll()
        return self.signal_group(0)


class SystemBackend:
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    utc = staticmethod(utc_now)
    spawn = staticmethod(Child)
    listeners = staticmethod(listeners)
    fetch = staticmethod(fetch_status)
    thp_self = staticmethod(thp_enabled_self)

    def __init__(self) -> None:
        self._oc = oc.SystemBackend()

    def sample(self) -> dict:
        return self._oc.sample()

    @staticmethod
    def llama_pids() -> list[int]:
        return process_ids("llama-server")

    @staticmethod
    def kill(pid: int, signum: int) -> bool:
        try:
            os.kill(pid, signum)
            return True
        except ProcessLookupError:
            return False


# ---------------------------------------------------------------- the runtime's own output


def _starting_summary(startup: dict) -> dict[str, Any]:
    """The policy and component evidence from the runtime's ``starting`` line: numbers and fixed labels."""
    releases = {}
    for component, record in (startup.get("releases") or {}).items():
        files = (record or {}).get("files") or {}
        releases[component] = {role: (item or {}).get("result") for role, item in files.items()}
    disable = startup.get("thp_disable") or {}
    server = startup.get("scene_server")
    return {
        "memory_policy": startup.get("memory_policy"),
        "thp_scope": startup.get("thp_scope"),
        "thp_disable": {key: disable.get(key) for key in ("verified", "reason", "set_rc", "thp_enabled")}
        if disable else None,
        "releases": releases,
        "scene_server": {key: server.get(key) for key in ("state", "problem", "layers", "vision_on_gpu")}
        if isinstance(server, dict) else None,
        "scene_problem": startup.get("scene_problem"),
        "detector_problem": startup.get("detector_problem"),
        "face": startup.get("face"),  # sentinel run --face: validation_run and identities_enrolled (a count)
        "face_problem": startup.get("face_problem"),
        "face_memory_before": startup.get("face_memory_before"),
        "notifier_problems": startup.get("notifier_problems"),
        "scene_memory_before": startup.get("scene_memory_before"),
        "detector_memory_before": startup.get("detector_memory_before"),
    }


def read_run_output(path: Path) -> dict[str, Any]:
    """Line counts (D48: every stdout line is JSON), the starting summary and the stopped line."""
    lines = json_lines = 0
    starting = stopped = None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            item = json.loads(line)
        except ValueError:
            continue
        json_lines += 1
        if isinstance(item, dict) and item.get("run") == "starting" and starting is None:
            starting = _starting_summary(item.get("startup") or {})
        elif isinstance(item, dict) and item.get("run") == "stopped":
            shutdown = item.get("shutdown") or {}
            stopped = {"all_stopped": shutdown.get("all_stopped"), "stopped": shutdown.get("stopped"),
                       "database_closed": item.get("database_closed"),
                       "signals_not_recorded": shutdown.get("signals_not_recorded")}
    return {"lines": lines, "json_lines": json_lines, "non_json_lines": lines - json_lines,
            "starting": starting, "stopped": stopped}


def last_error_label(path: Path) -> str | None:
    """The runtime's last ``run: <label>`` line on stderr (a refusal names its reason there), else None."""
    try:
        lines = [line for line in path.read_text(errors="replace").splitlines() if line.startswith("run: ")]
    except OSError:
        return None
    return lines[-1][len("run: "):][:400] if lines else None


# ---------------------------------------------------------------- supervision


def preflight(backend, out: Path, step4_headroom: bool = False) -> str | None:
    if out.exists():
        return "out_exists"
    if backend.thp_self() != 1:
        return "thp_enabled_not_1"
    if backend.llama_pids():
        return "llama_server_running"
    if backend.listeners():
        return "port_in_use"
    sample = backend.sample()
    if oc.memory_problem(sample):
        return "memory_guard_already_met"
    if step4_headroom and oc.baseline_problem(sample):  # step 4's admission headroom, its values unchanged
        return "initial_headroom_refused"
    return None


def supervise(argv: list[str], out: Path, *, duration_s: float, from_launch: bool = False,
              ready_timeout_s: float = READY_TIMEOUT_S, stop_grace_s: float = STOP_GRACE_S,
              status_at: tuple[tuple[float, str], ...] = (), step4_headroom: bool = False, backend=None,
              interrupted=lambda: False, cues: tuple[tuple[float, str, str], ...] = (),
              say=lambda text: (sys.stderr.write(text + "\n"), sys.stderr.flush())) -> dict:
    backend = backend or SystemBackend()
    result: dict[str, Any] = {"schema_version": 1, "mode": "step5_guard", "status": None,
                              "parameters": {"duration_s": duration_s, "from_launch": from_launch,
                                             "ready_timeout_s": ready_timeout_s, "stop_grace_s": stop_grace_s,
                                             "step4_headroom": step4_headroom,
                                             "status_at": [{"at_s": at, "name": name} for at, name in status_at],
                                             "cues": [{"at_s": at, "name": name} for at, name, _ in cues]},
                              "limits": {"pressure_stop_bytes": oc.PRESSURE_STOP, "mem_free_floor_bytes": oc.FREE_FLOOR,
                                         "mem_available_floor_bytes": oc.AVAILABLE_FLOOR,
                                         "swap_counters": "unchanged", "max_sample_gap_s": 0.5}}
    refused = preflight(backend, out, step4_headroom)
    if refused:
        result["status"] = f"refused:{refused}"
        result["preflight_sample"] = oc.safe_sample(backend.sample())
        if refused != "out_exists":  # keep the refusal as evidence; never write into an existing directory
            out.mkdir(mode=0o700, parents=False)
        return result
    out.mkdir(mode=0o700, parents=False)
    baseline = backend.sample()
    result["baseline"] = oc.safe_sample(baseline)
    guard_file = open(out / "guard.jsonl", "w")
    guard = oc.PressureGuard(baseline, emit=lambda reading: guard_file.write(json.dumps(reading) + "\n"))
    stdout, stderr = out / "run.jsonl", out / "run.err"
    launch = backend.clock()
    result["child"] = {"launched_utc": backend.utc()}
    child = backend.spawn(argv, stdout, stderr)
    ready_at = None
    read_offset = 0
    stop = {"requested_by": None, "requested_at_s": None, "exited_after_stop_s": None, "forced": False}
    pending = sorted(status_at)
    cue_queue = sorted(cues)
    printed: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    workers: list[threading.Thread] = []
    returncode = None
    gaps, previous, min_available = [], None, None

    def capture(at: float, name: str, taken: float) -> None:  # off the sampling loop: a slow page never stalls it
        label, body = backend.fetch()
        if label == "saved":
            (out / f"status-{name}.json").write_bytes(body)
        captures.append({"name": name, "at_s": at, "taken_after_launch_s": taken, "status": label,
                         "bytes": len(body), "listeners": backend.listeners()})
    try:
        while True:
            now = backend.clock()
            sample = backend.sample()
            guard.observe(sample)
            available = sample.get("MemAvailable")
            if type(available) is int:
                min_available = available if min_available is None else min(min_available, available)
            stamp = sample.get("t_mono")
            if isinstance(stamp, (int, float)):
                if previous is not None:
                    gaps.append(stamp - previous)
                previous = stamp
            if ready_at is None:
                try:
                    with open(stdout, "rb") as handle:
                        handle.seek(read_offset)
                        chunk = handle.read()
                except OSError:
                    chunk = b""
                complete = chunk[:chunk.rfind(b"\n") + 1]
                read_offset += len(complete)
                for line in complete.splitlines():
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict) and item.get("run") == "starting":
                        ready_at = now
                        result["child"]["ready_after_s"] = round(now - launch, 3)
                        result["child"]["ready_mono_s"] = round(now, 6)
                        break
            returncode = child.poll()
            if returncode is not None:
                break
            base = launch if from_launch else ready_at
            if stop["requested_by"] is None:
                reason = None
                if guard.trigger is not None:
                    reason = "guard_stop"
                elif interrupted():
                    reason = "operator_interrupt"
                elif ready_at is None and not from_launch and now - launch >= ready_timeout_s:
                    reason = "ready_timeout"
                elif base is not None and now - base >= duration_s:
                    reason = "duration_elapsed"
                if reason:
                    child.signal(signal.SIGINT)  # sentinel run's Ctrl-C path: a bounded, clean shutdown
                    stop.update(requested_by=reason, requested_at_s=round(now - launch, 3))
            elif not stop["forced"] and now - launch - stop["requested_at_s"] >= stop_grace_s:
                stop["forced"] = True
                child.signal_group(signal.SIGTERM)
                deadline = backend.clock() + KILL_GRACE_S
                while child.group_alive() and backend.clock() < deadline:
                    backend.sleep(0.1)
                child.signal_group(signal.SIGKILL)
            while cue_queue and base is not None and stop["requested_by"] is None and now - base >= cue_queue[0][0]:
                at, name, text = cue_queue.pop(0)
                elapsed = int(now - base)
                say(f"\a[{elapsed // 60}:{elapsed % 60:02d}] {text}")
                printed.append({"name": name, "at_s": at, "printed_after_launch_s": round(now - launch, 3),
                                "printed_mono_s": round(now, 6)})
            while pending and base is not None and stop["requested_by"] is None and now - base >= pending[0][0]:
                at, name = pending.pop(0)
                worker = threading.Thread(target=capture, args=(at, name, round(now - launch, 3)), daemon=True)
                worker.start()
                workers.append(worker)
            backend.sleep(SAMPLE_S)
    finally:
        guard_file.close()
        for worker in workers:
            worker.join(STATUS_TIMEOUT_S + 1.0)
    ended = backend.clock()
    if stop["requested_at_s"] is not None:
        stop["exited_after_stop_s"] = round(ended - launch - stop["requested_at_s"], 3)
    result["child"].update(returncode=returncode, exited_utc=backend.utc(), ran_s=round(ended - launch, 3))
    for at, name in pending:
        captures.append({"name": name, "at_s": at, "status": "not_reached", "bytes": 0, "listeners": None})
    for at, name, _ in cue_queue:
        printed.append({"name": name, "at_s": at, "printed_after_launch_s": None, "printed_mono_s": None})
    orphans = backend.llama_pids()
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for pid in orphans:
            backend.kill(pid, signum)
        deadline = backend.clock() + KILL_GRACE_S
        while backend.llama_pids() and backend.clock() < deadline:
            backend.sleep(0.1)
    left = backend.llama_pids()
    group_left = child.group_alive()
    after = backend.listeners()
    result.update(
        stop=stop,
        guard={"samples": guard.samples, "largest_gap_s": round(max(gaps), 3) if gaps else None,
               "peak_pressure_bytes": guard.peak, "min_mem_free_bytes": guard.minimum_free,
               "min_mem_available_bytes": min_available, "trigger": guard.trigger,
               "trigger_after_stop_request": bool(guard.trigger and stop["requested_by"] not in (None, "guard_stop"))},
        captures=captures,
        cues=printed,
        run_output=read_run_output(stdout),
        run_error_label=last_error_label(stderr),
        leftovers={"llama_server_after_exit": len(orphans), "llama_server_left": len(left),
                   "runtime_group_left": group_left, "listeners_after": after},
    )
    clear = not left and not group_left and not after
    if not clear:
        status = "cleanup_failed"
    elif stop["forced"]:
        status = "forced_stop"
    elif stop["requested_by"] in ("guard_stop", "ready_timeout", "operator_interrupt"):
        status = stop["requested_by"]
    elif stop["requested_by"] == "duration_elapsed":
        status = "duration_stop"
    else:
        status = "child_exited"
    result["status"] = status
    result["cleanup_clear"] = clear and not orphans
    return result


def _status_at(text: str) -> tuple[float, str]:
    seconds, sep, name = text.partition(":")
    if not sep or not name.replace("-", "").isalnum() or len(name) > 32:
        raise argparse.ArgumentTypeError("use SECONDS:NAME with a short alphanumeric name")
    value = float(seconds)
    if not 0 <= value <= MAX_DURATION_S:
        raise argparse.ArgumentTypeError(f"SECONDS must be within 0-{MAX_DURATION_S:g}")
    return value, name


def _cue_at(text: str) -> tuple[float, str, str]:
    seconds, sep, rest = text.partition(":")
    name, sep2, words = rest.partition(":")
    if not (sep and sep2) or not name.replace("-", "").isalnum() or len(name) > 32:
        raise argparse.ArgumentTypeError("use SECONDS:NAME:TEXT with a short alphanumeric name")
    if not words or len(words) > 160 or not all(32 <= ord(ch) < 127 for ch in words):
        raise argparse.ArgumentTypeError("TEXT: 1-160 printable ASCII characters")
    try:
        value = float(seconds)
    except ValueError:
        raise argparse.ArgumentTypeError("SECONDS must be a number") from None
    if not 0 <= value <= MAX_DURATION_S:
        raise argparse.ArgumentTypeError(f"SECONDS must be within 0-{MAX_DURATION_S:g}")
    return value, name, words


def _seconds(limit: float):
    def parse(text: str) -> float:
        value = float(text)
        if not 0 < value <= limit:
            raise argparse.ArgumentTypeError(f"must be within (0, {limit:g}]")
        return value
    return parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a new directory for this part's evidence")
    parser.add_argument("--duration-s", type=_seconds(MAX_DURATION_S), required=True)
    parser.add_argument("--from-launch", action="store_true",
                        help="count the duration and the status offsets from launch instead of from readiness")
    parser.add_argument("--ready-timeout-s", type=_seconds(MAX_DURATION_S), default=READY_TIMEOUT_S)
    parser.add_argument("--stop-grace-s", type=_seconds(300.0), default=STOP_GRACE_S)
    parser.add_argument("--status-at", type=_status_at, action="append", default=[], metavar="SECONDS:NAME")
    parser.add_argument("--cue-at", type=_cue_at, action="append", default=[], metavar="SECONDS:NAME:TEXT",
                        help="print TEXT to the operator's terminal at this offset (counted like the duration)")
    parser.add_argument("--step4-headroom", action="store_true",
                        help="also refuse before launch unless step 4's admission headroom holds (operator_check: "
                             "pressure below 2.0 GB, MemFree at least 3.5 GB, MemAvailable at least 4.0 GB)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the sentinel run command")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    flag = {"set": False}

    def interrupt(*_):
        flag["set"] = True

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = supervise(command, args.out, duration_s=args.duration_s, from_launch=args.from_launch,
                           ready_timeout_s=args.ready_timeout_s, stop_grace_s=args.stop_grace_s,
                           status_at=tuple(args.status_at), step4_headroom=args.step4_headroom,
                           interrupted=lambda: flag["set"], cues=tuple(args.cue_at))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    text = json.dumps(result, indent=2)
    if result["status"] != "refused:out_exists" and args.out.is_dir():
        (args.out / "result.json").write_text(text + "\n")
    print(text)
    clean = result.get("cleanup_clear") is True and (
        result["status"] == "child_exited"
        or (result["status"] == "duration_stop" and (result.get("child") or {}).get("returncode") == 0))
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
