#!/usr/bin/env python3
"""Runs one rehearsal step as a workload of its own and watches the device's memory while it runs.

usage: guarded_run.py --name NAME --out DIR [--min-available-bytes N] [--min-free-bytes N] [--interval S]
                      [--max-gap S] [--term-grace S] [--kill-grace S] [--timeout S] [--stdin-text TEXT] -- COMMAND...

- The command is started in a new session (its own process group), with this process as the one the kernel hands
  orphaned descendants to. What this step owns is therefore known by construction: every process in that session,
  and every process that descends from this monitor. Nothing else is ever signalled, and nothing is found by name.
- MemAvailable and MemFree are read from /proc/meminfo every --interval seconds (default 0.2). The workload is
  stopped when MemAvailable falls below --min-available-bytes (default 2 GiB), when MemFree falls below
  --min-free-bytes (default 0: not watched; added for the resumed engine export, 2026-11-14), when a reading fails,
  when two readings are more than --max-gap seconds apart (default 0.5: the monitoring itself failed), or at
  --timeout. With a limit already crossed at the start, nothing is started.
- Stop: SIGTERM to every owned process (through a handle on the process, after its identity is checked), wait
  --term-grace seconds (default 20), SIGKILL to those left, wait --kill-grace seconds (default 10), then report what
  is still there. After a normal end the step's leftover processes are treated the same way.
- Writes DIR/NAME.log (the command's output), DIR/NAME.mem.csv (the samples) and DIR/NAME.result.json.
Monitoring is a precaution. It cannot prevent the kernel's own out-of-memory handling between two samples.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PR_SET_CHILD_SUBREAPER = 36


def meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemAvailable", "MemFree", "SwapFree", "SwapTotal", "MemTotal"):
                values[key] = int(rest.split()[0]) * 1024
    for key in ("MemAvailable", "MemFree"):
        if key not in values:
            raise OSError(f"{key} is not in /proc/meminfo")
    return values


def process(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        return {"pid": pid, "comm": text[text.index("(") + 1:text.rindex(")")], "state": fields[0],
                "ppid": int(fields[1]), "pgrp": int(fields[2]), "sid": int(fields[3]), "start": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def owned(session: int, session_start: int) -> list[dict]:
    """This step's processes: members of its session (created by this monitor's own child, and started no earlier
    than it), and anything that descends from this monitor."""
    table = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            info = process(int(entry))
            if info is not None and info["state"] != "Z":
                table[info["pid"]] = info
    mine = {pid for pid, info in table.items() if info["sid"] == session and info["start"] >= session_start}
    me = os.getpid()
    changed = True
    while changed:  # descendants of this monitor, whatever session they moved to
        changed = False
        for pid, info in table.items():
            if pid not in mine and pid != me and (info["ppid"] == me or info["ppid"] in mine):
                mine.add(pid)
                changed = True
    return [table[pid] for pid in sorted(mine)]


def send(info: dict, signum: int) -> bool:
    """Signal exactly that process (number and start time), through a handle on it; never a later one."""
    try:
        handle = os.pidfd_open(info["pid"])
    except OSError:
        return False
    try:
        now = process(info["pid"])
        if now is None or now["start"] != info["start"]:
            return False
        signal.pidfd_send_signal(handle, signum)
        return True
    except OSError:
        return False
    finally:
        os.close(handle)


LEADER = {"pid": 0, "poll": None}  # the command itself: collected only through its own handle, so its status is kept


def reap() -> None:
    """Collect ended processes that were handed to this monitor; the command itself only through its own handle."""
    try:
        while True:
            seen = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if seen is None:
                return
            if seen.si_pid == LEADER["pid"]:
                LEADER["poll"]()
                if os.waitid(os.P_PID, seen.si_pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    return  # not collected by its handle: leave it, never loop on it
            else:
                os.waitpid(seen.si_pid, os.WNOHANG)
    except ChildProcessError:
        return


def stop_owned(session: int, session_start: int, term_grace: float, kill_grace: float, log) -> list[dict]:
    for signum, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, kill_grace)):
        group = owned(session, session_start)
        if not group:
            return []
        log(f"sending {signal.Signals(signum).name} to {len(group)} owned process(es): "
            + " ".join(f"{p['pid']}({p['comm']})" for p in group[:12]))
        for info in group:
            send(info, signum)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            reap()
            if not owned(session, session_start):
                return []
            time.sleep(0.1)
    reap()
    return owned(session, session_start)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-available-bytes", type=int, default=2 * 1024 ** 3)
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--max-gap", type=float, default=0.5)
    parser.add_argument("--term-grace", type=float, default=20.0)
    parser.add_argument("--kill-grace", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--stdin-text", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval > 0.5:
        parser.error("a command is needed, and the interval may not exceed 0.5 s")
    args.out.mkdir(parents=True, exist_ok=True)
    log_path, csv_path = args.out / f"{args.name}.log", args.out / f"{args.name}.mem.csv"
    result_path = args.out / f"{args.name}.result.json"
    for path in (log_path, csv_path, result_path):
        if path.exists():
            parser.error(f"{path} exists: each step keeps its own evidence")
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        print("STOP: orphaned descendants could not be routed to this monitor; nothing was started", file=sys.stderr)
        return 3
    notes: list[str] = []

    def note(text: str) -> None:
        notes.append(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {text}")
        print(f"[guard] {text}", flush=True)

    first = meminfo()
    if first["MemAvailable"] < args.min_available_bytes:
        note(f"not started: MemAvailable {first['MemAvailable']} B is already below {args.min_available_bytes} B")
        result_path.write_text(json.dumps({"name": args.name, "started": False, "reason": "below_threshold_at_start",
                                           "mem_at_start": first, "notes": notes}, indent=1) + "\n")
        return 4
    if first["MemFree"] < args.min_free_bytes:
        note(f"not started: MemFree {first['MemFree']} B is already below {args.min_free_bytes} B")
        result_path.write_text(json.dumps({"name": args.name, "started": False,
                                           "reason": "memfree_below_threshold_at_start", "mem_at_start": first,
                                           "notes": notes}, indent=1) + "\n")
        return 4
    started_utc, t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    with open(log_path, "wb") as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE if args.stdin_text is not None else subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info = process(child.pid)
    session, session_start = child.pid, info["start"] if info else 0
    LEADER.update(pid=child.pid, poll=child.poll)
    if args.stdin_text is not None:
        try:
            child.stdin.write(args.stdin_text.encode())
            child.stdin.close()
        except OSError:
            pass
    reason, low, samples, max_gap, last = None, first["MemAvailable"], 0, 0.0, time.monotonic()
    low_free, low_swap = first.get("MemFree", 0), first.get("SwapFree", 0)
    with open(csv_path, "w", encoding="ascii") as csv:
        csv.write("elapsed_s,MemAvailable,MemFree,SwapFree\n")
        while True:
            time.sleep(args.interval)
            now = time.monotonic()
            gap, last = now - last, now
            max_gap = max(max_gap, gap)
            try:
                mem = meminfo()
            except (OSError, ValueError) as error:
                reason = f"monitoring_failed:{type(error).__name__}"
                break
            samples += 1
            low = min(low, mem["MemAvailable"])
            low_free, low_swap = min(low_free, mem.get("MemFree", 0)), min(low_swap, mem.get("SwapFree", 0))
            csv.write(f"{now - t0:.2f},{mem['MemAvailable']},{mem.get('MemFree', 0)},{mem.get('SwapFree', 0)}\n")
            if gap > args.max_gap:
                reason = f"monitoring_failed:sampling_gap_{gap:.2f}s"
                break
            if mem["MemAvailable"] < args.min_available_bytes:
                reason = f"memavailable_below_threshold:{mem['MemAvailable']}"
                break
            if mem["MemFree"] < args.min_free_bytes:
                reason = f"memfree_below_threshold:{mem['MemFree']}"
                break
            if args.timeout and now - t0 > args.timeout:
                reason = "timeout"
                break
            if child.poll() is not None:
                break
    if reason is not None:
        note(f"stopping the step: {reason}")
    elif owned(session, session_start):
        note("the command ended but processes of the step are still running")
    leftovers = stop_owned(session, session_start, args.term_grace, args.kill_grace, note)
    try:
        code = child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        code = None
    reap()
    leftovers = owned(session, session_start)
    result = {
        "name": args.name, "command": command, "started": True, "started_utc": started_utc,
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_s": round(time.monotonic() - t0, 1),
        "exit_code": code, "stopped_by_monitor": reason,
        "thresholds": {"min_available_bytes": args.min_available_bytes, "min_free_bytes": args.min_free_bytes,
                       "interval_s": args.interval,
                       "max_gap_s": args.max_gap, "term_grace_s": args.term_grace, "kill_grace_s": args.kill_grace,
                       "timeout_s": args.timeout},
        "mem_at_start": first, "lowest_mem_available_bytes": low, "lowest_mem_free_bytes": low_free,
        "lowest_swap_free_bytes": low_swap, "samples": samples, "max_sampling_gap_s": round(max_gap, 3),
        "leftover_processes": [{k: p[k] for k in ("pid", "comm", "start")} for p in leftovers], "notes": notes,
    }
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"[guard] {args.name}: exit={code} stopped_by_monitor={reason} elapsed={result['elapsed_s']}s "
          f"lowest_MemAvailable={low} samples={samples} max_gap={result['max_sampling_gap_s']}s "
          f"leftovers={len(leftovers)}", flush=True)
    if leftovers:
        return 5
    if reason is not None:
        return 6
    return code if code is not None else 7


if __name__ == "__main__":
    sys.exit(main())
