#!/usr/bin/env python3
"""Memory around the operator's page-cache drop and immediately before a guarded launch (D103, session 87).

D103 extends D37's operator-run ``sync && sudo sysctl -w vm.drop_caches=1`` to one F2 package as a one-time,
operator-confirmed preparation step, and adds a launch floor that the launch block checks after its questions,
immediately before the workload starts. This tool only reads /proc (meminfo with the guard's own parser, the boot ID
and the clocks) and the files it is given, and writes the record it is asked for; it never calls sudo, never writes a
sysctl and never drops a cache: the operator's block does that (D37).

    launch_memory.py derive --reference-result RESULT.json
    launch_memory.py snapshot --label before_drop|after_drop|prepared --out FILE [--since FILE]
    launch_memory.py check --session-dir DIR --reference-result RESULT.json --out FILE [--prefix XX_]
    launch_memory.py report --session-dir DIR --out FILE [--result RESULT.json]

The launch floor is 5,569,835,008 B of MemFree: the guard's MemFree floor (1,073,741,824 B; operator_check.FREE_FLOOR)
plus the largest MemFree consumption from launch that the successful ten-minute diagnostic trk-20261017T103831Z
recorded (its launch baseline 5,745,963,008 B less its lowest MemFree 1,249,869,824 B: 4,496,093,184 B). It is an
empirical minimum from that one run: no safety reserve is added, and passing it guarantees nothing about the run that
follows. ``derive`` and ``check`` recompute it from that run's sealed result.json (its SHA-256 pinned here) and
refuse when it does not reproduce.

``check`` judges, in order: the evidence, then the memory. The evidence: the session's provenance shows f-F passed,
then exactly one drop attempt whose sync and drop exited 0 and whose block passed, then f-O passed with its prepared
record named; the before-drop, after-drop and prepared records exist, are valid, are this boot's and are in time
order; the reference result reproduces the floor. The memory, read last: MemFree now at or above the floor passes
(the guard stops only below its own floor). It prints fixed ``key=value`` lines (the block appends them to the
session's provenance), explains on stderr, writes its record to FILE (never over an existing file) and exits 0
(passed), 1 (MemFree below the floor) or 2 (evidence or telemetry missing or invalid; FILE exists; usage). The
elapsed times and the changes since the drop and since f-O are recorded, never judged. ``report`` describes the
records and the guard's result after a launch; it judges nothing.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import demo_profile as profile
import operator_check as oc

SCHEMA = "sentinel-launch-memory/1"
CONDITION = "page_cache_dropped_once"
REFERENCE_SESSION = "trk-20261017T103831Z"
# its sealed tv/result.json
REFERENCE_RESULT_SHA256 = "57578d19e802aff212749d70382e74db5ca5d75307b860b4cbcda5b9e94e9e67"
REFERENCE_BASELINE_MEMFREE = 5_745_963_008  # that run's launch baseline (the guard's first sample)
REFERENCE_MIN_MEMFREE = 1_249_869_824  # its lowest MemFree over the run (guard.min_mem_free_bytes)
LAUNCH_FLOOR_BYTES = 5_569_835_008  # oc.FREE_FLOOR + REFERENCE_BASELINE_MEMFREE - REFERENCE_MIN_MEMFREE
FLOOR_BASIS = ("an empirical minimum from one successful ten-minute diagnostic (trk-20261017T103831Z): no safety "
               "reserve is added and passing it guarantees nothing")
LABELS = ("before_drop", "after_drop", "prepared")
RECORD_NAMES = {"before_drop": "memory-before-drop.json", "after_drop": "memory-after-drop.json"}
PREPARED_NAME = re.compile(r"memory-prepared-\d{8}T\d{6}Z\.json")
KEYS = (*profile.MEMINFO_KEYS, "Buffers", "SReclaimable", "Dirty", "Writeback")
REQUIRED = ("MemTotal", "MemFree", "MemAvailable", "Cached")
CHANGED = ("MemFree", "MemAvailable", "Cached")
BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
PREFIX = re.compile(r"[A-Z]{1,4}_")


class Refused(Exception):
    """An evidence or telemetry problem, named by a fixed label."""


class SystemHost:
    """The readings a record holds; tests give a stand-in. The monotonic clock is the guard's (CLOCK_MONOTONIC), so a
    record and the guard's baseline sample can be placed on one time line within a boot."""

    @staticmethod
    def meminfo() -> dict[str, int]:
        return profile.read_meminfo(Path("/proc/meminfo"), KEYS)

    @staticmethod
    def boot_id() -> str:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def boottime() -> float:
        return time.clock_gettime(time.CLOCK_BOOTTIME)

    @staticmethod
    def utc() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def reading(host: Any) -> dict:
    """One reading of memory now, with the boot and both clocks; Refused when meminfo lacks a required field."""
    try:
        values = host.meminfo()
        boot = host.boot_id()
    except OSError:
        raise Refused("meminfo_unreadable") from None
    if any(key not in values for key in REQUIRED):
        raise Refused("meminfo_unreadable")
    return {"utc": host.utc(), "boot_id": boot, "monotonic_s": host.monotonic(), "boottime_s": host.boottime(),
            "meminfo": dict(values)}


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _seconds(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def snapshot(label: str, host: Any) -> dict:
    return {"schema": SCHEMA, "kind": "snapshot", "label": label, **reading(host)}


def load_record(path: Path, label: str | None = None) -> dict:
    """A snapshot record of this label (any of LABELS when None), checked field by field; Refused
    (<label>_record_missing|_invalid) otherwise."""
    name = label or "snapshot"
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        raise Refused(f"{name}_record_missing") from None
    except (OSError, UnicodeDecodeError, ValueError):
        raise Refused(f"{name}_record_invalid") from None
    meminfo = record.get("meminfo") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA or record.get("kind") != "snapshot"
            or record.get("label") not in ((label,) if label else LABELS) or not isinstance(record.get("boot_id"), str)
            or not BOOT_ID.fullmatch(record["boot_id"]) or not isinstance(record.get("utc"), str)
            or not _seconds(record.get("monotonic_s")) or not _seconds(record.get("boottime_s"))
            or not isinstance(meminfo, dict) or not all(_count(meminfo.get(key)) for key in REQUIRED)):
        raise Refused(f"{name}_record_invalid")
    return record


def derivation(result: Any) -> dict:
    """The launch floor recomputed from the reference run's result; Refused (reference_result_invalid) unless that run
    completed (the duration's stop, no guard trigger, the runtime's exit 0) under the guard's floor still in force and
    its numbers reproduce LAUNCH_FLOOR_BYTES."""
    try:
        floor = result["limits"]["mem_free_floor_bytes"]
        baseline = result["baseline"]["MemFree"]
        lowest = result["guard"]["min_mem_free_bytes"]
        complete = (result["mode"] == "step5_guard" and result["status"] == "duration_stop"
                    and result["guard"]["trigger"] is None and result["child"]["returncode"] == 0)
    except (KeyError, TypeError):
        raise Refused("reference_result_invalid") from None
    if not (complete and _count(floor) and _count(baseline) and _count(lowest) and floor == oc.FREE_FLOOR
            and baseline > lowest and floor + baseline - lowest == LAUNCH_FLOOR_BYTES):
        raise Refused("reference_result_invalid")
    return {"bytes": LAUNCH_FLOOR_BYTES, "guard_free_floor_bytes": floor, "reference_session": REFERENCE_SESSION,
            "reference_baseline_memfree_bytes": baseline, "reference_min_memfree_bytes": lowest,
            "reference_consumption_bytes": baseline - lowest, "reference_result_sha256": REFERENCE_RESULT_SHA256,
            "basis": FLOOR_BASIS}


def load_reference(path: Path) -> dict:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise Refused("reference_result_missing") from None
    except OSError:
        raise Refused("reference_result_unreadable") from None
    if hashlib.sha256(data).hexdigest() != REFERENCE_RESULT_SHA256:
        raise Refused("reference_result_differs")
    try:
        result = json.loads(data)
    except ValueError:
        raise Refused("reference_result_invalid") from None
    return derivation(result)


def _last(lines: list[str], pattern: str) -> int:
    """The index of the last line matching pattern (a full match), or -1."""
    found = [i for i, line in enumerate(lines) if re.fullmatch(pattern, line)]
    return found[-1] if found else -1


def preparation(provenance: Path) -> str:
    """The prepared record's file name, when the provenance shows f-F's pass, then exactly one drop attempt whose sync
    and drop exited 0 and whose block passed, then f-O's pass naming that record; Refused with the first gap."""
    try:
        lines = provenance.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        raise Refused("provenance_unreadable") from None
    attempts = [i for i, line in enumerate(lines) if line.startswith("FM_attempt=")]
    if not attempts:
        raise Refused("drop_not_attempted")
    if len(attempts) > 1:
        raise Refused("drop_attempted_more_than_once")
    attempt = attempts[0]
    passed_f = _last(lines, r"FF_exit=0")
    if passed_f < 0 or passed_f > attempt or _last(lines, r"FF_exit=.*") > attempt:
        raise Refused("ff_not_passed_before_drop")
    after = lines[attempt + 1:]
    sync = next((line for line in after if line.startswith("FM_sync_exit=")), None)
    if sync != "FM_sync_exit=0":
        raise Refused("sync_failed")
    drop = next((line for line in after if line.startswith("FM_drop_exit=")), None)
    if drop != "FM_drop_exit=0":
        raise Refused("drop_failed")
    passed_m = _last(lines, r"FM_exit=.*")
    if passed_m < attempt or lines[passed_m] != "FM_exit=0":
        raise Refused("drop_block_failed")
    passed_o = _last(lines, r"FO_exit=.*")
    if passed_o < passed_m or lines[passed_o] != "FO_exit=0":
        raise Refused("fo_not_passed_after_drop")
    named = _last(lines, r"FO_memory_record=.*")
    if not passed_m < named < passed_o or not PREPARED_NAME.fullmatch(lines[named].partition("=")[2]):
        raise Refused("prepared_record_not_named")
    return lines[named].partition("=")[2]


def _change(later: dict, earlier: dict) -> dict:
    return {key: later["meminfo"][key] - earlier["meminfo"][key] for key in CHANGED}


def _round(value: float) -> float:
    return round(value, 3)


def check(session: Path, reference: Path, out: Path, host: Any) -> tuple[list[tuple[str, str]], int, dict | None]:
    """The (key, value) lines, the exit status and the record written (None when FILE exists already)."""
    if out.exists() or out.is_symlink():
        lines = [("launch_check", "refused:evidence:launch_record_exists"), ("launch_memfree", "none"),
                 ("launch_floor", str(LAUNCH_FLOOR_BYTES)), ("launch_margin", "none"),
                 ("launch_since_drop_s", "none"), ("launch_since_prepared_s", "none")]
        return lines, 2, None
    problem = floor = None
    records: dict[str, dict] = {}
    names = dict(RECORD_NAMES)
    try:
        names["prepared"] = preparation(session / "provenance.txt")
        for label in LABELS:
            records[label] = load_record(session / names[label], label)
        floor = load_reference(reference)
    except Refused as refused:
        problem = str(refused)
    now = None
    try:
        now = reading(host)  # the memory last: immediately before the block launches the workload
    except Refused as refused:
        problem = problem or str(refused)
    if problem is None and any(r["boot_id"] != now["boot_id"] for r in records.values()):
        problem = "boot_changed"
    if problem is None and not (records["before_drop"]["boottime_s"] <= records["after_drop"]["boottime_s"]
                                <= records["prepared"]["boottime_s"] <= now["boottime_s"]):
        problem = "records_out_of_order"
    same_boot = now is not None and all(r["boot_id"] == now["boot_id"] for r in records.values())
    after, prepared = records.get("after_drop"), records.get("prepared")
    since_drop = _round(now["boottime_s"] - after["boottime_s"]) if same_boot and after else None
    since_prepared = _round(now["boottime_s"] - prepared["boottime_s"]) if same_boot and prepared else None
    memfree = now["meminfo"]["MemFree"] if now else None
    margin = memfree - LAUNCH_FLOOR_BYTES if memfree is not None else None
    if problem is not None:
        verdict, code = f"refused:evidence:{problem}", 2
    elif margin >= 0:
        verdict, code = "passed", 0
    else:
        verdict, code = "refused:below_floor", 1
    record = {"schema": SCHEMA, "kind": "launch_check", "condition": CONDITION, "verdict": verdict.partition(":")[0],
              "reason": verdict.partition(":")[2] or None,
              "floor": floor or {"bytes": LAUNCH_FLOOR_BYTES, "basis": FLOOR_BASIS,
                                 "reference_session": REFERENCE_SESSION},
              "now": now, "margin_bytes": margin, "since_drop_s": since_drop, "since_prepared_s": since_prepared,
              "change_since_drop": _change(now, after) if since_drop is not None else None,
              "change_since_prepared": _change(now, prepared) if since_prepared is not None else None,
              "records": {label: names.get(label) for label in LABELS}}
    lines = [("launch_check", verdict), ("launch_memfree", str(memfree) if memfree is not None else "none"),
             ("launch_floor", str(LAUNCH_FLOOR_BYTES)),
             ("launch_margin", str(margin) if margin is not None else "none"),
             ("launch_since_drop_s", f"{since_drop:.1f}" if since_drop is not None else "none"),
             ("launch_since_prepared_s", f"{since_prepared:.1f}" if since_prepared is not None else "none")]
    return lines, code, record


def write_new(path: Path, text: str) -> None:
    """Write a new file (mode 600); never replaces one (FileExistsError)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def _b(value: int) -> str:
    return f"{value:,} B"


def _signed(value: int) -> str:
    return f"{value:+,} B"


def derivation_text(floor: dict | None = None) -> str:
    floor = floor or {"guard_free_floor_bytes": oc.FREE_FLOOR, "reference_baseline_memfree_bytes":
                      REFERENCE_BASELINE_MEMFREE, "reference_min_memfree_bytes": REFERENCE_MIN_MEMFREE}
    consumption = floor["reference_baseline_memfree_bytes"] - floor["reference_min_memfree_bytes"]
    return (f"launch floor {_b(LAUNCH_FLOOR_BYTES)} of MemFree = the guard's MemFree floor "
            f"{_b(floor['guard_free_floor_bytes'])} + {REFERENCE_SESSION}'s largest MemFree consumption from its "
            f"launch, {_b(consumption)} (its launch baseline {_b(floor['reference_baseline_memfree_bytes'])} - its "
            f"lowest MemFree "
            f"{_b(floor['reference_min_memfree_bytes'])}); {FLOOR_BASIS}")


def _memory(record: dict | None) -> str:
    if not record:
        return "MISSING"
    m = record["meminfo"]
    return (f"MemFree {_b(m['MemFree'])}, MemAvailable {_b(m['MemAvailable'])}, Cached {_b(m['Cached'])} "
            f"({record['utc']})")


def _changes(change: dict) -> str:
    return ", ".join(f"{key} {_signed(change[key])}" for key in CHANGED)


def _try(load: Callable[[], dict]) -> dict | None:
    try:
        return load()
    except Refused:
        return None


def _launch_record(path: Path) -> dict | None:
    """The launch check's record when it holds a reading and its margin; None otherwise."""
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("kind") != "launch_check" or record.get("schema") != SCHEMA:
        return None
    now = record.get("now")
    usable = (isinstance(now, dict) and isinstance(now.get("meminfo"), dict) and isinstance(now.get("utc"), str)
              and all(_count(now["meminfo"].get(key)) for key in REQUIRED) and _seconds(now.get("monotonic_s"))
              and type(record.get("margin_bytes")) is int)
    return record if usable else None


def _result(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return result if isinstance(result, dict) and isinstance(result.get("baseline"), dict) else None


def report(session: Path, result_path: Path | None) -> list[str]:
    """Descriptive lines: the condition, the floor's derivation and each recorded state, the guard's launch baseline
    included; MISSING where a record or the result is absent or unreadable. Judges nothing."""
    lines = [f"CONDITION {CONDITION} (D103): the operator ran sync and sudo sysctl -w vm.drop_caches=1 once in this "
             "session before the launch; a changed test condition: report this session apart from every F2 session "
             "without it, never pooled", derivation_text()]
    named = None
    try:
        named = preparation(session / "provenance.txt")
    except Refused:
        pass
    before = _try(lambda: load_record(session / RECORD_NAMES["before_drop"], "before_drop"))
    after = _try(lambda: load_record(session / RECORD_NAMES["after_drop"], "after_drop"))
    prepared = _try(lambda: load_record(session / named, "prepared")) if named else None
    launch = _launch_record(session / "memory-launch-check.json")
    lines += [f"memory before the drop: {_memory(before)}", f"memory after the drop: {_memory(after)}"]
    if before and after:
        lines.append(f"change across the drop: {_changes(_change(after, before))}")
    lines.append(f"memory at f-O's prepared record: {_memory(prepared)}")
    now = launch["now"] if launch else None
    if launch:
        verdict = launch.get("verdict") if launch.get("verdict") in ("passed", "refused") else "unrecognized"
        reason = launch.get("reason") if isinstance(launch.get("reason"), str) else None
        lines.append(f"memory at the launch check: {_memory(now)}; {_signed(launch['margin_bytes'])} against the "
                     f"launch floor ({verdict}{':' + reason if reason else ''}); since the drop "
                     f"{launch.get('since_drop_s')} s, since f-O {launch.get('since_prepared_s')} s")
        change = launch.get("change_since_prepared")
        if isinstance(change, dict) and all(type(change.get(key)) is int for key in CHANGED):
            lines.append(f"change from f-O to the launch check: {_changes(change)}")
    else:
        lines.append("memory at the launch check: MISSING")
    result = _result(result_path)
    base = result["baseline"] if result else None
    if base and _count(base.get("MemFree")) and _seconds(base.get("t_mono")):
        text = (f"memory at the launch (the guard's baseline sample): MemFree {_b(base['MemFree'])}, "
                f"{_signed(base['MemFree'] - LAUNCH_FLOOR_BYTES)} against the launch floor")
        if isinstance(now, dict) and _seconds(now.get("monotonic_s")):
            text += (f"; {base['t_mono'] - now['monotonic_s']:.3f} s after the launch check, MemFree "
                     f"{_signed(base['MemFree'] - now['meminfo']['MemFree'])} since it")
        if after:
            text += f"; {base['t_mono'] - after['monotonic_s']:.1f} s after the drop"
        lines.append(text)
        guard = result.get("guard") if isinstance(result.get("guard"), dict) else {}
        lowest = guard.get("min_mem_free_bytes")
        if _count(lowest):
            lines.append(f"the run's lowest MemFree (the guard's): {_b(lowest)}, {_signed(lowest - oc.FREE_FLOOR)} "
                         f"against the guard's MemFree floor {_b(oc.FREE_FLOOR)}; the guard's status "
                         f"{result.get('status') if isinstance(result.get('status'), str) else 'unrecognized'}")
    else:
        lines.append("memory at the launch (the guard's baseline sample): MISSING")
    return lines


def _prefix(text: str) -> str:
    if text and not PREFIX.fullmatch(text):
        raise argparse.ArgumentTypeError("a prefix is 1-4 capital letters and an underscore")
    return text


def main(argv: list[str] | None = None, host: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive", help="recompute the launch floor from the reference run's result")
    derive.add_argument("--reference-result", type=Path, required=True)
    snap = commands.add_parser("snapshot", help="record memory now to a new file")
    snap.add_argument("--label", choices=LABELS, required=True)
    snap.add_argument("--out", type=Path, required=True)
    snap.add_argument("--since", type=Path, default=None, help="an earlier record: also print the change since it")
    chk = commands.add_parser("check", help="the evidence, then MemFree now against the launch floor")
    chk.add_argument("--session-dir", type=Path, required=True)
    chk.add_argument("--reference-result", type=Path, required=True)
    chk.add_argument("--out", type=Path, required=True)
    chk.add_argument("--prefix", type=_prefix, default="")
    rep = commands.add_parser("report", help="describe the records and the guard's launch baseline")
    rep.add_argument("--session-dir", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--result", type=Path, default=None)
    args = parser.parse_args(argv)
    host = host or SystemHost()
    if args.command == "derive":
        try:
            floor = load_reference(args.reference_result)
        except Refused as refused:
            print(f"STOP: the launch floor does not reproduce from {args.reference_result} ({refused})",
                  file=sys.stderr)
            return 2
        print(derivation_text(floor))
        return 0
    if args.command == "snapshot":
        try:
            record = snapshot(args.label, host)
            write_new(args.out, json.dumps(record, indent=1, sort_keys=True) + "\n")
        except Refused as refused:
            print(f"STOP: memory could not be read ({refused}); nothing was written", file=sys.stderr)
            return 2
        except OSError as error:
            print(f"STOP: {args.out} could not be written as a new file ({error.__class__.__name__})", file=sys.stderr)
            return 2
        print(f"memory {args.label}: {_memory(record)}")
        if args.since is not None:
            earlier = _try(lambda: load_record(args.since))
            print(f"change since {args.since.name}: {_changes(_change(record, earlier)) if earlier else 'unavailable'}")
        return 0
    if args.command == "check":
        lines, code, record = check(args.session_dir, args.reference_result, args.out, host)
        if record is not None:
            try:
                write_new(args.out, json.dumps(record, indent=1, sort_keys=True) + "\n")
            except OSError as error:
                print(f"STOP: {args.out} could not be written ({error.__class__.__name__})", file=sys.stderr)
                lines[0], code = ("launch_check", "refused:evidence:launch_record_unwritable"), 2
        for key, value in lines:
            print(f"{args.prefix}{key}={value}")
        print(derivation_text(record["floor"] if record and "guard_free_floor_bytes" in record["floor"] else None),
              file=sys.stderr)
        values = dict(lines)
        if values["launch_memfree"] != "none":
            print(f"MemFree now {_b(int(values['launch_memfree']))}: {_signed(int(values['launch_margin']))} against "
                  f"the launch floor; since the drop {values['launch_since_drop_s']} s, since f-O's prepared record "
                  f"{values['launch_since_prepared_s']} s; {values['launch_check']}", file=sys.stderr)
        return code
    lines = report(args.session_dir, args.result)
    try:
        write_new(args.out, "\n".join(lines) + "\n")
    except OSError as error:
        print(f"STOP: {args.out} could not be written as a new file ({error.__class__.__name__})", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
