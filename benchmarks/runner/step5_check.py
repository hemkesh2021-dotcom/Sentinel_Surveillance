#!/usr/bin/env python3
"""Checklist step 5 (D59): the predeclared reading of each part, from the files step5_guard wrote.

    step5_check.py {refusal,core,scene} PART_DIR --data-dir DATA_DIR

It prints one line per item, ``PASS``, ``FAIL`` or ``MISSING``, and then ``step5-<part>: validated`` or ``not
validated``. Missing evidence never passes. Exit status 0 means every item passed. ``DESCRIPTIVE`` lines are reported
and never judged. Only ``result.json`` and the saved ``status-*.json`` files are read, plus whether the database file
exists. Nothing is written, and no secret is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROFILE_ID = "step4cand-demo-20261007T090339Z"
EXPECTED_REFUSAL = (
    f"scene_not_admitted: llama-lfm2-vl-scene: resource profile {PROFILE_ID} measured memory policy "
    "(THP workload_disabled, model-file release post_load), not the runtime's (THP system, model-file release none)"
)
CANDIDATE_POLICY = {"thp": "workload_disabled", "model_file_release": "post_load"}
LAYERS = "offloaded 17/17 layers to GPU"
DATABASE = "sentinel.db"
GUARD_FIGURES = ("peak_pressure_bytes", "min_mem_free_bytes", "min_mem_available_bytes")
MISSING = object()


def _get(item: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(item, dict) or key not in item:
            return MISSING
        item = item[key]
    return item


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return MISSING


def _has_missing(value: Any) -> bool:
    if value is MISSING:
        return True
    if isinstance(value, dict):
        return any(_has_missing(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_missing(v) for v in value)
    return False


def _text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=lambda o: "missing" if o is MISSING else str(o))


class Reading:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ok = True

    def item(self, name: str, value: Any, passed: bool) -> None:
        label = "MISSING" if _has_missing(value) else ("PASS" if passed else "FAIL")
        self.ok = self.ok and label == "PASS"
        self.lines.append(f"{label} {name}: {_text(value)}")

    def describe(self, name: str, value: Any) -> None:
        self.lines.append(f"DESCRIPTIVE {name}: {_text(value)}")


def _supervision(r: Reading, result: dict, expected_status: str, expected_rc: int) -> None:
    status, rc = _get(result, "status"), _get(result, "child", "returncode")
    r.item("supervision status and runtime exit", {"status": status, "returncode": rc},
           status == expected_status and rc == expected_rc)
    trigger, forced = _get(result, "guard", "trigger"), _get(result, "stop", "forced")
    r.item("no memory-guard stop and no forced stop", {"trigger": trigger, "forced": forced},
           trigger is None and forced is False)
    left = _get(result, "leftovers")
    r.item("nothing left: no llama-server, runtime group or listener; cleanup clear",
           {"leftovers": left, "cleanup_clear": _get(result, "cleanup_clear")},
           isinstance(left, dict) and left.get("llama_server_after_exit") == 0 and left.get("llama_server_left") == 0
           and left.get("runtime_group_left") is False and left.get("listeners_after") == []
           and _get(result, "cleanup_clear") is True)


def _clean_stop(r: Reading, result: dict) -> None:
    stopped = _get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped, database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True)
    output = _get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and (counts["lines"] or 0) >= 2)


def _listeners(r: Reading, result: dict, ports: set[int]) -> None:
    captures = _get(result, "captures")
    seen = [c.get("listeners") for c in captures] if isinstance(captures, list) else MISSING
    flat = [item for group in seen for item in (group or [])] if seen is not MISSING else []
    r.item(f"listeners at each capture: loopback only, ports {sorted(ports)}", seen,
           seen is not MISSING and bool(seen) and all(group for group in seen)
           and all(item.get("scope") == "loopback" for item in flat)
           and all({item.get("port") for item in group} == ports for group in seen))


def _startup(r: Reading, result: dict, *, scope: dict, releases: dict) -> dict:
    start = _get(result, "run_output", "starting")
    r.item("memory policy selected: THP workload_disabled, release post_load", _get(start, "memory_policy"),
           _get(start, "memory_policy") == CANDIDATE_POLICY)
    r.item("THP scope at each startup checkpoint", _get(start, "thp_scope"), _get(start, "thp_scope") == scope)
    disable = _get(start, "thp_disable")
    r.item("THP disable verified in the runtime process", disable,
           isinstance(disable, dict) and disable.get("verified") is True and disable.get("reason") is None)
    r.item("post-load releases: every call returned 0", _get(start, "releases"), _get(start, "releases") == releases)
    r.item("detector loaded", _get(start, "detector_problem"), _get(start, "detector_problem") is None)
    return start if isinstance(start, dict) else {}


def _status(directory: Path, name: str) -> Any:
    return _load(directory / f"status-{name}.json")


def _deliveries(status: Any) -> Any:
    return _get(status, "delivery", "totals")


def read_refusal(directory: Path, data: Path) -> Reading:
    r = Reading()
    result = _load(directory / "result.json")
    _supervision(r, result, "child_exited", 1)
    r.item("refusal label", _get(result, "run_error_label"), _get(result, "run_error_label") == EXPECTED_REFUSAL)
    r.item("refused before startup: no starting or stopped line",
           {"starting": _get(result, "run_output", "starting"), "stopped": _get(result, "run_output", "stopped")},
           _get(result, "run_output", "starting") is None and _get(result, "run_output", "stopped") is None)
    r.item("no incident database created", (data / DATABASE).exists(), not (data / DATABASE).exists())
    return r


def read_core(directory: Path, data: Path) -> Reading:
    r = Reading()
    result = _load(directory / "result.json")
    _supervision(r, result, "duration_stop", 0)
    start = _startup(r, result, scope={"before_launch": {"runtime": 1}, "workload_verified": {"runtime": 0}},
                     releases={"detector": {"engine": "returned_0"}})
    r.item("Telegram channel available at startup", _get(start, "notifier_problems"),
           _get(start, "notifier_problems") == {})
    idle, alert = _status(directory, "idle"), _status(directory, "alert")
    r.item("idle (1:30): running, video fresh, occupancy empty",
           {"state": _get(idle, "runtime", "state"), "video": _get(idle, "live", "video"),
            "occupancy": _get(idle, "live", "occupancy")},
           _get(idle, "runtime", "state") == "running" and _get(idle, "live", "video") == "fresh"
           and _get(idle, "live", "occupancy") == "empty")
    r.item("idle (1:30): no incident and no delivery row",
           {"incidents": _get(idle, "incidents", "recent"), "delivery": _deliveries(idle)},
           _get(idle, "incidents", "recent") == []
           and isinstance(_deliveries(idle), dict) and not any(_deliveries(idle).values()))
    recent = _get(alert, "incidents", "recent")
    r.item("alert (3:30): exactly one incident", len(recent) if isinstance(recent, list) else MISSING,
           isinstance(recent, list) and len(recent) == 1)
    totals = _deliveries(alert)
    r.item("alert (3:30): exactly one delivery row, delivered, unambiguous", totals,
           isinstance(totals, dict) and totals.get("delivered") == 1 and totals.get("ambiguous") == 0
           and sum(totals.get(k) or 0 for k in ("queued", "attempted", "delivered", "failed")) == 1)
    _clean_stop(r, result)
    _listeners(r, result, {18090})
    r.describe("ready after (s)", _get(result, "child", "ready_after_s"))
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [_get(result, "guard", k) for k in GUARD_FIGURES])
    return r


def read_scene(directory: Path, data: Path) -> Reading:
    r = Reading()
    result = _load(directory / "result.json")
    _supervision(r, result, "duration_stop", 0)
    start = _startup(r, result,
                     scope={"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                            "workload_verified": {"runtime": 0, "scene_server": 1}},
                     releases={"scene": {"llm": "returned_0", "mmproj": "returned_0"},
                               "detector": {"engine": "returned_0"}})
    server = _get(start, "scene_server")
    r.item("scene server ready, 17/17 layers, vision encoder on GPU; scene admitted",
           {"scene_server": server, "scene_problem": _get(start, "scene_problem")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == LAYERS and server.get("vision_on_gpu") is True
           and _get(start, "scene_problem") is None)
    first, last = _status(directory, "scene-start"), _status(directory, "scene-end")
    r.item("scene analysis available and video fresh at both captures",
           [{"scene": _get(s, "components", "scene", "state"), "video": _get(s, "live", "video")}
            for s in (first, last)],
           all(_get(s, "components", "scene", "state") == "available" and _get(s, "live", "video") == "fresh"
               for s in (first, last)))
    delivered = [_get(s, "components", "scene", "worker", "delivered") for s in (first, last)]
    r.item("scene reports arrived between the captures", delivered,
           all(isinstance(v, int) for v in delivered) and delivered[1] > delivered[0])
    totals = _deliveries(last)
    r.item("no delivery row (no notification channel in this part)", totals,
           isinstance(totals, dict) and not any(totals.values()))
    _clean_stop(r, result)
    _listeners(r, result, {18081, 18090})
    span = [_get(c, "taken_after_launch_s") for c in _list(_get(result, "captures"))
            if isinstance(c, dict) and c.get("name") in ("scene-start", "scene-end")]
    if all(isinstance(v, int) for v in delivered) and len(span) == 2 and all(isinstance(v, float) for v in span):
        r.describe("scene reports between the captures; per 4 s",
                   [delivered[1] - delivered[0], round((delivered[1] - delivered[0]) * 4 / (span[1] - span[0]), 2)])
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [_get(result, "guard", k) for k in GUARD_FIGURES])
    r.describe("scene memory before (MemFree check), detector memory before",
               [_get(start, "scene_memory_before"), _get(start, "detector_memory_before")])
    recent = _get(last, "incidents", "recent")
    r.describe("incidents at the last capture", len(recent) if isinstance(recent, list) else MISSING)
    return r


READERS = {"refusal": read_refusal, "core": read_core, "scene": read_scene}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("part", choices=sorted(READERS))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reading = READERS[args.part](args.directory, args.data_dir)
    for line in reading.lines:
        print(line)
    print(f"step5-{args.part}: {'validated' if reading.ok else 'not validated'}")
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
