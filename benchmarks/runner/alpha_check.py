#!/usr/bin/env python3
"""The predeclared reading of the alpha's chat and combined device checks (D80): PASS, FAIL or MISSING; missing
evidence never passes. DESCRIPTIVE lines are never judged.

    alpha_check.py chat RUN_DIR                chat with scene analysis, the live view and monitoring (no face)
    alpha_check.py combined RUN_DIR F1_DIR     the same with face recognition and snapshots: this reading, then F2's
                                               (face_check, format 3) over the same run, both of which must pass

RUN_DIR holds the guard's files (result.json, guard.jsonl, run.jsonl, status-*.json) and alpha_load's
chat-load.jsonl and viewer.jsonl. Limits, all set before any run:

- **Memory (the existing limits, decimal bytes, MemTotal - MemAvailable from the guard's own samples):** every sample
  from STEADY_FROM_S after readiness to the stop at most 5,000,000,000; every sample of the run at most 5,400,000,000.
- **Scene deadlines (the existing scene.job_timeout_s):** no scene job missed its deadline and no waiting enrichment
  was skipped for it, over the run.
- **Chat-induced scene delay:** the largest wait of a scene job that arrived while a question ran, from its
  submission to its start on the scene worker, is at most MAX_CHAT_DELAY_MS, and the run used the declared grace
  (GRACE_MS: a periodic job lets a running question finish for that long, then ends it; an incident's job ends it at
  once; D81). Both PROPOSED (D80, D81): the maintainer decides them before any run; the reading names the values
  used. llama-server's own cancellation (it notices a closed request about once a second) shows in the next scene
  job's run time, not in its wait: the run times of the jobs that waited behind a question and of the others are
  reported (DESCRIPTIVE), and the deadline item bounds the whole.
- **Cancellation verified:** at least one scene job arrived while a question ran, ended it, and started; the chat
  load saw at least one question cancelled for it. Otherwise the item is MISSING (not exercised), never passing.
  alpha_load alternates a short and a long question so that both answers and cancellations are likely.
- **Real answers:** at least one question answered; every answered one names its frame (stream epoch, frame, ingest
  time) and its ages, is marked historical exactly when its frame was STALE_AFTER_MS or older when answered, and is
  1-600 characters long. The answers' text is never read or kept.
- **Monitoring kept first:** no incident signal lost, no detector failure, video fresh and scene analysis available
  at every capture.

Stdlib only; reads files only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import face_check  # noqa: E402
import step5_check as s5  # noqa: E402

STEADY_LIMIT = 5_000_000_000  # adapters.STEP4_TARGET_STEADY_BYTES
PEAK_LIMIT = 5_400_000_000  # adapters.STEP4_MAX_PEAK_BYTES
STEADY_FROM_S = 120.0  # the steady interval starts this long after readiness (model loading and settling excluded)
MAX_CHAT_DELAY_MS = 2000.0  # PROPOSED (D80): the maintainer's decision before any run
GRACE_MS = 1500  # analyzer.SIDE_GRACE_S, PROPOSED (D81): the maintainer's decision before any run
STALE_AFTER_MS = 2000  # freshness.stale_after_s
CANCELLED_FOR_SCENE = "a scene job is waiting: monitoring comes first"
VALIDATION_REASON = ("ask about the picture the camera is showing now (validation run: chat is not admitted; its "
                     "answers and their effect on monitoring are being checked)")
SCOPE = {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
         "workload_verified": {"runtime": 0, "scene_server": 1}}
RELEASES = {"scene": {"llm": "returned_0", "mmproj": "returned_0"}, "detector": {"engine": "returned_0"}}
USERINFO = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s@]+@")


def _jsonl(path: Path) -> list[dict] | Any:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return s5.MISSING
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            return s5.MISSING
        if isinstance(item, dict):
            out.append(item)
    return out


def _starting(run_dir: Path) -> Any:
    lines = _jsonl(run_dir / "run.jsonl")
    if lines is s5.MISSING:
        return s5.MISSING
    return next((line.get("startup") for line in lines if line.get("run") == "starting"), s5.MISSING)


def _captures(run_dir: Path, result: Any) -> list[tuple[str, Any]]:
    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    return [(name, s5._load(run_dir / f"status-{name}.json")) for name in names]


def _counter(counters: Any, key: str) -> Any:
    """A counter from a status counters object, which lists only those that have counted: absent is 0."""
    return counters.get(key, 0) if isinstance(counters, dict) else s5.MISSING


def _memory(r: s5.Reading, run_dir: Path, result: Any) -> None:
    samples = _jsonl(run_dir / "guard.jsonl")
    ready = s5._get(result, "child", "ready_mono_s")
    usable = [] if samples is s5.MISSING else [
        (s.get("t_mono"), s.get("pressure_bytes")) for s in samples
        if isinstance(s.get("t_mono"), (int, float)) and isinstance(s.get("pressure_bytes"), int)]
    if samples is s5.MISSING or not usable or not isinstance(ready, float):
        r.item("memory: steady samples at most 5,000,000,000 B, every sample at most 5,400,000,000 B", s5.MISSING,
               False)
        return
    steady = [p for t, p in usable if t - ready >= STEADY_FROM_S]
    peak = max(p for _, p in usable)
    value = {"samples": len(usable), "steady_samples": len(steady), "steady_max_bytes": max(steady) if steady else None,
             "run_peak_bytes": peak, "steady_from_s_after_ready": STEADY_FROM_S}
    r.item("memory: steady samples at most 5,000,000,000 B, every sample at most 5,400,000,000 B",
           value if steady else s5.MISSING, bool(steady) and max(steady) <= STEADY_LIMIT and peak <= PEAK_LIMIT)


def read_chat(run_dir: Path, *, face: bool = False) -> s5.Reading:
    r = s5.Reading()
    result = s5._load(run_dir / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._startup(r, result, scope=SCOPE, releases=RELEASES) if not face else s5._get(result, "run_output",
                                                                                              "starting")
    server = s5._get(start, "scene_server")
    r.item("scene server ready, 17/17 layers, vision encoder on GPU; scene admitted",
           {"scene_server": server, "scene_problem": s5._get(start, "scene_problem")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == s5.LAYERS and server.get("vision_on_gpu") is True
           and s5._get(start, "scene_problem") is None)
    starting = _starting(run_dir)
    r.item("chat offered as a validation run (not admitted); the live view on",
           {"chat": s5._get(starting, "chat"), "live_preview": isinstance(s5._get(starting, "live_preview"), dict)},
           s5._get(starting, "chat") == {"state": "available", "reason": VALIDATION_REASON}
           and isinstance(s5._get(starting, "live_preview"), dict))
    captures = _captures(run_dir, result)
    seen = [{"capture": name, "scene": s5._get(c, "components", "scene", "state"), "video": s5._get(c, "live", "video"),
             "chat_validation_run": s5._get(c, "components", "chat", "validation_run"),
             "live_view": s5._get(c, "preview", "enabled")} for name, c in captures]
    r.item("every capture: scene analysis available, video fresh, chat in its validation run, the live view on",
           seen if seen else s5.MISSING,
           bool(seen) and not s5._has_missing(seen) and all(
               s["scene"] == "available" and s["video"] == "fresh" and s["chat_validation_run"] is True
               and s["live_view"] is True for s in seen))
    s5._listeners(r, result, {18081, 18090})
    _memory(r, run_dir, result)
    last = captures[-1][1] if captures else s5.MISSING
    lane = s5._get(last, "components", "scene", "lane")
    missed = {k: (lane.get(k, 0) if isinstance(lane, dict) else s5.MISSING)
              for k in ("deadline_missed", "enrichment_skipped")}
    r.item("scene deadlines: none missed and no enrichment skipped, over the run (last capture)", missed,
           all(v == 0 for v in missed.values()))
    waits = s5._get(last, "components", "chat", "scene_waits_behind_chat")
    largest = s5._get(waits, "max_ever_ms")
    exercised = isinstance(s5._get(waits, "n"), int) and s5._get(waits, "n") >= 1
    shown = {k: s5._get(waits, k) for k in ("n", "median_ms", "max_ms", "max_ever_ms", "grace_ms")}
    r.item(f"chat-induced scene delay: the largest wait behind a question at most {MAX_CHAT_DELAY_MS:g} ms, with "
           f"the declared {GRACE_MS} ms grace (PROPOSED, D80 and D81)", shown if exercised else s5.MISSING,
           exercised and isinstance(largest, (int, float)) and largest <= MAX_CHAT_DELAY_MS
           and s5._get(waits, "grace_ms") == GRACE_MS)
    worker = s5._get(last, "components", "scene", "worker")
    load = _jsonl(run_dir / "chat-load.jsonl")
    asked = [] if load is s5.MISSING else [x for x in load if "state" in x]
    cancelled = [x for x in asked if x.get("state") == "cancelled" and x.get("reason") == CANCELLED_FOR_SCENE]
    preempted = _counter(worker, "side_preempted")
    value = {"side_preempted": preempted, "after_grace": _counter(worker, "side_preempted_after_grace"),
             "jobs_waited_for_side": _counter(worker, "jobs_waited_for_side"),
             "questions_cancelled_for_a_scene_job": len(cancelled)}
    r.item("cancellation verified: a scene job ended a running question and started; the question says why",
           value if isinstance(preempted, int) and preempted >= 1 and cancelled else s5.MISSING,
           isinstance(preempted, int) and preempted >= 1 and isinstance(value["jobs_waited_for_side"], int)
           and value["jobs_waited_for_side"] >= 1 and bool(cancelled))
    answered = [x for x in asked if x.get("state") == "answered"]
    named = [x for x in answered
             if isinstance(x.get("stream_epoch"), int) and isinstance(x.get("frame_seq"), int)
             and isinstance(x.get("ingest_utc"), str) and isinstance(x.get("age_ms_when_asked"), int)
             and isinstance(x.get("age_ms_when_answered"), int) and isinstance(x.get("historical"), bool)
             and x["historical"] == (x["age_ms_when_answered"] >= STALE_AFTER_MS)
             and isinstance(x.get("answer_chars"), int) and 1 <= x["answer_chars"] <= 600]
    r.item("real answers: at least one, each naming its frame and ages, historical exactly when its frame was "
           f"{STALE_AFTER_MS} ms or older when answered, 1-600 characters",
           {"answered": len(answered), "named_and_consistent": len(named)} if load is not s5.MISSING else s5.MISSING,
           bool(answered) and len(named) == len(answered))
    incidents = s5._get(last, "components", "incidents")
    frames = s5._get(last, "frames")
    value = {"signals_dropped": s5._get(incidents, "signals_dropped"), "frames_failed": s5._get(frames, "failed")}
    r.item("monitoring kept first: no incident signal lost, no detector failure", value,
           value["signals_dropped"] == 0 and value["frames_failed"] == 0)
    viewer = _jsonl(run_dir / "viewer.jsonl")
    pictures = None if viewer is s5.MISSING else sum(x.get("pictures", 0) for x in viewer
                                                     if isinstance(x.get("pictures"), int))
    r.item("the live view delivered pictures to a viewer during the run", pictures if pictures is not None
           else s5.MISSING, bool(pictures))
    s5._clean_stop(r, result)
    stopped = s5._get(result, "run_output", "stopped", "stopped")
    r.item("the live view's encoder stopped with the runtime", s5._get(stopped, "live_view"),
           s5._get(stopped, "live_view") is True)
    counts = {str(p.relative_to(run_dir)): len(USERINFO.findall(p.read_text(errors="replace")))
              for p in sorted(run_dir.rglob("*")) if p.is_file()}
    r.item("no URL with userinfo in any saved file", sum(counts.values()), all(v == 0 for v in counts.values()))
    # Descriptive only.
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    r.describe("rates at each capture (captured and processed fps)",
               [[name, s5._get(c, "rates", "captured_fps"), s5._get(c, "rates", "processed_fps")]
                for name, c in captures])
    states: dict[str, dict[str, int]] = {}
    for x in asked:
        kind = states.setdefault(str(x.get("kind")), {})
        kind[str(x.get("state"))] = kind.get(str(x.get("state")), 0) + 1
    r.describe("chat load: questions by kind and final state", states)
    r.describe("scene run time (ms) of the jobs that waited behind a question and of the others (the latest kept)",
               s5._get(waits, "run_ms"))
    r.describe("questions that ended by themselves while a scene job waited (within the grace), and accepted while "
               "a scene job ran", [_counter(worker, "side_finished_in_grace"),
                                    _counter(worker, "side_accepted_behind_scene")])
    times = sorted(x["answer_ms"] for x in answered if isinstance(x.get("answer_ms"), int))
    r.describe("chat: answer time (ms) median and largest",
               [times[len(times) // 2], times[-1]] if times else None)
    r.describe("chat: the gate at the last capture", s5._get(last, "components", "chat", "gate"))
    if viewer is not s5.MISSING:
        seconds = [x for x in viewer if isinstance(x.get("pictures"), int)]
        ages = [x["max_age_ms_when_sent"] for x in seconds if isinstance(x.get("max_age_ms_when_sent"), int)]
        r.describe("live view: pictures per second (mean over the seconds it ran), largest age when sent (ms)",
                   [round(pictures / len(seconds), 2) if seconds else None, max(ages) if ages else None])
    r.describe("live view encoding (ms) at the last capture", s5._get(last, "components", "preview", "encode_ms"))
    return r


def read_combined(run_dir: Path, f1: Path) -> tuple[s5.Reading, Any]:
    """This reading, with the incident and snapshot items of the alpha flow, and F2's own reading of the same run."""
    r = read_chat(run_dir, face=True)
    result = s5._load(run_dir / "result.json")
    captures = _captures(run_dir, result)
    last = captures[-1][1] if captures else s5.MISSING
    writer = s5._get(last, "components", "snapshots", "writer")
    recent = s5._get(last, "incidents", "recent")
    value = {"incidents": len(recent) if isinstance(recent, list) else s5.MISSING,
             "snapshots_saved": s5._get(writer, "saved")}
    r.item("the alpha flow ran alongside: at least one incident recorded, with its snapshot saved", value,
           isinstance(value["incidents"], int) and value["incidents"] >= 1
           and isinstance(value["snapshots_saved"], int) and value["snapshots_saved"] >= 1)
    return r, face_check.read_f2(run_dir, f1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a chat or combined device check's saved evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat").add_argument("run_dir", type=Path)
    combined = commands.add_parser("combined")
    combined.add_argument("run_dir", type=Path)
    combined.add_argument("f1", type=Path)
    args = parser.parse_args(argv)
    if args.command == "chat":
        reading = read_chat(args.run_dir)
        print("\n".join(reading.lines))
        print(f"alpha chat check: {'passed' if reading.ok else 'not passed'}")
        return 0 if reading.ok else 1
    reading, f2 = read_combined(args.run_dir, args.f1)
    print("\n".join(reading.lines))
    print("--- F2 (face recognition), over the same run")
    print("\n".join(f2.lines))
    both = reading.ok and f2.ok
    print(f"alpha combined check: {'passed' if both else 'not passed'} (alpha items {'passed' if reading.ok else 'not passed'}, "
          f"F2 {'passed' if f2.ok else 'not passed'})")
    return 0 if both else 1


if __name__ == "__main__":
    sys.exit(main())
