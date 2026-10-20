#!/usr/bin/env python3
"""The predeclared reading of the alpha's chat and combined device checks (D80): PASS, FAIL or MISSING; missing
evidence never passes. DESCRIPTIVE lines are never judged.

    alpha_check.py chat RUN_DIR       chat with scene analysis, the live view and monitoring (no face)
    alpha_check.py combined RUN_DIR --run FACE_RUN --identity-dir DIR --recheck-root DIR
                                      the same with face recognition: this reading, then F2's own (face_check, with
                                      its F1-basis gate) over the same run; both must pass
    alpha_check.py integration RUN_DIR
                                      session 103: the validation-only face + chat integration run (fci), read in four
                                      separate parts (below); never F2, never the combined run

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

**The integration run (session 103).** Face recognition (a validation run of the normal path, one enrolled identity)
and the scene model serving both scene analysis and chat (a validation run) together for INTEGRATION_DURATION_S from
readiness, no live view, snapshots or zone, read in four separately reported parts:

- **Execution:** the guard's supervision, the run's duration and its ``face_chat`` readiness, the memory policy and
  startup, face, scene and chat as configured at startup and at every capture, listeners, scene deadlines,
  monitoring first, the identity records, a clean shutdown with the face worker, the account locked, and the privacy
  counts.
- **Chat completion:** the chat-induced scene delay, the cancellation and the real answers, as in the chat check.
- **Recognition behaviour:** only that face recognition executed alongside is judged (at least MIN_FACE_APPLIED
  results applied, D63's minimum reused; no failed result; no worker error). Everything else is DESCRIPTIVE: what the
  runtime decided in this run, never recognition accuracy, retention or false acceptance, and never F2.
- **Memory:** the existing alpha limits; the guard's figures and the launch floor's basis described. The floor came
  from earlier face workloads without chat, so it guarantees no headroom for this run.

An integration pass says that these ran together within these limits; it implies neither recognition accuracy nor F2
acceptance, admits nothing and is not the combined run (which still needs a validated F2).

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
INTEGRATION_DURATION_S = 480.0  # the fci run: eight minutes from readiness (session 103)
INTEGRATION_READINESS = "face_chat"  # step5_guard.READINESS's value for this run
MIN_FACE_APPLIED = face_check.MIN_APPLIED  # D63's 30 applied results, reused as the floor for "face executed"
INTEGRATION_SCOPE = ("validation-only face + chat integration run (session 103): not F2 and not the combined run; it "
                     "assesses no recognition accuracy, retention or false acceptance; nothing is admitted or accepted")
RECOGNITION_SCOPE = ("recognition behaviour in this run only, as the runtime decided it: never recognition accuracy, "
                     "retention or false acceptance, never F2")
LAUNCH_FLOOR_NOTE = ("the launch floor 5,569,835,008 B (D103) is an empirical minimum from an earlier face workload "
                     "without chat (trk-20261017T103831Z); it guarantees no headroom for this combined run")
INTEGRATION_PARTS = ("execution", "chat completion", "recognition behaviour", "memory")


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


def _scene_deadlines(r: s5.Reading, last: Any) -> None:
    lane = s5._get(last, "components", "scene", "lane")
    missed = {k: (lane.get(k, 0) if isinstance(lane, dict) else s5.MISSING)
              for k in ("deadline_missed", "enrichment_skipped")}
    r.item("scene deadlines: none missed and no enrichment skipped, over the run (last capture)", missed,
           all(v == 0 for v in missed.values()))


def _chat_delay(r: s5.Reading, last: Any) -> Any:
    waits = s5._get(last, "components", "chat", "scene_waits_behind_chat")
    largest = s5._get(waits, "max_ever_ms")
    exercised = isinstance(s5._get(waits, "n"), int) and s5._get(waits, "n") >= 1
    shown = {k: s5._get(waits, k) for k in ("n", "median_ms", "max_ms", "max_ever_ms", "grace_ms")}
    r.item(f"chat-induced scene delay: the largest wait behind a question at most {MAX_CHAT_DELAY_MS:g} ms, with "
           f"the declared {GRACE_MS} ms grace (PROPOSED, D80 and D81)", shown if exercised else s5.MISSING,
           exercised and isinstance(largest, (int, float)) and largest <= MAX_CHAT_DELAY_MS
           and s5._get(waits, "grace_ms") == GRACE_MS)
    return waits


def _cancellation(r: s5.Reading, last: Any, asked: list[dict]) -> Any:
    worker = s5._get(last, "components", "scene", "worker")
    cancelled = [x for x in asked if x.get("state") == "cancelled" and x.get("reason") == CANCELLED_FOR_SCENE]
    preempted = _counter(worker, "side_preempted")
    value = {"side_preempted": preempted, "after_grace": _counter(worker, "side_preempted_after_grace"),
             "jobs_waited_for_side": _counter(worker, "jobs_waited_for_side"),
             "questions_cancelled_for_a_scene_job": len(cancelled)}
    r.item("cancellation verified: a scene job ended a running question and started; the question says why",
           value if isinstance(preempted, int) and preempted >= 1 and cancelled else s5.MISSING,
           isinstance(preempted, int) and preempted >= 1 and isinstance(value["jobs_waited_for_side"], int)
           and value["jobs_waited_for_side"] >= 1 and bool(cancelled))
    return worker


def _answers(r: s5.Reading, load: Any, asked: list[dict]) -> list[dict]:
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
    return answered


def _monitoring(r: s5.Reading, last: Any) -> None:
    incidents = s5._get(last, "components", "incidents")
    frames = s5._get(last, "frames")
    value = {"signals_dropped": s5._get(incidents, "signals_dropped"), "frames_failed": s5._get(frames, "failed")}
    r.item("monitoring kept first: no incident signal lost, no detector failure", value,
           value["signals_dropped"] == 0 and value["frames_failed"] == 0)


def _no_userinfo(r: s5.Reading, run_dir: Path) -> None:
    counts = {str(p.relative_to(run_dir)): len(USERINFO.findall(p.read_text(errors="replace")))
              for p in sorted(run_dir.rglob("*")) if p.is_file()}
    r.item("no URL with userinfo in any saved file", sum(counts.values()), all(v == 0 for v in counts.values()))


def _chat_described(r: s5.Reading, asked: list[dict], answered: list[dict], waits: Any, worker: Any,
                    last: Any) -> None:
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


def _asked(run_dir: Path) -> tuple[Any, list[dict]]:
    load = _jsonl(run_dir / "chat-load.jsonl")
    return load, ([] if load is s5.MISSING else [x for x in load if "state" in x])


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
    _scene_deadlines(r, last)
    waits = _chat_delay(r, last)
    load, asked = _asked(run_dir)
    worker = _cancellation(r, last, asked)
    answered = _answers(r, load, asked)
    _monitoring(r, last)
    viewer = _jsonl(run_dir / "viewer.jsonl")
    pictures = None if viewer is s5.MISSING else sum(x.get("pictures", 0) for x in viewer
                                                     if isinstance(x.get("pictures"), int))
    r.item("the live view delivered pictures to a viewer during the run", pictures if pictures is not None
           else s5.MISSING, bool(pictures))
    s5._clean_stop(r, result)
    stopped = s5._get(result, "run_output", "stopped", "stopped")
    r.item("the live view's encoder stopped with the runtime", s5._get(stopped, "live_view"),
           s5._get(stopped, "live_view") is True)
    _no_userinfo(r, run_dir)
    # Descriptive only.
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    r.describe("rates at each capture (captured and processed fps)",
               [[name, s5._get(c, "rates", "captured_fps"), s5._get(c, "rates", "processed_fps")]
                for name, c in captures])
    _chat_described(r, asked, answered, waits, worker, last)
    if viewer is not s5.MISSING:
        seconds = [x for x in viewer if isinstance(x.get("pictures"), int)]
        ages = [x["max_age_ms_when_sent"] for x in seconds if isinstance(x.get("max_age_ms_when_sent"), int)]
        r.describe("live view: pictures per second (mean over the seconds it ran), largest age when sent (ms)",
                   [round(pictures / len(seconds), 2) if seconds else None, max(ages) if ages else None])
    r.describe("live view encoding (ms) at the last capture", s5._get(last, "components", "preview", "encode_ms"))
    return r


def read_integration(run_dir: Path) -> dict[str, s5.Reading]:
    """The fci run's four parts, each its own Reading (INTEGRATION_PARTS), so that one never stands for another."""
    execution, chat, recognition, memory = (s5.Reading() for _ in INTEGRATION_PARTS)
    result = s5._load(run_dir / "result.json")
    captures = _captures(run_dir, result)
    last = captures[-1][1] if captures else s5.MISSING
    starting = _starting(run_dir)

    # ---- execution
    r = execution
    s5._supervision(r, result, "duration_stop", 0)
    readiness = s5._get(result, "readiness")
    value = {"duration_s": s5._get(result, "parameters", "duration_s"),
             "require_ready": s5._get(result, "parameters", "require_ready"),
             "readiness": {k: s5._get(readiness, k) for k in ("required", "met", "reason")}}
    r.item(f"the run: {INTEGRATION_DURATION_S:g} s from readiness, the {INTEGRATION_READINESS} readiness required and "
           "met before T0", value,
           value["duration_s"] == INTEGRATION_DURATION_S and value["require_ready"] == INTEGRATION_READINESS
           and value["readiness"] == {"required": INTEGRATION_READINESS, "met": True, "reason": None})
    start = s5._startup(r, result, scope=face_check.SCENE_SCOPE, releases=face_check.RELEASES)
    server = s5._get(start, "scene_server")
    r.item("scene server ready, 17/17 layers, vision encoder on GPU; scene admitted",
           {"scene_server": server, "scene_problem": s5._get(start, "scene_problem")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == s5.LAYERS and server.get("vision_on_gpu") is True
           and s5._get(start, "scene_problem") is None)
    r.item("face at startup: a validation run of the normal path with one enrolled identity, no face problem",
           {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") == face_check.F2_FACE and s5._get(start, "face_problem") is None)
    r.item("chat offered as a validation run (not admitted); no live view",
           {"chat": s5._get(starting, "chat"), "live_preview": isinstance(s5._get(starting, "live_preview"), dict)},
           s5._get(starting, "chat") == {"state": "available", "reason": VALIDATION_REASON}
           and isinstance(starting, dict) and "live_preview" not in starting)
    seen = [{"capture": name, "scene": s5._get(c, "components", "scene", "state"), "video": s5._get(c, "live", "video"),
             "face": s5._get(c, "components", "face", "state"),
             "face_validation_run": s5._get(c, "components", "face", "validation_run"),
             "enrolled": s5._get(c, "components", "face", "identities_enrolled"),
             "chat_validation_run": s5._get(c, "components", "chat", "validation_run")} for name, c in captures]
    r.item("every capture: scene analysis available, video fresh, face available (validation run, 1 enrolled), chat in "
           "its validation run", seen if seen else s5.MISSING,
           bool(seen) and not s5._has_missing(seen) and all(
               x["scene"] == "available" and x["video"] == "fresh" and x["face"] == "available"
               and x["face_validation_run"] is True and x["enrolled"] == 1 and x["chat_validation_run"] is True
               for x in seen))
    s5._listeners(r, result, {18081, 18090})
    _scene_deadlines(r, last)
    _monitoring(r, last)
    records, problems, _, _ = face_check.read_records(run_dir / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {face_check.FULL_SCALE_SCHEMA}, valid, none dropped",
           {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped} if total else s5.MISSING,
           total > 0 and not problems and schemas == [face_check.FULL_SCALE_SCHEMA] and dropped == 0
           and total < face_check.MAX_IDENTITY_RECORDS)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    output = s5._get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else s5.MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and (counts["lines"] or 0) >= 2)
    account, problem, fmt = face_check._account(run_dir)
    r.item("account (format 11) saved and locked before any result was shown", problem or {"format": fmt},
           account is not None and fmt == 11)
    _no_userinfo(r, run_dir)
    face_check._counts_items(r, run_dir)
    r.describe("rates at each capture (captured and processed fps)",
               [[name, s5._get(c, "rates", "captured_fps"), s5._get(c, "rates", "processed_fps")]
                for name, c in captures])

    # ---- chat completion
    waits = _chat_delay(chat, last)
    load, asked = _asked(run_dir)
    worker = _cancellation(chat, last, asked)
    answered = _answers(chat, load, asked)
    _chat_described(chat, asked, answered, waits, worker, last)

    # ---- recognition behaviour: face executed alongside (judged); the rest descriptive
    r = recognition
    results = s5._get(last, "components", "face", "results")
    counters = s5._get(last, "components", "face", "worker", "counters")
    value = {"results_applied": _counter(results, "results_applied"),
             "results_failed": _counter(results, "results_failed"), "worker_errors": _counter(counters, "errors")}
    r.item(f"face executed alongside: at least {MIN_FACE_APPLIED} results applied (D63's minimum, reused), no failed "
           "result, no worker error (last capture)", value,
           isinstance(value["results_applied"], int) and value["results_applied"] >= MIN_FACE_APPLIED
           and value["results_failed"] == 0 and value["worker_errors"] == 0)
    r.describe("scope", RECOGNITION_SCOPE)
    r.describe("face results by outcome (last capture)",
               {k: v for k, v in sorted(results.items()) if k.startswith("results_")} if isinstance(results, dict)
               else None)
    r.describe("votes and persons by outcome (last capture)",
               {k: v for k, v in sorted(results.items()) if not k.startswith("results_")} if isinstance(results, dict)
               else None)
    r.describe("face worker: ticks, offers and runs (last capture)", counters)
    r.describe("face processing ms at the last capture", s5._get(last, "components", "face", "worker", "processing_ms"))
    moves: dict[str, int] = {}
    for item in records.get("transition", []):
        key = f"{item.get('from')}->{item.get('to')}"
        moves[key] = moves.get(key, 0) + 1
    r.describe("identity transitions by from and to (records)", dict(sorted(moves.items())))
    r.describe("the account's answers (unknown accepted; they qualify these lines only)",
               {k: account.get(k) for k in ("stayed_seated", "worked_as_usual", "looked_at_camera_on_purpose",
                                            "other_faces_in_view")} if account is not None else None)

    # ---- memory
    _memory(memory, run_dir, result)
    memory.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
                    [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    memory.describe("launch floor", LAUNCH_FLOOR_NOTE)
    return dict(zip(INTEGRATION_PARTS, (execution, chat, recognition, memory)))


def read_combined(run_dir: Path, face_run: Path, identity_dir: Path,
                  recheck_root: Path | None) -> tuple[s5.Reading, Any]:
    """This reading, with the incident and snapshot items of the alpha flow, and F2's own reading of the same run,
    built on FACE_RUN's F1 (its F1-basis gate included, as face_check f2 --run reads it)."""
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
    return r, face_check.read_f2(run_dir, face_run / "f1", (face_run, identity_dir, recheck_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a chat or combined device check's saved evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat").add_argument("run_dir", type=Path)
    combined = commands.add_parser("combined")
    combined.add_argument("run_dir", type=Path)
    combined.add_argument("--run", type=Path, required=True, help="the face run whose F1 this F2 builds on")
    combined.add_argument("--identity-dir", type=Path, required=True)
    combined.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are")
    commands.add_parser("integration").add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "chat":
        reading = read_chat(args.run_dir)
        print("\n".join(reading.lines))
        print(f"alpha chat check: {'passed' if reading.ok else 'not passed'}")
        return 0 if reading.ok else 1
    if args.command == "integration":
        parts = read_integration(args.run_dir)
        for name, part in parts.items():
            print(f"=== {name}")
            print("\n".join(part.lines))
            print(f"integration {name}: {'passed' if part.ok else 'not passed'}"
                  + (" (only that face executed alongside is judged; no recognition accuracy)"
                     if name == "recognition behaviour" else ""))
        every = all(part.ok for part in parts.values())
        print(f"alpha integration check: {'passed' if every else 'not passed'} ({INTEGRATION_SCOPE})")
        return 0 if every else 1
    reading, f2 = read_combined(args.run_dir, args.run, args.identity_dir, args.recheck_root)
    print("\n".join(reading.lines))
    print("--- F2 (face recognition), over the same run")
    print("\n".join(f2.lines))
    both = reading.ok and f2.ok

    def said(ok: bool) -> str:
        return "passed" if ok else "not passed"

    print(f"alpha combined check: {said(both)} (alpha items {said(reading.ok)}, F2 {said(f2.ok)})")
    return 0 if both else 1


if __name__ == "__main__":
    sys.exit(main())
