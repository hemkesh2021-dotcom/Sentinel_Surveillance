"""The multi-pose comparison's execution check: one guarded session (B, E_M or E_P) judged before anything reads its
results (PROPOSED; validated with synthetic data only).

usage: mpe_check.py RUN_DIR --f1 F1_DIR --session B|E_M|E_P [--frozen EXP_ROOT/candidate.frozen.json]
                    [--run F1_RUN --identity-dir DIR --recheck-root DIR]

RUN_DIR is the run (the guard's result.json, run.jsonl, status captures, counts, provenance with the account lock);
its parent the session (face.yaml, marks.txt, provenance with the launch lines). V3's execution items are judged with
range_check's and face_check's own helpers, unchanged (range_check itself is not touched), with the multi-pose run's
own expectations: the startup face entry and the guard's readiness of that kind (B: the raw capture; E: the shadow
comparison of the frozen candidate, its digest, and its track lives), the session's cues and duration, its status
captures, the records read with the experiment's own rules (B: exactly one raw capture record, written and verified,
nothing dropped, every compared one-person face kept; E: every compared face with a decided shadow, the track lives
recorded), the face.yaml as the launch block ran it, the marks, the privacy counts, and the locked account (the account
step comes first: this check is a reading, so it refuses without it). E_P also shows no picture, preview, chat or
feature control in the runtime's starting line (D102). The spoken cues, when used, are described, never judged.

Prints PASS/FAIL/MISSING items and ``mpe-execution: passed`` only when every item passed; exits 1 otherwise. The
builder and the comparison refuse to read a session that has not passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import face_check as fc
import mpe_account
import mpe_schedule
import mpe_speak
import range_check as rc
import step5_check as s5

MISSING = s5.MISSING
READINESS = {"B": "face_windows_capture_raw", "E_M": "face_windows_shadow", "E_P": "face_windows_shadow"}
EXPERIMENT = {"B": "capture_raw", "E_M": "shadow", "E_P": "shadow"}
LAUNCH_PREFIX = "MV"  # the package's launch block records MV_config_sha256 before MV_launch_utc


def expected_face(kind: str, frozen: dict | None) -> dict:
    if kind == "B":
        return {**fc.WINDOWS_FACE, "capture": True, "capture_raw": True}
    return {**fc.WINDOWS_FACE, "shadow": True, "shadow_digest": (frozen or {}).get("prototypes_digest"), "lives": True}


def _compared_one_person(records: dict[str, list[dict]]) -> int:
    return sum(1 for r in records.get("result", []) if r.get("outcome") == "applied" and len(r["persons"]) == 1
               and r["persons"][0]["label"] in ("match", "unknown", "margin"))


def read_mpe_execution(directory: Path, f1: Path, kind: str, frozen: dict | None = None,
                       gate: tuple[Path, Path, Path | None] | None = None) -> fc.Reading:
    directory = Path(directory)
    r = fc.Reading()
    if gate is not None:
        ok, lines = fc.f1_gate(*gate)
        shown = {k: ("recorded" if k == "f1_identity_id" else v)  # never the identity ID itself
                 for k, v in (line.split("=", 1) for line in lines if "=" in line)}
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", shown, ok and f1.resolve() == (gate[0] / "f1").resolve())
    if kind != "B":
        r.item("the frozen candidate's record", "present" if frozen else MISSING,
               bool(frozen) and frozen.get("format") == "mpe-frozen-1")
    try:  # the account step: before anything of this session is judged or shown
        account = mpe_account.require(directory, kind)
        r.item("account saved and locked before any result was shown (the account step)", "locked", True)
    except mpe_account.AccountRefused as exc:
        r.item("account saved and locked before any result was shown (the account step)", exc.label, False)
        r.lines.append("mpe-execution: not passed (STOP: enter the account first; nothing else is read)")
        return r
    r.item("the account: nobody else in view", account["others_in_view"], account["others_in_view"] == "no")
    result = s5._load(directory / "result.json")
    r.item(f"the session's schedule ({kind}: its cue names and times)", mpe_schedule.schedule_of(result) or MISSING,
           mpe_schedule.schedule_of(result) == kind)
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("memory policy selected: THP workload_disabled, release post_load", s5._get(start, "memory_policy"),
           s5._get(start, "memory_policy") == s5.CANDIDATE_POLICY)
    r.item("THP scope at each startup checkpoint (no scene server)", s5._get(start, "thp_scope"),
           s5._get(start, "thp_scope") == rc.FACE_ONLY_SCOPE)
    disable = s5._get(start, "thp_disable")
    r.item("THP disable verified in the runtime process", disable,
           isinstance(disable, dict) and disable.get("verified") is True and disable.get("reason") is None)
    r.item("post-load releases, the detector's and the face weights': every call returned 0",
           s5._get(start, "releases"), s5._get(start, "releases") == rc.FACE_ONLY_RELEASES)
    r.item("no scene server or scene analysis; detector loaded",
           {k: s5._get(start, k) for k in ("scene_server", "scene_problem", "detector_problem")},
           isinstance(start, dict) and start.get("scene_server") is None and start.get("scene_problem") is None
           and start.get("detector_problem") is None)
    want = expected_face(kind, frozen)
    face = s5._get(start, "face")
    r.item(f"face admitted for this validation run on route B, the {EXPERIMENT[kind]} run's own entry (the frozen "
           "candidate's digest for a shadow run): one enrolled identity, no problem",
           {"face": {k: ("recorded" if k == "shadow_digest" else v) for k, v in face.items()}
            if isinstance(face, dict) else face, "face_problem": s5._get(start, "face_problem")},
           face == want and s5._get(start, "face_problem") is None)
    r.item("no notification channel (Telegram disabled)", s5._get(start, "notifier_problems"),
           s5._get(start, "notifier_problems") == {})
    readiness = s5._get(result, "readiness")
    seen_ready = {k: s5._get(readiness, k) for k in ("required", "met", "reason")}
    r.item(f"the guard's readiness: {READINESS[kind]} met before T0", seen_ready,
           seen_ready == {"required": READINESS[kind], "met": True, "reason": None})
    if kind != "B":
        given = s5._get(result, "parameters", "shadow_digest")
        r.item("the guard was given the frozen candidate's digest", "equal" if given == want["shadow_digest"] else
               given if given is MISSING else "differs", given == want["shadow_digest"] and given is not None)
    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    captures = [s5._load(directory / f"status-{name}.json") for name in names]
    seen = [{"face": s5._get(c, "components", "face", "state"),
             "validation_run": s5._get(c, "components", "face", "validation_run"),
             "enrolled": s5._get(c, "components", "face", "identities_enrolled"),
             "stream": s5._get(c, "components", "capture", "stream", "profile"),
             "scene": s5._get(c, "components", "scene", "state"), "video": s5._get(c, "live", "video"),
             "channels": s5._get(c, "components", "notifications", "channels"),
             "delivery": s5._get(c, "delivery", "totals")} for c in captures]
    wanted_names = [name for _, name in mpe_schedule.STATUS_AT[kind]]
    r.item(f"status captures {', '.join(wanted_names)}: face available (validation run, 1 enrolled), stream "
           "main_single, scene disabled, video fresh, no channel, no delivery row", seen if seen else MISSING,
           names == wanted_names and bool(seen) and not s5._has_missing(seen)
           and all(s["face"] == "available" and s["validation_run"] is True and s["enrolled"] == 1
                   and s["stream"] == fc.STREAM_MAIN_SINGLE and s["scene"] == "disabled" and s["video"] == "fresh"
                   and s["channels"] == {} and isinstance(s["delivery"], dict) and not any(s["delivery"].values())
                   for s in seen))
    s5._listeners(r, result, rc.STATUS_PORTS)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and isinstance(stopped.get("stopped"), dict) and stopped["stopped"].get("face_worker") is True)
    output = s5._get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and isinstance(counts["lines"], int)
           and counts["lines"] >= 2)
    config = directory.parent / "face.yaml"
    sha = hashlib.sha256(rc.RANGE_CONFIG.encode()).hexdigest()
    same = config.is_file() and config.read_bytes() == rc.RANGE_CONFIG.encode()
    launched = _before_launch(directory.parent, f"{LAUNCH_PREFIX}_config_sha256={sha}")
    r.item("the session's face.yaml, as the launch block ran it (its SHA-256 recorded just before the launch line): route "
           "B's stream profile and the face adapter only, every identity and tracker setting the runtime's default",
           {"file": "as required" if same else "differs" if config.is_file() else MISSING,
            "recorded_before_launch": launched}, same and launched)

    records, problems, starting, _ = fc.read_records(directory / "run.jsonl", EXPERIMENT[kind])
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind_ in records.values() for item in kind_})
    if schemas and schemas != [fc.WINDOWS_SCHEMA]:
        problems.append(f"schema {schemas}, not route B's {fc.WINDOWS_SCHEMA}")
    problems.extend(fc._route_b_startup_problems(start, starting, True))
    if schemas == [fc.WINDOWS_SCHEMA]:
        problems.extend(fc.frame_problems(records, EXPERIMENT[kind]))
    last = captures[-1] if captures else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {fc.WINDOWS_SCHEMA}, complete, none dropped; route B's startup markers and every "
           f"frame reference consistent; the {EXPERIMENT[kind]} run's own records",
           {"records": {k: len(v) for k, v in records.items()}, "problems": problems[:5], "dropped": dropped}
           if total else MISSING,
           total > 0 and not problems and dropped == 0 and total < fc.MAX_IDENTITY_RECORDS
           and not records.get("comparison") and not records.get("capture"))
    if kind == "B":
        raw = (records.get("capture_raw") or [MISSING])[0]
        compared = _compared_one_person(records)
        r.item("the raw capture: written and read back, nothing refused, nothing dropped, every compared one-person "
               "face kept", {k: raw.get(k) for k in ("written", "verified", "refused", "faces_kept",
                                                      "faces_dropped_over_cap", "faces_dropped_other_boot")}
               | {"compared_one_person_results": compared} if isinstance(raw, dict) else MISSING,
               isinstance(raw, dict) and raw.get("written") is True and raw.get("verified") is True
               and raw.get("refused") is None and raw.get("faces_dropped_over_cap") == 0
               and raw.get("faces_dropped_other_boot") == 0 and raw.get("faces_kept") == compared)
    fc._route_b_items(r, records)
    cues = fc._cue_offsets(result)
    duration = s5._get(result, "parameters", "duration_s")
    schedule = {name: at for name, at, _ in mpe_schedule.SESSIONS[kind]}
    r.item("T0: the t0 cue printed when the runtime was ready", {"t0_s_after_ready": cues.get("t0", MISSING)},
           "t0" in cues and 0 <= cues["t0"] <= fc.CUE_TOLERANCE_S)
    late = {name: round(cues[name] - at, 3) for name, at in schedule.items()
            if name in cues and not 0 <= cues[name] - at <= fc.CUE_TOLERANCE_S}
    absent = sorted(set(schedule) - set(cues))
    extra = sorted(set(cues) - set(schedule))
    run_s = mpe_schedule.duration_s(mpe_schedule.SESSIONS[kind])
    r.item(f"the run's {run_s} s and all {len(schedule)} cues of the schedule, each printed within "
           f"{fc.CUE_TOLERANCE_S:g} s of its time", {"duration_s": duration, "late": late, "absent": absent,
                                                     "extra": extra},
           duration == run_s and not (late or absent or extra))
    marks, why = _marks(directory.parent)
    r.item("the marks, lens to face, typed and recorded before the launch (4.0 and 5.0 m within 0.1 m; the seat; "
           "tape or laser; an out-of-view spot)", {**marks, "recorded": why} if marks else MISSING if why.startswith("no ")
           else why, marks is not None and all(rc._mark_ok(marks, m) for m in ("seat", "mark4", "mark5")))
    if kind == "E_P":
        present = sorted(fc._startup_keys(directory / "run.jsonl") & set(fc.NOT_PICTURED))
        r.item("the participant's run: no picture, live view, chat, feature control or scene server in the runtime's "
               "starting line (D102)", present or "none", not present)
    fc._counts_items(r, directory)
    enrolled = fc._enrolled_id(f1)
    r.item("the enrolled identity, from F1's list", "one identity" if enrolled else MISSING, enrolled is not None)
    spoken_log = directory.parent / "spoken.jsonl"  # the session's, beside the cues log: the run's is the guard's
    if spoken_log.is_file():  # optional: described, never judged
        spoken = [json.loads(line) for line in spoken_log.read_text("ascii").splitlines() if line.strip()]
        r.describe("spoken cues (optional; the printed lines are the record)", mpe_speak.check(spoken, result))
    r.lines.append("mpe-execution: passed" if r.ok else "mpe-execution: not passed")
    return r


def _marks(session: Path) -> tuple[dict[str, Any] | None, str]:
    """range_check's marks (its file, keys, labels and tolerances), recorded before this package's own launch line
    (range_check's helper looks for V3's). Typed values outside the fixed labels are never printed."""
    path = session / rc.MARKS_FILE
    if not path.is_file() or not (session / "provenance.txt").is_file():
        return None, f"no {rc.MARKS_FILE} or provenance.txt in the session directory"
    data = path.read_bytes()
    values = fc.parse_account(data.decode("utf-8", "replace"))
    if set(values) != set(rc.MARKS_KEYS):
        return None, f"{rc.MARKS_FILE} keys differ from {list(rc.MARKS_KEYS)}"
    if not _before_launch(session, f"marks_sha256={hashlib.sha256(data).hexdigest()}"):
        return None, f"{rc.MARKS_FILE} is not the file recorded in the session's provenance before the launch"
    out: dict[str, Any] = {
        "measured_with": values["measured_with"] if values["measured_with"] in rc.MEASURED_WITH else "other",
        "out_of_view_spot": values["out_of_view_spot"] if values["out_of_view_spot"] in ("yes", "no") else "other"}
    for key in ("seat_m", "mark4_m", "mark5_m"):
        try:
            out[key] = float(values[key])
        except ValueError:
            return None, f"{key} is not a number"
    return out, "recorded before the launch"


def _before_launch(session: Path, wanted: str) -> bool:
    lines = rc._provenance_lines(session)
    launch = [i for i, line in enumerate(lines) if line.startswith(f"{LAUNCH_PREFIX}_launch_utc=")]
    return wanted in lines and len(launch) == 1 and lines.index(wanted) < launch[0]


def passed(directory: Path, f1: Path, kind: str, frozen: dict | None = None, gate=None) -> tuple[bool, list[str]]:
    reading = read_mpe_execution(directory, f1, kind, frozen, gate)
    return reading.ok, reading.lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--f1", type=Path, required=True)
    parser.add_argument("--session", required=True, choices=tuple(mpe_schedule.SESSIONS))
    parser.add_argument("--frozen", type=Path, default=None)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--identity-dir", type=Path)
    parser.add_argument("--recheck-root", type=Path)
    args = parser.parse_args(argv)
    frozen = json.loads(args.frozen.read_text("ascii")) if args.frozen else None
    gate = (args.run, args.identity_dir, args.recheck_root) if args.run else None
    reading = read_mpe_execution(args.run_dir, args.f1, args.session, frozen, gate)
    print("\n".join(reading.lines))
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
