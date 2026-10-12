#!/usr/bin/env python3
"""The V2-25 device check's predeclared readings (F1 consented enrollment, F2 guarded live validation).

    face_check.py f1 F1_DIR --identity-dir DIR
    face_check.py f2 F2_DIR --f1 F1_DIR [--run RUN_DIR --identity-dir DIR --recheck-root ROOT]
    face_check.py f1-recheck RECHECK_DIR --run RUN_DIR --identity-dir DIR
    face_check.py gate RUN_DIR --identity-dir DIR [--recheck-root ROOT]

``f1-recheck`` is a corrected F1 reading made afterwards (face_recheck.py writes its directory); the run's own
result stays as recorded. ``gate`` says whether F2 may build on a run's F1: its own F1 validated, or exactly one
corrected F1 reading of it that the maintainer explicitly accepted (face_recheck.py accept) and that still matches
the run, its manifest, the enrolled identity and the audit log, and still validates. The privacy counts come from
face_privacy.py (face_counts.sh) and must cover every evidence file with each of their keys, nonconforming included.

Each item prints PASS, FAIL, MISSING or NOT_EXERCISED, then ``face-<part>: validated`` or ``not validated``; exit
status 0 only when every item passed. Missing evidence never passes, and NOT_EXERCISED (the run did not test what the
item is about, e.g. retention while new matching evidence kept arriving) never passes either. DESCRIPTIVE lines are
reported and never judged. Nothing is written, and no secret, name or embedding is read or shown; the account's notes
are never printed.

**F2's time base.** T0 is the moment the guard printed its ``t0`` cue, which it does when the runtime's ``starting``
line appears, i.e. once every model is loaded and released (``result.json`` ``cues``, CLOCK_MONOTONIC like the
runtime's records). Judged windows come from the account, never from the schedule, with a washout of WASHOUT_S
(an assumption, not a measurement). The account's header names its format (face_account):

- **Format 1** (the face-0f7855e package and earlier), read exactly as it always was: the operator answered whether
  the stopwatch was started at the cue, and each reading is taken as the exact time of its movement after T0 (the
  washout also absorbed the stopwatch's start delay). Each movement is the interval [t, t].
- **Format 2** (D66; uncalibrated, for this bounded test only): the stopwatch was started d s after the cue, give
  or take v s, and each reading r may be off by u s (u and v whole seconds 1..5). The delay lies in
  [max(0, d - v), d + v]; the clamp at 0 assumes the stopwatch cannot be started before a cue that comes at
  readiness. A movement then lies in the interval [r - u + max(0, d - v), r + u + d + v] after T0. The account's
  times are aligned to T0 here, never by the operator. Item 15 needs the delay's upper end d + v <= DELAY_MAX_S.
- **Formats 3 and 4** (D79 cued; D85 the same, with the 4:00 head turn stated in the cue and the account): no
  stopwatch; each movement lies in [cue, cue + allowance] from the guard's own cue times, where the account says it
  was done within its allowance and every cue was printed on time. Format 4 is read exactly as format 3.

With lo and hi each movement's interval ends (lo = hi = t in format 1), every window starts W after the latest
time its opening movement could have happened and ends W before the earliest time its closing movement could have
happened, so that, if the account's bounds hold, every judged instant is at least W from every movement:

    E1 empty     hi(out_of_view_1) + W .. lo(seated_facing_1) - W
    K1 facing    hi(seated_facing_1) + W .. lo(turned_away) - W
    A  away      hi(turned_away) + W .. lo(stood_up) - W          (the same track, face turned away)
    E2 empty     hi(out_of_view_2) + W .. lo(seated_facing_2) - W
    K2 return    hi(seated_facing_2) + W .. duration - W

The bounds describe the operator's timing only. The camera-to-ingest delay is not measured or bounded by them, and
the washout remains an assumption for it and for the movements themselves.

The uncertainty enters every item that uses an account time, and only those:
- 16, 19-21, 25 and 26 and retention's continuity and window end: the windows above;
- 27: every identity known on an interval that starts before hi(out_of_view_2), the latest possible leaving time,
  must end before E2 starts;
- 28: an identity known in K2 must start at or after hi(out_of_view_2), and the votes that made it known must all
  have frame times at or after it;
- 29: exactly one known interval of the enrolled identity overlaps the turn interval [lo, hi](turned_away), and it
  covers all of it; otherwise retention is NOT_EXERCISED. Its timing is judged from vote frame times, unchanged;
  match votes are counted (descriptively) from hi(turned_away), and those inside the turn interval apart;
- the time to known after sitting down (descriptive) is reported from hi and from lo of the seated interval.

Ticks and votes are placed by their frame's time; a result applied after its frame (in flight across a boundary)
counts where its frame was. Whether a state was justified at a moment uses only results applied by that moment.
Identity rules follow ``sentinel.identity.state`` with the runtime's configured values (a test ties the constants
below to the runtime's defaults).

**Identity record schemas.** Schema 1 is read exactly as it always was. Schema 2 (D84: what face validation runs
write) is schema 1 plus four numbers per result person (runtime.FaceDiagnostics): the person's box and the
detector's confidence, the assigned face's size in frame pixels and the face detector's score, and the vote's
similarity to the closest enrolled identity. A run's records share one schema. Every item is judged the same way for
both; schema 2's numbers must agree with their labels (a face only where one was assigned, a similarity only where
one was computed, a match at or above MATCH_THRESHOLD and an unknown vote at or below it), and they add DESCRIPTIVE
diagnostics per judged window, which are never judged. Schema 3 (D86) adds each result's full-scale pass. Schema 4 (D87,
``--face-comparison`` runs) is schema 3 plus "comparison" records, each checked against its own disposition. Schema 5
(D88, the full-scale detection candidate) is schema 2 plus each result's own detection numbers. Schema 6 (D90, what
the candidate writes from D90 on) is schema 5 plus, on each person, the position of the prototype behind a computed
similarity, the assigned face's geometry (the corrected box, the detector's own unclipped box or null, the overhang
past each edge or null, the eyes' angle and spacing or null; runtime.face_geometry_numbers) and "shadow" (D91: the
similarity to an experiment's working gallery, or null); its numbers must fit their bounds and labels, and the
readings add one DESCRIPTIVE geometry line, never judged.

**F2 for the candidate (D89).** The F2 reading reads a run of the candidate by the same criteria: its startup's face
entry is production's plus ``detection: full_scale``, which must agree with the records' schema (5 or 6 with it, never
without it) and appear in every status capture; three item names say so, and nothing else differs.

**The comparison reading** (``face_check.py comparison``, D87; descriptive, validates nothing): execution items first
(the run, the records in schema 4, the cues, a locked format-6 account in which every answer is yes, no or unknown, the
privacy counts); then, once the account is locked, DESCRIPTIVE lines over window W (WASHOUT_S after T0 to WASHOUT_S
before the stop cue). Each line names the account answers it depends on and says whether they support it, leave it
QUALIFIED (unknown) or CONTRADICTED; an answer never suppresses a line that does not depend on it. A failed comparison
is counted as failed, never as a nonmatch; a similarity is compared with MATCH_THRESHOLD descriptively only.

**The candidate reading** (``face_check.py candidate``, D88; descriptive, validates nothing, F2's criteria untouched):
the same execution items for a run with the full-scale detection candidate selected (the startup line's face
``detection``, the records in schema 5: schema 2 plus each result's own detection numbers), then, only for such a run,
DESCRIPTIVE lines over the same W: production's work and time, monitoring and memory, detection and votes per person,
the schema-2 diagnostics and the identity time. **One denominator (D88's correction):** the per-person line counts the
person on each applied result whose frame had exactly one person, and counts results with more people apart, so
result-level numbers (D86's pass, the candidate's own) and person-level ones (the face, the vote) count the same people;
the natural and comparison readings gain the same line beside their older ones, whose denominators are unchanged and
named (people, or results).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Any

import face_account
import face_evidence
import step5_check as s5
from operator_account import parse as parse_account
from operator_account import seconds as stopwatch_seconds

SCHEMA = 1
DIAGNOSTIC_SCHEMA = 2  # runtime.IDENTITY_DIAGNOSTIC_SCHEMA: schema 1 plus each result person's diagnostic numbers
FULL_SCALE_SCHEMA = 3  # runtime.IDENTITY_FULL_SCALE_SCHEMA: schema 2 plus each result's full-scale pass (D86)
COMPARISON_SCHEMA = 4  # runtime.IDENTITY_COMPARISON_SCHEMA: schema 3 plus comparison records (D87)
CANDIDATE_SCHEMA = 5  # runtime.IDENTITY_CANDIDATE_SCHEMA: schema 2 plus the detection candidate's numbers (D88)
GEOMETRY_SCHEMA = 6  # runtime.IDENTITY_GEOMETRY_SCHEMA: schema 5 plus each person's prototype, geometry, shadow (D90)
SCHEMAS = (SCHEMA, DIAGNOSTIC_SCHEMA, FULL_SCALE_SCHEMA, COMPARISON_SCHEMA, CANDIDATE_SCHEMA, GEOMETRY_SCHEMA)
CANDIDATE_SCHEMAS = (CANDIDATE_SCHEMA, GEOMETRY_SCHEMA)  # what runs of the candidate write (D88; D90 from 6)
F2_FACE = {"validation_run": True, "identities_enrolled": 1}  # the startup's face entry of a production F2 run
FULL_SCALE_KEYS = {"faces", "assigned", "best_score", "best_w_px", "ms"}
DETECTION_KEYS = {"best_below_score", "best_below_w_px", "detect_ms", "embed_ms"}  # runtime.detection_numbers (D88)
DETECTION_FULL_SCALE = "full_scale"  # legacy_deepface.DETECTION_FULL_SCALE: the startup line's face "detection"
RECORDING_FLOOR = 0.3  # legacy_deepface.FULL_SCALE_FLOOR: what the full-scale passes report under 0.9
ID = re.compile(r"^idn-[0-9a-f]{12}$")
NS = 1_000_000_000
# The runtime's identity values (config.IdentityConfig defaults; test-tied).
VOTE_TTL_S = 30.0
FRESH_S = 3.0
CONFIRMATIONS = 2
MATCH_THRESHOLD = 0.70  # cosine similarity; schema 2's similarities are checked against it, never judged
# This check's proposed values (maintainer review, 2026-10-08).
WASHOUT_S = 15.0
MIN_WINDOW_S = 45.0
MIN_TICKS_EMPTY = 30
MIN_OFFERED = 30
MIN_APPLIED = 30
MIN_MATCHES_FACING = 10
MIN_CONTINUITY = 0.9  # share of the away window's offered ticks that list the retained track
MIN_COMPLETION = 0.95  # approved 2026-10-08 for this bounded check: completed / offered, with 0 processing errors
# F1's package minimum (maintainer, 2026-10-08): at least 3 usable photos. The enrollment API's own minimum
# (sentinel.identity.enroll.MIN_PHOTOS, 2) is separate and unchanged; this check is stricter.
MIN_USABLE_PHOTOS = 3
MAX_USABLE_PHOTOS = 8  # MAX_PROTOTYPES
STATE_TOLERANCE_S = 2.0  # a state change may lag its cause by the loop's step and a capture
EXACT_S = 0.01  # a state change never precedes its cause (rounding only)
CUE_TOLERANCE_S = 0.5
# D66 (maintainer, 2026-10-08), format-2 accounts only; uncalibrated, for this bounded test only: the stopwatch
# start delay's upper end (delay + its uncertainty). Each uncertainty is 1..5 s (face_account.UNCERTAINTY_S).
DELAY_MAX_S = 5
MAX_IDENTITY_RECORDS = 10_000  # the runtime's cap
FACE_RELEASES = {"facenet512_weights.h5": "returned_0", "face_detection_yunet_2023mar.onnx": "returned_0"}
SCENE_SCOPE = {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
               "workload_verified": {"runtime": 0, "scene_server": 1}}
RELEASES = {"scene": {"llm": "returned_0", "mmproj": "returned_0"}, "detector": {"engine": "returned_0"},
            "face": FACE_RELEASES}
TICK_KEYS = {"identity", "schema", "stream_epoch", "frame_seq", "frame_mono_ns", "tick", "tracks"}
RESULT_KEYS = {"identity", "schema", "stream_epoch", "frame_seq", "frame_mono_ns", "applied_mono_ns", "outcome",
               "faces", "processing_ms", "error", "persons"}
PERSON_KEYS = {"track_id", "ownership", "vote", "identity_id", "label"}
DIAGNOSTIC_PERSON_KEYS = PERSON_KEYS | {"box", "confidence", "face", "similarity"}  # schema 2
GEOMETRY_PERSON_KEYS = DIAGNOSTIC_PERSON_KEYS | {"prototype", "geometry", "shadow"}  # schema 6 (D90)
GEOMETRY_KEYS = {"box", "raw_box", "overhang_px", "eyes"}  # runtime.face_geometry_numbers
EYES_KEYS = {"roll_deg", "spacing_px"}
SHADOW_KEYS = {"similarity", "prototype"}  # D91: the similarity to an experiment's working gallery
RAW_BOUNDS = (-1.0, 2.0)  # runtime.RAW_BOUNDS: what a recorded raw box may span, in frame widths and heights
MAX_PROTOTYPES = 8  # matching.MAX_PROTOTYPES: a prototype position is 0 to 7
MAX_PX = 4096.0  # a generous bound on any recorded pixel distance (frames are at most 640 px wide on these paths)
FACE_KEYS = {"w_px", "h_px", "score"}
SIMILARITY_LABELS = {"match", "unknown", "margin"}  # the labels for which the runtime computed a similarity
TRANSITION_KEYS = {"identity", "schema", "stream_epoch", "track_id", "from", "to", "identity_id", "basis",
                   "last_vote_age_ms", "reason", "mono_ns"}
COMPARISON_KEYS = {"identity", "schema", "stream_epoch", "frame_seq", "frame_mono_ns", "track_id", "kind",
                   "disposition", "score", "w_px", "similarity", "ms", "error"}  # runtime.ComparisonRecord (D87)
CAPTURE_KEYS = {"identity", "schema", "written", "refused", "faces_kept", "faces_dropped_over_cap", "faces_in_span",
                "prototypes", "min_best_to_others", "similarity_to_original", "verified"}  # D91, Capture.finish
CAPTURE_REFUSALS = ("too_few_faces", "too_few_prototypes", "inconsistent_capture", "capture_target_has_gallery",
                    "capture_not_verified", "capture_write_failed")  # identity.experiment.CAPTURE_REFUSALS
CAPTURE_MIN_PROTOTYPES = 4  # identity.experiment.CAPTURE_MIN_PROTOTYPES
COMPARISON_KINDS = ("full_scale_only", "both")
ATTEMPTED = ("completed", "failed", "production_not_applied")  # dispositions with a time; the rest were skipped
COMPARISON_DISPOSITIONS = (*ATTEMPTED, "skipped_one_per_result", "skipped_stopping", "skipped_production_waiting",
                           "skipped_no_landmarks", "skipped_stale", "skipped_cap", "skipped_interval", "skipped_quota")
ERROR_LABEL = re.compile(r"^[a-z_]+(:[A-Za-z0-9_]+)?$")
TICKS = {"offered", "skipped_no_person", "stopping"}
OUTCOMES = {"applied", "failed", "rejected_not_live", "rejected_boot", "rejected_age", "ignored_face_unavailable"}
STATES = {"unresolved", "unknown", "known", "uncertain", "cleared"}
MISSING = s5.MISSING
WINDOWS = ("E1", "K1", "A", "E2", "K2")


class Reading(s5.Reading):
    def not_exercised(self, name: str, value: Any) -> None:
        self.ok = False
        self.lines.append(f"NOT_EXERCISED {name}: {s5._text(value)}")


def _int(value: Any) -> bool:
    return type(value) is int


# ---------------------------------------------------------------- F2 records


def read_records(path: Path) -> tuple[dict[str, list[dict]], list[str], dict | None, dict | None]:
    """(records by kind, schema problems, starting line, stopped line) from the runtime's stdout file."""
    records: dict[str, list[dict]] = {"tick": [], "result": [], "transition": []}  # "comparison" only when present
    problems: list[str] = []
    starting = stopped = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return records, ["run.jsonl unreadable"], None, None
    for number, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except ValueError:
            continue  # D48 is judged separately
        if not isinstance(item, dict):
            continue
        if item.get("run") == "starting" and starting is None:
            starting = item.get("startup")
        elif item.get("run") == "stopped":
            stopped = item
        kind = item.get("identity")
        if not isinstance(kind, str):
            continue  # not an identity record (e.g. a status line)
        problem = _record_problem(kind, item)
        if problem is not None:
            problems.append(f"line {number}: {problem}")
        else:
            records.setdefault(kind, []).append(item)
    return records, problems, starting, stopped


def _record_problem(kind: Any, item: dict) -> str | None:
    keys = {"tick": TICK_KEYS, "result": RESULT_KEYS, "transition": TRANSITION_KEYS,
            "comparison": COMPARISON_KEYS, "capture": CAPTURE_KEYS}.get(kind)
    if keys is None:
        return f"unknown identity record {kind!r}"
    schema = item.get("schema")
    if schema not in SCHEMAS or type(schema) is not int:
        return f"{kind}: schema {schema!r}, not one of {list(SCHEMAS)}"
    if kind == "comparison":
        return _comparison_problem(item)
    if kind == "capture":
        return _capture_problem(item)
    if kind == "result" and schema in (FULL_SCALE_SCHEMA, COMPARISON_SCHEMA):
        keys = keys | {"full_scale"}
    elif kind == "result" and schema in CANDIDATE_SCHEMAS:
        keys = keys | {"detection"}
    if set(item) != keys:
        return f"{kind}: fields {sorted(set(item) ^ keys)} differ from schema {schema}"
    if kind == "tick":
        if not (_int(item["frame_mono_ns"]) and item["tick"] in TICKS and isinstance(item["tracks"], list)
                and all(_int(t) for t in item["tracks"])):
            return "tick: a field has the wrong type or value"
    elif kind == "result":
        if not (_int(item["frame_mono_ns"]) and _int(item["applied_mono_ns"]) and item["outcome"] in OUTCOMES
                and isinstance(item["persons"], list)):
            return "result: a field has the wrong type or value"
        if schema in (FULL_SCALE_SCHEMA, COMPARISON_SCHEMA) and not _full_scale_fits(item["full_scale"]):
            return "result: its full-scale numbers are malformed"
        if schema in CANDIDATE_SCHEMAS and not _detection_fits(item["detection"], item["faces"], item["outcome"]):
            return "result: its detection numbers are malformed"
        person_keys = (PERSON_KEYS if schema == SCHEMA else GEOMETRY_PERSON_KEYS if schema == GEOMETRY_SCHEMA
                       else DIAGNOSTIC_PERSON_KEYS)
        for person in item["persons"]:
            if not isinstance(person, dict) or set(person) != person_keys or not _int(person["track_id"]) \
                    or person["vote"] not in ("match", "unknown", "none") \
                    or (person["vote"] == "match") != (person["identity_id"] is not None) \
                    or (person["identity_id"] is not None and not ID.match(str(person["identity_id"]))):
                return f"result: a person entry differs from schema {schema}"
            if schema >= DIAGNOSTIC_SCHEMA and not _diagnostics_fit(person):
                return "result: a person entry's diagnostic numbers do not fit its labels"
            if schema == GEOMETRY_SCHEMA and not _geometry_fits(person):
                return "result: a person entry's geometry, prototype or shadow numbers do not fit"
    else:
        if not (_int(item["mono_ns"]) and _int(item["track_id"]) and item["from"] in STATES and item["to"] in STATES
                and (item["identity_id"] is None or ID.match(str(item["identity_id"])))
                and ((item["to"] == "known") == (item["identity_id"] is not None))):
            return "transition: a field has the wrong type or value"
    return None


def _comparison_problem(item: dict) -> str | None:
    """D87: a comparison record (schema 4 only) whose fields fit its disposition: a similarity exactly when completed,
    a time exactly when attempted, a fixed error label exactly when failed, a candidate counted at 0.9 or above."""
    if item["schema"] != COMPARISON_SCHEMA:
        return f"comparison: schema {item['schema']!r}; comparison records are schema {COMPARISON_SCHEMA}"
    if set(item) != COMPARISON_KEYS:
        return f"comparison: fields {sorted(set(item) ^ COMPARISON_KEYS)} differ from schema {COMPARISON_SCHEMA}"
    disposition = item["disposition"]
    if not (all(_int(item[k]) for k in ("stream_epoch", "frame_seq", "frame_mono_ns", "track_id", "w_px"))
            and item["w_px"] >= 0 and item["kind"] in COMPARISON_KINDS and disposition in COMPARISON_DISPOSITIONS
            and _number(item["score"], 0.9, 1.0)):
        return "comparison: a field has the wrong type or value"
    completed, attempted, failed = disposition == "completed", disposition in ATTEMPTED, disposition == "failed"
    if (item["similarity"] is not None) != completed or (completed and not _number(item["similarity"], -1.0, 1.0)):
        return "comparison: a similarity exactly when completed"
    if (item["ms"] is not None) != attempted or (attempted and not _number(item["ms"], 0.0, math.inf)):
        return "comparison: a time exactly when attempted"
    if (item["error"] is not None) != failed or (failed and not (isinstance(item["error"], str)
                                                                 and ERROR_LABEL.match(item["error"]))):
        return "comparison: a fixed error label exactly when failed"
    return None


def _capture_problem(item: dict) -> str | None:
    """D91: the capture's record (schema 6 only): written exactly when nothing refused it, verified only when written,
    a fixed refusal label, counts that fit together, and one similarity to the original per prototype (or none)."""
    if item["schema"] != GEOMETRY_SCHEMA:
        return f"capture: schema {item['schema']!r}; the capture's record is schema {GEOMETRY_SCHEMA}"
    if set(item) != CAPTURE_KEYS:
        return f"capture: fields {sorted(set(item) ^ CAPTURE_KEYS)} differ from schema {GEOMETRY_SCHEMA}"
    counts = ("faces_kept", "faces_dropped_over_cap", "faces_in_span", "prototypes")
    similarities = item["similarity_to_original"]
    if not (all(_int(item[k]) and item[k] >= 0 for k in counts) and item["faces_in_span"] <= item["faces_kept"]
            and item["prototypes"] <= MAX_PROTOTYPES and type(item["written"]) is bool
            and type(item["verified"]) is bool and (item["refused"] is None or item["refused"] in CAPTURE_REFUSALS)
            and (item["min_best_to_others"] is None or _number(item["min_best_to_others"], -1.0, 1.0))
            and isinstance(similarities, list) and all(_number(v, -1.0, 1.0) for v in similarities)
            and len(similarities) in (0, item["prototypes"])):
        return "capture: a field has the wrong type or value"
    if item["written"] != (item["refused"] is None) or (item["verified"] and not item["written"]):
        return "capture: written exactly when nothing refused it, and verified only when written"
    return None


def _number(value: Any, low: float, high: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _full_scale_fits(numbers: Any) -> bool:
    """Schema 3's full-scale pass of one result (absent: null), or counts that fit together."""
    if numbers is None:
        return True
    if not (isinstance(numbers, dict) and set(numbers) == FULL_SCALE_KEYS and _int(numbers["faces"])
            and _int(numbers["assigned"]) and 0 <= numbers["assigned"] <= numbers["faces"]
            and _number(numbers["ms"], 0.0, math.inf)):
        return False
    best, width = numbers["best_score"], numbers["best_w_px"]
    if best is None:
        return width is None and numbers["faces"] == 0
    return _number(best, 0.0, 1.0) and _int(width) and width >= 0 and (numbers["faces"] == 0 or best >= 0.9)


def _detection_fits(numbers: Any, faces: Any, outcome: Any) -> bool:
    """Schema 5's candidate numbers of one result (D88): absent exactly when the analysis failed; otherwise the best
    detection under 0.9 (at or above the recording floor; rounding can only reach 0.9 itself) with its width, or
    neither, the detection's time, and an embedding time whenever a face was kept."""
    if numbers is None:
        return outcome == "failed" or faces is None
    if not (isinstance(numbers, dict) and set(numbers) == DETECTION_KEYS and _number(numbers["detect_ms"], 0.0, math.inf)
            and (numbers["embed_ms"] is None or _number(numbers["embed_ms"], 0.0, math.inf))):
        return False
    if _int(faces) and faces > 0 and numbers["embed_ms"] is None:
        return False
    best, width = numbers["best_below_score"], numbers["best_below_w_px"]
    if best is None:
        return width is None
    return _number(best, RECORDING_FLOOR, 0.9) and _int(width) and width >= 0


def _geometry_fits(person: dict) -> bool:
    """Schema 6 (D90): a prototype position exactly when a similarity was computed; the geometry exactly when a face
    was assigned: the corrected box inside the frame, the detector's raw box within RAW_BOUNDS or null, the overhang
    (four pixel distances, 0 or more) exactly when the raw box is there, the eyes' angle and spacing or null; the
    shadow (D91) null, or a similarity and a prototype position only where a similarity was computed."""
    prototype, geometry, shadow = person["prototype"], person["geometry"], person["shadow"]
    computed = person["similarity"] is not None
    if (prototype is not None) != computed or (computed and not (_int(prototype) and 0 <= prototype < MAX_PROTOTYPES)):
        return False
    if (geometry is None) != (person["face"] is None):
        return False
    if geometry is not None:
        if not (isinstance(geometry, dict) and set(geometry) == GEOMETRY_KEYS and _box_fits(geometry["box"], 0.0, 1.0)):
            return False
        raw, overhang, eyes = geometry["raw_box"], geometry["overhang_px"], geometry["eyes"]
        if (raw is None) != (overhang is None):
            return False
        if raw is not None and not (_box_fits(raw, *RAW_BOUNDS) and isinstance(overhang, list) and len(overhang) == 4
                                    and all(_number(v, 0.0, MAX_PX) for v in overhang)):
            return False
        if eyes is not None and not (isinstance(eyes, dict) and set(eyes) == EYES_KEYS
                                     and _number(eyes["roll_deg"], -180.0, 180.0)
                                     and _number(eyes["spacing_px"], 0.0, MAX_PX)):
            return False
    if shadow is not None and not (computed and isinstance(shadow, dict) and set(shadow) == SHADOW_KEYS
                                   and _number(shadow["similarity"], -1.0, 1.0) and _int(shadow["prototype"])
                                   and 0 <= shadow["prototype"] < MAX_PROTOTYPES):
        return False
    return True


def _box_fits(box: Any, low: float, high: float) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(_number(v, low, high) for v in box)
            and box[0] < box[2] and box[1] < box[3])


def _diagnostics_fit(person: dict) -> bool:
    """Schema 2's numbers on one result person, against that person's own labels: a box inside the frame and the
    detector's confidence; a face (pixels and score) exactly when one was assigned; a similarity exactly when the
    runtime computed one, on the threshold's side its label says (rounding can only reach the threshold itself)."""
    box, face, similarity = person["box"], person["face"], person["similarity"]
    if not (isinstance(box, list) and len(box) == 4 and all(_number(v, 0.0, 1.0) for v in box)
            and box[0] < box[2] and box[1] < box[3] and _number(person["confidence"], 0.0, 1.0)):
        return False
    if (face is None) != (person["ownership"] != "assigned"):
        return False
    if face is not None and not (isinstance(face, dict) and set(face) == FACE_KEYS and _int(face["w_px"])
                                 and _int(face["h_px"]) and face["w_px"] >= 0 and face["h_px"] >= 0
                                 and _number(face["score"], 0.0, 1.0)):
        return False
    if (similarity is None) != (person["label"] not in SIMILARITY_LABELS):
        return False
    if similarity is None:
        return True
    if not _number(similarity, -1.0, 1.0):
        return False
    if person["label"] == "unknown":
        return similarity <= MATCH_THRESHOLD  # below it, or rounded up to it
    return similarity >= MATCH_THRESHOLD  # a match, or a margin refusal (decided only at or above it)


class Timeline:
    """Seconds after T0, per (epoch, track): applied votes and KNOWN intervals."""

    def __init__(self, records: dict[str, list[dict]], t0_ns: int, end_s: float) -> None:
        self.t0_ns = t0_ns
        self.end_s = end_s
        self.ticks = [dict(r, t=self.rel(r["frame_mono_ns"])) for r in records["tick"]]
        self.results = [dict(r, t=self.rel(r["frame_mono_ns"]), applied=self.rel(r["applied_mono_ns"]))
                        for r in records["result"]]
        self.transitions = sorted((dict(r, t=self.rel(r["mono_ns"])) for r in records["transition"]),
                                  key=lambda r: r["t"])
        self.votes: dict[tuple[int, int], list[dict]] = {}
        for result in self.results:
            if result["outcome"] != "applied":
                continue
            for person in result["persons"]:
                if person["vote"] in ("match", "unknown"):
                    key = (result["stream_epoch"], person["track_id"])
                    self.votes.setdefault(key, []).append(
                        {"t": result["t"], "applied": result["applied"], "vote": person["vote"],
                         "identity_id": person["identity_id"]})
        for votes in self.votes.values():
            votes.sort(key=lambda v: v["t"])
        self.known: dict[tuple[int, int], list[dict]] = {}
        open_: dict[tuple[int, int], dict] = {}
        for transition in self.transitions:
            key = (transition["stream_epoch"], transition["track_id"])
            if transition["to"] == "known":
                if key not in open_ or open_[key]["identity_id"] != transition["identity_id"]:
                    if key in open_:
                        open_[key]["end"] = transition["t"]
                    open_[key] = {"start": transition["t"], "end": None, "identity_id": transition["identity_id"],
                                  "basis_changes": [(transition["t"], transition["basis"])],
                                  "end_to": None, "end_reason": None}
                    self.known.setdefault(key, []).append(open_[key])
                else:
                    open_[key]["basis_changes"].append((transition["t"], transition["basis"]))
            elif key in open_:
                interval = open_.pop(key)
                interval.update(end=transition["t"], end_to=transition["to"], end_reason=transition["reason"])
        for interval in open_.values():
            interval["end"] = None  # still KNOWN when the records end

    def rel(self, ns: int) -> float:
        return (ns - self.t0_ns) / NS

    def in_window(self, items: list[dict], window: tuple[float, float]) -> list[dict]:
        return [item for item in items if window[0] <= item["t"] < window[1]]

    def known_overlapping(self, window: tuple[float, float]) -> list[tuple[tuple[int, int], dict]]:
        out = []
        for key, intervals in self.known.items():
            for interval in intervals:
                end = self.end_s if interval["end"] is None else interval["end"]
                if interval["start"] < window[1] and end > window[0]:
                    out.append((key, interval))
        return out

    def known_at(self, key: tuple[int, int], t: float) -> dict | None:
        for interval in self.known.get(key, ()):
            end = self.end_s if interval["end"] is None else interval["end"]
            if interval["start"] <= t < end:
                return interval
        return None

    def votes_known_by(self, key: tuple[int, int], t: float) -> list[dict]:
        """Applied votes on the track that the runtime had at time t, in frame order, still within the TTL."""
        return [v for v in self.votes.get(key, ()) if v["applied"] <= t + 1e-6 and t - v["t"] < VOTE_TTL_S]

    def justified(self, key: tuple[int, int], t: float, identity_id: str) -> bool:
        votes = self.votes_known_by(key, t)
        return len(votes) >= CONFIRMATIONS and all(
            v["vote"] == "match" and v["identity_id"] == identity_id for v in votes[-CONFIRMATIONS:])


# ---------------------------------------------------------------- F2 reading


def _account(directory: Path) -> tuple[dict[str, str] | None, str | None, int | None]:
    """(the values, a problem, the format: 1, 2 or None for an unrecognised header)."""
    path = directory / face_account.ACCOUNT
    if not path.is_file():
        return None, "no account.txt", None
    data = path.read_bytes()
    locked = face_account.lock_state(directory)
    if locked is None:
        return None, "the account is not locked (no account_sha256 in provenance.txt)", None
    if hashlib.sha256(data).hexdigest() != locked:
        return None, "account.txt differs from its recorded SHA-256", None
    text = data.decode("utf-8", "replace")
    known = face_account.account_format(text)
    return parse_account(text), None, None if known is None else known[0]


def _delay(account: dict[str, str]) -> tuple[int, int] | None:
    """Format 2: the stopwatch start delay's interval [max(0, d - v), d + v], or None if either value is unknown."""
    d = face_account.delay_seconds(account.get(face_account.DELAY))
    v = face_account.uncertainty_seconds(account.get(face_account.DELAY_UNCERTAINTY))
    return None if d is None or v is None else (max(0, d - v), d + v)


def _cue_offsets(result: Any) -> dict[str, float]:
    """Each cue the guard printed, in seconds after readiness (T0), on CLOCK_MONOTONIC; only well-formed ones."""
    ready = s5._get(result, "child", "ready_mono_s")
    offsets = {}
    for cue in s5._list(s5._get(result, "cues")):
        if isinstance(cue, dict) and isinstance(cue.get("name"), str) and isinstance(cue.get("printed_mono_s"), float) \
                and isinstance(ready, float):
            offsets[cue["name"]] = cue["printed_mono_s"] - ready
    return offsets


def _cues_on_time(offsets: dict[str, float]) -> bool:
    """Format 3: every movement cue printed within CUE_TOLERANCE_S after its offset (D79)."""
    return all(name in offsets and 0 <= offsets[name] - at <= CUE_TOLERANCE_S
               for name, at in face_account.CUE_AT_S.items())


def _bounds(account: dict[str, str], fmt: int, cues: dict[str, float] | None = None) -> dict[str, tuple[float, float]] | None:
    """Each movement's interval (lo, hi) in seconds after T0; None if any reading, uncertainty or the delay is
    unknown or malformed. Format 1: [t, t], the reading taken as exact, as it always was. Formats 3 and 4: [cue, cue +
    allowance] from the guard's own cue times, only where the account says the movement was done within its allowance
    after its cue (yes), and only when every cue was printed on time."""
    if fmt in face_account.CUED_FORMATS:
        if cues is None or not _cues_on_time(cues):
            return None
        bounds3 = {}
        for key in face_account.TIMES:
            if account.get(face_account.within_key(key)) != "yes":
                return None
            at = cues[face_account.CUE_OF[key]]
            bounds3[key] = (at, at + face_account.ALLOWANCE_S[key])
        return bounds3
    if fmt == 1:
        s = {key: stopwatch_seconds(account.get(key)) for key in face_account.TIMES}
        return None if any(value is None for value in s.values()) else {k: (v, v) for k, v in s.items()}
    delay = _delay(account)
    if delay is None:
        return None
    bounds = {}
    for key in face_account.TIMES:
        reading = stopwatch_seconds(account.get(key))
        u = face_account.uncertainty_seconds(account.get(face_account.uncertainty_key(key)))
        if reading is None or u is None:
            return None
        bounds[key] = (reading - u + delay[0], reading + u + delay[1])
    return bounds


def _windows(bounds: dict[str, tuple[int, int]], duration_s: float) -> dict[str, tuple[float, float]]:
    """Each window from the far edges of its movements' intervals (see the time base above)."""
    w = WASHOUT_S
    lo = {key: b[0] for key, b in bounds.items()}
    hi = {key: b[1] for key, b in bounds.items()}
    return {"E1": (hi["out_of_view_1"] + w, lo["seated_facing_1"] - w),
            "K1": (hi["seated_facing_1"] + w, lo["turned_away"] - w),
            "A": (hi["turned_away"] + w, lo["stood_up"] - w),
            "E2": (hi["out_of_view_2"] + w, lo["seated_facing_2"] - w),
            "K2": (hi["seated_facing_2"] + w, duration_s - w)}


def _enrolled_id(f1: Path) -> str | None:
    listing = s5._load(f1 / "list.json")
    identities = s5._get(listing, "identities")
    if not isinstance(identities, list) or len(identities) != 1:
        return None
    identity_id = identities[0].get("identity_id") if isinstance(identities[0], dict) else None
    return identity_id if isinstance(identity_id, str) and ID.match(identity_id) else None


def _counts(directory: Path, name: str) -> Any:
    """``<path> <key>=<n>`` lines of a counts file, or MISSING."""
    path = directory / name
    if not path.is_file():
        return MISSING
    out = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9._/-]+) ([a-z_]+)=(\d+)", line)
        if match is None:
            return MISSING
        out[f"{match[1]} {match[2]}"] = int(match[3])
    return out or MISSING


def read_f2(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """``gate`` (run directory, identity directory, recheck root): F2 also needs that run's F1 basis (f1_gate)."""
    r = Reading()
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this F2: the run's own F1 validated, or the maintainer's accepted corrected F1 reading of "
               "it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("memory policy selected: THP workload_disabled, release post_load", s5._get(start, "memory_policy"),
           s5._get(start, "memory_policy") == s5.CANDIDATE_POLICY)
    r.item("THP scope at each startup checkpoint", s5._get(start, "thp_scope"), s5._get(start, "thp_scope") == SCENE_SCOPE)
    disable = s5._get(start, "thp_disable")
    r.item("THP disable verified in the runtime process", disable,
           isinstance(disable, dict) and disable.get("verified") is True and disable.get("reason") is None)
    r.item("post-load releases, the face weights' included: every call returned 0", s5._get(start, "releases"),
           s5._get(start, "releases") == RELEASES)
    server = s5._get(start, "scene_server")
    r.item("scene server ready, 17/17 layers, vision encoder on GPU; scene and detector admitted",
           {"scene_server": server, "scene_problem": s5._get(start, "scene_problem"),
            "detector_problem": s5._get(start, "detector_problem")},
           isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
           and server.get("layers") == s5.LAYERS and server.get("vision_on_gpu") is True
           and s5._get(start, "scene_problem") is None and s5._get(start, "detector_problem") is None)
    # D89: a run of the full-scale detection candidate (D88) is read by the same criteria; only its startup marker,
    # the marker in each capture and its records' schema say which detection produced the faces. Every other run is
    # read exactly as before.
    candidate = s5._get(start, "face") == CANDIDATE_FACE
    r.item("face admitted for this validation run: one enrolled identity, no problem" if not candidate else
           "face admitted for this validation run with the full-scale detection candidate (D88): one enrolled "
           "identity, no problem",
           {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") in (F2_FACE, CANDIDATE_FACE) and s5._get(start, "face_problem") is None)
    r.item("no notification channel (Telegram disabled)", s5._get(start, "notifier_problems"),
           s5._get(start, "notifier_problems") == {})

    names = [c.get("name") for c in s5._list(s5._get(result, "captures")) if isinstance(c, dict)]
    captures = [s5._load(directory / f"status-{name}.json") for name in names]  # in the guard's time order
    seen = [{"face": s5._get(c, "components", "face", "state"),
             "validation_run": s5._get(c, "components", "face", "validation_run"),
             "enrolled": s5._get(c, "components", "face", "identities_enrolled"),
             **({"detection": s5._get(c, "components", "face", "detection")} if candidate else {}),
             "scene": s5._get(c, "components", "scene", "state"), "video": s5._get(c, "live", "video"),
             "channels": s5._get(c, "components", "notifications", "channels"),
             "delivery": s5._get(c, "delivery", "totals")} for c in captures]
    r.item("each status capture: face available (validation run, 1 enrolled), scene available, video fresh, "
           "no channel, no delivery row" + ("; the candidate's detection" if candidate else ""),
           seen if seen else MISSING,
           bool(seen) and not s5._has_missing(seen) and all(s["face"] == "available" and s["validation_run"] is True and s["enrolled"] == 1
                              and s["scene"] == "available" and s["video"] == "fresh" and s["channels"] == {}
                              and isinstance(s["delivery"], dict) and not any(s["delivery"].values())
                              and (not candidate or s["detection"] == DETECTION_FULL_SCALE) for s in seen))
    s5._listeners(r, result, {18081, 18090})
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    output = s5._get(result, "run_output")
    counts = {k: output.get(k) for k in ("lines", "non_json_lines")} if isinstance(output, dict) else MISSING
    r.item("D48: every stdout line is JSON", counts,
           isinstance(counts, dict) and counts["non_json_lines"] == 0 and (counts["lines"] or 0) >= 2)

    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    if len(schemas) > 1:
        problems.append(f"the records mix schemas {schemas}")
    schema = schemas[0] if len(schemas) == 1 else SCHEMA
    if schemas and candidate != (schema in CANDIDATE_SCHEMAS):  # D89: the marker and the records must agree
        problems.append(f"schema {schema} records with the startup's face {s5._get(start, 'face')}")
    last = captures[-1] if captures else MISSING
    dropped = s5._get(last, "components", "face", "records", "dropped")
    r.item(f"identity records: schema {schema}, complete, none dropped",
           {"records": {k: len(v) for k, v in records.items()}, "problems": problems[:5], "dropped": dropped}
           if total else MISSING,
           total > 0 and not problems and dropped == 0 and total < MAX_IDENTITY_RECORDS)

    cue = next((c for c in s5._list(s5._get(result, "cues")) if isinstance(c, dict) and c.get("name") == "t0"),
               MISSING)
    ready = s5._get(result, "child", "ready_mono_s")
    printed = s5._get(cue, "printed_mono_s")
    r.item("T0: the t0 cue printed when the runtime was ready", {"printed_mono_s": printed, "ready_mono_s": ready},
           isinstance(printed, float) and isinstance(ready, float) and 0 <= printed - ready <= CUE_TOLERANCE_S)
    account, problem, fmt = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked",
           account is not None)
    if account is None:
        return r
    if fmt is None:
        r.item("account complete and in order", "the account's header names none of formats 1 to 4", False)
        return r
    fields = {1: face_account.FIELDS_V1, 2: face_account.FIELDS, 3: face_account.FIELDS_V3,
              4: face_account.FIELDS_V4}.get(fmt)
    if fields is None:
        r.item("account complete and in order", f"format {fmt} is not an F2 account", False)
        return r
    review = face_account.review(account, fields)
    r.item("account complete and in order", review, review[-1] == "Complete and in order: yes")
    cues = _cue_offsets(result) if fmt in face_account.CUED_FORMATS else None
    if fmt == 1:
        declared = {k: account.get(k) for k in ("stopwatch_started_at_t0", "stayed_until_stop", "other_faces_in_view")}
        r.item("stopwatch started at the T0 cue; stayed until the stop; no other face in view", declared,
               declared == {"stopwatch_started_at_t0": "yes", "stayed_until_stop": "yes", "other_faces_in_view": "no"})
    elif fmt == 2:
        _declared(r, account)
    else:
        _declared_cued(r, account, cues or {})
    duration = s5._get(result, "parameters", "duration_s")
    bounds = _bounds(account, fmt, cues)
    windows = _windows(bounds, duration) if bounds is not None and isinstance(duration, (int, float)) else None
    if fmt in face_account.CUED_FORMATS and bounds is not None:
        r.describe("account movements from the guard's cues: the cue (s after T0) and the interval its allowance "
                   "allows", {k: [face_account.CUE_OF[k], round(bounds[k][0], 2), round(bounds[k][1], 2)]
                              for k in face_account.TIMES})
    if fmt == 2 and bounds is not None:
        r.describe("account times aligned to T0: stopwatch reading, its uncertainty (s), the interval it allows "
                   "(s after T0)", {"stopwatch_start_delay_s": list(_delay(account)),
                                    **{k: [account[k], int(account[face_account.uncertainty_key(k)]), *bounds[k]]
                                       for k in face_account.TIMES}})
    name = (f"judged windows from the account (washout {WASHOUT_S:g} s), each at least {MIN_WINDOW_S:g} s" if fmt == 1
            else f"judged windows from the account's intervals (washout {WASHOUT_S:g} s after the latest and before "
                 f"the earliest possible time of each movement), each at least {MIN_WINDOW_S:g} s")
    r.item(name, {k: [round(a, 1), round(b, 1)] for k, (a, b) in windows.items()} if windows else MISSING,
           windows is not None and all(b - a >= MIN_WINDOW_S for a, b in windows.values()))
    enrolled = _enrolled_id(f1)
    r.item("the enrolled identity, from F1's list", enrolled if enrolled else MISSING, enrolled is not None)
    if windows is None or enrolled is None or not isinstance(printed, float) or problems:
        return r
    t0_ns = round(printed * NS)
    line = Timeline(records, t0_ns, float(duration))
    _completion(r, line)
    _sampling(r, line, windows, enrolled)
    _identity(r, line, windows, enrolled, account, bounds, fmt)
    _counts_items(r, directory)
    _describe(r, result, line, windows, last, bounds, fmt)
    if schema >= DIAGNOSTIC_SCHEMA:
        _diagnose(r, line, windows, full_scale=schema == FULL_SCALE_SCHEMA, geometry=schema == GEOMETRY_SCHEMA)
    return r



NATURAL_WINDOWS = ("N1", "LOOK", "TURN", "N2")  # between the natural check's cues t0, look, turn, n2 and the stop


def _natural_windows(cues: dict[str, float], duration: float) -> dict[str, tuple[float, float]]:
    """Each phase from its cue's printed time + the movement's allowance + WASHOUT_S, to WASHOUT_S before the next
    cue (the stop: the run's duration). At T0 the operator is already seated, so N1 has no allowance."""
    order = list(face_account.NATURAL_CUE_AT_S)
    out = {}
    for index, (name, cue) in enumerate(zip(NATURAL_WINDOWS, order)):
        start = cues[cue] + (0 if cue == "t0" else face_account.NATURAL_ALLOWANCE_S) + WASHOUT_S
        end = (cues[order[index + 1]] if index + 1 < len(order) else duration) - WASHOUT_S
        out[name] = (start, end)
    return out


def read_natural(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """D86, the natural seated check: execution items (PASS, FAIL or MISSING) establish that the run and the account
    are usable, then DESCRIPTIVE diagnostics per phase. It validates nothing and judges no recognition."""
    r = Reading()
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("face admitted for this validation run: one enrolled identity, no problem",
           {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") == {"validation_run": True, "identities_enrolled": 1}
           and s5._get(start, "face_problem") is None)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    saved = [s5._load(directory / f"status-{c.get('name')}.json") for c in s5._list(s5._get(result, "captures"))
             if isinstance(c, dict) and (directory / f"status-{c.get('name')}.json").is_file()]
    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    dropped = s5._get(saved[-1], "components", "face", "records", "dropped") if saved else MISSING
    r.item(f"identity records: schema {FULL_SCALE_SCHEMA}, complete, none dropped",
           {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped} if total else MISSING,
           total > 0 and not problems and schemas == [FULL_SCALE_SCHEMA] and dropped == 0
           and total < MAX_IDENTITY_RECORDS)
    ready = s5._get(result, "child", "ready_mono_s")
    cues = _cue_offsets(result)
    printed = (cues.get("t0") + ready) if isinstance(ready, float) and "t0" in cues else None
    on_time = all(name in cues and 0 <= cues[name] - at <= CUE_TOLERANCE_S
                  for name, at in face_account.NATURAL_CUE_AT_S.items())
    account, problem, fmt = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    if account is None:
        return r
    if fmt != 5:
        r.item("account complete and in order", f"format {fmt} is not the natural check's account", False)
        return r
    review = face_account.review(account, face_account.FIELDS_V5)
    r.item("account complete and in order", review, review[-1] == "Complete and in order: yes")
    declared = {k: account.get(k) for k in (*(f.key for f in face_account.FIELDS_V5[:4]), "other_faces_in_view")}
    as_cued = (on_time and all(declared[f.key] == "yes" for f in face_account.FIELDS_V5[:4])
               and declared["other_faces_in_view"] == "no")
    r.item(f"every cue printed within {CUE_TOLERANCE_S:g} s of its time; each phase done as cued; no other face "
           "in view", {"cues_s_after_t0": {k: round(v, 2) for k, v in cues.items()}, **declared}, as_cued)
    _counts_items(r, directory)
    duration = s5._get(result, "parameters", "duration_s")
    if not (as_cued and isinstance(duration, (int, float)) and printed is not None) or problems:
        return r  # phases are described only when the account says each was done as cued
    windows = _natural_windows(cues, float(duration))
    r.describe("phases from the guard's cues (s after T0: allowance and washout applied)",
               {k: [round(a, 1), round(b, 1)] for k, (a, b) in windows.items()})
    line = Timeline(records, round(printed * NS), float(duration))
    _diagnose(r, line, windows, names=NATURAL_WINDOWS, full_scale=True)
    r.describe(PER_PERSON + ", and D86's pass on the same people",
               {name: _per_person(line, windows[name], "full_scale") for name in NATURAL_WINDOWS})
    r.describe(IDENTITY_TIME, {name: _identity_time(line, windows[name]) for name in NATURAL_WINDOWS})
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    return r

# D87: the account answers each comparison line depends on, and the answer that supports it.
ATTRIBUTION = {"other_faces_in_view": "no"}  # the faces compared are the operator's
SEATED_CONTEXT = {"stayed_seated": "yes", "worked_as_usual": "yes", "looked_at_camera_on_purpose": "no"}


def _support(account: dict[str, str], expected: dict[str, str]) -> str:
    """How the account bears on a line: SUPPORTED, QUALIFIED (an answer unknown) or CONTRADICTED (an answer differs)."""
    if not expected:
        return "depends on no account answer"
    contradicted = [f"{k}={account.get(k)}" for k, v in expected.items()
                    if account.get(k) not in (v, face_account.UNKNOWN, None)]
    unknown = [f"{k}=unknown" for k, v in expected.items() if account.get(k) in (face_account.UNKNOWN, None)]
    if contradicted:
        return "CONTRADICTED by " + ", ".join(contradicted + unknown)
    if unknown:
        return "QUALIFIED: " + ", ".join(unknown)
    return "SUPPORTED by " + ", ".join(f"{k}={v}" for k, v in expected.items())


def _say(r: Reading, depends: str, name: str, value: Any) -> None:
    """A DESCRIPTIVE line with the account answers it depends on, always right after DESCRIPTIVE."""
    r.lines.append(f"DESCRIPTIVE [{depends}] {name}: {s5._text(value)}")


def _comparison_window(cues: dict[str, float]) -> tuple[float, float]:
    """W: WASHOUT_S after T0 to WASHOUT_S before the stop cue (an assumption, as every washout here)."""
    return cues["t0"] + WASHOUT_S, cues["stop"] - WASHOUT_S


def read_comparison(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """D87, the natural recognition comparison: execution items (PASS, FAIL or MISSING), then, once the account is
    locked, DESCRIPTIVE lines each with the answers it depends on. It validates nothing and judges no recognition."""
    r = Reading()
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("face admitted for this validation run with the full-scale comparison: one enrolled identity, no problem",
           {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") == {"validation_run": True, "identities_enrolled": 1, "comparison": True}
           and s5._get(start, "face_problem") is None)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    saved = [s5._load(directory / f"status-{c.get('name')}.json") for c in s5._list(s5._get(result, "captures"))
             if isinstance(c, dict) and (directory / f"status-{c.get('name')}.json").is_file()]
    last = saved[-1] if saved else MISSING
    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    dropped = s5._get(last, "components", "face", "records", "dropped")
    worker = s5._get(last, "components", "face", "worker", "counters")
    outbox = (worker.get("comparison_outcomes_dropped", 0) if isinstance(worker, dict) else MISSING)
    r.item(f"identity records: schema {COMPARISON_SCHEMA}, complete, none dropped (records or comparison outcomes)",
           {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped, "comparison_outcomes_dropped": outbox} if total else MISSING,
           total > 0 and not problems and schemas == [COMPARISON_SCHEMA] and dropped == 0 and outbox == 0
           and total < MAX_IDENTITY_RECORDS)
    ready = s5._get(result, "child", "ready_mono_s")
    cues = _cue_offsets(result)
    duration = s5._get(result, "parameters", "duration_s")
    r.item(f"the run's {face_account.COMPARISON_DURATION_S} s and every cue printed within {CUE_TOLERANCE_S:g} s of "
           "its time", {"duration_s": duration, "cues_s_after_t0": {k: round(v, 2) for k, v in cues.items()}},
           duration == face_account.COMPARISON_DURATION_S
           and all(name in cues and 0 <= cues[name] - at <= CUE_TOLERANCE_S
                   for name, at in face_account.COMPARISON_CUE_AT_S.items()))
    account, problem, fmt = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    if account is not None:
        review = (face_account.review_separate(account, face_account.FIELDS_V6) if fmt == 6
                  else [f"format {fmt} is not the comparison's account (format 6)"])
        r.item("account in format 6, every answer yes, no or unknown (unknown is accepted)", review,
               fmt == 6 and review[-1].endswith(": yes"))
    _counts_items(r, directory)
    if account is None or fmt != 6 or problems or not isinstance(ready, float) or "t0" not in cues \
            or "stop" not in cues:
        return r  # described only from usable records, a time base and a locked account (whatever its answers)
    window = _comparison_window(cues)
    line = Timeline(records, round((cues["t0"] + ready) * NS), float(duration))
    for t in records.get("comparison", ()):
        t["t"] = line.rel(t["frame_mono_ns"])
    comparisons = records.get("comparison", [])
    in_w = [c for c in comparisons if window[0] <= c["t"] < window[1]]
    r.describe("window W (s after T0: the washout after T0 and before the stop cue)",
               [round(window[0], 1), round(window[1], 1)])
    r.describe("account answers (each line below names those it depends on)",
               {k: account.get(k) for k in (*SEATED_CONTEXT, *ATTRIBUTION)})
    none = _support(account, {})
    _say(r, none, "comparison accounting, whole run and W, by kind and disposition",
               {scope: {kind: {d: sum(1 for c in items if c["kind"] == kind and c["disposition"] == d)
                               for d in COMPARISON_DISPOSITIONS
                               if any(c["kind"] == kind and c["disposition"] == d for c in items)}
                        for kind in COMPARISON_KINDS}
                for scope, items in (("run", comparisons), ("W", in_w))})
    _say(r, none, "comparison failures by label, whole run (counted as failed, never as nonmatches)",
               {label: sum(1 for c in comparisons if c["error"] == label)
                for label in sorted({c["error"] for c in comparisons if c["error"]})})
    _comparison_cost(r, result, last, line, comparisons, window, none)
    attribution = _support(account, ATTRIBUTION)
    only = [c for c in in_w if c["kind"] == "full_scale_only" and c["disposition"] == "completed"]
    values = [c["similarity"] for c in only]
    _say(r, attribution, f"full-scale-only faces in W: similarity of each completed comparison to the enrollment, "
         f"[min, median, max], and how many at or above / below the unchanged {MATCH_THRESHOLD:g} (descriptive; never "
         f"a vote)",
               {"completed": len(only), "similarity": _spread(values),
                "at_or_above": sum(1 for v in values if v >= MATCH_THRESHOLD),
                "below": sum(1 for v in values if v < MATCH_THRESHOLD),
                "failed": sum(1 for c in in_w if c["kind"] == "full_scale_only" and c["disposition"] == "failed"),
                "production_not_applied": sum(1 for c in in_w if c["kind"] == "full_scale_only"
                                              and c["disposition"] == "production_not_applied")})
    _say(r, attribution, f"controls in W: the comparison's similarity minus production's on the same frame and "
         f"person, [min, median, max], and whether both fall on the same side of {MATCH_THRESHOLD:g}",
               _controls(in_w, records["result"]))
    _comparison_sampling(r, in_w, none)
    context = _support(account, SEATED_CONTEXT)
    diagnostics = Reading()
    _diagnose(diagnostics, line, {"W": window}, names=("W",), full_scale=True)
    for text in diagnostics.lines:
        depends = (_support(account, {**SEATED_CONTEXT, **ATTRIBUTION}) if "similarity to the closest" in text
                   else context)
        r.lines.append(text.replace("DESCRIPTIVE ", f"DESCRIPTIVE [{depends}] ", 1))  # the same place
    both = _support(account, {**SEATED_CONTEXT, **ATTRIBUTION})
    _say(r, both, PER_PERSON + ", and D86's pass on the same people, in W", _per_person(line, window, "full_scale"))
    _say(r, both, IDENTITY_TIME + ", in W", _identity_time(line, window))
    stop = s5._get(result, "stop")
    stop = stop if isinstance(stop, dict) else {}
    requested, after_launch = stop.get("requested_at_s"), s5._get(result, "child", "ready_after_s")
    _say(r, none, "the run's end (s after T0): the stop warning cue, the guard's stop request, the exit after it",
         {"stop_cue": round(cues["stop"], 3),
          "stop_requested": round(requested - after_launch, 3) if isinstance(requested, (int, float))
          and isinstance(after_launch, (int, float)) else None,
          "exited_after_stop_request_s": stop.get("exited_after_stop_s"), "requested_by": stop.get("requested_by")})
    return r


CANDIDATE_FACE = {"validation_run": True, "identities_enrolled": 1, "detection": DETECTION_FULL_SCALE}


def read_candidate(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """D88, the full-scale detection candidate's natural recognition check: the comparison check's execution items (the
    same 300 s seated run, cues and format-6 account), then, once the account is locked, DESCRIPTIVE lines over the
    same window W, each with the answers it depends on. It validates nothing, judges no recognition and changes no
    F2 criterion: production numbers only, with no shadow pass beside them."""
    r = Reading()
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item("face admitted for this validation run with the full-scale detection candidate: one enrolled identity, no "
           "problem", {"face": s5._get(start, "face"), "face_problem": s5._get(start, "face_problem")},
           s5._get(start, "face") == CANDIDATE_FACE and s5._get(start, "face_problem") is None)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    saved = [s5._load(directory / f"status-{c.get('name')}.json") for c in s5._list(s5._get(result, "captures"))
             if isinstance(c, dict) and (directory / f"status-{c.get('name')}.json").is_file()]
    last = saved[-1] if saved else MISSING
    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    dropped = s5._get(last, "components", "face", "records", "dropped")
    schema = schemas[0] if len(schemas) == 1 and schemas[0] in CANDIDATE_SCHEMAS else CANDIDATE_SCHEMA
    r.item(f"identity records: schema {schema}, complete, none dropped, no comparison record",
           {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped} if total else MISSING,
           total > 0 and not problems and schemas == [schema] and dropped == 0
           and not records.get("comparison") and total < MAX_IDENTITY_RECORDS)
    ready = s5._get(result, "child", "ready_mono_s")
    cues = _cue_offsets(result)
    duration = s5._get(result, "parameters", "duration_s")
    r.item(f"the run's {face_account.COMPARISON_DURATION_S} s and every cue printed within {CUE_TOLERANCE_S:g} s of "
           "its time", {"duration_s": duration, "cues_s_after_t0": {k: round(v, 2) for k, v in cues.items()}},
           duration == face_account.COMPARISON_DURATION_S
           and all(name in cues and 0 <= cues[name] - at <= CUE_TOLERANCE_S
                   for name, at in face_account.COMPARISON_CUE_AT_S.items()))
    account, problem, fmt = _account(directory)
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    if account is not None:
        review = (face_account.review_separate(account, face_account.FIELDS_V6) if fmt == 6
                  else [f"format {fmt} is not this check's account (format 6)"])
        r.item("account in format 6, every answer yes, no or unknown (unknown is accepted)", review,
               fmt == 6 and review[-1].endswith(": yes"))
    _counts_items(r, directory)
    if account is None or fmt != 6 or problems or not isinstance(ready, float) or "t0" not in cues \
            or "stop" not in cues or schemas != [schema] or s5._get(start, "face") != CANDIDATE_FACE:
        return r  # described only from the candidate's usable records, a time base and a locked account
    window = _comparison_window(cues)
    line = Timeline(records, round((cues["t0"] + ready) * NS), float(duration))
    r.describe("window W (s after T0: the washout after T0 and before the stop cue)",
               [round(window[0], 1), round(window[1], 1)])
    r.describe("account answers (each line below names those it depends on)",
               {k: account.get(k) for k in (*SEATED_CONTEXT, *ATTRIBUTION)})
    none = _support(account, {})
    _candidate_cost(r, line, window, last, none)
    _monitoring(r, result, last, none)
    context, both = _support(account, SEATED_CONTEXT), _support(account, {**SEATED_CONTEXT, **ATTRIBUTION})
    _say(r, both, PER_PERSON + ", and where no face was assigned the best detection under 0.9, in W",
         _per_person(line, window, "candidate"))
    diagnostics = Reading()
    _diagnose(diagnostics, line, {"W": window}, names=("W",), geometry=schema == GEOMETRY_SCHEMA)
    for text in diagnostics.lines:
        depends = both if "similarity to the closest" in text or "geometry" in text else context
        r.lines.append(text.replace("DESCRIPTIVE ", f"DESCRIPTIVE [{depends}] ", 1))
    _say(r, both, IDENTITY_TIME + ", in W", _identity_time(line, window))
    stop = s5._get(result, "stop")
    stop = stop if isinstance(stop, dict) else {}
    requested, after_launch = stop.get("requested_at_s"), s5._get(result, "child", "ready_after_s")
    _say(r, none, "the run's end (s after T0): the stop warning cue, the guard's stop request, the exit after it",
         {"stop_cue": round(cues["stop"], 3),
          "stop_requested": round(requested - after_launch, 3) if isinstance(requested, (int, float))
          and isinstance(after_launch, (int, float)) else None,
          "exited_after_stop_request_s": stop.get("exited_after_stop_s"), "requested_by": stop.get("requested_by")})
    return r


def _candidate_cost(r: Reading, line: Timeline, window: tuple[float, float], last: Any, none: str) -> None:
    """D88: production's own work and time with the candidate, from the records (whole run, and W), and the worker's
    counters at the last capture beside the ticks offered by then (never mixed with the whole run's counts)."""
    worker = s5._get(last, "components", "face", "worker", "counters")
    applied = [x for x in line.results if x["outcome"] == "applied"]
    in_w = [x for x in line.in_window(line.results, window)]
    detections = [x["detection"] for x in applied if x.get("detection") is not None]
    _say(r, none, "production with the candidate, whole run (records): offered and skipped ticks, results by outcome, "
         "error labels, results applied per offered tick; processing, detection and embedding ms (embedding only "
         "where a face was kept) and frame-to-applied ms; the worker's processing time as a share of W (results in "
         "W); and the worker's counters at the last capture",
         {"offered": sum(1 for t in line.ticks if t["tick"] == "offered"),
          "skipped_no_person": sum(1 for t in line.ticks if t["tick"] == "skipped_no_person"),
          "results": {o: sum(1 for x in line.results if x["outcome"] == o) for o in sorted(OUTCOMES)
                      if any(x["outcome"] == o for x in line.results)},
          "errors": {e: sum(1 for x in line.results if x["error"] == e)
                     for e in sorted({x["error"] for x in line.results if x["error"]})},
          "applied_per_offered": round(len(applied) / max(1, sum(1 for t in line.ticks if t["tick"] == "offered")), 3),
          "processing_ms": _spread([x["processing_ms"] for x in applied if x["processing_ms"] is not None]),
          "detect_ms": _spread([d["detect_ms"] for d in detections]),
          "embed_ms": _spread([d["embed_ms"] for d in detections if d["embed_ms"] is not None]),
          "frame_to_applied_ms": _spread([(x["applied"] - x["t"]) * 1000 for x in applied]),
          "processing_share_of_w": round(sum(x["processing_ms"] or 0 for x in in_w) / 1000 / (window[1] - window[0]), 4)
          if window[1] > window[0] else None,
          "worker_at_last_capture": {k: worker.get(k, 0) for k in ("offered", "started", "completed", "errors",
                                                                    "replaced", "expired_before_start",
                                                                    "fallback_dropped")}
          if isinstance(worker, dict) else MISSING})


def _monitoring(r: Reading, result: Any, last: Any, none: str) -> None:
    frames, rates = s5._get(last, "frames"), s5._get(last, "rates")
    _say(r, none, "monitoring at the last capture (frames processed, replaced, failed; processed fps) and the guard's "
         "memory figures (B)",
               {"frames": {k: frames.get(k) for k in ("captured", "processed", "replaced", "failed")}
                if isinstance(frames, dict) else MISSING,
                "processed_fps": rates.get("processed_fps") if isinstance(rates, dict) else MISSING,
                "guard": [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES]})


def _comparison_cost(r: Reading, result: Any, last: Any, line: Timeline, comparisons: list[dict],
                     window: tuple[float, float], none: str) -> None:
    """D87: what the comparison cost, and what production lost to it (never the cost of a production replacement)."""
    worker = s5._get(last, "components", "face", "worker", "counters")
    offered = sum(1 for t in line.ticks if t["tick"] == "offered")
    attempted = [c for c in comparisons if c["disposition"] in ATTEMPTED]
    in_w = [c for c in attempted if window[0] <= c["t"] < window[1]]
    applied = [x for x in line.results if x["outcome"] == "applied"]
    _say(r, none, "production and the comparison's cost, whole run: offered ticks, results by outcome, ticks yielded to "
         "comparisons and the ticks offered by then (both at the last capture), production processing ms and "
         "frame-to-applied ms, comparison ms, and the comparisons' share of W's time",
               {"offered": offered,
                "results": {o: sum(1 for x in line.results if x["outcome"] == o) for o in sorted(OUTCOMES)
                            if any(x["outcome"] == o for x in line.results)},
                "yielded_to_comparison": worker.get("yielded_to_comparison", 0) if isinstance(worker, dict) else MISSING,
                "offered_at_last_capture": worker.get("offered", 0) if isinstance(worker, dict) else MISSING,
                "production_processing_ms": _spread([x["processing_ms"] for x in applied
                                                     if x["processing_ms"] is not None]),
                "frame_to_applied_ms": _spread([(x["applied"] - x["t"]) * 1000 for x in applied]),
                "comparison_ms": _spread([c["ms"] for c in attempted]),
                "comparison_share_of_w": round(sum(c["ms"] for c in in_w) / 1000 / (window[1] - window[0]), 4)
                if window[1] > window[0] else None})
    _monitoring(r, result, last, none)


def _controls(comparisons: list[dict], results: list[dict]) -> dict:
    """Pairs a completed control with production's similarity for the same frame and person, when it has one."""
    production = {(x["stream_epoch"], x["frame_seq"], p["track_id"]): p.get("similarity")
                  for x in results if x["outcome"] == "applied" for p in x["persons"]}
    pairs, unpaired = [], 0
    for c in comparisons:
        if c["kind"] != "both" or c["disposition"] != "completed":
            continue
        theirs = production.get((c["stream_epoch"], c["frame_seq"], c["track_id"]))
        if theirs is None:
            unpaired += 1
        else:
            pairs.append((c["similarity"], theirs))
    same = sum(1 for mine, theirs in pairs if (mine >= MATCH_THRESHOLD) == (theirs >= MATCH_THRESHOLD))
    return {"pairs": len(pairs), "unpaired": unpaired,
            "difference": _spread([round(mine - theirs, 3) for mine, theirs in pairs]),
            "same_side": same, "opposite_side": len(pairs) - same}


def _comparison_sampling(r: Reading, comparisons: list[dict], none: str) -> None:
    """Whether the compared full-scale-only faces differ from the eligible ones that were not compared."""
    eligible = [c for c in comparisons if c["kind"] == "full_scale_only"]
    compared = [c for c in eligible if c["disposition"] in ATTEMPTED]
    skipped = [c for c in eligible if c["disposition"] not in ATTEMPTED]
    _say(r, none, "sampling in W, full-scale-only: compared and not compared, each with the candidates' score and width "
         "[min, median, max] and the first and last time (s after T0)",
               {name: {"n": len(items), "score": _spread([c["score"] for c in items]),
                       "w_px": _spread([c["w_px"] for c in items]),
                       "first_last": None if not items else [round(items[0]["t"], 1), round(items[-1]["t"], 1)]}
                for name, items in (("compared", compared), ("not_compared", skipped))})


def _declared(r: Reading, account: dict[str, str]) -> None:
    """Item 15 for a format-2 account (D66): the start delay known, with its upper end at most DELAY_MAX_S."""
    d = face_account.delay_seconds(account.get(face_account.DELAY))
    v = face_account.uncertainty_seconds(account.get(face_account.DELAY_UNCERTAINTY))
    upper = None if d is None or v is None else d + v
    value = {face_account.DELAY: account.get(face_account.DELAY),
             face_account.DELAY_UNCERTAINTY: account.get(face_account.DELAY_UNCERTAINTY), "delay_upper_s": upper,
             "stayed_until_stop": account.get("stayed_until_stop"),
             "other_faces_in_view": account.get("other_faces_in_view")}
    r.item(f"stopwatch start delay known, its upper end (delay + uncertainty) at most {DELAY_MAX_S} s; stayed until "
           "the stop; no other face in view", value,
           upper is not None and upper <= DELAY_MAX_S and value["stayed_until_stop"] == "yes"
           and value["other_faces_in_view"] == "no")


def _declared_cued(r: Reading, account: dict[str, str], cues: dict[str, float]) -> None:
    """Item 15 for a format-3 or format-4 account (D79, D85): every movement cue printed on time by the guard, every
    movement done within its allowance after its cue; stayed until the stop; no other face in view."""
    value = {"cues_s_after_t0": {name: (round(cues[name], 2) if name in cues else None)
                                 for name in face_account.CUE_AT_S},
             "within_allowance": {key: account.get(face_account.within_key(key)) for key in face_account.TIMES},
             "stayed_until_stop": account.get("stayed_until_stop"),
             "other_faces_in_view": account.get("other_faces_in_view")}
    r.item(f"every movement cue printed within {CUE_TOLERANCE_S:g} s of its time and each movement done within its "
           "allowance after it; stayed until the stop; no other face in view", value,
           _cues_on_time(cues) and all(v == "yes" for v in value["within_allowance"].values())
           and value["stayed_until_stop"] == "yes" and value["other_faces_in_view"] == "no")


def _completion(r: Reading, line: Timeline) -> None:
    offered = sum(1 for t in line.ticks if t["tick"] == "offered")
    completed = sum(1 for x in line.results if x["outcome"] != "failed")
    errors = sum(1 for x in line.results if x["outcome"] == "failed")
    value = {"offered": offered, "completed": completed, "processing_errors": errors,
             "completed_per_offered": round(completed / offered, 4) if offered else None}
    r.item(f"completion: completed / offered >= {MIN_COMPLETION} with 0 processing errors "
           "(the approved bounded-check criterion; not the replay profile's 0.95 Hz)",
           value, offered > 0 and completed <= offered and completed / offered >= MIN_COMPLETION and errors == 0)


def _sampling(r: Reading, line: Timeline, windows: dict, enrolled: str) -> None:
    for name in ("E1", "E2"):
        ticks = line.in_window(line.ticks, windows[name])
        r.item(f"{name} sampling: at least {MIN_TICKS_EMPTY} ticks", len(ticks), len(ticks) >= MIN_TICKS_EMPTY)
    for name in ("K1", "A", "K2"):
        ticks = line.in_window(line.ticks, windows[name])
        offered = sum(1 for t in ticks if t["tick"] == "offered")
        applied = sum(1 for x in line.in_window(line.results, windows[name]) if x["outcome"] == "applied")
        r.item(f"{name} sampling: at least {MIN_OFFERED} offered ticks and {MIN_APPLIED} applied results",
               {"offered": offered, "applied": applied}, offered >= MIN_OFFERED and applied >= MIN_APPLIED)
    for name in ("K1", "K2"):
        matches = sum(1 for x in line.in_window(line.results, windows[name]) if x["outcome"] == "applied"
                      for p in x["persons"] if p["vote"] == "match" and p["identity_id"] == enrolled)
        r.item(f"{name}: at least {MIN_MATCHES_FACING} match votes for the enrolled identity", matches,
               matches >= MIN_MATCHES_FACING)


def _identity(r: Reading, line: Timeline, windows: dict, enrolled: str, account: dict[str, str], bounds: dict,
              fmt: int) -> None:
    others = sorted({p["identity_id"] for x in line.results for p in x["persons"]
                     if p["vote"] == "match" and p["identity_id"] != enrolled}
                    | {t["identity_id"] for t in line.transitions if t["to"] == "known" and t["identity_id"] != enrolled})
    r.item("only the enrolled identity is ever matched or known", others, not others)
    spans = [(key, i["start"], line.end_s if i["end"] is None else i["end"], i["identity_id"])
             for key, intervals in line.known.items() for i in intervals]
    overlaps = [(a[0], b[0]) for n, a in enumerate(spans) for b in spans[n + 1:]
                if a[0] != b[0] and a[3] == b[3] and a[1] < b[2] and b[1] < a[2]]
    r.item("one identity is never known on two tracks at once", [list(map(list, o)) for o in overlaps], not overlaps)
    unjustified = [(t["stream_epoch"], t["track_id"], round(t["t"], 1)) for t in line.transitions
                   if t["to"] == "known" and t["from"] != "known"
                   and not line.justified((t["stream_epoch"], t["track_id"]), t["t"], t["identity_id"])]
    r.item(f"every change to known follows {CONFIRMATIONS} applied matches on that track within {VOTE_TTL_S:g} s",
           unjustified, not unjustified)
    for name in ("E1", "E2"):
        known = [(k, round(i["start"], 1)) for k, i in line.known_overlapping(windows[name])]
        r.item(f"{name}: no track is known while nobody is in view", known, not known)
    for name in ("K1", "K2"):
        reached = [(k, i) for k, i in line.known_overlapping(windows[name])
                   if i["identity_id"] == enrolled and i["basis_changes"][0][1] == "fresh"]
        r.item(f"{name}: the enrolled identity becomes known (fresh) while facing the camera",
               [(list(k), round(i["start"], 1)) for k, i in reached], bool(reached))
    _return(r, line, windows, bounds, fmt)
    _retention(r, line, windows, enrolled, account, bounds, fmt)


def _return(r: Reading, line: Timeline, windows: dict, bounds: dict, fmt: int) -> None:
    """Items 27-28 at ``left``: the latest time the operator could have left (format 1: the reading itself)."""
    left = bounds["out_of_view_2"][1]
    before = [(k, i) for k, intervals in line.known.items() for i in intervals if i["start"] < left]
    carried = [(list(k), round(i["start"], 1)) for k, i in before
               if i["end"] is None or i["end"] >= windows["E2"][0]]
    r.item("every identity known before leaving is cleared before the empty window E2" if fmt == 1 else
           "every identity known before the latest possible leaving time is cleared before the empty window E2",
           carried, not carried)
    late = []
    for key, interval in line.known_overlapping(windows["K2"]):
        votes = line.votes_known_by(key, interval["start"])[-CONFIRMATIONS:]
        if interval["start"] < left or any(v["t"] < left for v in votes):
            late.append((list(key), round(interval["start"], 1)))
    r.item("K2: nothing carried over: known again only from votes after the return" if fmt == 1 else
           "K2: nothing carried over: known again only from votes after the latest possible leaving time",
           late, not late)


def _retention(r: Reading, line: Timeline, windows: dict, enrolled: str, account: dict[str, str], bounds: dict,
               fmt: int) -> None:
    """Item 29. The turn is the interval [lo, hi] of turned_away (format 1: [t, t], the reading itself): exactly one
    known interval of the enrolled identity may overlap it, and it must cover all of it."""
    lo, hi = bounds["turned_away"]
    away = windows["A"]
    overlapping = [(key, interval) for key, intervals in line.known.items() for interval in intervals
                   if interval["identity_id"] == enrolled and interval["start"] <= hi
                   and (interval["end"] is None or interval["end"] > lo)]
    held = [(key, interval) for key, interval in overlapping
            if interval["start"] <= lo and (interval["end"] is None or interval["end"] > hi)]
    name = "A: a known identity is retained on the same track, then cleared when its older confirming vote is " \
           f"{VOTE_TTL_S:g} s old"
    if account.get("face_hidden_while_turned") != "yes":
        r.not_exercised(name, "the account does not say the face was hidden while turned away")
        return
    if fmt == 1 and len(held) != 1:
        r.not_exercised(name, f"{len(held)} tracks known at the turn")
        return
    if fmt != 1 and (len(overlapping) != 1 or len(held) != 1):
        r.not_exercised(name, {"turn_interval_s": [lo, hi], "known_throughout_it": len(held),
                               "known_in_part_of_it": len(overlapping) - len(held)})
        return
    key, interval = held[0]
    offered = [t for t in line.in_window(line.ticks, away) if t["tick"] == "offered"]
    continuity = sum(1 for t in offered if key[1] in t["tracks"]) / len(offered) if offered else 0.0
    matches = [v for v in line.votes.get(key, ()) if v["vote"] == "match" and hi <= v["t"] < away[1]]
    if fmt == 1:
        r.describe("A: new matching evidence on the retained track after the turn (match votes)", len(matches))
    else:
        r.describe("A: new matching evidence on the retained track after the latest possible turn time (match votes)",
                   len(matches))
        r.describe("A: match votes on the retained track inside the turn interval",
                   sum(1 for v in line.votes.get(key, ()) if v["vote"] == "match" and lo <= v["t"] < hi))
    if continuity < MIN_CONTINUITY or interval["end_to"] == "cleared":
        r.not_exercised(name, {"track": list(key), "continuity": round(continuity, 3), "ended": interval["end_to"]})
        return
    if interval["end"] is None or interval["end"] > away[1]:
        recent = [v for v in matches if v["t"] >= away[1] - VOTE_TTL_S]
        value = {"still_known_at_window_end": True, "recent_match_votes": len(recent)}
        if recent:  # new matching evidence kept it known: retention itself was not tested
            r.not_exercised(name, value)
        else:  # no new evidence for 30 s, yet still known
            r.item(name, value, False)
        return
    support = line.votes_known_by(key, interval["end"] - 1e-3)[-CONFIRMATIONS:]
    older = support[0]["t"] if len(support) == CONFIRMATIONS else None
    expected = None if older is None else older + VOTE_TTL_S
    newest = support[-1]["t"] if support else None
    retained_at = next((t for t, basis in interval["basis_changes"] if basis == "retained"), None)
    value = {"track": list(key), "continuity": round(continuity, 3), "older_vote_s": None if older is None
             else round(older, 1), "cleared_s": round(interval["end"], 1), "cleared_to": interval["end_to"],
             "expected_clear_s": None if expected is None else round(expected, 1),
             "retained_from_s": None if retained_at is None else round(retained_at, 1)}
    if interval["end_to"] == "uncertain":
        r.not_exercised(name, {**value, "why": "a contradicting vote ended it"})
        return
    r.item(name, value, expected is not None and interval["end_to"] == "unresolved"
           and expected - EXACT_S <= interval["end"] <= expected + STATE_TOLERANCE_S
           and retained_at is not None and newest is not None
           and newest + FRESH_S - EXACT_S <= retained_at <= newest + FRESH_S + STATE_TOLERANCE_S)


COUNT_FILES = ("secret-counts-camera.txt", "secret-counts-identity.txt")
NOT_COUNTED = face_evidence.NOT_COUNTED  # written after the counts, or the counts themselves
# face_privacy's keys per counts file; "nonconforming" counts the parts of a file that do not fit its profile
COUNT_KEYS = {"secret-counts-camera.txt": ("userinfo",),
              "secret-counts-identity.txt": ("passphrase", "name", "nonconforming")}
F1_COUNT_KEYS = ("passphrase", "name", "filenames", "nonconforming")


def _count_check(counts: Any, expected: set[str], keys: tuple[str, ...]) -> tuple[Any, bool]:
    """(the value shown, passed): every expected file counted with exactly these keys, all 0."""
    if not isinstance(counts, dict):
        return MISSING, False
    by_file: dict[str, set[str]] = {}
    for line in counts:
        path, key = line.rsplit(" ", 1)
        by_file.setdefault(path, set()).add(key)
    incomplete = sorted(path for path in expected if by_file.get(path) != set(keys))
    nonzero = [k for k, v in counts.items() if v]
    return {"lines": len(counts), "nonzero": nonzero, "incomplete": incomplete}, not nonzero and not incomplete


def _counts_items(r: Reading, directory: Path) -> None:
    expected = {p.name for p in directory.iterdir() if p.is_file() and p.name not in NOT_COUNTED}
    for name, label in zip(COUNT_FILES, ("camera userinfo", "passphrase, name and nonconforming")):
        value, passed = _count_check(_counts(directory, name), expected, COUNT_KEYS[name])
        r.item(f"{label} counts: every evidence file counted with each, all 0", value, passed)


def _describe(r: Reading, result: Any, line: Timeline, windows: dict, last: Any, bounds: dict, fmt: int) -> None:
    r.describe("guard peak pressure, min MemFree, min MemAvailable (B)",
               [s5._get(result, "guard", k) for k in s5.GUARD_FIGURES])
    r.describe("face processing ms at the last capture", s5._get(last, "components", "face", "worker", "processing_ms"))
    span = max(line.end_s, 1e-9)
    r.describe("scheduling rate: ticks per second over the run, and per window (s^-1)",
               {"run": round(len([t for t in line.ticks if 0 <= t["t"] < span]) / span, 3),
                **{k: round(len(line.in_window(line.ticks, w)) / (w[1] - w[0]), 3)
                   if w[1] > w[0] or (fmt == 1 and w[1] != w[0]) else None  # format 1 as before; never / 0
                   for k, w in windows.items()}})
    r.describe("results by outcome", {o: sum(1 for x in line.results if x["outcome"] == o) for o in sorted(OUTCOMES)})
    r.describe("offered ticks in the empty windows (person detections while nobody was in view)",
               {k: sum(1 for t in line.in_window(line.ticks, windows[k]) if t["tick"] == "offered")
                for k in ("E1", "E2")})
    usable = {}
    for k in ("K1", "A", "K2"):
        applied = [x for x in line.in_window(line.results, windows[k]) if x["outcome"] == "applied"]
        assigned = sum(1 for x in applied if any(p["ownership"] == "assigned" for p in x["persons"]))
        usable[k] = None if not applied else round(assigned / len(applied), 3)
    r.describe("usable-face rate (applied results with an assigned face) per window", usable)
    firsts = {}
    for k, start_key in (("K1", "seated_facing_1"), ("K2", "seated_facing_2")):
        known = sorted(i["start"] for _, i in line.known_overlapping((windows[k][0] - WASHOUT_S, windows[k][1])))
        if fmt == 1:
            firsts[k] = None if not known else round(known[0] - (windows[k][0] - WASHOUT_S), 1)
        else:  # from the latest and from the earliest possible sitting time
            lo, hi = bounds[start_key]
            firsts[k] = None if not known else [round(known[0] - hi, 1), round(known[0] - lo, 1)]
    r.describe("time to known after sitting down facing the camera (s)" if fmt == 1 else
               "time to known after sitting down facing the camera, from the latest and the earliest possible "
               "sitting time (s)", firsts)


def _spread(values: list[float]) -> list[float] | None:
    """[min, median, max], rounded; None when there is no value."""
    if not values:
        return None
    return [round(min(values), 3), round(statistics.median(values), 3), round(max(values), 3)]


def _diagnose(r: Reading, line: Timeline, windows: dict, names: tuple[str, ...] = WINDOWS,
              full_scale: bool = False, geometry: bool = False) -> None:
    """Schema 2 (D84): DESCRIPTIVE diagnostics per judged window, never judged. Whether a person was detected on each
    tick and on which track; where the people on applied results were and how sure the person detector was; how many
    of them had a face from the face detector, its size and score; and how close each vote came to the enrolled
    identity, by vote."""
    continuity, people, faces, similarity, full = {}, {}, {}, {}, {}
    cleared = [(t["t"], t["track_id"]) for t in line.transitions if t["to"] == "cleared"]
    for name in names:
        ticks = line.in_window(line.ticks, windows[name])
        run = longest = 0
        for tick in ticks:
            run = run + 1 if tick["tick"] == "skipped_no_person" else 0
            longest = max(longest, run)
        continuity[name] = {"offered": sum(1 for t in ticks if t["tick"] == "offered"),
                            "skipped_no_person": sum(1 for t in ticks if t["tick"] == "skipped_no_person"),
                            "longest_skipped_run": longest,
                            "track_ids": sorted({k for t in ticks if t["tick"] == "offered" for k in t["tracks"]}),
                            # an identity cleared when its track stopped being current, the same track ID offered
                            # again later (the tracker kept it longer than the runtime's track expiry)
                            "cleared_then_offered": sorted({k for at, k in cleared for t in ticks
                                                            if t["tick"] == "offered" and t["t"] > at
                                                            and k in t["tracks"]}),
                            "known_entries": sum(1 for t in line.in_window(line.transitions, windows[name])
                                                 if t["to"] == "known" and t["from"] != "known")}
        persons = [p for x in line.in_window(line.results, windows[name]) if x["outcome"] == "applied"
                   for p in x["persons"]]
        assigned = [p["face"] for p in persons if p["face"] is not None]
        people[name] = None if not persons else {
            "n": len(persons), "box_median": [round(statistics.median(p["box"][i] for p in persons), 3)
                                              for i in range(4)],
            "confidence": _spread([p["confidence"] for p in persons])}
        faces[name] = None if not persons else {
            "faces": len(assigned), "people": len(persons), "w_px": _spread([f["w_px"] for f in assigned]),
            "h_px": _spread([f["h_px"] for f in assigned]), "score": _spread([f["score"] for f in assigned])}
        applied = [x for x in line.in_window(line.results, windows[name]) if x["outcome"] == "applied"]
        passes = [x["full_scale"] for x in applied if x.get("full_scale") is not None]
        none_found = [x["full_scale"] for x in applied if x.get("full_scale") is not None and not x["faces"]]
        full[name] = None if not applied else {
            "results": len(applied), "with_pass": len(passes),
            "production_face": sum(1 for x in applied if any(p["ownership"] == "assigned" for p in x["persons"])),
            "full_scale_face": sum(1 for f in passes if f["assigned"]),
            "full_scale_only": sum(1 for f in none_found if f["assigned"]),
            "best_score_where_production_found_none": _spread([f["best_score"] for f in none_found
                                                               if f["best_score"] is not None]),
            "full_scale_w_px": _spread([f["best_w_px"] for f in passes if f["faces"]]),
            "ms": _spread([f["ms"] for f in passes])}
        votes = {vote: [p["similarity"] for p in persons if p["label"] == vote] for vote in ("match", "unknown")}
        similarity[name] = None if not persons else {
            "match_votes": len(votes["match"]), "match": _spread(votes["match"]),
            "unknown_votes": len(votes["unknown"]), "unknown": _spread(votes["unknown"])}
    r.describe("diagnostics: person continuity per window (offered and skipped ticks, the longest run of skipped "
               "ticks, track IDs on offered ticks)", continuity)
    r.describe("diagnostics: people on applied results per window (median box [x1, y1, x2, y2] as fractions of the "
               "frame; person detector confidence [min, median, max])", people)
    r.describe("diagnostics: assigned faces per window (faces among the people on applied results; width and height "
               "in frame pixels and the face detector's score, each [min, median, max])", faces)
    r.describe("diagnostics: similarity to the closest enrolled identity per window, by vote ([min, median, max]; "
               f"the runtime's match threshold is {MATCH_THRESHOLD:g})", similarity)
    if full_scale:
        r.describe("diagnostics: the full-scale face pass per window (D86: the same YuNet at native scale, never a "
                   "vote; results with a pass, results with a face assigned by production and by the full-scale pass "
                   "at 0.9, those found only at full scale, the best score where production found none, the face "
                   "width in frame pixels and the pass's ms, each [min, median, max])", full)
    if geometry:
        r.describe(GEOMETRY_LINE, {name: _geometry_by_vote(line, windows[name]) for name in names})


GEOMETRY_LINE = ("diagnostics: the assigned faces' geometry per window, by vote (D90; geometry, never a cause): faces "
                 "with the detector's own box, those reaching past the left, top, right and bottom edges and how far "
                 "(px [min, median, max]), the eye line's angle (degrees) and the eyes' spacing (px), each [min, "
                 "median, max], and the votes per prototype position")


def _geometry_by_vote(line: Timeline, window: tuple[float, float]) -> dict | None:
    persons = [p for x in line.in_window(line.results, window) if x["outcome"] == "applied" for p in x["persons"]
               if p.get("geometry") is not None]
    if not persons:
        return None
    out = {}
    for vote in ("match", "unknown"):
        group = [p for p in persons if p["label"] == vote]
        raw = [p["geometry"] for p in group if p["geometry"]["raw_box"] is not None]
        eyes = [p["geometry"]["eyes"] for p in group if p["geometry"]["eyes"] is not None]
        edges = {}
        for index, edge in enumerate(("left", "top", "right", "bottom")):
            past = [g["overhang_px"][index] for g in raw if g["overhang_px"][index] > 0]
            edges[edge] = {"n": len(past), "px": _spread(past)}
        positions: dict[str, int] = {}
        for p in group:
            if p["prototype"] is not None:
                positions[str(p["prototype"])] = positions.get(str(p["prototype"]), 0) + 1
        out[vote] = {"faces": len(group), "raw_box": len(raw), "past_edge": edges,
                     "roll_deg": _spread([e["roll_deg"] for e in eyes]),
                     "spacing_px": _spread([e["spacing_px"] for e in eyes]),
                     "prototype": dict(sorted(positions.items()))}
    return out


def _per_person(line: Timeline, window: tuple[float, float], kind: str) -> dict:
    """One denominator for every share in it (D88's correction): the person on each applied result in the window
    whose frame had exactly one person, so result-level numbers (D86's pass, the candidate's own) and person-level
    ones (the assigned face, the vote) count the same people. Results with more people are counted apart, never mixed
    in. ``kind``: "full_scale" (schemas 3 and 4: D86's pass beside production), "candidate" (schema 5) or "plain"."""
    applied = [x for x in line.in_window(line.results, window) if x["outcome"] == "applied"]
    single = [x for x in applied if len(x["persons"]) == 1]
    people = [x["persons"][0] for x in single]
    face = [p["ownership"] == "assigned" for p in people]
    labels = sorted({p["label"] for p in people})
    out: dict[str, Any] = {
        "applied_results": len(applied), "one_person": len(single), "more_people_apart": len(applied) - len(single),
        "face_assigned": sum(face), "ambiguous": sum(1 for p in people if p["ownership"] == "ambiguous"),
        "votes": {label: sum(1 for p in people if p["label"] == label) for label in labels}}
    if kind == "full_scale":
        passes = [(f, x["full_scale"]["assigned"] > 0) for f, x in zip(face, single) if x.get("full_scale") is not None]
        out["full_scale_pass"] = {"results": len(passes), "both": sum(1 for f, g in passes if f and g),
                                  "production_only": sum(1 for f, g in passes if f and not g),
                                  "full_scale_only": sum(1 for f, g in passes if g and not f),
                                  "neither": sum(1 for f, g in passes if not (f or g))}
    elif kind == "candidate":
        without = [x["detection"] for f, x in zip(face, single) if not f and x.get("detection") is not None]
        below = [d["best_below_score"] for d in without if d["best_below_score"] is not None]
        out["without_a_face"] = {"n": len(without), "best_below_0.9": _spread(below),
                                 "no_detection_at_the_floor": len(without) - len(below)}
    return out


def _identity_time(line: Timeline, window: tuple[float, float]) -> dict:
    """How long identities were KNOWN within the window, in person-seconds over every track (each KNOWN interval clipped
    to the window; one still open when the records end runs to the run's end), the KNOWN intervals overlapping it,
    the entries into KNOWN within it, and the state each interval ending within it went to."""
    start, end = window
    intervals = [i for _, i in line.known_overlapping(window)]
    known = sum(min(end, line.end_s if i["end"] is None else i["end"]) - max(start, i["start"]) for i in intervals)
    ended = [i["end_to"] for i in intervals if i["end"] is not None and start <= i["end"] < end]
    return {"window_s": round(end - start, 1), "known_s": round(known, 1), "known_intervals": len(intervals),
            "entries_into_known": sum(1 for t in line.in_window(line.transitions, window)
                                      if t["to"] == "known" and t["from"] != "known"),
            "ended_to": {state: ended.count(state) for state in sorted(set(ended))}}


PER_PERSON = ("detection and votes per person (D88's single denominator: the person on each applied result whose frame "
              "had exactly one person; results with more people counted apart): faces assigned, ambiguous, votes by "
              "label")
IDENTITY_TIME = ("identity time (person-seconds KNOWN within the window, KNOWN intervals overlapping it, entries into "
                 "KNOWN, and the state each interval ending within it went to)")


# ---------------------------------------------------------------- D91: the camera-side enrollment experiment

CAPTURE_FACE = {**CANDIDATE_FACE, "capture": True}  # the startup's face entry of a capture run
SHADOW_FACE = {**CANDIDATE_FACE, "shadow": True}  # ... and of a run compared with the working gallery
CAPTURE_CONDITIONS = {"consented_to_capture": "yes", "stayed_seated": "yes", "other_faces_in_view": "no"}
NON_ENROLLED_CONTEXT = {"participant_not_enrolled": "yes", "participant_consented": "yes",
                        "participant_stayed_seated": "yes", "only_participant_in_view": "yes"}
LOOK_WINDOW_S = 30.0  # the look cue's interval described apart: from the cue, its 10 s allowance and a margin
EXPERIMENT_CUES = {6: face_account.COMPARISON_CUE_AT_S, 8: face_account.NON_ENROLLED_CUE_AT_S}


def _experiment_execution(r: Reading, directory: Path, f1: Path, gate: Any, face: dict, name: str,
                          formats: tuple[int, ...], records_check: Any) -> tuple[Any, ...] | None:
    """The execution items the capture and the shadow runs share with the candidate's reading; None when nothing can
    be described (no usable records, time base or locked account), else what the descriptions need."""
    if gate is not None:
        ok, lines = f1_gate(*gate)
        r.item("F1 basis for this run: the run's own F1 validated, or the maintainer's accepted corrected F1 reading "
               "of it", dict(line.split("=", 1) for line in lines), ok and f1.resolve() == (gate[0] / "f1").resolve())
    result = s5._load(directory / "result.json")
    s5._supervision(r, result, "duration_stop", 0)
    start = s5._get(result, "run_output", "starting")
    r.item(f"face admitted for this validation run with the full-scale detection candidate and {name}: one "
           "enrolled identity, no problem", {"face": s5._get(start, "face"), "face_problem": s5._get(start,
                                                                                                   "face_problem")},
           s5._get(start, "face") == face and s5._get(start, "face_problem") is None)
    stopped = s5._get(result, "run_output", "stopped")
    r.item("clean shutdown: all stopped (the face worker included), database closed", stopped,
           isinstance(stopped, dict) and stopped.get("all_stopped") is True and stopped.get("database_closed") is True
           and (stopped.get("stopped") or {}).get("face_worker") is True)
    saved = [s5._load(directory / f"status-{c.get('name')}.json") for c in s5._list(s5._get(result, "captures"))
             if isinstance(c, dict) and (directory / f"status-{c.get('name')}.json").is_file()]
    last = saved[-1] if saved else MISSING
    records, problems, _, _ = read_records(directory / "run.jsonl")
    total = sum(len(v) for v in records.values())
    schemas = sorted({item["schema"] for kind in records.values() for item in kind})
    dropped = s5._get(last, "components", "face", "records", "dropped")
    extra = records_check(records)
    r.item(f"identity records: schema {GEOMETRY_SCHEMA}, complete, none dropped, no comparison record; {extra[0]}",
           {"records": {k: len(v) for k, v in records.items()}, "schemas": schemas, "problems": problems[:5],
            "dropped": dropped} if total else MISSING,
           total > 0 and not problems and schemas == [GEOMETRY_SCHEMA] and dropped == 0
           and not records.get("comparison") and extra[1] and total < MAX_IDENTITY_RECORDS)
    ready = s5._get(result, "child", "ready_mono_s")
    cues = _cue_offsets(result)
    duration = s5._get(result, "parameters", "duration_s")
    account, problem, fmt = _account(directory)
    expected = EXPERIMENT_CUES.get(fmt, face_account.COMPARISON_CUE_AT_S)
    r.item(f"the run's {face_account.COMPARISON_DURATION_S} s and every cue printed within {CUE_TOLERANCE_S:g} s of "
           "its time", {"duration_s": duration, "cues_s_after_t0": {k: round(v, 2) for k, v in cues.items()}},
           duration == face_account.COMPARISON_DURATION_S
           and all(name in cues and 0 <= cues[name] - at <= CUE_TOLERANCE_S for name, at in expected.items()))
    r.item("account saved and locked before any result was shown", problem or "locked", account is not None)
    if account is not None:
        fields = {f: face_account.FORMATS[h][1] for h, (f, _) in face_account.FORMATS.items()}.get(fmt)
        review = (face_account.review_separate(account, fields) if fmt in formats
                  else [f"format {fmt} is not this run's account (format {' or '.join(map(str, formats))})"])
        r.item(f"account in format {' or '.join(map(str, formats))}, every answer yes, no or unknown (unknown is "
               "accepted)", review, fmt in formats and review[-1].endswith(": yes"))
    _counts_items(r, directory)
    if account is None or fmt not in formats or problems or not isinstance(ready, float) or "t0" not in cues \
            or "stop" not in cues or schemas != [GEOMETRY_SCHEMA] or s5._get(start, "face") != face:
        return None
    window = _comparison_window(cues)
    line = Timeline(records, round((cues["t0"] + ready) * NS), float(duration))
    r.describe("window W (s after T0: the washout after T0 and before the stop cue)",
               [round(window[0], 1), round(window[1], 1)])
    return line, window, account, fmt, records, cues


def read_capture(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """D91, the capture: the candidate's execution items, then whether the working gallery it wrote may be used (written,
    read back and verified, at least CAPTURE_MIN_PROTOTYPES consistent prototypes, and the account's consent, seat and
    no other face), then DESCRIPTIVE numbers. It validates no recognition."""
    r = Reading()

    def one_capture(records: dict) -> tuple[str, bool]:
        return "exactly one capture record", len(records.get("capture", [])) == 1

    found = _experiment_execution(r, directory, f1, gate, CAPTURE_FACE, "the capture", (7,), one_capture)
    if found is None:
        return r
    line, window, account, _, records, _ = found
    (capture,) = records["capture"]
    r.item(f"the working gallery: written, read back and verified, from at least {CAPTURE_MIN_PROTOTYPES} prototypes "
           "consistent at the match threshold", {k: capture[k] for k in ("written", "refused", "verified",
                                                                         "prototypes")},
           capture["written"] is True and capture["verified"] is True and capture["refused"] is None
           and capture["prototypes"] >= CAPTURE_MIN_PROTOTYPES)
    conditions = {k: account.get(k) for k in CAPTURE_CONDITIONS}
    r.item("the capture's conditions (consent to the capture, stayed seated, no other face in view); otherwise the "
           "working gallery must not be used: purge it and capture again", conditions, conditions == CAPTURE_CONDITIONS)
    depends = _support(account, CAPTURE_CONDITIONS)
    _say(r, depends, "the capture's numbers: faces kept and over the cap, faces in the selection span, prototypes, the "
         "consistency check's smallest best similarity among them, and each prototype's best similarity to the "
         "original gallery (descriptive; nothing was selected by it)",
         {k: capture[k] for k in ("faces_kept", "faces_dropped_over_cap", "faces_in_span", "prototypes",
                                  "min_best_to_others", "similarity_to_original")})
    context = _support(account, {**CAPTURE_CONDITIONS, "worked_as_usual": "yes"})
    _say(r, context, PER_PERSON + ", and where no face was assigned the best detection under 0.9, in W",
         _per_person(line, window, "candidate"))
    _say(r, context, IDENTITY_TIME + ", in W (the original gallery's, as the runtime decided)",
         _identity_time(line, window))
    r.describe(GEOMETRY_LINE.replace("diagnostics: ", "", 1), {"W": _geometry_by_vote(line, window)})
    return r


def _label(similarity: Any) -> str | None:
    return None if similarity is None else "match" if similarity >= MATCH_THRESHOLD else "unknown"


def read_experiment(directory: Path, f1: Path, gate: tuple[Path, Path, Path | None] | None = None) -> Reading:
    """D91, an evaluation session with the working gallery compared read only: a genuine session (format 6, the
    enrolled person working as usual) or a non-enrolled participant's (format 8). Execution items, then DESCRIPTIVE
    paired numbers on the same faces: the original's votes (as the runtime decided), the working gallery's at the
    unchanged MATCH_THRESHOLD, and either's (the larger similarity, what one gallery holding both prototype sets would
    give; such a gallery would need more than MAX_PROTOTYPES, so it is a comparison, not a configuration), and KNOWN
    time replayed under the unchanged identity rule for each. It validates nothing and changes no gallery."""
    r = Reading()

    def shadows(records: dict) -> tuple[str, bool]:
        compared = [p for x in records.get("result", []) if x["outcome"] == "applied" for p in x["persons"]
                    if p.get("similarity") is not None]
        return ("no capture record, and every compared face compared with the working gallery too",
                not records.get("capture") and all(p.get("shadow") is not None for p in compared))

    found = _experiment_execution(r, directory, f1, gate, SHADOW_FACE, "the working gallery compared read only",
                                  (6, 8), shadows)
    if found is None:
        return r
    line, window, account, fmt, _, cues = found
    if fmt == 6:
        kind, context = "genuine", _support(account, {**SEATED_CONTEXT, **ATTRIBUTION})
    else:
        kind, context = "non-enrolled", _support(account, NON_ENROLLED_CONTEXT)
    r.describe("session kind and account answers (each line below names those it depends on)",
               {"kind": kind, **{k: v for k, v in account.items() if k != "notes"}})
    _say(r, context, PER_PERSON + ", and where no face was assigned the best detection under 0.9, in W",
         _per_person(line, window, "candidate"))
    _say(r, context, "paired comparison on the same faces (one-person results in W with a computed similarity; "
         f"MATCH_THRESHOLD {MATCH_THRESHOLD:g} unchanged): faces at or above it with the original (as the runtime "
         "voted), with the working gallery and with either; both, the original only, the working gallery only, "
         "neither; each gallery's similarities [min, median, max]", _paired(line, window))
    if fmt == 8 and "look" in cues:
        look = (cues["look"], cues["look"] + LOOK_WINDOW_S)
        _say(r, _support(account, {**NON_ENROLLED_CONTEXT, "participant_looked_at_cue": "yes"}),
             f"the same, within {LOOK_WINDOW_S:g} s of the look cue (s after T0 {round(look[0], 1)})",
             _paired(line, look))
    _say(r, context, "identity time replayed by the unchanged rule (2 consecutive matches within 30 s on a track), "
         "in W: KNOWN person-seconds, intervals and entries with the original (the runtime's own, and its replay, "
         "which must agree), with the working gallery and with either", {
             "runtime_original": _identity_time(line, window),
             "replay_original": _replay_known(line, window, lambda p: p["label"] if p["label"] in ("match", "unknown")
                                              else None),
             "replay_working": _replay_known(line, window, lambda p: _label((p.get("shadow") or {}).get("similarity"))),
             "replay_either": _replay_known(line, window, lambda p: _label(None if p.get("similarity") is None else
                                                                           max(p["similarity"],
                                                                               (p.get("shadow") or {}).get(
                                                                                   "similarity", -1.0))))})
    r.describe(GEOMETRY_LINE.replace("diagnostics: ", "", 1), {"W": _geometry_by_vote(line, window)})
    return r


def _paired(line: Timeline, window: tuple[float, float]) -> dict:
    applied = [x for x in line.in_window(line.results, window) if x["outcome"] == "applied" and len(x["persons"]) == 1]
    people = [x["persons"][0] for x in applied if x["persons"][0].get("similarity") is not None]
    original = [p["similarity"] for p in people]
    working = [p["shadow"]["similarity"] for p in people if p.get("shadow") is not None]
    pairs = [(p["similarity"] >= MATCH_THRESHOLD, p["shadow"]["similarity"] >= MATCH_THRESHOLD) for p in people
             if p.get("shadow") is not None]
    return {"faces": len(people), "with_working": len(pairs),
            "at_or_above": {"original": sum(a for a, _ in pairs), "working": sum(b for _, b in pairs),
                            "either": sum(a or b for a, b in pairs)},
            "both": sum(a and b for a, b in pairs), "original_only": sum(a and not b for a, b in pairs),
            "working_only": sum(b and not a for a, b in pairs), "neither": sum(not (a or b) for a, b in pairs),
            "similarity": {"original": _spread(original), "working": _spread(working)}}


def _replay_known(line: Timeline, window: tuple[float, float], label_of: Any) -> dict:
    """KNOWN under one gallery's labels, replayed by the runtime's rule: per track, the votes applied by each moment, in
    frame order, within VOTE_TTL_S of their frame; KNOWN while the newest CONFIRMATIONS are matches; a clear the
    runtime recorded drops the track's votes. Person-seconds within the window, the KNOWN intervals overlapping it and
    the entries into KNOWN inside it. A track the runtime lost without recording a clear (its state was unresolved
    with the original) is not cut here: the original's replay, compared with the runtime's own time, shows whether
    that mattered."""
    votes: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    for x in line.results:
        if x["outcome"] != "applied":
            continue
        for p in x["persons"]:
            label = label_of(p)
            if label is not None:
                votes.setdefault((x["stream_epoch"], p["track_id"]), []).append((x["t"], x["applied"], label))
    clears: dict[tuple[int, int], list[float]] = {}
    for t in line.transitions:
        if t["to"] == "cleared":
            clears.setdefault((t["stream_epoch"], t["track_id"]), []).append(t["t"])
    start, end = window[0], min(window[1], line.end_s)  # nothing is known after the run ends
    total, intervals, entries = 0.0, 0, 0
    for key, track in votes.items():
        track.sort()
        cut = sorted(clears.get(key, ()))

        def known(at: float) -> bool:
            last_clear = max((c for c in cut if c <= at), default=None)
            held = [v for v in track if v[1] <= at + 1e-9 and 0 <= at - v[0] < VOTE_TTL_S
                    and (last_clear is None or v[1] > last_clear)]
            return len(held) >= CONFIRMATIONS and all(v[2] == "match" for v in held[-CONFIRMATIONS:])

        events = sorted({v[1] for v in track} | {v[0] + VOTE_TTL_S for v in track} | set(cut))
        state, since = False, None
        for at in events + [math.inf]:
            now = known(at) if at != math.inf else False
            if now and not state:
                since = at
                if start <= at < end:
                    entries += 1
            elif state and not now:
                lo, hi = max(start, since), min(end, at)
                if hi > lo:
                    total += hi - lo
                    intervals += 1
                elif since < end and at > start:
                    intervals += 1
            state = now
    return {"known_s": round(total, 1), "known_intervals": intervals, "entries_into_known": entries}


# ---------------------------------------------------------------- F1 reading


def _provenance(directory: Path) -> dict[str, str]:
    path = directory / "provenance.txt"
    if not path.is_file():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _mode(path: Path) -> Any:
    try:
        info = os.lstat(path)
    except OSError:
        return MISSING
    return {"mode": oct(stat.S_IMODE(info.st_mode)), "owned": info.st_uid == os.getuid(),
            "regular": stat.S_ISREG(info.st_mode), "directory": stat.S_ISDIR(info.st_mode)}


def read_f1(directory: Path, identity_dir: Path) -> Reading:
    r = Reading()
    prov = _provenance(directory)
    exits = {k: prov.get(k) for k in ("dryrun_exit", "enroll_exit", "list_exit")}
    r.item("each identity command exited 0", exits if all(exits.values()) else MISSING,
           exits == {"dryrun_exit": "0", "enroll_exit": "0", "list_exit": "0"})
    supplied = prov.get("photos_supplied")
    supplied_n = int(supplied) if supplied and supplied.isdigit() else MISSING
    dry = s5._load(directory / "dryrun.json")
    r.item(f"dry run: photos reported by index, at least {MIN_USABLE_PHOTOS} usable, nothing written",
           {"photos": s5._get(dry, "photos"), "accepted": s5._get(dry, "accepted"),
            "dry_run": s5._get(dry, "dry_run"), "supplied": supplied_n,
            "gallery_before": prov.get("gallery_before_dryrun", MISSING),
            "gallery_after": prov.get("gallery_after_dryrun", MISSING)},
           s5._get(dry, "dry_run") is True and "identity_id" not in (dry if isinstance(dry, dict) else {})
           and s5._get(dry, "photos") == supplied_n and isinstance(s5._get(dry, "accepted"), int)
           and s5._get(dry, "accepted") >= MIN_USABLE_PHOTOS
           and [x.get("photo") for x in s5._list(s5._get(dry, "results"))] == list(range(1, (supplied_n or 0) + 1))
           and prov.get("gallery_before_dryrun") == "absent" and prov.get("gallery_after_dryrun") == "absent")
    enroll = s5._load(directory / "enroll.json")
    identity_id = s5._get(enroll, "identity_id")
    r.item(f"enrollment verified from {MIN_USABLE_PHOTOS}-{MAX_USABLE_PHOTOS} usable photos; exactly the supplied "
           "copies deleted; the folder removed",
           {k: s5._get(enroll, k) for k in ("verified", "identity_id", "photos", "accepted", "photos_deleted",
                                             "photos_changed_not_deleted", "folder_entries_left")},
           s5._get(enroll, "verified") is True and isinstance(identity_id, str) and bool(ID.match(identity_id))
           and s5._get(enroll, "photos") == supplied_n and s5._get(enroll, "photos_deleted") == supplied_n
           and s5._get(enroll, "photos_changed_not_deleted") == 0 and s5._get(enroll, "folder_entries_left") == 0
           and isinstance(s5._get(enroll, "accepted"), int)
           and MIN_USABLE_PHOTOS <= s5._get(enroll, "accepted") <= MAX_USABLE_PHOTOS
           and "refused" not in (enroll if isinstance(enroll, dict) else {}))
    listing = s5._get(s5._load(directory / "list.json"), "identities")
    r.item("the gallery holds exactly the enrolled identity, its prototypes and consent date", listing,
           isinstance(listing, list) and len(listing) == 1 and isinstance(listing[0], dict)
           and listing[0].get("identity_id") == identity_id
           and listing[0].get("prototypes") == s5._get(enroll, "accepted")
           and listing[0].get("consent_date") == prov.get("consent_date"))
    folder = prov.get("inbox_folder_name")
    states = {"identity_dir": _mode(identity_dir), "gallery": _mode(identity_dir / "gallery.sealed"),
              "inbox": _mode(identity_dir / "inbox"),
              "inbox_entries": len(list((identity_dir / "inbox").iterdir())) if (identity_dir / "inbox").is_dir()
              else MISSING,
              "dedicated_folder_gone": None if not folder else not (identity_dir / "inbox" / folder).exists()}
    r.item("private storage: directory and inbox 0700, gallery 0600, owned; the inbox empty", states,
           states["identity_dir"] != MISSING and states["gallery"] != MISSING and states["inbox"] != MISSING
           and states["identity_dir"]["mode"] == "0o700" and states["identity_dir"]["owned"]
           and states["inbox"]["mode"] == "0o700" and states["gallery"]["mode"] == "0o600"
           and states["gallery"]["regular"] and states["inbox_entries"] == 0 and states["dedicated_folder_gone"] is True)
    audit = []
    if (identity_dir / "audit.jsonl").is_file():
        audit = [json.loads(x) for x in (identity_dir / "audit.jsonl").read_text().splitlines() if x.strip()]
    enrolls = [a for a in audit if a.get("action") == "enroll"]
    r.item("audit: one enroll line for that identity, with its counts", enrolls if enrolls else MISSING,
           len(enrolls) == 1 and enrolls[0].get("identity_id") == identity_id
           and enrolls[0].get("photos_used") == s5._get(enroll, "accepted")
           and enrolls[0].get("photos_rejected") == (supplied_n if isinstance(supplied_n, int) else -1)
           - (s5._get(enroll, "accepted") or 0))
    value, passed = _count_check(_counts(directory, "secret-counts-identity.txt"), _f1_files(directory), F1_COUNT_KEYS)
    r.item(f"{PRIVACY_ITEM}, and nonconforming parts: every evidence file counted with each, all 0", value, passed)
    return r


PRIVACY_ITEM = "passphrase, name and photo-file-name counts"  # how F1's privacy item begins, before and after 95ce253


def _f1_files(directory: Path) -> set[str]:
    """F1's counted evidence: its directory's files and the identity directory's audit log."""
    return {p.name for p in directory.iterdir() if p.is_file() and p.name not in NOT_COUNTED} | {"audit.jsonl"}


# ---------------------------------------------------------------- F1 recheck and the F1 gate for F2

RUN_EVIDENCE = ("provenance.txt", "f1/dryrun.json", "f1/enroll.json", "f1/list.json", "f1/provenance.txt",
                "f1/check.txt", "f1/secret-counts-identity.txt")
NAME_COUNTS = "secret-counts-names.txt"
CARRIED_SOURCE = "f1/secret-counts-identity.txt"
RECHECK_VERDICT = "face-f1-recheck: validated"
ACCEPTANCE = "acceptance.txt"


def _sha256(path: Path) -> Any:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else MISSING


def _manifest(directory: Path, ignore: frozenset[str] = frozenset()) -> tuple[dict[str, str] | None, list[str]]:
    """The directory's SHA256SUMS as {relative path: SHA-256}, and its problems: a listed file missing or changed,
    or a file present but not listed (``ignore``: names never listed)."""
    try:
        text = (directory / "SHA256SUMS").read_text()
    except OSError:
        return None, ["SHA256SUMS missing"]
    listed = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (?:\./)?(.+)", line)
        if match is None:
            return None, ["SHA256SUMS malformed"]
        listed[match[2]] = match[1]
    problems = [f"{'missing' if not (directory / rel).is_file() else 'changed'}: {rel}"
                for rel, digest in sorted(listed.items()) if _sha256(directory / rel) != digest]
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    problems += [f"not listed: {rel}" for rel in sorted(present - set(listed) - {"SHA256SUMS"} - ignore)]
    return listed, problems


def _last(path: Path) -> Any:
    lines = path.read_text().splitlines() if path.is_file() else []
    return lines[-1] if lines else MISSING


def _key_values(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line) if path.is_file() else {}


def read_f1_recheck(recheck: Path, run: Path, identity_dir: Path) -> Reading:
    """A corrected F1 reading of a run, made afterwards: the run's own result stays as recorded."""
    r = Reading()
    listed, problems = _manifest(run)
    unlisted = [e for e in RUN_EVIDENCE if e not in (listed or {})]
    r.item("the run's evidence unchanged: its SHA256SUMS verifies every file, lists F1's evidence, counts and reading, "
           "and nothing was added", {"problems": problems, "not_listed": unlisted},
           listed is not None and not problems and not unlisted)
    exits = [line.split("=", 1)[1] for line in (run / "provenance.txt").read_text().splitlines()
             if line.startswith("F1_exit=")] if (run / "provenance.txt").is_file() else []
    r.describe("the run's own result, kept as recorded", {"F1_exit": exits[-1] if exits else MISSING,
                                                          "f1/check.txt": _last(run / "f1/check.txt")})
    original = (run / "f1/check.txt").read_text().splitlines() if (run / "f1/check.txt").is_file() else []
    fresh = read_f1(run / "f1", identity_dir).lines

    def others(lines: list[str]) -> list[str]:
        return [x for x in lines if not x.startswith("face-f1:") and not x.split(" ", 1)[-1].startswith(PRIVACY_ITEM)]

    same = bool(others(original)) and others(original) == others(fresh)
    r.item("every original F1 item other than the privacy counts is reproduced unchanged by this code, and passes",
           {"items": len(others(fresh)), "identical": same}, same and all(x.startswith("PASS ") for x in others(fresh)))
    expected = _f1_files(run / "f1")
    value, passed = _count_check(_counts(recheck, NAME_COUNTS), expected, ("name", "nonconforming"))
    r.item("names counted again with the corrected method, and every file fits its profile: every F1 evidence file "
           "counted with each, all 0", value, passed)
    source = _sha256(run / CARRIED_SOURCE)
    in_manifest = listed is not None and listed.get(CARRIED_SOURCE) == source
    carried = _counts(run / "f1", "secret-counts-identity.txt")
    if isinstance(carried, dict):
        carried = {k: v for k, v in carried.items() if k.rsplit(" ", 1)[1] in ("passphrase", "filenames")}
    value, passed = _count_check(carried, expected, ("passphrase", "filenames"))
    if isinstance(value, dict):
        value.update(source=CARRIED_SOURCE, sha256=source, verified_by_run_manifest=in_manifest, rescanned=False)
    r.item("passphrase and photo-file-name counts carried forward from the run's own counts file, not rescanned: the "
           "file verified by the run's SHA256SUMS, every F1 evidence file counted, all 0", value,
           passed and in_manifest)
    identity = _enrolled_id(run / "f1")
    want = {"run": run.name, "run_manifest_sha256": _sha256(run / "SHA256SUMS"), "identity_id": identity,
            "audit_sha256": _sha256(identity_dir / "audit.jsonl"), "carried_source_sha256": source,
            "names": "recounted", "passphrase_counts": "carried_forward", "filename_counts": "carried_forward"}
    recorded = _key_values(recheck / "provenance.txt")
    got = {key: recorded.get(key, MISSING) for key in want}
    r.item("the recheck's provenance names this run, its manifest, the enrolled identity and the audit log as they are "
           "now, and what was carried forward", got, got == want and identity is not None)
    return r


def f1_gate(run: Path, identity_dir: Path, recheck_root: Path | None) -> tuple[bool, list[str]]:
    """Whether F2 may build on this run's F1, as ``key=value`` lines: its own F1 validated, or exactly one corrected
    F1 reading of it that the maintainer explicitly accepted and that still matches the run, its manifest, the
    enrolled identity and the audit log, and still validates."""

    def refused(reason: str) -> tuple[bool, list[str]]:
        return False, ["f1_gate=refused", f"f1_gate_reason={reason}"]

    listed, problems = _manifest(run)
    if listed is None or problems or not all(e in listed for e in RUN_EVIDENCE):
        return refused("run_evidence_changed")
    identity = _enrolled_id(run / "f1")
    if identity is None:
        return refused("no_enrolled_identity")
    if not (identity_dir / "gallery.sealed").is_file():
        return refused("gallery_missing")
    exits = [line.split("=", 1)[1] for line in (run / "provenance.txt").read_text().splitlines()
             if line.startswith("F1_exit=")]
    head = ["f1_gate=passed", f"run={run.name}"]
    if exits and exits[-1] == "0":
        if _last(run / "f1/check.txt") != "face-f1: validated":
            return refused("f1_check_not_validated")
        return True, [*head, "f1_basis=original", f"f1_identity_id={identity}"]
    candidates = sorted(p for p in recheck_root.glob(f"{run.name}-f1-*") if (p / ACCEPTANCE).is_file()) \
        if recheck_root is not None and recheck_root.is_dir() else []
    if len(candidates) != 1:
        return refused("no_accepted_recheck" if not candidates else "several_accepted_rechecks")
    recheck = candidates[0]
    if not re.fullmatch(face_evidence.RECHECK_NAME, recheck.name):
        return refused("recheck_name")
    rlisted, rproblems = _manifest(recheck, ignore=frozenset({ACCEPTANCE}))
    if rlisted is None or rproblems:
        return refused("recheck_changed")
    if _last(recheck / "check.txt") != RECHECK_VERDICT:
        return refused("recheck_not_validated")
    if not read_f1_recheck(recheck, run, identity_dir).ok:
        return refused("recheck_no_longer_validates")
    accepted = _key_values(recheck / ACCEPTANCE)
    want = {"accepted_reading": "f1-recheck", "run": run.name, "identity_id": identity,
            "run_manifest_sha256": _sha256(run / "SHA256SUMS"),
            "recheck_manifest_sha256": _sha256(recheck / "SHA256SUMS"),
            "recheck_check_sha256": _sha256(recheck / "check.txt"),
            "audit_sha256": _sha256(identity_dir / "audit.jsonl"), "confirmation": f"ACCEPT {run.name} {identity}"}
    if any(accepted.get(k) != v for k, v in want.items()) or not face_evidence.ISO.fit(accepted.get("accepted_utc"))[0]:
        return refused("acceptance_does_not_match")
    return True, [*head, "f1_basis=accepted_recheck", f"f1_recheck={recheck.name}",
                  f"f1_accepted_utc={accepted['accepted_utc']}", f"f1_identity_id={identity}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="part", required=True)
    f1 = commands.add_parser("f1", help="read the consented enrollment's evidence")
    f1.add_argument("directory", type=Path)
    f1.add_argument("--identity-dir", type=Path, required=True)
    f2 = commands.add_parser("f2", help="read the guarded live validation's evidence")
    f2.add_argument("directory", type=Path)
    f2.add_argument("--f1", type=Path, required=True, help="F1's evidence directory (the enrolled ID)")
    f2.add_argument("--run", type=Path, help="with --identity-dir: also require this run's F1 basis (see gate)")
    f2.add_argument("--identity-dir", type=Path)
    f2.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are (see gate)")
    natural = commands.add_parser("natural", help="read the natural seated check's evidence (descriptive, D86)")
    natural.add_argument("directory", type=Path)
    natural.add_argument("--f1", type=Path, required=True, help="F1's evidence directory")
    natural.add_argument("--run", type=Path, help="with --identity-dir: also require this run's F1 basis (see gate)")
    natural.add_argument("--identity-dir", type=Path)
    natural.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are (see gate)")
    comparison = commands.add_parser("comparison", help="read the natural recognition comparison (descriptive, D87)")
    comparison.add_argument("directory", type=Path)
    comparison.add_argument("--f1", type=Path, required=True, help="F1's evidence directory")
    comparison.add_argument("--run", type=Path, help="with --identity-dir: also require this run's F1 basis (see gate)")
    comparison.add_argument("--identity-dir", type=Path)
    comparison.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are (see gate)")
    candidate = commands.add_parser("candidate", help="read the full-scale detection candidate's natural check "
                                     "(descriptive, D88)")
    candidate.add_argument("directory", type=Path)
    candidate.add_argument("--f1", type=Path, required=True, help="F1's evidence directory")
    candidate.add_argument("--run", type=Path, help="with --identity-dir: also require this run's F1 basis (see gate)")
    candidate.add_argument("--identity-dir", type=Path)
    candidate.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are (see gate)")
    for part, text in (("capture", "read the enrollment experiment's capture (D91): whether its working gallery may "
                                    "be used"),
                       ("experiment", "read an enrollment experiment session (D91, descriptive): genuine (format 6) "
                                      "or a non-enrolled participant's (format 8)")):
        sub = commands.add_parser(part, help=text)
        sub.add_argument("directory", type=Path)
        sub.add_argument("--f1", type=Path, required=True, help="F1's evidence directory")
        sub.add_argument("--run", type=Path, help="with --identity-dir: also require this run's F1 basis (see gate)")
        sub.add_argument("--identity-dir", type=Path)
        sub.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are (see gate)")
    recheck = commands.add_parser("f1-recheck", help="a corrected F1 reading of a run, made afterwards")
    recheck.add_argument("directory", type=Path, help="the recheck directory (face_recheck.py run)")
    recheck.add_argument("--run", type=Path, required=True)
    recheck.add_argument("--identity-dir", type=Path, required=True)
    gate = commands.add_parser("gate", help="whether F2 may build on a run's F1 (key=value lines; exit 0 if so)")
    gate.add_argument("run", type=Path)
    gate.add_argument("--identity-dir", type=Path, required=True)
    gate.add_argument("--recheck-root", type=Path)
    args = parser.parse_args(argv)
    if args.part == "gate":
        ok, lines = f1_gate(args.run, args.identity_dir, args.recheck_root)
        print("\n".join(lines))
        return 0 if ok else 1
    if args.part == "f1":
        reading, verdict = read_f1(args.directory, args.identity_dir), "face-f1"
    elif args.part == "f1-recheck":
        reading, verdict = read_f1_recheck(args.directory, args.run, args.identity_dir), "face-f1-recheck"
    else:
        if (args.run is None) != (args.identity_dir is None):
            parser.error(f"{args.part}: --run and --identity-dir go together")
        gate_args = None if args.run is None else (args.run, args.identity_dir, args.recheck_root)
        if args.part in ("natural", "comparison", "candidate", "experiment"):
            read = {"natural": read_natural, "comparison": read_comparison, "candidate": read_candidate,
                    "experiment": read_experiment}[args.part]
            reading = read(args.directory, args.f1, gate_args)
            for line in reading.lines:
                print(line)
            print(f"face-{args.part}: " + ("execution complete (descriptive only; it validates nothing)"
                                           if reading.ok else "execution incomplete"))
            return 0 if reading.ok else 1
        if args.part == "capture":
            reading = read_capture(args.directory, args.f1, gate_args)
            for line in reading.lines:
                print(line)
            print("face-capture: " + ("the working gallery may be used in the experiment (it validates nothing)"
                                      if reading.ok else "the working gallery is not usable"))
            return 0 if reading.ok else 1
        reading, verdict = read_f2(args.directory, args.f1, gate_args), "face-f2"
    for line in reading.lines:
        print(line)
    print(f"{verdict}: {'validated' if reading.ok else 'not validated'}")
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
