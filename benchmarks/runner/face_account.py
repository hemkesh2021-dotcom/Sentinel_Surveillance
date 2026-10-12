#!/usr/bin/env python3
"""The F2 operator account (V2-25 live validation): what the operator actually did, entered before any result.

    python3 benchmarks/runner/face_account.py enter F2_DIR [--format 3|4]
    python3 benchmarks/runner/face_account.py check F2_DIR

``enter`` asks for each item in turn and writes ``F2_DIR/account.txt`` (key=value) in **format 2** (D66):
- how many whole seconds after the T0 cue the stopwatch was started (0 if at the cue, as far as the operator can tell),
  then how many whole seconds (1 to 5) that may be off by;
- each movement's stopwatch reading, m:ss: the stopwatch's cumulative time since it was started (never a lap's own
  duration), then, asked separately, how many whole seconds (1 to 5) that reading may be off by. A reading the
  operator cannot bound within 5 s is ``unknown``; the uncertainty of an ``unknown`` reading is ``unknown`` without
  being asked.

``unknown`` is always accepted, and nothing is ever filled in from the schedule: scheduled times appear in the prompts
as targets only. The helper does no arithmetic; face_check aligns the readings to T0. It:
- refuses until the guarded run has finished (``result.json`` exists) and whenever the account is locked;
- reads no runtime output: not ``run.jsonl``, the status captures or ``result.json``'s contents, so no identity
  result can be seen before the account is saved;
- shows incomplete, invalid and out-of-order items before saving, and saves only on an explicit SAVE;
- **locks the account as it saves it**: ``account_sha256=`` is appended to ``provenance.txt``. A locked account is
  never changed, and face_check reads F2 only when the file still matches that hash.

**Format 1** (the ``face-0f7855e`` package and earlier: a yes/no stopwatch answer and readings taken as exact) is kept
read only, with its keys, comments and header unchanged, so accounts saved in it keep their original reading:
``check`` and face_check tell the formats apart by the header line.

**Format 3** (D79, PROPOSED, only with the maintainer's approval; ``enter --format 3``): cued, with no stopwatch. The
guard prints each movement's cue and records when (face_check reads those times from ``result.json``); the operator
only answers, for each movement, whether it was done within its allowance after its cue, and not before it (yes / no /
unknown), plus the same three questions about the face and the room. Each movement's interval is then [cue, cue +
allowance]; the judged windows, their washout and minimum, and every other criterion are unchanged.

**Format 4** (D85, the maintainer's instruction, 2026-10-12; ``enter --format 4``): format 3 with the 4:00 posture
stated, in the cue and in the account alike: a head turn within 5 s, seated, the torso upright where it was and the
chair not moved until 6:00, and no face, not even in profile, visible to the camera. Same keys, cues, allowances,
windows and criteria as format 3; only those two questions' wording differs.

**Format 6** (D87; ``enter --format 6``): the natural recognition comparison's account, not an F2 account. No timed
movement: the operator sits and works as usual for the whole run. Four separate questions (stayed seated; worked as
usual; looked toward the camera on purpose; other faces in view), each yes / no / unknown. ``unknown`` is an accepted
answer here, not an incomplete account: face_check's comparison reading qualifies only the lines that depend on it.

**Formats 7 and 8** (D91, the camera-side enrollment experiment; ``enter --format 7`` or ``8``): the capture's account
(consent to the capture, stayed seated, worked as usual, other faces in view) and a non-enrolled participant's session
(not enrolled, consented, stayed seated, looked toward the camera at the 2:30 cue, nobody else in view), each a set of
separate yes / no / unknown answers as in format 6. Never a name.

``check`` shows a saved account, read only, with its lock state. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from operator_account import UNKNOWN, Field, parse, seconds
from operator_account import accepted as _accepted

ACCOUNT = "account.txt"
LOCK_KEY = "account_sha256"
YES_NO = ("yes", "no", UNKNOWN)
TIMES = ("out_of_view_1", "seated_facing_1", "turned_away", "stood_up", "out_of_view_2", "seated_facing_2")
STRICT = (True, True, True, False, True)  # out_1 < facing_1 < turned < stood <= out_2 < facing_2

# Format 1, read only: the accounts of the face-0f7855e package and earlier. Keys, comments and header verbatim.
HEADER_V1 = "# F2 account entered with benchmarks/runner/face_account.py; values as typed"
FIELDS_V1 = (
    Field("stopwatch_started_at_t0", "choice", "yes / no: you started the stopwatch when the T0 cue appeared", YES_NO),
    Field("out_of_view_1", "time", "m:ss from which you were sure the camera could not see you (schedule: by 0:20)"),
    Field("seated_facing_1", "time", "m:ss from which you sat at the Mac with your face toward the camera "
          "(schedule 2:00)"),
    Field("turned_away", "time", "m:ss from which your face was turned away from the camera, still seated "
          "(schedule 4:00)"),
    Field("face_hidden_while_turned", "choice", "yes / no: the camera could not see your face at all while turned "
          "away", YES_NO),
    Field("stood_up", "time", "m:ss when you stood up to leave (schedule 6:00)"),
    Field("out_of_view_2", "time", "m:ss from which you were sure the camera could not see you again"),
    Field("seated_facing_2", "time", "m:ss from which you sat facing the camera again (schedule 8:00)"),
    Field("stayed_until_stop", "choice", "yes / no: you stayed seated facing the camera until the stop cue (10:00)",
          YES_NO),
    Field("other_faces_in_view", "choice", "yes / no: anyone else, or a photo or screen showing a face, was in the "
          "camera's view at any time", YES_NO),
    Field("notes", "text", ""),
)

# Format 2 (D66, maintainer, 2026-10-08; uncalibrated, for this bounded test only).
HEADER = "# F2 account, format 2, entered with benchmarks/runner/face_account.py; values as typed"
DELAY = "stopwatch_start_delay_s"
DELAY_ENTRY_MAX_S = 59  # what can be typed; face_check's criterion is the upper end (delay + uncertainty) <= 5
UNCERTAINTY_S = (1, 2, 3, 4, 5)  # whole seconds a reading or the delay may be off by (D66)
_UNCERTAINTY_NOTE = "how many whole seconds (1 to 5) that {} may be off by"


def uncertainty_key(key: str) -> str:
    return f"{key}_uncertainty_s"


DELAY_UNCERTAINTY = "stopwatch_start_delay_uncertainty_s"
_MOVES = (
    ("out_of_view_1", "from which you were sure the camera could not see you (target: by 0:20)"),
    ("seated_facing_1", "from which you sat at the Mac with your face toward the camera (target 2:00)"),
    ("turned_away", "from which your face was turned away from the camera, still seated (target 4:00)"),
    ("stood_up", "when you stood up to leave (target 6:00)"),
    ("out_of_view_2", "from which you were sure the camera could not see you again"),
    ("seated_facing_2", "from which you sat facing the camera again (target 8:00)"),
)


def _move(key: str) -> tuple[Field, Field]:
    what = dict(_MOVES)[key]
    return (Field(key, "time", f"stopwatch m:ss, cumulative, {what}"),
            Field(uncertainty_key(key), "uncertainty", _UNCERTAINTY_NOTE.format("reading")))


FIELDS = (
    Field(DELAY, "seconds", "whole seconds after the T0 cue when you started the stopwatch (0 if at the cue, as far "
          "as you can tell)"),
    Field(DELAY_UNCERTAINTY, "uncertainty", _UNCERTAINTY_NOTE.format("delay")),
    *_move("out_of_view_1"), *_move("seated_facing_1"), *_move("turned_away"),
    FIELDS_V1[4],  # face_hidden_while_turned
    *_move("stood_up"), *_move("out_of_view_2"), *_move("seated_facing_2"),
    *FIELDS_V1[8:],  # stayed_until_stop, other_faces_in_view, notes
)
# An uncertainty is asked only when its reading is known: the uncertainty key -> its reading's key.
PAIRED = {DELAY_UNCERTAINTY: DELAY, **{uncertainty_key(key): key for key in TIMES}}
UNCERTAINTY_OF = {reading: uncertainty for uncertainty, reading in PAIRED.items()}

# Format 3 (D79, PROPOSED: used only with the maintainer's approval; uncalibrated, for this bounded test only).
HEADER_V3 = "# F2 account, format 3 (cued), entered with benchmarks/runner/face_account.py; values as typed"
CUE_OF = {"out_of_view_1": "t0", "seated_facing_1": "k1", "turned_away": "a", "stood_up": "e2",
          "out_of_view_2": "e2", "seated_facing_2": "k2"}  # the guard's cue that starts each movement
CUE_AT_S = {"t0": 0, "k1": 120, "a": 240, "e2": 360, "k2": 480}  # the cues' offsets from T0
ALLOWANCE_S = {"out_of_view_1": 20, "seated_facing_1": 20, "turned_away": 5, "stood_up": 5, "out_of_view_2": 20,
               "seated_facing_2": 20}  # each movement is done within this long after its cue


def within_key(key: str) -> str:
    return f"{key}_within_{ALLOWANCE_S[key]}_s"


_CUED = {
    "out_of_view_1": "within 20 s after the T0 cue you were where the camera surely could not see you",
    "seated_facing_1": "within 20 s after the 2:00 cue you sat at the Mac with your face toward the camera",
    "turned_away": "within 5 s after the 4:00 cue your face was turned away from the camera, still seated",
    "stood_up": "within 5 s after the 6:00 cue you stood up to leave",
    "out_of_view_2": "within 20 s after the 6:00 cue you were again where the camera surely could not see you",
    "seated_facing_2": "within 20 s after the 8:00 cue you sat facing the camera again",
}


def _cued(key: str) -> Field:
    return Field(within_key(key), "choice", f"yes / no: {_CUED[key]}, and not before that cue", YES_NO)


FIELDS_V3 = (
    *(_cued(key) for key in TIMES[:3]), FIELDS_V1[4],  # face_hidden_while_turned
    *(_cued(key) for key in TIMES[3:]), *FIELDS_V1[8:],  # stayed_until_stop, other_faces_in_view, notes
)
INTRO_V3 = (
    "Your F2 account (format 3, cued). For each movement: did you do it within its allowance after its cue, and not\n"
    "before the cue? Answer yes or no, or unknown if you are not sure. Then the three questions about your face and the\n"
    "room. Nothing here shows an identity result."
)

# Format 4 (D85, the maintainer's instruction, 2026-10-12): format 3 with the 4:00 posture stated. The same keys, cues,
# allowances, windows and criteria; only the wording of the turn and of the face-hidden question differs, so that the
# account asks exactly what the 4:00 cue asks. Formats 1-3 are read exactly as before.
HEADER_V4 = ("# F2 account, format 4 (cued, head turned), entered with benchmarks/runner/face_account.py; "
             "values as typed")
TURN_V4 = Field(within_key("turned_away"), "choice", "yes / no: within 5 s after the 4:00 cue, and not before it, you "
                "turned your head away from the camera, and until the 6:00 cue you stayed seated with your torso "
                "upright where it was while you faced the camera and the chair not moved", YES_NO)
HIDDEN_V4 = Field("face_hidden_while_turned", "choice", "yes / no: from your turn until the 6:00 cue the camera could "
                  "not see your face at all, not even in profile", YES_NO)
FIELDS_V4 = tuple({TURN_V4.key: TURN_V4, HIDDEN_V4.key: HIDDEN_V4}.get(f.key, f) for f in FIELDS_V3)
INTRO_V4 = (
    "Your F2 account (format 4, cued). For each movement: did you do it within its allowance after its cue, and not\n"
    "before the cue? At 4:00 the movement is a head turn: seated, torso upright and still, the chair not moved,\n"
    "no part of your face toward the camera, not even in profile. Answer yes or no, or unknown if you are not\n"
    "sure. Then the three questions about your face and the room. Nothing here shows an identity result."
)
CUED_FORMATS = (3, 4)  # face_check reads both alike: each movement's interval from the guard's cue times

# Format 5 (D86): the natural seated check's account, not an F2 account. The operator sits and moves as usual; the
# guard cues four phases; the reading is descriptive (face_check natural) and validates nothing.
HEADER_V5 = ("# natural seated check account, format 5 (cued), entered with benchmarks/runner/face_account.py; "
             "values as typed")
NATURAL_CUE_AT_S = {"t0": 0, "look": 120, "turn": 210, "n2": 300}  # the guard's cues; the run stops at 390 s
NATURAL_ALLOWANCE_S = 10  # each cued movement is done within this long after its cue
NATURAL_MOVES = {"looked_at_camera_within_10_s": "look", "swivelled_away_within_10_s": "turn",
                 "back_as_usual_within_10_s": "n2"}
FIELDS_V5 = (
    Field("seated_as_usual_from_t0", "choice", "yes / no: from the T0 cue until the 2:00 cue you sat and worked at the "
          "Mac as you normally do, looking at the Mac", YES_NO),
    Field("looked_at_camera_within_10_s", "choice", "yes / no: within 10 s after the 2:00 cue, and not before it, you "
          "looked toward the camera, and kept looking toward it until the 3:30 cue", YES_NO),
    Field("swivelled_away_within_10_s", "choice", "yes / no: within 10 s after the 3:30 cue, and not before it, you "
          "swivelled the chair away as you normally would, and stayed turned away until the 5:00 cue", YES_NO),
    Field("back_as_usual_within_10_s", "choice", "yes / no: within 10 s after the 5:00 cue, and not before it, you "
          "swivelled back and worked at the Mac as you normally do until the stop", YES_NO),
    *FIELDS_V1[9:],  # other_faces_in_view, notes
)
INTRO_V5 = (
    "Your natural seated check account (format 5). For each phase: did you do as its cue said, within 10 s after the\n"
    "cue and not before it? Answer yes or no, or unknown if you are not sure. Then whether anyone else was in view.\n"
    "Nothing here shows an identity result."
)
# Format 6 (D87): the natural recognition comparison's account. Separate observations about the whole run, so an
# uncertain answer qualifies only the readings that depend on it (face_check comparison); unknown is accepted.
HEADER_V6 = ("# natural recognition comparison account, format 6, entered with benchmarks/runner/face_account.py; "
             "values as typed")
FIELDS_V6 = (
    Field("stayed_seated", "choice", "yes / no: from the T0 cue until the run stopped by itself you stayed seated in "
          "your usual chair and place, and never stood up or left", YES_NO),
    Field("worked_as_usual", "choice", "yes / no: from the T0 cue until the run stopped you worked at the Mac as you "
          "normally do (glances anywhere are part of that)", YES_NO),
    Field("looked_at_camera_on_purpose", "choice", "yes / no: at any time from the T0 cue until the run stopped you "
          "looked toward the camera on purpose", YES_NO),
    *FIELDS_V1[9:],  # other_faces_in_view, notes
)
INTRO_V6 = (
    "Your natural recognition comparison account (format 6). Four separate questions about the whole run. Answer each\n"
    "on its own: yes or no, or unknown if you are not sure; an unknown answer affects only the readings that depend\n"
    "on it. Nothing here shows an identity result."
)
COMPARISON_CUE_AT_S = {"t0": 0, "stop": 290}  # the guard's cues: T0, and the stop warning 10 s before the stop
COMPARISON_DURATION_S = 300  # the run stops by itself 300 s after T0
# Format 7 (D91): the camera-side enrollment capture's account: the enrolled person, alone, seated and working as
# usual, and their consent to this capture being kept in a separate working gallery. Separate answers; unknown accepted.
HEADER_V7 = ("# camera-side enrollment capture account, format 7, entered with benchmarks/runner/face_account.py; "
             "values as typed")
FIELDS_V7 = (
    Field("consented_to_capture", "choice", "yes / no: before the run you (the enrolled person) agreed that numbers "
          "computed from your face during this run may be kept in a separate working gallery for the enrollment "
          "experiment, revocable at any time", YES_NO),
    *FIELDS_V6[:2],  # stayed_seated, worked_as_usual
    *FIELDS_V1[9:],  # other_faces_in_view, notes
)
INTRO_V7 = (
    "Your camera-side enrollment capture account (format 7). Four separate questions about the whole run. Answer each\n"
    "on its own: yes or no, or unknown if you are not sure; the reading says what an unknown answer leaves open.\n"
    "Nothing here shows an identity result."
)
# Format 8 (D91): a session in which a consenting person who is not enrolled sits at the seat (the enrolled person out
# of view), for the experiment's false-acceptance numbers. Separate answers; unknown accepted.
HEADER_V8 = ("# non-enrolled participant session account, format 8, entered with benchmarks/runner/face_account.py; "
             "values as typed")
NON_ENROLLED_CUE_AT_S = {"t0": 0, "look": 150, "stop": 290}  # the guard's cues; the run stops at 300 s
NON_ENROLLED_LOOK_S = 10  # the participant looks toward the camera for about this long after the look cue
FIELDS_V8 = (
    Field("participant_not_enrolled", "choice", "yes / no: the person seated in the camera's view during the run is "
          "not enrolled in Sentinel (not the enrolled person, and never enrolled)", YES_NO),
    Field("participant_consented", "choice", "yes / no: before the run the participant agreed to sit in the camera's "
          "view for this check, knowing that only numbers are kept (no picture, no name)", YES_NO),
    Field("participant_stayed_seated", "choice", "yes / no: from the T0 cue until the run stopped by itself the "
          "participant stayed seated at the usual seat and did not leave", YES_NO),
    Field("participant_looked_at_cue", "choice", "yes / no: at the 2:30 cue the participant looked toward the camera "
          "for about 10 s, then carried on as before", YES_NO),
    Field("only_participant_in_view", "choice", "yes / no: from the T0 cue until the run stopped nobody else (you "
          "included), and no photo or screen showing a face, was in the camera's view", YES_NO),
    FIELDS_V1[-1],  # notes
)
INTRO_V8 = (
    "The non-enrolled participant session's account (format 8), answered with the participant. Five separate\n"
    "questions about the whole run: yes or no, or unknown if you are not sure. Never type a name. Nothing here shows\n"
    "an identity result."
)
SEPARATE_FORMATS = (6, 7, 8)  # formats whose answers are read one by one: unknown is accepted, never "incomplete"
FORMATS = {HEADER_V1: (1, FIELDS_V1), HEADER: (2, FIELDS), HEADER_V3: (3, FIELDS_V3), HEADER_V4: (4, FIELDS_V4),
           HEADER_V5: (5, FIELDS_V5), HEADER_V6: (6, FIELDS_V6), HEADER_V7: (7, FIELDS_V7), HEADER_V8: (8, FIELDS_V8)}
INTRO = (
    "Your F2 account (format 2). Answer from what you noted; type unknown for anything you did not note or are not\n"
    "sure of. First: how many whole seconds after the T0 cue you started the stopwatch (0 if at the cue, as far as\n"
    "you can tell). Then each stopwatch reading: the stopwatch's cumulative time since you started it (not a lap's\n"
    "own duration), m:ss with two-digit seconds (e.g. 3:07). After the delay and after each reading you are asked\n"
    "how many whole seconds (1 to 5) it may be off by; if you cannot say within 5 seconds, the reading is unknown.\n"
    "Never enter a target time you did not read. Nothing here shows an identity result."
)
_DELAY = re.compile(r"\d{1,2}")


def delay_seconds(value: str | None) -> int | None:
    """The stopwatch start delay as typed (whole seconds, 0..59); None for unknown, missing or malformed."""
    if value is None or not _DELAY.fullmatch(value) or str(int(value)) != value:
        return None
    return int(value) if int(value) <= DELAY_ENTRY_MAX_S else None


def uncertainty_seconds(value: str | None) -> int | None:
    """An uncertainty as typed (whole seconds, 1..5); None for unknown, missing or malformed."""
    return int(value) if value in {str(s) for s in UNCERTAINTY_S} else None


def accepted(field: Field, raw: str) -> tuple[str | None, str | None]:
    """(value to write, None) for an acceptable answer, else (None, why it was not accepted)."""
    text = raw.strip()
    if field.kind not in ("seconds", "uncertainty"):
        return _accepted(field, raw)
    if text.lower() in ("unknown", "u"):
        return UNKNOWN, None
    if field.kind == "seconds":
        if _DELAY.fullmatch(text) and int(text) <= DELAY_ENTRY_MAX_S:
            return str(int(text)), None
        return None, f"whole seconds from 0 to {DELAY_ENTRY_MAX_S} (e.g. 2), or unknown"
    if text in {str(s) for s in UNCERTAINTY_S}:
        return text, None
    return None, "whole seconds from 1 to 5 (e.g. 2); if you cannot say within 5 seconds, type unknown"


def account_format(text: str) -> tuple[int, tuple[Field, ...]] | None:
    """(format number, its fields) from an account's header line; None for an unrecognised header."""
    lines = text.splitlines()
    return FORMATS.get(lines[0]) if lines else None


def order_problems(values: dict[str, str]) -> list[str]:
    known = [(i, key, seconds(values.get(key))) for i, key in enumerate(TIMES)]
    known = [(i, key, s) for i, key, s in known if s is not None]
    problems = []
    for a, (i, first, s1) in enumerate(known):
        for j, second, s2 in known[a + 1:]:
            strict = any(STRICT[i:j])
            if s1 > s2 or (strict and s1 == s2):
                problems.append(f"{second} {values[second]} should be {'after' if strict else 'at or after'} "
                                f"{first} {values[first]}")
    return problems


def review(values: dict[str, str], fields: tuple[Field, ...] = FIELDS) -> list[str]:
    """Incomplete, invalid and out-of-order items: the form of the answers, not whether they are true. The order is
    that of the readings as typed (format 2 aligns them all by the same delay, so their order is the same)."""
    unknown, invalid = [], []
    for field in fields:
        if field.kind == "text":
            continue
        value = values.get(field.key)
        if value in (None, "", UNKNOWN):
            unknown.append(field.key + (" (missing)" if value is None else ""))
        elif accepted(field, value)[0] != value:
            invalid.append(f"{field.key}={value}")
    order = order_problems(values)
    return [
        "Incomplete (unknown): " + (", ".join(unknown) if unknown else "none"),
        "Not an accepted value: " + ("; ".join(invalid) if invalid else "none"),
        "Times not in order: " + ("; ".join(order) if order else "none among the times given"),
        "Complete and in order: " + ("yes" if not (unknown or invalid or order) else "no"),
    ]


def review_separate(values: dict[str, str], fields: tuple[Field, ...] = FIELDS_V6) -> list[str]:
    """Formats 6 to 8: every answer present and accepted (yes, no or unknown); unknown answers are listed, not
    refused."""
    unknown, invalid = [], []
    for field in fields:
        if field.kind == "text":
            continue
        value = values.get(field.key)
        if value is None or value == "" or accepted(field, value)[0] != value:
            invalid.append(f"{field.key}={value if value is not None else '(missing)'}")
        elif value == UNKNOWN:
            unknown.append(field.key)
    return [
        "Unknown (accepted; each reading names the answers it depends on): " + (", ".join(unknown) or "none"),
        "Not an accepted value: " + ("; ".join(invalid) if invalid else "none"),
        "Answered (yes, no or unknown everywhere): " + ("yes" if not invalid else "no"),
    ]


def render(values: dict[str, str], fields: tuple[Field, ...] = FIELDS, header: str = HEADER) -> str:
    lines = [header] + [f"{f.key}={values[f.key]}" + (f"   # {f.comment}" if f.comment else "") for f in fields]
    return "\n".join(lines) + "\n"


def listing(values: dict[str, str], fields: tuple[Field, ...] = FIELDS) -> list[str]:
    width = max(26, *(len(f.key) for f in fields))  # 26: format 1's listing, unchanged
    return [f"  {n:2d} {f.key:<{width}} {values.get(f.key, '(missing)') or '(empty)'}" for n, f in enumerate(fields, 1)]


def lock_state(run_dir: Path) -> str | None:
    provenance = run_dir / "provenance.txt"
    if not provenance.is_file():
        return None
    for line in provenance.read_text().splitlines():
        if line.startswith(LOCK_KEY + "="):
            return line.split("=", 1)[1].strip()
    return None


def not_ready(run_dir: Path) -> str | None:
    if not run_dir.is_dir():
        return f"{run_dir} is not a run directory"
    if not (run_dir / "provenance.txt").is_file():
        return "provenance.txt is missing, so the account cannot be locked"
    if not (run_dir / "result.json").is_file():
        return "the guarded run has not finished (no result.json)"
    return None


_HINTS = {"time": "m:ss or unknown", "text": "one line, may be empty", "seconds": "whole seconds, or unknown",
          "uncertainty": "1 to 5, or unknown"}


def _ask(field: Field, number: int, ask: Callable[[str], str], say: Callable[[str], None], total: int) -> str:
    hint = _HINTS.get(field.kind, " / ".join(field.choices))
    say(f"[{number}/{total}] {field.key}" + (f": {field.comment}" if field.comment else ""))
    while True:
        value, why = accepted(field, ask(f"  ({hint}) > "))
        if why is None and (value or field.kind == "text"):
            return value
        say("  not accepted: " + (why or "type a value, or unknown"))


def _answer(index: int, values: dict[str, str], ask: Callable[[str], str], say: Callable[[str], None],
            fields: tuple[Field, ...] = FIELDS) -> str:
    field = fields[index]
    reading = PAIRED.get(field.key)
    if reading is not None and values.get(reading) == UNKNOWN:
        say(f"[{index + 1}/{len(fields)}] {field.key}: unknown, because {reading} is unknown")
        return UNKNOWN
    return _ask(field, index + 1, ask, say, len(fields))


def enter(run_dir: Path, ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
          account_format_number: int = 2) -> int:
    """Format 2 (D66), or format 3 (D79, cued) or 4 (D85, cued, the 4:00 head turn stated) when asked for."""
    fields, header, intro = {3: (FIELDS_V3, HEADER_V3, INTRO_V3), 4: (FIELDS_V4, HEADER_V4, INTRO_V4),
                             5: (FIELDS_V5, HEADER_V5, INTRO_V5), 6: (FIELDS_V6, HEADER_V6, INTRO_V6),
                             7: (FIELDS_V7, HEADER_V7, INTRO_V7),
                             8: (FIELDS_V8, HEADER_V8, INTRO_V8)}.get(account_format_number, (FIELDS, HEADER, INTRO))
    checker = review_separate if account_format_number in SEPARATE_FORMATS else review
    problem = not_ready(run_dir)
    if problem is not None:
        say(f"STOP: {problem}. Nothing written.")
        return 2
    path = run_dir / ACCOUNT
    if lock_state(run_dir) is not None or path.exists():
        say("STOP: an account already exists or is locked; it is never changed. Nothing written.")
        return 2
    position = {field.key: n for n, field in enumerate(fields)}
    try:
        say(intro)
        values: dict[str, str] = {}
        for index in range(len(fields)):
            values[fields[index].key] = _answer(index, values, ask, say, fields)
        while True:
            say("Your account (not saved yet):")
            for line in listing(values, fields) + checker(values, fields):
                say(line)
            choice = ask("Type SAVE to write and lock it, a number to re-enter that item, or QUIT: ").strip()
            if choice == "SAVE":
                break
            if choice == "QUIT":
                say("Nothing written.")
                return 1
            if choice.isdigit() and 1 <= int(choice) <= len(fields):
                key = fields[int(choice) - 1].key
                values[key] = _answer(int(choice) - 1, values, ask, say, fields)
                if key in UNCERTAINTY_OF and UNCERTAINTY_OF[key] in position:  # a re-entered reading: ask again
                    values[UNCERTAINTY_OF[key]] = _answer(position[UNCERTAINTY_OF[key]], values, ask, say, fields)
            else:
                say("Not understood: type SAVE, QUIT or an item number.")
    except (EOFError, KeyboardInterrupt):
        say("\nAborted. Nothing written.")
        return 1
    data = render(values, fields, header).encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        say("STOP: an account appeared meanwhile. Nothing written.")
        return 2
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    with open(run_dir / "provenance.txt", "a") as provenance:
        provenance.write(f"{LOCK_KEY}={hashlib.sha256(data).hexdigest()}\n")
    say(f"Saved and locked: {path} (its SHA-256 is now in provenance.txt).")
    return 0


def check(run_dir: Path, say: Callable[[str], None] = print) -> int:
    path = run_dir / ACCOUNT
    if not path.is_file():
        say(f"No account: {path} does not exist.")
        return 1
    data = path.read_bytes()
    text = data.decode("utf-8", "replace")
    known = account_format(text)
    values = parse(text)
    say(f"Account in {path} (read only), " + (f"format {known[0]}:" if known else "an unrecognised format:"))
    fields = known[1] if known else FIELDS
    checker = review_separate if known and known[0] in SEPARATE_FORMATS else review
    for line in listing(values, fields) + checker(values, fields):
        say(line)
    recorded = lock_state(run_dir)
    if recorded is None:
        say("Locked: no")
    elif hashlib.sha256(data).hexdigest() == recorded:
        say("Locked: yes; the file matches its recorded SHA-256")
    else:
        say("Locked: yes; THE FILE DIFFERS FROM ITS RECORDED SHA-256")
    return 0 if known else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enter or check the F2 operator account; shows no identity result.")
    commands = parser.add_subparsers(dest="command", required=True)
    enter_parser = commands.add_parser("enter", help="answer each item; saving locks the account")
    enter_parser.add_argument("run_dir", type=Path)
    enter_parser.add_argument("--format", type=int, choices=(2, 3, 4, 5, 6, 7, 8), default=2,
                              help="2 (D66, the default), 3 (D79, cued), 4 (D85, cued, the 4:00 head turn stated), 5 "
                                   "(D86, the natural seated check), 6 (D87, the natural recognition comparison), 7 "
                                   "(D91, the enrollment experiment's capture) or 8 (D91, a non-enrolled participant's "
                                   "session); 3 to 8 only with the maintainer's approval")
    commands.add_parser("check", help="show a saved account and its lock state").add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "enter":
        return enter(args.run_dir, account_format_number=args.format)
    return check(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
