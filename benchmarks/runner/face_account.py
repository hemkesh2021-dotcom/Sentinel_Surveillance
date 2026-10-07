#!/usr/bin/env python3
"""The F2 operator account (V2-25 live validation): what the operator actually did, entered before any result.

    python3 benchmarks/runner/face_account.py enter F2_DIR
    python3 benchmarks/runner/face_account.py check F2_DIR

``enter`` asks for each item in turn and writes ``F2_DIR/account.txt`` (key=value). Times are the operator's own
stopwatch readings, m:ss cumulative from the T0 cue (the guard prints it when the runtime is ready); ``unknown`` is
always accepted and nothing is ever filled in from the schedule. It:
- refuses until the guarded run has finished (``result.json`` exists) and whenever the account is locked;
- reads no runtime output: not ``run.jsonl``, the status captures or ``result.json``'s contents, so no identity
  result can be seen before the account is saved;
- shows incomplete, invalid and out-of-order items before saving, and saves only on an explicit SAVE;
- **locks the account as it saves it**: ``account_sha256=`` is appended to ``provenance.txt``. A locked account is
  never changed, and face_check reads F2 only when the file still matches that hash.

``check`` shows a saved account, read only, with its lock state. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path

from operator_account import UNKNOWN, Field, accepted, parse, seconds

ACCOUNT = "account.txt"
LOCK_KEY = "account_sha256"
HEADER = "# F2 account entered with benchmarks/runner/face_account.py; values as typed"
YES_NO = ("yes", "no", UNKNOWN)
FIELDS = (
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
TIMES = ("out_of_view_1", "seated_facing_1", "turned_away", "stood_up", "out_of_view_2", "seated_facing_2")
STRICT = (True, True, True, False, True)  # out_1 < facing_1 < turned < stood <= out_2 < facing_2
INTRO = (
    "Your F2 account. Answer from what you noted; type unknown for anything you did not note or are not sure of.\n"
    "Stopwatch readings are cumulative from 0:00, when the T0 cue appeared, written m:ss with two-digit seconds\n"
    "(e.g. 3:07). Never enter a schedule time you did not read. Nothing here shows an identity result."
)


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


def review(values: dict[str, str]) -> list[str]:
    """Incomplete, invalid and out-of-order items: the form of the answers, not whether they are true."""
    unknown, invalid = [], []
    for field in FIELDS:
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


def render(values: dict[str, str]) -> str:
    lines = [HEADER] + [f"{f.key}={values[f.key]}" + (f"   # {f.comment}" if f.comment else "") for f in FIELDS]
    return "\n".join(lines) + "\n"


def listing(values: dict[str, str]) -> list[str]:
    return [f"  {n:2d} {f.key:<26} {values.get(f.key, '(missing)') or '(empty)'}" for n, f in enumerate(FIELDS, 1)]


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


def _ask(field: Field, number: int, ask: Callable[[str], str], say: Callable[[str], None]) -> str:
    hint = {"time": "m:ss or unknown", "text": "one line, may be empty"}.get(field.kind, " / ".join(field.choices))
    say(f"[{number}/{len(FIELDS)}] {field.key}" + (f": {field.comment}" if field.comment else ""))
    while True:
        value, why = accepted(field, ask(f"  ({hint}) > "))
        if why is None and (value or field.kind == "text"):
            return value
        say("  not accepted: " + (why or "type a value, or unknown"))


def enter(run_dir: Path, ask: Callable[[str], str] = input, say: Callable[[str], None] = print) -> int:
    problem = not_ready(run_dir)
    if problem is not None:
        say(f"STOP: {problem}. Nothing written.")
        return 2
    path = run_dir / ACCOUNT
    if lock_state(run_dir) is not None or path.exists():
        say("STOP: an account already exists or is locked; it is never changed. Nothing written.")
        return 2
    try:
        say(INTRO)
        values = {field.key: _ask(field, n, ask, say) for n, field in enumerate(FIELDS, 1)}
        while True:
            say("Your account (not saved yet):")
            for line in listing(values) + review(values):
                say(line)
            choice = ask("Type SAVE to write and lock it, a number to re-enter that item, or QUIT: ").strip()
            if choice == "SAVE":
                break
            if choice == "QUIT":
                say("Nothing written.")
                return 1
            if choice.isdigit() and 1 <= int(choice) <= len(FIELDS):
                field = FIELDS[int(choice) - 1]
                values[field.key] = _ask(field, int(choice), ask, say)
            else:
                say("Not understood: type SAVE, QUIT or an item number.")
    except (EOFError, KeyboardInterrupt):
        say("\nAborted. Nothing written.")
        return 1
    data = render(values).encode()
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
    values = parse(data.decode("utf-8", "replace"))
    say(f"Account in {path} (read only):")
    for line in listing(values) + review(values):
        say(line)
    recorded = lock_state(run_dir)
    if recorded is None:
        say("Locked: no")
    elif hashlib.sha256(data).hexdigest() == recorded:
        say("Locked: yes; the file matches its recorded SHA-256")
    else:
        say("Locked: yes; THE FILE DIFFERS FROM ITS RECORDED SHA-256")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enter or check the F2 operator account; shows no identity result.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("enter", help="answer each item; saving locks the account").add_argument("run_dir", type=Path)
    commands.add_parser("check", help="show a saved account and its lock state").add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    return enter(args.run_dir) if args.command == "enter" else check(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
