"""Operator account entry for a 3TC run (session 36): prompts instead of editing the raw template.

    python3 benchmarks/runner/operator_account.py enter RUN_DIR
    python3 benchmarks/runner/operator_account.py check RUN_DIR

``enter`` asks for each item of 3TC-5's account in turn and writes
``RUN_DIR/observations.txt`` in the template's own format (same keys, order and
comments), so 3TC-6 reads it unchanged. It:
- refuses before the run has finished (no ``end_utc`` in ``marks.txt``) and
  whenever the account is locked (``observations_sha256`` recorded in
  ``provenance.txt``); a locked account is never changed;
- replaces an unlocked account only after showing it and an explicit REPLACE,
  and only if it has not changed meanwhile;
- accepts ``unknown`` for every item and never fills in a value: each
  stopwatch reading is the operator's own m:ss reading, cumulative from 0:00;
- shows every incomplete, invalid or out-of-order item before anything is
  saved, and saves only on an explicit SAVE.

``check`` prints the same review of a saved account, read only, with its lock
state; it can run after the account is locked and before any count is shown.
Neither command reads ``b.json`` or shows a detection count. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ACCOUNT = "observations.txt"
LOCK_KEY = "observations_sha256"
MAX_NOTES = 500
HEADER = "# 3TC account entered with benchmarks/runner/operator_account.py; values as typed"
TIME = re.compile(r"(\d{1,2}):([0-5]\d)")
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Field:
    key: str
    kind: str  # "choice", "time" or "text"
    comment: str
    choices: tuple[str, ...] = ()


YES_NO = ("yes", "no", UNKNOWN)
FIELDS = (  # 3TC-5's template: keys, order and comments verbatim
    Field("stopwatch_started_at_prompt", "choice", "yes / no: you started it when NOW appeared", YES_NO),
    Field("out_of_view_1", "time", "stopwatch m:ss from which you were sure the camera could not see you, after 0:00"),
    Field("started_walking_in", "time", "m:ss when you started walking back in (schedule 3:00)"),
    Field("in_view_from", "time", "m:ss from which you were sure the camera could see you"),
    Field("in_view_until", "time",
          "m:ss when you stood up to leave (schedule 4:30); you were surely visible until then"),
    Field("out_of_view_2", "time", "m:ss from which you were sure the camera could not see you again"),
    Field("returned_at", "time", "m:ss when you came back into view (schedule 7:00)"),
    Field("anyone_else_entered_view", "choice", "anyone, or a pet, as far as you know: yes / no / unknown", YES_NO),
    Field("finished_shown_on_return", "choice", "FINISHED was on the screen when you came back: yes / no / unknown",
          YES_NO),
    Field("person_like_objects_or_reflections_noticed", "choice", "yes / no / unknown; describe in notes", YES_NO),
    Field("changes_noticed_on_return", "choice",
          "lighting, objects, screens or camera: yes / no / unknown; describe in notes", YES_NO),
    Field("followed_schedule", "choice", "yes / roughly / no", ("yes", "roughly", "no", UNKNOWN)),
    Field("notes", "text", ""),
)
BY_KEY = {field.key: field for field in FIELDS}
TIMES = tuple(field.key for field in FIELDS if field.kind == "time")
STRICT = (True, False, True, False, True)  # as 3TC-6: out_1 < walking_in <= in_from < in_until <= out_2 < returned
DESCRIBE = ("person_like_objects_or_reflections_noticed", "changes_noticed_on_return")
INTRO = (
    "Your 3TC account. Answer each item from what you noted; type unknown for anything you did not note or are\n"
    "not sure of. Stopwatch readings are cumulative from 0:00 (when NOW appeared), written m:ss with two-digit\n"
    "seconds, e.g. 3:07. Never enter a schedule time you did not read. Nothing here shows a detection count."
)


def seconds(value: str | None) -> int | None:
    """Seconds of an m:ss reading; None for unknown, missing or malformed."""
    match = TIME.fullmatch(value or "")
    return None if match is None else int(match[1]) * 60 + int(match[2])


def accepted(field: Field, raw: str) -> tuple[str | None, str | None]:
    """(value to write, None) for an acceptable answer, else (None, why it was not accepted)."""
    text = raw.strip()
    if field.kind == "text":
        if "#" in text:
            return None, "# is not allowed: it starts a comment, so everything after it would be lost"
        if any(not ch.isprintable() for ch in text):
            return None, "only printable characters on one line"
        if len(text) > MAX_NOTES:
            return None, f"at most {MAX_NOTES} characters"
        return text, None
    lowered = text.lower()
    if lowered in ("unknown", "u"):
        return UNKNOWN, None
    if field.kind == "time":
        if seconds(text) is None:
            return None, "write m:ss with two-digit seconds (e.g. 3:07), or unknown"
        return text, None
    if lowered not in field.choices:
        return None, "answer " + " / ".join(field.choices)
    return lowered, None


def parse(text: str) -> dict[str, str]:
    """key=value pairs as 3TC-6 reads them: the text after # is a comment."""
    out = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def render(values: dict[str, str]) -> str:
    lines = [HEADER]
    for field in FIELDS:
        value = values[field.key]
        lines.append(f"{field.key}={value}" + (f"   # {field.comment}" if field.comment else ""))
    return "\n".join(lines) + "\n"


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
    """Incomplete, invalid and out-of-order items; it checks the form of the answers, not whether they are true."""
    unknown, invalid = [], []
    for field in FIELDS:
        if field.kind == "text":
            continue
        value = values.get(field.key)
        if value is None or value == "" or value == UNKNOWN:
            unknown.append(field.key + (" (missing)" if value is None else " (empty)" if value == "" else ""))
        else:
            form, why = accepted(field, value)
            if form != value:  # 3TC-6 compares the exact text
                why = why or f"3TC-6 reads the exact text; the accepted form is {form}"
                invalid.append(f"{field.key}={value} ({why})")
    order = order_problems(values)
    lines = [
        "Incomplete (unknown): " + (", ".join(unknown) if unknown else "none"),
        "Not an accepted value: " + ("; ".join(invalid) if invalid else "none"),
        "Times not in order: " + ("; ".join(order) if order else "none among the times given"),
    ]
    asked = [key for key in DESCRIBE if values.get(key) == "yes"]
    if asked and not values.get("notes"):
        lines.append("Notes: empty, although " + " and ".join(asked) + " is yes (describe in notes)")
    lines.append("Complete and in order: " + ("yes" if not (unknown or invalid or order) else "no"))
    return lines


def listing(values: dict[str, str]) -> list[str]:
    return [f"  {n:2d} {field.key:<44} {values.get(field.key, '(missing)') or '(empty)'}"
            for n, field in enumerate(FIELDS, 1)]


def lock_state(run_dir: Path) -> str | None:
    """The recorded account hash, or None if provenance.txt records none."""
    for line in (run_dir / "provenance.txt").read_text().splitlines():
        if line.startswith(LOCK_KEY + "="):
            return line.split("=", 1)[1].strip()
    return None


def not_ready(run_dir: Path) -> str | None:
    if not run_dir.is_dir():
        return f"{run_dir} is not a run directory"
    if not (run_dir / "provenance.txt").is_file():
        return "provenance.txt is missing, so the lock state cannot be checked"
    marks = run_dir / "marks.txt"
    if not marks.is_file() or "end_utc" not in parse(marks.read_text()):
        return "the run has not finished (no end_utc in marks.txt)"
    return None


def _ask_field(field: Field, number: int, ask: Callable[[str], str], say: Callable[[str], None]) -> str:
    hint = {"time": "m:ss or unknown", "text": "one line, may be empty"}.get(field.kind, " / ".join(field.choices))
    say(f"[{number}/{len(FIELDS)}] {field.key}" + (f": {field.comment}" if field.comment else ""))
    while True:
        value, why = accepted(field, ask(f"  ({hint}) > "))
        if why is None and (value or field.kind == "text"):
            return value
        say("  not accepted: " + (why or "type a value, or unknown"))


def _write(path: Path, data: bytes, previous: bytes | None) -> str | None:
    """Write the account; never over a file that appeared or changed since it was shown."""
    if previous is None:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            return f"{ACCOUNT} appeared meanwhile"
        target = None
    else:
        if not path.is_file() or path.read_bytes() != previous:
            return f"{ACCOUNT} changed meanwhile"
        target = path.with_name(f".{ACCOUNT}.{os.getpid()}.tmp")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if target is not None:
        os.replace(target, path)
    return None


def enter(run_dir: Path, ask: Callable[[str], str] = input, say: Callable[[str], None] = print) -> int:
    problem = not_ready(run_dir)
    if problem is not None:
        say(f"STOP: {problem}. Nothing written.")
        return 2
    if lock_state(run_dir) is not None:
        say("STOP: this account is locked (observations_sha256 is recorded in provenance.txt); "
            "a locked account is never changed. Nothing written.")
        return 2
    path = run_dir / ACCOUNT
    previous = path.read_bytes() if path.exists() else None
    try:
        if previous is not None:
            shown = parse(previous.decode("utf-8", "replace"))
            say(f"An unlocked account already exists in {path}:")
            for line in listing(shown) + review(shown):
                say(line)
            if ask("Type REPLACE to write a new account in its place, or press Enter to leave it unchanged: ").strip() \
                    != "REPLACE":
                say("Unchanged. Nothing written.")
                return 1
        say(INTRO)
        values = {field.key: _ask_field(field, n, ask, say) for n, field in enumerate(FIELDS, 1)}
        while True:
            say("Your account (not saved yet):")
            for line in listing(values) + review(values):
                say(line)
            choice = ask("Type SAVE to write it, a number to re-enter that item, or QUIT to leave without writing: ")
            choice = choice.strip()
            if choice == "SAVE":
                break
            if choice == "QUIT":
                say("Nothing written.")
                return 1
            if choice.isdigit() and 1 <= int(choice) <= len(FIELDS):
                field = FIELDS[int(choice) - 1]
                values[field.key] = _ask_field(field, int(choice), ask, say)
            else:
                say("Not understood: type SAVE, QUIT or an item number.")
    except (EOFError, KeyboardInterrupt):
        say("\nAborted. Nothing written.")
        return 1
    if lock_state(run_dir) is not None:
        say("STOP: the account was locked meanwhile. Nothing written.")
        return 2
    problem = _write(path, render(values).encode(), previous)
    if problem is not None:
        say(f"STOP: {problem}. Nothing written.")
        return 2
    say(f"Saved to {path}. It is locked when its SHA-256 is recorded in provenance.txt "
        "(3TC-6 records it before showing any count).")
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
    recorded = lock_state(run_dir) if (run_dir / "provenance.txt").is_file() else None
    if recorded is None:
        say("Locked: no (no observations_sha256 in provenance.txt)")
    elif hashlib.sha256(data).hexdigest() == recorded:
        say("Locked: yes; the file matches its recorded SHA-256")
    else:
        say("Locked: yes; THE FILE DIFFERS FROM ITS RECORDED SHA-256")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enter or check a 3TC operator account; shows no detection count.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("enter", help="answer the account's items one by one; never changes a locked account") \
        .add_argument("run_dir", type=Path)
    commands.add_parser("check", help="show a saved account's incomplete, invalid or out-of-order items") \
        .add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    return enter(args.run_dir) if args.command == "enter" else check(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
