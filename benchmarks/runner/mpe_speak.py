"""Optional spoken cues for the multi-pose comparison's sessions (PROPOSED; timing validated synthetically only).

usage:
  mpe_speak.py follow CUES_LOG --session B|E_M|E_P --tts "COMMAND" [--log SPOKEN.jsonl]
  mpe_speak.py test --tts "COMMAND"
  mpe_speak.py check SPOKEN.jsonl RESULT_JSON

The guard prints each cue as it falls due (``\\a[m:ss] TEXT``, step5_guard) and records its printed time; the launch
block copies the guard's terminal output to CUES_LOG as it is written. ``follow`` reads that file as it grows, takes
only lines that are one of the session's own cues (the text at its scheduled second, mpe_schedule), and starts the
local speech command with the cue's text at once (``say`` on the Mac, ``spd-say -w`` on the Jetson, or any command
that takes the text as its last argument). The command runs in its own process, so a slow voice never delays the next
cue; it never receives anything but the schedule's fixed text. Each spoken cue is logged with the monotonic time it
was seen and the speech started. ``check`` (same machine) compares those times with the guard's printed_mono_s: every
printed cue spoken once, in order, within MAX_LAG_S. Speech is a convenience: the guard's printed lines stay the
record, and nobody needs to memorize a card or start a stopwatch. ``test`` speaks one fixed sentence so the operator
can hear the voice before a session.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import mpe_schedule

LINE = re.compile(r"\a?\[(\d+):(\d{2})\] (.+)")
POLL_S = 0.05
MAX_LAG_S = 0.5  # the guard samples every 0.2 s; a spoken cue starts within this after its printed line
TEST_SENTENCE = "Spoken cues are on. Follow each bell's words as they come."


def cue_of(kind: str, line: str) -> tuple[str, int, str] | None:
    """The session's cue a printed line is: its text at the second printed (the guard prints the elapsed second, at
    most one past the cue's own), or None for any other line."""
    found = LINE.fullmatch(line.rstrip("\n"))
    if found is None:
        return None
    printed_s = int(found.group(1)) * 60 + int(found.group(2))
    text = found.group(3)
    for name, at, cue_text in mpe_schedule.SESSIONS[kind]:
        if cue_text == text and at <= printed_s <= at + 1:
            return name, at, text
    return None


def follow(log: Path, kind: str, tts: list[str], spoken_log: Path | None = None, *,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
           spawn: Callable[[list[str]], object] = lambda argv: subprocess.Popen(  # noqa: S603 - fixed text only
               argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
           give_up_s: float | None = None) -> list[dict]:
    """Speak each of the session's cues as its line appears in ``log``; return (and log) what was spoken. Ends after
    the stop cue, or when the log has not grown for ``give_up_s`` seconds (None: wait for the stop cue)."""
    spoken: list[dict] = []
    voices: list[object] = []  # the speech processes still running: reaped as they end, waited for at the end
    names = {name for name, _, _ in mpe_schedule.SESSIONS[kind]}
    position, buffer, quiet_since = 0, "", clock()
    try:
        return _follow(log, kind, tts, spoken_log, clock, sleep, spawn, give_up_s, spoken, voices, names,
                       position, buffer, quiet_since)
    finally:
        for voice in voices:
            wait = getattr(voice, "wait", None)
            if wait is not None:
                try:
                    wait(timeout=30)
                except subprocess.TimeoutExpired:
                    voice.kill()
                    voice.wait()


def _follow(log, kind, tts, spoken_log, clock, sleep, spawn, give_up_s, spoken, voices, names, position, buffer,
            quiet_since) -> list[dict]:
    while True:
        voices[:] = [v for v in voices if getattr(v, "poll", lambda: 0)() is None]
        try:
            with open(log, "r", encoding="ascii", errors="replace") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            quiet_since = clock()
            buffer += chunk
            *lines, buffer = buffer.split("\n")
            for line in lines:
                seen = clock()
                cue = cue_of(kind, line)
                if cue is None or any(s["name"] == cue[0] for s in spoken):
                    continue
                voices.append(spawn([*tts, cue[2]]))
                entry = {"name": cue[0], "at_s": cue[1], "seen_mono_s": round(seen, 6),
                         "spoken_mono_s": round(clock(), 6)}
                spoken.append(entry)
                if spoken_log is not None:
                    with open(spoken_log, "a", encoding="ascii") as out:
                        out.write(json.dumps(entry) + "\n")
                if cue[0] == "stop" or len(spoken) == len(names):
                    return spoken
        elif give_up_s is not None and clock() - quiet_since > give_up_s:
            return spoken
        sleep(POLL_S)


def check(spoken: list[dict], result: dict) -> dict:
    """The spoken cues against the guard's record (the same machine's monotonic clock): each printed cue spoken once,
    in order, its speech started within MAX_LAG_S of its printed time."""
    printed = [c for c in result.get("cues", []) if isinstance(c.get("printed_mono_s"), (int, float))]
    by_name = {s["name"]: s for s in spoken}
    lags = {c["name"]: round(by_name[c["name"]]["spoken_mono_s"] - c["printed_mono_s"], 3)
            for c in printed if c["name"] in by_name}
    missing = [c["name"] for c in printed if c["name"] not in by_name]
    order = [s["name"] for s in spoken] == [c["name"] for c in printed if c["name"] in by_name]
    late = {name: lag for name, lag in lags.items() if not 0 <= lag <= MAX_LAG_S}
    return {"spoken": len(lags), "printed": len(printed), "missing": missing, "late": late, "in_order": order,
            "max_lag_s": max(lags.values()) if lags else None,
            "ok": not missing and not late and order and len(spoken) == len(lags)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("follow")
    f.add_argument("log", type=Path)
    f.add_argument("--session", required=True, choices=tuple(mpe_schedule.SESSIONS))
    f.add_argument("--tts", required=True, help='the speech command, e.g. "say" or "spd-say -w"')
    f.add_argument("--log", dest="spoken_log", type=Path, default=None)
    t = sub.add_parser("test")
    t.add_argument("--tts", required=True)
    c = sub.add_parser("check")
    c.add_argument("spoken_log", type=Path)
    c.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    if args.command == "test":
        subprocess.run([*shlex.split(args.tts), TEST_SENTENCE], check=False)  # noqa: S603 - fixed text only
        return 0
    if args.command == "follow":
        spoken = follow(args.log, args.session, shlex.split(args.tts), args.spoken_log)
        print(f"spoken cues: {len(spoken)}")
        return 0
    spoken = [json.loads(line) for line in args.spoken_log.read_text("ascii").splitlines() if line.strip()]
    result = check(spoken, json.loads(args.result.read_text("utf-8")))
    for key, value in result.items():
        print(f"{key}={value}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
