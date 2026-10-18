#!/usr/bin/env python3
"""An operator block's verdict on a guarded run (session 81): judged from the guard's own result.json and its exit
status, never from the block's last command.

The device run trk-20261016T100149Z showed why: the guard stopped the runtime 18.7 s after T0 (MemFree under its
floor), exited 1 and printed that exit to the terminal only, and the block's ``TV_exit`` was the status of its last
``tee``, 0. The guard's own exit 0 is not enough either: it also covers a runtime that exited on its own before the
duration (``child_exited``).

    guard_verdict.py RESULT_JSON --guard-exit N --cues NAME[,NAME...] [--require-ready tracking_diagnostic|landmark_diagnostic]
                     [--prefix XX_]

prints fixed ``key=value`` lines, each name with the prefix (the block appends them to the session's provenance):

    status=<the guard's status | missing | unreadable | malformed | unrecognized>
    returncode=<the runtime's exit | none>
    readiness=met | unmet:<reason> | not_required | missing
    cues_printed=<printed>/<expected>
    cleanup=clear | not_clear
    verdict=complete | incomplete:<reason>[;<reason>...]

and exits 0 only for ``verdict=complete``: the guard exited 0; its status is ``duration_stop`` (the runtime ran the
whole duration and the guard then stopped it); the runtime exited 0; no memory-guard trigger and no forced stop;
readiness met when required (the guard's own record); exactly the expected cues declared to the guard, in that
order, and each printed; nothing left (no llama-server, runtime group or listener) and cleanup clear. Anything else,
a missing, unreadable or malformed result included, is ``incomplete`` with its reasons, and exits 1 (2: usage).
Every value printed is a number or a fixed label; a text in the result that is not one is printed as
``unrecognized``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

STATUSES = ("duration_stop", "child_exited", "cleanup_failed", "forced_stop", "guard_stop", "ready_timeout",
            "operator_interrupt", "readiness_refused")  # step5_guard.supervise's statuses, besides refused:<label>
LABEL = re.compile(r"[a-z][a-z0-9_]{0,63}")
REASON = re.compile(r"[A-Za-z0-9_:() .,-]{1,160}")  # step5_guard.readiness_problem's labels (a detector problem's too)
PREFIX = re.compile(r"[A-Z]{1,4}_")
READINESS = ("tracking_diagnostic", "landmark_diagnostic")  # step5_guard.READINESS (sessions 80, 90)


def _status(value: Any) -> str:
    if value in STATUSES:
        return value
    if isinstance(value, str) and value.startswith("refused:") and LABEL.fullmatch(value[len("refused:"):]):
        return value
    return "unrecognized"


def _int(value: Any) -> bool:
    return type(value) is int


def verdict(result: Any, guard_exit: int, cues: tuple[str, ...], require_ready: str | None) -> list[tuple[str, str]]:
    """The (key, value) lines for one guarded run; the last is the verdict."""
    reasons: list[str] = [] if guard_exit == 0 else [f"guard_exit_{guard_exit}"]
    if isinstance(result, str):  # missing, unreadable: nothing else can be judged
        return [("status", result), ("returncode", "none"), ("readiness", "missing"), ("cues_printed", f"0/{len(cues)}"),
                ("cleanup", "not_clear"), ("verdict", "incomplete:" + ";".join([*reasons, f"result_{result}"]))]
    if not isinstance(result, dict) or result.get("mode") != "step5_guard":
        return [("status", "malformed"), ("returncode", "none"), ("readiness", "missing"),
                ("cues_printed", f"0/{len(cues)}"), ("cleanup", "not_clear"),
                ("verdict", "incomplete:" + ";".join([*reasons, "result_malformed"]))]
    status = _status(result.get("status"))
    if status != "duration_stop":
        reasons.append(f"status_{status}")
    child = result.get("child") if isinstance(result.get("child"), dict) else {}
    returncode = child.get("returncode")
    if returncode != 0 or not _int(returncode):
        reasons.append(f"runtime_exit_{returncode if _int(returncode) else 'none'}")
    guard = result.get("guard") if isinstance(result.get("guard"), dict) else None
    if guard is None or "trigger" not in guard:
        reasons.append("guard_record_missing")
    elif guard["trigger"] is not None:
        reasons.append("memory_guard_stop")
    stop = result.get("stop") if isinstance(result.get("stop"), dict) else {}
    if stop.get("forced") is not False:
        reasons.append("forced_stop" if stop.get("forced") is True else "stop_record_missing")
    readiness = result.get("readiness")
    if require_ready is None:
        ready = "not_required"
    elif not isinstance(readiness, dict):
        ready = "missing"
        reasons.append("readiness_missing")
    elif readiness.get("required") == require_ready and readiness.get("met") is True and readiness.get("reason") is None:
        ready = "met"
    else:
        reason = readiness.get("reason")
        reason = reason if isinstance(reason, str) and REASON.fullmatch(reason) else (
            "required_elsewhere" if readiness.get("required") != require_ready else "unrecognized")
        ready = f"unmet:{reason}"
        reasons.append(f"readiness_{ready}")
    parameters = result.get("parameters") if isinstance(result.get("parameters"), dict) else {}
    declared = [c.get("name") for c in parameters.get("cues") or [] if isinstance(c, dict)]
    if declared != list(cues):
        reasons.append("cues_not_as_expected")
    printed_by_name = {c.get("name"): c.get("printed_mono_s") for c in result.get("cues") or [] if isinstance(c, dict)}
    printed = [name for name in cues if isinstance(printed_by_name.get(name), (int, float))
               and not isinstance(printed_by_name.get(name), bool)]
    reasons += [f"cue_not_printed:{name}" for name in cues if name not in printed]
    left = result.get("leftovers") if isinstance(result.get("leftovers"), dict) else None
    clear = (result.get("cleanup_clear") is True and left is not None and left.get("llama_server_after_exit") == 0
             and left.get("llama_server_left") == 0 and left.get("runtime_group_left") is False
             and left.get("listeners_after") == [])
    if not clear:
        reasons.append("cleanup_not_clear")
    return [("status", status), ("returncode", str(returncode) if _int(returncode) else "none"), ("readiness", ready),
            ("cues_printed", f"{len(printed)}/{len(cues)}"), ("cleanup", "clear" if clear else "not_clear"),
            ("verdict", "complete" if not reasons else "incomplete:" + ";".join(reasons))]


def load(path: Path) -> Any:
    """The parsed result, or the label ``missing`` or ``unreadable``."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return "missing"
    except (OSError, UnicodeDecodeError):
        return "unreadable"
    try:
        return json.loads(text)
    except ValueError:
        return "unreadable"


def _cues(text: str) -> tuple[str, ...]:
    names = tuple(text.split(","))
    if not names or not all(LABEL.fullmatch(name) for name in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("give the expected cue names, comma-separated, each once")
    return names


def _prefix(text: str) -> str:
    if text and not PREFIX.fullmatch(text):
        raise argparse.ArgumentTypeError("a prefix is 1-4 capital letters and an underscore")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result", type=Path, help="the guard's result.json")
    parser.add_argument("--guard-exit", type=int, required=True, help="the guard's own exit status")
    parser.add_argument("--cues", type=_cues, required=True, help="the cue names the block passed, in order")
    parser.add_argument("--require-ready", choices=READINESS, default=None,
                        help="the readiness the block required of the guard (its record must show it met)")
    parser.add_argument("--prefix", type=_prefix, default="", help="for each line's name, e.g. IV_")
    args = parser.parse_args(argv)
    lines = verdict(load(args.result), args.guard_exit, args.cues, args.require_ready)
    for key, value in lines:
        print(f"{args.prefix}{key}={value}")
    return 0 if lines[-1][1] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
