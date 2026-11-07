"""The multi-pose comparison's account step: entered once, right after a session's run, locked, and required before
any reading shows anything (PROPOSED; validated with synthetic data only).

usage:
  mpe_account.py enter RUN_DIR --session B|E_M|E_P [--others-in-view no|yes|unknown --not-as-cued none|NAME,...]
  mpe_account.py check RUN_DIR --session B|E_M|E_P

``enter`` asks the person (or takes the two flags, for a block that has already asked): was anyone else in the
camera's view at any time, and which pose windows or holds were not done as cued (their names as the step lists them
or ``none``). It writes RUN_DIR/mpe-account.txt once (never over an existing file), read only (0400), and
appends to RUN_DIR/provenance.txt ``account_sha256=`` and ``account_saved_utc=``: the lock. It refuses once a reading
has started (``reading_started_utc=`` in the provenance): the account always comes before any result is shown.

Every reading (the builder, the comparison, the execution check) calls ``require`` first. It refuses, with a fixed
label and before anything else is read or printed, unless the account is there, parses, is the session's, matches its
lock byte for byte and was saved before any earlier reading began; then it appends ``reading_started_utc=``. A
missing, edited, unlocked or late account therefore stops the reading with "STOP: enter the account first", never with
a result. The lock shows only that the account was not changed after it was saved; that it was saved before any result
was seen rests on this ordering (the same limit as V3's account).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import mpe_schedule

ACCOUNT_NAME = mpe_schedule.ACCOUNT_NAME
LOCK_KEY = "account_sha256"
SAVED_KEY = "account_saved_utc"
READING_KEY = "reading_started_utc"
ANSWERS = ("no", "yes", "unknown")


class AccountRefused(Exception):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def allowed_names(kind: str) -> set[str]:
    """The window or hold names a session's account may list as not done as cued."""
    if kind == "B":
        return {w.name for w in mpe_schedule.B_WINDOWS}
    if kind == "E_M":
        return {h.name for h in mpe_schedule.EM_HOLDS}
    if kind == "E_P":
        return {w.name for w in mpe_schedule.EP_WINDOWS} | {h.name for h in mpe_schedule.EP_HOLDS}
    raise AccountRefused("unknown_session")


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _provenance(run_dir: Path) -> list[tuple[str, str]]:
    path = run_dir / "provenance.txt"
    if not path.is_file():
        return []
    return [tuple(line.split("=", 1)) for line in path.read_text("utf-8").splitlines() if "=" in line]


def _append(run_dir: Path, lines: list[str]) -> None:
    fd = os.open(run_dir / "provenance.txt", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, "".join(f"{line}\n" for line in lines).encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)


def enter(run_dir: Path, kind: str, *, others_in_view: str | None = None, not_as_cued: str | None = None,
          ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
          utc_now: Callable[[], str] = _utc) -> str:
    """Write the account once and lock it; returns its SHA-256. AccountRefused with a fixed label otherwise."""
    run_dir = Path(run_dir)
    names = allowed_names(kind)
    if any(key == READING_KEY for key, _ in _provenance(run_dir)):
        raise AccountRefused("reading_already_started")
    if any(key == LOCK_KEY for key, _ in _provenance(run_dir)) or os.path.lexists(run_dir / ACCOUNT_NAME):
        raise AccountRefused("account_already_saved")
    if others_in_view is None:
        say(f"Was anyone other than {mpe_schedule.WHO[kind]} in the camera's view at any time? (no / yes / unknown)")
        others_in_view = ask("> ").strip()
    if not_as_cued is None:
        say(f"This session's windows and holds: {', '.join(sorted(names))}")
        say("Which of them were not done as the spoken or printed cues said? Their names as listed, separated by "
            "commas, or none")
        not_as_cued = ask("> ").strip()
    listed = [] if not_as_cued == "none" else [n.strip() for n in not_as_cued.split(",")]
    if others_in_view not in ANSWERS or any(not n or n not in names for n in listed) or len(set(listed)) != len(listed):
        raise AccountRefused("account_invalid")
    text = (f"format={mpe_schedule.ACCOUNT_FORMAT}\nsession={kind}\nothers_in_view={others_in_view}\n"
            f"not_as_cued={','.join(listed) if listed else 'none'}\n").encode("ascii")
    fd = os.open(run_dir / ACCOUNT_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, text)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(run_dir / ACCOUNT_NAME, 0o400)
    digest = hashlib.sha256(text).hexdigest()
    _append(run_dir, [f"{LOCK_KEY}={digest}", f"{SAVED_KEY}={utc_now()}"])
    return digest


def require(run_dir: Path, kind: str, *, utc_now: Callable[[], str] = _utc) -> dict:
    """Before a reading shows anything: the session's locked account, saved before any reading began. Returns the
    parsed account and marks the reading begun; AccountRefused with a fixed label otherwise (nothing else is read)."""
    account = verify(run_dir, kind)
    _append(Path(run_dir), [f"{READING_KEY}={utc_now()}"])
    return account


def verify(run_dir: Path, kind: str) -> dict:
    """``require``'s checks without marking a reading begun (a reading reads the account again with it)."""
    run_dir = Path(run_dir)
    provenance = _provenance(run_dir)
    keys = [key for key, _ in provenance]
    if LOCK_KEY not in keys or SAVED_KEY not in keys:
        raise AccountRefused("account_not_locked")
    if keys.count(LOCK_KEY) != 1 or keys.count(SAVED_KEY) != 1:
        raise AccountRefused("account_lock_invalid")
    if READING_KEY in keys and keys.index(READING_KEY) < keys.index(LOCK_KEY):
        raise AccountRefused("account_saved_after_a_reading")
    path = run_dir / ACCOUNT_NAME
    if not path.is_file() or path.is_symlink():
        raise AccountRefused("account_missing")
    lock = dict(provenance)[LOCK_KEY]
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock:
        raise AccountRefused("account_changed")
    try:
        return mpe_schedule.read_account(run_dir, kind, allowed_names(kind))
    except mpe_schedule.AccountError as exc:
        raise AccountRefused(exc.label) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("enter")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--session", required=True, choices=("B", "E_M", "E_P"))
    e.add_argument("--others-in-view", choices=ANSWERS)
    e.add_argument("--not-as-cued")
    c = sub.add_parser("check")
    c.add_argument("run_dir", type=Path)
    c.add_argument("--session", required=True, choices=("B", "E_M", "E_P"))
    args = parser.parse_args(argv)
    try:
        if args.command == "enter":
            enter(args.run_dir, args.session, others_in_view=args.others_in_view, not_as_cued=args.not_as_cued)
            print("account saved and locked; the readings may now run")
        else:
            require(args.run_dir, args.session)
            print("account locked")
    except AccountRefused as exc:
        print(f"STOP: {exc.label}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
