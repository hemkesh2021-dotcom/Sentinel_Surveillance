"""The multi-pose comparison's retention (decision 11) and withdrawal (decision 10), as the maintainer approved them:
every deadline measured from collection, and the only deletions of sealed evidence this package makes (the approved
exception to keeping sealed evidence, for these folders only). Nothing runs on its own: the maintainer pastes the
package's block by each deadline (mp-K checks, mp-Z closes out, mp-W withdraws), and mp-R refuses a new session while
a deadline has passed with data still present.

usage: mpe_retention.py check     --root EXP_ROOT --sessions SESSIONS_DIR --data DATA_DIR --pin PIN
       mpe_retention.py close-out --root EXP_ROOT --sessions SESSIONS_DIR --data DATA_DIR --pin PIN
                                  --identity-dir ORIGINAL [--validation-finished]
       mpe_retention.py withdraw  (the same) --who participant|maintainer --requested YYYY-MM-DD
       mpe_retention.py closed    --root EXP_ROOT

Each deadline runs from a session's start, which its name records: mp-R names the session before anything of it is
collected, so every deadline here falls at or before the one measured from the first collected face. The same
deadlines hold whatever the outcome, for a session that stopped or was abandoned, and for an experiment abandoned
before it has an outcome.

- **Raw capture** (``raw-<session>``) and **working copy** (``wc-<session>``), in the experiment root: deleted by the
  build or at the session's close (mp-H; mp-C after an abandoned session); never later than 24 hours after the session
  started. The close-out runs the manifest's recovery, which deletes any left once its writer has exited.
- **Frozen candidate** (``candidate-<B session>/``) and its **frozen record**: kept privately until the maintainer
  records the follow-up validation finished, or 30 days after B started, whichever comes first. Keeping it authorizes
  no production adoption; an extension is a separate decision (this tool never extends a deadline). Its deletion
  closes the experiment: no session of this package starts after it.
- **E_P session folder** (SESSIONS_DIR/mp-ep-*) and its **data folder** (DATA_DIR/mp-ep-*): 30 days after E_P
  started; at once on the participant's withdrawal (the maintainer pastes mp-W within 2 days of the request).
- **B and E_M session folders and their data folders**: 90 days after each started; at once on the maintainer's
  withdrawal, which also deletes the candidate and E_P.
- **Kept** (no face data): the experiment root with MANIFEST.tsv and RETENTION.tsv (this tool's log of every
  deletion and withdrawal, outside every folder it deletes), the decisions and the outcome record.
- **Closed:** a withdrawal, or the candidate's deletion, ends the experiment: no session of this package starts after
  it and no outcome is recorded after it (RETENTION.tsv says why).

Standard library only: the package runs it, with mpe_manifest and mpe_outcome, from its own checksummed copies, so a
deadline is never missed because the worktree moved on from the pin.

A session folder is this package's only when its name is mp-(b|em|ep)-<UTC> directly in SESSIONS_DIR, it is a
directory (never a link) and its provenance's first package_pin is PIN (or the log names it as one this tool began to
delete); its data folder is DATA_DIR/<the same name>. Nothing else is ever deleted here: never the original
enrollment, the experiment root, another package's session; nothing while a session of this package is still open in
its shell, and nothing while a runtime, builder, comparison or manifest tool runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import mpe_manifest
from mpe_outcome import SESSION_RE, package_sessions, provenance, session_time

RAW_HOURS = 24
CANDIDATE_DAYS = 30
EP_DAYS = 30
B_EM_DAYS = 90
WITHDRAWAL_DAYS = 2
LOG_NAME = "RETENTION.tsv"
ROOT_ITEM_RE = re.compile(r"(raw|wc)-(mp-(?:b|em|ep)-\d{8}T\d{6}Z)")
KIND_OF = {"b": "B", "em": "E_M", "ep": "E_P"}
LEFT = ("left_changed", "left_not_empty", "left_outside_root", "left_protected", "left_sealed")


class Refused(Exception):
    pass


@dataclass(frozen=True)
class Item:
    what: str  # e.g. "E_P_session_folder", "raw_capture", "candidate"
    path: Path
    deadline: datetime
    session: str  # the session its deadline runs from
    kind: str = ""  # B, E_M or E_P for a session or data folder


def _utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def log(root: Path, *fields: str) -> None:
    """Append one row to RETENTION.tsv in the experiment root (append-only, private; the root is never deleted)."""
    if any("\t" in f or "\n" in f for f in fields):
        raise Refused("a log field holds a tab or a newline")
    if not root.is_dir():
        root.mkdir(mode=0o700, parents=True)
    fd = os.open(root / LOG_NAME, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, ("\t".join(fields) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def log_rows(root: Path) -> list[list[str]]:
    try:
        return [line.split("\t") for line in (root / LOG_NAME).read_text("utf-8").splitlines() if line]
    except FileNotFoundError:
        return []


def closed(root: Path) -> list[str]:
    """Why the experiment is closed (its candidate deleted, or a withdrawal); empty while it is open. The manifest's own
    deletion line for the frozen record closes it too (durable even if this log's row was never written: a power cut
    between the deletion and the row); a manifest that cannot be read counts as closed."""
    reasons = [row[1] for row in log_rows(root) if row[0] == "closed" and len(row) > 1]
    try:
        entry = mpe_manifest.Manifest(root).entries().get(root / mpe_manifest.FROZEN_NAME)
    except (mpe_manifest.ManifestError, OSError, ValueError):
        return reasons + ["manifest_unreadable"]
    if entry is not None and entry.deleted and not any(r.startswith("candidate_deleted") for r in reasons):
        reasons.append("candidate_deleted")
    return reasons


def _sealed(folder: Path) -> bool:
    return (folder / mpe_manifest.SEAL_NAME).exists()


def live(folder: Path) -> bool:
    """The session's shell is still open: the same process (PID and start time) on this boot, as mp-R recorded it."""
    prov = provenance(folder)
    pid, start, boot = ((prov.get(k) or [None])[0] for k in ("shell_pid", "shell_start", "shell_boot_id"))
    try:
        if not pid or boot != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
            return False
        text = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return False
    fields = text[text.rindex(")") + 2:].split()
    return len(fields) > 19 and fields[19] == start


def busy(sessions: Path, pin: str, processes: list[list[bytes]] | None = None) -> str | None:
    """Why nothing may be deleted now, or None."""
    for argv in (mpe_manifest._process_argvs() if processes is None else processes):
        if b"run" in argv and any(a.endswith(b"sentinel") or a == b"sentinel.cli" for a in argv):
            return "a runtime is running"
        if any(a.endswith(n) for a in argv for n in (b"mpe_build.py", b"mpe_compare.py", b"mpe_manifest.py")):
            return "a builder, comparison or manifest tool is running"
    for _, folder in package_sessions(sessions, pin):
        if not _sealed(folder) and live(folder):
            return f"session {folder.name} is still open in its shell: end it there (mp-H) first"
    return None


def _deadline(kind: str, name: str) -> datetime:
    return session_time(name) + timedelta(days=EP_DAYS if kind == "E_P" else B_EM_DAYS)


def items(root: Path, sessions: Path, data: Path, pin: str) -> list[Item]:
    """Everything of this package still present that a deadline applies to."""
    out: list[Item] = []
    seen: set[str] = set()
    for kind, folder in package_sessions(sessions, pin):
        seen.add(folder.name)
        out.append(Item(f"{kind}_session_folder", folder, _deadline(kind, folder.name), folder.name, kind))
    begun = {Path(row[1]).name for row in log_rows(root) if row[0] in ("deleting", "deleted") and len(row) > 1}
    for name in sorted(begun - seen):  # a deletion that stopped part-way: the log attributes what is left
        match = SESSION_RE.fullmatch(name)
        folder = sessions / name
        if match and os.path.lexists(folder) and not folder.is_symlink():
            kind = KIND_OF[match.group(1)]
            out.append(Item(f"{kind}_session_folder", folder, _deadline(kind, name), name, kind))
            seen.add(name)
    for name in sorted(seen | begun):
        match = SESSION_RE.fullmatch(name)
        folder = data / name
        if match and os.path.lexists(folder) and not folder.is_symlink():
            kind = KIND_OF[match.group(1)]
            out.append(Item(f"{kind}_data_folder", folder, _deadline(kind, name), name, kind))
    if root.is_dir():
        for path in sorted(root.iterdir()):
            match = ROOT_ITEM_RE.fullmatch(path.name)
            if match:
                what = "raw_capture" if match.group(1) == "raw" else "working_copy"
                out.append(Item(what, path, session_time(match.group(2)) + timedelta(hours=RAW_HOURS), match.group(2)))
            elif mpe_manifest.CANDIDATE_RE.fullmatch(path.name):
                b = path.name[len("candidate-"):]
                out.append(Item("candidate", path, session_time(b) + timedelta(days=CANDIDATE_DAYS), b))
        frozen = root / mpe_manifest.FROZEN_NAME
        if os.path.lexists(frozen):
            b = _frozen_b(frozen)
            deadline = session_time(b) + timedelta(days=CANDIDATE_DAYS) if b else datetime.min.replace(
                tzinfo=timezone.utc)  # a record naming no B session is overdue at once
            out.append(Item("frozen_record", frozen, deadline, b or "unknown"))
    return out


def _frozen_b(frozen: Path) -> str | None:
    try:
        name = json.loads(frozen.read_text("ascii")).get("candidate")
    except (OSError, ValueError, AttributeError):
        return None
    return name[len("candidate-"):] if isinstance(name, str) and mpe_manifest.CANDIDATE_RE.fullmatch(name) else None


def unattributed(sessions: Path, data: Path, pin: str) -> list[Path]:
    """Folders with a session's name that this package does not count as its own: listed for review, never deleted."""
    mine = {folder.name for _, folder in package_sessions(sessions, pin)}
    out = []
    for parent in (sessions, data):
        if parent.is_dir():
            out += [p for p in sorted(parent.iterdir()) if SESSION_RE.fullmatch(p.name) and p.name not in mine
                    and not (parent == data and (sessions / p.name).exists())]
    return out


def _left(delta: timedelta) -> str:
    return f"{delta.days} days {delta.seconds // 3600} h left"


def check(root: Path, sessions: Path, data: Path, pin: str, now: datetime | None = None) -> tuple[list[str], bool]:
    """Each item and its deadline; True when every item present is within its deadline."""
    now = now or _utc()
    lines, ok = [], True
    for item in sorted(items(root, sessions, data, pin), key=lambda i: (i.deadline, str(i.path))):
        overdue = now >= item.deadline
        ok = ok and not overdue
        lines.append(f"retention: {item.what} {item.path.name}: deadline {item.deadline:%Y-%m-%d %H:%M} UTC "
                     f"({'OVERDUE' if overdue else _left(item.deadline - now)})")
    for path in unattributed(sessions, data, pin):
        lines.append(f"retention: not this package's (another package's or unattributed; left, for review): {path}")
    lines += [f"retention: the experiment is closed ({reason}): no session of this package starts"
              for reason in closed(root)]
    lines.append("retention: every deadline met" if ok else
                 "retention: a deadline has passed with data still present (OVERDUE above)")
    return lines, ok


def _delete_tree(path: Path, parent: Path, protected: tuple[Path, ...]) -> None:
    """Remove one session or data folder: exactly PARENT/<a session's name>, a directory reached without a link, never
    at, under or holding a protected path; links inside are removed, never followed."""
    if path.parent != parent or SESSION_RE.fullmatch(path.name) is None:
        raise Refused(f"not a session folder of {parent}: {path}")
    if parent.resolve() != parent or path.resolve() != path:
        raise Refused(f"a link on the way: {path}")
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise Refused(f"not a directory: {path}")
    if any(path == p or p in path.parents or path in p.parents for p in protected):
        raise Refused(f"a protected path: {path}")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise Refused("this platform's tree removal may follow links")
    shutil.rmtree(path)


def _remove(root: Path, item: Item, reason: str, protected: tuple[Path, ...], sessions: Path, data: Path,
            now: datetime, report: dict) -> None:
    parent = sessions if item.what.endswith("_session_folder") else data
    log(root, "deleting", str(item.path), item.what, reason, item.deadline.isoformat(), now.isoformat())
    try:
        _delete_tree(item.path, parent, protected)
    except (Refused, OSError) as exc:
        report["left"].append(f"{item.path} ({type(exc).__name__}: {exc})")
        return
    log(root, "deleted", str(item.path), item.what, reason, item.deadline.isoformat(), _utc().isoformat())
    report["deleted"].append(f"{item.what} {item.path}")


def _recover(root: Path, sessions: Path, pin: str, identity: Path, report: dict) -> None:
    """Raw captures and working copies (and an unfrozen candidate): the manifest's recovery, as mp-C runs it."""
    if not (root / mpe_manifest.MANIFEST_NAME).exists():
        return
    manifest = mpe_manifest.Manifest(root)
    runs = [folder / "mv" for _, folder in package_sessions(sessions, pin)
            if not _sealed(folder) and (folder / "mv" / "result.json").is_file()]
    exited = all(mpe_manifest.runtime_exited(mpe_manifest._result(run)) for run in runs) if runs else \
        mpe_manifest.runtime_exited({"child": {"returncode": 0}})
    written = any(mpe_manifest._recorded_written(run) for run in runs)
    frozen = root / mpe_manifest.FROZEN_NAME
    result = mpe_manifest.recover(manifest, writer_exited=exited, protected=(identity,),
                                  recorded_written={p: written for p, r in manifest.reservations().items()
                                                    if r.writer == "raw"}, frozen=frozen)
    for path in result["cleanup"]["deleted"]:
        log(root, "deleted", path, "experiment_file", "recovery", "-", _utc().isoformat())
        report["deleted"].append(f"experiment_file {path}")
    report["left"] += [f"{p} ({k})" for k in LEFT for p in result["cleanup"][k]]


def _retire(root: Path, identity: Path, reason: str, deadline: str, report: dict) -> None:
    result = mpe_manifest.retire_candidate(mpe_manifest.Manifest(root), protected=(identity,))
    for path in result["deleted"]:
        log(root, "deleted", path, "candidate", reason, deadline, _utc().isoformat())
        report["deleted"].append(f"candidate {path}")
    report["left"] += [f"{p} ({k})" for k in LEFT for p in result[k]]
    if not os.path.lexists(root / mpe_manifest.FROZEN_NAME) and not any(
            mpe_manifest.CANDIDATE_RE.fullmatch(p.name) for p in root.iterdir()):
        log(root, "closed", f"candidate_deleted_{reason}", _utc().isoformat())


def _phase(report: dict, name: str, step, *args) -> None:
    try:
        step(*args)
    except (mpe_manifest.ManifestError, OSError, ValueError) as exc:
        report["left"].append(f"{name} stopped: {type(exc).__name__}: {exc}")


def close_out(root: Path, sessions: Path, data: Path, pin: str, identity: Path, *, validation_finished: bool = False,
              now: datetime | None = None, processes: list[list[bytes]] | None = None) -> dict:
    """Apply the deadlines now: recover raw captures and working copies; delete the frozen candidate and its record when
    due (or when the maintainer records the follow-up validation finished); delete each session and data folder whose
    deadline has passed (the data folder first: the session folder holds the provenance that attributes it)."""
    reason = busy(sessions, pin, processes)
    if reason:
        raise Refused(reason)
    now = now or _utc()
    report: dict = {"deleted": [], "left": []}
    _phase(report, "recovery", _recover, root, sessions, pin, identity, report)  # each phase apart: one that stops
    present = items(root, sessions, data, pin)  # never keeps the folders below from their deadline
    candidate = [i for i in present if i.what in ("candidate", "frozen_record")]
    if candidate and (validation_finished or any(now >= i.deadline for i in candidate)):
        if validation_finished:
            log(root, "validation_finished", now.isoformat())
        due = min(i.deadline for i in candidate)
        _phase(report, "candidate", _retire, root, identity,
               "validation_finished" if validation_finished and now < due else "deadline", due.isoformat(), report)
    for item in sorted((i for i in present if i.kind),
                       key=lambda i: (i.what.endswith("_session_folder"), str(i.path))):
        if now >= item.deadline:
            _remove(root, item, "deadline", (identity, root), sessions, data, now, report)
    return report


def withdraw(root: Path, sessions: Path, data: Path, pin: str, identity: Path, who: str, requested: date, *,
             today: date | None = None, processes: list[list[bytes]] | None = None,
             now: datetime | None = None) -> dict:
    """A withdrawal, at once: the participant's deletes every E_P session and data folder of this package; the
    maintainer's deletes the candidate, its record, any raw capture or working copy and every session and data folder
    of this package. Either closes the experiment; an outcome already recorded is marked withdrawn in the log."""
    if who not in ("participant", "maintainer"):
        raise Refused("who withdraws: participant or maintainer")
    today = today or date.today()
    if requested > today:
        raise Refused("the request date is in the future")
    reason = busy(sessions, pin, processes)
    if reason:
        raise Refused(reason)
    now = now or _utc()
    within = (today - requested).days <= WITHDRAWAL_DAYS
    log(root, "withdrawal", who, requested.isoformat(), now.isoformat(), f"within_{WITHDRAWAL_DAYS}_days="
        f"{'yes' if within else 'no'}")
    report: dict = {"deleted": [], "left": [], "within_two_days": within}
    try:
        if who == "maintainer":
            _phase(report, "recovery", _recover, root, sessions, pin, identity, report)
            if os.path.lexists(root / mpe_manifest.FROZEN_NAME):
                _phase(report, "candidate", _retire, root, identity, "maintainer_withdrawal", now.isoformat(), report)
        kinds = ("E_P",) if who == "participant" else ("B", "E_M", "E_P")
        targets = [i for i in items(root, sessions, data, pin) if i.kind in kinds]
        for item in sorted(targets, key=lambda i: (i.what.endswith("_session_folder"), str(i.path))):
            _remove(root, item, f"{who}_withdrawal", (identity, root), sessions, data, now, report)
        outcome = sessions / f"mp-outcome-{pin[:7]}.txt"
        if outcome.exists():
            log(root, "outcome_marked_withdrawn", str(outcome), who, now.isoformat())
    finally:
        log(root, "closed", f"{who}_withdrawal", now.isoformat())  # whatever stopped part-way: no session starts after
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "close-out", "withdraw", "closed"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name == "closed":
            continue
        p.add_argument("--sessions", type=Path, required=True)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--pin", required=True)
        if name in ("close-out", "withdraw"):
            p.add_argument("--identity-dir", type=Path, required=True)
        if name == "close-out":
            p.add_argument("--validation-finished", action="store_true")
        if name == "withdraw":
            p.add_argument("--who", choices=("participant", "maintainer"), required=True)
            p.add_argument("--requested", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    if args.command == "closed":
        reasons = closed(root)
        for reason in reasons:
            print(f"closed={reason}")
        return 0 if reasons else 1
    sessions, data = args.sessions.absolute(), args.data.absolute()
    if args.command == "check":
        lines, ok = check(root, sessions, data, args.pin)
        print("\n".join(lines))
        return 0 if ok else 1
    try:
        if args.command == "close-out":
            report = close_out(root, sessions, data, args.pin, args.identity_dir.absolute(),
                               validation_finished=args.validation_finished)
        else:
            report = withdraw(root, sessions, data, args.pin, args.identity_dir.absolute(), args.who, args.requested)
            print(f"withdrawal_within_two_days={'yes' if report['within_two_days'] else 'no'}")
    except Refused as exc:
        print(f"STOP: {exc}; nothing was deleted")
        return 1
    except mpe_manifest.ManifestError as exc:
        print(f"STOP: {exc} (the log above names what was deleted before it)")
        return 1
    print(f"deleted={len(report['deleted'])}")
    for line in report["deleted"]:
        print(f"  {line}")
    print(f"left={len(report['left'])}")
    for line in report["left"]:
        print(f"  {line}")
    lines, ok = check(root, sessions, data, args.pin)
    print("\n".join(lines))
    return 0 if ok and not report["left"] else 1


if __name__ == "__main__":
    sys.exit(main())
