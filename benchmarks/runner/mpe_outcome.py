"""The multi-pose comparison's outcome (decision 12, as the maintainer approved it): one label for the frozen candidate,
read from the sealed evaluation sessions in a fixed order of precedence; the first that applies is the result.

usage: mpe_outcome.py status --sessions SESSIONS_DIR --pin PIN
       mpe_outcome.py record --sessions SESSIONS_DIR --pin PIN --frozen EXP_ROOT/candidate.frozen.json --out FILE [--end]
       mpe_outcome.py day-check --frozen EXP_ROOT/candidate.frozen.json   (decision 9: E_M never on B's local day)

1. **REJECTED**: an E_P session of this package that passed its execution check (sealed, its seal verified) shows any
   candidate match vote, or a candidate KNOWN, on the participant: every compared face of that run, the account saying
   nobody else was in view, the shadow the frozen candidate. This holds whatever the other evidence or the replay
   shows: match votes are the recorded shadow decisions and need no replay, and a KNOWN needs match votes. A session
   that passed but was sealed before its comparison (a closed shell, a power cut: mp-C seals it as it is) is read for
   this too, from its sealed records, the same way mpe_compare counts them.
2. **INCONCLUSIVE**: any required evidence is missing, invalid or incomplete: no valid frozen record; E_M or E_P not
   run, not compared, not sealed, its seal not verified, its execution check not passed (the locked account included)
   or compared with another candidate; any of the comparison's own INCONCLUSIVE labels (incomplete or inconsistent
   track-life records, a replay that does not reproduce the recorded history, compared faces without a shadow
   decision, a shadow that is not the frozen candidate, an account that does not say nobody else was in view); fewer
   than two valid holds in a condition; the participant not seen in a pose window that gave the candidate a prototype
   (no compared face there, or the window excluded by the account).
3. **NOT_SUPPORTED**: the evidence is complete and valid (E_M), but an improvement criterion fails: at either mark
   (4.0 m or 5.0 m) the candidate's KNOWN share of the natural holds is less than GAIN_POINTS percentage points above
   the original's (losses counted: the shares are net); a facing hold where the original acquired KNOWN at the mark and
   the candidate did not; the pooled facing KNOWN share more than FACING_LOSS_POINTS points below the original's.
   Shares are pooled over the kept holds' time and compared in integer nanoseconds (no rounding decides).
4. **SUPPORTED**: only when every criterion above is met.

SUPPORTED allows only proposing to adopt this exact candidate (a separate enrollment decision) and a live validation of
it; it is not beta readiness or adequate natural recognition at 4-5 m. One enrolled person and one participant, one
session each: k of n, never a rate. The record holds labels, the digests of what it read and E_M's criterion numbers;
nothing of the participant but the labels, and no similarity value or identity ID. It is written once and never changed,
and only while every evaluation session of this package is sealed. It names the earliest instant the participant's
counts may enter the Git record: 7 days after E_P's launch plus 2 hours (a bound on the run's end).

Standard library only: the package runs it, with mpe_retention and mpe_manifest, from its own checksummed copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FORMAT = "mpe-outcome-1"
COMPARE_FORMAT = "mpe-compare-1"
FROZEN_FORMAT = "mpe-frozen-1"
OUTCOMES = ("REJECTED", "INCONCLUSIVE", "NOT_SUPPORTED", "SUPPORTED")  # the order of precedence
GAIN_POINTS = 20  # natural KNOWN share at each mark: the candidate at least this many points above the original
FACING_LOSS_POINTS = 5  # pooled facing KNOWN share: the candidate at most this many points below the original
GIT_DELAY = timedelta(days=7, hours=2)  # after E_P's launch: 7 days after the run's end at the earliest
ACCOUNT_NAME = "mpe-account.txt"  # mpe_schedule's, in the run's directory
SESSION_RE = re.compile(r"mp-(b|em|ep)-(\d{8}T\d{6}Z)")
KIND_OF = {"b": "B", "em": "E_M", "ep": "E_P"}
PASSED_LINE = "mpe-execution: passed"
EXCLUDED = "excluded_not_as_cued"  # mpe_compare's label for an E_P window the account excludes


def session_time(name: str) -> datetime:
    """A session's start (UTC), from its own name: mp-R takes it before anything of the session is collected."""
    match = SESSION_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not a session name: {name!r}")
    return datetime.strptime(match.group(2), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def provenance(directory: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        text = (directory / "provenance.txt").read_text("utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out.setdefault(key, []).append(value)
    return out


def package_sessions(sessions: Path, pin: str) -> list[tuple[str, Path]]:
    """(kind, folder) of each session folder this package made: its name, not a link, its first recorded pin."""
    out = []
    if not sessions.is_dir():
        return out
    for directory in sorted(sessions.iterdir()):
        match = SESSION_RE.fullmatch(directory.name)
        if match is None or directory.is_symlink() or not directory.is_dir():
            continue
        if (provenance(directory).get("package_pin") or [None])[0] == pin:
            out.append((KIND_OF[match.group(1)], directory))
    return out


def seal_verifies(directory: Path) -> bool:
    """The session's SHA256SUMS lists every file now in it, and each still has its listed SHA-256."""
    seal = directory / "SHA256SUMS"
    if seal.is_symlink() or not seal.is_file():
        return False
    listed: dict[str, str] = {}
    for line in seal.read_text("utf-8", errors="replace").splitlines():
        digest, sep, name = line.partition("  ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest) or not name.startswith("./") or name in listed:
            return False
        listed[name] = digest
    present = set()
    for current, dirs, files in os.walk(directory, followlinks=False):
        for name in files:
            path = Path(current) / name
            rel = "./" + str(path.relative_to(directory))
            if rel != "./SHA256SUMS":
                present.add(rel)
        for name in dirs:
            if (Path(current) / name).is_symlink():
                return False
    if present != set(listed):
        return False
    for rel, digest in listed.items():
        path = directory / rel[2:]
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True


def evidence(sessions: Path, pin: str, kind: str) -> dict[str, Any]:
    """What this package's sessions of ``kind`` (E_M or E_P) give the outcome: the one compared (MX_started), sealed
    with a seal that verifies, whose execution check passed and whose comparison record is there; else a state label."""
    folders = [d for k, d in package_sessions(sessions, pin) if k == kind]
    if not folders:
        return {"kind": kind, "state": "not_run"}
    compared = [d for d in folders if "yes" in provenance(d).get("MX_started", [])]
    # E_P: every session that passed its execution check but was sealed before its comparison, read for REJECTED
    uncompared = [{"name": d.name, **sealed_run_votes(d / "mv")} for d in folders
                  if kind == "E_P" and d not in compared and _passed(d)]
    if len(compared) > 1:
        return {"kind": kind, "state": "compared_more_than_once", "passed_uncompared": uncompared}
    if not compared:
        return {"kind": kind, "state": "not_compared", "passed_uncompared": uncompared}
    folder = compared[0]
    out: dict[str, Any] = {"kind": kind, "name": folder.name, "passed_uncompared": uncompared,
                           "launch_utc": (provenance(folder).get("MV_launch_utc") or [None])[-1]}
    if not (folder / "SHA256SUMS").exists():
        return {**out, "state": "not_sealed"}
    if not seal_verifies(folder):
        return {**out, "state": "seal_not_verified"}
    prov = provenance(folder)
    check = folder / "mv" / "check.txt"
    lines = [line for line in check.read_text("utf-8", errors="replace").splitlines() if line.strip()] \
        if check.is_file() else []
    if (prov.get("MPE_exit") or [None])[-1] != "0" or not lines or lines[-1] != PASSED_LINE:
        return {**out, "state": "execution_not_passed"}
    try:
        record = json.loads((folder / "compare.json").read_text("ascii"))
    except (OSError, ValueError):
        return {**out, "state": "comparison_missing"}
    if record.get("format") != COMPARE_FORMAT or record.get("session") != kind:
        return {**out, "state": "comparison_missing"}
    seal_sha = hashlib.sha256((folder / "SHA256SUMS").read_bytes()).hexdigest()
    return {**out, "state": "compared", "compare": record, "seal_sha256": seal_sha}


def _account_others(run: Path) -> str | None:
    try:
        lines = (run / ACCOUNT_NAME).read_text("ascii").splitlines()
    except (OSError, ValueError):
        return None
    return next((line.split("=", 1)[1] for line in lines if line.startswith("others_in_view=")), None)


def sealed_run_votes(run: Path) -> dict[str, Any]:
    """From a sealed run that passed its execution check: the candidate match votes as mpe_compare counts them (each
    compared face, label match, unknown or margin, whose shadow decision is match; no replay), the shadow digest the run
    started with, and the account's answer on anyone else in view. Read only."""
    votes, digest, started = 0, None, False
    try:
        text = (run / "run.jsonl").read_text("utf-8", errors="replace")
    except OSError:
        return {"readable": False}
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("run") == "starting" and not started:
            started = True
            digest = (((record.get("startup") or {}).get("face")) or {}).get("shadow_digest")
        if record.get("identity") == "result" and record.get("outcome") == "applied":
            for person in record.get("persons") or []:
                shadow = person.get("shadow") if isinstance(person, dict) else None
                if person.get("label") in ("match", "unknown", "margin") and isinstance(shadow, dict) \
                        and shadow.get("decision") == "match":
                    votes += 1
    return {"readable": True, "candidate_match_votes": votes, "shadow_digest": digest,
            "others_in_view": _account_others(run)}


def _passed(folder: Path) -> bool:
    """Sealed, the seal verified, the execution check passed (its last MPE_exit 0 and check.txt's verdict line)."""
    if not (folder / "SHA256SUMS").exists() or not seal_verifies(folder):
        return False
    check = folder / "mv" / "check.txt"
    lines = [line for line in check.read_text("utf-8", errors="replace").splitlines() if line.strip()] \
        if check.is_file() else []
    return (provenance(folder).get("MPE_exit") or [None])[-1] == "0" and bool(lines) and lines[-1] == PASSED_LINE


def _sums(rows: list[dict]) -> tuple[int, int, int]:
    return (sum(r["length_ns"] for r in rows), sum(r["known_ns"]["baseline"] for r in rows),
            sum(r["known_ns"]["candidate"] for r in rows))


def _points(length: int, base: int, cand: int) -> float | None:
    return None if length <= 0 else 100.0 * (cand - base) / length


def em_criteria(compare: dict) -> dict[str, dict]:
    """The improvement criteria on E_M's kept holds (not excluded): each with its number and whether it is met. Raises
    KeyError or TypeError when a kept hold carries no replay numbers (a withheld replay: never evaluated)."""
    kept = [r for r in compare["holds"] if not r["excluded"]]
    out: dict[str, dict] = {}
    for spot in ("4", "5"):
        rows = [r for r in kept if r["spot"] == spot and r["kind"] == "natural"]
        length, base, cand = _sums(rows)
        out[f"natural_gain_{spot}_0m"] = {"points": _points(length, base, cand), "bar": f">= +{GAIN_POINTS}",
                                           "met": length > 0 and 100 * (cand - base) >= GAIN_POINTS * length}
    facing = [r for r in kept if r["kind"] == "facing"]
    lost = sum(1 for r in facing if r["at_mark"]["baseline"] == "acquired" and r["at_mark"]["candidate"] != "acquired")
    out["facing_acquisitions_lost"] = {"count": lost, "bar": "0", "met": lost == 0}
    length, base, cand = _sums(facing)
    out["pooled_facing"] = {"points": _points(length, base, cand), "bar": f">= -{FACING_LOSS_POINTS}",
                            "met": length > 0 and 100 * (cand - base) >= -FACING_LOSS_POINTS * length}
    return out


def _frozen_ok(frozen: dict | None) -> bool:
    return isinstance(frozen, dict) and frozen.get("format") == FROZEN_FORMAT and \
        isinstance(frozen.get("digest"), str) and isinstance(frozen.get("prototypes_digest"), str)


def evaluate(frozen: dict | None, em: dict, ep: dict) -> dict[str, Any]:
    """The outcome, in the order of precedence (REJECTED, INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED), with its reasons."""
    digest = frozen.get("prototypes_digest") if _frozen_ok(frozen) else None
    sessions = {e["kind"]: {k: e.get(k) for k in ("name", "state", "seal_sha256", "launch_utc")} for e in (em, ep)}
    base: dict[str, Any] = {"format": FORMAT, "sessions": sessions,
                            "frozen_digest": frozen.get("digest") if _frozen_ok(frozen) else None}
    reasons: list[str] = []
    # 1. REJECTED: the participant-match condition, on an E_P session that passed its execution check, whose shadow
    # was the frozen candidate: first any that was sealed before its comparison (its sealed records), then the compared
    for run in ep.get("passed_uncompared") or []:
        votes = run.get("candidate_match_votes") or 0
        shadow_ok = run.get("shadow_digest") is not None and (digest is None or run["shadow_digest"] == digest)
        if votes > 0 and shadow_ok and run.get("others_in_view") == "no":
            return {**base, "outcome": "REJECTED", "reasons": ["e_p_candidate_match_on_the_participant",
                                                               f"e_p_session_sealed_before_its_comparison_{run['name']}"]}
        if votes > 0 or not run.get("readable", True):
            reasons.append(f"e_p_uncompared_session_not_attributable_{run['name']}")
    if ep["state"] == "compared":
        c = ep["compare"]
        labels = set(c.get("inconclusive") or [])
        shadow_ok = not labels & {"frozen_record_missing", "shadow_gallery_not_the_frozen_candidate"} and \
            (digest is None or c.get("frozen_prototypes_digest") == digest)
        p = c.get("participant") or {}
        matched = (p.get("candidate_match_votes") or 0) > 0 or (p.get("candidate_known_entries") or 0) > 0
        if matched and shadow_ok and "account_others_in_view_not_no" not in labels:
            return {**base, "outcome": "REJECTED", "reasons": ["e_p_candidate_match_on_the_participant"]}
        if matched:
            reasons.append("e_p_candidate_match_not_attributable_to_the_frozen_candidate_and_the_participant")
    # 2. INCONCLUSIVE: required evidence missing, invalid or incomplete
    if digest is None:
        reasons.append("frozen_record_missing")
    for e in (em, ep):
        tag = e["kind"].lower()
        if e["state"] != "compared":
            reasons.append(f"{tag}_{e['state']}")
            continue
        c = e["compare"]
        if digest is not None and c.get("frozen_prototypes_digest") != digest:
            reasons.append(f"{tag}_compared_with_another_candidate")
        reasons += [f"{tag}:{label}" for label in c.get("inconclusive") or []]
        if not c.get("replay_valid"):
            reasons.append(f"{tag}:replay_not_valid")
        reasons += [f"{tag}:fewer_than_two_valid_holds_{x['spot']}_{x['kind']}" for x in c.get("conditions") or []
                    if x.get("inconclusive")]
        if e["kind"] == "E_P":
            exposure = c.get("exposure") or {}
            windows = [name for name, _ in (frozen or {}).get("windows", [])] if digest is not None else list(exposure)
            reasons += [f"e_p:participant_not_seen_in_window_{name}" for name in windows
                        if not isinstance(exposure.get(name), int) or exposure.get(name) == 0]
    criteria = None
    if not reasons:
        try:
            criteria = em_criteria(em["compare"])
        except (KeyError, TypeError):
            reasons.append("e_m:replay_numbers_withheld")
    if reasons:
        return {**base, "outcome": "INCONCLUSIVE", "reasons": sorted(set(reasons), key=reasons.index)}
    # 3. NOT_SUPPORTED / 4. SUPPORTED: the improvement criteria on valid, complete evidence
    failed = [name for name, value in criteria.items() if not value["met"]]
    return {**base, "outcome": "NOT_SUPPORTED" if failed else "SUPPORTED", "reasons": failed or ["every_criterion_met"],
            "criteria": criteria}


def _fmt_points(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}"


def record_lines(out: dict[str, Any], pin: str, utc_now: datetime) -> list[str]:
    lines = [f"format={FORMAT}", f"pin={pin}", f"recorded_utc={utc_now.isoformat(timespec='seconds')}",
             f"frozen_digest={out['frozen_digest'] or 'none'}"]
    for kind in ("E_M", "E_P"):
        s = out["sessions"][kind]
        tag = kind.lower()
        lines += [f"session_{tag}={s.get('name') or 'none'}", f"session_{tag}_state={s['state']}",
                  f"session_{tag}_seal_sha256={s.get('seal_sha256') or 'none'}"]
    lines.append(f"outcome={out['outcome']}")
    lines += [f"reason={r}" for r in out["reasons"]]
    for name, value in (out.get("criteria") or {}).items():
        number = f"count {value['count']}" if "count" in value else f"{_fmt_points(value['points'])} points"
        lines.append(f"criterion_{name}={'met' if value['met'] else 'not_met'} ({number}; bar {value['bar']})")
    ep_name = out["sessions"]["E_P"].get("name")
    if ep_name:
        launch = out["sessions"]["E_P"].get("launch_utc")
        try:
            base = datetime.strptime(launch, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            base = session_time(ep_name) + timedelta(days=1)  # no launch time recorded: a day later, never earlier
        lines.append(f"e_p_counts_enter_git_not_before={(base + GIT_DELAY):%Y-%m-%dT%H:%MZ}")
    lines.append("adoption=not_authorized_by_this_outcome (SUPPORTED allows only proposing adoption, a separate "
                 "enrollment decision, and a live validation)")
    return lines


def b_day(frozen: dict) -> date | None:
    """B's local calendar day (this machine's time zone), from the session the frozen record's candidate names."""
    name = frozen.get("candidate") if isinstance(frozen, dict) else None
    if not isinstance(name, str) or not name.startswith("candidate-"):
        return None
    try:
        return session_time(name[len("candidate-"):]).astimezone().date()
    except ValueError:
        return None


def write_once(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        os.write(fd, text.encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "record"):
        p = sub.add_parser(name)
        p.add_argument("--sessions", type=Path, required=True)
        p.add_argument("--pin", required=True)
        if name == "record":
            p.add_argument("--frozen", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--end", action="store_true",
                           help="record now although E_M or E_P is not compared (the maintainer's typed choice)")
    day = sub.add_parser("day-check")
    day.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "day-check":
        try:
            b = b_day(json.loads(args.frozen.read_text("ascii")))
        except (OSError, ValueError):
            b = None
        today = date.today()
        print(f"b_day={b or 'unknown'} today={today}")
        return 0 if b is not None and b != today else 1
    em, ep = (evidence(args.sessions, args.pin, kind) for kind in ("E_M", "E_P"))
    both = em["state"] == "compared" and ep["state"] == "compared"
    if args.command == "status":
        print(f"e_m={em['state']}\ne_p={ep['state']}")
        return 0 if both else 1
    if os.path.lexists(args.out):
        print(f"STOP: the outcome is recorded already ({args.out.name}); it is never changed")
        return 1
    unsealed = [d.name for k, d in package_sessions(args.sessions, args.pin)
                if k in ("E_M", "E_P") and not (d / "SHA256SUMS").exists()]
    if unsealed:
        print(f"STOP: evaluation session {', '.join(unsealed)} is not sealed: end it with mp-H in its shell (or, if its "
              "shell is gone, start a new SSH shell: mp-R stops and asks for mp-C, which seals it); nothing recorded")
        return 1
    if not both and not args.end:
        print(f"STOP: e_m={em['state']} e_p={ep['state']}: both evaluation sessions are not compared and sealed yet; "
              "nothing recorded")
        return 1
    try:
        frozen = json.loads(args.frozen.read_text("ascii"))
    except (OSError, ValueError):
        frozen = None
    out = evaluate(frozen, em, ep)
    lines = record_lines(out, args.pin, datetime.now(timezone.utc).replace(microsecond=0))
    write_once(args.out, "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"recorded: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
