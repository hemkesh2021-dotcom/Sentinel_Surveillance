#!/usr/bin/env python3
"""The offline F1 recheck of one V2-25 face run, the offline recheck of an enrollment-experiment capture (D91), and
the maintainer's explicit acceptance of each.

    printf 'name=%s\\0name=%s\\0' "$N1" "${N2-}" | face_recheck.py run RUN_DIR --identity-dir DIR --out RECHECK_DIR \\
        --pin PIN --commit COMMIT
    face_recheck.py accept RECHECK_DIR --run RUN_DIR --identity-dir DIR      (interactive: the maintainer's decision)
    printf 'name=%s\\0name=%s\\0' "$N1" "${N2-}" | face_recheck.py capture SESSION_DIR --f1 F1_DIR --run RUN_DIR \\
        --identity-dir DIR [--recheck-root ROOT] --working-gallery XDIR --out RECHECK_DIR --pin PIN --commit COMMIT
    face_recheck.py accept-capture RECHECK_DIR SESSION_DIR --f1 F1_DIR --run RUN_DIR --identity-dir DIR \\
        [--recheck-root ROOT] --working-gallery XDIR                         (interactive: the maintainer's decision)

``run`` reads the run directory and the identity directory's audit log only: no photo, no gallery decryption, no
model. The names arrive on stdin (face_counts.sh's face_recheck_f1). It writes a new RECHECK_DIR (mode 700, outside
the run) and changes nothing of the run:
- ``secret-counts-names.txt``: the names counted now in F1's evidence with face_privacy's corrected method, and the
  parts of each file that do not fit its profile. Nothing else is rescanned;
- ``provenance.txt``: the run, the SHA-256 of its manifest and whether it verified, the enrolled identity, the audit
  log's SHA-256 now (the run's manifest does not cover it), the commit and pin, and that the passphrase and
  photo-file-name counts are carried forward from the run's own counts file (its SHA-256, and whether the run's
  manifest verifies it): not rescanned;
- ``check.txt`` (face_check's f1-recheck reading), ``README.txt`` and ``SHA256SUMS``.
The run's own result (F1_exit, f1/check.txt) stays as recorded. This is a reading made afterwards with code changed
afterwards, for the maintainer's decision. Exit 0 only if the reading validates.

``accept`` is that decision. It refuses unless the recheck directory is unchanged and its reading still validates
now, shows what would be accepted, and writes ``acceptance.txt`` (mode 600, never overwritten) only after the exact
typed line ``ACCEPT <run> <identity>``. face_check's gate lets F2 build on a run whose own F1 did not validate only
through such an acceptance, and only while it still matches.

``capture`` does the same for an experiment session whose capture reading did not say the working gallery may be
used. It reads the session directory, the F1 basis (as ``gate`` does: the run directory and the identity directory's
audit log and listing, read only) and the working gallery's SHA-256 only: no gallery is decrypted, no model loaded.
The names arrive on stdin (face_counts.sh's face_recheck_capture). RECHECK_DIR must be named
``<session>-xc-<pin>-<UTC>``; it gets ``secret-counts-names.txt`` (the names and the nonconforming parts, counted in
the capture's evidence with the current profiles), ``provenance.txt``, ``check.txt`` (face_check's capture recheck),
``README.txt`` and ``SHA256SUMS``. The session's own result (XC_check_exit, xc/check.txt) stays as recorded.
``accept-capture`` is the maintainer's decision, written as ``acceptance.txt`` only after the exact typed line
``ACCEPT <session> capture``. It adopts nothing: the working gallery never votes.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import face_check as fc
import face_privacy

README = """\
Offline F1 recheck of {run} (V2-25 face device check), made {utc} at {commit}.

The run's own result is unchanged and stays as recorded: F1_exit={f1_exit} in its provenance.txt, and its
f1/check.txt ends "{verdict}". Nothing in the run directory was changed.

This directory holds a corrected reading made afterwards, with code changed afterwards (the privacy counts' method):
- the names were counted again in F1's evidence (secret-counts-names.txt; whole tokens in content for short names,
  substrings for longer ones), with every file read against its profile (nonconforming parts counted);
- the passphrase and photo-file-name counts were NOT rescanned: check.txt takes them from the run's own
  f1/secret-counts-identity.txt, verified by the run's SHA256SUMS (provenance.txt records its SHA-256);
- every other F1 item is re-evaluated by the same code and compared with the run's f1/check.txt.

Whether this reading stands as F1's result is the maintainer's decision (face_recheck.py accept). Until then F2
cannot build on this run.
"""


def _write(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def run(run_dir: Path, identity_dir: Path, out: Path, *, pin: str, commit: str, stdin: bytes,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> tuple[bool, list[str]]:
    if out.exists():
        raise face_privacy.Refused(f"{out} exists")
    if not (run_dir / "SHA256SUMS").is_file():
        raise face_privacy.Refused("the run has no SHA256SUMS")
    names = face_privacy.run("f1", run_dir / "f1", stdin, audit=identity_dir / "audit.jsonl", keys=("name",))
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    out.mkdir(mode=0o700)
    utc = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    _write(out / fc.NAME_COUNTS, "".join(f"{line}\n" for line in names))
    listed, problems = fc._manifest(run_dir)
    source = fc._sha256(run_dir / fc.CARRIED_SOURCE)
    provenance = {
        "recheck_utc": utc, "run": run_dir.name, "run_manifest_sha256": fc._sha256(run_dir / "SHA256SUMS"),
        "run_manifest_check": "verified" if listed is not None and not problems else f"failed ({len(problems)})",
        "identity_id": fc._enrolled_id(run_dir / "f1") or "unknown",
        "audit_sha256": fc._sha256(identity_dir / "audit.jsonl"), "audit_in_run_manifest": "no",
        "commit": commit, "pin_commit": pin, "names": "recounted",
        "passphrase_counts": "carried_forward", "filename_counts": "carried_forward",
        "carried_source": fc.CARRIED_SOURCE, "carried_source_sha256": source,
        "carried_source_in_run_manifest": "yes" if listed is not None and listed.get(fc.CARRIED_SOURCE) == source
        else "no",
    }
    _write(out / "provenance.txt", "".join(f"{k}={v}\n" for k, v in provenance.items()))
    reading = fc.read_f1_recheck(out, run_dir, identity_dir)
    lines = [*reading.lines, f"face-f1-recheck: {'validated' if reading.ok else 'not validated'}"]
    _write(out / "check.txt", "".join(f"{line}\n" for line in lines))
    exits = [x.split("=", 1)[1] for x in (run_dir / "provenance.txt").read_text().splitlines()
             if x.startswith("F1_exit=")] if (run_dir / "provenance.txt").is_file() else []
    _write(out / "README.txt", README.format(run=run_dir.name, utc=utc, commit=commit,
                                             f1_exit=exits[-1] if exits else "(none)",
                                             verdict=fc._last(run_dir / "f1/check.txt")))
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    _write(out / "SHA256SUMS", "".join(f"{fc._sha256(out / name)}  ./{name}\n" for name in files))
    return reading.ok, [*lines, f"recheck: {out}", f"recheck SHA256SUMS: {fc._sha256(out / 'SHA256SUMS')}"]


def accept(recheck: Path, run_dir: Path, identity_dir: Path, *, ask: TextIO, say: Callable[[str], None],
           utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    if (recheck / fc.ACCEPTANCE).exists():
        say(f"STOP: {recheck / fc.ACCEPTANCE} exists; an acceptance is never changed")
        return 1
    listed, problems = fc._manifest(recheck, ignore=frozenset({fc.ACCEPTANCE}))
    if listed is None or problems:
        say(f"STOP: the recheck directory changed since it was written ({'; '.join(problems)})")
        return 1
    if fc._last(recheck / "check.txt") != fc.RECHECK_VERDICT:
        say("STOP: the recheck's reading did not validate; there is nothing to accept")
        return 1
    reading = fc.read_f1_recheck(recheck, run_dir, identity_dir)
    if not reading.ok:
        say("STOP: the recheck no longer validates now:")
        for line in reading.lines:
            say(f"  {line}")
        return 1
    identity = fc._enrolled_id(run_dir / "f1")
    expected = f"ACCEPT {run_dir.name} {identity}"
    say(f"Run: {run_dir}\nEnrolled identity (opaque): {identity}\nCorrected F1 reading: {recheck}\n")
    for line in reading.lines:
        say(f"  {line}")
    say("\nAccepting records your decision that this corrected reading stands as F1's result for this run and this "
        "identity, so that F2 may build on it. The run's own result stays as recorded. Any later change to the run, "
        "the recheck or the audit log voids it.")
    say(f"To accept, type exactly: {expected}")
    typed = ask.readline().rstrip("\n")
    if typed != expected:
        say("Not accepted; nothing was written.")
        return 1
    fields = {
        "accepted_reading": "f1-recheck", "run": run_dir.name, "identity_id": identity,
        "run_manifest_sha256": fc._sha256(run_dir / "SHA256SUMS"),
        "recheck_manifest_sha256": fc._sha256(recheck / "SHA256SUMS"),
        "recheck_check_sha256": fc._sha256(recheck / "check.txt"),
        "audit_sha256": fc._sha256(identity_dir / "audit.jsonl"),
        "accepted_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"), "confirmation": typed,
    }
    _write(recheck / fc.ACCEPTANCE, "".join(f"{k}={v}\n" for k, v in fields.items()))
    say(f"Accepted: {recheck / fc.ACCEPTANCE} (SHA-256 {fc._sha256(recheck / fc.ACCEPTANCE)})")
    return 0


CAPTURE_README = """\
Offline capture recheck of {session} (the enrollment experiment's capture, D91), made {utc} at {commit}.

The session's own result is unchanged and stays as recorded: XC_check_exit={exit} in its provenance.txt, and its
xc/check.txt ends "{verdict}". Nothing in the session directory or in the working gallery was changed.

This directory holds a corrected reading made afterwards, with code changed afterwards (the privacy profiles):
- the names were counted again in the capture's evidence, with every file read against the current profiles
  (secret-counts-names.txt; nonconforming parts counted);
- the passphrase and camera userinfo counts were NOT rescanned: they search every byte whatever the profiles say, and
  check.txt takes them from the session's own xc/ counts files, verified by the session's SHA256SUMS;
- every other capture item was read again by the same code and compared with the session's xc/check.txt;
- the working gallery was checked by its SHA-256 only to be the file the capture wrote.

Whether this reading stands as the capture's result is the maintainer's decision (face_recheck.py accept-capture).
It adopts nothing: the working gallery never votes, and face recognition stays PENDING_VALIDATION.
"""


def run_capture(session: Path, f1: Path, run_dir: Path, identity_dir: Path, recheck_root: Path | None,
                working_gallery: Path, out: Path, *, pin: str, commit: str, stdin: bytes,
                utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> tuple[bool, list[str]]:
    if not re.fullmatch(r"exp-\d{8}T\d{6}Z", session.name):
        raise face_privacy.Refused("the session is not an experiment session (exp-<UTC>)")
    if not re.fullmatch(rf"{session.name}-xc-[0-9a-f]{{7}}-\d{{8}}T\d{{6}}Z", out.name):
        raise face_privacy.Refused("the recheck directory's name must be <session>-xc-<pin>-<UTC>")
    if out.exists():
        raise face_privacy.Refused(f"{out} exists")
    if not (session / "SHA256SUMS").is_file():
        raise face_privacy.Refused("the session has no SHA256SUMS")
    names = face_privacy.run("f2", session / "xc", stdin, keys=("name",))
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    out.mkdir(mode=0o700)
    utc = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    _write(out / fc.NAME_COUNTS, "".join(f"{line}\n" for line in names))
    listed, problems = fc._manifest(session)
    provenance = {
        "recheck_utc": utc, "session": session.name, "session_manifest_sha256": fc._sha256(session / "SHA256SUMS"),
        "session_manifest_check": "verified" if listed is not None and not problems else f"failed ({len(problems)})",
        "working_gallery_sha256": fc._sha256(working_gallery / "gallery.sealed"), "f1_run": run_dir.name,
        "commit": commit, "pin_commit": pin, "names": "recounted", "passphrase_counts": "carried_forward",
        "userinfo_counts": "carried_forward",
        "carried_identity_counts_sha256": fc._sha256(session / "xc/secret-counts-identity.txt"),
        "carried_camera_counts_sha256": fc._sha256(session / "xc/secret-counts-camera.txt"),
    }
    _write(out / "provenance.txt", "".join(f"{k}={v}\n" for k, v in provenance.items()))
    reading = fc.read_capture_recheck(out, session, f1, (run_dir, identity_dir, recheck_root), working_gallery)
    lines = [*reading.lines, fc.CAPTURE_RECHECK_VERDICT if reading.ok else "face-capture-recheck: not validated"]
    _write(out / "check.txt", "".join(f"{line}\n" for line in lines))
    exits = fc._values(session / "provenance.txt", "XC_check_exit")
    _write(out / "README.txt", CAPTURE_README.format(session=session.name, utc=utc, commit=commit,
                                                     exit=exits[-1] if exits else "(none)",
                                                     verdict=fc._last(session / "xc/check.txt")))
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    _write(out / "SHA256SUMS", "".join(f"{fc._sha256(out / name)}  ./{name}\n" for name in files))
    return reading.ok, [*lines, f"recheck: {out}", f"recheck SHA256SUMS: {fc._sha256(out / 'SHA256SUMS')}"]


def accept_capture(recheck: Path, session: Path, f1: Path, run_dir: Path, identity_dir: Path,
                   recheck_root: Path | None, working_gallery: Path, *, ask: TextIO, say: Callable[[str], None],
                   utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    if (recheck / fc.ACCEPTANCE).exists():
        say(f"STOP: {recheck / fc.ACCEPTANCE} exists; an acceptance is never changed")
        return 1
    listed, problems = fc._manifest(recheck, ignore=frozenset({fc.ACCEPTANCE}))
    if listed is None or problems:
        say(f"STOP: the recheck directory changed since it was written ({'; '.join(problems)})")
        return 1
    if fc._last(recheck / "check.txt") != fc.CAPTURE_RECHECK_VERDICT:
        say("STOP: the recheck's reading did not validate; there is nothing to accept")
        return 1
    reading = fc.read_capture_recheck(recheck, session, f1, (run_dir, identity_dir, recheck_root), working_gallery)
    if not reading.ok:
        say("STOP: the recheck no longer validates now:")
        for line in reading.lines:
            say(f"  {line}")
        return 1
    expected = f"ACCEPT {session.name} capture"
    say(f"Session: {session}\nCorrected capture reading: {recheck}\n")
    for line in reading.lines:
        say(f"  {line}")
    say("\nAccepting records your decision that this corrected reading stands as the capture's result for this "
        "session, so that its working gallery may be compared read only in the experiment's evaluation sessions. The "
        "session's own result stays as recorded. It adopts nothing: the working gallery never votes, and face "
        "recognition stays PENDING_VALIDATION. Any later change to the session, the recheck or the working gallery "
        "voids it.")
    say(f"To accept, type exactly: {expected}")
    typed = ask.readline().rstrip("\n")
    if typed != expected:
        say("Not accepted; nothing was written.")
        return 1
    fields = {
        "accepted_reading": "capture-recheck", "session": session.name,
        "session_manifest_sha256": fc._sha256(session / "SHA256SUMS"),
        "recheck_manifest_sha256": fc._sha256(recheck / "SHA256SUMS"),
        "recheck_check_sha256": fc._sha256(recheck / "check.txt"),
        "working_gallery_sha256": fc._sha256(working_gallery / "gallery.sealed"),
        "accepted_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"), "confirmation": typed,
    }
    _write(recheck / fc.ACCEPTANCE, "".join(f"{k}={v}\n" for k, v in fields.items()))
    say(f"Accepted: {recheck / fc.ACCEPTANCE} (SHA-256 {fc._sha256(recheck / fc.ACCEPTANCE)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("run", help="write the offline F1 recheck (names on stdin)")
    make.add_argument("run", type=Path)
    make.add_argument("--identity-dir", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--pin", required=True)
    make.add_argument("--commit", required=True)
    decide = commands.add_parser("accept", help="the maintainer's explicit acceptance (typed)")
    decide.add_argument("recheck", type=Path)
    decide.add_argument("--run", type=Path, required=True)
    decide.add_argument("--identity-dir", type=Path, required=True)
    capture = commands.add_parser("capture", help="write the offline recheck of an experiment's capture (names on stdin)")
    capture.add_argument("session", type=Path)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--pin", required=True)
    capture.add_argument("--commit", required=True)
    decide_capture = commands.add_parser("accept-capture", help="the maintainer's explicit acceptance (typed)")
    decide_capture.add_argument("recheck", type=Path)
    decide_capture.add_argument("session", type=Path)
    for sub in (capture, decide_capture):
        sub.add_argument("--f1", type=Path, required=True, help="F1's evidence directory (the run's f1/)")
        sub.add_argument("--run", type=Path, required=True, help="the run whose F1 the experiment builds on")
        sub.add_argument("--identity-dir", type=Path, required=True, help="the enrolled gallery's directory (read only)")
        sub.add_argument("--recheck-root", type=Path, help="where accepted corrected F1 readings are")
        sub.add_argument("--working-gallery", type=Path, required=True, help="the experiment's working gallery directory")
    args = parser.parse_args(argv)
    if args.command == "accept":
        return accept(args.recheck, args.run, args.identity_dir, ask=sys.stdin, say=print)
    if args.command == "accept-capture":
        return accept_capture(args.recheck, args.session, args.f1, args.run, args.identity_dir, args.recheck_root,
                              args.working_gallery, ask=sys.stdin, say=print)
    try:
        if args.command == "capture":
            ok, lines = run_capture(args.session, args.f1, args.run, args.identity_dir, args.recheck_root,
                                    args.working_gallery, args.out, pin=args.pin, commit=args.commit,
                                    stdin=sys.stdin.buffer.read())
        else:
            ok, lines = run(args.run, args.identity_dir, args.out, pin=args.pin, commit=args.commit,
                            stdin=sys.stdin.buffer.read())
    except face_privacy.Refused as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
