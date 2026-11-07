"""The multi-pose comparison's builder: one frozen, deployable candidate from session B's raw capture (PROPOSED;
validated with synthetic data only, never run on real faces).

usage:
  mpe_build.py build --session B_DIR --raw RAW_DIR --identity-dir ORIGINAL --candidate NEW_DIR --root EXP_ROOT
                     --config FACE_YAML
  mpe_build.py verify-frozen --candidate DIR --root EXP_ROOT --identity-dir ORIGINAL

The passphrase comes from SENTINEL_IDENTITY_PASSPHRASE or a hidden prompt, as `sentinel run` takes it.

**The candidate** is a complete working gallery of the enrolled person, at most MAX_PROTOTYPES (8) prototypes, one per
cued pose window of session B (mpe_schedule.B_WINDOWS: the seat's lens and down, each mark's lens, left and right),
written through D91's writer (identity.experiment.write_working_gallery) into a new private directory, never the
original's. The original gallery is only read (its sealed file's SHA-256 is compared before and after).

**Selection uses B's build data only, by time:** in each window, from POSE_SETTLE_S after its pose bell to the next
bell (the bells' printed times), the face nearest the middle (the earlier on a tie). Never a label, vote, identity,
similarity or anything from an evaluation session. A window the account marks not done as cued gives nothing; an empty
window gives nothing and nothing replaces it.

**Refused before anything is written** (fixed labels): a path that is relative, passes through a link or holds '..'
(``path_not_canonical``); a raw capture or candidate not strictly inside the experiment root, or an original gallery or
session inside it, holding it or overlapping either (``raw_capture_outside_root``, ``candidate_outside_root``,
``paths_overlap``); a frozen record or a candidate directory already there or listed before; the session is not B's
schedule; B's account is missing, invalid or says anyone else was in view; the raw capture is missing, unreadable, not a
private directory holding exactly its one sealed file, from another boot, of another identity or pipeline (its
compatibility record or a face's length), or holds a face that is not one of B's own applied one-person compared
results; fewer than CAPTURE_MIN_FACES faces inside the windows (``too_few_faces``) or fewer than CAPTURE_MIN_PROTOTYPES
prototypes (``too_few_prototypes``); enrollment's consistency rule unchanged (``inconsistent_capture``: a prototype
whose best similarity to the others is under the match threshold); a second build from everything read again that
differs (``build_not_reproducible``). Each of these reads NOT_BUILT. Anything unexpected is refused with a fixed label.

**Listed, then frozen:** the raw capture's directory and file are listed in the manifest (class raw) only after its own
checks, never a tree the builder did not make; the candidate's new directory and everything the writer put in it
(class candidate) right after the write. The candidate's digest (SHA-256 over the chosen windows, their faces' frame
times and the prototypes' exact values, never over the sealed file) and its prototypes' digest (the one a route B
shadow run records at its start) go into EXP_ROOT/candidate.frozen.json (no vector; class output) after the written
gallery reads back with exactly those prototypes. ``verify-frozen`` recomputes both from the gallery, for an evaluation
session's gate. Only then is this build's raw capture deleted, through the manifest (mpe_manifest), and nothing else.

Prints counts and fixed labels only: never an embedding, a similarity value, an identity ID or the passphrase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mpe_account
import mpe_manifest
import mpe_schedule
from sentinel.identity.experiment import (
    CAPTURE_MIN_FACES,
    CAPTURE_MIN_PROTOTYPES,
    RAW_FILE,
    ExperimentRefused,
    open_raw_capture,
    prototypes_digest,
    write_working_gallery,
)
from sentinel.identity.gallery import GalleryDocument, IdentityStore, unit
from sentinel.identity.matching import MAX_PROTOTYPES, unit_cosine
from sentinel.identity.vault import Sealer, Secret, VaultError, write_private_atomic

FROZEN_NAME = "candidate.frozen.json"
FROZEN_FORMAT = "mpe-frozen-1"
ACCOUNT_NAME = mpe_schedule.ACCOUNT_NAME
COMPARED_LABELS = ("match", "unknown", "margin")  # a vote that computed a similarity: what the capture is offered


class BuildRefused(Exception):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def read_account(session: Path, kind: str, windows: set[str]) -> dict[str, Any]:
    """The session's locked account (mpe_account.verify: there, the session's, unchanged since it was saved, saved
    before any reading began), its fixed labels as refusals."""
    try:
        return mpe_account.verify(session, kind)
    except mpe_account.AccountRefused as exc:
        raise BuildRefused(exc.label) from None


def provenance_boot(session: Path) -> str | None:
    path = session / "provenance.txt"
    if not path.is_file():
        return None
    for line in path.read_text("utf-8").splitlines():
        if line.startswith("boot_id="):
            return line.partition("=")[2].strip()
    return None


def compared_results(session: Path) -> set[tuple[int, int, int]]:
    """(stream epoch, frame number, frame time) of every applied result with exactly one person whose vote computed a
    similarity: the faces a capture in that session could have been offered."""
    out = set()
    for line in (session / "run.jsonl").read_text("utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("identity") != "result" or record.get("outcome") != "applied":
            continue
        persons = record.get("persons") or []
        if len(persons) == 1 and persons[0].get("label") in COMPARED_LABELS:
            out.add((record["stream_epoch"], record["frame_seq"], record["frame_mono_ns"]))
    return out


def select_windows(faces: list[dict[str, Any]], printed: dict[str, int],
                   excluded: set[str]) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """The predeclared rule: per B window (in schedule order) not excluded, the face nearest the middle of
    [pose bell + POSE_SETTLE_S, next bell), the earlier on a tie: ([(window, index into faces)], faces per window)."""
    settle = round(mpe_schedule.POSE_SETTLE_S * 1e9)
    chosen, counts = [], {}
    for window in mpe_schedule.B_WINDOWS:
        if window.name in excluded:
            continue
        start, end = printed[window.bell] + settle, printed[window.until]
        inside = [i for i, face in enumerate(faces) if start <= face["t"] < end]
        counts[window.name] = len(inside)
        if inside:
            middle = (start + end) / 2
            chosen.append((window.name, min(inside, key=lambda i: (abs(faces[i]["t"] - middle), faces[i]["t"], i))))
    return chosen, counts


def candidate_digest(windows: list[tuple[str, int]], prototypes: list[tuple[float, ...]]) -> str:
    """SHA-256 over the chosen windows, their faces' frame times and the prototypes' exact values (repr round-trips)."""
    canonical = json.dumps({"windows": [[name, t] for name, t in windows],
                            "prototypes": [[float(x) for x in p] for p in prototypes]},
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _paths_problem(session: Path, raw_dir: Path, identity_dir: Path, candidate_dir: Path, root: Path) -> str | None:
    """Before anything is read or listed: every path absolute and canonical (no link, no '..'); the raw capture and the
    candidate strictly inside the experiment root and apart from each other; the original gallery and the session
    neither inside the root nor holding it, and never the raw capture's or the candidate's place."""
    for path in (session, raw_dir, identity_dir, candidate_dir, root):
        if not path.is_absolute() or path.resolve() != path:
            return "path_not_canonical"
    if raw_dir == root or not _inside(raw_dir, root):
        return "raw_capture_outside_root"
    if candidate_dir == root or not _inside(candidate_dir, root):
        return "candidate_outside_root"
    if _inside(raw_dir, candidate_dir) or _inside(candidate_dir, raw_dir):
        return "paths_overlap"
    for other in (identity_dir, session):
        if _inside(other, root) or _inside(root, other) or _inside(raw_dir, other) or _inside(candidate_dir, other):
            return "paths_overlap"
    return None


def _read_inputs(session: Path, raw_dir: Path, *, secret: Secret, sealer: Sealer) -> tuple[dict, dict, dict]:
    """result.json, the account and the raw capture, each read afresh (the second build reads them all again)."""
    result = json.loads((session / "result.json").read_text("utf-8"))
    if mpe_schedule.schedule_of(result) != "B":
        raise BuildRefused("not_a_build_session")
    account = read_account(session, "B", {w.name for w in mpe_schedule.B_WINDOWS})
    try:
        raw = open_raw_capture(raw_dir, sealer, secret)
    except ExperimentRefused as exc:
        raise BuildRefused(exc.label) from None
    return result, account, raw


def _one_build(faces: list[dict[str, Any]], printed: dict[str, int], excluded: set[str]):
    chosen, counts = select_windows(faces, printed, excluded)
    prototypes = [unit(faces[i]["e"]) for _, i in chosen]
    windows = [(name, faces[i]["t"]) for name, i in chosen]
    return windows, prototypes, counts


def build(session: Path, raw_dir: Path, identity_dir: Path, candidate_dir: Path, root: Path, *, secret: Secret,
          sealer: Sealer, match_threshold: float,
          utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc).replace(microsecond=0)) -> dict[str, Any]:
    """Build, freeze and clean as the module says. The report: counts and fixed labels only. Never raises: anything
    unexpected is refused with a fixed label (its class name), never its text."""
    session, raw_dir, identity_dir, candidate_dir, root = (Path(x) for x in (session, raw_dir, identity_dir,
                                                                             candidate_dir, root))
    report: dict[str, Any] = {"outcome": "NOT_BUILT", "refused": None, "prototypes": 0, "faces_in_windows": 0,
                              "faces_per_window": {}, "windows_excluded": [], "original_unchanged": None,
                              "frozen": False, "raw_deleted": False}
    original_file = IdentityStore(identity_dir, sealer).gallery_path
    before = mpe_manifest.sha256_file(original_file) if original_file.is_file() else None
    reserved = made = False  # made: this build reserved (and so owns) the candidate directory
    try:
        problem = _paths_problem(session, raw_dir, identity_dir, candidate_dir, root)
        if problem is not None:
            raise BuildRefused(problem)
        if len(mpe_schedule.B_WINDOWS) > MAX_PROTOTYPES:
            raise BuildRefused("too_many_windows")
        try:  # the account step comes first: nothing of the session is read or shown before its locked account
            mpe_account.require(session, "B")
        except mpe_account.AccountRefused as exc:
            raise BuildRefused(exc.label) from None
        manifest = mpe_manifest.Manifest(root)
        frozen_path = root / FROZEN_NAME
        listed = manifest.entries()
        if os.path.lexists(frozen_path) or frozen_path in listed:
            raise BuildRefused("frozen_record_exists")
        if os.path.lexists(candidate_dir) or candidate_dir in listed or candidate_dir in manifest.reservations():
            raise BuildRefused("candidate_target_exists")
        if not os.path.lexists(raw_dir):
            raise BuildRefused("raw_capture_missing")
        raw_entry = listed.get(raw_dir)  # the launch block reserved it before the run and claimed it after
        if raw_entry is None or raw_entry.cls != "raw" or raw_entry.deleted or raw_dir / RAW_FILE not in listed:
            raise BuildRefused("raw_capture_not_claimed")
        result, account, raw = _read_inputs(session, raw_dir, secret=secret, sealer=sealer)  # its own checks first
        if account["others_in_view"] != "no":
            raise BuildRefused("others_in_view")
        report["windows_excluded"] = sorted(account["not_as_cued"])
        original = IdentityStore(identity_dir, sealer).load(secret)
        if len(original.identities) != 1:
            raise BuildRefused("original_needs_one_identity")
        if raw["identity_id"] != original.identities[0].identity_id:
            raise BuildRefused("raw_capture_other_identity")
        dimension = original.compatibility.dimension
        if raw["compatibility"] != original.compatibility.revision or any(
                len(f["e"]) != dimension or not all(isinstance(x, float) for x in f["e"]) for f in raw["faces"]):
            raise BuildRefused("raw_capture_other_pipeline")
        boot = provenance_boot(session)
        if boot is None or raw["boot_id"] != boot:
            raise BuildRefused("raw_capture_other_boot")
        offered = compared_results(session)
        faces = raw["faces"]
        if not faces or any((f["epoch"], f["seq"], f["t"]) not in offered for f in faces):
            raise BuildRefused("raw_capture_not_this_session")
        windows, prototypes, counts = _one_build(faces, mpe_schedule.printed_ns(result), account["not_as_cued"])
        report.update(faces_per_window=counts, faces_in_windows=sum(counts.values()), prototypes=len(prototypes))
        if report["faces_in_windows"] < CAPTURE_MIN_FACES:
            raise BuildRefused("too_few_faces")
        if len(prototypes) < CAPTURE_MIN_PROTOTYPES:
            raise BuildRefused("too_few_prototypes")
        if len(prototypes) > MAX_PROTOTYPES:
            raise BuildRefused("too_many_prototypes")
        best_to_others = [max(unit_cosine(p, q) for j, q in enumerate(prototypes) if j != i)
                          for i, p in enumerate(prototypes)]
        if min(best_to_others) < match_threshold:
            raise BuildRefused("inconsistent_capture")
        digest = candidate_digest(windows, prototypes)
        result2, account2, raw2 = _read_inputs(session, raw_dir, secret=secret, sealer=sealer)  # read again, built again
        again = _one_build(raw2["faces"], mpe_schedule.printed_ns(result2), account2["not_as_cued"])
        if candidate_digest(again[0], again[1]) != digest:
            raise BuildRefused("build_not_reproducible")
        store = IdentityStore(candidate_dir, sealer)
        manifest.reserve(candidate_dir, "candidate", "gallery", "mpe_build.py build")  # a kill from here: recoverable
        reserved = made = True
        written, refused = write_working_gallery(store, secret, original, raw["consent_date"], prototypes, utc_now(),
                                                 faces_in_span=report["faces_in_windows"])
        if os.path.lexists(candidate_dir):  # what D91's writer left in the reserved directory, nothing else
            manifest_claim = mpe_manifest.claim(manifest, candidate_dir, writer_exited=True)
            if manifest_claim["unexpected"]:
                raise BuildRefused("candidate_unexpected_files")
        else:
            mpe_manifest.close_absent(manifest, candidate_dir, writer_exited=True, recorded_written=False)
        reserved = False
        if not written or refused is not None:
            raise BuildRefused(refused or "capture_write_failed")
        back = store.load(secret)
        read_back = [tuple(p) for p in back.identities[0].prototypes]
        if read_back != [tuple(p) for p in prototypes] or candidate_digest(windows, read_back) != digest:
            raise BuildRefused("candidate_not_verified")
        frozen = {"format": FROZEN_FORMAT, "digest": digest, "prototypes_digest": prototypes_digest(read_back),
                  "prototypes": len(prototypes), "windows": [[name, t] for name, t in windows],
                  "faces_per_window": counts, "faces_in_windows": report["faces_in_windows"],
                  "windows_excluded": report["windows_excluded"], "built_utc": utc_now().isoformat(timespec="seconds")}
        write_private_atomic(frozen_path, (json.dumps(frozen, sort_keys=True, indent=1) + "\n").encode("ascii"))
        manifest.add(frozen_path, "output", "mpe_build.py build")
        report.update(outcome="BUILT", frozen=True, digest=digest)
        cleaned = mpe_manifest.cleanup(manifest, ("raw",), within=raw_dir,  # this build's raw capture only
                                       protected=(identity_dir, session))
        report["raw_deleted"] = not os.path.lexists(raw_dir) and not cleaned["left_changed"] \
            and not cleaned["left_not_empty"]
        if not report["raw_deleted"]:
            report["refused"] = "raw_capture_left"
    except BuildRefused as exc:
        report["refused"] = exc.label
    except VaultError as exc:
        report["refused"] = f"vault:{exc.label}"
    except mpe_manifest.ManifestError:
        report["refused"] = "manifest_refused"
    except Exception as exc:  # noqa: BLE001 - a fixed label only: an exception's text can hold a path or an ID
        report["refused"] = f"failed:{type(exc).__name__}"
    finally:
        if report["outcome"] != "BUILT" and made:  # only what this build made; never an earlier build's candidate
            report["candidate_removed"] = _remove_failed_candidate(candidate_dir, root, reserved,
                                                                   (identity_dir, session))
        after = mpe_manifest.sha256_file(original_file) if original_file.is_file() else None
        report["original_unchanged"] = before is not None and before == after
    return report


def _remove_failed_candidate(candidate_dir: Path, root: Path, reserved: bool, protected: tuple[Path, ...]) -> bool:
    """A build that did not freeze its candidate removes what it wrote: only listed candidate entries under the
    candidate's own directory (claimed first if the write stopped half-way), never anything protected. True when no
    candidate directory is left."""
    try:
        manifest = mpe_manifest.Manifest(root)
        if reserved and candidate_dir in manifest.reservations() and \
                manifest.reservations()[candidate_dir].closed is None:
            if os.path.lexists(candidate_dir):
                mpe_manifest.claim(manifest, candidate_dir, writer_exited=True)
            else:
                mpe_manifest.close_absent(manifest, candidate_dir, writer_exited=True, recorded_written=False)
        if candidate_dir in manifest.entries():
            mpe_manifest.cleanup(manifest, ("candidate",), within=candidate_dir, protected=protected)
    except Exception:  # noqa: BLE001 - a fixed outcome only; recovery (mpe_manifest recover) finishes the work
        pass
    return not os.path.lexists(candidate_dir)


def verify_frozen(candidate_dir: Path, root: Path, identity_dir: Path, *, secret: Secret, sealer: Sealer) -> dict:
    """For an evaluation session's gate: the candidate in ``candidate_dir`` is exactly the frozen one (its prototypes'
    digest with the frozen windows), opens as a shadow of the original, and holds at most MAX_PROTOTYPES prototypes."""
    frozen = json.loads((Path(root) / FROZEN_NAME).read_text("ascii"))
    if frozen.get("format") != FROZEN_FORMAT:
        return {"frozen_candidate": "invalid_record"}
    document: GalleryDocument = IdentityStore(candidate_dir, sealer).load(secret)
    original = IdentityStore(identity_dir, sealer).load(secret)
    prototypes = [tuple(p) for identity in document.identities for p in identity.prototypes]
    same_identity = [i.identity_id for i in document.identities] == [i.identity_id for i in original.identities]
    digest = candidate_digest([tuple(w) for w in frozen["windows"]], prototypes)
    ok = digest == frozen["digest"] and prototypes_digest(prototypes) == frozen.get("prototypes_digest") \
        and len(prototypes) == frozen["prototypes"] <= MAX_PROTOTYPES and same_identity \
        and document.compatibility.first_difference(original.compatibility) is None
    return {"frozen_candidate": "matches" if ok else "differs", "prototypes": len(prototypes)}


def _secret() -> Secret:
    from sentinel.identity.vault import prompt_secret, take_secret_from_environment

    return take_secret_from_environment() or prompt_secret(confirm=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for flag in ("--session", "--raw", "--identity-dir", "--candidate", "--root", "--config", "--f1"):
        b.add_argument(flag, type=Path, required=True)
    for flag in ("--run", "--f1-identity-dir", "--recheck-root"):  # the F1 gate, as the launch block ran it
        b.add_argument(flag, type=Path, default=None)
    v = sub.add_parser("verify-frozen")
    for flag in ("--candidate", "--root", "--identity-dir"):
        v.add_argument(flag, type=Path, required=True)
    args = parser.parse_args(argv)
    secret = _secret()
    sealer = Sealer()
    if args.command == "verify-frozen":
        result = verify_frozen(args.candidate, args.root, args.identity_dir, secret=secret, sealer=sealer)
        print(f"frozen_candidate={result['frozen_candidate']}")
        return 0 if result["frozen_candidate"] == "matches" else 1
    from sentinel.config import load_config

    import mpe_check

    session = args.session.absolute()
    if not (session.parent / "face.yaml").is_file() or args.config.read_bytes() != (session.parent /
                                                                                     "face.yaml").read_bytes():
        print("STOP: config_not_the_sessions (--config must be the session's own face.yaml)", file=sys.stderr)
        return 1
    gate = (args.run, args.f1_identity_dir, args.recheck_root) if args.run else None
    ok, lines = mpe_check.passed(session, args.f1, "B", None, gate)  # the execution check first (and the account)
    if not ok:
        print("\n".join(lines))
        print("STOP: execution_not_passed (nothing is read or built from a session that has not passed)",
              file=sys.stderr)
        return 1
    print(lines[-1])  # mpe-execution: passed
    threshold = load_config(args.config).identity.match_threshold
    report = build(args.session.absolute(), args.raw.absolute(), args.identity_dir.absolute(),
                   args.candidate.absolute(), args.root.absolute(), secret=secret, sealer=sealer,
                   match_threshold=threshold)
    for key in ("outcome", "refused", "prototypes", "faces_in_windows", "windows_excluded", "original_unchanged",
                "frozen", "raw_deleted"):
        print(f"{key}={report[key]}")
    for name, count in report["faces_per_window"].items():
        print(f"faces_in_window {name}={count}")
    return 0 if report["outcome"] == "BUILT" and report["refused"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
