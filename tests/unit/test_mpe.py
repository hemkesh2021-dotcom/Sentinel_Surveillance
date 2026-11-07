"""The multi-pose comparison's tooling, with synthetic embeddings only (no image, camera or model of anyone).

The real runtime (DemoRuntime on route B, through test_face_check.simulate) runs each session's schedule: B with the raw
capture, then the builder freezes one candidate of at most 8 prototypes, then E_M and E_P with the candidate compared
in shadow; the reader pairs the full-precision decisions and replays the identity history; the manifest cleanup
deletes only what it listed. Faces are fixed 512-value directions with seeded noise: the original gallery is one
photo-like direction, the camera faces another, the cued turns and the participant others."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from face_runs import COMPAT, ENROLLED, FACE_BOX, GALLERY, RUNNERS, real_face_startup, startup_line, vec
from sentinel.config import parse_config
from sentinel.contracts import FrameKey, NormalizedBox
from sentinel.identity.association import FaceObservation, Ownership
from sentinel.identity.experiment import RAW_FILE, ExperimentRefused, RawCapture, open_raw_capture, open_shadow
from sentinel.identity.gallery import CONSENT_SCOPE, GalleryDocument, GalleryIdentity, IdentityStore, unit
from sentinel.identity.legacy_deepface import FaceRun
from sentinel.identity.matching import MAX_PROTOTYPES, EnrolledIdentity, Enrollment, best_match
from sentinel.identity.state import IdentityPolicy, IdentityResolver, decide, decision_label
from sentinel.identity.vault import Sealer, Secret, VaultError
from sentinel.runtime import identity_policy

SECRET = Secret("a long synthetic passphrase")
CONSENT = "2026-11-07"
BOOT = "fake-boot-1"  # FakeClock's
POLICY = identity_policy(parse_config({"config_version": 1, "camera": {"id": "cam-1"}}).identity)

# The faces (directions in the 512-value space; the original's one prototype is the first axis)
PHOTO = vec(0.9, 0.0, 0.0, 0.0, 0.0, 0.43)  # a close, photo-like face: the original matches it, the camera's lens not
LENS = vec(0.75, 0.66)  # the camera's frontal face at range: the original matches it (0.75)
LEFT = vec(0.5, 0.2, 0.84)  # cued turns: the original does not match them
RIGHT = vec(0.5, 0.2, 0.0, 0.84)
DOWN = vec(0.6, 0.5, 0.0, 0.0, 0.62)
STRANGER = vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # the participant: nothing matches
IMPOSTOR_LEFT = vec(0.3, 0.2, 0.9)  # the participant turned left, close to the candidate's left prototype
POSE_FACES = {"lens": LENS, "left": LEFT, "right": RIGHT, "down": DOWN}


@pytest.fixture(scope="module")
def sealer() -> Sealer:
    made = Sealer(kdf_n=2**14)
    try:
        made.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    return made


@pytest.fixture(scope="module")
def mods():
    import sys

    sys.path.insert(0, str(RUNNERS))
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # test_face_check's simulate: the real runtime's run
    try:
        return SimpleNamespace(schedule=importlib.import_module("mpe_schedule"),
                               manifest=importlib.import_module("mpe_manifest"),
                               build=importlib.import_module("mpe_build"),
                               compare=importlib.import_module("mpe_compare"),
                               account=importlib.import_module("mpe_account"),
                               check=importlib.import_module("mpe_check"),
                               sim=importlib.import_module("test_face_check"))
    finally:
        sys.path.remove(str(RUNNERS))
        sys.path.remove(str(Path(__file__).resolve().parent))


def noisy(face, seed: int, spread: float = 0.004):  # per value: about 0.09 in all over 512 values
    rng = random.Random(seed)
    return unit(tuple(x + rng.gauss(0.0, spread) for x in face))


def observed(embedding, quality: float = 0.95) -> FaceObservation:
    frame = FrameKey(camera_id="cam-1", boot_id=BOOT, run_id="run-1", stream_epoch=1, frame_seq=1)
    return FaceObservation(frame=frame, box=NormalizedBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2), quality=quality,
                           embedding=embedding)


# ---------------------------------------------------------------- full-precision decisions (state.decide)


def mods_key():
    from dataclasses import make_dataclass

    return make_dataclass("Key", [("boot_id", str)], frozen=True)(BOOT)


def at_cosine(c: float) -> tuple[float, ...]:
    return (c, math.sqrt(1.0 - c * c)) + (0.0,) * 510


def test_the_shadow_decides_with_the_production_function_at_full_precision_never_from_the_rounded_score() -> None:
    gallery = Enrollment(model_revision="r", dimension=512,
                         identities=(EnrolledIdentity(identity_id=ENROLLED, prototypes=(vec(1.0),)),))
    from sentinel.identity.experiment import Shadow

    shadow = Shadow(gallery, 1)
    for c, label in ((0.70 - 1e-9, "unknown"), (0.70 + 1e-9, "match"), (0.6996, "unknown"), (0.7004, "match")):
        face = observed(at_cosine(c))
        numbers = shadow.numbers(face, POLICY)
        assert numbers["similarity"] == round(best_match(face.embedding, gallery).similarity, 3)
        assert numbers["decision"] == label
        # parity: the label the face's vote gets when this gallery votes, and the shared function's
        resolver = IdentityResolver(gallery, POLICY)
        person = SimpleNamespace(key=mods_key(), observed_mono_ns=1)
        vote = resolver.observe([person], [face], SimpleNamespace(person_ownership={person.key: Ownership.ASSIGNED},
                                                                   face_of={person.key: 0}))[0]
        assert vote.label == numbers["decision"] == decision_label(decide(face, gallery, POLICY)[1])
    # rounded to the same 0.700, decided apart
    below, above = (shadow.numbers(observed(at_cosine(0.70 - 1e-9)), POLICY),
                    shadow.numbers(observed(at_cosine(0.70 + 1e-9)), POLICY))
    assert below["similarity"] == above["similarity"] == 0.7 and below["decision"] != above["decision"]
    assert shadow.numbers(observed(at_cosine(0.9), quality=0.5), POLICY)["decision"] == "low_quality"


def test_the_shared_decision_is_the_production_rule_written_out_independently() -> None:
    """The rule from the guide, written out here without the code under test: quality first, then the best cosine over
    each identity's prototypes, the threshold, then the margin over the runner-up identity."""
    rng = random.Random(4)
    prototypes = {ENROLLED: (vec(1.0), vec(0.2, 1.0)), "idn-00000000000b": (vec(0.0, 0.0, 1.0),)}
    gallery = Enrollment(model_revision="r", dimension=512, identities=tuple(
        EnrolledIdentity(identity_id=k, prototypes=v) for k, v in prototypes.items()))
    policy = IdentityPolicy(match_threshold=0.7, margin=0.05, min_quality=0.6)
    seen = set()
    faces = [observed(unit(tuple(rng.gauss(0, 1) for _ in range(3)) + (0.0,) * 509), quality=rng.choice([0.5, 0.95]))
             for _ in range(400)] + [observed(vec(1.0, 0.0, 1.0))]  # the last: as close to both identities (margin)
    for face in faces:
        best = sorted(((max(sum(a * b for a, b in zip(face.embedding, q)) for q in qs), k)
                       for k, qs in prototypes.items()), reverse=True)
        if face.quality < 0.6:
            expected = "low_quality"
        elif best[0][0] < 0.7:
            expected = "unknown"
        elif best[0][0] - best[1][0] < 0.05:
            expected = "margin"
        else:
            expected = "match"
        vote, note, similarity, _ = decide(face, gallery, policy)
        assert decision_label(note) == expected
        assert vote == (best[0][1] if expected == "match" else "" if expected == "unknown" else None)
        seen.add(expected)
    assert seen == {"low_quality", "unknown", "margin", "match"}  # every branch exercised


# ---------------------------------------------------------------- sessions run through the real runtime


def presence(mods, kind: str) -> tuple[tuple, list]:
    """The script (who is in view, on which track) and the windows (start, end, face) from a session's cues: on the
    spot 3 s after the walk bell, out 3 s after the end bell; faces only from the first pose bell or the hold bell."""
    cues = dict((name, at) for name, at, _ in mods.schedule.SESSIONS[kind])
    script, windows, track = [], [], 0
    groups = [("pose", w) for w in (mods.schedule.B_WINDOWS if kind == "B" else
                                     mods.schedule.EP_WINDOWS if kind == "E_P" else ())]
    blocks = {}
    for _, w in groups:
        blocks.setdefault(w.session + w.spot, []).append(w)
    for prefix, ws in blocks.items():
        track += 1
        script.append((cues[f"{prefix}go"] + 3, cues[f"{prefix}end"] + 3, "facing", track))
        for w in ws:
            windows.append((cues[w.bell], cues[w.until], ("pose", w.spot, w.pose)))
    for hold in (mods.schedule.EM_HOLDS if kind == "E_M" else mods.schedule.EP_HOLDS if kind == "E_P" else ()):
        track += 1
        script.append((cues[hold.go] + 3, cues[hold.end] + 3, "facing", track))
        windows.append((cues[hold.hold], cues[hold.end], ("hold", hold.spot, hold.kind)))
    return tuple(sorted(script)), windows


def run_session(mods, root: Path, kind: str, face_of, *, shadow=None, capture=None, tail: bool = False,
                approach: str | None = None, lives: bool = True) -> Path:
    """A session through the real runtime on route B, then result.json (the cues as the guard prints them, 0.1 s
    late), provenance.txt (the boot) and an account. ``tail``: the person stays in view after the last end bell and
    faces the camera until the run ends (back at the Mac). ``approach``: in that hold the face is seen from the walk
    bell's arrival on, before the hold bell."""
    script, windows = presence(mods, kind)
    cues_at = dict((name, at) for name, at, _ in mods.schedule.SESSIONS[kind])
    end_s = mods.schedule.duration_s(mods.schedule.SESSIONS[kind])
    if tail:
        last = max(script, key=lambda segment: segment[1])
        script = tuple(sorted([seg for seg in script if seg is not last] + [(last[0], end_s + 1, last[2], last[3])]))
        windows = windows + [(max(w[1] for w in windows), end_s + 1, ("hold", "5", "facing"))]
    if approach is not None:
        hold = next(h for h in mods.schedule.EM_HOLDS + mods.schedule.EP_HOLDS if h.name == approach)
        windows = [(cues_at[hold.go] + 4, cues_at[hold.hold], ("hold", hold.spot, hold.kind))] + windows

    def analyze_for(t0):
        def analyze(frame, image):
            t = (frame.ingest_mono_ns - t0) / 1e9
            what = next((w[2] for w in windows if w[0] <= t < w[1]), None)
            face = None if what is None else face_of(what, t)
            if face is None:
                return FaceRun((), 0, 1)
            return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=face),), 1, 0)
        return analyze

    cues = mods.schedule.SESSIONS[kind]
    startup = {"run": "starting", "startup": {"face": {"validation_run": True, "identities_enrolled": 1,
                                                      "detection": "head_windows",
                                                      **({"shadow": True, "shadow_digest": shadow.digest,
                                                          "lives": True} if shadow is not None and lives else {}),
                                                      **({"capture": True, "capture_raw": True} if capture else {})}}}
    out = mods.sim.simulate(root, startup,
                            script=script, analyze=analyze_for, duration=float(mods.schedule.duration_s(cues)),
                            route_b=True, scene=False, shadow=shadow, experiment_capture=capture,
                            lives=shadow is not None and lives, captures=mods.schedule.STATUS_AT[kind])
    (out / "result.json").write_text(json.dumps({"cues": [{"name": n, "at_s": float(a), "printed_mono_s": 1.0 + a + 0.1}
                                                          for n, a, _ in cues], "child": {"ready_mono_s": 1.0}}))
    (out / "provenance.txt").write_text(f"boot_id={BOOT}\n")
    mods.account.enter(out, kind, others_in_view="no", not_as_cued="none")  # the account step, locked
    return out


def build_face(what, t):
    kind, spot, pose = what
    return noisy(POSE_FACES[pose], round(t * 1000))


def em_face(what, t):
    _, spot, kind = what
    n = round(t * 1000)
    if spot == "s":
        return noisy(PHOTO, n)  # the seat: photo-like, which the camera-side candidate does not cover (losses)
    if kind == "facing":
        return noisy(LENS, n)
    return noisy((LEFT, RIGHT, DOWN, LENS)[int(t // 5) % 4], n)  # natural: turns the original misses (gains)


def ep_face(what, t):
    kind, spot, detail = what
    if kind == "pose" and spot == "4" and detail == "left":
        return noisy(IMPOSTOR_LEFT, round(t * 1000))  # the participant, pose-matched to the candidate's left
    return noisy(STRANGER, round(t * 1000), 0.01)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, sealer, mods):
    """B -> build -> E_M and E_P, once for the module (each session runs the real runtime through its schedule)."""
    base = tmp_path_factory.mktemp("mpe")
    original = IdentityStore(base / "identity", sealer)
    original.ensure_directory()
    original.save(GALLERY, SECRET)
    raw = base / "experiment" / "raw"
    (base / "experiment").mkdir(mode=0o700)
    mods.manifest.Manifest(base / "experiment").reserve(raw, "raw", "raw", "test: the B launch block")
    capture = RawCapture(IdentityStore(raw, sealer), SECRET, GALLERY, CONSENT)
    (base / "b").mkdir()
    b = run_session(mods, base / "b", "B", build_face, capture=capture)
    raw_report = capture.finished
    mods.manifest.claim(mods.manifest.Manifest(base / "experiment"), raw, writer_exited=True)
    candidate = base / "experiment" / "candidate"
    report = mods.build.build(b, raw, base / "identity", candidate, base / "experiment", secret=SECRET, sealer=sealer,
                              match_threshold=POLICY.match_threshold)
    shadow = open_shadow(IdentityStore(candidate, sealer), SECRET, GALLERY) if report["outcome"] == "BUILT" else None
    (base / "em").mkdir()
    (base / "ep").mkdir()
    em = run_session(mods, base / "em", "E_M", em_face, shadow=shadow)
    ep = run_session(mods, base / "ep", "E_P", ep_face, shadow=shadow)
    frozen = json.loads((base / "experiment" / mods.build.FROZEN_NAME).read_text())
    return SimpleNamespace(base=base, b=b, raw=raw, raw_report=raw_report, candidate=candidate, report=report,
                           em=em, ep=ep, frozen=frozen, original=original)


def test_the_schedules_are_the_planned_lengths_with_alphanumeric_cues(mods) -> None:
    lengths = {kind: mods.schedule.duration_s(cues) for kind, cues in mods.schedule.SESSIONS.items()}
    assert lengths == {"B": 355, "E_M": 1215, "E_P": 925}  # 5:55, 20:15, 15:25
    assert all(name.isalnum() and len(text) <= 160 for cues in mods.schedule.SESSIONS.values() for name, _, text in cues)
    assert len(mods.schedule.B_WINDOWS) == MAX_PROTOTYPES  # one prototype per window, never more than a gallery holds
    assert len(set(mods.schedule.SCHEDULE_DIGESTS.values())) == 3


def test_the_raw_capture_keeps_every_compared_face_whatever_the_vote_sealed_and_never_in_plain_text(pipeline, mods,
                                                                                                    sealer) -> None:
    assert pipeline.raw_report["written"] and pipeline.raw_report["refused"] is None
    # what B's records say was compared, match or unknown alike (the turns are unknown to the original)
    votes = [p["vote"] for line in (pipeline.b / "run.jsonl").read_text().splitlines()
             if (r := json.loads(line)).get("identity") == "result" and r["outcome"] == "applied"
             for p in r["persons"] if p["label"] in ("match", "unknown")]
    assert {"match", "unknown"} <= set(votes)
    assert pipeline.raw_report["faces_kept"] == len(votes)
    assert pipeline.report["raw_deleted"] and not pipeline.raw.exists()


def test_the_build_freezes_one_deployable_candidate_of_at_most_eight_prototypes_from_build_windows_only(pipeline,
                                                                                                         sealer) -> None:
    report = pipeline.report
    assert report["outcome"] == "BUILT" and report["refused"] is None and report["frozen"]
    assert report["prototypes"] == 8 <= MAX_PROTOTYPES
    assert report["original_unchanged"] is True
    document = IdentityStore(pipeline.candidate, sealer).load(SECRET)
    assert len(document.identities) == 1 and document.identities[0].identity_id == ENROLLED
    assert len(document.identities[0].prototypes) == 8
    assert document.compatibility == COMPAT
    frozen = pipeline.frozen
    assert [w for w, _ in frozen["windows"]] == ["bs:lens", "bs:down", "b4:lens", "b4:left", "b4:right", "b5:lens",
                                                 "b5:left", "b5:right"]
    assert set(frozen) == {"format", "digest", "prototypes_digest", "prototypes", "windows", "faces_per_window",
                           "faces_in_windows", "windows_excluded", "built_utc"}  # no vector anywhere in the record
    assert all(isinstance(name, str) and isinstance(t, int) for name, t in frozen["windows"])  # names, frame times


def test_verify_frozen_matches_only_the_exact_candidate(pipeline, mods, sealer, tmp_path) -> None:
    root = pipeline.base / "experiment"
    assert mods.build.verify_frozen(pipeline.candidate, root, pipeline.base / "identity", secret=SECRET,
                                    sealer=sealer)["frozen_candidate"] == "matches"
    other = IdentityStore(tmp_path / "other", sealer)
    other.ensure_directory()
    document = IdentityStore(pipeline.candidate, sealer).load(SECRET)
    changed = list(document.identities[0].prototypes)
    changed[3] = unit(tuple(x + (1e-6 if i == 7 else 0.0) for i, x in enumerate(changed[3])))
    other.save(document.model_copy(update={"identities": (document.identities[0].model_copy(
        update={"prototypes": tuple(changed)}),)}), SECRET)
    assert mods.build.verify_frozen(tmp_path / "other", root, pipeline.base / "identity", secret=SECRET,
                                    sealer=sealer)["frozen_candidate"] == "differs"


def test_the_comparison_reports_gains_and_losses_on_the_same_faces_from_full_precision_decisions(pipeline,
                                                                                                  mods) -> None:
    out = mods.compare.compare(pipeline.em, POLICY, pipeline.frozen)
    assert out["session"] == "E_M" and out["inconclusive"] == []
    conditions = {(c["spot"], c["kind"]): c for c in out["conditions"]}
    seat = conditions[("s", "facing")]
    assert seat["loss"] == seat["compared"] > 0 and seat["gain"] == 0  # the photo-like seat face: the candidate loses
    assert seat["known_share"]["candidate"] < seat["known_share"]["baseline"] and seat["lost_acquisitions"] == 3
    natural = conditions[("4", "natural")]
    assert natural["gain"] > 0 and natural["loss"] == 0 and natural["both"] > 0
    assert natural["known_share"]["candidate"] > natural["known_share"]["baseline"]
    facing = conditions[("4", "facing")]
    assert facing["both"] == facing["compared"] and facing["loss"] == facing["gain"] == 0
    # every compared face is counted once, in one class
    assert all(h["both"] + h["loss"] + h["gain"] + h["neither"] == h["compared"] for h in out["holds"])
    assert out["pending"] == "the gain bar and the facing-loss limit"


def test_the_chronological_replay_reproduces_the_recorded_production_history(pipeline, mods) -> None:
    for session in (pipeline.em, pipeline.ep):
        out = mods.compare.compare(session, POLICY, pipeline.frozen)
        replay = out["replay"]
        assert replay["recorded"] > 0 and replay["matched"] == replay["recorded"]
        assert replay["unmatched"] == replay["extra"] == 0


def test_the_replay_check_fails_when_the_rules_differ_from_the_runtimes(pipeline, mods) -> None:
    for policy in (IdentityPolicy(match_threshold=0.7, confirmations=3, vote_ttl_ns=POLICY.vote_ttl_ns,
                                  fresh_window_ns=POLICY.fresh_window_ns),
                   IdentityPolicy(match_threshold=0.7, vote_ttl_ns=POLICY.vote_ttl_ns,
                                  fresh_window_ns=1_000_000_000)):  # a fresh window of 1 s, not the runtime's 3 s
        out = mods.compare.compare(pipeline.em, policy, pipeline.frozen)
        assert "replay_does_not_reproduce_the_recorded_history" in out["inconclusive"]


def test_the_hold_reset_analysis_is_printed_apart_and_never_as_runtime_behaviour(pipeline, mods, capsys) -> None:
    out = mods.compare.compare(pipeline.em, POLICY, pipeline.frozen)
    mods.compare.print_report(out)
    text = capsys.readouterr().out
    chronological = text.index("CHRONOLOGICAL")
    analysis = text.index("ANALYSIS hold reset")
    assert chronological < analysis and "NOT runtime behaviour, never a criterion" in text
    assert "PENDING the gain bar and the facing-loss limit" in text and "SUPPORTED or NOT_SUPPORTED verdict" in text
    for h in out["holds"]:
        assert set(h["hold_reset_share"]) == {"baseline", "candidate"}  # its own field, never the KNOWN share's
    assert ENROLLED not in text and "similarity" not in text.lower()


def test_the_participants_session_reports_candidate_matches_as_rejected_and_its_exposure(pipeline, mods) -> None:
    out = mods.compare.compare(pipeline.ep, POLICY, pipeline.frozen)
    assert out["session"] == "E_P" and out["inconclusive"] == []
    p = out["participant"]
    assert p["baseline_match_votes"] == 0 and p["baseline_known_entries"] == 0
    assert p["candidate_match_votes"] > 0 and out["rejected"]
    assert set(out["exposure"]) == {w for w, _ in pipeline.frozen["windows"]}
    assert all(count and count > 0 for count in out["exposure"].values()) and out["not_exercised"] == []


def test_the_reader_refuses_a_session_that_is_not_an_evaluation_schedule(pipeline, mods) -> None:
    assert mods.compare.compare(pipeline.b, POLICY, pipeline.frozen)["inconclusive"] == ["not_an_evaluation_session"]


# ---------------------------------------------------------------- the builder's refusals


def _unlock(session: Path) -> None:
    """Test-only: take a session's account and its lock and reading lines out, to enter another one."""
    account = session / "mpe-account.txt"
    if account.exists():
        account.chmod(0o600)
        account.unlink()
    provenance = session / "provenance.txt"
    keep = [line for line in provenance.read_text().splitlines()
            if not line.startswith(("account_sha256=", "account_saved_utc=", "reading_started_utc="))]
    provenance.write_text("".join(f"{line}\n" for line in keep))


def reenter(session: Path, kind: str, *, others_in_view: str = "no", not_as_cued: str = "none") -> None:
    _unlock(session)
    MPE_ACCOUNT().enter(session, kind, others_in_view=others_in_view, not_as_cued=not_as_cued)


def force_account(session: Path, text: str) -> None:
    """Test-only: an account the entry step would refuse, written and locked as if it had been saved."""
    _unlock(session)
    (session / "mpe-account.txt").write_text(text)
    with open(session / "provenance.txt", "a") as provenance:
        provenance.write(f"account_sha256={hashlib.sha256(text.encode()).hexdigest()}\n"
                         "account_saved_utc=2026-11-07T00:00:00+00:00\n")


def MPE_MANIFEST():
    import sys

    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module("mpe_manifest")
    finally:
        sys.path.remove(str(RUNNERS))


def MPE_ACCOUNT():
    import sys

    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module("mpe_account")
    finally:
        sys.path.remove(str(RUNNERS))


def copy_session(source: Path, target: Path) -> Path:
    target.mkdir()
    for name in ("result.json", "run.jsonl", "provenance.txt", "mpe-account.txt"):
        (target / name).write_bytes((source / name).read_bytes())
    return target


def raw_from(pipeline, sealer, directory: Path, *, extra=None, boot: str = BOOT) -> Path:
    """A raw capture of B's own compared faces (rebuilt from the records' frame numbers with the build faces), with an
    optional face that B's records never offered."""
    manifest = MPE_MANIFEST().Manifest(directory.parent)
    manifest.reserve(directory, "raw", "raw", "test: the B launch block")
    capture = RawCapture(IdentityStore(directory, sealer), SECRET, GALLERY, CONSENT)
    for line in (pipeline.b / "run.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("identity") == "result" and r["outcome"] == "applied" and len(r["persons"]) == 1 \
                and r["persons"][0]["label"] in ("match", "unknown"):
            capture.offer(r["frame_mono_ns"], LENS, frame_seq=r["frame_seq"], stream_epoch=r["stream_epoch"],
                          track_id=r["persons"][0]["track_id"], boot_id=boot)
    if extra is not None:
        capture.offer(extra, LENS, frame_seq=10**6, stream_epoch=1, track_id=1, boot_id=boot)
    assert capture.finish()["written"]
    MPE_MANIFEST().claim(manifest, directory, writer_exited=True)
    return directory


@pytest.mark.parametrize("case, label", [
    ("schedule", "not_a_build_session"), ("account_missing", "account_missing"),
    ("others", "others_in_view"), ("account_bad", "account_invalid"), ("boot", "raw_capture_other_boot"),
    ("foreign_face", "raw_capture_not_this_session"), ("target", "candidate_target_exists"),
    ("original_dir", "candidate_outside_root"),
])
def test_the_builder_refuses_before_writing_anything(pipeline, mods, sealer, tmp_path, case, label) -> None:
    session = copy_session(pipeline.b, tmp_path / "b")
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    raw = raw_from(pipeline, sealer, root / "raw", extra=123 if case == "foreign_face" else None,
                   boot="another-boot" if case == "boot" else BOOT)
    candidate = root / "candidate"
    if case == "schedule":
        result = json.loads((session / "result.json").read_text())
        result["cues"][3]["at_s"] += 1.0
        (session / "result.json").write_text(json.dumps(result))
    elif case == "account_missing":
        (session / "mpe-account.txt").unlink()
    elif case == "others":
        reenter(session, "B", others_in_view="unknown")
    elif case == "account_bad":
        force_account(session, "format=mpe-account-1\nsession=B\nothers_in_view=no\nnot_as_cued=b9:sideways\n")
    elif case == "target":
        candidate.mkdir()
    elif case == "original_dir":
        candidate = pipeline.base / "identity"
    before = sorted(p.name for p in root.iterdir())
    report = mods.build.build(session, raw, pipeline.base / "identity", candidate, root, secret=SECRET, sealer=sealer,
                              match_threshold=POLICY.match_threshold)
    assert report["outcome"] == "NOT_BUILT" and report["refused"] == label
    assert not (root / mods.build.FROZEN_NAME).exists() and report["original_unchanged"] is True
    assert not (candidate / "gallery.sealed").exists() or case == "original_dir"
    after = sorted(p.name for p in root.iterdir())
    assert set(after) - set(before) <= {"MANIFEST.tsv"}  # nothing but the manifest's listing of the raw capture
    assert raw.exists()  # kept (listed) for the cleanup step, never deleted by a refused build


def test_a_window_not_done_as_cued_gives_no_prototype_and_too_few_or_inconsistent_faces_build_nothing(
        pipeline, mods, sealer, tmp_path) -> None:
    session = copy_session(pipeline.b, tmp_path / "b")
    reenter(session, "B", not_as_cued="b5:left,b5:right")
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    raw = raw_from(pipeline, sealer, root / "raw")
    report = mods.build.build(session, raw, pipeline.base / "identity", root / "candidate", root, secret=SECRET,
                              sealer=sealer, match_threshold=POLICY.match_threshold)
    assert report["outcome"] == "BUILT" and report["prototypes"] == 6
    assert report["windows_excluded"] == ["b5:left", "b5:right"]
    assert set(report["faces_per_window"]) == {"bs:lens", "bs:down", "b4:lens", "b4:left", "b4:right", "b5:lens"}
    # all eight windows excluded but three: under the minimum of four prototypes
    session2 = copy_session(pipeline.b, tmp_path / "b2")
    reenter(session2, "B", not_as_cued="bs:lens,bs:down,b4:lens,b4:left,b5:lens")
    root2 = tmp_path / "exp2"
    root2.mkdir(mode=0o700)
    report = mods.build.build(session2, raw_from(pipeline, sealer, root2 / "raw"), pipeline.base / "identity",
                              root2 / "candidate", root2, secret=SECRET, sealer=sealer,
                              match_threshold=POLICY.match_threshold)
    assert report["refused"] == "too_few_prototypes" and not (root2 / "candidate").exists()
    # the real B capture against a threshold no turn reaches: the consistency rule refuses it, unchanged
    root3 = tmp_path / "exp3"
    root3.mkdir(mode=0o700)
    report = mods.build.build(copy_session(pipeline.b, tmp_path / "b3"), raw_from(pipeline, sealer, root3 / "raw"),
                              pipeline.base / "identity", root3 / "candidate", root3, secret=SECRET, sealer=sealer,
                              match_threshold=1.01)
    assert report["refused"] == "inconsistent_capture" and not (root3 / "candidate").exists()


def test_the_selection_never_depends_on_what_the_original_decided(pipeline, mods, sealer, tmp_path,
                                                                    monkeypatch) -> None:
    """The same B faces with an original that matches none of them: the same frozen digest (time alone chooses)."""
    other = GALLERY.model_copy(update={"identities": (GALLERY.identities[0].model_copy(
        update={"prototypes": (vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),)}),)})
    monkeypatch.setattr(mods.sim, "GALLERY", other)  # the simulated runtime votes with this original instead
    store = IdentityStore(tmp_path / "identity2", sealer)
    store.ensure_directory()
    store.save(other, SECRET)
    raw = tmp_path / "exp" / "raw"
    (tmp_path / "exp").mkdir(mode=0o700)
    mods.manifest.Manifest(tmp_path / "exp").reserve(raw, "raw", "raw", "test: the B launch block")
    capture = RawCapture(IdentityStore(raw, sealer), SECRET, other, CONSENT)
    (tmp_path / "b").mkdir()
    b = run_session(mods, tmp_path / "b", "B", build_face, capture=capture)
    mods.manifest.claim(mods.manifest.Manifest(tmp_path / "exp"), raw, writer_exited=True)
    votes = {p["vote"] for line in (b / "run.jsonl").read_text().splitlines()
             if (r := json.loads(line)).get("identity") == "result" for p in r["persons"]}
    assert "match" not in votes  # the original decided "unknown" for every face this time
    report = mods.build.build(b, raw, tmp_path / "identity2", tmp_path / "exp" / "candidate", tmp_path / "exp",
                              secret=SECRET, sealer=sealer, match_threshold=POLICY.match_threshold)
    assert report["outcome"] == "BUILT" and report["digest"] == pipeline.report["digest"]


# ---------------------------------------------------------------- the manifest cleanup


def test_the_cleanup_deletes_only_listed_experiment_files_and_reports_the_rest(pipeline, mods, sealer,
                                                                              tmp_path) -> None:
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    manifest = mods.manifest.Manifest(root)
    candidate = root / "candidate"
    store = IdentityStore(candidate, sealer)
    store.ensure_directory()
    store.save(GALLERY, SECRET)
    manifest.add_tree(candidate, "candidate", "test")
    evidence = tmp_path / "sessions" / "e_m"
    evidence.mkdir(parents=True)
    (evidence / "run.jsonl").write_text("{}\n")
    manifest.add(evidence, "evidence", "test")
    manifest.add(evidence / "run.jsonl", "evidence", "test")
    # unrelated and unexpected things: a crash report, a backup copy of the sealed gallery, an unlisted file inside
    crash = tmp_path / "crash"
    crash.mkdir()
    (crash / "_usr_bin_python3.1000.crash").write_text("synthetic")
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "gallery.sealed").write_bytes((candidate / "gallery.sealed").read_bytes())
    (candidate / "stray.txt").write_text("not listed")
    report = mods.manifest.copy_report(manifest, [tmp_path / "backup", tmp_path / "sessions"], crash)
    assert report["copies_with_a_listed_digest"] == [str(backup / "gallery.sealed")]
    assert report["unlisted_inside_listed_dirs"] == [str(candidate / "stray.txt")]
    assert report["crash_reports_since_first_entry"] == ["_usr_bin_python3.1000.crash"]
    with pytest.raises(mods.manifest.ManifestError):
        mods.manifest.cleanup(manifest, ("evidence",))
    result = mods.manifest.cleanup(manifest, ("candidate",))
    assert not (candidate / "gallery.sealed").exists()  # listed: deleted
    assert (candidate / "stray.txt").exists() and str(candidate) in result["left_not_empty"]  # unlisted: left
    assert result["unlisted_inside"] == [str(candidate / "stray.txt")]
    assert (evidence / "run.jsonl").exists() and (backup / "gallery.sealed").exists()  # never touched
    assert (crash / "_usr_bin_python3.1000.crash").exists()


def test_the_cleanup_leaves_a_listed_path_that_is_no_longer_the_file_it_listed(mods, tmp_path) -> None:
    root = tmp_path / "exp"
    manifest = mods.manifest.Manifest(root)
    root.mkdir(mode=0o700)
    target = root / "raw"
    target.mkdir()
    (target / RAW_FILE).write_bytes(b"synthetic")
    manifest.add_tree(target, "raw", "test")
    (target / RAW_FILE).unlink()  # replaced by another file of the same name: a different inode
    (target / RAW_FILE).write_bytes(b"something else")
    result = mods.manifest.cleanup(manifest, ("raw",))
    assert result["left_changed"] == [str(target / RAW_FILE)] and (target / RAW_FILE).exists()
    link = root / "link"
    link.symlink_to(target)
    with pytest.raises(mods.manifest.ManifestError):
        manifest.add(link, "raw", "test")
    with pytest.raises(mods.manifest.ManifestError):
        manifest.add(Path("relative"), "raw", "test")


# ---------------------------------------------------------------- the raw capture's own rules


def test_a_raw_capture_refuses_an_existing_target_mixes_no_boots_and_reads_back_only_its_own_format(tmp_path,
                                                                                                    sealer) -> None:
    (tmp_path / "exists").mkdir()
    with pytest.raises(ExperimentRefused, match="face_capture_raw_target_exists"):
        RawCapture(IdentityStore(tmp_path / "exists", sealer), SECRET, GALLERY, CONSENT)
    capture = RawCapture(IdentityStore(tmp_path / "raw", sealer), SECRET, GALLERY, CONSENT)
    capture.offer(1, LENS, frame_seq=1, stream_epoch=1, track_id=1, boot_id="a")
    capture.offer(2, LENS, frame_seq=2, stream_epoch=1, track_id=1, boot_id="b")
    report = capture.finish()
    assert report["written"] and report["faces_kept"] == 1 and report["faces_dropped_other_boot"] == 1
    raw = (tmp_path / "raw" / RAW_FILE).read_bytes()
    assert b"0.75" not in raw and repr(LENS[0]).encode() not in raw  # sealed: no value in plain text
    document = open_raw_capture(tmp_path / "raw", sealer, SECRET)
    assert [f["e"] for f in document["faces"]] == [list(LENS)] and document["boot_id"] == "a"
    (tmp_path / "raw" / "other").write_text("x")
    with pytest.raises(ExperimentRefused, match="raw_capture_unexpected_files"):
        open_raw_capture(tmp_path / "raw", sealer, SECRET)
    empty = RawCapture(IdentityStore(tmp_path / "none", sealer), SECRET, GALLERY, CONSENT)
    assert empty.finish()["refused"] == "no_faces" and not (tmp_path / "none").exists()


# ---------------------------------------------------------------- the review's regressions


def test_the_selection_rule_is_time_alone_nearest_the_middle_the_earlier_on_a_tie(mods) -> None:
    printed = {name: at * 1_000_000_000 for name, at, _ in mods.schedule.B_CUES}
    window = mods.schedule.B_WINDOWS[0]  # bs:lens: from its bell + 3 s to the next bell
    start = printed[window.bell] + 3_000_000_000
    end = printed[window.until]
    middle = (start + end) // 2
    faces = [{"t": start - 1, "e": LENS}, {"t": middle + 500_000_000, "e": LEFT}, {"t": middle - 500_000_000, "e": RIGHT},
             {"t": end, "e": DOWN}, {"t": start, "e": LENS}]
    chosen, counts = mods.build.select_windows(faces, printed, set())
    assert counts["bs:lens"] == 3  # start included, before the settle and at the next bell excluded
    assert dict(chosen)["bs:lens"] == 2  # equidistant from the middle: the earlier one
    assert counts["bs:down"] == 0  # the face at the next bell lies inside that window's settle: in neither
    chosen, counts = mods.build.select_windows(faces, printed, {"bs:lens"})
    assert "bs:lens" not in counts and "bs:lens" not in dict(chosen)


def test_a_wrong_raw_path_is_refused_before_anything_is_listed_and_a_build_deletes_only_its_own_raw(pipeline, mods,
                                                                                                   sealer,
                                                                                                   tmp_path) -> None:
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    session = copy_session(pipeline.b, tmp_path / "b")
    identity = pipeline.base / "identity"
    original = (identity / "gallery.sealed").read_bytes()

    def build(raw, candidate=root / "candidate", session=session):
        return mods.build.build(session, raw, identity, candidate, root, secret=SECRET, sealer=sealer,
                                match_threshold=POLICY.match_threshold)

    assert build(identity)["refused"] == "raw_capture_outside_root"  # the original's directory named as --raw
    assert build(pipeline.base)["refused"] == "raw_capture_outside_root"
    decoy = root / "decoy"  # inside the root but not a raw capture: a copy of a gallery's directory
    decoy.mkdir(mode=0o700)
    (decoy / "gallery.sealed").write_bytes(original)
    assert build(decoy)["refused"] == "raw_capture_not_claimed"  # never reserved and claimed: never read or listed
    assert build(Path("relative/raw"))["refused"] == "path_not_canonical"
    assert build(root / "raw", candidate=root / "raw" / "candidate")["refused"] == "paths_overlap"
    assert not (root / "MANIFEST.tsv").exists() or not mods.manifest.Manifest(root).entries()  # nothing listed
    old = raw_from(pipeline, sealer, root / "old", boot="another-boot")  # refused after its own checks: listed
    assert build(old)["refused"] == "raw_capture_other_boot"
    raw = raw_from(pipeline, sealer, root / "raw")
    report = build(raw)
    assert report["outcome"] == "BUILT" and report["raw_deleted"] and not raw.exists()
    assert (old / RAW_FILE).exists() and (decoy / "gallery.sealed").exists()  # another build's, and never listed
    assert (identity / "gallery.sealed").read_bytes() == original and report["original_unchanged"]


def test_a_frozen_record_already_there_refuses_the_build_before_the_candidate_is_written(pipeline, mods, sealer,
                                                                                        tmp_path) -> None:
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    (root / mods.build.FROZEN_NAME).write_text("{}")
    report = mods.build.build(copy_session(pipeline.b, tmp_path / "b"), raw_from(pipeline, sealer, root / "raw"),
                              pipeline.base / "identity", root / "candidate", root, secret=SECRET, sealer=sealer,
                              match_threshold=POLICY.match_threshold)
    assert report["refused"] == "frozen_record_exists" and not (root / "candidate").exists()


def test_the_replay_stops_at_the_runs_end_and_at_the_mark_sees_a_known_from_the_approach(pipeline, mods,
                                                                                        tmp_path) -> None:
    """In view at the stop (back at the Mac, facing it) and a face on the approach of hold m4f1: the replay still
    reproduces the history (it never predicts past the run's last step) and V3's rule reads that hold not_at_mark."""
    shadow = open_shadow(IdentityStore(pipeline.candidate, Sealer(kdf_n=2**14)), SECRET, GALLERY)
    (tmp_path / "em").mkdir()
    em = run_session(mods, tmp_path / "em", "E_M", em_face, shadow=shadow, tail=True, approach="m4f1")
    out = mods.compare.compare(em, POLICY, pipeline.frozen)
    assert out["inconclusive"] == [] and out["replay"]["extra"] == out["replay"]["unmatched"] == 0
    hold = next(h for h in out["holds"] if h["hold"] == "m4f1")
    assert hold["at_mark"]["baseline"] == "not_at_mark" and hold["at_mark"]["candidate"] == "not_at_mark"
    tail = [r for r in mods.compare.read_session(em)["transitions"] if r["to"] == "known"]
    assert tail  # the person was KNOWN at the end, votes current past it


def test_the_replay_check_fails_on_a_missing_or_moved_recorded_transition(pipeline, mods) -> None:
    data = mods.compare.read_session(pipeline.em)
    run = mods.compare.replay(data, POLICY, "baseline")
    others = [i for i, r in enumerate(data["transitions"]) if r["to"] != "cleared"]
    missing = [r for i, r in enumerate(data["transitions"]) if i != others[3]]
    assert mods.compare.check_replay(run, missing, clears=True)["extra"] == 1
    moved = [dict(r, mono_ns=r["mono_ns"] + 1_000_000_000) if i == others[5] else r
             for i, r in enumerate(data["transitions"])]
    result = mods.compare.check_replay(run, moved, clears=True)
    assert result["unmatched"] == 1 and result["extra"] >= 1


def test_the_reading_is_inconclusive_without_the_frozen_record_its_shadow_digest_or_the_account(pipeline, mods,
                                                                                               tmp_path) -> None:
    assert "frozen_record_missing" in mods.compare.compare(pipeline.em, POLICY, None)["inconclusive"]
    other = dict(pipeline.frozen, prototypes_digest="0" * 64)
    assert "shadow_gallery_not_the_frozen_candidate" in mods.compare.compare(pipeline.em, POLICY, other)["inconclusive"]
    session = copy_session(pipeline.em, tmp_path / "em")
    (session / "mpe-account.txt").unlink()
    assert "account_missing" in mods.compare.compare(session, POLICY, pipeline.frozen)["inconclusive"]
    reenter(session, "E_M", not_as_cued="m4n2")
    out = mods.compare.compare(session, POLICY, pipeline.frozen)
    assert out["inconclusive"] == []
    hold = next(h for h in out["holds"] if h["hold"] == "m4n2")
    assert hold["not_as_cued"] and hold["excluded"]
    natural = next(c for c in out["conditions"] if (c["spot"], c["kind"]) == ("4", "natural"))
    assert natural["not_as_cued"] == 1 and natural["excluded"] == 1 and "gained_acquisitions" in natural
    reenter(session, "E_M", others_in_view="yes")
    assert "account_others_in_view_not_no" in mods.compare.compare(session, POLICY, pipeline.frozen)["inconclusive"]


def test_the_manifest_never_lists_itself_or_its_root_and_a_cleanup_can_be_limited_to_one_path(mods, tmp_path) -> None:
    root = tmp_path / "exp"
    manifest = mods.manifest.Manifest(root)
    root.mkdir(mode=0o700)
    (root / "a").mkdir()
    (root / "a" / "f").write_bytes(b"a")
    (root / "b").mkdir()
    (root / "b" / "f").write_bytes(b"b")
    manifest.add_tree(root / "a", "raw", "test")
    manifest.add_tree(root / "b", "raw", "test")
    for path in (root, manifest.path):
        with pytest.raises(mods.manifest.ManifestError):
            manifest.add(path, "raw", "test")
    result = mods.manifest.cleanup(manifest, ("raw",), within=root / "a")
    assert not (root / "a").exists() and (root / "b" / "f").exists() and sorted(result["deleted"]) == sorted(
        [str(root / "a" / "f"), str(root / "a")])
    assert all(e.sha256 for e in manifest.entries().values() if e.kind == "file")  # listed with its digest at once


# ---------------------------------------------------------------- per-session working copies of the original


def original_enrollment(sealer, directory: Path) -> dict[str, str]:
    store = IdentityStore(directory, sealer)
    store.ensure_directory()
    store.save(GALLERY, SECRET)
    store.audit("enroll", ENROLLED, photos=3)
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("gallery.sealed", "audit.jsonl")}


def test_a_working_copy_is_listed_and_its_cleanup_never_reaches_the_original_or_sealed_evidence(mods, sealer,
                                                                                                tmp_path) -> None:
    original = tmp_path / "identity"
    before = original_enrollment(sealer, original)
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    (root / "work").mkdir(mode=0o700)
    manifest = mods.manifest.Manifest(root)
    copy = root / "work" / "e_m-identity"
    made = mods.manifest.make_working_copy(manifest, original, copy, "test")
    assert [(e.kind, e.cls) for e in made] == [("dir", "working_copy"), ("file", "working_copy"),
                                                ("file", "working_copy")]
    assert IdentityStore(copy, sealer).load(SECRET) == GALLERY  # the session runs on the copy, the same enrollment
    # a sealed session inside the root, listed (wrongly) as a working copy: under its seal, never deleted
    sealed = root / "sessions" / "e_m"
    sealed.mkdir(parents=True)
    (sealed / "run.jsonl").write_text("{}\n")
    (sealed / "SHA256SUMS").write_text("0  run.jsonl\n")
    manifest.add(sealed / "run.jsonl", "working_copy", "forged")
    # the original's own sealed file, forged into the manifest (add() refuses a deletable entry outside the root)
    with pytest.raises(mods.manifest.ManifestError):
        manifest.add(original / "gallery.sealed", "working_copy", "forged")
    info = os.lstat(original / "gallery.sealed")
    manifest._append("add", str(original / "gallery.sealed"), "file", "working_copy", "forged",
                     "2026-11-07T00:00:00+00:00", f"{info.st_dev}:{info.st_ino}", before["gallery.sealed"])
    result = mods.manifest.cleanup(manifest, ("working_copy",))  # no protected list at all: still never reached
    assert not copy.exists() and sorted(result["deleted"]) == sorted(
        [str(copy / "gallery.sealed"), str(copy / "audit.jsonl"), str(copy)])
    assert result["left_sealed"] == [str(sealed / "run.jsonl")] and (sealed / "run.jsonl").exists()
    assert result["left_outside_root"] == [str(original / "gallery.sealed")]
    after = {name: hashlib.sha256((original / name).read_bytes()).hexdigest() for name in before}
    assert after == before  # the original enrollment, byte for byte


def test_a_protected_path_is_never_deleted_even_when_listed_inside_the_root(mods, sealer, tmp_path) -> None:
    original = tmp_path / "identity"
    original_enrollment(sealer, original)
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    manifest = mods.manifest.Manifest(root)
    copy = root / "e_p-identity"
    mods.manifest.make_working_copy(manifest, original, copy, "test")
    result = mods.manifest.cleanup(manifest, ("working_copy",), protected=(copy,))
    assert sorted(result["left_protected"]) == sorted([str(copy), str(copy / "gallery.sealed"),
                                                       str(copy / "audit.jsonl")])
    assert (copy / "gallery.sealed").exists() and result["deleted"] == []
    result = mods.manifest.cleanup(manifest, ("working_copy",), protected=(copy / "audit.jsonl",))
    assert (copy / "audit.jsonl").exists() and str(copy) in result["left_protected"]  # nor what holds a protected path


@pytest.mark.parametrize("case", ["same", "inside_original", "outside_root", "original_inside_root", "exists",
                                  "relative", "no_gallery"])
def test_a_working_copy_is_refused_unless_it_is_new_inside_the_root_and_apart_from_the_original(mods, sealer, tmp_path,
                                                                                               case) -> None:
    original = tmp_path / "identity"
    before = original_enrollment(sealer, original)
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    manifest = mods.manifest.Manifest(root)
    copy = root / "copy"
    if case == "same":
        copy = original
    elif case == "inside_original":
        copy = original / "copy"
    elif case == "outside_root":
        copy = tmp_path / "elsewhere"
    elif case == "original_inside_root":
        original = root / "identity"
        original_enrollment(sealer, original)
    elif case == "exists":
        copy.mkdir()
    elif case == "relative":
        copy = Path("exp/copy")
    elif case == "no_gallery":
        original = tmp_path / "empty"
        original.mkdir(mode=0o700)
    with pytest.raises(mods.manifest.ManifestError):
        mods.manifest.make_working_copy(manifest, original, copy, "test")
    assert not manifest.entries()
    if case not in ("original_inside_root", "no_gallery"):
        assert {n: hashlib.sha256((tmp_path / "identity" / n).read_bytes()).hexdigest() for n in before} == before


# ---------------------------------------------------------------- track lives: recorded, replayed, and harmless


TWO = GalleryDocument(compatibility=COMPAT, identities=(
    GalleryIdentity(identity_id=ENROLLED, enrolled_utc=GALLERY.identities[0].enrolled_utc, consent_date="2026-10-08",
                    consent_scope=CONSENT_SCOPE, prototypes=(vec(1.0),)),
    GalleryIdentity(identity_id="idn-00000000000b", enrolled_utc=GALLERY.identities[0].enrolled_utc,
                    consent_date="2026-10-08", consent_scope=CONSENT_SCOPE, prototypes=(vec(0.0, 1.0),))))
BETWEEN = vec(1.0, 1.0)  # as close to both enrolled identities: the original's margin rule, so no vote at all
NEAR_A = vec(1.0, 0.1)  # the first identity's, for the original; the candidate (its prototype BETWEEN) matches it too


def gap_run(mods, root: Path, shadow, *, lives: bool, script=None, face_at=None) -> Path:
    """Track 7 in view, a one-face first life, gone 1.5 s (longer than the 1.0 s track expiry: its life ends), back
    under the same number with faces every tick."""
    script = script or ((60, 66, "facing", 7), (66, 67.5, "absent", None), (67.5, 85, "facing", 7))

    def analyze_for(t0):
        def analyze(frame, image):
            t = (frame.ingest_mono_ns - t0) / 1e9
            face = face_at(t) if face_at else (BETWEEN if 60.5 <= t < 61.5 else NEAR_A if t >= 67.5 else None)
            if face is None:
                return FaceRun((), 0, 1)
            return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=face),), 1, 0)
        return analyze

    root.mkdir()
    startup = {"run": "starting", "startup": {"face": {"validation_run": True, "identities_enrolled": 2,
                                                      "detection": "head_windows", "shadow": True,
                                                      **({"lives": True} if lives else {})}}}
    out = mods.sim.simulate(root, startup, script=script, analyze=analyze_for, duration=90.0, route_b=True,
                            scene=False, shadow=shadow, lives=lives)
    (out / "result.json").write_text(json.dumps({"cues": []}))
    return out


@pytest.fixture
def two_identities(mods, sealer, tmp_path, monkeypatch):
    monkeypatch.setattr(mods.sim, "GALLERY", TWO)  # the simulated runtime's original: two enrolled identities
    store = IdentityStore(tmp_path / "candidate", sealer)
    store.ensure_directory()
    store.save(GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
        identity_id=ENROLLED, enrolled_utc=GALLERY.identities[0].enrolled_utc, consent_date="2026-11-07",
        consent_scope=CONSENT_SCOPE, prototypes=(BETWEEN,)),)), SECRET)
    return open_shadow(store, SECRET, TWO)


def records(run: Path, kind: str | None = None) -> list[dict]:
    lines = [json.loads(line) for line in (run / "run.jsonl").read_text().splitlines()]
    return [r for r in lines if kind is None or r.get("identity") == kind]


def test_life_records_change_nothing_else_the_runtime_writes_or_decides(mods, two_identities, tmp_path) -> None:
    on = gap_run(mods, tmp_path / "on", two_identities, lives=True)
    off = gap_run(mods, tmp_path / "off", two_identities, lives=False)
    def timeless(run_records):  # processing_ms is the face stage's wall-clock time: it differs between any two runs
        return [{k: v for k, v in r.items() if k != "processing_ms"} for r in run_records]

    without = timeless(r for r in records(on) if r.get("identity") != "life" and r.get("run") != "starting")
    plain = timeless(r for r in records(off) if r.get("run") != "starting")
    assert len(without) == len(plain) > 100
    assert without == plain  # every result, vote, shadow decision, tick, transition, status line and the stop: equal
    assert records(on, "life") and not records(off, "life")


def test_a_returning_track_number_is_recorded_as_a_life_ended_and_a_new_life_begun(mods, two_identities,
                                                                                  tmp_path) -> None:
    run = gap_run(mods, tmp_path / "on", two_identities, lives=True)
    lives = [(r["event"], r["track_id"], r["since_mono_ns"]) for r in records(run, "life")]
    sevens = [life for life in lives if life[1] == 7]
    assert [e for e, _, _ in sevens][:3] == ["began", "ended", "began"]
    first, ended, again = sevens[:3]
    assert first[2] == ended[2] < again[2]  # the same life ended; the number began a new one later
    assert all(r["schema"] == 8 for r in records(run, "life"))


def test_a_stale_vote_that_moves_only_the_candidate_is_caught_by_the_lives_never_by_baseline_agreement(
        mods, two_identities, tmp_path) -> None:
    """The first life's one face is a margin for the original (no vote) and a match for the candidate. The number
    returns: without the life boundaries the candidate keeps that stale vote and reads KNOWN one face early, while the
    baseline's replay still agrees with every recorded transition. Only the recorded lives make the candidate right."""
    run = gap_run(mods, tmp_path / "on", two_identities, lives=True)
    data = mods.compare.read_session(run)
    assert mods.compare.lives_problem(data) is None
    with_lives = {which: mods.compare.replay(data, POLICY, which, "lives") for which in ("baseline", "candidate")}
    legacy = {which: mods.compare.replay(data, POLICY, which, "legacy") for which in ("baseline", "candidate")}
    # the baseline agrees with the record in both modes (legacy: clears as inputs; lives: clears compared too)
    check = mods.compare.check_replay(legacy["baseline"], data["transitions"])
    assert check["unmatched"] == check["extra"] == 0 and check["recorded"] > 0
    check = mods.compare.check_replay(with_lives["baseline"], data["transitions"], clears=True)
    assert check["unmatched"] == check["extra"] == 0
    # the candidate: one face earlier without the boundaries (the stale vote from the ended life)
    entry = {mode: min(t for t, key, _ in runs["candidate"].entries if key.track_id == 7)
             for mode, runs in (("lives", with_lives), ("legacy", legacy))}
    assert entry["legacy"] < entry["lives"]
    assert entry["lives"] - entry["legacy"] >= 500_000_000  # a face tick apart, not a timing tolerance
    # the reading: without life records the candidate's history is never taken from baseline agreement
    stripped = copy_session_records(run, tmp_path / "stripped", drop_lives=True)
    data = mods.compare.read_session(stripped)
    assert mods.compare.lives_problem(data) == "life_records_missing"


def copy_session_records(run: Path, target: Path, *, drop_lives: bool) -> Path:
    target.mkdir()
    lines = []
    for record in records(run):
        if drop_lives and record.get("identity") == "life":
            continue
        if drop_lives and record.get("run") == "starting":
            record = json.loads(json.dumps(record))
            record["startup"]["face"].pop("lives", None)
        lines.append(json.dumps(record))
    (target / "run.jsonl").write_text("\n".join(lines) + "\n")
    for name in ("result.json", "provenance.txt", "mpe-account.txt"):
        if (run / name).exists():
            (target / name).write_bytes((run / name).read_bytes())
    return target


def test_without_life_records_the_reading_withholds_every_replay_number_though_the_baseline_agrees(pipeline, mods,
                                                                                                  tmp_path) -> None:
    stripped = copy_session_records(pipeline.em, tmp_path / "em", drop_lives=True)
    out = mods.compare.compare(stripped, POLICY, pipeline.frozen)
    assert "life_records_missing" in out["inconclusive"] and not out["replay_valid"]
    assert out["replay"]["unmatched"] == out["replay"]["extra"] == 0  # the baseline still agrees (legacy mode)
    assert all(h["known_share"] is None and h["at_mark"] is None for h in out["holds"])
    assert all(c["known_share"] is None and c["lost_acquisitions"] is None for c in out["conditions"])
    assert all(h["compared"] > 0 for h in out["holds"])  # the per-face pairs need no replay: still reported
    full = mods.compare.compare(pipeline.em, POLICY, pipeline.frozen)
    assert full["replay_valid"] and full["mode"] == "lives" and full["replay"]["clears_compared"]


@pytest.mark.parametrize("damage, label", [("ended_unbegun", "life_records_inconsistent"),
                                           ("tick_without_life", "life_records_incomplete")])
def test_life_records_that_do_not_hold_together_make_the_reading_inconclusive(pipeline, mods, tmp_path, damage,
                                                                              label) -> None:
    data = mods.compare.read_session(pipeline.em)
    if damage == "ended_unbegun":
        data["lives"] = [r for r in data["lives"] if not (r["event"] == "began" and r is data["lives"][0])]
    else:
        tick = next(t for t in data["ticks"] if t.get("tracks"))
        data["ticks"].append(dict(tick, tracks=[999]))
    assert mods.compare.lives_problem(data) == label


# ---------------------------------------------------------------- the account step: before any result, locked


def test_the_account_is_entered_once_read_only_and_locked_by_its_digest(mods, tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "provenance.txt").write_text(f"boot_id={BOOT}\n")
    asked = iter(["no", "m4n2, m5f1"])
    digest = mods.account.enter(run, "E_M", ask=lambda prompt: next(asked), say=lambda text: None)
    text = (run / "mpe-account.txt").read_bytes()
    assert hashlib.sha256(text).hexdigest() == digest and oct((run / "mpe-account.txt").stat().st_mode)[-3:] == "400"
    assert f"account_sha256={digest}" in (run / "provenance.txt").read_text()
    with pytest.raises(mods.account.AccountRefused, match="account_already_saved"):
        mods.account.enter(run, "E_M", others_in_view="no", not_as_cued="none")
    account = mods.account.require(run, "E_M")
    assert account == {"others_in_view": "no", "not_as_cued": {"m4n2", "m5f1"}}
    assert "reading_started_utc=" in (run / "provenance.txt").read_text()


@pytest.mark.parametrize("answers, label", [(("maybe", "none"), "account_invalid"),
                                            (("no", "m9n9"), "account_invalid"),
                                            (("no", "m4n2,m4n2"), "account_invalid")])
def test_the_account_step_refuses_answers_it_cannot_lock(mods, tmp_path, answers, label) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(mods.account.AccountRefused, match=label):
        mods.account.enter(run, "E_M", others_in_view=answers[0], not_as_cued=answers[1])
    assert not (run / "mpe-account.txt").exists()


def test_no_reading_shows_anything_before_the_locked_account_and_none_after_an_edit_or_a_late_account(
        pipeline, mods, sealer, tmp_path, capsys) -> None:
    session = copy_session(pipeline.em, tmp_path / "em")
    _unlock(session)  # no account yet
    out = mods.compare.compare(session, POLICY, pipeline.frozen)
    mods.compare.print_report(out)
    text = capsys.readouterr().out
    assert out["stopped"] == "account_not_locked" and text.startswith("STOP: enter the account first")
    assert "CHRONOLOGICAL" not in text and "compared" not in text and "KNOWN" not in text
    # a reading started before the account: the account step refuses, and every later reading refuses too
    provenance = session / "provenance.txt"
    provenance.write_text(provenance.read_text() + "reading_started_utc=2026-11-07T00:00:00+00:00\n")
    with pytest.raises(mods.account.AccountRefused, match="reading_already_started"):
        mods.account.enter(session, "E_M", others_in_view="no", not_as_cued="none")
    force_account(session, "format=mpe-account-1\nsession=E_M\nothers_in_view=no\nnot_as_cued=none\n")
    provenance.write_text("reading_started_utc=2026-11-07T00:00:00+00:00\n" + provenance.read_text())
    assert mods.compare.compare(session, POLICY, pipeline.frozen)["stopped"] == "account_saved_after_a_reading"
    # an account changed after it was locked
    reenter(session, "E_M")
    account = session / "mpe-account.txt"
    account.chmod(0o600)
    account.write_text(account.read_text().replace("not_as_cued=none", "not_as_cued=m4n1"))
    assert mods.compare.compare(session, POLICY, pipeline.frozen)["stopped"] == "account_changed"
    # the builder: refused before anything is read or listed
    b = copy_session(pipeline.b, tmp_path / "b")
    _unlock(b)
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    raw = raw_from(pipeline, sealer, root / "raw")  # registered at its creation, as the launch block does
    listed = set(mods.manifest.Manifest(root).entries())
    report = mods.build.build(b, raw, pipeline.base / "identity", root / "candidate", root, secret=SECRET,
                              sealer=sealer, match_threshold=POLICY.match_threshold)
    assert report["refused"] == "account_not_locked"
    assert set(mods.manifest.Manifest(root).entries()) == listed == {raw, raw / RAW_FILE}  # the builder added nothing


# ---------------------------------------------------------------- reservation, recovery and scoped cleanup


def recovery_setup(sealer, tmp_path):
    original = tmp_path / "identity"
    before = original_enrollment(sealer, original)
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    sealed = tmp_path / "sessions" / "mp-20261107T100000Z"
    sealed.mkdir(parents=True)
    (sealed / "run.jsonl").write_text("{}\n")
    (sealed / "SHA256SUMS").write_text("0  run.jsonl\n")
    return original, before, root, sealed


def unchanged(original: Path, before: dict) -> bool:
    return {n: hashlib.sha256((original / n).read_bytes()).hexdigest() for n in before} == before


def test_an_aborted_capture_is_claimed_only_after_the_runtime_exits_and_its_temporary_is_cleaned(mods, sealer,
                                                                                                tmp_path) -> None:
    original, before, root, sealed = recovery_setup(sealer, tmp_path)
    manifest = mods.manifest.Manifest(root)
    raw = root / "raw-mp-1"
    manifest.reserve(raw, "raw", "raw", "test")
    raw.mkdir(mode=0o700)  # the runtime was killed while sealing: its temporary is all there is
    (raw / ".raw-capture.sealed.4242.tmp").write_bytes(b"partial sealed bytes")
    (raw / "notes.txt").write_text("not the writer's")
    with pytest.raises(mods.manifest.ManifestError, match="writer has exited"):
        mods.manifest.claim(manifest, raw, writer_exited=False)
    assert not mods.manifest.runtime_exited({"child": {"returncode": None}}, processes=[])
    assert not mods.manifest.runtime_exited({"child": {"returncode": 0}},
                                            processes=[[b"/venv/bin/sentinel", b"run", b"face.yaml"]])
    assert not mods.manifest.runtime_exited({"child": {"returncode": 0}},  # the package's own launch
                                            processes=[[b"/env/bin/python", b"-m", b"sentinel.cli", b"run", b"f.yaml"]])
    assert mods.manifest.runtime_exited({"child": {"returncode": 0}}, processes=[[b"bash"]])
    result = mods.manifest.claim(manifest, raw, writer_exited=True)
    assert result["listed"] == [str(raw), str(raw / ".raw-capture.sealed.4242.tmp")]
    assert result["unexpected"] == [str(raw / "notes.txt")]
    cleaned = mods.manifest.cleanup(manifest, ("raw",), within=raw, protected=(original, sealed))
    assert not (raw / ".raw-capture.sealed.4242.tmp").exists() and (raw / "notes.txt").exists()
    assert cleaned["left_not_empty"] == [str(raw)]  # the unexpected file is reported, never deleted
    assert unchanged(original, before) and (sealed / "run.jsonl").exists()


def test_a_reservation_is_once_inside_the_root_and_closes_as_absent_only_when_nothing_was_recorded_written(
        mods, sealer, tmp_path) -> None:
    original, before, root, sealed = recovery_setup(sealer, tmp_path)
    manifest = mods.manifest.Manifest(root)
    for bad in (root / "exists", tmp_path / "outside", Path("relative")):
        if bad.name == "exists":
            bad.mkdir()
        with pytest.raises(mods.manifest.ManifestError):
            manifest.reserve(bad, "raw", "raw", "test")
    with pytest.raises(mods.manifest.ManifestError):
        manifest.reserve(root / "x", "evidence", "raw", "test")
    raw = root / "raw-mp-2"
    manifest.reserve(raw, "raw", "raw", "test")
    with pytest.raises(mods.manifest.ManifestError, match="reserved once"):
        manifest.reserve(raw, "raw", "raw", "test")
    with pytest.raises(mods.manifest.ManifestError, match="recorded a write"):
        mods.manifest.close_absent(manifest, raw, writer_exited=True, recorded_written=True)
    mods.manifest.close_absent(manifest, raw, writer_exited=True, recorded_written=False)
    assert manifest.reservations()[raw].closed == "absent"
    with pytest.raises(mods.manifest.ManifestError, match="reserved once"):
        manifest.reserve(raw, "raw", "raw", "test")  # a retry takes a new name


def test_a_failed_build_removes_only_the_candidate_it_wrote(pipeline, mods, sealer, tmp_path, monkeypatch) -> None:
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    session = copy_session(pipeline.b, tmp_path / "b")
    raw = raw_from(pipeline, sealer, root / "raw")

    def half_written(store, *args, **kwargs):  # the writer stops after its staging file and a vault temporary
        store.directory.mkdir(mode=0o700)
        (store.directory / ".gallery.sealed.staged").write_bytes(b"sealed but never verified")
        (store.directory / "..gallery.sealed.staged.777.tmp").write_bytes(b"half")
        return False, "capture_not_verified"

    monkeypatch.setattr(mods.build, "write_working_gallery", half_written)
    report = mods.build.build(session, raw, pipeline.base / "identity", root / "candidate", root, secret=SECRET,
                              sealer=sealer, match_threshold=POLICY.match_threshold)
    assert report["refused"] == "capture_not_verified" and report["candidate_removed"] is True
    assert not (root / "candidate").exists() and report["original_unchanged"]
    assert (raw / RAW_FILE).exists()  # the raw capture waits for the closing block's scoped cleanup
    # an earlier build's frozen candidate in place: a later refused build never touches it
    (root / "candidate").mkdir()
    (root / "candidate" / "gallery.sealed").write_bytes(b"an earlier candidate")
    report = mods.build.build(session, raw, pipeline.base / "identity", root / "candidate", root, secret=SECRET,
                              sealer=sealer, match_threshold=POLICY.match_threshold)
    assert report["refused"] == "candidate_target_exists" and (root / "candidate" / "gallery.sealed").exists()


def test_recovery_after_a_killed_build_and_an_abandoned_session_claims_cleans_and_keeps_what_it_must(
        mods, sealer, tmp_path) -> None:
    original, before, root, sealed = recovery_setup(sealer, tmp_path)
    manifest = mods.manifest.Manifest(root)
    copy = root / "wc-mp-3"
    mods.manifest.make_working_copy(manifest, original, copy, "test: m-F")  # listed at creation
    raw = root / "raw-mp-3"
    manifest.reserve(raw, "raw", "raw", "test: m-V")
    raw.mkdir(mode=0o700)
    (raw / "raw-capture.sealed").write_bytes(b"sealed raw faces")
    candidate = root / "candidate-mp-3"
    manifest.reserve(candidate, "candidate", "gallery", "test: m-X")  # the build was killed while writing
    candidate.mkdir(mode=0o700)
    (candidate / "gallery.sealed").write_bytes(b"sealed candidate")
    (candidate / "inbox").mkdir()
    unsealed = sealed.parent / "mp-20261107T120000Z"
    unsealed.mkdir()
    status = mods.manifest.status(manifest, sealed.parent)
    assert status["open_reservations"] == sorted([str(raw), str(candidate)])
    assert status["working_copies_left"] and status["unsealed_sessions"] == [str(unsealed)]
    with pytest.raises(mods.manifest.ManifestError):
        mods.manifest.recover(manifest, writer_exited=False, protected=(original, sealed.parent))
    report = mods.manifest.recover(manifest, writer_exited=True, protected=(original, sealed.parent))
    assert sorted(report["claimed"]) == sorted([str(raw), str(candidate)])
    assert not raw.exists() and not candidate.exists() and not copy.exists()
    after = mods.manifest.status(manifest, sealed.parent)
    assert not (after["open_reservations"] or after["raw_left"] or after["working_copies_left"]
                or after["unfrozen_candidate_left"])
    assert after["unsealed_sessions"] == [str(unsealed)]  # evidence: sealed as it is by m-H, never deleted
    assert unchanged(original, before) and (sealed / "run.jsonl").exists() and unsealed.exists()


def test_recovery_never_deletes_a_frozen_candidate(mods, sealer, tmp_path) -> None:
    original, before, root, sealed = recovery_setup(sealer, tmp_path)
    manifest = mods.manifest.Manifest(root)
    candidate = root / "candidate"
    manifest.reserve(candidate, "candidate", "gallery", "test")
    candidate.mkdir(mode=0o700)
    (candidate / "gallery.sealed").write_bytes(b"frozen candidate")
    mods.manifest.claim(manifest, candidate, writer_exited=True)
    frozen = root / "candidate.frozen.json"
    frozen.write_text("{}")
    mods.manifest.recover(manifest, writer_exited=True, protected=(original,), frozen=frozen)
    assert (candidate / "gallery.sealed").exists()  # waits for the retention decision (m-Z)
    assert mods.manifest.status(manifest, frozen=frozen)["unfrozen_candidate_left"] == []


# ---------------------------------------------------------------- optional spoken cues, timed synthetically


def guard_like_printer(mods, kind: str, log: Path, scale: float, extra_lines=()) -> list[dict]:
    """Print the session's cues into ``log`` as the guard does (``\\a[m:ss] TEXT`` at each cue's time, scaled down),
    recording each printed_mono_s as the guard's result.json does."""
    import threading
    import time as _time

    printed: list[dict] = []
    start = _time.monotonic()

    def run():
        with open(log, "a", encoding="ascii") as out:
            for line in extra_lines:
                out.write(line + "\n")
                out.flush()
            for name, at, text in mods.schedule.SESSIONS[kind]:
                while _time.monotonic() - start < at * scale:
                    _time.sleep(0.002)
                out.write(f"\a[{at // 60}:{at % 60:02d}] {text}\n")
                out.flush()
                printed.append({"name": name, "at_s": float(at), "printed_mono_s": _time.monotonic()})
                if name == "bs:lens" or at == 0:
                    out.write("a line the guard prints that is no cue\n")
                    out.flush()

    thread = threading.Thread(target=run)
    thread.start()
    return printed, thread


def test_spoken_cues_follow_the_printed_schedule_within_half_a_second_and_never_wait_for_a_slow_voice(
        mods, tmp_path) -> None:
    import sys as _sys
    import threading

    speak = MPE_SPEAK()
    log = tmp_path / "cues.log"
    log.write_text("")
    voice = [_sys.executable, "-c", "import sys, time; time.sleep(1.5)"]  # a slow voice: 1.5 s per cue
    spoken_log = tmp_path / "spoken.jsonl"
    spoken: list = []
    follower = threading.Thread(target=lambda: spoken.extend(
        speak.follow(log, "B", voice, spoken_log, give_up_s=5.0)))
    follower.start()
    printed, printer = guard_like_printer(mods, "B", log, scale=0.02,
                                          extra_lines=["STOP: not a cue", "\a[9:99] Walk to your seat"])
    printer.join()
    follower.join(timeout=30)
    assert not follower.is_alive()
    result = speak.check(spoken, {"cues": printed})
    assert result["ok"], result
    assert result["spoken"] == len(mods.schedule.B_CUES) and result["missing"] == [] and result["in_order"]
    assert result["max_lag_s"] < speak.MAX_LAG_S
    logged = [json.loads(line) for line in spoken_log.read_text().splitlines()]
    assert [e["name"] for e in logged] == [n for n, _, _ in mods.schedule.B_CUES]  # each once, in order


def test_a_cue_is_matched_only_at_its_own_time_and_text(mods) -> None:
    speak = MPE_SPEAK()
    assert speak.cue_of("B", "\a[1:15] Now face the camera lens until the next bell")[0] == "bslens"
    assert speak.cue_of("B", "\a[1:16] Now face the camera lens until the next bell")[0] == "bslens"  # one late
    assert speak.cue_of("B", "\a[1:17] Now face the camera lens until the next bell") is None
    assert speak.cue_of("B", "[1:15] Now face the camera lens until the next bell")[0] == "bslens"  # no bell char
    assert speak.cue_of("E_M", "\a[1:15] Now face the camera lens until the next bell") is None  # another session's
    assert speak.cue_of("B", "STOP: the camera probe failed") is None
    late = speak.check([{"name": "t0", "spoken_mono_s": 10.9}], {"cues": [{"name": "t0", "printed_mono_s": 10.0}]})
    assert not late["ok"] and late["late"] == {"t0": 0.9}


def MPE_SPEAK():
    import sys

    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module("mpe_speak")
    finally:
        sys.path.remove(str(RUNNERS))


# ---------------------------------------------------------------- the execution check on guarded sessions (mpe_check)

GUARD_REQUIRED = {"B": "face_windows_capture_raw", "E_M": "face_windows_shadow", "E_P": "face_windows_shadow"}
MARKS = {"seat_m": "0.90", "mark4_m": "4.02", "mark5_m": "4.97", "measured_with": "tape", "out_of_view_spot": "yes"}
STATUS_LISTENERS = [{"family": "tcp", "port": 18090, "scope": "loopback"}]  # the status page alone: no scene server
GUARDED_BOOT = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"  # the launch block records the kernel's boot UUID (the privacy
# profile admits nothing else); FakeClock's "fake-boot-1" in the records is a test name the execution check never reads


def RUNNER(name: str):
    import sys

    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(RUNNERS))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def guarded(mods, run: Path, kind: str, target: Path, accepted_scene, *, digest: str | None = None,
            given_digest: str | None = None, account: bool = True, face: dict | None = None,
            required: str | None = None, launch_last: bool = True, edit_lines=None) -> Path:
    """A pipeline session (the real runtime's run.jsonl and status captures) laid out as the package leaves it: the
    runtime's full starting line on route B with the session's own face entry (``face`` replaces it), the guard's
    result.json (its readiness, the frozen candidate's digest it was given, the cues printed 0.1 s late, the status
    captures), the session's face.yaml, marks and launch lines, then the account and the counts, in the package's
    order. ``edit_lines`` changes the run's records (a list of JSON lines) before the guard's summary reads them."""
    session = target / f"mp-{kind}"
    out = session / "run"
    out.mkdir(parents=True)
    for p in run.iterdir():
        if p.name == "run.jsonl" or p.name.startswith("status-"):
            (out / p.name).write_bytes(p.read_bytes())
    startup_root = target / "startup"
    startup_root.mkdir()
    real = real_face_startup(startup_root, accepted_scene, windows=True)
    own = {"capture": True, "capture_raw": True} if kind == "B" else {"shadow": True, "shadow_digest": digest,
                                                                       "lives": True}
    real["face"] = face if face is not None else {**real["face"], **own}
    lines = (out / "run.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["run"] == "starting"
    lines[0] = json.dumps(startup_line(real, scene=False))
    (out / "run.jsonl").write_text("\n".join(edit_lines(lines) if edit_lines else lines) + "\n")
    cues, statuses = mods.schedule.SESSIONS[kind], mods.schedule.STATUS_AT[kind]
    duration = float(mods.schedule.duration_s(cues))
    readiness = {"required": required or GUARD_REQUIRED[kind], "met": True, "reason": None, "starting_after_s": 70.0}
    parameters = {"duration_s": duration, "status_at": [{"at_s": float(a), "name": n} for a, n in statuses],
                  "cues": [{"at_s": float(a), "name": n} for n, a, _ in cues], "require_ready": readiness["required"],
                  **({} if kind == "B" else {"shadow_digest": given_digest or digest})}
    guard = SimpleNamespace(guard=RUNNER("step5_guard"), check=RUNNER("face_check"))
    mods.sim.finish(guard, out, account=None, counts={}, cues={n: float(a) for n, a, _ in cues}, duration_s=duration,
                    captures=statuses, readiness=readiness, provenance=f"boot_id={GUARDED_BOOT}\n",
                    result_update={"parameters": parameters, "captures": [
                        {"name": n, "at_s": float(a), "status": "saved", "listeners": STATUS_LISTENERS}
                        for a, n in statuses]})
    rc = mods.check.rc
    (session / "face.yaml").write_text(rc.RANGE_CONFIG)
    marks = "".join(f"{k}={v}\n" for k, v in MARKS.items())
    (session / "marks.txt").write_text(marks)
    launch = [f"marks_sha256={_sha(marks)}", f"MV_config_sha256={_sha(rc.RANGE_CONFIG)}",
              "MV_launch_utc=2026-11-07T09:00:00Z"]
    (session / "provenance.txt").write_text("".join(f"{line}\n" for line in (launch if launch_last else launch[::-1])))
    if account:
        mods.account.enter(out, kind, others_in_view="no", not_as_cued="none")
    for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                           ("face_counts_f2", "secret-counts-identity.txt")):  # after the account, as the block does
        done = mods.sim.bash_counts(function, str(out))
        assert done.returncode == 0, done.stderr
        (out / name).write_text(done.stdout)
    return out


@pytest.fixture
def f1_list(tmp_path) -> Path:
    f1 = tmp_path / "f1"
    f1.mkdir()
    (f1 / "list.json").write_text(json.dumps({"identity": "list", "identities": [
        {"identity_id": ENROLLED, "prototypes": 3, "consent_date": "2026-10-08",
         "enrolled_utc": "2026-10-08T09:00:00+00:00"}]}))
    return f1


def execution(mods, out: Path, kind: str, f1: Path, frozen=None) -> tuple[bool, str]:
    reading = mods.check.read_mpe_execution(out, f1, kind, frozen)
    return reading.ok, "\n".join(reading.lines)


def test_each_guarded_session_passes_the_execution_check_with_its_own_entry_records_and_account(
        pipeline, mods, tmp_path, accepted_scene, f1_list) -> None:
    digest = pipeline.frozen["prototypes_digest"]
    for kind, run in (("B", pipeline.b), ("E_M", pipeline.em), ("E_P", pipeline.ep)):
        out = guarded(mods, run, kind, tmp_path / kind, accepted_scene, digest=digest)
        ok, text = execution(mods, out, kind, f1_list, None if kind == "B" else pipeline.frozen)
        assert ok, text
        assert text.splitlines()[-1] == "mpe-execution: passed"
        assert digest not in text and ENROLLED not in text  # neither the digest nor the identity ID is printed
        assert "FAIL" not in text and "MISSING" not in text
        provenance = (out / "provenance.txt").read_text().splitlines()
        assert [line.split("=")[0] for line in provenance].count("reading_started_utc") == 1  # a reading, marked


@pytest.mark.parametrize("kind, change, expected", [
    ("E_M", "no_account", "account_not_locked"),
    ("B", "readiness", "the guard's readiness"),
    ("B", "ordinary_capture_entry", "face admitted for this validation run"),
    ("B", "raw_face_dropped", "the raw capture"),
    ("E_M", "given_digest", "the guard was given the frozen candidate's digest"),
    ("E_M", "lives_dropped", "identity records"),
    ("E_P", "launched_first", "face.yaml"),
])
def test_the_execution_check_fails_a_session_that_is_not_the_runs_own(pipeline, mods, tmp_path, accepted_scene,
                                                                         f1_list, kind, change, expected) -> None:
    digest = pipeline.frozen["prototypes_digest"]
    run = {"B": pipeline.b, "E_M": pipeline.em, "E_P": pipeline.ep}[kind]
    options = {}
    if change == "no_account":
        options["account"] = False
    elif change == "readiness":
        options["required"] = "face_windows"
    elif change == "ordinary_capture_entry":  # the ordinary gallery capture's entry: never route B's raw capture run
        options["face"] = {**mods.check.fc.WINDOWS_FACE, "capture": True}
    elif change == "raw_face_dropped":
        def fewer(lines):
            out = []
            for line in lines:
                record = json.loads(line)
                if record.get("identity") == "capture_raw":
                    record["faces_kept"] -= 1
                out.append(json.dumps(record))
            return out
        options["edit_lines"] = fewer
    elif change == "given_digest":
        options["given_digest"] = "0" * 64
    elif change == "lives_dropped":
        options["edit_lines"] = lambda lines: [line for line in lines if json.loads(line).get("identity") != "life"]
    elif change == "launched_first":
        options["launch_last"] = False
    out = guarded(mods, run, kind, tmp_path, accepted_scene, digest=digest, **options)
    ok, text = execution(mods, out, kind, f1_list, None if kind == "B" else pipeline.frozen)
    assert not ok and text.splitlines()[-1].startswith("mpe-execution: not passed")
    failed = [line for line in text.splitlines() if line.startswith(("FAIL", "MISSING"))]
    assert failed and any(expected in line for line in failed), text
    assert digest not in text and "0" * 64 not in text
    if change == "no_account":  # nothing of the session is judged or shown: the account item alone
        assert len(failed) == 1 and "STOP: enter the account first" in text
        assert "status captures" not in text and "identity records" not in text
        assert "reading_started_utc" not in (out / "provenance.txt").read_text()


def test_the_builder_and_the_comparison_refuse_a_session_that_has_not_passed_or_another_config(
        pipeline, mods, tmp_path, accepted_scene, f1_list, monkeypatch, capsys) -> None:
    digest = pipeline.frozen["prototypes_digest"]
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(pipeline.frozen))
    em = guarded(mods, pipeline.em, "E_M", tmp_path / "em", accepted_scene, digest=digest)
    other = tmp_path / "other.yaml"
    other.write_text(mods.check.rc.RANGE_CONFIG + "# changed\n")
    # the comparison: another configuration, then a session without its account, then the session as run
    assert mods.compare.main([str(em), "--config", str(other), "--frozen", str(frozen), "--f1", str(f1_list)]) == 1
    assert "STOP: config_not_the_sessions" in capsys.readouterr().out
    unaccounted = guarded(mods, pipeline.em, "E_M", tmp_path / "em2", accepted_scene, digest=digest, account=False)
    config = unaccounted.parent / "face.yaml"
    assert mods.compare.main([str(unaccounted), "--config", str(config), "--frozen", str(frozen),
                              "--f1", str(f1_list)]) == 1
    text = capsys.readouterr().out
    assert "STOP: execution_not_passed" in text and "STOP: enter the account first" in text
    assert "compared" not in text and "CHRONOLOGICAL" not in text and "KNOWN" not in text
    mods.compare.main([str(em), "--config", str(em.parent / "face.yaml"), "--frozen", str(frozen),
                       "--f1", str(f1_list)])
    text = capsys.readouterr().out
    assert "mpe-execution: passed" in text and "STOP" not in text and "CHRONOLOGICAL" in text
    # the builder: refused before anything is read or listed
    monkeypatch.setenv("SENTINEL_IDENTITY_PASSPHRASE", "a long synthetic passphrase")
    b = guarded(mods, pipeline.b, "B", tmp_path / "b", accepted_scene, account=False)
    root = tmp_path / "exp"
    root.mkdir(mode=0o700)
    common = ["build", "--session", str(b), "--raw", str(root / "raw"), "--identity-dir",
              str(pipeline.base / "identity"), "--candidate", str(root / "candidate"), "--root", str(root),
              "--f1", str(f1_list)]
    assert mods.build.main(common + ["--config", str(other)]) == 1
    assert "STOP: config_not_the_sessions" in capsys.readouterr().err
    monkeypatch.setenv("SENTINEL_IDENTITY_PASSPHRASE", "a long synthetic passphrase")
    assert mods.build.main(common + ["--config", str(b.parent / "face.yaml")]) == 1
    seen = capsys.readouterr()
    assert "STOP: execution_not_passed" in seen.err and "STOP: enter the account first" in seen.out
    assert sorted(p.name for p in root.iterdir()) == []  # nothing reserved, listed or written
