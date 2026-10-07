"""The V2-25 device check's readings (face_check.py) and the F2 account helper (face_account.py).

F2's evidence is produced by the real runtime: DemoRuntime, EdgeCore and a real FaceWorker run a scripted ten
minutes on a fake clock; the identity records come from drain_identity_records(), the status lines from the CLI's
status_line(), the captures from the status page's build_status(), the startup's face fields from assemble() and its
face release from post_load_release(), and the guard summary from step5_guard.read_run_output(). Only the camera,
the detector and DeepFace are fakes, with synthetic embeddings. F1's evidence comes from the real `sentinel identity`
commands with a real sealed gallery. So a change in the runtime's output schema fails these tests (unlike session
47's hand-written layers value).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel import demo_runtime
from sentinel.adapters import FACE_ADAPTER_ID, FaceAdmission, FaceAdmissionStatus
from sentinel.cli import main, status_line
from sentinel.config import IdentityConfig, parse_config
from sentinel.contracts import NormalizedBox, PixelFormat, StreamIdentity
from sentinel.demo_runtime import (
    DemoRuntime,
    Devices,
    FaceOptions,
    FaceRuntime,
    MemoryOps,
    RunOptions,
    SceneRuntime,
    assemble,
    file_facts,
    post_load_release,
)
from sentinel.identity.association import FaceObservation
from sentinel.identity.gallery import (
    CONSENT_SCOPE,
    FaceCompatibility,
    GalleryDocument,
    GalleryIdentity,
    IdentityStore,
    unit,
)
from sentinel.identity.legacy_deepface import EnrollmentFace, FaceRun, static_compatibility_fields
from sentinel.identity.vault import SECRET_ENV, Sealer, Secret, VaultError
from sentinel.identity.worker import TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING, FaceWorker
from sentinel.incidents.service import IncidentService
from sentinel.media.capture import CapturedFrame, CaptureState, CaptureStatus, LatestFrame
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.memory_policy import RELEASE_FILE_ROLES
from sentinel.jobs import WorkerOutcome
from sentinel.runtime import FaceResultRecord, IdentityTransition, tick_record
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.status_page import build_status, read_store
from sentinel.storage.database import Database
from sentinel.tracking.tracker import PersonTracker, RawTrack

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
NOW = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)
ENROLLED = "idn-00000000000a"
COMPAT = FaceCompatibility(**static_compatibility_fields(), opencv_version="4.13.0", tensorflow_version="2.21.0",
                           tf_keras_version="2.21.0", numpy_version="2.2.6")


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


GALLERY = GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
    identity_id=ENROLLED, enrolled_utc=NOW, consent_date="2026-10-08", consent_scope=CONSENT_SCOPE,
    prototypes=(vec(1.0),)),))
FACE = vec(0.97, 0.1, 0.0, 0.05)
BOX = (100.0, 100.0, 200.0, 400.0)  # a person on the 640x480 frame
FACE_BOX = NormalizedBox(x1=0.19, y1=0.22, x2=0.28, y2=0.35)  # in that person's head region
# What the operator did (seconds after the T0 cue), and the account they would give.
SCRIPT = ((0, 12, "facing", 3), (12, 120, "absent", None), (120, 240, "facing", 1), (240, 360, "away", 1),
          (360, 480, "absent", None), (480, 601, "facing", 2))
ACCOUNT = {"stopwatch_started_at_t0": "yes", "out_of_view_1": "0:12", "seated_facing_1": "2:00",
           "turned_away": "4:00", "face_hidden_while_turned": "yes", "stood_up": "6:00", "out_of_view_2": "6:01",
           "seated_facing_2": "8:00", "stayed_until_stop": "yes", "other_faces_in_view": "no",
           "notes": "a note that is never printed"}
REPORT = json.dumps({"persons_visible": 1, "fire_or_smoke": False, "threat": "low", "observations": ["a person"],
                     "uncertainty": "low", "summary": "A synthetic report."})
CAPTURES = ((60.0, "e1"), (180.0, "k1"), (300.0, "a"), (420.0, "e2"), (540.0, "k2"), (590.0, "end"))
LOOPBACK = [{"family": "tcp", "port": 18081, "scope": "loopback"}, {"family": "tcp", "port": 18090, "scope": "loopback"}]


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(check=importlib.import_module("face_check"), account=importlib.import_module("face_account"),
                           guard=importlib.import_module("step5_guard"))


# ---------------------------------------------------------------- the real startup's face fields


def real_face_startup(root: Path, accepted_scene) -> dict[str, Any]:
    """assemble()'s face fields and post_load_release()'s face record, as `sentinel run --face` writes them."""
    scene = accepted_scene()
    weights = root / "weights"
    weights.mkdir()
    facts = {}
    for name in RELEASE_FILE_ROLES["face"]:
        (weights / name).write_bytes(name.encode())
        facts[name] = dataclasses.replace(file_facts(weights / name), sha256=hashlib.sha256(name.encode()).hexdigest())
    admission = FaceAdmission(profile_id=scene.profile.profile_id, status=FaceAdmissionStatus.PENDING_VALIDATION,
                              face_hz=1.0, model_weights=facts["facenet512_weights.h5"],
                              detector_weights=facts["face_detection_yunet_2023mar.onnx"], deepface_version="0.0.99")
    backend = SimpleNamespace(weight_files={n: weights / n for n in RELEASE_FILE_ROLES["face"]}, load=lambda: None,
                              compatibility=lambda: COMPAT, faces=lambda *a: FaceRun((), 0, 0))
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": [{
        "adapter_id": FACE_ADAPTER_ID, "contract_version": 1, "implementation_revision": "1", "enabled": True,
        "input_kinds": ["frame"], "output_kinds": ["face.observation"], "model_revision": "facenet512-yunet",
        "resource_profile_id": scene.profile.profile_id, "timeout_ms": 3000}]})
    devices = Devices(capture_source=lambda c: SimpleNamespace(),
                      tracker_backend=lambda e: SimpleNamespace(load=lambda: None, track=lambda i: [], reset=lambda: None),
                      scene_server=lambda o, p: None, scene_request=lambda p, t: None,
                      meminfo=lambda: {"MemFree": 5_000_000_000, "MemAvailable": 6_000_000_000},
                      notifiers=lambda c: ({}, {}), face_backend=lambda w: backend,
                      identity_store=lambda d: SimpleNamespace(exists=lambda: True, load=lambda s: GALLERY))
    (root / "data").mkdir()
    assembly = assemble(config, RunOptions(root / "data", root / "x.engine", face=FaceOptions(
        root / "identity", Secret("a long synthetic passphrase"), validation_run=True)), devices, FakeClock(),
        profiles=scene.profiles, face_admissions={admission.profile_id: admission})
    assembly.database.close()
    release = post_load_release(MemoryOps(release=lambda path: {"result": "returned_0", "returncode": 0, "error": None,
                                                                "bytes": 10, "elapsed_s": 0.001},
                                          sleep=lambda s: None), "face", backend.weight_files)
    return {key: assembly.startup[key] for key in ("face", "face_problem", "face_memory_before")} | {"face_release": release}


def startup_line(face: dict[str, Any]) -> dict[str, Any]:
    release = {"settle_s": 15.0, "files": {}}
    return {"run": "starting", "startup": {
        "memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"},
        "thp_scope": {"before_launch": {"runtime": 1}, "llama_ready": {"runtime": 1, "scene_server": 1},
                      "workload_verified": {"runtime": 0, "scene_server": 1}},
        "thp_disable": {"verified": True, "reason": None, "set_rc": 0, "thp_enabled": 0},
        "releases": {"scene": {**release, "files": {"llm": {"result": "returned_0"}, "mmproj": {"result": "returned_0"}}},
                     "detector": {**release, "files": {"engine": {"result": "returned_0"}}},
                     "face": face["face_release"]},
        "scene_server": {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True},
        "scene_problem": None, "detector_problem": None, "notifier_problems": {},
        "face": face["face"], "face_problem": face["face_problem"], "face_memory_before": face["face_memory_before"],
        "status_page": "http://127.0.0.1:18090/"}}


# ---------------------------------------------------------------- the scripted live run


class Image:
    shape = (480, 640, 3)

    def copy(self) -> "Image":
        return Image()


class Capture:
    def __init__(self, clock: FakeClock) -> None:
        self.stamper = FrameStamper("cam-1", clock)
        self.stamper.connect()

    @property
    def connected(self) -> StreamIdentity | None:
        return self.stamper.current_stream

    def status(self) -> CaptureStatus:
        return CaptureStatus(CaptureState.STREAMING, self.connected, 0, 1, 0, 0, None, None)

    def start(self) -> None:
        pass

    def stop(self, timeout_s: float) -> bool:
        return True


class Tracker:
    def __init__(self) -> None:
        self.next: list[RawTrack] = []

    def load(self) -> None:
        pass

    def track(self, image: Any) -> list[RawTrack]:
        return self.next

    def reset(self) -> None:
        pass


def segment(script, t: float) -> tuple[str, int | None]:
    for start, end, state, track in script:
        if start <= t < end:
            return state, track
    return "absent", None


def simulate(root: Path, startup: dict, *, script=SCRIPT, away_face=None, fps: float = 4.0,
             duration: float = 600.0) -> Path:
    """Run the real runtime through ``script`` and write the F2 directory as the package's blocks would."""
    out = root / "f2"
    out.mkdir()
    clock = FakeClock(utc=NOW)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    slot, capture, tracker = LatestFrame(), Capture(clock), Tracker()
    t0 = clock.monotonic_ns()

    def analyze(frame, image):
        state, _ = segment(script, (frame.ingest_mono_ns - t0) / 1e9)
        embedding = FACE if state == "facing" else away_face if state == "away" else None
        if embedding is None:
            return FaceRun((), 0, 1)  # DeepFace's no-face fallback, dropped
        return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=embedding),), 1, 0)

    worker = FaceWorker(analyze, clock)
    images = RecentImages()
    scene = SceneRuntime(ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                               revision="1+lfm2-vl-1.6b-q4_0"), images)
    db = Database.open(root / "sentinel.db")
    runtime = DemoRuntime(config, clock, capture=capture, slot=slot,
                          incidents=IncidentService(db, clock, incidents=config.incidents,
                                                    notifications=config.notifications),
                          tracker=PersonTracker(tracker), scene=scene, face=FaceRuntime(worker, 1, True),
                          enrollment=GALLERY.enrollment(), status_refresh_s=0.0)
    runtime.start()
    lines = [json.dumps(startup)]
    pending, captures, next_status = list(CAPTURES), [], 0.0
    for n in range(1, int(duration * fps) + 1):
        t = n / fps
        state, track = segment(script, t)
        tracker.next = [] if state == "absent" else [RawTrack(track, *BOX, 0.9)]
        clock.advance(ns=t0 + round(n * 1e9 / fps) - clock.monotonic_ns())
        frame = capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        slot.publish(CapturedFrame(frame, Image()))
        runtime.step(timeout_s=0)
        deadline = time.monotonic() + 5
        while worker.status()["busy"] or worker.status()["waiting"]:
            assert time.monotonic() < deadline
            time.sleep(0.0005)
        runtime.step(timeout_s=0)
        lines += [json.dumps(record) for record in runtime.drain_identity_records()]
        if t >= next_status:
            lines.append(json.dumps(status_line(runtime.snapshot())))
            next_status += 30.0
        while pending and t >= pending[0][0]:
            at, name = pending.pop(0)
            status = build_status(runtime.snapshot(), read_store(root / "sentinel.db"), clock.utc_now())
            (out / f"status-{name}.json").write_text(json.dumps(status))
            captures.append({"name": name, "at_s": at, "status": "saved", "listeners": LOOPBACK})
    shutdown = runtime.shutdown()
    lines += [json.dumps(record) for record in runtime.drain_identity_records()]
    lines.append(json.dumps({"run": "stopped", "shutdown": shutdown, "database_closed": True,
                             "status": status_line(runtime.snapshot())}))
    db.close()
    (out / "run.jsonl").write_text("\n".join(lines) + "\n")
    return out


def finish(mods, out: Path, *, cue_delay_s: float = 0.1, account: dict[str, str] | None = ACCOUNT,
           counts: dict[str, int] | None = None) -> None:
    """result.json as the guard writes it, then the account and the counts, in the package's order."""
    ready = 1.0  # FakeClock's default start (seconds), the runtime's T0
    result = {
        "schema_version": 1, "mode": "step5_guard", "status": "duration_stop",
        "parameters": {"duration_s": 600.0, "cues": [{"at_s": 0.0, "name": "t0"}]},
        "child": {"returncode": 0, "ready_after_s": 70.0, "ready_mono_s": ready},
        "guard": {"trigger": None, "peak_pressure_bytes": 1, "min_mem_free_bytes": 2, "min_mem_available_bytes": 3},
        "stop": {"forced": False}, "cleanup_clear": True,
        "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0, "runtime_group_left": False,
                      "listeners_after": []},
        "captures": json.loads((out / "captures.json").read_text()) if (out / "captures.json").exists() else
        [{"name": name, "at_s": at, "status": "saved", "listeners": LOOPBACK} for at, name in CAPTURES],
        "cues": [{"name": "t0", "at_s": 0.0, "printed_after_launch_s": 70.1, "printed_mono_s": ready + cue_delay_s}],
        "run_output": mods.guard.read_run_output(out / "run.jsonl"),
    }
    (out / "result.json").write_text(json.dumps(result))
    (out / "provenance.txt").write_text("part=f2\n")
    if account is not None:
        answers = iter([account[f.key] for f in mods.account.FIELDS] + ["SAVE"])
        assert mods.account.enter(out, ask=lambda prompt: next(answers), say=lambda text: None) == 0
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in mods.check.NOT_COUNTED)
    counts = counts or {}
    (out / "secret-counts-camera.txt").write_text("".join(f"{n} userinfo={counts.get('userinfo', 0)}\n" for n in names))
    (out / "secret-counts-identity.txt").write_text("".join(
        f"{n} passphrase={counts.get('passphrase', 0)}\n{n} name={counts.get('name', 0)}\n" for n in names))


@pytest.fixture
def startup(tmp_path, accepted_scene) -> dict:
    root = tmp_path / "startup"
    root.mkdir()
    return startup_line(real_face_startup(root, accepted_scene))


@pytest.fixture
def f1_list(tmp_path) -> Path:
    f1 = tmp_path / "f1"
    f1.mkdir()
    (f1 / "list.json").write_text(json.dumps({"identity": "list", "identities": [
        {"identity_id": ENROLLED, "prototypes": 3, "consent_date": "2026-10-08", "enrolled_utc": NOW.isoformat()}]}))
    return f1


def reading(mods, out, f1):
    result = mods.check.read_f2(out, f1)
    return result, "\n".join(result.lines)


def line_for(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.split(" ", 1)[1].startswith(start))


# ---------------------------------------------------------------- F2


def test_the_scripted_live_run_is_validated_from_the_runtimes_own_output(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert "MISSING" not in text and "FAIL" not in text and "NOT_EXERCISED" not in text
    retention = line_for(text, "A: a known identity is retained")
    assert retention.startswith("PASS") and '"cleared_to": "unresolved"' in retention
    assert "a note that is never printed" not in text
    assert "DESCRIPTIVE scheduling rate" in text and "DESCRIPTIVE time to known" in text


def test_a_face_still_visible_while_turned_away_does_not_exercise_retention(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, away_face=FACE)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "A: a known identity is retained").startswith("NOT_EXERCISED")


def test_retention_is_not_judged_when_the_account_says_the_face_was_visible(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account={**ACCOUNT, "face_hidden_while_turned": "no"})
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "A: a known identity is retained").startswith("NOT_EXERCISED")


def test_a_tracker_reusing_the_id_after_the_return_carries_nothing_over(mods, tmp_path, startup, f1_list) -> None:
    script = SCRIPT[:-1] + ((480, 601, "facing", 1),)
    out = simulate(tmp_path, startup, script=script)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text


def test_the_account_is_required_locked_and_unchanged(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account=None)
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "account saved and locked").startswith("FAIL")
    assert "completion" not in text  # nothing about identity is shown without an account
    (tmp_path / "b").mkdir()
    out2 = simulate(tmp_path / "b", startup)
    finish(mods, out2)
    (out2 / "account.txt").write_text((out2 / "account.txt").read_text().replace("turned_away=4:00", "turned_away=4:10"))
    result, text = reading(mods, out2, f1_list)
    assert not result.ok and "differs from its recorded SHA-256" in text


@pytest.mark.parametrize(("change", "item", "label"), [
    (lambda a: {**a, "out_of_view_1": "unknown"}, "account complete and in order", "FAIL"),
    (lambda a: {**a, "other_faces_in_view": "yes"}, "stopwatch started at the T0 cue", "FAIL"),
    (lambda a: {**a, "stood_up": "4:40", "out_of_view_2": "4:41"}, "judged windows from the account", "FAIL"),
])
def test_account_problems_fail_the_reading(mods, tmp_path, startup, f1_list, change, item, label) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, account=change(ACCOUNT))
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, item).startswith(label)


def _nth_result(lines: list[dict], n: int, change) -> list[dict]:
    indices = [i for i, line in enumerate(lines) if line.get("identity") == "result"]
    return [change(line) if i == indices[n] else line for i, line in enumerate(lines)]


def _mutate(out: Path, edit) -> None:
    lines = [json.loads(line) for line in (out / "run.jsonl").read_text().splitlines()]
    lines = edit(lines)
    (out / "run.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


@pytest.mark.parametrize(("edit", "item", "label"), [
    # a field missing from one result: the schema changed
    (lambda ls: _nth_result(ls, 0, lambda l: {k: v for k, v in l.items() if k != "frame_mono_ns"}),
     "identity records: schema 1", "FAIL"),
    # another schema version
    (lambda ls: [dict(l, schema=2) if l.get("identity") == "tick" else l for l in ls], "identity records", "FAIL"),
    # most results lost: completion below 0.95
    (lambda ls: [l for n, l in enumerate(ls) if not (l.get("identity") == "result" and n % 3)], "completion", "FAIL"),
    # one processing error
    (lambda ls: _nth_result(ls, 4, lambda l: dict(l, outcome="failed", persons=[])), "completion", "FAIL"),
    # another identity known on the return
    (lambda ls: [dict(l, identity_id="idn-00000000000b") if l.get("identity") == "transition"
                 and l["to"] == "known" and l["track_id"] == 2 else l for l in ls], "only the enrolled", "FAIL"),
    # no offered tick and no result while facing in K2 (the person was not seen)
    (lambda ls: [l for l in ls if not (l.get("identity") in ("tick", "result") and l.get("frame_mono_ns", 0) > 495e9)],
     "K2 sampling", "FAIL"),
])
def test_runtime_output_deviations_fail_the_reading(mods, tmp_path, startup, f1_list, edit, item, label) -> None:
    out = simulate(tmp_path, startup)
    _mutate(out, edit)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, item).startswith(label), text


def test_a_known_change_without_two_matches_or_on_two_tracks_fails(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)

    def inject(lines):
        start = next(l for l in lines if l.get("identity") == "transition" and l["to"] == "known" and l["track_id"] == 1)
        fake = dict(start, track_id=9, mono_ns=start["mono_ns"] + 5_000_000_000)
        return lines + [fake]

    _mutate(out, inject)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert line_for(text, "one identity is never known on two tracks").startswith("FAIL")
    assert line_for(text, "every change to known follows").startswith("FAIL") and not result.ok


def test_t0_secrets_and_startup_problems_fail_the_reading(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, cue_delay_s=3.0, counts={"name": 1})
    result, text = reading(mods, out, f1_list)
    assert line_for(text, "T0: the t0 cue").startswith("FAIL")
    assert line_for(text, "passphrase and name counts").startswith("FAIL")
    broken = dict(startup, startup={**startup["startup"], "face": {"validation_run": False, "identities_enrolled": 1}})
    (tmp_path / "x").mkdir()
    out2 = simulate(tmp_path / "x", broken, duration=10.0)
    finish(mods, out2)
    _, text = reading(mods, out2, f1_list)
    assert line_for(text, "face admitted for this validation run").startswith("FAIL")


def test_the_checker_matches_the_runtimes_schema_and_values(mods) -> None:
    check = mods.check
    frame = SimpleNamespace(stream_epoch=0, frame_seq=1, ingest_mono_ns=5)
    assert set(tick_record(frame, [], TICK_OFFERED)) == check.TICK_KEYS
    record = FaceResultRecord(0, 1, 5, 6, "applied", 1, ((1, "assigned", "match", ENROLLED, "match"),)).record()
    assert set(record) == check.RESULT_KEYS and set(record["persons"][0]) == check.PERSON_KEYS
    assert set(IdentityTransition(0, 1, "unresolved", "known", ENROLLED, "fresh", 0, "r", 5).record()) == \
        check.TRANSITION_KEYS
    assert check.TICKS == {TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING}
    defaults = IdentityConfig()
    assert (check.VOTE_TTL_S, check.FRESH_S, check.CONFIRMATIONS) == (
        defaults.vote_ttl_s, defaults.face_result_max_age_s, defaults.confirmations)
    assert check.MAX_IDENTITY_RECORDS == demo_runtime.MAX_IDENTITY_RECORDS
    assert set(check.FACE_RELEASES) == set(RELEASE_FILE_ROLES["face"])
    assert check.MIN_COMPLETION == 0.95 and check.SCHEMA == 1


# ---------------------------------------------------------------- F2 account helper


def test_the_account_waits_for_the_run_saves_once_and_locks(mods, tmp_path) -> None:
    run = tmp_path / "f2"
    run.mkdir()
    (run / "provenance.txt").write_text("part=f2\n")
    said: list[str] = []
    assert mods.account.enter(run, ask=lambda p: "", say=said.append) == 2 and "not finished" in said[-1]
    (run / "result.json").write_text("{}")
    answers = iter(["yes", "0:12", "unknown", "2:00", "4:00", "yes", "6:00", "6:01", "8:00", "yes", "no", "",
                    "3", "1:58", "SAVE"])
    assert mods.account.enter(run, ask=lambda p: next(answers), say=said.append) == 0
    saved = (run / "account.txt").read_text()
    assert "seated_facing_1=1:58" in saved and "unknown" not in saved
    locked = (run / "provenance.txt").read_text().splitlines()[-1]
    assert locked == "account_sha256=" + hashlib.sha256(saved.encode()).hexdigest()
    assert mods.account.enter(run, ask=lambda p: "", say=said.append) == 2  # never changed once saved
    said.clear()
    assert mods.account.check(run, say=said.append) == 0 and said[-1].startswith("Locked: yes; the file matches")


def test_the_account_review_reports_order_and_unknowns(mods) -> None:
    values = {**ACCOUNT, "seated_facing_1": "5:00", "stood_up": "unknown"}
    review = mods.account.review(values)
    assert "stood_up" in review[0] and "turned_away 4:00 should be after seated_facing_1 5:00" in review[2]
    assert review[-1] == "Complete and in order: no"


# ---------------------------------------------------------------- F1


class Pipeline:
    def load(self) -> None:
        pass

    def compatibility(self) -> FaceCompatibility:
        return COMPAT

    def enrollment_photo(self, path: Path):
        kind = path.read_text()
        if kind == "none":
            return None, "no_face"
        return EnrollmentFace(vec(1.0, 0.05 * int(kind)), 0.9, 120), "face_found"


def run_f1(mods, tmp_path, monkeypatch, capsys, *, kinds=("1", "2", "3"), break_step=None) -> tuple[Path, Path]:
    sealer = Sealer(kdf_n=2**14)
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    identity_dir = tmp_path / "identity"
    store = IdentityStore(identity_dir, sealer)
    store.ensure_directory()
    folder = store.inbox / "f1-20261008T090000Z"
    folder.mkdir(mode=0o700)
    for n, kind in enumerate(kinds):
        (folder / f"Jane Doe {n}.jpg").write_text(kind)
    f1 = tmp_path / "f1"
    f1.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    provenance = [f"photos_supplied={len(kinds)}", "consent_date=2026-10-08", f"inbox_folder_name={folder.name}",
                  "gallery_before_dryrun=absent"]

    def identity(*args):
        monkeypatch.setenv(SECRET_ENV, "a long synthetic passphrase")
        code = main(["identity", *args[:1], str(config), *args[1:], "--identity-dir", str(identity_dir)],
                     face_backend=lambda w: Pipeline(), identity_store=lambda d: store)
        return code, capsys.readouterr().out

    common = (str(folder), "--consent-confirmed", "--consent-date", "2026-10-08")
    code, out = identity("enroll", *common, "--dry-run")
    (f1 / "dryrun.json").write_text(out)
    provenance += [f"dryrun_exit={code}", f"gallery_after_dryrun={'present' if store.exists() else 'absent'}"]
    if break_step == "dryrun_writes":
        provenance[-1] = "gallery_after_dryrun=present"
    code, out = identity("enroll", *common)
    (f1 / "enroll.json").write_text(out)
    provenance.append(f"enroll_exit={code}")
    code, out = identity("list")
    (f1 / "list.json").write_text(out)
    provenance.append(f"list_exit={code}")
    (f1 / "provenance.txt").write_text("\n".join(provenance) + "\n")
    names = sorted(p.name for p in f1.iterdir() if p.is_file()) + ["audit.jsonl"]
    bad = 1 if break_step == "name_found" else 0
    (f1 / "secret-counts-identity.txt").write_text("".join(
        f"{n} passphrase=0\n{n} name={bad}\n{n} filenames=0\n" for n in names))
    return f1, identity_dir


def test_a_consented_enrollment_is_validated(mods, tmp_path, monkeypatch, capsys) -> None:
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys)
    result = mods.check.read_f1(f1, identity_dir)
    text = "\n".join(result.lines)
    assert result.ok, text
    assert "Jane" not in text and "Jane" not in "".join(p.read_text() for p in f1.iterdir())


@pytest.mark.parametrize(("break_step", "item"), [
    ("dryrun_writes", "dry run"),
    ("name_found", "passphrase, name and photo-file-name counts"),
])
def test_f1_deviations_fail(mods, tmp_path, monkeypatch, capsys, break_step, item) -> None:
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys, break_step=break_step)
    result = mods.check.read_f1(f1, identity_dir)
    assert not result.ok and line_for("\n".join(result.lines), item).startswith("FAIL")


def test_two_usable_photos_fail_this_package_and_three_pass(mods, tmp_path, monkeypatch, capsys) -> None:
    """The package needs 3 usable photos (maintainer, 2026-10-08), although the enrollment API accepts 2."""
    from sentinel.identity.enroll import MIN_PHOTOS

    assert (MIN_PHOTOS, mods.check.MIN_USABLE_PHOTOS) == (2, 3)
    two = tmp_path / "two"
    two.mkdir()
    f1, identity_dir = run_f1(mods, two, monkeypatch, capsys, kinds=("1", "2", "none"))
    assert json.loads((f1 / "enroll.json").read_text())["verified"] is True  # the API itself accepted 2
    result = mods.check.read_f1(f1, identity_dir)
    text = "\n".join(result.lines)
    assert not result.ok
    assert line_for(text, "dry run").startswith("FAIL") and line_for(text, "enrollment verified").startswith("FAIL")
    assert '"accepted": 2' in line_for(text, "dry run")
    others = [line for line in text.splitlines()[:-1]
              if not line.split(" ", 1)[1].startswith(("dry run", "enrollment verified"))]
    assert all(line.startswith("PASS") for line in others), text  # every other condition holds
    for kinds in (("1", "2", "3"), ("1", "2", "3", "none")):
        root = tmp_path / f"three-{len(kinds)}"
        root.mkdir()
        f1, identity_dir = run_f1(mods, root, monkeypatch, capsys, kinds=kinds)
        result = mods.check.read_f1(f1, identity_dir)
        assert result.ok, "\n".join(result.lines)


def test_f1_fails_if_a_supplied_copy_was_left_or_storage_is_not_private(mods, tmp_path, monkeypatch, capsys) -> None:
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys, kinds=("1", "2", "3", "none"))
    text = "\n".join(mods.check.read_f1(f1, identity_dir).lines)
    assert line_for(text, "dry run").startswith("PASS")  # 3 usable of 4
    (identity_dir / "gallery.sealed").chmod(0o644)
    (identity_dir / "inbox" / "stray.jpg").write_text("x")
    text = "\n".join(mods.check.read_f1(f1, identity_dir).lines)
    assert line_for(text, "private storage").startswith("FAIL")
