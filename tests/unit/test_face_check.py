"""The V2-25 device check's readings (face_check.py) and the F2 account helper (face_account.py).

F2's evidence is produced by the real runtime: DemoRuntime, EdgeCore and a real FaceWorker run a scripted ten
minutes on a fake clock; the identity records come from drain_identity_records(), the status lines from the CLI's
status_line(), the captures from the status page's build_status(), the startup's face fields from assemble() and its
face release from post_load_release(), and the guard summary from step5_guard.read_run_output(). Only the camera,
the detector and DeepFace are fakes, with synthetic embeddings. F1's evidence comes from the real `sentinel identity`
commands with a real sealed gallery. So a change in the runtime's output schema fails these tests (unlike session
47's hand-written layers value).

The privacy counts come from face_counts.sh's functions, run by bash as the package's blocks run them (only the
interpreter differs), on that evidence: initials and short names against the evidence's own schema text, injected
leaks, malformed evidence and unexpected fields. The offline F1 recheck, the maintainer's acceptance and the F1 gate
for F2 are exercised on a run directory built as F1f and FH left face-20261007T215900Z.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
import os
import shutil
import string
import subprocess
import sys
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
REPO = RUNNERS.parents[1]
# The synthetic operator's terms: what FN, F1c/F2p, F1b and F2b would hold in the shell.
TERMS = {"N1": "Jane", "N2": "D", "SENTINEL_IDENTITY_PASSPHRASE": "a long synthetic passphrase",
         "SENTINEL_RTSP_URL": "rtsp://cam-user:cam-secret-9@camera.invalid:554/sub"}
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


def bash_counts(function: str, *args: str, env: dict[str, str] | None = None, photos: tuple[str, ...] = (),
                prelude: str = "") -> subprocess.CompletedProcess:
    """One face_counts.sh function, run as the package's blocks run it: bash sources the file from the repository
    root and the terms are shell variables. Only the interpreter differs (FACE_PY: this test's Python)."""
    script = ('source benchmarks/runner/face_counts.sh; FACE_PY=("$TEST_PY"); '
              '[ -z "${TEST_PHOTOS-}" ] || mapfile -t PHOTO_NAMES <<< "$TEST_PHOTOS"; ' + prelude +
              f' set -uo pipefail; {function} "$@"')
    environment = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/"), "TEST_PY": sys.executable,
                   **({"TEST_PHOTOS": "\n".join(photos)} if photos else {}), **(TERMS if env is None else env)}
    return subprocess.run(["bash", "-c", script, "bash", *args], cwd=REPO, env=environment, capture_output=True,
                          text=True, timeout=120)


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(check=importlib.import_module("face_check"), account=importlib.import_module("face_account"),
                           guard=importlib.import_module("step5_guard"), evidence=importlib.import_module("face_evidence"),
                           privacy=importlib.import_module("face_privacy"), recheck=importlib.import_module("face_recheck"))


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


F2_PROVENANCE = ("f2_launch_utc=2026-10-08T09:00:00Z\nboot_id=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0\n"
                 "commit=95ce253c7b0d2784b6fecc9380bc10492905f252\npin_commit=95ce253c7b0d2784b6fecc9380bc10492905f252\n")


def finish(mods, out: Path, *, cue_delay_s: float = 0.1, account: dict[str, str] | None = ACCOUNT,
           counts: dict[str, int] | None = None) -> None:
    """result.json as the guard writes it, then the account and the counts, in the package's order (the counts by
    face_counts.sh unless ``counts`` fakes them)."""
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
    (out / "provenance.txt").write_text(F2_PROVENANCE)
    if account is not None:
        answers = iter([account[f.key] for f in mods.account.FIELDS] + ["SAVE"])
        assert mods.account.enter(out, ask=lambda prompt: next(answers), say=lambda text: None) == 0
    if counts is None:  # F2e's counts, by the real workflow
        for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                               ("face_counts_f2", "secret-counts-identity.txt")):
            done = bash_counts(function, str(out))
            assert done.returncode == 0, done.stderr
            (out / name).write_text(done.stdout)
        return
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in mods.check.NOT_COUNTED)
    (out / "secret-counts-camera.txt").write_text("".join(f"{n} userinfo={counts.get('userinfo', 0)}\n" for n in names))
    (out / "secret-counts-identity.txt").write_text("".join(
        f"{n} passphrase={counts.get('passphrase', 0)}\n{n} name={counts.get('name', 0)}\n"
        f"{n} nonconforming={counts.get('nonconforming', 0)}\n" for n in names))


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
    assert line_for(text, "passphrase, name and nonconforming counts").startswith("FAIL")
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


PHOTOS = tuple(f"Jane Doe {n}.jpg" for n in range(8))


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
    folder = store.inbox / "face-20261008T090000Z"
    folder.mkdir(mode=0o700)
    for name, kind in zip(PHOTOS, kinds):
        (folder / name).write_text(kind)
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
    if break_step == "name_found":
        provenance.append("operator_note=Jane D.")  # a name where nothing may hold one
    (f1 / "provenance.txt").write_text("\n".join(provenance) + "\n")
    done = bash_counts("face_counts_f1", str(f1), str(identity_dir / "audit.jsonl"), photos=PHOTOS[:len(kinds)])
    assert done.returncode == 0, done.stderr
    (f1 / "secret-counts-identity.txt").write_text(done.stdout)
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


# ================================================================ privacy counts (face_counts.sh, face_privacy, face_evidence)

F1_FILES = ("dryrun.json", "enroll.json", "list.json", "provenance.txt", "audit.jsonl")
TRT = "[10/08/2026-09:00:01] [TRT] [I] Loaded engine size: 13 MiB"
TRT_W = ("[10/08/2026-09:00:01] [TRT] [W] Using an engine plan file across different models of devices is not "
         "recommended and is likely to affect performance or even cause errors.")
WARNING = ("/home/user/onvif_env/lib/python3.10/site-packages/torch/cuda/__init__.py:827: UserWarning: Can't "
           "initialize NVML")


def parse(stdout: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for line in stdout.splitlines():
        label, pair = line.rsplit(" ", 1)
        key, value = pair.split("=")
        out.setdefault(label, {})[key] = int(value)
    return out


def stdin_of(terms: dict[str, list[str]]) -> bytes:
    return b"".join(f"{kind}={value}\0".encode() for kind, values in terms.items() for value in values)


def counts(mods, command: str, directory: Path, terms: dict[str, list[str]], **kwargs) -> dict[str, dict[str, int]]:
    """face_privacy's own API, for sweeps; the shell workflow is exercised separately."""
    return parse("\n".join(mods.privacy.run(command, directory, stdin_of(terms), **kwargs)))


def explain(*args: str, terms: dict[str, list[str]]) -> list[str]:
    done = subprocess.run([sys.executable, str(RUNNERS / "face_privacy.py"), *args, "--explain"],
                          input=stdin_of(terms), capture_output=True, timeout=60, check=True)
    return done.stdout.decode().splitlines()


@pytest.fixture
def f1_evidence(mods, tmp_path, monkeypatch, capsys):
    """Two real enrollments: all photos accepted, and one rejected (its reason label is no_face)."""
    out = []
    for name, kinds in (("plain", ("1", "2", "3")), ("rejected", ("1", "2", "3", "none"))):
        root = tmp_path / name
        root.mkdir()
        out.append(run_f1(mods, root, monkeypatch, capsys, kinds=kinds))
    return out


@pytest.fixture
def f2_evidence(mods, tmp_path, startup):
    out = simulate(tmp_path, startup)
    (out / "run.err").write_text("\n".join((WARNING, '  warnings.warn("Can\'t initialize NVML")', TRT, TRT_W)) + "\n")
    (out / "guard.jsonl").write_text(json.dumps({"MemAvailable": 1, "MemFree": 2, "pressure_bytes": 3,
                                                 "pswpin": 0, "pswpout": 0, "t_mono": 1.5}) + "\n")
    finish(mods, out)
    return out


# ---------------------------------------------------------------- the defect and its correction, F1


def test_no_initial_or_short_name_matches_f1_schema_text(mods, f1_evidence) -> None:
    """Every letter (an initial) and two-letter names drawn from the schema's own words count 0 and every file fits
    its profile, on the real enrollment's evidence, while the old substring search flagged nearly every letter."""
    for f1, identity_dir in f1_evidence:
        audit = identity_dir / "audit.jsonl"
        for name in [*string.ascii_letters, "No", "Id", "Px", "Ed", "Al", "Dry"[:2]]:
            found = counts(mods, "f1", f1, {"name": ["Jane", name]}, audit=audit, keys=("name",))
            assert sorted(found) == sorted(F1_FILES)
            assert all(v == {"name": 0, "nonconforming": 0} for v in found.values()), (name, found)
        texts = [p.read_text().casefold() for p in (*(f1 / n for n in F1_FILES[:4]), audit)]
        flagged = [c for c in string.ascii_lowercase if any(c in t for t in texts)]
        assert len(flagged) >= 20  # grep -ciF's view: an initial "leaked" almost whatever it was


def test_the_f1_workflow_counts_through_bash_and_validates(mods, f1_evidence) -> None:
    f1, identity_dir = f1_evidence[1]
    done = bash_counts("face_counts_f1", str(f1), str(identity_dir / "audit.jsonl"), photos=PHOTOS[:4],
                       env={**TERMS, "N2": "K"})
    assert done.returncode == 0 and done.stderr == ""
    found = parse(done.stdout)
    assert found == {n: {"passphrase": 0, "name": 0, "filenames": 0, "nonconforming": 0} for n in F1_FILES}
    assert all(TERMS[k] not in done.stdout for k in ("N1", "SENTINEL_IDENTITY_PASSPHRASE")) and "Jane" not in done.stdout
    (f1 / "secret-counts-identity.txt").write_text(done.stdout)
    assert mods.check.read_f1(f1, identity_dir).ok


def _copy(f1: Path, identity_dir: Path, tmp: Path) -> tuple[Path, Path]:
    f1c, identity = tmp / "f1", tmp / "identity"
    shutil.copytree(f1, f1c)
    identity.mkdir()
    shutil.copy2(identity_dir / "audit.jsonl", identity / "audit.jsonl")
    return f1c, identity / "audit.jsonl"


def _edit_json(path: Path, change) -> None:
    item = json.loads(path.read_text())
    change(item)
    path.write_text(json.dumps(item))


LEAKS = {  # what an initial "D" (or the long name, passphrase, file name) looks like where it does not belong
    "a value holding the initial": ("enroll.json", lambda p: _edit_json(p, lambda d: d["results"][0].update(
        reason="D"))),
    "an unexpected key": ("dryrun.json", lambda p: _edit_json(p, lambda d: d.update({"D": 1}))),
    "a file-name-like value": ("provenance.txt", lambda p: p.write_text(
        p.read_text().replace("inbox_folder_name=face-20261008T090000Z", "inbox_folder_name=D_selfies"))),
    "an extra file": ("notes.txt", lambda p: p.write_text("photos by D.\n")),
    "a duplicated key": ("list.json", lambda p: p.write_text(p.read_text().replace('{"identity": "list"',
                                                                                   '{"identity": "D", "identity": "list"'))),
    "an appended audit line": ("audit.jsonl", lambda p: p.write_text(
        p.read_text() + '{"action":"enroll","identity_id":"D","utc":"2026-10-08T09:00:00+00:00"}\n')),
}


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_an_initial_out_of_place_and_unexpected_content_are_counted(mods, f1_evidence, tmp_path, leak) -> None:
    f1, identity_dir = f1_evidence[0]
    f1c, audit = _copy(f1, identity_dir, tmp_path)
    name, change = LEAKS[leak]
    change(audit if name == "audit.jsonl" else f1c / name)
    done = bash_counts("face_counts_f1", str(f1c), str(audit), photos=PHOTOS[:3])
    assert done.returncode == 0, done.stderr
    found = parse(done.stdout)[name]
    assert found["name"] >= 1 and found["nonconforming"] >= 1, found
    where = explain("f1", str(f1c), "--audit", str(audit), "--keys", "name", terms={"name": ["Jane", "D"]})
    assert any(x.startswith(f"{name} name: ") for x in where) and not any('"D"' in x or "Jane" in x for x in where)
    (f1c / "secret-counts-identity.txt").write_text(done.stdout)
    text = "\n".join(mods.check.read_f1(f1c, audit.parent).lines)
    assert next(x for x in text.splitlines() if "photo-file-name counts" in x).startswith("FAIL")


def test_longer_names_passphrases_and_file_names_keep_their_byte_searches(mods, f1_evidence, tmp_path) -> None:
    f1, identity_dir = f1_evidence[0]
    f1c, audit = _copy(f1, identity_dir, tmp_path)
    secret = 'pass"phrase-with-a-quote'
    raw = (f1c / "enroll.json").read_text()[:-1]
    (f1c / "enroll.json").write_text(raw[:-1] + ', "x1": "JaneDoe", "x2": "Zo\\u00eb", "x3": '
                                     + json.dumps(secret) + ', "x4": "Jane Doe 1.jpg"}\n')
    found = counts(mods, "f1", f1c, {"passphrase": [secret], "name": ["Jane", "Zoë"], "filename": list(PHOTOS[:3])},
                   audit=audit)["enroll.json"]
    assert found["name"] >= 2  # a substring of a value, and a non-ASCII name only visible decoded
    assert found["passphrase"] == 1  # only visible decoded: its quote is escaped in the bytes
    assert found["filenames"] == 1 and found["nonconforming"] == 4
    quality = counts(mods, "f1", f1, {"name": ["Ali"]}, audit=audit, keys=("name",))  # "Ali" is inside "quality"
    assert quality["dryrun.json"]["name"] >= 1  # unchanged: a longer name still fails closed on such a collision


MALFORMED = {
    "truncated JSON": lambda p: p.write_text(p.read_text()[:40]),
    "a wrong type": lambda p: _edit_json(p, lambda d: d.update(photos=str(d["photos"]))),
    "a value outside its vocabulary": lambda p: _edit_json(p, lambda d: d["results"][0].update(result="maybe")),
    "a root that is not an object": lambda p: p.write_text("[1, 2]\n"),
    "bytes that are not UTF-8": lambda p: p.write_bytes(p.read_bytes() + b"\xff\xfe"),
}


@pytest.mark.parametrize("kind", sorted(MALFORMED))
def test_malformed_evidence_fails_closed(mods, f1_evidence, tmp_path, kind) -> None:
    f1, identity_dir = f1_evidence[0]
    f1c, audit = _copy(f1, identity_dir, tmp_path)
    MALFORMED[kind](f1c / "dryrun.json")
    done = bash_counts("face_counts_f1", str(f1c), str(audit), photos=PHOTOS[:3])
    assert done.returncode == 0 and parse(done.stdout)["dryrun.json"]["nonconforming"] >= 1
    (f1c / "secret-counts-identity.txt").write_text(done.stdout)
    assert not mods.check.read_f1(f1c, audit.parent).ok


@pytest.mark.parametrize(("env", "photos", "stop"), [
    ({k: v for k, v in TERMS.items() if k != "SENTINEL_IDENTITY_PASSPHRASE"}, PHOTOS[:3], "no passphrase"),
    ({**TERMS, "N1": "", "N2": ""}, PHOTOS[:3], "no name"),
    (TERMS, (), "no filename"),
    ({**TERMS, "N1": "1984"}, PHOTOS[:3], "a name without a letter"),
])
def test_missing_terms_refuse_with_no_counts(f1_evidence, env, photos, stop) -> None:
    f1, identity_dir = f1_evidence[0]
    done = bash_counts("face_counts_f1", str(f1), str(identity_dir / "audit.jsonl"), env=env, photos=photos)
    assert (done.returncode, done.stdout) == (2, "") and stop in done.stderr


@pytest.mark.parametrize(("stdin", "stop"), [
    (b"passphrase=a\0passphrase=b\0name=Jane\0filename=x\0", "more than one passphrase"),
    (b"password=a\0name=Jane\0filename=x\0", "unknown kind"),
    (b"passphrase=a\0name=Jane\0filename=x", "NUL-terminated"),
    (b"passphrase=\xff\0name=Jane\0filename=x\0", "not UTF-8"),
    (b"passphrase=a\0name=Jane\0filename=x\0url=rtsp://h/x\0", "no userinfo"),
    (b"passphrase=a\0name=Jane\0filename=x\0url=rtsp://u:p@h/x\0", "does not use"),
])
def test_malformed_terms_refuse(mods, f1_evidence, stdin, stop) -> None:
    f1, identity_dir = f1_evidence[0]
    with pytest.raises(mods.privacy.Refused, match=stop):
        mods.privacy.run("f1", f1, stdin, audit=identity_dir / "audit.jsonl")


# ---------------------------------------------------------------- F2 and FH


def test_f2_schema_text_never_counts_and_free_text_always_can(mods, f2_evidence) -> None:
    files = sorted(p.name for p in f2_evidence.iterdir() if p.name not in mods.evidence.NOT_COUNTED)
    assert {"run.jsonl", "run.err", "result.json", "guard.jsonl", "account.txt", "provenance.txt",
            "status-e1.json"} <= set(files)
    for letter in string.ascii_lowercase:
        found = counts(mods, "f2", f2_evidence, {"name": ["Jane", letter]}, keys=("name",))
        assert sorted(found) == files and all(v["nonconforming"] == 0 for v in found.values()), found
        hits = {name: v["name"] for name, v in found.items() if v["name"]}
        if letter != "a":
            assert hits == {}, (letter, hits)  # TRT's [I] and [W], "Can't", every key, label and template: schema
    where = {x.split(": ", 1)[1] for x in explain("f2", str(f2_evidence), "--keys", "name",
                                                   terms={"name": ["Jane", "a"]}) if " name: " in x}
    # "A synthetic report." (the scene model's summary) and the account's notes are free text: content
    assert where == {"live.scene_report.summary (content)", "line 12: notes (content)"}, where


def test_the_f2_workflow_counts_through_bash_and_validates(mods, f2_evidence, f1_list) -> None:
    files = {p.name for p in f2_evidence.iterdir() if p.name not in mods.evidence.NOT_COUNTED}
    for name, keys in (("secret-counts-camera.txt", {"userinfo"}),
                       ("secret-counts-identity.txt", {"passphrase", "name", "nonconforming"})):
        found = parse((f2_evidence / name).read_text())
        assert set(found) == files and all(set(v) == keys and not any(v.values()) for v in found.values()), found
    result = mods.check.read_f2(f2_evidence, f1_list)
    assert result.ok, "\n".join(result.lines)


F2_LEAKS = {
    "an unexpected status field": ("status-k1.json", lambda p: _edit_json(p, lambda d: d["live"].update(operator="D"))),
    "a non-JSON stdout line": ("run.jsonl", lambda p: p.write_text(p.read_text() + "hello D\n")),
    "an unexpected capture name": ("result.json", lambda p: _edit_json(
        p, lambda d: d["captures"][0].update(name="D"))),
    "an extra file": ("extra.txt", lambda p: p.write_text("D\n")),
    "a provenance line": ("provenance.txt", lambda p: p.write_text(p.read_text() + "operator=D\n")),
}


@pytest.mark.parametrize("leak", sorted(F2_LEAKS))
def test_f2_unexpected_fields_and_lines_are_counted(mods, f2_evidence, f1_list, leak) -> None:
    name, change = F2_LEAKS[leak]
    change(f2_evidence / name)
    done = bash_counts("face_counts_f2", str(f2_evidence))
    found = parse(done.stdout)[name]
    assert found["name"] >= 1 and found["nonconforming"] >= 1, found
    (f2_evidence / "secret-counts-identity.txt").write_text(done.stdout)
    camera = bash_counts("face_counts_camera", str(f2_evidence))
    (f2_evidence / "secret-counts-camera.txt").write_text(camera.stdout)
    text = "\n".join(mods.check.read_f2(f2_evidence, f1_list).lines)
    assert next(x for x in text.splitlines() if "passphrase, name and nonconforming counts" in x).startswith("FAIL")


def test_f2_free_text_and_secrets_in_logs_are_counted(f2_evidence) -> None:
    lines = (f2_evidence / "account.txt").read_text().replace("notes=a note that is never printed", "notes=asked D")
    (f2_evidence / "account.txt").write_text(lines)
    with (f2_evidence / "run.err").open("a") as err:
        err.write(f"open failed: {TERMS['SENTINEL_RTSP_URL']}\n[10/08/2026-09:00:02] [TRT] [E] tensor named D\n")
    identity = parse(bash_counts("face_counts_f2", str(f2_evidence)).stdout)
    assert identity["account.txt"] == {"passphrase": 0, "name": 1, "nonconforming": 0}  # notes: content, not structure
    assert identity["run.err"] == {"passphrase": 0, "name": 1, "nonconforming": 0}  # a log message: content
    camera = parse(bash_counts("face_counts_camera", str(f2_evidence)).stdout)
    assert camera["run.err"] == {"userinfo": 1} and sum(v["userinfo"] for v in camera.values()) == 1


def test_fh_counts_the_session_files_and_needs_a_url(tmp_path) -> None:
    session = tmp_path / "session"
    (session / "f1").mkdir(parents=True)
    (session / "face.yaml").write_text("camera: {id: cam-1}\n")
    (session / "validate.txt").write_text("ok\n")
    (session / "camera-probe.json").write_text(json.dumps({"note": "cam-user:cam-secret-9"}))
    (session / "f1" / "dryrun.json").write_text("{}")
    (session / "secret-counts-top.txt").write_text("")
    done = bash_counts("face_counts_top", str(session))
    assert parse(done.stdout) == {"camera-probe.json": {"userinfo": 1}, "face.yaml": {"userinfo": 0},
                                  "validate.txt": {"userinfo": 0}, "f1/dryrun.json": {"userinfo": 0}}
    assert bash_counts("face_counts_top", str(session), env={k: v for k, v in TERMS.items()
                                                             if k != "SENTINEL_RTSP_URL"}).stdout == ""


def test_terms_reach_the_counter_on_stdin_only(f1_evidence, f2_evidence, tmp_path) -> None:
    """No process gets a term as an argument (FH included); the counter gets them on stdin."""
    recorder = tmp_path / "recorder.py"
    record = tmp_path / "record.jsonl"
    recorder.write_text("import json, sys\n"
                        f"open({str(record)!r}, 'a').write(json.dumps({{'argv': sys.argv, "
                        "'stdin': sys.stdin.buffer.read().decode()}) + '\\n')\n")
    f1, identity_dir = f1_evidence[0]
    session = tmp_path / "session"
    session.mkdir()
    for function, args in (("face_counts_f1", (str(f1), str(identity_dir / "audit.jsonl"))),
                           ("face_counts_f2", (str(f2_evidence),)), ("face_counts_camera", (str(f2_evidence),)),
                           ("face_counts_top", (str(session),)),
                           ("face_recheck_f1", (str(f1), str(identity_dir), str(tmp_path / "r"), "0" * 40))):
        done = bash_counts(function, *args, photos=PHOTOS[:3], prelude=f'FACE_PY=("$TEST_PY" {recorder});')
        assert done.returncode == 0, done.stderr
    calls = [json.loads(x) for x in record.read_text().splitlines()]
    assert len(calls) == 5
    secrets = [TERMS["N1"], TERMS["SENTINEL_IDENTITY_PASSPHRASE"], "cam-secret-9", *PHOTOS[:3]]
    for call in calls:
        assert not any(s in arg for s in secrets for arg in call["argv"]), call["argv"]
    assert "passphrase=a long synthetic passphrase\0name=Jane\0name=D\0filename=Jane Doe 0.jpg\0" in calls[0]["stdin"]
    assert calls[3]["stdin"] == f"url={TERMS['SENTINEL_RTSP_URL']}\0" and calls[4]["stdin"] == "name=Jane\0name=D\0"


# ---------------------------------------------------------------- the profiles follow the pinned code


def test_the_vocabularies_are_the_codes(mods) -> None:
    from operator_check import MEMORY_KEYS

    from sentinel.demo_runtime import MAX_PENDING_SIGNALS, PENDING_DURABILITY, RuntimeState
    from sentinel.identity import state
    from sentinel.identity.association import Ownership
    from sentinel.identity.worker import TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING
    from sentinel.incidents.service import IncidentStatus
    from sentinel.live_state import Capability, Occupancy, SceneStatus
    from sentinel.media.capture import CaptureState
    from sentinel.media.health import VideoState
    from sentinel.memory_policy import CANDIDATE_POLICY, DEFAULT_POLICY, RELEASE_FILE_ROLES, RELEASE_RESULTS
    from sentinel.rules.scene_hazard import Severity
    from sentinel.scene.report import Threat, Uncertainty
    from sentinel.scene.server import ServerState

    fe = mods.evidence
    assert set(fe.MEMORY_KEYS) == set(MEMORY_KEYS)
    for vocab, enum in ((fe.CAPABILITY, Capability), (fe.OCCUPANCY, Occupancy), (fe.SCENE_STATUS, SceneStatus),
                        (fe.VIDEO, VideoState), (fe.CAPTURE, CaptureState), (fe.SERVER, ServerState),
                        (fe.RUNTIME, RuntimeState), (fe.BASIS, state.Basis), (fe.SEVERITY, Severity),
                        (fe.INCIDENT_STATUS, IncidentStatus), (fe.THREAT, Threat), (fe.UNCERTAINTY, Uncertainty)):
        assert vocab.values == {e.value for e in enum}, enum
    assert set(fe.IDENTITY_STATES) == {e.value for e in state.IdentityState}
    assert set(fe.OWNERSHIP) == {e.value for e in Ownership}
    assert set(fe.VOTE_NOTES) == set(state._NOTE_LABELS) and set(fe.VOTE_LABELS) == set(state._NOTE_LABELS.values())
    assert fe.TICKS.values == {TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING}
    assert fe.RELEASE_RESULTS.values == set(RELEASE_RESULTS)
    assert fe.RELEASE_COMPONENTS.values == set(RELEASE_FILE_ROLES)
    assert fe.RELEASE_ROLES.values == {r for roles in RELEASE_FILE_ROLES.values() for r in roles}
    assert {DEFAULT_POLICY.thp, CANDIDATE_POLICY.thp} == fe.THP.values
    assert {DEFAULT_POLICY.model_file_release, CANDIDATE_POLICY.model_file_release} == fe.RELEASE.values
    assert fe.PENDING_DURABILITY.fit(PENDING_DURABILITY) == (True, [str(MAX_PENDING_SIGNALS)])


def test_the_runtimes_fixed_phrases_are_templates_and_their_variable_parts_content(mods) -> None:
    from sentinel.demo_runtime import degradation

    snapshot = {"runtime": {"state": "running"}, "live": {"video": "stale"},
                "components": {"capture": {"ready": False, "state": "waiting", "problem": "read_timeout"},
                               "detector": {"state": "unavailable", "problem": "engine_missing"},
                               "scene": {"state": "unavailable", "problem": "server_failed"},
                               "face": {"state": "unavailable", "problem": "gallery_locked"},
                               "incidents": {"pending_signals": 2, "problem": "database_locked", "signals_dropped": 1},
                               "notifications": {"channels": {"telegram": {"state": "unavailable", "problem": "http_5"}},
                                                 "worker": {"state": "stopped", "problem": "lease"}}}}
    phrases = degradation(snapshot)
    assert len(phrases) == 10
    for phrase in phrases:
        assert mods.evidence.DEGRADED.fit(phrase)[0], phrase
    assert mods.evidence.DEGRADED.fit("capture waiting (read_timeout)") == (True, ["waiting", "read_timeout"])
    assert mods.evidence.TRANSITION_REASON.fit("2 consistent matches") == (True, ["2"])
    assert mods.evidence.TRANSITION_REASON.fit("matched by D") == (False, [])
    assert mods.evidence.INCIDENT_TITLE.fit("Person in restricted zone 'whole-view'") == (True, ["whole-view"])
    assert mods.evidence.OCCUPANCY_REASON.fit("nobody detected on fresh video") == (True, [])


# ================================================================ the offline F1 recheck, its acceptance, the F1 gate

PIN = "0123456789abcdef0123456789abcdef01234567"
FH_MANIFEST = "find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"


def seal(directory: Path) -> None:
    """The run's SHA256SUMS, by FH's own command line."""
    subprocess.run(["bash", "-c", FH_MANIFEST], cwd=directory, check=True)


def tree(*roots: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for root in roots
            for p in ([root] if root.is_file() else sorted(root.rglob("*"))) if p.is_file()}


@pytest.fixture
def face_run(mods, tmp_path, monkeypatch, capsys):
    """A run as F1f and FH left face-20261007T215900Z: the old substring counts flagged the family initial in every
    file (name=1, no nonconforming key), so F1_exit=1 and f1/check.txt says not validated; then its SHA256SUMS."""
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys)
    run = tmp_path / "runs" / "face-20261008T090000Z"
    run.mkdir(parents=True)
    shutil.move(f1, run / "f1")
    files = sorted(mods.check._f1_files(run / "f1"))
    (run / "f1" / "secret-counts-identity.txt").write_text(
        "".join(f"{n} passphrase=0\n{n} name=1\n{n} filenames=0\n" for n in files))
    original = mods.check.read_f1(run / "f1", identity_dir)
    assert not original.ok
    (run / "f1" / "check.txt").write_text("\n".join([*original.lines, "face-f1: not validated"]) + "\n")
    (run / "face.yaml").write_text("config_version: 1\ncamera: {id: cam-1}\n")
    (run / "validate.txt").write_text("config ok\n")
    (run / "secret-counts-top.txt").write_text("")
    (run / "provenance.txt").write_text(f"pin_commit={PIN}\nFB_exit=0\nF1_exit=1\n")
    seal(run)
    return SimpleNamespace(run=run, identity=identity_dir, root=tmp_path / "recheck")


def recheck(face_run, name: str = "20261008T100000Z", env: dict[str, str] | None = None):
    out = face_run.root / f"{face_run.run.name}-f1-{PIN[:7]}-{name}"
    return out, bash_counts("face_recheck_f1", str(face_run.run), str(face_run.identity), str(out), PIN, env=env)


def accept(mods, face_run, out: Path, typed: str | None = None) -> tuple[int, str]:
    said: list[str] = []
    identity = mods.check._enrolled_id(face_run.run / "f1")
    line = typed if typed is not None else f"ACCEPT {face_run.run.name} {identity}"
    code = mods.recheck.accept(out, face_run.run, face_run.identity, ask=io.StringIO(line + "\n"), say=said.append)
    return code, "\n".join(said)


def test_the_offline_recheck_validates_and_changes_nothing_of_the_run(mods, face_run) -> None:
    before = tree(face_run.run, face_run.identity / "audit.jsonl")
    out, done = recheck(face_run)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "face-f1-recheck: validated" in done.stdout and "Jane" not in done.stdout
    assert tree(face_run.run, face_run.identity / "audit.jsonl") == before  # the run and the audit log unchanged
    assert sorted(p.name for p in out.iterdir()) == ["README.txt", "SHA256SUMS", "check.txt", "provenance.txt",
                                                      "secret-counts-names.txt"]
    assert oct(out.stat().st_mode & 0o777) == "0o700" and all(oct(p.stat().st_mode & 0o777) == "0o600"
                                                                for p in out.iterdir())
    provenance = mods.check._key_values(out / "provenance.txt")
    assert provenance["names"] == "recounted" and provenance["passphrase_counts"] == "carried_forward"
    assert provenance["filename_counts"] == "carried_forward" and provenance["carried_source_in_run_manifest"] == "yes"
    assert provenance["run_manifest_check"] == "verified" and provenance["pin_commit"] == PIN
    check = (out / "check.txt").read_text()
    carried = next(x for x in check.splitlines() if "carried forward" in x)
    assert carried.startswith("PASS") and '"rescanned": false' in carried
    assert '"F1_exit": "1"' in check and "F1_exit=1" in (out / "README.txt").read_text()
    assert not any("Jane" in p.read_text() for p in out.iterdir())
    assert mods.check._manifest(out) == ({p: hashlib.sha256((out / p).read_bytes()).hexdigest()
                                          for p in sorted(x.name for x in out.iterdir()) if p != "SHA256SUMS"}, [])


def _changed_run_file(face_run) -> None:
    with (face_run.run / "face.yaml").open("a") as handle:
        handle.write("# edited after the run\n")


def _added_run_file(face_run) -> None:
    (face_run.run / "f1" / "extra.json").write_text("{}")


def _carried_passphrase(face_run) -> None:
    counts = face_run.run / "f1" / "secret-counts-identity.txt"
    counts.write_text(counts.read_text().replace("list.json passphrase=0", "list.json passphrase=1"))
    seal(face_run.run)


def _original_item_differs(face_run) -> None:
    check = face_run.run / "f1" / "check.txt"
    check.write_text(check.read_text().replace("PASS each identity command", "FAIL each identity command", 1))
    seal(face_run.run)


@pytest.mark.parametrize(("change", "item"), [
    (_changed_run_file, "the run's evidence unchanged"),
    (_added_run_file, "the run's evidence unchanged"),
    (_carried_passphrase, "passphrase and photo-file-name counts carried forward"),
    (_original_item_differs, "every original F1 item other than the privacy counts"),
    (None, "names counted again"),  # a first name that collides with schema text: fails closed
])
def test_the_recheck_fails_closed(mods, face_run, change, item) -> None:
    if change is not None:
        change(face_run)
    out, done = recheck(face_run, env={**TERMS, "N1": "Ali"} if change is None else None)
    assert done.returncode == 1 and "face-f1-recheck: not validated" in done.stdout
    assert next(x for x in (out / "check.txt").read_text().splitlines() if item in x).startswith("FAIL")
    assert accept(mods, face_run, out)[0] == 1 and not (out / "acceptance.txt").exists()


def test_acceptance_needs_the_exact_typed_line_and_is_written_once(mods, face_run) -> None:
    out, done = recheck(face_run)
    assert done.returncode == 0
    code, said = accept(mods, face_run, out, typed="ACCEPT")
    assert code == 1 and "nothing was written" in said and not (out / "acceptance.txt").exists()
    code, said = accept(mods, face_run, out)
    assert code == 0, said
    acceptance = out / "acceptance.txt"
    assert oct(acceptance.stat().st_mode & 0o777) == "0o600"
    fields = mods.check._key_values(acceptance)
    assert fields["run"] == face_run.run.name and fields["accepted_reading"] == "f1-recheck"
    assert fields["recheck_check_sha256"] == hashlib.sha256((out / "check.txt").read_bytes()).hexdigest()
    assert accept(mods, face_run, out)[0] == 1  # never overwritten


def test_the_gate_lets_f2_build_only_on_an_accepted_recheck_that_still_matches(mods, face_run, tmp_path) -> None:
    gate = mods.check.f1_gate
    assert gate(face_run.run, face_run.identity, face_run.root) == (False, ["f1_gate=refused",
                                                                         "f1_gate_reason=no_accepted_recheck"])
    out, _ = recheck(face_run)
    assert gate(face_run.run, face_run.identity, face_run.root)[1][-1] == "f1_gate_reason=no_accepted_recheck"
    assert accept(mods, face_run, out)[0] == 0
    ok, lines = gate(face_run.run, face_run.identity, face_run.root)
    assert ok and lines[:3] == ["f1_gate=passed", f"run={face_run.run.name}", "f1_basis=accepted_recheck"]
    provenance = tmp_path / "provenance.txt"  # F2c writes these lines into F2's provenance, which is counted
    provenance.write_text(F2_PROVENANCE + "\n".join(lines) + "\n")
    assert mods.evidence.read(provenance, "f2").nonconforming == []
    cli = subprocess.run([sys.executable, str(RUNNERS / "face_check.py"), "gate", str(face_run.run), "--identity-dir",
                          str(face_run.identity), "--recheck-root", str(face_run.root)], capture_output=True, text=True)
    assert cli.returncode == 0 and cli.stdout.splitlines() == lines

    def refused(reason: str) -> None:
        assert gate(face_run.run, face_run.identity, face_run.root) == (False, ["f1_gate=refused",
                                                                             f"f1_gate_reason={reason}"])

    audit = face_run.identity / "audit.jsonl"
    saved = audit.read_bytes()
    audit.write_bytes(saved + b'{"action":"revoke","identities_left":0,"identity_id":"idn-000000000001",'
                              b'"utc":"2026-10-08T11:00:00+00:00"}\n')
    refused("recheck_no_longer_validates")  # the identity's audit log changed after the acceptance
    audit.write_bytes(saved)
    check = out / "check.txt"
    kept = check.read_bytes()
    check.write_bytes(kept + b"\n")
    refused("recheck_changed")
    check.write_bytes(kept)
    acceptance = out / "acceptance.txt"
    text = acceptance.read_text()
    acceptance.write_text(text.replace("confirmation=ACCEPT", "confirmation=accept"))
    refused("acceptance_does_not_match")
    acceptance.write_text(text)
    other, _ = recheck(face_run, name="20261008T110000Z")
    shutil.copy2(acceptance, other / "acceptance.txt")
    refused("several_accepted_rechecks")
    shutil.rmtree(other)
    (face_run.identity / "gallery.sealed").rename(face_run.identity / "gallery.moved")
    refused("gallery_missing")
    (face_run.identity / "gallery.moved").rename(face_run.identity / "gallery.sealed")
    _changed_run_file(face_run)
    refused("run_evidence_changed")


def test_the_gate_accepts_a_run_whose_own_f1_validated(mods, face_run) -> None:
    f1 = face_run.run / "f1"
    done = bash_counts("face_counts_f1", str(f1), str(face_run.identity / "audit.jsonl"), photos=PHOTOS[:3])
    (f1 / "secret-counts-identity.txt").write_text(done.stdout)
    reading = mods.check.read_f1(f1, face_run.identity)
    assert reading.ok
    (f1 / "check.txt").write_text("\n".join([*reading.lines, "face-f1: validated"]) + "\n")
    (face_run.run / "provenance.txt").write_text(f"pin_commit={PIN}\nFB_exit=0\nF1_exit=0\n")
    seal(face_run.run)
    ok, lines = mods.check.f1_gate(face_run.run, face_run.identity, None)
    assert ok and "f1_basis=original" in lines


def test_the_f2_reading_requires_the_f1_basis_when_given(mods, face_run, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out)
    gate = (face_run.run, face_run.identity, face_run.root)
    first = mods.check.read_f2(out, face_run.run / "f1", gate).lines[0]
    assert first.startswith("FAIL F1 basis for this F2") and "no_accepted_recheck" in first
    recheck_dir, _ = recheck(face_run)
    assert accept(mods, face_run, recheck_dir)[0] == 0
    assert mods.check.read_f2(out, face_run.run / "f1", gate).lines[0].startswith("PASS F1 basis for this F2")
    assert mods.check.read_f2(out, f1_list, gate).lines[0].startswith("FAIL F1 basis")  # another run's F1
    bad = subprocess.run([sys.executable, str(RUNNERS / "face_check.py"), "f2", str(out), "--f1", str(f1_list),
                          "--run", str(face_run.run)], capture_output=True, text=True)
    assert bad.returncode == 2 and "--run and --identity-dir go together" in bad.stderr
