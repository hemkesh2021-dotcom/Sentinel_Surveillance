"""`sentinel run --face` (V2-25 demo form): admission, the sealed gallery, load order, the loop and the CLI's secret.

Fakes stand in for the camera, the detector and DeepFace; the profile and face admission used here are SYNTHETIC
fixtures passed explicitly (never the registry). One startup test opens a gallery sealed with the real helper.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel import demo_runtime
from sentinel.adapters import FACE_ADAPTER_ID, FaceAdmission, FaceAdmissionStatus, FileFacts
from sentinel.cli import main
from sentinel.config import parse_config
from sentinel.contracts import NormalizedBox, PixelFormat
from sentinel.demo_runtime import (
    DemoRuntime,
    Devices,
    FaceOptions,
    FaceRuntime,
    RunOptions,
    StartupRefused,
    assemble,
    file_facts,
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
from sentinel.identity.legacy_deepface import FaceBackendError, FaceRun, static_compatibility_fields
from sentinel.identity.vault import SECRET_ENV, Sealer, Secret, VaultError
from sentinel.identity.worker import FaceWorker
from sentinel.incidents.service import IncidentService
from sentinel.live_state import Capability
from sentinel.media.capture import CapturedFrame, CaptureState, CaptureStatus, LatestFrame, SourceError
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.memory_policy import CANDIDATE_POLICY
from sentinel.storage.database import Database
from sentinel.tracking.tracker import PersonTracker, RawTrack, TrackerError

SECRET = Secret("a long synthetic passphrase")
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
PERSON = [RawTrack(1, 100.0, 100.0, 200.0, 400.0, 0.9)]  # on a 640x480 frame
FACE_BOX = NormalizedBox(x1=0.19, y1=0.22, x2=0.28, y2=0.35)  # inside that person's head region


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


A = vec(1.0)
COMPAT = FaceCompatibility(**static_compatibility_fields(), opencv_version="4.13.0",
                           tensorflow_version="2.21.0", tf_keras_version="2.21.0", numpy_version="2.2.6")
GALLERY = GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
    identity_id="idn-00000000000a", enrolled_utc=NOW, consent_date="2026-10-07", consent_scope=CONSENT_SCOPE,
    prototypes=(A,)),))


class Image:
    shape = (480, 640, 3)

    def copy(self) -> "Image":
        return Image()


class Tracker:
    def __init__(self, calls: list[str] | None = None) -> None:
        self.script: list[list[RawTrack]] = []
        self.calls = calls if calls is not None else []

    def load(self) -> None:
        self.calls.append("tracker_load")

    def track(self, image: Any) -> list[RawTrack]:
        return self.script.pop(0) if self.script else []

    def reset(self) -> None:
        pass


def analyze_with(embedding):
    def analyze(frame, image):
        return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=embedding),), 1, 0)
    return analyze


# ---------------------------------------------------------------- the loop


class Capture:
    def __init__(self, clock: FakeClock, slot: LatestFrame) -> None:
        self.clock, self.slot = clock, slot
        self.stamper = FrameStamper("cam-1", clock)
        self.stamper.connect()

    @property
    def connected(self):
        return self.stamper.current_stream

    def status(self) -> CaptureStatus:
        return CaptureStatus(CaptureState.STREAMING, self.connected, 0, 1, 0, 0, None, None)

    def start(self) -> None:
        pass

    def stop(self, timeout_s: float) -> bool:
        return True


class Loop:
    def __init__(self, tmp_path: Path, *, analyze=analyze_with(vec(0.97, 0.1, 0.0, 0.05)), face_problem=None,
                 with_face: bool = True, validation: bool = True, comparator=None,
                 detection: str = "deepface", shadow=None, capture=None, continuity=None,
                 feature_controls: bool = False, perf_counter=None) -> None:
        self.clock = FakeClock(utc=NOW)
        self.config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
        self.slot = LatestFrame()
        self.capture = Capture(self.clock, self.slot)
        self.tracker = Tracker()
        self.db = Database.open(tmp_path / "sentinel.db")
        timing = {} if perf_counter is None else {"perf_counter": perf_counter}  # the worker's own timer by default
        self.worker = FaceWorker(analyze, self.clock, comparator=comparator, **timing)
        face = (FaceRuntime(self.worker, 1, validation, comparator is not None, detection, shadow, capture, continuity)
                if with_face else None)
        self.runtime = DemoRuntime(
            self.config, self.clock, capture=self.capture, slot=self.slot,
            incidents=IncidentService(self.db, self.clock, incidents=self.config.incidents,
                                      notifications=self.config.notifications),
            tracker=PersonTracker(self.tracker), face=face, face_problem=face_problem,
            enrollment=GALLERY.enrollment(), status_refresh_s=0.0, feature_controls=feature_controls)
        self.runtime.start()

    def frame(self, persons=PERSON) -> None:
        self.tracker.script.append(list(persons))
        self.clock.advance(1 / 15)
        frame = self.capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.slot.publish(CapturedFrame(frame, Image()))
        self.runtime.step(timeout_s=0)
        deadline = time.monotonic() + 5
        while self.worker.status()["busy"] or self.worker.status()["waiting"]:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        self.runtime.step(timeout_s=0)  # takes the face result, if any

    def close(self) -> dict:
        result = self.runtime.shutdown()
        self.db.close()
        return result


def test_the_loop_samples_live_frames_and_reports_opaque_identity_transitions(tmp_path) -> None:
    loop = Loop(tmp_path)
    for _ in range(40):  # about 2.7 s at 15 fps: three face ticks
        loop.frame()
    records = loop.runtime.drain_identity_records()
    transitions = [r for r in records if r["identity"] == "transition"]
    assert [(t["from"], t["to"], t["identity_id"], t["basis"]) for t in transitions] == [
        ("unresolved", "known", "idn-00000000000a", "fresh")]
    ticks = [r for r in records if r["identity"] == "tick"]
    results = [r for r in records if r["identity"] == "result"]
    assert [(r["tick"], r["tracks"]) for r in ticks] == [("offered", [1])] * 3
    assert [r["outcome"] for r in results] == ["applied"] * 3
    assert [r["frame_mono_ns"] for r in results] == [r["frame_mono_ns"] for r in ticks]  # votes at their frame time
    assert all(r["persons"][0]["vote"] == "match" and r["processing_ms"] is not None for r in results)
    assert records.index(transitions[0]) > records.index(results[1])  # known only after the second match
    snapshot = loop.runtime.snapshot()
    face = snapshot["components"]["face"]
    assert face["state"] == "available" and face["identities_enrolled"] == 1 and face["validation_run"] is True
    assert (face["worker"]["counters"]["ticks"], face["worker"]["counters"]["offered"]) == (3, 3)
    assert face["results"]["results_applied"] == 3 and face["results"]["votes_match"] == 3
    assert snapshot["live"]["identity"]["known"] == 1 and snapshot["live"]["identity"]["fresh"] == 1
    text = json.dumps(snapshot)
    assert "embedding" not in text and "0.97" not in text and "idn-00000000000a" not in text  # counts only
    assert loop.runtime.drain_identity_records() == []
    assert loop.close()["stopped"]["face_worker"] is True


@pytest.mark.parametrize(("validation", "schema"), [(True, 3), (False, 1)])
def test_only_a_validation_runs_records_carry_diagnostic_numbers(tmp_path, validation, schema) -> None:
    """D84: a --face-validation run's identity records are schema 2, each result person with its numbers (the person's
    box and confidence, its face's size and score, its similarity); any other face run keeps schema 1 unchanged."""
    loop = Loop(tmp_path, validation=validation)
    for _ in range(20):  # two face ticks: a known identity, so a transition too
        loop.frame()
    records = loop.runtime.drain_identity_records()
    assert {r["identity"] for r in records} == {"tick", "result", "transition"}
    assert {r["schema"] for r in records} == {schema}
    result = next(r for r in records if r["identity"] == "result")
    person = result["persons"][0]
    assert result.get("full_scale", "absent") == (None if validation else "absent")  # D86: none from this fake
    plain = {"track_id": 1, "ownership": "assigned", "vote": "match", "identity_id": "idn-00000000000a",
             "label": "match"}
    if validation:  # PERSON is 100-200 x 100-400 px of 640x480; FACE_BOX is 0.09 x 0.13 of it
        assert person == {**plain, "box": [0.156, 0.208, 0.312, 0.833], "confidence": 0.9,
                          "face": {"w_px": 58, "h_px": 62, "score": 0.9}, "similarity": 0.993}
    else:
        assert person == plain
    loop.close()


def test_an_empty_room_runs_no_face_analysis(tmp_path) -> None:
    loop = Loop(tmp_path)
    for _ in range(40):
        loop.frame(persons=[])
    counters = loop.runtime.snapshot()["components"]["face"]["worker"]["counters"]
    assert (counters["ticks"], counters["skipped_no_person"], counters.get("offered", 0)) == (3, 3, 0)
    loop.close()


def test_identity_records_are_capped_and_counted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo_runtime, "MAX_IDENTITY_RECORDS", 4)
    loop = Loop(tmp_path)
    for _ in range(40):
        loop.frame()
    assert len(loop.runtime.drain_identity_records()) == 4
    assert loop.runtime.snapshot()["components"]["face"]["records"] == {"kept": 4, "dropped": 3, "limit": 4}
    loop.close()


def test_unavailable_face_recognition_is_shown_and_degrades(tmp_path) -> None:
    loop = Loop(tmp_path, with_face=False, face_problem="no_identities_enrolled")
    loop.frame()
    snapshot = loop.runtime.snapshot()
    assert snapshot["components"]["face"] == {"state": "unavailable", "problem": "no_identities_enrolled"}
    assert "face recognition unavailable (no_identities_enrolled)" in snapshot["degraded"]
    assert snapshot["live"]["face_recognition"] == "unavailable"
    loop.close()


# ---------------------------------------------------------------- assemble()


class FaceBackend:
    def __init__(self, weights: Path, calls: list[str], *, error: Exception | None = None,
                 compatibility: FaceCompatibility = COMPAT) -> None:
        self.weights, self.calls, self.error, self._compatibility = weights, calls, error, compatibility

    @property
    def weight_files(self) -> dict[str, Path]:
        return {name: self.weights / name for name in ("facenet512_weights.h5", "face_detection_yunet_2023mar.onnx")}

    def load(self) -> None:
        self.calls.append("face_load")
        if self.error is not None:
            raise self.error

    def compatibility(self) -> FaceCompatibility:
        return self._compatibility

    def faces(self, frame, width, height, image):
        return FaceRun((), 0, 0)


class Store:
    def __init__(self, document: GalleryDocument | None = GALLERY, error: VaultError | None = None) -> None:
        self.document, self.error = document, error

    def exists(self) -> bool:
        return self.document is not None or self.error is not None

    def load(self, secret: Secret) -> GalleryDocument:
        if self.error is not None:
            raise self.error
        return self.document  # type: ignore[return-value]


@pytest.fixture
def face_setup(tmp_path, accepted_scene):
    scene = accepted_scene()
    weights = tmp_path / "weights"
    weights.mkdir()
    facts = {}
    for name, content in (("facenet512_weights.h5", b"model"), ("face_detection_yunet_2023mar.onnx", b"detector")):
        (weights / name).write_bytes(content)
        facts[name] = dataclasses.replace(file_facts(weights / name), sha256=hashlib.sha256(content).hexdigest())
    admission = FaceAdmission(profile_id=scene.profile.profile_id, status=FaceAdmissionStatus.PENDING_VALIDATION,
                              face_hz=1.0, model_weights=facts["facenet512_weights.h5"],
                              detector_weights=facts["face_detection_yunet_2023mar.onnx"], deepface_version="0.0.99")
    manifest = {"adapter_id": FACE_ADAPTER_ID, "contract_version": 1, "implementation_revision": "1",
                "enabled": True, "input_kinds": ["frame"], "output_kinds": ["face.observation"],
                "model_revision": "facenet512-yunet", "resource_profile_id": scene.profile.profile_id,
                "timeout_ms": 3000}
    return SimpleNamespace(scene=scene, weights=weights, admissions={admission.profile_id: admission},
                           manifest=manifest, tmp=tmp_path)


def start(setup, *, store=None, backend_error=None, compatibility=COMPAT, validation=True, memfree=7e9,
          identity=None, policy=None, manifests=None, comparison=False, backend_class=None, detection=None,
          stores=None, face_extra=None, tracker_backend=None):
    calls: list[str] = []
    data = setup.tmp / "data"
    data.mkdir(exist_ok=True)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "identity": identity or {},
                           "adapters": [setup.manifest] if manifests is None else manifests})
    devices = Devices(
        capture_source=lambda capture: SimpleNamespace(),
        tracker_backend=lambda engine: Tracker(calls) if tracker_backend is None else tracker_backend(calls),
        scene_server=lambda options, port: None,
        scene_request=lambda port, timeout: None,
        meminfo=lambda: {"MemFree": int(memfree), "MemAvailable": 7_000_000_000},
        notifiers=lambda config: ({}, {}),
        face_backend=lambda weights_dir: (backend_class or FaceBackend)(setup.weights, calls, error=backend_error,
                                                                        compatibility=compatibility),
        identity_store=lambda directory: ((stores or {}).get(Path(directory))
                                          or (store if store is not None else Store())),
    )
    options = RunOptions(data, setup.tmp / "x.engine", face=FaceOptions(setup.tmp / "identity", SECRET,
                                                                       validation_run=validation,
                                                                       comparison=comparison,
                                                                       **({"detection": detection} if detection
                                                                          else {}), **(face_extra or {})),
                         **({"memory_policy": policy} if policy is not None else {}))
    assembly = assemble(config, options, devices, FakeClock(), profiles=setup.scene.profiles,
                        face_admissions=setup.admissions)
    return assembly, calls


def test_a_validation_run_loads_face_after_the_detector(face_setup) -> None:
    assembly, calls = start(face_setup)
    assert calls == ["tracker_load", "face_load"]
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1}
    assert assembly.startup["face_problem"] is None and assembly.startup["face_memory_before"]["MemFree"] == 7_000_000_000
    assert assembly.runtime.snapshot()["components"]["face"]["state"] == "available"
    assembly.database.close()


@pytest.mark.parametrize(("change", "label"), [
    ({"validation": False}, "pending validation; only a guarded --face-validation run may use it"),
    ({"manifests": []}, f"no {FACE_ADAPTER_ID} adapter in the configuration"),
    ({"identity": {"face_interval_s": 2.0}}, "measured 1 Hz, not the configured 2 s interval"),
    ({"policy": CANDIDATE_POLICY}, "measured memory policy"),
])
def test_face_is_refused_before_anything_starts_without_its_admission(face_setup, change, label) -> None:
    with pytest.raises(StartupRefused) as refused:
        start(face_setup, **change)
    assert refused.value.label.startswith("face_not_admitted: ") and label in refused.value.label
    assert not (face_setup.tmp / "data" / "sentinel.db").exists()


def test_changed_face_weights_are_refused(face_setup) -> None:
    (face_setup.weights / "face_detection_yunet_2023mar.onnx").write_bytes(b"another detector")
    with pytest.raises(StartupRefused, match="does not have the profiled SHA-256"):
        start(face_setup)


def test_a_sealing_error_refuses_startup_before_the_database_opens(face_setup) -> None:
    with pytest.raises(StartupRefused) as refused:
        start(face_setup, store=Store(error=VaultError("authentication_failed")))
    assert refused.value.label == "identity_gallery: authentication_failed"
    assert not (face_setup.tmp / "data" / "sentinel.db").exists()


@pytest.mark.parametrize(("kwargs", "problem", "loaded"), [
    ({"store": Store(document=None)}, "no_identities_enrolled", False),
    ({"store": Store(document=GALLERY.model_copy(update={"identities": ()}))}, "no_identities_enrolled", False),
    ({"store": Store(document=GALLERY.model_copy(update={"compatibility": COMPAT.model_copy(
        update={"detector_backend": "retinaface"})}))}, "gallery_incompatible:detector_backend", False),
    ({"memfree": 1e9}, "memfree_below_minimum", False),
    ({"backend_error": FaceBackendError("tensorflow_gpu_visible")}, "tensorflow_gpu_visible", True),
    ({"compatibility": COMPAT.model_copy(update={"opencv_version": "4.12.0"})}, "gallery_incompatible:opencv_version",
     True),
])
def test_face_is_unavailable_with_its_reason_and_core_monitoring_starts(face_setup, kwargs, problem, loaded) -> None:
    assembly, calls = start(face_setup, **kwargs)
    assert assembly.startup["face_problem"] == problem and ("face_load" in calls) is loaded
    component = assembly.runtime.snapshot()["components"]["face"]
    assert component == {"state": "unavailable", "problem": problem}
    assert assembly.runtime.core.tick(None).state.face_recognition is Capability.UNAVAILABLE
    assembly.database.close()


def test_startup_opens_a_gallery_sealed_with_the_real_helper(face_setup) -> None:
    sealer = Sealer(kdf_n=2**14)
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    store = IdentityStore(face_setup.tmp / "identity", sealer)
    store.ensure_directory()
    store.save(GALLERY, SECRET)
    assembly, _ = start(face_setup, store=store)
    assert assembly.runtime.snapshot()["components"]["face"]["identities_enrolled"] == 1
    assembly.database.close()
    wrong = IdentityStore(face_setup.tmp / "identity", sealer)
    with pytest.raises(StartupRefused, match="identity_gallery: authentication_failed"):
        start(face_setup, store=SimpleNamespace(exists=wrong.exists,
                                                load=lambda secret: wrong.load(Secret("another long passphrase"))))


# ---------------------------------------------------------------- the CLI's secret


def _cli(tmp_path, *extra: str) -> list[str]:
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    return ["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", str(tmp_path / "x.engine"),
            "--status-port", "0", *extra]


def _refusing_devices() -> Devices:
    def refuse(capture):
        raise SourceError("rtsp_url_missing")
    return Devices(capture_source=refuse, tracker_backend=lambda e: None, scene_server=lambda o, p: None,
                   scene_request=lambda p, t: None, meminfo=lambda: None)


def test_the_passphrase_is_removed_from_the_environment_even_without_face(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path), devices=_refusing_devices()) == 1
    assert SECRET_ENV not in os.environ
    assert "typed silently" not in capsys.readouterr().err


def test_face_needs_the_passphrase_and_validation_needs_face(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(SECRET_ENV, raising=False)
    assert main(_cli(tmp_path, "--face"), devices=_refusing_devices()) == 1
    assert f"identity_secret_missing ({SECRET_ENV} is not set)" in capsys.readouterr().err
    assert main(_cli(tmp_path, "--face-validation"), devices=_refusing_devices()) == 1
    assert "--face-validation needs --face" in capsys.readouterr().err


def test_a_secret_in_the_options_never_prints(tmp_path) -> None:
    options = FaceOptions(tmp_path, Secret("never printed passphrase"))
    assert "never printed" not in repr(options) and "never printed" not in str(RunOptions(tmp_path, tmp_path,
                                                                                            face=options))


def test_a_failing_face_worker_stop_is_reported(tmp_path) -> None:
    release = threading.Event()

    def slow(frame, image):
        release.wait(5)
        return FaceRun((), 0, 0)

    loop = Loop(tmp_path, analyze=slow)
    loop.tracker.script.append(list(PERSON))
    loop.clock.advance(1 / 15)
    frame = loop.capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    loop.slot.publish(CapturedFrame(frame, Image()))
    loop.runtime.step(timeout_s=0)
    deadline = time.monotonic() + 5
    while not loop.worker.status()["busy"]:
        assert time.monotonic() < deadline
        time.sleep(0.002)
    original = demo_runtime.FACE_STOP_TIMEOUT_S
    demo_runtime.FACE_STOP_TIMEOUT_S = 0.05
    try:
        result = loop.runtime.shutdown()
    finally:
        demo_runtime.FACE_STOP_TIMEOUT_S = original
        release.set()
        loop.db.close()
    assert result["stopped"]["face_worker"] is False and result["all_stopped"] is False


def test_the_adapters_full_scale_pass_reaches_the_validation_record(tmp_path) -> None:
    """D86: the worker carries the adapter's full-scale pass from its FaceRun to the result record (schema 3)."""
    from sentinel.identity.legacy_deepface import FullScale

    def analyze(frame, image):
        full = FullScale(((0.97, FACE_BOX), (0.42, NormalizedBox(x1=0.5, y1=0.5, x2=0.6, y2=0.6))), 21.0)
        return FaceRun((), 0, 1, full)  # the production pass found nothing; the full-scale pass found a face

    loop = Loop(tmp_path, analyze=analyze)
    for _ in range(3):
        loop.frame()
    result = next(r for r in loop.runtime.drain_identity_records() if r["identity"] == "result")
    assert result["schema"] == 3 and result["faces"] == 0
    assert result["full_scale"] == {"faces": 1, "assigned": 1, "best_score": 0.97, "best_w_px": 58, "ms": 21.0}
    loop.close()


# ---------------------------------------------------------------- D87: the full-scale comparison


class ComparingBackend(FaceBackend):
    """A face backend with D86's pass and D87's comparison (enable order recorded)."""

    def enable_full_scale(self) -> None:
        self.calls.append("enable_full_scale")

    def enable_comparison(self) -> None:
        self.calls.append("enable_comparison")

    def embed_full_scale(self, width, height, image, row):
        return vec(0.97, 0.1, 0.0, 0.05)


def test_a_comparison_run_writes_schema_4_with_comparison_records_and_changes_no_identity(tmp_path) -> None:
    """D87: a full-scale-only face is embedded after its frame's production result, within the budget (one in 4 s),
    and recorded with its similarity; production found no face, so the identity stays unresolved."""
    from sentinel.identity.association import AssociationGeometry
    from sentinel.identity.comparison import FullScaleComparator
    from sentinel.identity.legacy_deepface import FullScale, FullScaleRow

    row = FullScaleRow(121.0, 106.0, 58.0, 62.0, (135.0, 125.0), (160.0, 126.0), 0.97)

    def analyze(frame, image):
        return FaceRun((), 0, 1, FullScale(((0.97, FACE_BOX),), 21.0, (row,)))

    comparator = FullScaleComparator(lambda frame, image, r: vec(0.97, 0.1, 0.0, 0.05), AssociationGeometry())
    loop = Loop(tmp_path, analyze=analyze, comparator=comparator)
    for _ in range(40):  # three face ticks in about 2.7 s of frame time
        loop.frame()
    records = loop.runtime.drain_identity_records()
    assert {r["schema"] for r in records} == {4}
    assert [r["identity"] for r in records].count("result") == 3 and "transition" not in {r["identity"] for r in records}
    comparisons = [r for r in records if r["identity"] == "comparison"]
    assert [(c["kind"], c["disposition"]) for c in comparisons] == [
        ("full_scale_only", "completed"), ("full_scale_only", "skipped_interval"),
        ("full_scale_only", "skipped_interval")]
    assert comparisons[0]["similarity"] == 0.993 and comparisons[0]["w_px"] == 58 and comparisons[0]["ms"] >= 0
    results = [r for r in records if r["identity"] == "result"]
    assert [r["frame_seq"] for r in results] == [c["frame_seq"] for c in comparisons]
    assert all(records.index(c) > records.index(r) for c, r in zip(comparisons, results))  # production first
    snapshot = loop.runtime.snapshot()
    face = snapshot["components"]["face"]
    assert face["comparison_results"] == {"full_scale_only:completed": 1, "full_scale_only:skipped_interval": 2}
    assert face["worker"]["comparison"]["counters"]["attempted_full_scale_only"] == 1
    assert face["results"].get("votes_match", 0) == 0 and snapshot["live"]["identity"]["known"] == 0
    assert "embedding" not in json.dumps(snapshot) and "0.993" not in json.dumps(snapshot)
    loop.close()


def test_the_comparison_is_enabled_after_the_full_scale_pass_and_before_the_load(face_setup) -> None:
    assembly, calls = start(face_setup, comparison=True, backend_class=ComparingBackend)
    assert calls == ["tracker_load", "enable_full_scale", "enable_comparison", "face_load"]
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1, "comparison": True}
    assembly.database.close()
    plain, _ = start(face_setup, backend_class=ComparingBackend)
    assert plain.startup["face"] == {"validation_run": True, "identities_enrolled": 1}  # unchanged without it
    plain.database.close()


def test_the_comparison_is_refused_outside_validation_runs_and_without_support(face_setup) -> None:
    with pytest.raises(StartupRefused, match="face_comparison_needs_validation"):
        start(face_setup, comparison=True, validation=False, backend_class=ComparingBackend)
    with pytest.raises(StartupRefused, match="face_comparison_unavailable"):
        start(face_setup, comparison=True)  # this backend has no comparison


def test_the_comparison_flag_needs_validation(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-comparison"), devices=_refusing_devices()) == 1
    assert "--face-comparison needs --face-validation" in capsys.readouterr().err


# ---------------------------------------------------------------- D88: the full-scale detection candidate


class CandidateBackend(FaceBackend):
    """A face backend with D86's pass, D87's comparison and D88's candidate (each call recorded)."""

    def enable_full_scale(self) -> None:
        self.calls.append("enable_full_scale")

    def enable_comparison(self) -> None:
        self.calls.append("enable_comparison")

    def select_full_scale_detection(self) -> None:
        self.calls.append("select_full_scale_detection")


def test_the_candidate_replaces_the_detection_before_the_load_and_is_marked_at_startup(face_setup) -> None:
    assembly, calls = start(face_setup, detection="full_scale", backend_class=CandidateBackend)
    assert calls == ["tracker_load", "select_full_scale_detection", "face_load"]  # D86's pass is not enabled with it
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1, "detection": "full_scale"}
    assert assembly.runtime.snapshot()["components"]["face"]["detection"] == "full_scale"
    assembly.database.close()
    plain, calls = start(face_setup, backend_class=CandidateBackend)  # without the selection: production unchanged
    assert calls == ["tracker_load", "enable_full_scale", "face_load"]
    assert plain.startup["face"] == {"validation_run": True, "identities_enrolled": 1}
    assert "detection" not in plain.runtime.snapshot()["components"]["face"]
    plain.database.close()


def test_the_candidate_is_refused_outside_validation_runs_with_the_comparison_and_without_support(face_setup) -> None:
    with pytest.raises(StartupRefused, match="face_detection_needs_validation"):
        start(face_setup, detection="full_scale", validation=False, backend_class=CandidateBackend)
    with pytest.raises(StartupRefused, match="face_detection_excludes_comparison"):
        start(face_setup, detection="full_scale", comparison=True, backend_class=CandidateBackend)
    with pytest.raises(StartupRefused, match="face_detection_unavailable"):
        start(face_setup, detection="full_scale")  # this backend has no candidate
    with pytest.raises(StartupRefused, match="face_detection_unknown"):
        start(face_setup, detection="half_scale", backend_class=CandidateBackend)


def test_the_candidate_flag_needs_validation_and_excludes_the_comparison(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-full-scale-detection"), devices=_refusing_devices()) == 1
    assert "--face-full-scale-detection needs --face-validation" in capsys.readouterr().err
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-validation", "--face-comparison", "--face-full-scale-detection"),
                devices=_refusing_devices()) == 1
    assert "use one" in capsys.readouterr().err


def test_a_candidate_run_writes_schema_6_with_its_numbers_and_the_same_identity_rules(tmp_path) -> None:
    """D88: the worker carries the candidate's numbers from its FaceRun to the result record; the faces go through the
    unchanged association and votes (a match, then a face that does not match). D90: from the same FaceRun, each
    person's prototype position and the assigned face's geometry (schema 6; no working gallery, so no shadow)."""
    from sentinel.identity.legacy_deepface import CandidateDetection, FaceGeometry

    embeddings = [vec(0.97, 0.1, 0.0, 0.05), vec(0.97, 0.1, 0.0, 0.05), vec(0.0, 0.1, 0.99)]
    geometry = FaceGeometry((121.0, 104.0, 58.5, 63.0), (138.0, 128.0), (161.5, 126.5))

    def analyze(frame, image):
        embedding = embeddings.pop(0)
        face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.93, embedding=embedding)
        below = (0.71, NormalizedBox(x1=0.5, y1=0.5, x2=0.55, y2=0.56))
        return FaceRun((face,), 1, 0, detection=CandidateDetection(below, 48.5, 870.25), geometry=(geometry,))

    loop = Loop(tmp_path, analyze=analyze, detection="full_scale")
    for _ in range(40):  # three face ticks
        loop.frame()
    records = loop.runtime.drain_identity_records()
    assert {r["schema"] for r in records} == {6}
    results = [r for r in records if r["identity"] == "result"]
    assert [r["persons"][0]["label"] for r in results] == ["match", "match", "unknown"]
    for result in results:
        person = result["persons"][0]
        assert person["prototype"] == 0 and person["shadow"] is None  # one prototype enrolled; no working gallery
        assert person["geometry"] == {"box": [0.19, 0.22, 0.28, 0.35],
                                      "raw_box": [round(121.0 / 640, 3), round(104.0 / 480, 3),
                                                  round(179.5 / 640, 3), round(167.0 / 480, 3)],
                                      "overhang_px": [0.0, 0.0, 0.0, 0.0],
                                      "eyes": {"roll_deg": round(math.degrees(math.atan2(-1.5, 23.5)), 1),
                                               "spacing_px": round(math.hypot(23.5, -1.5), 1)}}
    assert all(r["detection"] == {"best_below_score": 0.71, "best_below_w_px": 32, "detect_ms": 48.5,
                                  "embed_ms": 870.2} for r in results)  # rounded to 0.1 ms
    assert all("full_scale" not in r for r in results)
    assert [(t["from"], t["to"]) for t in records if t["identity"] == "transition"] == [
        ("unresolved", "known"), ("known", "uncertain")]
    face = loop.runtime.snapshot()["components"]["face"]
    assert face["detection"] == "full_scale" and "comparison_results" not in face
    loop.close()


# ---------------------------------------------------------------- D91: the camera-side enrollment experiment


WORK = Path("/synthetic/identity-experiment")  # a working gallery's directory (the stores are fakes or tmp)
CANDIDATE = {"detection": "full_scale", "backend_class": None}


def _candidate(face_setup, **kwargs):
    return start(face_setup, detection="full_scale", backend_class=CandidateBackend, **kwargs)


@pytest.mark.parametrize("change, label", [
    (dict(face_extra={"shadow_dir": WORK}, detection=None), "face_experiment_needs_candidate"),
    (dict(face_extra={"shadow_dir": WORK, "capture_dir": WORK, "capture_consent_date": "2026-10-14"}),
     "face_experiment_one_kind"),
    (dict(face_extra={"shadow_dir": "SELF"}), "face_experiment_same_gallery"),
    (dict(face_extra={"capture_dir": WORK}), "face_capture_consent_date"),
    (dict(face_extra={"capture_dir": WORK, "capture_consent_date": "14 October"}), "face_capture_consent_date"),
    (dict(face_extra={"shadow_dir": WORK}, stores={WORK: Store(None)}), "face_shadow_missing"),
    (dict(face_extra={"shadow_dir": WORK}, stores={WORK: Store(GALLERY.model_copy(update={
        "compatibility": COMPAT.model_copy(update={"numpy_version": "2.3.0"})}))}),
     "face_shadow_incompatible:numpy_version"),
    (dict(face_extra={"shadow_dir": WORK}, stores={WORK: Store(GALLERY.model_copy(update={"identities": (
        GALLERY.identities[0].model_copy(update={"identity_id": "idn-00000000000b"}),)}))}),
     "face_shadow_unknown_identity"),
    (dict(face_extra={"capture_dir": WORK, "capture_consent_date": "2026-10-14"}, stores={WORK: Store(GALLERY)}),
     "face_capture_target_has_gallery"),
    (dict(face_extra={"shadow_dir": WORK}, store=Store(None), stores={WORK: Store(GALLERY)}),
     "face_experiment_needs_enrollment"),
])
def test_the_experiment_is_refused_before_anything_loads_unless_it_fits(face_setup, change, label) -> None:
    change = dict(change)
    if change.get("face_extra", {}).get("shadow_dir") == "SELF":
        change["face_extra"] = {"shadow_dir": face_setup.tmp / "identity" / "." }  # the original's own directory
    detection = change.pop("detection", "full_scale")
    with pytest.raises(StartupRefused, match=label):
        start(face_setup, detection=detection, backend_class=CandidateBackend, **change)
    assert not (face_setup.tmp / "data" / "sentinel.db").exists()  # refused before the database opens


@pytest.mark.parametrize("kind", ["shadow", "capture"])
def test_an_experiment_run_is_marked_at_startup_and_in_the_status(face_setup, kind) -> None:
    extra = {"shadow_dir": WORK} if kind == "shadow" else {"capture_dir": WORK, "capture_consent_date": "2026-10-14"}
    assembly, calls = _candidate(face_setup, face_extra=extra, stores={WORK: Store(GALLERY if kind == "shadow"
                                                                                   else None)})
    assert calls == ["tracker_load", "select_full_scale_detection", "face_load"]
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1, "detection": "full_scale",
                                        kind: True}
    assert assembly.runtime.snapshot()["components"]["face"]["experiment"] == kind
    assembly.database.close()
    plain, _ = _candidate(face_setup)
    assert "experiment" not in plain.runtime.snapshot()["components"]["face"]
    plain.database.close()


def _candidate_analyze(embeddings, people_per_result=None):
    from sentinel.identity.legacy_deepface import CandidateDetection, FaceGeometry

    geometry = FaceGeometry((121.0, 104.0, 58.5, 63.0), (138.0, 128.0), (161.5, 126.5))

    def analyze(frame, image):
        embedding = embeddings(frame) if callable(embeddings) else embeddings.pop(0)
        face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.93, embedding=embedding)
        return FaceRun((face,), 1, 0, detection=CandidateDetection(None, 48.0, 861.0), geometry=(geometry,))
    return analyze


def _step_timer(step_s: float):
    """A deterministic stand-in for the worker's wall-clock timer: each reading is `step_s` after the one before, so
    every face job (one reading at its start, one at its end) measures exactly `step_s`."""
    readings = iter(range(1, 1_000_000))
    return lambda: next(readings) * step_s


def test_a_shadow_run_records_the_working_gallerys_numbers_and_votes_only_from_the_original(tmp_path) -> None:
    from sentinel.identity.experiment import Shadow
    from sentinel.identity.matching import EnrolledIdentity, Enrollment

    near, other = vec(0.97, 0.1, 0.0, 0.05), vec(0.0, 0.1, 0.99)
    work = Enrollment(model_revision="synthetic", dimension=512, identities=(EnrolledIdentity(
        identity_id="idn-00000000000a", prototypes=(vec(0.0, 0.0, 0.0, 1.0), near)),))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    # Each run gets the same deterministic timer: processing_ms is the worker's wall-clock measurement, which differs
    # between any two real runs (0.0 against 0.1 ms on a CI runner) and says nothing about the shadow's isolation.
    shadowed = Loop(tmp_path / "a", analyze=_candidate_analyze([near, near, other]), detection="full_scale",
                    shadow=Shadow(work, 2), perf_counter=_step_timer(0.0025))
    plain = Loop(tmp_path / "b", analyze=_candidate_analyze([near, near, other]), detection="full_scale",
                 perf_counter=_step_timer(0.0025))
    for _ in range(40):  # three face ticks
        shadowed.frame()
        plain.frame()
    with_shadow, without = shadowed.runtime.drain_identity_records(), plain.runtime.drain_identity_records()
    strip = [{**r, "persons": [{**p, "shadow": None} for p in r["persons"]]} if r["identity"] == "result" else r
             for r in with_shadow]
    assert strip == without  # only the shadow numbers differ: the votes, labels, states, transitions and times are the same
    assert [r["processing_ms"] for r in with_shadow if r["identity"] == "result"] == [2.5] * 3  # the measurement, carried
    shadows = [r["persons"][0]["shadow"] for r in with_shadow if r["identity"] == "result"]
    assert shadows[:2] == [{"similarity": 1.0, "prototype": 1}] * 2  # the working gallery's own copy of that face
    assert shadows[2]["prototype"] in (0, 1) and shadows[2]["similarity"] < 0.70
    assert [r["persons"][0]["label"] for r in with_shadow if r["identity"] == "result"] == ["match", "match", "unknown"]
    shadowed.close()
    plain.close()


def test_a_capture_run_offers_only_the_one_persons_compared_faces_and_writes_once_at_the_stop(tmp_path) -> None:
    from sentinel.identity.experiment import CAPTURE_SEGMENTS, Capture
    from sentinel.identity.gallery import IdentityStore

    sealer = Sealer(kdf_n=2**14)
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    store = IdentityStore(tmp_path / "work", sealer)
    capture = Capture(store, SECRET, GALLERY, "2026-10-14", 0.70)
    (tmp_path / "run").mkdir()
    faces = {"n": 0}

    def embedding(frame):
        faces["n"] += 1
        return vec(0.97, 0.1, 0.0, 0.05 + 0.001 * (faces["n"] % 7))

    loop = Loop(tmp_path / "run", analyze=_candidate_analyze(embedding), detection="full_scale", capture=capture)
    for _ in range(int(80 * 15)):  # 80 s at 15 fps: a face tick a second
        loop.frame()
    kept = capture.kept
    assert 75 <= kept <= 81 and not store.exists()  # nothing written while the run lasts
    two = PERSON + [RawTrack(2, 400.0, 100.0, 500.0, 400.0, 0.9)]
    for _ in range(45):  # three ticks with two people: the face is not the only person's, so it is not kept
        loop.frame(persons=two)
    assert capture.kept == kept
    loop.close()
    (record,) = [r for r in loop.runtime.drain_identity_records() if r["identity"] == "capture"]
    assert record["written"] is True and record["verified"] is True and record["refused"] is None
    assert record["schema"] == 6 and record["prototypes"] == CAPTURE_SEGMENTS and record["faces_kept"] == kept
    (identity,) = store.load(SECRET).identities
    assert identity.identity_id == "idn-00000000000a" and len(identity.prototypes) == CAPTURE_SEGMENTS
    assert "embedding" not in json.dumps(record) and capture.kept == 0


def test_the_experiment_flags_need_the_candidate_and_go_one_at_a_time(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-validation", "--face-shadow-gallery", str(tmp_path / "w")),
                devices=_refusing_devices()) == 1
    assert "--face-shadow-gallery needs --face-validation and --face-full-scale-detection" in capsys.readouterr().err
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-validation", "--face-full-scale-detection", "--face-shadow-gallery",
                     str(tmp_path / "w"), "--face-capture-gallery", str(tmp_path / "x"), "--face-capture-consent-date",
                     "2026-10-14"), devices=_refusing_devices()) == 1
    assert "separate runs" in capsys.readouterr().err
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-validation", "--face-full-scale-detection", "--face-capture-gallery",
                     str(tmp_path / "x")), devices=_refusing_devices()) == 1
    assert "go together" in capsys.readouterr().err


# ---------------------------------------------------------------- D95: each person's state for the dashboard


def test_the_status_lists_each_tracked_persons_state_on_its_track_with_no_identity_id(tmp_path) -> None:
    """D95: the dashboard's rows are the core's own per-track identity (state, basis, newest vote's age), keyed by the
    track, with the configured rule's numbers; no identity ID, embedding or similarity (D61); bounded, the rest
    counted; a track that ends takes its row with it."""
    loop = Loop(tmp_path)
    for _ in range(40):  # three matching face ticks: known on track 1
        loop.frame()
    live = loop.runtime.snapshot()["live"]
    (row,) = live["faces"]
    assert {k: row[k] for k in ("track_id", "stream_epoch", "track_status", "predicted", "state", "basis")} == {
        "track_id": 1, "stream_epoch": 1, "track_status": "confirmed", "predicted": False, "state": "known",
        "basis": "fresh"}
    assert row["reason"] == "2 consistent matches" and 0 <= row["vote_age_ms"] <= 3000
    assert set(row) == {"track_id", "stream_epoch", "track_status", "predicted", "last_measured_age_ms", "state",
                        "basis", "vote_age_ms", "reason"} and live["faces_not_shown"] == 0
    assert datetime.fromisoformat(live["published_utc"]) <= datetime.fromisoformat(
        loop.runtime.snapshot()["runtime"]["updated_utc"])
    assert loop.runtime.snapshot()["components"]["face"]["policy"] == {"confirmations": 2, "vote_ttl_s": 30.0,
                                                                       "fresh_window_s": 3.0}
    text = json.dumps(loop.runtime.snapshot())
    assert "idn-" not in text and "embedding" not in text and "similarity" not in text
    for _ in range(40):  # nobody: the track expires and its row goes; nothing moves to another track
        loop.frame(persons=[])
    assert loop.runtime.snapshot()["live"]["faces"] == []
    loop.close()


def test_the_rows_are_bounded_and_ordered_by_track(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo_runtime, "MAX_FACE_ROWS", 2)
    loop = Loop(tmp_path, analyze=analyze_with(None))
    people = [RawTrack(n, 40.0 + 60 * n, 100.0, 90.0 + 60 * n, 400.0, 0.9) for n in (5, 2, 3)]
    for _ in range(5):
        loop.frame(persons=people)
    live = loop.runtime.snapshot()["live"]
    assert [row["track_id"] for row in live["faces"]] == [2, 3] and live["faces_not_shown"] == 1
    assert all(row["state"] == "unresolved" and row["basis"] is None for row in live["faces"])
    loop.close()


def test_without_face_recognition_the_status_has_no_rows(tmp_path) -> None:
    loop = Loop(tmp_path, with_face=False)
    loop.frame()
    live = loop.runtime.snapshot()["live"]
    assert "faces" not in live and "faces_not_shown" not in live and "published_utc" in live
    loop.close()


# ---------------------------------------------------------------- D97: the continuity rule (validation runs only)


def test_the_continuity_rule_is_refused_outside_plain_validation_runs(face_setup) -> None:
    with pytest.raises(StartupRefused, match="face_continuity_needs_validation"):
        start(face_setup, validation=False, face_extra={"continuity": True})
    with pytest.raises(StartupRefused, match="face_continuity_production_only"):
        start(face_setup, detection="full_scale", backend_class=CandidateBackend, face_extra={"continuity": True})
    with pytest.raises(StartupRefused, match="face_continuity_production_only"):
        start(face_setup, comparison=True, backend_class=ComparingBackend, face_extra={"continuity": True})


def test_the_continuity_flag_needs_validation_and_production_detection(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-continuity"), devices=_refusing_devices()) == 1
    assert "--face-continuity needs --face-validation" in capsys.readouterr().err
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-validation", "--face-continuity", "--face-full-scale-detection"),
                devices=_refusing_devices()) == 1
    assert "production detection" in capsys.readouterr().err


def test_a_continuity_run_is_marked_at_startup_with_its_rule(face_setup) -> None:
    assembly, _ = start(face_setup, face_extra={"continuity": True})
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1, "continuity": {
        "max_gap_s": 0.5, "overlap_iou": 0.3, "min_step_iou": 0.4, "retention_s": 120.0}}
    policy = assembly.runtime.snapshot()["components"]["face"]["policy"]
    assert policy["continuity"] == assembly.startup["face"]["continuity"] and policy["vote_ttl_s"] == 30.0
    assembly.database.close()
    plain, _ = start(face_setup)  # without the flag: the startup and the status exactly as before
    assert plain.startup["face"] == {"validation_run": True, "identities_enrolled": 1}
    assert "continuity" not in plain.runtime.snapshot()["components"]["face"]["policy"]
    plain.database.close()


def test_a_continuity_runs_breaks_come_before_the_transitions_they_cause_and_fit_the_evidence_profile(
        tmp_path, monkeypatch) -> None:
    from sentinel.config import parse_config as _parse

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks/runner"))
    import face_evidence

    policy = _parse({"config_version": 1, "camera": {"id": "cam-1"}}).identity.continuity_policy()
    loop = Loop(tmp_path, continuity=policy)
    for _ in range(40):  # three matching face ticks: known on track 1
        loop.frame()
    (row,) = loop.runtime.snapshot()["live"]["faces"]
    assert row["state"] == "known" and row["followed_ms"] >= 2000
    loop.frame(persons=[RawTrack(1, 420.0, 100.0, 520.0, 400.0, 0.9)])  # the same number on a box far away: a jump
    records = loop.runtime.drain_identity_records()
    kinds = [(r["identity"], r.get("cause") or r.get("to")) for r in records if r["identity"] in ("break", "transition")]
    assert kinds[-2:] == [("break", "jump"), ("transition", "unresolved")]
    transition = records[-1] if records[-1]["identity"] == "transition" else next(
        r for r in reversed(records) if r["identity"] == "transition")
    assert transition["reason"] == "not carried over a break in tracking (the box jumped)"
    status = loop.runtime.snapshot()
    (row,) = status["live"]["faces"]
    assert row["state"] == "unresolved" and row["reason"] == transition["reason"] and row["followed_ms"] < 1000
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    from sentinel.status_page import build_status, read_store

    (out / "status-n1.json").write_text(json.dumps(build_status(status, read_store(tmp_path / "sentinel.db"), NOW)))
    assert face_evidence.read(out / "run.jsonl", "f2").nonconforming == []
    assert face_evidence.read(out / "status-n1.json", "f2").nonconforming == []
    loop.close()


# ---------------------------------------------------------------- D99: face recognition turned off and on at runtime


def test_face_turned_off_presents_nothing_earlier_as_current_and_on_again_starts_unresolved(tmp_path) -> None:
    loop = Loop(tmp_path, feature_controls=True)
    for _ in range(40):  # three matching face ticks: known on track 1
        loop.frame()
    assert loop.runtime.snapshot()["live"]["faces"][0]["state"] == "known"
    loop.runtime.drain_identity_records()
    assert loop.runtime.controls.request("face", "off")[0] == 202
    loop.frame()
    status = loop.runtime.snapshot()
    assert status["components"]["face"]["state"] == "disabled" and status["components"]["face"]["paused"] is True
    assert "faces" not in status["live"] and status["live"]["face_recognition"] == "disabled"
    assert status["live"]["identity"]["known"] == 0  # nothing earlier is presented as current
    ended = [r for r in loop.runtime.drain_identity_records() if r["identity"] == "transition"]
    assert [(r["from"], r["to"], r["reason"]) for r in ended] == [("known", "unresolved", "face recognition disabled")]
    offered = loop.worker.status()["counters"]["offered"]
    for _ in range(30):  # two seconds: no frame is offered, so no face is analysed
        loop.frame()
    assert loop.worker.status()["counters"]["offered"] == offered
    assert status["features"]["items"]["face"]["validation"] == "pending_validation"
    assert loop.runtime.controls.request("face", "on")[0] == 202
    loop.frame()
    (row,) = loop.runtime.snapshot()["live"]["faces"]
    assert row["state"] == "unresolved"  # every earlier vote was forgotten
    for _ in range(40):
        loop.frame()
    assert loop.runtime.snapshot()["live"]["faces"][0]["state"] == "known"  # from new matches only
    loop.close()


def test_a_face_result_in_flight_when_face_is_turned_off_is_ignored() -> None:
    from sentinel.contracts import TrackStatus
    from sentinel.contracts import TrackObservation as Observation
    from sentinel.identity.matching import EnrolledIdentity, Enrollment
    from sentinel.runtime import EdgeCore

    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stream = stamper.connect()
    enrollment = Enrollment(model_revision="synthetic", dimension=4, identities=(
        EnrolledIdentity(identity_id="idn-00000000000a", prototypes=((1.0, 0.0, 0.0, 0.0),)),))
    core = EdgeCore(parse_config({"config_version": 1, "camera": {"id": "cam-1"}}), clock, None,
                    face_recognition=Capability.AVAILABLE, enrollment=enrollment)
    clock.advance(0.1)
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    person = Observation.detected(frame, track_id=1, box=NormalizedBox(x1=0.15, y1=0.2, x2=0.32, y2=0.85),
                                  confidence=0.9, status=TrackStatus.CONFIRMED)
    core.on_frame(frame, [person], stream)
    core.set_face_recognition(Capability.DISABLED)  # turned off while that frame's analysis runs
    face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.95, embedding=(0.99, 0.1, 0.0, 0.0))
    output = core.on_face_outcome(frame, [person], [face], stream)
    assert output.face_result.outcome == "ignored_face_unavailable" and output.face_result.persons == ()
    core.set_face_recognition(Capability.AVAILABLE)
    clock.advance(0.1)
    assert core.tick(stream).state.people[0].identity.value == "unresolved"


# ---------------------------------------------------------------- session 79: the tracking diagnostic (validation runs only)


def test_the_tracking_diagnostic_is_refused_outside_plain_validation_runs_and_without_a_reading_backend(
        face_setup) -> None:
    from face_runs import DiagnosticTracker

    with pytest.raises(StartupRefused, match="tracking_diagnostic_needs_validation"):
        start(face_setup, validation=False, face_extra={"tracking_diagnostic": True})
    with pytest.raises(StartupRefused, match="tracking_diagnostic_plain_only"):
        start(face_setup, face_extra={"tracking_diagnostic": True, "continuity": True})
    with pytest.raises(StartupRefused, match="tracking_diagnostic_plain_only"):
        start(face_setup, detection="full_scale", backend_class=CandidateBackend,
              face_extra={"tracking_diagnostic": True})
    with pytest.raises(StartupRefused, match="tracking_diagnostic_plain_only"):
        start(face_setup, comparison=True, backend_class=ComparingBackend, face_extra={"tracking_diagnostic": True})
    with pytest.raises(StartupRefused, match="tracking_diagnostic_unavailable"):  # the plain test Tracker reads nothing
        start(face_setup, face_extra={"tracking_diagnostic": True})
    assembly, _ = start(face_setup, face_extra={"tracking_diagnostic": True},
                        tracker_backend=lambda calls: DiagnosticTracker())
    assembly.database.close()


def test_a_tracking_diagnostic_run_is_marked_at_startup_and_f2s_marker_is_otherwise_unchanged(face_setup) -> None:
    from face_runs import TRACKER_SETTINGS, DiagnosticTracker

    backends: list[DiagnosticTracker] = []

    def make(calls):
        backends.append(DiagnosticTracker())
        return backends[-1]

    assembly, _ = start(face_setup, face_extra={"tracking_diagnostic": True}, tracker_backend=make)
    face = assembly.startup["face"]
    assert {k: v for k, v in face.items() if k != "tracking_diagnostic"} == {"validation_run": True,
                                                                             "identities_enrolled": 1}
    marker = face["tracking_diagnostic"]
    assert marker["tracker"] == TRACKER_SETTINGS and marker["shadow_conf"] == 0.1
    assert marker["max_records"] == 20_000 and marker["max_line_bytes"] == 4096 and marker["max_bytes"] == 32_000_000
    assert backends[-1].enabled == 0.1  # enabled before the backend loaded
    status = assembly.runtime.snapshot()["components"]["detector"]["tracking"]
    assert status == {"kept": 0, "dropped": 0, "truncated_lines": 0, "bytes": 0, "max_line_bytes_seen": 0,
                      "limit": 20_000, "byte_limit": 32_000_000, "problems": {}}
    assembly.database.close()
    plain, _ = start(face_setup)  # without the flag: the startup and the status exactly as before
    assert plain.startup["face"] == {"validation_run": True, "identities_enrolled": 1}
    assert "tracking" not in plain.runtime.snapshot()["components"]["detector"]
    plain.database.close()


def test_a_tracking_diagnostic_run_refuses_to_start_unless_its_detector_loaded_with_the_hooks_and_settings(
        face_setup) -> None:
    """Session 80: the device run trk-20261016T032337Z started with detector_problem set and ran its cues against no
    detector. A diagnostic run now refuses at startup (tracking_diagnostic_not_ready:<reason>), before the face model
    loads, whenever the detector cannot load with the diagnostic's hooks or the tracker's settings are incomplete."""
    from face_runs import TRACKER_SETTINGS, DiagnosticTracker

    class Backend(DiagnosticTracker):
        def __init__(self, calls, *, error=None, settings=TRACKER_SETTINGS):
            super().__init__()
            self.calls, self.error, self.settings = calls, error, settings

        def load(self):
            self.calls.append("tracker_load")
            if self.error is not None:
                raise self.error

        def diagnostic_settings(self):
            return None if self.settings is None else dict(self.settings)

    flag = {"tracking_diagnostic": True}
    seen: dict[str, list[str]] = {}

    def make(calls, **kwargs):
        seen["calls"] = calls
        return Backend(calls, **kwargs)

    with pytest.raises(StartupRefused, match=r"tracking_diagnostic_not_ready:diagnostic_unsupported \(callback_order\)"):
        start(face_setup, face_extra=flag,
              tracker_backend=lambda calls: make(calls, error=TrackerError("diagnostic_unsupported", "callback_order")))
    assert seen["calls"] == ["tracker_load"]  # the face model never loaded: the refusal came first
    with pytest.raises(StartupRefused, match=r"tracking_diagnostic_not_ready:load_failed \(RuntimeError\)"):
        start(face_setup, face_extra=flag,
              tracker_backend=lambda calls: make(calls, error=TrackerError("load_failed", "RuntimeError")))
    seen.clear()
    with pytest.raises(StartupRefused, match="tracking_diagnostic_not_ready:memfree_below_minimum"):
        start(face_setup, face_extra=flag, memfree=0, tracker_backend=make)
    assert seen == {}  # no backend made, nothing loaded: the detector's memory check refused first
    with pytest.raises(StartupRefused, match="tracking_diagnostic_not_ready:tracker_settings_missing"):
        start(face_setup, face_extra=flag, tracker_backend=lambda calls: make(calls, settings=None))
    assert seen["calls"] == ["tracker_load"]
    with pytest.raises(StartupRefused, match="tracking_diagnostic_not_ready:tracker_settings_incomplete:iou"):
        start(face_setup, face_extra=flag, tracker_backend=lambda calls: make(
            calls, settings={k: v for k, v in TRACKER_SETTINGS.items() if k != "iou"}))
    assembly, calls = start(face_setup, face_extra=flag, tracker_backend=make)  # ready: as in session 79
    assert calls == ["tracker_load", "face_load"] and assembly.startup["detector_problem"] is None
    assert assembly.startup["face"]["tracking_diagnostic"]["tracker"] == TRACKER_SETTINGS
    assembly.database.close()


def test_without_the_diagnostic_a_detector_that_fails_to_load_still_starts_as_before(face_setup) -> None:
    class Failing:
        def __init__(self, calls):
            self.calls = calls

        def load(self):
            self.calls.append("tracker_load")
            raise TrackerError("diagnostic_unsupported", "callback_order")

        def track(self, image):
            return []

        def reset(self):
            pass

    assembly, calls = start(face_setup, tracker_backend=Failing)
    assert calls == ["tracker_load", "face_load"]  # the detector unavailable with its reason; everything else loads
    assert assembly.startup["detector_problem"] == "diagnostic_unsupported (callback_order)"
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1}
    detector = assembly.runtime.snapshot()["components"]["detector"]
    assert detector == {"state": "unavailable", "problem": "diagnostic_unsupported (callback_order)"}
    assembly.database.close()


def test_the_tracking_diagnostic_flag_needs_validation_and_the_plain_rules(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--tracking-diagnostic"), devices=_refusing_devices()) == 1
    assert "--tracking-diagnostic needs --face-validation" in capsys.readouterr().err
    for extra in (("--face-continuity",), ("--face-full-scale-detection",), ("--face-comparison",)):
        monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
        assert main(_cli(tmp_path, "--face", "--face-validation", "--tracking-diagnostic", *extra),
                    devices=_refusing_devices()) == 1
        assert "plain identity rules" in capsys.readouterr().err


# ---------------------------------------------------------------- session 90: the landmark diagnostic


class LandmarkBackend(FaceBackend):
    """A face backend with D86's pass and the landmark diagnostic's opt-in (each call recorded)."""

    def enable_full_scale(self) -> None:
        self.calls.append("enable_full_scale")

    def enable_landmarks(self) -> None:
        self.calls.append("enable_landmarks")


def test_the_landmark_diagnostic_is_enabled_after_the_full_scale_pass_and_marked_at_startup(face_setup) -> None:
    from sentinel.identity.landmarks import marker
    from sentinel.runtime import IDENTITY_FULL_SCALE_SCHEMA, IDENTITY_LANDMARK_SCHEMA

    assembly, calls = start(face_setup, backend_class=LandmarkBackend, face_extra={"landmark_diagnostic": True})
    assert calls == ["tracker_load", "enable_full_scale", "enable_landmarks", "face_load"]
    assert assembly.startup["face"] == {"validation_run": True, "identities_enrolled": 1, "landmark_diagnostic": marker()}
    assert assembly.runtime._identity_schema == IDENTITY_LANDMARK_SCHEMA
    assembly.database.close()
    plain, calls = start(face_setup, backend_class=LandmarkBackend)  # without it: D86's pass alone, as before
    assert calls == ["tracker_load", "enable_full_scale", "face_load"]
    assert plain.startup["face"] == {"validation_run": True, "identities_enrolled": 1}
    assert plain.runtime._identity_schema == IDENTITY_FULL_SCALE_SCHEMA
    plain.database.close()


@pytest.mark.parametrize(("extra", "label"), [
    ({"validation": False}, "landmark_diagnostic_needs_validation"),
    ({"comparison": True}, "landmark_diagnostic_plain_only"),
    ({"detection": "full_scale"}, "landmark_diagnostic_plain_only"),
    ({"face_extra": {"continuity": True}}, "landmark_diagnostic_plain_only"),
    ({"face_extra": {"tracking_diagnostic": True}}, "landmark_diagnostic_plain_only"),
    ({"backend_class": FaceBackend}, "landmark_diagnostic_unavailable"),
])
def test_the_landmark_diagnostic_is_refused_outside_plain_validation_runs_and_without_support(face_setup, extra,
                                                                                              label) -> None:
    options = {"backend_class": LandmarkBackend, **extra}
    options["face_extra"] = {"landmark_diagnostic": True, **options.get("face_extra", {})}
    with pytest.raises(StartupRefused, match=label):
        start(face_setup, **options)


def test_the_landmark_diagnostic_flag_needs_validation_and_the_plain_rules(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
    assert main(_cli(tmp_path, "--face", "--face-landmark-diagnostic"), devices=_refusing_devices()) == 1
    assert "--face-landmark-diagnostic needs --face-validation" in capsys.readouterr().err
    for extra in (("--face-continuity",), ("--face-full-scale-detection",), ("--face-comparison",),
                  ("--tracking-diagnostic",)):
        monkeypatch.setenv(SECRET_ENV, "typed silently before the launch")
        assert main(_cli(tmp_path, "--face", "--face-validation", "--face-landmark-diagnostic", *extra),
                    devices=_refusing_devices()) == 1
        assert "plain identity rules" in capsys.readouterr().err
