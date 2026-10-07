"""`sentinel run --face` (V2-25 demo form): admission, the sealed gallery, load order, the loop and the CLI's secret.

Fakes stand in for the camera, the detector and DeepFace; the profile and face admission used here are SYNTHETIC
fixtures passed explicitly (never the registry). One startup test opens a gallery sealed with the real helper.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
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
from sentinel.tracking.tracker import PersonTracker, RawTrack

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
                 with_face: bool = True) -> None:
        self.clock = FakeClock(utc=NOW)
        self.config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
        self.slot = LatestFrame()
        self.capture = Capture(self.clock, self.slot)
        self.tracker = Tracker()
        self.db = Database.open(tmp_path / "sentinel.db")
        self.worker = FaceWorker(analyze, self.clock)
        face = FaceRuntime(self.worker, 1, True) if with_face else None
        self.runtime = DemoRuntime(
            self.config, self.clock, capture=self.capture, slot=self.slot,
            incidents=IncidentService(self.db, self.clock, incidents=self.config.incidents,
                                      notifications=self.config.notifications),
            tracker=PersonTracker(self.tracker), face=face, face_problem=face_problem,
            enrollment=GALLERY.enrollment(), status_refresh_s=0.0)
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
          identity=None, policy=None, manifests=None):
    calls: list[str] = []
    data = setup.tmp / "data"
    data.mkdir(exist_ok=True)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "identity": identity or {},
                           "adapters": [setup.manifest] if manifests is None else manifests})
    devices = Devices(
        capture_source=lambda capture: SimpleNamespace(),
        tracker_backend=lambda engine: Tracker(calls),
        scene_server=lambda options, port: None,
        scene_request=lambda port, timeout: None,
        meminfo=lambda: {"MemFree": int(memfree), "MemAvailable": 7_000_000_000},
        notifiers=lambda config: ({}, {}),
        face_backend=lambda weights_dir: FaceBackend(setup.weights, calls, error=backend_error,
                                                     compatibility=compatibility),
        identity_store=lambda directory: store if store is not None else Store(),
    )
    options = RunOptions(data, setup.tmp / "x.engine", face=FaceOptions(setup.tmp / "identity", SECRET,
                                                                       validation_run=validation),
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
