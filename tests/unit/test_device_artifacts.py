"""The device-artifacts route (session 116, part 41): a tester's own files, identified by a record, for experimental
validation runs only.

Everything here is synthetic: small files in a temporary directory stand in for the detector engine, the llama.cpp
build, the model files and the face weights; the resource profile, the face admission and the scene model set are
fixtures passed explicitly or patched for one test (never the registry's); fakes stand in for the camera, the
detector, the scene server and DeepFace. Nothing is built, loaded or opened.

Covered: a valid record is read and says what it is; a malformed record, or one that claims more than "pending" for
resource validation, is refused; a record whose built files were not made by the package's steps with the pinned
settings is refused; a changed, missing, replaced or re-timed file is refused; the default path decides exactly as
before and still refuses the device's own files; the record is refused outside a validation run; every other
admission rule still applies with a record; the starting line says that nothing was validated.

A pass here says the route's rules behave as written. It is not a run on another device.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from sentinel import demo_runtime, device_artifacts
from sentinel.adapters import (
    FACE_ADAPTER_ID,
    FACE_ADMISSIONS,
    RESOURCE_PROFILES,
    STEP4_CANDIDATE_CRITERIA_ID,
    CANDIDATE_RUNTIME_LIMITATION,
    STARTUP_IDENTITY_LIMITATION,
    FaceAdmission,
    FaceAdmissionStatus,
    FileFacts,
    ProfileStatus,
)
from sentinel.cli import main
from sentinel.config import parse_config
from sentinel.demo_runtime import (
    Devices,
    FaceOptions,
    RunOptions,
    SceneOptions,
    StartupRefused,
    assemble,
    device_engine_problem,
    face_admission,
    file_facts,
    scene_admission,
)
from sentinel.device_artifacts import DeviceArtifactsError
from sentinel.identity.gallery import CONSENT_SCOPE, FaceCompatibility, GalleryDocument, GalleryIdentity, unit
from sentinel.identity.legacy_deepface import FaceRun, static_compatibility_fields
from sentinel.identity.vault import Secret
from sentinel.media.clock import FakeClock
from sentinel.memory_policy import CANDIDATE_POLICY, DEFAULT_POLICY
from sentinel.scene import models
from sentinel.scene.llama_server import PROFILED_FLAGS, PROMPT_CACHE_FLAGS

REPO = Path(__file__).resolve().parents[2]
LIBRARIES = ("libggml-base.so.0.27.0", "libggml-cuda.so.0.27.0", "libllama.so.0.7.0")
LARGE = 64  # bytes: above the patched startup hash limit, so compared by name, size and modification time
WEIGHTS = ("facenet512_weights.h5", "face_detection_yunet_2023mar.onnx")
DOWNLOAD = {"observed": device_artifacts.DOWNLOADED, "step": "fetch-assets.sh", "utc": "2026-11-13T19:00:00Z",
            "url": "https://example.invalid/file", "revision": "a-synthetic-revision"}
INSPECTED = {"observed": device_artifacts.INSPECTED, "statement": device_artifacts.INSPECTED_STATEMENT}


class fakes:  # noqa: N801 - stand-ins for DeepFace, the sealed gallery and the detector, as test_face_runtime has them
    SECRET = Secret("a long synthetic passphrase")
    COMPAT = FaceCompatibility(**static_compatibility_fields(), opencv_version="4.13.0", tensorflow_version="2.21.0",
                               tf_keras_version="2.21.0", numpy_version="2.2.6")
    GALLERY = GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
        identity_id="idn-00000000000a", enrolled_utc=datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc),
        consent_date="2026-10-07", consent_scope=CONSENT_SCOPE, prototypes=(unit([1.0] + [0.0] * 511),)),))

    class Tracker:
        def load(self) -> None:
            pass

        def track(self, image: Any) -> list:
            return []

        def reset(self) -> None:
            pass

    class FaceBackend:
        def __init__(self, weights: Path, calls: list) -> None:
            self.weights = weights

        @property
        def weight_files(self) -> dict[str, Path]:
            return {name: self.weights / name for name in WEIGHTS}

        def load(self) -> None:
            pass

        def compatibility(self) -> FaceCompatibility:
            return fakes.COMPAT

        def faces(self, frame, width, height, image):
            return FaceRun((), 0, 0)

    class Store:
        def exists(self) -> bool:
            return True

        def load(self, secret: Secret) -> GalleryDocument:
            return fakes.GALLERY


def hashed(path: Path) -> FileFacts:
    return dataclasses.replace(file_facts(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def entry(path: Path) -> dict[str, Any]:
    facts = hashed(path)
    return {"name": facts.name, "bytes": facts.bytes, "sha256": facts.sha256, "mtime_utc": facts.mtime_utc}


class Device:
    """A synthetic tester's device: its own files, a record of them, and the reference device's records they are
    not (a synthetic accepted profile, face admission and scene model set)."""

    def __init__(self, tmp: Path, accepted_scene, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp
        monkeypatch.setattr(demo_runtime, "STARTUP_HASH_LIMIT_BYTES", 32)  # small files hashed, LARGE ones by metadata
        own, reference = tmp / "device", tmp / "reference"
        for folder in (own, reference, own / "weights", reference / "weights"):
            folder.mkdir()
        real = models.SCENE_MODEL_SETS["lfm3-vl"]
        names = {"engine": "yolov8n.engine", "binary": "llama-server", "model": real.model.name, "mmproj": real.mmproj.name}
        sizes = {"engine": 12, "binary": 8, "model": LARGE, "mmproj": LARGE, LIBRARIES[0]: 16, LIBRARIES[1]: LARGE,
                 LIBRARIES[2]: 20, "weights/" + WEIGHTS[0]: LARGE, "weights/" + WEIGHTS[1]: 24}
        for key, size in sizes.items():
            name = names.get(key, key)
            downloaded = key in ("model", "mmproj") or key.startswith("weights/")
            (reference / name).write_bytes(b"R" * size if not downloaded else b"D" * size)
            (own / name).write_bytes(b"T" * size if not downloaded else b"D" * size)  # a download has the same bytes
            os.utime(reference / name, (1_700_000_000, 1_700_000_000))  # ... and another time on each device
            os.utime(own / name, (1_800_000_000, 1_800_000_000))
        self.own, self.reference = own, reference
        self.engine, self.binary = own / names["engine"], own / names["binary"]
        self.model, self.mmproj, self.weights = own / names["model"], own / names["mmproj"], own / "weights"
        # the reference device's records: what the default path compares with
        self.set = dataclasses.replace(real, directory=reference, model=hashed(reference / names["model"]),
                                       mmproj=hashed(reference / names["mmproj"]))
        monkeypatch.setattr(models, "SCENE_MODEL_SETS", MappingProxyType({**models.SCENE_MODEL_SETS, "lfm3-vl": self.set}))
        fixture = accepted_scene(
            llama_flags=PROFILED_FLAGS + PROMPT_CACHE_FLAGS + self.set.server_flags,
            scene_request_sha256=self.set.scene_request_sha256, llm=self.set.model, mmproj=self.set.mmproj,
            llama_server=hashed(reference / "llama-server"),
            llama_libraries=tuple(hashed(reference / name) for name in LIBRARIES),
            criteria_id=STEP4_CANDIDATE_CRITERIA_ID, memory_policy=CANDIDATE_POLICY, memory_policy_verified=True,
            limitations=(STARTUP_IDENTITY_LIMITATION, CANDIDATE_RUNTIME_LIMITATION))
        self.profile, self.profiles = fixture.profile, fixture.profiles
        self.admissions = {self.profile.profile_id: FaceAdmission(
            profile_id=self.profile.profile_id, status=FaceAdmissionStatus.PENDING_VALIDATION, face_hz=1.0,
            model_weights=hashed(reference / "weights" / WEIGHTS[0]),
            detector_weights=hashed(reference / "weights" / WEIGHTS[1]), deepface_version="0.0.99")}
        scene = {**fixture.manifest, "model_revision": self.set.model_revision}
        face = {"adapter_id": FACE_ADAPTER_ID, "contract_version": 1, "implementation_revision": "1", "enabled": True,
                "input_kinds": ["frame"], "output_kinds": ["face.observation"], "model_revision": "facenet512-yunet",
                "resource_profile_id": self.profile.profile_id, "timeout_ms": 3000}
        self.manifests = [scene, face]
        self.scene = SceneOptions(self.binary, self.model, self.mmproj, model_set="lfm3-vl")

    def config(self, **sections: Any):
        return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": self.manifests, **sections})

    def document(self) -> dict[str, Any]:
        """The record the package's steps would write for these files (lib/artifacts.py)."""
        return {
            "schema": device_artifacts.SCHEMA, "created_utc": "2026-11-13T22:00:00Z",
            "package": "sentinel-personal-beta-synthetic", "purpose": device_artifacts.PURPOSE,
            "resource_validation": "pending",
            "platform": {"device_model": "a synthetic device", "machine": "aarch64", "kernel": "5.15.148-tegra",
                         "l4t": "36.4.7", "os": "ubuntu 22.04", "cuda_toolkit": "12.6.11-1",
                         "tensorrt": "10.3.0.30-1+cuda12.5", "cudnn": "9.3.0.75-1", "python": "3.10.12",
                         "packages": {"torch": "2.9.1", "ultralytics": "8.4.25"},
                         "preflight": "not_verified_acknowledged"},
            "artifacts": {
                "detector_engine": {"file": entry(self.engine), "provenance": {
                    "observed": device_artifacts.BUILT, "step": "build-engine.sh", "utc": "2026-11-13T21:00:00Z",
                    "package": "sentinel-personal-beta-synthetic",
                    "weights_sha256": device_artifacts.DETECTOR_WEIGHTS_SHA256,
                    "export": dict(device_artifacts.ENGINE_EXPORT),
                    "tools": {"ultralytics": "8.4.25", "tensorrt": "10.3.0", "torch": "2.9.1"}}},
                "llama_server": {"file": entry(self.binary), "libraries": [entry(self.own / name) for name in LIBRARIES],
                                 "provenance": {
                    "observed": device_artifacts.BUILT, "step": "build-llama.sh", "utc": "2026-11-13T20:00:00Z",
                    "package": "sentinel-personal-beta-synthetic", "repository": device_artifacts.LLAMA_REPOSITORY, "commit": device_artifacts.LLAMA_COMMITS["lfm3-vl"],
                    "cmake": list(device_artifacts.LLAMA_CMAKE_OPTIONS), "target": "llama-server",
                    "reports": "version: 0.7.0-dev (build 11938, commit 19cce6919)", "tools": {"cmake": "cmake version 3.28.3"}}},
                "scene_model": {"file": entry(self.model), "provenance": dict(DOWNLOAD)},
                "scene_mmproj": {"file": entry(self.mmproj), "provenance": {
                    "observed": device_artifacts.VERIFIED, "pinned_url": "https://example.invalid/mmproj",
                    "revision": "a-synthetic-revision", "statement": device_artifacts.VERIFIED_STATEMENT}},
                "face_model_weights": {"file": entry(self.weights / WEIGHTS[0]), "provenance": dict(DOWNLOAD)},
                "face_detector_weights": {"file": entry(self.weights / WEIGHTS[1]), "provenance": dict(DOWNLOAD)},
            },
        }

    def record(self, change=None) -> Path:
        document = self.document()
        if change is not None:
            change(document)
        path = self.tmp / "device-artifacts.json"
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        return path

    def artifacts(self, change=None) -> device_artifacts.DeviceArtifacts:
        return device_artifacts.load(self.record(change))

    def scene_problem(self, artifacts=None, config=None, scene=None, policy=CANDIDATE_POLICY, profiles=None) -> str | None:
        return scene_admission(config or self.config(), scene or self.scene, profiles=profiles or self.profiles,
                               policy=policy, artifacts=artifacts)[0]

    def face_problem(self, artifacts=None, validation: bool = True) -> str | None:
        return face_admission(self.config(), FaceOptions(self.tmp / "identity", fakes.SECRET, validation_run=validation),
                              profiles=self.profiles, admissions=self.admissions, policy=CANDIDATE_POLICY,
                              weights_dir=self.weights, artifacts=artifacts)

    def assemble(self, artifacts=None, *, face: bool = True, validation: bool = True, scene: bool = True):
        """assemble() with stand-ins for every device-side part; what was called, and with what, is recorded."""
        calls: list[Any] = []

        def tracker_backend(engine: Path, **keywords: Any) -> Any:
            calls.append(("tracker", keywords))
            return fakes.Tracker()

        def scene_server(options: Any, port: int) -> Any:
            calls.append("scene_server")
            ready = SimpleNamespace(state=SimpleNamespace(value="ready"), problem=None, layers="25/25", vision_on_gpu=True)
            return SimpleNamespace(start=lambda timeout: ready, status=lambda: ready, stop=lambda grace: True, pid=None)

        thp = {"off": False}  # the candidate memory policy's calls, answered as a device answers them; nothing real
        memory = demo_runtime.MemoryOps(
            disable_thp=lambda: thp.update(off=True) or {"verified": True},
            thp_enabled=lambda who: (0 if thp["off"] else 1) if who == "self" else 1,
            release=lambda path: {"result": "returned_0", "returncode": 0, "bytes": 1, "elapsed_s": 0.0, "error": None},
            sleep=lambda seconds: None)
        devices = Devices(
            capture_source=lambda capture: SimpleNamespace(), tracker_backend=tracker_backend, scene_server=scene_server,
            scene_request=lambda port, timeout, model: (lambda job, image: None),
            meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000},
            notifiers=lambda config: ({}, {}), memory=memory,
            face_backend=lambda weights_dir: fakes.FaceBackend(self.weights, []),
            identity_store=lambda directory: fakes.Store())
        options = RunOptions(
            self.tmp / "data", self.engine, scene=self.scene if scene else None,
            face=FaceOptions(self.tmp / "identity", fakes.SECRET, validation_run=validation) if face else None,
            memory_policy=CANDIDATE_POLICY, device_artifacts=artifacts)
        (self.tmp / "data").mkdir(exist_ok=True)
        return (lambda: assemble(self.config(), options, devices, FakeClock(), profiles=self.profiles,
                                 face_admissions=self.admissions)), calls


@pytest.fixture
def device(tmp_path, accepted_scene, monkeypatch) -> Device:
    return Device(tmp_path, accepted_scene, monkeypatch)


# ---------------------------------------------------------------- the record: read strictly, and honest


def test_a_valid_record_is_read_and_says_that_it_identifies_files_and_validates_nothing(device: Device) -> None:
    path = device.record()
    record = device_artifacts.load(path)
    assert record.record_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(record.files) == set(device_artifacts.ROLES) and [lib.name for lib in record.libraries] == list(LIBRARIES)
    assert record.files["llama_server"] == hashed(device.binary) and record.files["scene_model"] == hashed(device.model)
    assert record.provenance["detector_engine"]["observed"] == "built_by_package_step"
    assert record.provenance["scene_mmproj"]["observed"] == "verified_against_pinned_checksum"
    said = record.startup_entry(scene=True)
    assert said == {
        "schema": "sentinel-device-artifacts/1", "record_sha256": record.record_sha256,
        "created_utc": "2026-11-13T22:00:00Z", "run": "experimental_validation_run_only",
        "compared_with_the_record": ["detector_engine", "face_weights", "llama_build", "scene_model_files"],
        "resource_validation": "pending", "resource_profile_measures_this_device": False,
        "recorded_platform_preflight": "not_verified_acknowledged",
        "detector_engine_sha256": hashed(device.engine).sha256, "llama_server_sha256": hashed(device.binary).sha256}
    without_scene = record.startup_entry(scene=False)  # a run without scene analysis compared no llama.cpp build
    assert without_scene["compared_with_the_record"] == ["detector_engine", "face_weights"]
    assert "llama_server_sha256" not in without_scene and without_scene["resource_validation"] == "pending"
    assert str(device.tmp) not in json.dumps(said) and str(device.tmp) not in path.read_text()  # names only, no paths
    assert "not a resource profile and measures nothing" in device_artifacts.PURPOSE


def _drop(*keys: str):
    def change(document: dict) -> None:
        target = document
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]
    return change


def _put(value: Any, *keys: str):
    def change(document: dict) -> None:
        target = document
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return change


MALFORMED = [
    (_put("sentinel-device-artifacts/2", "schema"), "schema is not sentinel-device-artifacts/1"),
    (_put("passed", "resource_validation"), 'resource_validation is not "pending"'),
    (_put("validated", "resource_validation"), "a record identifies files and validates nothing"),
    (_put(None, "resource_validation"), 'resource_validation is not "pending"'),
    (_put("this device passed its resource profile", "purpose"), "purpose is not the schema's own sentence"),
    (_put({"steady_max_bytes": 4_397_879_296}, "resource_profile"), "the record does not hold exactly"),
    (_put(True, "resource_validated"), "the record does not hold exactly"),
    (_drop("platform"), "the record does not hold exactly"),
    (_drop("artifacts", "detector_engine"), "artifacts does not hold exactly"),
    (_put({}, "artifacts", "another_model"), "artifacts does not hold exactly"),
    (_drop("artifacts", "llama_server", "libraries"), "artifacts.llama_server does not hold exactly"),
    (_put([], "artifacts", "llama_server", "libraries"), "libraries is not a list of files"),
    (_put("0" * 63, "artifacts", "scene_model", "file", "sha256"), "artifacts.scene_model.file.sha256"),
    (_put("A" * 64, "artifacts", "scene_model", "file", "sha256"), "artifacts.scene_model.file.sha256"),
    (_put(True, "artifacts", "scene_model", "file", "bytes"), "bytes is not a positive whole number"),
    (_put(0, "artifacts", "detector_engine", "file", "bytes"), "bytes is not a positive whole number"),
    (_put(1.5, "artifacts", "detector_engine", "file", "bytes"), "bytes is not a positive whole number"),
    (_put("../yolov8n.engine", "artifacts", "detector_engine", "file", "name"), "is not a file name"),
    (_put("2026-11-13 20:00", "artifacts", "detector_engine", "file", "mtime_utc"), "file.mtime_utc"),
    (_put("measured", "artifacts", "llama_server", "provenance", "observed"), "provenance.observed is not one of"),
    (_put("inspected_only", "artifacts", "scene_model", "provenance", "observed"), "provenance.observed is not one of"),
    (_put("built_by_package_step", "artifacts", "scene_model", "provenance", "observed"), "provenance.observed is not one of"),
    (_drop("platform", "tensorrt"), "platform.tensorrt is missing"),
    (_put("verified", "platform", "preflight"), "platform.preflight is not one of"),
    (_put("yesterday", "created_utc"), "created_utc is not as the schema requires"),
    (_put({"a": {"b": {"c": {"d": 1}}}}, "artifacts", "detector_engine", "provenance", "tools"), "does not allow"),
    # no claim or measurement below the top level either: provenance and platform hold exactly the schema's keys
    (_put("passed", "artifacts", "detector_engine", "provenance", "resource_validation"),
     "artifacts.detector_engine.provenance does not hold exactly"),
    (_put({"steady_max_bytes": 4_397_879_296}, "artifacts", "llama_server", "provenance", "resource_profile"),
     "artifacts.llama_server.provenance does not hold exactly"),
    (_put("validated", "artifacts", "scene_model", "provenance", "validation"), "provenance does not hold exactly"),
    (_put(True, "platform", "resource_validated"), "platform does not hold exactly"),
    (_put("another sentence", "platform", "test_fixture"), "platform does not hold exactly"),
    (_put("validated on this device", "package"), "package is not as the schema requires"),
    (_put("this file was validated", "artifacts", "scene_mmproj", "provenance", "statement"),
     "provenance.statement is not as the schema requires"),
    (_put({"observed": "inspected_only", "statement": "built with the pinned options"}, "artifacts", "llama_server",
          "provenance"), "provenance.statement is not as the schema requires"),
    (_put("inspected_only", "artifacts", "llama_server", "provenance", "observed"), "provenance does not hold exactly"),
    (_drop("artifacts", "detector_engine", "provenance", "export"), "provenance does not hold exactly"),
    (_drop("artifacts", "llama_server", "provenance", "cmake"), "provenance does not hold exactly"),
    (_put("build-llama.sh", "artifacts", "detector_engine", "provenance", "step"), "provenance.step is not as the schema requires"),
    (_put({"cmake": "validated \u2713"}, "artifacts", "llama_server", "provenance", "tools"), "provenance.tools"),
    # a time is a real time, in ASCII digits
    (_put("9999-99-99T99:99:99Z", "created_utc"), "created_utc is not as the schema requires"),
    (_put("\uff12\uff10\uff12\uff16-11-13T22:00:00Z", "created_utc"), "created_utc is not as the schema requires"),
    (_put("2026-13-45T25:61:61+00:00", "artifacts", "llama_server", "file", "mtime_utc"), "file.mtime_utc"),
    (_put("2026-02-30T00:00:00Z", "artifacts", "detector_engine", "provenance", "utc"), "provenance.utc"),
]


@pytest.mark.parametrize(("change", "told"), MALFORMED, ids=[told[:38] + str(n) for n, (_, told) in enumerate(MALFORMED)])
def test_a_malformed_record_or_one_that_claims_a_validation_is_refused(device: Device, change, told: str) -> None:
    with pytest.raises(DeviceArtifactsError) as refused:
        device.artifacts(change)
    assert refused.value.label.startswith("device_artifacts_malformed: ") and told in refused.value.label
    assert str(device.tmp) not in refused.value.label


@pytest.mark.parametrize(("data", "told"), [
    (b"not json", "not a JSON document"), (b"[]", "the record is not an object"), (b"\xff\xfe", "not a JSON document"),
    (b'{"schema": "sentinel-device-artifacts/1", "schema": "sentinel-device-artifacts/1"}', "a key is given twice"),
    (b'{"schema": NaN}', "a number is not finite"), (b" " * 262_145, "larger than a record can be"),
])
def test_bytes_that_are_not_a_record_are_refused(data: bytes, told: str) -> None:
    with pytest.raises(DeviceArtifactsError) as refused:
        device_artifacts.parse(data)
    assert told in refused.value.label


def test_a_record_that_is_not_there_is_refused_without_its_path(tmp_path) -> None:
    with pytest.raises(DeviceArtifactsError) as refused:
        device_artifacts.load(tmp_path / "absent.json")
    assert refused.value.label == "device_artifacts_unreadable: FileNotFoundError"


# ---------------------------------------------------------------- the default path is the same, and still refuses


def test_without_a_record_the_devices_own_files_are_refused_exactly_as_before(device: Device) -> None:
    assert device.scene_problem() == "llama-lfm2-vl-scene: llama-server llama-server does not have the profiled SHA-256"
    told = device.face_problem()
    assert told.startswith(f"{FACE_ADAPTER_ID}: face model weights facenet512_weights.h5 (64 B, 2027-01-15T08:00:00+00:00) "
                           "is not the profiled facenet512_weights.h5 (64 B, 2023-11-14T22:13:20+00:00)")
    start, calls = device.assemble()
    with pytest.raises(StartupRefused) as refused:
        start()
    assert refused.value.label.startswith("scene_not_admitted: ") and calls == []
    assert not (device.tmp / "data" / "sentinel.db").exists()
    # the reference device's own files, without a record: admitted, as before
    reference = SceneOptions(device.reference / "llama-server", device.set.model_path, device.set.mmproj_path,
                             model_set="lfm3-vl")
    assert device.scene_problem(scene=reference) is None


def test_the_registry_the_pins_and_the_default_decisions_do_not_know_the_route() -> None:
    """Nothing the default path reads was changed: the registry's entries, the engine pin, the face admission."""
    from sentinel.inference.legacy_ultralytics import LEGACY_ENGINE_SHA256

    cap = RESOURCE_PROFILES["step4cand-lfm3cap-20261028T174508Z"]
    assert cap.status is ProfileStatus.ACCEPTED and cap.engine_sha256 == LEGACY_ENGINE_SHA256
    assert LEGACY_ENGINE_SHA256 == "08370639f961d2c67148c19562718ef80527c7085e88d2d923176180f1b98637"
    assert cap.llama_server.sha256 == "282105a90e18ee6fdfad9b39ef8e63f4392d094b26ab8ca9c83f7c72e3f50157"
    assert FACE_ADMISSIONS["step4cand-demo-20261007T090339Z"].status is FaceAdmissionStatus.PENDING_VALIDATION
    assert RunOptions(Path("d"), Path("e")).device_artifacts is None
    for function in (demo_runtime.profile_mismatch, scene_admission, face_admission):
        assert function.__kwdefaults__["artifacts"] is None


# ---------------------------------------------------------------- with a record, in a validation run


def test_with_its_record_the_devices_own_files_are_identified_and_nothing_else_is_relaxed(device: Device) -> None:
    record = device.artifacts()
    assert device_engine_problem(device.engine, record) is None
    assert device.scene_problem(record) is None
    assert device.face_problem(record) is None
    # face recognition is still pending validation: the record does not stand in for the validation flag
    assert "pending validation; only a guarded --face-validation run may use it" in device.face_problem(record, validation=False)
    # the configuration must still be the one the accepted profile measured
    assert "measured a 4 s scene interval, not the configured 2 s" in device.scene_problem(
        record, config=device.config(scene={"interval_s": 2.0}))
    assert "measured memory policy" in device.scene_problem(record, policy=DEFAULT_POLICY)
    # ... by a profile that is accepted, and that measured the pinned engine
    pending = {device.profile.profile_id: dataclasses.replace(device.profile, status=ProfileStatus.PENDING)}
    assert "is pending, not accepted" in device.scene_problem(record, profiles=pending)
    other = {device.profile.profile_id: dataclasses.replace(device.profile, engine_sha256="0" * 64)}
    assert device.scene_problem(record, profiles=other).endswith("measured another detector engine than the pinned one")
    unprofiled = dataclasses.replace(device.set, profiled=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(models, "SCENE_MODEL_SETS", MappingProxyType({**models.SCENE_MODEL_SETS, "lfm3-vl": unprofiled}))
        assert "has no accepted resource profile (D46)" in device.scene_problem(record)


def changed(device: Device, case: str) -> None:
    later = (1_800_000_500, 1_800_000_500)
    if case == "binary_rebuilt":
        device.binary.write_bytes(b"X" * 8)
    elif case == "small_library_changed":
        (device.own / LIBRARIES[0]).write_bytes(b"X" * 16)
    elif case == "library_missing":
        (device.own / LIBRARIES[2]).unlink()
    elif case == "large_library_rewritten":
        os.utime(device.own / LIBRARIES[1], later)
    elif case == "large_library_other_size":
        (device.own / LIBRARIES[1]).write_bytes(b"T" * (LARGE + 1))
    elif case == "model_copied_again":
        os.utime(device.model, later)
    elif case == "model_other_size":
        device.model.write_bytes(b"D" * (LARGE + 8))
    elif case == "projector_missing":
        device.mmproj.unlink()
    elif case == "engine_rebuilt":
        device.engine.write_bytes(b"X" * 12)
    elif case == "engine_missing":
        device.engine.unlink()
    elif case == "face_weights_copied_again":
        os.utime(device.weights / WEIGHTS[0], later)
    elif case == "face_detector_changed":
        (device.weights / WEIGHTS[1]).write_bytes(b"X" * 24)


@pytest.mark.parametrize(("case", "where", "told"), [
    ("binary_rebuilt", "scene", "llama-server llama-server does not have the recorded SHA-256"),
    ("small_library_changed", "scene", f"llama.cpp library {LIBRARIES[0]} does not have the recorded SHA-256"),
    ("library_missing", "scene", f"llama.cpp library {LIBRARIES[2]} is missing"),
    ("large_library_rewritten", "scene", f"llama.cpp library {LIBRARIES[1]} (64 B, 2027-01-15T08:08:20+00:00) is not the recorded"),
    ("large_library_other_size", "scene", f"llama.cpp library {LIBRARIES[1]} (65 B,"),
    ("model_copied_again", "scene", "is not the recorded LFM3-VL-2B-QAD-Q4_0.gguf (64 B, 2027-01-15T08:00:00+00:00)"),
    ("model_other_size", "scene", "scene model LFM3-VL-2B-QAD-Q4_0.gguf (72 B,"),
    ("projector_missing", "scene", "scene projector mmproj-LFM3-VL-2B-Q8_0.gguf is missing"),
    ("engine_rebuilt", "engine", "detector engine yolov8n.engine does not have the recorded SHA-256"),
    ("engine_missing", "engine", "detector engine yolov8n.engine is missing"),
    ("face_weights_copied_again", "face", "face model weights facenet512_weights.h5 (64 B, 2027-01-15T08:08:20+00:00) is not the recorded"),
    ("face_detector_changed", "face", "face detector weights face_detection_yunet_2023mar.onnx does not have the recorded SHA-256"),
])
def test_a_file_that_is_missing_or_changed_since_the_record_is_refused(device: Device, case: str, where: str, told: str) -> None:
    record = device.artifacts()  # made while the files were as built
    changed(device, case)
    problem = {"scene": lambda: device.scene_problem(record), "face": lambda: device.face_problem(record),
               "engine": lambda: device_engine_problem(device.engine, record)}[where]()
    assert problem is not None and told in problem and str(device.tmp) not in problem
    start, calls = device.assemble(record)  # and the run refuses before anything opens or loads
    with pytest.raises(StartupRefused) as refused:
        start()
    assert told in refused.value.label and calls == [] and not (device.tmp / "data" / "sentinel.db").exists()
    assert refused.value.label.split(":")[0] in {"scene": ("scene_not_admitted",), "face": ("face_not_admitted",),
                                                 "engine": ("device_artifacts_refused",)}[where]


INCOMPATIBLE = [
    (_put(INSPECTED, "artifacts", "detector_engine", "provenance"), "engine", "the detector engine was only inspected"),
    (_put(device_artifacts.TEST_FIXTURE, "platform", "test_fixture"), "engine",
     "the record's platform facts were read from a test fixture, not from a device: no run uses it"),
    (_put({"format": "engine", "imgsz": 640, "batch": True, "half": 0, "int8": 0, "dynamic": 0, "simplify": 1, "nms": 0,
           "device": False}, "artifacts", "detector_engine", "provenance", "export"), "engine",
     "export settings are not the pinned ones"),
    (_put("0" * 64, "artifacts", "detector_engine", "provenance", "weights_sha256"), "engine",
     "was not exported from the pinned yolov8n.pt"),
    (_put({**device_artifacts.ENGINE_EXPORT, "half": True}, "artifacts", "detector_engine", "provenance", "export"), "engine",
     "export settings are not the pinned ones"),
    (_put({**device_artifacts.ENGINE_EXPORT, "imgsz": 1280}, "artifacts", "detector_engine", "provenance", "export"), "engine",
     "export settings are not the pinned ones"),
    (_put({"ultralytics": "8.4.26", "tensorrt": "10.3.0"}, "artifacts", "detector_engine", "provenance", "tools"), "engine",
     "was not exported with ultralytics 8.4.25"),
    (_put({"ultralytics": "8.4.25", "tensorrt": "10.7.0"}, "artifacts", "detector_engine", "provenance", "tools"), "engine",
     "was not exported with tensorrt 10.3.0"),
    (_put(INSPECTED, "artifacts", "llama_server", "provenance"), "scene", "the llama.cpp build was only inspected"),
    (_put("98dc1418e0000000000000000000000000000000", "artifacts", "llama_server", "provenance", "commit"), "scene",
     "is not https://github.com/ggml-org/llama.cpp at the pinned commit 19cce6919992fdffd92e0ac152caaf1e2b8f254e"),
    (_put("https://example.invalid/fork", "artifacts", "llama_server", "provenance", "repository"), "scene",
     "at the pinned commit"),
    (_put(["-DCMAKE_BUILD_TYPE=Debug", "-DGGML_CUDA=ON"], "artifacts", "llama_server", "provenance", "cmake"), "scene",
     "build's options are not the pinned ones"),
    (_put(list(reversed(device_artifacts.LLAMA_CMAKE_OPTIONS)), "artifacts", "llama_server", "provenance", "cmake"), "scene",
     "build's options are not the pinned ones"),
    (_put("llama-cli", "artifacts", "llama_server", "provenance", "target"), "scene", "build's options are not the pinned ones"),
    (lambda d: d["artifacts"]["llama_server"]["libraries"].pop(), "scene", "libraries are not the ones the pinned build consists of"),
    (_put("f" * 64, "artifacts", "scene_model", "file", "sha256"), "scene",
     "the recorded scene model is not the pinned LFM3-VL-2B-QAD-Q4_0.gguf"),
    (_put("another.gguf", "artifacts", "scene_mmproj", "file", "name"), "scene", "the recorded scene projector is not the pinned"),
    (_put("f" * 64, "artifacts", "face_model_weights", "file", "sha256"), "face",
     "device artifacts: the recorded face model weights is not the pinned facenet512_weights.h5"),
    (_put(25, "artifacts", "face_detector_weights", "file", "bytes"), "face",
     "the recorded face detector weights is not the pinned face_detection_yunet_2023mar.onnx"),
]


@pytest.mark.parametrize(("change", "where", "told"), INCOMPATIBLE,
                         ids=[f"{where}-{n}" for n, (_, where, _told) in enumerate(INCOMPATIBLE)])
def test_a_record_of_files_that_are_not_the_pinned_ones_or_not_the_packages_builds_is_refused(
        device: Device, change, where: str, told: str) -> None:
    record = device.artifacts(change)
    problem = {"scene": lambda: device.scene_problem(record), "face": lambda: device.face_problem(record),
               "engine": lambda: device_engine_problem(device.engine, record)}[where]()
    assert problem is not None and told in problem
    assert "device artifacts: " in problem


def test_a_scene_model_set_without_a_pinned_build_for_the_route_is_refused(device: Device, monkeypatch) -> None:
    record = device.artifacts()
    monkeypatch.setattr(device_artifacts, "LLAMA_COMMITS", MappingProxyType({"another-set": "0" * 40}))
    told = device.scene_problem(record)
    assert "scene model set lfm3-vl has no llama.cpp build pinned for this route (only: another-set)" in told


# ---------------------------------------------------------------- validation runs only, and said in the run's own line


@pytest.mark.parametrize("run", [{"face": False}, {"validation": False}, {"face": False, "scene": False}])
def test_a_record_is_refused_outside_a_face_validation_run_before_anything_opens(device: Device, run: dict) -> None:
    start, calls = device.assemble(device.artifacts(), **run)
    with pytest.raises(StartupRefused) as refused:
        start()
    assert refused.value.label.startswith("device_artifacts_needs_validation: ") and "it admits nothing" in refused.value.label
    assert calls == [] and not (device.tmp / "data" / "sentinel.db").exists()


def test_a_validation_run_on_a_record_starts_and_its_first_line_says_nothing_was_validated(device: Device) -> None:
    record = device.artifacts()
    start, calls = device.assemble(record)
    assembly = start()
    try:
        # the detector's own pin is the recorded engine's checksum, and nothing else was passed to it
        assert calls == ["scene_server", ("tracker", {"expected_sha256": hashed(device.engine).sha256})]
        startup = assembly.startup
        assert startup["device_artifacts"] == record.startup_entry(scene=True)
        assert startup["device_artifacts"]["compared_with_the_record"] == ["detector_engine", "face_weights", "llama_build",
                                                                       "scene_model_files"]
        assert startup["device_artifacts"]["resource_validation"] == "pending"
        assert startup["device_artifacts"]["resource_profile_measures_this_device"] is False
        # what the guard's readiness check reads is as in every validation run: the record adds nothing there
        assert startup["face"] == {"validation_run": True, "identities_enrolled": 1}
        assert startup["scene_problem"] is None and startup["detector_problem"] is None and startup["face_problem"] is None
        assert startup["memory_policy"] == CANDIDATE_POLICY.labels()  # the accepted profile's policy, as in every run
        assert json.loads(json.dumps(startup["device_artifacts"])) == startup["device_artifacts"]
    finally:
        assembly.database.close()
    # without scene analysis the line names only what was compared: no llama.cpp build, no scene files
    (device.tmp / "data" / "sentinel.db").unlink()
    start, calls = device.assemble(record, scene=False)
    assembly = start()
    assert assembly.startup["device_artifacts"] == record.startup_entry(scene=False)
    assert "llama_server_sha256" not in assembly.startup["device_artifacts"]
    assembly.database.close()
    # the same run without a record gives the detector no other pin, and says nothing about a record
    monkey = pytest.MonkeyPatch()
    try:
        own = {device.profile.profile_id: dataclasses.replace(
            device.profile, llama_server=hashed(device.binary), llm=dataclasses.replace(device.set.model, mtime_utc=hashed(device.model).mtime_utc),
            mmproj=dataclasses.replace(device.set.mmproj, mtime_utc=hashed(device.mmproj).mtime_utc),
            llama_libraries=tuple(hashed(device.own / name) for name in LIBRARIES))}
        monkey.setattr(device, "profiles", own)
        monkey.setattr(device, "admissions", {device.profile.profile_id: dataclasses.replace(
            device.admissions[device.profile.profile_id], model_weights=hashed(device.weights / WEIGHTS[0]),
            detector_weights=hashed(device.weights / WEIGHTS[1]))})
        (device.tmp / "data" / "sentinel.db").unlink()
        start, calls = device.assemble(None)
        assembly = start()
        assert calls == ["scene_server", ("tracker", {})] and "device_artifacts" not in assembly.startup
        assembly.database.close()
    finally:
        monkey.undo()


def test_the_run_command_takes_a_record_only_with_the_validation_flags_and_refuses_one_it_cannot_read(
        device: Device, tmp_path, capsys, monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    record = device.record()
    base = ["run", str(config), "--data-dir", str(tmp_path / "db"), "--engine", str(device.engine), "--status-port", "0"]

    def run(*extra: str, environ: dict | None = None) -> tuple[int, str]:
        monkeypatch.setenv("SENTINEL_IDENTITY_PASSPHRASE", "a synthetic passphrase, not one")
        code = main([*base, *extra], capture_source=lambda capture: pytest.fail("the camera source was made"),
                    tracker_backend=lambda engine, **k: pytest.fail("the detector was constructed"),
                    meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000})
        return code, capsys.readouterr().err.strip()

    assert run("--device-artifacts", str(record)) == (1, "run: --device-artifacts needs --face --face-validation")
    assert run("--device-artifacts", str(record), "--face") == (1, "run: --device-artifacts needs --face --face-validation")
    assert run("--device-artifacts", str(record), "--scene") == (1, "run: --device-artifacts needs --face --face-validation")
    flags = ("--face", "--face-validation")
    assert run("--device-artifacts", str(tmp_path / "absent.json"), *flags) == (
        1, "run: device_artifacts_unreadable: FileNotFoundError")
    overclaiming = device.record(_put("passed", "resource_validation"))
    code, told = run("--device-artifacts", str(overclaiming), *flags)
    assert code == 1 and told.startswith('run: device_artifacts_malformed: resource_validation is not "pending"')
    assert not (tmp_path / "db").exists()  # nothing was opened for any of them
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--device-artifacts FILE" in text and "experimental validation runs only" in text
    assert "It admits nothing and measures nothing" in text and "resource validation is pending" in text


def test_the_real_detector_is_given_the_recorded_checksum_and_refuses_another_engine(device: Device) -> None:
    """The route changes which checksum the detector requires, not whether it requires one (no GPU: it stops at the
    checksum, before any import)."""
    from sentinel.cli import _legacy_tracker
    from sentinel.tracking.tracker import TrackerError

    recorded = hashed(device.engine).sha256
    for expected, label in ((None, "engine_hash_mismatch"), ("0" * 64, "engine_hash_mismatch"), (recorded, "reached_the_import")):
        backend = _legacy_tracker(device.engine, expected) if expected else _legacy_tracker(device.engine)
        backend._driver_check = lambda: None
        backend._importer = lambda: (_ for _ in ()).throw(ImportError("stopped here: nothing is imported in a test"))
        with pytest.raises(TrackerError) as refused:
            backend.load()
        assert refused.value.label == (label if label != "reached_the_import" else "runtime_unavailable")


# ---------------------------------------------------------------- the package's admission check, for a run on a record


def admission_check(device: Device, monkeypatch, capsys, *extra: str) -> tuple[int, list[str], str]:
    """packaging/beta-install/lib/admission_check.py's own main(), on the synthetic device and its synthetic
    registry (the script's decisions are the runtime's functions; only the registry they read is the fixture's)."""
    import functools
    import importlib.util
    import sys

    import yaml

    spec = importlib.util.spec_from_file_location("admission_check_under_test",
                                                  REPO / "packaging/beta-install/lib/admission_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "scene_admission", functools.partial(scene_admission, profiles=device.profiles))
    monkeypatch.setattr(module, "face_admission", functools.partial(face_admission, profiles=device.profiles,
                                                                    admissions=device.admissions))
    config = device.tmp / "live.yaml"
    config.write_text(yaml.safe_dump({"config_version": 1, "camera": {"id": "cam-1"},
                                      "capture": {"stream_profile": "main_single"}, "adapters": device.manifests}))
    monkeypatch.setattr(sys, "argv", [
        "admission_check.py", str(config), str(device.tmp / "identity"), "--engine", str(device.engine),
        "--llama-server", str(device.binary), "--scene-model", str(device.model), "--scene-mmproj", str(device.mmproj),
        "--weights-dir", str(device.weights), *extra])
    code = module.main()
    said = capsys.readouterr()
    return code, said.out.splitlines(), said.err


def test_the_admission_check_decides_as_the_runtime_does_and_never_calls_a_record_a_validation(
        device: Device, monkeypatch, capsys) -> None:
    # without a record: the default admission, which refuses this device's own files, as it always has
    code, out, err = admission_check(device, monkeypatch, capsys)
    assert code == 1 and "detector_engine=refused" in out and "scene_admission=refused" in out
    assert out[-1] == "admission_check=failed:detector_engine,scene_admission,face_admission_validation_run"
    assert "engine_hash_mismatch" in err and not any("device_artifacts" in line or "resource_validation" in line for line in out)
    # with its record: the same decisions for a run on that record, the default's own answers beside them
    record = device.record()
    code, out, err = admission_check(device, monkeypatch, capsys, "--device-artifacts", str(record))
    assert code == 0 and err == "", (out, err)
    for line in ("stream_profile=main_single", "notification_channels=0", "thresholds=unchanged",
                 "default_admission_detector_engine=refused", "default_admission_scene=refused",
                 f"device_artifacts_record_sha256={hashlib.sha256(record.read_bytes()).hexdigest()}",
                 "detector_engine=the_recorded_engine_validation_run_only",
                 "scene_admission=files_as_recorded_validation_run_only",
                 "face_admission_without_validation_flag=refused", "face_admission_validation_run=allowed",
                 "identity_retention_rule=proposed_defaults", "identity_retention_without_validation_flag=refused",
                 "display_labels=0", "chat_admitted=False", "chat_validation_run_allowed=True"):
        assert line in out, line
    assert out[-2:] == ["resource_validation=pending", "admission_check=as_expected_for_device_artifacts"]
    assert "scene_admission=admitted" not in out and "detector_engine=the_pinned_engine" not in out
    # a file changed since the record, and a record that cannot be used: refused, with the runtime's own reason
    os.utime(device.model, (1_800_000_900, 1_800_000_900))
    code, out, err = admission_check(device, monkeypatch, capsys, "--device-artifacts", str(record))
    assert code == 1 and "scene_admission=refused" in out and out[-1] == "admission_check=failed:scene_admission"
    assert "scene_refusal: llama-lfm2-vl-scene: scene model LFM3-VL-2B-QAD-Q4_0.gguf" in err and "is not the recorded" in err
    assert "resource_validation=pending" not in out
    overclaiming = device.record(_put("passed", "resource_validation"))
    code, out, err = admission_check(device, monkeypatch, capsys, "--device-artifacts", str(overclaiming))
    assert (code, out) == (1, ["device_artifacts=refused", "admission_check=failed:device_artifacts"])
    assert 'device_artifacts_refusal: device_artifacts_malformed: resource_validation is not "pending"' in err
    inspected = device.record(_put(INSPECTED, "artifacts", "detector_engine", "provenance"))
    os.utime(device.model, (1_800_000_000, 1_800_000_000))
    code, out, err = admission_check(device, monkeypatch, capsys, "--device-artifacts", str(inspected))
    assert code == 1 and "detector_engine=refused" in out and "the detector engine was only inspected" in err


# ---------------------------------------------------------------- the pins are the package's, and the build's own


def test_the_routes_pins_are_the_asset_manifests_and_the_accepted_builds() -> None:
    manifest = json.loads((REPO / "packaging/beta-install/manifest/assets.json").read_text())
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    llama, engine = assets["llama_server"], assets["detector_engine"]
    assert device_artifacts.LLAMA_COMMITS == {"lfm3-vl": models.LFM3_LLAMA_CPP_COMMIT} == {"lfm3-vl": llama["commit"]}
    assert (llama["repo"], llama["cmake_target"], tuple(llama["cmake"])) == (
        device_artifacts.LLAMA_REPOSITORY, device_artifacts.LLAMA_TARGET, device_artifacts.LLAMA_CMAKE_OPTIONS)
    cap = RESOURCE_PROFILES["step4cand-lfm3cap-20261028T174508Z"]
    assert sorted(llama["libraries"]) == sorted(library.name for library in cap.llama_libraries)
    assert engine["export"] == dict(device_artifacts.ENGINE_EXPORT) and engine["tools"] == dict(device_artifacts.ENGINE_TOOLS)
    assert assets["detector_weights"]["sha256"] == device_artifacts.DETECTOR_WEIGHTS_SHA256
    chosen = models.SCENE_MODEL_SETS["lfm3-vl"]
    for asset_id, pinned in (("scene_model", chosen.model), ("scene_mmproj", chosen.mmproj),
                             ("face_weights", FACE_ADMISSIONS["step4cand-demo-20261007T090339Z"].model_weights),
                             ("face_detector", FACE_ADMISSIONS["step4cand-demo-20261007T090339Z"].detector_weights)):
        asset = assets[asset_id]
        assert (Path(asset["path"]).name, asset["size"], asset["sha256"]) == (pinned.name, pinned.bytes, pinned.sha256)
    script = (REPO / "packaging/beta-install/build-engine.sh").read_text()  # the export is given the manifest's settings
    assert "asset --package \"$PKG\" detector_engine export" in script
    assert "YOLO(sys.argv[1]).export(**json.loads(sys.argv[2]))" in script
    # the build is told its target by name (the helper's own "target" field is where an asset is put, a path)
    assert 'TARGET="$(asset cmake_target)"' in (REPO / "packaging/beta-install/build-llama.sh").read_text()
    assert "target" not in llama


def test_the_evidence_profile_knows_the_starting_lines_new_entry(device: Device) -> None:
    import sys

    sys.path.insert(0, str(REPO / "benchmarks/runner"))
    import face_evidence

    entry_ = device.artifacts().startup_entry(scene=True)
    for key, value in entry_.items():
        assert f"device_artifacts.{key}{'[]' if isinstance(value, list) else ''}" in face_evidence.OPTIONAL_STARTUP, key
        assert face_evidence.RUN.known(f"startup.device_artifacts.{key}{'[]' if isinstance(value, list) else ''}"), key
    assert len([key for key in face_evidence.OPTIONAL_STARTUP if key.startswith("device_artifacts.")]) == len(entry_)
