"""The real runtime's evidence for the device-check readings' tests (face_check, features_check): the synthetic
operator's terms and gallery, the startup's face fields from the real assemble(), the fake camera and detector the
scripted runs drive, the privacy counts by face_counts.sh as the blocks run them, and the launch block's provenance.
Shared by test_face_check and test_features_check (moved from test_face_check unchanged)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sentinel.adapters import FACE_ADAPTER_ID, FaceAdmission, FaceAdmissionStatus
from sentinel.config import parse_config
from sentinel.contracts import NormalizedBox, StreamIdentity
from sentinel.demo_runtime import Devices, FaceOptions, MemoryOps, RunOptions, assemble, file_facts, post_load_release
from sentinel.identity.gallery import CONSENT_SCOPE, FaceCompatibility, GalleryDocument, GalleryIdentity, unit
from sentinel.identity.legacy_deepface import FaceRun, static_compatibility_fields
from sentinel.identity.vault import Secret
from sentinel.media.capture import CaptureState, CaptureStatus
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.memory_policy import RELEASE_FILE_ROLES
from sentinel.tracking.tracker import RawTrack

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"  # tests/support/..
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
REPORT = json.dumps({"persons_visible": 1, "fire_or_smoke": False, "threat": "low", "observations": ["a person"],
                     "uncertainty": "low", "summary": "A synthetic report."})
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


# ---------------------------------------------------------------- the real startup's face fields


def real_face_startup(root: Path, accepted_scene, *, comparison: bool = False,
                      candidate: bool = False, experiment: str | None = None,
                      continuity: bool = False, tracking: bool = False) -> dict[str, Any]:
    """assemble()'s face fields and post_load_release()'s face record, as `sentinel run --face` writes them (with
    ``comparison``, as `--face-validation --face-comparison` writes them, D87; with ``candidate``, as
    `--face-validation --face-full-scale-detection` writes them, D88; with ``continuity``, as `--face-validation
    --face-continuity` writes them, D97)."""
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
                              compatibility=lambda: COMPAT, faces=lambda *a: FaceRun((), 0, 0),
                              **({"enable_full_scale": lambda: None, "enable_comparison": lambda: None,
                                  "embed_full_scale": lambda *a: FACE} if comparison else {}),
                              **({"select_full_scale_detection": lambda: None} if candidate else {}))
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": [{
        "adapter_id": FACE_ADAPTER_ID, "contract_version": 1, "implementation_revision": "1", "enabled": True,
        "input_kinds": ["frame"], "output_kinds": ["face.observation"], "model_revision": "facenet512-yunet",
        "resource_profile_id": scene.profile.profile_id, "timeout_ms": 3000}]})
    devices = Devices(capture_source=lambda c: SimpleNamespace(),
                      tracker_backend=lambda e: (DiagnosticTracker() if tracking else
                                                 SimpleNamespace(load=lambda: None, track=lambda i: [],
                                                                 reset=lambda: None)),
                      scene_server=lambda o, p: None, scene_request=lambda p, t: None,
                      meminfo=lambda: {"MemFree": 5_000_000_000, "MemAvailable": 6_000_000_000},
                      notifiers=lambda c: ({}, {}), face_backend=lambda w: backend,
                      identity_store=lambda d: SimpleNamespace(
                          exists=lambda: Path(d).name != "capture-target", load=lambda s: GALLERY))
    (root / "data").mkdir()
    extra = ({"shadow_dir": root / "working"} if experiment == "shadow" else
             {"capture_dir": root / "capture-target", "capture_consent_date": "2026-10-14"}
             if experiment == "capture" else {})
    assembly = assemble(config, RunOptions(root / "data", root / "x.engine", face=FaceOptions(
        root / "identity", Secret("a long synthetic passphrase"), validation_run=True, comparison=comparison,
        continuity=continuity, tracking_diagnostic=tracking, **({"detection": "full_scale"} if candidate else {}),
        **extra)),
        devices, FakeClock(),
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


# ---------------------------------------------------------------- the scripted live run's fakes


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


# Session 79: the tracker's settings as the pinned backend reads them from the loaded objects
# (legacy_ultralytics.LegacyUltralyticsTracker.diagnostic_settings), for a synthetic diagnostic run's startup marker.
TRACKER_SETTINGS = {"conf": 0.4, "match_thresh": 0.8, "track_high_thresh": 0.25, "track_low_thresh": 0.1,
                    "new_track_thresh": 0.25, "track_buffer": 30, "max_time_lost": 30, "fuse_score": True,
                    "unconfirmed_thresh": 0.7, "shadow_conf": 0.1, "shadow_max_nms": 30000, "max_det": 300, "iou": 0.7}


class DiagnosticTracker(Tracker):
    """A Tracker that also hands the boundary a scripted reading of each frame (the tracking diagnostic)."""

    def __init__(self) -> None:
        super().__init__()
        self.diag: Any = None
        self.enabled: float | None = None

    def enable_diagnostic(self, shadow_conf: float = 0.1) -> None:
        self.enabled = shadow_conf

    def diagnostic_settings(self) -> dict[str, Any]:
        return dict(TRACKER_SETTINGS)

    def last_diagnostics(self) -> Any:
        return self.diag


def segment(script, t: float) -> tuple[str, int | None]:
    for start, end, state, track in script:
        if start <= t < end:
            return state, track
    return "absent", None


F2_PROVENANCE = ("f2_launch_utc=2026-10-08T09:00:00Z\nboot_id=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0\n"
                 "commit=95ce253c7b0d2784b6fecc9380bc10492905f252\npin_commit=95ce253c7b0d2784b6fecc9380bc10492905f252\n")
# What the enrollment experiment's -r2 and later launch blocks add to the run's provenance: the F1 gate's lines
# for an accepted recheck.
GATE_LINES = ("f1_gate=passed\nrun=face-20261008T090000Z\nf1_basis=accepted_recheck\n"
              "f1_recheck=face-20261008T090000Z-f1-0123456-20261008T100000Z\nf1_accepted_utc=2026-10-08T10:05:00Z\n"
              f"f1_identity_id={ENROLLED}\n")
