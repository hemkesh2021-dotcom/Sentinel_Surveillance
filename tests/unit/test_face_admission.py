"""Face admission (V2-25 demo form): a separate PENDING_VALIDATION record keyed to the accepted replay profile, whose
entry and limitations stay as accepted. A profile alone never admits the face adapter."""

from __future__ import annotations

import dataclasses

import pytest

from sentinel.adapters import (
    CANDIDATE_RUNTIME_LIMITATION,
    FACE_ADAPTER_ID,
    FACE_ADMISSIONS,
    FACE_RUNTIME_LIMITATION,
    RESOURCE_PROFILES,
    AdapterState,
    FaceAdmissionStatus,
    ProfileStatus,
    face_admission_problem,
    resolve,
)
from sentinel.config import parse_config
from sentinel.identity.legacy_deepface import DEEPFACE_VERSION, DETECTOR_WEIGHTS, MODEL_WEIGHTS

ACCEPTED = "step4cand-demo-20261007T090339Z"


def face_manifest(profile: str | None) -> dict:
    manifest = {"adapter_id": FACE_ADAPTER_ID, "contract_version": 1, "implementation_revision": "1", "enabled": True,
                "input_kinds": ["frame"], "output_kinds": ["face.observation"], "model_revision": "facenet512-yunet",
                "timeout_ms": 3000}
    return manifest if profile is None else {**manifest, "resource_profile_id": profile}


def statuses(profile: str | None, **kwargs):
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": [face_manifest(profile)]})
    return resolve(config.adapters, **kwargs)


def test_the_accepted_replay_profile_entry_is_unchanged() -> None:
    profile = RESOURCE_PROFILES[ACCEPTED]
    assert profile.status is ProfileStatus.ACCEPTED and profile.criteria_passed is True
    assert (profile.peak_bytes, profile.steady_max_bytes, profile.face_hz, profile.face_errors) == (
        4_297_113_600, 4_297_113_600, 1.0, 0)
    assert CANDIDATE_RUNTIME_LIMITATION in profile.limitations
    assert "runs no face model and releases no face files" in CANDIDATE_RUNTIME_LIMITATION
    assert not hasattr(profile, "face_admission") and "face_status" not in {f.name for f in dataclasses.fields(profile)}


def test_the_face_record_is_separate_pending_and_names_the_profiled_weights() -> None:
    admission = FACE_ADMISSIONS[ACCEPTED]
    assert admission.status is FaceAdmissionStatus.PENDING_VALIDATION and admission.face_hz == 1.0
    assert (admission.model_weights.name, admission.model_weights.sha256) == MODEL_WEIGHTS
    assert (admission.detector_weights.name, admission.detector_weights.sha256) == DETECTOR_WEIGHTS
    assert (admission.model_weights.bytes, admission.detector_weights.bytes) == (94_955_648, 232_589)
    assert admission.deepface_version == DEEPFACE_VERSION and admission.limitation == FACE_RUNTIME_LIMITATION
    for gap in ("live camera faces", "identity correctness", "the runtime's memory with face", "no enrolled identities"):
        assert gap in FACE_RUNTIME_LIMITATION
    assert set(FACE_ADMISSIONS) == {ACCEPTED}


def test_pending_validation_admits_only_a_validation_run() -> None:
    assert "pending validation" in face_admission_problem(ACCEPTED, validation_run=False)
    assert face_admission_problem(ACCEPTED, validation_run=True) is None
    (status,) = statuses(ACCEPTED)
    assert status.state is AdapterState.ENABLED and "--face-validation runs only" in status.reason


@pytest.mark.parametrize("profile", [None, "provisional-demo-20261003T085010Z", "no-such-profile"])
def test_a_profile_alone_never_admits_face(profile) -> None:
    assert face_admission_problem(profile, validation_run=True) is not None
    (status,) = statuses(profile)
    assert status.state is AdapterState.UNAVAILABLE


def test_an_accepted_profile_without_a_face_record_does_not_admit_face() -> None:
    assert "no face admission record" in face_admission_problem(ACCEPTED, validation_run=True, admissions={})
    (status,) = statuses(ACCEPTED, face_admissions={})
    assert status.state is AdapterState.UNAVAILABLE


def test_a_validated_record_admits_without_the_flag() -> None:
    validated = {ACCEPTED: dataclasses.replace(FACE_ADMISSIONS[ACCEPTED], status=FaceAdmissionStatus.VALIDATED)}
    assert face_admission_problem(ACCEPTED, validation_run=False, admissions=validated) is None
    (status,) = statuses(ACCEPTED, face_admissions=validated)
    assert (status.state, status.reason) == (AdapterState.ENABLED, "enabled")
