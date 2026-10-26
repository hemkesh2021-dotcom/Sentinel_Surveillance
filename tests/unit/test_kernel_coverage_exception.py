"""D105: a kernel-coverage exception substitutes for ONE profiled run's own marker proof (K_kernel), never as a rule.

Profiles and exceptions here are SYNTHETIC (the ``accepted_scene`` fixture, tests/unit/conftest.py, and the records
built below), passed explicitly; none is, or may become, a runtime record. The step-4 tooling's own check is unchanged:
fed the measured run's marker times, ``kernel_evidence()`` still reports the coverage as uncertain.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path

import pytest

from sentinel import adapters
from sentinel.adapters import (
    KERNEL_COVERAGE_EXCEPTIONS,
    KernelCoverageException,
    ProfileStatus,
    accepted_profile_problem,
)

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"


def exception_for(profile: adapters.ResourceProfile, **overrides: object) -> KernelCoverageException:
    record = KernelCoverageException(
        profile_id=profile.profile_id, run_dir=profile.run_dir, boot_id=profile.boot_id,
        recorded_kernel_status="uncertain", recorded_reasons=("run markers do not bracket the interval",),
        recorded_verdict="SYNTHETIC: not eligible: K_kernel; LCd_exit=1",
        uncovered_utc=("2099-12-31T23:59:57.000000Z", "2099-12-31T23:59:58.119178Z"),
        evidence="SYNTHETIC", evidence_sha256=(("SHA256SUMS", "a" * 64),),
        oom_candidates=0, nvmap_candidates=0, decision="SYNTHETIC", limitations="SYNTHETIC")
    return dataclasses.replace(record, **overrides)


def problem(profile: adapters.ResourceProfile, exceptions: dict) -> str | None:
    return accepted_profile_problem(profile.profile_id, {profile.profile_id: profile}, exceptions)


def test_unobserved_coverage_without_its_own_exception_is_refused_as_before(accepted_scene) -> None:
    profile = accepted_scene(kernel_coverage="uncertain").profile
    refusal = f"resource profile {profile.profile_id} has no observed kernel-log coverage"
    assert problem(profile, {}) == refusal
    assert accepted_profile_problem(profile.profile_id, {profile.profile_id: profile}) == refusal  # the runtime's
    for status in ("unavailable", "truncated", None):
        assert problem(accepted_scene(kernel_coverage=status).profile, {}) == refusal


def test_the_profiles_own_exception_covers_exactly_its_recorded_coverage(accepted_scene) -> None:
    profile = accepted_scene(kernel_coverage="uncertain").profile
    assert problem(profile, {profile.profile_id: exception_for(profile)}) is None


@pytest.mark.parametrize("key, change, reason", [
    ("other", {}, "has no observed kernel-log coverage"),
    (None, {"profile_id": "other"}, "has no observed kernel-log coverage"),
    (None, {"run_dir": "demo-profile-20990101T000000Z"}, "'s kernel-coverage exception names another run"),
    (None, {"boot_id": "00000000-0000-4000-8000-000000000001"}, "'s kernel-coverage exception names another run"),
    (None, {"recorded_kernel_status": "observed"}, "does not record the run's own verdict"),
    (None, {"recorded_reasons": ()}, "does not record the run's own verdict"),
    (None, {"recorded_verdict": ""}, "does not record the run's own verdict"),
    (None, {"recorded_kernel_status": "unavailable"}, "records kernel coverage uncertain, not the unavailable"),
    (None, {"evidence": ""}, "lacks its evidence, checksums or decision"),
    (None, {"evidence_sha256": ()}, "lacks its evidence, checksums or decision"),
    (None, {"evidence_sha256": (("SHA256SUMS", "not-a-digest"),)}, "lacks its evidence, checksums or decision"),
    (None, {"evidence_sha256": (("", "a" * 64),)}, "lacks its evidence, checksums or decision"),
    (None, {"decision": ""}, "lacks its evidence, checksums or decision"),
    (None, {"oom_candidates": 1}, "does not establish 0 OOM and 0 NvMap candidates"),
    (None, {"nvmap_candidates": 2}, "does not establish 0 OOM and 0 NvMap candidates"),
])
def test_an_exception_that_does_not_fit_its_profile_admits_nothing(accepted_scene, key, change, reason) -> None:
    profile = accepted_scene(kernel_coverage="uncertain").profile
    found = problem(profile, {key or profile.profile_id: exception_for(profile, **change)})
    assert found is not None and reason in found, found


@pytest.mark.parametrize("change, reason", [
    ({"steady_max_bytes": 5_100_000_000}, "steady max 5100000000 exceeds"),
    ({"oom_candidates": 1}, "OOM candidate lines 1 exceeds 0"),
    ({"nvmap_candidates": None}, "lacks its recorded NvMap candidate lines"),
    ({"identity_status": "changed"}, "has no verified identity"),
    ({"status": ProfileStatus.PENDING}, "is pending, not accepted"),
    ({"criteria_passed": None}, "has no recorded pass"),
    ({"scene_valid": 100}, "strictly valid scene reports"),
])
def test_an_exception_never_covers_any_other_check(accepted_scene, change, reason) -> None:
    profile = accepted_scene(kernel_coverage="uncertain", **change).profile
    found = problem(profile, {profile.profile_id: exception_for(dataclasses.replace(profile, oom_candidates=0,
                                                                                    nvmap_candidates=0))})
    assert found is not None and reason in found, found


def test_exceptions_are_static_records_that_runtime_code_cannot_add_or_change() -> None:
    assert all(key == record.profile_id for key, record in KERNEL_COVERAGE_EXCEPTIONS.items())
    with pytest.raises(TypeError):
        KERNEL_COVERAGE_EXCEPTIONS["x"] = None  # type: ignore[index]
    record = exception_for(adapters.RESOURCE_PROFILES["step4cand-demo-20261007T090339Z"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.oom_candidates = 0  # type: ignore[misc]


# ---------------------------------------------------------------- the step-4 tooling's own check is unchanged


class Completed:
    def __init__(self, lines: list[dict]) -> None:
        self.status, self.returncode, self.cleanup_clear, self.errors = "completed", 0, True, b""
        self.output = "".join(json.dumps(line) + "\n" for line in lines).encode()


class Journal:
    """journalctl stand-in: the run's two markers, two kernel lines inside the window, no journald loss lines."""

    def __init__(self, start_us: int, end_us: int, tag: str) -> None:
        self.markers = [{"MESSAGE": f"step4cand start {tag}", "__REALTIME_TIMESTAMP": str(start_us)},
                        {"MESSAGE": f"step4cand end {tag}", "__REALTIME_TIMESTAMP": str(end_us)}]

    def run(self, argv: list[str], timeout: float, capture_stderr: bool = False) -> Completed:
        if "-k" in argv:
            return Completed([{"MESSAGE": "block nvme1n1: No UUID available providing old NGUID",
                               "__REALTIME_TIMESTAMP": "1793008472802031"}])
        return Completed(self.markers if "-t" in argv else [])


@pytest.mark.parametrize("start_us, status, reasons", [
    (1_793_008_048_119_178, "uncertain", ["run markers do not bracket the interval"]),  # the measured run's marker
    (1_793_008_047_900_000, "observed", []),  # 0.1 s before the start's second: as before, observed
])
def test_the_tooling_still_requires_the_runs_own_markers_to_bracket_its_window(monkeypatch, start_us, status,
                                                                               reasons) -> None:
    """The measured run's start marker (09:47:28.119178Z, the profiler's start truncated to 09:47:28Z) still gives
    "uncertain" with no counts: D105 changed the registry, not the K criterion or the tooling."""
    monkeypatch.syspath_prepend(str(RUNNERS))
    operator = importlib.import_module("operator_check")
    boot = "6745ddbf-fc39-4700-a820-6ee8e825caad"
    backend = type("Backend", (), {"boot_id": staticmethod(lambda: boot)})()
    manifest = {"boot_id": boot, "started_utc": "2026-10-26T09:47:28Z", "finished_utc": "2026-10-26T10:01:59Z"}
    tag = "sentinel-operator-tvw_tug2"
    kernel = operator.kernel_evidence(Journal(start_us, 1_793_008_982_900_963, tag), backend, manifest, tag, True,
                                      marker_tag=operator.STEP4CAND_MARKER_TAG, marker_prefix="step4cand")
    assert (kernel["status"], kernel["reasons"]) == (status, reasons)
    if status == "observed":
        assert (kernel["oom_candidates"], kernel["nvmap_candidates"]) == (0, 0)
    else:  # unavailable is never zero: lower bounds only
        assert (kernel["oom_candidates"], kernel["nvmap_candidates"]) == (None, None)
        assert (kernel["oom_candidates_lower_bound"], kernel["nvmap_candidates_lower_bound"]) == (0, 0)
