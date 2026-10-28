"""D106 (2026-10-28): the LFM3-VL resource profile admits scene analysis only for the configuration it measured.

Portable first: the registry's entry is the run's record (generated from its manifest, profile, step-4 report and
identity snapshot), passes the registry's acceptance rules, carries its limitations (the guarded runs' pressure stop
26.8 MB above the sampled peak, the untagged build with its log level, the NVRM lines), and refuses configurations it
did not measure. Then, on this device only (skipped elsewhere): the measured files and build are admitted, and a
swapped model, projector or binary is refused. Nothing is started: no server, model or camera.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from sentinel.adapters import (
    CANDIDATE_RUNTIME_LIMITATION,
    LFM3_BUILD_LIMITATION,
    LFM3_MEMORY_MARGIN_LIMITATION,
    LFM3_NVRM_OBSERVATION,
    RESOURCE_PROFILES,
    STARTUP_IDENTITY_LIMITATION,
    accepted_profile_problem,
)
from sentinel.config import parse_config
from sentinel.demo_runtime import SceneOptions, profile_mismatch, scene_admission
from sentinel.memory_policy import CANDIDATE_POLICY, DEFAULT_POLICY
from sentinel.scene import models
from sentinel.scene.llama_server import PROFILED_FLAGS, PROMPT_CACHE_FLAGS

LFM3_PROFILE = "step4cand-lfm3-20261028T111716Z"
LFM3 = models.SCENE_MODEL_SETS["lfm3-vl"]
PROFILE = RESOURCE_PROFILES[LFM3_PROFILE]
GUARD_PRESSURE_STOP = 4_800_000_000  # operator_check.PRESSURE_STOP, the guarded runs' stop


def config(profile_id: str = LFM3_PROFILE, revision: str | None = LFM3.model_revision, interval_s: float = 4.0):
    manifest: dict[str, Any] = {"adapter_id": "llama-lfm2-vl-scene", "contract_version": 1,
                                "implementation_revision": "1", "enabled": True, "input_kinds": ["frame"],
                                "output_kinds": ["scene.report"], "resource_profile_id": profile_id, "timeout_ms": 8000}
    if revision is not None:
        manifest["model_revision"] = revision
    return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "scene": {"interval_s": interval_s},
                         "adapters": [manifest]})


def options(**paths: Path) -> SceneOptions:
    return SceneOptions(paths.get("binary", LFM3.llama_server), paths.get("model", LFM3.model_path),
                        paths.get("mmproj", LFM3.mmproj_path), model_set=LFM3.key)


def test_the_entry_is_the_runs_record_and_passes_the_registry_rules() -> None:
    assert accepted_profile_problem(LFM3_PROFILE) is None
    assert PROFILE.run_dir == "demo-profile-20261028T111716Z"
    assert PROFILE.commit == "595e1df090531ef380d31c542d69ee1740662455"
    assert PROFILE.boot_id == "c8c02fc5-71d8-467d-82a9-6b42f5ef71a0"
    assert PROFILE.llama_flags == PROFILED_FLAGS + PROMPT_CACHE_FLAGS + LFM3.server_flags
    assert (PROFILE.llm, PROFILE.mmproj) == (LFM3.model, LFM3.mmproj)
    assert PROFILE.scene_request_sha256 == LFM3.scene_request_sha256
    assert (PROFILE.steady_max_bytes, PROFILE.peak_bytes) == (4_771_803_136, 4_773_228_544)
    assert GUARD_PRESSURE_STOP - PROFILE.peak_bytes == 26_771_456  # the observed margin, recorded as a limitation
    assert PROFILE.kernel_coverage == "observed" and (PROFILE.oom_candidates, PROFILE.nvmap_candidates) == (0, 0)
    assert PROFILE.llama_server.name == "llama-server" and PROFILE.llama_server.bytes == 72_456
    assert "libllama-server-impl.so" in {lib.name for lib in PROFILE.llama_libraries}
    assert len(PROFILE.llama_libraries) == 8
    assert PROFILE.memory_policy == CANDIDATE_POLICY and PROFILE.memory_policy_verified is True
    for limitation in (STARTUP_IDENTITY_LIMITATION, CANDIDATE_RUNTIME_LIMITATION, LFM3_MEMORY_MARGIN_LIMITATION,
                       LFM3_BUILD_LIMITATION, LFM3_NVRM_OBSERVATION):
        assert limitation in PROFILE.limitations
    assert "26,771,456 B" in LFM3_MEMORY_MARGIN_LIMITATION and "580.178.04" in LFM3_NVRM_OBSERVATION
    assert LFM3.profiled is True and models.DEFAULT_SCENE_MODEL_SET == LFM3.key


def test_the_other_profiles_are_unchanged_and_never_admit_lfm3() -> None:
    assert RESOURCE_PROFILES["step4cand-lfm25-20261026T094728Z"].scene_request_sha256 != LFM3.scene_request_sha256
    assert RESOURCE_PROFILES["step4cand-demo-20261007T090339Z"].scene_request_sha256 != LFM3.scene_request_sha256
    problem, _ = scene_admission(config(profile_id="step4cand-lfm25-20261026T094728Z"), options(),
                                 policy=CANDIDATE_POLICY)
    assert problem is not None


@pytest.mark.parametrize("cfg, policy, reason", [
    (None, DEFAULT_POLICY, f"resource profile {LFM3_PROFILE} measured memory policy"),
    (config(interval_s=5.0), CANDIDATE_POLICY, f"resource profile {LFM3_PROFILE} measured a 4 s scene interval"),
    (config(revision="lfm2.5-vl-1.6b-q4_0"), CANDIDATE_POLICY, "the manifest declares model_revision"),
])
def test_a_configuration_the_profile_did_not_measure_is_refused(cfg, policy, reason) -> None:
    problem, revision = scene_admission(cfg or config(), options(), policy=policy)
    assert problem is not None and reason in problem and revision is None, problem


def test_a_profile_without_the_log_level_does_not_admit_the_set(tmp_path) -> None:
    import dataclasses

    without = dataclasses.replace(PROFILE, llama_flags=PROFILED_FLAGS + PROMPT_CACHE_FLAGS)
    problem = profile_mismatch(without, config(), options(), policy=CANDIDATE_POLICY)
    assert problem is not None and "measured llama-server flags" in problem


REAL = all(p.exists() for p in (LFM3.llama_server, LFM3.model_path, LFM3.mmproj_path))
device = pytest.mark.skipif(not REAL, reason="needs this device's LFM3 build and model files")


@device
def test_the_measured_files_and_build_are_admitted() -> None:
    assert scene_admission(config(), options(), policy=CANDIDATE_POLICY) == (None, config().adapters[0].producer_revision)


def sparse(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


@device
@pytest.mark.parametrize("swap, reason", [
    ("model-other-size", "scene model LFM3-VL-2B-QAD-Q4_0.gguf (1 B"),
    ("model-other-time", "is not the profiled LFM3-VL-2B-QAD-Q4_0.gguf (1049615648 B, 2026-10-27T20:12:38+00:00)"),
    ("mmproj-other-size", "scene projector mmproj-LFM3-VL-2B-Q8_0.gguf (1 B"),
    ("binary", "llama-server llama-server does not have the profiled SHA-256"),
    ("b8932", "llama-server llama-server does not have the profiled SHA-256"),
])
def test_a_swapped_model_projector_or_binary_is_refused(tmp_path, swap, reason) -> None:
    if swap == "model-other-size":
        paths = {"model": sparse(tmp_path / LFM3.model.name, 1)}
    elif swap == "model-other-time":
        stand_in = sparse(tmp_path / LFM3.model.name, LFM3.model.bytes)
        os.utime(stand_in, (1_700_000_000, 1_700_000_000))
        paths = {"model": stand_in}
    elif swap == "mmproj-other-size":
        paths = {"mmproj": sparse(tmp_path / LFM3.mmproj.name, 1)}
    elif swap == "binary":
        stand_in = tmp_path / "llama-server"
        stand_in.write_bytes(b"\0" * PROFILE.llama_server.bytes)  # the measured size, other bytes
        paths = {"binary": stand_in}
    else:  # the b8932 build the other sets use
        paths = {"binary": models.LLAMA_SERVER_B8932}
    problem, revision = scene_admission(config(), options(**paths), policy=CANDIDATE_POLICY)
    assert problem is not None and reason in problem and revision is None, problem
