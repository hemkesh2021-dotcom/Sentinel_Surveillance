"""D105: the accepted LFM2.5-VL resource profile admits scene analysis only for the configuration it measured.

The first part is portable: the registry's real entry against configurations that differ before any file is looked
at (memory policy, scene interval, request fingerprint, declared model revision, the other profile). The second part
needs this device's real files (the b8932 build and both model sets) and is skipped elsewhere: the measured files are
admitted, and a swapped model, projector, binary or engine is refused. Nothing is started: no server, model or camera;
only metadata is read, and files up to 32,000,000 B (the llama.cpp build) are hashed, as at startup.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest

from sentinel.adapters import RESOURCE_PROFILES
from sentinel.config import parse_config
from sentinel.demo_runtime import SceneOptions, profile_mismatch, scene_admission
from sentinel.memory_policy import CANDIDATE_POLICY, DEFAULT_POLICY
from sentinel.scene import models

LFM25 = "step4cand-lfm25-20261026T094728Z"  # accepted under D105
LFM2 = "step4cand-demo-20261007T090339Z"  # accepted under D59
NEW, OLD = models.SCENE_MODEL_SETS["lfm2.5-vl"], models.SCENE_MODEL_SETS["lfm2-vl"]
BINARY = Path.home() / "llama.cpp/build/bin/llama-server"


def config(profile_id: str = LFM25, revision: str | None = NEW.model_revision, interval_s: float = 4.0):
    manifest: dict[str, Any] = {"adapter_id": "llama-lfm2-vl-scene", "contract_version": 1,
                                "implementation_revision": "1", "enabled": True, "input_kinds": ["frame"],
                                "output_kinds": ["scene.report"], "resource_profile_id": profile_id, "timeout_ms": 8000}
    if revision is not None:
        manifest["model_revision"] = revision
    return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "scene": {"interval_s": interval_s},
                         "adapters": [manifest]})


def options(model_set: models.SceneModelSet = NEW, **paths: Path) -> SceneOptions:
    return SceneOptions(paths.get("binary", BINARY), paths.get("model", model_set.model_path),
                        paths.get("mmproj", model_set.mmproj_path), model_set=model_set.key)


def admission(cfg=None, scene=None, policy=CANDIDATE_POLICY) -> tuple[str | None, str | None]:
    return scene_admission(cfg or config(), scene or options(), policy=policy)


# ---------------------------------------------------------------- portable: the configuration, before any file


def test_the_entry_is_the_default_sets_measurement_and_the_rollback_keeps_its_own() -> None:
    lfm25, lfm2 = RESOURCE_PROFILES[LFM25], RESOURCE_PROFILES[LFM2]
    assert (NEW.model, NEW.mmproj, NEW.scene_request_sha256) == (lfm25.llm, lfm25.mmproj, lfm25.scene_request_sha256)
    assert (OLD.model, OLD.mmproj, OLD.scene_request_sha256) == (lfm2.llm, lfm2.mmproj, lfm2.scene_request_sha256)
    assert models.DEFAULT_SCENE_MODEL_SET == NEW.key and models.ROLLBACK_SCENE_MODEL_SET == OLD.key
    assert NEW.profiled and OLD.profiled
    # the same build, flags, engine and memory policy; only the model set differs
    for field in ("llama_flags", "cache_ram_mib", "scene_interval_s", "engine_sha256", "llama_server",
                  "llama_libraries", "memory_policy", "criteria_id"):
        assert getattr(lfm25, field) == getattr(lfm2, field), field


@pytest.mark.parametrize("cfg, scene, policy, reason", [
    (None, None, DEFAULT_POLICY, f"resource profile {LFM25} measured memory policy"),
    (config(interval_s=5.0), None, CANDIDATE_POLICY, f"resource profile {LFM25} measured a 4 s scene interval"),
    (config(revision=OLD.model_revision), options(OLD), CANDIDATE_POLICY,  # the rollback set on the new profile
     f"resource profile {LFM25} measured another scene request than the runtime sends"),
    (config(profile_id=LFM2), None, CANDIDATE_POLICY,  # the default set on the old profile
     f"resource profile {LFM2} measured another scene request than the runtime sends"),
    (config(revision=OLD.model_revision), None, CANDIDATE_POLICY,
     "the manifest declares model_revision lfm2-vl-1.6b-q4_0, not lfm2.5-vl-1.6b-q4_0"),
    (config(profile_id="step4cand-lfm25-20261026T094626Z"), None, CANDIDATE_POLICY,  # the session folder's name
     "resource profile step4cand-lfm25-20261026T094626Z"),
])
def test_a_configuration_the_profile_did_not_measure_is_refused(cfg, scene, policy, reason) -> None:
    problem, revision = admission(cfg, scene, policy)
    assert problem is not None and reason in problem and revision is None, problem


def test_the_profile_needs_its_own_kernel_exception() -> None:
    from sentinel import adapters

    without = {pid: p for pid, p in adapters.KERNEL_COVERAGE_EXCEPTIONS.items() if pid != LFM25}
    assert adapters.accepted_profile_problem(LFM25, kernel_exceptions=without) == (
        f"resource profile {LFM25} has no observed kernel-log coverage")
    moved = {LFM2: dataclasses.replace(adapters.KERNEL_COVERAGE_EXCEPTIONS[LFM25], profile_id=LFM2)}
    assert adapters.accepted_profile_problem(LFM25, kernel_exceptions=moved) == (
        f"resource profile {LFM25} has no observed kernel-log coverage")


# ---------------------------------------------------------------- this device's real files


REAL = all(path.exists() for path in (BINARY, NEW.model_path, NEW.mmproj_path, OLD.model_path, OLD.mmproj_path))
device = pytest.mark.skipif(not REAL, reason="needs this device's llama.cpp build and both scene model sets")


@device
def test_the_measured_files_are_admitted_on_the_default_set() -> None:
    assert admission() == (None, config().adapters[0].producer_revision)


@device
def test_the_rollback_stays_admitted_on_its_own_profile() -> None:
    assert admission(config(profile_id=LFM2, revision=OLD.model_revision), options(OLD))[0] is None


def sparse(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)  # a sparse stand-in: no data written
    return path


@device
@pytest.mark.parametrize("swap, reason", [
    ({"model": OLD.model_path}, "scene model LFM2-VL-1.6B-Q4_0.gguf (695750048 B"),
    ({"mmproj": OLD.mmproj_path}, "scene projector mmproj-LFM2-VL-1.6B-Q8_0.gguf (564115648 B"),
    ("model-other-size", "scene model LFM2.5-VL-1.6B-Q4_0.gguf (1 B"),
    ("model-other-time", "is not the profiled LFM2.5-VL-1.6B-Q4_0.gguf (695752480 B, 2026-10-25T20:33:16+00:00)"),
    ("mmproj-other-size", "scene projector mmproj-LFM2.5-VL-1.6b-Q8_0.gguf (1 B"),
    ("binary", "llama-server llama-server does not have the profiled SHA-256"),
])
def test_a_swapped_model_projector_or_binary_is_refused(tmp_path, swap, reason) -> None:
    if swap == "model-other-size":
        swap = {"model": sparse(tmp_path / NEW.model.name, 1)}
    elif swap == "model-other-time":  # the measured name and size, another modification time
        stand_in = sparse(tmp_path / NEW.model.name, NEW.model.bytes)
        os.utime(stand_in, (1_700_000_000, 1_700_000_000))
        swap = {"model": stand_in}
    elif swap == "mmproj-other-size":
        swap = {"mmproj": sparse(tmp_path / NEW.mmproj.name, 1)}
    elif swap == "binary":
        stand_in = tmp_path / "llama-server"
        stand_in.write_bytes(b"\0" * RESOURCE_PROFILES[LFM25].llama_server.bytes)  # the measured size, other bytes
        swap = {"binary": stand_in}
    problem, revision = admission(scene=options(**swap))
    assert problem is not None and reason in problem and revision is None, problem


@device
def test_another_detector_engine_is_refused() -> None:
    problem = profile_mismatch(RESOURCE_PROFILES[LFM25], config(), options(), engine_sha256="f" * 64,
                               policy=CANDIDATE_POLICY)
    assert problem == f"resource profile {LFM25} measured another detector engine than the pinned one"
