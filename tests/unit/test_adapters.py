from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from sentinel.adapters import (
    AdapterManifest,
    AdapterOutputError,
    AdapterRole,
    AdapterSpec,
    AdapterState,
    evidence_from_adapter,
    load,
    resolve,
)
from sentinel.cli import main
from sentinel.config import ConfigError, load_config, parse_config
from sentinel.contracts import Evidence, EvidenceStatus, FrameRef, PixelFormat
from sentinel.live_state import Capability, Occupancy, SceneStatus
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.runtime import EdgeCore

SCENE = {
    "adapter_id": "demo-scene",
    "contract_version": 1,
    "implementation_revision": "1",
    "enabled": True,
    "input_kinds": ["frame"],
    "output_kinds": ["scene.report"],
    "model_revision": "lfm2-vl-1.6b-q4_0",
    "resource_profile_id": "orin-nano-8gb-scene-small",
    "timeout_ms": 8000,
}
REGISTRY = {
    "demo-scene": AdapterSpec(
        adapter_id="demo-scene",
        role=AdapterRole.SCENE_ANALYZER,
        module="sentinel_probe_module_that_does_not_exist",
        attribute="Analyzer",
        input_kinds=frozenset({"frame"}),
        output_kinds=frozenset({"scene.report"}),
    ),
    "demo-telegram": AdapterSpec(
        adapter_id="demo-telegram",
        role=AdapterRole.NOTIFIER,
        module="json",
        attribute="dumps",
        input_kinds=frozenset({"incident.alert"}),
        output_kinds=frozenset({"delivery.result"}),
    ),
}
PROFILES = frozenset({"orin-nano-8gb-scene-small"})


def manifest(**changes: object) -> AdapterManifest:
    return AdapterManifest.model_validate({**SCENE, **changes})


def config_problems(adapters: list[dict]) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": adapters})
    return caught.value.problems


def test_manifests_load_from_yaml_with_the_guide_fields(tmp_path: Path) -> None:
    path = tmp_path / "sentinel.yaml"
    path.write_text(
        "config_version: 1\ncamera: {id: cam-1}\nadapters:\n"
        "  - {adapter_id: demo-scene, contract_version: 1, implementation_revision: '1',\n"
        "     input_kinds: [frame], output_kinds: [scene.report], timeout_ms: 8000}\n",
        encoding="utf-8",
    )
    (loaded,) = load_config(path).adapters
    assert (loaded.enabled, loaded.priority, loaded.model_revision) == (False, 50, None)
    assert loaded.producer_revision == "1"
    assert manifest().producer_revision == "1+lfm2-vl-1.6b-q4_0"


def test_unknown_major_versions_and_duplicate_ids_fail_configuration() -> None:
    assert config_problems([{**SCENE, "contract_version": 2}]) == [
        "adapters.0.contract_version: unsupported adapter contract version 2; this build supports 1"
    ]
    assert config_problems([SCENE, {**SCENE, "enabled": False}]) == [
        "adapters: duplicate adapter_id 'demo-scene'"
    ]
    assert config_problems([{**SCENE, "contract_version": "1"}]) == [
        "adapters.0.contract_version: Input should be a valid integer (got '1')"
    ]


def test_a_typo_inside_an_adapter_entry_is_named_with_a_suggestion() -> None:
    entry = {k: v for k, v in SCENE.items() if k != "timeout_ms"} | {"timout_ms": 8000}
    assert sorted(config_problems([entry])) == [
        "adapters.0.timeout_ms: required setting is missing",
        "adapters.0.timout_ms: unknown setting (did you mean 'timeout_ms'?)",
    ]


def test_each_adapter_resolves_to_a_state_with_a_visible_reason() -> None:
    notifier = manifest(
        adapter_id="demo-telegram",
        input_kinds=["incident.alert"],
        output_kinds=["delivery.result"],
        model_revision=None,
        resource_profile_id=None,
    )
    cases = [
        (manifest(enabled=False), AdapterState.DISABLED, "disabled in configuration"),
        (manifest(adapter_id="other"), AdapterState.UNAVAILABLE, "not a built-in adapter"),
        (
            manifest(output_kinds=["scene.report", "scene.caption"]),
            AdapterState.UNAVAILABLE,
            "implementation does not handle kinds: scene.caption",
        ),
        (
            manifest(resource_profile_id="unmeasured"),
            AdapterState.UNAVAILABLE,
            "no measured resource profile; unknown profiles are unavailable",
        ),
        (manifest(resource_profile_id=None), AdapterState.UNAVAILABLE, "no measured resource profile"),
        (notifier, AdapterState.ENABLED, "enabled"),  # notifiers load no model
        (manifest(), AdapterState.ENABLED, "enabled"),
    ]
    statuses = resolve([m for m, _, _ in cases], registry=REGISTRY, known_profiles=PROFILES)
    for status, (_, state, reason) in zip(statuses, cases, strict=True):
        assert status.state is state and status.reason.startswith(reason)

    old = {"demo-scene": AdapterSpec(**{**vars(REGISTRY["demo-scene"]), "contract_versions": frozenset()})}
    (status,) = resolve([manifest()], registry=old, known_profiles=PROFILES)
    assert status.reason == "implementation does not support contract version 1"


def test_resolving_imports_nothing_and_a_failed_load_only_disables_that_adapter() -> None:
    # The scene spec names a module that does not exist: resolving must not notice.
    scene, notifier = resolve(
        [manifest(), manifest(adapter_id="demo-telegram", input_kinds=["incident.alert"],
                              output_kinds=["delivery.result"], resource_profile_id=None)],
        registry=REGISTRY,
        known_profiles=PROFILES,
    )
    assert scene.state is AdapterState.ENABLED and notifier.state is AdapterState.ENABLED
    implementation, after = load(scene)
    assert implementation is None and after.state is AdapterState.UNAVAILABLE
    assert after.reason == (
        "failed to load: ModuleNotFoundError: "
        "No module named 'sentinel_probe_module_that_does_not_exist'"
    )
    implementation, after = load(notifier)
    assert implementation is json.dumps and after is notifier
    disabled = resolve([manifest(enabled=False)], registry=REGISTRY)[0]
    assert load(disabled) == (None, disabled)


def adapter_evidence(frame: FrameRef, **changes: object) -> str:
    evidence = Evidence.observed_on(
        frame,
        evidence_id="demo-scene-1.observed",
        kind="scene.report",
        status=EvidenceStatus.OBSERVED,
        producer="demo-scene",
        producer_revision="1+lfm2-vl-1.6b-q4_0",
        ttl_ns=10 * NS_PER_SECOND,
        correlation_group="demo-scene",
        reason="scene report",
        value={"summary": "empty room"},
    )
    data = json.loads(evidence.model_dump_json())
    return json.dumps({**data, **changes})


def test_adapter_evidence_is_parsed_strictly_and_checked_against_its_manifest(
    next_frame: Callable[..., FrameRef],
) -> None:
    frame = next_frame()
    (enabled,) = resolve([manifest()], registry=REGISTRY, known_profiles=PROFILES)
    assert evidence_from_adapter(enabled, adapter_evidence(frame)).producer == "demo-scene"

    rejected = [
        (enabled, adapter_evidence(frame, contract_version=2), "invalid evidence: contract_version: literal_error"),
        (enabled, adapter_evidence(frame, kind="scene.caption"), "kind 'scene.caption' is not declared"),
        (enabled, adapter_evidence(frame, producer="other"), "producer 'other' is not adapter"),
        (enabled, adapter_evidence(frame, producer_revision="2"), "producer revision does not match"),
        (enabled, adapter_evidence(frame, surprise=1), "invalid evidence: surprise: extra_forbidden"),
        (enabled, '{"token=s3cret" ', "invalid evidence: <evidence>: json_invalid"),
    ]
    disabled = resolve([manifest(enabled=False)], registry=REGISTRY)[0]
    rejected.append((disabled, adapter_evidence(frame), "adapter 'demo-scene' is disabled"))
    for status, payload, message in rejected:
        with pytest.raises(AdapterOutputError) as caught:
            evidence_from_adapter(status, payload)
        assert str(caught.value).startswith(message)
        assert "s3cret" not in str(caught.value)


def test_core_monitoring_runs_with_every_optional_adapter_disabled() -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    assert config.adapters == []
    clock = FakeClock()
    core = EdgeCore(config, clock, None)  # no scene analyzer, no face recognition
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    for _ in range(90):  # 6 s of empty frames: past several scene intervals
        clock.advance(1 / 15)
        frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        state = core.on_frame(frame, [], stamper.current_stream).state
    assert (state.occupancy, state.scene, state.scene_analysis, state.face_recognition) == (
        Occupancy.EMPTY,
        SceneStatus.NO_CURRENT_RESULT,
        Capability.DISABLED,
        Capability.DISABLED,
    )
    assert state.scene_reason == "scene analysis disabled" and core.lane is None
    assert core.diagnostics()["scene_jobs_in_flight"] == 0
    # Scene-worker calls are harmless no-ops rather than errors.
    assert core.request_enrichment(frame, "inc-1", stamper.current_stream).evidence == ()


def test_config_validate_reports_each_adapter_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "sentinel.yaml"
    path.write_text(
        "config_version: 1\ncamera: {id: cam-1}\nadapters:\n"
        "  - {adapter_id: llama-scene, contract_version: 1, implementation_revision: '1', enabled: true,\n"
        "     input_kinds: [frame], output_kinds: [scene.report], timeout_ms: 8000}\n"
        "  - {adapter_id: telegram, contract_version: 1, implementation_revision: '1',\n"
        "     input_kinds: [incident.alert], output_kinds: [delivery.result], timeout_ms: 5000}\n",
        encoding="utf-8",
    )
    assert main(["config", "validate", str(path)]) == 0  # optional adapters never block startup
    lines = capsys.readouterr().out.splitlines()
    assert lines[1:] == [
        "  adapter llama-scene: unavailable (not a built-in adapter)",
        "  adapter telegram: disabled (disabled in configuration)",
    ]
