"""D46: `sentinel run --scene` needs an accepted combined profile measured with --cache-ram 0 that
matches the selected configuration, checked before the database opens or any scene process starts.

Accepted profiles here are the SYNTHETIC `accepted_scene` fixture (tests/unit/conftest.py), passed
explicitly; none of them is, or may become, a runtime acceptance record.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel import adapters
from sentinel.adapters import (
    RESOURCE_PROFILES,
    STEP4_CRITERIA_ID,
    AdapterManifest,
    AdapterState,
    ProfileStatus,
    accepted_profile_problem,
    known_profile_ids,
    resolve,
)
from sentinel.cli import main
from sentinel.config import parse_config
from sentinel.demo_runtime import Devices, RunOptions, StartupRefused, assemble
from sentinel.media.clock import FakeClock

PROVISIONAL = "provisional-demo-20261003T085010Z"
REPORT = '{"persons_visible": 0}'


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.status = SimpleNamespace(state=SimpleNamespace(value="ready"), problem=None, layers="17/17",
                                      vision_on_gpu=True)

    def devices(self) -> Devices:
        recorder = self

        class Backend:
            def load(self) -> None:
                recorder.calls.append("detector_load")

            def track(self, image: object) -> list:
                return []

            def reset(self) -> None:
                pass

        def scene_server(options: Any, port: int) -> Any:
            recorder.calls.append("scene_server")
            return SimpleNamespace(start=lambda timeout: recorder.status, status=lambda: recorder.status,
                                   stop=lambda grace: True)

        def scene_request(port: int, timeout: float) -> Any:
            recorder.calls.append("scene_request")
            return lambda job, image: None

        return Devices(
            capture_source=lambda config: object(),
            tracker_backend=lambda engine: Backend(),
            scene_server=scene_server,
            scene_request=scene_request,
            meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000},
            notifiers=lambda config: ({}, {}),
        )


def config_with(*adapters_: dict[str, Any], **sections: Any):
    return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": list(adapters_), **sections})


def refused(tmp_path: Path, config, scene, profiles=RESOURCE_PROFILES) -> tuple[str, Recorder]:
    recorder = Recorder()
    data = tmp_path / "data"
    with pytest.raises(StartupRefused) as error:
        assemble(config, RunOptions(data, tmp_path / "x.engine", scene=scene), recorder.devices(), FakeClock(),
                 profiles=profiles)
    assert recorder.calls == []  # no scene server, no scene request worker, no model load
    assert not (data / "sentinel.db").exists()  # refused before the database opens
    label = error.value.label
    assert label.startswith("scene_not_admitted: ")
    assert str(tmp_path) not in label  # file names only, never paths
    return label, recorder


# ---------------------------------------------------------------- default off, core only


def test_scene_is_off_by_default_and_core_runs_without_any_scene_profile(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    provisional = {**fixture.manifest, "resource_profile_id": PROVISIONAL}
    for config in (config_with(), config_with(provisional)):
        recorder = Recorder()
        assembly = assemble(config, RunOptions(tmp_path / str(len(recorder.calls)), tmp_path / "x.engine"),
                            recorder.devices(), FakeClock())
        assert recorder.calls == ["detector_load"]
        assert assembly.runtime.snapshot()["components"]["scene"] == {"state": "disabled", "problem": None}
        assembly.database.close()


def test_the_cli_refuses_scene_without_a_profile_and_starts_no_process(tmp_path, capsys, monkeypatch,
                                                                       accepted_scene) -> None:
    fixture = accepted_scene()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    config = tmp_path / "config.yaml"
    config.write_text(
        "config_version: 1\ncamera: {id: cam-1}\nadapters: [{adapter_id: llama-lfm2-vl-scene, contract_version: 1, "
        "implementation_revision: '1', enabled: true, input_kinds: [frame], output_kinds: [scene.report], "
        f"resource_profile_id: {PROVISIONAL}, timeout_ms: 8000}}]\n"
    )
    loads: list[str] = []
    code = main(
        ["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", "x", "--status-port", "0", "--scene",
         "--llama-server", str(fixture.options.binary), "--scene-model", str(fixture.model),
         "--scene-mmproj", str(fixture.mmproj)],
        capture_source=lambda config: object(),
        tracker_backend=lambda engine: loads.append("tracker") or pytest.fail("the detector was constructed"),
        meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000},
    )
    err = capsys.readouterr().err.strip()
    assert code == 1 and loads == []
    assert err == ("run: scene_not_admitted: llama-lfm2-vl-scene unavailable: resource profile "
                   f"{PROVISIONAL} is provisional (measured with the default prompt cache); scene needs an accepted "
                   "combined profile measured with --cache-ram 0 (operator checklist step 4)")


def test_run_has_no_option_that_bypasses_scene_admission(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    options = set(re.findall(r"--[a-z][a-z-]+", capsys.readouterr().out))
    assert not {o for o in options if re.search(r"allow|force|skip|unsafe|override|profile|admit", o)}


# ---------------------------------------------------------------- rejected profiles


def test_the_check_8_provisional_cache_enabled_profile_is_rejected(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    label, _ = refused(tmp_path, config_with({**fixture.manifest, "resource_profile_id": PROVISIONAL}), fixture.options)
    assert f"resource profile {PROVISIONAL} is provisional (measured with the default prompt cache)" in label


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"status": ProfileStatus.PENDING}, "is pending, not accepted"),
        ({"status": ProfileStatus.FAILED}, "is failed, not accepted"),
        ({"status": ProfileStatus.PROVISIONAL, "cache_ram_mib": 0}, "is provisional (measured with --cache-ram 0)"),
        ({"cache_ram_mib": None}, "was measured with the default prompt cache, not --cache-ram 0"),
        ({"cache_ram_mib": 8192}, "was measured with --cache-ram 8192, not --cache-ram 0"),
        ({"criteria_id": "step4-combined-cache-off-v0"}, f"was not judged against {STEP4_CRITERIA_ID}"),
        ({"criteria_id": None}, f"was not judged against {STEP4_CRITERIA_ID}"),
        ({"criteria_passed": False}, f"has no recorded pass against {STEP4_CRITERIA_ID}"),
        ({"criteria_passed": None}, f"has no recorded pass against {STEP4_CRITERIA_ID}"),
        ({"run_dir": ""}, "lacks its run directory"),
        ({"run_dir": "../demo-profile-20991231T235959Z"}, "lacks its run directory"),
        ({"commit": "d85eb1e"}, "lacks its full commit"),
        ({"boot_id": ""}, "lacks its boot ID"),
        ({"gpu_guard_ok": None}, "has no recorded GPU guard pass"),
        ({"gpu_guard_ok": False}, "has no recorded GPU guard pass"),
        ({"prompt_cache_disabled": None}, "has no record of the prompt cache being disabled"),
        ({"peak_bytes": None}, "lacks its recorded run peak"),
        ({"peak_bytes": 5_400_000_001}, "run peak 5400000001 B exceeds 5400000000 B"),
        ({"steady_p95_bytes": 5_400_000_001}, "steady p95 5400000001 B exceeds 5400000000 B"),
        ({"steady_slope_bytes_per_min": None}, "lacks its recorded steady slope per minute"),
        ({"steady_slope_bytes_per_min": 10_000_001}, "steady slope per minute 10000001 B exceeds 10000000 B"),
        ({"scene_errors": 1}, "lacks a record of zero scene request errors"),
        ({"scene_errors": None}, "lacks a record of zero scene request errors"),
    ],
)
def test_missing_pending_failed_or_insufficient_evidence_is_rejected(tmp_path, accepted_scene, change, reason) -> None:
    fixture = accepted_scene(**change)
    label, _ = refused(tmp_path, config_with(fixture.manifest), fixture.options, fixture.profiles)
    assert f"resource profile {fixture.profile.profile_id} {reason}" in label, label


def test_an_unknown_or_unnamed_profile_and_a_disabled_or_absent_manifest_are_rejected(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    cases = [
        (config_with({**fixture.manifest, "resource_profile_id": "demo-profile-20991231T235959Z"}),
         "resource profile demo-profile-20991231T235959Z is not in the registry"),
        (config_with({**fixture.manifest, "resource_profile_id": None}), "the manifest names no resource_profile_id"),
        (config_with({**fixture.manifest, "enabled": False}), "llama-lfm2-vl-scene disabled: disabled in configuration"),
        (config_with(), "no llama-lfm2-vl-scene adapter in the configuration"),
    ]
    for index, (config, reason) in enumerate(cases):
        label, _ = refused(tmp_path / str(index), config, fixture.options, fixture.profiles)
        assert reason in label, label


def test_a_registry_entry_filed_under_another_id_is_rejected(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    profiles = {"some-other-id": fixture.profile}
    label, _ = refused(tmp_path, config_with({**fixture.manifest, "resource_profile_id": "some-other-id"}),
                       fixture.options, profiles)
    assert "resource profile some-other-id is not in the registry" in label


# ---------------------------------------------------------------- configuration matching


def test_flags_that_differ_from_the_runtime_are_rejected(tmp_path, accepted_scene) -> None:
    flags = ("--n-gpu-layers", "999", "--ctx-size", "1024", "--parallel", "1", "--cache-ram", "0")
    fixture = accepted_scene(llama_flags=flags)
    label, _ = refused(tmp_path, config_with(fixture.manifest), fixture.options, fixture.profiles)
    assert "measured llama-server flags --n-gpu-layers 999 --ctx-size 1024" in label
    assert "not the runtime's --n-gpu-layers 999 --ctx-size 2048 --parallel 1 --cache-ram 0" in label


def test_a_different_scene_interval_is_rejected(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    config = config_with(fixture.manifest, scene={"interval_s": 2.0})
    label, _ = refused(tmp_path, config, fixture.options, fixture.profiles)
    assert label.endswith("measured a 4 s scene interval, not the configured 2 s")


def test_model_files_that_are_not_the_profiled_ones_are_rejected(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    config = config_with(fixture.manifest)
    other = fixture.model.with_name("LFM2-VL-1.6B-Q8_0.gguf")
    other.write_bytes(fixture.model.read_bytes())
    label, _ = refused(tmp_path / "a", config, dataclasses.replace(fixture.options, model=other), fixture.profiles)
    assert "scene model LFM2-VL-1.6B-Q8_0.gguf (64 B," in label and "is not the profiled LFM2-VL-1.6B-Q4_0.gguf" in label
    missing = dataclasses.replace(fixture.options, mmproj=fixture.mmproj.with_name("absent.gguf"))
    label, _ = refused(tmp_path / "b", config, missing, fixture.profiles)
    assert label.endswith("scene projector absent.gguf is missing")
    fixture.mmproj.write_bytes(b"\0" * 33)  # same name, different size
    label, _ = refused(tmp_path / "c", config, fixture.options, fixture.profiles)
    assert "scene projector mmproj-LFM2-VL-1.6B-Q8_0.gguf (33 B," in label
    fixture.mmproj.write_bytes(b"\0" * 32)
    stat = fixture.model.stat()
    os.utime(fixture.model, (stat.st_atime, stat.st_mtime + 60))  # same size, replaced later
    label, _ = refused(tmp_path / "d", config, fixture.options, fixture.profiles)
    assert "scene model LFM2-VL-1.6B-Q4_0.gguf (64 B," in label and "is not the profiled" in label


def test_a_profile_of_another_detector_engine_is_rejected(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene(engine_sha256="0" * 64)
    label, _ = refused(tmp_path, config_with(fixture.manifest), fixture.options, fixture.profiles)
    assert label.endswith("measured another detector engine than the pinned one")


# ---------------------------------------------------------------- admission


def test_a_valid_synthetic_accepted_profile_admits_scene(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    recorder = Recorder()
    assembly = assemble(config_with(fixture.manifest), RunOptions(tmp_path, tmp_path / "x.engine", scene=fixture.options),
                        recorder.devices(), FakeClock(), profiles=fixture.profiles)
    assert recorder.calls == ["scene_server", "scene_request", "detector_load"]
    assert assembly.runtime.core.lane is not None
    assert assembly.runtime.snapshot()["components"]["scene"]["state"] == "available"
    assert assembly.startup["scene_problem"] is None
    assembly.database.close()


# ---------------------------------------------------------------- the runtime registry


def test_the_runtime_registry_holds_no_synthetic_and_no_admissible_scene_profile() -> None:
    """Acceptance only by a maintainer-approved commit that adds an entry with step-4 evidence (D46).
    That commit is expected to change the second assertion; nothing at runtime can."""
    for profile_id, profile in RESOURCE_PROFILES.items():
        assert profile.profile_id == profile_id
        assert "synthetic" not in profile_id and "SYNTHETIC" not in profile.note
        assert re.match(r"^demo-profile-\d{8}T\d{6}Z$", profile.run_dir)
        assert profile.commit != "0" * 40
    assert [pid for pid in RESOURCE_PROFILES if accepted_profile_problem(pid) is None] == []
    with pytest.raises(TypeError):
        RESOURCE_PROFILES["x"] = RESOURCE_PROFILES[PROVISIONAL]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        RESOURCE_PROFILES[PROVISIONAL].status = ProfileStatus.ACCEPTED  # type: ignore[misc]


def test_the_provisional_profile_still_admits_the_detector_and_config_validate_explains_scene(capsys) -> None:
    assert known_profile_ids() == adapters.KNOWN_RESOURCE_PROFILES == frozenset({PROVISIONAL})
    detector = {"adapter_id": "legacy-yolov8n-bytetrack", "contract_version": 1, "implementation_revision": "1",
                "enabled": True, "input_kinds": ["frame"], "output_kinds": ["person.track"],
                "resource_profile_id": PROVISIONAL, "timeout_ms": 1000}
    scene = {**detector, "adapter_id": "llama-lfm2-vl-scene", "output_kinds": ["scene.report"]}
    statuses = resolve([AdapterManifest.model_validate(m) for m in (detector, scene)])
    assert [s.state for s in statuses] == [AdapterState.ENABLED, AdapterState.UNAVAILABLE]
    assert "is provisional" in statuses[1].reason
