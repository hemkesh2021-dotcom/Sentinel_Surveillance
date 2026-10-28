"""Session 115: the scene model sets. LFM2.5-VL-1.6B is the default; LFM2-VL-1.6B, the first accepted profile's model, is
the rollback. One set serves scene analysis and chat alike. D46 is unchanged (the maintainer's decision, 2026-10-26): a set
that no accepted profile measured is refused by ``sentinel run --scene``, and nothing gets around that. Since D105
(2026-10-26) an accepted profile, step4cand-lfm25-20261026T094728Z, measured the default set; the refusal is shown here
with a SYNTHETIC unprofiled copy of it.

Profiles here are the SYNTHETIC ``accepted_scene`` fixture (tests/unit/conftest.py) or the registry, read only; no
model, server or camera is used.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel import cli
from sentinel.adapters import RESOURCE_PROFILES
from sentinel.config import parse_config
from sentinel.demo_runtime import Devices, RunOptions, StartupRefused, assemble
from sentinel.media.clock import FakeClock
from sentinel.scene import models
from sentinel.scene.chat import ChatQuestion, LlamaChatRequest, chat_request_body
from sentinel.scene.llama_server import (
    IMAGE_SIZE,
    JPEG_QUALITY,
    MAX_TOKENS,
    MODEL_NAME,
    SCENE_PROMPT,
    SCENE_REQUEST_SHA256,
    SCENE_SYSTEM,
    TEMPERATURE,
    LlamaSceneRequest,
    request_fingerprint,
    scene_request_body,
)

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = "step4cand-demo-20261007T090339Z"
LFM25_PROFILE = "step4cand-lfm25-20261026T094728Z"  # accepted under D105
OLD, NEW = models.SCENE_MODEL_SETS["lfm2-vl"], models.SCENE_MODEL_SETS["lfm2.5-vl"]
LFM3 = models.SCENE_MODEL_SETS["lfm3-vl"]  # session 116: the default; profiled under D106
JPEG = b"\xff\xd8synthetic\xff\xd9"


def d46_refusal(key: str) -> str:
    return (f"scene_not_admitted: llama-lfm2-vl-scene: scene model set {key} has no accepted resource profile "
            "(D46): it needs a step-4 profile run that measures it and the maintainer's acceptance; no other set is "
            "substituted (--scene-model-set lfm2.5-vl selects the explicit fallback, lfm2-vl the rollback, each "
            "admitted only by its own accepted profile)")


D46_REFUSAL = d46_refusal("lfm2.5-vl")  # the synthetic unprofiled() case


def config_with(*adapters_: dict[str, Any]):
    return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "adapters": list(adapters_)})


class Recorder:
    """Devices that record what the runtime asked for, with the request model each factory was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def devices(self) -> Devices:
        recorder = self

        class Backend:
            def load(self) -> None:
                recorder.calls.append(("detector_load",))

            def track(self, image: object) -> list:
                return []

            def reset(self) -> None:
                pass

        status = SimpleNamespace(state=SimpleNamespace(value="ready"), problem=None, layers="17/17", vision_on_gpu=True)

        def scene_server(options: Any, port: int) -> Any:
            recorder.calls.append(("scene_server", options.model_set))
            return SimpleNamespace(start=lambda timeout: status, status=lambda: status, stop=lambda grace: True)

        def scene_request(port: int, timeout: float, chosen: Any) -> Any:
            recorder.calls.append(("scene_request", chosen.request_model))
            return lambda job, image: None

        def chat_request(port: int, timeout: float, chosen: Any) -> Any:
            recorder.calls.append(("chat_request", chosen.request_model))
            return lambda *args, **kwargs: None

        return Devices(capture_source=lambda config: object(), tracker_backend=lambda engine: Backend(),
                       scene_server=scene_server, scene_request=scene_request, chat_request=chat_request,
                       meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000},
                       notifiers=lambda config: ({}, {}))


def refusal(tmp_path: Path, manifest: dict[str, Any], scene: Any, profiles: Any) -> str:
    recorder = Recorder()
    with pytest.raises(StartupRefused) as error:
        assemble(config_with(manifest), RunOptions(tmp_path / "data", tmp_path / "x.engine", scene=scene),
                 recorder.devices(), FakeClock(), profiles=profiles)
    assert recorder.calls == []  # no scene server, no request worker, no model load
    assert not (tmp_path / "data" / "sentinel.db").exists()
    return error.value.label


# ---------------------------------------------------------------- the two sets


def test_the_rollback_set_is_exactly_what_the_accepted_profile_measured() -> None:
    profile = RESOURCE_PROFILES[ACCEPTED]
    assert models.ROLLBACK_SCENE_MODEL_SET == OLD.key == "lfm2-vl" and OLD.profiled is True
    assert (OLD.model, OLD.mmproj) == (profile.llm, profile.mmproj)
    assert OLD.request_model == MODEL_NAME == "lfm2-vl"
    assert OLD.scene_request_sha256 == SCENE_REQUEST_SHA256
    # session 115, parts 17 and 20: the request gained its top-level grammar field, so the profile that measured the
    # request without it (its fingerprint, as recorded) no longer admits the set until a new measurement is accepted
    assert profile.scene_request_sha256 == "3057dacd4c0ed54193b36b10cfccbac65efaae38ab40e7a6bb142be341595045"
    assert profile.scene_request_sha256 != SCENE_REQUEST_SHA256
    assert OLD.model_revision == "lfm2-vl-1.6b-q4_0"
    # the earlier defaults of sentinel run, unchanged for the rollback
    assert OLD.model_path == Path.home() / "models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf"
    assert OLD.mmproj_path == Path.home() / "models/lfm2-vl/mmproj-LFM2-VL-1.6B-Q8_0.gguf"


def test_the_fallback_set_is_lfm25_vl_at_a_pinned_revision_with_its_own_projector() -> None:
    assert models.FALLBACK_SCENE_MODEL_SET == NEW.key == "lfm2.5-vl" and NEW.profiled is True  # D105
    profile = RESOURCE_PROFILES[LFM25_PROFILE]  # the accepted profile measured exactly these files ...
    assert (NEW.model, NEW.mmproj) == (profile.llm, profile.mmproj)
    # ... and the request before session 115, part 17, whose fingerprint it keeps: it no longer admits the set
    assert profile.scene_request_sha256 == "1eb816afada9944ac398e46cb73ce43026ca0a475781f0fd8546f96175d322a3"
    assert NEW.scene_request_sha256 != profile.scene_request_sha256
    assert re.fullmatch(r"[0-9a-f]{40}", models.LFM25_REVISION)
    assert NEW.source == f"LiquidAI/LFM2.5-VL-1.6B-GGUF at {models.LFM25_REVISION}"
    assert (NEW.model.name, NEW.model.bytes, NEW.model.sha256) == (
        "LFM2.5-VL-1.6B-Q4_0.gguf", 695_752_480, "8186364a4e7c3ad30f6dd3d3b7a4e0074c77dd91eed6cad5d8be9090ce285804")
    assert (NEW.mmproj.name, NEW.mmproj.bytes, NEW.mmproj.sha256) == (
        "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf", 583_109_888,
        "2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d")
    assert NEW.mmproj.name != OLD.mmproj.name and NEW.mmproj.sha256 != OLD.mmproj.sha256  # never the old projector
    assert NEW.directory == Path.home() / "models/lfm2.5-vl" != OLD.directory  # beside the old files, never over them
    assert (NEW.request_model, NEW.model_revision) == ("lfm2.5-vl", "lfm2.5-vl-1.6b-q4_0")
    assert NEW.scene_request_sha256 != SCENE_REQUEST_SHA256
    assert NEW.scene_request_sha256 == request_fingerprint(
        system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY,
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model="lfm2.5-vl")


def test_an_unknown_set_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown scene model set 'lfm3'"):
        models.scene_model_set("lfm3")
    with pytest.raises(TypeError):
        models.SCENE_MODEL_SETS["x"] = OLD  # type: ignore[index]


# ---------------------------------------------------------------- requests: the set's model name, nothing else


def test_scene_and_chat_requests_carry_the_sets_model_and_differ_in_nothing_else() -> None:
    old, new = json.loads(scene_request_body(JPEG)), json.loads(scene_request_body(JPEG, NEW.request_model))
    assert old["model"] == "lfm2-vl" and new["model"] == "lfm2.5-vl"
    assert {**old, "model": None} == {**new, "model": None}  # prompt, schema, tokens, temperature, picture
    question = ChatQuestion("What is on the table?")
    old_chat = json.loads(chat_request_body(JPEG, question))
    new_chat = json.loads(chat_request_body(JPEG, question, 60, NEW.request_model))
    assert old_chat["model"] == "lfm2-vl" and new_chat["model"] == "lfm2.5-vl" and new_chat["max_tokens"] == 60
    assert {**old_chat, "model": None, "max_tokens": None} == {**new_chat, "model": None, "max_tokens": None}


class Transport:
    def __init__(self, reply: bytes) -> None:
        self.reply, self.bodies = reply, []

    def post(self, body: bytes, **kwargs: Any) -> bytes:
        self.bodies.append(json.loads(body))
        return self.reply


def test_the_request_classes_send_the_model_they_were_given() -> None:
    report = json.dumps({"persons_visible": 0, "fire_or_smoke": False, "threat": "none", "observations": [],
                         "uncertainty": "low", "summary": "Nobody is in view."})
    reply = json.dumps({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": report}}],
                        "timings": {"prompt_ms": 400.0, "predicted_ms": 600.0, "predicted_n": 30}}).encode()
    for model in (OLD.request_model, NEW.request_model):
        transport = Transport(reply)
        LlamaSceneRequest(transport, encode=lambda image: JPEG, model=model)(
            SimpleNamespace(job_id="scene-1"), object())
        LlamaChatRequest(transport, encode=lambda image: JPEG, model=model)(ChatQuestion("Is anyone here?"), object())
        assert [body["model"] for body in transport.bodies] == [model, model]


# ---------------------------------------------------------------- admission: D46 unchanged


def unprofiled(monkeypatch) -> None:
    """SYNTHETIC: the default set as it was before D105, with no accepted profile behind it."""
    monkeypatch.setattr(models, "SCENE_MODEL_SETS", {**models.SCENE_MODEL_SETS,
                                                     NEW.key: dataclasses.replace(NEW, profiled=False)})


def test_an_unprofiled_set_is_refused_before_any_process_whatever_the_profile_says(tmp_path, accepted_scene,
                                                                                  monkeypatch) -> None:
    unprofiled(monkeypatch)
    fixture = accepted_scene()
    manifest = {**fixture.manifest, "model_revision": NEW.model_revision}
    scene = dataclasses.replace(fixture.options, model_set=NEW.key)
    assert refusal(tmp_path / "a", manifest, scene, fixture.profiles) == D46_REFUSAL
    # A profile that names the new files does not admit them either: only a maintainer-approved commit that records
    # an accepted profile which measured them, and marks the set profiled, can (as D105's did).
    measured = accepted_scene(llm=dataclasses.replace(fixture.profile.llm, name=NEW.model.name),
                              mmproj=dataclasses.replace(fixture.profile.mmproj, name=NEW.mmproj.name))
    assert refusal(tmp_path / "b", manifest, scene, measured.profiles) == D46_REFUSAL
    without_revision = {k: v for k, v in manifest.items() if k != "model_revision"}
    assert refusal(tmp_path / "c", without_revision, scene, fixture.profiles) == D46_REFUSAL


def test_a_declared_model_revision_must_be_the_selected_sets(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    label = refusal(tmp_path / "a", {**fixture.manifest, "model_revision": NEW.model_revision}, fixture.options,
                    fixture.profiles)
    assert label == ("scene_not_admitted: llama-lfm2-vl-scene: the manifest declares model_revision "
                     "lfm2.5-vl-1.6b-q4_0, not lfm2-vl-1.6b-q4_0 (scene model set lfm2-vl)")
    label = refusal(tmp_path / "b", fixture.manifest, dataclasses.replace(fixture.options, model_set=NEW.key),
                    fixture.profiles)
    assert label == ("scene_not_admitted: llama-lfm2-vl-scene: the manifest declares model_revision "
                     "lfm2-vl-1.6b-q4_0, not lfm2.5-vl-1.6b-q4_0 (scene model set lfm2.5-vl)")


def test_the_rollback_set_is_admitted_as_before_and_scene_and_chat_name_its_model(tmp_path, accepted_scene) -> None:
    fixture = accepted_scene()
    assert fixture.options.model_set == OLD.key  # SceneOptions' default: the accepted profile's set
    for name, manifest in (("declared", fixture.manifest),
                           ("undeclared", {k: v for k, v in fixture.manifest.items() if k != "model_revision"})):
        recorder = Recorder()
        assembly = assemble(config_with(manifest), RunOptions(tmp_path / name, tmp_path / "x.engine",
                                                              scene=fixture.options, chat=True, chat_validation=True),
                            recorder.devices(), FakeClock(), profiles=fixture.profiles)
        assert recorder.calls == [("scene_server", "lfm2-vl"), ("scene_request", "lfm2-vl"), ("detector_load",),
                                  ("chat_request", "lfm2-vl")]
        assert assembly.startup["scene_problem"] is None and "scene_model" not in assembly.startup
        assembly.database.close()


def test_the_runtime_gives_scene_and_chat_the_selected_sets_name_not_a_constant(tmp_path, accepted_scene,
                                                                                monkeypatch) -> None:
    """A SYNTHETIC profiled set (never in the registry) shows the wiring: both requests carry the set's own name."""
    synthetic = dataclasses.replace(OLD, key="synthetic-set", request_model="synthetic-model")
    fixture = accepted_scene(scene_request_sha256=synthetic.scene_request_sha256)  # measured with that request
    monkeypatch.setattr("sentinel.demo_runtime.scene_model_set", lambda key: synthetic)
    recorder = Recorder()
    options = RunOptions(tmp_path, tmp_path / "x.engine", scene=fixture.options, chat=True, chat_validation=True)
    assembly = assemble(config_with(fixture.manifest), options, recorder.devices(), FakeClock(),
                        profiles=fixture.profiles)
    assert ("scene_request", "synthetic-model") in recorder.calls
    assert ("chat_request", "synthetic-model") in recorder.calls
    assembly.database.close()


def test_a_future_accepted_profile_of_another_set_needs_data_only(tmp_path, accepted_scene, monkeypatch) -> None:
    """SYNTHETIC: if a maintainer-approved commit records a profile that measured another set and marks that set
    profiled, the runtime admits it with no code change, because a profile's request fingerprint is compared with the
    selected set's; the old request's fingerprint does not admit it."""
    future = dataclasses.replace(NEW, profiled=True, model_revision=OLD.model_revision)  # the fixture's manifest
    monkeypatch.setattr("sentinel.demo_runtime.scene_model_set", lambda key: future)
    measured = accepted_scene(scene_request_sha256=NEW.scene_request_sha256)
    recorder = Recorder()
    assembly = assemble(config_with(measured.manifest), RunOptions(tmp_path / "a", tmp_path / "x.engine",
                                                                   scene=measured.options),
                        recorder.devices(), FakeClock(), profiles=measured.profiles)
    assert ("scene_request", "lfm2.5-vl") in recorder.calls
    assembly.database.close()
    fixture = accepted_scene()  # the profiled request's fingerprint, SCENE_REQUEST_SHA256
    label = refusal(tmp_path / "b", fixture.manifest, fixture.options, fixture.profiles)
    assert label.endswith("measured another scene request than the runtime sends")


# ---------------------------------------------------------------- the CLI: the default, the rollback, no bypass


def scene_options_from_cli(tmp_path: Path, monkeypatch, *flags: str) -> Any:
    captured: dict[str, Any] = {}

    def fake_assemble(config, options, devices, clock, **kwargs):
        captured["scene"] = options.scene
        raise StartupRefused("captured")

    monkeypatch.setattr(cli, "assemble", fake_assemble)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    tmp_path.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    code = cli.main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", "x", "--status-port", "0",
                     "--scene", *flags], capture_source=lambda config: object(),
                    tracker_backend=lambda engine: pytest.fail("the detector was constructed"),
                    meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000})
    assert code == 1
    return captured["scene"]


def test_sentinel_run_selects_lfm3_by_default_with_its_own_build_and_the_others_explicitly(tmp_path,
                                                                                         monkeypatch) -> None:
    default = scene_options_from_cli(tmp_path / "a", monkeypatch)
    assert (default.model_set, default.model, default.mmproj) == (LFM3.key, LFM3.model_path, LFM3.mmproj_path)
    assert default.binary == LFM3.llama_server == Path.home() / "llama.cpp-lfm3/build/bin/llama-server"
    fallback = scene_options_from_cli(tmp_path / "f", monkeypatch, "--scene-model-set", "lfm2.5-vl")
    assert (fallback.model_set, fallback.model, fallback.mmproj) == (NEW.key, NEW.model_path, NEW.mmproj_path)
    assert fallback.binary == NEW.llama_server == Path.home() / "llama.cpp/build/bin/llama-server"
    rollback = scene_options_from_cli(tmp_path / "b", monkeypatch, "--scene-model-set", "lfm2-vl")
    assert (rollback.model_set, rollback.model, rollback.mmproj) == (OLD.key, OLD.model_path, OLD.mmproj_path)
    assert rollback.binary == OLD.llama_server == NEW.llama_server
    given = scene_options_from_cli(tmp_path / "g", monkeypatch, "--llama-server", "/b/llama-server")
    assert given.binary == Path("/b/llama-server") and given.model_set == LFM3.key
    explicit = scene_options_from_cli(tmp_path / "c", monkeypatch, "--scene-model", "/m/a.gguf",
                                      "--scene-mmproj", "/m/b.gguf")
    assert (explicit.model, explicit.mmproj) == (Path("/m/a.gguf"), Path("/m/b.gguf"))


def cli_scene_run(tmp_path: Path, monkeypatch, profile_id: str, *flags: str,
                  model_revision: str = "lfm2.5-vl-1.6b-q4_0") -> int:
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        "config_version: 1\ncamera: {id: cam-1}\nadapters: [{adapter_id: llama-lfm2-vl-scene, contract_version: 1, "
        "implementation_revision: '1', enabled: true, input_kinds: [frame], output_kinds: [scene.report], "
        f"model_revision: {model_revision}, resource_profile_id: {profile_id}, timeout_ms: 8000}}]\n")
    code = cli.main(["run", str(config), "--data-dir", str(tmp_path / "data"), "--engine", "x", "--status-port", "0",
                     "--scene", *flags], capture_source=lambda config: object(),
                    tracker_backend=lambda engine: pytest.fail("the detector was constructed"),
                    meminfo=lambda: {"MemFree": 7_000_000_000, "MemAvailable": 7_000_000_000})
    assert not (tmp_path / "data" / "sentinel.db").exists()
    return code


def test_the_cli_refuses_an_unprofiled_default_with_the_d46_reason_and_starts_nothing(tmp_path, capsys,
                                                                                    monkeypatch) -> None:
    """D106: the default, lfm3-vl, is profiled now; the synthetic unprofiled case keeps the D46 refusal."""
    for profile_id in (ACCEPTED, LFM25_PROFILE):  # another set's profile never admits it
        capsys.readouterr()
        assert cli_scene_run(tmp_path / profile_id, monkeypatch, profile_id, model_revision=LFM3.model_revision) == 1
        error = capsys.readouterr().err.strip()
        assert error.startswith("run: scene_not_admitted: llama-lfm2-vl-scene: ") and "D46" not in error, error
    unprofiled(monkeypatch)
    capsys.readouterr()
    assert cli_scene_run(tmp_path / "synthetic", monkeypatch, LFM25_PROFILE, "--scene-model-set", "lfm2.5-vl") == 1
    assert capsys.readouterr().err.strip() == f"run: {D46_REFUSAL}"


def test_the_cli_checks_the_profiled_default_against_its_profile_and_starts_nothing(tmp_path, capsys,
                                                                                 monkeypatch) -> None:
    """D105: past D46, the fallback set meets its profile's checks; without the measured memory policy it is
    refused."""
    assert cli_scene_run(tmp_path, monkeypatch, LFM25_PROFILE, "--scene-model-set", "lfm2.5-vl") == 1
    error = capsys.readouterr().err.strip()
    assert error.startswith("run: scene_not_admitted: llama-lfm2-vl-scene: resource profile "
                            f"{LFM25_PROFILE} measured memory policy") and "D46" not in error


def test_run_still_has_no_option_that_bypasses_scene_admission(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    options = set(re.findall(r"--[a-z][a-z-]+", capsys.readouterr().out))
    assert "--scene-model-set" in options and "--scene-validation" not in options
    assert not {o for o in options if re.search(r"allow|force|skip|unsafe|override|profile|admit|scene-valid", o)}


# ---------------------------------------------------------------- the smoke tool's own checks (no server)


def smoke_module() -> Any:
    sys.path.insert(0, str(ROOT / "benchmarks/runner"))
    try:
        import scene_smoke
    finally:
        sys.path.remove(str(ROOT / "benchmarks/runner"))
    return scene_smoke


def test_the_smoke_tool_checks_the_chatml_shape_and_the_pictures_features() -> None:
    smoke = smoke_module()
    prompt = ("<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<__media__>T<|im_end|>\n"
              "<|im_start|>assistant\n")
    assert smoke.chatml_problem(prompt, "S", "T") is None
    assert smoke.chatml_problem(prompt.replace("system\nS", "system\nX"), "S", "T") == "system_or_user_header"
    assert smoke.chatml_problem(prompt[:-len("<|im_start|>assistant\n")], "S", "T") == "user_text_or_generation_prompt"
    assert smoke.picture_mentions("A red disc, a BLUE square; the word Exit.") == {
        "red": True, "blue": True, "round_shape": True, "square_shape": True, "word_exit": True}
    assert smoke.picture_mentions("A predicament: blueberry exits.") == {
        "red": False, "blue": False, "round_shape": False, "square_shape": False, "word_exit": False}
