"""Session 115, part 20: the scene request carries its report schema's grammar, pinned to the one canonical schema.

On llama-server b8932 the top-level ``json_schema`` field (part 17) was converted but typed as an output-format grammar
and prefilled with the chat template's generation prompt, which the conversion cannot accept, so every LFM2.5-VL scene
request failed before decoding (part 19: HTTP 400, "Failed to initialize samplers"). A request's own ``grammar`` is
typed USER and never prefilled; where the template builds its own grammar (LFM2-VL) that one is kept. Portably, here:
- the packaged grammar is the pinned conversion of scene_report_schema(), and both pins hold;
- a changed schema or a changed file is refused before a request is built (no stale grammar is ever sent);
- the body carries ``grammar`` and ``response_format``, never ``json_schema``, for both sets; the rest is unchanged;
- the parser's limits are unchanged; ordinary chat carries no constraint;
- the fingerprint changed by the grammar alone, so the profiles measured before it still admit neither set.
On the device, when llama.cpp's own converter and grammar validator are installed: the converter regenerates the file
byte for byte, and the grammar accepts five observations, rejects six and rejects the generation prompt as a prefix.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from sentinel.adapters import RESOURCE_PROFILES, accepted_profile_problem
from sentinel.config import parse_config
from sentinel.demo_runtime import SceneOptions, profile_mismatch
from sentinel.memory_policy import CANDIDATE_POLICY
from sentinel.scene import completion, models
from sentinel.scene.chat import ChatQuestion, chat_request_body
from sentinel.scene.completion import (
    SCENE_REPORT_GRAMMAR_SHA256,
    SCENE_REPORT_SCHEMA_SHA256,
    scene_grammar_problem,
    scene_report_grammar,
    scene_report_schema,
    scene_report_schema_sha256,
    scene_response_format,
)
from sentinel.scene.llama_server import (
    IMAGE_SIZE,
    JPEG_QUALITY,
    MAX_TOKENS,
    SCENE_PROMPT,
    SCENE_REQUEST_SHA256,
    SCENE_SYSTEM,
    TEMPERATURE,
    request_fingerprint,
    scene_request_body,
)
from sentinel.scene.report import SceneReport, SceneReportError, parse_scene_report

ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_FILE = ROOT / "src/sentinel/scene/scene_report.gbnf"
JPEG = b"\xff\xd8synthetic\xff\xd9"
SETS = ("lfm2-vl", "lfm2.5-vl")
# the fingerprints of the request before any grammar, as the accepted profiles recorded them (D47)
BEFORE = {"step4cand-demo-20261007T090339Z": "3057dacd4c0ed54193b36b10cfccbac65efaae38ab40e7a6bb142be341595045",
          "step4cand-lfm25-20261026T094728Z": "1eb816afada9944ac398e46cb73ce43026ca0a475781f0fd8546f96175d322a3"}
PROFILE_OF = {"step4cand-demo-20261007T090339Z": "lfm2-vl", "step4cand-lfm25-20261026T094728Z": "lfm2.5-vl"}
# part 17's fingerprints (the json_schema field), measured once and failed: never admitted
PART_17 = {"lfm2-vl": "b03fdbac", "lfm2.5-vl": "50085b37"}
REPORT = {"persons_visible": 0, "fire_or_smoke": False, "threat": "low", "uncertainty": "low", "summary": "Shapes."}
LLAMA_CPP = Path.home() / "llama.cpp"
CONVERTER = LLAMA_CPP / "examples/json_schema_to_grammar.py"
VALIDATOR = LLAMA_CPP / "build/bin/test-gbnf-validator"
GENERATION_PROMPT = "<|im_start|>assistant\n"


def test_the_packaged_grammar_is_the_pinned_conversion_of_the_canonical_schema() -> None:
    text = GRAMMAR_FILE.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == SCENE_REPORT_GRAMMAR_SHA256
    assert scene_report_schema_sha256() == SCENE_REPORT_SCHEMA_SHA256
    assert scene_grammar_problem() is None
    assert scene_report_grammar() == text and len(text) == 1148 and text.endswith("\n")
    # the shape the server's converter gives the schema: one root over the six fields in order, the bounds as rules
    assert text.startswith("boolean ::= ")
    assert 'root ::= "{" space persons-visible-kv "," space fire-or-smoke-kv "," space threat-kv "," space ' \
           'observations-kv "," space uncertainty-kv "," space summary-kv "}" space\n' in text
    assert 'observations ::= "[" space (observations-item ("," space observations-item){0,4})? "]" space' in text
    assert 'observations-item ::= "\\"" char{1,80} "\\"" space' in text
    assert 'summary ::= "\\"" char{0,160} "\\"" space' in text
    assert 'persons-visible ::= ([0-9] | ([1-4] [0-9] | [5] "0")) space' in text
    assert scene_report_schema() == SceneReport.model_json_schema()  # the one source, unchanged


def test_a_changed_schema_or_file_is_refused_before_any_request_is_built(monkeypatch) -> None:
    changed = dict(scene_report_schema(), description="another report")
    monkeypatch.setattr(completion, "scene_report_schema", lambda: changed)
    assert scene_grammar_problem() == "scene_report_grammar_stale"
    with pytest.raises(RuntimeError, match="scene_report_grammar_stale"):
        scene_report_grammar()
    with pytest.raises(RuntimeError, match="scene_report_grammar_stale"):
        scene_request_body(JPEG)
    monkeypatch.undo()
    monkeypatch.setattr(completion, "SCENE_REPORT_GRAMMAR_SHA256", "0" * 64)
    assert scene_grammar_problem() == "scene_report_grammar_changed"
    with pytest.raises(RuntimeError, match="scene_report_grammar_changed"):
        scene_request_body(JPEG)
    monkeypatch.undo()
    assert scene_grammar_problem() is None


def test_the_body_carries_the_grammar_and_the_response_format_and_never_the_field() -> None:
    for key in SETS:
        body = json.loads(scene_request_body(JPEG, models.scene_model_set(key).request_model))
        assert sorted(body) == ["grammar", "max_tokens", "messages", "model", "response_format", "stream",
                                "temperature"]
        assert body["grammar"] == scene_report_grammar()
        assert body["response_format"] == scene_response_format()
        assert body["response_format"]["json_schema"]["schema"] == scene_report_schema()
    old, new = (json.loads(scene_request_body(JPEG, models.scene_model_set(k).request_model)) for k in SETS)
    assert {**old, "model": None} == {**new, "model": None}  # the rollback set sends the same fields


def test_the_rest_of_the_request_is_unchanged() -> None:
    body = json.loads(scene_request_body(JPEG))
    assert (body["max_tokens"], body["temperature"], body["stream"]) == (MAX_TOKENS, TEMPERATURE, False)
    assert (MAX_TOKENS, TEMPERATURE) == (200, 0.05)
    assert hashlib.sha256(SCENE_SYSTEM.encode()).hexdigest() == \
        "216f3fbc82843b2cb582182e52ddcfbad09891ea2edb9a3a17049661d8d73ccf"
    assert hashlib.sha256(SCENE_PROMPT.encode()).hexdigest() == \
        "3c1637dd746d906752fe04caa3d93b3bf0d48b6d9fb95b73ef970690285ea81c"
    assert (IMAGE_SIZE, JPEG_QUALITY) == ((480, 360), 60)


def test_five_observations_pass_and_six_are_rejected_by_the_unchanged_limits() -> None:
    schema = json.loads(scene_request_body(JPEG))["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["observations"]["maxItems"] == 5
    assert schema["properties"]["observations"]["items"]["maxLength"] == 80
    assert schema["properties"]["summary"]["maxLength"] == 160
    five = parse_scene_report(json.dumps({**REPORT, "observations": ["a", "b", "c", "d", "e"]}))
    assert len(five.observations) == 5
    with pytest.raises(SceneReportError, match="observations: too_long"):
        parse_scene_report(json.dumps({**REPORT, "observations": ["a", "b", "c", "d", "e", "f"]}))


def test_ordinary_chat_is_unaffected() -> None:
    for key in SETS:
        chat = json.loads(chat_request_body(JPEG, ChatQuestion("What is on the table?"), 120,
                                            models.scene_model_set(key).request_model))
        assert sorted(chat) == ["max_tokens", "messages", "model", "stream", "temperature"]
        assert not ({"json_schema", "response_format", "grammar"} & set(chat))


def fingerprint_without_grammar(model: str) -> str:
    """D47's canonical form before session 115, part 17: the stored profiles' fingerprints."""
    canonical = {"system": SCENE_SYSTEM, "prompt": SCENE_PROMPT, "image_size": list(IMAGE_SIZE),
                 "jpeg_quality": JPEG_QUALITY, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "model": model,
                 "stream": False, "response_format": scene_response_format()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_the_fingerprint_changed_by_the_grammar_alone_and_the_old_profiles_still_admit_neither(tmp_path) -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    for profile_id, before in BEFORE.items():
        key = PROFILE_OF[profile_id]
        chosen, profile = models.scene_model_set(key), RESOURCE_PROFILES[profile_id]
        assert profile.scene_request_sha256 == before == fingerprint_without_grammar(chosen.request_model)
        assert chosen.scene_request_sha256 != before
        assert not chosen.scene_request_sha256.startswith(PART_17[key])
        assert chosen.scene_request_sha256 == request_fingerprint(
            system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model=chosen.request_model)
        assert accepted_profile_problem(profile_id) is None  # the record stays accepted: it measured another request
        options = SceneOptions(tmp_path / "no-server", tmp_path / "no-model", tmp_path / "no-mmproj", model_set=key)
        assert profile_mismatch(profile, config, options, policy=CANDIDATE_POLICY) == (
            f"resource profile {profile_id} measured another scene request than the runtime sends")
    assert SCENE_REQUEST_SHA256 == models.scene_model_set("lfm2-vl").scene_request_sha256


def test_the_step4_workload_sends_the_runtimes_request(monkeypatch) -> None:
    """The measurement sends exactly what the runtime would, the grammar included (D47)."""
    spec = importlib.util.spec_from_file_location("v2_26_demo_workload", ROOT / "benchmarks/runner/demo_workload.py")
    workload = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(workload)  # type: ignore[union-attr]
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def read(self, n: int = -1) -> bytes:
            return b'{"choices": []}'

    def fake_urlopen(request: Any, timeout: float | None = None) -> Response:
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        IMWRITE_JPEG_QUALITY=1, resize=lambda frame, size: frame,
        imencode=lambda ext, image, params: (True, types.SimpleNamespace(tobytes=lambda: JPEG))))
    workload.scene_request(0, object())
    assert captured["body"] == json.loads(scene_request_body(JPEG, workload.SCENE_REQUEST_MODEL))
    assert captured["body"]["grammar"] == scene_report_grammar() and "json_schema" not in captured["body"]
    assert workload.scene_request_sha256() == models.scene_model_set("lfm2-vl").scene_request_sha256


# ---------------------------------------------------------------- on the device: llama.cpp's own converter and validator


@pytest.mark.skipif(not CONVERTER.is_file(), reason="llama.cpp's json_schema_to_grammar.py is not installed here")
def test_llamacpps_converter_regenerates_the_packaged_grammar_byte_for_byte(tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(scene_report_schema()))
    produced = subprocess.run(["python3", "-I", str(CONVERTER), str(schema)], capture_output=True, text=True,
                              timeout=60)
    assert produced.returncode == 0, produced.stderr[-400:]
    assert produced.stdout == GRAMMAR_FILE.read_text(encoding="utf-8")


def validate(text: str, tmp_path: Path) -> bool:
    grammar, sample = tmp_path / "scene_report.gbnf", tmp_path / "sample.txt"
    grammar.write_text(scene_report_grammar(), encoding="utf-8")
    sample.write_text(text, encoding="utf-8")
    run = subprocess.run([str(VALIDATOR), str(grammar), str(sample)], capture_output=True, text=True, timeout=60)
    assert "valid according to the grammar" in run.stdout, run.stdout[-400:]
    return "Input string is valid according to the grammar" in run.stdout


@pytest.mark.skipif(not VALIDATOR.is_file(), reason="llama.cpp's test-gbnf-validator is not installed here")
def test_the_grammar_accepts_five_observations_rejects_six_and_rejects_the_generation_prompt(tmp_path) -> None:
    five = json.dumps({"persons_visible": 0, "fire_or_smoke": False, "threat": "low",
                       "observations": ["a", "b", "c", "d", "e"], "uncertainty": "low", "summary": "Shapes."})
    six = json.dumps({"persons_visible": 0, "fire_or_smoke": False, "threat": "low",
                      "observations": ["a", "b", "c", "d", "e", "f"], "uncertainty": "low", "summary": "Shapes."})
    assert validate(five, tmp_path) is True
    assert validate(six, tmp_path) is False
    assert validate(GENERATION_PROMPT + five, tmp_path) is False  # why a prefill of it fails (part 19)
