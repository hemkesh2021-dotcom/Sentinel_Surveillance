"""Session 115, part 17: the scene request carries its report schema twice, from one source, and the fingerprint knows.

On llama-server b8932 the LFM2.5-VL chat template's special case builds a grammar only for tools and ignores
``response_format``, so the schema never reached the sampler and reports with more than five observations came back
(session 115, part 16). The request now also carries llama.cpp's own top-level ``json_schema`` field, which the server
converts itself whenever the template produced no grammar; where the template did (LFM2-VL), that grammar is kept. That
server path was reproduced offline on the device against the installed build (session 115 log, part 17). Portably, here:
- the body sends the same schema in both places, for both model sets, and nothing else changed;
- the parser's limits are unchanged: five observations pass, six are rejected;
- ordinary chat is unaffected;
- the fingerprint changed by that field alone, so the profiles measured before it no longer admit either set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
from sentinel.scene import models
from sentinel.scene.chat import ChatQuestion, chat_request_body
from sentinel.scene.completion import scene_report_schema, scene_response_format
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
JPEG = b"\xff\xd8synthetic\xff\xd9"
SETS = ("lfm2-vl", "lfm2.5-vl")
# the fingerprints of the request before this field, as the accepted profiles recorded them (D47)
BEFORE = {"step4cand-demo-20261007T090339Z": "3057dacd4c0ed54193b36b10cfccbac65efaae38ab40e7a6bb142be341595045",
          "step4cand-lfm25-20261026T094728Z": "1eb816afada9944ac398e46cb73ce43026ca0a475781f0fd8546f96175d322a3"}
PROFILE_OF = {"step4cand-demo-20261007T090339Z": "lfm2-vl", "step4cand-lfm25-20261026T094728Z": "lfm2.5-vl"}
REPORT = {"persons_visible": 0, "fire_or_smoke": False, "threat": "low", "uncertainty": "low", "summary": "Shapes."}


def test_the_body_sends_the_same_schema_in_both_places_from_one_source() -> None:
    for key in SETS:
        body = json.loads(scene_request_body(JPEG, models.scene_model_set(key).request_model))
        assert body["json_schema"] == body["response_format"]["json_schema"]["schema"] == scene_report_schema()
        assert body["json_schema"] == SceneReport.model_json_schema()
        assert body["response_format"] == scene_response_format()
    old, new = (json.loads(scene_request_body(JPEG, models.scene_model_set(k).request_model)) for k in SETS)
    assert {**old, "model": None} == {**new, "model": None}  # the rollback set sends the same fields


def test_the_rest_of_the_request_is_unchanged() -> None:
    body = json.loads(scene_request_body(JPEG))
    assert sorted(body) == ["json_schema", "max_tokens", "messages", "model", "response_format", "stream",
                            "temperature"]
    assert (body["max_tokens"], body["temperature"], body["stream"]) == (MAX_TOKENS, TEMPERATURE, False)
    assert (MAX_TOKENS, TEMPERATURE) == (200, 0.05)
    assert hashlib.sha256(SCENE_SYSTEM.encode()).hexdigest() == \
        "216f3fbc82843b2cb582182e52ddcfbad09891ea2edb9a3a17049661d8d73ccf"
    assert hashlib.sha256(SCENE_PROMPT.encode()).hexdigest() == \
        "3c1637dd746d906752fe04caa3d93b3bf0d48b6d9fb95b73ef970690285ea81c"
    assert (IMAGE_SIZE, JPEG_QUALITY) == ((480, 360), 60)


def test_five_observations_pass_and_six_are_rejected_by_the_unchanged_limits() -> None:
    schema = json.loads(scene_request_body(JPEG))["json_schema"]
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


def fingerprint_before_the_field(model: str) -> str:
    """D47's canonical form as it was before session 115, part 17: the stored profiles' fingerprints."""
    canonical = {"system": SCENE_SYSTEM, "prompt": SCENE_PROMPT, "image_size": list(IMAGE_SIZE),
                 "jpeg_quality": JPEG_QUALITY, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "model": model,
                 "stream": False, "response_format": scene_response_format()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_the_fingerprint_changed_by_the_field_alone_and_the_old_profiles_no_longer_admit(tmp_path: Path) -> None:
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    for profile_id, before in BEFORE.items():
        key = PROFILE_OF[profile_id]
        chosen, profile = models.scene_model_set(key), RESOURCE_PROFILES[profile_id]
        assert profile.scene_request_sha256 == before == fingerprint_before_the_field(chosen.request_model)
        assert chosen.scene_request_sha256 != before
        assert chosen.scene_request_sha256 == request_fingerprint(
            system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model=chosen.request_model)
        assert accepted_profile_problem(profile_id) is None  # the record stays accepted: it measured another request
        options = SceneOptions(tmp_path / "no-server", tmp_path / "no-model", tmp_path / "no-mmproj", model_set=key)
        assert profile_mismatch(profile, config, options, policy=CANDIDATE_POLICY) == (
            f"resource profile {profile_id} measured another scene request than the runtime sends")
    assert SCENE_REQUEST_SHA256 == models.scene_model_set("lfm2-vl").scene_request_sha256


def test_the_step4_workload_sends_the_runtimes_request(monkeypatch) -> None:
    """The measurement sends exactly what the runtime would, the new field included (D47)."""
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
    assert workload.scene_request_sha256() == models.scene_model_set("lfm2-vl").scene_request_sha256
