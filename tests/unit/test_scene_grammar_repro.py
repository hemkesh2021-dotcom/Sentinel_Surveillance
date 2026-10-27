"""Session 115, part 19: the standalone reproduction of the LFM2.5-VL scene request's HTTP errors
(benchmarks/runner/scene_grammar_repro.py), against the fake llama-server.

- The three bodies are the runtime's: the chat request, and the scene request with and without its top-level
  ``json_schema`` member, those two differing in that member only.
- A server that rejects the field as b8932 rejects a sampler it cannot initialise (HTTP 400, the error envelope, the
  grammar prefill's message) is read as ``field_rejected_at_sampler_init``, with the message's category, and never
  its text, kept.
- The result holds no prompt, picture, answer, report text or server message.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fake_llama_server import ANSWER, BUILD, REPORT, FakeLlamaServer, FakeServerProcess, free_port

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
# b8932's message for the LFM2.5-VL scene request with the field: common_sampler_init prefills the generation prompt
# into the OUTPUT_FORMAT grammar, whose first character must be "{" (offline, session 115, part 19)
PREFILL_MESSAGE = "Failed to initialize samplers: Unexpected empty grammar stack after accepting piece: <|im_start|> (6)"
SIX = dict(REPORT, observations=["one", "two", "three", "four", "five", "six"])


@pytest.fixture(scope="module")
def repro():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield importlib.import_module("scene_grammar_repro")
    finally:
        sys.path.remove(str(RUNNERS))


def ran(repro, tmp_path: Path, *, rounds: int = 2, **fake_options: Any) -> tuple[dict[str, Any], FakeLlamaServer]:
    port = free_port()
    fake = FakeLlamaServer(port, decode_s={}, **fake_options)
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent/llama-server"), port=port,
                              rounds=rounds, ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "repro.json")
    result = repro.run(args, server_factory=FakeServerProcess(fake), servers=lambda: [], in_use=lambda port: False,
                       meminfo=lambda: None, picture=lambda: "a synthetic picture",
                       encode=lambda image: b"\xff\xd8x\xff\xd9", vmstat=lambda: {"drop_pagecache": 1},
                       sleep=lambda seconds: None)
    return result, fake


def test_the_three_bodies_are_the_runtimes_and_the_scene_bodies_differ_in_the_field_only(repro) -> None:
    from sentinel.scene.llama_server import scene_request_body
    from sentinel.scene.models import scene_model_set

    from sentinel.scene.completion import scene_report_schema

    jpeg = b"\xff\xd8x\xff\xd9"
    bodies = repro.case_bodies(jpeg, scene_model_set("lfm2.5-vl").request_model)
    assert list(bodies) == ["basic", "response_format", "json_schema"]
    sent = json.loads(scene_request_body(jpeg, "lfm2.5-vl"))
    with_field, without = json.loads(bodies["json_schema"]), json.loads(bodies["response_format"])
    assert without == {key: value for key, value in sent.items() if key != "grammar"}  # the profiled request
    assert "json_schema" not in without and "response_format" in without and "grammar" not in without
    # part 17's request: the schema where the grammar now goes, in the same place
    assert list(with_field) == [("json_schema" if key == "grammar" else key) for key in sent]
    assert with_field["json_schema"] == scene_report_schema() and "grammar" not in with_field
    assert dict(without, json_schema=with_field["json_schema"]) == with_field
    assert repro.body_difference(bodies["response_format"], bodies["json_schema"]) == {
        "differing_members": ["json_schema"], "only_json_schema": True, "json_schema_in_response_format": True}
    basic = json.loads(bodies["basic"])
    assert "response_format" not in basic and "json_schema" not in basic and basic["model"] == "lfm2.5-vl"


def test_a_server_rejecting_the_field_at_sampler_init_is_read_so_without_its_message(repro, tmp_path) -> None:
    result, fake = ran(repro, tmp_path, json_schema_error=PREFILL_MESSAGE)
    assert result["problems"] == [] and len(result["requests"]) == 6
    assert result["comparison"]["outcome"] == "field_rejected_at_sampler_init"
    cases = result["comparison"]["cases"]
    assert cases["basic"]["statuses"] == ["200"] and cases["response_format"]["statuses"] == ["200"]
    assert cases["json_schema"] == {"requests": 2, "statuses": ["400"], "transport": [],
                                    "error_categories": ["sampler_init_failed"], "grammar_prefill_rejected": 2,
                                    "finish_reasons": [], "valid_reports": 0}
    rejected = [item for item in result["requests"] if item["case"] == "json_schema"][0]
    assert rejected["error"]["type"] == "invalid_request_error"
    assert rejected["error"]["piece_is_generation_prompt_start"] is True
    assert rejected["error"]["message_chars"] == len(PREFILL_MESSAGE)
    assert rejected["answer"] is None
    answered = [item for item in result["requests"] if item["case"] == "response_format"][0]
    assert answered["answer"]["finish_reason"] == "stop" and answered["answer"]["validation"]["valid"] is True
    assert answered["answer"]["usage"]["completion_tokens"] == 60
    basic = [item for item in result["requests"] if item["case"] == "basic"][0]
    assert basic["answer"]["validation"] == "not_applicable" and basic["answer"]["content_chars"] == len(ANSWER)
    assert result["scene_bodies_differ"]["only_json_schema"] is True
    assert result["bodies"]["json_schema"]["is_part_17s_request"] is True
    assert result["bodies"]["response_format"]["is_scene_request_body_without_grammar"] is True
    assert result["build_info"] == BUILD and result["stop"]["stopped"] is True
    assert fake.completed == {"chat": 2, "scene": 2}
    text = json.dumps(result)
    for secret in (PREFILL_MESSAGE, "<|im_start|>", ANSWER, "synthetic picture", REPORT["summary"], "data:image"):
        assert secret not in text


def test_a_server_answering_every_request_is_read_as_all_answered(repro, tmp_path) -> None:
    result, _ = ran(repro, tmp_path, rounds=1)
    assert result["comparison"]["outcome"] == "all_answered"
    assert [item["status"] for item in result["requests"]] == [200, 200, 200]


def test_a_rejected_report_keeps_its_field_paths_and_categories_only(repro, tmp_path) -> None:
    result, _ = ran(repro, tmp_path, rounds=1, scene_text=json.dumps(SIX))
    scene = [item for item in result["requests"] if item["case"] != "basic"]
    for item in scene:
        assert item["answer"]["validation"] == {"valid": False, "reason": "invalid_report",
                                                "details": [{"loc": "observations", "type": "too_long"}]}
    assert result["comparison"]["outcome"] == "all_answered"
    assert result["comparison"]["cases"]["json_schema"]["valid_reports"] == 0
    assert "six" not in json.dumps(result)


@pytest.mark.parametrize("payload, expected", [
    ({"error": {"code": 400, "message": PREFILL_MESSAGE, "type": "invalid_request_error"}},
     {"type": "invalid_request_error", "category": "sampler_init_failed", "grammar_prefill_rejected": True,
      "piece_is_generation_prompt_start": True}),
    ({"error": {"code": 400, "message": '"json_schema": Undefined rule', "type": "invalid_request_error"}},
     {"type": "invalid_request_error", "category": "json_schema_conversion_failed",
      "grammar_prefill_rejected": False}),
    ({"error": {"code": 400, "message": "request (3000 tokens) exceeds the available context size (2048 tokens)",
                "type": "exceed_context_size_error"}},
     {"type": "exceed_context_size_error", "category": "exceeds_context_size", "grammar_prefill_rejected": False}),
    ({"error": {"code": 500, "message": "something else entirely", "type": "made_up"}},
     {"type": "other", "category": "other", "grammar_prefill_rejected": False}),
    ("not an envelope", {"type": "other", "category": "other", "grammar_prefill_rejected": False, "envelope": False}),
])
def test_the_error_envelope_is_categorised_and_never_kept(repro, payload, expected) -> None:
    facts = repro.error_facts(payload)
    for key, value in expected.items():
        assert facts[key] == value
    assert set(facts) <= {"type", "envelope", "category", "grammar_prefill_rejected", "piece_is_generation_prompt_start",
                          "sampler_init_sliced", "message_sha256", "message_chars"}
    assert "message" not in facts


@pytest.mark.parametrize("statuses, category, outcome", [
    ({"basic": [500], "response_format": [200], "json_schema": [400]}, "other", "basic_request_failed"),
    ({"basic": [200], "response_format": [200], "json_schema": [500]}, "other", "field_rejected_otherwise"),
    # the sampler's message with another status, or without the prefill's text, is not the sampler-init outcome
    ({"basic": [200], "response_format": [200], "json_schema": [500]}, "sampler_init_failed",
     "field_rejected_otherwise"),
    ({"basic": [200], "response_format": [200], "json_schema": [400]}, "sampler_init_failed",
     "field_rejected_at_sampler_init"),
    ({"basic": [200], "response_format": [400], "json_schema": [400]}, "sampler_init_failed", "mixed"),
    ({"basic": [200], "response_format": [200], "json_schema": [400, 200]}, "sampler_init_failed", "mixed"),
])
def test_the_other_outcomes(repro, statuses, category, outcome) -> None:
    requests = []
    for name, codes in statuses.items():
        for code in codes:
            item = {"case": name, "status": code, "transport": None, "error": None, "answer": None}
            if code == 200:
                item["answer"] = {"finish_reason": "stop", "validation": "not_applicable"}
            else:
                item["error"] = {"category": category, "grammar_prefill_rejected": category != "other"}
            requests.append(item)
    assert repro.comparison(requests)["outcome"] == outcome


def test_the_sampler_init_outcome_needs_the_prefill_text_on_every_rejection(repro) -> None:
    answered = {"case": "basic", "status": 200, "transport": None, "error": None,
                "answer": {"finish_reason": "stop", "validation": "not_applicable"}}
    requests = [answered, dict(answered, case="response_format")]
    for rejected in (True, False):
        requests.append({"case": "json_schema", "status": 400, "transport": None, "answer": None,
                         "error": {"category": "sampler_init_failed", "grammar_prefill_rejected": rejected}})
    assert repro.comparison(requests)["outcome"] == "field_rejected_otherwise"


def test_a_transport_failure_is_a_fixed_label_and_an_incomplete_result(repro, tmp_path) -> None:
    port = free_port()  # nothing listens: every request fails at the connection
    fake = FakeLlamaServer(port, decode_s={})
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent/llama-server"), port=port,
                              rounds=1, ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "repro.json")
    failing = FakeServerProcess(fake)
    failing.start = lambda ready_timeout_s: (setattr(failing, "_state", failing._state.READY) or failing.status())
    result = repro.run(args, server_factory=failing, servers=lambda: [], in_use=lambda port: False,
                       meminfo=lambda: None, picture=lambda: "a synthetic picture",
                       encode=lambda image: b"\xff\xd8x\xff\xd9", vmstat=lambda: {}, sleep=lambda seconds: None)
    assert [item["transport"] for item in result["requests"]] == ["request_failed:ConnectionRefusedError"] * 3
    assert result["problems"] == [f"{name}_1:request_failed:ConnectionRefusedError" for name in repro.CASES]


def test_main_refuses_an_existing_output_and_names_its_commands(repro, tmp_path, capsys) -> None:
    out = tmp_path / "repro.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        repro.main(["run", "--model-set", "lfm2.5-vl", "--out", str(out)])
    assert "exists" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        repro.main(["run", "--model-set", "lfm2.5-vl", "--rounds", "6", "--out", str(tmp_path / "new.json")])
