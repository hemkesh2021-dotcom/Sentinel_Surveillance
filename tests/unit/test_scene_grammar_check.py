"""Session 115, part 20: the guarded verification that the scene request's grammar constrains generation
(benchmarks/runner/scene_grammar_check.py), against the fake llama-server.

- The three bodies are the runtime's: the chat request, and the scene request without and with its ``grammar``
  member (the pinned grammar), those two differing in that member only.
- A server whose unconstrained answer runs to six observations while the grammar request's answer is valid is read as
  ``enforced``; one whose control stays within the limit as ``enforced_control_not_exercised``; a grammar request
  answered with six observations as ``grammar_reports_rejected``; a grammar request refused (HTTP 400) as
  ``grammar_requests_failed``.
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
SIX = json.dumps(dict(REPORT, observations=["one", "two", "three", "four", "five", "six"]))
SLICED_MESSAGE = "Failed to initialize samplers: std::exception"


@pytest.fixture(scope="module")
def check():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield importlib.import_module("scene_grammar_check")
    finally:
        sys.path.remove(str(RUNNERS))


def ran(check, tmp_path: Path, *, rounds: int = 2, **fake_options: Any) -> tuple[dict[str, Any], FakeLlamaServer]:
    port = free_port()
    fake = FakeLlamaServer(port, decode_s={}, **fake_options)
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent/llama-server"), port=port,
                              rounds=rounds, ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "check.json")
    result = check.run(args, server_factory=FakeServerProcess(fake), servers=lambda: [], in_use=lambda port: False,
                       meminfo=lambda: None, picture=lambda: "a synthetic picture",
                       encode=lambda image: b"\xff\xd8x\xff\xd9", vmstat=lambda: {"drop_pagecache": 1},
                       sleep=lambda seconds: None)
    return result, fake


def test_the_bodies_are_the_runtimes_and_differ_in_the_pinned_grammar_only(check) -> None:
    from sentinel.scene.completion import scene_report_grammar
    from sentinel.scene.llama_server import scene_request_body

    jpeg = b"\xff\xd8x\xff\xd9"
    bodies = check.case_bodies(jpeg, "lfm2.5-vl")
    assert list(bodies) == ["chat", "unconstrained", "grammar"]
    assert bodies["grammar"] == scene_request_body(jpeg, "lfm2.5-vl")
    with_grammar, without = json.loads(bodies["grammar"]), json.loads(bodies["unconstrained"])
    assert "grammar" not in without and "response_format" in without and "json_schema" not in without
    assert dict(without, grammar=scene_report_grammar()) == with_grammar
    assert check.body_difference(bodies["unconstrained"], bodies["grammar"]) == {
        "differing_members": ["grammar"], "only_grammar": True, "grammar_is_pinned": True}
    chat = json.loads(bodies["chat"])
    assert not ({"grammar", "json_schema", "response_format"} & set(chat)) and chat["model"] == "lfm2.5-vl"


def test_a_control_over_the_limit_and_valid_grammar_answers_read_as_enforced(check, tmp_path) -> None:
    result, fake = ran(check, tmp_path, scene_text=SIX, grammar_scene_text=json.dumps(REPORT))
    assert result["problems"] == [] and len(result["requests"]) == 6 and result["grammar_problem"] is None
    assert result["comparison"]["outcome"] == "enforced"
    cases = result["comparison"]["cases"]
    assert cases["chat"]["statuses"] == ["200"] and cases["chat"]["finish_reasons"] == ["stop"]
    assert cases["unconstrained"] == {"requests": 2, "statuses": ["200"], "transport": [], "error_categories": [],
                                      "finish_reasons": ["stop"], "valid_reports": 0, "rejected_reports": 2,
                                      "rejection_reasons": ["invalid_report"], "observations_over_limit": 2,
                                      "observations_max": None}
    assert cases["grammar"]["valid_reports"] == 2 and cases["grammar"]["observations_max"] == 1
    chat = [item for item in result["requests"] if item["case"] == "chat"][0]
    assert chat["answer"]["parses_as_report"] is False and chat["answer"]["content_chars"] == len(ANSWER)
    assert chat["answer"]["validation"] == "not_applicable"
    control = [item for item in result["requests"] if item["case"] == "unconstrained"][0]
    assert control["answer"]["validation"]["details"] == [{"loc": "observations", "type": "too_long"}]
    assert result["bodies"]["grammar"]["is_scene_request_body"] is True
    assert result["build_info"] == BUILD and result["stop"]["stopped"] is True
    assert fake.completed == {"chat": 2, "scene": 4}
    text = json.dumps(result)
    for secret in (ANSWER, "synthetic picture", REPORT["summary"], "data:image", "six", "root ::="):
        assert secret not in text


def test_a_control_within_the_limit_is_not_exercised(check, tmp_path) -> None:
    result, _ = ran(check, tmp_path, rounds=1)
    assert result["comparison"]["outcome"] == "enforced_control_not_exercised"
    assert result["comparison"]["cases"]["unconstrained"]["observations_over_limit"] == 0


def test_a_grammar_answer_over_the_limit_is_rejected_and_read_so(check, tmp_path) -> None:
    result, _ = ran(check, tmp_path, rounds=1, scene_text=SIX)  # six observations whatever the request carries
    assert result["comparison"]["outcome"] == "grammar_reports_rejected"
    assert result["comparison"]["cases"]["grammar"]["observations_over_limit"] == 1


def test_a_refused_grammar_request_is_read_with_its_category(check, tmp_path) -> None:
    result, _ = ran(check, tmp_path, rounds=1, grammar_error=SLICED_MESSAGE)
    assert result["comparison"]["outcome"] == "grammar_requests_failed"
    grammar = result["comparison"]["cases"]["grammar"]
    assert grammar["statuses"] == ["400"] and grammar["error_categories"] == ["sampler_init_failed"]
    item = [item for item in result["requests"] if item["case"] == "grammar"][0]
    assert item["error"]["sampler_init_sliced"] is True
    assert SLICED_MESSAGE not in json.dumps(result)


def test_a_stale_grammar_refuses_before_any_server_starts(check, tmp_path, monkeypatch) -> None:
    import sentinel.scene.completion as completion

    monkeypatch.setattr(completion, "SCENE_REPORT_GRAMMAR_SHA256", "0" * 64)
    port = free_port()
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent/llama-server"), port=port,
                              rounds=1, ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "check.json")
    started = []
    result = check.run(args, server_factory=lambda *a: started.append(a), servers=lambda: [],
                       in_use=lambda port: False, meminfo=lambda: None, picture=lambda: None,
                       encode=lambda image: b"", vmstat=lambda: {}, sleep=lambda seconds: None)
    assert result["problems"] == ["grammar:scene_report_grammar_changed"] and started == []


@pytest.mark.parametrize("chat, control, grammar, outcome", [
    ([500], [200], [200], "chat_failed"),
    ([200], [500], [200], "mixed"),
    ([200], [200], [200, 400], "grammar_requests_failed"),
])
def test_the_other_outcomes(check, chat, control, grammar, outcome) -> None:
    requests = []
    for name, codes in (("chat", chat), ("unconstrained", control), ("grammar", grammar)):
        for code in codes:
            item = {"case": name, "status": code, "transport": None, "error": None, "answer": None}
            if code == 200:
                item["answer"] = {"finish_reason": "stop", "validation": "not_applicable" if name == "chat" else
                                  {"valid": True, "observations": 2}}
            else:
                item["error"] = {"category": "other"}
            requests.append(item)
    assert check.comparison(requests)["outcome"] == outcome
