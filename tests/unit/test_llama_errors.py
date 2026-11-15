"""Session 115, part 20: llama-server b8932's error envelope read into fixed labels (benchmarks/runner/llama_errors.py),
and the step-4 workload keeping an HTTP error's category, never its message."""

from __future__ import annotations

import importlib
import io
import json
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
PREFILL_MESSAGE = "Failed to initialize samplers: Unexpected empty grammar stack after accepting piece: <|im_start|> (6)"
SLICED_MESSAGE = "Failed to initialize samplers: std::exception"  # what the live server sent (part 19's reproduction)


@pytest.fixture(scope="module")
def errors():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield importlib.import_module("llama_errors")
    finally:
        sys.path.remove(str(RUNNERS))


def envelope(message: str, kind: str = "invalid_request_error", code: int = 400) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "type": kind}}


@pytest.mark.parametrize("payload, expected", [
    (envelope(PREFILL_MESSAGE), {"type": "invalid_request_error", "category": "sampler_init_failed",
                                 "grammar_prefill_rejected": True, "piece_is_generation_prompt_start": True,
                                 "sampler_init_sliced": False}),
    (envelope(SLICED_MESSAGE), {"type": "invalid_request_error", "category": "sampler_init_failed",
                                "grammar_prefill_rejected": False, "sampler_init_sliced": True,
                                "message_chars": 45}),
    (envelope('"json_schema": Undefined rule'), {"category": "json_schema_conversion_failed"}),
    (envelope("Cannot use both json_schema and grammar"), {"category": "json_schema_and_grammar"}),
    (envelope("request (3000 tokens) exceeds the available context size (2048 tokens)",
              "exceed_context_size_error"), {"type": "exceed_context_size_error", "category": "exceeds_context_size"}),
    (envelope("failed to process image", "server_error", 500), {"type": "server_error",
                                                               "category": "image_processing_failed"}),
    (envelope("something else entirely", "made_up"), {"type": "other", "category": "other"}),
    ("not an envelope", {"type": "other", "category": "other", "envelope": False}),
])
def test_the_envelope_is_categorised_and_the_message_never_kept(errors, payload, expected) -> None:
    facts = errors.error_facts(payload)
    for key, value in expected.items():
        assert facts[key] == value, key
    assert set(facts) <= {"type", "envelope", "category", "grammar_prefill_rejected", "sampler_init_sliced",
                          "piece_is_generation_prompt_start", "message_sha256", "message_chars"}
    assert "message" not in facts
    assert facts["category"] in errors.CATEGORY_LABELS


def test_an_errors_body_gives_one_fixed_label(errors) -> None:
    assert errors.error_category(json.dumps(envelope(SLICED_MESSAGE)).encode()) == "sampler_init_failed"
    assert errors.error_category(b"<html>not json") == "unparsable"
    assert errors.error_category(b"{}") == "other"


def workload_module():
    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module("demo_workload")
    finally:
        sys.path.remove(str(RUNNERS))


def test_the_workload_counts_an_http_errors_category_and_keeps_no_message(monkeypatch) -> None:
    w = workload_module()
    body = json.dumps(envelope(SLICED_MESSAGE)).encode()

    def failing_urlopen(request: Any, timeout: float | None = None) -> Any:
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        IMWRITE_JPEG_QUALITY=1, resize=lambda frame, size: frame,
        imencode=lambda ext, image, params: (True, types.SimpleNamespace(tobytes=lambda: b"\xff\xd8x\xff\xd9"))))
    stats = w.Stats()
    outcome = w.scene_attempt(18081, object(), stats)
    assert outcome == {"error": "http", "error_category": "sampler_init_failed"}
    second = w.scene_attempt(18081, object(), stats)
    assert second["error_category"] == "sampler_init_failed"
    with stats.lock:
        assert stats.vlm_errors == {"HTTP 400": 2}
        assert stats.vlm_http_categories == {"sampler_init_failed": 2}
    summary = stats.snapshot(2.0, 1.0)["scene"] if hasattr(stats, "snapshot") else None
    if summary is None:  # the stats' own summary method, whatever its name, carries the categories
        summary = next(method for name in dir(stats) if callable(method := getattr(stats, name))
                       and name.startswith("summ"))(2.0, 1.0)["scene"]
    assert summary["http_error_categories"] == {"sampler_init_failed": 2}
    assert summary["http_errors"] == 2 and summary["errors"] == {"HTTP 400": 2}
    assert "std::exception" not in json.dumps(summary)
    assert w.http_error_category(urllib.error.HTTPError("http://127.0.0.1", 500, "x", {}, io.BytesIO(b"nope"))) == \
        "unparsable"


def test_the_profilers_sanitizer_keeps_the_categories_and_the_summary_names_them() -> None:
    sys.path.insert(0, str(RUNNERS))
    try:
        profile = importlib.import_module("demo_profile")
    finally:
        sys.path.remove(str(RUNNERS))
    scene = {"http_errors": 2, "errors": {"HTTP 400": 2}, "http_error_categories": {"sampler_init_failed": 2},
             "error_category": "sampler_init_failed", "completed": 0}
    kept = profile.sanitize_diagnostic({"scene": scene})["scene"]
    assert kept["http_error_categories"] == {"sampler_init_failed": 2}
    assert kept["error_category"] == "sampler_init_failed"
    assert kept["errors"] == {"HTTP 400": 2}
