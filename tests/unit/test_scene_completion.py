from __future__ import annotations

import json

import pytest

from sentinel.scene.completion import (
    SceneCompletionError,
    parse_scene_completion,
    scene_completion_content,
    scene_completion_finish_reason,
    scene_response_format,
)
from sentinel.scene.report import SceneReport, SceneReportError

REPORT = {
    "persons_visible": 0, "fire_or_smoke": False, "threat": "none",
    "observations": [], "uncertainty": "high", "summary": "No person is visible.",
}


def completion(content: object = None, finish_reason: object = "stop") -> dict:
    return {"choices": [{
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": json.dumps(REPORT) if content is None else content},
    }]}


def test_request_schema_is_the_unchanged_report_schema_and_not_shared_mutable_state() -> None:
    response_format = scene_response_format()
    assert response_format["type"] == "json_schema"
    wrapper = response_format["json_schema"]
    assert wrapper["strict"] is True
    assert wrapper["schema"] == SceneReport.model_json_schema()
    schema = wrapper["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(REPORT)
    assert schema["properties"]["summary"]["maxLength"] == 160
    assert schema["properties"]["observations"]["maxItems"] == 5
    assert schema["properties"]["observations"]["items"]["maxLength"] == 80
    schema["properties"].clear()
    assert scene_response_format()["json_schema"]["schema"] == SceneReport.model_json_schema()


@pytest.mark.parametrize("fenced", [False, True])
def test_complete_responses_at_the_limits_are_not_clipped_or_presumed_truncated(fenced: bool) -> None:
    summary = "a" * 160
    observations = ["b" * 80] * 5
    content = json.dumps({**REPORT, "summary": summary, "observations": observations})
    payload = completion(f"```json\n{content}\n```" if fenced else content)
    report = parse_scene_completion(payload)
    assert report.summary == summary
    assert report.observations == tuple(observations)
    assert payload["choices"][0]["message"]["content"].endswith("```" if fenced else "}")


@pytest.mark.parametrize("content", [json.dumps(REPORT), '{"summary": "unfinished', ""])
def test_length_finished_responses_are_rejected_even_if_the_json_happens_to_be_valid(content: str) -> None:
    with pytest.raises(SceneCompletionError) as caught:
        parse_scene_completion(completion(content, "length"))
    assert caught.value.reason == "truncated"


@pytest.mark.parametrize("reason", [None, "tool_calls", "content_filter", "unknown", [], 1])
def test_missing_or_non_stop_finish_reason_never_accepts_a_report(reason: object) -> None:
    with pytest.raises(SceneCompletionError, match="incomplete_completion"):
        parse_scene_completion(completion(finish_reason=reason))


@pytest.mark.parametrize("payload", [
    None, [], {}, {"choices": []}, {"choices": [None]}, {"choices": [1, 2]},
    {"choices": [{"finish_reason": "stop"}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "user", "content": json.dumps(REPORT)}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ["text"]}}]},
])
def test_malformed_envelopes_are_rejected(payload: object) -> None:
    with pytest.raises(SceneCompletionError, match="malformed_response"):
        parse_scene_completion(payload)


@pytest.mark.parametrize("message_field", [{"refusal": "private diagnostic"}, {"tool_calls": [{}]}])
def test_refusals_and_tool_calls_are_not_scene_reports(message_field: dict) -> None:
    payload = completion()
    payload["choices"][0]["message"].update(message_field)
    with pytest.raises(SceneCompletionError, match="unsupported_completion"):
        parse_scene_completion(payload)


def test_server_error_never_echoes_its_body_or_accepts_a_spurious_valid_choice() -> None:
    payload = {**completion(), "error": {"message": "http://user:private@camera/"}}
    with pytest.raises(SceneCompletionError) as caught:
        parse_scene_completion(payload)
    assert str(caught.value) == "server_error"


@pytest.mark.parametrize("content", [
    '{"summary": "unfinished', json.dumps(REPORT)[:-1],
    json.dumps({**REPORT, "summary": "s" * 161}),
    json.dumps({**REPORT, "observations": ["s" * 81]}),
    json.dumps({**REPORT, "observations": ["s"] * 6}),
    json.dumps({**REPORT, "persons_visible": "1"}),
    json.dumps({**REPORT, "fire_or_smoke": "false"}),
    json.dumps({**REPORT, "extra": "ignored?"}),
    json.dumps(REPORT) + json.dumps(REPORT),
])
def test_stop_finished_content_still_requires_unchanged_strict_validation(content: str) -> None:
    with pytest.raises(SceneReportError):
        parse_scene_completion(completion(content))


@pytest.mark.parametrize(("payload", "expected"), [
    (None, "missing"), ({"choices": []}, "missing"), (completion(finish_reason=None), "missing"),
    (completion(finish_reason="stop"), "stop"), (completion(finish_reason="length"), "length"),
    (completion(finish_reason="private camera description"), "other"),
    (completion(finish_reason={"private": "text"}), "other"),
])
def test_finish_reason_diagnostics_have_only_fixed_bounded_labels(payload: object, expected: str) -> None:
    assert scene_completion_finish_reason(payload) == expected


def test_structural_validity_does_not_establish_image_accuracy() -> None:
    observed_label = 2
    report = parse_scene_completion(completion())
    assert report.persons_visible == 0
    assert report.persons_visible != observed_label


def test_the_envelope_check_returns_report_text_unparsed_for_the_lane_to_validate() -> None:
    # V2-26: the worker checks the envelope; the scene lane parses the report strictly.
    assert scene_completion_content(completion('{"persons_visible": "two"}')) == '{"persons_visible": "two"}'
    with pytest.raises(SceneReportError):
        parse_scene_completion(completion('{"persons_visible": "two"}'))
    for payload, reason in ((completion(finish_reason="length"), "truncated"), ({"error": {}}, "server_error")):
        with pytest.raises(SceneCompletionError) as caught:
            scene_completion_content(payload)
        assert caught.value.reason == reason
