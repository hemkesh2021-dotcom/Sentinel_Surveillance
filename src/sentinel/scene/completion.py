"""Schema-constrained requests and fail-closed, non-streaming scene completions.

Generation constraints do not replace strict parsing or establish scene accuracy.
A complete response can hit a character bound; that alone is not truncation.
"""

from __future__ import annotations

from collections.abc import Mapping

from .report import SceneReport, SceneReportError, parse_scene_report


class SceneCompletionError(SceneReportError):
    """An unusable completion envelope; reason is a fixed, safe diagnostic code."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def scene_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "scene_report", "strict": True, "schema": SceneReport.model_json_schema()},
    }


def scene_completion_finish_reason(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return "missing"
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return "missing"
    reason = choices[0].get("finish_reason")
    if reason is None:
        return "missing"
    return reason if reason in ("stop", "length") else "other"


def parse_scene_completion(payload: object) -> SceneReport:
    return parse_scene_report(scene_completion_content(payload))


def scene_completion_content(payload: object) -> str:
    """The report text of a complete, non-streaming completion; SceneCompletionError otherwise.

    The text itself is not parsed here: parse_scene_report() does that, wherever it runs.
    """
    if not isinstance(payload, Mapping):
        raise SceneCompletionError("malformed_response")
    if "error" in payload:
        raise SceneCompletionError("server_error")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise SceneCompletionError("malformed_response")
    reason = choices[0].get("finish_reason")
    if reason == "length":
        raise SceneCompletionError("truncated")
    if reason != "stop":
        raise SceneCompletionError("incomplete_completion")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise SceneCompletionError("malformed_response")
    if message.get("refusal") is not None or message.get("tool_calls"):
        raise SceneCompletionError("unsupported_completion")
    content = message.get("content")
    if not isinstance(content, str):
        raise SceneCompletionError("malformed_response")
    return content
