"""Schema-constrained requests and fail-closed, non-streaming scene completions.

Generation constraints do not replace strict parsing or establish scene accuracy.
A complete response can hit a character bound; that alone is not truncation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import resources

from .report import SceneReport, SceneReportError, parse_scene_report

# The report schema's grammar (scene_report.gbnf), as llama-server b8932's own converter produces it from
# scene_report_schema() (json_schema_to_grammar; byte for byte examples/json_schema_to_grammar.py's output; session 115,
# part 20). The request sends it as llama.cpp's top-level ``grammar`` field: a request's own grammar is typed USER by
# the server and never prefilled with the chat template's generation prompt, which is what made the ``json_schema``
# field fail before decoding on LFM2.5-VL (part 19). Both pins guard the one source: the schema's canonical SHA-256
# records what the file was generated from, and the file's SHA-256 records the file; scene_report_grammar() refuses
# when either differs, so a changed report model or a changed file never sends a stale grammar.
SCENE_REPORT_SCHEMA_SHA256 = "5fe9e4f17c125253c07dbdbe274e1b9992befc1c84738c47560601b5fff070c0"
SCENE_REPORT_GRAMMAR_SHA256 = "3cadd9f1401204007290c15ac3536756acfa51226dd59fcad2e3c816fdcb2133"


def scene_report_schema_sha256() -> str:
    """SHA-256 of the canonical JSON of scene_report_schema(): what the pinned grammar was generated from."""
    return hashlib.sha256(json.dumps(scene_report_schema(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scene_grammar_problem() -> str | None:
    """None when the packaged grammar is the pinned conversion of the current schema; else a fixed label."""
    try:
        text = resources.files(__package__).joinpath("scene_report.gbnf").read_text(encoding="utf-8")
    except (OSError, TypeError):
        return "scene_report_grammar_missing"
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != SCENE_REPORT_GRAMMAR_SHA256:
        return "scene_report_grammar_changed"
    if scene_report_schema_sha256() != SCENE_REPORT_SCHEMA_SHA256:
        return "scene_report_grammar_stale"
    return None


def scene_report_grammar() -> str:
    """The pinned grammar text for the scene request; RuntimeError when scene_grammar_problem() finds one."""
    problem = scene_grammar_problem()
    if problem is not None:
        raise RuntimeError(f"{problem}: regenerate scene_report.gbnf from scene_report_schema() and pin both hashes")
    return resources.files(__package__).joinpath("scene_report.gbnf").read_text(encoding="utf-8")


class SceneCompletionError(SceneReportError):
    """An unusable completion envelope; reason is a fixed, safe diagnostic code."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def scene_report_schema() -> dict[str, object]:
    """The report's JSON schema: the one source for both places a scene request carries it (session 115, part 17)."""
    return SceneReport.model_json_schema()


def scene_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "scene_report", "strict": True, "schema": scene_report_schema()},
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
