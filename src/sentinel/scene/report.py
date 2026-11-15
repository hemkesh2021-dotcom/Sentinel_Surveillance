"""Scene report schema for VLM output (guide chapter 13).

Model text is untrusted input. It must be exactly one JSON object matching
SceneReport, optionally inside a single fenced code block. Extra fields, wrong
types (a string "true" is not a boolean) and oversized values are rejected,
never repaired, and nothing in the text is executed or used as an instruction.
Rejection messages name the offending field but never echo the model's text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any

from pydantic import Field, StringConstraints, ValidationError

from ..contracts import Contract
from ..redaction import redact_line

MAX_REPORT_CHARS = 4_096

Observation = Annotated[str, StringConstraints(min_length=1, max_length=80)]


class Threat(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Uncertainty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SceneReport(Contract):
    """What the scene model reports about one image. A label, not a calibrated probability."""

    persons_visible: Annotated[int, Field(ge=0, le=50)]
    fire_or_smoke: bool
    threat: Threat
    observations: Annotated[tuple[Observation, ...], Field(max_length=5)]
    uncertainty: Uncertainty
    summary: Annotated[str, StringConstraints(max_length=160)]


class SceneReportError(ValueError):
    """The text is not a valid scene report; the message is safe to log."""


_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(?P<body>.*)\n```\Z", re.DOTALL)


def parse_scene_report(text: str) -> SceneReport:
    if len(text) > MAX_REPORT_CHARS:
        raise SceneReportError(f"report longer than {MAX_REPORT_CHARS} characters")
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group("body").strip()
    try:
        return SceneReport.model_validate_json(body)
    except ValidationError as exc:
        problems = [_describe(error) for error in exc.errors(include_input=False)]
        raise SceneReportError("invalid scene report: " + "; ".join(problems[:3])) from None


def _describe(error: Mapping[str, Any]) -> str:
    # An unexpected key is model text: bound and redact it like any other output.
    loc = redact_line(".".join(str(part) for part in error["loc"]), 40) or "<report>"
    if error["type"] == "json_invalid":
        return "not a single JSON object"
    if error["type"] == "extra_forbidden":
        return f"{loc}: unexpected field"
    return f"{loc}: {error['type']}"
