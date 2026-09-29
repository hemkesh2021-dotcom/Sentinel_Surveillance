"""Redaction for anything that may reach logs, diagnostics or the UI.

Messages from configuration, providers and workers can embed URLs with
credentials or bot tokens. Redact before storing or showing them, and bound
their length so a misbehaving producer cannot flood a log line.
"""

from __future__ import annotations

import re

_PATTERNS = (
    (re.compile(r"(?<=://)[^/@\s]+@"), "<redacted>@"),  # URL userinfo
    (re.compile(r"\bbot\d+:[A-Za-z0-9_-]+"), "bot<redacted>"),  # Telegram bot token in a path
    (re.compile(r"(?i)\b(token|password|passwd|secret|api[_-]?key)=[^&\s]+"), r"\1=<redacted>"),
)


def redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_line(text: str, limit: int = 200) -> str:
    """One redacted line of at most ``limit`` characters."""
    if limit < 4:
        raise ValueError("limit must be at least 4")
    line = " ".join(redact(text).split())
    return line if len(line) <= limit else line[: limit - 3] + "..."
