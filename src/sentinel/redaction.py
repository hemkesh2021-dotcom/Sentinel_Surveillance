"""Redaction for anything that may reach logs, diagnostics or the UI.

Messages from configuration, providers and workers can embed URLs with
credentials, bot tokens, Authorization headers or JSON with credential fields.
Redact before storing or showing them, and bound their length so a misbehaving
producer cannot flood a log line.

What is redacted: URL userinfo; Telegram bot tokens; an Authorization (or
Proxy-Authorization) value, whatever its scheme; a Bearer token or Basic
credentials anywhere; and the value of any key ending in a credential name
(``access_token``, ``client_secret``, ``db_password``, ``secret_key``,
``AWS_SECRET_ACCESS_KEY``...) written ``key=value``, ``key: value``,
``"key": "value"`` or ``'key': 'value'``, a quoted value whole (escapes
included, even when the message cuts it off). Credentials in any other form
(a cookie, a PEM block, a bare value with no name) are not recognized.

Every pattern is linear: separators are bounded, and a value is consumed
whole, so nothing is rescanned at every position.
"""

from __future__ import annotations

import re


def _userinfo(match: re.Match[str]) -> str:
    # A password may hold an unencoded '/' or '@', so everything up to the last
    # '@' of the token after the scheme counts as userinfo. A path or query that
    # holds an '@' is hidden too: the safe side. The token is consumed whole, so
    # the scan stays linear.
    _, at, rest = match.group(1).rpartition("@")
    return f"://<redacted>@{rest}" if at else match.group(0)


_NAMES = r"token|password|passwd|secret|api[_-]?key|(?:secret|access|private)[_-]?key|credentials?"
_SEPARATOR = r"""["']?[ \t]{0,16}[:=][ \t]{0,16}"""  # key=, key:, "key": and 'key': (closing quote first)
_QUOTED = r'''"(?:[^"\\\r\n]|\\.)*"?|'(?:[^'\\\r\n]|\\.)*'?'''  # to its closing quote, or the line's end

_PATTERNS = (
    (re.compile(r"://(\S+)"), _userinfo),  # URL userinfo
    (re.compile(r"\bbot\d+:[A-Za-z0-9_-]+"), "bot<redacted>"),  # Telegram bot token in a path
    # An Authorization value, whatever its scheme (Bearer, Basic, token ...), up to the end of the value
    (re.compile(rf"""(?i)\b((?:proxy-)?authorization{_SEPARATOR})(?:{_QUOTED}|[^\r\n,;}}\]"']*)"""),
     r"\1<redacted>"),
    (re.compile(r"(?i)\b(bearer)[ \t]{1,16}[A-Za-z0-9._~+/-]+=*"), r"\1 <redacted>"),  # a bearer token anywhere
    (re.compile(r"(?i)\b(basic)[ \t]{1,16}[A-Za-z0-9+/]{8,}={0,2}"), r"\1 <redacted>"),  # Basic credentials
    # Any key ending in one of the names: access_token, client_secret, db_password, secret_key...
    (re.compile(rf"""(?i)((?:{_NAMES}){_SEPARATOR})(?:{_QUOTED}|[^\s&,;}}\]"']+)"""), r"\1<redacted>"),
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
