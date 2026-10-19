"""Redaction of credentials in text that may reach logs, records or the UI.

The values are synthetic. Each case keeps one secret per line so a partial
leak (a fragment of a password left behind) fails the test too.
"""

from __future__ import annotations

import time

import pytest

from sentinel.redaction import redact, redact_line


@pytest.mark.parametrize(
    ("text", "secret", "expected"),
    [
        ("GET /x?access_token=XYZsecret1&page=2", "XYZsecret1", "GET /x?access_token=<redacted>&page=2"),
        ("auth_token=tok-2 failed", "tok-2", "auth_token=<redacted> failed"),
        ("refresh_token=r3fresh", "r3fresh", "refresh_token=<redacted>"),
        ("client_secret=cl1ent", "cl1ent", "client_secret=<redacted>"),
        ("db_password=hunter2", "hunter2", "db_password=<redacted>"),
        ("X-Api-Key=k3y", "k3y", "X-Api-Key=<redacted>"),
        ("token=s3cret", "s3cret", "token=<redacted>"),
    ],
    ids=["access_token", "auth_token", "refresh_token", "client_secret", "db_password", "x-api-key", "plain"],
)
def test_a_key_ending_in_a_credential_name_is_redacted_with_its_prefix_kept(text, secret, expected) -> None:
    assert redact(text) == expected
    assert secret not in redact_line(text)


@pytest.mark.parametrize(
    ("text", "secret", "expected"),
    [
        ("open rtsp://admin:hunter2@192.0.2.10/live failed", "hunter2", "open rtsp://<redacted>@192.0.2.10/live failed"),
        ("rtsp://admin:pa/ss@192.0.2.10/live", "pa/ss", "rtsp://<redacted>@192.0.2.10/live"),
        ("rtsp://admin:p@ss@192.0.2.10/live", "p@ss", "rtsp://<redacted>@192.0.2.10/live"),
        ("rtsp://admin:p@s/s@192.0.2.10:554/live?x=1", "s/s", "rtsp://<redacted>@192.0.2.10:554/live?x=1"),
        ("http://u:p4ss@127.0.0.1:8080 refused; retry http://127.0.0.1:8080/x",
         "p4ss", "http://<redacted>@127.0.0.1:8080 refused; retry http://127.0.0.1:8080/x"),
    ],
    ids=["plain", "slash-in-password", "at-in-password", "both", "two-urls"],
)
def test_url_userinfo_is_redacted_up_to_the_last_at_sign(text, secret, expected) -> None:
    assert redact(text) == expected
    assert secret not in redact_line(text)


def test_a_url_without_userinfo_is_left_alone() -> None:
    text = "GET https://example.com/a/b?q=1 and rtsp://192.0.2.10/live"
    assert redact(text) == text


def test_long_adversarial_text_is_redacted_in_linear_time() -> None:
    # Error text is not bounded before redaction (only the stored line is), so
    # the patterns must not rescan the rest of a token at every position.
    hostile = "x://" * 50_000 + "a-" * 50_000 + "token" * 20_000
    started = time.monotonic()
    redact_line(hostile)
    assert time.monotonic() - started < 2.0
