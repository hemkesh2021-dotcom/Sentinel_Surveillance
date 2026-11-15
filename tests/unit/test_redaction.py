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
        ("open rtsp://admin:hunter2@192.0.2.10/live", "hunter2", "open rtsp://<redacted>@192.0.2.10/live"),
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


# ---------------------------------------------------------------- Authorization values, JSON fields, *_key names


@pytest.mark.parametrize(("text", "secret", "kept"), [
    ("POST /v1 Authorization: Bearer eyJhbGci.SYNTH.sig", "SYNTH", "Authorization: <redacted>"),
    ('headers {"Authorization": "Bearer sk-SYNTH-1 more"}', "SYNTH", '{"Authorization": <redacted>}'),
    ("{'Proxy-Authorization': 'Basic dXNlcjpTWU5USA=='}", "dXNlcjpTWU5USA==", "{'Proxy-Authorization': <redacted>}"),
    ("authorization=token SYNTH-gh-1", "SYNTH", "authorization=<redacted>"),
    ("retry with bearer SYNTH.tok-en_1 failed", "SYNTH", "bearer <redacted> failed"),
    ("auth Basic dXNlcjpTWU5USA== rejected", "dXNlcjpTWU5USA==", "Basic <redacted> rejected"),
    ('body {"password": "correct horse SYNTH", "user": "u"}', "SYNTH", '{"password": <redacted>, "user": "u"}'),
    ('{"client_secret":"SYNTH-3","api_key": "SYNTH-4"}', "SYNTH", '{"client_secret":<redacted>,"api_key": <redacted>}'),
    ("{'access_token': 'SYNTH-5'}", "SYNTH", "{'access_token': <redacted>}"),
    ('{"password": "esc\\"aped SYNTH-6"}', "SYNTH", '{"password": <redacted>}'),
    ("secret_key=SYNTH-7 region=x", "SYNTH", "secret_key=<redacted> region=x"),
    ("AWS_SECRET_ACCESS_KEY: SYNTH-8", "SYNTH", "AWS_SECRET_ACCESS_KEY: <redacted>"),
    ("private_key='SYNTH 9' next", "SYNTH", "private_key=<redacted> next"),
    ("credentials: SYNTH-10", "SYNTH", "credentials: <redacted>"),
    ('{"password": "unterminated SYNTH-11', "SYNTH", '{"password": <redacted>'),
], ids=["header", "json-header", "dict-proxy-basic", "header-token-scheme", "bare-bearer", "bare-basic",
        "json-spaces", "json-compact", "dict-repr", "json-escaped-quote", "secret_key", "aws-env", "kwarg-repr",
        "credentials", "cut-off-json"])
def test_authorization_values_json_fields_and_key_names_are_redacted(text, secret, kept) -> None:
    assert secret not in redact(text) and kept in redact(text)


@pytest.mark.parametrize(("text", "secret", "kept"), [
    ("login failed: password=ab;SYNTH-1", "SYNTH", "password=<redacted>"),
    ("api_key=x,SYNTH-2 rejected", "SYNTH", "api_key=<redacted> rejected"),
    ("token=a}SYNTH-3", "SYNTH", "token=<redacted>"),
    ("secret=a]SYNTH-4", "SYNTH", "secret=<redacted>"),
    ('passwd=a"SYNTH-5', "SYNTH", "passwd=<redacted>"),
    ("password: a'SYNTH-6 next", "SYNTH", "password: <redacted> next"),
    ("password=ab&SYNTH-7", "SYNTH", "password=<redacted>"),  # an & that starts no other parameter
    ("GET /x?access_token=ab;SYNTH-8&page=2", "SYNTH", "access_token=<redacted>&page=2"),
    ("Authorization: token ab;SYNTH-9", "SYNTH", "Authorization: <redacted>"),
    ('Authorization: Digest username="u", response="SYNTH-10"', "SYNTH", "Authorization: <redacted>"),
], ids=["semicolon", "comma", "brace", "bracket", "double-quote", "single-quote", "ampersand", "query",
        "authorization-semicolon", "authorization-digest"])
def test_a_credential_value_with_punctuation_is_redacted_whole(text, secret, kept) -> None:
    # Follow-up review A, finding 1: an unquoted value stopped at , ; } ] " or ', so the rest of the secret leaked.
    assert secret not in redact(text) and kept in redact(text)
    assert secret not in redact_line(text)


@pytest.mark.parametrize("text", [
    "token_count: 3 tokens processed", "max_tokens=512", '{"token_type": "bearer"}', "basic check failed",
    "the secretary=front desk", "no credentials here", "Bearer", "keyword: value",
])
def test_names_that_only_look_like_credentials_are_left_alone(text) -> None:
    assert redact(text) == text


@pytest.mark.parametrize("hostile", [
    '"password": "' + "\\" * 200_000,  # escapes in a quoted value, never closed
    '"' * 200_000,
    "password" * 40_000,  # names with no separator
    "password:" * 40_000,
    "secret_key=" + "=" * 200_000,
    "Authorization:" + " " * 200_000,
    "Bearer " * 50_000,
    "bearer" + "\t" * 200_000 + "x",
    "basic " + "A" * 200_000,
    "'token'" * 40_000,
    "password=" + "&a" * 100_000,  # an & that starts no other parameter, again and again
    "token=&" + "a" * 200_000,
    "token=x" + "&a=" * 60_000,
    "Authorization: " + "," * 200_000,
], ids=["escapes", "quotes", "names", "names-colon", "equals", "header-spaces", "bearers", "tabs", "basic-run",
        "quoted-names", "ampersands", "long-parameter-name", "parameters", "header-commas"])
def test_the_new_patterns_stay_linear_on_hostile_text(hostile) -> None:
    started = time.monotonic()
    redact_line(hostile)
    assert time.monotonic() - started < 1.0
