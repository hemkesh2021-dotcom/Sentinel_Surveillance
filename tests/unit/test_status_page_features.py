"""D99: the feature controls' requests and their authorization, without a browser.

Two POST routes, /features/unlock and /features, exist only with ``sentinel run --feature-controls``. Each must come
from this page exactly as POST /chat must (the loopback Host, an Origin of exactly this origin, Sec-Fetch-Site
same-origin when sent, a JSON body with a Content-Length within the bound, the page's token) AND carry the operator's
control key, which only the runtime's 0600 key file holds: reaching the page through the SSH tunnel authorizes
nothing. Payloads are exact; wrong keys lock every control for a while; nothing in a request is echoed, and the key is
in no page, status or response. FeatureControls itself: requests coalesce, are bounded and logged with fixed labels.
"""

from __future__ import annotations

import http.client
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from status_fixtures import incident_store

from sentinel import status_page
from sentinel.features import FEATURES, RATE_LIMIT, FeatureControls
from sentinel.media.clock import FakeClock
from sentinel.status_page import (
    CONTROL_KEY_NAME,
    StatusServer,
    new_control_key,
    normalized_key,
    remove_control_key,
    write_control_key,
)

SNAPSHOT = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": "2026-10-14T12:00:00+00:00"},
            "degraded": [], "components": {}}
ALL = {name: True for name in FEATURES}


class Rig:
    def __init__(self, tmp_path: Path, *, controls: bool = True, switchable: dict[str, bool] | None = None) -> None:
        incident_store(tmp_path / "sentinel.db", count=1)
        self.clock = FakeClock()
        self.controls = FeatureControls(self.clock, ALL if switchable is None else switchable)
        self.key = new_control_key()
        self.server = StatusServer(0, lambda: SNAPSHOT, tmp_path / "sentinel.db")
        if controls:
            self.server.enable_controls(lambda: self.controls, self.key)
        self.server.start()
        self.port = self.server.address[1]
        self.origin = f"http://127.0.0.1:{self.port}"

    def request(self, method: str, path: str, body: bytes | None = None, **headers: str | None) -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            values = {"Host": f"127.0.0.1:{self.port}", **{k.replace("_", "-"): v for k, v in headers.items()}}
            if body is not None and "Content-Length" not in values and "Transfer-Encoding" not in values:
                values["Content-Length"] = str(len(body))
            for name, value in values.items():
                if value is not None:
                    conn.putheader(name, value)
            conn.endheaders()
            if body is not None:
                conn.send(body)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def good(self) -> dict[str, str]:
        return {"Origin": self.origin, "Content_Type": "application/json", "X_Sentinel_Token": self.server.chat_token,
                "Sec_Fetch_Site": "same-origin", "X_Sentinel_Control": self.key}

    def change(self, feature: object = "face", state: object = "off", *, raw: bytes | None = None,
               **headers: str | None) -> tuple[int, dict[str, Any]]:
        body = raw if raw is not None else json.dumps({"feature": feature, "state": state}).encode()
        code, _, data = self.request("POST", "/features", body, **{**self.good(), **headers})
        return code, json.loads(data) if data.startswith(b"{") else {}

    def unlock(self, key: str | None = None, body: bytes = b"{}", **headers: str | None) -> tuple[int, dict[str, Any]]:
        code, _, data = self.request("POST", "/features/unlock", body,
                                     **{**self.good(), "X_Sentinel_Control": key or self.key, **headers})
        return code, json.loads(data) if data.startswith(b"{") else {}

    def close(self) -> None:
        assert self.server.stop(3.0)


@pytest.fixture
def rig(tmp_path: Path):
    made = []

    def make(**options: Any) -> Rig:
        folder = tmp_path / f"r{len(made)}"
        folder.mkdir()
        made.append(Rig(folder, **options))
        return made[-1]

    yield make
    for r in made:
        r.close()


# ---------------------------------------------------------------- the routes


def test_without_feature_controls_both_routes_say_off_and_change_nothing(rig) -> None:
    r = rig(controls=False)
    assert r.change()[0] == 404 and r.unlock()[0] == 404
    assert r.controls.requested("face") is True and r.controls.view()["log"] == []
    assert r.server.controls_view() == {"enabled": False, "locked_s": 0, "key_file": None}


def test_the_right_key_unlocks_and_a_change_is_accepted_once(rig) -> None:
    r = rig()
    assert r.unlock() == (200, {"state": "unlocked"})
    assert r.unlock(key=r.key.lower().replace("-", "")) == (200, {"state": "unlocked"})  # as an operator may type it
    code, document = r.change("face", "off")
    assert code == 202 and document == {"state": "accepted", "change": 1, "feature": "face", "requested": "off"}
    assert r.controls.requested("face") is False and r.controls.waiting("face")
    assert r.change("face", "off") == (200, {"state": "unchanged", "feature": "face", "requested": "off"})


@pytest.mark.parametrize(("headers", "code", "counted"), [
    ({"X_Sentinel_Control": None}, 403, "refused_wrong_key"),  # the SSH tunnel and the page token alone: refused
    ({"X_Sentinel_Control": "AAAA-BBBB-CCCC-DDDD-EEEE-FFFF"}, 403, "refused_wrong_key"),
    ({"Origin": "http://evil.example"}, 403, "refused_not_this_page"),
    ({"Origin": None}, 403, "refused_not_this_page"),
    ({"Sec_Fetch_Site": "cross-site"}, 403, "refused_not_this_page"),
    ({"X_Sentinel_Token": "0" * 32}, 403, "refused_not_this_page"),
    ({"Content_Type": "text/plain"}, 415, "refused_not_this_page"),
    ({"Transfer_Encoding": "chunked"}, 411, "refused_not_this_page"),
])
def test_a_change_without_this_page_or_the_key_is_refused_and_counted(rig, headers, code, counted) -> None:
    r = rig()
    got, document = r.change(**headers)
    assert got == code and document["state"] in ("refused", "invalid")
    assert r.controls.requested("face") is True and r.controls.view()["counters"].get(counted) == 1
    assert r.key not in json.dumps(document)


def test_the_host_the_body_bound_and_other_methods_are_refused(rig) -> None:
    r = rig()
    assert r.request("POST", "/features", b"{}", **{**r.good(), "Host": "evil.example:80"})[0] == 421
    assert r.change(raw=b"x" * 300)[0] == 413
    assert r.request("POST", "/features", None, **r.good())[0] == 411  # no Content-Length
    for method in ("OPTIONS", "PUT", "DELETE", "PATCH"):  # no preflight can pass, no other verb exists
        assert r.request(method, "/features", None, **r.good())[0] == 405
    assert r.request("GET", "/features")[0] == 404 and r.request("GET", "/features/unlock")[0] == 404
    assert r.controls.requested("face") is True


@pytest.mark.parametrize("raw", [
    b'{"feature": "face"}', b'{"feature": "face", "state": "off", "why": "x"}', b'{"feature": "camera", "state": "off"}',
    b'{"feature": "face", "state": "maybe"}', b'{"feature": "face", "state": true}', b'["face", "off"]', b"face off",
    b'\xff\xfe',
])
def test_a_payload_must_be_exactly_a_feature_and_a_state(rig, raw) -> None:
    r = rig()
    code, document = r.change(raw=raw)
    assert code == 400 and document["state"] == "invalid" and r.controls.view()["log"] == []
    assert r.unlock(body=b'{"key": "x"}')[0] == 400


def test_wrong_keys_lock_every_control_for_a_while_even_the_right_key(rig, monkeypatch) -> None:
    r = rig()
    now = [1000.0]
    monkeypatch.setattr(r.server, "_monotonic", lambda: now[0])
    for _ in range(status_page.KEY_FAILURES_ALLOWED):
        assert r.unlock(key="AAAA-BBBB-CCCC-DDDD-EEEE-FFFF")[0] == 403
    code, document = r.unlock()
    assert code == 429 and document["state"] == "locked" and r.change()[0] == 429
    assert r.server.controls_view()["locked_s"] == round(status_page.KEY_LOCKOUT_S)
    now[0] += status_page.KEY_LOCKOUT_S + 1
    assert r.unlock()[0] == 200 and r.change()[0] == 202
    assert r.controls.view()["counters"]["refused_locked"] == 2


def test_the_key_is_in_no_page_status_or_response(rig) -> None:
    r = rig()
    r.unlock()
    r.change()
    responses = [r.request("GET", path) for path in ("/", "/dashboard", "/status.json", "/status.json?page=1")]
    responses.append(r.request("POST", "/features", b"{}", **{**r.good(), "X_Sentinel_Control": None}))
    responses.append(r.request("POST", "/features", b'{"feature": "face", "state": "on"}', **r.good()))
    seen = b"".join(repr(headers).encode() + data for _, headers, data in responses)
    for form in (r.key, r.key.replace("-", ""), r.key.lower()):
        assert form.encode() not in seen
    document = json.loads(r.request("GET", "/status.json?page=1")[2])
    assert document["controls"] == {"enabled": True, "locked_s": 0, "key_file": CONTROL_KEY_NAME}


# ---------------------------------------------------------------- the key and its file


def test_a_key_is_120_random_bits_in_six_groups_compared_without_case_or_separators() -> None:
    keys = {new_control_key() for _ in range(50)}
    assert len(keys) == 50 and all(len(k) == 29 and k.count("-") == 5 for k in keys)
    key = keys.pop()
    assert normalized_key(key) == normalized_key(key.lower().replace("-", " ")) == key.replace("-", "").encode()


def test_the_key_file_is_private_replaces_a_stale_one_and_never_follows_a_link(tmp_path) -> None:
    path = write_control_key(tmp_path, "AAAA-BBBB")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600 and path.read_text() == "AAAA-BBBB\n"
    assert write_control_key(tmp_path, "CCCC-DDDD").read_text() == "CCCC-DDDD\n"  # a key a crash left is replaced
    assert remove_control_key(path) and not path.exists() and remove_control_key(path)
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    (tmp_path / CONTROL_KEY_NAME).symlink_to(target)
    with pytest.raises(OSError):
        write_control_key(tmp_path, "EEEE-FFFF")
    assert target.read_text() == "keep"


# ---------------------------------------------------------------- FeatureControls


def test_only_what_the_run_started_can_be_switched(rig) -> None:
    r = rig(switchable={"face": True})
    code, document = r.change("scene", "off")
    assert code == 409 and "restart" in document["reason"]
    assert r.controls.requested("scene") is False  # never started, so never on


def test_repeated_requests_coalesce_and_the_loop_applies_the_last_in_order() -> None:
    clock = FakeClock()
    controls = FeatureControls(clock, ALL)
    assert controls.request("face", "off")[0] == 202
    assert controls.request("scene", "off")[0] == 202
    assert controls.request("face", "on")[0] == 202  # replaces the waiting "off"
    taken = controls.take()
    assert [(c.feature, c.on) for c in taken] == [("scene", False), ("face", True)] and controls.take() == []
    for change in taken:
        controls.done(change, "applied" if change.feature == "scene" else "unchanged")
    log = controls.view()["log"]
    assert [(e["change"], e["feature"], e["requested"], e["result"]) for e in log] == [
        (1, "face", "off", "accepted"), (2, "scene", "off", "accepted"), (1, "face", "off", "replaced"),
        (3, "face", "on", "accepted"), (2, "scene", "off", "applied"), (3, "face", "on", "unchanged")]
    records = controls.drain_records()
    assert [r["result"] for r in records] == ["applied", "unchanged"] and controls.drain_records() == []
    assert set(records[0]) == {"control", "change", "at_utc", "feature", "requested", "result", "mono_ns"}
    with pytest.raises(ValueError):
        controls.done(taken[0], "accepted")


def test_changes_are_bounded_per_window_and_the_log_is_bounded() -> None:
    clock = FakeClock()
    controls = FeatureControls(clock, ALL)
    for n in range(RATE_LIMIT):
        assert controls.request("face", "off" if n % 2 == 0 else "on")[0] == 202
    assert controls.request("face", "off" if RATE_LIMIT % 2 == 0 else "on")[0] == 429
    clock.advance(10.1)
    assert controls.request("scene", "off")[0] == 202
    for n in range(40):
        clock.advance(2.0)
        controls.request("chat", "off" if n % 2 == 0 else "on")
    assert len(controls.view()["log"]) == 20


def test_names_and_states_are_exact() -> None:
    controls = FeatureControls(FakeClock(), ALL)
    for feature, state in (("Face", "off"), ("face", "OFF"), ("face", 0), (None, "off"), ("live view", "off")):
        assert controls.request(feature, state)[0] == 400
    with pytest.raises(ValueError):
        FeatureControls(FakeClock(), {"camera": True})


# ---------------------------------------------------------------- the panel, as the server renders it


def features_status(items: dict[str, dict[str, Any]], *, enabled: bool = True, log: list | None = None) -> dict:
    from datetime import datetime, timezone

    from sentinel.status_page import build_status

    snapshot = {**SNAPSHOT, "features": {"accepting": True, "items": items, "log": log or [], "counters": {},
                                         "rate": {"limit": 6, "window_s": 10.0}}}
    status = build_status(snapshot, {"state": "available"}, datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc))
    return {**status, "controls": {"enabled": enabled, "locked_s": 0, "key_file": CONTROL_KEY_NAME}}


def entry(requested: str, actual: str, why: str, *, switchable: bool = True, validation: str | None = None,
          problem: str | None = None) -> dict[str, Any]:
    return {"requested": requested, "actual": actual, "why": why, "switchable": switchable, "validation": validation,
            "problem": problem}


def text_of(html: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def test_the_panel_shows_requested_apart_from_actual_and_why_with_switches_the_script_enables() -> None:
    from sentinel.status_page import status_regions

    html = status_regions(features_status({
        "face": entry("on", "running", "running", validation="pending_validation"),
        "scene": entry("off", "stopping", "finishing"),
        "chat": entry("on", "running", "running", validation="validation_run"),
        "live_view": entry("off", "off", "turned_off"),
        "snapshots": entry("off", "off", "not_started", switchable=False),
    }))["features"]
    words = text_of(html)
    assert "Requested: on · now: running" in words and "Requested: off · now: stopping" in words
    assert "Stopping: the scene check in progress ends by its own deadline" in words
    assert "Experimental · pending validation: turning it on does not admit or validate it." in words
    assert "Validation run: chat is not admitted, and turning it on does not admit it." in words
    assert "Face recognition and chat are both on. They have been checked separately only; together they are not " \
           "validated." in words
    assert "This run did not start incident snapshots." in words and "Restart required to turn on incident snapshots" \
        in words and "--snapshots" in words and "Nothing restarts by itself" in words
    assert html.count("class=switch") == 5 and html.count(" disabled ") == 5  # the script enables them, never the HTML
    assert 'data-feature=snapshots data-want=on data-allowed=0' in html  # not started: never switchable
    assert 'aria-checked=true data-feature=face data-want=off data-allowed=1' in html
    assert "data-controls=on data-accepting=1" in html


def test_without_feature_controls_the_panel_is_read_only_and_says_how_features_are_chosen() -> None:
    from sentinel.status_page import status_regions

    status = features_status({}, enabled=False)
    status.pop("features")
    status["components"] = {"face": {"state": "available"}, "scene": {"state": "disabled"}}
    html = status_regions(status)["features"]
    words = text_of(html)
    assert "class=switch" not in html and "data-controls=off" in html
    assert "Read only: this run was started without --feature-controls" in words
    assert "How features are chosen at launch" in words and "Requested:" not in words


def test_the_change_log_is_shown_newest_first_with_fixed_labels_only() -> None:
    from sentinel.status_page import status_regions

    log = [{"change": 1, "at_utc": "2026-10-14T11:59:00+00:00", "feature": "face", "requested": "off",
            "result": "accepted"},
           {"change": 1, "at_utc": "2026-10-14T11:59:00.300000+00:00", "feature": "face", "requested": "off",
            "result": "applied"}]
    html = status_regions(features_status({"face": entry("off", "off", "turned_off")}, log=log))["features"]
    words = text_of(html)
    assert "Recent changes" in words and words.index("applied") < words.index("accepted")
    assert "No key or address is ever logged." in words
