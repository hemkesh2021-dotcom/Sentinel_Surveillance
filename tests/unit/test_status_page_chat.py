"""D74: the status page's chat routes, without a browser: only this page may ask, nothing else changes.

POST /chat is the page's only non-GET route. It needs the loopback Host, an Origin of exactly this origin (and
Sec-Fetch-Site same-origin when sent), a JSON body with a Content-Length within the bound, and the token this server
put in its page; GET /chat/<job> needs the token too. OPTIONS stays refused, so no cross-origin preflight can pass.
Chat is off without ``--chat`` and unavailable in this build (no admission); neither ever reaches the model.
"""

from __future__ import annotations

import http.client
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
from status_fixtures import Camera, incident_store

from sentinel.config import FreshnessConfig
from sentinel.media.health import assess_frame
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.chat import CHAT_ADMISSION, CHAT_MAX_BODY, ChatAdmission, ChatService
from sentinel.status_page import StatusServer

ADMITTED = ChatAdmission(True, "admitted for this test", "a synthetic fixture")
SNAPSHOT = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": "2026-10-09T12:00:00+00:00"},
            "degraded": [], "components": {}}


class Rig:
    def __init__(self, tmp_path: Path, admission: ChatAdmission = ADMITTED, *, enabled: bool = True) -> None:
        incident_store(tmp_path / "sentinel.db", count=1)
        self.camera = Camera()
        self.questions: list[str] = []
        self.analyzer = ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="test")
        self.analyzer.start()
        started = self.camera.clock.mono()
        self.service = ChatService(
            enabled=True, admission=admission, scene=self.analyzer, latest=self.camera.frames.latest,
            assess=lambda f: assess_frame(f, self.camera.stamper.current_stream, self.camera.clock.mono(), started,
                                          FreshnessConfig()),
            clock=self.camera.clock, run=self.answer, stale_after_s=2.0)
        self.server = StatusServer(0, lambda: SNAPSHOT, tmp_path / "sentinel.db", preview=lambda: self.camera.preview,
                                   chat=(lambda: self.service) if enabled else None)
        self.server.start()
        self.port = self.server.address[1]
        self.origin = f"http://127.0.0.1:{self.port}"

    def answer(self, question: Any, image: Any, abort: Any = None) -> tuple[str, bool]:
        self.questions.append(question.question)
        return f"Seen: {question.question}", False

    def request(self, method: str, path: str, body: bytes | None = None, **headers: str) -> tuple[int, dict, bytes]:
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

    def ask(self, question: str = "Is the door open?", **headers: str) -> tuple[int, dict, bytes]:
        good = {"Origin": self.origin, "Content_Type": "application/json", "X_Sentinel_Token": self.server.chat_token,
                "Sec_Fetch_Site": "same-origin"}
        return self.request("POST", "/chat", json.dumps({"question": question}).encode(), **{**good, **headers})

    def close(self) -> None:
        assert self.server.stop(3.0) and self.analyzer.stop(3.0)


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


def body(response: tuple[int, dict, bytes]) -> dict[str, Any]:
    return json.loads(response[2])


def test_a_question_from_this_page_is_answered_and_read_back_with_the_token(rig) -> None:
    r = rig()
    code, headers, raw = r.ask()
    accepted = json.loads(raw)
    assert code == 202 and accepted["state"] in ("waiting", "answering") and headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers
    for _ in range(200):
        answer = body(r.request("GET", f"/chat/{accepted['job']}", X_Sentinel_Token=r.server.chat_token))
        if answer["state"] == "answered":
            break
        time.sleep(0.01)
    assert answer["answer"] == "Seen: Is the door open?" and answer["frame"]["frame_seq"] == 0
    assert r.request("GET", f"/chat/{accepted['job']}")[0] == 403  # no token, no answer
    assert r.request("GET", "/chat/" + "0" * 16, X_Sentinel_Token=r.server.chat_token)[0] == 404
    assert r.request("GET", "/chat/../status.json", X_Sentinel_Token=r.server.chat_token)[0] == 404


@pytest.mark.parametrize(("headers", "code"), [
    ({"Origin": None}, 403),
    ({"Origin": "http://evil.example"}, 403),
    ({"Origin": "http://127.0.0.1:1"}, 403),
    ({"Origin": "null"}, 403),
    ({"Sec_Fetch_Site": "cross-site"}, 403),
    ({"Sec_Fetch_Site": "same-site"}, 403),
    ({"Content_Type": "text/plain"}, 415),
    ({"Content_Type": "application/x-www-form-urlencoded"}, 415),
    ({"X_Sentinel_Token": None}, 403),
    ({"X_Sentinel_Token": "0" * 32}, 403),
    ({"X_Sentinel_Token": "\u00e9" * 32}, 403),  # not ASCII: refused like any wrong token, not an error
    ({"Host": "evil.example"}, 421),
    ({"Transfer_Encoding": "chunked"}, 411),
])
def test_only_this_page_may_ask(rig, headers: dict[str, Any], code: int) -> None:
    r = rig()
    response = r.ask(**headers)
    assert response[0] == code, response
    assert r.questions == [] and "Is the door open" not in response[2].decode()  # never echoed


def test_bodies_are_bounded_and_validated_and_nothing_else_takes_a_post(rig) -> None:
    r = rig()
    good = {"Origin": r.origin, "Content_Type": "application/json", "X_Sentinel_Token": r.server.chat_token}
    assert r.request("POST", "/chat", b"x" * (CHAT_MAX_BODY + 1), **good)[0] == 413
    response = r.request("POST", "/chat", b"{not json", **good)
    assert response[0] == 400 and body(response) == {"state": "invalid", "reason": "request is not JSON"}
    assert r.request("POST", "/chat", json.dumps({"question": "q", "act": "ack"}).encode(), **good)[0] == 400
    for path in ("/", "/status.json", "/incidents/x", "/chat/0123456789abcdef"):
        assert r.request("POST", path, b"{}", **good)[0] == 405
    options = r.request("OPTIONS", "/chat", **{"Origin": "http://evil.example",
                                               "Access_Control_Request_Method": "POST"})
    assert options[0] == 405 and "Access-Control-Allow-Origin" not in options[1]
    assert r.questions == []


def test_chat_off_or_not_admitted_says_why_and_never_asks(rig) -> None:
    off = rig(enabled=False)
    response = off.ask()
    assert response[0] == 503 and body(response)["state"] == "off"
    built = rig(admission=CHAT_ADMISSION)
    response = built.ask()
    assert response[0] == 503 and body(response) == {"state": "unavailable", "reason": CHAT_ADMISSION.reason}
    page = built.request("GET", "/")[2].decode()
    assert 'data-chat-state="unavailable"' in page and "operator-run resource and interference checks" in page
    assert f'data-token="{built.server.chat_token}"' in page and "<button type=submit class=button id=chat-ask disabled>" in page
    assert built.questions == [] and off.questions == []


def test_chat_state_reaches_the_page_documents_only_and_no_question_is_kept(rig) -> None:
    r = rig()
    r.ask("A private question")
    assert "chat" not in body(r.request("GET", "/status.json"))  # the plain document is unchanged
    page = body(r.request("GET", "/status.json?page=1"))
    assert page["chat"]["state"] == "available" and 'data-chat-state="available"' in page["page"]["regions"]["chat-state"]
    assert "A private question" not in json.dumps(page) and "A private question" not in r.request("GET", "/")[2].decode()
    assert "A private question" not in json.dumps(r.service.summary())
    assert re.search(r'data-token="[0-9a-f]{32}"', r.request("GET", "/")[2].decode())
