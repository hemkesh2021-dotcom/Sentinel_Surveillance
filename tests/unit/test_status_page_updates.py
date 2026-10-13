"""D73: what the status pages' in-place updates rely on, without a browser.

``/status.json`` keeps its shape (the guards and evidence profiles read it) and never encodes a picture.
``/status.json?page=1`` adds the page's regions, rendered by the same functions the full page embeds, and orders every
document within one server instance. The incident page has the same, from ``/incidents/<id>.json?page=1``. The one
script is served from this origin, GET only, under the same Host rule.
"""

from __future__ import annotations

import http.client
import json
import re
from pathlib import Path
from typing import Any

import pytest
from status_fixtures import Camera, incident_store

from sentinel.status_page import SCRIPT, StatusServer
from sentinel.storage.database import Database
from sentinel.web import head_script, script

SNAPSHOT = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": "2026-10-09T12:00:00+00:00"},
            "degraded": [], "components": {}}


def get(server: StatusServer, path: str, *, host: str | None = None, method: str = "GET") -> tuple[int, Any, bytes]:
    address, port = server.address
    conn = http.client.HTTPConnection(address, port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
        conn.endheaders()
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.fixture
def served(tmp_path: Path):
    ids = incident_store(tmp_path / "sentinel.db", count=3)
    camera = Camera()
    server = StatusServer(0, lambda: SNAPSHOT, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield server, ids, camera
    finally:
        assert server.stop(3.0)


def regions_in(page: str) -> dict[str, str]:
    """What a page embeds in each region: the inner HTML of every ``data-region`` element that is not the frame."""
    found = {}
    for match in re.finditer(r'<(div|footer) (?:class=(?:"[^"]*"|[\w-]+) )?data-region=([a-z-]+)>', page):
        tag, name = match.groups()
        if name == "frame-empty":
            continue
        start, depth, pos = match.end(), 1, match.end()
        pattern = re.compile(rf"<{tag}\b|</{tag}>")
        while depth:
            step = pattern.search(page, pos)
            depth += 1 if step.group().startswith(f"<{tag}") else -1
            pos = step.end()
        found[name] = page[start:step.start()]
    return found


def test_the_plain_status_document_is_unchanged_and_never_encodes(served) -> None:
    server, _, camera = served
    document = json.loads(get(server, "/status.json")[2])
    assert "page" not in document and document["preview"]["picture"] is None
    assert camera.preview.summary()["counters"].get("encodes", 0) == 0


def test_the_page_document_carries_the_same_regions_the_page_embeds_and_is_ordered(served) -> None:
    server, ids, camera = served
    page = get(server, "/")[2].decode()
    first = json.loads(get(server, "/status.json?page=1")[2])
    second = json.loads(get(server, "/status.json?page=1&theme=dark")[2])
    marker = re.search(r"data-instance=([0-9a-f]{16}) data-sequence=(\d+)", page)
    assert marker and first["page"]["instance"] == second["page"]["instance"] == marker[1]
    assert int(marker[1] and marker[2]) < first["page"]["sequence"] < second["page"]["sequence"]
    assert set(first["page"]["regions"]) == {"pills", "fresh", "live-head", "live-facts", "caps", "incidents",
                                             "incidents-meta", "faces", "chat-state", "health", "delivery", "foot",
                                             "features", "hidden"}  # D99, D100
    embedded = regions_in(page)
    assert set(embedded) == set(first["page"]["regions"])
    for name in ("live-head", "caps", "incidents", "delivery"):  # the parts that do not carry a time or a sequence
        assert embedded[name] == first["page"]["regions"][name], name
    assert first["preview"]["picture"] is None  # D75: pictures come from /live/frame, never from a page update
    assert camera.preview.summary()["counters"]["encodes"] == 1  # the page's own first picture only
    assert first["page"]["token"] == second["page"]["token"] and re.fullmatch(r"[0-9a-f]{32}", first["page"]["token"])
    assert f'href="/incidents/{ids[0]}?theme=dark"' in second["page"]["regions"]["incidents"]
    assert "<script" not in json.dumps(first["page"]) and first["page"]["frame_empty"].startswith("<div class=frame>")


def test_another_server_instance_numbers_its_documents_apart(tmp_path: Path) -> None:
    Database.open(tmp_path / "sentinel.db").close()
    one, two = (StatusServer(0, lambda: SNAPSHOT, tmp_path / "sentinel.db") for _ in range(2))
    try:
        assert one.marker()["instance"] != two.marker()["instance"]
        assert [one.marker()["sequence"] for _ in range(3)] == [2, 3, 4]
    finally:
        assert one.stop(1.0) and two.stop(1.0)  # never started: this closes their sockets


@pytest.mark.parametrize("query", ["page=2", "page=1x", "pages=1", "PAGE=1", "x=page=1"])
def test_only_the_exact_page_flag_asks_for_regions(served, query: str) -> None:
    server, _, _ = served
    assert "page" not in json.loads(get(server, f"/status.json?{query}")[2])


def test_fragments_are_escaped_like_the_page(tmp_path: Path) -> None:
    Database.open(tmp_path / "sentinel.db").close()
    hostile = {**SNAPSHOT, "degraded": ["<script>alert(1)</script>", "<img src=x onerror=alert(2)>"]}
    server = StatusServer(0, lambda: hostile, tmp_path / "sentinel.db")
    server.start()
    try:
        regions = json.loads(get(server, "/status.json?page=1")[2])["page"]["regions"]
        joined = "".join(regions.values())
        assert "<script>" not in joined and "&lt;script&gt;alert(1)&lt;/script&gt;" in joined
        assert "<img src=x" not in joined and "&lt;img src=x onerror=alert(2)&gt;" in joined
    finally:
        assert server.stop(3.0)


def test_the_incident_page_document_carries_its_regions(served) -> None:
    server, ids, _ = served
    page = get(server, f"/incidents/{ids[0]}")[2].decode()
    document = json.loads(get(server, f"/incidents/{ids[0]}.json?page=1")[2])
    assert document["incident"]["incident_id"] == ids[0]
    assert set(document["page"]["regions"]) == {"title", "snap", "summary", "ack", "delivery", "lifecycle", "scene",
                                                "tech"}
    assert set(regions_in(page)) == set(document["page"]["regions"])
    assert f'data-source="/incidents/{ids[0]}.json?page=1"' in page
    assert "page" not in json.loads(get(server, f"/incidents/{ids[0]}.json")[2])  # unchanged without the flag


def test_the_script_is_served_from_this_origin_only_and_read_only(served) -> None:
    server, _, _ = served
    status, headers, body = get(server, "/assets/dashboard.js")
    assert (status, headers["Content-Type"]) == (200, "text/javascript; charset=utf-8") and body == SCRIPT == script()
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["Cache-Control"] == "no-store"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert get(server, "/assets/dashboard.js", method="POST")[0] == 405
    assert get(server, "/assets/dashboard.js", host="evil.example")[0] == 421
    text = body.decode()
    # It talks to this origin only, keeps nothing but the viewer's appearance choices (D96: one key, read and written
    # in two places), and runs no code it is sent: never the conversation, the draft, a frame or a token in storage.
    for forbidden in ("eval(", "new Function", "localStorage", "sessionStorage", "indexedDB", "document.cookie",
                      "http://", "https://", "XMLHttpRequest", "postMessage", "innerHTML ="):
        if forbidden == "innerHTML =":
            assert text.count(forbidden) == 1 and "template.innerHTML = html" in text  # inert template parsing only
        elif forbidden in ("http://",):
            assert text.count(forbidden) == text.count("http://www.w3.org/2000/svg")  # the SVG namespace only
        elif forbidden == "localStorage":
            assert text.count(forbidden) == 2 and 'const STORE = "sentinel.appearance";' in text
            assert "window.localStorage.getItem(STORE)" in text and "window.localStorage.setItem(STORE, " in text
        else:
            assert forbidden not in text, forbidden


def test_the_head_script_only_reads_the_saved_appearance(served) -> None:
    """D96: the small head script applies the viewer's saved appearance before the first paint; it reads one key,
    writes nothing, takes only exact values and leaves a page whose address chose its appearance alone."""
    server, _, _ = served
    status, headers, body = get(server, "/assets/appearance.js")
    assert (status, headers["Content-Type"]) == (200, "text/javascript; charset=utf-8") and body == head_script()
    assert get(server, "/assets/appearance.js", method="POST")[0] == 405
    text = body.decode()
    assert text.count("localStorage") == 1 and 'getItem("sentinel.appearance")' in text and "setItem" not in text
    for forbidden in ("eval(", "new Function", "sessionStorage", "indexedDB", "document.cookie", "http", "fetch(",
                      "innerHTML", "document.write"):
        assert forbidden not in text, forbidden
    assert 'root.hasAttribute("data-pref")' in text and 'palettes.indexOf(saved.palette) > 0' in text
    page = get(server, "/")[2].decode()
    assert page.index('<script src="/assets/appearance.js"></script>') < page.index("<body>")


def test_pages_no_longer_reload_themselves_and_say_so_without_script(served) -> None:
    server, ids, _ = served
    for path in ("/", f"/incidents/{ids[0]}"):
        page = get(server, path)[2].decode()
        assert "http-equiv=refresh" not in page and '<script src="/assets/dashboard.js" defer></script>' in page
        assert "<noscript><p class='note noscript'>Automatic updates need JavaScript" in page
