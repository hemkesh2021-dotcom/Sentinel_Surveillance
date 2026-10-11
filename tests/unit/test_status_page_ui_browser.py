"""D72: the styled status pages in a real browser (headless Chromium; ``browser_harness``), with synthetic data only.

Covered: incidents beside the live view on desktop and below it on mobile; no horizontal overflow with long content;
light and dark themes, from the browser's preference and from the page's own choice; keyboard reachability with a
visible focus ring; reduced motion (transitions off, the D71 watchdog still running); and the watchdog failing closed
when animations do not run at all. The page's self-reload is held off here (a long refresh interval), so that what is
on screen can be followed; how updates keep focus and reading state is tested with the partial-update increment.
Screenshots can be saved for review with SENTINEL_TEST_SCREENSHOTS; they never decide a result.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser
from status_fixtures import Camera, Runtime, incident_store

from sentinel import status_page
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.incidents.service import IncidentService
from sentinel.media.clock import FakeClock
from sentinel.status_page import PAGE_CURRENT_S, StatusServer
from sentinel.storage.database import Database

RECTS = """(() => { const r = id => { const b = document.getElementById(id).getBoundingClientRect();
  return {left: b.left, right: b.right, top: b.top, bottom: b.bottom, width: b.width}; };
  return {live: r('live'), incidents: r('incidents'), scroll: document.documentElement.scrollWidth,
          client: document.documentElement.clientWidth}; })()"""


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 3600)  # no self-reload while a test follows the screen
    ids = incident_store(tmp_path / "sentinel.db", count=15, long_titles=True)
    runtime, camera = Runtime(), Camera()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "ids": ids, "runtime": runtime, "camera": camera}
    finally:
        assert server.stop(3.0)


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if page.evaluate("document.readyState") == "complete":
            return page
        time.sleep(0.05)
    raise AssertionError("the page did not load")


def test_incidents_are_beside_the_camera_on_desktop_and_below_it_on_mobile_with_no_sideways_scroll(
        browser: Browser, site: dict[str, Any]) -> None:
    for name, width, height, mobile in (("desktop", 1280, 900, False), ("mobile", 390, 844, True)):
        for theme in ("light", "dark"):
            page = opened(browser, f"{site['url']}/?theme={theme}", width=width, height=height, mobile=mobile)
            try:
                rects = page.evaluate(RECTS)
                assert rects["scroll"] <= rects["client"], (name, theme, rects)
                live, incidents = rects["live"], rects["incidents"]
                if mobile:
                    assert incidents["top"] >= live["bottom"] - 1 and abs(incidents["width"] - live["width"]) < 2, rects
                else:
                    assert incidents["left"] >= live["right"] - 1 and incidents["top"] < live["bottom"], rects
                    assert live["width"] > incidents["width"]  # the live view is the larger panel
                page.save_screenshot(f"status-{name}-{theme}", full=True)
                detail = opened(browser, f"{site['url']}/incidents/{site['ids'][1]}?theme={theme}",
                                width=width, height=height, mobile=mobile)
                try:
                    sizes = detail.evaluate("({s: document.documentElement.scrollWidth, "
                                            "c: document.documentElement.clientWidth})")
                    assert sizes["s"] <= sizes["c"], (name, theme, sizes)
                    detail.save_screenshot(f"incident-{name}-{theme}", full=True)
                finally:
                    detail.close()
            finally:
                page.close()


@pytest.mark.parametrize(("preference", "query", "background"), [
    ("light", "", "rgb(247, 239, 229)"), ("dark", "", "rgb(28, 29, 28)"),  # Carbon & Cream (D82)
    ("light", "?theme=dark", "rgb(28, 29, 28)"), ("dark", "?theme=light", "rgb(247, 239, 229)"),
])
def test_light_and_dark_follow_the_browser_unless_the_page_chooses(
        browser: Browser, site: dict[str, Any], preference: str, query: str, background: str) -> None:
    page = opened(browser, f"{site['url']}/{query}", media={"prefers-color-scheme": preference})
    try:
        seen = page.evaluate("""(() => ({bg: getComputedStyle(document.body).backgroundColor,
            selected: [...document.querySelectorAll('.themes a[aria-current=true]')].map(a => [a.innerText.trim(),
              getComputedStyle(a).backgroundColor])}))()""")
        assert seen["bg"] == background
        expected = {"": "Auto", "?theme=dark": "Dark", "?theme=light": "Light"}[query]
        accent = "rgb(46, 48, 47)" if background == "rgb(247, 239, 229)" else "rgb(237, 220, 201)"
        assert seen["selected"] == [[expected, accent]]  # the theme's accent (carbon, or cream when dark) marks it
    finally:
        page.close()


def test_a_section_link_brings_its_section_into_view_on_a_phone(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/", width=390, height=844, mobile=True)
    try:
        page.evaluate("document.querySelector('.jump a[href=\"#chat\"]').click()")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            seen = page.evaluate("""(() => { const r = document.getElementById('chat').getBoundingClientRect();
              return {hash: location.hash, top: Math.round(r.top), sideways: document.documentElement.scrollWidth
                      > document.documentElement.clientWidth}; })()""")
            if seen["hash"] == "#chat" and 0 <= seen["top"] <= 40:
                break
            time.sleep(0.05)
        assert seen["hash"] == "#chat" and 0 <= seen["top"] <= 40 and not seen["sideways"], seen
    finally:
        page.close()


FOCUSED = """(() => { const e = document.activeElement; const s = getComputedStyle(e);
  return {tag: e.tagName, cls: e.className, text: (e.innerText || '').trim().slice(0, 60),
          outline: s.outlineStyle, width: parseFloat(s.outlineWidth), inLive: !!e.closest('#live'),
          inIncidents: !!e.closest('#incidents')}; })()"""


def test_every_control_is_reachable_by_keyboard_with_a_visible_focus_ring(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/", width=1280, height=900)
    try:
        page.key("Tab")
        first = page.evaluate(FOCUSED)
        assert first["cls"] == "skip" and first["text"] == "Skip to the live view"
        assert page.evaluate("document.activeElement.getBoundingClientRect().top") >= 0  # shown when focused
        page.key("Enter")
        page.key("Tab")
        assert page.evaluate(FOCUSED)["inLive"]  # the skip link moved the keyboard to the live view
        page.navigate(f"{site['url']}/")
        time.sleep(0.5)
        seen = []
        for _ in range(80):  # every stop once, until focus leaves the document again
            page.key("Tab")
            focus = page.evaluate(FOCUSED)
            if focus["tag"] == "BODY" or (seen and focus["cls"] == "skip"):
                break
            seen.append(focus)
            assert focus["outline"] != "none" and focus["width"] >= 2, focus  # a visible ring on every stop
        assert seen and seen[0]["cls"] == "skip" and len(seen) < 80
        texts = [f["text"] for f in seen]
        assert {"Auto", "Light", "Dark"} <= set(texts) and "About this picture" in texts
        assert sum(1 for f in seen if f["cls"] == "inc") >= 10 and any(f["inIncidents"] for f in seen)
        summary = next(i for i, f in enumerate(seen) if f["text"] == "About this picture")
        page.navigate(f"{site['url']}/")
        time.sleep(0.5)
        for _ in range(summary + 1):
            page.key("Tab")
        page.key("Enter")
        assert page.evaluate("document.getElementById('d-picture').open") is True  # opened from the keyboard
    finally:
        page.close()


TEXT = "document.body.innerText"
PLAYER = "document.getElementById('player-state').innerText.trim()"


def test_reduced_motion_turns_transitions_off_and_keeps_the_freshness_watchdog(browser: Browser, site: dict[str, Any]) -> None:
    for preference, transition in (("no-preference", "0.18s"), ("reduce", "0s")):
        page = opened(browser, f"{site['url']}/", media={"prefers-reduced-motion": preference})
        try:
            started = time.monotonic()
            assert page.evaluate("getComputedStyle(document.querySelector('.inc')).transitionDuration").startswith(
                transition)
            text = page.evaluate(TEXT)
            assert time.monotonic() - started < PAGE_CURRENT_S - 1
            assert "This page is current" in text and page.evaluate(PLAYER) == "LIVE"  # the player's claim (D75)
            time.sleep(PAGE_CURRENT_S + 0.5)  # nothing refreshes it here, and the camera stand-in makes no new frame
            text = page.evaluate(TEXT)
            assert f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s" in text
            assert "This page is current" not in text and "All components working" not in text
            assert page.evaluate(PLAYER) != "LIVE" and "the newest frame of the connected stream" not in text
        finally:
            page.close()


def test_a_browser_that_runs_no_animation_never_shows_a_current_claim(browser: Browser, site: dict[str, Any]) -> None:
    page = browser.page()
    try:
        page.call("Page.enable")
        page.call("Page.setBypassCSP", {"enabled": True})  # only so the test can add its own stylesheet
        page.navigate(f"{site['url']}/")
        time.sleep(0.8)
        page.evaluate("""(() => { const s = document.createElement('style');
            s.textContent = '*, *::before, *::after { animation: none !important; }';
            document.head.appendChild(s); })()""")
        text = page.evaluate(TEXT)
        assert "NOT CURRENT" in text and "This page is current" not in text
        assert "All components working" not in text and "the newest frame of the connected stream" not in text
        assert page.evaluate("parseFloat(getComputedStyle(document.querySelector('.live .frame img')).opacity)") < 0.5
    finally:
        page.close()


def test_the_incident_page_updates_in_place_keeping_focus_expanded_provenance_and_the_snapshot(
        browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_page, "DETAIL_REFRESH_S", 1)
    ids = incident_store(tmp_path / "sentinel.db", count=2, acknowledge=False)
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1"}}, tmp_path / "sentinel.db")
    server.start()
    page = None
    try:
        page = opened(browser, f"http://127.0.0.1:{server.address[1]}/incidents/{ids[0]}", width=1280, height=900)
        assert "Not acknowledged" in page.evaluate(TEXT)
        page.evaluate("""(() => { document.getElementById('d-provenance').open = true;
            const link = document.querySelector('a.button[download]'); link.focus(); link.__testMark = 'kept';
            document.querySelector('#snapshot img').__testMark = 'kept'; })()""")
        origin = page.evaluate("performance.timeOrigin")
        db = Database.open(tmp_path / "sentinel.db")  # what `sentinel incident ack` does, while this page is open
        try:
            IncidentService(db, FakeClock(), incidents=IncidentsConfig(), notifications=NotificationsConfig()
                            ).acknowledge(ids[0], actor="operator:tester", reason="seen in the rehearsal")
        finally:
            db.close()
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline and "by operator:tester: seen in the rehearsal" not in page.evaluate(TEXT):
            time.sleep(0.1)
        state = page.evaluate("""(() => ({origin: performance.timeOrigin,
            focus: document.activeElement.__testMark || null, open: document.getElementById('d-provenance').open,
            img: (document.querySelector('#snapshot img') || {}).__testMark || null,
            text: document.body.innerText}))()""")
        assert "by operator:tester: seen in the rehearsal" in state["text"] and "Not acknowledged" not in state["text"]
        assert state["origin"] == origin and state["focus"] == "kept" and state["open"] and state["img"] == "kept"
    finally:
        if page is not None:
            page.close()
        assert server.stop(3.0)
