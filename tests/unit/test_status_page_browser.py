"""D71 and D73: the status page's freshness in a real browser, with in-place updates.

Headless Chromium (``browser_harness``) loads the real StatusServer; the runtime and the camera are stand-ins
(``status_fixtures``) whose report and frames the test controls. Requests can be held, delayed, replayed, corrupted
or failed through the DevTools Fetch domain, without changing the server. What a viewer sees is read from the page's
visible text (``innerText``, which leaves out hidden elements) and computed styles.

Covered: updates in place (no reload; the picture never blanks); a stalled connection turning the page NOT CURRENT,
with bounded, non-overlapping retries, and recovering; a delayed answer getting less time and a too-late one none;
out-of-order, unreadable, invalid and failed answers restarting nothing; a picture that fails to load, and a frame
that is no longer live, both kept in place and marked as old; a hidden tab not polled and updated at once on return;
scroll, keyboard focus, expanded sections, the selected incident and the theme kept across updates; and the read-only
page without script.
"""

from __future__ import annotations

import base64
import contextlib
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, DevToolsError, Page, headless_browser, wait_until
from status_fixtures import Camera, Runtime, incident_store

from sentinel.status_page import PAGE_CURRENT_S, REFRESH_S, StatusServer
from sentinel.storage.database import Database

PAGE_CURRENT = "This page is current"
PAGE_NOT_CURRENT = f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s"
HEALTHY = "All components working"
HEALTHY_LAST_REPORTED = "Last reported: all components working"
LIVE = "the newest frame of the connected stream"  # the LIVE banner's reason, in its current layer
LIVE_LAST_REPORTED = "NOT CURRENT: last reported LIVE"
PICTURE_OLD = f"NOT CURRENT: this picture is old: no newer one has arrived for {PAGE_CURRENT_S} s."
WINDOW = PAGE_CURRENT_S
PROBE = """(() => {
  const figure = document.querySelector('#frame figure.frame');
  const img = figure && figure.querySelector('img');
  let opacity = null;
  if (img) { opacity = 1; for (let node = img; node; node = node.parentElement) opacity *= parseFloat(getComputedStyle(node).opacity); }
  const report = document.querySelector('.report');
  return {document: performance.timeOrigin, ready: document.readyState, text: document.body ? document.body.innerText : '',
          sequence: document.documentElement.dataset.sequence || null,
          sync: (document.getElementById('sync') || {}).textContent || '',
          report_opacity: report ? parseFloat(getComputedStyle(report).opacity) : null,
          generated: (document.querySelector('.freshbar time') || {getAttribute: () => null}).getAttribute('datetime'),
          picture: img ? {opacity: opacity, loaded: img.complete && img.naturalWidth > 0, src: img.getAttribute('src'),
                          state: figure.dataset.state || null, frame: figure.dataset.frame || null} : null};
})()"""


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def live(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A served page with live video (a new frame every 0.25 s)."""
    Database.open(tmp_path / "sentinel.db").close()
    runtime, camera = Runtime(), Camera()
    stop = camera.run()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "runtime": runtime, "camera": camera, "stop": stop,
               "db": tmp_path / "sentinel.db"}
    finally:
        stop.set()
        runtime.answering.set()
        assert server.stop(3.0)


def read(page: Page) -> dict[str, Any]:
    return page.evaluate(PROBE)


def wait_for(page: Page, predicate: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    seen: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        with contextlib.suppress(DevToolsError):
            seen = read(page)
            if seen["ready"] == "complete" and predicate(seen):
                return seen
        time.sleep(0.05)
    raise AssertionError(f"the page never reached the expected state; last seen: {seen!r}")


def current(seen: dict[str, Any]) -> bool:
    return all(claim in seen["text"] for claim in (PAGE_CURRENT, HEALTHY, LIVE))


def current_with_picture(seen: dict[str, Any]) -> bool:
    return current(seen) and bool((seen["picture"] or {}).get("loaded")) and seen["picture"]["state"] in (None, "live")


def not_current(seen: dict[str, Any]) -> bool:
    return PAGE_NOT_CURRENT in seen["text"] and not any(c in seen["text"] for c in (PAGE_CURRENT, HEALTHY, LIVE))


def assert_current(seen: dict[str, Any]) -> None:
    assert current(seen), seen["text"][:300]
    for label in (PAGE_NOT_CURRENT, HEALTHY_LAST_REPORTED, LIVE_LAST_REPORTED, PICTURE_OLD):
        assert label not in seen["text"], label
    assert seen["report_opacity"] == 1 and seen["picture"]["opacity"] == 1


def assert_not_current(seen: dict[str, Any]) -> None:
    assert not_current(seen), seen["text"][:300]
    for label in (HEALTHY_LAST_REPORTED, LIVE_LAST_REPORTED):
        assert label in seen["text"], label
    assert seen["report_opacity"] < 1


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    wait_for(page, current_with_picture, 30.0)
    return page


def test_updates_arrive_in_place_and_a_new_picture_replaces_the_old_without_blanking(
        browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    try:
        first = read(page)
        served = live["runtime"].served
        frames, blank = set(), 0
        deadline = time.monotonic() + 3 * REFRESH_S + 1
        while time.monotonic() < deadline:
            seen = read(page)
            if not (seen["picture"] and seen["picture"]["loaded"]):
                blank += 1  # no picture shown, or one that has not decoded: a flash
            frames.add(seen["picture"]["frame"] if seen["picture"] else None)
            assert seen["document"] == first["document"], "the page reloaded"
            assert_current(seen)
            time.sleep(0.03)
        assert blank == 0 and len(frames) >= 3  # new pictures arrived, and never a gap between them
        assert live["runtime"].served >= served + 2 and int(read(page)["sequence"]) > int(first["sequence"])
        assert "http-equiv" not in page.evaluate("document.head.innerHTML")  # no document-wide reload
    finally:
        page.close()


def test_a_stalled_connection_turns_the_page_not_current_and_only_a_fresh_update_restores_it(
        browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    runtime = live["runtime"]
    try:
        origin = read(page)["document"]
        runtime.answering.clear()  # as a stalled SSH forward: requests arrive and are never answered
        stalled = time.monotonic()
        seen = wait_for(page, not_current, WINDOW + REFRESH_S + 1.0)
        assert time.monotonic() - stalled <= WINDOW + REFRESH_S + 1.0
        assert_not_current(seen)
        assert PICTURE_OLD in seen["text"] and seen["picture"]["loaded"]  # the last frame stays, marked as old
        assert seen["picture"]["opacity"] < 0.5
        time.sleep(9.0)
        seen = read(page)
        assert_not_current(seen) and "No current update (no answer in time)" in seen["sync"]
        arrivals = [t for t in runtime.arrivals if t >= stalled]
        assert 1 <= len(arrivals) <= 4  # one at a time, abandoned after the window, then backing off
        assert all(b - a >= REFRESH_S for a, b in zip(arrivals, arrivals[1:]))
        runtime.answering.set()
        seen = wait_for(page, current_with_picture, 16.0)
        assert_current(seen) and seen["document"] == origin and seen["sync"] == ""
    finally:
        page.close()


def fulfil(page: Page, event: dict[str, Any], body: bytes, *, status: int = 200,
           kind: str = "application/json") -> None:
    page.call("Fetch.fulfillRequest", {"requestId": event["requestId"], "responseCode": status,
                                       "responseHeaders": [{"name": "Content-Type", "value": kind}],
                                       "body": base64.b64encode(body).decode()})


def test_a_delayed_answer_is_current_for_less_time_and_a_late_one_is_not_applied(
        browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    try:
        page.call("Fetch.enable", {"patterns": [{"urlPattern": "*status.json?page=1*", "requestStage": "Response"}]})
        event = page.paused(REFRESH_S + 3.0)
        page.call("Fetch.continueRequest", {"requestId": event["requestId"]})
        # Delayed: the answer is held 3.5 s. Its window still ends WINDOW s after its request, about 1.5 s after
        # it is applied; the next request is then held, so nothing renews it.
        event = page.paused(REFRESH_S + 3.0)
        held = time.monotonic()
        time.sleep(3.5)
        page.call("Fetch.continueRequest", {"requestId": event["requestId"]})
        applied = wait_for(page, lambda s: current(s) and s["sequence"] is not None, 2.0)
        before = int(applied["sequence"])
        waiting = page.paused(REFRESH_S + 3.0)  # the next request: held, never answered
        seen = wait_for(page, not_current, 3.0)
        assert time.monotonic() - held < WINDOW + 1.0  # ended by the delayed answer's own window, not later
        # Too late: the held request is abandoned after the window; answering it now changes nothing.
        time.sleep(WINDOW + 0.5)
        with contextlib.suppress(DevToolsError):
            page.call("Fetch.continueRequest", {"requestId": waiting["requestId"]})
        time.sleep(0.5)
        seen = read(page)
        assert_not_current(seen) and int(seen["sequence"]) == before
        page.call("Fetch.disable")
        assert_current(wait_for(page, current_with_picture, 12.0))
    finally:
        page.close()


def test_out_of_order_unreadable_invalid_and_failed_answers_restart_nothing(
        browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    try:
        page.call("Fetch.enable", {"patterns": [{"urlPattern": "*status.json?page=1*", "requestStage": "Response"}]})
        event = page.paused(REFRESH_S + 3.0)
        old = base64.b64decode(page.call("Fetch.getResponseBody", {"requestId": event["requestId"]})["body"])
        page.call("Fetch.continueRequest", {"requestId": event["requestId"]})
        event = page.paused(REFRESH_S + 3.0)
        last_valid = time.monotonic()
        page.call("Fetch.continueRequest", {"requestId": event["requestId"]})
        newest = int(wait_for(page, lambda s: int(s["sequence"] or 0) > json.loads(old)["page"]["sequence"], 3.0)
                     ["sequence"])
        stale_document = dict(json.loads(old), generated_utc="2001-01-01T00:00:00+00:00")
        answers = [
            (json.dumps(json.loads(old)).encode(), 200, "application/json"),  # an older document, replayed
            (json.dumps(stale_document).encode(), 200, "application/json"),  # older still, with any content
            (b"{not json", 200, "application/json"),
            (json.dumps({"page": {"instance": "zz", "sequence": 10 ** 6, "regions": {}}}).encode(), 200,
             "application/json"),
            (b"<html>not the API</html>", 200, "text/html"),
            (b"status unavailable\n", 500, "text/plain"),
        ]
        for body, status, kind in answers:
            event = page.paused(20.0)
            assert event is not None, "the page stopped asking"
            fulfil(page, event, body, status=status, kind=kind)
            seen = read(page)
            assert int(seen["sequence"]) == newest  # nothing older, unreadable or invalid was applied
        seen = read(page)
        assert time.monotonic() - last_valid > WINDOW
        assert_not_current(seen)  # and none of it renewed a claim
        assert "2001-01-01" not in page.evaluate("document.querySelector('.freshbar').innerHTML")
        page.call("Fetch.disable")
        assert_current(wait_for(page, current_with_picture, 20.0))
    finally:
        page.close()


def test_fast_failures_back_off_instead_of_flooding_the_runtime(browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    try:
        page.call("Fetch.enable", {"patterns": [{"urlPattern": "*status.json?page=1*", "requestStage": "Request"}]})
        started, asked = time.monotonic(), []
        while time.monotonic() - started < 13.0:  # every update fails at once, as a refused or busy answer would
            event = page.paused(0.5)
            if event:
                asked.append(time.monotonic())
                fulfil(page, event, b"busy\n", status=503, kind="text/plain")
        assert 2 <= len(asked) <= 4, asked  # 2, 4, 8 s apart: not one request per period, and never a flood
        gaps = [b - a for a, b in zip(asked, asked[1:])]
        assert all(later >= earlier * 1.6 for earlier, later in zip(gaps, gaps[1:])), gaps
        assert "No current update (HTTP 503); trying again in" in read(page)["sync"]
        page.call("Fetch.disable")
        assert_current(wait_for(page, current_with_picture, 20.0))
    finally:
        page.close()


def test_a_picture_that_fails_to_load_keeps_the_last_frame_marked_old_until_a_new_one_loads(
        browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    try:
        kept = read(page)["picture"]["src"]
        page.call("Fetch.enable", {"patterns": [{"urlPattern": "*/preview/*", "requestStage": "Request"}]})
        deadline = time.monotonic() + 3 * REFRESH_S + 2
        seen = read(page)
        while time.monotonic() < deadline and seen["picture"]["state"] != "failed":
            event = page.paused(0.5)
            if event:
                page.call("Fetch.failRequest", {"requestId": event["requestId"], "errorReason": "Failed"})
            seen = read(page)
        assert seen["picture"]["state"] == "failed" and seen["picture"]["src"] == kept and seen["picture"]["loaded"]
        assert "OLD FRAME: the newest picture could not be loaded; this one was received at" in seen["text"]
        assert seen["picture"]["opacity"] < 0.5 and current(seen)  # the status itself is still current
        page.call("Fetch.disable")
        seen = wait_for(page, lambda s: s["picture"]["state"] == "live" and s["picture"]["src"] != kept, 10.0)
        assert_current(seen) and "OLD FRAME" not in seen["text"]
    finally:
        page.close()


def test_a_frame_that_is_no_longer_live_stays_in_place_marked_as_an_old_frame(
        browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    try:
        live["stop"].set()  # the camera stops delivering frames
        time.sleep(0.3)
        kept = read(page)["picture"]["src"]
        live["camera"].clock.advance(3.0)  # the newest frame is now older than freshness.stale_after_s
        seen = wait_for(page, lambda s: s["picture"] and s["picture"]["state"] == "retained", 2 * REFRESH_S + 2)
        assert seen["picture"]["src"] == kept and seen["picture"]["loaded"] and seen["picture"]["opacity"] < 0.5
        assert "OLD FRAME: the camera has no current frame; this one was received at" in seen["text"]
        assert "STALE: not current" in seen["text"] and LIVE not in seen["text"] and PAGE_CURRENT in seen["text"]
    finally:
        page.close()


def test_a_hidden_tab_is_not_polled_and_updates_at_once_when_shown_again(browser: Browser, live: dict[str, Any]) -> None:
    page = opened(browser, f"{live['url']}/")
    runtime = live["runtime"]
    try:
        other = browser.page()  # another tab in front, and no focus emulation: headless reports the page hidden
        other.call("Page.bringToFront")
        page.call("Emulation.setFocusEmulationEnabled", {"enabled": False})
        assert wait_until(lambda: page.evaluate("document.visibilityState") == "hidden", 3.0)
        time.sleep(1.0)
        hidden = runtime.served
        time.sleep(WINDOW + 1.5)
        assert runtime.served <= hidden + 1  # at most a request already under way
        assert not_current(read(page))  # nothing renewed it meanwhile
        shown = time.monotonic()
        page.call("Page.bringToFront")
        page.call("Emulation.setFocusEmulationEnabled", {"enabled": True})
        other.close()
        seen = wait_for(page, current_with_picture, 3.0)
        assert time.monotonic() - shown < 3.0
        assert any(arrived >= shown for arrived in runtime.arrivals)  # made current by an update asked for on return
        assert_current(seen)
    finally:
        page.close()


MARK = """(() => { const e = document.activeElement; e.__testMark = 'kept'; return e.className; })()"""  # a property, not an attribute
STATE = """(() => ({y: Math.round(scrollY), focused: document.activeElement.__testMark || null,
  focusedClass: document.activeElement.className, details: [...document.querySelectorAll('details[open]')].map(d => d.id),
  selected: [...document.querySelectorAll('.inc-list > li.selected')].map(li => li.dataset.key),
  pressed: [...document.querySelectorAll('.peek[aria-pressed=true]')].length,
  preview: document.getElementById('inc-preview').hidden ? null : document.querySelector('#inc-preview h3').textContent,
  theme: document.documentElement.dataset.theme || null, sequence: Number(document.documentElement.dataset.sequence),
  origin: performance.timeOrigin}))()"""


def test_updates_keep_scroll_focus_expanded_sections_the_selected_incident_and_the_theme(
        browser: Browser, tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=12)
    runtime, camera = Runtime(), Camera()
    stop = camera.run()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    page = None
    try:
        page = browser.page(width=1280, height=700)
        page.navigate(f"http://127.0.0.1:{server.address[1]}/?theme=dark")
        wait_for(page, current_with_picture, 30.0)
        page.evaluate("document.getElementById('d-picture').open = true; document.getElementById('d-technical').open = true")
        page.evaluate(f"document.querySelector('.peek[data-select=\"{ids[3]}\"]').click()")
        wait_until(lambda: page.evaluate("!document.getElementById('inc-preview').hidden"), 5.0)
        page.evaluate("window.scrollTo(0, 420)")
        page.evaluate(f"document.querySelector('li[data-key=\"i-{ids[5]}\"] a.inc').focus()")
        assert page.evaluate(MARK) == "inc"
        before = page.evaluate(STATE)
        assert before["selected"] == [f"i-{ids[3]}"] and before["preview"] and before["y"] == 420
        deadline = time.monotonic() + 3 * REFRESH_S + 1
        while time.monotonic() < deadline:
            time.sleep(0.25)
            after = page.evaluate(STATE)
            assert after["focused"] == "kept" and after["y"] == 420, after
            assert set(after["details"]) >= {"d-picture", "d-technical"}, after
            assert after["selected"] == [f"i-{ids[3]}"] and after["pressed"] == 1 and after["preview"] == before["preview"]
            assert after["theme"] == "dark" and after["origin"] == before["origin"]
        assert page.evaluate(STATE)["sequence"] >= before["sequence"] + 2  # several updates came and went
    finally:
        if page is not None:
            page.close()
        stop.set()
        assert server.stop(3.0)


def test_without_script_the_page_is_read_only_and_marks_itself_not_current(browser: Browser, live: dict[str, Any]) -> None:
    page = browser.page()
    try:
        page.call("Emulation.setScriptExecutionDisabled", {"value": True})
        page.navigate(f"{live['url']}/")
        seen = wait_for(page, current_with_picture, 15.0)
        assert "Automatic updates need JavaScript" in seen["text"]
        served = live["runtime"].served
        seen = wait_for(page, not_current, WINDOW + 1.0)
        assert_not_current(seen) and live["runtime"].served == served  # nothing reloads or polls
        page.call("Page.reload")
        assert_current(wait_for(page, current_with_picture, 15.0))  # a reload that succeeds makes it current again
    finally:
        page.close()
