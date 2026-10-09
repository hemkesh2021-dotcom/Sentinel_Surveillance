"""D71: the status page in a real browser. A page that stops refreshing drops its current claims; only a fresh page restores them.

Headless Chromium loads the real StatusServer (``browser_harness``). The runtime is a stand-in whose report the test
sets (``status_fixtures``).

- **Refresh loss** is a refresh request that gets no answer, as through a stalled SSH forward: the browser keeps
  showing the old page. While that request is pending, Chromium holds DevTools messages for the page, so the test
  reads what is on screen from a screenshot (composited by the browser): the page's green, which only current claims
  use, must be gone, and the NOT CURRENT warning shown. A refresh the browser gives up (``Page.stopLoading``) leaves
  the same old page, whose visible text (``innerText``, which leaves out hidden elements) is then read.
- **Recovery** is a late answer to the pending refresh, the page's next refresh, or a reload. A fresh page whose
  report is stale stays without current claims: the server-side freshness checks still decide.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from browser_harness import DevToolsError, Page, headless_browser, rgb_pixels, wait_until
from status_fixtures import Camera, Runtime

from sentinel.status_page import PAGE_CURRENT_S, REFRESH_S, StatusServer
from sentinel.storage.database import Database

# What a viewer can read: the page's own freshness, the runtime's health, and the live view.
PAGE_CURRENT = "This page is current"
PAGE_NOT_CURRENT = f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s"
HEALTHY = "All components working"
HEALTHY_LAST_REPORTED = "Last reported: all components working"
LIVE = "the newest frame of the connected stream"  # the LIVE banner's reason, in its current layer
LIVE_LAST_REPORTED = "NOT CURRENT: last reported LIVE"
PICTURE_NOT_CURRENT = f"NOT CURRENT: this picture is old; the page has not refreshed for {PAGE_CURRENT_S} s"
PROBE = """(() => {
  const img = document.querySelector('.live .frame img');
  let opacity = null;
  if (img) {
    opacity = 1;
    for (let node = img; node; node = node.parentElement) opacity *= parseFloat(getComputedStyle(node).opacity);
  }
  const report = document.querySelector('.report');
  return {document: performance.timeOrigin, since_navigation_ms: performance.now(), ready: document.readyState,
          text: document.body ? document.body.innerText : '',
          report_opacity: report ? parseFloat(getComputedStyle(report).opacity) : null,
          generated: (document.querySelector('.freshbar time') || {getAttribute: () => null}).getAttribute('datetime'),
          picture: img ? {opacity: opacity, loaded: img.complete && img.naturalWidth > 0} : null};
})()"""
VIEWPORT = (1000, 1000)
SHOT_SCALE = 0.25
BAD_RGB = (0x7A, 0x2E, 0x12)  # STYLE's --stale-bg, light theme: undimmed only in the page's own NOT CURRENT warning
GREEN_TINT = 2  # g - max(r, b): only current claims are green (--ok), dimmed or not; text is grey (no LCD fringes)
# Quarter-scale resampling leaves at most a few stray green-tinted pixels (1 measured on a hung page); the smallest
# single current claim, the 'Page current' pill alone, measured 218 and the whole current page 572 (session 54).
GREEN_NOISE_PX = 24


def read_page(page: Page) -> dict[str, Any]:
    return page.evaluate(PROBE)


def screenshot_colours(page: Page) -> dict[str, int]:
    """What is on screen (works while a load is pending): green-tinted pixels, and the warning's colour."""
    pixels = rgb_pixels(page.screenshot(scale=SHOT_SCALE))
    return {"green": sum(g - max(r, b) >= GREEN_TINT for r, g, b in pixels), "warning": pixels.count(BAD_RGB)}


def wait_for(page: Page, predicate: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    seen: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        with contextlib.suppress(DevToolsError):  # a page being replaced has no context for a moment
            seen = read_page(page)
            if seen["ready"] == "complete" and predicate(seen):
                return seen
        time.sleep(0.1)
    raise AssertionError(f"the page never reached the expected state; last seen: {seen!r}")


def claims_current(seen: dict[str, Any]) -> bool:
    return PAGE_CURRENT in seen["text"] and HEALTHY in seen["text"] and LIVE in seen["text"]


def current_with_picture(seen: dict[str, Any]) -> bool:
    return claims_current(seen) and bool((seen["picture"] or {}).get("loaded"))


def generated(seen: dict[str, Any]) -> datetime:
    """When the server built the page shown, from its own time element."""
    assert seen["generated"], seen["text"][:200]
    return datetime.fromisoformat(seen["generated"])


def assert_current(seen: dict[str, Any]) -> None:
    for label in (PAGE_NOT_CURRENT, HEALTHY_LAST_REPORTED, LIVE_LAST_REPORTED, PICTURE_NOT_CURRENT):
        assert label not in seen["text"], label
    assert seen["picture"]["opacity"] == 1 and seen["report_opacity"] == 1


def assert_not_current(seen: dict[str, Any]) -> None:
    assert seen["since_navigation_ms"] >= PAGE_CURRENT_S * 1000  # never sooner than PAGE_CURRENT_S after loading began
    for claim in (PAGE_CURRENT, HEALTHY, LIVE):
        assert claim not in seen["text"], claim
    for label in (PAGE_NOT_CURRENT, HEALTHY_LAST_REPORTED, LIVE_LAST_REPORTED, PICTURE_NOT_CURRENT):
        assert label in seen["text"], label
    assert seen["picture"]["opacity"] < 0.2 and seen["report_opacity"] < 1


def lose_refresh(runtime: Runtime) -> datetime:
    """The page's next refresh gets no answer; returns when it was cut off. The page shown was built before then."""
    cut_off = datetime.now(timezone.utc)
    runtime.answering.clear()
    before = runtime.unanswered
    assert wait_until(lambda: runtime.unanswered > before, REFRESH_S + 3.0)  # the browser asked, and waits
    return cut_off


def test_a_page_that_stops_refreshing_drops_every_current_claim_until_a_fresh_page_loads(tmp_path: Path) -> None:
    Database.open(tmp_path / "sentinel.db").close()
    runtime, camera = Runtime(), Camera()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        with headless_browser() as browser:
            page = browser.page(width=VIEWPORT[0], height=VIEWPORT[1])
            page.navigate(f"http://127.0.0.1:{server.address[1]}/")

            # 1. Current: the page, the runtime's health and the live view say so; the picture is shown.
            assert_current(wait_for(page, current_with_picture, 30.0))
            on_screen = screenshot_colours(page)
            assert on_screen["green"] > GREEN_NOISE_PX and on_screen["warning"] == 0, on_screen

            # 2. Refresh loss: the refresh hangs and the browser keeps the old page. Once that page is older than
            # PAGE_CURRENT_S, nothing green (no current claim, dimmed or not) is on screen, and the warning is.
            lose_refresh(runtime)
            time.sleep(PAGE_CURRENT_S + 1.0)
            on_screen = screenshot_colours(page)
            assert on_screen["green"] <= GREEN_NOISE_PX and on_screen["warning"] > 0, on_screen

            # A late answer, carrying a stale report, gives a current page that claims nothing current: the
            # server-side checks still decide what is live and healthy.
            runtime.lag_s = 30.0
            camera.clock.advance(3.0)  # the newest frame is now older than freshness.stale_after_s
            runtime.answering.set()
            answered = wait_for(page, lambda s: PAGE_CURRENT in s["text"], 10.0)
            assert "STALE: not current" in answered["text"] and "runtime status not updated for 30 s" in answered["text"]
            for claim in (HEALTHY, LIVE, PAGE_NOT_CURRENT, HEALTHY_LAST_REPORTED, LIVE_LAST_REPORTED):
                assert claim not in answered["text"], claim
            assert answered["picture"] is None

            # 3. A fresh report: the page's next refresh restores every current claim and the picture.
            runtime.lag_s = 0.0
            camera.frame()
            restored = wait_for(page, current_with_picture, REFRESH_S + 5.0)
            assert restored["document"] != answered["document"]
            assert_current(restored)
            assert screenshot_colours(page)["green"] > GREEN_NOISE_PX

            # 4. A refresh the browser gives up: the same old page stays, and its visible text says it is not
            # current, with the runtime's report as last reported.
            cut_off = lose_refresh(runtime)
            page.call("Page.stopLoading")
            stale = wait_for(page, lambda s: PAGE_NOT_CURRENT in s["text"], PAGE_CURRENT_S + 3.0)
            assert generated(stale) <= cut_off  # built before the loss: never replaced
            assert_not_current(stale)
            time.sleep(1.0)
            still = read_page(page)
            assert still["document"] == stale["document"]
            assert_not_current(still)

            # Only a refresh that succeeds (here the operator's reload) makes it current again.
            runtime.answering.set()
            page.call("Page.reload")
            reloaded = wait_for(page, current_with_picture, 10.0)
            assert reloaded["document"] != stale["document"] and generated(reloaded) > cut_off
            assert_current(reloaded)
    finally:
        runtime.answering.set()
        assert server.stop(3.0)
