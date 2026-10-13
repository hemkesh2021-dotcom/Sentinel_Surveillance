"""D100: the dashboard's browser-only viewing preferences, kept apart from the runtime's features.

The live picture fits its panel or shows at its own pixels; the optional panels (Ask, Health, Alerts) can be hidden in
this browser. Both are saved with the appearance, applied before the first paint, and change nothing the runtime does:
no request is made for them, hiding Ask leaves chat as it is, and the picture is the same one. A hidden panel's
warning stays in view (the strip under the freshness bar), and so do the freshness bar and NOT CURRENT.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser, wait_until
from status_fixtures import Camera, Runtime, incident_store

from sentinel import status_page
from sentinel.status_page import StatusServer, build_status, render_html, status_regions

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def page_status(*, degraded: list[str] | None = None, totals: dict[str, int] | None = None) -> dict:
    snapshot = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": NOW.isoformat()},
                "degraded": degraded or [], "components": {}}
    store = {"state": "available", "delivery": {"totals": totals or {}, "recent": []}}
    return build_status(snapshot, store, NOW)


# ---------------------------------------------------------------- as the server renders it


def test_a_hidden_panels_warning_is_rendered_for_the_strip_and_nothing_else_is() -> None:
    assert status_regions(page_status())["hidden"] == ""
    html = status_regions(page_status(degraded=["video stale", "detector unavailable (engine)"],
                                      totals={"failed": 1, "ambiguous": 2}))["hidden"]
    words = re.sub(r"<[^>]+>", " ", html)
    assert "Health is hidden here, and the runtime reports a problem: video stale and 1 more." in words
    assert "Alerts is hidden here: 1 failed (needs the operator) and 2 ambiguous alert deliveries." in words
    assert html.count("data-show=") == 2 and 'data-panel=health' in html and 'data-panel=delivery' in html


def test_the_controls_are_script_only_and_say_they_change_no_feature() -> None:
    html = render_html(page_status())
    assert re.search(r'<div class="segs view-size js-only" role=group aria-label="Picture size in this browser">', html)
    buttons = re.findall(r"<button type=button class=seg data-view=(\w+) aria-pressed=(\w+)>", html)
    assert buttons == [("fit", "true"), ("actual", "false")] * 2  # beside the picture and in Appearance
    assert all(f"data-panel={name}" in html for name in ("chat", "health", "delivery"))
    assert "data-panel=live" not in html and "data-panel=faces" not in html and "data-panel=incidents" not in html
    assert "they change no feature (a hidden panel's feature keeps running; use Features for that)" in html
    assert '<div class="hidden-panels js-only" data-region=hidden>' in html


# ---------------------------------------------------------------- in a real browser


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)
    incident_store(tmp_path / "sentinel.db", count=3)
    runtime = Runtime(degraded=["video stale"])
    camera = Camera()
    stop = camera.run(0.1)
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "runtime": runtime}
    finally:
        stop.set()
        assert server.stop(3.0)
        camera.preview.close(2.0)


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0), "the page did not load"
    return page


def shown(page: Page, selector: str) -> bool:
    return page.evaluate(f"(() => {{ const e = document.querySelector({selector!r}); "
                         "return !!e && getComputedStyle(e).display !== 'none'; })()")


def test_actual_size_shows_the_picture_at_its_own_pixels_and_is_kept_on_reload(browser: Browser,
                                                                               site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        frame = "document.querySelector('.live .frame canvas, .live .frame img')"
        assert wait_until(lambda: page.evaluate(f"!!{frame}"), 8.0)
        fit = page.evaluate(f"{frame}.getBoundingClientRect().width")
        page.evaluate("document.querySelector('.view-size button[data-view=actual]').click()")
        natural = page.evaluate(f"{frame}.width || {frame}.naturalWidth")
        assert wait_until(lambda: abs(page.evaluate(f"{frame}.getBoundingClientRect().width") - natural) < 1.5, 3.0)
        assert fit > natural  # fit had scaled the small synthetic picture up to the panel
        assert page.evaluate("document.querySelector('.view-size button[data-view=actual]').getAttribute('aria-pressed')") \
            == "true"
        saved = json.loads(page.evaluate("localStorage.getItem('sentinel.appearance')"))
        assert saved["view"] == "actual" and set(saved) <= {"theme", "palette", "density", "rail", "view", "hide"}
        page.navigate(f"{site['url']}/dashboard")
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 10.0)
        assert page.evaluate("document.documentElement.dataset.view") == "actual"  # applied before the first paint
        page.evaluate("document.querySelector('.view-size button[data-view=fit]').click()")
        assert page.evaluate("document.documentElement.dataset.view") is None
    finally:
        page.evaluate("localStorage.clear()")
        page.close()


def test_hiding_optional_panels_keeps_their_warnings_in_view_and_changes_no_feature(browser: Browser,
                                                                                   site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: shown(page, "#health"), 5.0)
        assert not shown(page, ".hw[data-panel=health]")  # the warning is in Health itself while it is shown
        for name in ("health", "chat"):
            page.evaluate(f"document.querySelector('button[data-panel={name}]').click()")
        assert not shown(page, "#health") and not shown(page, "#chat")
        assert page.evaluate("document.querySelector('button[data-panel=health] .pstate').textContent") == ": hidden"
        assert page.evaluate("document.querySelector('button[data-panel=delivery] .pstate').textContent") == ": shown"
        assert not shown(page, ".nav a[data-nav=health]") and not shown(page, ".nav a[data-nav=chat]")
        assert shown(page, ".hw[data-panel=health]")  # the degraded reason stays in view
        assert "the runtime reports a problem: video stale" in page.evaluate(
            "document.querySelector('.hw[data-panel=health]').innerText")
        assert shown(page, ".freshbar") and shown(page, "#live") and shown(page, "#faces") and shown(page, "#features")
        page.navigate(f"{site['url']}/dashboard")  # saved: still hidden after a reload, before the first paint
        assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 10.0)
        assert page.evaluate("document.documentElement.dataset.hide") == "chat health"
        assert not shown(page, "#health")
        page.evaluate("document.querySelector('.hw[data-panel=health] button[data-show=health]').click()")
        assert shown(page, "#health") and not shown(page, "#chat")
        requests = page.evaluate("performance.getEntriesByType('resource').map(e => e.name)")
        assert not [name for name in requests if "/chat" in name or "/features" in name]  # no feature was asked for
    finally:
        page.evaluate("localStorage.clear()")
        page.close()
