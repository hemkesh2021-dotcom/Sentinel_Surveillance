"""D99 in a real browser (headless Chromium; ``browser_harness``), with a stand-in runtime and synthetic data only.

The Features panel: switches stay disabled until the operator's key unlocks them in this tab; a wrong key is refused
and says so; a switch asks the runtime for a change (one request at a time) and shows the requested state while the
actual one follows from the runtime; a page that stops being current disables every switch by itself, and they come
back with a current update; Lock forgets the key; a runtime that restarts (a new server instance and key) makes the
page forget the old key. The key is never stored in the browser. The switch works from the keyboard.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser, wait_until
from status_fixtures import Runtime, incident_store

from sentinel import status_page
from sentinel.features import FeatureControls
from sentinel.media.clock import FakeClock
from sentinel.status_page import PAGE_CURRENT_S, StatusServer, new_control_key


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)
    incident_store(tmp_path / "sentinel.db", count=2)
    controls = FeatureControls(FakeClock(), {"face": True, "scene": True, "live_view": True})
    runtime = Runtime(controls=controls)
    key = new_control_key()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db")
    server.enable_controls(lambda: controls, key)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "runtime": runtime, "controls": controls, "key": key,
               "server": server}
    finally:
        runtime.answering.set()
        assert server.stop(3.0)


def opened(browser: Browser, url: str, **options: Any) -> Page:
    page = browser.page(**options)
    page.navigate(url)
    assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0), "the page did not load"
    return page


def switches(page: Page) -> dict[str, dict[str, Any]]:
    return page.evaluate("Object.fromEntries([...document.querySelectorAll('#features button.switch')].map(b => "
                         "[b.dataset.feature, {disabled: b.disabled, checked: b.getAttribute('aria-checked')}]))")


def note(page: Page) -> str:
    return page.evaluate("document.getElementById('controls-note').textContent")


def unlock(page: Page, key: str) -> None:
    page.evaluate(f"document.getElementById('control-key').value = {key!r}")
    page.evaluate("document.getElementById('controls-unlock').requestSubmit()")


def test_switches_stay_disabled_until_the_operators_key_unlocks_them(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: switches(page).get("face") is not None, 5.0)
        assert all(s["disabled"] for s in switches(page).values())
        assert page.evaluate("!document.getElementById('controls-unlock').hidden")
        unlock(page, "AAAA-BBBB-CCCC-DDDD-EEEE-FFFF")
        assert wait_until(lambda: "Not unlocked: the control key is not this run's" in note(page), 5.0)
        assert all(s["disabled"] for s in switches(page).values())
        unlock(page, site["key"].lower())  # as an operator may type it
        assert wait_until(lambda: "Unlocked" in note(page), 5.0)
        assert wait_until(lambda: not switches(page)["face"]["disabled"], 3.0)
        state = switches(page)
        assert state["chat"]["disabled"] and state["snapshots"]["disabled"]  # not started: never switchable
        assert page.evaluate("document.getElementById('control-key').value") == ""  # not left in the field
        stored = page.evaluate("JSON.stringify(Object.assign({}, localStorage)) + JSON.stringify(Object.assign({}, "
                               "sessionStorage)) + document.cookie")
        assert site["key"] not in stored and site["key"].lower() not in stored  # never stored in the browser
    finally:
        page.close()


def test_a_switch_asks_for_a_change_and_the_actual_state_follows(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: switches(page).get("face") is not None, 5.0)
        unlock(page, site["key"])
        assert wait_until(lambda: not switches(page)["face"]["disabled"], 5.0)
        page.evaluate("document.querySelector('#features button.switch[data-feature=face]').focus()")
        page.key("Enter")  # the keyboard works: it is a real button
        assert wait_until(lambda: "Requested: face off (change 1)" in note(page), 5.0)
        assert wait_until(lambda: switches(page)["face"]["checked"] == "false", 5.0)
        panel = page.evaluate("document.querySelector('#features li[data-key=\"f-face\"]').innerText")
        assert "Requested: off · now: off" in panel and "Turned off: earlier recognition is cleared" in panel
        assert site["controls"].view()["log"][-1]["result"] == "applied"
        page.evaluate("document.querySelector('#features button.switch[data-feature=face]').click()")
        assert wait_until(lambda: switches(page)["face"]["checked"] == "true", 5.0)
    finally:
        page.close()


def test_a_page_that_is_not_current_disables_every_switch_until_a_current_update(browser: Browser,
                                                                                 site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: switches(page).get("face") is not None, 5.0)
        unlock(page, site["key"])
        assert wait_until(lambda: not switches(page)["face"]["disabled"], 5.0)
        site["runtime"].answering.clear()  # the tunnel pauses: no update arrives
        assert wait_until(lambda: switches(page)["face"]["disabled"], PAGE_CURRENT_S + 3)
        assert "This page is not current" in note(page)
        site["runtime"].answering.set()
        assert wait_until(lambda: not switches(page)["face"]["disabled"], 20.0)  # without a reload
        assert site["controls"].view()["log"] == []  # nothing was changed meanwhile
    finally:
        site["runtime"].answering.set()
        page.close()


def test_lock_forgets_the_key_and_a_restarted_runtime_needs_its_new_key(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard", width=1440, height=900)
    try:
        assert wait_until(lambda: switches(page).get("face") is not None, 5.0)
        unlock(page, site["key"])
        assert wait_until(lambda: not switches(page)["face"]["disabled"], 5.0)
        page.evaluate("document.getElementById('controls-lock').click()")
        assert all(s["disabled"] for s in switches(page).values()) and "Locked" in note(page)
        unlock(page, site["key"])
        assert wait_until(lambda: not switches(page)["face"]["disabled"], 5.0)
        server: StatusServer = site["server"]
        server._instance = secrets.token_hex(8)  # as a restarted runtime's server: a new instance and a new key
        server.enable_controls(lambda: site["controls"], new_control_key())
        assert wait_until(lambda: "The runtime restarted" in note(page), 6.0)
        assert all(s["disabled"] for s in switches(page).values())
        assert page.evaluate("!document.getElementById('controls-unlock').hidden")
    finally:
        page.close()
