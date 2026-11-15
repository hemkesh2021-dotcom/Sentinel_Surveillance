"""Session 114: the maintainer's approved preview (v5) on the dashboard, in a real browser (headless Chromium;
``browser_harness``), with synthetic data only; session 115: the approved preview v6.

Carbon & Cream's dark mode is v5's deep black (a deep page, recessed panels, raised charcoal keys, a faint grain from
this origin); every light mode and every other palette keep their earlier look. Every control answers as in session
114: hover and keyboard focus send the highlight once around its edge and a glow settles; a press goes down with a
pulse; selected is a steady thin line; the edge circles only while the control's own request is pending; a disabled
control never glows. Navigation items are the maintainer's exception: no ring or outline in any state, keyboard focus
changes nothing, selected shows the two-layer glow alone. The decorations survive in-place updates; the camera picture
is never decorated; reduced motion keeps every state and moves nothing.

v6 (session 115): the trace takes 800 ms and the glow settles over 400 ms (480 ms after the hover or focus began); a
selected control (navigation's aria-current, a choice's aria-current or aria-pressed, a switch's aria-checked) keeps
its selected look under the pointer and on keyboard focus: no trace, no added glow, its label unchanged, keyboard focus
still drawing its ring. A selection that changes under the pointer, a tap on a phone or a key takes that look at once;
a press, pending, disabled and reduced motion answer as before.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from browser_harness import Browser, Page, headless_browser, wait_until
from status_fixtures import Camera, Runtime, incident_store

from sentinel import status_page
from sentinel.features import FeatureControls
from sentinel.media.clock import FakeClock
from sentinel.scene.chat import ADVISORY
from sentinel.status_page import StatusServer, new_control_key
from sentinel.web import palettes

SETTLED_S = 1.0  # past the trace (800 ms) and the glow's settling (from 480 ms to 880 ms after the hover or focus)
FRAME = {"stream_epoch": 1, "frame_seq": 3, "ingest_utc": "2026-10-25T12:00:00.200000Z", "age_ms_when_asked": 200}
STYLE = """(selector => { const e = document.querySelector(selector); const s = getComputedStyle(e);
  const rect = e.querySelector(':scope > .edge rect'); const ring = e.querySelector(':scope > .ring');
  return {shadow: s.boxShadow, outline: `${s.outlineStyle} ${s.outlineWidth} ${s.outlineColor}`, color: s.color,
          transform: s.transform, edge: rect ? getComputedStyle(rect).animationName : 'none',
          dash: rect ? getComputedStyle(rect).strokeDasharray : null,
          ring: ring ? getComputedStyle(ring).display : null,
          ringAnimation: ring ? getComputedStyle(ring).animationName : null, busy: e.getAttribute('aria-busy'),
          transition: s.transitionDuration}; })"""


class Slow:
    """A chat stand-in whose one answer waits until the test releases it: the question stays pending till then."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def availability(self) -> dict[str, str]:
        return {"state": "available", "reason": "ask about the picture the camera is showing now"}

    def ask(self, question: Any) -> tuple[int, dict[str, Any]]:
        return 202, {"job": "0" * 15 + "1", "state": "waiting", "frame": FRAME, "advisory": ADVISORY}

    def poll(self, job_id: str) -> tuple[int, dict[str, Any]]:
        if not self.release.is_set():
            return 200, {"job": job_id, "state": "waiting", "frame": FRAME, "advisory": ADVISORY}
        return 200, {"job": job_id, "state": "answered", "answer": "Nobody is in the picture (synthetic).",
                     "frame": FRAME, "age_ms_when_answered": 900, "historical": True, "advisory": ADVISORY}


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with headless_browser() as connection:
        yield connection


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setattr(status_page, "REFRESH_S", 1)  # updates every second, so a test sees several
    incident_store(tmp_path / "sentinel.db", count=4)
    controls = FeatureControls(FakeClock(), {"face": True, "scene": True, "live_view": True})
    runtime, camera, chat = Runtime(controls=controls), Camera(), Slow()
    stop = camera.run(0.1)
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview,
                          chat=lambda: chat)
    key = new_control_key()
    server.enable_controls(lambda: controls, key)
    server.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.address[1]}", "key": key, "chat": chat}
    finally:
        stop.set()
        chat.release.set()
        runtime.answering.set()
        assert server.stop(3.0)
        camera.preview.close(2.0)


def opened(browser: Browser, url: str, *, width: int = 1280, height: int = 900, touch: bool = False,
           **options: Any) -> Page:
    page = browser.page(width=width, height=height, mobile=width < 600, **options)
    if touch:  # a phone's touch screen: a tap, and the hover the browser then keeps on what was tapped
        page.call("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
    page.navigate(url)
    assert wait_until(lambda: page.evaluate("document.readyState") == "complete", 15.0), "the page did not load"
    assert wait_until(lambda: not page.evaluate("document.documentElement.classList.contains('enter')"), 5.0)
    assert wait_until(lambda: page.evaluate("!!document.querySelector('.nav a[aria-current]')"), 5.0)
    return page


# Part 28: Appearance and Features are in the Settings dialog. A control inside one of them is reached by opening the
# dialog on its section (as a viewer does, from the sidebar); a control on the page behind is reached by closing it.
REVEAL = """((selector, close) => {
  const dialog = document.getElementById('settings-dialog');
  if (!dialog || !document.documentElement.classList.contains('sd')) return;
  const node = document.querySelector(selector);
  const part = node && node.closest('#settings, #features');
  if (part) {
    if (!dialog.open || dialog.dataset.section !== part.id) {
      document.querySelector(`#sidebar .nav a[data-nav=${part.id}]`).click();
    }
  } else if (close && dialog.open && node && !dialog.contains(node)) {
    dialog.close();
  }
})"""


def reveal(page: Page, selector: str, *, close: bool = True) -> None:
    page.evaluate(f"({REVEAL})({selector!r}, {'true' if close else 'false'})")


def style(page: Page, selector: str) -> dict[str, Any]:
    reveal(page, selector, close=False)
    return page.evaluate(f"({STYLE})({selector!r})")


def drawn(page: Page, selector: str) -> dict[str, Any]:
    """What is drawn of a control: its shadows, outline, label colour, position and edge."""
    seen = style(page, selector)
    return {key: seen[key] for key in ("shadow", "outline", "color", "transform", "edge")}


def centre(page: Page, selector: str) -> tuple[float, float]:
    reveal(page, selector)
    page.evaluate(f"document.querySelector({selector!r}).scrollIntoView({{block: 'center', behavior: 'instant'}})")
    time.sleep(0.1)
    x, y = page.evaluate(f"(() => {{ const r = document.querySelector({selector!r}).getBoundingClientRect(); "
                         "return [r.left + r.width / 2, r.top + r.height / 2]; })()")
    return x, y


def mouse(page: Page, kind: str, x: float, y: float) -> None:
    extra = {"button": "left", "clickCount": 1} if kind in ("mousePressed", "mouseReleased") else {}
    page.call("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, **extra})


def hover(page: Page, selector: str) -> tuple[float, float]:
    """The pointer over the control, aimed again if an update moved the page under it."""
    for _ in range(20):
        x, y = centre(page, selector)
        mouse(page, "mouseMoved", x, y)
        if page.evaluate(f"document.querySelector({selector!r}).matches(':hover')"):
            return x, y
    raise AssertionError(f"could not hover {selector}")


def away(page: Page) -> None:
    mouse(page, "mouseMoved", 1, 1)
    page.evaluate("document.activeElement && document.activeElement.blur()")
    time.sleep(0.6)  # past every transition back (the shadow's 260 ms, the label's 400 ms)


def tap(page: Page, selector: str) -> None:
    x, y = centre(page, selector)
    page.call("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    page.call("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


def attribute(page: Page, selector: str, name: str) -> str | None:
    return page.evaluate(f"document.querySelector({selector!r}).getAttribute({name!r})")


def keyboard_focus(page: Page, selector: str) -> None:
    """What a keyboard user gets on ``selector``: keyboard modality first (a Tab), then the focus."""
    reveal(page, selector)
    page.key("Tab")
    page.evaluate(f"document.querySelector({selector!r}).focus({{preventScroll: true}})")
    assert page.evaluate(f"document.querySelector({selector!r}).matches(':focus-visible')"), selector
    time.sleep(SETTLED_S)


def rgb(colour: str) -> str:
    return "rgb({}, {}, {})".format(*(int(colour[i:i + 2], 16) for i in (1, 3, 5))) if colour.startswith("#") \
        else colour


def layers(token: str) -> list[str]:
    """A token's shadow layers ("0 0 12px rgba(...), ...") as the browser computes them."""
    return [f"{colour} 0px 0px {blur}px 0px" for blur, colour in re.findall(r"0 0 (\d+)px (rgba\([^)]*\))", token)]


def ring_layers(shadow: str, colours: list[str]) -> list[str]:
    """Every 1 px ring or line (inside or outside the edge) drawn in one of ``colours``."""
    return [c for c in colours if re.search(re.escape(c) + r" 0px 0px 0px 1px", shadow)]


def unlock(page: Page, key: str) -> None:
    page.evaluate(f"(() => {{ document.getElementById('control-key').value = {key!r}; "
                  "document.getElementById('controls-unlock').requestSubmit(); })()")
    assert wait_until(lambda: page.evaluate("[...document.querySelectorAll('#features button.switch')]"
                                            ".some(b => !b.disabled)"), 8.0)


# ---------------------------------------------------------------- the deep black, Carbon & Cream's dark mode only


def test_carbon_and_cream_dark_takes_the_deep_black_and_every_other_look_keeps_its_own(
        browser: Browser, site: dict[str, Any]) -> None:
    for query, look in (("?theme=dark", "deep"), ("?theme=light", "light"),
                        ("?theme=dark&palette=teal-coconut", "teal")):
        page = opened(browser, f"{site['url']}/dashboard{query}")
        try:
            seen = page.evaluate("""(() => { const s = e => getComputedStyle(document.querySelector(e));
              return {body: s('body').backgroundColor, grain: s('body').backgroundImage, panel: s('#live')
              .backgroundColor, well: s('#live').boxShadow, edge: s('#live').borderTopColor, key: s(
              '.nav a[data-nav=faces]').backgroundImage, loaded: performance.getEntriesByType('resource').some(
              r => r.name.endsWith('/assets/grain.svg'))}; })()""")
            if look == "deep":
                assert (seen["body"], seen["panel"], seen["edge"]) == ("rgb(6, 6, 5)", "rgb(14, 13, 11)",
                                                                      "rgba(0, 0, 0, 0)"), seen
                assert "/assets/grain.svg" in seen["grain"] and seen["loaded"], seen  # from this origin (CSP)
                assert seen["well"].startswith("rgba(0, 0, 0, 0.85) 0px 2px 7px 0px inset"), seen  # recessed
                assert "linear-gradient(170deg, rgb(46, 44, 40), rgb(35, 34, 31) 55%, rgb(27, 26, 23))" in seen["key"]
            else:
                tokens = palettes.tokens("carbon-cream" if look == "light" else "teal-coconut",
                                         "light" if look == "light" else "dark")
                assert (seen["body"], seen["panel"]) == (rgb(tokens["canvas"]), rgb(tokens["panel"])), seen
                assert set(seen["grain"].split(", ")) == {"none"} and seen["key"] == "none" and not seen["loaded"], seen
                assert "inset" not in seen["well"].split(",")[0], seen  # the earlier raised panel
        finally:
            page.close()


# ---------------------------------------------------------------- navigation items: the maintainer's exception


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_navigation_items_show_no_ring_and_keyboard_focus_changes_nothing(
        browser: Browser, site: dict[str, Any], mode: str) -> None:
    tokens = palettes.tokens("carbon-cream", mode)
    rings = [rgb(tokens[name]) for name in ("hl", "hl-ring", "marker", "marker-line", "focus")]
    item, current = ".nav a[data-nav=health]", ".nav a[aria-current]"
    page = opened(browser, f"{site['url']}/dashboard?theme={mode}")
    try:
        away(page)
        hover(page, item)  # first used: its decorations are added (drawn only on hover)
        hover(page, current)
        away(page)
        idle, selected = drawn(page, item), drawn(page, current)
        assert idle["outline"].startswith("none") and selected["outline"].startswith("none")
        assert all(layer in selected["shadow"] for layer in layers(tokens["glow-selected"])), selected  # the glow
        assert ring_layers(selected["shadow"], rings) == [] and selected["color"] == rgb(tokens["ink"]), selected

        keyboard_focus(page, item)  # keyboard focus changes nothing, on a plain item and on the selected one
        assert drawn(page, item) == idle
        keyboard_focus(page, current)
        assert drawn(page, current) == selected
        away(page)

        x, y = hover(page, item)
        time.sleep(0.1)
        assert style(page, item)["edge"] == "trace"  # the trip around the edge, once
        time.sleep(SETTLED_S)
        hovered = style(page, item)
        assert hovered["color"] == rgb(tokens["hl"]) and f"{tokens['hl-glow']} 0px 0px 14px 0px" in hovered["shadow"]
        assert ring_layers(hovered["shadow"], rings) == [] and hovered["outline"].startswith("none"), hovered
        mouse(page, "mousePressed", x, y)
        time.sleep(0.2)
        pressed = style(page, item)
        assert ring_layers(pressed["shadow"], rings) == [] and pressed["outline"].startswith("none"), pressed
        assert pressed["ring"] == "none" and pressed["transform"] != "none", pressed  # down, with no pulse ring
        mouse(page, "mouseReleased", x, y)
        away(page)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.3)  # the clicked section stays marked for a moment; then the section in view again
        hover(page, current)
        time.sleep(0.1)
        assert style(page, current)["edge"] == "none"  # v6: no trip around a selected item
        time.sleep(SETTLED_S)
        assert drawn(page, current) == selected  # selected and hovered: the glow alone, its label in ink (v6)
    finally:
        page.close()


# ---------------------------------------------------------------- every other control, as in session 114


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_choices_switches_buttons_and_the_question_box_answer_as_approved(
        browser: Browser, site: dict[str, Any], mode: str) -> None:
    tokens = palettes.tokens("carbon-cream", mode)
    halo = [f"{tokens['hl-ring']} 0px 0px 0px 1px", f"{tokens['hl-glow']} 0px 0px 14px 0px"]
    line = f"{rgb(tokens['marker-line'])} 0px 0px 0px 1px inset"
    ring = f"solid 2px {rgb(tokens['focus'])}"
    page = opened(browser, f"{site['url']}/dashboard?theme={mode}")
    try:
        choice = "#settings [data-set=density][data-value=compact]"
        chosen = "#settings [data-set=density][aria-current]"
        away(page)
        selected = style(page, chosen)
        assert line in selected["shadow"] and not any(h in selected["shadow"] for h in halo), selected  # a line only
        hover(page, choice)
        time.sleep(0.1)
        assert style(page, choice)["edge"] == "trace"
        time.sleep(SETTLED_S)
        hovered = style(page, choice)
        assert all(h in hovered["shadow"] for h in halo) and hovered["color"] == rgb(tokens["hl"]), hovered
        x, y = hover(page, chosen)  # pressed in place: the choice already made, so nothing changes but the look
        mouse(page, "mousePressed", x, y)
        time.sleep(0.05)
        pressed = style(page, chosen)
        assert pressed["transform"] != "none" and pressed["ringAnimation"] == "pulse", pressed  # down, a short pulse
        assert halo[0] in pressed["shadow"] and pressed["color"] == rgb(tokens["ink"]), pressed  # its press ring (v6)
        mouse(page, "mouseReleased", x, y)
        away(page)
        keyboard_focus(page, choice)
        focused = style(page, choice)
        assert focused["outline"] == ring and all(h in focused["shadow"] for h in halo), focused
        away(page)

        unlock(page, site["key"])
        away(page)
        switch = "#features button.switch[data-feature=scene]"
        on = drawn(page, switch)
        assert line in on["shadow"] and not any(h in on["shadow"] for h in halo), on
        hover(page, switch)
        time.sleep(SETTLED_S)
        assert drawn(page, switch) == on  # on and hovered: as at rest (v6)
        away(page)

        keyboard_focus(page, "#chat-q")
        assert style(page, "#chat-q")["outline"] == ring
        field = style(page, "#chat-form .field")
        assert field["edge"] == "trace" or all(h in field["shadow"] for h in halo), field
        time.sleep(0.5)
        assert all(h in style(page, "#chat-form .field")["shadow"] for h in halo)
        page.evaluate("(() => { const q = document.getElementById('chat-q'); q.value = 'Is the door open?'; "
                      "q.dispatchEvent(new Event('input')); })()")
        away(page)
        hover(page, "#chat-ask")
        time.sleep(SETTLED_S)
        ask = style(page, "#chat-ask")
        label = tokens["hl"] if mode == "dark" else tokens["accent-ink"]  # a filled button keeps its label colour
        assert all(h in ask["shadow"] for h in halo) and ask["color"] == rgb(label), ask
    finally:
        page.close()


def test_a_disabled_control_never_glows_and_pending_lasts_exactly_as_long_as_its_request(
        browser: Browser, site: dict[str, Any]) -> None:
    tokens = palettes.tokens("carbon-cream", "dark")
    page = opened(browser, f"{site['url']}/dashboard?theme=dark")
    try:
        locked = "#features button.switch[data-feature=scene]"  # on, and disabled until the controls are unlocked
        assert page.evaluate(f"document.querySelector({locked!r}).disabled") and \
            attribute(page, locked, "aria-checked") == "true"
        hover(page, locked)
        time.sleep(SETTLED_S)
        seen = style(page, locked)
        assert tokens["hl-glow"] not in seen["shadow"] and tokens["hl-ring"] not in seen["shadow"], seen
        assert seen["edge"] == "none", seen
        away(page)
        assert page.evaluate("document.getElementById('chat-ask').disabled")  # nothing typed yet
        hover(page, "#chat-ask")
        time.sleep(SETTLED_S)
        disabled = style(page, "#chat-ask")
        assert tokens["hl-glow"] not in disabled["shadow"] and tokens["hl-ring"] not in disabled["shadow"], disabled
        assert disabled["edge"] == "none" and disabled["color"] == rgb(tokens["faint"]), disabled
        away(page)
        assert style(page, "#chat-ask")["busy"] is None
        page.evaluate("(() => { const q = document.getElementById('chat-q'); q.value = 'Is anyone in the room?'; "
                      "q.dispatchEvent(new Event('input')); document.getElementById('chat-ask').click(); })()")
        assert wait_until(lambda: style(page, "#chat-ask")["busy"] == "true", 3.0)
        pending = style(page, "#chat-ask")
        assert pending["edge"] == "orbit" and tokens["hl-glow"] not in pending["shadow"], pending
        time.sleep(2.2)  # polls come back "waiting": still pending
        assert style(page, "#chat-ask")["busy"] == "true"
        site["chat"].release.set()
        assert wait_until(lambda: page.evaluate("(document.querySelector('#chat-log .exchange') || {dataset: {}})"
                                                ".dataset.state") == "answered", 5.0)
        done = style(page, "#chat-ask")
        assert done["busy"] is None and done["edge"] == "none", done  # over with the request, not before or after
    finally:
        page.close()


# ---------------------------------------------------------------- v6: a selected control keeps its look (session 115)


def selections(page: Page, key: str) -> dict[str, str | None]:
    """Every kind of selection on the dashboard, by its own state attribute, each with an unselected control of the
    same kind (None where every one is selected): a choice (aria-current), a palette (aria-current), the picture's
    size (aria-pressed), a shown panel (aria-pressed), a previewed incident (aria-pressed), a switch that is on
    (aria-checked). The focus view's button (aria-pressed) is checked last, by itself."""
    unlock(page, key)
    page.evaluate("document.querySelector('.inc-list > li:nth-child(2) .peek').click()")
    assert wait_until(lambda: page.evaluate("!!document.querySelector('.inc-list > li.selected > .peek')"), 5.0)
    return {"#settings [data-set=density][aria-current]": "#settings [data-set=density]:not([aria-current])",
            "#settings .swatch[aria-current]": "#settings .swatch:not([aria-current])",
            "#live .view-size [aria-pressed=true]": "#live .view-size [aria-pressed=false]",
            "#settings button.seg[data-panel=chat][aria-pressed=true]": None,
            ".inc-list > li.selected > .peek[aria-pressed=true]": ".inc-list > li:not(.selected) > .peek",
            "#features button.switch[data-feature=scene][aria-checked=true]": None}


@pytest.mark.parametrize("mode", ["dark", "light"])
@pytest.mark.parametrize("width", [1280, 390], ids=["desktop", "phone"])
def test_a_selected_control_keeps_its_look_under_the_pointer_and_on_keyboard_focus(
        browser: Browser, site: dict[str, Any], mode: str, width: int) -> None:
    tokens = palettes.tokens("carbon-cream", mode)
    halo = [f"{tokens['hl-ring']} 0px 0px 0px 1px", f"{tokens['hl-glow']} 0px 0px 14px 0px"]
    ring = f"solid 2px {rgb(tokens['focus'])}"
    page = opened(browser, f"{site['url']}/dashboard?theme={mode}", width=width, height=900 if width > 600 else 844)
    try:
        if width < 600:  # navigation is in the drawer
            page.evaluate("document.querySelector('.menu').click()")
            assert wait_until(lambda: page.evaluate("document.documentElement.classList.contains('drawer-open')"), 3.0)
            time.sleep(0.4)
        current, item = ".nav a[aria-current]", ".nav a[data-nav=health]:not([aria-current])"
        hover(page, current)  # its decorations are added on first use
        mouse(page, "mouseMoved", width - 2, 2)
        time.sleep(0.6)
        rest = drawn(page, current)
        hover(page, current)
        time.sleep(0.1)
        assert style(page, current)["edge"] == "none", current  # no trip around a selected item
        time.sleep(SETTLED_S)
        assert drawn(page, current) == rest, current  # the glow alone, its label in ink
        hover(page, item)
        time.sleep(SETTLED_S)
        assert style(page, item)["color"] == rgb(tokens["hl"])  # an item not selected still answers the pointer
        if width < 600:
            page.key("Escape")
            assert wait_until(lambda: not page.evaluate("document.documentElement.classList.contains('drawer-open')"),
                              3.0)

        def keeps_its_look(chosen: str) -> None:
            hover(page, chosen)
            away(page)
            rest = drawn(page, chosen)
            hover(page, chosen)
            time.sleep(0.1)
            assert page.evaluate(f"!!document.querySelector({chosen!r}).querySelector(':scope > .edge rect')")
            assert style(page, chosen)["edge"] == "none", chosen  # no trip around the edge
            time.sleep(SETTLED_S)
            assert drawn(page, chosen) == rest, chosen  # no added glow, its label unchanged
            away(page)
            keyboard_focus(page, chosen)
            focused = drawn(page, chosen)
            assert focused["outline"] == ring, (chosen, focused)  # keyboard focus still draws its ring ...
            assert {**focused, "outline": rest["outline"]} == rest, (chosen, focused)  # ... and nothing else
            away(page)

        for chosen, other in selections(page, site["key"]).items():
            keeps_its_look(chosen)
            if other:  # beside it, a control of the same kind that is not selected answers as before
                hover(page, other)
                time.sleep(0.1)
                assert style(page, other)["edge"] == "trace", other
                time.sleep(SETTLED_S)
                hovered = style(page, other)
                assert all(h in hovered["shadow"] for h in halo) and hovered["color"] == rgb(tokens["hl"]), \
                    (other, hovered)
                away(page)
        page.evaluate("document.getElementById('focus-toggle').click()")  # last: the focus view hides the others
        assert attribute(page, "#focus-toggle", "aria-pressed") == "true"
        keeps_its_look("#focus-toggle")
    finally:
        page.close()


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_a_selection_made_under_the_pointer_or_from_the_keyboard_shows_the_selected_look_at_once(
        browser: Browser, site: dict[str, Any], mode: str) -> None:
    tokens = palettes.tokens("carbon-cream", mode)
    halo = [f"{tokens['hl-ring']} 0px 0px 0px 1px", f"{tokens['hl-glow']} 0px 0px 14px 0px"]
    ring = f"solid 2px {rgb(tokens['focus'])}"
    page = opened(browser, f"{site['url']}/dashboard?theme={mode}")
    try:
        fit, actual = "#live .view-size [data-view=fit]", "#live .view-size [data-view=actual]"
        hover(page, fit)
        away(page)
        selected = drawn(page, fit)  # Fit is selected when the page opens
        x, y = hover(page, actual)
        time.sleep(SETTLED_S)
        assert all(h in style(page, actual)["shadow"] for h in halo)
        mouse(page, "mousePressed", x, y)
        time.sleep(0.25)
        pressed = style(page, actual)
        assert halo[0] in pressed["shadow"] and halo[1] not in pressed["shadow"], pressed  # the press, not the hover
        mouse(page, "mouseReleased", x, y)
        assert attribute(page, actual, "aria-pressed") == "true"
        assert page.evaluate(f"document.querySelector({actual!r}).matches(':hover')")  # the pointer has not moved
        time.sleep(0.1)
        assert style(page, actual)["edge"] == "none"  # the trip stops at once
        time.sleep(0.6)
        assert drawn(page, actual) == selected  # the selected look under the pointer

        away(page)  # from the keyboard: Space on the focused choice
        keyboard_focus(page, fit)
        page.key(" ")
        assert wait_until(lambda: attribute(page, fit, "aria-pressed") == "true", 2.0)
        time.sleep(0.6)
        assert drawn(page, fit) == {**selected, "outline": ring}  # the selected look and the focus ring
        away(page)

        # a switch: pending while its request is in flight, then off (not selected) and on again, under the pointer
        unlock(page, site["key"])
        switch = "#features button.switch[data-feature=scene]"
        hover(page, switch)
        away(page)
        on = drawn(page, switch)
        page.call("Fetch.enable", {"patterns": [{"urlPattern": "*/features", "requestStage": "Request"}]})
        try:
            x, y = hover(page, switch)
            mouse(page, "mousePressed", x, y)
            mouse(page, "mouseReleased", x, y)
            held = page.paused(5.0)
            assert held is not None and held["request"]["method"] == "POST", held
            pending = style(page, switch)
            assert pending["busy"] == "true" and pending["edge"] == "orbit", pending  # selected and pending
            assert not any(h in pending["shadow"] for h in halo), pending
            page.call("Fetch.continueRequest", {"requestId": held["requestId"]})
        finally:
            page.call("Fetch.disable")
        assert wait_until(lambda: attribute(page, switch, "aria-checked") == "false"
                          and not page.evaluate(f"document.querySelector({switch!r}).disabled"), 8.0)
        time.sleep(SETTLED_S)
        off = style(page, switch)
        assert off["edge"] == "trace" and all(h in off["shadow"] for h in halo), off  # off and hovered: the hover
        x, y = hover(page, switch)
        mouse(page, "mousePressed", x, y)
        mouse(page, "mouseReleased", x, y)
        assert wait_until(lambda: attribute(page, switch, "aria-checked") == "true"
                          and not page.evaluate(f"document.querySelector({switch!r}).disabled"), 8.0)
        time.sleep(0.6)
        assert drawn(page, switch) == on  # on again: the selected look, the pointer still over it
    finally:
        page.close()


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_on_a_phone_a_tapped_choice_shows_its_selected_look_though_the_browser_keeps_it_hovered(
        browser: Browser, site: dict[str, Any], mode: str) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme={mode}", width=390, height=844, touch=True)
    try:
        fit, actual = "#live .view-size [data-view=fit]", "#live .view-size [data-view=actual]"
        tap(page, fit)  # already selected: the tap changes nothing but dresses it
        time.sleep(SETTLED_S)
        page.evaluate("document.activeElement && document.activeElement.blur()")
        tap(page, "#live .phead h2")  # the hover moves to the heading
        time.sleep(0.6)
        selected = drawn(page, fit)
        tap(page, actual)
        assert attribute(page, actual, "aria-pressed") == "true"
        assert page.evaluate(f"document.querySelector({actual!r}).matches(':hover')")  # kept by the browser
        time.sleep(SETTLED_S)
        page.evaluate("document.activeElement && document.activeElement.blur()")
        time.sleep(0.6)
        assert drawn(page, actual) == selected  # no glow, no trace, its label unchanged
    finally:
        page.close()


def test_the_trace_and_the_glow_take_v6s_timings_and_a_press_and_the_return_keep_theirs(
        browser: Browser, site: dict[str, Any]) -> None:
    timing = """(selector => { const e = document.querySelector(selector); const s = getComputedStyle(e);
      const rect = e.querySelector(':scope > .edge rect'); const d = s.transitionDuration.split(', ');
      const w = s.transitionDelay.split(', ');
      return {trace: rect ? getComputedStyle(rect).animationDuration : null, transitions: Object.fromEntries(
        s.transitionProperty.split(', ').map((p, i) => [p, [d[i], w[i]]]))}; })"""
    page = opened(browser, f"{site['url']}/dashboard?theme=dark")
    try:
        choice = "#settings [data-set=density]:not([aria-current])"
        hover(page, choice)
        away(page)
        rest = page.evaluate(f"({timing})({choice!r})")["transitions"]
        assert rest["box-shadow"] == ["0.26s", "0s"] and rest["color"] == ["0.4s", "0s"], rest  # the return
        x, y = hover(page, choice)
        hovered = page.evaluate(f"({timing})({choice!r})")
        assert hovered["trace"] == "0.8s", hovered  # one trip around the edge in 800 ms
        assert hovered["transitions"]["box-shadow"] == ["0.4s", "0.48s"], hovered  # the glow settles as it ends
        assert hovered["transitions"]["color"] == ["0.4s", "0s"], hovered
        mouse(page, "mousePressed", x, y)
        pressed = page.evaluate(f"({timing})({choice!r})")["transitions"]
        assert pressed["box-shadow"] == ["0.14s", "0s"] and pressed["transform"] == ["0.14s", "0s"], pressed
        mouse(page, "mouseMoved", 1, 1)
        mouse(page, "mouseReleased", 1, 1)  # released elsewhere: no choice made
        assert attribute(page, choice, "aria-current") is None
    finally:
        page.close()


# ---------------------------------------------------------------- updates, the camera picture, reduced motion


def test_a_controls_decorations_survive_in_place_updates(browser: Browser, site: dict[str, Any]) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme=dark")
    try:
        peek = ".inc-list > li:first-child .peek"  # inside the incidents region, which every update re-renders
        hover(page, peek)
        time.sleep(0.5)
        page.evaluate(f"document.querySelector({peek!r}).querySelector(':scope > .edge').__testMark = 'kept'")
        before = page.evaluate("document.documentElement.dataset.sequence")
        assert wait_until(lambda: int(page.evaluate("document.documentElement.dataset.sequence")) >= int(before) + 2,
                          8.0)
        seen = page.evaluate(f"""(() => {{ const b = document.querySelector({peek!r});
          return {{edges: b.querySelectorAll(':scope > .edge').length, rings: b.querySelectorAll(':scope > .ring')
          .length, mark: b.querySelector(':scope > .edge').__testMark || null, label: b.querySelector('span')
          .textContent, last: b.lastElementChild.className}}; }})()""")
        assert seen == {"edges": 1, "rings": 1, "mark": "kept", "label": "Preview", "last": "ring"}, seen
    finally:
        page.close()


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_the_camera_picture_is_never_decorated(browser: Browser, site: dict[str, Any], mode: str) -> None:
    page = opened(browser, f"{site['url']}/dashboard?theme={mode}")
    try:
        picture = ".live .frame canvas, .live .frame img"
        assert wait_until(lambda: page.evaluate(f"!!document.querySelector({picture!r})"), 8.0)
        away(page)
        frame = style(page, ".live .frame")
        hover(page, picture)
        time.sleep(SETTLED_S)
        assert style(page, ".live .frame") == frame  # hovering the picture changes nothing around it
        seen = page.evaluate(f"""(() => {{ const p = getComputedStyle(document.querySelector({picture!r}));
          return {{shadow: p.boxShadow, outline: p.outlineStyle, decorations: document.querySelectorAll(
          '#frame .edge, #frame .ring').length}}; }})()""")
        assert seen == {"shadow": "none", "outline": "none", "decorations": 0}, seen
    finally:
        page.close()


def test_reduced_motion_keeps_every_state_and_moves_nothing(browser: Browser, site: dict[str, Any]) -> None:
    tokens = palettes.tokens("carbon-cream", "dark")
    page = opened(browser, f"{site['url']}/dashboard?theme=dark", media={"prefers-reduced-motion": "reduce"})
    try:
        choice = "#settings [data-set=density][data-value=compact]"
        hover(page, choice)
        time.sleep(0.2)
        hovered = style(page, choice)
        assert f"{tokens['hl-glow']} 0px 0px 14px 0px" in hovered["shadow"] and hovered["edge"] == "none", hovered
        assert set(hovered["transition"].split(", ")) == {"0s"}, hovered
        chosen = "#settings [data-set=density][aria-current]"
        away(page)
        rest = drawn(page, chosen)
        x, y = hover(page, chosen)
        time.sleep(0.2)
        assert drawn(page, chosen) == rest  # selected: as at rest under the pointer (v6)
        mouse(page, "mousePressed", x, y)
        time.sleep(0.05)
        pressed = style(page, chosen)
        assert pressed["transform"] == "none" and pressed["ringAnimation"] == "none", pressed  # inset, nothing moves
        assert "rgba(0, 0, 0, 0.9) 0px 2px 7px 0px inset" in pressed["shadow"], pressed  # v5's pressed shadow
        mouse(page, "mouseReleased", x, y)
        page.evaluate("(() => { const q = document.getElementById('chat-q'); q.value = 'Is anyone in the room?'; "
                      "q.dispatchEvent(new Event('input')); document.getElementById('chat-ask').click(); })()")
        assert wait_until(lambda: style(page, "#chat-ask")["busy"] == "true", 3.0)
        pending = style(page, "#chat-ask")
        assert pending["edge"] == "none" and pending["dash"] == "2px, 3px", pending  # a still, dotted edge
    finally:
        site["chat"].release.set()
        page.close()
