"""D72: the status pages' styling and structure, without a browser.

Contrast is computed from the stylesheet's own tokens (WCAG 2.x relative luminance): text at least 4.5:1, focus
rings and control borders at least 3:1, in both themes. States carry a text label and an icon, never colour alone.
Images are never filtered or recoloured. Reduced motion turns transitions off and keeps the D71 watchdog.
"""

from __future__ import annotations

import http.client
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from status_fixtures import incident_store

from sentinel import status_page
from sentinel.status_page import (
    STYLE,
    STYLE_SOURCE,
    StatusServer,
    build_status,
    read_incident,
    read_store,
    render_html,
    render_incident_html,
    theme_from_query,
)
from sentinel.web import ICONS, csp_hash, stylesheet

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
REFERENCE = {"--coconut-milk": "#f0ede5", "--white-smoke": "#eddcc9", "--stretch-limo": "#2b2c30",
             "--carbon-fibre": "#2e302f", "--green-sheen": "#d9ce52"}
TEXT_PAIRS = [("ink", s) for s in ("bg", "surface", "surface-2", "surface-3")] + \
    [("muted", s) for s in ("bg", "surface", "surface-2", "surface-3")] + \
    [("link", s) for s in ("bg", "surface", "surface-2", "surface-3")] + \
    [("accent-ink", "accent"), ("ok-ink", "ok-bg"), ("warn-ink", "warn-bg"), ("crit-ink", "crit-bg"),
     ("off-ink", "off-bg"), ("stale-ink", "stale-bg")]
UI_PAIRS = [("focus", s) for s in ("bg", "surface", "surface-2")] + \
    [("border", s) for s in ("bg", "surface", "surface-2", "surface-3")]


def tokens(block: str, base: dict[str, str] | None = None) -> dict[str, str]:
    values = dict(base or {})
    for name, value in re.findall(r"--([a-z0-9-]+):\s*([^;]+);", block):
        values[name] = value.strip()
    resolved = {}
    for name, value in values.items():
        while value.startswith("var(--"):
            value = values[value[6:-1]]
        resolved[name] = value
    return resolved


def luminance(colour: str) -> float:
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def themes() -> dict[str, dict[str, str]]:
    css = stylesheet(5)
    light = tokens(css[css.index(":root {"):css.index("}", css.index(":root {"))])
    dark_block = css[css.index(":root[data-theme=dark] {"):]
    dark = tokens(dark_block[:dark_block.index("}")], light)
    return {"light": light, "dark": dark}


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_and_controls_meet_their_contrast_in_both_themes(theme: str) -> None:
    values = themes()[theme]
    low = {f"{fg} on {bg}": round(contrast(values[fg], values[bg]), 2) for fg, bg in TEXT_PAIRS
           if contrast(values[fg], values[bg]) < 4.5}
    assert not low, low
    weak = {f"{fg} on {bg}": round(contrast(values[fg], values[bg]), 2) for fg, bg in UI_PAIRS
            if contrast(values[fg], values[bg]) < 3.0}
    assert not weak, weak


def test_the_reference_colours_are_named_tokens_and_both_theme_blocks_agree() -> None:
    css = stylesheet(5)
    for name, value in REFERENCE.items():
        assert f"{name}: {value};" in css, name
    light, dark = themes()["light"], themes()["dark"]
    assert (light["bg"], light["ink"], light["surface-2"], light["accent"]) == ("#f0ede5", "#2b2c30", "#eddcc9", "#d9ce52")
    assert (dark["bg"], dark["surface"], dark["ink"]) == ("#2b2c30", "#2e302f", "#f0ede5")
    # The automatic dark block (prefers-color-scheme) and the explicit one (?theme=dark) set the same values.
    auto = css[css.index(":root:not([data-theme=light]) {"):]
    assert tokens(auto[:auto.index("}")], light) == dark


def test_images_are_never_recoloured_and_reduced_motion_keeps_the_watchdog() -> None:
    assert not re.search(r"\b(filter|backdrop-filter|mix-blend-mode|mask-image)\s*:", STYLE)
    reduced = STYLE[STYLE.index("@media (prefers-reduced-motion: reduce)"):]
    assert "transition-duration: 0s" in reduced and "animation" not in reduced
    assert ".cur { visibility: hidden; animation: cur 5s linear var(--fresh-delay, 0s); }" in STYLE  # untouched
    assert STYLE == stylesheet(status_page.PAGE_CURRENT_S) and "CURRENT" not in STYLE.replace("NOT CURRENT", "")


def test_every_state_has_a_label_and_an_icon_and_kinds_are_distinct() -> None:
    snapshot = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": NOW.isoformat()},
                "degraded": ["detector unavailable (libcuda_not_l4t)"],
                "components": {"capture": {"state": "waiting", "problem": "open_failed"},
                               "detector": {"state": "unavailable", "problem": "libcuda_not_l4t"},
                               "scene": {"state": "disabled"}, "face": {"state": "available"},
                               "incidents": {"state": "ok"}}}
    page = render_html(build_status(snapshot, {"state": "available"}, NOW))
    badges = re.findall(r'<span class="badge k-(\w+)[^"]*"><svg[^>]*><use href="#i-(\w+)"/></svg>([^<]+)</span>', page)
    assert len(badges) == page.count('<span class="badge ')  # every badge: icon, then a text label
    kinds = {kind: icon for kind, icon, _ in badges}
    assert {"ok", "warn", "crit", "off"} <= set(kinds) and len({kinds[k] for k in ("ok", "warn", "crit", "off")}) == 4
    assert all(text.strip() for _, _, text in badges)
    assert set(re.findall(r'<use href="#i-(\w+)"/>', page)) <= set(ICONS)
    assert 'aria-hidden="true" focusable="false"' in page  # decorative: the label carries the meaning


def test_the_csp_allows_exactly_the_inline_stylesheet_and_no_other_inline_code(tmp_path: Path) -> None:
    incident_store(tmp_path / "sentinel.db", count=1)
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        for path in ("/", "/assets/icon.svg", "/status.json"):
            status, headers, body = get(server, path)
            assert status == 200, path
            policy = headers["Content-Security-Policy"]
            assert f"style-src {STYLE_SOURCE};" in policy and "unsafe" not in policy
            assert "script-src 'self';" in policy and "connect-src 'self';" in policy  # D73: one same-origin script
            assert "default-src 'none'" in policy and "frame-ancestors 'none'" in policy
        page = get(server, "/")[2].decode()
        (style,) = re.findall(r"<style>(.*?)</style>", page, flags=re.S)
        assert csp_hash(style) == STYLE_SOURCE and " style=" not in page
        assert re.findall(r"<script[^>]*>(.*?)</script>", page, flags=re.S) == [""]  # no inline script
        assert '<script src="/assets/dashboard.js" defer></script>' in page and "http-equiv=refresh" not in page
    finally:
        assert server.stop(3.0)


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


@pytest.mark.parametrize(("query", "theme"), [
    ("theme=light", "light"), ("theme=dark", "dark"), ("", None), ("theme=", None), ("theme=Dark", None),
    ("theme=dark%22%3E%3Cscript%3E", None), ("x=1&theme=dark", "dark"), ("theme=sepia", None),
])
def test_the_theme_choice_is_one_of_two_exact_values_and_never_echoed(query: str, theme: str | None) -> None:
    assert theme_from_query(query) == theme


def test_the_chosen_theme_is_applied_and_kept_in_every_link(tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=2)
    server = StatusServer(0, lambda: {"runtime": {"state": "running", "camera_id": "cam-1"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        page = get(server, "/?theme=dark")[2].decode()
        assert '<html lang=en data-page=status data-theme=dark data-source="/status.json?page=1&amp;theme=dark"' in page
        assert f'href="/incidents/{ids[0]}?theme=dark"' in page and '<a href="/?theme=dark" aria-current="true">' in page
        assert '<a href="/">' in page and '<a href="/?theme=light">' in page  # the other choices stay reachable
        page = get(server, "/?theme=%22%3E%3Cscript%3Ealert(1)%3C/script%3E")[2].decode()
        assert '<html lang=en data-page=status data-source="/status.json?page=1"' in page and "alert(1)" not in page
        detail = get(server, f"/incidents/{ids[0]}?theme=light")[2].decode()
        assert "data-theme=light" in detail and '<a class=back href="/?theme=light">' in detail
        assert f'href="/incidents/{ids[0]}?theme=light" aria-current="true"' in detail
    finally:
        assert server.stop(3.0)


def test_the_tab_icon_is_served_read_only_on_the_loopback_host(tmp_path: Path) -> None:
    server = StatusServer(0, lambda: {"runtime": {"state": "running"}}, tmp_path / "sentinel.db")
    server.start()
    try:
        status, headers, body = get(server, "/assets/icon.svg")
        assert (status, headers["Content-Type"], headers["X-Content-Type-Options"]) == (200, "image/svg+xml", "nosniff")
        assert body.startswith(b"<svg") and b"<script" not in body and headers["Cache-Control"] == "no-store"
        assert get(server, "/assets/icon.svg", method="POST")[0] == 405
        assert get(server, "/assets/icon.svg", host="evil.example")[0] == 421
        assert get(server, "/assets/other.svg")[0] == 404
    finally:
        assert server.stop(3.0)


def test_the_incident_page_exposes_snapshot_download_reason_times_lifecycle_acknowledgment_and_delivery(
        tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=2)
    page = render_incident_html({"generated_utc": NOW.isoformat(), **read_incident(tmp_path / "sentinel.db", ids[0])})
    sections = re.findall(r'<section (?:id=\w+ )?class="panel p-(\w+)" aria-label="?([^">]+)"?>', page)
    assert [s for s, _ in sections] == ["snap", "summary", "ack", "delivery", "lifecycle", "scene", "tech"]
    assert f'<img src="/incidents/{ids[0]}/snapshot.jpg"' in page
    assert f'<a class=button href="/incidents/{ids[0]}/snapshot/download" download>' in page
    assert '<p class=reason>track 3 stayed in the zone for 1.0 s (at least 1.0 s)</p>' in page  # why it opened
    for label in ("first seen (UTC)", "latest (UTC)", "recorded (UTC)", "updated (UTC)"):
        assert f"<th>{label}</th><td><time datetime=" in page, label
    assert "<ol class=timeline>" in page and "open → acknowledged" in page
    assert "acknowledged" in page and "by operator:maintainer: seen on the dashboard" in page
    assert "delivered" in page and "provider confirmed: yes" in page
    # Technical provenance is kept, behind expandable sections.
    assert page.count("<details class=tech") == 2 and "Snapshot provenance" in page and "every stored row" in page
    failed = render_incident_html({"generated_utc": NOW.isoformat(), **read_incident(tmp_path / "sentinel.db", ids[1])})
    assert "failed" in failed and "provider confirmed: no" in failed and "Not acknowledged" in failed


def test_the_main_page_puts_incidents_beside_the_live_view_and_keeps_technical_detail_behind_expanders(
        tmp_path: Path) -> None:
    ids = incident_store(tmp_path / "sentinel.db", count=3)
    page = render_html(build_status({"runtime": {"state": "running", "camera_id": "cam-1"}},
                                    read_store(tmp_path / "sentinel.db"), NOW))
    order = re.findall(r'<(?:section|aside) id=([\w-]+) class="panel', page)
    assert order == ["live", "incidents", "health", "delivery"]  # the mobile order; desktop puts incidents beside
    assert '<a class=skip href="#live">' in page
    assert all(f'href="/incidents/{i}"' in page for i in ids)
    assert page.count("<details class=tech") == 2  # the runtime's counters, the alert rows (the live view adds one)
    assert re.search(r"\.board \{[^}]*grid-template-areas: \"live\" \"incidents\" \"health\" \"delivery\"", STYLE)
    assert '"live incidents" "health incidents" "delivery incidents"' in STYLE
