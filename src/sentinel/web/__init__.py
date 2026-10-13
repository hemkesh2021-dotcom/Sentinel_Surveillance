"""Static assets of the D-2 status pages (D72, D96): the stylesheet, the scripts, the icon sprite and the tab icon.

Self-hosted (guide ch. 15: the interface must work on a disconnected LAN), with no font, image or script from
elsewhere. The stylesheet (every palette's tokens from ``palettes``, then ``dashboard.css``) is inlined in every page so
that the D71 watchdog rules always arrive with the markup they govern; pages allow it through the CSP by its SHA-256
and allow no other inline style. Two same-origin scripts: ``appearance.js`` in the head applies the viewer's saved
appearance before the first paint, and ``dashboard.js`` (deferred) does everything else. Icons are simple stroked SVG
symbols drawn for these pages on one 24-unit grid, referenced with ``<use>``, decorative only (every state and every
control also has a text label). Portable: stdlib only.
"""

from __future__ import annotations

import base64
import hashlib
from importlib import resources

PAGE_CURRENT_TOKEN = "CURRENT"  # replaced by the watchdog window in seconds


def _text(name: str) -> str:
    return resources.files(__name__).joinpath(name).read_text(encoding="utf-8")


PALETTE_TOKEN = "/* PALETTES */"  # replaced by every palette's tokens (D96)


def stylesheet(current_s: int) -> str:
    """The page stylesheet: every palette's tokens and swatches, then the rules, with the D71 window filled in."""
    if not isinstance(current_s, int) or isinstance(current_s, bool) or not 1 <= current_s <= 60:
        raise ValueError("the watchdog window must be a whole number of seconds from 1 to 60")
    from .palettes import palette_css, swatch_css

    rules = _text("dashboard.css").replace(PAGE_CURRENT_TOKEN + "s", f"{current_s}s")
    if rules.count(PALETTE_TOKEN) != 1:
        raise ValueError("the stylesheet must hold one palette marker")
    return rules.replace(PALETTE_TOKEN, palette_css() + swatch_css())


def script() -> bytes:
    """The pages' main script: in-place updates (D73), the live player (D75), navigation and appearance (D96)."""
    return _text("dashboard.js").encode("utf-8")


def head_script() -> bytes:
    """The pages' small head script (D96): the viewer's saved appearance, applied before the first paint."""
    return _text("appearance.js").encode("utf-8")


def csp_hash(text: str) -> str:
    """The CSP source expression that allows exactly this inline style or script."""
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


# 24×24, stroked with currentColor; ``solid`` parts are filled.
ICONS = {
    # Session 68 (D98): the Sentinel mark, an S drawn from two tangent elliptical arcs inside a faint lens ring (an
    # aperture motif); restrained, and legible from 16 px on light and dark tiles (rendered and inspected at 16-96 px).
    "logo": '<path d="M16.2 7.3A4.4 3.7 0 1 0 12 12a4.4 3.7 0 1 1-4.2 4.7"/>'
            '<circle cx="12" cy="12" r="10.1" stroke-width="1.15" opacity=".5"/>',
    "video": '<rect x="2.5" y="6" width="13" height="12" rx="2.5"/><path d="M15.5 10.2 21.5 7v10l-6-3.2z"/>',
    "bell": '<path d="M6 16.5V11a6 6 0 0 1 12 0v5.5l1.5 2h-15z"/><path d="M10 21a2.2 2.2 0 0 0 4 0"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m7.8 12.4 2.8 2.8 5.6-5.8"/>',
    "warn": '<path d="M12 3.6 21.6 20H2.4z"/><path d="M12 9.6v4.6"/><circle class="solid" cx="12" cy="17.2" r="1.1"/>',
    "crit": '<path d="M8.3 3h7.4L21 8.3v7.4L15.7 21H8.3L3 15.7V8.3z"/><path d="m9.2 9.2 5.6 5.6m0-5.6-5.6 5.6"/>',
    "off": '<circle cx="12" cy="12" r="9"/><path d="M5.7 5.7 18.3 18.3"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7.2V12l3.4 2.1"/>',
    "download": '<path d="M12 3.8v11.4"/><path d="m7.2 10.6 4.8 4.8 4.8-4.8"/><path d="M4.5 19.8h15"/>',
    "chevron": '<path d="m9 5.5 6.5 6.5L9 18.5"/>',
    "image": '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><circle cx="8.6" cy="9.8" r="1.7"/>'
             '<path d="m3.8 17.8 5.4-5.2 3.7 3.4 2.6-2.4 4.7 4.4"/>',
    "back": '<path d="M15 5.5 8.5 12l6.5 6.5"/>',
    "sun": '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.8v2.2M12 19v2.2M2.8 12h2.2M19 12h2.2M5.5 5.5l1.6 1.6'
           'M16.9 16.9l1.6 1.6M5.5 18.5l1.6-1.6M16.9 7.1l1.6-1.6"/>',
    "moon": '<path d="M20.2 14.6A8.4 8.4 0 0 1 9.4 3.8a8.4 8.4 0 1 0 10.8 10.8z"/>',
    "auto": '<circle cx="12" cy="12" r="8.6"/><path class="solid" d="M12 3.4a8.6 8.6 0 0 1 0 17.2z"/>',
    "shield": '<path d="M12 2.8 19.5 6v5.4c0 4.6-3.1 8.4-7.5 9.8-4.4-1.4-7.5-5.2-7.5-9.8V6z"/>'
              '<path d="m8.6 12 2.4 2.4 4.4-4.6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.4"/><circle class="solid" cx="12" cy="7.7" r="1.1"/>',
    "send": '<path d="M3.5 11.2 20.5 4l-7.2 17-2.3-7.5z"/><path d="m11 13.5 9.5-9.5"/>',
    "chat": '<path d="M4 5.5h16v10.5H9.5L5 20v-4H4z"/>',
    "list": '<path d="M8.5 6.5h12M8.5 12h12M8.5 17.5h12"/><circle class="solid" cx="4.4" cy="6.5" r="1.2"/>'
            '<circle class="solid" cx="4.4" cy="12" r="1.2"/><circle class="solid" cx="4.4" cy="17.5" r="1.2"/>',
    # D96: navigation, recognition and appearance
    "scan": '<path d="M3.5 8V6.2a2.7 2.7 0 0 1 2.7-2.7H8M16 3.5h1.8a2.7 2.7 0 0 1 2.7 2.7V8M20.5 16v1.8a2.7 2.7 0 0 1'
            '-2.7 2.7H16M8 20.5H6.2a2.7 2.7 0 0 1-2.7-2.7V16"/><circle cx="12" cy="10" r="2.8"/>'
            '<path d="M7.4 17.4a5 5 0 0 1 9.2 0"/>',
    "user": '<circle cx="12" cy="8.4" r="3.8"/><path d="M4.8 20.2a7.2 7.2 0 0 1 14.4 0"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.4a2.6 2.6 0 0 1 5 .9c0 1.8-2.5 2.2-2.5 3.9"/>'
            '<circle class="solid" cx="12" cy="17.3" r="1.1"/>',
    "flask": '<path d="M9.2 3.5h5.6M10.2 3.5v5.6L4.9 18.2a1.6 1.6 0 0 0 1.4 2.3h11.4a1.6 1.6 0 0 0 1.4-2.3L13.8 9.1V3.5"/>'
             '<path d="M7.6 14.5h8.8"/>',
    "sliders": '<path d="M4 7h9.5M17.5 7H20M4 17h3.5M11.5 17H20"/><circle cx="15.5" cy="7" r="2"/>'
               '<circle cx="9.5" cy="17" r="2"/>',
    "menu": '<path d="M4 6.5h16M4 12h16M4 17.5h16"/>',
    "close": '<path d="m6 6 12 12M18 6 6 18"/>',
    "sidebar": '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9.5 4v16"/>',
    "expand": '<path d="M14.5 3.5h6v6M9.5 20.5h-6v-6M20.5 3.5l-7 7M3.5 20.5l7-7"/>',
    "shrink": '<path d="M20.5 9.5h-6v-6M3.5 14.5h6v6M14.5 9.5l6-6M9.5 14.5l-6 6"/>',
    "filter": '<path d="M4 5h16l-6.2 7.3v5.9l-3.6 1.8v-7.7z"/>',
    # D99: the feature controls
    "toggle": '<rect x="2.5" y="7" width="19" height="10" rx="5"/><circle class="solid" cx="16.5" cy="12" r="3"/>',
}

SPRITE = (
    '<svg class="sprite" aria-hidden="true" focusable="false"><defs>'
    + "".join(f'<symbol id="i-{name}" viewBox="0 0 24 24">{body}</symbol>' for name, body in ICONS.items())
    + "</defs></svg>"
)


def icon(name: str, cls: str = "ic") -> str:
    """A decorative icon from the sprite; its meaning is always also given in text."""
    if name not in ICONS:
        raise KeyError(name)
    return f'<svg class="{cls}" aria-hidden="true" focusable="false"><use href="#i-{name}"/></svg>'


TAB_ICON = (  # the same mark on the Carbon tile, drawn heavier for a 16 px browser tab (D98)
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#2e302f"/>'
    '<g transform="translate(4 4)" fill="none" stroke="#eddcc9" stroke-linecap="round">'
    '<path d="M16.2 7.3A4.4 3.7 0 1 0 12 12a4.4 3.7 0 1 1-4.2 4.7" stroke-width="2.5"/>'
    '<circle cx="12" cy="12" r="10.1" stroke-width="1.4" opacity=".55"/></g></svg>'
).encode("ascii")
