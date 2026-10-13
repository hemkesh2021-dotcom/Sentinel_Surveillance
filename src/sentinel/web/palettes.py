"""Colour palettes for the status pages (D76, offered in D96): shared design tokens, derived per preset and mode.

Each preset names two or three base colours. Every token the stylesheet uses (surfaces, text, borders, focus, the
accent and its text) is derived from them for a light and a dark mode, in OKLab, and then moved in lightness only as
far as needed to meet its contrast floor (WCAG 2.x relative luminance) against every surface it appears on: text 4.5:1
(primary text 7:1), focus rings, control borders and selection markers 3:1. ``contrast_report`` lists every pairing,
and the tests check all of them for all presets in both modes.

States keep one meaning in every preset: their colours (healthy and live green, warning amber, error red, the
not-current banner) are fixed per mode and never derived from a palette, a palette's accent never marks a state, and
every state is also given by its text and icon. Camera pictures and snapshots are never recoloured by any of this.

Carbon & Cream is the default; its dark surfaces are the maintainer's: canvas #1C1D1C, sidebar #181918, panels
#232423, raised surfaces #2E302F. Portable: stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

DEFAULT_PALETTE = "carbon-cream"
SURFACES = ("canvas", "sidebar", "panel", "raised", "sunken")
TEXT_FLOOR = 4.5
INK_FLOOR = 7.0
UI_FLOOR = 3.0

# States: fixed per mode, the same in every preset (text on its own background; contrast checked in the tests).
STATES = {
    "light": {"ok-bg": "#dcebda", "ok-ink": "#1d6a3b", "warn-bg": "#f6e3bf", "warn-ink": "#734400",
              "crit-bg": "#f8dad5", "crit-ink": "#a11d1d", "stale-bg": "#7a2e12", "stale-ink": "#fdf6e8"},
    "dark": {"ok-bg": "#1e3326", "ok-ink": "#8fd3a0", "warn-bg": "#3a2e19", "warn-ink": "#f2c76e",
             "crit-bg": "#3f2320", "crit-ink": "#ff9e94", "stale-bg": "#f2a65a", "stale-ink": "#1c1d1c"},
}


# ---------------------------------------------------------------- colour arithmetic


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    text = hex_colour.lstrip("#")
    return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in rgb)


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _from_linear(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def luminance(hex_colour: str) -> float:
    r, g, b = (_to_linear(c) for c in _rgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _oklab(hex_colour: str) -> tuple[float, float, float]:
    r, g, b = (_to_linear(c) for c in _rgb(hex_colour))
    l_ = math.copysign(abs(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3), 1)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _from_oklab(lightness: float, a: float, b: float) -> tuple[float, float, float] | None:
    """Gamma-encoded sRGB, or None outside the sRGB gamut."""
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    linear = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
              -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
              -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    if any(c < -1e-4 or c > 1 + 1e-4 for c in linear):
        return None
    return tuple(_from_linear(min(1.0, max(0.0, c))) for c in linear)  # type: ignore[return-value]


def _lch(lightness: float, chroma: float, hue: float) -> str:
    """OKLab lightness, chroma and hue as hex, the chroma reduced (hue kept) until it is inside the sRGB gamut."""
    lightness = min(1.0, max(0.0, lightness))
    while True:
        rgb = _from_oklab(lightness, chroma * math.cos(hue), chroma * math.sin(hue))
        if rgb is not None:
            return _hex(rgb)
        if chroma <= 0:
            return _hex((lightness, lightness, lightness))
        chroma = max(0.0, chroma - 0.002)


def tone(seed: str, lightness: float, max_chroma: float | None = None) -> str:
    """``seed``'s hue at OKLab ``lightness``, with its chroma (at most ``max_chroma``), reduced into the sRGB gamut."""
    _, a, b = _oklab(seed)
    chroma = math.hypot(a, b)
    if max_chroma is not None:
        chroma = min(chroma, max_chroma)
    return _lch(lightness, chroma, math.atan2(b, a))


def mix(a: str, b: str, weight: float) -> str:
    """``weight`` of ``b`` mixed into ``a``, in OKLab."""
    la, lb = _oklab(a), _oklab(b)
    lightness, ca, cb = (x + (y - x) * weight for x, y in zip(la, lb))
    return _lch(lightness, math.hypot(ca, cb), math.atan2(cb, ca))


def at_least(colour: str, against: tuple[str, ...], floor: float, *, lighter: bool) -> str:
    """``colour``, moved lighter or darker in OKLab lightness only as far as needed to reach ``floor`` against all of
    ``against``. Raises if no lightness reaches it."""
    if all(contrast(colour, other) >= floor + 0.02 for other in against):
        return colour.lower()
    lightness, a, b = _oklab(colour)
    chroma, hue = math.hypot(a, b), math.atan2(b, a)
    step = 0.004 if lighter else -0.004
    while 0.0 <= lightness <= 1.0:
        candidate = _lch(lightness, chroma, hue)
        if all(contrast(candidate, other) >= floor + 0.02 for other in against):
            return candidate
        lightness += step
    raise ValueError(f"no lightness of {colour} reaches {floor}:1 against {against}")


def best_text(fill: str, choices: tuple[str, ...]) -> str:
    """The choice with the most contrast on ``fill``."""
    return max(choices, key=lambda choice: contrast(fill, choice))


def rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (round(c * 255) for c in _rgb(hex_colour))
    return f"rgba({r}, {g}, {b}, {alpha:g})"


# ---------------------------------------------------------------- presets


@dataclass(frozen=True)
class Seeds:
    """One mode's starting colours. ``tint``: the surfaces' hue; ``accent``: fills of selected and primary controls;
    ``marker``: selection bars and focus rings; ``link``: link text (default: the marker); ``lift``: the accent's
    OKLab lightness in this mode, when the base colour itself is too dark or too light for it; ``surfaces``: exact
    surfaces that replace the derived ones."""

    tint: str
    ink: str
    accent: str
    marker: str
    link: str | None = None
    lift: float | None = None
    surfaces: dict[str, str] = field(default_factory=dict)
    accent_ink: str | None = None  # text on the accent, used when it reaches the floor (else the best of the rest)


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    colours: tuple[str, ...]  # as named by the maintainer; shown as the preset's swatches
    brand: str  # the monogram's fill: the palette's character, never a state
    light: Seeds
    dark: Seeds
    brand_ink: str | None = None  # the monogram's letter, used when it reaches 3:1 on the brand


PRESETS: tuple[Preset, ...] = (  # in the maintainer's order; Carbon & Cream first (the default, emitted first)
    Preset("carbon-cream", "Carbon & Cream", ("#2E302F", "#EDDCC9"), "#2E302F", brand_ink="#EDDCC9",
           light=Seeds(tint="#EDDCC9", ink="#2E302F", accent="#2E302F", marker="#2E302F", accent_ink="#EDDCC9"),
           dark=Seeds(tint="#2E302F", ink="#EDDCC9", accent="#EDDCC9", marker="#EDDCC9",
                      surfaces={"canvas": "#1C1D1C", "sidebar": "#181918", "panel": "#232423", "raised": "#2E302F",
                                "sunken": "#191A19"})),
    Preset("sesame-mango", "Sesame & Mango", ("#1F1D1A", "#F4B54A"), "#F4B54A",
           light=Seeds(tint="#F4B54A", ink="#1F1D1A", accent="#F4B54A", marker="#1F1D1A", link="#F4B54A"),
           dark=Seeds(tint="#1F1D1A", ink="#F7EEDF", accent="#F4B54A", marker="#F4B54A")),
    Preset("matcha-cream", "Matcha & Cream", ("#4E5529", "#F0E2C6"), "#4E5529", brand_ink="#F0E2C6",
           light=Seeds(tint="#F0E2C6", ink="#4E5529", accent="#4E5529", marker="#4E5529", accent_ink="#F0E2C6"),
           dark=Seeds(tint="#4E5529", ink="#F0E2C6", accent="#4E5529", marker="#4E5529", lift=0.76)),
    Preset("teal-coconut", "Teal & Coconut", ("#174F4A", "#FDFAF5"), "#174F4A", brand_ink="#FDFAF5",
           light=Seeds(tint="#FDFAF5", ink="#174F4A", accent="#174F4A", marker="#174F4A", accent_ink="#FDFAF5"),
           dark=Seeds(tint="#174F4A", ink="#FDFAF5", accent="#174F4A", marker="#174F4A", lift=0.74)),
    Preset("burgundy-rose", "Burgundy & Rose", ("#501627", "#AF144B", "#FACCD7"), "#AF144B", brand_ink="#FDEFF2",
           light=Seeds(tint="#FACCD7", ink="#501627", accent="#AF144B", marker="#AF144B", accent_ink="#FDEFF2"),
           dark=Seeds(tint="#501627", ink="#FBE8EC", accent="#AF144B", marker="#FACCD7", accent_ink="#FDEFF2")),
    Preset("port-fern-mauve", "Port, Fern & Mauve", ("#5C2C35", "#9AA067", "#E7DCD9"), "#9AA067",
           light=Seeds(tint="#E7DCD9", ink="#5C2C35", accent="#5C2C35", marker="#5C2C35", accent_ink="#F3ECEA"),
           dark=Seeds(tint="#5C2C35", ink="#F1E9E7", accent="#5C2C35", marker="#5C2C35", lift=0.74)),
    Preset("charcoal-sheen", "Charcoal & Sheen", ("#2B2C30", "#D9CE52", "#F0EDE5"), "#D9CE52",
           light=Seeds(tint="#F0EDE5", ink="#2B2C30", accent="#D9CE52", marker="#2B2C30", link="#D9CE52"),
           dark=Seeds(tint="#2B2C30", ink="#F0EDE5", accent="#D9CE52", marker="#D9CE52")),
)
PALETTES = {preset.key: preset for preset in PRESETS}

# Lightness and chroma ceilings of the derived surfaces (OKLab), per mode: shallow steps, so panels separate by a fine
# border and a soft shadow rather than by strong colour.
_LIGHT_SURFACES = {"canvas": (0.955, 0.016), "sidebar": (0.93, 0.024), "panel": (0.982, 0.008),
                   "raised": (0.995, 0.004), "sunken": (0.925, 0.02)}
_DARK_SURFACES = {"canvas": (0.215, 0.012), "sidebar": (0.197, 0.012), "panel": (0.245, 0.012),
                  "raised": (0.285, 0.012), "sunken": (0.19, 0.012)}


@lru_cache(maxsize=None)
def tokens(key: str, mode: str) -> dict[str, str]:
    """Every colour token of preset ``key`` in ``mode`` (light or dark)."""
    preset = PALETTES[key]
    seeds = preset.light if mode == "light" else preset.dark
    lighter = mode == "dark"
    plan = _DARK_SURFACES if lighter else _LIGHT_SURFACES
    out = {name: seeds.surfaces.get(name) or tone(seeds.tint, lightness, chroma)
           for name, (lightness, chroma) in plan.items()}
    surfaces = tuple(out[name] for name in SURFACES)
    out["ink"] = at_least(seeds.ink, surfaces, INK_FLOOR, lighter=lighter)
    out["muted"] = at_least(mix(out["ink"], out["panel"], 0.42), surfaces, TEXT_FLOOR, lighter=lighter)
    out["border"] = at_least(mix(out["ink"], out["panel"], 0.62), surfaces, UI_FLOOR, lighter=lighter)
    accent = tone(seeds.accent, seeds.lift, 0.12) if seeds.lift is not None else seeds.accent
    out["accent"] = accent
    preferred = seeds.accent_ink
    out["accent-ink"] = (preferred.lower() if preferred and contrast(accent, preferred) >= TEXT_FLOOR + 0.2 else
                         best_text(accent, (out["raised"], out["ink"], out["canvas"], "#ffffff", "#1c1d1c")))
    if contrast(accent, out["accent-ink"]) < TEXT_FLOOR:
        raise ValueError(f"{key} {mode}: no text reaches {TEXT_FLOOR}:1 on its accent")
    marker = tone(seeds.marker, seeds.lift, 0.12) if seeds.lift is not None and seeds.marker == seeds.accent \
        else seeds.marker
    out["marker"] = at_least(marker, surfaces, UI_FLOOR, lighter=lighter)
    out["focus"] = out["marker"]
    out["link"] = at_least(seeds.link or out["marker"], surfaces, TEXT_FLOOR, lighter=lighter)
    out["brand"] = preset.brand
    out["brand-ink"] = (preset.brand_ink.lower() if preset.brand_ink and contrast(preset.brand, preset.brand_ink) >= 4.5
                        else best_text(preset.brand, ("#ffffff", "#1c1d1c", out["ink"], out["raised"])))
    out["line"] = rgba(out["ink"], 0.11 if lighter else 0.13)
    out["line-strong"] = rgba(out["ink"], 0.2 if lighter else 0.24)
    out["shadow-lo"] = rgba("#000000", 0.5) if lighter else rgba(mix(out["ink"], out["canvas"], 0.35), 0.16)
    out["shadow-hi"] = rgba("#ffffff", 0.03) if lighter else rgba("#ffffff", 0.85)
    out["off-bg"] = out["sunken"]
    out["off-ink"] = out["muted"]
    out["done-bg"] = out["raised"]
    out["done-ink"] = out["ink"]
    out.update(STATES[mode])
    return out


def contrast_report(key: str, mode: str) -> list[tuple[str, str, float, float]]:
    """(foreground, background, ratio, floor) for every pairing the stylesheet relies on."""
    t = tokens(key, mode)
    pairs: list[tuple[str, str, float]] = []
    for surface in SURFACES:
        pairs += [("ink", surface, INK_FLOOR), ("muted", surface, TEXT_FLOOR), ("link", surface, TEXT_FLOOR),
                  ("border", surface, UI_FLOOR), ("focus", surface, UI_FLOOR), ("marker", surface, UI_FLOOR)]
    pairs += [("accent-ink", "accent", TEXT_FLOOR), ("brand-ink", "brand", UI_FLOOR), ("off-ink", "off-bg", TEXT_FLOOR),
              ("done-ink", "done-bg", TEXT_FLOOR)]
    pairs += [(f"{state}-ink", f"{state}-bg", TEXT_FLOOR) for state in ("ok", "warn", "crit", "stale")]
    return [(fg, bg, round(contrast(t[fg], t[bg]), 2), floor) for fg, bg, floor in pairs]


def _block(selector: str, values: dict[str, str]) -> str:
    body = " ".join(f"--{name}: {value};" for name, value in values.items())
    return f"{selector} {{ {body} }}"


@lru_cache(maxsize=None)
def palette_css() -> str:
    """The tokens of every preset, light and dark (the browser's preference, or data-theme), as CSS."""
    lines = ["/* D76: generated by sentinel.web.palettes; every pairing's contrast is checked in the tests. */"]
    for preset in PRESETS:
        light, dark = tokens(preset.key, "light"), tokens(preset.key, "dark")
        selector = f":root[data-palette={preset.key}]"
        if preset.key == DEFAULT_PALETTE:
            selector = f":root, {selector}"
        lines.append(_block(selector, {"scheme": "light", **light}).replace("--scheme: light;", "color-scheme: light;"))
        lines.append("@media (prefers-color-scheme: dark) { "
                     + _block(", ".join(f"{part.strip()}:not([data-theme=light])" for part in selector.split(",")),
                              {"scheme": "dark", **dark}).replace("--scheme: dark;", "color-scheme: dark;") + " }")
        lines.append(_block(", ".join(f"{part.strip()}[data-theme=dark]" for part in selector.split(",")),
                            {"scheme": "dark", **dark}).replace("--scheme: dark;", "color-scheme: dark;"))
    return "\n".join(lines) + "\n"


def swatch_css() -> str:
    """Each preset's named colours as the appearance panel's swatch dots (D96); classes, so no inline style is needed.
    The dots show the palette's own colours; they are never a state."""
    return "".join(f".sw-{preset.key} .d{i} {{ background: {colour.lower()}; }}\n"
                   for preset in PRESETS for i, colour in enumerate(preset.colours, 1))
