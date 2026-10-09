"""D71: the status page in a real browser. A page that stops refreshing drops its current claims; only a fresh page restores them.

Headless Chromium loads the real StatusServer and is driven over its DevTools pipe (stdlib only: no Selenium,
Playwright or websocket package). The runtime is a stand-in whose report the test sets.

- **Refresh loss** is a refresh request that gets no answer, as through a stalled SSH forward: the browser keeps
  showing the old page. While that request is pending, Chromium holds DevTools messages for the page, so the test
  reads what is on screen from a screenshot (composited by the browser): the page's green, which only current claims
  use, must be gone, and the red NOT CURRENT warning shown. A refresh the browser gives up (``Page.stopLoading``)
  leaves the same old page, whose visible text (``innerText``, which leaves out hidden elements) is then read.
- **Recovery** is a late answer to the pending refresh, the page's next refresh, or a reload. A fresh page whose
  report is stale stays without current claims: the server-side freshness checks still decide.

Skipped where no Chromium-family browser is found or it does not start; SENTINEL_TEST_BROWSER names one. The
browser gets its own profile, home and temporary directory, removed afterwards, and every host name resolves to
nothing, so it reaches only the loopback page.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import os
import re
import select
import shutil
import signal
import struct
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from sentinel.config import FreshnessConfig
from sentinel.contracts import PixelFormat
from sentinel.live_preview import LivePreview
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import EncodedJpeg
from sentinel.media.recent import RecentFrames
from sentinel.status_page import PAGE_CURRENT_S, REFRESH_S, StatusServer
from sentinel.storage.database import Database

BROWSER_ENV = "SENTINEL_TEST_BROWSER"
# The chromium snap's own binary, run directly: /snap/bin/chromium needs snap confinement, which a sandboxed shell
# may not allow. Ubuntu's /usr/bin/chromium-browser is never used: that wrapper changes desktop settings.
SNAP_CHROMIUM = "/snap/chromium/current/usr/lib/chromium-browser/chrome"
BROWSER_NAMES = ("google-chrome", "google-chrome-stable", "chromium")
# An 8×8 grey JPEG (OpenCV, quality 50), so the browser has a real picture to decode and show.
PICTURE = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBk"
    "eFxlZ2P/wAALCAAIAAgBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEG"
    "E1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJ"
    "ipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/ACv/2Q=="
)
# What a viewer can read: the page's own freshness, the runtime's health, and the live view.
PAGE_CURRENT = "This page is current"
PAGE_NOT_CURRENT = f"NOT CURRENT: this page has not refreshed for {PAGE_CURRENT_S} s"
HEALTHY = "All components working"
HEALTHY_LAST_REPORTED = "Last reported: all components working"
LIVE = "LIVE: the newest frame of the connected stream"
LIVE_LAST_REPORTED = "NOT CURRENT: last reported LIVE"
PICTURE_NOT_CURRENT = f"NOT CURRENT: this picture is old; the page has not refreshed for {PAGE_CURRENT_S} s"
PROBE = """(() => {
  const img = document.querySelector('.pv img');
  let opacity = null;
  if (img) {
    opacity = 1;
    for (let node = img; node; node = node.parentElement) opacity *= parseFloat(getComputedStyle(node).opacity);
  }
  const report = document.querySelector('.report');
  return {document: performance.timeOrigin, since_navigation_ms: performance.now(), ready: document.readyState,
          text: document.body ? document.body.innerText : '',
          report_opacity: report ? parseFloat(getComputedStyle(report).opacity) : null,
          picture: img ? {opacity: opacity, loaded: img.complete && img.naturalWidth > 0} : null};
})()"""
VIEWPORT = (1000, 1000)
SHOT_SCALE = 0.25
BAD_RGB = (0xFD, 0xEC, 0xEA)  # STYLE's --bad, light scheme: undimmed only in the page's own NOT CURRENT warning
GREEN_TINT = 2  # g - max(r, b): only current claims are green (--ok), dimmed or not; text is grey (no LCD fringes)
GENERATED = re.compile(r"generated at (\d{4}-\d\d-\d\dT[0-9:.]+\+00:00)")


class DevToolsError(RuntimeError):
    pass


class DevTools:
    """The Chrome DevTools protocol over --remote-debugging-pipe: NUL-terminated JSON, one flat page session."""

    def __init__(self, write_fd: int, read_fd: int) -> None:
        self._write, self._read = write_fd, read_fd
        self._buffer = b""
        self._id = 0
        self._session: str | None = None

    def call(self, method: str, params: dict[str, Any] | None = None, *, page: bool = False,
             timeout_s: float = 15.0) -> dict[str, Any]:
        self._id += 1
        message: dict[str, Any] = {"id": self._id, "method": method, "params": params or {}}
        if page:
            message["sessionId"] = self._session
        data = json.dumps(message).encode() + b"\0"
        while data:
            data = data[os.write(self._write, data):]
        deadline = time.monotonic() + timeout_s
        while True:
            reply = self._receive(deadline)
            if reply.get("id") == self._id:  # events and other replies are skipped
                if "error" in reply:
                    raise DevToolsError(str(reply["error"].get("message")))
                return reply.get("result", {})

    def _receive(self, deadline: float) -> dict[str, Any]:
        while b"\0" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no reply from the browser")
            if select.select([self._read], [], [], remaining)[0]:
                chunk = os.read(self._read, 1 << 16)
                if not chunk:
                    raise EOFError("the browser closed its DevTools pipe")
                self._buffer += chunk
        message, _, self._buffer = self._buffer.partition(b"\0")
        return json.loads(message)

    def open(self, url: str) -> None:
        target = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        self._session = self.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", {"width": VIEWPORT[0], "height": VIEWPORT[1],
                                                         "deviceScaleFactor": 1, "mobile": False}, page=True)
        self.call("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-color-scheme", "value": "light"}]},
                  page=True)
        self.call("Page.navigate", {"url": url}, page=True)

    def screenshot_colours(self) -> dict[str, int]:
        """What is on screen (works while a load is pending): green-tinted pixels, and the warning's red ones."""
        clip = {"x": 0, "y": 0, "width": VIEWPORT[0], "height": VIEWPORT[1], "scale": SHOT_SCALE}
        png = base64.b64decode(self.call("Page.captureScreenshot", {"format": "png", "clip": clip}, page=True)["data"])
        pixels = rgb_pixels(png)
        return {"green": sum(g - max(r, b) >= GREEN_TINT for r, g, b in pixels), "warning": pixels.count(BAD_RGB)}

    def read_page(self) -> dict[str, Any]:
        result = self.call("Runtime.evaluate", {"expression": PROBE, "returnByValue": True}, page=True)
        if "exceptionDetails" in result:
            raise DevToolsError("the probe failed")
        return result["result"]["value"]


def rgb_pixels(png: bytes) -> list[tuple[int, int, int]]:
    """The (r, g, b) of every pixel of an 8-bit, non-interlaced RGB or RGBA PNG (what Chromium's screenshots are)."""
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    pos, header, data = 8, b"", b""
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos:pos + 8])
        body, pos = png[pos + 8:pos + 8 + length], pos + 12 + length
        header, data = (body, data) if kind == b"IHDR" else (header, data + body if kind == b"IDAT" else data)
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", header)
    assert depth == 8 and colour in (2, 6) and interlace == 0, (depth, colour, interlace)
    step = 3 if colour == 2 else 4
    raw, stride = zlib.decompress(data), width * step
    previous, pixels = bytearray(stride), []
    for y in range(height):
        kind, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - step] if i >= step else 0
            b, c = previous[i], previous[i - step] if i >= step else 0
            if kind == 1:
                line[i] = (line[i] + a) & 255
            elif kind == 2:
                line[i] = (line[i] + b) & 255
            elif kind == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif kind == 4:
                guess = a + b - c
                near = min((abs(guess - a), 0, a), (abs(guess - b), 1, b), (abs(guess - c), 2, c))[2]
                line[i] = (line[i] + near) & 255
        pixels += [tuple(line[i:i + 3]) for i in range(0, stride, step)]
        previous = line
    return pixels


def find_browser() -> str | None:
    named = os.environ.get(BROWSER_ENV)
    if named:
        return named
    for path in (SNAP_CHROMIUM, *(shutil.which(name) for name in BROWSER_NAMES)):
        if path and os.path.isfile(path) and os.access(path, os.X_OK) and os.path.realpath(path) != "/usr/bin/snap":
            return path
    return None


@contextlib.contextmanager
def headless_browser(executable: str) -> Iterator[DevTools]:
    """A headless browser of its own, in its own session; DevTools on fds 3 (commands) and 4 (replies).

    Its files are in a short temporary directory: the browser makes a Unix socket under TMPDIR, and a socket's
    path must stay under 108 bytes, which a deep pytest directory can exceed.
    """
    if not (hasattr(os, "posix_spawn") and hasattr(os, "waitid")):
        pytest.skip("needs os.posix_spawn and os.waitid")
    with tempfile.TemporaryDirectory(prefix="sentinel-browser-") as name:
        yield from _browser(executable, Path(name))


def _browser(executable: str, root: Path) -> Iterator[DevTools]:
    home, profile = root / "home", root / "profile"
    home.mkdir()
    profile.mkdir()
    to_browser, commands = os.pipe()
    replies, from_browser = os.pipe()
    child_in = fcntl.fcntl(to_browser, fcntl.F_DUPFD_CLOEXEC, 10)  # above 4, so the dup2s below never collide
    child_out = fcntl.fcntl(from_browser, fcntl.F_DUPFD_CLOEXEC, 10)
    os.close(to_browser)
    os.close(from_browser)
    args = [executable, "--headless", "--remote-debugging-pipe", f"--user-data-dir={profile}", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--disable-background-networking",
            "--disable-component-update", "--disable-default-apps", "--disable-gpu", "--mute-audio",
            "--password-store=basic", "--use-mock-keychain", "--disable-lcd-text", "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1", "about:blank"]
    env = {"HOME": str(home), "TMPDIR": str(root), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
    actions = [
        (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
        (os.POSIX_SPAWN_OPEN, 1, str(root / "browser.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
        (os.POSIX_SPAWN_DUP2, 1, 2),
        (os.POSIX_SPAWN_DUP2, child_in, 3),
        (os.POSIX_SPAWN_DUP2, child_out, 4),
    ]
    try:
        pid = os.posix_spawn(executable, args, env, file_actions=actions, setsid=True)
    except (OSError, NotImplementedError) as exc:
        os.close(commands)
        os.close(replies)
        pytest.skip(f"the browser did not start ({type(exc).__name__})")
    finally:
        os.close(child_in)
        os.close(child_out)
    tools = DevTools(commands, replies)
    try:
        try:
            tools.call("Browser.getVersion", timeout_s=60.0)
        except (TimeoutError, EOFError, OSError, DevToolsError) as exc:
            log = (root / "browser.log").read_text(errors="replace").strip().splitlines()[-1:]
            pytest.skip(f"the browser did not answer on its DevTools pipe ({type(exc).__name__}; log: {log})")
        yield tools
    finally:
        with contextlib.suppress(TimeoutError, EOFError, OSError, DevToolsError):
            tools.call("Browser.close", timeout_s=5.0)
        os.close(commands)
        os.close(replies)
        wait_until(lambda: os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None, 10.0)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)  # its helpers too: the group is still its own until it is reaped
        os.waitpid(pid, 0)


class Runtime:
    """The runtime stand-in: how old its last update is, and whether the page's requests are answered."""

    def __init__(self) -> None:
        self.answering = threading.Event()
        self.answering.set()
        self.lag_s = 0.0
        self.unanswered = 0  # page requests that arrived while the runtime was not answering

    def snapshot(self) -> dict[str, Any]:
        if not self.answering.is_set():
            self.unanswered += 1
            self.answering.wait(60.0)
        updated = datetime.now(timezone.utc) - timedelta(seconds=self.lag_s)
        return {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": updated.isoformat()},
                "degraded": [], "components": {"capture": {"state": "streaming", "stream_epoch": 1}}}


class Camera:
    """A live view over a real FrameStamper and RecentFrames on a FakeClock; the encoder returns PICTURE."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.frames = RecentFrames()
        self.preview = LivePreview(self.frames, self.clock, FreshnessConfig(), lambda: self.stamper.current_stream,
                                   encode=lambda *_, **__: EncodedJpeg(PICTURE, 8, 8))
        self.stamper.connect()
        self.frame()

    def frame(self) -> None:
        self.clock.advance(1 / 15)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.frames.put(frame, object())


def wait_for(browser: DevTools, predicate: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    seen: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        with contextlib.suppress(DevToolsError):  # a page being replaced has no context for a moment
            seen = browser.read_page()
            if seen["ready"] == "complete" and predicate(seen):
                return seen
        time.sleep(0.1)
    raise AssertionError(f"the page never reached the expected state; last seen: {seen!r}")


def wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def claims_current(seen: dict[str, Any]) -> bool:
    return PAGE_CURRENT in seen["text"] and HEALTHY in seen["text"] and LIVE in seen["text"]


def current_with_picture(seen: dict[str, Any]) -> bool:
    return claims_current(seen) and bool((seen["picture"] or {}).get("loaded"))


def generated(seen: dict[str, Any]) -> datetime:
    """When the server built the page shown, from its own text."""
    found = GENERATED.search(seen["text"])
    assert found, seen["text"][:200]
    return datetime.fromisoformat(found[1])


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
    executable = find_browser()
    if executable is None:
        pytest.skip(f"no Chromium-family browser found (set {BROWSER_ENV})")
    Database.open(tmp_path / "sentinel.db").close()
    runtime, camera = Runtime(), Camera()
    server = StatusServer(0, runtime.snapshot, tmp_path / "sentinel.db", preview=lambda: camera.preview)
    server.start()
    try:
        with headless_browser(executable) as browser:
            browser.open(f"http://127.0.0.1:{server.address[1]}/")

            # 1. Current: the page, the runtime's health and the live view say so; the picture is shown.
            assert_current(wait_for(browser, current_with_picture, 30.0))
            on_screen = browser.screenshot_colours()
            assert on_screen["green"] > 0 and on_screen["warning"] == 0, on_screen

            # 2. Refresh loss: the refresh hangs and the browser keeps the old page. Once that page is older than
            # PAGE_CURRENT_S, nothing green (no current claim, dimmed or not) is on screen, and the warning is.
            lose_refresh(runtime)
            time.sleep(PAGE_CURRENT_S + 1.0)
            on_screen = browser.screenshot_colours()
            assert on_screen["green"] == 0 and on_screen["warning"] > 0, on_screen

            # A late answer, carrying a stale report, gives a current page that claims nothing current: the
            # server-side checks still decide what is live and healthy.
            runtime.lag_s = 30.0
            camera.clock.advance(3.0)  # the newest frame is now older than freshness.stale_after_s
            runtime.answering.set()
            answered = wait_for(browser, lambda s: PAGE_CURRENT in s["text"], 10.0)
            assert "STALE: not current" in answered["text"] and "runtime status not updated for 30 s" in answered["text"]
            for claim in (HEALTHY, LIVE, PAGE_NOT_CURRENT, HEALTHY_LAST_REPORTED, LIVE_LAST_REPORTED):
                assert claim not in answered["text"], claim
            assert answered["picture"] is None

            # 3. A fresh report: the page's next refresh restores every current claim and the picture.
            runtime.lag_s = 0.0
            camera.frame()
            restored = wait_for(browser, current_with_picture, REFRESH_S + 5.0)
            assert restored["document"] != answered["document"]
            assert_current(restored)
            assert browser.screenshot_colours()["green"] > 0

            # 4. A refresh the browser gives up: the same old page stays, and its visible text says it is not
            # current, with the runtime's report as last reported.
            cut_off = lose_refresh(runtime)
            browser.call("Page.stopLoading", page=True)
            stale = wait_for(browser, lambda s: PAGE_NOT_CURRENT in s["text"], PAGE_CURRENT_S + 3.0)
            assert generated(stale) <= cut_off  # built before the loss: never replaced
            assert_not_current(stale)
            time.sleep(1.0)
            still = browser.read_page()
            assert still["document"] == stale["document"]
            assert_not_current(still)

            # Only a refresh that succeeds (here the operator's reload) makes it current again.
            runtime.answering.set()
            browser.call("Page.reload", page=True)
            reloaded = wait_for(browser, current_with_picture, 10.0)
            assert reloaded["document"] != stale["document"] and generated(reloaded) > cut_off
            assert_current(reloaded)
    finally:
        runtime.answering.set()
        assert server.stop(3.0)
