"""Headless Chromium for the status-page browser tests, driven over its DevTools pipe (stdlib only).

No Selenium, Playwright or websocket package, and no dependency: the Chrome DevTools protocol travels as
NUL-terminated JSON over ``--remote-debugging-pipe`` (fds 3 and 4 of the browser). The browser gets its own profile,
home and temporary directory, removed afterwards; every host name resolves to nothing, so it reaches only loopback
pages; LCD text smoothing is off, so text has no coloured fringes. Tests skip, with the reason and the browser log's
last line, when no Chromium-family browser is found or it does not start; SENTINEL_TEST_BROWSER names one.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import os
import select
import shutil
import signal
import struct
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

BROWSER_ENV = "SENTINEL_TEST_BROWSER"
# The chromium snap's own binary, run directly: /snap/bin/chromium needs snap confinement, which a sandboxed shell
# may not allow. Ubuntu's /usr/bin/chromium-browser is never used: that wrapper changes desktop settings.
SNAP_CHROMIUM = "/snap/chromium/current/usr/lib/chromium-browser/chrome"
BROWSER_NAMES = ("google-chrome", "google-chrome-stable", "chromium")
SHOTS_ENV = "SENTINEL_TEST_SCREENSHOTS"  # a directory: tests may save the screenshots they take there, for review


class DevToolsError(RuntimeError):
    pass


class Browser:
    """One browser connection; ``page()`` opens a tab with its own flat DevTools session."""

    def __init__(self, write_fd: int, read_fd: int) -> None:
        self._write, self._read = write_fd, read_fd
        self._buffer = b""
        self._id = 0

    def call(self, method: str, params: dict[str, Any] | None = None, *, session: str | None = None,
             timeout_s: float = 15.0) -> dict[str, Any]:
        self._id += 1
        message: dict[str, Any] = {"id": self._id, "method": method, "params": params or {}}
        if session is not None:
            message["sessionId"] = session
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

    def page(self, *, width: int = 1000, height: int = 1000, mobile: bool = False,
             media: dict[str, str] | None = None) -> Page:
        target = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = self.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        page = Page(self, target, session)
        page.call("Emulation.setFocusEmulationEnabled", {"enabled": True})  # keyboard focus as in a focused window
        page.viewport(width, height, mobile=mobile)
        page.media({"prefers-color-scheme": "light", **(media or {})})
        return page


class Page:
    def __init__(self, browser: Browser, target: str, session: str) -> None:
        self.browser, self.target, self.session = browser, target, session
        self.width = self.height = 0

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float = 15.0) -> dict[str, Any]:
        return self.browser.call(method, params, session=self.session, timeout_s=timeout_s)

    def viewport(self, width: int, height: int, *, mobile: bool = False) -> None:
        self.width, self.height = width, height
        self.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1,
                                                         "mobile": mobile})

    def media(self, features: dict[str, str]) -> None:
        self.call("Emulation.setEmulatedMedia", {"features": [{"name": k, "value": v} for k, v in features.items()]})

    def navigate(self, url: str) -> None:
        self.call("Page.navigate", {"url": url})

    def evaluate(self, expression: str) -> Any:
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise DevToolsError(f"evaluation failed: {result['exceptionDetails'].get('text')}")
        return result["result"].get("value")

    def key(self, key: str, *, shift: bool = False) -> None:
        """Press and release one key (Tab, Enter, Space, ...), as a keyboard user would."""
        codes = {"Tab": 9, "Enter": 13, " ": 32, "Escape": 27, "ArrowDown": 40, "ArrowUp": 38}
        base = {"key": key, "code": "Space" if key == " " else key, "windowsVirtualKeyCode": codes.get(key, 0),
                "modifiers": 8 if shift else 0}
        text = {"Enter": "\r", " ": " "}.get(key)
        self.call("Input.dispatchKeyEvent", {**base, "type": "keyDown", **({"text": text} if text else {})})
        self.call("Input.dispatchKeyEvent", {**base, "type": "keyUp"})

    def screenshot(self, *, scale: float = 1.0, full: bool = False) -> bytes:
        params: dict[str, Any] = {"format": "png"}
        if full:
            params["captureBeyondViewport"] = True
        else:
            params["clip"] = {"x": 0, "y": 0, "width": self.width, "height": self.height, "scale": scale}
        return base64.b64decode(self.call("Page.captureScreenshot", params, timeout_s=30.0)["data"])

    def save_screenshot(self, name: str, **options: Any) -> None:
        """For review only, when SENTINEL_TEST_SCREENSHOTS names a directory; never part of a pass or a failure."""
        folder = os.environ.get(SHOTS_ENV)
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            (Path(folder) / f"{name}.png").write_bytes(self.screenshot(**options))

    def close(self) -> None:
        with contextlib.suppress(TimeoutError, EOFError, OSError, DevToolsError):
            self.browser.call("Target.closeTarget", {"targetId": self.target})


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


def wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@contextlib.contextmanager
def headless_browser(executable: str | None = None) -> Iterator[Browser]:
    """A headless browser of its own, in its own session; DevTools on fds 3 (commands) and 4 (replies).

    Its files are in a short temporary directory: the browser makes a Unix socket under TMPDIR, and a socket's
    path must stay under 108 bytes, which a deep pytest directory can exceed.
    """
    executable = executable or find_browser()
    if executable is None:
        pytest.skip(f"no Chromium-family browser found (set {BROWSER_ENV})")
    if not (hasattr(os, "posix_spawn") and hasattr(os, "waitid")):
        pytest.skip("needs os.posix_spawn and os.waitid")
    with tempfile.TemporaryDirectory(prefix="sentinel-browser-") as name:
        yield from _browser(executable, Path(name))


def _browser(executable: str, root: Path) -> Iterator[Browser]:
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
            "--password-store=basic", "--use-mock-keychain", "--disable-lcd-text",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1", "about:blank"]
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
    browser = Browser(commands, replies)
    try:
        try:
            browser.call("Browser.getVersion", timeout_s=60.0)
        except (TimeoutError, EOFError, OSError, DevToolsError) as exc:
            log = (root / "browser.log").read_text(errors="replace").strip().splitlines()[-1:]
            pytest.skip(f"the browser did not answer on its DevTools pipe ({type(exc).__name__}; log: {log})")
        yield browser
    finally:
        with contextlib.suppress(TimeoutError, EOFError, OSError, DevToolsError):
            browser.call("Browser.close", timeout_s=5.0)
        os.close(commands)
        os.close(replies)
        wait_until(lambda: os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None, 10.0)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)  # its helpers too: the group is still its own until it is reaped
        os.waitpid(pid, 0)
