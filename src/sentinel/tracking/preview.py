"""Operator-only live preview for `sentinel track probe --preview` (diagnostic P1, session 26; not a product feature).

While a track probe runs, it shows the operator where the person boxes are (U23):
exactly the boxes the probe counts, drawn on the frame each result was computed
from, with track ID, confidence and CONFIRMED (C) or TENTATIVE (T), and the
frame's stream epoch, sequence and ingest time. Detections the adapter drops
(below the threshold, or without an activated track) are not shown: they are
not counted either.

- **The counted boxes on their own frame.** TrackProbe passes each counted
  result with its frame to PreviewFeed.observe after counting. An update holds
  that frame's image, the same object the tracker processed (OpenCV returns a
  new array per read, so capture never writes into it), and the published
  observations. Drawing happens on a copy.
- **Bounded; never in the way of inference.** PreviewFeed keeps one update; a
  newer one replaces it and the older is counted as dropped. The probe thread
  only swaps a reference under a lock. Drawing and JPEG encoding run in the
  viewer's HTTP thread, at most MAX_FPS frames per second. At most one viewer
  at a time. Per connection the kernel send buffer is fixed (SEND_BUFFER_BYTES;
  Linux doubles it), one JPEG is written at a time, and a viewer that stops
  reading for WRITE_TIMEOUT_S is disconnected. So at most two frame images are
  held for the preview (the pending one and the one being drawn, with its copy).
- **Operator only.** It binds 127.0.0.1 and nothing else, before any socket is
  opened. Requests must name 127.0.0.1 or localhost with the port in their Host
  header (as D-2), and the path must start with the operator's token
  (SENTINEL_PREVIEW_TOKEN, compared in constant time). It is reached over an
  SSH port forward. Nothing is logged: no request lines and no error output.
  The token and the URL never appear in any output or in the summary.
- **Saves nothing.** Images exist in memory and on the wire only; responses
  are not cached and may not be framed. The summary holds counts and timings.
  Timings and resource figures from a preview run include the preview's own
  work, so they are not a performance baseline.

Portable: cv2 is imported only by render_jpeg, the device renderer, when a
viewer is connected.
"""

from __future__ import annotations

import hmac
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from ..contracts import TrackStatus
from ..media.capture import CapturedFrame
from ..media.probe import MAX_SAMPLES, percentiles_ms
from .tracker import FrameOutcome, TrackingResult

LOOPBACK_HOST = "127.0.0.1"
PREVIEW_PORT = 18091
MAX_FPS = 5.0
TOKEN_ENV = "SENTINEL_PREVIEW_TOKEN"
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22,128}$")
WRITE_TIMEOUT_S = 5.0  # also bounds how long a request may take to arrive
SEND_BUFFER_BYTES = 262_144
WAIT_S = 0.25  # how often a waiting viewer thread checks for the stop
JPEG_QUALITY = 80
HEADER_PX = 24  # a strip above the frame for the header, so it never covers a box
BOUNDARY = "frame"
VIEWER_COUNTS = {  # summary name: counter
    "served": "viewers_served",
    "refused_busy": "viewers_refused_busy",
    "disconnected": "viewers_disconnected",
    "write_timeouts": "viewer_write_timeouts",
    "ended_by_stop": "viewers_ended_by_stop",
}
NOTE = (
    "timings, CPU and memory in this report include the preview's work; "
    "they are not a performance baseline"
)
PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
}
PAGE = b"""<!doctype html>
<html lang=en><head><meta charset=utf-8><title>Sentinel preview</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}p{margin:8px}</style></head>
<body><p>Live, not recorded. Boxes are those the track probe counts: #ID, confidence, C confirmed or T tentative.
One viewer at a time; at most 5 frames per second. The stream ends with the probe.</p>
<img src="stream.mjpg" alt="preview"></body></html>
"""


def valid_token(token: str | None) -> bool:
    return token is not None and TOKEN_PATTERN.match(token) is not None


@dataclass(frozen=True)
class PreviewBox:
    """One counted observation, in normalized native-image coordinates."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    confirmed: bool


@dataclass(frozen=True)
class PreviewUpdate:
    """One counted result and the image it was computed from."""

    image: Any
    width: int
    height: int
    stream_epoch: int
    frame_seq: int
    ingest_utc: datetime
    failed: bool
    boxes: tuple[PreviewBox, ...]


def preview_update(captured: CapturedFrame, result: TrackingResult) -> PreviewUpdate | None:
    """The update for a processed or failed frame; None for a skipped one (it was not counted)."""
    if result.outcome is FrameOutcome.SKIPPED:
        return None
    frame = captured.frame
    boxes = tuple(
        PreviewBox(
            track_id=person.track_id,
            x1=person.box.x1,
            y1=person.box.y1,
            x2=person.box.x2,
            y2=person.box.y2,
            confidence=float(person.detector_confidence or 0.0),
            confirmed=person.status is TrackStatus.CONFIRMED,
        )
        for person in result.persons
    )
    return PreviewUpdate(
        image=captured.image,
        width=frame.native_width,
        height=frame.native_height,
        stream_epoch=frame.stream_epoch,
        frame_seq=frame.frame_seq,
        ingest_utc=frame.ingest_utc,
        failed=result.outcome is FrameOutcome.FAILED,
        boxes=boxes,
    )


def pixel_box(box: PreviewBox, width: int, height: int) -> tuple[int, int, int, int]:
    """The box in image pixels, inside the image."""

    def at(value: float, size: int) -> int:
        return min(max(round(value * size), 0), size - 1)

    return at(box.x1, width), at(box.y1, height), at(box.x2, width), at(box.y2, height)


def box_label(box: PreviewBox) -> str:
    return f"#{box.track_id} {box.confidence:.2f} {'C' if box.confirmed else 'T'}"


def header_text(update: PreviewUpdate) -> str:
    stamp = update.ingest_utc.strftime("%H:%M:%S.%f")[:-3]
    text = f"epoch {update.stream_epoch} seq {update.frame_seq} {stamp}Z boxes {len(update.boxes)}"
    return text + " FAILED" if update.failed else text


def render_jpeg(update: PreviewUpdate) -> bytes:
    """Device renderer: the frame below a header strip, with each counted box and its label."""
    import cv2  # device only (~/onvif_env); never imported by portable code paths

    canvas = cv2.copyMakeBorder(update.image, HEADER_PX, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # a copy
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in update.boxes:
        x1, y1, x2, y2 = pixel_box(box, update.width, update.height)
        y1, y2 = y1 + HEADER_PX, y2 + HEADER_PX
        color = (0, 200, 0) if box.confirmed else (0, 215, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = box_label(box)
        (width, height), base = cv2.getTextSize(label, font, 0.5, 1)
        top = y1 - height - base - 2 if y1 - height - base - 2 >= HEADER_PX else y1 + 2
        cv2.rectangle(canvas, (x1, top), (x1 + width + 4, top + height + base + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 2, top + height + 1), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    text_color = (60, 60, 255) if update.failed else (255, 255, 255)
    cv2.putText(canvas, header_text(update), (4, HEADER_PX - 7), font, 0.5, text_color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("jpeg_encode_failed")
    return encoded.tobytes()


class PreviewFeed:
    """One-slot hand-off of the latest counted result from the probe thread to the viewer.

    publish() never waits on a viewer: it replaces the pending update. Counters
    satisfy published == shown + dropped + pending.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: PreviewUpdate | None = None
        self._closed = False
        self._counts = {"published": 0, "shown": 0, "dropped": 0}

    def observe(self, captured: CapturedFrame, result: TrackingResult) -> None:
        """TrackProbe observer: offer this counted result to the viewer."""
        update = preview_update(captured, result)
        if update is not None:
            self.publish(update)

    def publish(self, update: PreviewUpdate) -> None:
        with self._cond:
            if self._closed:
                return
            self._counts["published"] += 1
            if self._pending is not None:
                self._counts["dropped"] += 1
            self._pending = update
            self._cond.notify_all()

    def take(self, timeout_s: float) -> PreviewUpdate | None:
        """The newest update not yet shown, waiting up to ``timeout_s``; None on timeout or close."""
        with self._cond:
            self._cond.wait_for(lambda: self._pending is not None or self._closed, timeout_s)
            update, self._pending = self._pending, None
            if update is not None:
                self._counts["shown"] += 1
            return update

    def close(self) -> None:
        """Release the pending image and wake the viewer; later updates are ignored."""
        with self._cond:
            self._closed = True
            if self._pending is not None:
                self._counts["dropped"] += 1
                self._pending = None
            self._cond.notify_all()

    def counts(self) -> dict[str, int]:
        with self._cond:
            return {**self._counts, "pending": int(self._pending is not None)}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False  # server_close() would join handlers without a timeout; stop() waits with one
    preview: PreviewServer

    def handle_error(self, request: Any, client_address: Any) -> None:  # no traceback output
        self.preview._count("handler_errors")


class PreviewServer:
    """Serves the preview on 127.0.0.1:``port`` from its own thread; ``port`` 0 picks a free port (tests)."""

    def __init__(
        self,
        port: int,
        feed: PreviewFeed,
        token: str,
        *,
        host: str = LOOPBACK_HOST,
        render: Callable[[PreviewUpdate], bytes] = render_jpeg,
        max_fps: float = MAX_FPS,
        write_timeout_s: float = WRITE_TIMEOUT_S,
        send_buffer_bytes: int = SEND_BUFFER_BYTES,
    ) -> None:
        if host != LOOPBACK_HOST:
            raise ValueError(f"the preview binds {LOOPBACK_HOST} only")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if not valid_token(token):
            raise ValueError("the token must be 22-128 characters of A-Z, a-z, 0-9, '-' and '_'")
        if not 0 < max_fps <= MAX_FPS:
            raise ValueError(f"max_fps must be above 0 and at most {MAX_FPS}")
        if not 0 < write_timeout_s <= 30:
            raise ValueError("write_timeout_s must be above 0 and at most 30")
        self._feed = feed
        self._token = token.encode("ascii")
        self._render = render
        self._interval_s = 1.0 / max_fps
        self._max_fps = max_fps
        self._send_buffer = send_buffer_bytes
        self._stop = threading.Event()
        self._lock = threading.Condition()
        self._viewer = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._active = 0
        self._render_ns: list[int] = []
        self._counts = {
            "page": 0, "viewers_served": 0, "viewers_refused_busy": 0, "viewers_disconnected": 0,
            "viewer_write_timeouts": 0, "viewers_ended_by_stop": 0, "refused_host": 0, "not_found": 0,
            "refused_method": 0, "handler_errors": 0, "render_failures": 0, "frames_sent": 0, "bytes_sent": 0,
        }
        handler = type("PreviewHandler", (_Handler,), {"preview": self, "timeout": write_timeout_s})
        self._httpd = _Server((LOOPBACK_HOST, port), handler)
        self._httpd.preview = self
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the preview can be started once")
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": WAIT_S}, name="sentinel-preview", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float) -> bool:
        """Stop serving, end the viewer and close every socket; True if nothing is left running."""
        self._stop.set()
        self._feed.close()
        with self._lock:
            sockets = list(self._connections)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # unblocks a stalled write or a request that never arrives
            except OSError:
                pass
        deadline = time.monotonic() + timeout_s
        if self._thread is not None:
            stopper = threading.Thread(target=self._httpd.shutdown, daemon=True)
            stopper.start()
            stopper.join(timeout_s)
            self._thread.join(max(0.0, deadline - time.monotonic()))
        self._httpd.server_close()
        with self._lock:
            self._lock.wait_for(lambda: self._active == 0, max(0.0, deadline - time.monotonic()))
            self._closed = self._active == 0 and (self._thread is None or not self._thread.is_alive())
            return self._closed

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            render = percentiles_ms(self._render_ns)
            closed, active = self._closed, self._active
        return {
            "enabled": True,
            "bind": LOOPBACK_HOST,
            "port": self.address[1],
            "max_fps": self._max_fps,
            "updates": self._feed.counts(),
            "viewers": {name: counts[key] for name, key in VIEWER_COUNTS.items()},
            "requests": {name: counts[name] for name in ("page", "refused_host", "not_found", "refused_method",
                                                          "handler_errors")},
            "frames_sent": counts["frames_sent"],
            "bytes_sent": counts["bytes_sent"],
            "render_failures": counts["render_failures"],
            "render_ms": render,
            "closed": closed,
            "handlers_open": active,
            "note": NOTE,
        }

    # -- used by the handler

    def _count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += amount

    def _enter(self, sock: socket.socket) -> bool:
        """Register a connection; False once stopping (the handler then closes it at once)."""
        with self._lock:
            self._active += 1
            self._connections.add(sock)
            return not self._stop.is_set()

    def _leave(self, sock: socket.socket) -> None:
        with self._lock:
            self._active -= 1
            self._connections.discard(sock)
            self._lock.notify_all()

    def _route(self, path: str) -> str | None:
        """'page' or 'stream' for the token's two paths, else None."""
        if not path.startswith("/"):
            return None
        token, slash, rest = path[1:].partition("/")
        if not slash or not hmac.compare_digest(token.encode("utf-8", "replace"), self._token):
            return None
        return {"": "page", "stream.mjpg": "stream"}.get(rest)

    def _stream(self, handler: _Handler) -> None:
        if not self._viewer.acquire(blocking=False):
            self._count("viewers_refused_busy")
            handler.send_plain(503, b"another viewer is connected\n")
            return
        try:
            self._count("viewers_served")
            handler.send_response(200)
            handler.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            for name, value in PAGE_HEADERS.items():
                handler.send_header(name, value)
            handler.end_headers()
            self._count(self._send_frames(handler))
        except TimeoutError:
            self._count("viewer_write_timeouts")
        except OSError:
            self._count("viewers_ended_by_stop" if self._stop.is_set() else "viewers_disconnected")
        finally:
            self._viewer.release()

    def _send_frames(self, handler: _Handler) -> str:
        """Write frames until the stop; returns the counter for how the stream ended."""
        next_send = 0.0
        while not self._stop.is_set():
            delay = next_send - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            update = self._feed.take(WAIT_S)
            if update is None:
                continue
            next_send = time.monotonic() + self._interval_s
            started = time.perf_counter_ns()
            try:
                jpeg = self._render(update)
            except Exception:  # noqa: BLE001 - counted; never shown
                self._count("render_failures")
                continue
            finally:
                del update  # the viewer holds no image while it writes
            elapsed = time.perf_counter_ns() - started
            with self._lock:
                if len(self._render_ns) < MAX_SAMPLES:
                    self._render_ns.append(elapsed)
            head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
            handler.wfile.write(head + jpeg + b"\r\n")
            with self._lock:
                self._counts["frames_sent"] += 1
                self._counts["bytes_sent"] += len(head) + len(jpeg) + 2
        return "viewers_ended_by_stop"


class _Handler(BaseHTTPRequestHandler):
    preview: PreviewServer
    server_version = "sentinel-preview"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.preview._send_buffer)
        except OSError:
            pass

    def handle(self) -> None:
        try:
            if self.preview._enter(self.connection):
                super().handle()
        finally:
            self.preview._leave(self.connection)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        port = self.server.server_address[1]
        if self.headers.get("Host", "") not in (f"{LOOPBACK_HOST}:{port}", f"localhost:{port}"):
            self.preview._count("refused_host")
            self.send_plain(421, b"misdirected request\n")
            return
        route = self.preview._route(urlsplit(self.path).path)
        if route is None:
            self.preview._count("not_found")
            self.send_plain(404, b"not found\n")
        elif route == "page":
            self.preview._count("page")
            self._send(200, "text/html; charset=utf-8", PAGE)
        else:
            self.preview._stream(self)

    def _refuse(self) -> None:
        self.preview._count("refused_method")
        self._send(405, "text/plain; charset=utf-8", b"GET only\n", allow="GET")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse  # noqa: N815

    def send_plain(self, code: int, body: bytes) -> None:
        self._send(code, "text/plain; charset=utf-8", body)

    def _send(self, code: int, kind: str, body: bytes, *, allow: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in PAGE_HEADERS.items():
            self.send_header(name, value)
        if allow is not None:
            self.send_header("Allow", allow)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - request lines hold the token
        return
