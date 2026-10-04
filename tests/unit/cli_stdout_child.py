"""Child process for test_cli_stdout.py: one `sentinel` command with noisy fake devices (no camera, no GPU).

Usage: cli_stdout_child.py SCENARIO WORK_DIR. The fake model load writes to stdout
the four ways libraries do: print, a logging handler bound to sys.stdout before
the command (as Ultralytics binds its handler at import), os.write on descriptor 1,
and libc's buffered stdio (as a native logger may). Before and after the command
the child writes its own markers to stdout. It saves descriptor checks to
WORK_DIR/checks.json and exits with main()'s code, or 3 if main() raised.

The track_preview scenarios (P1) also run a viewer thread against the preview
and record what it received, the token and whether the port was closed after.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from sentinel.cli import main
from sentinel.contracts import PixelFormat
from sentinel.demo_runtime import Devices
from sentinel.media.capture import DecodedFrame
from sentinel.tracking.tracker import RawTrack, TrackerError

LIBC = ctypes.CDLL(None)
LIBRARY = logging.getLogger("fake_library")
LIBRARY.addHandler(logging.StreamHandler(sys.stdout))
LIBRARY.propagate = False
IMAGE = SimpleNamespace(shape=(480, 640, 3))
PLENTY = {"MemFree": 4_000_000_000, "MemAvailable": 5_000_000_000}
LOW = {"MemFree": 1_000_000_000, "MemAvailable": 2_000_000_000}
TOKEN = "Child_token-0123456789abcdef"  # synthetic
PREVIEW: dict[str, object] = {}


def noise(tag: str) -> None:
    print(f"NOISE-print-{tag}")
    LIBRARY.warning("NOISE-logging-%s", tag)
    os.write(1, f"NOISE-oswrite-{tag}\n".encode())
    LIBC.puts(f"NOISE-native-{tag}".encode())  # stays in libc's buffer: stdout is a pipe


class Source:
    endpoint = None

    def __init__(self, stop_after: int = 0, sig: signal.Signals = signal.SIGTERM) -> None:
        self.stop_after = stop_after
        self.sig = sig
        self.reads = 0

    def open(self) -> None:
        pass

    def read(self) -> DecodedFrame:
        self.reads += 1
        time.sleep(0.01)
        if self.reads == self.stop_after:
            os.kill(os.getpid(), self.sig)
        return DecodedFrame(IMAGE, 640, 480, PixelFormat.BGR, self.reads * 66_667)

    def close(self) -> None:
        pass


class Backend:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def load(self) -> None:
        noise("load")
        if self.failure is not None:
            raise self.failure

    def track(self, image: object) -> list[RawTrack]:
        self.calls += 1
        if self.calls == 1:
            noise("track")
        return [RawTrack(1, 64, 48, 320, 480, 0.8)]

    def reset(self) -> None:
        pass


def command(scenario: str, work: Path) -> int:
    failures: dict[str, BaseException] = {
        "load_failed": TrackerError("load_failed", "RuntimeError"),
        "raises": RuntimeError("unexpected"),
        "interrupted": KeyboardInterrupt(),
    }
    kind, _, variant = scenario.partition("_")
    backend = Backend(failures.get(variant))
    if variant in ("preview", "previewsigint"):
        return preview_command(work, backend, interrupt=variant == "previewsigint")
    if kind == "track":
        return main(
            ["track", "probe", str(work / "config.yaml"), "--engine", str(work / "x.engine"), "--seconds", "1"],
            capture_source=lambda config: Source(),
            tracker_backend=lambda engine: backend,
            meminfo=lambda: LOW if variant == "memfree" else PLENTY,
        )
    devices = Devices(
        capture_source=lambda config: Source(stop_after=30),
        tracker_backend=lambda engine: backend,
        scene_server=lambda options, port: None,
        scene_request=lambda port, timeout: None,
        meminfo=lambda: PLENTY,
        notifiers=lambda config: ({}, {}),
    )
    return main(
        ["run", str(work / "config.yaml"), "--data-dir", str(work / "data"), "--engine", str(work / "x.engine"),
         "--status-interval-s", "1", "--status-port", "0"],
        devices=devices,
    )


def viewer(port: int) -> None:
    """Read the page and two stream parts, as a browser would; record what came back."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and "parts" not in PREVIEW:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(f"GET /{TOKEN}/stream.mjpg HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
                with sock.makefile("rb") as stream:
                    PREVIEW["status"] = stream.readline().split()[1].decode()
                    while stream.readline() not in (b"\r\n", b""):
                        pass
                    parts = []
                    while len(parts) < 2 and stream.readline() == b"--frame\r\n":
                        length = 0
                        while (line := stream.readline()) not in (b"\r\n", b""):
                            if line.lower().startswith(b"content-length:"):
                                length = int(line.split(b":")[1])
                        parts.append(stream.read(length).decode())
                        stream.read(2)
                    PREVIEW["parts"] = parts
        except OSError:
            time.sleep(0.05)


def preview_command(work: Path, backend: Backend, *, interrupt: bool) -> int:
    os.environ["SENTINEL_PREVIEW_TOKEN"] = TOKEN
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    PREVIEW["token"] = TOKEN
    thread = threading.Thread(target=viewer, args=(port,), daemon=True)
    thread.start()
    source = Source(stop_after=40, sig=signal.SIGINT) if interrupt else Source()
    try:
        return main(
            ["track", "probe", str(work / "config.yaml"), "--engine", str(work / "x.engine"),
             "--seconds", "30" if interrupt else "2", "--preview", "--box-summary"],
            capture_source=lambda config: source,
            tracker_backend=lambda engine: backend,
            meminfo=lambda: PLENTY,
            preview_port=port,
            preview_render=lambda update: f"frame-{update.frame_seq}".encode(),
        )
    finally:
        thread.join(5)
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            PREVIEW["port_closed_after"] = False
        except ConnectionRefusedError:
            PREVIEW["port_closed_after"] = True


def descriptors() -> dict[str, object]:
    def identity(fd: int) -> list[int]:
        st = os.fstat(fd)
        return [st.st_dev, st.st_ino]

    return {"open": sorted(os.listdir("/dev/fd")), "stdout": identity(1), "stderr": identity(2)}


def child(scenario: str, work: Path) -> int:
    (work / "config.yaml").write_text("config_version: 1\ncamera: {id: cam-1}\n")
    print("BEFORE-print")  # still buffered: the switch must flush it to the real stdout first
    LIBC.puts(b"BEFORE-native")
    before = descriptors()
    raised = None
    try:
        code = command(scenario, work)
    except BaseException as exc:  # noqa: BLE001 - the test checks what happens to stdout on any exit
        code, raised = 3, type(exc).__name__
    after = descriptors()
    (work / "checks.json").write_text(json.dumps({"before": before, "after": after, "raised": raised,
                                                  "preview": PREVIEW}))
    print("AFTER-print")
    os.write(1, b"AFTER-oswrite\n")
    LIBC.puts(b"AFTER-native")
    return code


if __name__ == "__main__":
    sys.exit(child(sys.argv[1], Path(sys.argv[2])))
