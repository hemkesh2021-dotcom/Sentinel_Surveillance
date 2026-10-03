"""Capture probe: what `sentinel capture probe` reports (V2-05 demo-form device check).

It runs the capture worker for a bounded time with a consumer that takes every
frame it can, and summarizes cadence, frame age at hand-off, timestamp quality,
process CPU and this host's TCP connections to the camera's RTSP endpoint. The
connection count is the demo form's evidence of a single upstream session. It
sees only this host, not other clients of the camera.

The summary holds numbers and fixed labels only: never the URL, host or frames.
Timing is ingest-based (U3): there is no capture clock, so camera-to-ingest delay
is not measured. Source PTS is diagnostic: ``pts_none_reasons`` says why frames
were stamped without a usable PTS, and ``pts_step_ms`` is the PTS step between
consecutive frames (OpenCvSource reports microseconds).
"""

from __future__ import annotations

import ipaddress
import math
import sys
import time
from pathlib import Path
from typing import Any

from ..contracts import FrameRef, SourceTimeQuality
from .capture import CapturedFrame, CaptureWorker, LatestFrame
from .clock import NS_PER_SECOND, Clock
from .opencv_source import RTSP_FFMPEG_OPTIONS, RtspEndpoint

MAX_SAMPLES = 20_000
_TCP_ESTABLISHED = "01"
LIMITATIONS = (
    "software decode (D24); ingest-based timing, camera-to-ingest delay unknown; "
    "connections counted for this host only; not hardware acceptance"
)


def percentiles_ms(values_ns: list[int]) -> dict[str, float | int | None]:
    """Nearest-rank p50/p95 and max, in milliseconds."""
    if not values_ns:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values_ns)

    def rank(q: int) -> float:
        return round(ordered[max(0, math.ceil(len(ordered) * q / 100) - 1)] / 1e6, 3)

    return {"count": len(ordered), "p50": rank(50), "p95": rank(95), "max": round(ordered[-1] / 1e6, 3)}


def established_connections(endpoint: RtspEndpoint, proc_net: Path = Path("/proc/net")) -> int | None:
    """Established TCP connections from this host to the endpoint, or None if not countable.

    Reads /proc/net/tcp and tcp6 (all processes). Needs an IP-literal host; a
    host name is not resolved here.
    """
    target = endpoint.ip
    if target is None:
        return None
    count = 0
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc_net / name).read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != _TCP_ESTABLISHED:
                continue
            remote = _proc_address(fields[2])
            if remote is not None and remote == (target, endpoint.port):
                count += 1
    return count


def _proc_address(text: str) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int] | None:
    """Decode a /proc/net/tcp{,6} 'ADDR:PORT': the address is 32-bit words in host byte order."""
    addr, _, port = text.partition(":")
    try:
        raw = b"".join(int(addr[i : i + 8], 16).to_bytes(4, sys.byteorder) for i in range(0, len(addr), 8))
        ip = ipaddress.ip_address(raw)
        number = int(port, 16)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip, number


class ProbeStats:
    """Per-frame observations of the probe's consumer, bounded to MAX_SAMPLES each."""

    def __init__(self) -> None:
        self._intervals: list[int] = []
        self._ages: list[int] = []
        self._quality = {q.value: 0 for q in SourceTimeQuality}
        self._sizes: set[tuple[int, int]] = set()
        self._formats: set[str] = set()
        self._epochs: set[tuple[str, int]] = set()
        self._last: CapturedFrame | None = None
        self._last_pts: int | None = None  # last known PTS of an unbroken run of consumed frames
        self._pts_steps: list[int] = []
        self._none_reasons = {"missing": 0, "repeated": 0, "backwards": 0, "unattributed": 0}
        self.consumed = 0

    def observe(self, captured: CapturedFrame, taken_mono_ns: int) -> None:
        frame = captured.frame
        self.consumed += 1
        self._quality[frame.source_time_quality.value] += 1
        self._sizes.add((frame.native_width, frame.native_height))
        self._formats.add(frame.pixel_format.value)
        self._epochs.add((frame.run_id, frame.stream_epoch))
        if len(self._ages) < MAX_SAMPLES:
            self._ages.append(taken_mono_ns - frame.ingest_mono_ns)
        last = self._last.frame if self._last is not None else None
        previous = None  # the frame stamped just before this one, if the probe consumed it
        if last is not None and last.stream == frame.stream and frame.frame_seq == last.frame_seq + 1:
            previous = last
        # Ingest cadence from consecutive frames only, so skipped frames do not inflate it.
        if previous is not None and len(self._intervals) < MAX_SAMPLES:
            self._intervals.append(frame.ingest_mono_ns - previous.ingest_mono_ns)
        self._observe_pts(frame, previous)
        self._last = captured

    def _observe_pts(self, frame: FrameRef, previous: FrameRef | None) -> None:
        if previous is None:
            self._last_pts = None  # the stamper compared with a frame the probe did not see
        pts, known = frame.source_pts, self._last_pts
        if frame.source_time_quality is SourceTimeQuality.NONE:
            if pts is None:
                reason = "missing"
            elif known is None:
                reason = "unattributed"
            else:
                reason = "repeated" if pts == known else "backwards"
            self._none_reasons[reason] += 1
        elif previous is not None and previous.source_pts is not None and len(self._pts_steps) < MAX_SAMPLES:
            self._pts_steps.append((frame.source_pts - previous.source_pts) * 1000)  # type: ignore[operator]
        if pts is not None:
            self._last_pts = pts

    def summary(self) -> dict[str, Any]:
        return {
            "consumed": self.consumed,
            "epochs": len(self._epochs),
            "native_sizes": sorted([w, h] for w, h in self._sizes),
            "pixel_formats": sorted(self._formats),
            "pts_quality": dict(self._quality),
            "pts_none_reasons": dict(self._none_reasons),
            "pts_step_ms": percentiles_ms(self._pts_steps),
            "ingest_interval_ms": percentiles_ms(self._intervals),
            "handoff_age_ms": percentiles_ms(self._ages),
        }


def run_probe(
    worker: CaptureWorker,
    slot: LatestFrame,
    clock: Clock,
    seconds: float,
    *,
    endpoint: RtspEndpoint | None = None,
    stop_timeout_s: float = 20.0,
    proc_net: Path = Path("/proc/net"),
) -> dict[str, Any]:
    """Run ``worker`` for ``seconds`` with a fast consumer and summarize; always stops the worker."""
    stats = ProbeStats()
    before = established_connections(endpoint, proc_net) if endpoint is not None else None
    during: list[int] = []
    start = clock.mono()
    duration_ns = round(seconds * NS_PER_SECOND)
    next_sample = start.ns + NS_PER_SECOND
    cpu_start = time.process_time()
    worker.start()
    try:
        while clock.mono().ns_since(start) < duration_ns:
            captured = slot.take(timeout_s=0.25)
            now = clock.mono()
            if captured is not None:
                stats.observe(captured, now.ns)
            if endpoint is not None and now.ns >= next_sample:
                count = established_connections(endpoint, proc_net)
                if count is not None:
                    during.append(count)
                next_sample += NS_PER_SECOND
    finally:
        stopped = worker.stop(stop_timeout_s)
    elapsed_s = clock.mono().ns_since(start) / NS_PER_SECOND
    cpu_s = time.process_time() - cpu_start
    status = worker.status()
    counts = slot.counts()
    if endpoint is None:
        connections: dict[str, Any] = {"status": "not_applicable"}
    elif before is None:
        connections = {"status": "unavailable", "reason": "host_not_ip_literal"}
    else:
        connections = {
            "status": "observed",
            "before": before,
            "samples": len(during),
            "min": min(during, default=None),
            "max": max(during, default=None),
        }
    return {
        "probe": "capture",
        "status": "frames_received" if stats.consumed else "no_frames",
        "seconds": round(elapsed_s, 3),
        "worker": {
            "state": status.state.value,
            "stopped": stopped,
            "connects": status.connects,
            "open_failures": status.open_failures,
            "stream_ends": status.stream_ends,
            "problem": status.problem,
        },
        "frames": {"captured": counts["published"], **{k: counts[k] for k in ("delivered", "replaced", "discarded")}},
        "captured_fps": round(counts["published"] / elapsed_s, 3) if elapsed_s > 0 else None,
        **stats.summary(),
        "cpu": {
            "process_s": round(cpu_s, 3),
            "core_equivalents": round(cpu_s / elapsed_s, 3) if elapsed_s > 0 else None,
        },
        "max_rss_bytes": _max_rss_bytes(),
        "upstream_connections": connections,
        "ffmpeg_options": RTSP_FFMPEG_OPTIONS if endpoint is not None else None,
        "limitations": LIMITATIONS,
    }


def _max_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # not on Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if sys.platform.startswith("linux") else peak
