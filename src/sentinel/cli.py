"""``sentinel`` command line (guide chapter 18). Only implemented commands exist."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .adapters import resolve
from .config import CaptureConfig, ConfigError, SentinelConfig, load_config
from .media.capture import CaptureWorker, LatestFrame, SourceError, VideoSource
from .media.clock import SystemClock
from .media.frames import FrameStamper
from .media.probe import run_probe
from .tracking.probe import TrackProbe, read_meminfo
from .tracking.tracker import PersonTracker, TrackerError

PROBE_MAX_S = 300.0
GB = 1_000_000_000


def _seconds(text: str) -> float:
    value = float(text)
    if not 1.0 <= value <= PROBE_MAX_S:
        raise argparse.ArgumentTypeError(f"must be between 1 and {PROBE_MAX_S:g}")
    return value


def _min_free_gb(text: str) -> float:
    value = float(text)
    if not 0.5 <= value <= 7.0:
        raise argparse.ArgumentTypeError("must be between 0.5 and 7")
    return value


def _live_source(config: CaptureConfig) -> VideoSource:
    from .media.opencv_source import OpenCvSource

    return OpenCvSource.from_environment(config)


def _legacy_tracker(engine: Path) -> Any:
    from .inference.legacy_ultralytics import LegacyUltralyticsTracker

    return LegacyUltralyticsTracker(engine)


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_source: Callable[[CaptureConfig], VideoSource] = _live_source,
    tracker_backend: Callable[[Path], Any] = _legacy_tracker,
    meminfo: Callable[[], dict[str, int] | None] = read_meminfo,
) -> int:
    parser = argparse.ArgumentParser(prog="sentinel")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("config", help="configuration tools")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="validate a configuration file")
    validate.add_argument("path", help="YAML configuration file")
    capture_parser = commands.add_parser("capture", help="camera ingest tools")
    capture_commands = capture_parser.add_subparsers(dest="capture_command", required=True)
    probe = capture_commands.add_parser(
        "probe",
        help="read the camera (SENTINEL_RTSP_URL) for a while and print a numbers-only JSON summary",
    )
    probe.add_argument("path", help="YAML configuration file")
    probe.add_argument("--seconds", type=_seconds, default=30.0, help="1-300, default 30")
    track_parser = commands.add_parser("track", help="person detection and tracking tools")
    track_commands = track_parser.add_subparsers(dest="track_command", required=True)
    track_probe = track_commands.add_parser(
        "probe",
        help="run the legacy detector + ByteTrack on the camera (SENTINEL_RTSP_URL) for a while and "
        "print a numbers-only JSON summary; needs the GPU with L4T's libcuda preloaded (D27)",
    )
    track_probe.add_argument("path", help="YAML configuration file")
    track_probe.add_argument("--engine", type=Path, required=True, help="the profiled yolov8n.engine")
    track_probe.add_argument("--seconds", type=_seconds, default=30.0, help="1-300, default 30")
    track_probe.add_argument(
        "--min-free-gb", type=_min_free_gb, default=1.5,
        help="refuse to load the model below this MemFree (decimal GB; default 1.5; see U18)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.path)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.command == "capture":
        return _capture_probe(config, args.seconds, capture_source)
    if args.command == "track":
        return _track_probe(
            config, args.seconds, args.engine, round(args.min_free_gb * GB), capture_source, tracker_backend, meminfo
        )
    print(
        f"{args.path}: valid Sentinel configuration "
        f"(version {config.config_version}, camera {config.camera.id})"
    )
    # Optional adapters never block startup; show why any cannot be used.
    statuses = resolve(config.adapters)
    if not statuses:
        print("  no optional adapters configured: core monitoring only")
    for status in statuses:
        print(f"  adapter {status.manifest.adapter_id}: {status.state.value} ({status.reason})")
    for zone in config.zones:
        if zone.schedule is None:
            when = "always active"
        else:
            windows = ", ".join(f"{w.start}-{w.end}" for w in zone.schedule.windows)
            when = f"{windows} {zone.schedule.timezone}"
        state = "enabled" if zone.enabled else "disabled"
        print(f"  zone {zone.zone_id}: {zone.rule}, {state}, {zone.severity}, {when}")
    return 0


def _capture_probe(
    config: SentinelConfig, seconds: float, capture_source: Callable[[CaptureConfig], VideoSource]
) -> int:
    capture = config.capture
    try:
        source = capture_source(capture)
    except SourceError as exc:
        print(f"capture probe: {exc.reason}", file=sys.stderr)
        return 1
    clock = SystemClock()
    slot = LatestFrame()
    worker = CaptureWorker(source, FrameStamper(config.camera.id, clock), slot, capture)
    summary = run_probe(
        worker,
        slot,
        clock,
        seconds,
        endpoint=getattr(source, "endpoint", None),
        stop_timeout_s=capture.open_timeout_s + capture.read_timeout_s + 1.0,
    )
    summary["settings"] = capture.model_dump()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "frames_received" and summary["worker"]["stopped"] else 1


def _track_probe(
    config: SentinelConfig,
    seconds: float,
    engine: Path,
    min_free_bytes: int,
    capture_source: Callable[[CaptureConfig], VideoSource],
    tracker_backend: Callable[[Path], Any],
    meminfo: Callable[[], dict[str, int] | None],
) -> int:
    capture = config.capture
    try:
        source = capture_source(capture)  # checks the URL before any model is loaded
    except SourceError as exc:
        print(f"track probe: {exc.reason}", file=sys.stderr)
        return 1
    # GPU allocations failed beyond MemFree on this device (U18); a provisional probe guard, not a policy.
    before = meminfo()
    if before is not None and before["MemFree"] < min_free_bytes:
        refusal = {"probe": "track", "status": "refused", "reason": "memfree_below_minimum",
                   "memory_before": before, "min_free_bytes": min_free_bytes}
        print(json.dumps(refusal, indent=2))
        return 1
    backend = tracker_backend(engine)
    started = time.monotonic()
    try:
        backend.load()
    except TrackerError as exc:
        detail = f" ({exc.error_type})" if exc.error_type else ""
        print(f"track probe: {exc.label}{detail}", file=sys.stderr)
        return 1
    load_s = time.monotonic() - started
    after = meminfo()
    clock = SystemClock()
    slot = LatestFrame()
    worker = CaptureWorker(source, FrameStamper(config.camera.id, clock), slot, capture)
    summary = run_probe(
        worker,
        slot,
        clock,
        seconds,
        endpoint=getattr(source, "endpoint", None),
        stop_timeout_s=capture.open_timeout_s + capture.read_timeout_s + 1.0,
        consumer=TrackProbe(PersonTracker(backend), clock),
    )
    summary["load"] = {"seconds": round(load_s, 2), "memory_before": before, "memory_after": after,
                       "min_free_bytes": min_free_bytes}
    summary["settings"] = capture.model_dump()
    print(json.dumps(summary, indent=2))
    tracking = summary["tracking"]
    return 0 if tracking["processed"] and not tracking["failed"] and summary["worker"]["stopped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
