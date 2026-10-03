"""``sentinel`` command line (guide chapter 18). Only implemented commands exist."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

from . import __version__
from .adapters import resolve
from .config import CaptureConfig, ConfigError, SentinelConfig, load_config
from .media.capture import CaptureWorker, LatestFrame, SourceError, VideoSource
from .media.clock import SystemClock
from .media.frames import FrameStamper
from .media.probe import run_probe

PROBE_MAX_S = 300.0


def _seconds(text: str) -> float:
    value = float(text)
    if not 1.0 <= value <= PROBE_MAX_S:
        raise argparse.ArgumentTypeError(f"must be between 1 and {PROBE_MAX_S:g}")
    return value


def _live_source(config: CaptureConfig) -> VideoSource:
    from .media.opencv_source import OpenCvSource

    return OpenCvSource.from_environment(config)


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_source: Callable[[CaptureConfig], VideoSource] = _live_source,
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
    args = parser.parse_args(argv)

    try:
        config = load_config(args.path)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.command == "capture":
        return _capture_probe(config, args.seconds, capture_source)
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


if __name__ == "__main__":
    raise SystemExit(main())
