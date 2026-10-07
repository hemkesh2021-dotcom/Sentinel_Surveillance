"""``sentinel`` command line (guide chapter 18). Only implemented commands exist."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TextIO

from . import __version__
from .adapters import resolve
from .config import CaptureConfig, ConfigError, SentinelConfig, load_config
from .demo_runtime import DATABASE_NAME, Devices, FaceOptions, RunOptions, SceneOptions, StartupRefused, assemble
from .identity.gallery import DEFAULT_IDENTITY_DIR
from .identity.vault import (
    MIN_NEW_SECRET_CHARS,
    SECRET_ENV,
    Secret,
    VaultError,
    prompt_secret,
    take_secret_from_environment,
)
from .media.capture import CaptureWorker, LatestFrame, SourceError, VideoSource
from .media.clock import SystemClock
from .media.frames import FrameStamper
from .memory_policy import policy_from_flags
from .media.probe import run_probe
from .tracking.probe import TrackProbe, read_meminfo
from .tracking.tracker import PersonTracker, TrackerError

PROBE_MAX_S = 300.0
GB = 1_000_000_000
PREVIEW_STOP_TIMEOUT_S = 3.0
HOME = Path.home()


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


def _scene_server(options: SceneOptions, port: int) -> Any:
    from .scene.server import LlamaServerProcess

    return LlamaServerProcess(options.binary, options.model, options.mmproj, port,
                              require_system_thp=options.require_system_thp)


def _face_backend(weights_dir: Path | None) -> Any:
    from .identity.legacy_deepface import LegacyDeepFaceBackend

    return LegacyDeepFaceBackend(weights_dir)


def _identity_store(directory: Path) -> Any:
    from .identity.gallery import IdentityStore
    from .identity.vault import Sealer

    return IdentityStore(directory, Sealer())


def _scene_request(port: int, timeout_s: float) -> Any:
    from .scene.llama_server import LlamaSceneRequest, LoopbackTransport

    return LlamaSceneRequest(LoopbackTransport(port, timeout_s))


def _status_port(text: str) -> int:
    value = int(text)
    if value != 0 and not 1024 <= value <= 65535:
        raise argparse.ArgumentTypeError("must be 0 (no status page) or between 1024 and 65535")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not 1.0 <= value <= 3600.0:
        raise argparse.ArgumentTypeError("must be between 1 and 3600")
    return value


def _flush_streams(*streams: TextIO) -> None:
    """Flush Python's streams and libc's stdio buffers; a failing flush never stops the caller's restore."""
    for stream in (*streams, sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()
    with contextlib.suppress(AttributeError, OSError):
        ctypes.CDLL(None).fflush(None)  # native writers such as TensorRT's logger buffer in libc


@contextlib.contextmanager
def _json_stdout() -> Iterator[TextIO]:
    """Keep stdout for this command's JSON; whatever else is written to it goes to stderr meanwhile.

    Loading the legacy model writes diagnostics to stdout: Ultralytics' logger (a
    handler bound to sys.stdout at import) and TensorRT's logger, possibly from
    native code. That made the track probe's stdout invalid JSON on the device
    (checklist step 3, session 21). While the command runs, descriptor 1 points
    at stderr for the whole process (every thread, library, handler and child
    that inherits it), and the command writes its JSON to a private duplicate of
    the original stdout. Buffers are flushed before the switch and before the
    restore; the restore and descriptor cleanup run in finally blocks, so an
    exception or Ctrl-C leaves stdout as it was. Logging handlers are not touched.
    Without real descriptors 1 and 2 behind sys.stdout and sys.stderr (in-process
    tests), nothing is redirected.
    """
    try:
        redirect = sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
    except (AttributeError, OSError, ValueError):
        redirect = False
    if not redirect:
        yield sys.stdout
        return
    _flush_streams()
    saved = os.dup(1)
    try:
        out = open(saved, "w", encoding="utf-8", closefd=False)
        try:
            os.dup2(2, 1)
            try:
                yield out
            finally:
                _flush_streams(out)
                os.dup2(saved, 1)
        finally:
            with contextlib.suppress(OSError, ValueError):
                out.close()
    finally:
        os.close(saved)


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_source: Callable[[CaptureConfig], VideoSource] = _live_source,
    tracker_backend: Callable[[Path], Any] = _legacy_tracker,
    meminfo: Callable[[], dict[str, int] | None] = read_meminfo,
    devices: Devices | None = None,
    face_backend: Callable[[Path | None], Any] = _face_backend,
    identity_store: Callable[[Path], Any] = _identity_store,
    preview_port: int | None = None,
    preview_render: Callable[[Any], bytes] | None = None,
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
    track_probe.add_argument(
        "--preview", action="store_true",
        help="operator-only live preview on 127.0.0.1:18091 (P1): the counted boxes on their frames with track ID, "
        "confidence and C/T, at most 5 frames/s; path token from SENTINEL_PREVIEW_TOKEN; saves nothing. "
        "Reach it over SSH: ssh -L 18091:127.0.0.1:18091 <device>",
    )
    track_probe.add_argument(
        "--box-summary", action="store_true",
        help="add track_boxes to the JSON: numbers-only box statistics per track, at most 64 tracks",
    )
    track_probe.add_argument(
        "--multi-person-frames", action="store_true",
        help="add multi_person_frames to the JSON: numbers-only records of frames with two or more persons "
        "(frame identity, times, track IDs, boxes, confidences, pairwise overlap), at most 64 records",
    )
    identity_parser = commands.add_parser(
        "identity", help="the sealed identity gallery (V2-25 demo form): consented enrollment, list, revoke, purge")
    identity_commands = identity_parser.add_subparsers(dest="identity_command", required=True)
    enroll_parser = identity_commands.add_parser(
        "enroll",
        help="enroll one consenting person from 2-8 photos in a folder inside <identity dir>/inbox/ (one face each); "
        f"the gallery passphrase comes from {SECRET_ENV} or a hidden prompt; the photos are deleted afterwards",
    )
    enroll_parser.add_argument("path", help="YAML configuration file (its identity section)")
    enroll_parser.add_argument("photos", type=Path, help="the photo folder, inside <identity dir>/inbox/")
    enroll_parser.add_argument("--consent-confirmed", action="store_true",
                               help="the person explicitly consented to this enrollment (required)")
    enroll_parser.add_argument("--consent-date", required=True, help="the date of that consent, YYYY-MM-DD")
    enroll_parser.add_argument("--dry-run", action="store_true", help="report each photo; write and delete nothing")
    list_parser = identity_commands.add_parser("list", help="opaque IDs, prototype counts and dates (no names)")
    list_parser.add_argument("path", help="YAML configuration file")
    revoke_parser = identity_commands.add_parser("revoke", help="remove one identity from the gallery")
    revoke_parser.add_argument("path", help="YAML configuration file")
    revoke_parser.add_argument("identity_id", help="the opaque ID (idn-...)")
    purge_parser = identity_commands.add_parser("purge", help="delete the gallery and the inbox")
    purge_parser.add_argument("path", help="YAML configuration file")
    purge_parser.add_argument("--yes", action="store_true", help="required: confirms the deletion")
    for sub_parser in (enroll_parser, list_parser, revoke_parser, purge_parser):
        sub_parser.add_argument("--identity-dir", type=Path, default=DEFAULT_IDENTITY_DIR,
                                help=f"the private (0700) identity directory (default {DEFAULT_IDENTITY_DIR})")
    run = commands.add_parser(
        "run",
        help="run the demo runtime (D-1): camera (SENTINEL_RTSP_URL) -> detector/tracker -> rules -> "
        "incidents -> outbox; needs the GPU with L4T's libcuda preloaded (D27). Stop with Ctrl-C or SIGTERM",
    )
    run.add_argument("path", help="YAML configuration file")
    run.add_argument("--data-dir", type=Path, required=True, help="directory for the incident database")
    run.add_argument("--engine", type=Path, required=True, help="the profiled yolov8n.engine")
    run.add_argument(
        "--min-free-gb", type=_min_free_gb, default=1.5,
        help="do not load the detector below this MemFree (decimal GB; default 1.5; provisional, see U18)",
    )
    run.add_argument(
        "--scene", action="store_true",
        help="also start the scene server and scene analysis; off by default, and refused unless the "
        "configuration lists an enabled, admitted llama-lfm2-vl-scene adapter",
    )
    run.add_argument("--llama-server", type=Path, default=HOME / "llama.cpp/build/bin/llama-server")
    run.add_argument("--scene-model", type=Path, default=HOME / "models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf")
    run.add_argument("--scene-mmproj", type=Path, default=HOME / "models/lfm2-vl/mmproj-LFM2-VL-1.6B-Q8_0.gguf")
    run.add_argument("--scene-ready-timeout-s", type=_positive_seconds, default=180.0)
    run.add_argument(
        "--scene-min-free-gb", type=_min_free_gb, default=3.0,
        help="do not start the scene server below this MemFree (decimal GB; default 3.0, as check 8)",
    )
    run.add_argument(
        "--workload-thp-disable", action="store_true",
        help="D58 memory policy (off by default): after any scene server has started and before the detector "
        "loads, disable transparent huge pages for this process alone (prctl), verified; the scene server keeps the "
        "system setting; startup is refused if it cannot be verified. With --scene, the profile must have measured it",
    )
    run.add_argument(
        "--post-load-release", action="store_true",
        help="D58 memory policy (off by default): after each model loads and settles (15 s), release its model files "
        "from the page cache (posix_fadvise DONTNEED); startup is refused unless every call returns 0. With --scene, "
        "the profile must have measured it",
    )
    run.add_argument(
        "--face", action="store_true",
        help="V2-25 demo form (off by default): 1 Hz face recognition against the sealed gallery in --identity-dir, "
        f"opened with the passphrase in {SECRET_ENV} (typed silently beforehand; removed from this process's "
        "environment at once). Needs an enabled, admitted legacy-deepface-facenet512-yunet adapter",
    )
    run.add_argument(
        "--face-validation", action="store_true",
        help="with --face: allow a face admission that is PENDING_VALIDATION, for the guarded device validation only",
    )
    run.add_argument("--identity-dir", type=Path, default=DEFAULT_IDENTITY_DIR,
                     help=f"the private (0700) identity directory (default {DEFAULT_IDENTITY_DIR})")
    run.add_argument(
        "--status-port", type=_status_port, default=18090,
        help="read-only status page on 127.0.0.1:PORT (D-2; default 18090; 0 = none). Reach it over SSH: "
        "ssh -L 18090:127.0.0.1:18090 <device>",
    )
    run.add_argument("--status-interval-s", type=_positive_seconds, default=30.0,
                     help="print a numbers-only status line this often (default 30)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.path)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.command == "capture":
        return _capture_probe(config, args.seconds, capture_source)
    if args.command == "track":
        with _json_stdout() as out:
            return _track_probe(
                config, args.seconds, args.engine, round(args.min_free_gb * GB), capture_source, tracker_backend,
                meminfo, out, preview=args.preview, box_summary=args.box_summary,
                multi_person_frames=args.multi_person_frames, preview_port=preview_port, preview_render=preview_render,
            )
    if args.command == "identity":
        with _json_stdout() as out:
            return _identity(config, args, face_backend, identity_store, out)
    if args.command == "run":
        devices = devices or Devices(
            capture_source=capture_source,
            tracker_backend=tracker_backend,
            scene_server=_scene_server,
            scene_request=_scene_request,
            meminfo=meminfo,
            face_backend=_face_backend,
            identity_store=_identity_store,
        )
        with _json_stdout() as out:
            return _run(config, args, devices, out)
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
    out: TextIO,
    *,
    preview: bool = False,
    box_summary: bool = False,
    multi_person_frames: bool = False,
    preview_port: int | None = None,
    preview_render: Callable[[Any], bytes] | None = None,
) -> int:
    capture = config.capture
    try:
        source = capture_source(capture)  # checks the URL before any model is loaded
    except SourceError as exc:
        print(f"track probe: {exc.reason}", file=sys.stderr)
        return 1
    token = None
    if preview:
        from .tracking.preview import TOKEN_ENV, valid_token

        token = os.environ.get(TOKEN_ENV)
        if not valid_token(token):  # never printed: it is the preview's access secret
            print(f"track probe: preview_token_{'invalid' if token else 'missing'}", file=sys.stderr)
            return 1
    # GPU allocations failed beyond MemFree on this device (U18); a provisional probe guard, not a policy.
    before = meminfo()
    if before is not None and before["MemFree"] < min_free_bytes:
        refusal = {"probe": "track", "status": "refused", "reason": "memfree_below_minimum",
                   "memory_before": before, "min_free_bytes": min_free_bytes}
        print(json.dumps(refusal, indent=2), file=out, flush=True)
        return 1
    observers: list[Callable[[Any, Any], None]] = []
    boxes = None
    if box_summary:
        from .tracking.box_summary import BoxSummary

        boxes = BoxSummary()
        observers.append(boxes.observe)
    multi = None
    if multi_person_frames:
        from .tracking.multi_person import MultiPersonFrames

        multi = MultiPersonFrames()
        observers.append(multi.observe)
    server = None
    stop: threading.Event | None = None
    handlers: dict[int, Any] = {}
    summary: dict[str, Any] | None = None
    try:
        if preview:
            # Bound before the model loads, so a busy port refuses fast; it stops in the finally below on any exit.
            from .tracking.preview import PREVIEW_PORT, PreviewFeed, PreviewServer, render_jpeg

            feed = PreviewFeed()
            try:
                server = PreviewServer(PREVIEW_PORT if preview_port is None else preview_port, feed, token or "",
                                       render=preview_render or render_jpeg)
            except OSError as exc:
                print(f"track probe: preview_port_unavailable:{type(exc).__name__}", file=sys.stderr)
                return 1
            observers.append(feed.observe)
            stop = threading.Event()
            handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
            server.start()
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
        if stop is None or not stop.is_set():
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
                consumer=TrackProbe(PersonTracker(backend), clock, observers),
                stop=stop,
            )
    finally:
        if server is not None:
            server.stop(PREVIEW_STOP_TIMEOUT_S)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    interrupted = stop is not None and stop.is_set()
    if summary is None:  # a signal arrived during the model load: the camera was never opened
        stopped = {"probe": "track", "status": "interrupted", "reason": "signal_before_capture",
                   "preview": None if server is None else {**server.summary(), "ended_by": "signal"}}
        print(json.dumps(stopped, indent=2), file=out, flush=True)
        return 130
    summary["load"] = {"seconds": round(load_s, 2), "memory_before": before, "memory_after": after,
                       "min_free_bytes": min_free_bytes}
    summary["settings"] = capture.model_dump()
    if boxes is not None:
        summary.update(boxes.summary())
    if multi is not None:
        summary.update(multi.summary())
    if server is not None:
        summary["preview"] = {**server.summary(), "ended_by": "signal" if interrupted else "duration"}
    print(json.dumps(summary, indent=2), file=out, flush=True)
    if interrupted:
        return 130
    tracking = summary["tracking"]
    return 0 if tracking["processed"] and not tracking["failed"] and summary["worker"]["stopped"] else 1


def _identity_secret(store: Any) -> Secret:
    """The gallery passphrase: SENTINEL_IDENTITY_PASSPHRASE (removed at once) or a hidden prompt, confirmed for a new
    gallery, which also needs MIN_NEW_SECRET_CHARS."""
    new = not store.exists()
    secret = take_secret_from_environment()
    if secret is None:
        return prompt_secret(confirm=new)
    if new and len(secret.reveal()) < MIN_NEW_SECRET_CHARS:
        raise VaultError("secret_too_short")
    return secret


def _identity(config: SentinelConfig, args: argparse.Namespace, face_backend: Callable[[Path | None], Any],
              identity_store: Callable[[Path], Any], out: TextIO) -> int:
    """``sentinel identity``: JSON on stdout (opaque IDs, counts and labels); errors as one label on stderr."""
    from .identity.enroll import EnrollmentRefused, enroll
    from .identity.legacy_deepface import FaceBackendError

    store = identity_store(args.identity_dir.expanduser())
    command = args.identity_command
    try:
        if command == "purge":
            if not args.yes:
                print("identity: purge needs --yes", file=sys.stderr)
                return 1
            print(json.dumps({"identity": "purge", **store.purge()}), file=out)
            return 0
        if command == "enroll":
            if not args.consent_confirmed:
                print("identity: consent_not_confirmed (the person must explicitly consent; pass --consent-confirmed)",
                      file=sys.stderr)
                return 1
            try:
                date.fromisoformat(args.consent_date)
            except ValueError:
                print("identity: consent_date_invalid (use YYYY-MM-DD)", file=sys.stderr)
                return 1
        if command == "list" and not store.exists():
            print(json.dumps({"identity": "list", "identities": []}), file=out)
            return 0
        secret = _identity_secret(store)
        if command == "enroll":
            report = enroll(store, face_backend(None), args.photos, secret=secret, consent_date=args.consent_date,
                            min_quality=config.identity.min_quality, match_threshold=config.identity.match_threshold,
                            dry_run=args.dry_run)
            print(json.dumps({"identity": "enroll", **report}), file=out)
            return 0
        if command == "list":
            document = store.load(secret)
            print(json.dumps({"identity": "list", "identities": [
                {"identity_id": i.identity_id, "prototypes": len(i.prototypes), "consent_date": i.consent_date,
                 "enrolled_utc": i.enrolled_utc.isoformat()} for i in document.identities]}), file=out)
            return 0
        revoked = store.revoke(args.identity_id, secret)
        print(json.dumps({"identity": "revoke", "identity_id": args.identity_id, "revoked": revoked}), file=out)
        return 0 if revoked else 1
    except EnrollmentRefused as refused:
        print(json.dumps({"identity": "enroll", "refused": refused.label, **refused.report}), file=out)
        print(f"identity: {refused.label}", file=sys.stderr)
        return 1
    except (VaultError, FaceBackendError) as exc:
        print(f"identity: {exc.label}", file=sys.stderr)
        return 1


def _run(config: SentinelConfig, args: argparse.Namespace, devices: Devices, out: TextIO) -> int:
    secret = take_secret_from_environment()  # always removed first: no child (llama-server) may inherit it
    if args.face_validation and not args.face:
        print("run: --face-validation needs --face", file=sys.stderr)
        return 1
    face = None
    if args.face:
        if secret is None:
            print(f"run: identity_secret_missing ({SECRET_ENV} is not set)", file=sys.stderr)
            return 1
        face = FaceOptions(identity_dir=args.identity_dir.expanduser(), secret=secret,
                           validation_run=args.face_validation)
    del secret
    scene = None
    if args.scene:
        scene = SceneOptions(
            binary=args.llama_server, model=args.scene_model, mmproj=args.scene_mmproj,
            ready_timeout_s=args.scene_ready_timeout_s, min_free_bytes=round(args.scene_min_free_gb * GB),
        )
    options = RunOptions(
        data_dir=args.data_dir.expanduser(), engine=args.engine, min_free_bytes=round(args.min_free_gb * GB), scene=scene,
        face=face,
        memory_policy=policy_from_flags(workload_thp_disable=args.workload_thp_disable,
                                        post_load_release=args.post_load_release),
    )
    clock = SystemClock()
    holder: dict[str, Any] = {}
    page = None
    if args.status_port:
        # Bound before any model loads: a busy port refuses fast, and the page shows "starting" meanwhile.
        from .status_page import StatusServer

        def snapshot() -> dict[str, Any]:
            runtime = holder.get("runtime")
            return runtime.snapshot() if runtime is not None else {
                "runtime": {"state": "starting", "camera_id": config.camera.id}}

        try:
            page = StatusServer(args.status_port, snapshot, options.data_dir / DATABASE_NAME)
        except OSError as exc:
            print(f"run: status_port_unavailable:{type(exc).__name__}", file=sys.stderr)
            return 1
        page.start()
    try:
        assembly = assemble(config, options, devices, clock)
    except (StartupRefused, KeyboardInterrupt) as exc:
        if page is not None:
            page.stop(2.0)
        if isinstance(exc, KeyboardInterrupt):
            print("run: interrupted during startup", file=sys.stderr)
            return 130
        print(f"run: {exc.label}", file=sys.stderr)
        return 1
    except BaseException:
        if page is not None:
            page.stop(2.0)
        raise
    runtime = assembly.runtime
    holder["runtime"] = runtime
    startup = {**assembly.startup, "status_page": None if page is None else "http://%s:%d/" % page.address}
    print(json.dumps({"run": "starting", "startup": startup}), file=out, flush=True)
    handlers = {sig: signal.signal(sig, lambda *_: runtime.request_stop()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        runtime.start()
        next_line = time.monotonic()
        while not runtime.stop_requested:
            runtime.step()
            for record in runtime.drain_identity_transitions():  # opaque IDs and numbers only
                print(json.dumps(record), file=out, flush=True)
            if time.monotonic() >= next_line:
                print(json.dumps(status_line(runtime.snapshot())), file=out, flush=True)
                next_line = time.monotonic() + args.status_interval_s
    finally:
        shutdown = runtime.shutdown()
        if page is not None:
            shutdown["stopped"]["status_page"] = page.stop(2.0)
            shutdown["all_stopped"] = all(shutdown["stopped"].values())
        closed = assembly.close(shutdown)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    for record in runtime.drain_identity_transitions():
        print(json.dumps(record), file=out, flush=True)
    print(json.dumps({"run": "stopped", "shutdown": shutdown, "database_closed": closed,
                      "status": status_line(runtime.snapshot())}), file=out, flush=True)
    return 0 if shutdown["all_stopped"] and closed else 2


def status_line(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The periodic one-line summary: numbers and fixed labels only."""
    live = snapshot.get("live") or {}
    return {
        "updated_utc": snapshot["runtime"]["updated_utc"],
        "state": snapshot["runtime"]["state"],
        "video": live.get("video"),
        "occupancy": live.get("occupancy"),
        "scene": live.get("scene"),
        "identity": live.get("identity"),
        "rates": snapshot["rates"],
        "capture": {k: snapshot["components"]["capture"][k] for k in ("state", "stream_epoch", "reconnects")},
        "pending_signals": snapshot["components"]["incidents"]["pending_signals"],
        "degraded": snapshot["degraded"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
