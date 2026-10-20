#!/usr/bin/env python3
"""What the V2-25 device check's evidence files may contain, field by field (for the privacy counts; stdlib only).

face_privacy.py counts search terms in F1's and F2's evidence. Passphrases, photo file names, camera userinfo and
names of three or more letters or digits are searched in every byte. A shorter name (an initial, a two-letter
name) cannot be told apart from the evidence's own fixed text by spelling: one letter occurs in nearly every key.
It is told apart by structure. Each file is read against its profile here:

- every key must be a declared field of its file, or fit the declared key vocabulary of a field whose keys are data
  (memory counter names, release roles, component names); such keys are schema text;
- every value must fit its field: a type, a closed vocabulary, a structured identifier or a fixed template. Values
  that fit are schema text, except a template's variable parts, which are content;
- TEXT fields are content: the scene model's summary, problem and error labels, log messages, the account's notes;
- anything that does not fit is nonconforming: an unexpected key, value or line, a duplicate key, malformed JSON,
  undecodable bytes, a file without a profile. It is counted, and all of its text becomes content.

face_privacy.py searches only the content for short names, as whole tokens, and requires nonconforming = 0 through
face_check. The vocabularies mirror the pinned code; tests tie them to its enums and to the real runtime's output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import face_account

# Files written after the counts, or the counts themselves; never counted (face_check uses the same set).
NOT_COUNTED = frozenset({"SHA256SUMS", "check.txt", "secret-counts-camera.txt", "secret-counts-identity.txt"})
STATUS_NAMES = ("e1", "k1", "a", "e2", "k2", "end", "n1", "look", "turn", "n2",
                "c1", "c2", "c3",  # F2's, the natural check's and the comparison check's (D87)
                "seated", "turned", "out", "returned", "faceoff1", "faceoff2", "faceon", "liveoff1", "liveoff2",
                "liveon", "asked", "sceneoff1", "sceneoff2", "sceneon", "chatoff1", "chatoff2", "chaton",
                "snapoff",  # the feature-controls check's (D99, features_check)
                "i1")  # the tracking diagnostic's short integration check's (session 81; with "end")
CUE_NAMES = ("t0", "k1", "a", "e2", "k2", "stop", "look", "turn", "n2",  # F2's, the natural check's, D91's
             "leave", "back", "pause", "resume",  # D95's live dashboard check
             "away", "faceoff", "faceon", "liveoff", "liveon", "ask", "sceneoff", "askoff", "sceneon", "chatoff",
             "chaton", "snapoff", "snapon")  # the feature-controls check's (D99, features_check)
RECHECK_NAME = r"face-\d{8}T\d{6}Z-f1-[0-9a-f]{7}-\d{8}T\d{6}Z"  # ~/sentinel-runs/recheck/<run>-f1-<pin>-<UTC>
SESSION_NAME = r"exp-\d{8}T\d{6}Z"  # an enrollment-experiment session (D91)
CAPTURE_RECHECK_NAME = SESSION_NAME + r"-xc-[0-9a-f]{7}-\d{8}T\d{6}Z"  # ~/sentinel-runs/recheck/<session>-xc-<pin>-<UTC>


# ---------------------------------------------------------------- field specs


class Spec:
    def fit(self, value: Any) -> tuple[bool, list[str]]:
        """(whether the value fits this field, the parts of it that are content)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Kind(Spec):
    name: str  # int, number, bool, null

    def fit(self, value: Any) -> tuple[bool, list[str]]:
        ok = {"int": type(value) is int, "number": type(value) in (int, float), "bool": type(value) is bool,
              "null": value is None}[self.name]
        return ok, []


@dataclass(frozen=True)
class Vocab(Spec):
    """A closed vocabulary: a value in it is schema text."""

    values: frozenset[str]

    def fit(self, value: Any) -> tuple[bool, list[str]]:
        return isinstance(value, str) and value in self.values, []


@dataclass(frozen=True)
class Form(Spec):
    """A structured identifier or machine format (an opaque ID, a time, a hash), matched as a whole."""

    pattern: re.Pattern

    def fit(self, value: Any) -> tuple[bool, list[str]]:
        return isinstance(value, str) and self.pattern.fullmatch(value) is not None, []


@dataclass(frozen=True)
class Template(Spec):
    """Fixed text with variable parts: the fixed text is schema text, the groups are content."""

    patterns: tuple[re.Pattern, ...]

    def fit(self, value: Any) -> tuple[bool, list[str]]:
        if isinstance(value, str):
            for pattern in self.patterns:
                match = pattern.fullmatch(value)
                if match:
                    return True, [g for g in match.groups() if g]
        return False, []


@dataclass(frozen=True)
class Text(Spec):
    """Free text: content."""

    def fit(self, value: Any) -> tuple[bool, list[str]]:
        return isinstance(value, str), [value] if isinstance(value, str) else []


@dataclass(frozen=True)
class Either(Spec):
    options: tuple[Spec, ...]

    def fit(self, value: Any) -> tuple[bool, list[str]]:
        for option in self.options:
            ok, content = option.fit(value)
            if ok:
                return True, content
        return False, []


INT, NUM, BOOL, NULL, TEXT = Kind("int"), Kind("number"), Kind("bool"), Kind("null"), Text()


def vocab(*values: str) -> Vocab:
    return Vocab(frozenset(values))


def form(regex: str) -> Form:
    return Form(re.compile(regex))


def template(*regexes: str) -> Template:
    return Template(tuple(re.compile(r, re.DOTALL) for r in regexes))


def maybe(spec: Spec) -> Either:
    return Either((spec, NULL))


def fixed(text: str) -> str:
    """A template alternative with no variable part."""
    return re.escape(text)


IDN = form(r"idn-[0-9a-f]{12}")
DATE = form(r"\d{4}-\d{2}-\d{2}")
ISO = form(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})?")
DIGITS = form(r"-?\d+")
HEX40, HEX64 = form(r"[0-9a-f]{40}"), form(r"[0-9a-f]{64}")
UUID = form(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RUN_NAME = form(r"face-\d{8}T\d{6}Z")
LAYERS = form(r"\d+/\d+")
LOOPBACK_URL = form(r"http://127\.0\.0\.1:\d{1,5}/")
INCIDENT_ID = form(r"inc-\d{8}T\d{6}Z-[0-9a-f]{12}")
COUNT = maybe(INT)
SECONDS = maybe(NUM)

# Vocabularies of the pinned code (tests compare each with its enum or constant).
MEMORY_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "Shmem", "KReclaimable",
               "SUnreclaim", "Mlocked", "Unevictable", "CmaFree", "AnonPages", "Mapped", "Active(file)",
               "Inactive(file)", "pswpin", "pswpout")  # operator_check.MEMORY_KEYS
SAMPLE_KEYS = vocab(*MEMORY_KEYS, "t_mono", "pressure_bytes")  # operator_check.safe_sample
MEMFREE_KEYS = vocab("MemAvailable", "MemFree")  # demo_runtime.memfree_problem's reading
CAPABILITY = vocab("available", "unavailable", "disabled")  # live_state.Capability
OCCUPANCY = vocab("empty", "occupied", "unknown")  # live_state.Occupancy
SCENE_STATUS = vocab("reported", "unknown", "no_current_result")  # live_state.SceneStatus
VIDEO = vocab("starting", "fresh", "stale", "offline")  # media.health.VideoState
CAPTURE = vocab("starting", "connecting", "streaming", "waiting", "stopped", "failed")  # media.capture.CaptureState
SERVER = vocab("not_started", "starting", "ready", "failed", "exited", "stopped")  # scene.server.ServerState
RUNTIME = vocab("starting", "running", "stopping", "stopped")  # demo_runtime.RuntimeState
WORKER = vocab("not_started", "running", "stopped", "not_configured")  # worker/outbox status, or no outbox
IDENTITY_STATES = ("unresolved", "unknown", "known", "uncertain")  # identity.state.IdentityState
BASIS = vocab("fresh", "retained")  # identity.state.Basis
OWNERSHIP = ("assigned", "no_face", "no_person", "ambiguous")  # identity.association.Ownership
VOTE_LABELS = ("match", "unknown", "margin", "low_quality", "incompatible", "no_enrollment",  # state._NOTE_LABELS
               "before_break",  # D97: state.BEFORE_BREAK (continuity runs only)
               "before_expiry")  # state.BEFORE_EXPIRY: a late result from the track's earlier life
CONTINUITY_CAUSES = ("gap", "jump", "overlap")  # identity.continuity.Cause (D97)
NOT_CARRIED = tuple(f"not carried over a break in tracking ({why})" for why in (  # identity.continuity.NOT_CARRIED
    "a detection gap", "the box jumped", "another detection overlapped"))
CONTINUITY_POLICY = ("max_gap_s", "overlap_iou", "min_step_iou", "retention_s")  # ContinuityPolicy.numbers (D97)
VOTE_NOTES = ("face matched", "face matched nobody", "face matches more than one identity", "face quality too low",
              "embedding from an incompatible model", "no identities enrolled")  # state._NOTE_LABELS' keys
TICKS = vocab("offered", "skipped_no_person", "stopping")  # identity.worker TICK_*
COMPARISON_KINDS = ("full_scale_only", "both")  # identity.comparison.Kind
CAPTURE_REFUSALS = ("too_few_faces", "too_few_prototypes", "inconsistent_capture", "capture_target_has_gallery",
                    "capture_not_verified", "capture_write_failed",
                    "capture_audit_incomplete")  # identity.experiment.CAPTURE_REFUSALS (D91)
COMPARISON_DISPOSITIONS = ("completed", "failed", "production_not_applied", "skipped_one_per_result",
                           "skipped_stopping", "skipped_production_waiting", "skipped_no_landmarks",
                           "skipped_stale", "skipped_cap", "skipped_interval",
                           "skipped_quota")  # identity.comparison.Disposition
OUTCOMES = vocab("applied", "failed", "rejected_not_live", "rejected_boot", "rejected_age",
                 "ignored_face_unavailable")  # runtime.FaceResultRecord.outcome
SEVERITY = vocab("info", "warning", "critical")  # rules.scene_hazard.Severity
INCIDENT_STATUS = vocab("candidate", "open", "acknowledged", "resolved", "dismissed")  # incidents IncidentStatus
THREAT = vocab("none", "low", "medium", "high")  # scene.report.Threat
UNCERTAINTY = vocab("low", "medium", "high")  # scene.report.Uncertainty
THP = vocab("system", "workload_disabled")  # memory_policy THP labels
RELEASE = vocab("none", "post_load")  # memory_policy model-file release labels
RELEASE_RESULTS = vocab("returned_0", "returned_error", "open_failed", "unsupported")  # memory_policy
RELEASE_COMPONENTS = vocab("scene", "detector", "face")  # memory_policy.RELEASE_FILE_ROLES
RELEASE_ROLES = vocab("llm", "mmproj", "engine", "facenet512_weights.h5", "face_detection_yunet_2023mar.onnx")
STOPPED_PARTS = vocab("capture", "face_worker", "scene_server", "scene_worker", "outbox", "status_page",
                      "snapshots", "live_view",  # only with --snapshots and --live-preview (D70, D75)
                      "control_key_removed",  # D99, only with --feature-controls
                      # demo_runtime.SHUTDOWN_STEPS: only as false, when that shutdown step raised
                      "status", "face_capture", "tracking_summary", "flush")
CHANNELS = vocab("telegram")
GUARD_STATUS = Either((vocab("duration_stop", "child_exited", "cleanup_failed", "forced_stop", "guard_stop",
                             "ready_timeout", "operator_interrupt", "readiness_refused"), template(r"refused:(.+)")))
STOP_BY = maybe(vocab("duration_elapsed", "guard_stop", "ready_timeout", "operator_interrupt", "readiness_refused"))
READINESS = vocab("tracking_diagnostic", "landmark_diagnostic")  # step5_guard.READINESS (sessions 80 and 90)
CAPTURE_RESULT = Either((vocab("saved", "too_large", "not_json", "not_reached"), template(r"unavailable:(.+)")))
LISTENER = {"port": INT, "family": vocab("tcp", "tcp6"), "scope": vocab("loopback", "any", "other")}

PENDING_DURABILITY = template(
    r"rule observations waiting to be recorded are held in memory only \(at most (\d+)\); "
    r"they are lost if the process stops abruptly: not a crash-safe spool")
OCCUPANCY_REASON = template(r"no fresh video \((.+)\)", r"person detector (.+)",
                            fixed("detector has not processed a recent frame"),
                            r"(\d+) confirmed person\(s\) on fresh video", fixed("nobody detected on fresh video"))
SCENE_REASON = template(fixed("scene analysis disabled"), fixed("scene analysis turned off"),  # D99
                        r"scene analysis unavailable: (.+)",
                        fixed("current scene report"), r"last scene check ([^:]+): (.*)", fixed("no scene result yet"),
                        r"last scene result not current \((.+)\)")
DEGRADED = template(r"capture (\S+)(?: \((.+)\))?", r"video (\S+)", r"detector unavailable \((.*)\)",
                    r"scene analysis unavailable \((.*)\)", r"face recognition unavailable \((.*)\)",
                    r"(\d+) rule observation\(s\) not yet recorded \((.*)\)",
                    r"(\d+) rule observation\(s\) lost: not recorded", r"(\S+) unavailable \((.*)\); its alerts stay "
                    r"queued", r"delivery worker problem \((.*)\)", fixed("delivery worker stopped"))
TRANSITION_REASON = template(*(fixed(t) for t in (
    *VOTE_NOTES, "no face visible", "face ownership ambiguous", *OWNERSHIP, "no face observed yet",
    "consistent faces matched nobody", "contradictory face evidence", "track no longer current", *NOT_CARRIED,
    "face recognition disabled")),  # D99: identities end when face recognition is turned off
    r"(\d+) consistent matches")
TRACK_STATUS = vocab("tentative", "confirmed")  # contracts.TrackStatus
# A person's identity reason (D95): the core's own (identity.state, runtime.EdgeCore._identity), never free text.
IDENTITY_REASON = template(*(fixed(t) for t in (
    *VOTE_NOTES, "no face visible", "face ownership ambiguous", *OWNERSHIP, "no face observed yet",
    "consistent faces matched nobody", "contradictory face evidence", "no identities enrolled",
    "face recognition disabled", "face recognition unavailable", *NOT_CARRIED)),
    r"(\d+) consistent matches")
INCIDENT_TITLE = template(r"Person in restricted zone '(.*)'", r"Person dwelling in zone '(.*)'",
                          fixed("Possible fire or smoke (scene model only, unconfirmed)"))

# D78: what --live-preview (D69, D75), --snapshots (D70) and --chat [--chat-validation] (D74, D77) add to a run's
# evidence, field by field, so that a combined run is read as strictly as a face-only one. A run without those flags
# has none of these fields. Fixed text is declared as fixed text, never as free text.
PREVIEW_STATE = vocab("live", "stale", "offline", "starting", "paused")  # live_preview.STATE_NAMES, PAUSED (D99)
PREVIEW_REASON = template(
    fixed("the newest frame of the connected stream"), fixed("no frame yet since the runtime started"),
    fixed("no frame received"), r"no new frame for (\d+) s \(offline after ([\d.]+) s\)",
    r"no new frame for ([\d.]+) s \(stale after ([\d.]+) s\)",
    fixed("the stream connection this frame came from has ended; reconnecting"),
    fixed("the live view is turned off in Features"))  # D99: live_preview.PAUSED_REASON
TIME_QUALITY = vocab("none", "stream_relative", "capture_synced")  # contracts.SourceTimeQuality
TIME_LABEL = template(*(fixed("ingest time: when this device received and decoded the frame, not when the camera "
                              f"captured it; {text} (source time quality: {quality})") for quality, text in (
    ("none", "the camera's capture time is not known"),
    ("stream_relative", "the camera's capture time is not known (its stream timestamps map to no clock)"),
    ("capture_synced", "a capture time is known through a validated clock sync, but is not shown here"))))
PREVIEW_BOUNDS = {"max_width": INT, "max_height": INT, "jpeg_quality": INT, "max_encodes_per_s": NUM,
                  "max_jpeg_bytes": INT, "pictures_kept": INT, "yield_encodes_per_s": NUM, "stream_wait_s": NUM,
                  "viewer_idle_s": NUM, "stale_after_ms": INT}  # LivePreview.bounds
PREVIEW_COUNTERS = vocab(  # LivePreview counters
    "reused", "encode_busy", "rate_limited", "encodes", "encode_failures", "served", "not_kept", "refused_not_current",
    "stream_requests", "stream_pictures_sent", "stream_no_newer_picture", "stream_not_live", "yielded",
    "encoder_starts", "paused", "stream_refused_paused")  # the last two D99
SNAPSHOT_POLICY = {"retention_s": NUM, "quota_bytes": INT, "max_count": INT, "reserve_min_bytes": INT,
                   "reserve_fraction": NUM, "max_bytes": INT, "max_width": INT, "max_height": INT,
                   "jpeg_quality": INT, "prune_interval_s": NUM}  # incidents.snapshots.SnapshotPolicy
SNAPSHOT_MISSING = ("frame_not_retained", "too_many_waiting", "queue_full", "frame_identity_mismatch",
                    "incident_id_not_file_safe", "not_an_image", "unexpected_channels", "unsupported_pixel_format",
                    "jpeg_encode_failed", "encode_failed", "too_large", "retention_failed", "free_space_reserve",
                    "write_failed", "snapshots_off")  # SnapshotStore._missing labels, before any ':' (D99 last)
SNAPSHOT_STORE_COUNTERS = vocab("saved", "duplicate", *(f"missing:{label}" for label in SNAPSHOT_MISSING),
                                "pruned:retention_age", "pruned:retention_quota")
SNAPSHOT_WRITER_COUNTERS = vocab("queued", "not_queued_full", "not_queued_stopped", "saved", "missing", "duplicate",
                                 "errors")
CHAT_STATE = vocab("off", "unavailable", "available", "paused")  # ChatService.availability
GATE_REASON = template(
    fixed("memory pressure cannot be read, so chat waits until it can"),
    r"device memory pressure is above ([\d.]+) GB, so chat is stopped and a running question cancelled",
    r"device memory pressure has been above ([\d.]+) GB for (\d+) samples, so new questions wait",
    r"scene analysis missed its deadline (\d+) times within ([\d.]+) s, so new questions wait")
CHAT_REASON = Either((template(
    fixed("chat is off; start sentinel run with --chat to offer it"),
    fixed("not admitted: concurrent scene analysis and chat have not passed the operator-run resource and "
          "interference checks, so chat stays unavailable"),
    fixed("the scene model is not running (start sentinel run with --scene)"),
    fixed("ask about the picture the camera is showing now"),
    fixed("ask about the picture the camera is showing now (validation run: chat is not admitted; its answers and "
          "their effect on monitoring are being checked)"),
    fixed("monitoring comes first"), fixed("chat is turned off in Features")), GATE_REASON))  # the last D99
CHAT_ADVISORY = template(fixed(
    "Answers come from the on-device scene model and describe one frame: they are advisory. Chat cannot acknowledge "
    "incidents, change rules, control the camera, run commands or identify anyone."))
CHAT_COUNTERS = vocab(  # ChatService counters
    "refused_unavailable", "refused_paused", "busy_pending", "busy_interval", "no_frame", "no_fresh_frame",
    "busy_scene", "accepted", "gave_up_waiting", "timeouts", "answered", "errors", "cancelled",
    "cancelled_operator_off")  # D99
GATE_COUNTERS = vocab("samples", "closed", "reopened", "cancel_requests")  # ChatGate counters
GATE_LIMITS = {"stop_new_bytes": INT, "stop_samples": INT, "cancel_bytes": INT, "readmit_bytes": INT,
               "readmit_s": NUM, "misses_window_s": NUM, "misses_to_stop": INT}
# D99: --feature-controls adds the features block (DemoRuntime._features_view), the startup's switchable features and
# one "control" line per applied change (features.FeatureControls); fixed labels and numbers only, never the key.
FEATURES = ("face", "scene", "chat", "live_view", "snapshots")  # features.FEATURES
FEATURE_STATES = ("on", "off")
FEATURE_ACTUAL = ("running", "starting", "stopping", "off", "busy", "unavailable", "error")
FEATURE_WHY = ("running", "pending", "finishing", "turned_off", "not_started", "failed_at_start", "not_admitted",
               "paused_by_gate", "worker_stopped", "server_not_ready", "writer_problem")
FEATURE_VALIDATION = ("pending_validation", "validation_run", "not_admitted")
FEATURE_RESULTS = ("accepted", "replaced", "applied", "unchanged", "refused")  # features.OUTCOMES
FEATURE_COUNTERS = vocab("accepted", "replaced", "applied", "unchanged", "refused", "records_dropped",
                         "refused_invalid", "refused_not_switchable", "refused_busy", "refused_not_this_page",
                         "refused_wrong_key", "refused_locked")
FEATURES_STATUS = {
    "features.accepting": BOOL,
    "features.items.*.requested": vocab(*FEATURE_STATES), "features.items.*.actual": vocab(*FEATURE_ACTUAL),
    "features.items.*.why": vocab(*FEATURE_WHY), "features.items.*.problem": maybe(TEXT),
    "features.items.*.switchable": BOOL, "features.items.*.validation": maybe(vocab(*FEATURE_VALIDATION)),
    "features.log[].change": INT, "features.log[].at_utc": ISO, "features.log[].feature": vocab(*FEATURES),
    "features.log[].requested": vocab(*FEATURE_STATES), "features.log[].result": vocab(*FEATURE_RESULTS),
    "features.counters.*": INT, "features.rate.limit": INT, "features.rate.window_s": NUM,
    "components.face.paused": BOOL, "components.scene.paused": BOOL,
}
FEATURE_RECORDS = {  # features.FeatureControls.done: one per change the loop applied, found unchanged or refused
    "control": vocab("feature"), "change": INT, "at_utc": ISO, "feature": vocab(*FEATURES),
    "requested": vocab(*FEATURE_STATES), "result": vocab(*FEATURE_RESULTS),
}
# Session 79: the tracking diagnostic's startup marker inside the face entry (demo_runtime.assemble): the bounds
# (tracking.diagnostic.DiagnosticBounds.numbers) and the tracker's settings as the loaded objects hold them
# (legacy_ultralytics.LegacyUltralyticsTracker.diagnostic_settings; null when the detector did not load)
TRACKING_BOUNDS = ("shadow_conf", "max_records", "max_bytes", "max_line_bytes", "max_det", "max_shadow", "max_lost",
                   "max_assign_tracks", "max_assign_dets")
TRACKER_SETTINGS = ("conf", "match_thresh", "track_high_thresh", "track_low_thresh", "new_track_thresh",
                    "track_buffer", "max_time_lost", "unconfirmed_thresh", "shadow_conf", "shadow_max_nms", "max_det",
                    "iou")
TRACKING_STARTUP = {
    **{f"face.tracking_diagnostic.{k}": NUM for k in TRACKING_BOUNDS},
    "face.tracking_diagnostic.tracker": NULL,
    **{f"face.tracking_diagnostic.tracker.{k}": NUM for k in TRACKER_SETTINGS},
    "face.tracking_diagnostic.tracker.fuse_score": BOOL,
}
LANDMARK_MARKER = ("version", "score", "min_eye_spacing_px", "persons_max")  # identity.landmarks.marker (session 90)
LANDMARK_STARTUP = {"face.landmark_diagnostic": NULL, **{f"face.landmark_diagnostic.{k}": NUM for k in LANDMARK_MARKER}}
LANDMARK_STATUSES = ("measured", "invalid", "no_candidate", "ambiguous", "unavailable", "not_recorded")  # landmarks
LANDMARK_REASONS = ("malformed_row", "landmark_outside_frame", "eye_spacing_too_small", "eyes_not_left_to_right",
                    "mouth_not_below_eyes", "pass_failed", "persons_bound")  # landmarks.INVALID/OTHER_REASONS
LANDMARK_MEASURES = ("nose_lateral", "nose_vertical", "mouth_vertical", "nose_position", "mouth_lateral",
                     "eye_line_deg", "eye_spacing_px", "eye_spacing_ratio")  # landmarks.MEASURES
OPTIONAL_STARTUP = {  # assemble's startup fields for the live view, snapshots and chat
    **{f"live_preview.{k}": v for k, v in PREVIEW_BOUNDS.items()},
    **{f"snapshots.{k}": INT for k in ("removed_temporary", "removed_unrecorded", "unexpected_files", "pruned")},
    **{f"snapshots.policy.{k}": v for k, v in SNAPSHOT_POLICY.items()},
    "chat.state": CHAT_STATE, "chat.reason": CHAT_REASON,
    "feature_controls.*": BOOL,  # D99
}
OPTIONAL_STATUS = {  # build_status's preview, and DemoRuntime's preview, snapshots and chat components
    **FEATURES_STATUS,
    "preview.enabled": BOOL, "preview.state": PREVIEW_STATE, "preview.reason": PREVIEW_REASON, "preview.age_ms": COUNT,
    "preview.frame": NULL, **{f"preview.frame.{k}": INT for k in ("stream_epoch", "frame_seq", "native_width",
                                                                  "native_height")},
    "preview.frame.ingest_utc": ISO, "preview.frame.source_time_quality": TIME_QUALITY,
    "preview.time_label": maybe(TIME_LABEL), "preview.picture": NULL, "preview.problem": maybe(TEXT),
    **{f"preview.bounds.{k}": v for k, v in PREVIEW_BOUNDS.items()},
    "components.preview.state": vocab("enabled", "paused"), "components.preview.pictures_kept": INT,
    "components.preview.problem": maybe(TEXT), "components.preview.encoder.running": BOOL,
    "components.preview.encoder.nice": COUNT, "components.preview.encode_ms": NULL,
    "components.preview.encode_ms.last": INT, "components.preview.encode_ms.median": NUM,
    "components.preview.encode_ms.max": NUM, "components.preview.counters.*": INT,
    "components.snapshots.state": vocab("not_started", "running", "stopped"), "components.snapshots.queued": INT,
    "components.snapshots.held_frames": INT, "components.snapshots.problem": maybe(TEXT),
    "components.snapshots.writer.*": INT, "components.snapshots.store.*": INT,
    **{f"components.snapshots.policy.{k}": v for k, v in SNAPSHOT_POLICY.items()},
    "components.snapshots.pinned_frames": INT, "components.snapshots.not_queued": INT,
    "components.chat.state": CHAT_STATE, "components.chat.reason": CHAT_REASON, "components.chat.counters.*": INT,
    "components.chat.advisory": CHAT_ADVISORY, "components.chat.validation_run": BOOL,
    "components.chat.gate.open": BOOL, "components.chat.gate.reason": maybe(GATE_REASON),
    "components.chat.gate.pressure_bytes": COUNT, "components.chat.gate.peak_pressure_bytes": COUNT,
    "components.chat.gate.misses_in_window": INT, **{f"components.chat.gate.limits.{k}": v
                                                     for k, v in GATE_LIMITS.items()},
    "components.chat.gate.counters.*": INT,
    "components.chat.scene_waits_behind_chat.n": INT,
    **{f"components.chat.scene_waits_behind_chat.{k}": SECONDS for k in ("median_ms", "max_ms", "last_ms",
                                                                         "max_ever_ms")},
    "components.chat.scene_waits_behind_chat.grace_ms": INT,  # D81
    **{f"components.chat.scene_waits_behind_chat.run_ms.{group}.{k}": (INT if k == "n" else SECONDS)
       for group in ("behind_side", "others") for k in ("n", "median_ms", "max_ms")},
}
OPTIONAL_STATUS_KEYS = {"features.items": vocab(*FEATURES), "features.counters": FEATURE_COUNTERS,
                        "components.preview.counters": PREVIEW_COUNTERS,
                        "components.snapshots.writer": SNAPSHOT_WRITER_COUNTERS,
                        "components.snapshots.store": SNAPSHOT_STORE_COUNTERS,
                        "components.chat.counters": CHAT_COUNTERS, "components.chat.gate.counters": GATE_COUNTERS}


# ---------------------------------------------------------------- profiles


def nest(prefix: str, fields: dict[str, Spec]) -> dict[str, Spec]:
    return {f"{prefix}.{path}" if path else prefix: spec for path, spec in fields.items()}


def _prefixes(path: str) -> Iterator[str]:
    for match in re.finditer(r"\.|\[\]", path):
        yield path[:match.start()]
        if match.group() == "[]" and match.end() < len(path):
            yield path[:match.end()]


@dataclass(frozen=True)
class JsonProfile:
    """Leaf fields by path (``a.b``, ``a[]`` for list items, ``a.*`` under a data-keyed object) and the key
    vocabulary of each data-keyed object."""

    fields: dict[str, Spec]
    keys: dict[str, Spec] = field(default_factory=dict)
    lines: bool = False  # JSON lines: one object per line
    containers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        found = {p for path in [*self.fields, *self.keys] for p in _prefixes(path) if p}
        object.__setattr__(self, "containers", frozenset(found | set(self.keys)))

    def known(self, path: str) -> bool:
        return path in self.fields or path in self.containers


@dataclass(frozen=True)
class KeyValueProfile:
    """``key=value`` lines (provenance.txt)."""

    fields: dict[str, Spec]


@dataclass(frozen=True)
class AccountProfile:
    """F2's account.txt, as face_account renders it: its header, then ``key=value   # comment`` per field. The
    header names the format (1 to 10, as face_account.FORMATS lists them), and each value is checked against that
    format's field."""


@dataclass(frozen=True)
class LogProfile:
    """Free-text log lines (the runtime's stderr). A line in a known log format has its timestamp and severity
    as schema text and its message as content; any other line is content as a whole."""

    formats: tuple[re.Pattern, ...]


ENROLL = JsonProfile({
    "identity": vocab("enroll"), "photos": INT, "dry_run": BOOL, "accepted": INT,
    "results[].photo": INT, "results[].result": vocab("accepted", "rejected"),
    "results[].reason": vocab("face_found", "no_face", "several_faces", "unreadable_photo", "low_quality"),
    "results[].quality": NUM, "results[].face_width_px": INT, "inconsistent_photos[]": INT,
    "refused": Either((vocab("too_many_photos", "too_few_usable_photos", "inconsistent_photos",
                             "close_to_enrolled_identity", "photos_changed"), template(r"gallery_incompatible:(.+)"))),
    "identity_id": IDN, "verified": BOOL, "photos_deleted": INT, "photos_changed_not_deleted": INT,
    "folder_entries_left": INT,
})
LISTING = JsonProfile({
    "identity": vocab("list"), "identities[].identity_id": IDN, "identities[].prototypes": INT,
    "identities[].consent_date": DATE, "identities[].enrolled_utc": ISO,
})
AUDIT = JsonProfile({  # identity.gallery: a change's started line, then its completed (or aborted) one
    "utc": ISO, "action": vocab("enroll", "revoke", "purge", "enroll_started", "enroll_aborted", "revoke_started",
                                "revoke_aborted", "purge_started", "purge_incomplete"), "identity_id": maybe(IDN),
    "photos_used": INT, "photos_rejected": INT, "identities_left": INT, "gallery": INT, "inbox_files": INT,
    "left": INT,  # purge_incomplete: what could not be deleted
}, lines=True)
F1_PROVENANCE = KeyValueProfile({
    "inbox_folder_name": RUN_NAME, "gallery_before_dryrun": vocab("absent", "present"),
    "gallery_after_dryrun": vocab("absent", "present"), "photos_supplied": DIGITS, "consent_date": DATE,
    "dryrun_exit": DIGITS, "enroll_exit": DIGITS, "list_exit": DIGITS,
})

THP_SCOPE = {f"{checkpoint}.{process}": COUNT for checkpoint in ("before_launch", "llama_ready", "workload_verified")
             for process in ("runtime", "scene_server")}
SCENE_SERVER = {"": NULL, "state": SERVER, "problem": maybe(TEXT), "layers": maybe(LAYERS),
                "vision_on_gpu": maybe(BOOL)}
STARTUP = {  # the runtime's ``starting`` line (demo_runtime.assemble and cli)
    "memory_policy.thp": THP, "memory_policy.model_file_release": RELEASE,
    **nest("thp_scope", THP_SCOPE),
    "scene_memory_before": NULL, "scene_memory_before.*": COUNT,
    "detector_memory_before": NULL, "detector_memory_before.*": COUNT,
    "face_memory_before": NULL, "face_memory_before.*": COUNT,
    **nest("scene_server", SCENE_SERVER),
    "releases.*.settle_s": SECONDS, "releases.*.files.*.result": maybe(RELEASE_RESULTS),
    "releases.*.files.*.returncode": COUNT, "releases.*.files.*.error": maybe(TEXT),
    "releases.*.files.*.bytes": COUNT, "releases.*.files.*.elapsed_s": SECONDS,
    "thp_disable.t_mono": SECONDS, "thp_disable.model_modules_loaded[]": TEXT, "thp_disable.set_rc": COUNT,
    "thp_disable.set_errno": COUNT, "thp_disable.get_value": COUNT, "thp_disable.thp_enabled": COUNT,
    "thp_disable.anon_huge_pages_bytes": COUNT, "thp_disable.verified": maybe(BOOL),
    "thp_disable.reason": maybe(TEXT),
    "detector_problem": maybe(TEXT), "scene_problem": maybe(TEXT), "face_problem": maybe(TEXT),
    "notifier_problems.*": maybe(TEXT), "face.validation_run": BOOL, "face.identities_enrolled": INT,
    "face.comparison": BOOL,  # D87, only in --face-comparison runs
    "face.detection": vocab("full_scale"),  # D88, only in --face-full-scale-detection runs
    "face.shadow": BOOL, "face.capture": BOOL,  # D91, only in the enrollment experiment's runs
    **{f"face.continuity.{k}": NUM for k in CONTINUITY_POLICY},  # D97, only in --face-continuity runs
    **TRACKING_STARTUP,  # session 79, only in --tracking-diagnostic runs
    **LANDMARK_STARTUP,  # session 90, only in --face-landmark-diagnostic runs
    "status_page": maybe(LOOPBACK_URL),
    **OPTIONAL_STARTUP,
}
STARTUP_KEYS = {"scene_memory_before": MEMFREE_KEYS, "detector_memory_before": MEMFREE_KEYS,
                "face_memory_before": MEMFREE_KEYS, "releases": RELEASE_COMPONENTS, "releases.*.files": RELEASE_ROLES,
                "notifier_problems": CHANNELS, "feature_controls": vocab(*FEATURES)}
STATUS_LINE = {  # cli.status_line
    "updated_utc": ISO, "state": RUNTIME, "video": maybe(VIDEO), "occupancy": maybe(OCCUPANCY),
    "scene": maybe(SCENE_STATUS), "identities": NULL,
    **{f"identities.{k}": INT for k in (*IDENTITY_STATES, "fresh", "retained")},
    **{f"rates.{k}": SECONDS for k in ("window_s", "captured_fps", "processed_fps", "failed_per_s")},
    "capture.state": CAPTURE, "capture.stream_epoch": COUNT, "capture.reconnects": INT, "pending_signals": INT,
    "degraded[]": DEGRADED,
}
IDENTITY_RECORDS = {  # runtime.tick_record, FaceResultRecord, IdentityTransition, ComparisonRecord
    "identity": vocab("tick", "result", "transition", "comparison", "capture", "break"), "schema": INT,
    "stream_epoch": INT, "cause": vocab(*CONTINUITY_CAUSES),  # D97: runtime.break_record (continuity runs only)
    "frame_seq": INT,
    "frame_mono_ns": INT, "tick": TICKS, "tracks[]": INT, "applied_mono_ns": INT, "outcome": OUTCOMES,
    "faces": COUNT, "processing_ms": SECONDS, "error": maybe(TEXT), "persons[].track_id": INT,
    "persons[].ownership": vocab(*OWNERSHIP), "persons[].vote": vocab("none", "unknown", "match"),
    "persons[].identity_id": maybe(IDN), "persons[].label": vocab(*VOTE_LABELS, *OWNERSHIP),
    # schema 2 (a validation run's runtime.FaceDiagnostics): numbers only
    "persons[].box[]": NUM, "persons[].confidence": maybe(NUM), "persons[].face": NULL,
    "persons[].face.w_px": INT, "persons[].face.h_px": INT, "persons[].face.score": NUM,
    "persons[].similarity": maybe(NUM),
    # schema 3 (D86, runtime.full_scale_numbers): counts and numbers only
    "full_scale": NULL, "full_scale.faces": INT, "full_scale.assigned": INT, "full_scale.best_score": maybe(NUM),
    "full_scale.best_w_px": COUNT, "full_scale.ms": NUM,
    # schema 5 (D88, runtime.detection_numbers): numbers only
    "detection": NULL, "detection.best_below_score": maybe(NUM), "detection.best_below_w_px": COUNT,
    "detection.detect_ms": NUM, "detection.embed_ms": maybe(NUM),
    # schema 6 (D90 runtime.face_geometry_numbers and the prototype position; D91 the working gallery's similarity):
    # numbers only, each list of four numbers or null
    "persons[].prototype": COUNT, "persons[].geometry": NULL, "persons[].geometry.box[]": NUM,
    "persons[].geometry.raw_box": NULL, "persons[].geometry.raw_box[]": NUM,
    "persons[].geometry.overhang_px": NULL, "persons[].geometry.overhang_px[]": NUM,
    "persons[].geometry.eyes": NULL, "persons[].geometry.eyes.roll_deg": NUM,
    "persons[].geometry.eyes.spacing_px": NUM,
    "persons[].shadow": NULL, "persons[].shadow.similarity": NUM, "persons[].shadow.prototype": INT,
    # schema 7 (session 90, landmarks.landmark_records and FaceResultRecord.landmark_ms): fixed labels and numbers only
    "landmark_ms": maybe(NUM), "persons[].landmarks": NULL,
    "persons[].landmarks.status": vocab(*LANDMARK_STATUSES), "persons[].landmarks.reason": maybe(vocab(*LANDMARK_REASONS)),
    "persons[].landmarks.score": maybe(NUM), "persons[].landmarks.face_iou": maybe(NUM),
    **{f"persons[].landmarks.{k}": maybe(NUM) for k in LANDMARK_MEASURES},
    # D91, the capture's record at the stop (identity.experiment.Capture.finish): fixed labels and numbers only
    "written": BOOL, "refused": maybe(vocab(*CAPTURE_REFUSALS)), "faces_kept": INT, "faces_dropped_over_cap": INT,
    "faces_in_span": INT, "prototypes": INT, "min_best_to_others": maybe(NUM), "similarity_to_original[]": NUM,
    "verified": BOOL,
    "track_id": INT, "from": vocab(*IDENTITY_STATES), "to": vocab(*IDENTITY_STATES, "cleared"),
    "identity_id": maybe(IDN), "basis": maybe(BASIS), "last_vote_age_ms": COUNT, "reason": TRANSITION_REASON,
    "mono_ns": INT,
    # schema 4 (D87, runtime.ComparisonRecord): fixed labels and numbers only ("error" is shared with results)
    "kind": vocab(*COMPARISON_KINDS), "disposition": vocab(*COMPARISON_DISPOSITIONS), "score": NUM, "w_px": INT,
    "similarity": maybe(NUM), "ms": maybe(NUM),
}
# Session 79, the tracking diagnostic (tracking.diagnostic.frame_record and summary_record, only in
# --tracking-diagnostic runs): numbers and fixed labels only; boxes are fractions of the frame; track numbers are the
# tracker's; nothing here is a picture, a crop or an identity.
TRACKING_PUBLISHED = ("continued", "refound", "confirmed")  # tracking.diagnostic.PUBLISHED_OUTCOMES
TRACKING_UNPUBLISHED = ("new_unconfirmed", "unmatched")  # tracking.diagnostic.UNPUBLISHED_OUTCOMES
TRACKING_PROBLEMS = ("capture_missing", "capture_failed", "tracker_replaced", "raw_missing", "clone_failed",
                     "shadow_failed", "layout_unknown", "snapshot_failed", "assign_failed",
                     "no_backend_diagnostics")  # tracking.diagnostic.PROBLEMS
TRACKING_SHORTENED = ("shadow", "assign", "unpub", "lost", "pub", "det")  # tracking.diagnostic.SHORTENED
TRACKING_TIMINGS = ("backend", "capture", "clone", "shadow", "snapshot", "record")  # tracking.diagnostic.TIMINGS
TRACKING_RECORDS = {
    "tracking": vocab("frame", "summary"), "n": INT, "tf": COUNT,
    "det[].i": INT, "det[].s": NUM, "det[].b[]": NUM,
    "pub[].id": INT, "pub[].i": COUNT, "pub[].o": vocab(*TRACKING_PUBLISHED), "pub[].b[]": NUM,
    "unpub[].i": INT, "unpub[].o": vocab(*TRACKING_UNPUBLISHED), "unpub[].id": COUNT,
    "lost[].id": INT, "lost[].lost": INT, "lost[].b[]": NUM, "removed[]": INT,
    "assign[].stage": vocab("pool", "unconfirmed"), "assign[].thresh": NUM, "assign[].tracks[].id": INT,
    "assign[].tracks[].st": vocab("tracked", "lost"), "assign[].tracks[].lost": INT, "assign[].dets[]": INT,
    "assign[].rows[].id": INT, "assign[].rows[].c[]": NUM, "assign[].cut": BOOL,
    "ss": INT, "shadow[].s": NUM, "shadow[].b[]": NUM, "sh_hi": INT, "sh_agree": INT, "sh_cand": COUNT,
    "sh_lim": maybe(BOOL), "cur[]": INT, **{f"ns.{k}": COUNT for k in TRACKING_TIMINGS},
    "problems[]": vocab(*TRACKING_PROBLEMS), "trunc.*": INT,
    # the summary line at the stop
    "kept": INT, "dropped": INT, "bytes": INT, "truncated_lines": INT, "max_line_bytes_seen": INT, "processed": INT,
    "problem_counts.*": INT, "limits.max_records": INT, "limits.max_bytes": INT, "limits.max_line_bytes": INT,
}
TRACKING_KEYS = {"trunc": vocab(*TRACKING_SHORTENED), "problem_counts": vocab(*TRACKING_PROBLEMS)}
RUN = JsonProfile({
    "run": vocab("starting", "stopped"), **nest("startup", STARTUP),
    **STATUS_LINE, **IDENTITY_RECORDS, **FEATURE_RECORDS, **TRACKING_RECORDS,
    "shutdown.stopped.*": BOOL, "shutdown.all_stopped": BOOL, "shutdown.signals_not_recorded": INT,
    "database_closed": BOOL, **nest("status", STATUS_LINE),
}, keys={**{f"startup.{k}": v for k, v in STARTUP_KEYS.items()}, "shutdown.stopped": STOPPED_PARTS,
         **TRACKING_KEYS}, lines=True)

GUARD = JsonProfile({"*": maybe(NUM)}, keys={"": SAMPLE_KEYS}, lines=True)  # step5_guard's samples

STARTING_SUMMARY = {  # step5_guard._starting_summary
    "": NULL, "memory_policy": NULL, "memory_policy.thp": THP, "memory_policy.model_file_release": RELEASE,
    "thp_scope": NULL, **nest("thp_scope", THP_SCOPE),
    "thp_disable": NULL, "thp_disable.verified": maybe(BOOL), "thp_disable.reason": maybe(TEXT),
    "thp_disable.set_rc": COUNT, "thp_disable.thp_enabled": COUNT,
    "releases.*.*": maybe(RELEASE_RESULTS), **nest("scene_server", SCENE_SERVER),
    "scene_problem": maybe(TEXT), "detector_problem": maybe(TEXT), "face_problem": maybe(TEXT),
    "notifier_problems": NULL, "notifier_problems.*": maybe(TEXT),
    "scene_memory_before": NULL, "scene_memory_before.*": COUNT,
    "detector_memory_before": NULL, "detector_memory_before.*": COUNT,
    "face_memory_before": NULL, "face_memory_before.*": COUNT,
    "face": NULL, "face.validation_run": BOOL, "face.identities_enrolled": INT, "face.comparison": BOOL,
    "face.detection": vocab("full_scale"),  # D88
    "face.shadow": BOOL, "face.capture": BOOL,  # D91
    **{f"face.continuity.{k}": NUM for k in CONTINUITY_POLICY},  # D97: the guard keeps the startup's face entry
    **TRACKING_STARTUP,  # session 79: likewise
    **LANDMARK_STARTUP,  # session 90: likewise
}
STARTING_SUMMARY_KEYS = {"releases": RELEASE_COMPONENTS, "releases.*": RELEASE_ROLES, "notifier_problems": CHANNELS,
                         "scene_memory_before": MEMFREE_KEYS, "detector_memory_before": MEMFREE_KEYS,
                         "face_memory_before": MEMFREE_KEYS}
# The memory guard's first stopping sample (operator_check.PressureGuard._stop; null when nothing stopped it): its
# conditions as operator_check.memory_conditions and _condition write them, numbers and fixed texts only. Session 81:
# declared as text before, so the device run trk-20261016T100149Z's guard stop was the only nonconforming location in
# its result.json.
GUARD_STOPS = vocab("telemetry_unavailable", "telemetry_invalid", "sampled_pressure_stop", "free_or_available_stop",
                    "swap_counter_change", "clock_unavailable", "sampling_gap")
GUARD_TRIGGER = {
    "guard.trigger": NULL, "guard.trigger.stop": GUARD_STOPS, "guard.trigger.conditions[].stop": GUARD_STOPS,
    "guard.trigger.conditions[].condition": template(
        r"MemTotal - MemAvailable >= \d+", r"Mem(?:Free|Available) < \d+", r"pswp(?:in|out) != \d+",
        r"t_mono interval [<>] (?:0|0\.5)", fixed("required field missing or invalid"),
        fixed("MemTotal <= 0 or MemFree/MemAvailable > MemTotal"), fixed("t_mono missing or not finite")),
    "guard.trigger.conditions[].field": vocab("MemTotal - MemAvailable", "MemFree", "MemAvailable", "pswpin",
                                              "pswpout", "t_mono interval"),
    "guard.trigger.conditions[].comparison": vocab(">=", "<", ">", "!="),
    "guard.trigger.conditions[].threshold": NUM, "guard.trigger.conditions[].value": NUM,
    "guard.trigger.conditions[].fields[]": vocab("MemTotal", "MemAvailable", "MemFree", "pswpin", "pswpout",
                                                 "t_mono"),
    "guard.trigger.t_mono": maybe(NUM), "guard.trigger.guard_jsonl_line": INT, "guard.trigger.sample.*": maybe(NUM),
    "guard.trigger.basis": template(fixed("the sample that stopped the guard, before the owned group's cleanup")),
}
RESULT = JsonProfile({  # step5_guard.supervise
    "schema_version": INT, "mode": vocab("step5_guard"), "status": GUARD_STATUS,
    "parameters.duration_s": NUM, "parameters.from_launch": BOOL, "parameters.ready_timeout_s": NUM,
    "parameters.stop_grace_s": NUM, "parameters.step4_headroom": BOOL,
    "parameters.status_at[].at_s": NUM, "parameters.status_at[].name": vocab(*STATUS_NAMES),
    "parameters.cues[].at_s": NUM, "parameters.cues[].name": vocab(*CUE_NAMES),
    "parameters.require_ready": maybe(READINESS),
    # session 80: the readiness requirement's outcome on the starting line (step5_guard.readiness_problem's label)
    "readiness.required": READINESS, "readiness.met": BOOL, "readiness.reason": maybe(TEXT),
    "readiness.starting_after_s": SECONDS,
    "limits.pressure_stop_bytes": INT, "limits.mem_free_floor_bytes": INT, "limits.mem_available_floor_bytes": INT,
    "limits.swap_counters": vocab("unchanged"), "limits.max_sample_gap_s": NUM,
    "preflight_sample.*": maybe(NUM), "baseline.*": maybe(NUM),
    "stop.requested_by": STOP_BY, "stop.requested_at_s": SECONDS, "stop.exited_after_stop_s": SECONDS,
    "stop.forced": BOOL,
    "child.launched_utc": maybe(ISO), "child.exited_utc": maybe(ISO), "child.returncode": COUNT,
    "child.ran_s": SECONDS, "child.ready_after_s": SECONDS, "child.ready_mono_s": SECONDS,
    "guard.samples": INT, "guard.largest_gap_s": SECONDS, "guard.peak_pressure_bytes": COUNT,
    "guard.min_mem_free_bytes": COUNT, "guard.min_mem_available_bytes": COUNT, **GUARD_TRIGGER,
    "guard.trigger_after_stop_request": BOOL,
    "captures[].name": vocab(*STATUS_NAMES), "captures[].at_s": NUM, "captures[].taken_after_launch_s": SECONDS,
    "captures[].status": CAPTURE_RESULT, "captures[].bytes": INT, "captures[].listeners": NULL,
    **{f"captures[].listeners[].{k}": v for k, v in LISTENER.items()},
    "cues[].name": vocab(*CUE_NAMES), "cues[].at_s": NUM, "cues[].printed_after_launch_s": SECONDS,
    "cues[].printed_mono_s": SECONDS,
    "run_output.lines": INT, "run_output.json_lines": INT, "run_output.non_json_lines": INT,
    **nest("run_output.starting", STARTING_SUMMARY),
    "run_output.stopped": NULL, "run_output.stopped.all_stopped": maybe(BOOL),
    "run_output.stopped.stopped": NULL, "run_output.stopped.stopped.*": maybe(BOOL),
    "run_output.stopped.database_closed": maybe(BOOL), "run_output.stopped.signals_not_recorded": COUNT,
    "run_error_label": maybe(TEXT),
    "leftovers.llama_server_after_exit": INT, "leftovers.llama_server_left": INT,
    "leftovers.runtime_group_left": BOOL, **{f"leftovers.listeners_after[].{k}": v for k, v in LISTENER.items()},
    "cleanup_clear": BOOL,
}, keys={"preflight_sample": SAMPLE_KEYS, "baseline": SAMPLE_KEYS, "guard.trigger.sample": SAMPLE_KEYS,
         **{f"run_output.starting.{k}": v for k, v in STARTING_SUMMARY_KEYS.items()},
         "run_output.stopped.stopped": STOPPED_PARTS})

FACE_COUNTERS = vocab(*(f"persons_{o}" for o in OWNERSHIP), *(f"faces_{o}" for o in OWNERSHIP),
                      *(f"results_{o}" for o in sorted(OUTCOMES.values)), "votes_match", "votes_unknown",
                      "no_vote_margin", "no_vote_low_quality", "no_vote_incompatible", "no_vote_no_enrollment",
                      "persons_before_break", *(f"breaks_{c}" for c in CONTINUITY_CAUSES),  # D97
                      "persons_before_expiry")  # state.BEFORE_EXPIRY
STATUS = JsonProfile({  # demo_runtime.DemoRuntime.snapshot through status_page.build_status
    "generated_utc": ISO,
    "runtime.camera_id": TEXT, "runtime.state": RUNTIME, "runtime.updated_utc": ISO, "runtime.status_age_s": SECONDS,
    "runtime.shutdown": NULL, "runtime.shutdown.stopped.*": BOOL, "runtime.shutdown.all_stopped": BOOL,
    "runtime.shutdown.signals_not_recorded": INT,
    "live": NULL, "live.sequence": INT, "live.video": VIDEO, "live.last_frame_age_ms": COUNT,
    "live.detector": CAPABILITY, "live.face_recognition": CAPABILITY, "live.scene_analysis": CAPABILITY,
    "live.occupancy": OCCUPANCY, "live.occupancy_reason": OCCUPANCY_REASON, "live.people": INT,
    "live.confirmed_people": INT, **{f"live.identity.{k}": INT for k in (*IDENTITY_STATES, "fresh", "retained")},
    "live.published_utc": ISO,  # D95: when the core published this live state
    # D95: each listed person's recognition state, only while face recognition runs (demo_runtime._face_rows)
    "live.faces[].track_id": INT, "live.faces[].stream_epoch": INT, "live.faces[].track_status": TRACK_STATUS,
    "live.faces[].predicted": BOOL, "live.faces[].last_measured_age_ms": INT,
    "live.faces[].state": vocab(*IDENTITY_STATES), "live.faces[].basis": maybe(BASIS),
    "live.faces[].vote_age_ms": COUNT, "live.faces[].reason": IDENTITY_REASON,  # no identity ID (D61, D95)
    "live.faces[].followed_ms": COUNT,  # D97: time followed without a break (continuity runs only)
    "live.faces_not_shown": INT,
    "live.scene": SCENE_STATUS, "live.scene_reason": SCENE_REASON, "live.scene_report": NULL,
    "live.scene_report.persons_visible": INT, "live.scene_report.fire_or_smoke": BOOL,
    "live.scene_report.threat": THREAT, "live.scene_report.uncertainty": UNCERTAINTY,
    "live.scene_report.summary": TEXT,
    **{f"rates.{k}": SECONDS for k in ("window_s", "captured_fps", "processed_fps", "failed_per_s")},
    **{f"frames.{k}": INT for k in ("captured", "replaced", "discarded", "taken", "processed", "failed", "skipped",
                                    "not_live")},
    "components.capture.state": CAPTURE, "components.capture.ready": BOOL, "components.capture.problem": maybe(TEXT),
    **{f"components.capture.{k}": COUNT for k in ("connects", "open_failures", "reconnects", "stream_ends",
                                                  "stream_epoch")},
    "components.capture.retry_delay_s": SECONDS,
    "components.detector.state": CAPABILITY, "components.detector.problem": maybe(TEXT),
    "components.detector.counters.*": INT,
    # session 79: the tracking diagnostic's budget (DemoRuntime._tracking_status), only in its runs
    **{f"components.detector.tracking.{k}": INT for k in ("kept", "dropped", "truncated_lines", "bytes",
                                                           "max_line_bytes_seen", "limit", "byte_limit")},
    "components.detector.tracking.problems.*": INT,
    "components.scene.state": CAPABILITY, "components.scene.problem": maybe(TEXT),
    **nest("components.scene.server", SCENE_SERVER), "components.scene.lane.*": INT, "components.scene.worker.*": INT,
    "components.face.state": CAPABILITY, "components.face.problem": maybe(TEXT),
    "components.face.identities_enrolled": INT, "components.face.validation_run": BOOL,
    "components.face.policy.confirmations": INT, "components.face.policy.vote_ttl_s": NUM,  # D95
    "components.face.policy.fresh_window_s": NUM,
    **{f"components.face.policy.continuity.{k}": NUM for k in CONTINUITY_POLICY},  # D97
    "components.face.detection": vocab("full_scale"),  # D88
    "components.face.experiment": vocab("shadow", "capture"),  # D91
    **{f"components.face.records.{k}": INT for k in ("kept", "dropped", "limit")},
    "components.face.results.*": INT, "components.face.worker.state": WORKER, "components.face.worker.busy": BOOL,
    "components.face.worker.waiting": BOOL, "components.face.worker.counters.*": INT,
    **{f"components.face.worker.processing_ms.{k}": SECONDS for k in ("n", "p50", "p95", "p99", "max")},
    # D87: the comparator's status and the runtime's final dispositions (counts and numbers only)
    "components.face.worker.comparison.counters.*": INT,
    **{f"components.face.worker.comparison.ms.{k}": SECONDS for k in ("n", "min", "p50", "max")},
    **{f"components.face.worker.comparison.policy.{k}": NUM for k in ("interval_s", "cap", "control_share",
                                                                       "max_frame_age_s")},
    "components.face.comparison_results.*": INT,
    "components.incidents.state": vocab("ok", "degraded"), "components.incidents.problem": maybe(TEXT),
    "components.incidents.annotation_problem": maybe(TEXT), "components.incidents.durability": PENDING_DURABILITY,
    **{f"components.incidents.{k}": INT for k in ("pending_limit", "pending_signals", "signals_dropped")},
    "components.notifications.min_severity": SEVERITY, "components.notifications.channels.*.state": CAPABILITY,
    "components.notifications.channels.*.problem": maybe(TEXT), "components.notifications.worker.state": WORKER,
    "components.notifications.worker.passes": INT, "components.notifications.worker.problem": maybe(TEXT),
    "components.notifications.worker.counters.*": INT,
    "database.state": CAPABILITY, "database.problem": maybe(TEXT), "degraded[]": DEGRADED,
    "incidents.unresolved": INT, "incidents.recent[].incident_id": INCIDENT_ID, "incidents.recent[].kind": TEXT,
    "incidents.recent[].zone_id": TEXT, "incidents.recent[].severity": SEVERITY,
    "incidents.recent[].status": INCIDENT_STATUS, "incidents.recent[].title": INCIDENT_TITLE,
    "incidents.recent[].first_observed_utc": ISO, "incidents.recent[].last_observed_utc": ISO,
    "incidents.recent[].annotations": INT,
    "delivery.totals.*": INT, "delivery.recent[].outbox_id": INT, "delivery.recent[].incident_id": INCIDENT_ID,
    "delivery.recent[].channel": TEXT, "delivery.recent[].message_kind": TEXT, "delivery.recent[].state": TEXT,
    "delivery.recent[].attempts": INT, "delivery.recent[].in_flight": BOOL, "delivery.recent[].ambiguous": BOOL,
    "delivery.recent[].next_attempt_utc": maybe(ISO), "delivery.recent[].updated_utc": ISO,
    "delivery.recent[].last_error": maybe(TEXT),
    **OPTIONAL_STATUS,
}, keys={
    **OPTIONAL_STATUS_KEYS,
    "runtime.shutdown.stopped": STOPPED_PARTS,
    "components.detector.counters": Either((  # tracking.tracker counters; skip and failure labels are content
        vocab("processed", "epoch_resets", "failed", "failure_resets", "boxes_dropped_small",
              "boxes_dropped_overflow"), template(r"skipped_([a-z0-9_]+)", r"failed_([a-z0-9_]+)"))),
    "components.detector.tracking.problems": vocab(*TRACKING_PROBLEMS),  # session 79
    "components.scene.lane": vocab(  # scene.lane counters (JobPurpose: periodic, enrichment)
        "submitted_periodic", "submitted_enrichment", "enrichment_replaced", "result_for_unknown_job", "late_result",
        "enrichment_skipped", "submit_failed", "cancel_failed", "timed_out_job_forgotten",
        "deadline_missed",  # D77
        "enrichment_skipped_paused"),  # D99
    "components.scene.worker": vocab(  # scene.analyzer counters
        "submitted", "delivered", "image_not_retained", "replaced_waiting", "cancelled_waiting", "cancelled_running",
        "dropped_at_stop", "inbox_dropped",
        # chat's side requests (D74) and their preemption (D77)
        "side_accepted", "side_refused", "side_cancelled", "side_completed", "side_dropped_at_stop",
        "side_callback_failed", "side_preempted", "side_aborted", "jobs_waited_for_side",
        # D81: accepted behind a scene job, and the periodic job's grace
        "side_accepted_behind_scene", "side_grace_started", "side_finished_in_grace", "side_preempted_after_grace"),
    "components.face.results": FACE_COUNTERS,  # runtime face_counters
    "components.face.worker.counters": Either((  # identity.worker counters; an error's class name is content
        vocab("ticks", "skipped_no_person", "replaced", "offered", "abandoned_at_stop", "expired_before_start",
              "started", "results_replaced", "errors", "completed", "faces_detected", "fallback_dropped",
              "faces_returned",
              "yielded_to_comparison", "comparison_outcomes_dropped", "comparison_plan_errors",  # D87
              "discarded_off"),  # D99
        template(r"error:([A-Za-z0-9_]+)"))),  # one per exception class (FaceWorker._process)
    "components.face.worker.comparison.counters": Either((  # identity.comparison counters; failure labels are content
        vocab(*(f"{stage}_{kind}" for stage in ("eligible", "attempted", "completed", "failed")
                for kind in COMPARISON_KINDS), *(d for d in COMPARISON_DISPOSITIONS if d.startswith("skipped_"))),
        template(r"failed:([a-z_]+(?::[A-Za-z0-9_]+)?)"))),
    "components.face.comparison_results": vocab(*(f"{kind}:{d}" for kind in COMPARISON_KINDS
                                                  for d in COMPARISON_DISPOSITIONS)),
    "components.notifications.channels": CHANNELS,
    "components.notifications.worker.counters": vocab("sent", "retry", "dead", "late_outcome", "lease_expired"),
    "delivery.totals": vocab("queued", "attempted", "delivered", "failed", "ambiguous"),
})

F2_PROVENANCE = KeyValueProfile({
    "f2_launch_utc": ISO, "boot_id": UUID, "commit": HEX40, "pin_commit": HEX40, "run": RUN_NAME,
    "f1_gate": vocab("passed"), "f1_basis": vocab("original", "accepted_recheck"), "f1_recheck": form(RECHECK_NAME),
    "f1_accepted_utc": ISO, "f1_identity_id": IDN, "account_sha256": HEX64,
    # D91's framing record: the capture block's fixed line, and the answer an evaluation block accepts before writing
    "framing_baseline": vocab("this_capture"), "framing_since_capture": vocab("none", "changed", "unsure"),
    # D91's evaluation sessions (experiment_launch.sh): the room light answered before the launch, and the capture gate
    "lighting": vocab("daylight", "lamps", "both", "unsure"), "capture_gate": vocab("passed"),
    "capture_session": form(SESSION_NAME), "capture_basis": vocab("original", "accepted_recheck"),
    "capture_recheck": form(CAPTURE_RECHECK_NAME), "capture_accepted_utc": ISO,
})
RUN_ERR = LogProfile((
    re.compile(r"\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\] \[TRT\] \[[VIWEF]\] (.*)"),  # TensorRT's logger
    re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+: [IWEF] (.*)"),  # TensorFlow's C++ logger
    re.compile(r"[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ (.*)"),  # abseil's logger
))
ACCOUNT = AccountProfile()

# D80: alpha_load's records in a combined run (chat questions and a live viewer): numbers and fixed labels only; the
# answers' text is never recorded. Every reason the runtime can give a question is declared as fixed text.
CHAT_JOB_REASON = Either((CHAT_REASON, template(
    fixed("another question is being answered; ask again when it is done"), r"one question every ([\d.]+) s",
    fixed("no frame has been received yet, so there is nothing to ask about"),
    fixed("the newest frame's stream has ended; ask again when the camera is live"),
    fixed("the newest frame's stream is no longer connected; ask again when the camera is live"),
    r"the newest frame is ([\d.]+) s old \(stale after ([\d.]+) s\); ask again when the camera is live",
    fixed("the scene model is busy; monitoring comes first, so ask again shortly"),
    fixed("the scene model stayed busy; monitoring comes first"), fixed("no answer in time"),
    fixed("a scene job is waiting: monitoring comes first"), fixed("device memory pressure: monitoring comes first"),
    fixed("the runtime is stopping"), fixed("cancelled"), fixed("the model did not answer in time"),
    r"the model could not be reached \(([A-Za-z0-9_:]+)\)", fixed("the model's answer was not usable"),
    fixed("the model gave an empty answer"), r"the frame could not be encoded \(([A-Za-z0-9_]+)\)",
    fixed("the question could not be answered"), fixed("only this page may ask"),
    fixed("a Content-Length is required"), fixed("a JSON body is required"), fixed("request too large"),
    fixed("request is not JSON"), fixed("request has unexpected fields"), fixed("question missing"),
    fixed("question empty"), r"question longer than (\d+) characters",
    r"history must be a list of at most (\d+) exchanges", fixed("history items need a question and an answer"),
    fixed("the runtime is starting"), fixed("the question could not be taken"))))
CHAT_FINAL = vocab("answered", "error", "timeout", "busy", "cancelled", "unavailable", "paused", "no_fresh_frame",
                   "refused", "invalid", "off", "unknown", "unreadable")  # alpha_load.FINAL and its own unreadable
LOAD_EVENT = {"event": vocab("started", "page_never_answered", "page_gone", "stopped_asking"), "mono_s": NUM,
              "questions.short": template(fixed("What is visible in this picture? Answer in one sentence.")),
              "questions.long": template(fixed("Describe everything visible in this picture in detail, object by "
                                               "object, with its colour and where it is.")), "every_s": NUM}
CHAT_LOAD = JsonProfile({
    **LOAD_EVENT, "asked_mono_s": NUM, "kind": vocab("short", "long"), "http_status": INT, "state": CHAT_FINAL,
    "reason": maybe(CHAT_JOB_REASON),
    "stream_epoch": COUNT, "frame_seq": COUNT, "ingest_utc": maybe(ISO), "age_ms_when_asked": COUNT,
    "age_ms_when_answered": COUNT, "historical": maybe(BOOL), "answer_ms": COUNT, "answer_chars": COUNT,
    "cut_short": maybe(BOOL),
}, lines=True)
VIEWER = JsonProfile({
    **LOAD_EVENT, "second_mono_s": NUM, "pictures": INT, "bytes": INT, "max_age_ms_when_sent": COUNT,
    "states.*": INT,
}, keys={"states": Either((vocab("live", "stale", "offline", "starting", "closed", "busy", "refused", "invalid",
                                 "off", "error"), template(r"http_(\d{3})")))}, lines=True)

F1_PROFILES: dict[str, Any] = {"dryrun.json": ENROLL, "enroll.json": ENROLL, "list.json": LISTING,
                               "provenance.txt": F1_PROVENANCE, "audit.jsonl": AUDIT}
F2_PROFILES: dict[str, Any] = {"result.json": RESULT, "guard.jsonl": GUARD, "run.jsonl": RUN, "run.err": RUN_ERR,
                               "account.txt": ACCOUNT, "provenance.txt": F2_PROVENANCE,
                               "chat-load.jsonl": CHAT_LOAD, "viewer.jsonl": VIEWER,  # a combined run's load (D80)
                               **{f"status-{name}.json": STATUS for name in STATUS_NAMES}}
PROFILES = {"f1": F1_PROFILES, "f2": F2_PROFILES}


# ---------------------------------------------------------------- reading


@dataclass
class Reading:
    """One file read against its profile: its content and its nonconforming parts, by location only."""

    content: list[tuple[str, str]] = field(default_factory=list)  # (location, text)
    nonconforming: list[str] = field(default_factory=list)  # locations
    decoded: list[str] = field(default_factory=list)  # JSON strings whose encoding differs from their text

    def bad(self, where: str, node: Any) -> None:
        self.nonconforming.append(where)
        self.content.extend((where, text) for text in _texts(node))


def _texts(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _texts(value)
    elif isinstance(node, (list, tuple)):  # tuples: a duplicate key's (key, value) pairs
        for value in node:
            yield from _texts(value)
    elif isinstance(node, str):
        yield node


class _Duplicate(dict):
    """An object that had a key twice: every pair is kept, so nothing is hidden by the last one."""

    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__(pairs)
        self.pairs = pairs


def _object(pairs: list[tuple[str, Any]]) -> dict:
    return _Duplicate(pairs) if len({k for k, _ in pairs}) != len(pairs) else dict(pairs)


def _strings(node: Any, out: list[str]) -> None:
    for text in _texts(node):
        if json.dumps(text)[1:-1] != text:
            out.append(text)


def decoded_strings(text: str, lines: bool) -> list[str]:
    """For files read without a profile: JSON strings whose encoding differs from their text."""
    out: list[str] = []
    for chunk in text.splitlines() if lines else [text]:
        try:
            _strings(json.loads(chunk), out)
        except ValueError:
            continue
    return out


def _walk(profile: JsonProfile, node: Any, path: str, where: Callable[[str], str], reading: Reading) -> None:
    if isinstance(node, _Duplicate):
        reading.bad(where(f"{path}.<duplicate key>" if path else "<duplicate key>"), node.pairs)
        return
    if isinstance(node, dict):
        if path and path not in profile.containers:
            reading.bad(where(path), node)
            return
        dynamic = profile.keys.get(path)
        for key, child in node.items():
            if dynamic is not None:
                ok, content = dynamic.fit(key)
                child_path = f"{path}.*" if path else "*"
                if not ok:
                    reading.bad(where(f"{path}.<key>" if path else "<key>"), {key: child})
                    continue
                reading.content.extend((where(f"{path}.<key>"), c) for c in content)
            else:
                child_path = f"{path}.{key}" if path else key
                if not profile.known(child_path):
                    reading.bad(where(f"{path}.<unexpected key>" if path else "<unexpected key>"), {key: child})
                    continue
            _walk(profile, child, child_path, where, reading)
    elif isinstance(node, list):
        child_path = f"{path}[]"
        if not profile.known(child_path):
            reading.bad(where(path or "<root>"), node)
            return
        for item in node:
            _walk(profile, item, child_path, where, reading)
    else:
        spec = profile.fields.get(path)
        ok, content = spec.fit(node) if spec is not None else (False, [])
        if not ok:
            reading.bad(where(path or "<root>"), node)
            return
        reading.content.extend((where(path), c) for c in content)


def _read_json(profile: JsonProfile, text: str, reading: Reading) -> None:
    documents = []
    if profile.lines:
        for number, line in enumerate(text.splitlines(), start=1):
            documents.append((lambda p, n=number: f"line {n}: {p}", line))
    else:
        documents.append((lambda p: p, text))
    for where, chunk in documents:
        try:
            node = json.loads(chunk, object_pairs_hook=_object)
        except ValueError:
            reading.bad(where("<not JSON>"), chunk)
            continue
        if not isinstance(node, dict):
            reading.bad(where("<root is not an object>"), node)
            continue
        _strings(node, reading.decoded)
        _walk(profile, node, "", where, reading)


def _read_key_values(profile: KeyValueProfile, text: str, reading: Reading) -> None:
    for number, line in enumerate(text.splitlines(), start=1):
        key, sep, value = line.partition("=")
        spec = profile.fields.get(key) if sep else None
        ok, content = spec.fit(value) if spec is not None else (False, [])
        if not ok:
            reading.bad(f"line {number}" + (f": {key}" if spec is not None else ""), line)
        else:
            reading.content.extend((f"line {number}: {key}", c) for c in content)


def _read_account(text: str, reading: Reading) -> None:
    lines = text.splitlines()
    known = face_account.account_format(text)
    if known is None:
        reading.bad("line 1: header", lines[0] if lines else "")
    fields = {f.key: f for f in (known[1] if known else face_account.FIELDS)}
    for number, line in enumerate(lines[1:], start=2):
        key, sep, rest = line.partition("=")
        item = fields.get(key) if sep else None
        suffix = f"   # {item.comment}" if item is not None and item.comment else ""
        if item is None or not rest.endswith(suffix):
            reading.bad(f"line {number}", line)
            continue
        value = rest[:len(rest) - len(suffix)] if suffix else rest
        if item.kind == "text" or face_account.accepted(item, value)[0] != value:
            reading.content.append((f"line {number}: {key}", value))  # notes, or an answer as typed: content


def _read_log(profile: LogProfile, text: str, reading: Reading) -> None:
    for number, line in enumerate(text.splitlines(), start=1):
        message = next((m.group(1) for m in (f.fullmatch(line) for f in profile.formats) if m), line)
        if message:
            reading.content.append((f"line {number}", message))


def read(path: Path, part: str, name: str | None = None) -> Reading:
    """Read one evidence file of F1 or F2 (``part``) against the profile for its name."""
    reading = Reading()
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        reading.bad("<not UTF-8>", data.decode("utf-8", errors="replace"))
        return reading
    profile = PROFILES[part].get(name or path.name)
    if isinstance(profile, JsonProfile):
        _read_json(profile, text, reading)
    elif isinstance(profile, KeyValueProfile):
        _read_key_values(profile, text, reading)
    elif isinstance(profile, AccountProfile):
        _read_account(text, reading)
    elif isinstance(profile, LogProfile):
        _read_log(profile, text, reading)
    else:
        reading.bad("<no profile for this file>", text)
    return reading
