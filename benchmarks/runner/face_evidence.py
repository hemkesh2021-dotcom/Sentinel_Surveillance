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
from operator_account import accepted

# Files written after the counts, or the counts themselves; never counted (face_check uses the same set).
NOT_COUNTED = frozenset({"SHA256SUMS", "check.txt", "secret-counts-camera.txt", "secret-counts-identity.txt"})
STATUS_NAMES = ("e1", "k1", "a", "e2", "k2", "end")  # F2c's --status-at names
CUE_NAMES = ("t0", "k1", "a", "e2", "k2", "stop")  # F2c's --cue-at names
RECHECK_NAME = r"face-\d{8}T\d{6}Z-f1-[0-9a-f]{7}-\d{8}T\d{6}Z"  # ~/sentinel-runs/recheck/<run>-f1-<pin>-<UTC>


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
VOTE_LABELS = ("match", "unknown", "margin", "low_quality", "incompatible", "no_enrollment")  # state._NOTE_LABELS
VOTE_NOTES = ("face matched", "face matched nobody", "face matches more than one identity", "face quality too low",
              "embedding from an incompatible model", "no identities enrolled")  # state._NOTE_LABELS' keys
TICKS = vocab("offered", "skipped_no_person", "stopping")  # identity.worker TICK_*
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
STOPPED_PARTS = vocab("capture", "face_worker", "scene_server", "scene_worker", "outbox", "status_page")
CHANNELS = vocab("telegram")
GUARD_STATUS = Either((vocab("duration_stop", "child_exited", "cleanup_failed", "forced_stop", "guard_stop",
                             "ready_timeout", "operator_interrupt"), template(r"refused:(.+)")))
STOP_BY = maybe(vocab("duration_elapsed", "guard_stop", "ready_timeout", "operator_interrupt"))
CAPTURE_RESULT = Either((vocab("saved", "too_large", "not_json", "not_reached"), template(r"unavailable:(.+)")))
LISTENER = {"port": INT, "family": vocab("tcp", "tcp6"), "scope": vocab("loopback", "any", "other")}

PENDING_DURABILITY = template(
    r"rule observations waiting to be recorded are held in memory only \(at most (\d+)\); "
    r"they are lost if the process stops abruptly: not a crash-safe spool")
OCCUPANCY_REASON = template(r"no fresh video \((.+)\)", r"person detector (.+)",
                            fixed("detector has not processed a recent frame"),
                            r"(\d+) confirmed person\(s\) on fresh video", fixed("nobody detected on fresh video"))
SCENE_REASON = template(fixed("scene analysis disabled"), r"scene analysis unavailable: (.+)",
                        fixed("current scene report"), r"last scene check ([^:]+): (.*)", fixed("no scene result yet"),
                        r"last scene result not current \((.+)\)")
DEGRADED = template(r"capture (\S+)(?: \((.+)\))?", r"video (\S+)", r"detector unavailable \((.*)\)",
                    r"scene analysis unavailable \((.*)\)", r"face recognition unavailable \((.*)\)",
                    r"(\d+) rule observation\(s\) not yet recorded \((.*)\)",
                    r"(\d+) rule observation\(s\) lost: not recorded", r"(\S+) unavailable \((.*)\); its alerts stay "
                    r"queued", r"delivery worker problem \((.*)\)", fixed("delivery worker stopped"))
TRANSITION_REASON = template(*(fixed(t) for t in (
    *VOTE_NOTES, "no face visible", "face ownership ambiguous", *OWNERSHIP, "no face observed yet",
    "consistent faces matched nobody", "contradictory face evidence", "track no longer current")),
    r"(\d+) consistent matches")
INCIDENT_TITLE = template(r"Person in restricted zone '(.*)'", r"Person dwelling in zone '(.*)'",
                          fixed("Possible fire or smoke (scene model only, unconfirmed)"))


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
    """F2's account.txt, as face_account renders it: its header, then ``key=value   # comment`` per field."""


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
                             "close_to_enrolled_identity"), template(r"gallery_incompatible:(.+)"))),
    "identity_id": IDN, "verified": BOOL, "photos_deleted": INT, "photos_changed_not_deleted": INT,
    "folder_entries_left": INT,
})
LISTING = JsonProfile({
    "identity": vocab("list"), "identities[].identity_id": IDN, "identities[].prototypes": INT,
    "identities[].consent_date": DATE, "identities[].enrolled_utc": ISO,
})
AUDIT = JsonProfile({
    "utc": ISO, "action": vocab("enroll", "revoke", "purge"), "identity_id": maybe(IDN), "photos_used": INT,
    "photos_rejected": INT, "identities_left": INT, "gallery": INT, "inbox_files": INT,
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
    "status_page": maybe(LOOPBACK_URL),
}
STARTUP_KEYS = {"scene_memory_before": MEMFREE_KEYS, "detector_memory_before": MEMFREE_KEYS,
                "face_memory_before": MEMFREE_KEYS, "releases": RELEASE_COMPONENTS, "releases.*.files": RELEASE_ROLES,
                "notifier_problems": CHANNELS}
STATUS_LINE = {  # cli.status_line
    "updated_utc": ISO, "state": RUNTIME, "video": maybe(VIDEO), "occupancy": maybe(OCCUPANCY),
    "scene": maybe(SCENE_STATUS), "identities": NULL,
    **{f"identities.{k}": INT for k in (*IDENTITY_STATES, "fresh", "retained")},
    **{f"rates.{k}": SECONDS for k in ("window_s", "captured_fps", "processed_fps", "failed_per_s")},
    "capture.state": CAPTURE, "capture.stream_epoch": COUNT, "capture.reconnects": INT, "pending_signals": INT,
    "degraded[]": DEGRADED,
}
IDENTITY_RECORDS = {  # runtime.tick_record, FaceResultRecord, IdentityTransition
    "identity": vocab("tick", "result", "transition"), "schema": INT, "stream_epoch": INT, "frame_seq": INT,
    "frame_mono_ns": INT, "tick": TICKS, "tracks[]": INT, "applied_mono_ns": INT, "outcome": OUTCOMES,
    "faces": COUNT, "processing_ms": SECONDS, "error": maybe(TEXT), "persons[].track_id": INT,
    "persons[].ownership": vocab(*OWNERSHIP), "persons[].vote": vocab("none", "unknown", "match"),
    "persons[].identity_id": maybe(IDN), "persons[].label": vocab(*VOTE_LABELS, *OWNERSHIP),
    "track_id": INT, "from": vocab(*IDENTITY_STATES), "to": vocab(*IDENTITY_STATES, "cleared"),
    "identity_id": maybe(IDN), "basis": maybe(BASIS), "last_vote_age_ms": COUNT, "reason": TRANSITION_REASON,
    "mono_ns": INT,
}
RUN = JsonProfile({
    "run": vocab("starting", "stopped"), **nest("startup", STARTUP),
    **STATUS_LINE, **IDENTITY_RECORDS,
    "shutdown.stopped.*": BOOL, "shutdown.all_stopped": BOOL, "shutdown.signals_not_recorded": INT,
    "database_closed": BOOL, **nest("status", STATUS_LINE),
}, keys={**{f"startup.{k}": v for k, v in STARTUP_KEYS.items()}, "shutdown.stopped": STOPPED_PARTS}, lines=True)

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
    "face": NULL, "face.validation_run": BOOL, "face.identities_enrolled": INT,
}
STARTING_SUMMARY_KEYS = {"releases": RELEASE_COMPONENTS, "releases.*": RELEASE_ROLES, "notifier_problems": CHANNELS,
                         "scene_memory_before": MEMFREE_KEYS, "detector_memory_before": MEMFREE_KEYS,
                         "face_memory_before": MEMFREE_KEYS}
RESULT = JsonProfile({  # step5_guard.supervise
    "schema_version": INT, "mode": vocab("step5_guard"), "status": GUARD_STATUS,
    "parameters.duration_s": NUM, "parameters.from_launch": BOOL, "parameters.ready_timeout_s": NUM,
    "parameters.stop_grace_s": NUM, "parameters.step4_headroom": BOOL,
    "parameters.status_at[].at_s": NUM, "parameters.status_at[].name": vocab(*STATUS_NAMES),
    "parameters.cues[].at_s": NUM, "parameters.cues[].name": vocab(*CUE_NAMES),
    "limits.pressure_stop_bytes": INT, "limits.mem_free_floor_bytes": INT, "limits.mem_available_floor_bytes": INT,
    "limits.swap_counters": vocab("unchanged"), "limits.max_sample_gap_s": NUM,
    "preflight_sample.*": maybe(NUM), "baseline.*": maybe(NUM),
    "stop.requested_by": STOP_BY, "stop.requested_at_s": SECONDS, "stop.exited_after_stop_s": SECONDS,
    "stop.forced": BOOL,
    "child.launched_utc": maybe(ISO), "child.exited_utc": maybe(ISO), "child.returncode": COUNT,
    "child.ran_s": SECONDS, "child.ready_after_s": SECONDS, "child.ready_mono_s": SECONDS,
    "guard.samples": INT, "guard.largest_gap_s": SECONDS, "guard.peak_pressure_bytes": COUNT,
    "guard.min_mem_free_bytes": COUNT, "guard.min_mem_available_bytes": COUNT, "guard.trigger": maybe(TEXT),
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
}, keys={"preflight_sample": SAMPLE_KEYS, "baseline": SAMPLE_KEYS,
         **{f"run_output.starting.{k}": v for k, v in STARTING_SUMMARY_KEYS.items()},
         "run_output.stopped.stopped": STOPPED_PARTS})

FACE_COUNTERS = vocab(*(f"persons_{o}" for o in OWNERSHIP), *(f"faces_{o}" for o in OWNERSHIP),
                      *(f"results_{o}" for o in sorted(OUTCOMES.values)), "votes_match", "votes_unknown",
                      "no_vote_margin", "no_vote_low_quality", "no_vote_incompatible", "no_vote_no_enrollment")
STATUS = JsonProfile({  # demo_runtime.DemoRuntime.snapshot through status_page.build_status
    "generated_utc": ISO,
    "runtime.camera_id": TEXT, "runtime.state": RUNTIME, "runtime.updated_utc": ISO, "runtime.status_age_s": SECONDS,
    "runtime.shutdown": NULL, "runtime.shutdown.stopped.*": BOOL, "runtime.shutdown.all_stopped": BOOL,
    "runtime.shutdown.signals_not_recorded": INT,
    "live": NULL, "live.sequence": INT, "live.video": VIDEO, "live.last_frame_age_ms": COUNT,
    "live.detector": CAPABILITY, "live.face_recognition": CAPABILITY, "live.scene_analysis": CAPABILITY,
    "live.occupancy": OCCUPANCY, "live.occupancy_reason": OCCUPANCY_REASON, "live.people": INT,
    "live.confirmed_people": INT, **{f"live.identity.{k}": INT for k in (*IDENTITY_STATES, "fresh", "retained")},
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
    "components.scene.state": CAPABILITY, "components.scene.problem": maybe(TEXT),
    **nest("components.scene.server", SCENE_SERVER), "components.scene.lane.*": INT, "components.scene.worker.*": INT,
    "components.face.state": CAPABILITY, "components.face.problem": maybe(TEXT),
    "components.face.identities_enrolled": INT, "components.face.validation_run": BOOL,
    **{f"components.face.records.{k}": INT for k in ("kept", "dropped", "limit")},
    "components.face.results.*": INT, "components.face.worker.state": WORKER, "components.face.worker.busy": BOOL,
    "components.face.worker.waiting": BOOL, "components.face.worker.counters.*": INT,
    **{f"components.face.worker.processing_ms.{k}": SECONDS for k in ("n", "p50", "p95", "p99", "max")},
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
}, keys={
    "runtime.shutdown.stopped": STOPPED_PARTS,
    "components.detector.counters": Either((  # tracking.tracker counters; skip and failure labels are content
        vocab("processed", "epoch_resets", "failed", "failure_resets", "boxes_dropped_small",
              "boxes_dropped_overflow"), template(r"skipped_([a-z0-9_]+)", r"failed_([a-z0-9_]+)"))),
    "components.scene.lane": vocab(  # scene.lane counters (JobPurpose: periodic, enrichment)
        "submitted_periodic", "submitted_enrichment", "enrichment_replaced", "result_for_unknown_job", "late_result",
        "enrichment_skipped", "submit_failed", "cancel_failed", "timed_out_job_forgotten"),
    "components.scene.worker": vocab(  # scene.analyzer counters
        "submitted", "delivered", "image_not_retained", "replaced_waiting", "cancelled_waiting", "cancelled_running",
        "dropped_at_stop", "inbox_dropped"),
    "components.face.results": FACE_COUNTERS,  # runtime face_counters
    "components.face.worker.counters": vocab(  # identity.worker counters
        "ticks", "skipped_no_person", "replaced", "offered", "abandoned_at_stop", "expired_before_start", "started",
        "results_replaced", "errors", "completed", "faces_detected", "fallback_dropped", "faces_returned"),
    "components.notifications.channels": CHANNELS,
    "components.notifications.worker.counters": vocab("sent", "retry", "dead", "late_outcome", "lease_expired"),
    "delivery.totals": vocab("queued", "attempted", "delivered", "failed", "ambiguous"),
})

F2_PROVENANCE = KeyValueProfile({
    "f2_launch_utc": ISO, "boot_id": UUID, "commit": HEX40, "pin_commit": HEX40, "run": RUN_NAME,
    "f1_gate": vocab("passed"), "f1_basis": vocab("original", "accepted_recheck"), "f1_recheck": form(RECHECK_NAME),
    "f1_accepted_utc": ISO, "f1_identity_id": IDN, "account_sha256": HEX64,
})
RUN_ERR = LogProfile((
    re.compile(r"\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\] \[TRT\] \[[VIWEF]\] (.*)"),  # TensorRT's logger
    re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+: [IWEF] (.*)"),  # TensorFlow's C++ logger
    re.compile(r"[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ (.*)"),  # abseil's logger
))
ACCOUNT = AccountProfile()

F1_PROFILES: dict[str, Any] = {"dryrun.json": ENROLL, "enroll.json": ENROLL, "list.json": LISTING,
                               "provenance.txt": F1_PROVENANCE, "audit.jsonl": AUDIT}
F2_PROFILES: dict[str, Any] = {"result.json": RESULT, "guard.jsonl": GUARD, "run.jsonl": RUN, "run.err": RUN_ERR,
                               "account.txt": ACCOUNT, "provenance.txt": F2_PROVENANCE,
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
    if not lines or lines[0] != face_account.HEADER:
        reading.bad("line 1: header", lines[0] if lines else "")
    fields = {f.key: f for f in face_account.FIELDS}
    for number, line in enumerate(lines[1:], start=2):
        key, sep, rest = line.partition("=")
        item = fields.get(key) if sep else None
        suffix = f"   # {item.comment}" if item is not None and item.comment else ""
        if item is None or not rest.endswith(suffix):
            reading.bad(f"line {number}", line)
            continue
        value = rest[:len(rest) - len(suffix)] if suffix else rest
        if item.kind == "text" or accepted(item, value)[0] != value:
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
