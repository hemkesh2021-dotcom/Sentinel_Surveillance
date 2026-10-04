"""Adapter manifests and the built-in adapter registry (guide chapter 27; V2-49).

Optional capabilities (scene analysis, detectors, face recognition, notifiers)
are adapters. Configuration declares each one with a versioned manifest;
the implementation must come from BUILTIN_ADAPTERS, a static table in this
code base. Configuration names an adapter ID, never a module, so it cannot load
arbitrary code.

Three levels of failure:

- An invalid manifest (unknown major contract version, duplicate ID, malformed
  field) is invalid configuration, and nothing starts.
- A valid manifest that cannot be honoured (not built in, kinds or version
  the implementation does not support, unknown resource profile) leaves that
  adapter UNAVAILABLE with a visible reason while core monitoring runs.
- Enumerating and resolving adapters imports nothing. An adapter module, and
  any ML framework it needs, is imported only by ``load()`` for an enabled
  adapter. ``load()`` turns an import error into UNAVAILABLE, but an import
  that aborts the whole process (as TensorRT does when it resolves the wrong
  libcuda, see U13) cannot be caught in-process: heavy adapters belong in a
  worker process (V2-54).

Evidence from an adapter is untrusted: ``evidence_from_adapter()`` parses it
strictly (an unknown evidence contract version is rejected) and checks it
against the adapter's manifest.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from .contracts import Evidence, EvidenceKind, Identifier
from .redaction import redact_line

SUPPORTED_CONTRACT_VERSIONS = frozenset({1})

Revision = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Kinds = Annotated[list[EvidenceKind], Field(min_length=1, max_length=8)]


class AdapterManifest(BaseModel):
    """One adapter as configured (guide ch. 27 fields)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    adapter_id: Identifier
    contract_version: int
    implementation_revision: Revision
    enabled: bool = False
    input_kinds: Kinds
    output_kinds: Kinds
    model_revision: Revision | None = None
    resource_profile_id: Identifier | None = None
    timeout_ms: Annotated[int, Field(ge=1, le=600_000)]
    priority: Annotated[int, Field(ge=0, le=100)] = 50

    @field_validator("contract_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value not in SUPPORTED_CONTRACT_VERSIONS:
            supported = ", ".join(str(v) for v in sorted(SUPPORTED_CONTRACT_VERSIONS))
            raise ValueError(
                f"unsupported adapter contract version {value}; this build supports {supported}"
            )
        return value

    @property
    def producer_revision(self) -> str:
        """The revision its evidence must carry: implementation, plus model if any."""
        if self.model_revision is None:
            return self.implementation_revision
        return f"{self.implementation_revision}+{self.model_revision}"


def check_unique_ids(manifests: Iterable[AdapterManifest]) -> None:
    seen: set[str] = set()
    for manifest in manifests:
        if manifest.adapter_id in seen:
            raise ValueError(f"duplicate adapter_id {manifest.adapter_id!r}")
        seen.add(manifest.adapter_id)


class AdapterRole(str, Enum):
    SCENE_ANALYZER = "scene_analyzer"
    DETECTOR = "detector"
    FACE = "face"
    NOTIFIER = "notifier"

    @property
    def uses_models(self) -> bool:
        """Model adapters need a measured resource profile to be admitted."""
        return self is not AdapterRole.NOTIFIER


@dataclass(frozen=True)
class AdapterSpec:
    """A built-in implementation. ``module`` is imported only by load()."""

    adapter_id: str
    role: AdapterRole
    module: str
    attribute: str
    input_kinds: frozenset[str]
    output_kinds: frozenset[str]
    contract_versions: frozenset[int] = SUPPORTED_CONTRACT_VERSIONS
    # Needs an ACCEPTED combined profile that passed the step-4 criteria, not just a known one (D46).
    requires_accepted_profile: bool = False


# Real adapters join with their packages (face V2-25, notifiers V2-15 still to
# come). Keep this table explicit.
BUILTIN_ADAPTERS: Mapping[str, AdapterSpec] = {
    # V2-26 demo form: LFM2-VL-1.6B on llama-server, loopback only, prompt cache off (D41, D42).
    "llama-lfm2-vl-scene": AdapterSpec(
        adapter_id="llama-lfm2-vl-scene",
        role=AdapterRole.SCENE_ANALYZER,
        module="sentinel.scene.llama_server",
        attribute="LlamaSceneRequest",
        input_kinds=frozenset({"frame"}),
        output_kinds=frozenset({"scene.report"}),
        requires_accepted_profile=True,
    ),
    # V2-09/V2-10 demo form: v1's engine through Ultralytics track() with ByteTrack.
    "legacy-yolov8n-bytetrack": AdapterSpec(
        adapter_id="legacy-yolov8n-bytetrack",
        role=AdapterRole.DETECTOR,
        module="sentinel.inference.legacy_ultralytics",
        attribute="LegacyUltralyticsTracker",
        input_kinds=frozenset({"frame"}),
        output_kinds=frozenset({"person.track"}),
    ),
}

# ---------------------------------------------------------------- resource profiles (guide ch. 27; D28, D33, D46)


class ProfileStatus(str, Enum):
    PROVISIONAL = "provisional"  # admitted for the demo with its limits recorded (D28, D33); never scene admission
    ACCEPTED = "accepted"  # passed the predeclared step-4 criteria; added by a maintainer-approved commit
    PENDING = "pending"  # measured or planned, not yet judged
    FAILED = "failed"  # did not pass its criteria


@dataclass(frozen=True)
class FileFacts:
    """A model file as the profiling run's manifest recorded it (``files.<label>``)."""

    name: str
    bytes: int
    mtime_utc: str  # ISO 8601, whole seconds, UTC
    sha256: str | None = None  # from the manifest's ``sha256``; not recomputed at startup


@dataclass(frozen=True)
class ResourceProfile:
    """One measured combined run of the demo model components, copied from its run directory.

    Entries are static code. Nothing at runtime creates, edits or accepts one:
    a profile becomes ACCEPTED only through a separate, maintainer-approved
    commit that adds its entry with the run directory, commit, boot ID and its
    recorded pass against STEP4_CRITERIA_ID (status record, D46).
    """

    profile_id: str
    status: ProfileStatus
    run_dir: str  # the run's directory name under ~/sentinel-runs/
    commit: str  # repository commit of the run (manifest ``repository.commit``)
    boot_id: str  # manifest ``boot_id``
    llama_flags: tuple[str, ...]  # manifest ``llama_server.flags``
    cache_ram_mib: int | None  # manifest ``llama_server.cache_ram_mib``; None means the server default (cache on)
    scene_interval_s: float  # manifest ``parameters.scene_interval_s``
    llm: FileFacts
    mmproj: FileFacts
    engine_sha256: str  # manifest ``sha256.engine``
    criteria_id: str | None = None  # the predeclared criteria it was judged against
    criteria_passed: bool | None = None  # the maintainer's recorded judgment of every criterion
    # Copied from the step-4 report (operator_check step4 ``criteria``) and profile.json; checked again here.
    gpu_guard_ok: bool | None = None  # D27: every layer and the vision encoder on CUDA0, L4T libcuda, workload cuInit 0
    cache_verdict: str | None = None  # must be "disabled_verified" (manifest, running command line, build, log, 0 updates)
    steady_status: str | None = None  # "complete": monotonic boundaries, teardown excluded
    steady_coverage: float | None = None
    steady_max_bytes: int | None = None  # every steady sample, decimal bytes
    steady_seconds_above_target: float | None = None  # time above 5,000,000,000 B in the steady interval
    peak_bytes: int | None = None  # sampled cold-load/runtime peak
    steady_slope_bytes_per_min: int | None = None
    unique_fps: float | None = None
    min_window_fps: float | None = None
    schedule_age_p95_ms: float | None = None  # replay scheduling age; not camera-to-result
    schedule_age_p99_ms: float | None = None
    face_hz: float | None = None
    face_errors: int | None = None  # the workload's face ``error_count`` (a total, not the sanitized names)
    scene_attempts: int | None = None
    scene_valid: int | None = None  # strict U20/SceneReport parse; structural validity, not accuracy
    scene_truncated: int | None = None  # completions that did not finish "stop"
    scene_errors: int | None = None  # HTTP, transport and client-timeout errors
    scene_over_deadline: int | None = None  # completions over the 8 s job timeout
    kernel_coverage: str | None = None  # must be "observed"
    oom_candidates: int | None = None
    nvmap_candidates: int | None = None
    identity_status: str | None = None  # "verified": snapshot before the cache drop equals the end-of-run hashes
    replay_clip_verified: bool | None = None  # the clip's hash equals the recorded check 8 clip
    llama_server: FileFacts | None = None  # with sha256
    llama_libraries: tuple[FileFacts, ...] = ()  # each with sha256
    scene_request_sha256: str | None = None
    limitations: tuple[str, ...] = ()  # must include STARTUP_IDENTITY_LIMITATION
    note: str = ""


# Step-4 criteria (D47; the same values as benchmarks/runner/step4_criteria.py, which a test enforces).
# Demo criteria only: they do not establish the guide's 1080p beta gates.
STEP4_CRITERIA_ID = "step4-combined-cache-off-v2"
STEP4_TARGET_STEADY_BYTES = 5_000_000_000
STEP4_MAX_PEAK_BYTES = 5_400_000_000
STEP4_MAX_STEADY_SLOPE_BYTES_PER_MIN = 10_000_000
STEP4_MIN_STEADY_COVERAGE = 0.95
STEP4_MIN_UNIQUE_FPS = 14.5
STEP4_MIN_WINDOW_FPS = 13.5
STEP4_MAX_SCHEDULE_AGE_P95_MS = 150.0
STEP4_MAX_SCHEDULE_AGE_P99_MS = 250.0
STEP4_MIN_FACE_HZ = 0.95
STEP4_MIN_SCENE_ATTEMPTS = 140
STEP4_MIN_SCENE_STRICT_VALID_SHARE = 0.95
# What `sentinel run --scene` checks at startup, and what it cannot (D47). An accepted profile must carry this text.
STARTUP_HASH_LIMIT_BYTES = 32_000_000
STARTUP_IDENTITY_LIMITATION = (
    "startup hashes the llama-server binary and build libraries up to 32,000,000 B; larger libraries "
    "(libggml-cuda) and the model files are checked by name, size and modification time only, so a "
    "same-size replacement that keeps its modification time is not detected at startup (demo limitation)"
)
_RUN_DIR = re.compile(r"^demo-profile-\d{8}T\d{6}Z$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_CHECK8_LLAMA_FLAGS = ("--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1")

RESOURCE_PROFILES: Mapping[str, ResourceProfile] = MappingProxyType({
    # Check 8 (session 5): admitted as the provisional demo profile with its exceedance recorded (D28, D33).
    # Measured with llama-server's default prompt cache, so it cannot admit the scene adapter (D41, D46).
    "provisional-demo-20261003T085010Z": ResourceProfile(
        profile_id="provisional-demo-20261003T085010Z",
        status=ProfileStatus.PROVISIONAL,
        run_dir="demo-profile-20261003T085010Z",
        commit="d85eb1e2d1c9b7fedc2cad5f67f7eb367b415640",
        boot_id="2dfc802c-4b59-4b76-ad53-a7f3a995589e",
        llama_flags=_CHECK8_LLAMA_FLAGS,
        cache_ram_mib=None,
        scene_interval_s=4.0,
        llm=FileFacts("LFM2-VL-1.6B-Q4_0.gguf", 695_750_048, "2026-04-25T18:45:52+00:00",
                      "ce0d4b122d328d14390ef160785da3a51a527f96844f392a04cb2db96f134e5d"),
        mmproj=FileFacts("mmproj-LFM2-VL-1.6B-Q8_0.gguf", 564_115_648, "2026-04-25T18:39:39+00:00",
                         "65ec437db88d65fff93f472d00c145e09880769ac67fedff5cd1c0f8d8301d87"),
        engine_sha256="08370639f961d2c67148c19562718ef80527c7085e88d2d923176180f1b98637",
        note="check 8, face at 2 Hz, default prompt cache; steady median 5.036 GB, p95 5.347 GB, peak 5.350 GB",
    ),
})


def known_profile_ids(profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES) -> frozenset[str]:
    """Profiles that admit model adapters at all: provisional or accepted."""
    return frozenset(
        pid for pid, p in profiles.items() if p.status in (ProfileStatus.PROVISIONAL, ProfileStatus.ACCEPTED)
    )


KNOWN_RESOURCE_PROFILES: frozenset[str] = known_profile_ids()


def accepted_profile_problem(
    profile_id: str | None, profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES
) -> str | None:
    """None if ``profile_id`` is an ACCEPTED combined profile measured with --cache-ram 0 whose
    recorded evidence passes the step-4 criteria; otherwise the first missing prerequisite."""
    if profile_id is None:
        return "the manifest names no resource_profile_id; scene needs an accepted combined profile"
    profile = profiles.get(profile_id)
    if profile is None or profile.profile_id != profile_id:
        return f"resource profile {profile_id} is not in the registry"
    if profile.status is ProfileStatus.PROVISIONAL:
        cache = "the default prompt cache" if profile.cache_ram_mib is None else f"--cache-ram {profile.cache_ram_mib}"
        return (f"resource profile {profile_id} is provisional (measured with {cache}); scene needs an accepted "
                "combined profile measured with --cache-ram 0 (operator checklist step 4)")
    if profile.status is not ProfileStatus.ACCEPTED:
        return f"resource profile {profile_id} is {profile.status.value}, not accepted"
    if profile.cache_ram_mib != 0:
        cache = "the default prompt cache" if profile.cache_ram_mib is None else f"--cache-ram {profile.cache_ram_mib}"
        return f"resource profile {profile_id} was measured with {cache}, not --cache-ram 0"
    if profile.criteria_id != STEP4_CRITERIA_ID:
        return f"resource profile {profile_id} was not judged against {STEP4_CRITERIA_ID}"
    if profile.criteria_passed is not True:
        return f"resource profile {profile_id} has no recorded pass against {STEP4_CRITERIA_ID}"
    if not _RUN_DIR.match(profile.run_dir):
        return f"resource profile {profile_id} lacks its run directory"
    if not _COMMIT.match(profile.commit):
        return f"resource profile {profile_id} lacks its full commit"
    if not _BOOT_ID.match(profile.boot_id):
        return f"resource profile {profile_id} lacks its boot ID"
    if profile.gpu_guard_ok is not True:
        return f"resource profile {profile_id} has no recorded GPU guard pass"
    if profile.cache_verdict != "disabled_verified":
        return f"resource profile {profile_id} has no verified record of the prompt cache being off"
    if profile.steady_status != "complete" or profile.steady_coverage is None or profile.steady_coverage < STEP4_MIN_STEADY_COVERAGE:
        return f"resource profile {profile_id} lacks a complete, covered steady interval"
    checks = (
        ("steady max", profile.steady_max_bytes, STEP4_TARGET_STEADY_BYTES, "max"),
        ("seconds above the steady target", profile.steady_seconds_above_target, 0, "max"),
        ("run peak", profile.peak_bytes, STEP4_MAX_PEAK_BYTES, "max"),
        ("steady slope per minute", profile.steady_slope_bytes_per_min, STEP4_MAX_STEADY_SLOPE_BYTES_PER_MIN, "max"),
        ("unique frames per second", profile.unique_fps, STEP4_MIN_UNIQUE_FPS, "min"),
        ("lowest 10 s window frames per second", profile.min_window_fps, STEP4_MIN_WINDOW_FPS, "min"),
        ("scheduling age p95 ms", profile.schedule_age_p95_ms, STEP4_MAX_SCHEDULE_AGE_P95_MS, "max"),
        ("scheduling age p99 ms", profile.schedule_age_p99_ms, STEP4_MAX_SCHEDULE_AGE_P99_MS, "max"),
        ("face rate Hz", profile.face_hz, STEP4_MIN_FACE_HZ, "min"),
        ("face errors", profile.face_errors, 0, "max"),
        ("scene attempts", profile.scene_attempts, STEP4_MIN_SCENE_ATTEMPTS, "min"),
        ("scene request errors", profile.scene_errors, 0, "max"),
        ("scene completions over the deadline", profile.scene_over_deadline, 0, "max"),
        ("truncated scene completions", profile.scene_truncated, 0, "max"),
        ("OOM candidate lines", profile.oom_candidates, 0, "max"),
        ("NvMap candidate lines", profile.nvmap_candidates, 0, "max"),
    )
    for label, value, limit, kind in checks:
        if value is None:
            return f"resource profile {profile_id} lacks its recorded {label}"
        if (value > limit) if kind == "max" else (value < limit):
            relation = "exceeds" if kind == "max" else "is below"
            return f"resource profile {profile_id} {label} {value} {relation} {limit}"
    if profile.scene_valid is None or profile.scene_valid < STEP4_MIN_SCENE_STRICT_VALID_SHARE * profile.scene_attempts:
        return f"resource profile {profile_id} lacks {STEP4_MIN_SCENE_STRICT_VALID_SHARE:.0%} strictly valid scene reports"
    if profile.kernel_coverage != "observed":
        return f"resource profile {profile_id} has no observed kernel-log coverage"
    if profile.identity_status != "verified" or profile.replay_clip_verified is not True:
        return f"resource profile {profile_id} has no verified identity (files and replay clip)"
    if (profile.llama_server is None or not profile.llama_server.sha256 or not profile.llama_libraries
            or any(not lib.sha256 for lib in profile.llama_libraries)
            or profile.llm.sha256 is None or profile.mmproj.sha256 is None):
        return f"resource profile {profile_id} lacks its llama.cpp build or model hashes"
    if not profile.scene_request_sha256 or not re.fullmatch(r"[0-9a-f]{64}", profile.scene_request_sha256):
        return f"resource profile {profile_id} lacks its scene request fingerprint"
    if STARTUP_IDENTITY_LIMITATION not in profile.limitations:
        return f"resource profile {profile_id} does not carry the startup identity limitation"
    return None


class AdapterState(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"  # switched off in configuration
    UNAVAILABLE = "unavailable"  # configured on, but cannot be used; see reason


@dataclass(frozen=True)
class AdapterStatus:
    manifest: AdapterManifest
    state: AdapterState
    reason: str
    spec: AdapterSpec | None = None


def resolve(
    manifests: Iterable[AdapterManifest],
    *,
    registry: Mapping[str, AdapterSpec] = BUILTIN_ADAPTERS,
    known_profiles: frozenset[str] | None = None,
    profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES,
) -> tuple[AdapterStatus, ...]:
    """Decide each adapter's state without importing anything."""
    known = known_profile_ids(profiles) if known_profiles is None else known_profiles
    statuses = []
    for manifest in manifests:
        statuses.append(_resolve_one(manifest, registry, known, profiles))
    return tuple(statuses)


def _resolve_one(
    manifest: AdapterManifest,
    registry: Mapping[str, AdapterSpec],
    known_profiles: frozenset[str],
    profiles: Mapping[str, ResourceProfile],
) -> AdapterStatus:
    def unavailable(reason: str, spec: AdapterSpec | None = None) -> AdapterStatus:
        return AdapterStatus(manifest, AdapterState.UNAVAILABLE, reason, spec)

    if not manifest.enabled:
        return AdapterStatus(manifest, AdapterState.DISABLED, "disabled in configuration")
    spec = registry.get(manifest.adapter_id)
    if spec is None:
        return unavailable("not a built-in adapter")
    if manifest.contract_version not in spec.contract_versions:
        version = manifest.contract_version
        return unavailable(f"implementation does not support contract version {version}", spec)
    missing_in = set(manifest.input_kinds) - spec.input_kinds
    missing_out = set(manifest.output_kinds) - spec.output_kinds
    if missing_in or missing_out:
        kinds = ", ".join(sorted(missing_in | missing_out))
        return unavailable(f"implementation does not handle kinds: {kinds}", spec)
    if spec.requires_accepted_profile:
        problem = accepted_profile_problem(manifest.resource_profile_id, profiles)
        if problem is not None:
            return unavailable(problem, spec)
    elif spec.role.uses_models and manifest.resource_profile_id not in known_profiles:
        return unavailable("no measured resource profile; unknown profiles are unavailable", spec)
    return AdapterStatus(manifest, AdapterState.ENABLED, "enabled", spec)


def load(status: AdapterStatus) -> tuple[Any | None, AdapterStatus]:
    """Import an enabled adapter's implementation; failure makes it UNAVAILABLE."""
    if status.state is not AdapterState.ENABLED or status.spec is None:
        return None, status
    try:
        module = importlib.import_module(status.spec.module)
        implementation = getattr(module, status.spec.attribute)
    except Exception as exc:  # a broken optional adapter must not stop core monitoring
        reason = redact_line(f"failed to load: {type(exc).__name__}: {exc}")
        return None, AdapterStatus(status.manifest, AdapterState.UNAVAILABLE, reason, status.spec)
    return implementation, status


class AdapterOutputError(ValueError):
    """Adapter output was rejected; the message is safe to log."""


def output_problem(status: AdapterStatus, evidence: Evidence) -> str | None:
    """Why ``evidence`` may not be accepted from this adapter, or None if it may."""
    manifest = status.manifest
    if status.state is not AdapterState.ENABLED:
        return f"adapter {manifest.adapter_id!r} is {status.state.value}"
    if evidence.kind not in manifest.output_kinds:
        return f"kind {evidence.kind!r} is not declared by adapter {manifest.adapter_id!r}"
    if evidence.producer != manifest.adapter_id:
        return f"producer {evidence.producer!r} is not adapter {manifest.adapter_id!r}"
    if evidence.producer_revision != manifest.producer_revision:
        return "producer revision does not match the adapter manifest"
    return None


def evidence_from_adapter(status: AdapterStatus, payload: str | bytes) -> Evidence:
    """Parse serialized evidence from an adapter and check it against its manifest."""
    try:
        evidence = Evidence.model_validate_json(payload)
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        where = redact_line(".".join(str(part) for part in first["loc"]) or "<evidence>", 60)
        raise AdapterOutputError(f"invalid evidence: {where}: {first['type']}") from None
    problem = output_problem(status, evidence)
    if problem is not None:
        raise AdapterOutputError(redact_line(problem))
    return evidence
