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
from .memory_policy import CANDIDATE_POLICY, DEFAULT_POLICY, RELEASE_POST_LOAD, THP_SYSTEM, MemoryPolicy
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
    # Needs its own face admission record keyed to an accepted profile (V2-25 demo form), never a profile alone.
    requires_face_admission: bool = False


FACE_ADAPTER_ID = "legacy-deepface-facenet512-yunet"

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
    # V2-25 demo form: the face path the accepted replay profile measured (YuNet + Facenet512, TensorFlow on the CPU,
    # 1 Hz). Admitted only through a FACE_ADMISSIONS record, which starts PENDING_VALIDATION.
    FACE_ADAPTER_ID: AdapterSpec(
        adapter_id=FACE_ADAPTER_ID,
        role=AdapterRole.FACE,
        module="sentinel.identity.legacy_deepface",
        attribute="LegacyDeepFaceBackend",
        input_kinds=frozenset({"frame"}),
        output_kinds=frozenset({"face.observation"}),
        requires_face_admission=True,
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
    recorded pass against an admissible identity (status record, D46, D59).
    Each candidate result needs its own such commit; none is admitted by its
    identity alone.
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
    # D58: the memory policy the run measured (manifest ``memory_policy``; earlier runs: the default, which is what
    # they ran). It must be the one its criteria identity's procedure measures, and the runtime's selected one.
    memory_policy: MemoryPolicy = DEFAULT_POLICY
    # D59: the run's evidence that a non-default policy held (the step-4 report's R inputs): every release recorded
    # and returned 0 (post_load); the disable verified before the detector load, THP_enabled as expected at every
    # checkpoint, no AnonHugePages above the value at the check and the THP settings unchanged (workload_disabled).
    memory_policy_verified: bool | None = None


# Step-4 criteria (D47; the same values as benchmarks/runner/step4_criteria.py, which a test enforces).
# Demo criteria only: they do not establish the guide's 1080p beta gates.
STEP4_CRITERIA_ID = "step4-combined-cache-off-v2"
# D54's PLR variant and D58's candidate: the same rules and thresholds under their own identities; each identity's
# procedure measures one memory policy.
STEP4PLR_CRITERIA_ID = "step4plr-combined-cache-off-v2"
STEP4_CANDIDATE_CRITERIA_ID = "step4cand-wtd-plr-combined-cache-off-v2"
CRITERIA_MEMORY_POLICIES: Mapping[str, MemoryPolicy] = MappingProxyType({
    STEP4_CRITERIA_ID: DEFAULT_POLICY,
    STEP4PLR_CRITERIA_ID: MemoryPolicy(THP_SYSTEM, RELEASE_POST_LOAD),
    STEP4_CANDIDATE_CRITERIA_ID: CANDIDATE_POLICY,
})
# D59: the identities whose accepted profile can admit scene analysis. Each admits only with the memory policy its
# procedure measures, and `sentinel run` must select and establish that policy (profile_mismatch, assemble).
# The PLR identity (D54) and every diagnostic stay non-admissible.
SCENE_ADMISSIBLE_CRITERIA_IDS: tuple[str, ...] = (STEP4_CRITERIA_ID, STEP4_CANDIDATE_CRITERIA_ID)
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
# D59: what an accepted candidate-identity profile does not measure about `sentinel run`. Such a profile must carry it.
CANDIDATE_RUNTIME_LIMITATION = (
    "measured by the profiler's replay workload, not by sentinel run: the runtime applies the same memory policy "
    "through the same code (THP disabled in the detector process alone, verified; each loaded model's files released "
    "after a 15 s settle) but runs no face model and releases no face files, adds camera capture, rules, incidents, "
    "the outbox and the status page, spawns llama-server as its child (the profiler: a sibling; both before the "
    "disable), waits no 5 s after a release, and has no steady-end THP checkpoint or THP-settings read; measured "
    "headless with no dev tools, over one 600 s steady interval on one boot (demo limitation)"
)
_RUN_DIR = re.compile(r"^demo-profile-\d{8}T\d{6}Z$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_CHECK8_LLAMA_FLAGS = ("--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1")
_STEP4_LLAMA_FLAGS = _CHECK8_LLAMA_FLAGS + ("--cache-ram", "0")

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
    # The D58 candidate run (session 46), accepted by the maintainer under D59 (session 47): it admits scene analysis
    # only for `sentinel run --scene --workload-thp-disable --post-load-release`, which must establish both policies.
    "step4cand-demo-20261007T090339Z": ResourceProfile(
        profile_id="step4cand-demo-20261007T090339Z",
        status=ProfileStatus.ACCEPTED,
        run_dir="demo-profile-20261007T090339Z",
        commit="269de8670f117c8a72e725a5b799ec53372250e5",
        boot_id="83d3fc26-6543-4697-834f-924a23152605",
        llama_flags=_STEP4_LLAMA_FLAGS,
        cache_ram_mib=0,
        scene_interval_s=4.0,
        llm=FileFacts("LFM2-VL-1.6B-Q4_0.gguf", 695_750_048, "2026-04-25T18:45:52+00:00",
                      "ce0d4b122d328d14390ef160785da3a51a527f96844f392a04cb2db96f134e5d"),
        mmproj=FileFacts("mmproj-LFM2-VL-1.6B-Q8_0.gguf", 564_115_648, "2026-04-25T18:39:39+00:00",
                         "65ec437db88d65fff93f472d00c145e09880769ac67fedff5cd1c0f8d8301d87"),
        engine_sha256="08370639f961d2c67148c19562718ef80527c7085e88d2d923176180f1b98637",
        criteria_id=STEP4_CANDIDATE_CRITERIA_ID,
        criteria_passed=True,
        gpu_guard_ok=True,
        cache_verdict="disabled_verified",
        steady_status="complete",
        steady_coverage=0.9993,
        steady_max_bytes=4_297_113_600,
        steady_seconds_above_target=0.0,
        peak_bytes=4_297_113_600,
        steady_slope_bytes_per_min=4_024_059,
        unique_fps=15.0,
        min_window_fps=15.0,
        schedule_age_p95_ms=59.1,
        schedule_age_p99_ms=72.5,
        face_hz=1.0,
        face_errors=0,
        scene_attempts=150,
        scene_valid=150,
        scene_truncated=0,
        scene_errors=0,
        scene_over_deadline=0,
        kernel_coverage="observed",
        oom_candidates=0,
        nvmap_candidates=0,
        identity_status="verified",
        replay_clip_verified=True,
        llama_server=FileFacts("llama-server", 9_080_480, "2026-04-25T19:41:59+00:00",
                              "3d6cbfe061043d6c3bf09e20b61f998cbf80e23a5c59e94049e9142269bc0f85"),
        llama_libraries=(
            FileFacts("libggml-base.so.0.10.0", 805_336, "2026-04-25T19:21:01+00:00",
                      "afc19920f759337f8f983c09263f66bf7e5cbbd2cc4f99c64ad729fcb8b5a155"),
            FileFacts("libggml-cpu.so.0.10.0", 995_032, "2026-04-25T19:21:22+00:00",
                      "b6b3a68a0d1c06a0f2f457469f6d39be848d2276e3e5061edb43646d4c2a6545"),
            FileFacts("libggml-cuda.so.0.10.0", 199_749_520, "2026-04-25T19:35:35+00:00",
                      "7a85864039b9a9980e2b9308302ce15146d732ae03f0d88d51ae7a604f38c54b"),
            FileFacts("libggml.so.0.10.0", 77_880, "2026-04-25T19:35:39+00:00",
                      "ccb98bd7f02198d6345e469c1e1aafa2cd9735791e80eec526eb8952c8e90b74"),
            FileFacts("libllama-common.so.0.0.8932", 5_086_776, "2026-04-25T19:38:56+00:00",
                      "1588061895ad551b4360b24037624fc408cf24e02159c0cfbe4eecd62b9f238c"),
            FileFacts("libllama.so.0.0.8932", 3_033_896, "2026-04-25T19:37:19+00:00",
                      "4f6c659419ee77c8002bc928867327093edaab101201370cb12f1c2d9ab4b629"),
            FileFacts("libmtmd.so.0.0.8932", 1_171_384, "2026-04-25T19:38:04+00:00",
                      "b66ff78faedf3c69ef50645f9e08232a3d4531e2f49cecd1518e5c25a3cee76c"),
        ),
        scene_request_sha256="3057dacd4c0ed54193b36b10cfccbac65efaae38ab40e7a6bb142be341595045",
        limitations=(STARTUP_IDENTITY_LIMITATION, CANDIDATE_RUNTIME_LIMITATION),
        note=("D58 candidate run step4cand-20261007T082510Z: all 15 criteria of "
              "step4cand-wtd-plr-combined-cache-off-v2 pass (status record, session 46); accepted by the maintainer "
              "under D59 (session 47) for the bounded 640x480 replay demo profile only"),
        memory_policy=CANDIDATE_POLICY,
        memory_policy_verified=True,
    ),
})


def known_profile_ids(profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES) -> frozenset[str]:
    """Profiles that admit model adapters at all: provisional or accepted."""
    return frozenset(
        pid for pid, p in profiles.items() if p.status in (ProfileStatus.PROVISIONAL, ProfileStatus.ACCEPTED)
    )


KNOWN_RESOURCE_PROFILES: frozenset[str] = known_profile_ids()


# ---------------------------------------------------------------- face admission (V2-25 demo form)


class FaceAdmissionStatus(str, Enum):
    PENDING_VALIDATION = "pending_validation"  # only an explicit, guarded --face-validation run may use it
    VALIDATED = "validated"  # set by a separate, maintainer-approved commit after the live device validation


# What the replay profile's face measurement does not establish about the live identity pipeline.
FACE_RUNTIME_LIMITATION = (
    "the profile measured this face model on replay in the profiler's workload: YuNet and Facenet512 on whole 640x480 "
    "frames at 1.0 Hz with TensorFlow on the CPU (600 runs, 0 errors, p50 919.0 ms, a 439,967,744 B load delta), with "
    "no enrolled identities; it does not measure live camera faces, sentinel run's face worker (scheduling, skipped "
    "empty frames, frame copies, result rejection), association with live tracks, gallery matching or the sealed "
    "gallery, identity correctness, or the runtime's memory with face; those need the separate live device validation"
)


@dataclass(frozen=True)
class FaceAdmission:
    """A face admission record, separate from the resource profile it names (whose entry is never edited)."""

    profile_id: str
    status: FaceAdmissionStatus
    face_hz: float  # the measured cadence; the runtime's identity.face_interval_s must equal 1 / face_hz
    model_weights: FileFacts  # with sha256
    detector_weights: FileFacts  # with sha256
    deepface_version: str
    limitation: str = FACE_RUNTIME_LIMITATION
    note: str = ""


FACE_ADMISSIONS: Mapping[str, FaceAdmission] = MappingProxyType({
    # Maintainer decision (2026-10-07): a separate record keyed to the accepted replay profile, starting
    # PENDING_VALIDATION; the profile entry, its acceptance and its limitations stay as recorded.
    "step4cand-demo-20261007T090339Z": FaceAdmission(
        profile_id="step4cand-demo-20261007T090339Z",
        status=FaceAdmissionStatus.PENDING_VALIDATION,
        face_hz=1.0,
        model_weights=FileFacts("facenet512_weights.h5", 94_955_648, "2026-03-23T14:43:46+00:00",
                                "3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f"),
        detector_weights=FileFacts("face_detection_yunet_2023mar.onnx", 232_589, "2026-04-25T22:14:17+00:00",
                                   "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
        deepface_version="0.0.99",
        note="weights as that run's provenance recorded them; face workload_steady: 600 runs at 1.0 Hz, 0 errors",
    ),
})


def face_admission_problem(
    profile_id: str | None,
    *,
    validation_run: bool,
    profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES,
    admissions: Mapping[str, FaceAdmission] = FACE_ADMISSIONS,
) -> str | None:
    """None if face recognition may run on ``profile_id``: an accepted profile with a face admission record that is
    VALIDATED, or PENDING_VALIDATION in a ``--face-validation`` run; otherwise the first missing prerequisite."""
    if profile_id is None:
        return "the manifest names no resource_profile_id; face needs a face admission record"
    problem = accepted_profile_problem(profile_id, profiles)
    if problem is not None:
        return problem
    admission = admissions.get(profile_id)
    if admission is None or admission.profile_id != profile_id:
        return f"resource profile {profile_id} has no face admission record"
    if admission.status is FaceAdmissionStatus.PENDING_VALIDATION and not validation_run:
        return (f"the face admission for {profile_id} is pending validation; only a guarded --face-validation run "
                "may use it")
    if admission.status not in (FaceAdmissionStatus.PENDING_VALIDATION, FaceAdmissionStatus.VALIDATED):
        return f"the face admission for {profile_id} is {admission.status.value}"
    return None


def accepted_profile_problem(
    profile_id: str | None, profiles: Mapping[str, ResourceProfile] = RESOURCE_PROFILES
) -> str | None:
    """None if ``profile_id`` is an ACCEPTED combined profile measured with --cache-ram 0 whose
    recorded evidence passes the step-4 criteria under an admissible identity (D59: step 4, or
    the D58 candidate with its policy evidence and runtime limitation); otherwise the first
    missing prerequisite. The runtime's policy is matched separately (profile_mismatch)."""
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
    if profile.criteria_id not in SCENE_ADMISSIBLE_CRITERIA_IDS:  # D59: step 4 or the D58 candidate
        return (f"resource profile {profile_id} was not judged against an admissible identity "
                f"({' or '.join(SCENE_ADMISSIBLE_CRITERIA_IDS)})")
    measured = CRITERIA_MEMORY_POLICIES[profile.criteria_id]
    if profile.memory_policy != measured:  # D58: an identity and its record must agree
        return (f"resource profile {profile_id} records memory policy ({profile.memory_policy.describe()}), not the "
                f"one {profile.criteria_id} measures ({measured.describe()})")
    if profile.memory_policy != DEFAULT_POLICY and profile.memory_policy_verified is not True:  # D59
        return f"resource profile {profile_id} has no recorded evidence that its memory policy held"
    if profile.criteria_passed is not True:
        return f"resource profile {profile_id} has no recorded pass against {profile.criteria_id}"
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
    if profile.criteria_id == STEP4_CANDIDATE_CRITERIA_ID and CANDIDATE_RUNTIME_LIMITATION not in profile.limitations:
        return f"resource profile {profile_id} does not carry the candidate's runtime limitation"
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
    face_admissions: Mapping[str, FaceAdmission] = FACE_ADMISSIONS,
) -> tuple[AdapterStatus, ...]:
    """Decide each adapter's state without importing anything."""
    known = known_profile_ids(profiles) if known_profiles is None else known_profiles
    statuses = []
    for manifest in manifests:
        statuses.append(_resolve_one(manifest, registry, known, profiles, face_admissions))
    return tuple(statuses)


def _resolve_one(
    manifest: AdapterManifest,
    registry: Mapping[str, AdapterSpec],
    known_profiles: frozenset[str],
    profiles: Mapping[str, ResourceProfile],
    face_admissions: Mapping[str, FaceAdmission] = FACE_ADMISSIONS,
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
    if spec.requires_face_admission:
        problem = face_admission_problem(manifest.resource_profile_id, validation_run=True, profiles=profiles,
                                         admissions=face_admissions)
        if problem is not None:
            return unavailable(problem, spec)
        if face_admissions[manifest.resource_profile_id].status is FaceAdmissionStatus.PENDING_VALIDATION:  # type: ignore[index]
            return AdapterStatus(manifest, AdapterState.ENABLED,
                                 "enabled for guarded --face-validation runs only (admission pending validation)", spec)
        return AdapterStatus(manifest, AdapterState.ENABLED, "enabled", spec)
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
