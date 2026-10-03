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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
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


# Real adapters join with their packages (detector V2-09, face V2-25, scene
# V2-26, notifiers V2-15). Keep this table explicit.
BUILTIN_ADAPTERS: Mapping[str, AdapterSpec] = {
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

# Resource profiles with measured admission data (guide ch. 27).
# provisional-demo-20261003T085010Z: check 8's combined run of the three demo model
# components (D28, D33). Admits the demo adapters for the Oct 20 demo only, with its
# exceedance recorded; not a benchmark, Gate B record or beta-gate result.
KNOWN_RESOURCE_PROFILES: frozenset[str] = frozenset({"provisional-demo-20261003T085010Z"})


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
    known_profiles: frozenset[str] = KNOWN_RESOURCE_PROFILES,
) -> tuple[AdapterStatus, ...]:
    """Decide each adapter's state without importing anything."""
    statuses = []
    for manifest in manifests:
        statuses.append(_resolve_one(manifest, registry, known_profiles))
    return tuple(statuses)


def _resolve_one(
    manifest: AdapterManifest,
    registry: Mapping[str, AdapterSpec],
    known_profiles: frozenset[str],
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
    if spec.role.uses_models and manifest.resource_profile_id not in known_profiles:
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
