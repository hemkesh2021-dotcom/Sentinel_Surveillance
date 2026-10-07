"""Typed Sentinel configuration (guide chapters 6 and 18).

Invalid configuration fails before anything starts, with one line per problem
naming where it is. Credentials never belong in this file; later slices will
reference secrets by ID.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .adapters import AdapterManifest, check_unique_ids
from .contracts import Identifier
from .media.clock import NS_PER_SECOND
from .redaction import redact_line
from .rules.geometry import Anchor, polygon_problem
from .rules.schedule import Schedule, load_timezone, parse_hhmm

CONFIG_VERSION = 1

Seconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CameraConfig(_Section):
    id: Identifier


class CaptureConfig(_Section):
    """Camera ingest (V2-05 demo form: OpenCV/FFmpeg software decode, D24). Proposed starting values.

    The stream URL comes from the SENTINEL_RTSP_URL environment variable, never this file.
    """

    open_timeout_s: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 10.0
    # A connection that delivers no frame for this long is closed and reopened (new stream epoch).
    read_timeout_s: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 5.0
    # Wait before reopening; doubles after each connection that delivered no frame.
    reconnect_initial_s: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 1.0
    reconnect_max_s: Annotated[float, Field(gt=0, le=300, allow_inf_nan=False)] = 15.0
    # FFmpeg frame threading delays each frame by up to (threads - 1) frames.
    decode_threads: Annotated[int, Field(ge=1, le=4)] = 1

    @field_validator("reconnect_max_s")
    @classmethod
    def _max_at_least_initial(cls, value: float, info: ValidationInfo) -> float:
        initial = info.data.get("reconnect_initial_s")
        if initial is not None and value < initial:
            raise ValueError(f"must be at least reconnect_initial_s ({value} < {initial})")
        return value


class FreshnessConfig(_Section):
    """Guide chapter 6 starting targets, to be verified on the actual network."""

    stale_after_s: Seconds = 2.0
    offline_after_s: Seconds = 10.0
    track_expiry_s: Seconds = 1.0

    @field_validator("offline_after_s")
    @classmethod
    def _offline_after_stale(cls, value: float, info: ValidationInfo) -> float:
        stale = info.data.get("stale_after_s")
        if stale is not None and value <= stale:
            raise ValueError(f"must be greater than stale_after_s ({value} <= {stale})")
        return value

    @property
    def stale_after_ns(self) -> int:
        return round(self.stale_after_s * NS_PER_SECOND)

    @property
    def offline_after_ns(self) -> int:
        return round(self.offline_after_s * NS_PER_SECOND)

    @property
    def track_expiry_ns(self) -> int:
        return round(self.track_expiry_s * NS_PER_SECOND)


class SceneConfig(_Section):
    """Scene lane (VLM) timing. Proposed starting values; measure latency on the device (V2-26)."""

    interval_s: Seconds = 4.0  # between periodic scene checks, with or without people (v1: 4 s)
    job_timeout_s: Seconds = 8.0  # from the source frame's ingest to the job deadline
    evidence_ttl_s: Seconds = 10.0  # how long an on-time report stays current

    @field_validator("evidence_ttl_s")
    @classmethod
    def _ttl_covers_timeout(cls, value: float, info: ValidationInfo) -> float:
        timeout = info.data.get("job_timeout_s")
        if timeout is not None and value < timeout:
            raise ValueError(
                f"must be at least job_timeout_s ({value} < {timeout}); "
                "otherwise results that meet their deadline are already expired"
            )
        return value

    @property
    def interval_ns(self) -> int:
        return round(self.interval_s * NS_PER_SECOND)

    @property
    def job_timeout_ns(self) -> int:
        return round(self.job_timeout_s * NS_PER_SECOND)

    @property
    def evidence_ttl_ns(self) -> int:
        return round(self.evidence_ttl_s * NS_PER_SECOND)


class SceneServerConfig(_Section):
    """The llama-server behind the demo scene adapter (V2-26 demo form; D41, D42).

    It always listens on 127.0.0.1, and its model flags are fixed in code (the
    provisional demo profile's plus ``--cache-ram 0``); nothing here changes either.
    """

    port: Annotated[int, Field(ge=1024, le=65535)] = 18081
    # Bounds each blocking HTTP step of one request. At least scene.job_timeout_s, so the
    # job deadline decides what is on time; a reply after it only annotates history.
    request_timeout_s: Annotated[float, Field(gt=0, le=120, allow_inf_nan=False)] = 20.0


class HazardConfig(_Section):
    """Fire/smoke candidates from scene reports; VLM-only candidates are capped at warning."""

    confirmations: Annotated[int, Field(ge=2, le=10)] = 2
    max_gap_s: Seconds = 12.0  # between the source frames of consecutive positive reports

    @property
    def max_gap_ns(self) -> int:
        return round(self.max_gap_s * NS_PER_SECOND)


class IdentityConfig(_Section):
    """Face association and identity policy. Starting values; calibrate on held-out identities (V2-25)."""

    # Cosine similarity, not a probability. 0.70 is DeepFace 0.0.99's pretuned Facenet512 value (cosine distance
    # 0.30, config/threshold.py, verified when distance <= threshold); uncalibrated for this camera and these people.
    match_threshold: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] = 0.70
    margin: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0.05  # uncalibrated starting value
    min_quality: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.6  # YuNet confidence; uncalibrated
    confirmations: Annotated[int, Field(ge=1, le=5)] = 2
    vote_ttl_s: Seconds = 30.0
    head_fraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.4
    min_face_inside: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.6
    # The 1 Hz face path (V2-25 demo form, D34); starting values: face p99 was 989 ms in the replay profile.
    face_interval_s: Seconds = 1.0  # between face ticks, in frame time
    face_pending_max_age_s: Seconds = 2.0  # a waiting frame older than this is not analysed
    face_result_max_age_s: Seconds = 3.0  # a result for an older frame is rejected; also the "fresh" window

    @model_validator(mode="after")
    def _face_bounds(self) -> IdentityConfig:
        if not self.face_pending_max_age_s <= self.face_result_max_age_s:
            raise ValueError("face_pending_max_age_s must not exceed face_result_max_age_s")
        return self

    @property
    def vote_ttl_ns(self) -> int:
        return round(self.vote_ttl_s * NS_PER_SECOND)

    @property
    def face_interval_ns(self) -> int:
        return round(self.face_interval_s * NS_PER_SECOND)

    @property
    def face_pending_max_age_ns(self) -> int:
        return round(self.face_pending_max_age_s * NS_PER_SECOND)

    @property
    def face_result_max_age_ns(self) -> int:
        return round(self.face_result_max_age_s * NS_PER_SECOND)


ZoneId = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ScheduleWindowConfig(_Section):
    """Daily local-time window [start, end); an end not after the start spans midnight."""

    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _time_of_day(cls, value: str) -> str:
        parse_hhmm(value)
        return value

    @field_validator("end")
    @classmethod
    def _not_empty(cls, value: str, info: ValidationInfo) -> str:
        if info.data.get("start") == value:
            raise ValueError("must differ from start; omit the schedule for a rule that is always active")
        return value


class ScheduleConfig(_Section):
    timezone: str  # IANA name, e.g. Asia/Kolkata
    windows: list[ScheduleWindowConfig] = Field(min_length=1, max_length=8)

    @field_validator("timezone")
    @classmethod
    def _iana(cls, value: str) -> str:
        load_timezone(value)
        return value

    def build(self) -> Schedule:
        return Schedule.parse(self.timezone, [(w.start, w.end) for w in self.windows])


class ZoneConfig(_Section):
    """A zone rule over current person tracks (guide ch. 9). Starting values; calibrate on labelled replays (V2-07)."""

    zone_id: ZoneId
    rule: Literal["restricted", "dwell"] = "restricted"
    # Normalized native-image points [x, y]; x to the right, y downwards.
    polygon: list[Annotated[list[Fraction], Field(min_length=2, max_length=2)]]
    anchor: Literal["bottom_center", "center"] = "bottom_center"
    # restricted: how long a person must be seen in the zone before it counts as an entry;
    # dwell: how long before it counts as dwelling.
    min_duration_s: Seconds = 1.0
    # How long a person may be unseen in the zone (out of it, predicted only, or outside
    # the schedule) before the presence ends.
    gap_tolerance_s: Seconds = 1.0
    severity: Literal["info", "warning", "critical"] = "warning"
    schedule: ScheduleConfig | None = None  # None: always active
    enabled: bool = True

    @field_validator("polygon")
    @classmethod
    def _simple_polygon(cls, value: list[list[float]]) -> list[list[float]]:
        problem = polygon_problem([(x, y) for x, y in value])
        if problem is not None:
            raise ValueError(problem)
        return value

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return tuple((x, y) for x, y in self.polygon)

    @property
    def anchor_kind(self) -> Anchor:
        return Anchor(self.anchor)

    @property
    def min_duration_ns(self) -> int:
        return round(self.min_duration_s * NS_PER_SECOND)

    @property
    def gap_tolerance_ns(self) -> int:
        return round(self.gap_tolerance_s * NS_PER_SECOND)

    @property
    def revision(self) -> str:
        """Changes whenever any setting of this zone changes; recorded with its observations."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class IncidentsConfig(_Section):
    """How rule observations become incidents (guide ch. 9 and 10). Proposed starting value."""

    # A new observation of the same rule and zone within this long of the incident's
    # latest one, in the same boot, joins that open incident instead of opening another.
    merge_window_s: Seconds = 120.0

    @property
    def merge_window_ns(self) -> int:
        return round(self.merge_window_s * NS_PER_SECOND)


class NotificationsConfig(_Section):
    """External notification channels; all disabled unless listed (guide ch. 10)."""

    channels: list[Literal["telegram"]] = Field(default_factory=list, max_length=4)
    min_severity: Literal["info", "warning", "critical"] = "warning"
    # Delivery (guide ch. 10). Proposed starting values; they do not change the policy revision.
    lease_s: Seconds = 60.0  # a worker that holds a row longer is presumed dead; the row is retried
    request_timeout_s: Seconds = 10.0  # must stay well below lease_s
    max_attempts: Annotated[int, Field(ge=1, le=50)] = 8
    max_age_s: Seconds = 6 * 3600.0  # older undelivered rows are dead-lettered
    backoff_base_s: Seconds = 5.0
    backoff_max_s: Seconds = 600.0
    starvation_s: Seconds = 300.0  # rows due this long are served before newer urgent ones

    @field_validator("request_timeout_s")
    @classmethod
    def _timeout_inside_lease(cls, value: float, info: ValidationInfo) -> float:
        lease = info.data.get("lease_s")
        if lease is not None and value * 2 > lease:
            raise ValueError(f"must be at most half of lease_s ({value} > {lease} / 2)")
        return value

    @field_validator("channels")
    @classmethod
    def _unique_channels(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each channel may be listed once")
        return value

    @property
    def revision(self) -> str:
        """Policy revision recorded with each outbox row; a changed policy may notify again."""
        policy = {"channels": self.channels, "min_severity": self.min_severity}
        return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]


class SentinelConfig(_Section):
    config_version: Literal[1]
    camera: CameraConfig
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    scene_server: SceneServerConfig = Field(default_factory=SceneServerConfig)
    hazard: HazardConfig = Field(default_factory=HazardConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    zones: list[ZoneConfig] = Field(default_factory=list, max_length=16)
    incidents: IncidentsConfig = Field(default_factory=IncidentsConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    # Optional adapters; core monitoring runs with none enabled (guide ch. 27).
    adapters: list[AdapterManifest] = Field(default_factory=list, max_length=32)

    @field_validator("adapters")
    @classmethod
    def _unique_adapter_ids(cls, value: list[AdapterManifest]) -> list[AdapterManifest]:
        check_unique_ids(value)
        return value

    @field_validator("zones")
    @classmethod
    def _unique_zone_ids(cls, value: list[ZoneConfig]) -> list[ZoneConfig]:
        seen: set[str] = set()
        for zone in value:
            if zone.zone_id in seen:
                raise ValueError(f"duplicate zone_id {zone.zone_id!r}")
            seen.add(zone.zone_id)
        return value

    @model_validator(mode="after")
    def _request_outlasts_job(self) -> SentinelConfig:
        timeout, job = self.scene_server.request_timeout_s, self.scene.job_timeout_s
        if timeout < job:
            raise ValueError(
                f"scene_server.request_timeout_s must be at least scene.job_timeout_s ({timeout} < {job})"
            )
        return self


class ConfigError(Exception):
    """Configuration could not be loaded; ``problems`` has one entry per issue."""

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        count = f"{len(problems)} problem" + ("" if len(problems) == 1 else "s")
        details = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"{source}: invalid configuration ({count})\n{details}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys; PyYAML otherwise keeps the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key: the base class reports it
                continue
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: str | os.PathLike[str]) -> SentinelConfig:
    source = os.fspath(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(source, [f"cannot read file ({exc.strerror or exc})"]) from None
    except UnicodeDecodeError:
        raise ConfigError(source, ["file is not valid UTF-8"]) from None
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(source, [_describe_yaml_error(exc)]) from None
    return parse_config({} if data is None else data, source=source)


def parse_config(data: object, *, source: str = "<config>") -> SentinelConfig:
    if not isinstance(data, Mapping):
        raise ConfigError(
            source, [f"top level must be a mapping of settings, not {type(data).__name__}"]
        )
    try:
        return SentinelConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(source, [_describe(error) for error in exc.errors()]) from None


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc)
    if mark is None:
        return f"YAML error: {problem}"
    return f"YAML error at line {mark.line + 1}, column {mark.column + 1}: {problem}"


def _describe(error: Mapping[str, Any]) -> str:
    loc = tuple(error["loc"])
    where = ".".join(str(part) for part in loc) or "<top level>"
    kind = error["type"]
    if kind == "extra_forbidden":
        return f"{where}: unknown setting{_suggestion(loc)}"
    if kind == "missing":
        return f"{where}: required setting is missing"
    if kind == "value_error":
        return f"{where}: {error['ctx']['error']}"
    if kind == "literal_error" and loc == ("config_version",):
        return (
            f"{where}: unsupported version {_show(error['input'])}; "
            f"this build reads version {CONFIG_VERSION}"
        )
    return f"{where}: {error['msg']} (got {_show(error['input'])})"


def _suggestion(loc: tuple[Any, ...]) -> str:
    model: type[BaseModel] = SentinelConfig
    for part in loc[:-1]:
        if isinstance(part, int):  # an item of a list of sections, e.g. adapters[0]
            continue
        field = model.model_fields.get(part)
        annotation = field.annotation if field is not None else None
        if get_origin(annotation) is list:
            (annotation,) = get_args(annotation)
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            return ""
        model = annotation
    matches = difflib.get_close_matches(str(loc[-1]), list(model.model_fields), n=1)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def _show(value: Any) -> str:
    """Short repr of an offending value with any URL credentials removed."""
    return redact_line(repr(value), limit=60)
