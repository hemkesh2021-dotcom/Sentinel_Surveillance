"""Typed Sentinel configuration (guide chapters 6 and 18).

Invalid configuration fails before anything starts, with one line per problem
naming where it is. Credentials never belong in this file; later slices will
reference secrets by ID.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from .contracts import Identifier
from .media.clock import NS_PER_SECOND
from .redaction import redact_line

CONFIG_VERSION = 1

Seconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CameraConfig(_Section):
    id: Identifier


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


class HazardConfig(_Section):
    """Fire/smoke candidates from scene reports; VLM-only candidates are capped at warning."""

    confirmations: Annotated[int, Field(ge=2, le=10)] = 2
    max_gap_s: Seconds = 12.0  # between the source frames of consecutive positive reports

    @property
    def max_gap_ns(self) -> int:
        return round(self.max_gap_s * NS_PER_SECOND)


class SentinelConfig(_Section):
    config_version: Literal[1]
    camera: CameraConfig
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    hazard: HazardConfig = Field(default_factory=HazardConfig)


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
        field = model.model_fields.get(part) if isinstance(part, str) else None
        annotation = field.annotation if field is not None else None
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            return ""
        model = annotation
    matches = difflib.get_close_matches(str(loc[-1]), list(model.model_fields), n=1)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def _show(value: Any) -> str:
    """Short repr of an offending value with any URL credentials removed."""
    return redact_line(repr(value), limit=60)
