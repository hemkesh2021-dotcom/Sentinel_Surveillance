"""Published live state (guide chapters 2, 6 and 15).

The edge runtime publishes one LiveState per step, including steps with no
people, no frames or no scene result, so consumers never have to infer state
from silence. It keeps apart what the UI must distinguish: "nobody detected on
fresh video" (EMPTY), "no fresh video" and "the detector is unavailable" (both
UNKNOWN, with the reason), and a scene that is reported, unknown (the last
check failed) or has no current result.
"""

from __future__ import annotations

from enum import Enum

from pydantic import NonNegativeInt

from .contracts import (
    Contract,
    Identifier,
    NormalizedBox,
    TrackStatus,
    UtcDatetime,
)
from .identity.state import Basis, IdentityState
from .media.health import VideoState
from .scene.report import SceneReport


class Capability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"  # expected but not working: failed, not loaded or not admitted
    DISABLED = "disabled"  # switched off by configuration


class Occupancy(str, Enum):
    EMPTY = "empty"  # nobody detected on fresh video by a working detector
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"  # see occupancy_reason


class SceneStatus(str, Enum):
    REPORTED = "reported"  # a current scene report
    UNKNOWN = "unknown"  # the current scene check failed, timed out or was not possible
    NO_CURRENT_RESULT = "no_current_result"  # none yet, or the last one is no longer current


class PersonState(Contract):
    track_id: NonNegativeInt
    stream_epoch: NonNegativeInt
    box: NormalizedBox
    status: TrackStatus
    predicted: bool
    last_measured_age_ms: NonNegativeInt
    identity: IdentityState  # context only: never an access decision by itself
    identity_id: Identifier | None  # set only when KNOWN
    identity_reason: str
    identity_basis: Basis | None = None  # KNOWN/UNKNOWN: a fresh decision or one retained on the same track
    identity_vote_age_ms: NonNegativeInt | None = None  # the newest current vote's age


class LiveState(Contract):
    camera_id: Identifier
    sequence: NonNegativeInt  # increases with every publication of this runtime
    boot_id: Identifier
    published_mono_ns: NonNegativeInt
    published_utc: UtcDatetime
    video: VideoState
    last_frame_age_ms: NonNegativeInt | None
    detector: Capability
    face_recognition: Capability
    scene_analysis: Capability
    occupancy: Occupancy
    occupancy_reason: str
    people: tuple[PersonState, ...]
    scene: SceneStatus
    scene_reason: str
    scene_report: SceneReport | None
    scene_evidence_id: Identifier | None
