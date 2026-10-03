"""Replays a timeline through EdgeCore with zone rules configured, at a chosen UTC start time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sentinel.config import parse_config
from sentinel.contracts import NormalizedBox
from sentinel.identity.association import FaceObservation
from sentinel.identity.matching import Enrollment
from sentinel.live_state import Capability, LiveState
from sentinel.media.clock import FakeClock
from sentinel.replay import ReplayDriver, Timeline
from sentinel.rules.zones import ZoneObservation, ZonePhase
from sentinel.runtime import EdgeCore
from sentinel.scene.lane import SceneAnalyzer

# A floor zone in the lower right of the image, e.g. in front of a door.
DOOR_ZONE = [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]]
IN_ZONE = [0.65, 0.2, 0.85, 0.9]  # feet (bottom centre 0.75, 0.9) inside the zone
OUTSIDE = [0.05, 0.2, 0.25, 0.9]  # elsewhere in the room
# Upper body overlaps the zone, feet (0.575, 0.95) outside it: standing beside the door.
BESIDE_ZONE = [0.45, 0.3, 0.7, 0.95]


def person(box: list[float], track: int = 1, **fields: object) -> list[dict]:
    return [{"track": track, "box": box, **fields}]


def zone(**overrides: object) -> dict:
    return {"zone_id": "door", "polygon": DOOR_ZONE, **overrides}


@dataclass
class ZoneRun:
    observations: list[tuple[int, ZoneObservation]] = field(default_factory=list)
    states: list[tuple[int, LiveState]] = field(default_factory=list)
    core: EdgeCore | None = None

    def phase(self, phase: ZonePhase) -> list[tuple[int, ZoneObservation]]:
        return [(ms, o) for ms, o in self.observations if o.phase is phase]

    @property
    def entered(self) -> list[tuple[int, ZoneObservation]]:
        return self.phase(ZonePhase.ENTERED)

    @property
    def ended(self) -> list[tuple[int, ZoneObservation]]:
        return self.phase(ZonePhase.ENDED)

    def at(self, at_ms: int) -> LiveState:
        return [state for ms, state in self.states if ms <= at_ms][-1]


def run_zones(
    timeline: Timeline,
    *,
    zones: list[dict],
    utc_start: datetime,
    analyzer: SceneAnalyzer | None = None,
    enrollment: Enrollment | None = None,
) -> ZoneRun:
    """Scene analysis is off unless an analyzer is given; faces are used only with an enrollment."""
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "zones": zones})
    driver = ReplayDriver(timeline, clock=FakeClock(utc=utc_start))
    origin = driver.clock.monotonic_ns()
    core = EdgeCore(
        config,
        driver.clock,
        analyzer,
        face_recognition=Capability.AVAILABLE if enrollment is not None else Capability.DISABLED,
        enrollment=enrollment,
    )
    run = ZoneRun(core=core)
    for step in driver.steps():
        at_ms = (step.now.ns - origin) // 1_000_000
        if step.frame is None:
            output = core.tick(step.stream)
        else:
            faces = tuple(
                FaceObservation(
                    frame=step.frame.key,
                    box=NormalizedBox(x1=f.box[0], y1=f.box[1], x2=f.box[2], y2=f.box[3]),
                    quality=f.quality,
                    embedding=f.embedding,
                )
                for f in step.faces
            )
            output = core.on_frame(step.frame, step.persons, step.stream, faces if enrollment else None)
        run.states.append((at_ms, output.state))
        run.observations += [(at_ms, observation) for observation in output.zones]
    return run
