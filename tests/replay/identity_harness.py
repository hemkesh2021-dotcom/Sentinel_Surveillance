"""Replays faces and people through EdgeCore's identity path and the B0 v1 face worker.

Embeddings are synthetic 4-D vectors; no real face data is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from b0_v1_snapshot import V1FaceWorker
from core_harness import CONFIG

from sentinel.contracts import NormalizedBox
from sentinel.identity.association import FaceObservation
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import IdentityState
from sentinel.jobs import AnalysisJob
from sentinel.live_state import Capability, LiveState, PersonState
from sentinel.replay import ReplayDriver, Timeline
from sentinel.runtime import EdgeCore

ALICE = (1.0, 0.0, 0.0, 0.0)
BOB = (0.0, 1.0, 0.0, 0.0)
ALICE_FACE = (0.95, 0.1, 0.0, 0.05)  # cosine ~0.99 with ALICE
BOB_FACE = (0.05, 0.97, 0.1, 0.0)
NOBODY_FACE = (0.0, 0.1, 0.99, 0.0)  # matches nobody enrolled
ENROLLMENT = Enrollment(
    model_revision="stub-embedder-1",
    dimension=4,
    identities=(
        EnrolledIdentity(identity_id="alice", prototypes=(ALICE,)),
        EnrolledIdentity(identity_id="bob", prototypes=(BOB,)),
    ),
)
V1_DB = {"alice": ALICE, "bob": BOB}
# v1 pads each person crop by 20 px (L355-357) on its 640x480 frame.
V1_PAD_X, V1_PAD_Y = 20 / 640, 20 / 480


class IdleAnalyzer:
    revision = "idle"

    def submit(self, job: AnalysisJob) -> None:
        pass

    def cancel(self, job_id: str) -> None:
        pass


@dataclass
class IdentityRun:
    states: list[tuple[int, LiveState]] = field(default_factory=list)
    v1: V1FaceWorker | None = None
    core: EdgeCore | None = None

    def people_at(self, at_ms: int) -> dict[int, PersonState]:
        state = [s for ms, s in self.states if ms <= at_ms][-1]
        return {person.track_id: person for person in state.people}

    def history(self, track_id: int) -> list[IdentityState]:
        return [
            person.identity
            for _, state in self.states
            for person in state.people
            if person.track_id == track_id
        ]


def run_identity(
    timeline: Timeline,
    *,
    enrollment: Enrollment | None = ENROLLMENT,
    v1_db: dict[str, tuple[float, ...]] = V1_DB,
) -> IdentityRun:
    driver = ReplayDriver(timeline)
    origin = driver.clock.monotonic_ns()
    core = EdgeCore(
        CONFIG,
        driver.clock,
        IdleAnalyzer(),
        face_recognition=Capability.AVAILABLE,
        enrollment=enrollment,
    )
    run = IdentityRun(v1=V1FaceWorker(v1_db), core=core)
    for step in driver.steps():
        at_ms = (step.now.ns - origin) // 1_000_000
        if step.frame is None:
            run.states.append((at_ms, core.tick(step.stream).state))
            continue
        faces = tuple(
            FaceObservation(
                frame=step.frame.key,
                box=NormalizedBox(x1=f.box[0], y1=f.box[1], x2=f.box[2], y2=f.box[3]),
                quality=f.quality,
                embedding=f.embedding,
            )
            for f in step.faces
        )
        run.states.append((at_ms, core.on_frame(step.frame, step.persons, step.stream, faces).state))
        with_embedding = [f for f in step.faces if f.embedding is not None]
        crops = {
            p.track_id: [f.embedding for f in with_embedding if _in_padded(f.box, p.box)]
            for p in step.persons
        }
        run.v1.run(crops, [f.embedding for f in with_embedding])
    return run


def _in_padded(face: tuple[float, ...], person: NormalizedBox) -> bool:
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    return (
        person.x1 - V1_PAD_X <= cx <= person.x2 + V1_PAD_X
        and person.y1 - V1_PAD_Y <= cy <= person.y2 + V1_PAD_Y
    )
