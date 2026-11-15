"""Session 115, part 9: an answer about a frame the runtime replaced is recorded with both frames and judged against the
frame sent; nothing missing or contradictory passes, and a record that states no replacement is judged as before.

Since session 114 a question whose frame went stale while it waited is sent the newest live frame (scene.chat). Here a
real ChatService and scene worker on a FakeClock produce the documents: a question that waits behind a scene job until
its frame is stale, and one whose frame is still live. alpha_load's own record function records them, face_evidence's
chat-load profile reads the records, and alpha_check judges them and records that differ from them in one fact each.
The nine records of the device session fci25-20261026T124349Z are not used and nothing of theirs is reconstructed; the
shape they have (no replacement stated, no age when asked) is shown to stay not consistent.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel.config import FreshnessConfig
from sentinel.contracts import PixelFormat
from sentinel.jobs import AnalysisJob, JobPurpose, WorkerOutcome
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.health import assess_frame
from sentinel.media.recent import RecentFrames
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.chat import ChatAdmission, ChatQuestion, ChatService

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
ANSWER = "A synthetic answer: a chair stands by the window."
QUESTION = "What is visible in this picture? Answer in one sentence."


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield SimpleNamespace(check=importlib.import_module("alpha_check"), load=importlib.import_module("alpha_load"),
                              evidence=importlib.import_module("face_evidence"),
                              s5=importlib.import_module("step5_check"))
    finally:
        sys.path.remove(str(RUNNERS))


class Live:
    """A real scene worker and ChatService on a FakeClock; the model answers at once, or takes ``answer_s``."""

    def __init__(self, answer_s: float = 0.0) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stamper.connect()
        self.frames, self.images, self.hold = RecentFrames(), RecentImages(), threading.Event()
        self.analyzer = ThreadedSceneAnalyzer(self.scene_job, self.images, revision="test")
        self.analyzer.start()
        started = self.clock.mono()

        def model(question: Any, image: Any, abort: Any = None, max_tokens: Any = None) -> tuple[str, bool]:
            self.clock.advance(answer_s)
            return ANSWER, False

        self.service = ChatService(
            enabled=True, admission=ChatAdmission(True, "admitted for this test", "a synthetic fixture"),
            scene=self.analyzer, latest=self.frames.latest,
            assess=lambda frame: assess_frame(frame, self.stamper.current_stream, self.clock.mono(), started,
                                              FreshnessConfig()),
            clock=self.clock, run=model, stale_after_s=2.0)

    def scene_job(self, job: AnalysisJob, image: Any) -> WorkerOutcome:
        self.hold.wait(10.0)
        return WorkerOutcome.completed(job.job_id, "{}")

    def frame(self, dt: float = 1 / 15) -> None:
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        self.frames.put(frame, f"image-{frame.frame_seq}")

    def asked_while_a_scene_job_runs(self, waited_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ask while a periodic scene job runs; the camera goes on for ``waited_s``, then the job ends."""
        self.frame()
        frame = self.frames.latest()[0]
        self.images.put(frame.key, object())  # the periodic job names a held frame
        self.analyzer.submit(AnalysisJob(job_id="scene-1", frame=frame, purpose=JobPurpose.PERIODIC, incident_id=None,
                                         deadline_mono_ns=frame.ingest_mono_ns + 8 * NS_PER_SECOND))
        until(self.analyzer.busy)
        code, accepted = self.service.ask(ChatQuestion(QUESTION))
        assert code == 202, accepted
        for _ in range(round(waited_s / 0.5)):
            self.frame(0.5)
        self.hold.set()
        deadline = time.monotonic() + 5
        while True:
            code, document = self.service.poll(accepted["job"])
            if document["state"] not in ("sending", "waiting", "answering"):
                return accepted, document
            assert time.monotonic() < deadline, document
            time.sleep(0.005)

    def close(self) -> None:
        self.hold.set()
        assert self.analyzer.stop(5.0)


def until(condition: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def answered(waited_s: float, answer_s: float = 0.0) -> tuple[dict[str, Any], dict[str, Any]]:
    live = Live(answer_s)
    try:
        return live.asked_while_a_scene_job_runs(waited_s)
    finally:
        live.close()


@pytest.fixture(scope="module")
def runtime() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """The runtime's documents: (as accepted, as answered) for each case."""
    return {"unchanged": answered(1.5),  # still live when the worker was free: the frame asked about is sent
            "replaced": answered(2.5),  # stale by then: the newest live frame is sent instead
            "replaced, answered late": answered(2.5, answer_s=2.5)}


def recorded(mods, document: dict[str, Any]) -> dict[str, Any]:
    return mods.load._record(12.0, "short", document, 200)


def fits_the_chat_load_profile(mods, tmp_path: Path, *records: dict[str, Any]) -> list[str]:
    path = tmp_path / "chat-load.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return mods.evidence.read(path, "f2").nonconforming


# ---------------------------------------------------------------- the runtime's own documents, recorded


def test_a_replaced_frame_is_recorded_with_both_frames_and_its_ages_never_the_text(mods, runtime, tmp_path) -> None:
    accepted, document = runtime["replaced"]
    assert document["frame_replaced"] is True  # the runtime's own statement (scene.chat, session 114)
    record = recorded(mods, document)
    assert record["frame_replaced"] is True
    assert record["asked_frame"] == accepted["frame"]  # the original request's frame, exactly as it was accepted
    sent = (record["stream_epoch"], record["frame_seq"], record["ingest_utc"])
    assert sent == (document["frame"]["stream_epoch"], document["frame"]["frame_seq"], document["frame"]["ingest_utc"])
    assert sent[:2] != (accepted["frame"]["stream_epoch"], accepted["frame"]["frame_seq"])  # the frame sent is newer
    assert (record["age_ms_when_asked"], record["age_ms_when_sent"]) == (None, 0)  # never asked about; live when sent
    assert (record["age_ms_when_answered"], record["historical"]) == (0, False)
    assert ANSWER not in json.dumps(record) and record["answer_chars"] == len(ANSWER)
    assert fits_the_chat_load_profile(mods, tmp_path, record) == []
    assert mods.check.answer_problem(record) is None


def test_an_unchanged_frame_is_recorded_and_judged_as_before(mods, runtime, tmp_path) -> None:
    accepted, document = runtime["unchanged"]
    assert "frame_replaced" not in document and "asked_frame" not in document
    record = recorded(mods, document)
    assert (record["frame_replaced"], record["asked_frame"], record["age_ms_when_sent"]) == (False, None, None)
    assert {k: record[k] for k in accepted["frame"]} == accepted["frame"]  # the frame asked about is the frame sent
    assert (record["age_ms_when_answered"], record["historical"]) == (record["age_ms_when_asked"] + 1500, False)
    assert ANSWER not in json.dumps(record)
    assert fits_the_chat_load_profile(mods, tmp_path, record) == []
    assert mods.check.answer_problem(record) is None


def test_a_replaced_frame_answered_late_is_historical_by_the_age_of_the_frame_sent(mods, runtime) -> None:
    record = recorded(mods, runtime["replaced, answered late"][1])
    assert record["frame_replaced"] is True and record["age_ms_when_sent"] == 0
    assert (record["age_ms_when_answered"], record["historical"]) == (2500, True)
    assert mods.check.answer_problem(record) is None
    assert mods.check.answer_problem({**record, "historical": False}).startswith("historical not exactly")


# ---------------------------------------------------------------- the rule: one fact changed at a time


def changed(record: dict[str, Any], change: Any) -> dict[str, Any]:
    out = copy.deepcopy(record)
    change(out)
    return out


REPLACED_VARIANTS = [
    ("replacement stated as text", lambda r: r.update(frame_replaced="yes"),
     "the replacement neither stated nor denied"),
    ("replacement stated as null", lambda r: r.update(frame_replaced=None),
     "the replacement neither stated nor denied"),
    ("no frame asked about", lambda r: r.update(asked_frame=None), "a replacement without the frame asked about"),
    ("frame asked about without its age", lambda r: r["asked_frame"].update(age_ms_when_asked=None),
     "a replacement without the frame asked about"),
    ("frame asked about with a JSON true as its age", lambda r: r["asked_frame"].update(age_ms_when_asked=True),
     "a replacement without the frame asked about"),
    ("frame asked about without its ingest time", lambda r: r["asked_frame"].pop("ingest_utc"),
     "a replacement without the frame asked about"),
    ("the same frame named twice", lambda r: r["asked_frame"].update(stream_epoch=r["stream_epoch"],
                                                                      frame_seq=r["frame_seq"]),
     "a replacement whose frame sent is the frame asked about"),
    ("the frame sent with an age when asked", lambda r: r.update(age_ms_when_asked=40),
     "a replacement whose frame sent has an age when asked"),
    ("no age when sent", lambda r: r.update(age_ms_when_sent=None), "a replacement without a frame sent while live"),
    ("sent at the freshness limit", lambda r: r.update(age_ms_when_sent=2000),
     "a replacement without a frame sent while live"),
    ("a negative age when sent", lambda r: r.update(age_ms_when_sent=-1),
     "a replacement without a frame sent while live"),
    ("answered before it was sent", lambda r: r.update(age_ms_when_sent=600, age_ms_when_answered=500),
     "a replacement without an age when answered at or after its sending"),
    ("no age when answered", lambda r: r.update(age_ms_when_answered=None),
     "a replacement without an age when answered at or after its sending"),
    ("historical flag flipped", lambda r: r.update(historical=True),
     "historical not exactly whether the frame sent was 2000 ms or older when answered"),
    ("an empty answer", lambda r: r.update(answer_chars=0), "not 1-600 characters"),
    ("too long an answer", lambda r: r.update(answer_chars=601), "not 1-600 characters"),
    ("the frame sent not named", lambda r: r.update(frame_seq=None), "the frame sent is not named"),
]
UNCHANGED_VARIANTS = [
    ("a frame asked about without a replacement", lambda r: r.update(asked_frame={"stream_epoch": 1, "frame_seq": 1,
                                                                                    "ingest_utc": r["ingest_utc"],
                                                                                    "age_ms_when_asked": 0}),
     "replacement facts without a replacement"),
    ("an age when sent without a replacement", lambda r: r.update(age_ms_when_sent=0),
     "replacement facts without a replacement"),
    ("no age when asked and no replacement", lambda r: r.update(age_ms_when_asked=None),
     "no age when asked, and no replacement stated"),
    ("no age when answered", lambda r: r.update(age_ms_when_answered=None), "no age when answered"),
    ("historical flag flipped", lambda r: r.update(historical=True),
     "historical not exactly whether the frame sent was 2000 ms or older when answered"),
]


@pytest.mark.parametrize(("name", "change", "problem"), REPLACED_VARIANTS, ids=[v[0] for v in REPLACED_VARIANTS])
def test_a_replacement_with_missing_or_contradictory_facts_is_not_consistent(mods, runtime, name, change,
                                                                             problem) -> None:
    record = recorded(mods, runtime["replaced"][1])
    assert mods.check.answer_problem(changed(record, change)) == problem


@pytest.mark.parametrize(("name", "change", "problem"), UNCHANGED_VARIANTS, ids=[v[0] for v in UNCHANGED_VARIANTS])
def test_an_unchanged_frame_with_missing_or_contradictory_facts_is_not_consistent(mods, runtime, name, change,
                                                                                  problem) -> None:
    record = recorded(mods, runtime["unchanged"][1])
    assert mods.check.answer_problem(changed(record, change)) == problem


def test_a_record_from_before_the_replacement_facts_is_read_as_before(mods, runtime) -> None:
    """alpha_load before session 115, part 9 wrote no replacement facts: a complete record still passes, and the shape
    of fci25-20261026T124349Z's nine (no age when asked, no replacement stated) still does not."""
    old = {k: v for k, v in recorded(mods, runtime["unchanged"][1]).items()
           if k not in ("frame_replaced", "asked_frame", "age_ms_when_sent")}
    assert mods.check.answer_problem(old) is None
    stale = {**old, "age_ms_when_asked": None}
    assert mods.check.answer_problem(stale) == "no age when asked, and no replacement stated"


# ---------------------------------------------------------------- the reading's item


def item_line(mods, load: Any, records: list[dict[str, Any]]) -> str:
    reading = mods.s5.Reading()
    mods.check._answers(reading, load, records)
    (line,) = reading.lines
    return line


def test_the_item_passes_only_when_every_answer_is_consistent_and_counts_each_reason(mods, runtime) -> None:
    unchanged, replaced = (recorded(mods, runtime[k][1]) for k in ("unchanged", "replaced"))
    line = item_line(mods, [unchanged, replaced], [unchanged, replaced])
    assert line.startswith("PASS real answers") and '"sent_frame_replaced": 1' in line
    assert '"named_and_consistent": 2' in line and '"not_consistent": {}' in line
    stale = {k: v for k, v in unchanged.items() if k not in ("frame_replaced", "asked_frame", "age_ms_when_sent")}
    stale["age_ms_when_asked"] = None
    line = item_line(mods, [unchanged, replaced, stale], [unchanged, replaced, stale])
    assert line.startswith("FAIL real answers") and '"answered": 3, "named_and_consistent": 2' in line
    assert '"no age when asked, and no replacement stated": 1' in line


def test_missing_evidence_stays_missing_and_no_answer_never_passes(mods, runtime) -> None:
    assert item_line(mods, mods.s5.MISSING, []).startswith("MISSING real answers")
    cancelled = {**recorded(mods, runtime["unchanged"][1]), "state": "cancelled"}
    assert item_line(mods, [cancelled], [cancelled]).startswith("FAIL real answers")
