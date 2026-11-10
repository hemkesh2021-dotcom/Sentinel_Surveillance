"""Zooming into a region of a pinned frame, and the scan's tiles (session 116, part 28; scene.roi).

The crop a region question sends is cut from exactly the stated rectangle of exactly the pinned frame's decoded
picture, before it is fitted to the model's image size: shown here with stand-in pictures that record what was cut
from them (the repository's .venv has no numpy), and, where numpy and OpenCV are installed (the device's Python), with
real pixels whose values encode their own coordinates, decoded back from the JPEG that was sent. Bounds, expiry,
replacement, reconnects and the refusals are checked through the real ChatService and scene worker.
"""

from __future__ import annotations

import gc
import http.client
import json
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import pytest
from status_fixtures import incident_store

from sentinel.config import FreshnessConfig
from sentinel.contracts import NormalizedBox, PixelFormat
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.health import assess_frame
from sentinel.media.jpeg import EncodedJpeg, fit_within
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.scene.chat import (
    CHAT_MIN_INTERVAL_S,
    PIN_EXPIRED,
    PIN_MIN_INTERVAL_S,
    REGION_NOTE,
    ChatAdmission,
    ChatQuestion,
    ChatRequestError,
    ChatService,
    LlamaChatRequest,
    chat_request_body,
    parse_chat_request,
)
from sentinel.scene.llama_server import IMAGE_SIZE
from sentinel.scene.roi import (
    MAIN_STREAM,
    MIN_ROI_PX,
    PIN_KEEP_S,
    PINS_KEPT,
    SCAN_OVERLAP,
    WORKING,
    PinnedFrames,
    PreparedCrop,
    RoiError,
    roi_pixels,
    scan_tiles,
)
from sentinel.status_page import StatusServer

ADMITTED = ChatAdmission(True, "admitted for this test", "a synthetic fixture")
NATIVE, WORK = (2304, 1296), (640, 480)


# ---------------------------------------------------------------- the rectangle, in whole pixels of the source


def test_a_region_maps_outward_to_whole_pixels_and_stays_inside_the_picture() -> None:
    assert roi_pixels(2304, 1296, (0.25, 0.25, 0.5, 0.5)) == (576, 324, 1152, 648)
    assert roi_pixels(2304, 1296, (0.0, 0.0, 1.0, 1.0)) == (0, 0, 2304, 1296)
    assert roi_pixels(640, 480, (0.1001, 0.2001, 0.2999, 0.4999)) == (64, 96, 192, 240)  # floor the start, ceil the end
    assert roi_pixels(2304, 1296, (0.97, 0.94, 1.0, 1.0)) == (2234, 1218, 2304, 1296)  # clamped at the far edges
    x1, y1, x2, y2 = roi_pixels(1000, 1000, (0.3333, 0.3333, 0.6667, 0.6667))
    assert (x1, y1) == (333, 333) and (x2, y2) == (667, 667)


@pytest.mark.parametrize("box", [
    (0.5, 0.5, 0.5, 0.9), (0.6, 0.1, 0.4, 0.9), (0.1, 0.9, 0.5, 0.2), (-0.1, 0.0, 0.5, 0.5), (0.0, 0.0, 1.1, 0.5),
    (float("nan"), 0.0, 0.5, 0.5), (float("inf"), 0.0, 0.5, 0.5), (0.1, 0.1, 0.5), ("a", 0, 1, 1), None,
])
def test_a_region_outside_the_picture_or_malformed_is_refused(box: Any) -> None:
    with pytest.raises(RoiError):
        roi_pixels(2304, 1296, box)


def test_a_region_too_small_to_hold_detail_is_refused_rather_than_enlarged() -> None:
    with pytest.raises(RoiError, match="smaller than 64 recorded pixels"):
        roi_pixels(2304, 1296, (0.5, 0.5, 0.52, 0.9))  # 47 pixels wide on the main stream
    with pytest.raises(RoiError):
        roi_pixels(640, 480, (0.5, 0.5, 0.59, 0.9))  # 58 pixels on the substream: the same selection is too small
    assert roi_pixels(2304, 1296, (0.5, 0.5, 0.53, 0.56)) == (1152, 648, 1222, 726)  # 70 x 78: just enough
    assert MIN_ROI_PX == 64


def test_the_scans_tiles_cover_the_whole_picture_and_overlap_their_neighbours() -> None:
    tiles = scan_tiles()
    assert [name for name, _ in tiles] == ["top left", "top right", "bottom left", "bottom right"]
    boxes = [box for _, box in tiles]
    assert min(b[0] for b in boxes) == 0.0 and min(b[1] for b in boxes) == 0.0
    assert max(b[2] for b in boxes) == 1.0 and max(b[3] for b in boxes) == 1.0
    left, right = boxes[0], boxes[1]
    width = left[2] - left[0]
    assert right[0] < left[2] and abs((left[2] - right[0]) / width - SCAN_OVERLAP) < 1e-3  # a tenth of a tile shared
    for x1, y1, x2, y2 in boxes:  # every tile is a usable region of the main stream and of the substream
        assert roi_pixels(*NATIVE, (x1, y1, x2, y2)) and roi_pixels(*WORK, (x1, y1, x2, y2))
    covered = set()
    for box in boxes:
        x1, y1, x2, y2 = roi_pixels(100, 100, box) if False else (round(box[0] * 100), round(box[1] * 100),
                                                                  round(box[2] * 100), round(box[3] * 100))
        covered |= {(x, y) for x in range(x1, x2) for y in range(y1, y2)}
    assert len(covered) == 100 * 100  # no part of the picture is left out
    assert fit_within(1213, 683, *IMAGE_SIZE) == (480, 270)  # a main-stream tile as sent: its own shape, within the cap


# ---------------------------------------------------------------- stand-in pictures that record what was cut from them


class Picture:
    """A decoded picture's stand-in: a size, a name, and a record of every rectangle cut from it."""

    def __init__(self, name: str, width: int, height: int, origin: tuple[int, int] = (0, 0)) -> None:
        self.name, self.shape, self.origin = name, (height, width, 3), origin
        self.nbytes = width * height * 3
        self.cuts: list[tuple[int, int, int, int]] = []

    def __getitem__(self, key: Any) -> Picture:
        rows, cols = key
        cut = (cols.start, rows.start, cols.stop, rows.stop)
        self.cuts.append(cut)
        return Picture(f"{self.name}[{cut[0]},{cut[1]},{cut[2]},{cut[3]}]", cols.stop - cols.start,
                       rows.stop - rows.start, (self.origin[0] + cols.start, self.origin[1] + rows.start))


class Person:
    def __init__(self, track: int, box: tuple[float, float, float, float]) -> None:
        self.track_id, self.box = track, NormalizedBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])


class Rig:
    """The real ChatService and scene worker over a fake clock, stand-in pictures and a model that keeps what it got."""

    def __init__(self, *, main: bool = True) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stamper.connect()
        self.main = main
        self.pins = PinnedFrames()
        self.encoded: list[tuple[str, int, int, int, int]] = []
        self.sent: list[tuple[str, Any]] = []
        self.analyzer = ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="test")
        self.analyzer.start()
        started = self.clock.mono()
        self.gate_open = True
        rig = self

        class Gate:
            def allows_new(self) -> tuple[bool, str | None]:
                return rig.gate_open, None if rig.gate_open else "memory pressure: monitoring comes first"

            def summary(self) -> dict:
                return {}

        self.service = ChatService(
            enabled=True, admission=ADMITTED, scene=self.analyzer, latest=lambda: None,
            assess=lambda f: assess_frame(f, self.stamper.current_stream, self.clock.mono(), started,
                                          FreshnessConfig()),
            clock=self.clock, run=self.model, stale_after_s=2.0, gate=Gate(),  # type: ignore[arg-type]
            pins=self.pins, encode=self.encode)
        self.n = 0
        self.frame()

    def encode(self, image: Any, pixel_format: Any, *, max_width: int, max_height: int, quality: int) -> EncodedJpeg:
        height, width = image.shape[:2]
        out = fit_within(width, height, max_width, max_height)
        self.encoded.append((image.name, max_width, max_height, quality, pixel_format))
        return EncodedJpeg(f"jpeg of {image.name} at {out[0]}x{out[1]}".encode(), *out)

    def model(self, question: Any, image: Any, abort: Any = None, max_tokens: Any = None) -> tuple[str, bool]:
        self.sent.append((question.question, image))
        return "A cup stands on the shelf.", False

    def frame(self, dt: float = 0.1, people: tuple[Person, ...] = ()) -> Any:
        self.clock.advance(dt)
        self.n += 1
        frame = self.stamper.stamp(native_width=NATIVE[0] if self.main else WORK[0],
                                   native_height=NATIVE[1] if self.main else WORK[1], pixel_format=PixelFormat.BGR)
        work, main = Picture(f"work{self.n}", *WORK), Picture(f"main{self.n}", *NATIVE) if self.main else None
        # weak references to this frame's pictures: the store's are the only strong ones, as on the device
        self.newest = (weakref.ref(work), weakref.ref(main) if main is not None else None)
        self.pins.latest(frame, work, main, people)
        return frame

    def hold(self) -> dict[str, Any]:
        code, document = self.service.hold()
        assert code == 200, document
        return document

    def ask(self, picture: str, box: tuple[float, ...], question: str = "What is on the shelf?",
            tile: str | None = None) -> tuple[int, dict[str, Any]]:
        code, document = self.service.ask(ChatQuestion(question, picture=picture, region=box, tile=tile))
        deadline = time.monotonic() + 5
        while code == 202 and document["state"] in ("waiting", "answering"):
            assert time.monotonic() < deadline
            time.sleep(0.005)
            _, document = self.service.poll(document["job"])
        return code, document

    def close(self) -> None:
        assert self.analyzer.stop(3.0)


@pytest.fixture
def rig():
    made: list[Rig] = []

    def make(**options: Any) -> Rig:
        made.append(Rig(**options))
        return made[-1]

    yield make
    for item in made:
        item.close()


def test_a_region_is_cut_from_the_pinned_frames_main_stream_picture_then_fitted_to_the_models_size(rig) -> None:
    r = rig()
    held = r.hold()
    assert held["source"] == MAIN_STREAM and held["source_size"] == [2304, 1296] and held["shown_size"] == [640, 480]
    assert held["keep_s"] == PIN_KEEP_S and held["min_px"] == MIN_ROI_PX and len(held["tiles"]) == 4
    for _ in range(30):  # three more seconds of video: the loop's newest frame moved on thirty times
        r.frame()
    code, document = r.ask(held["picture"], (0.25, 0.25, 0.5, 0.5))
    assert code == 202 and document["state"] == "answered"
    (question, image), = r.sent
    assert isinstance(image, PreparedCrop)
    assert image.jpeg == b"jpeg of main1[576,324,1152,648] at 480x270"  # frame 1's main picture, that rectangle
    assert (image.width, image.height) == (480, 270)  # 576x324 fitted inside 480x360, its own shape kept
    assert r.encoded == [("main1[576,324,1152,648]", 480, 360, 60, PixelFormat.BGR)]  # cut first, then encoded
    assert document["region"] == {"box": [0.25, 0.25, 0.5, 0.5], "pixels": [576, 324, 1152, 648],
                                  "source": MAIN_STREAM, "source_size": [2304, 1296], "sent_size": [480, 270]}
    assert document["picture"] == held["picture"] and document["frame"]["frame_seq"] == held["frame"]["frame_seq"]
    assert document["frame"]["ingest_utc"] == held["frame"]["ingest_utc"]
    assert document["frame"]["age_ms_when_asked"] >= 3000 and document["historical"] is True
    assert "frame_replaced" not in document and r.service.counters["regions"] == 1


def test_a_small_region_is_sent_at_its_own_size_never_enlarged(rig) -> None:
    r = rig()
    held = r.hold()
    _, document = r.ask(held["picture"], (0.5, 0.5, 0.6, 0.6))  # 231 x 130 recorded pixels
    assert document["region"]["pixels"] == [1152, 648, 1383, 778] and document["region"]["sent_size"] == [231, 130]
    assert r.sent[0][1].jpeg == b"jpeg of main1[1152,648,1383,778] at 231x130"
    r.clock.advance(CHAT_MIN_INTERVAL_S)
    r.frame()
    code, refused = r.ask(held["picture"], (0.5, 0.5, 0.52, 0.9))
    assert code == 400 and refused["state"] == "invalid" and "smaller than 64 recorded pixels" in refused["reason"]
    assert len(r.sent) == 1 and r.service.counters["region_refused"] == 1


def test_on_the_substream_a_region_is_cut_from_the_working_picture_and_says_so(rig) -> None:
    r = rig(main=False)
    held = r.hold()
    assert held["source"] == WORKING and held["source_size"] == [640, 480]
    _, document = r.ask(held["picture"], (0.25, 0.25, 0.75, 0.75))
    assert document["region"]["source"] == WORKING and document["region"]["pixels"] == [160, 120, 480, 360]
    assert r.sent[0][1].jpeg == b"jpeg of work1[160,120,480,360] at 320x240"  # no more pixels than were recorded


def test_the_pin_is_that_frame_only_and_a_pin_that_is_gone_is_refused_never_replaced(rig) -> None:
    r = rig()
    first = r.hold()
    r.clock.advance(PIN_MIN_INTERVAL_S)
    r.frame()
    second = r.hold()
    r.clock.advance(PIN_MIN_INTERVAL_S)
    r.frame()
    third = r.hold()  # only PINS_KEPT are held: the first is pushed out
    assert PINS_KEPT == 2 and r.pins.numbers(r.clock.monotonic_ns())["pinned"] == 2
    code, document = r.ask(first["picture"], (0.1, 0.1, 0.9, 0.9))
    assert (code, document) == (410, {"state": "picture_expired", "reason": PIN_EXPIRED})
    assert "it will be a different picture" in PIN_EXPIRED and r.sent == []
    _, answered = r.ask(second["picture"], (0.1, 0.1, 0.9, 0.9))
    assert r.sent[0][1].jpeg.startswith(b"jpeg of main2[")  # the second pin's own frame, not the newest
    r.clock.advance(PIN_KEEP_S + 1)
    r.frame()
    assert r.ask(third["picture"], (0.1, 0.1, 0.9, 0.9))[0] == 410  # expired
    assert r.pins.numbers(r.clock.monotonic_ns()) == {"pinned": 0, "limit": 2, "bytes": 0}
    assert r.ask("0" * 16, (0.1, 0.1, 0.9, 0.9))[0] == 410 and answered["state"] == "answered"


def test_the_pinned_stores_memory_is_bounded_and_reported(rig) -> None:
    r = rig()
    r.hold()
    r.clock.advance(PIN_MIN_INTERVAL_S)
    r.frame()
    r.hold()
    numbers = r.service.pin_numbers()
    one = NATIVE[0] * NATIVE[1] * 3 + WORK[0] * WORK[1] * 3  # 8,957,952 + 921,600 bytes
    assert numbers == {"pinned": 2, "limit": 2, "bytes": 2 * one} and one == 9_879_552
    r.service.set_operator_off(True)  # turned off in Features: nothing stays held
    assert r.service.pin_numbers() == {"pinned": 0, "limit": 2, "bytes": 0}
    r.service.set_operator_off(False)
    r.clock.advance(PIN_MIN_INTERVAL_S)
    assert r.service.hold()[0] == 409  # and the loop's newest frame was released with them: nothing to pin yet
    r.frame()
    r.clock.advance(PIN_MIN_INTERVAL_S)
    r.frame()
    assert r.service.hold()[0] == 200
    r.service.release_pictures()
    assert r.service.pin_numbers()["pinned"] == 0


# ---------------------------------------------------------------- what is released, and when (the finished-code review)


def gone(picture: Any) -> bool:
    """Whether nothing references a picture any more. After a collection: a reference cycle is no reference. This is
    the release of the image itself, which a refused id alone would not show."""
    gc.collect()
    return picture() is None


def test_a_pin_ends_at_its_time_while_no_viewer_asks_anything_and_its_pictures_are_released(rig) -> None:
    r = rig()
    r.hold()
    work, main = r.newest
    r.clock.advance(PIN_KEEP_S - 1)
    r.frame()  # video went on: the newest-frame slot has a later frame
    r.service.summary()  # the runtime's status refresh; no page asks for anything
    assert not gone(main) and not gone(work)  # within its time the pin holds its own frame's pictures
    r.clock.advance(2.0)
    r.frame()
    r.service.summary()  # past its time: the pin ends here, without a request for it
    assert gone(main) and gone(work)
    assert r.service.pin_numbers() == {"pinned": 0, "limit": 2, "bytes": 0}


def test_a_pin_whose_frame_is_still_the_newest_is_released_too_once_the_camera_went_away(rig) -> None:
    r = rig()
    r.hold()
    work, main = r.newest
    r.clock.advance(PIN_KEEP_S + 1)  # no frame since: the newest-frame slot still has the pinned frame's pictures
    r.service.expire_pictures()  # the runtime's loop
    assert gone(main) and gone(work)
    assert r.service.hold()[0] == 409  # nothing to pin until a live frame arrives
    r.frame()
    assert r.service.hold()[0] == 200


def test_a_region_question_takes_its_crop_when_accepted_and_keeps_no_picture_of_the_frame(rig) -> None:
    r = rig()
    release, got = threading.Event(), []

    def slow(question: Any, image: Any, abort: Any = None, max_tokens: Any = None) -> tuple[str, bool]:
        got.append(image)
        release.wait(5.0)
        return "A cup stands on the shelf.", False

    r.service._run = slow
    first = r.hold()
    work, main = r.newest
    code, pending = r.service.ask(ChatQuestion("What is on the shelf?", picture=first["picture"],
                                               region=(0.25, 0.25, 0.5, 0.5)))
    assert code == 202
    assert r.encoded == [("main1[576,324,1152,648]", 480, 360, 60, PixelFormat.BGR)]  # cut and encoded on acceptance
    deadline = time.monotonic() + 5
    while not got:  # the question is on the model now, and stays there
        assert time.monotonic() < deadline
        time.sleep(0.005)
    for _ in range(PINS_KEPT):  # two more held pictures push the first out while its question runs
        r.clock.advance(PIN_MIN_INTERVAL_S)
        r.frame()
        r.hold()
    assert gone(main) and gone(work)  # the running question does not keep its frame alive: the pins are the bound
    assert r.service.pin_numbers()["pinned"] == PINS_KEPT
    release.set()
    while r.service.poll(pending["job"])[1]["state"] != "answered":
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert isinstance(got[0], PreparedCrop) and got[0].jpeg == b"jpeg of main1[576,324,1152,648] at 480x270"
    document = r.service.poll(pending["job"])[1]
    assert document["picture"] == first["picture"] and document["region"]["pixels"] == [576, 324, 1152, 648]


def test_a_region_that_cannot_be_encoded_is_refused_and_no_question_is_made(rig) -> None:
    r = rig()
    held = r.hold()

    def broken(*args: Any, **options: Any) -> EncodedJpeg:
        raise OSError("no encoder")

    r.service._encode = broken
    code, document = r.service.ask(ChatQuestion("What is on the shelf?", picture=held["picture"],
                                                region=(0.25, 0.25, 0.5, 0.5)))
    assert (code, document) == (500, {"state": "error", "reason": "the region could not be encoded (OSError)"})
    assert r.sent == [] and r.service.counters["region_refused"] == 1 and r.service.counters["accepted"] == 0


def test_while_chat_is_off_or_the_runtime_is_stopping_no_frame_is_kept_and_nothing_is_pinned(rig) -> None:
    r = rig()
    r.hold()
    r.service.set_operator_off(True)  # turned off in Features
    r.frame()  # a frame that still reaches the store
    assert gone(r.newest[1]) and gone(r.newest[0])  # is not kept
    r.service.set_operator_off(False)
    r.clock.advance(PIN_MIN_INTERVAL_S)
    r.frame()
    assert r.service.hold()[0] == 200  # turned on again, the store takes frames and pins again
    r.service.release_pictures()  # the runtime is stopping
    r.frame()
    assert gone(r.newest[1]) and gone(r.newest[0])
    r.clock.advance(PIN_MIN_INTERVAL_S)
    assert r.service.hold()[0] == 409 and r.service.pin_numbers()["pinned"] == 0
    r.service.set_operator_off(True)
    r.service.set_operator_off(False)  # a late switch does not reopen what the stop closed
    r.frame()
    assert gone(r.newest[1]) and r.service.hold()[0] == 409


def test_a_reconnect_leaves_a_pin_as_its_own_earlier_frame_marked_historical(rig) -> None:
    r = rig()
    held = r.hold()
    r.stamper.disconnect()
    r.stamper.connect()  # the stream came back as a new epoch
    r.frame()
    _, document = r.ask(held["picture"], (0.0, 0.0, 0.5, 0.5))
    assert document["state"] == "answered" and document["frame"]["stream_epoch"] == held["frame"]["stream_epoch"] == 1
    assert document["historical"] is True and r.sent[0][1].jpeg.startswith(b"jpeg of main1[")
    r.clock.advance(PIN_MIN_INTERVAL_S)
    r.frame()
    again = r.hold()
    assert again["frame"]["stream_epoch"] == 2  # a new pin is the new stream's frame


def test_nothing_is_pinned_from_a_stale_stream_under_the_memory_gate_or_too_often(rig) -> None:
    r = rig()
    r.clock.advance(5.0)  # no new frame for five seconds: stale
    code, document = r.service.hold()
    assert code == 409 and document["state"] == "no_fresh_frame" and r.service.counters["hold_not_live"] == 1
    r.frame()
    r.gate_open = False
    code, document = r.service.hold()
    assert code == 503 and document["state"] == "paused" and r.service.counters["hold_refused"] == 1
    r.gate_open = True
    assert r.service.hold()[0] == 200
    code, document = r.service.hold()
    assert code == 429 and document["state"] == "busy" and r.service.counters["hold_busy"] == 1
    assert r.pins.numbers(r.clock.monotonic_ns())["pinned"] == 1
    plain = ChatService(enabled=True, admission=ADMITTED, scene=r.analyzer, latest=lambda: None, clock=r.clock,
                        assess=lambda f: None, run=r.model, stale_after_s=2.0)  # type: ignore[arg-type,return-value]
    assert plain.hold()[0] == 503 and plain.held_picture("0" * 16) is None  # no pins, no encoder: unavailable
    assert plain.ask(ChatQuestion("x", picture="0" * 16, region=(0, 0, 1, 1)))[0] == 503


def test_a_region_question_shares_every_limit_of_a_question(rig) -> None:
    r = rig()
    held = r.hold()
    assert r.ask(held["picture"], (0.0, 0.0, 0.5, 0.5))[0] == 202
    code, document = r.service.ask(ChatQuestion("And here?", picture=held["picture"], region=(0.5, 0.5, 1.0, 1.0)))
    assert code == 429 and "one question every" in document["reason"]  # the same interval as any question
    r.clock.advance(CHAT_MIN_INTERVAL_S)
    r.frame()
    r.gate_open = False
    code, document = r.service.ask(ChatQuestion("And here?", picture=held["picture"], region=(0.5, 0.5, 1.0, 1.0)))
    assert code == 503 and document["state"] == "paused"  # and the same gate
    r.gate_open = True
    code, document = r.ask(held["picture"], scan_tiles()[3][1], "Describe this part.", tile="bottom right")
    assert code == 202 and document["region"]["tile"] == "bottom right"
    assert document["region"]["pixels"] == [1091, 613, 2304, 1296]


def test_the_trackers_people_on_the_pinned_frame_are_the_only_named_regions(rig) -> None:
    r = rig()
    r.clock.advance(0.1)
    r.frame(people=(Person(61, (0.4, 0.2, 0.6, 0.9)), Person(62, (0.1, 0.3, 0.2, 0.8))))
    held = r.hold()
    assert held["people"] == [{"track_id": 61, "box": [0.4, 0.2, 0.6, 0.9]}, {"track_id": 62, "box": [0.1, 0.3, 0.2, 0.8]}]
    r.frame()  # the next frame has nobody: the pin still has its own frame's people
    _, document = r.ask(held["picture"], tuple(held["people"][0]["box"]))
    assert document["region"]["pixels"] == [921, 259, 1383, 1167]


def test_the_held_picture_is_served_as_the_working_picture_within_its_preview_size(rig) -> None:
    r = rig()
    held = r.hold()
    assert r.service.held_picture(held["picture"]) == b"jpeg of work1 at 640x480"
    assert r.encoded[-1][:3] == ("work1", 640, 480)
    assert r.service.held_picture("f" * 16) is None and r.service.held_picture("../x") is None


# ---------------------------------------------------------------- the request, and what reaches the model


def test_a_region_request_is_validated_and_stands_by_itself() -> None:
    def parse(**fields: Any) -> ChatQuestion:
        return parse_chat_request(json.dumps({"question": "What is on the shelf?", **fields}).encode())

    good = parse(region={"picture": "0123456789abcdef", "box": [0.25, 0.25, 0.5, 0.5]})
    assert (good.picture, good.region, good.tile) == ("0123456789abcdef", (0.25, 0.25, 0.5, 0.5), None)
    assert parse(region={"picture": "0" * 16, "box": [0, 0, 1, 1], "tile": "top left"}).tile == "top left"
    for bad in ({"picture": "0" * 16}, {"box": [0, 0, 1, 1]}, {"picture": "nope", "box": [0, 0, 1, 1]},
                {"picture": "0" * 16, "box": [0, 0, 1]}, {"picture": "0" * 16, "box": [0, 0, 1, "1"]},
                {"picture": "0" * 16, "box": [0, 0, 1, True]}, {"picture": "0" * 16, "box": [0, 0, 1, 1], "tile": "x"},
                {"picture": "0" * 16, "box": [0, 0, 1, 1], "zoom": 4}, "top left"):
        with pytest.raises(ChatRequestError):
            parse(region=bad)
    with pytest.raises(ChatRequestError):  # no earlier conversation is sent with a crop
        parse(region={"picture": "0" * 16, "box": [0, 0, 1, 1]}, history=[{"question": "a", "answer": "b"}])
    with pytest.raises(ChatRequestError):
        parse(region={"picture": "0" * 16, "box": [0, 0, 1, 1]}, auto=True)


def test_the_model_is_told_it_sees_a_crop_and_gets_the_prepared_jpeg_unchanged() -> None:
    question = ChatQuestion("What is on the shelf?", picture="0" * 16, region=(0.25, 0.25, 0.5, 0.5))
    body = json.loads(chat_request_body(b"crop", question))
    assert body["messages"][1]["content"][1]["text"] == (
        f"The operator's question about this image: What is on the shelf?\n{REGION_NOTE}")
    assert body["max_tokens"] == 120  # the same answer limit as any question

    posted: list[bytes] = []

    class Transport:
        def post(self, data: bytes, abort: Any = None) -> bytes:
            posted.append(data)
            return json.dumps({"choices": [{"message": {"role": "assistant", "content": "A cup."},
                                            "finish_reason": "stop"}]}).encode()

    def never(image: Any) -> bytes:
        raise AssertionError("a prepared crop is never resized or encoded again")

    reply = LlamaChatRequest(Transport(), encode=never)(question, PreparedCrop(b"\xff\xd8crop", 480, 270))
    assert reply.answer == "A cup."
    url = json.loads(posted[0])["messages"][1]["content"][0]["image_url"]["url"]
    import base64

    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8crop"


# ---------------------------------------------------------------- the page's two routes


def test_holding_and_reading_the_held_picture_are_for_this_page_only(tmp_path: Path, rig) -> None:
    r = rig()
    incident_store(tmp_path / "sentinel.db", count=1)
    snapshot = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": "2026-11-10T12:00:00+00:00"},
                "degraded": [], "components": {}}
    server = StatusServer(0, lambda: snapshot, tmp_path / "sentinel.db", chat=lambda: r.service)
    server.start()
    port = server.address[1]
    good = {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}", "Content-Type": "application/json",
            "X-Sentinel-Token": server.chat_token, "Sec-Fetch-Site": "same-origin"}

    def call(method: str, path: str, headers: dict, body: bytes | None = None) -> tuple[int, str, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request(method, path, body, headers)
            response = conn.getresponse()
            return response.status, response.getheader("Content-Type") or "", response.read()
        finally:
            conn.close()

    try:
        for headers in ({**good, "Origin": "http://evil.example"}, {**good, "X-Sentinel-Token": "wrong"},
                        {**good, "Sec-Fetch-Site": "cross-site"}, {**good, "Content-Type": "text/plain"}):
            assert call("POST", "/chat/hold", headers, b"{}")[0] in (403, 415)
        assert r.service.counters["held"] == 0
        code, kind, body = call("POST", "/chat/hold", good, b"{}")
        held = json.loads(body)
        assert code == 200 and held["state"] == "held" and held["source"] == MAIN_STREAM
        code, kind, body = call("GET", f"/chat/picture/{held['picture']}", {k: good[k] for k in (
            "Host", "X-Sentinel-Token")})
        assert (code, kind, body) == (200, "image/jpeg", b"jpeg of work1 at 640x480")
        assert call("GET", f"/chat/picture/{held['picture']}", {"Host": good["Host"]})[0] == 403  # no token
        assert call("GET", f"/chat/picture/{'f' * 16}", {k: good[k] for k in ("Host", "X-Sentinel-Token")})[0] == 404
        code, _, body = call("POST", "/chat", good, json.dumps({
            "question": "What is here?", "region": {"picture": held["picture"], "box": [0.0, 0.0, 0.5, 0.5]}}).encode())
        assert code == 202 and json.loads(body)["region"]["pixels"] == [0, 0, 1152, 648]
    finally:
        assert server.stop(3.0)


# ---------------------------------------------------------------- real pixels (numpy and OpenCV: the device's Python)


def test_the_jpeg_sent_decodes_to_the_stated_region_of_the_stated_frame_and_no_other() -> None:
    """Two frames whose pixels encode their own coordinates (blue: column, green: row, red: the frame), with a
    patch of two-pixel stripes only the main stream can resolve. The JPEG a region question sends, decoded, is the stated
    rectangle of the pinned frame's main picture: not the other frame, not another rectangle, and not the working
    copy enlarged. Needs numpy and OpenCV: skipped in the repository's .venv; run with the device's Python."""
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from sentinel.media.jpeg import encode_jpeg
    from sentinel.media.opencv_source import area_resize

    def picture(frame_value: int) -> Any:
        ys, xs = np.mgrid[0:NATIVE[1], 0:NATIVE[0]]
        image = np.empty((NATIVE[1], NATIVE[0], 3), np.uint8)
        image[..., 0] = xs * 255 // (NATIVE[0] - 1)  # blue rises with the column
        image[..., 1] = ys * 255 // (NATIVE[1] - 1)  # green rises with the row
        image[..., 2] = frame_value  # red names the frame
        # two-pixel stripes in a patch of the region: detail the main stream records and the 640x480 copy cannot hold
        stripes = (xs[400:510, 700:910] // 2 % 2 * 80).astype(np.uint8)
        image[400:510, 700:910] = 60 + stripes[..., None]
        return image

    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    pins = PinnedFrames()
    sent: list[Any] = []
    analyzer = ThreadedSceneAnalyzer(lambda job, image: None, RecentImages(), revision="test")
    analyzer.start()
    started = clock.mono()
    service = ChatService(
        enabled=True, admission=ADMITTED, scene=analyzer, latest=lambda: None, clock=clock,
        assess=lambda f: assess_frame(f, stamper.current_stream, clock.mono(), started, FreshnessConfig()),
        run=lambda q, image, abort=None, max_tokens=None: (sent.append(image), ("ok", False))[1], stale_after_s=2.0,
        pins=pins, encode=encode_jpeg)
    try:
        frames = {}
        for value in (60, 200):  # the pinned frame, then a later one
            clock.advance(0.1)
            frame = stamper.stamp(native_width=NATIVE[0], native_height=NATIVE[1], pixel_format=PixelFormat.BGR)
            native = picture(value)
            frames[value] = native
            pins.latest(frame, area_resize(native, *WORK), native)
            if value == 60:
                code, held = service.hold()
                assert code == 200
        code, document = service.ask(ChatQuestion("What is here?", picture=held["picture"], region=(0.25, 0.25, 0.5, 0.5)))
        deadline = time.monotonic() + 10
        while document["state"] in ("waiting", "answering"):
            assert time.monotonic() < deadline
            time.sleep(0.01)
            _, document = service.poll(document["job"])
        assert document["state"] == "answered" and document["region"]["pixels"] == [576, 324, 1152, 648]
        (crop,) = sent
        assert (crop.width, crop.height) == (480, 270) == tuple(document["region"]["sent_size"])
        decoded = cv2.imdecode(np.frombuffer(crop.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (270, 480, 3) and decoded.shape[1] <= IMAGE_SIZE[0] and decoded.shape[0] <= IMAGE_SIZE[1]
        expected = cv2.resize(frames[60][324:648, 576:1152], (480, 270), interpolation=cv2.INTER_AREA).astype(int)
        other_frame = cv2.resize(frames[200][324:648, 576:1152], (480, 270), interpolation=cv2.INTER_AREA).astype(int)
        other_region = cv2.resize(frames[60][0:324, 0:576], (480, 270), interpolation=cv2.INTER_AREA).astype(int)
        whole = cv2.resize(frames[60], (480, 270), interpolation=cv2.INTER_AREA).astype(int)

        def error(reference: Any) -> float:
            return float(np.abs(decoded.astype(int) - reference).mean())

        assert error(expected) < 4.0  # the stated rectangle of the pinned frame, within JPEG loss
        assert error(other_frame) > 30 and error(other_region) > 30 and error(whole) > 20  # and nothing else
        # the corners' colours are the region's own coordinates: columns 576..1152 of 2304, rows 324..648 of 1296
        assert abs(int(decoded[5, 5, 0]) - 576 * 255 // 2303) <= 6 and abs(int(decoded[-5, -5, 0]) - 1151 * 255 // 2303) <= 6
        assert abs(int(decoded[5, 5, 1]) - 324 * 255 // 1295) <= 6 and abs(int(decoded[-5, -5, 1]) - 647 * 255 // 1295) <= 6
        # a smaller region is sent pixel for pixel: the two-pixel stripes survive, which no enlargement of the
        # 640x480 working copy could contain
        clock.advance(CHAT_MIN_INTERVAL_S)
        code, document = service.ask(ChatQuestion("And here?", picture=held["picture"],
                                                  region=(0.30, 0.30, 0.40, 0.40)))
        deadline = time.monotonic() + 10
        while document["state"] in ("waiting", "answering"):
            assert time.monotonic() < deadline
            time.sleep(0.01)
            _, document = service.poll(document["job"])
        x1, y1, x2, y2 = document["region"]["pixels"]
        fine = cv2.imdecode(np.frombuffer(sent[1].jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert fine.shape[:2] == (y2 - y1, x2 - x1) == (131, 231)  # not enlarged: one recorded pixel per pixel

        def stripe_contrast(image: Any) -> float:
            """How strongly columns two pixels apart differ inside the striped patch (its place in this crop)."""
            profile = image[20:115, 15:215].astype(int).mean(axis=(0, 2))
            return float(np.abs(profile[2:] - profile[:-2]).mean())

        enlarged = cv2.resize(area_resize(frames[60], *WORK)[144:192, 192:256], (231, 131))
        assert stripe_contrast(fine) > 40  # the recorded detail is in what was sent
        assert stripe_contrast(enlarged) < stripe_contrast(fine) / 4  # and the small copy, enlarged, never had it
    finally:
        assert analyzer.stop(3.0)
