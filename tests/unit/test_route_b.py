"""Route B (session 116, part 15): one main stream, its working copy for the tracker, and head windows of the same decoded
picture for faces. Portable fakes stand in for OpenCV, DeepFace and the camera: no image library, model or stream runs here,
except one test (skipped without numpy and cv2) that compares DeepFace's cut, transcribed, on a window and on the whole picture.

What is checked, by the risks the maintainer named: one source frame across tracking, window extraction and the worker;
coordinates between the 16:9 picture and the 640x480 working copy, window offsets, edges and landmarks; two people's
ambiguity unresolved; people the window budget leaves out not analysed and given no vote; no fallback embedding without a
face; thresholds unchanged.
"""

from __future__ import annotations

import dataclasses
import random
import threading
import time
from types import SimpleNamespace

import pytest

from sentinel.config import STREAM_MAIN_SINGLE, STREAM_SUB, WORKING_SIZE, CaptureConfig, parse_config
from sentinel.contracts import (
    FrameKey,
    NormalizedBox,
    PixelFormat,
    ResizeTransform,
    TrackObservation,
    TrackStatus,
)
from sentinel.identity.association import (
    AssociationGeometry,
    FaceObservation,
    Ownership,
    _is_candidate,
    associate,
    withhold,
)
from sentinel.identity.gallery import unit
from sentinel.identity.head_windows import Window, plan_windows
from sentinel.identity.legacy_deepface import (
    DETECTOR_WEIGHTS,
    FULL_SCALE_FLOOR,
    MODEL_WEIGHTS,
    YUNET_THRESHOLD,
    FaceBackendError,
    LegacyDeepFaceBackend,
    candidate_region,
    candidate_rows,
)
from sentinel.identity.matching import EnrolledIdentity, Enrollment
from sentinel.identity.state import IdentityState
from sentinel.identity.window_stage import (
    CONTEXT,
    WindowImages,
    WindowNumbers,
    WindowSettings,
    WindowStageError,
    area_to_frame,
    cut_inside,
    prepare,
    select,
)
from sentinel.identity.worker import TICK_OFFERED, TICK_PREPARE_FAILED, FaceWorker
from sentinel.live_state import Capability
from sentinel.media.capture import CapturedFrame, CaptureWorker, DecodedFrame, LatestFrame, WorkingCopy
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.opencv_source import OpenCvSource
from sentinel.runtime import IDENTITY_WINDOWS_SCHEMA, EdgeCore
from sentinel.tracking.tracker import (
    BYTETRACK_LOST_TRACK_FRAMES,
    FrameOutcome,
    PersonTracker,
    RawTrack,
    frame_counted_settings,
)

W, H = 2304, 1296
SQUEEZE = ResizeTransform(output_width=640, output_height=480, scale_x=640 / 2304, scale_y=480 / 1296)
CONFIG = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})


def vec(*head: float) -> tuple[float, ...]:
    return unit(tuple(head) + (0.0,) * (512 - len(head)))


ENROLLED = vec(1.0)
MATCHING = vec(0.97, 0.1, 0.0, 0.05)  # cosine 0.993 with ENROLLED
ENROLLMENT = Enrollment(model_revision="synthetic", dimension=512, identities=(
    EnrolledIdentity(identity_id="idn-00000000000a", prototypes=(ENROLLED,)),))


class Picture:
    """A picture's stand-in: its shape and where it came from, never pixels. Cuts are recorded."""

    def __init__(self, width: int, height: int, origin=(0, 0), scaled_from=None) -> None:
        self.shape = (height, width, 3)
        self.origin, self.scaled_from = origin, scaled_from
        self.cuts: list = []
        self.copies = 0
        self.writeable = True

    def __getitem__(self, key):
        rows, cols = key
        self.cuts.append((cols.start, rows.start, cols.stop, rows.stop))
        return Picture(cols.stop - cols.start, rows.stop - rows.start, (self.origin[0] + cols.start,
                                                                       self.origin[1] + rows.start))

    def copy(self) -> Picture:
        self.copies += 1
        made = Picture(self.shape[1], self.shape[0], self.origin, self.scaled_from)
        made.copied_from = self
        return made

    def setflags(self, write: bool) -> None:
        self.writeable = write

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]


def fake_resize(image: Picture, width: int, height: int) -> Picture:
    return Picture(width, height, image.origin, scaled_from=(image.shape[1], image.shape[0]))


def stamped(clock=None, *, resize=SQUEEZE, width=W, height=H):
    clock = clock or FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stream = stamper.connect()
    return stamper.stamp(native_width=width, native_height=height, pixel_format=PixelFormat.BGR, resize=resize), stream


def person(frame, track: int, box: NormalizedBox) -> TrackObservation:
    return TrackObservation.detected(frame, track_id=track, box=box, confidence=0.9, status=TrackStatus.CONFIRMED)


def nbox(x1, y1, x2, y2) -> NormalizedBox:
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2)


# ---------------------------------------------------------------- slice 1: one frame, two images


class Script:
    """A VideoSource: these decoded pictures, 20 per second, then the stream ends."""

    def __init__(self, pictures, clock) -> None:
        self.pictures, self.clock = list(pictures), clock

    def open(self) -> None:
        pass

    def read(self):
        self.clock.advance(1 / 20)
        return self.pictures.pop(0) if self.pictures else None

    def close(self) -> None:
        pass


class Keeping(LatestFrame):
    def __init__(self) -> None:
        super().__init__()
        self.captured: list[CapturedFrame] = []

    def publish(self, captured: CapturedFrame) -> None:
        self.captured.append(captured)
        super().publish(captured)


def decoded(width=W, height=H, label=0) -> DecodedFrame:
    picture = Picture(width, height)
    picture.label = label
    return DecodedFrame(picture, width, height, PixelFormat.BGR, 50_000 * label)


def capture(pictures, working, profile=STREAM_MAIN_SINGLE):
    clock = FakeClock()
    slot = Keeping()
    worker = CaptureWorker(Script(pictures, clock), FrameStamper("cam-1", clock), slot,
                           CaptureConfig(stream_profile=profile), wait=lambda s: True, working=working)
    worker.run()
    return worker.status(), slot.captured


def test_each_decoded_picture_and_its_working_copy_share_one_frame() -> None:
    first, second = decoded(label=0), decoded(label=1)
    status, captured = capture([first, second], WorkingCopy(*WORKING_SIZE, fake_resize))
    assert [c.frame.frame_seq for c in captured] == [0, 1]
    for item, source in zip(captured, (first, second), strict=True):
        frame = item.frame
        assert (frame.native_width, frame.native_height) == (W, H)  # the decoded picture's own size
        assert frame.resize == SQUEEZE and frame.whole_picture  # the working copy: the whole picture, squeezed
        assert (frame.processed_width, frame.processed_height) == WORKING_SIZE
        assert item.full is source.image  # the face stage cuts from the decoded picture itself
        assert item.image.shape == (480, 640, 3) and item.image.scaled_from == (W, H)
    assert status.frames == 2 and status.problem == "no_frame"  # the script's end, nothing else


def test_a_picture_not_larger_than_the_working_copy_is_refused_and_nothing_is_published() -> None:
    status, captured = capture([decoded(640, 480)], WorkingCopy(*WORKING_SIZE, fake_resize))
    assert captured == [] and status.problem == "not_larger_than_working" and status.frames == 0


def test_a_failed_or_misshapen_working_copy_ends_the_connection_with_a_label() -> None:
    def raises(image, width, height):
        raise RuntimeError("never kept: it could name the camera")

    status, captured = capture([decoded()], WorkingCopy(*WORKING_SIZE, raises))
    assert captured == [] and status.problem == "working_copy_error:RuntimeError"
    status, captured = capture([decoded()], WorkingCopy(*WORKING_SIZE, lambda image, w, h: Picture(w, h - 1)))
    assert captured == [] and status.problem == "working_copy_size"


def test_without_a_working_copy_nothing_changes() -> None:
    source = decoded(640, 480)
    status, captured = capture([source], None, profile=STREAM_SUB)
    (item,) = captured
    assert item.image is source.image and item.full is None
    assert item.frame.resize == ResizeTransform.identity(640, 480) and item.frame.whole_picture


def test_the_profile_is_the_substream_by_default_and_main_single_decodes_with_two_threads() -> None:
    default = parse_config({"config_version": 1, "camera": {"id": "cam-1"}}).capture
    assert (default.stream_profile, default.decode_threads, default.stream_decode_threads) == (STREAM_SUB, 1, 1)
    main = parse_config({"config_version": 1, "camera": {"id": "cam-1"},
                         "capture": {"stream_profile": "main_single"}}).capture
    assert (main.main_decode_threads, main.stream_decode_threads) == (2, 2)
    one = CaptureConfig(stream_profile="main_single", main_decode_threads=1)
    assert one.stream_decode_threads == 1 and CaptureConfig(main_decode_threads=1).stream_decode_threads == 1
    for bad in ({"stream_profile": "main"}, {"main_decode_threads": 0}, {"main_decode_threads": 5}):
        with pytest.raises(ValueError):
            CaptureConfig(**bad)
    opened = []

    class Cap:
        def __init__(self, target, api, params) -> None:
            self.params = params
            opened.append(self)

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
            return True

        def release(self) -> None:
            pass

    cv2 = SimpleNamespace(CAP_PROP_OPEN_TIMEOUT_MSEC=53, CAP_PROP_READ_TIMEOUT_MSEC=54, CAP_PROP_N_THREADS=70,
                          CAP_FFMPEG=1900, VideoCapture=Cap)
    for config, threads in ((default, 1), (main, 2)):
        OpenCvSource("clip.mp4", config, cv2=cv2).open()
        assert opened[-1].params[4:] == [70, threads]


class Backend:
    def __init__(self, raw) -> None:
        self.raw = list(raw)

    def track(self, image):
        return list(self.raw)

    def reset(self) -> None:
        pass


class Image:
    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)


def test_the_tracker_works_on_the_working_copy_and_its_boxes_mean_the_same_place_in_the_picture() -> None:
    raw = [RawTrack(3, 64.0, 48.0, 320.0, 480.0, 0.75)]
    frame, _ = stamped()
    routed = PersonTracker(Backend(raw)).process(frame, Image(640, 480))
    substream, _ = stamped(resize=None, width=640, height=480)
    plain = PersonTracker(Backend(raw)).process(substream, Image(640, 480))
    assert routed.outcome is FrameOutcome.PROCESSED and routed.persons[0].frame == frame.key
    assert routed.persons[0].box == plain.persons[0].box == nbox(0.1, 0.1, 0.5, 1.0)
    # the same box in the decoded picture's pixels: 230.4-1152 x 129.6-1296 of 2304x1296
    box = routed.persons[0].box
    assert (box.x1 * W, box.x2 * W, box.y1 * H, box.y2 * H) == pytest.approx((230.4, 1152.0, 129.6, 1296.0))
    wrong = PersonTracker(Backend(raw)).process(stamped()[0], Image(W, H))  # the decoded picture is not its input
    assert wrong.outcome is FrameOutcome.FAILED and wrong.problem == "image_size_mismatch"


def test_the_frame_counted_tracker_settings_are_unchanged_and_their_duration_is_explicit() -> None:
    from sentinel.inference.legacy_ultralytics import TRACK_ARGS

    assert TRACK_ARGS["tracker"] == "bytetrack.yaml" and BYTETRACK_LOST_TRACK_FRAMES == 30  # nothing overridden
    reading = frame_counted_settings(2, 20.0)
    assert reading["frames"] == {"bytetrack_lost_track": 30, "confirm_detections": 2, "unconfirmed_after_misses": 4}
    assert reading["seconds"] == {"bytetrack_lost_track": 1.5, "confirm_detections": 0.1,
                                  "unconfirmed_after_misses": 0.2}
    assert reading["seconds_at_substream_fps"]["bytetrack_lost_track"] == 2.0
    assert frame_counted_settings(2, None)["seconds"] is None  # no rate measured yet: no duration claimed


# ---------------------------------------------------------------- slice 2: the windows


FAR = nbox(0.45, 0.30, 0.50, 0.62)  # about 115 px wide in the decoded picture: one 640x640 window, scale 1
NEAR = nbox(0.02, 0.05, 0.40, 1.0)  # about 876 px wide, beside FAR: a larger window, scaled down


def test_the_offer_copies_only_the_windows_at_their_detector_size() -> None:
    frame, _ = stamped()
    persons = (person(frame, 1, FAR), person(frame, 2, NEAR))
    picture = Picture(W, H)
    images = prepare(frame, persons, picture, resize=fake_resize, settings=WindowSettings())
    assert picture.copies == 0  # the decoded picture itself is never copied
    assert len(images.windows) == 2 and images.uncovered == ()
    for window, image in zip(images.windows, images.inputs, strict=True):
        w_in, h_in = window.input_size
        assert max(w_in, h_in) <= 640 and image.shape[:2] == (h_in, w_in) and image.writeable is False
        assert image.origin == (window.x, window.y)  # cut at the window's own offset
        if window.scale == 1.0:
            assert image.copied_from.shape[:2] == (window.h, window.w)
        else:
            assert image.scaled_from == (window.w, window.h)
    assert images.nbytes <= 4 * 640 * 640 * 3
    assert sorted(k.track_id for owners in images.owners for k in owners) == [1, 2]


def test_the_offer_refuses_a_padded_frame_another_image_and_another_frames_people() -> None:
    frame, _ = stamped()
    padded, _ = stamped(resize=ResizeTransform.letterbox(W, H, 640, 640))
    with pytest.raises(WindowStageError, match="not_whole_picture"):
        prepare(padded, (), Picture(W, H), resize=fake_resize)
    with pytest.raises(WindowStageError, match="image_size_mismatch"):
        prepare(frame, (person(frame, 1, FAR),), Picture(640, 480), resize=fake_resize)
    other, _ = stamped()
    with pytest.raises(WindowStageError, match="persons_from_another_frame"):
        prepare(frame, (person(other, 1, FAR),), Picture(W, H), resize=fake_resize)


def random_box(rng: random.Random) -> NormalizedBox:
    w, h = rng.uniform(0.02, 0.5), rng.uniform(0.1, 0.95)
    x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    return nbox(x, y, x + w, y + h)


def test_any_layout_stays_within_the_copy_bound_and_every_person_is_covered_or_listed() -> None:
    rng = random.Random(1161502)
    frame, _ = stamped()
    for _ in range(400):
        persons = tuple(person(frame, i + 1, random_box(rng)) for i in range(rng.randint(1, 9)))
        images = prepare(frame, persons, Picture(W, H), resize=fake_resize)
        assert images.nbytes <= 4 * 640 * 640 * 3 and len(images.windows) <= 4
        listed = [k for owners in images.owners for k in owners] + list(images.uncovered)
        assert sorted(k.track_id for k in listed) == [p.track_id for p in persons]  # each exactly once


def test_every_face_association_could_give_a_covered_person_lies_inside_that_persons_window() -> None:
    """The stated precondition: a candidate face at most 1.25 times as wide as the person box, and no taller than the box
    is wide (pixels), lies whole inside the window covering that person."""
    rng = random.Random(1161503)
    geometry = AssociationGeometry()
    checked = 0
    for _ in range(3000):
        box = random_box(rng)
        plan = plan_windows([box], W, H)
        (window,) = plan.windows
        pw = (box.x2 - box.x1) * W
        fw, fh = rng.uniform(0.05, 1.25) * pw, rng.uniform(0.05, 1.0) * pw
        cx = rng.uniform(box.x1, box.x2) * W
        cy = rng.uniform(box.y1, box.y1 + geometry.head_fraction * (box.y2 - box.y1)) * H
        x1, y1, x2, y2 = cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue  # beyond the picture: neither path sees it whole
        face = nbox(x1 / W, y1 / H, x2 / W, y2 / H)
        if not _is_candidate(face, box, geometry):
            continue
        checked += 1
        assert window.x <= x1 + 1e-6 and window.y <= y1 + 1e-6
        assert x2 <= window.x + window.w + 1e-6 and y2 <= window.y + window.h + 1e-6
    assert checked > 500


def test_a_detection_maps_back_to_the_decoded_picture_with_its_offset_scale_and_eyes() -> None:
    window = Window(1000, 200, 640, 640, 1.0, (0,))
    (row,) = candidate_rows([[300.25, 100.5, 40.0, 48.0, 312.0, 118.0, 328.0, 118.5, 0, 0, 0, 0, 0, 0, 0.95]])
    assert window.to_frame(row.x, row.y) == (1300.25, 300.5)  # an integer offset at scale 1
    assert area_to_frame(window, {"x": 300, "y": 100, "w": 40, "h": 48}) == {"x": 1300.0, "y": 300.0, "w": 40.0,
                                                                            "h": 48.0}
    assert window.to_frame(*row.eye_4) == (1312.0, 318.0) and window.to_frame(*row.eye_6) == (1328.0, 318.5)
    # a window wider than it is tall (the picture's height cut it) and scaled: each axis by its own exact scale
    wide = Window(0, 0, 1500, 1296, 640 / 1500, (0,))
    assert wide.input_size == (640, 553)
    x, y = wide.to_frame(320.0, 276.5)
    assert (x, y) == pytest.approx((750.0, 648.0))  # the input's centre is the window's centre
    assert wide.to_frame(640, 553) == pytest.approx((1500.0, 1296.0))  # and its far corner the window's


def test_a_cut_may_meet_black_beyond_the_picture_but_never_beyond_a_window_edge_inside_it() -> None:
    (row,) = candidate_rows([[10.0, 300.0, 40.0, 48.0, 20.0, 315.0, 36.0, 315.0, 0, 0, 0, 0, 0, 0, 0.95]])
    # grown by int(0.5 * 40) = 20 on each side, the cut starts at x = -10: past the window's left edge
    assert cut_inside(row, 640, 640, (False, True, True, True))  # that edge is the picture's: black there, as in D88
    assert not cut_inside(row, 640, 640, (True, True, True, True))  # inside the picture: pixels the window lacks
    (clear,) = candidate_rows([[20.0, 300.0, 40.0, 48.0, 30.0, 315.0, 46.0, 315.0, 0, 0, 0, 0, 0, 0, 0.95]])
    assert cut_inside(clear, 640, 640, (True, True, True, True))  # exactly at the edge: whole
    (bottom,) = candidate_rows([[300.0, 600.0, 40.0, 40.0, 310.0, 610.0, 326.0, 610.0, 0, 0, 0, 0, 0, 0, 0.95]])
    assert not cut_inside(bottom, 640, 640, (True, True, True, True))
    assert cut_inside(bottom, 640, 640, (True, True, True, False))


def test_at_scale_one_deepfaces_region_in_a_window_is_the_pictures_moved_by_the_window_offset() -> None:
    """DeepFace's cut depends on its integer region (box and eyes) and the pixels of the box grown by half; in a window at
    scale 1 every one of those integers is the picture's minus the window's offset, so the cut is the same pixels."""
    rng = random.Random(1161504)
    for _ in range(2000):
        wx, wy = rng.randint(0, W - 640), rng.randint(0, H - 640)
        x, y = rng.uniform(30, 560), rng.uniform(30, 560)
        w, h = rng.uniform(8, 60), rng.uniform(8, 70)
        local = [x, y, w, h, x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.41 * h, 0, 0, 0, 0, 0, 0, 0.93]
        whole = [v + (wx if i in (0, 4, 6) else wy if i in (1, 5, 7) else 0) if i < 8 and i not in (2, 3) else v
                 for i, v in enumerate(local)]
        (in_window,), (in_picture,) = candidate_rows([local]), candidate_rows([whole])
        a = candidate_region(in_window, 320, 320)  # the window's own border, int(0.5 * 640)
        b = candidate_region(in_picture, W // 2, H // 2)  # D88's border on the decoded picture
        assert (a["x"] - 320 + wx, a["y"] - 320 + wy, a["w"], a["h"]) == (b["x"] - W // 2, b["y"] - H // 2, b["w"], b["h"])
        for eye in ("left_eye", "right_eye"):
            assert (a[eye][0] - 320 + wx, a[eye][1] - 320 + wy) == (b[eye][0] - W // 2, b[eye][1] - H // 2)


def images_for(windows, owners=None, uncovered=()) -> WindowImages:
    owners = owners or tuple((key(i + 1),) for i in range(len(windows)))
    return WindowImages(W, H, tuple(windows), owners, tuple(uncovered),
                        tuple(Picture(*w.input_size) for w in windows))


def key(track: int):
    frame, _ = stamped()
    return person(frame, track, FAR).key


def row(x, y, w, h, score):
    (made,) = candidate_rows([[x, y, w, h, x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.41 * h, 0, 0, 0, 0, 0, 0,
                               score]])
    return made


def test_a_face_two_windows_see_whole_counts_once_and_its_partial_explains_nothing_away() -> None:
    a, b = Window(400, 200, 640, 640, 1.0, (0,)), Window(800, 200, 640, 640, 1.0, (1,))
    # the same face at picture x 900-940: in a at input x 500, in b at input x 100; both see it whole
    selection = select([[row(500.0, 300.0, 40.0, 48.0, 0.95)], [row(100.0, 300.0, 40.0, 48.0, 0.93)]],
                       images_for([a, b]), YUNET_THRESHOLD)
    assert selection.kept == ((0, 0),) and selection.dropped_duplicate == 1 and selection.not_analysed == ()
    # a face at picture x 1020-1060 is cut by a's right edge (1040) and whole in b: a's partial is part of b's face
    selection = select([[row(620.0, 300.0, 20.0, 48.0, 0.91)], [row(220.0, 300.0, 40.0, 48.0, 0.94)]],
                       images_for([a, b]), YUNET_THRESHOLD)
    assert selection.kept == ((1, 0),) and selection.dropped_at_window_edge == 1
    assert selection.unexplained_windows == () and selection.not_analysed == ()


def test_a_face_a_window_cannot_see_whole_withholds_everyone_that_window_covers() -> None:
    a, b = Window(400, 200, 640, 640, 1.0, (0,)), Window(1600, 200, 640, 640, 1.0, (1,))
    owners = ((key(1), key(2)), (key(3),))
    selection = select([[row(620.0, 300.0, 20.0, 48.0, 0.91), row(100.0, 100.0, 40.0, 48.0, 0.97)],
                        [row(100.0, 100.0, 40.0, 48.0, 0.96)]], images_for([a, b], owners, (key(4),)),
                       YUNET_THRESHOLD)
    assert selection.kept == ((0, 1), (1, 0)) and selection.unexplained_windows == (0,)
    assert [k.track_id for k in selection.not_analysed] == [1, 2, 4]  # a's people and the one left uncovered


def test_a_cut_face_beside_a_whole_face_of_its_own_window_is_never_explained_away() -> None:
    """Review finding (session 116, part 16): YuNet kept two overlapping faces apart in one window (IoU 0.294); the second
    one's cut reaches past the window's top edge. Explaining it by its neighbour would drop a second candidate and turn an
    AMBIGUOUS person into an assignment, so the window's people are not analysed."""
    window = Window(830, 145, 640, 640, 1.0, (0,))
    selection = select([[row(190.0, 155.0, 230.0, 260.0, 0.95), row(220.0, 93.0, 160.0, 190.0, 0.93)]],
                       images_for([window]), YUNET_THRESHOLD)
    assert selection.kept == ((0, 0),) and selection.dropped_at_window_edge == 1
    assert selection.unexplained_windows == (0,) and [k.track_id for k in selection.not_analysed] == [1]
    # nor by the neighbour's twin from another window: that face may be the neighbour, not the one cut
    other = Window(880, 160, 640, 640, 1.0, (1,))
    twin = row(140.0, 140.0, 230.0, 260.0, 0.97)  # the same neighbour (picture 1020, 300), whole in the other window
    selection = select([[row(190.0, 155.0, 230.0, 260.0, 0.95), row(220.0, 93.0, 160.0, 190.0, 0.93)], [twin]],
                       images_for([window, other]), YUNET_THRESHOLD)
    assert selection.kept == ((1, 0),) and selection.dropped_duplicate == 1
    assert selection.unexplained_windows == (0,) and [k.track_id for k in selection.not_analysed] == [1]


def test_only_detections_at_the_threshold_count() -> None:
    window = Window(400, 200, 640, 640, 1.0, (0,))
    selection = select([[row(100.0, 100.0, 40.0, 48.0, 0.8999), row(300.0, 100.0, 40.0, 48.0, YUNET_THRESHOLD)]],
                       images_for([window]), YUNET_THRESHOLD)
    assert selection.kept == ((0, 1),) and selection.detected == 1
    assert YUNET_THRESHOLD == 0.9 and CONTEXT == 0.5


# ---------------------------------------------------------------- the adapter's window path, with fakes


class Cut:
    def __init__(self, index: int) -> None:
        self.shape, self.index = (80, 64, 3), index

    def __truediv__(self, other):
        return self


class Detector:
    def __init__(self) -> None:
        self.per_call: list = []
        self.calls: list = []

    def setInputSize(self, size) -> None:  # noqa: N802
        self.calls.append(("size", size))

    def setScoreThreshold(self, value) -> None:  # noqa: N802
        self.calls.append(("threshold", value))

    def detect(self, image):
        self.calls.append(("detect", image))
        return 1, (self.per_call.pop(0) if self.per_call else [])


class Runtime:
    """Just enough of TensorFlow, OpenCV and DeepFace for load() and the window path; calls recorded."""

    def __init__(self) -> None:
        self.calls: list = []
        self.detector = Detector()
        self.model = object()
        self.cache: dict = {"facial_recognition": {}, "face_detector": {}}
        self.ns = SimpleNamespace(
            tf=SimpleNamespace(config=SimpleNamespace(get_visible_devices=lambda kind: []), __version__="2.21.0"),
            cv2=SimpleNamespace(__version__="4.13.0", BORDER_CONSTANT=0, copyMakeBorder=self._border),
            numpy=SimpleNamespace(__version__="2.2.6", zeros=lambda shape, dtype: ("zeros", shape), uint8="uint8"),
            tf_keras=SimpleNamespace(__version__="2.21.0"), deepface=SimpleNamespace(__version__="0.0.99"),
            DeepFace=SimpleNamespace(build_model=self._build, represent=self.represent),
            detection=SimpleNamespace(extract_face=self.extract_face),
            modeling=SimpleNamespace(cached_models=self.cache), FacialAreaRegion=_Region)

    def _build(self, name) -> None:
        self.cache["facial_recognition"][name] = self.model

    def _border(self, image, top, bottom, left, right, kind, value):
        self.calls.append(("border", image, top, bottom, left, right))
        return ("padded", image)

    def extract_face(self, facial_area, img, align, expand_percentage, width_border, height_border, detector_backend):
        self.calls.append(("extract", dataclasses.asdict(facial_area), width_border, height_border))
        index = sum(1 for c in self.calls if c[0] == "extract") - 1
        return SimpleNamespace(img=Cut(index), facial_area=_Region(
            x=facial_area.x - width_border, y=facial_area.y - height_border, w=facial_area.w, h=facial_area.h))

    def represent(self, img_path, model_name="VGG-Face", enforce_detection=True, detector_backend="opencv",
                  align=True, expand_percentage=0, normalization="base", **_):
        self.calls.append(("represent", detector_backend))
        if detector_backend != "skip":  # the warm-up: DeepFace builds and caches its YuNet client
            self.cache["face_detector"]["yunet"] = SimpleNamespace(model=self.detector)
            return [{"embedding": [0.0] * 511 + [1.0], "facial_area": {"x": 0, "y": 0, "w": 640, "h": 480},
                     "face_confidence": 0}]
        images = img_path if isinstance(img_path, list) else [img_path]
        out = [[{"embedding": list(MATCHING), "facial_area": {}, "face_confidence": 0}] for _ in images]
        return out[0] if len(images) == 1 else out


@dataclasses.dataclass
class _Region:
    x: int
    y: int
    w: int
    h: int
    left_eye: tuple | None = None
    right_eye: tuple | None = None
    confidence: float | None = None
    nose: tuple | None = None
    mouth_right: tuple | None = None
    mouth_left: tuple | None = None


def loaded(tmp_path, runtime: Runtime, *, select_windows: bool = True) -> LegacyDeepFaceBackend:
    for name, _ in (MODEL_WEIGHTS, DETECTOR_WEIGHTS):
        (tmp_path / name).write_bytes(b"synthetic")
    digests = {MODEL_WEIGHTS[0]: MODEL_WEIGHTS[1], DETECTOR_WEIGHTS[0]: DETECTOR_WEIGHTS[1]}
    made = LegacyDeepFaceBackend(tmp_path, importer=lambda: runtime.ns, sha256=lambda p: digests[p.name])
    if select_windows:
        made.select_window_detection()
    made.load()
    return made


def two_windows():
    frame, _ = stamped()
    persons = (person(frame, 1, FAR), person(frame, 2, NEAR))
    return frame, persons, prepare(frame, persons, Picture(W, H), resize=fake_resize)


def test_a_frame_without_a_face_at_the_threshold_is_never_embedded(tmp_path) -> None:
    runtime = Runtime()
    made = loaded(tmp_path, runtime)
    frame, persons, images = two_windows()
    runtime.detector.per_call = [[[100.0, 100.0, 40.0, 48.0, 112, 118, 128, 118, 0, 0, 0, 0, 0, 0, 0.6]], []]
    at_load = len(runtime.calls)
    run = made.window_faces(frame.key, images)
    assert runtime.calls[at_load:] == []  # no border, cut or represent: no fallback embedding
    assert run.faces == () and run.fallback_dropped == 0 and run.detection.embed_ms is None
    assert run.detection.best_below[0] == 0.6 and run.windows.detected == 0
    sizes = [c[1] for c in runtime.detector.calls if c[0] == "size"]
    assert sizes == [w.input_size for w in images.windows]  # each window at its own input size, at most 640
    assert {c[1] for c in runtime.detector.calls if c[0] == "threshold"} == {FULL_SCALE_FLOOR}


def test_a_face_in_a_window_is_reported_in_the_whole_pictures_coordinates(tmp_path) -> None:
    runtime = Runtime()
    made = loaded(tmp_path, runtime)
    frame, persons, images = two_windows()
    far = images.windows[[i for i, o in enumerate(images.owners) if o[0].track_id == 1][0]]
    assert far.scale == 1.0
    # the far person's face, 40x48 at picture (1060.6, 410.2): in its window at that minus the window's offset
    fx, fy = 1060.6 - far.x, 410.2 - far.y
    rows = [[fx, fy, 40.0, 48.0, fx + 12, fy + 18, fx + 28, fy + 18.4, 0, 0, 0, 0, 0, 0, 0.953]]
    runtime.detector.per_call = [rows, []] if images.windows[0] is far else [[], rows]
    at_load = len(runtime.calls)
    run = made.window_faces(frame.key, images)
    (face,) = run.faces
    assert face.frame == frame.key and face.quality == 0.95
    assert (face.box.x1, face.box.y1) == pytest.approx((1060 / W, 410 / H))  # extract's int() box, moved back
    assert (face.box.x2, face.box.y2) == pytest.approx((1100 / W, 458 / H))
    (geometry,) = run.geometry
    assert geometry.raw_box == pytest.approx((1060.6, 410.2, 40.0, 48.0))  # D90: YuNet's own, in picture pixels
    assert geometry.right_eye == pytest.approx((1072.6, 428.2)) and geometry.left_eye == pytest.approx((1088.6, 428.6))
    assert [c[0] for c in runtime.calls[at_load:]] == ["border", "extract", "represent"]  # one window, one batch
    association = associate(persons, run.faces)
    assert association.person_ownership[persons[0].key] is Ownership.ASSIGNED  # the far person's own face


def test_the_window_path_excludes_the_whole_picture_and_the_diagnostics(tmp_path) -> None:
    runtime = Runtime()
    made = loaded(tmp_path, runtime)
    with pytest.raises(FaceBackendError, match="windows_need_window_images"):
        made.faces(FrameKey(camera_id="cam-1", boot_id="b", run_id="r", stream_epoch=1, frame_seq=0), W, H, "x")
    first = LegacyDeepFaceBackend(tmp_path, importer=lambda: runtime.ns)
    first.select_window_detection()
    with pytest.raises(FaceBackendError, match="candidate_excludes_diagnostics"):
        first.enable_full_scale()
    second = LegacyDeepFaceBackend(tmp_path, importer=lambda: runtime.ns)
    second.enable_full_scale()
    with pytest.raises(FaceBackendError, match="candidate_excludes_diagnostics"):
        second.select_window_detection()
    plain = loaded(tmp_path, Runtime(), select_windows=False)
    frame, persons, images = two_windows()
    with pytest.raises(FaceBackendError, match="windows_not_selected"):
        plain.window_faces(frame.key, images)
    assert made.compatibility() == plain.compatibility()  # the enrollment's record still matches


def test_the_identity_thresholds_are_unchanged() -> None:
    identity = CONFIG.identity
    assert (identity.match_threshold, identity.margin, identity.min_quality, identity.confirmations,
            identity.vote_ttl_s, identity.head_fraction, identity.min_face_inside) == (0.70, 0.05, 0.6, 2, 30.0, 0.4,
                                                                                    0.6)
    assert YUNET_THRESHOLD == 0.9 and WindowSettings(head_fraction=identity.head_fraction).head_fraction == 0.4


# ---------------------------------------------------------------- the worker: one frame from offer to result


def test_the_offer_prepares_windows_and_the_result_keeps_its_own_frame_while_later_frames_arrive() -> None:
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    gate, started = threading.Event(), threading.Event()
    seen: list = []

    def analyze(frame, part):
        seen.append((frame.key, part))
        started.set()
        gate.wait(5)
        run_faces = (FaceObservation(frame=frame.key, box=nbox(0.46, 0.32, 0.48, 0.36), quality=0.95,
                                     embedding=MATCHING),)
        return SimpleNamespace(faces=run_faces, detected=1, fallback_dropped=0, windows="numbers")

    def refuse_copy(image):
        raise AssertionError("route B never copies the whole picture")

    worker = FaceWorker(analyze, clock, copy=refuse_copy,
                        prepare=lambda frame, persons, image: ("windows of", frame.key, tuple(p.key for p in persons),
                                                               image))
    worker.start()
    try:
        offered = stamper.stamp(native_width=W, native_height=H, pixel_format=PixelFormat.BGR, resize=SQUEEZE)
        people = (person(offered, 1, FAR),)
        picture = Picture(W, H)
        assert worker.consider(offered, people, picture) == TICK_OFFERED
        assert started.wait(5)
        later = []
        for _ in range(15):  # the loop moves on while the analysis runs
            clock.advance(1 / 20)
            frame = stamper.stamp(native_width=W, native_height=H, pixel_format=PixelFormat.BGR, resize=SQUEEZE)
            later.append(frame.key)
            worker.consider(frame, (person(frame, 1, FAR),), Picture(W, H))
        gate.set()
        deadline = time.monotonic() + 5
        result = None
        while result is None:
            assert time.monotonic() < deadline
            result = worker.drain()
            time.sleep(0.002)
        assert seen[0] == (offered.key, ("windows of", offered.key, (people[0].key,), picture))
        assert result.frame.key == offered.key and offered.key not in later
        assert [p.frame for p in result.persons] == [offered.key]
        assert [f.frame for f in result.faces] == [offered.key] and result.windows == "numbers"
    finally:
        gate.set()
        worker.stop(5)


def test_a_frame_whose_windows_cannot_be_made_is_not_offered() -> None:
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()

    def broken(frame, persons, image):
        raise WindowStageError("image_size_mismatch")

    worker = FaceWorker(lambda frame, part: None, clock, prepare=broken)
    frame = stamper.stamp(native_width=W, native_height=H, pixel_format=PixelFormat.BGR, resize=SQUEEZE)
    assert worker.consider(frame, (person(frame, 1, FAR),), Picture(640, 480)) == TICK_PREPARE_FAILED
    assert worker.counters["prepare_errors"] == 1 and worker.counters["prepare_error:image_size_mismatch"] == 1
    assert worker.counters["offered"] == 0 and worker.status()["waiting"] is False


# ---------------------------------------------------------------- the identity rules on route B's results


class World:
    """The real EdgeCore on main-stream frames, fed faces and window numbers as the window path makes them."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stream = self.stamper.connect()
        self.core = EdgeCore(CONFIG, self.clock, None, face_recognition=Capability.AVAILABLE, enrollment=ENROLLMENT)

    def step(self, people, faces, not_analysed=()):
        self.clock.advance(1.0)
        frame = self.stamper.stamp(native_width=W, native_height=H, pixel_format=PixelFormat.BGR, resize=SQUEEZE)
        persons = tuple(person(frame, track, box) for track, box in people)
        self.state = self.core.on_frame(frame, persons, self.stream).state
        observed = tuple(FaceObservation(frame=frame.key, box=box, quality=0.95, embedding=MATCHING) for box in faces)
        withheld = tuple(p.key for p in persons if p.track_id in not_analysed)
        numbers = WindowNumbers(1, ((640, 640),), (1.0,), len(persons), len(withheld), 0, withheld, len(faces), 0, 0,
                                0)
        output = self.core.on_face_outcome(frame, persons, observed, self.stream, processing_ms=900.0,
                                           windows=numbers)
        self.state = output.state
        return output.face_result.record(IDENTITY_WINDOWS_SCHEMA)

    def identity(self, track: int):
        return {p.track_id: p for p in self.state.people}[track].identity


A, B = nbox(0.40, 0.20, 0.48, 0.90), nbox(0.43, 0.22, 0.52, 0.90)  # overlapping people, side by side
SHARED = nbox(0.44, 0.24, 0.46, 0.30)  # in both head regions
OWN_B = nbox(0.49, 0.25, 0.51, 0.31)  # in B's only


def test_a_face_two_people_could_own_stays_unresolved_on_route_b() -> None:
    record = World().step([(1, A), (2, B)], [SHARED])
    assert [(p["ownership"], p["vote"]) for p in record["persons"]] == [("ambiguous", "none")] * 2
    assert record["schema"] == IDENTITY_WINDOWS_SCHEMA and record["windows"]["not_analysed"] == []


def test_a_person_not_analysed_stays_in_the_association_and_gets_no_vote() -> None:
    world = World()
    record = world.step([(1, A), (2, B)], [SHARED], not_analysed=(2,))
    by_track = {p["track_id"]: p for p in record["persons"]}
    assert (by_track[1]["ownership"], by_track[1]["vote"]) == ("ambiguous", "none")  # never handed to the other person
    assert (by_track[2]["ownership"], by_track[2]["vote"], by_track[2]["label"]) == ("not_analysed", "none",
                                                                                     "not_analysed")
    assert record["windows"]["not_analysed"] == [2]
    # what the runtime avoids: deciding over the analysed person alone would have assigned the shared face
    frame, _ = stamped()
    alone = associate([person(frame, 1, A)], [FaceObservation(frame=frame.key, box=SHARED, quality=0.95)])
    assert list(alone.person_ownership.values()) == [Ownership.ASSIGNED]
    assert world.core.face_counters["faces_ambiguous"] == 1
    assert world.core._identities.counters["persons_not_analysed"] == 1


def test_a_person_not_analysed_keeps_their_identity_without_a_new_vote_and_their_face_goes_to_nobody() -> None:
    world = World()
    for _ in range(2):
        world.step([(2, B)], [OWN_B])
    assert world.identity(2) is IdentityState.KNOWN
    record = world.step([(1, A), (2, B)], [OWN_B], not_analysed=(2,))
    by_track = {p["track_id"]: p for p in record["persons"]}
    assert (by_track[2]["ownership"], by_track[2]["vote"]) == ("not_analysed", "none")
    assert (by_track[1]["ownership"], by_track[1]["vote"]) == ("no_face", "none")  # B's face is never A's
    assert world.core.face_counters["faces_not_analysed"] == 1
    assert world.identity(2) is IdentityState.KNOWN and world.identity(1) is IdentityState.UNRESOLVED


def test_withholding_nobody_changes_nothing() -> None:
    frame, _ = stamped()
    persons = [person(frame, 1, A), person(frame, 2, B)]
    faces = [FaceObservation(frame=frame.key, box=OWN_B, quality=0.95)]
    decided = associate(persons, faces)
    assert withhold(decided, ()) is decided


# ---------------------------------------------------------------- real pixels: the cut, transcribed (device only)


def test_the_cut_from_a_window_is_the_cut_from_the_whole_picture_pixel_for_pixel() -> None:
    """DeepFace 0.0.99's ``extract_sub_image`` and ``align_img_wrt_eyes`` transcribed from the installed source (numpy and
    OpenCV only), on random pixels: D88's padded whole picture against the window path's padded window. Skipped where
    numpy or cv2 is missing (the repository's .venv); run on the device as test_tracking_diagnostic_installed.py says."""
    numpy = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from sentinel.media.opencv_source import area_resize

    def sub_image(img, x, y, w, h):  # detection.extract_sub_image
        rx, ry = int(0.5 * w), int(0.5 * h)
        x1, y1, x2, y2 = x - rx, y - ry, x + w + rx, y + h + ry
        if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
            return img[y1:y2, x1:x2], rx, ry
        out = numpy.zeros((h + 2 * ry, w + 2 * rx, img.shape[2]), dtype=img.dtype)
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        region = img[cy1:cy2, cx1:cx2]
        sx, sy = max(0, rx - x), max(0, ry - y)
        out[sy:sy + region.shape[0], sx:sx + region.shape[1]] = region
        return out, rx, ry

    def cut(padded, region):  # extract_face's aligned cut, before project_facial_area's crop
        img, _, _ = sub_image(padded, region["x"], region["y"], region["w"], region["h"])
        (lx, ly), (rx, ry) = region["left_eye"], region["right_eye"]
        angle = float(numpy.degrees(numpy.arctan2(ly - ry, lx - rx)))
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    rng = numpy.random.default_rng(1161505)
    picture = rng.integers(0, 256, size=(H, W, 3), dtype=numpy.uint8)
    padded_picture = cv2.copyMakeBorder(picture, H // 2, H // 2, W // 2, W // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    window = Window(1100, 300, 640, 640, 1.0, (0,))
    crop = picture[window.y:window.y + 640, window.x:window.x + 640].copy()
    padded_window = cv2.copyMakeBorder(crop, 320, 320, 320, 320, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    checked = 0
    for x, y, w, h in ((300.4, 200.7, 41.2, 52.9), (60.2, 70.9, 30.0, 36.6), (500.9, 480.3, 55.5, 60.1)):
        local = [x, y, w, h, x + 0.3 * w, y + 0.45 * h, x + 0.7 * w, y + 0.38 * h, 0, 0, 0, 0, 0, 0, 0.95]
        whole = [v + window.x if i in (0, 4, 6) else v + window.y if i in (1, 5, 7) else v for i, v in enumerate(local)]
        (in_window,), (in_picture,) = candidate_rows([local]), candidate_rows([whole])
        assert cut_inside(in_window, 640, 640, (True, True, True, True))
        a = cut(padded_window, candidate_region(in_window, 320, 320))
        b = cut(padded_picture, candidate_region(in_picture, W // 2, H // 2))
        assert a.shape == b.shape and numpy.array_equal(a, b)
        checked += 1
    assert checked == 3
    working = area_resize(picture, *WORKING_SIZE)
    assert working.shape == (480, 640, 3) and working is not picture and not numpy.shares_memory(working, picture)


def test_the_probes_refuse_main_single_before_opening_anything(tmp_path, capsys) -> None:
    """Review finding (session 116, part 16): the capture and track probes have no working copy, so under main_single the
    tracker would see the decoded picture itself. They refuse it before the source or the detector is touched."""
    import json

    from sentinel.cli import main

    config = tmp_path / "main.yaml"
    config.write_text(json.dumps({"config_version": 1, "camera": {"id": "cam-1"},
                                  "capture": {"stream_profile": "main_single"}}), encoding="utf-8")
    touched: list = []
    assert main(["capture", "probe", str(config), "--seconds", "1"], capture_source=touched.append) == 1
    assert "capture probe: stream_profile_not_supported" in capsys.readouterr().err
    assert main(["track", "probe", str(config), "--engine", str(tmp_path / "x.engine")], capture_source=touched.append,
                tracker_backend=touched.append, meminfo=lambda: {"MemFree": 9_000_000_000,
                                                                 "MemAvailable": 9_000_000_000}) == 1
    assert "track probe: stream_profile_not_supported" in capsys.readouterr().err
    assert touched == []
