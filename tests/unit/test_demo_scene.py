from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.scene.report import SceneReport
from sentinel.scene.completion import SceneCompletionError

REPORT = {
    "persons_visible": 0, "fire_or_smoke": False, "threat": "none",
    "observations": [], "uncertainty": "high", "summary": "No person is visible.",
}


@pytest.fixture
def workload():
    path = Path(__file__).resolve().parents[2] / "benchmarks/runner/demo_workload.py"
    spec = importlib.util.spec_from_file_location("demo_scene_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 100},
    }


def test_scene_http_request_sends_the_schema_and_short_prompt_with_no_real_network(workload, monkeypatch) -> None:
    encoded = SimpleNamespace(tobytes=lambda: b"synthetic JPEG stand-in")
    image_calls = []
    cv2 = SimpleNamespace(
        resize=lambda frame, size: image_calls.append((frame, size)) or frame,
        imencode=lambda extension, frame, options: (True, encoded), IMWRITE_JPEG_QUALITY=1,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    requests = []
    answer = payload(json.dumps(REPORT))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            assert limit == 1_000_001
            return json.dumps(answer).encode()

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(workload.urllib.request, "urlopen", urlopen)
    frame = object()
    assert workload.scene_request(18081, frame) == answer
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:18081/v1/chat/completions"
    assert timeout == 30.0
    assert body["max_tokens"] == 200 and body["stream"] is False
    assert body["response_format"]["json_schema"]["schema"] == SceneReport.model_json_schema()
    assert "up to 3 short phrases" in body["messages"][1]["content"][1]["text"]
    assert "one short complete sentence" in body["messages"][1]["content"][1]["text"]
    assert "under 100 characters" in body["messages"][1]["content"][1]["text"]
    assert image_calls == [(frame, (480, 360))]


@pytest.mark.parametrize("oversized", [False, True])
def test_scene_http_body_is_rejected_not_silently_clipped(workload, monkeypatch, oversized: bool) -> None:
    encoded = SimpleNamespace(tobytes=lambda: b"synthetic JPEG stand-in")
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        resize=lambda frame, size: frame, imencode=lambda *args: (True, encoded), IMWRITE_JPEG_QUALITY=1,
    ))
    data = json.dumps(payload(json.dumps(REPORT))).encode()
    data = data + b" " * 1_000_001 if oversized else data[:-1]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            return data[:limit]

    monkeypatch.setattr(workload.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    error = SceneCompletionError if oversized else json.JSONDecodeError
    with pytest.raises(error):
        workload.scene_request(18081, object())


def test_scene_loop_rejects_incomplete_output_and_counts_boundaries_without_saving_text(workload, monkeypatch) -> None:
    boundary = {**REPORT, "summary": "s" * 160, "observations": ["o" * 80]}
    replies = [
        payload(json.dumps(REPORT)), payload(json.dumps(boundary)),
        payload(json.dumps(REPORT), "length"), payload('{"summary": "unfinished'),
        payload(json.dumps({**REPORT, "summary": "s" * 161})),
        payload(json.dumps(REPORT), "private finish reason"), {},
        {**payload(json.dumps(REPORT)), "error": {"message": "http://u:private@camera/"}},
    ]
    stop = threading.Event()
    iterator = iter(replies)
    calls = 0

    def request(port, frame):
        nonlocal calls
        calls += 1
        if calls == len(replies):
            stop.set()
        return next(iterator)

    monkeypatch.setattr(workload, "scene_request", request)
    latest = workload.Latest()
    latest.put(object())
    stats = workload.Stats()
    workload.scene_loop(18081, latest, stats, 0.0, stop)
    scene = stats.summary(1.0)["scene"]
    assert scene["completed"] == 8
    assert (scene["valid_reports"], scene["invalid_reports"], scene["unchecked_reports"]) == (2, 6, 0)
    assert scene["rejected_reports_by_reason"] == {
        "truncated": 1, "invalid_report": 2, "incomplete_completion": 1,
        "malformed_response": 1, "server_error": 1,
    }
    assert scene["finish_reasons"] == {"stop": 5, "length": 1, "other": 1, "missing": 1}
    assert (scene["valid_summaries_at_limit"], scene["valid_observations_at_limit"]) == (1, 1)
    assert scene["errors"] == {}
    assert "not evaluated" in scene["accuracy"]
    assert "private" not in json.dumps(scene)
    assert all(value is not retained for value in replies for retained in stats.__dict__.values())
    stats.reset()
    reset = stats.summary(1.0)["scene"]
    assert reset["rejected_reports_by_reason"] == reset["finish_reasons"] == {}
    assert reset["valid_summaries_at_limit"] == reset["valid_observations_at_limit"] == 0


def test_validation_unavailable_fails_closed_before_request(workload, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentinel.scene.completion", None)
    stop = threading.Event()
    latest = workload.Latest()
    latest.put(object())
    stats = workload.Stats()
    monkeypatch.setattr(workload, "scene_request", lambda *args: pytest.fail("unvalidated request started"))
    monkeypatch.setattr(stop, "wait", lambda *args: stop.set())
    workload.scene_loop(18081, latest, stats, 0.0, stop)
    scene = stats.summary(1.0)["scene"]
    assert scene["errors"] == {"ModuleNotFoundError": 1}
    assert scene["valid_reports"] == scene["completed"] == scene["unchecked_reports"] == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_profile_summary_displays_completion_counts_and_preserves_legacy_runs(tmp_path, legacy: bool) -> None:
    path = Path(__file__).resolve().parents[2] / "benchmarks/runner/demo_profile.py"
    spec = importlib.util.spec_from_file_location("demo_scene_profile", path)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    manifest = {
        "run_id": "synthetic", "input": {"synthetic": True, "fps": 15},
        "parameters": {"face_hz": 1, "scene_interval_s": 4},
        "repository": {"commit": "synthetic", "tracked_changes": False},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "memory.csv").write_text(
        "t_mono,phase,mem_total,mem_free,mem_available,swap_total,swap_free\n"
        "1,baseline,8000000000,5000000000,7000000000,0,0\n"
        "2,steady,8000000000,3000000000,4000000000,0,0\n"
    )
    (tmp_path / "tegrastats.log").write_text("")
    scene = {"completed": 2, "valid_reports": 1, "invalid_reports": 1}
    if not legacy:
        scene.update({
            "finish_reasons": {"stop": 1, "length": 1}, "rejected_reports_by_reason": {"truncated": 1},
            "valid_summaries_at_limit": 1, "valid_observations_at_limit": 0,
        })
    (tmp_path / "events.jsonl").write_text(json.dumps({
        "event": "workload_stats", "seconds": 1.0, "input": "synthetic", "scene": scene,
    }) + "\n")
    summary = profile.summarize(tmp_path)
    if legacy:
        assert "finish reasons unavailable; rejections unavailable" in summary
        assert "at character limits unavailable/unavailable" in summary
    else:
        assert "finish reasons {'stop': 1, 'length': 1}; rejections {'truncated': 1}" in summary
        assert "at character limits 1/0" in summary
    assert "do not establish scene accuracy or semantic completeness" in summary
    assert json.loads((tmp_path / "profile.json").read_text())["workload_steady"]["scene"] == scene


def test_replay_source_counts_decodes_across_loops_not_unique_clip_frames(workload, monkeypatch) -> None:
    frames = [object(), object(), object()]

    class Capture:
        def __init__(self, clip):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=Capture))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    source = workload.Frames("synthetic-permitted-clip", 15.0)
    delivered = [source.read() for _ in range(7)]
    assert source.decoded == 7 and source.loops == 2
    assert delivered == frames + frames + frames[:1]
    assert len(set(delivered)) == 3


def test_face_counts_positive_calls_not_unique_frames_faces_or_identities(workload) -> None:
    frame = object()
    latest = workload.Latest()
    latest.put(frame)
    stats = workload.Stats()
    stop = threading.Event()
    calls = []

    def represent(img_path, **kwargs):
        calls.append(img_path)
        if len(calls) == 3:
            stop.set()
            return [{"face_confidence": 0}]
        return [{"face_confidence": 0.9}, {"face_confidence": 0.8}]

    workload.face_loop(SimpleNamespace(represent=represent), latest, stats, 1_000_000.0, stop)
    face = stats.summary(1.0)["face"]
    assert calls == [frame, frame, frame]
    assert face["runs"] == 3 and face["runs_with_face"] == 2
