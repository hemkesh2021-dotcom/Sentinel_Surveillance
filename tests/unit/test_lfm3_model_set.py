"""Session 116: LFM3-VL-2B is the default scene and chat model set, LFM2.5-VL the explicit fallback, with D46 unchanged.

- The set pins the official QAD Q4_0 checkpoint and its own Q8_0 projector at a Hub revision, a separate pinned
  llama.cpp build, and the template's documented no-think switch, sent in every request as chat_template_kwargs.
- The switch is part of the request fingerprint; the sets that send none keep their fingerprints.
- The request classes and the step-4 workload send it; a thinking block is never an answer (chat fails closed).
- No profile measures the set: admission refuses it, and never substitutes another set.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.contracts import PixelFormat
from sentinel.jobs import AnalysisJob, JobPurpose
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.scene import models
from sentinel.scene.chat import CHAT_MAX_TOKENS, ChatQuestion, ChatRequestError, LlamaChatRequest, chat_answer, \
    chat_request_body
from sentinel.scene.llama_server import LlamaSceneRequest, request_fingerprint, scene_request_body

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
LFM3, LFM25, LFM2 = (models.SCENE_MODEL_SETS[k] for k in ("lfm3-vl", "lfm2.5-vl", "lfm2-vl"))
JPEG = b"\xff\xd8synthetic\xff\xd9"
NO_THINK = {"enable_thinking": False}
# the fingerprints of session 115, part 20, which the grammar fix gave the two older sets: unchanged by this set
PART_20 = {"lfm2-vl": "6b65cd33", "lfm2.5-vl": "43494f59"}


def test_the_default_set_is_lfm3_vl_2b_qad_q4_0_with_its_own_projector_build_and_switch() -> None:
    assert models.DEFAULT_SCENE_MODEL_SET == LFM3.key == "lfm3-vl"
    assert models.FALLBACK_SCENE_MODEL_SET == LFM25.key and models.ROLLBACK_SCENE_MODEL_SET == LFM2.key
    assert LFM3.profiled is False  # part 7: its capped launch is unmeasured (D106 measured it uncapped); LFM2.5-VL's
    # profile is never its evidence either
    assert LFM3.request_model == "lfm3-vl" and LFM3.model_revision == "lfm3-vl-2b-qad-q4_0"
    assert (LFM3.model.name, LFM3.model.bytes, LFM3.model.sha256) == (
        "LFM3-VL-2B-QAD-Q4_0.gguf", 1_049_615_648,
        "621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6")
    assert (LFM3.mmproj.name, LFM3.mmproj.bytes, LFM3.mmproj.sha256) == (
        "mmproj-LFM3-VL-2B-Q8_0.gguf", 203_031_872,
        "c5c5b58d544a18da919172d2d0368090ea1a4a351b6feb7f8a326a264a5918ab")
    for facts in (LFM3.model, LFM3.mmproj):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", facts.mtime_utc)
    assert re.fullmatch(r"[0-9a-f]{40}", models.LFM3_REVISION) and re.fullmatch(r"[0-9a-f]{40}", models.LFM3_LLAMA_CPP_COMMIT)
    assert LFM3.source == f"LiquidAI/LFM3-VL-2B-GGUF at {models.LFM3_REVISION}"
    assert LFM3.directory == Path.home() / "models/lfm3-vl"
    assert LFM3.llama_server == Path.home() / "llama.cpp-lfm3/build/bin/llama-server" != LFM25.llama_server
    assert LFM25.llama_server == LFM2.llama_server == Path.home() / "llama.cpp/build/bin/llama-server"
    assert LFM3.llama_build.startswith("master 19cce6919") and LFM25.llama_build.startswith("b8932")
    assert dict(LFM3.chat_template_kwargs) == NO_THINK and LFM25.chat_template_kwargs is None
    assert LFM2.chat_template_kwargs is None
    assert (LFM3.log_verbosity, LFM25.log_verbosity, LFM2.log_verbosity) == (4, None, None)
    assert (LFM3.image_max_tokens, LFM25.image_max_tokens, LFM2.image_max_tokens) == (256, None, None)  # part 7


def test_every_request_of_the_set_carries_the_no_think_switch_and_nothing_else_changes() -> None:
    scene = json.loads(scene_request_body(JPEG, LFM3.request_model, LFM3.chat_template_kwargs))
    assert scene["chat_template_kwargs"] == NO_THINK and scene["model"] == "lfm3-vl"
    chat = json.loads(chat_request_body(JPEG, ChatQuestion("What is on the table?"), 60, LFM3.request_model,
                                        LFM3.chat_template_kwargs))
    assert chat["chat_template_kwargs"] == NO_THINK and chat["model"] == "lfm3-vl" and chat["max_tokens"] == 60
    other = json.loads(scene_request_body(JPEG, LFM25.request_model, LFM25.chat_template_kwargs))
    assert "chat_template_kwargs" not in other
    assert {k: v for k, v in scene.items() if k not in ("model", "chat_template_kwargs")} == \
        {k: v for k, v in other.items() if k != "model"}  # the grammar, the prompts, the picture, the budgets
    other_chat = json.loads(chat_request_body(JPEG, ChatQuestion("What is on the table?"), 60, LFM25.request_model))
    assert "chat_template_kwargs" not in other_chat
    assert {k: v for k, v in chat.items() if k not in ("model", "chat_template_kwargs")} == \
        {k: v for k, v in other_chat.items() if k != "model"}


def test_the_switch_is_part_of_the_fingerprint_and_the_other_sets_keep_theirs() -> None:
    assert LFM3.scene_request_sha256 != request_fingerprint(
        system=models.SCENE_SYSTEM, prompt=models.SCENE_PROMPT, image_size=models.IMAGE_SIZE,
        jpeg_quality=models.JPEG_QUALITY, max_tokens=models.MAX_TOKENS, temperature=models.TEMPERATURE,
        model=LFM3.request_model)  # the same set without the switch would be another request
    for key, prefix in PART_20.items():
        assert models.scene_model_set(key).scene_request_sha256.startswith(prefix)
    assert len({s.scene_request_sha256 for s in models.SCENE_MODEL_SETS.values()}) == 3


class Transport:
    def __init__(self, reply: dict) -> None:
        self.reply, self.bodies = reply, []

    def post(self, body: bytes, **kwargs) -> bytes:
        self.bodies.append(json.loads(body))
        return json.dumps(self.reply).encode()


def test_the_request_classes_send_the_sets_switch() -> None:
    report = {"persons_visible": 0, "fire_or_smoke": False, "threat": "none", "observations": [], "uncertainty": "low",
              "summary": "Nothing."}
    reply = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(report)}}],
             "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0, "predicted_n": 3}}
    transport = Transport(reply)
    stamper = FrameStamper("t", FakeClock())
    stamper.connect()
    frame = stamper.stamp(native_width=4, native_height=4, pixel_format=PixelFormat.BGR)
    job = AnalysisJob(job_id="j", purpose=JobPurpose.PERIODIC, frame=frame, deadline_mono_ns=frame.ingest_mono_ns + 1)
    scene = LlamaSceneRequest(transport, encode=lambda image: JPEG, model=LFM3.request_model,
                              chat_template_kwargs=LFM3.chat_template_kwargs)
    assert scene(job, object()).text == json.dumps(report)
    chat = LlamaChatRequest(transport, encode=lambda image: JPEG, model=LFM3.request_model,
                            chat_template_kwargs=LFM3.chat_template_kwargs)
    chat(ChatQuestion("q"), object(), max_tokens=CHAT_MAX_TOKENS)
    assert [b["chat_template_kwargs"] for b in transport.bodies] == [NO_THINK, NO_THINK]
    assert [b["model"] for b in transport.bodies] == ["lfm3-vl", "lfm3-vl"]


@pytest.mark.parametrize("content", ["<think>\nthe shapes\n</think>\nRed and blue.", "Red<think>", "</think>Red."])
def test_a_thinking_block_is_never_an_answer(content) -> None:
    payload = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
    with pytest.raises(ChatRequestError, match="not usable"):
        chat_answer(payload)
    fine = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Red and blue.",
                                                               "reasoning_content": ""}}]}
    assert chat_answer(fine) == ("Red and blue.", False)


def test_the_step4_workload_sends_the_runtimes_request_for_the_set(monkeypatch) -> None:
    sys.path.insert(0, str(RUNNERS))
    try:
        w = importlib.import_module("demo_workload")
        op = importlib.import_module("operator_check")
    finally:
        sys.path.remove(str(RUNNERS))
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b'{"choices": []}'

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        resize=lambda frame, size: frame, IMWRITE_JPEG_QUALITY=1,
        imencode=lambda ext, image, params: (True, SimpleNamespace(tobytes=lambda: JPEG))))
    monkeypatch.setattr(w.urllib.request, "urlopen", lambda request, timeout: sent.append(request) or Response())
    monkeypatch.setattr(w, "SCENE_REQUEST_MODEL", LFM3.request_model)
    monkeypatch.setattr(w, "SCENE_TEMPLATE_KWARGS", json.loads(json.dumps(dict(LFM3.chat_template_kwargs))))
    w.scene_request(18081, object())
    assert json.loads(sent[0].data) == json.loads(scene_request_body(JPEG, LFM3.request_model,
                                                                      LFM3.chat_template_kwargs))
    assert w.scene_request_sha256() == LFM3.scene_request_sha256
    argv = op.scene_model_args("lfm3-vl")
    assert argv[:2] == ["--llama", str(LFM3.llama_server)]
    assert argv[-6:-4] == ["--scene-template-kwargs", '{"enable_thinking": false}']
    assert json.loads(argv[-5]) == NO_THINK
    assert argv[-4:-2] == ["--llama-log-verbosity", "4"]  # session 116, part 2: the build hides its markers at 3
    assert argv[-2:] == ["--llama-image-max-tokens", "256"]  # part 7: the set's image-token cap
    for flag in ("--scene-template-kwargs", "--llama-log-verbosity", "--llama-image-max-tokens"):
        assert flag not in op.scene_model_args("lfm2.5-vl")
    assert op.SCENE_MODEL_SET_KEYS == tuple(sorted(models.SCENE_MODEL_SETS))


def test_a_profile_of_the_set_must_have_measured_its_launch_flags(tmp_path) -> None:
    """Session 116, parts 2 and 7: the set's log-verbosity and image-cap flags are part of what a profile must have
    measured."""
    import dataclasses

    from sentinel.adapters import RESOURCE_PROFILES
    from sentinel.config import parse_config
    from sentinel.demo_runtime import SceneOptions, profile_mismatch
    from sentinel.memory_policy import CANDIDATE_POLICY
    from sentinel.scene.llama_server import PROFILED_FLAGS, PROMPT_CACHE_FLAGS

    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    accepted = RESOURCE_PROFILES["step4cand-lfm25-20261026T094728Z"]
    measured = dataclasses.replace(accepted, scene_request_sha256=LFM3.scene_request_sha256,
                                   llama_flags=PROFILED_FLAGS + PROMPT_CACHE_FLAGS + LFM3.server_flags)
    without = dataclasses.replace(measured, llama_flags=PROFILED_FLAGS + PROMPT_CACHE_FLAGS)
    options = SceneOptions(tmp_path / "no-server", tmp_path / "no-model", tmp_path / "no-mmproj", model_set=LFM3.key)
    assert "measured llama-server flags" in profile_mismatch(without, config, options, policy=CANDIDATE_POLICY)
    assert "measured llama-server flags" not in (profile_mismatch(measured, config, options, policy=CANDIDATE_POLICY)
                                                 or "")
    fallback = SceneOptions(tmp_path / "no-server", tmp_path / "no-model", tmp_path / "no-mmproj", model_set=LFM25.key)
    assert "measured llama-server flags" not in (profile_mismatch(accepted, config, fallback, policy=CANDIDATE_POLICY)
                                                 or "")  # the other sets' expected flags are unchanged
