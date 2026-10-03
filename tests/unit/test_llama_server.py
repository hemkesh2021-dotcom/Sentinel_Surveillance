"""V2-26 demo form: llama-server launch, request and envelope handling with fakes (no server, no model)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

from sentinel.adapters import AdapterManifest, AdapterState, load, resolve
from sentinel.contracts import PixelFormat
from sentinel.jobs import AnalysisJob, JobPurpose, OutcomeKind
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.scene import llama_server as ls
from sentinel.scene.completion import scene_response_format
from sentinel.scene.llama_server import (
    LlamaSceneRequest,
    LoopbackTransport,
    TransportError,
    llama_server_command,
    llama_server_environment,
    scene_request_body,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = {
    "persons_visible": 1,
    "fire_or_smoke": False,
    "threat": "none",
    "observations": ["person near the door"],
    "uncertainty": "low",
    "summary": "One person stands near the door.",
}


def completion(content: object = None, *, finish: str = "stop", **extra: object) -> bytes:
    text = json.dumps(REPORT) if content is None else content
    choice = {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}
    return json.dumps({"choices": [choice], **extra}).encode()


def benchmark_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"v2_26_{name}", ROOT / "benchmarks/runner" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def job() -> AnalysisJob:
    clock = FakeClock()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    return AnalysisJob(job_id="scene-1", purpose=JobPurpose.PERIODIC, frame=frame,
                       deadline_mono_ns=frame.ingest_mono_ns + 8_000_000_000)


class FakeTransport:
    def __init__(self, reply: bytes | Exception) -> None:
        self.reply = reply
        self.bodies: list[bytes] = []

    def post(self, body: bytes) -> bytes:
        self.bodies.append(body)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# --- launch: loopback only, profiled flags, prompt cache off ------------------------------------

def test_the_server_command_binds_loopback_only_with_the_profiled_flags_and_no_prompt_cache() -> None:
    argv = llama_server_command(Path("/opt/llama-server"), Path("/m/model.gguf"), Path("/m/mmproj.gguf"), 18081)
    assert argv == [
        "/opt/llama-server", "--model", "/m/model.gguf", "--mmproj", "/m/mmproj.gguf",
        "--host", "127.0.0.1", "--port", "18081",
        "--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1", "--cache-ram", "0",
    ]
    assert argv.count("--host") == 1 and argv[argv.index("--host") + 1] == "127.0.0.1"
    for port in (80, 1023, 65536):
        with pytest.raises(ValueError):
            llama_server_command(Path("s"), Path("m"), Path("p"), port)


def test_the_model_flags_are_the_ones_check_8_profiled_plus_d41() -> None:
    profile = benchmark_module("demo_profile")
    assert list(ls.PROFILED_FLAGS) == profile.LLAMA_FLAGS
    assert list(ls.PROFILED_FLAGS) + ["--cache-ram", "0"] == profile.llama_flags(type("Args", (), {"llama_cache_ram": 0}))
    assert ls.L4T_LIBCUDA == profile.L4T_LIBCUDA


def test_the_server_environment_drops_llama_overrides_and_keeps_the_d27_preload() -> None:
    base = {"PATH": "/usr/bin", "LLAMA_ARG_HOST": "0.0.0.0", "LLAMA_ARG_CACHE_RAM": "8192", "LLAMA_ARG_PORT": "80"}
    env = llama_server_environment(base)
    assert env == {
        "PATH": "/usr/bin",
        "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1",
        "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
    }
    assert base["LLAMA_ARG_HOST"] == "0.0.0.0"  # the caller's mapping is not changed


def test_no_launch_path_in_this_repository_binds_llama_server_beyond_loopback() -> None:
    sources = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "benchmarks").rglob("*.py"))]
    hosts = 0
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "0.0.0.0" not in text, path
        for match in re.finditer(r'"--host",\s*([^,\]\s]+)', text):
            assert match.group(1) in ('"127.0.0.1"', "LOOPBACK_HOST"), (path, match.group(1))
            hosts += 1
    assert hosts >= 2  # this module and the benchmark profiler
    assert "0.0.0.0" not in (ROOT / "config/default.yaml").read_text(encoding="utf-8")
    launcher = (ROOT / "start_sentinel.sh").read_text(encoding="utf-8")  # v1, read only
    assert "--host 127.0.0.1" in launcher and "0.0.0.0" not in launcher


# --- the request S1 and check 8 measured ---------------------------------------------------------

def test_the_request_is_the_one_s1_and_check_8_measured() -> None:
    workload = benchmark_module("demo_workload")
    assert (ls.SCENE_SYSTEM, ls.SCENE_PROMPT) == (workload.SCENE_SYSTEM, workload.SCENE_PROMPT)
    assert (ls.IMAGE_SIZE, ls.JPEG_QUALITY, ls.MAX_TOKENS, ls.MAX_RESPONSE_BYTES) == (
        workload.VLM_IMAGE_SIZE, workload.VLM_JPEG_QUALITY, workload.VLM_MAX_TOKENS, workload.VLM_MAX_RESPONSE_BYTES,
    )
    body = json.loads(scene_request_body(b"\xff\xd8jpeg"))
    assert body == {
        "model": "lfm2-vl",
        "max_tokens": 200,
        "temperature": 0.05,
        "stream": False,
        "response_format": scene_response_format(),
        "messages": [
            {"role": "system", "content": workload.SCENE_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9hqcGVn"}},
                {"type": "text", "text": workload.SCENE_PROMPT},
            ]},
        ],
    }


# --- transport: 127.0.0.1 only, bounded, fixed labels -------------------------------------------

class FakeResponse:
    def __init__(self, status: int, data: bytes) -> None:
        self.status, self._data = status, data
        self.read_limit: int | None = None

    def read(self, limit: int) -> bytes:
        self.read_limit = limit
        return self._data[:limit]


class FakeConnection:
    made: list[FakeConnection] = []
    response: FakeResponse | Exception = FakeResponse(200, b"{}")
    fail_on_request: Exception | None = None

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.sent: tuple[str, str, bytes, dict] | None = None
        self.closed = False
        FakeConnection.made.append(self)

    def request(self, method: str, path: str, body: bytes, headers: dict) -> None:
        if FakeConnection.fail_on_request is not None:
            raise FakeConnection.fail_on_request
        self.sent = (method, path, body, headers)

    def getresponse(self) -> FakeResponse:
        if isinstance(FakeConnection.response, Exception):
            raise FakeConnection.response
        return FakeConnection.response

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def connection() -> type[FakeConnection]:
    FakeConnection.made = []
    FakeConnection.response = FakeResponse(200, b"{}")
    FakeConnection.fail_on_request = None
    return FakeConnection


def test_the_transport_posts_to_loopback_and_reads_a_bounded_body(connection: type[FakeConnection]) -> None:
    transport = LoopbackTransport(18081, 20.0, connection=connection)
    assert transport.post(b'{"x": 1}') == b"{}"
    (conn,) = connection.made
    assert (conn.host, conn.port, conn.timeout) == ("127.0.0.1", 18081, 20.0)
    assert conn.sent == ("POST", "/v1/chat/completions", b'{"x": 1}', {"Content-Type": "application/json"})
    assert connection.response.read_limit == 1_000_001 and conn.closed  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("setup", "label", "timed_out"),
    [
        (lambda c: setattr(c, "response", TimeoutError("timed out")), "request_timed_out", True),
        (lambda c: setattr(c, "fail_on_request", ConnectionRefusedError(111, "refused")),
         "request_failed:ConnectionRefusedError", False),
        (lambda c: setattr(c, "response", FakeResponse(503, b'{"error": "Loading model"}')), "http_503", False),
        (lambda c: setattr(c, "response", FakeResponse(200, b"x" * 1_000_001)), "response_too_large", False),
    ],
)
def test_transport_failures_are_labels(connection: type[FakeConnection], setup: Any, label: str, timed_out: bool) -> None:
    setup(connection)
    with pytest.raises(TransportError) as caught:
        LoopbackTransport(18081, 5.0, connection=connection).post(b"{}")
    assert (caught.value.label, caught.value.timed_out) == (label, timed_out)
    assert connection.made[0].closed


# --- one job: encode, request, check the envelope -------------------------------------------------

def test_a_complete_reply_passes_its_report_text_to_the_lane(job: AnalysisJob) -> None:
    transport = FakeTransport(completion())
    outcome = LlamaSceneRequest(transport, encode=lambda image: b"jpeg-of-" + image)(job, b"frame")
    assert outcome.kind is OutcomeKind.COMPLETED and outcome.job_id == "scene-1"
    assert json.loads(outcome.text) == REPORT  # type: ignore[arg-type]
    sent = json.loads(transport.bodies[0])
    assert sent["messages"][1]["content"][0]["image_url"]["url"].endswith("anBlZy1vZi1mcmFtZQ==")


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        (completion(finish="length"), "completion rejected: truncated"),
        (completion(finish=None), "completion rejected: incomplete_completion"),  # type: ignore[arg-type]
        (json.dumps({"error": {"message": "failed at /home/user/models/secret.gguf"}}).encode(),
         "completion rejected: server_error"),
        (b"<html>proxy page</html>", "completion rejected: malformed_response"),
        (b"\xff\xfe", "completion rejected: malformed_response"),
        (completion(usage={"prompt_tokens": 325, "completion_tokens": 96}), None),
        (completion("a" * 8_193), "output over 8192 chars"),
    ],
)
def test_unusable_replies_become_error_outcomes_without_model_or_server_text(
    job: AnalysisJob, reply: bytes, detail: str | None
) -> None:
    outcome = LlamaSceneRequest(FakeTransport(reply), encode=lambda image: b"j")(job, b"frame")
    if detail is None:  # extra top-level keys such as usage are not an envelope problem
        assert outcome.kind is OutcomeKind.COMPLETED
        return
    assert (outcome.kind, outcome.detail) == (OutcomeKind.ERROR, detail)
    assert "secret" not in outcome.model_dump_json() and "proxy" not in outcome.model_dump_json()


def test_a_timeout_and_a_failed_encoding_are_that_jobs_outcome(job: AnalysisJob) -> None:
    timed_out = LlamaSceneRequest(FakeTransport(TransportError("request_timed_out", timed_out=True)),
                                  encode=lambda image: b"j")(job, b"frame")
    assert (timed_out.kind, timed_out.detail) == (OutcomeKind.TIMEOUT, "request_timed_out")

    def broken(image: object) -> bytes:
        raise ValueError("cannot encode frame at /home/user/clips/private.mp4")

    transport = FakeTransport(completion())
    failed = LlamaSceneRequest(transport, encode=broken)(job, b"frame")
    assert (failed.kind, failed.detail) == (OutcomeKind.ERROR, "image encoding failed: ValueError")
    assert transport.bodies == []  # nothing was sent


def test_the_registry_admits_the_scene_adapter_only_with_the_provisional_profile() -> None:
    manifest = {
        "adapter_id": "llama-lfm2-vl-scene",
        "contract_version": 1,
        "implementation_revision": "1",
        "enabled": True,
        "input_kinds": ["frame"],
        "output_kinds": ["scene.report"],
        "model_revision": "lfm2-vl-1.6b-q4_0+mmproj-q8_0",
        "resource_profile_id": "provisional-demo-20261003T085010Z",
        "timeout_ms": 8000,
    }
    (status,) = resolve([AdapterManifest.model_validate(manifest)])
    implementation, loaded = load(status)  # imports the module only: no OpenCV
    assert implementation is LlamaSceneRequest and loaded.state is AdapterState.ENABLED
    (unmeasured,) = resolve([AdapterManifest.model_validate({**manifest, "resource_profile_id": "none-yet"})])
    assert unmeasured.state is AdapterState.UNAVAILABLE
