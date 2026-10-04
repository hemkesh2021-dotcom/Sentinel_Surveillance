"""llama-server scene requests for the Oct 20 demo (D24; V2-26 demo form, full acceptance pending).

The demo scene model is v1's LFM2-VL-1.6B (Q4_0 with the Q8_0 vision projector)
served by llama.cpp's llama-server (b8932, provisionally retained under D36).
This module fixes how that server is launched and how one scene job is asked
of it. It is not V2-26 proper, which compares models and records the choice.

- **Loopback only (D42).** The server is always started with
  ``--host 127.0.0.1``, which on the command line overrides any
  ``LLAMA_ARG_HOST``. Every ``LLAMA_ARG_*`` variable is removed from its
  environment, and the client connects only to 127.0.0.1 with ``http.client``,
  which never consults proxy settings. Frames never leave the device.
- **Fixed model flags.** These are the flags check 8 profiled for the
  provisional demo profile (D33), plus ``--cache-ram 0`` (D41). S1 showed that
  llama-server's default host-RAM prompt cache grew by about 5.4 MiB per
  request. Configuration chooses only the port and the request timeout.
- **The same request S1 and check 8 measured.** A 480×360 JPEG at quality 60,
  the v2 SceneReport prompt, schema-constrained output (U20), at most 200
  tokens, non-streaming.
- **Fail closed.** The response is read up to a bound. The envelope must be one
  complete ``stop`` choice (U20's ``scene_completion_content``). Only then does
  the report text go to the scene lane, which parses it strictly. Problems are
  fixed labels and exception class names; model text, image bytes and server
  messages are never copied into details.

Device adapter: OpenCV is imported only by the default JPEG encoder.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..jobs import AnalysisJob, OutcomeKind, WorkerOutcome
from .completion import SceneCompletionError, scene_completion_content, scene_response_format

LOOPBACK_HOST = "127.0.0.1"
COMPLETIONS_PATH = "/v1/chat/completions"
# check 8's provisional demo profile (start_sentinel.sh's flags), plus D41.
PROFILED_FLAGS = ("--n-gpu-layers", "999", "--ctx-size", "2048", "--parallel", "1")
PROMPT_CACHE_FLAGS = ("--cache-ram", "0")
L4T_LIBCUDA = "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"  # D27 preload
IMAGE_SIZE = (480, 360)  # v1's scene image; width, height
JPEG_QUALITY = 60
MAX_TOKENS = 200
TEMPERATURE = 0.05
MODEL_NAME = "lfm2-vl"
MAX_RESPONSE_BYTES = 1_000_000
SCENE_SYSTEM = (
    "You are the scene-analysis component of a home security camera. "
    "Reply with exactly one JSON object and nothing else."
)
SCENE_PROMPT = (
    "Describe this camera image as JSON with exactly these fields: "
    '"persons_visible" (integer 0-50), "fire_or_smoke" (true or false), '
    '"threat" ("none", "low", "medium" or "high"), '
    '"observations" (up to 3 short phrases, aim under 48 characters each), '
    '"uncertainty" ("low", "medium" or "high"), '
    '"summary" (one short complete sentence, aim under 100 characters). '
    "Finish descriptions well before the schema limits; do not fill the available space."
)


def request_fingerprint(
    *, system: str, prompt: str, image_size: tuple[int, int], jpeg_quality: int, max_tokens: int,
    temperature: float, model: str,
) -> str:
    """SHA-256 of everything that shapes a scene request except the image (D47).

    The step-4 workload records it for what it sent; an accepted profile keeps it, and
    ``sentinel run --scene`` refuses a profile whose fingerprint differs from SCENE_REQUEST_SHA256.
    """
    canonical = {
        "system": system, "prompt": prompt, "image_size": list(image_size), "jpeg_quality": jpeg_quality,
        "max_tokens": max_tokens, "temperature": temperature, "model": model, "stream": False,
        "response_format": scene_response_format(),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def llama_server_command(binary: Path, model: Path, mmproj: Path, port: int) -> list[str]:
    """The demo scene server's argv: loopback only, profiled flags, prompt cache off."""
    if not 1024 <= port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    return [
        str(binary), "--model", str(model), "--mmproj", str(mmproj),
        "--host", LOOPBACK_HOST, "--port", str(port), *PROFILED_FLAGS, *PROMPT_CACHE_FLAGS,
    ]


def llama_server_environment(base: Mapping[str, str]) -> dict[str, str]:
    """The server's environment: no ``LLAMA_ARG_*`` overrides, L4T libcuda (D27), unified memory as v1."""
    env = {key: value for key, value in base.items() if not key.startswith("LLAMA_ARG_")}
    env["LD_PRELOAD"] = L4T_LIBCUDA
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"  # v1's launcher and check 8 (session 4)
    return env


def scene_request_body(jpeg: bytes) -> bytes:
    image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    body = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
        "response_format": scene_response_format(),
        "messages": [
            {"role": "system", "content": SCENE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": SCENE_PROMPT},
                ],
            },
        ],
    }
    return json.dumps(body).encode("utf-8")


SCENE_REQUEST_SHA256 = request_fingerprint(
    system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY,
    max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model=MODEL_NAME,
)


def opencv_jpeg(image: Any) -> bytes:
    """v1's scene image: resized to 480×360 and JPEG-encoded at quality 60 (needs OpenCV)."""
    import cv2

    ok, jpeg = cv2.imencode(".jpg", cv2.resize(image, IMAGE_SIZE), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return jpeg.tobytes()


class TransportError(Exception):
    """A request failure described by a fixed label."""

    def __init__(self, label: str, *, timed_out: bool = False) -> None:
        super().__init__(label)
        self.label = label
        self.timed_out = timed_out


class LoopbackTransport:
    """POSTs a completion request to llama-server on 127.0.0.1; returns the bounded response body."""

    def __init__(
        self,
        port: int,
        timeout_s: float,
        *,
        connection: Callable[..., Any] = http.client.HTTPConnection,
    ) -> None:
        self._port = port
        self._timeout_s = timeout_s
        self._connection = connection

    def post(self, body: bytes) -> bytes:
        conn = self._connection(LOOPBACK_HOST, self._port, timeout=self._timeout_s)
        try:
            conn.request("POST", COMPLETIONS_PATH, body=body, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
        except (TimeoutError, socket.timeout):
            raise TransportError("request_timed_out", timed_out=True) from None
        except (OSError, http.client.HTTPException) as exc:
            raise TransportError(f"request_failed:{type(exc).__name__}") from None
        finally:
            conn.close()
        if status != 200:
            raise TransportError(f"http_{status}" if 100 <= status <= 599 else "http_invalid_status")
        if len(data) > MAX_RESPONSE_BYTES:
            raise TransportError("response_too_large")
        return data


class LlamaSceneRequest:
    """Runs one scene job against llama-server: encode, request, check the envelope.

    Called on the analyzer's worker thread; it blocks for the request.
    """

    def __init__(self, transport: Any, *, encode: Callable[[Any], bytes] = opencv_jpeg) -> None:
        self._transport = transport
        self._encode = encode

    def __call__(self, job: AnalysisJob, image: Any) -> WorkerOutcome:
        try:
            jpeg = self._encode(image)
        except Exception as exc:  # a bad frame is that job's error, not a worker crash
            return WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, f"image encoding failed: {type(exc).__name__}")
        try:
            data = self._transport.post(scene_request_body(jpeg))
        except TransportError as exc:
            kind = OutcomeKind.TIMEOUT if exc.timed_out else OutcomeKind.ERROR
            return WorkerOutcome.failed(job.job_id, kind, exc.label)
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, "completion rejected: malformed_response")
        try:
            content = scene_completion_content(payload)
        except SceneCompletionError as exc:
            return WorkerOutcome.failed(job.job_id, OutcomeKind.ERROR, f"completion rejected: {exc.reason}")
        return WorkerOutcome.completed(job.job_id, content)  # the lane parses the report strictly
