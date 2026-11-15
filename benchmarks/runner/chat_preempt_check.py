#!/usr/bin/env python3
"""A bounded, controlled check that the runtime's after-grace preemption ends a real in-flight chat request on this
device (session 115, part 11). Operator tooling, not part of the runtime, and separate evidence: it never changes an
integration run's verdict and never satisfies its "cancellation verified" item.

    chat_preempt_check.py run --model-set lfm3-vl --post-load-release --out DIR/chatpreempt.json
    chat_preempt_check.py read DIR        the reading of DIR/chatpreempt.json with the guard's DIR/result.json

It starts one llama-server through the runtime's own LlamaServerProcess, as scene_smoke.py does:
- the model set's files, the profiled flags with the prompt cache off;
- its readiness checks;
- after the runtime's MemFree precheck, and only if no other llama-server runs.

It refuses before anything starts if LLAMA_SERVER_SLOTS_DEBUG is set, because llama-server's /slots then carries text.
It then composes the runtime's own pieces, unchanged:

- **The scene worker:** ThreadedSceneAnalyzer with LlamaSceneRequest, the 8 s scene deadline and the runtime's grace
  (analyzer.SIDE_GRACE_S, 1.5 s). The lane's periodic interval is left out, as while scene analysis is paused
  (SideWindow None). So chat sizes no answer and each question may use the full 120-token cap, the case the grace's
  cut-off exists for. The scene jobs are submitted by this check.
- **Chat:** ChatService in its validation run (CHAT_ADMISSION, as --chat --chat-validation) with LlamaChatRequest. Its
  own freshness rule applies on a live synthetic stream: a FrameStamper on the system clock, the picture every 1/15 s.

The check is a short-answer control and at most TRIALS cancellation trials, each judged on its own:

- **The control:** alpha_load's short question. Once its request is in flight, one periodic scene job is submitted.
  Expected: it is answered, never cancelled, and the scene job then runs.
- **A trial:** alpha_load's long question, the same way. The scene job is submitted only when the request is genuinely
  in flight:
  - the chat transport's POST to llama-server has begun and not returned;
  - llama-server is decoding this request (DecodingWatch).

  **The decoding rule** (session 115, part 15) follows /slots as llama-server b8932 (VALIDATED_BUILD) fills it:
  - a slot takes a new task's id and its n_predict when the task is launched;
  - but it keeps the previous task's n_decoded and n_remain until the new task's prompt has been processed.

  So a reading counts only when it shows a slot processing a task whose id is not the one the slot held before the
  question. Decoding is observed only when two consecutive readings of that task show n_decoded rising, with n_decoded +
  n_remain equal to its n_predict and n_predict equal to the chat's cap (corroboration, never identity on its own).
  Another build, an unreadable or multi-slot /slots, or a second new task is ambiguous. A trial whose decoding was not
  observed is NOT EXERCISED.

  **A request that ends before that, or that ends by itself within the grace, is NOT EXERCISED, never passed.** Only an
  exercised trial can pass, and it passes only with all of these:
  - the question ended ``cancelled`` with the scene job's reason (analyzer.PREEMPTED);
  - the transport's request was aborted;
  - the worker counted one preemption after the grace;
  - the scene job started at most MAX_CHAT_DELAY_MS (D80's 2000 ms) after its submission, by this check's clock and by
    the worker's own measure;
  - the scene job completed within its deadline with a report the scene lane's strict parser accepts.

  A late scene job or an invalid report fails the trial, whatever happened to the question.
- **Also checked:** llama-server's /health after the trials and no slot left processing; the stop, with no llama-server
  left and the port closed. Memory is the guard's (step5_guard, its limits unchanged); ``read`` judges its result.

**Three things apart (session 116, part 11).** Besides those trials ("resume" below), for build b11938-19cce6919, LFM3-VL
and the 256-token image cap:
- **Waiting-question withdrawal.** A question asked while scene jobs keep the worker busy, given up after CHAT_START_S
  (chat's ``busy``, the worker's ``side_cancelled``). It must never have reached llama-server: no POST began, and no slot
  task with the chat's cap was seen. This cancels nothing on the GPU, because nothing started there.
- **In-flight cancellation.** A question ended through the runtime's own abort path with no scene job involved, once
  during its prompt and once during its decoding. The server-side evidence is what counts, since a client abort alone
  proves nothing about the GPU: llama-server's own ``cancel task`` line for this request's task and its ``stop
  processing`` line, and /slots leaving that task and never returning to it. Without those lines it is NOT EXERCISED.
- **Resume.** The trials above, with the same server-side evidence, the next task first seen, its decoding seen, and its
  completion's timings beside those of uninterrupted baseline scene jobs; then a second control.

**What timing can be measured** (TIMING_LIMITS, printed with every reading):
- the client's abort and the HTTP return, on this process's clock;
- when /slots last showed the task and first did not: each reading is known only between its sending and its return, and
  the server answers /slots between its decode steps, so the release lies between two instants, never at one;
- when the server's log lines reached this process: arrival times, after the log's own buffering;
- the kernel's GPU load (/sys/.../load), a coarse average that cannot tell one request's work from another's.
Nothing here observes CUDA kernels. From the pinned source, llama-server notices a closed connection only on its 1 s
poll (HTTP_POLLING_SECONDS) and only then posts the cancel, so generation may continue for up to about a second after
the client's abort, and a scene request meanwhile waits in its queue. The worker's scene start, which D80's 2000 ms is
measured on, is the moment the scene POST begins, which can be earlier than the server's start of that task. Both are
reported; D80's criterion is not changed here.

**Overall:** passed when both controls are as expected, every section asked for has at least one PASS and no FAIL, and
every other check passed; not exercised when nothing failed but a section has no PASS; failed otherwise. Exit 0, 3, 1.

The result keeps metadata only:
- states, the runtime's fixed reason labels, timings, token counts, lengths and counters;
- the server's build string (/props build_info, nothing else from /props);
- each scene response's finish reason and completion-token count;
- for a rejected report, the known field paths and pydantic's error categories, with an unexpected key masked;
- never an answer's or a report's text, a value, a parser message, a key the report should not have, or a prompt.

The picture is synthetic: labelled shapes on white, with nobody and no fire. No camera, no face and no private file is
read. Run it under the existing guard, from the worktree, with the runtime's Python (it has OpenCV):

    env PYTHONPATH=src ~/sentinel-surveillance/.venv/bin/python benchmarks/runner/step5_guard.py --out DIR \\
      --from-launch --duration-s 600 -- env LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \\
      PYTHONPATH=src ~/onvif_env/bin/python benchmarks/runner/chat_preempt_check.py run --model-set lfm3-vl \\
      --post-load-release --out DIR/chatpreempt.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from alpha_load import QUESTIONS  # noqa: E402  (the protocol's two questions)

from sentinel.config import FreshnessConfig  # noqa: E402
from sentinel.contracts import PixelFormat  # noqa: E402
from sentinel.demo_runtime import memfree_problem  # noqa: E402
from sentinel.jobs import AnalysisJob, JobPurpose, OutcomeKind  # noqa: E402
from sentinel.media.clock import SystemClock  # noqa: E402
from sentinel.media.frames import FrameStamper  # noqa: E402
from sentinel.media.health import assess_frame  # noqa: E402
from sentinel.media.recent import RecentFrames  # noqa: E402
from sentinel.memory_policy import release_component_files  # noqa: E402
from sentinel.scene.analyzer import PREEMPTED, SIDE_GRACE_S, RecentImages, ThreadedSceneAnalyzer  # noqa: E402
from sentinel.scene.chat import CHAT_ADMISSION, CHAT_MAX_TOKENS, CHAT_MIN_INTERVAL_S, CHAT_START_S, ChatQuestion, ChatService, \
    LlamaChatRequest  # noqa: E402
from sentinel.scene.llama_server import LOOPBACK_HOST, LlamaSceneRequest, LoopbackTransport, TransportError, \
    opencv_jpeg  # noqa: E402
from sentinel.scene.completion import scene_completion_finish_reason  # noqa: E402
from sentinel.scene.models import SCENE_MODEL_SETS, scene_model_set  # noqa: E402
from sentinel.scene.report import _FENCE as REPORT_FENCE  # noqa: E402  (the parser's own fence rule)
from sentinel.scene.report import MAX_REPORT_CHARS, SceneReport, SceneReportError, parse_scene_report  # noqa: E402
from sentinel.scene.server import LlamaServerProcess, health_ok, port_in_use  # noqa: E402
from sentinel.tracking.probe import read_meminfo  # noqa: E402

DEFAULT_BINARY = Path.home() / "llama.cpp/build/bin/llama-server"
PORT = 18081  # the configured scene port (config/default.yaml)
REQUEST_TIMEOUT_S = 20.0  # scene_server.request_timeout_s
JOB_TIMEOUT_MS = 8_000  # scene.job_timeout_s
MIN_FREE_BYTES = 3_000_000_000  # the runtime's precheck before the scene server
SETTLE_S = 15.0  # D58: the release follows the load and this settle
MAX_CHAT_DELAY_MS = 2000.0  # alpha_check.MAX_CHAT_DELAY_MS (D80): the longest a scene job may wait behind a question
GRACE_MS = round(SIDE_GRACE_S * 1000)  # D81, the runtime's
TRIALS = 3
IN_FLIGHT_WAIT_S = 10.0  # how long a request has to begin decoding before its trial is not exercised
POLL_S = 0.01
SLOTS_DEBUG = "LLAMA_SERVER_SLOTS_DEBUG"  # llama-server puts the prompt and the text generated into /slots when set
VALIDATED_BUILD = "b8932-98dc1418e"  # the llama-server build whose /slots the decoding rule was first checked against
# Session 116, part 11: LFM3-VL's own build, its tools/server/server-context.cpp read. A slot's reset() zeroes its stats
# and n_predict_max when a task ends, so an idle slot reads 0 decoded and -1 remaining rather than the last task's, and a
# new task reads 0 and its cap through the prompt. The rule below holds on both: it needs two rising readings of a new
# task. The source checks are in test_chat_preempt_check.py (on this device).
VALIDATED_LFM3_BUILD = "b11938-19cce6919"
VALIDATED_BUILDS = frozenset({VALIDATED_BUILD, VALIDATED_LFM3_BUILD})
BUILD_INFO = re.compile(r"b\d{1,6}-[0-9a-f]{7,40}")
ERROR_TYPE = re.compile(r"[a-z_]{1,40}")  # pydantic's error categories
REPORT_FIELDS = frozenset(SceneReport.model_fields)
MAX_DETAILS = 10
SCOPE = ("separate controlled evidence of withdrawal, in-flight cancellation and resume on this device's llama-server; "
         "it is not an integration run's 'cancellation verified' item, changes no verdict, marks no requirement "
         "satisfied and admits nothing; a client abort alone does not show that GPU work stopped")
TIMING_LIMITS = (
    "timing: the client's abort and the HTTP return are this process's own clock",
    "timing: a /slots reading is known only between its sending and its return, so a release lies between two instants",
    "timing: log times are arrival at this process, after the server's log buffering",
    "timing: GPU load is the kernel's coarse average and cannot tell one request from another; no CUDA kernel is observed",
    "timing: the server polls for a closed connection once a second (pinned source), so generation can outlast the abort",
    "timing: D80's scene start is when the worker's POST begins, which can precede the server's start of that task",
)
WITHDRAWALS = 2  # questions given up while they wait
CANCEL_PHASES = ("prompt", "decoding", "prompt", "decoding")  # questions ended in flight, no scene job involved
BASELINES = 3  # uninterrupted scene jobs, for their completion timings
WITHDRAWAL_JOBS = 5  # scene jobs that may be used to keep the worker busy past CHAT_START_S; a bound, not a target
ABORT_REASON = "the check ended this question"  # a fixed label: chat keeps the reason it is given
CANCEL_WAIT_S = 8.0  # how long a slot may take to leave a cancelled task before the trial fails: a bound, not a limit
TAIL_S = 2.0  # how long the readings continue after the slot left the task
TIMELINE_POLL_S = 0.025
LOAD_POLL_S = 0.05
TIMELINE_KEPT = 4000
LOAD_KEPT = 2000
LOG_EVENTS_KEPT = 200
GPU_LOAD = Path("/sys/devices/platform/bus@0/17000000.gpu/load")  # world-readable; the kernel's per-mille GPU load
# llama-server's own lines (tools/server/server-queue.cpp's stop() and server-context.cpp's release()); only these
# matches are kept, never a line
LOG_CANCEL = re.compile(r"cancel task, id_task = (\d+)")
LOG_RELEASE = re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)")
EXIT = {"passed": 0, "failed": 1, "not_exercised": 3}


def synthetic_picture() -> Any:
    """640x480 BGR: twelve labelled shapes in eight colours on white, so that a description runs long. Nobody, no
    fire."""
    import cv2
    import numpy as np

    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    colours = ((200, 0, 0), (0, 160, 0), (0, 0, 200), (0, 200, 200), (160, 0, 160), (90, 90, 90), (200, 200, 0),
               (0, 0, 0))
    words = ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE")
    for index, word in enumerate(words):
        column, row = index % 4, index // 4
        cx, cy, colour = 80 + column * 160, 90 + row * 150, colours[index % len(colours)]
        if index % 3 == 0:
            cv2.circle(image, (cx, cy), 40, colour, thickness=-1)
        elif index % 3 == 1:
            cv2.rectangle(image, (cx - 40, cy - 40), (cx + 40, cy + 40), colour, thickness=-1)
        else:
            points = np.array([[cx, cy - 45], [cx - 45, cy + 40], [cx + 45, cy + 40]], dtype=np.int32)
            cv2.fillPoly(image, [points], colour)
        cv2.putText(image, word, (cx - 45, cy + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
    return image


class InFlight:
    """The runtime's chat transport, noting when each POST to llama-server begins and ends and how it ended. Numbers and
    fixed labels only: the request and the response are never kept."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self.begun_ns: int | None = None
        self.ended_ns: int | None = None
        self.outcome: str | None = None

    def reset(self) -> None:
        with self._lock:
            self.begun_ns = self.ended_ns = self.outcome = None

    def post(self, body: bytes, **kwargs: Any) -> bytes:
        with self._lock:
            self.begun_ns, self.ended_ns, self.outcome = time.monotonic_ns(), None, None
        outcome = "returned"
        try:
            return self.inner.post(body, **kwargs)
        except TransportError as exc:
            outcome = "aborted" if exc.aborted else f"error:{exc.label}"
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then raised as before
            outcome = f"error:{type(exc).__name__}"
            raise
        finally:
            with self._lock:
                self.ended_ns, self.outcome = time.monotonic_ns(), outcome

    def state(self) -> tuple[int | None, int | None, str | None]:
        with self._lock:
            return self.begun_ns, self.ended_ns, self.outcome


def _get_json(port: int, path: str) -> tuple[int, Any]:
    conn = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=REQUEST_TIMEOUT_S)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        data = response.read(4_000_000)
        try:
            return response.status, json.loads(data)
        except ValueError:
            return response.status, None
    finally:
        conn.close()


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def slot_progress(port: int) -> dict[str, Any] | None:
    """From llama-server's /slots, numbers only: how many slots, how many processing, and the first processing slot's
    tokens decoded and remaining. Everything else in the response is dropped here."""
    try:
        code, data = _get_json(port, "/slots")
    except OSError:
        return None
    if code != 200 or not isinstance(data, list):
        return None
    busy = [slot for slot in data if isinstance(slot, Mapping) and slot.get("is_processing") is True]
    out: dict[str, Any] = {"slots": len(data), "processing": len(busy), "n_decoded": None, "n_remain": None}
    if busy:
        token = busy[0].get("next_token")
        token = token[0] if isinstance(token, list) and token else token
        if isinstance(token, Mapping):
            out["n_decoded"], out["n_remain"] = _count(token.get("n_decoded")), _count(token.get("n_remain"))
    return out


def slot_reading(port: int) -> dict[str, Any] | None:
    """From llama-server's /slots, numbers only, for its one slot: whether it is processing, the task it holds (its
    current task, or its last one once that ended: b8932's ``id_task``), that task's ``n_predict``, and the slot's
    tokens decoded and remaining. None when /slots cannot be read. Everything else in the response is dropped here."""
    try:
        code, data = _get_json(port, "/slots")
    except OSError:
        return None
    if code != 200 or not isinstance(data, list):
        return None
    out: dict[str, Any] = {"slots": len(data), "processing": None, "id_task": None, "n_predict": None,
                           "n_decoded": None, "n_remain": None}
    if len(data) != 1 or not isinstance(data[0], Mapping):
        return out
    slot, params = data[0], data[0].get("params")
    token = slot.get("next_token")
    token = token[0] if isinstance(token, list) and token else token
    out.update(processing=slot.get("is_processing") if isinstance(slot.get("is_processing"), bool) else None,
               id_task=_count(slot.get("id_task")),
               n_predict=_count(params.get("n_predict")) if isinstance(params, Mapping) else None)
    if isinstance(token, Mapping):
        out["n_decoded"], out["n_remain"] = _count(token.get("n_decoded")), _count(token.get("n_remain"))
    return out


def idle_slot(port: int, timeout_s: float = 5.0) -> dict[str, Any] | None:
    """The slot's reading once it is idle. llama-server sends a request's final response and then releases the slot
    (send_final_response, then slot.release()), so a reading right after the previous job may still show it
    processing. Still processing after ``timeout_s``, it is returned as it is (and the trial's attribution is
    ambiguous)."""
    deadline = time.monotonic() + timeout_s
    reading = slot_reading(port)
    while reading is not None and reading.get("processing") is True and time.monotonic() < deadline:
        time.sleep(POLL_S)
        reading = slot_reading(port)
    return reading


def server_build(port: int) -> str | None:
    """llama-server's build string from /props (``build_info``), when it has that form; nothing else is kept."""
    try:
        code, data = _get_json(port, "/props")
    except OSError:
        return None
    build = data.get("build_info") if code == 200 and isinstance(data, Mapping) else None
    return build if isinstance(build, str) and BUILD_INFO.fullmatch(build) else None


class DecodingWatch:
    """Whether llama-server is decoding the current chat request, from /slots readings alone (the module's decoding
    rule, checked against VALIDATED_BUILD's tools/server/server-context.cpp):

    - launch_slot_with_task gives the slot the new task (its id, its params) at once;
    - n_decoded is set to 0 only when that task's prompt has been processed, and n_remain is recomputed only once
      n_decoded > 0, so through the prompt both still hold the previous task's values;
    - once the task ends, /slots keeps showing it (task_prev).

    ``result`` is None while watching, "decoding_observed", or what made the attribution impossible."""

    def __init__(self, baseline: Mapping[str, Any] | None, build: str | None, max_tokens: int) -> None:
        self.baseline_task = None if baseline is None else baseline.get("id_task")
        self.build, self.max_tokens = build, max_tokens
        self.task: int | None = None
        self.n_predict: int | None = None
        self.readings = 0
        self.before: int | None = None
        self.last: Mapping[str, Any] | None = None
        self.result: str | None = None
        if build not in VALIDATED_BUILDS:
            self.result = "ambiguous:unvalidated_build"
        elif baseline is None or baseline.get("slots") != 1 or baseline.get("processing") is not False:
            self.result = "ambiguous:no_idle_slot_before_the_question"

    def feed(self, reading: Mapping[str, Any] | None) -> bool:
        """One reading; True once decoding of this request is observed."""
        if self.result is not None:
            return self.result == "decoding_observed"
        if reading is None or reading.get("slots") != 1 or reading.get("processing") is None:
            self.result = "ambiguous:slots_unreadable"
            return False
        if reading["processing"] is False:
            return False  # not launched yet (or ended: the trial sees that from the transport)
        task = reading.get("id_task")
        if task is None:
            self.result = "ambiguous:no_task_identity"
            return False
        if task == self.baseline_task:
            return False  # the slot still shows the task it held before the question
        if self.task is None:
            self.task, self.n_predict = task, reading.get("n_predict")
        elif task != self.task:
            self.result = "ambiguous:another_new_task"
            return False
        self.readings += 1
        decoded, remain = reading.get("n_decoded"), reading.get("n_remain")
        previous, self.last = (None if self.last is None else self.last.get("n_decoded")), reading
        if self.before is None:
            self.before = decoded
        rising = previous is not None and decoded is not None and decoded > previous and decoded >= 1
        if rising and remain is not None and self.n_predict == self.max_tokens == decoded + remain:
            self.result = "decoding_observed"
            return True
        return False

    def end(self, why: str) -> None:
        if self.result is None:
            self.result = why

    def record(self) -> dict[str, Any]:
        last = self.last or {}
        return {"result": self.result, "build": self.build, "baseline_task": self.baseline_task, "task": self.task,
                "n_predict": self.n_predict, "readings": self.readings, "n_decoded_first": self.before,
                "n_decoded": last.get("n_decoded"), "n_remain": last.get("n_remain")}


class SceneResponses:
    """The runtime's scene transport, keeping each response's finish reason (the runtime's own reading of it) and
    completion-token count. Never its text."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.last = inner, None

    def post(self, body: bytes, **kwargs: Any) -> bytes:
        self.last = None
        data = self.inner.post(body, **kwargs)
        self.last = completion_facts(data)
        return data


COMPLETION_TIMINGS = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_token_ms")


def completion_facts(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return {"finish_reason": "missing", "completion_tokens": None}
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    timings = payload.get("timings") if isinstance(payload, Mapping) else None
    return {"finish_reason": scene_completion_finish_reason(payload),
            "completion_tokens": _count(usage.get("completion_tokens")) if isinstance(usage, Mapping) else None,
            "timings": {key: round(timings[key], 2) for key in COMPLETION_TIMINGS
                        if isinstance(timings, Mapping) and isinstance(timings.get(key), (int, float))
                        and not isinstance(timings.get(key), bool)}}


def _memory() -> dict[str, int | None]:
    info = read_meminfo() or {}
    return {key: info.get(key) for key in ("MemTotal", "MemFree", "MemAvailable")}


def llama_servers() -> list[int]:
    found = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
    return [int(pid) for pid in found.stdout.split()]


def report_facts(text: str | None) -> dict[str, Any]:
    """Whether a scene job's text is a report the scene lane accepts, with its enumerations and lengths only."""
    if text is None:
        return {"valid": False, "problem": "no_text"}
    try:
        report = parse_scene_report(text)
    except SceneReportError:
        return {"valid": False, "problem": "report_rejected", "rejection": rejection_details(text)}
    dumped = report.model_dump(mode="json")
    return {"valid": True, "persons_visible": dumped.get("persons_visible"),
            "fire_or_smoke": dumped.get("fire_or_smoke"), "threat": dumped.get("threat"),
            "uncertainty": dumped.get("uncertainty"), "observations": len(dumped.get("observations") or []),
            "summary_chars": len(dumped.get("summary") or "")}


def rejection_details(text: str) -> list[dict[str, str]]:
    """Why parse_scene_report rejects ``text``, safely: the known field paths (an observation by its index) and
    pydantic's error categories only.
    - An unexpected key is masked.
    - No value, message, input or context is kept.

    It repeats the parser's own steps (its length bound, its fence) to reach the validation errors the parser hides."""
    if len(text) > MAX_REPORT_CHARS:
        return [{"loc": "<report>", "type": "report_too_long"}]
    body = text.strip()
    fenced = REPORT_FENCE.match(body)
    if fenced:
        body = fenced.group("body").strip()
    try:
        SceneReport.model_validate_json(body)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        return [{"loc": _safe_loc(error["loc"]),
                 "type": error["type"] if ERROR_TYPE.fullmatch(str(error["type"])) else "other"}
                for error in errors[:MAX_DETAILS]]
    return [{"loc": "<report>", "type": "rejected_otherwise"}]


def _safe_loc(loc: tuple[Any, ...]) -> str:
    """A validation error's path with only the report's own field names and indices: anything else is masked."""
    parts = []
    for index, part in enumerate(loc):
        if isinstance(part, int) and not isinstance(part, bool) and index > 0:
            parts.append(str(part))
        else:
            parts.append(part if index == 0 and part in REPORT_FIELDS else "<unexpected>")
    return ".".join(parts) or "<report>"


# ---------------------------------------------------------------- what the server shows


class ServerLogWatch:
    """Which tasks llama-server says it cancelled and released, and when each line reached this process. Only the two
    fixed patterns are read; every line is dropped at once. ``attached`` is False until a server's output runs through
    it: then the absence of a line means nothing, and a cancellation cannot be attributed."""

    def __init__(self) -> None:
        self.attached = False
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=LOG_EVENTS_KEPT)

    def feed(self, text: str) -> None:
        cancel, release = LOG_CANCEL.search(text), LOG_RELEASE.search(text)
        if cancel is None and release is None:
            return
        now = time.monotonic_ns()
        with self._lock:
            if cancel is not None:
                self._events.append({"kind": "cancel", "task": int(cancel[1]), "at_ns": now})
            if release is not None:
                self._events.append({"kind": "release", "task": int(release[1]), "n_tokens": int(release[2]), "at_ns": now})

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def popen(self, argv: Any, **kwargs: Any) -> Any:
        """subprocess.Popen, with the process's output read through this watch (LlamaServerProcess takes ``popen``)."""
        process = subprocess.Popen(argv, **kwargs)
        process.stdout = _TeedOutput(process.stdout, self.feed)
        self.attached = True
        return process


class _TeedOutput:
    def __init__(self, stream: Any, feed: Callable[[str], None]) -> None:
        self._stream, self._feed = stream, feed

    def readline(self, *args: Any) -> Any:
        raw = self._stream.readline(*args)
        self._feed(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw))
        return raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def read_gpu_load(path: Path = GPU_LOAD) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


class Observers:
    """For one trial, off its own thread: bounded /slots readings (each with when it was sent and returned) and GPU load
    samples. Numbers only."""

    def __init__(self, port: int, load: Callable[[], int | None] = read_gpu_load,
                 reader: Callable[[int], Mapping[str, Any] | None] | None = None) -> None:
        self.port, self._load = port, load
        self._reader = reader or slot_reading
        self.readings: deque[dict[str, Any]] = deque(maxlen=TIMELINE_KEPT)
        self.loads: deque[tuple[int, int | None]] = deque(maxlen=LOAD_KEPT)
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._read_slots, daemon=True, name="chatpreempt-slots"),
                         threading.Thread(target=self._read_load, daemon=True, name="chatpreempt-load")]

    def _read_slots(self) -> None:
        while not self._stop.is_set():
            sent = time.monotonic_ns()
            reading = self._reader(self.port)
            self.readings.append({"sent": sent, "received": time.monotonic_ns(), **(reading or {"slots": None})})
            self._stop.wait(TIMELINE_POLL_S)

    def _read_load(self) -> None:
        while not self._stop.is_set():
            self.loads.append((time.monotonic_ns(), self._load()))
            self._stop.wait(LOAD_POLL_S)

    def start(self) -> Observers:
        for thread in self._threads:
            thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(REQUEST_TIMEOUT_S + 1.0)

    def latest(self) -> Mapping[str, Any] | None:
        return self.readings[-1] if self.readings else None


def _ms(ns: int | None, origin: int) -> float | None:
    return None if ns is None else round((ns - origin) / 1e6, 1)


def server_timing(readings: Sequence[Mapping[str, Any]], loads: Sequence[tuple[int, int | None]],
                  events: Sequence[Mapping[str, Any]], *, task: int | None, abort_ns: int, attached: bool,
                  chat_cap: int = CHAT_MAX_TOKENS) -> dict[str, Any]:
    """What the server shows about one request's end, in milliseconds from the client's abort (TIMING_LIMITS says what
    each can and cannot mean). ``released_between_ms`` is the interval the slot's leaving this task lies in."""
    durations = sorted((r["received"] - r["sent"]) / 1e6 for r in readings)
    out: dict[str, Any] = {
        "task": task, "log_available": attached, "readings": len(readings),
        "reading_ms": None if not durations else {"p50": round(durations[len(durations) // 2], 1),
                                                  "max": round(durations[-1], 1)},
        "cancel_seen_ms": None, "release_seen_ms": None, "released_n_tokens": None,
    }
    for event in events:
        if event.get("task") != task:
            continue
        if event["kind"] == "cancel" and out["cancel_seen_ms"] is None:
            out["cancel_seen_ms"] = _ms(event["at_ns"], abort_ns)
        if event["kind"] == "release" and out["release_seen_ms"] is None:
            out["release_seen_ms"] = _ms(event["at_ns"], abort_ns)
            out["released_n_tokens"] = event.get("n_tokens")
    on_task = [r for r in readings if r.get("id_task") == task and r.get("processing") is True]
    last = on_task[-1] if on_task else None
    before = [r for r in on_task if r["received"] <= abort_ns]
    after = [] if last is None else [r for r in readings if r["sent"] >= last["received"]]
    left = next((r for r in after if r.get("slots") == 1
                 and (r.get("processing") is False or r.get("id_task") != task)), None)
    out.update({
        "n_predict": None if last is None else last.get("n_predict"),
        "n_decoded_at_abort": before[-1].get("n_decoded") if before else None,
        "n_decoded_last_seen": None if last is None else last.get("n_decoded"),
        "last_on_task_sent_ms": None if last is None else _ms(last["sent"], abort_ns),
        "first_off_task_received_ms": None if left is None else _ms(left["received"], abort_ns),
        "still_on_task_at_the_end": last is not None and left is None,
    })
    if last is not None and left is not None:
        out["released_between_ms"] = [max(0.0, _ms(last["sent"], abort_ns)), _ms(left["received"], abort_ns)]
    if isinstance(out["n_decoded_at_abort"], int) and isinstance(out["n_decoded_last_seen"], int):
        out["tokens_seen_after_abort"] = out["n_decoded_last_seen"] - out["n_decoded_at_abort"]
    next_task = None if left is None else next((r for r in after if r.get("processing") is True
                                                and r.get("id_task") not in (None, task)), None)
    if next_task is not None:
        out["next_task_first_seen_ms"] = _ms(next_task["received"], abort_ns)
        out["next_task_is_chat_cap"] = next_task.get("n_predict") == chat_cap
        same = [r for r in after if r.get("processing") is True and r.get("id_task") == next_task.get("id_task")]
        rising = next((b for a, b in zip(same, same[1:], strict=False)
                       if isinstance(a.get("n_decoded"), int) and isinstance(b.get("n_decoded"), int)
                       and b["n_decoded"] > a["n_decoded"]), None)
        out["next_task_decoding_seen_ms"] = None if rising is None else _ms(rising["received"], abort_ns)
    values = [(at, value) for at, value in loads if value is not None]
    if values:
        prior = [value for at, value in values if at <= abort_ns]
        later = [(at, value) for at, value in values if at > abort_ns]
        zero = next((at for at, value in later if value == 0), None)
        out["gpu_load"] = {
            "samples": len(values), "before_abort_p50": sorted(prior)[len(prior) // 2] if prior else None,
            "first_zero_after_abort_ms": _ms(zero, abort_ns),
            "max_after_that": None if zero is None else max(value for at, value in later if at >= zero)}
    else:
        out["gpu_load"] = None
    return out


# ---------------------------------------------------------------- one trial


class Rig:
    """The runtime's scene worker and chat service on a live synthetic stream, against llama-server on ``port``."""

    def __init__(self, port: int, request_model: str, revision: str, picture: Any,
                 transport: Callable[[int, float], Any] = LoopbackTransport,
                 encode: Callable[[Any], bytes] = opencv_jpeg, build: str | None = None,
                 chat_template_kwargs: Mapping[str, object] | None = None, log: ServerLogWatch | None = None,
                 load: Callable[[], int | None] = read_gpu_load) -> None:
        self.port, self.picture, self.build = port, picture, build
        self.log, self.load = log or ServerLogWatch(), load
        self.clock = SystemClock()
        self.stamper = FrameStamper("chatpreempt", self.clock)
        self.stamper.connect()
        self.frames, self.images = RecentFrames(), RecentImages()
        self.scene_started: dict[str, int] = {}
        self.scene_done: dict[str, int] = {}
        self.scene_completions: dict[str, Any] = {}
        responses = SceneResponses(transport(port, REQUEST_TIMEOUT_S))
        scene_request = LlamaSceneRequest(responses, encode=encode, model=request_model,
                                          chat_template_kwargs=chat_template_kwargs)

        def run_job(job: AnalysisJob, image: Any) -> Any:  # on the worker's one thread: one request at a time
            self.scene_started[job.job_id] = time.monotonic_ns()
            responses.last = None
            outcome = scene_request(job, image)
            self.scene_done[job.job_id] = time.monotonic_ns()  # its own end: a question may still be running after it
            self.scene_completions[job.job_id] = responses.last
            return outcome

        self.analyzer = ThreadedSceneAnalyzer(run_job, self.images, revision=revision)  # no lane interval: see above
        self.chat_transport = InFlight(transport(port, REQUEST_TIMEOUT_S))
        watching = self.clock.mono()
        freshness = FreshnessConfig()
        self.chat = ChatService(enabled=True, admission=CHAT_ADMISSION, scene=self.analyzer,
                                latest=self.frames.latest, clock=self.clock,
                                run=LlamaChatRequest(self.chat_transport, encode=encode, model=request_model,
                                                     chat_template_kwargs=chat_template_kwargs),
                                assess=lambda frame: assess_frame(frame, self.stamper.current_stream, self.clock.mono(),
                                                                  watching, freshness),
                                stale_after_s=freshness.stale_after_s, validation_run=True)
        self._stop = threading.Event()
        self._feeder = threading.Thread(target=self._feed, name="chatpreempt-frames", daemon=True)
        self.outcomes: dict[str, Any] = {}

    def _feed(self) -> None:
        while not self._stop.wait(1 / 15):
            frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
            self.frames.put(frame, self.picture)

    def start(self) -> None:
        self.analyzer.start()
        self._feeder.start()
        _until(lambda: self.frames.latest() is not None, 5.0)

    def stop(self) -> bool:
        self._stop.set()
        self._feeder.join(2.0)
        return self.analyzer.stop(10.0)

    def collect(self) -> None:
        for job_id, outcome in self.analyzer.drain():
            self.outcomes[job_id] = outcome

    def _scene_job(self, scene_id: str) -> int:
        """Submit one periodic scene job on the newest frame, as the lane does; the time of its submission."""
        frame = self.frames.latest()[0]
        self.images.put(frame.key, self.picture)  # the periodic job names a held frame, as the lane's do
        submitted = time.monotonic_ns()
        self.analyzer.submit(AnalysisJob(job_id=scene_id, frame=frame, purpose=JobPurpose.PERIODIC,
                                         deadline_mono_ns=frame.ingest_mono_ns + JOB_TIMEOUT_MS * 1_000_000))
        return submitted

    def _scene_facts(self, scene_id: str, submitted_ns: int) -> dict[str, Any]:
        """The finished scene job: when it started and completed after its submission, and its report's form."""
        _until(lambda: (self.collect() or True) and scene_id in self.outcomes, JOB_TIMEOUT_MS / 1000 + 20.0)
        started_ns, outcome = self.scene_started.get(scene_id), self.outcomes[scene_id]
        return {"start_delay_ms": None if started_ns is None else round((started_ns - submitted_ns) / 1e6, 1),
                "completed_after_submission_ms": round((self.scene_done.get(scene_id, time.monotonic_ns())
                                                        - submitted_ns) / 1e6, 1),
                "outcome": outcome.kind.value,
                **(self.scene_completions.get(scene_id) or {"finish_reason": None, "completion_tokens": None}),
                **report_facts(outcome.text)}

    def baseline(self, name: str) -> dict[str, Any]:
        """One scene job with no question anywhere near it: what an uninterrupted job's completion looks like."""
        _until(lambda: not self.analyzer.busy(), 10.0)
        observers = Observers(self.port, self.load).start()
        try:
            scene = self._scene_facts(name, self._scene_job(name))
        finally:
            observers.stop()
        problems = scene_ran_problems(scene)  # its time is what is being measured here, so it is never judged
        return {"name": name, "scene": scene, "verdict": "NOT AS EXPECTED" if problems else "AS EXPECTED",
                "reasons": problems}

    def withdrawal(self, name: str, question: str) -> dict[str, Any]:
        """One question asked while scene jobs keep the worker busy past CHAT_START_S: it waits, then is given up, and
        must never reach llama-server. Scene jobs go first by the runtime's own rule; this check only keeps one waiting."""
        _until(lambda: not self.analyzer.busy(), 10.0)
        before = dict(self.analyzer.counters)
        self.chat_transport.reset()
        observers = Observers(self.port, self.load).start()
        submitted = {f"{name}-scene-0": self._scene_job(f"{name}-scene-0")}
        item: dict[str, Any] = {"name": name, "kind": "withdrawal"}
        try:
            _until(lambda: next(iter(submitted)) in self.scene_started, 10.0)
            code, accepted = self.chat.ask(ChatQuestion(question))
            item["accepted"] = code
            if code != 202:
                item.update(verdict="FAIL", reasons=[f"not_accepted:{accepted.get('state')}"])
                return item
            job = accepted["job"]
            deadline = time.monotonic() + CHAT_START_S + JOB_TIMEOUT_MS / 1000 * 2
            document = self.chat.poll(job)[1]
            while document.get("state") == "waiting" and time.monotonic() < deadline:
                if list(submitted)[-1] in self.scene_started and len(submitted) < WITHDRAWAL_JOBS:
                    scene_id = f"{name}-scene-{len(submitted)}"
                    submitted[scene_id] = self._scene_job(scene_id)
                time.sleep(POLL_S)
                document = self.chat.poll(job)[1]
            if document.get("state") != "busy":
                document = self._wait_final(job, 40.0)
            scenes = [self._scene_facts(scene_id, at) for scene_id, at in submitted.items()]
        finally:
            observers.stop()
        after = dict(self.analyzer.counters)
        begun, _, outcome = self.chat_transport.state()
        readings = list(observers.readings)
        tasks = {r["id_task"]: r.get("n_predict") for r in readings
                 if r.get("processing") is True and r.get("id_task") is not None}
        item.update({
            "question_end": {"state": document.get("state"), "reason": document.get("reason"),
                             "wait_ms": document.get("wait_ms")},
            "request_began": begun is not None, "request_outcome": outcome,
            "tasks_seen": len(tasks), "chat_cap_task_seen": CHAT_MAX_TOKENS in tasks.values(),
            "scene_jobs": len(scenes), "scenes": scenes,
            "counters": {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(after) | set(before))
                         if after.get(key, 0) != before.get(key, 0)},
        })
        item["verdict"], item["reasons"] = judge_withdrawal(item)
        return item

    def cancellation(self, name: str, question: str, phase: str) -> dict[str, Any]:
        """One question ended in flight through the runtime's own abort path, in its prompt or its decoding, with no
        scene job involved: what the server shows after the client's abort."""
        _until(lambda: not self.analyzer.busy(), 10.0)
        before = dict(self.analyzer.counters)
        self.chat_transport.reset()
        watch = DecodingWatch(idle_slot(self.port), self.build, CHAT_MAX_TOKENS)
        observers = Observers(self.port, self.load).start()
        item: dict[str, Any] = {"name": name, "kind": "cancellation", "phase": phase}
        try:
            code, accepted = self.chat.ask(ChatQuestion(question))
            item["accepted"] = code
            if code != 202:
                item.update(verdict="FAIL", reasons=[f"not_accepted:{accepted.get('state')}"])
                return item
            job = accepted["job"]
            deadline = time.monotonic() + IN_FLIGHT_WAIT_S
            seen = 0
            while True:
                _, ended, _ = self.chat_transport.state()
                if ended is not None or self._final(job):
                    watch.end("ended_before_the_abort")
                    break
                if time.monotonic() > deadline:
                    watch.end("not_in_this_phase_in_time")
                    break
                if len(observers.readings) > seen:
                    seen = len(observers.readings)
                    reading = observers.latest()
                    if self.chat_transport.state()[0] is not None and reading.get("sent", 0) >= (
                            self.chat_transport.state()[0]):
                        watch.feed(reading)
                if watch.result is not None and watch.result != "decoding_observed":
                    break
                if phase == "prompt" and watch.task is not None:
                    break
                if phase == "decoding" and watch.result == "decoding_observed":
                    break
                time.sleep(POLL_S)
            abort_ns = time.monotonic_ns()
            aborted = self.chat.abort_running(ABORT_REASON)
            document = self._wait_final(job, 40.0)
            release_deadline = time.monotonic() + CANCEL_WAIT_S
            while time.monotonic() < release_deadline:
                current = observers.latest() or {}
                if current.get("sent", 0) > abort_ns and (
                        current.get("processing") is False or current.get("id_task") not in (None, watch.task)):
                    break
                time.sleep(POLL_S)
            time.sleep(TAIL_S)  # the readings continue: a task that came back would show here
        finally:
            observers.stop()
        after = dict(self.analyzer.counters)
        begun, ended, outcome = self.chat_transport.state()
        answer = document.get("answer")
        item.update({
            "decoding": watch.record(), "abort_returned": aborted,
            "request": {"outcome": outcome, "began_before_abort_ms": None if begun is None else _ms(abort_ns, begun),
                        "http_return_after_abort_ms": None if ended is None else _ms(ended, abort_ns)},
            "question_end": {"state": document.get("state"), "reason": document.get("reason"),
                             "answer_chars": len(answer) if isinstance(answer, str) else None},
            "server": server_timing(list(observers.readings), list(observers.loads), self.log.events(), task=watch.task,
                                    abort_ns=abort_ns, attached=self.log.attached),
            "counters": {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(after) | set(before))
                         if after.get(key, 0) != before.get(key, 0)},
        })
        item["verdict"], item["reasons"] = judge_cancellation(item)
        return item

    def trial(self, name: str, question: str) -> dict[str, Any]:
        """One question and one periodic scene job, measured (see the module's docstring)."""
        _until(lambda: not self.analyzer.busy(), 10.0)
        before = dict(self.analyzer.counters)
        self.chat_transport.reset()
        # the slot's task before the question, then the chat's cap (no lane interval: the full cap is planned)
        watch = DecodingWatch(idle_slot(self.port), self.build, CHAT_MAX_TOKENS)
        code, accepted = self.chat.ask(ChatQuestion(question))
        item: dict[str, Any] = {"name": name, "question": "short" if question == QUESTIONS["short"] else "long",
                                "accepted": code}
        if code != 202:
            item.update(verdict="FAIL", reasons=[f"not_accepted:{accepted.get('state')}"])
            return item
        job = accepted["job"]
        deadline = time.monotonic() + IN_FLIGHT_WAIT_S
        progress, in_flight = None, False
        observers = Observers(self.port, self.load).start()
        while watch.result is None:
            begun, ended, _ = self.chat_transport.state()
            if ended is not None or self._final(job):
                watch.end("ended_before_decoding_was_observed")
                break
            if time.monotonic() > deadline:
                watch.end("no_decoding_observed_in_time")
                break
            if begun is not None:
                progress = slot_reading(self.port)
                if watch.feed(progress):
                    begun, ended, _ = self.chat_transport.state()
                    in_flight = begun is not None and ended is None
                    break
            time.sleep(POLL_S)
        scene_id = f"{name}-scene"
        begun, ended, _ = self.chat_transport.state()
        in_flight = in_flight and ended is None
        submitted_ns = self._scene_job(scene_id)
        document = self._wait_final(job, 40.0)
        _until(lambda: (self.collect() or True) and scene_id in self.outcomes, JOB_TIMEOUT_MS / 1000 + 20.0)
        done_ns = self.scene_done.get(scene_id, time.monotonic_ns())
        observers.stop()
        begun, ended, outcome = self.chat_transport.state()
        after = dict(self.analyzer.counters)
        started_ns = self.scene_started.get(scene_id)
        scene_outcome = self.outcomes[scene_id]
        delays = self.analyzer.side_delays()
        answer = document.get("answer")
        item.update({
            "in_flight_at_submission": in_flight,
            "decoding": watch.record(),
            "slots_at_submission": progress,
            "submitted_after_request_began_ms": None if begun is None else round((submitted_ns - begun) / 1e6, 1),
            "request": {"outcome": outcome,
                        "ms": None if begun is None or ended is None else round((ended - begun) / 1e6, 1),
                        "ended_after_submission_ms": None if ended is None else round((ended - submitted_ns) / 1e6, 1)},
            "question_end": {"state": document.get("state"), "reason": document.get("reason"),
                             "answer_chars": len(answer) if isinstance(answer, str) else None,
                             "cut_short": document.get("cut_short"), "tokens": document.get("tokens"),
                             "max_tokens": document.get("max_tokens"), "wait_ms": document.get("wait_ms"),
                             "run_ms": document.get("run_ms")},
            "scene": {"start_delay_ms": None if started_ns is None else round((started_ns - submitted_ns) / 1e6, 1),
                      "waited_behind_question_ms": delays.get("last_ms") if after.get("jobs_waited_for_side", 0)
                      > before.get("jobs_waited_for_side", 0) else None,
                      "completed_after_submission_ms": round((done_ns - submitted_ns) / 1e6, 1),
                      "outcome": scene_outcome.kind.value,
                      **(self.scene_completions.get(scene_id) or {"finish_reason": None, "completion_tokens": None}),
                      **report_facts(scene_outcome.text)},
            "counters": {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(after) | set(before))
                         if after.get(key, 0) != before.get(key, 0)},
        })
        if ended is not None:  # from the abort's end of the request: the server's side, and the next task's start
            item["server"] = server_timing(list(observers.readings), list(observers.loads), self.log.events(),
                                           task=watch.task, abort_ns=ended, attached=self.log.attached)
            item["server"]["scene_submitted_before_request_end_ms"] = round((ended - submitted_ns) / 1e6, 1)
        item["verdict"], item["reasons"] = (judge_control if name.startswith("control") else judge_trial)(item)
        return item

    def _final(self, job: str) -> bool:
        return self.chat.poll(job)[1].get("state") not in ("sending", "waiting", "answering")

    def _wait_final(self, job: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            document = self.chat.poll(job)[1]
            if document.get("state") not in ("sending", "waiting", "answering") or time.monotonic() > deadline:
                return document
            time.sleep(POLL_S)


def _until(condition: Callable[[], bool], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("the check's own wait ran out")
        time.sleep(POLL_S)


# ---------------------------------------------------------------- the judging (predeclared)


def scene_problems(item: Mapping[str, Any]) -> list[str]:
    """Why the trial's scene job did not run as monitoring requires: late, not completed, or no valid report."""
    scene, problems = item["scene"], []
    if scene["outcome"] != OutcomeKind.COMPLETED.value:
        problems.append(f"scene_job_{scene['outcome']}")
    if not scene.get("valid"):
        problems.append(f"scene_report_{scene.get('problem', 'invalid')}")
    for key in ("start_delay_ms", "waited_behind_question_ms"):
        value = scene.get(key)
        if value is not None and value > MAX_CHAT_DELAY_MS:
            problems.append(f"scene_{key}_over_{MAX_CHAT_DELAY_MS:g}")
    if scene["start_delay_ms"] is None:
        problems.append("scene_job_never_started")
    if scene["completed_after_submission_ms"] > JOB_TIMEOUT_MS:
        problems.append("scene_job_past_its_deadline")
    return problems


def scene_ran_problems(scene: Mapping[str, Any]) -> list[str]:
    """Whether a scene job ran at all and gave a valid report, its times aside."""
    problems = []
    if scene["outcome"] != OutcomeKind.COMPLETED.value:
        problems.append(f"scene_job_{scene['outcome']}")
    if not scene.get("valid"):
        problems.append(f"scene_report_{scene.get('problem', 'invalid')}")
    if scene.get("start_delay_ms") is None:
        problems.append("scene_job_never_started")
    return problems


def judge_trial(item: Mapping[str, Any]) -> tuple[str, list[str]]:
    """PASS only for a request in flight when the scene job came that the grace then ended for it; NOT EXERCISED when
    it ended by itself (before the job, or within the grace); FAIL otherwise. A scene problem always fails."""
    problems = scene_problems(item)
    end, counters = item["question_end"], item["counters"]
    if problems:
        return "FAIL", problems
    if not item["in_flight_at_submission"]:
        why = (item.get("decoding") or {}).get("result")
        return "NOT EXERCISED", [f"the request was not in flight, decoding, when the scene job was submitted ({why})"]
    if end["state"] == "answered":
        if counters.get("side_finished_in_grace") == 1:
            return "NOT EXERCISED", ["the answer ended by itself within the grace"]
        return "FAIL", ["answered without ending within the grace"]
    if end["state"] != "cancelled":
        return "FAIL", [f"question_{end['state']}"]
    reasons = []
    if end["reason"] != PREEMPTED:
        reasons.append("cancelled_for_another_reason")
    if item["request"]["outcome"] != "aborted":
        reasons.append(f"request_{item['request']['outcome']}")
    if counters.get("side_preempted") != 1 or counters.get("side_preempted_after_grace") != 1:
        reasons.append("no_preemption_after_the_grace_counted")
    if reasons:
        return "FAIL", reasons
    server = item.get("server") or {}
    if server.get("log_available") and server.get("cancel_seen_ms") is None:
        return "NOT EXERCISED", ["the client aborted, but the server reported no cancellation of this task"]
    return "PASS", ["ended by the scene job after the grace, which then ran"]


def judge_withdrawal(item: Mapping[str, Any]) -> tuple[str, list[str]]:
    """PASS: the question waited, was given up as busy and never reached llama-server, and the scene jobs ran. NOT
    EXERCISED: a gap let it start. Anything else fails, as does any scene job that did not run as monitoring requires."""
    problems = [f"{index}:{problem}" for index, scene in enumerate(item["scenes"])
                for problem in scene_ran_problems(scene)]  # these jobs queue on purpose: their times are not judged
    if problems:
        return "FAIL", problems
    end = item["question_end"]
    if item["request_began"]:
        if end["state"] == "busy":
            return "FAIL", ["given up as busy after its request began"]
        return "NOT EXERCISED", [f"a gap let the question start ({end['state']})"]
    if end["state"] != "busy":
        return "FAIL", [f"question_{end['state']}"]
    if item["counters"].get("side_cancelled") != 1:
        return "FAIL", ["no_withdrawal_counted"]
    if item["chat_cap_task_seen"]:
        return "FAIL", ["a task with the chat's cap was seen on the server"]
    return "PASS", ["waited, given up as busy, withdrawn; no request began and no chat task was seen on the server"]


def judge_cancellation(item: Mapping[str, Any]) -> tuple[str, list[str]]:
    """PASS needs the server's own evidence: its cancel line for this task, its release line, and /slots leaving the task
    without coming back. A client abort alone is NOT EXERCISED, an answer that ended first too. Timing is reported,
    never judged: no limit on it exists yet, and none is invented here."""
    end, server = item["question_end"], item["server"]
    result = (item.get("decoding") or {}).get("result")
    if isinstance(result, str) and result.startswith("ambiguous"):
        return "NOT EXERCISED", [f"the task could not be attributed ({result})"]
    if not item["abort_returned"] or end["state"] == "answered":
        return "NOT EXERCISED", [f"the request had ended before the abort ({end['state']}, {result})"]
    reasons = []
    if end["state"] != "cancelled":
        reasons.append(f"question_{end['state']}")
    elif end["reason"] != ABORT_REASON:
        reasons.append("cancelled_for_another_reason")
    if item["request"]["outcome"] != "aborted":
        reasons.append(f"request_{item['request']['outcome']}")
    if reasons:
        return "FAIL", reasons
    if server["task"] is None:
        return "NOT EXERCISED", [f"the abort came before the task was seen on the slot ({result})"]
    if not server["log_available"]:
        return "NOT EXERCISED", ["the server's log was not available: a client abort alone proves nothing"]
    if server["cancel_seen_ms"] is None:
        return "NOT EXERCISED", ["the client aborted, but the server reported no cancellation of this task"]
    if server["release_seen_ms"] is None:
        return "FAIL", ["the server reported the cancellation but no release of the slot"]
    if server["still_on_task_at_the_end"]:
        return "FAIL", ["the slot was still on the task at the end of the readings"]
    return "PASS", ["the server reported cancelling and releasing this task, and /slots left it"]


def judge_control(item: Mapping[str, Any]) -> tuple[str, list[str]]:
    """As expected: the short question answered (never cancelled) and the scene job then ran as monitoring requires."""
    problems = scene_problems(item)
    if item["question_end"]["state"] != "answered":
        problems.append(f"control_{item['question_end']['state']}")
    return ("NOT AS EXPECTED", problems) if problems else ("AS EXPECTED", [
        "answered within the grace" if item["in_flight_at_submission"] else "answered before the scene job came"])


def section_verdict(items: Sequence[Mapping[str, Any]]) -> str:
    verdicts = [item["verdict"] for item in items]
    if "FAIL" in verdicts:
        return "failed"
    return "passed" if "PASS" in verdicts else "not_exercised"


def overall(result: Mapping[str, Any]) -> str:
    """Every section asked for has a PASS and no FAIL; both controls as expected; nothing else went wrong."""
    trials = result.get("trials", [])
    controls = [result.get("control", {})] + ([result["control_after"]] if "control_after" in result else [])
    if result["problems"] or not trials or any(c.get("verdict") != "AS EXPECTED" for c in controls) \
            or any(b.get("verdict") != "AS EXPECTED" for b in result.get("baselines", [])):
        return "failed"
    sections = [section_verdict(items) for items in (trials, result.get("withdrawal"), result.get("cancellation"))
                if items is not None]
    if "failed" in sections:
        return "failed"
    return "passed" if all(section == "passed" for section in sections) else "not_exercised"


# ---------------------------------------------------------------- the run


def run(args: argparse.Namespace, *, server_factory: Callable[..., Any] | None = None,
        environ: Mapping[str, str] = os.environ, transport: Callable[[int, float], Any] = LoopbackTransport,
        servers: Callable[[], list[int]] = llama_servers, meminfo: Callable[[], Any] = read_meminfo,
        picture: Callable[[], Any] = synthetic_picture, encode: Callable[[Any], bytes] = opencv_jpeg,
        load: Callable[[], int | None] = read_gpu_load) -> dict[str, Any]:
    """The check. The keyword arguments are the runtime's own pieces; tests and the package's dry run pass stand-ins
    (a fake llama-server, its process, the meminfo reader, a picture and an encoder without OpenCV)."""
    chosen = scene_model_set(args.model_set)
    model, mmproj = chosen.model_path, chosen.mmproj_path
    binary = args.llama_server or chosen.llama_server  # the set's own pinned build (session 116)
    result: dict[str, Any] = {"check": "chat_preempt", "scope": SCOPE, "model_set": chosen.key,
                              "request_model": chosen.request_model, "model_revision": chosen.model_revision,
                              "files": {"model": model.name, "mmproj": mmproj.name}, "binary": str(binary),
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "limits": {"grace_ms": GRACE_MS, "max_chat_delay_ms": MAX_CHAT_DELAY_MS,
                                         "job_timeout_ms": JOB_TIMEOUT_MS, "chat_max_tokens": CHAT_MAX_TOKENS,
                                         "trials": args.trials, "withdrawals": args.withdrawals,
                                         "cancellations": args.cancellations, "baselines": args.baselines},
                              "server_flags": list(chosen.server_flags),
                              "memory": {"before": _memory()}, "problems": []}
    problems: list[str] = result["problems"]
    if environ.get(SLOTS_DEBUG):
        problems.append("slots_debug_set")  # /slots would carry text: nothing is started
        return result
    if servers():
        problems.append("another_llama_server_running")
        return result
    if port_in_use(args.port):
        problems.append("port_in_use")
        return result
    problem, memory = memfree_problem(meminfo, MIN_FREE_BYTES)
    result["memory"]["precheck"] = memory
    if problem is not None:
        problems.append(f"memfree_precheck:{problem}")
        return result
    log = ServerLogWatch()
    if server_factory is None:  # the runtime's own process and readiness; only its output is also read here
        server = LlamaServerProcess(binary, model, mmproj, args.port, extra_flags=chosen.server_flags, popen=log.popen)
    else:
        server = server_factory(binary, model, mmproj, args.port, extra_flags=chosen.server_flags)
        if hasattr(server, "attach_log"):  # the fake's lines, the same two patterns
            server.attach_log(log)
    started = time.monotonic()
    rig = None
    try:  # from the start on: a guard stop (SIGINT) during the load still stops this check's own llama-server
        status = server.start(args.ready_timeout_s)
        result["readiness"] = {"state": status.state.value, "problem": status.problem, "layers": status.layers,
                               "vision_on_gpu": status.vision_on_gpu, "ready_s": round(time.monotonic() - started, 2)}
        if status.state.value != "ready":
            problems.append(f"not_ready:{status.problem}")
            return result
        result["memory"]["ready"] = _memory()
        if args.post_load_release:
            time.sleep(SETTLE_S)
            record = release_component_files("scene", {"llm": model, "mmproj": mmproj})
            result["release"] = {"problem": record["problem"],
                                 "results": {role: item.get("result") for role, item in record["files"].items()}}
            if not record["verified"]:
                problems.append(f"release_failed:{record['problem']}")
            result["memory"]["released"] = _memory()
        image = picture()
        jpeg = encode(image)
        result["picture"] = {"jpeg_bytes": len(jpeg), "sha256": hashlib.sha256(jpeg).hexdigest()}
        result["server_build"] = server_build(args.port)  # the decoding rule holds for VALIDATED_BUILD only
        rig = Rig(args.port, chosen.request_model, chosen.model_revision, image, transport, encode,
                  build=result["server_build"], chat_template_kwargs=chosen.chat_template_kwargs, log=log, load=load)
        rig.start()
        if args.baselines:
            result["baselines"] = [rig.baseline(f"baseline-{index}") for index in range(1, args.baselines + 1)]
        result["control"] = rig.trial("control", QUESTIONS["short"])
        result["memory"]["after_control"] = _memory()
        if args.withdrawals:
            result["withdrawal"] = []
            for index in range(1, args.withdrawals + 1):
                time.sleep(CHAT_MIN_INTERVAL_S)  # chat takes one question every 4 s
                result["withdrawal"].append(rig.withdrawal(f"withdrawal-{index}", QUESTIONS["long"]))
            result["memory"]["after_withdrawal"] = _memory()
        if args.cancellations:
            result["cancellation"] = []
            for index, phase in enumerate(CANCEL_PHASES[:args.cancellations], 1):
                time.sleep(CHAT_MIN_INTERVAL_S)
                result["cancellation"].append(rig.cancellation(f"cancel-{index}", QUESTIONS["long"], phase))
            result["memory"]["after_cancellation"] = _memory()
        result["trials"] = []
        for index in range(1, args.trials + 1):
            time.sleep(CHAT_MIN_INTERVAL_S)  # chat takes one question every 4 s
            result["trials"].append(rig.trial(f"trial-{index}", QUESTIONS["long"]))
            result["memory"][f"after_trial_{index}"] = _memory()
        if args.baselines:  # after everything: does a question and a scene job still run normally?
            time.sleep(CHAT_MIN_INTERVAL_S)
            result["control_after"] = rig.trial("control-after", QUESTIONS["short"])
            result["memory"]["after_control_after"] = _memory()
        progress = slot_progress(args.port)
        result["health"] = {"health_ok": health_ok(args.port), "slots": progress}
        if not result["health"]["health_ok"]:
            problems.append("server_not_healthy_after_the_trials")
        if progress is None or progress["processing"] != 0:
            problems.append("a_slot_still_processing_after_the_trials")
        return result
    except TimeoutError:
        problems.append("the_checks_own_wait_ran_out")
        return result
    finally:
        if rig is not None and not rig.stop():
            problems.append("scene_worker_not_stopped")
        stopped = server.stop(10.0)
        time.sleep(0.5)
        result["stop"] = {"stopped": stopped, "state": server.status().state.value,
                          "returncode": server.status().returncode, "llama_servers_left": len(servers()),
                          "port_open": port_in_use(args.port)}
        if not stopped or result["stop"]["llama_servers_left"] or result["stop"]["port_open"]:
            problems.append("unclean_stop")
        result["memory"]["after_stop"] = _memory()


# ---------------------------------------------------------------- the reading (with the guard's result)


def _trigger(trigger: Any) -> str | None:
    """The guard's stop condition in short, numbers only; None when nothing triggered."""
    if not trigger:
        return None
    if not isinstance(trigger, Mapping):
        return str(trigger)
    conditions = [f"{c.get('condition')} (value {c.get('value')})" for c in trigger.get("conditions") or []
                  if isinstance(c, Mapping)]
    return f"{trigger.get('stop')}: {'; '.join(conditions)}" if conditions else str(trigger.get("stop"))


def read(directory: Path) -> tuple[list[str], int]:
    """The check's lines, with the guard's memory and supervision; exit 0 passed, 3 not exercised, 1 otherwise.

    Each result is read on its own and a missing one is named. The guard's result is reported without the check's
    (a guard stop ends the check before it writes its result); nothing is judged without both."""
    lines: list[str] = []
    try:
        guard = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"MISSING the guard's result, result.json ({type(exc).__name__}): nothing is judged"], 1
    if str(guard.get("status")).startswith("refused:"):
        return [f"REFUSED the guard: {guard.get('status')}, before launch: nothing was started and nothing is "
                "judged"], 1
    g, stop, child = guard.get("guard") or {}, guard.get("stop") or {}, guard.get("child") or {}
    trigger = _trigger(g.get("trigger"))
    peak = g.get("peak_pressure_bytes")
    memory = isinstance(peak, int) and peak <= 5_400_000_000 and trigger is None
    held = "the guard's limits held (no stop)" if trigger is None else f"the guard stopped it on {trigger}"
    memory_line = f"{'PASS' if memory else 'FAIL'} memory: {held}; peak pressure {peak} B, at most 5,400,000,000 B"
    supervised = guard.get("status") == "child_exited" and stop.get("forced") is False and trigger is None
    try:
        check = json.loads((directory / "chatpreempt.json").read_text())
    except (OSError, ValueError) as exc:
        left = guard.get("leftovers") or {}
        stopped = (f"stop requested by {stop.get('requested_by')} {stop.get('requested_at_s')} s after launch"
                   if stop.get("requested_by") else "no stop requested")
        return [f"{'PASS' if supervised else 'FAIL'} the guard: status {guard.get('status')}, trigger {trigger}, "
                f"{stopped}, returncode {child.get('returncode')}, llama-server after the check's exit "
                f"{left.get('llama_server_after_exit')}, left {left.get('llama_server_left')}",
                memory_line,
                f"MISSING the check's result, chatpreempt.json ({type(exc).__name__}): not written; nothing else is "
                "judged"], 1
    supervision = supervised and child.get("returncode") == EXIT.get(check.get("verdict"))
    lines.append(f"{'PASS' if supervision else 'FAIL'} the guard: status {guard.get('status')}, trigger {trigger}, "
                 f"returncode {child.get('returncode')} for the check's {check.get('verdict')}")
    lines.append(memory_line)
    control = check.get("control") or {}
    lines.append(f"{control.get('verdict', 'MISSING')} control (short question): "
                 f"{'; '.join(control.get('reasons', []))} {json.dumps(_summary(control)) if control else ''}".rstrip())
    for baseline in check.get("baselines", []):
        lines.append(f"{baseline['verdict']} {baseline['name']} (uninterrupted scene job): "
                     f"{json.dumps(_scene_summary(baseline['scene']))}")
    if "withdrawal" in check:
        lines.append(f"--- waiting-question withdrawal: {section_verdict(check['withdrawal']).replace('_', ' ')}")
        for item in check["withdrawal"]:
            lines.append(f"{item['verdict']} {item['name']}: {'; '.join(item['reasons'])} {json.dumps(_withdrawal_summary(item))}")
    if "cancellation" in check:
        lines.append(f"--- in-flight cancellation: {section_verdict(check['cancellation']).replace('_', ' ')}")
        for item in check["cancellation"]:
            lines.append(f"{item['verdict']} {item['name']} (during its {item['phase']}): {'; '.join(item['reasons'])} "
                         f"{json.dumps(_cancel_summary(item))}")
    if "withdrawal" in check or "cancellation" in check:
        lines.append(f"--- resume after preemption: {section_verdict(check.get('trials', [])).replace('_', ' ')}")
    for trial in check.get("trials", []):
        lines.append(f"{trial['verdict']} {trial['name']} (long question): {'; '.join(trial['reasons'])} "
                     f"{json.dumps(_summary(trial))}")
    if "control_after" in check:
        control = check["control_after"]
        lines.append(f"{control.get('verdict', 'MISSING')} control after everything (short question): "
                     f"{'; '.join(control.get('reasons', []))} {json.dumps(_summary(control))}".rstrip())
    if "withdrawal" in check or "cancellation" in check:
        lines.extend(TIMING_LIMITS)
    cleanup = {"problems": check.get("problems"), "health": check.get("health"), "stop": check.get("stop")}
    lines.append(f"{'PASS' if not check.get('problems') else 'FAIL'} server, health and cleanup: {json.dumps(cleanup)}")
    verdict = check.get("verdict") if supervision and memory else "failed"
    lines.append(f"chat preemption check: {verdict.replace('_', ' ')} ({SCOPE})")
    return lines, EXIT.get(verdict, 1)


def _scene_summary(scene: Mapping[str, Any]) -> dict[str, Any]:
    return {key: scene.get(key) for key in ("outcome", "valid", "finish_reason", "completion_tokens", "start_delay_ms",
                                            "completed_after_submission_ms", "timings")}


def _withdrawal_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"state": (item.get("question_end") or {}).get("state"), "wait_ms": (item.get("question_end") or {}).get("wait_ms"),
            "request_began": item.get("request_began"), "chat_cap_task_seen": item.get("chat_cap_task_seen"),
            "scene_jobs": item.get("scene_jobs"), "scenes_valid": [s.get("valid") for s in item.get("scenes", [])],
            "counters": item.get("counters")}


def _cancel_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"state": (item.get("question_end") or {}).get("state"), "request": item.get("request"),
            "decoding": (item.get("decoding") or {}).get("result"), "server": item.get("server"),
            "counters": item.get("counters")}


def _summary(item: Mapping[str, Any]) -> dict[str, Any]:
    end, scene = item.get("question_end") or {}, item.get("scene") or {}
    decoding = item.get("decoding") or {}
    return {"in_flight": item.get("in_flight_at_submission"), "decoding": decoding.get("result"),
            "slot_task": [decoding.get("baseline_task"), decoding.get("task")],
            "slots": item.get("slots_at_submission"),
            "request": (item.get("request") or {}).get("outcome"), "state": end.get("state"),
            "reason": end.get("reason"), "tokens": end.get("tokens"),
            "scene_start_delay_ms": scene.get("start_delay_ms"),
            "scene_waited_ms": scene.get("waited_behind_question_ms"), "scene_report_valid": scene.get("valid"),
            "scene_finish": scene.get("finish_reason"), "scene_tokens": scene.get("completion_tokens"),
            "scene_rejection": scene.get("rejection"), "scene_timings": scene.get("timings"),
            "server": item.get("server"), "counters": item.get("counters")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--model-set", choices=sorted(SCENE_MODEL_SETS), required=True)
    go.add_argument("--llama-server", type=Path, default=None,
                        help="the llama-server binary (default: the selected set's own pinned build)")
    go.add_argument("--port", type=int, default=PORT)
    go.add_argument("--trials", type=int, default=TRIALS, choices=range(1, TRIALS + 1))
    go.add_argument("--withdrawals", type=int, default=WITHDRAWALS, choices=range(0, WITHDRAWALS + 1))
    go.add_argument("--cancellations", type=int, default=len(CANCEL_PHASES), choices=range(0, len(CANCEL_PHASES) + 1))
    go.add_argument("--baselines", type=int, default=BASELINES, choices=range(0, BASELINES + 1))
    go.add_argument("--ready-timeout-s", type=float, default=180.0)
    go.add_argument("--post-load-release", action="store_true")
    go.add_argument("--out", type=Path, required=True, help="a new JSON file for the result")
    reading = sub.add_parser("read")
    reading.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "read":
        lines, code = read(args.directory)
        print("\n".join(lines), flush=True)
        return code
    if args.out.exists():
        parser.error(f"{args.out} exists")
    result = run(args)
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["verdict"] = overall(result)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"chat_preempt_check": result["verdict"], "problems": result["problems"],
                      "control": (result.get("control") or {}).get("verdict"),
                      "trials": [t["verdict"] for t in result.get("trials", [])]}), flush=True)
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
