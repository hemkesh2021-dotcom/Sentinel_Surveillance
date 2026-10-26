#!/usr/bin/env python3
"""A bounded, controlled check that the runtime's after-grace preemption ends a real in-flight chat request on this
device (session 115, part 11). Operator tooling, not part of the runtime, and separate evidence: it never changes an
integration run's verdict and never satisfies its "cancellation verified" item.

    chat_preempt_check.py run --model-set lfm2.5-vl --post-load-release --out DIR/chatpreempt.json
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
  - llama-server's /slots shows a slot processing, with at least one answer token decoded.

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

**Overall:** passed when the control is as expected, at least one trial passed, no trial failed and every other check
passed; not exercised when no trial was exercised and nothing failed; failed otherwise. Exit 0, 3, 1.

The result keeps metadata only:
- states, the runtime's fixed reason labels, timings, token counts, lengths and counters;
- never an answer's or a report's text, and never a prompt.

The picture is synthetic: labelled shapes on white, with nobody and no fire. No camera, no face and no private file is
read. Run it under the existing guard, from the worktree, with the runtime's Python (it has OpenCV):

    env PYTHONPATH=src ~/sentinel-surveillance/.venv/bin/python benchmarks/runner/step5_guard.py --out DIR \\
      --from-launch --duration-s 600 -- env LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \\
      PYTHONPATH=src ~/onvif_env/bin/python benchmarks/runner/chat_preempt_check.py run --model-set lfm2.5-vl \\
      --post-load-release --out DIR/chatpreempt.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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
from sentinel.scene.chat import CHAT_ADMISSION, CHAT_MAX_TOKENS, CHAT_MIN_INTERVAL_S, ChatQuestion, ChatService, \
    LlamaChatRequest  # noqa: E402
from sentinel.scene.llama_server import LOOPBACK_HOST, LlamaSceneRequest, LoopbackTransport, TransportError, \
    opencv_jpeg  # noqa: E402
from sentinel.scene.models import SCENE_MODEL_SETS, scene_model_set  # noqa: E402
from sentinel.scene.report import SceneReportError, parse_scene_report  # noqa: E402
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
SCOPE = ("separate controlled evidence of the runtime's after-grace preemption on this device's llama-server; it is "
         "not an integration run's 'cancellation verified' item, changes no verdict and admits nothing")
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
        return {"valid": False, "problem": "report_rejected"}
    dumped = report.model_dump(mode="json")
    return {"valid": True, "persons_visible": dumped.get("persons_visible"),
            "fire_or_smoke": dumped.get("fire_or_smoke"), "threat": dumped.get("threat"),
            "uncertainty": dumped.get("uncertainty"), "observations": len(dumped.get("observations") or []),
            "summary_chars": len(dumped.get("summary") or "")}


# ---------------------------------------------------------------- one trial


class Rig:
    """The runtime's scene worker and chat service on a live synthetic stream, against llama-server on ``port``."""

    def __init__(self, port: int, request_model: str, revision: str, picture: Any,
                 transport: Callable[[int, float], Any] = LoopbackTransport,
                 encode: Callable[[Any], bytes] = opencv_jpeg) -> None:
        self.port, self.picture = port, picture
        self.clock = SystemClock()
        self.stamper = FrameStamper("chatpreempt", self.clock)
        self.stamper.connect()
        self.frames, self.images = RecentFrames(), RecentImages()
        self.scene_started: dict[str, int] = {}
        scene_request = LlamaSceneRequest(transport(port, REQUEST_TIMEOUT_S), encode=encode, model=request_model)

        def run_job(job: AnalysisJob, image: Any) -> Any:
            self.scene_started[job.job_id] = time.monotonic_ns()
            return scene_request(job, image)

        self.analyzer = ThreadedSceneAnalyzer(run_job, self.images, revision=revision)  # no lane interval: see above
        self.chat_transport = InFlight(transport(port, REQUEST_TIMEOUT_S))
        watching = self.clock.mono()
        freshness = FreshnessConfig()
        self.chat = ChatService(enabled=True, admission=CHAT_ADMISSION, scene=self.analyzer,
                                latest=self.frames.latest, clock=self.clock,
                                run=LlamaChatRequest(self.chat_transport, encode=encode, model=request_model),
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

    def trial(self, name: str, question: str) -> dict[str, Any]:
        """One question and one periodic scene job, measured (see the module's docstring)."""
        _until(lambda: not self.analyzer.busy(), 10.0)
        before = dict(self.analyzer.counters)
        self.chat_transport.reset()
        code, accepted = self.chat.ask(ChatQuestion(question))
        item: dict[str, Any] = {"name": name, "question": "short" if question == QUESTIONS["short"] else "long",
                                "accepted": code}
        if code != 202:
            item.update(verdict="FAIL", reasons=[f"not_accepted:{accepted.get('state')}"])
            return item
        job = accepted["job"]
        deadline = time.monotonic() + IN_FLIGHT_WAIT_S
        progress, in_flight = None, False
        while time.monotonic() < deadline:
            begun, ended, _ = self.chat_transport.state()
            if ended is not None or self._final(job):
                break
            if begun is not None:
                progress = slot_progress(self.port)
                if progress and progress["processing"] >= 1 and (progress["n_decoded"] or 0) >= 1:
                    begun, ended, _ = self.chat_transport.state()
                    in_flight = begun is not None and ended is None
                    break
            time.sleep(POLL_S)
        frame = self.frames.latest()[0]
        self.images.put(frame.key, self.picture)  # the periodic job names a held frame, as the lane's do
        scene_id = f"{name}-scene"
        submitted_ns = time.monotonic_ns()
        begun, ended, _ = self.chat_transport.state()
        in_flight = in_flight and ended is None
        self.analyzer.submit(AnalysisJob(job_id=scene_id, frame=frame, purpose=JobPurpose.PERIODIC,
                                         deadline_mono_ns=frame.ingest_mono_ns + JOB_TIMEOUT_MS * 1_000_000))
        document = self._wait_final(job, 40.0)
        _until(lambda: (self.collect() or True) and scene_id in self.outcomes, JOB_TIMEOUT_MS / 1000 + 20.0)
        done_ns = time.monotonic_ns()
        begun, ended, outcome = self.chat_transport.state()
        after = dict(self.analyzer.counters)
        started_ns = self.scene_started.get(scene_id)
        scene_outcome = self.outcomes[scene_id]
        delays = self.analyzer.side_delays()
        answer = document.get("answer")
        item.update({
            "in_flight_at_submission": in_flight,
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
                      "outcome": scene_outcome.kind.value, **report_facts(scene_outcome.text)},
            "counters": {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(after) | set(before))
                         if after.get(key, 0) != before.get(key, 0)},
        })
        item["verdict"], item["reasons"] = (judge_control if name == "control" else judge_trial)(item)
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


def judge_trial(item: Mapping[str, Any]) -> tuple[str, list[str]]:
    """PASS only for a request in flight when the scene job came that the grace then ended for it; NOT EXERCISED when
    it ended by itself (before the job, or within the grace); FAIL otherwise. A scene problem always fails."""
    problems = scene_problems(item)
    end, counters = item["question_end"], item["counters"]
    if problems:
        return "FAIL", problems
    if not item["in_flight_at_submission"]:
        return "NOT EXERCISED", ["the request was not in flight when the scene job was submitted"]
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
    return ("FAIL", reasons) if reasons else ("PASS", ["ended by the scene job after the grace, which then ran"])


def judge_control(item: Mapping[str, Any]) -> tuple[str, list[str]]:
    """As expected: the short question answered (never cancelled) and the scene job then ran as monitoring requires."""
    problems = scene_problems(item)
    if item["question_end"]["state"] != "answered":
        problems.append(f"control_{item['question_end']['state']}")
    return ("NOT AS EXPECTED", problems) if problems else ("AS EXPECTED", [
        "answered within the grace" if item["in_flight_at_submission"] else "answered before the scene job came"])


def overall(result: Mapping[str, Any]) -> str:
    trials = [t["verdict"] for t in result.get("trials", [])]
    if result["problems"] or result.get("control", {}).get("verdict") != "AS EXPECTED" or "FAIL" in trials \
            or not trials:
        return "failed"
    return "passed" if "PASS" in trials else "not_exercised"


# ---------------------------------------------------------------- the run


def run(args: argparse.Namespace, *, server_factory: Callable[..., Any] = LlamaServerProcess,
        environ: Mapping[str, str] = os.environ, transport: Callable[[int, float], Any] = LoopbackTransport,
        servers: Callable[[], list[int]] = llama_servers, meminfo: Callable[[], Any] = read_meminfo,
        picture: Callable[[], Any] = synthetic_picture, encode: Callable[[Any], bytes] = opencv_jpeg) -> dict[str, Any]:
    """The check. The keyword arguments are the runtime's own pieces; tests and the package's dry run pass stand-ins
    (a fake llama-server, its process, the meminfo reader, a picture and an encoder without OpenCV)."""
    chosen = scene_model_set(args.model_set)
    model, mmproj = chosen.model_path, chosen.mmproj_path
    result: dict[str, Any] = {"check": "chat_preempt", "scope": SCOPE, "model_set": chosen.key,
                              "request_model": chosen.request_model, "model_revision": chosen.model_revision,
                              "files": {"model": model.name, "mmproj": mmproj.name}, "binary": str(args.llama_server),
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "limits": {"grace_ms": GRACE_MS, "max_chat_delay_ms": MAX_CHAT_DELAY_MS,
                                         "job_timeout_ms": JOB_TIMEOUT_MS, "chat_max_tokens": CHAT_MAX_TOKENS,
                                         "trials": args.trials},
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
    server = server_factory(args.llama_server, model, mmproj, args.port)
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
        rig = Rig(args.port, chosen.request_model, chosen.model_revision, image, transport, encode)
        rig.start()
        result["control"] = rig.trial("control", QUESTIONS["short"])
        result["memory"]["after_control"] = _memory()
        result["trials"] = []
        for index in range(1, args.trials + 1):
            time.sleep(CHAT_MIN_INTERVAL_S)  # chat takes one question every 4 s
            result["trials"].append(rig.trial(f"trial-{index}", QUESTIONS["long"]))
            result["memory"][f"after_trial_{index}"] = _memory()
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
    for trial in check.get("trials", []):
        lines.append(f"{trial['verdict']} {trial['name']} (long question): {'; '.join(trial['reasons'])} "
                     f"{json.dumps(_summary(trial))}")
    cleanup = {"problems": check.get("problems"), "health": check.get("health"), "stop": check.get("stop")}
    lines.append(f"{'PASS' if not check.get('problems') else 'FAIL'} server, health and cleanup: {json.dumps(cleanup)}")
    verdict = check.get("verdict") if supervision and memory else "failed"
    lines.append(f"chat preemption check: {verdict.replace('_', ' ')} ({SCOPE})")
    return lines, EXIT.get(verdict, 1)


def _summary(item: Mapping[str, Any]) -> dict[str, Any]:
    end, scene = item.get("question_end") or {}, item.get("scene") or {}
    return {"in_flight": item.get("in_flight_at_submission"), "slots": item.get("slots_at_submission"),
            "request": (item.get("request") or {}).get("outcome"), "state": end.get("state"),
            "reason": end.get("reason"), "tokens": end.get("tokens"),
            "scene_start_delay_ms": scene.get("start_delay_ms"),
            "scene_waited_ms": scene.get("waited_behind_question_ms"), "scene_report_valid": scene.get("valid"),
            "counters": item.get("counters")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--model-set", choices=sorted(SCENE_MODEL_SETS), required=True)
    go.add_argument("--llama-server", type=Path, default=DEFAULT_BINARY)
    go.add_argument("--port", type=int, default=PORT)
    go.add_argument("--trials", type=int, default=TRIALS, choices=range(1, TRIALS + 1))
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
