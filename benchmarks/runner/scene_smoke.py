#!/usr/bin/env python3
"""A bounded smoke test of one scene model set with a synthetic picture (session 115). Operator tooling, not part of
the runtime; it proves that a set loads and answers, never how well.

It starts one llama-server through the runtime's own LlamaServerProcess (loopback only, the profiled flags with the
prompt cache off, L4T's libcuda, and its readiness checks: health, every layer and the vision encoder on the GPU,
libraries from the binary's own directory), after the runtime's MemFree precheck, and only if no other llama-server
runs. Then, with the runtime's own request classes and parsers:

- ``/props``: the server's image modality and the chat template it applies (compared with the model file's own);
- ``/apply-template``: how Sentinel's scene and chat messages are rendered (the ChatML shape, the picture's marker);
- REPEATS scene jobs about the synthetic picture (LlamaSceneRequest: the 480x360 JPEG at quality 60, the
  schema-constrained request, at most 200 tokens), each report parsed by the scene lane's strict parser;
- REPEATS chat questions about the same picture (LlamaChatRequest, at most 120 tokens), each answer checked as the
  runtime checks it, with llama-server's timings and the pace sample chat would take from it;
- the stop (SIGTERM, then SIGKILL after the grace), then no llama-server left and the port closed.

With ``--post-load-release`` the model files are released from the page cache 15 s after the server is ready, as
``sentinel run --post-load-release`` does (D58). The picture is synthetic (a red disc, a blue square and the word EXIT
on white; nobody in it), so its answers are kept in the result; no camera, no private file, no secret is read. The
picture checks only show that the image reached the model: a smoke test of image input, never a quality measure.

Run it under the existing guard, from the worktree (with the runtime's Python, which has OpenCV):

    env PYTHONPATH=src ~/sentinel-surveillance/.venv/bin/python benchmarks/runner/step5_guard.py --out DIR \\
      --from-launch --duration-s 600 -- env LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \\
      PYTHONPATH=src ~/onvif_env/bin/python benchmarks/runner/scene_smoke.py --model-set lfm2.5-vl \\
      --post-load-release --out DIR/smoke.json

Exit status 0 means every check passed; the result JSON says which did not.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinel.contracts import PixelFormat  # noqa: E402
from sentinel.demo_runtime import memfree_problem  # noqa: E402
from sentinel.jobs import AnalysisJob, JobPurpose  # noqa: E402
from sentinel.media.clock import SystemClock  # noqa: E402
from sentinel.media.frames import FrameStamper  # noqa: E402
from sentinel.memory_policy import release_component_files  # noqa: E402
from sentinel.scene.chat import CHAT_MAX_TOKENS, ChatQuestion, LlamaChatRequest, chat_request_body  # noqa: E402
from sentinel.scene.chat_budget import pace_sample  # noqa: E402
from sentinel.scene.llama_server import (  # noqa: E402
    LOOPBACK_HOST,
    MAX_TOKENS,
    LlamaSceneRequest,
    LoopbackTransport,
    opencv_jpeg,
    scene_request_body,
)
from sentinel.scene.models import SCENE_MODEL_SETS, scene_model_set  # noqa: E402
from sentinel.scene.report import SceneReportError, parse_scene_report  # noqa: E402
from sentinel.scene.server import LlamaServerProcess, port_in_use  # noqa: E402
from sentinel.tracking.probe import read_meminfo  # noqa: E402

DEFAULT_BINARY = Path.home() / "llama.cpp/build/bin/llama-server"
PORT = 18081  # the configured scene port (config/default.yaml)
REQUEST_TIMEOUT_S = 20.0  # scene_server.request_timeout_s
JOB_TIMEOUT_MS = 8_000  # scene.job_timeout_s: a scene job over it would be late in the runtime
MIN_FREE_BYTES = 3_000_000_000  # the runtime's precheck before the scene server
SETTLE_S = 15.0  # D58: the release follows the load and this settle
MAX_REPEATS = 5
QUESTION = "Which shapes and colours do you see in this picture, and is there any text?"
COLOURS = ("red", "blue")
SHAPES = (("circle", "disc", "disk", "round", "dot", "ball"), ("square", "rectangle", "box", "block"))
WORD = "exit"


def synthetic_picture() -> Any:
    """640x480 BGR: a red disc, a blue square and the word EXIT on white. Nobody, no fire."""
    import cv2
    import numpy as np

    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.circle(image, (170, 270), 110, (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (370, 170), (570, 370), (255, 0, 0), thickness=-1)
    cv2.putText(image, "EXIT", (230, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), thickness=8)
    return image


def picture_mentions(text: str) -> dict[str, bool]:
    """Which of the picture's known features an answer names (words, case-insensitive)."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {"red": "red" in words, "blue": "blue" in words, "round_shape": bool(words & set(SHAPES[0])),
            "square_shape": bool(words & set(SHAPES[1])), "word_exit": WORD in words}


def chatml_problem(prompt: str, system: str, text: str) -> str | None:
    """None if ``prompt`` is Sentinel's two messages in the ChatML shape both LFM2-VL templates produce."""
    expected_start = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
    start = prompt.find("<|im_start|>")
    if start < 0 or not prompt[start:].startswith(expected_start):
        return "system_or_user_header"
    if not prompt.endswith(f"{text}<|im_end|>\n<|im_start|>assistant\n"):
        return "user_text_or_generation_prompt"
    if prompt.count("<|im_start|>") != 3:
        return "turn_count"
    return None


class RecordingTransport:
    """The runtime's transport, keeping the last response body (for its timings and usage)."""

    def __init__(self, inner: LoopbackTransport) -> None:
        self.inner = inner
        self.last: bytes | None = None

    def post(self, body: bytes, **kwargs: Any) -> bytes:
        self.last = None
        self.last = self.inner.post(body, **kwargs)
        return self.last


def _get_json(port: int, method: str, path: str, body: bytes | None = None) -> tuple[int, Any]:
    conn = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=REQUEST_TIMEOUT_S)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read(4_000_000)
        try:
            return response.status, json.loads(data)
        except ValueError:
            return response.status, None
    finally:
        conn.close()


def _memory() -> dict[str, int | None]:
    info = read_meminfo() or {}
    return {key: info.get(key) for key in ("MemTotal", "MemFree", "MemAvailable")}


def _usage(raw: bytes | None) -> dict[str, Any]:
    try:
        payload = json.loads(raw or b"")
    except ValueError:
        return {}
    timings = payload.get("timings") if isinstance(payload, dict) else None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    keep = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_token_ms", "cache_n")
    return {"timings": {k: timings.get(k) for k in keep} if isinstance(timings, dict) else None,
            "usage": usage if isinstance(usage, dict) else None,
            "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason") if isinstance(payload, dict)
            else None,
            "think_tag": "<think>" in json.dumps(payload)}


def llama_servers() -> list[int]:
    found = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
    return [int(pid) for pid in found.stdout.split()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    chosen = scene_model_set(args.model_set)
    model, mmproj = chosen.model_path, chosen.mmproj_path
    binary = args.llama_server or chosen.llama_server  # the set's own pinned build (session 116)
    result: dict[str, Any] = {"model_set": chosen.key, "request_model": chosen.request_model,
                              "model_revision": chosen.model_revision, "source": chosen.source,
                              "files": {"model": model.name, "mmproj": mmproj.name}, "binary": str(binary),
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "memory": {"before": _memory()}, "problems": []}
    problems: list[str] = result["problems"]
    if llama_servers():
        problems.append("another_llama_server_running")
        return result
    if port_in_use(args.port):
        problems.append("port_in_use")
        return result
    problem, memory = memfree_problem(read_meminfo, MIN_FREE_BYTES)
    result["memory"]["precheck"] = memory
    if problem is not None:
        problems.append(f"memfree_precheck:{problem}")
        return result
    server = LlamaServerProcess(binary, model, mmproj, args.port)
    started = time.monotonic()
    status = server.start(args.ready_timeout_s)
    result["readiness"] = {"state": status.state.value, "problem": status.problem, "layers": status.layers,
                           "vision_on_gpu": status.vision_on_gpu, "ready_s": round(time.monotonic() - started, 2)}
    try:
        if status.state.value != "ready":
            problems.append(f"not_ready:{status.problem}")
            return result
        result["memory"]["ready"] = _memory()
        if args.post_load_release:
            time.sleep(SETTLE_S)
            record = release_component_files("scene", {"llm": model, "mmproj": mmproj})  # judged by memory_policy
            result["release"] = {"problem": record["problem"],
                                 "results": {role: item.get("result") for role, item in record["files"].items()}}
            if not record["verified"]:
                problems.append(f"release_failed:{record['problem']}")
            result["memory"]["released"] = _memory()
        code, props = _get_json(args.port, "GET", "/props")
        template = props.get("chat_template") if isinstance(props, dict) else None
        result["props"] = {
            "status": code, "vision": ((props or {}).get("modalities") or {}).get("vision") if isinstance(props, dict)
            else None, "build_info": (props or {}).get("build_info") if isinstance(props, dict) else None,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None}
        if result["props"]["vision"] is not True:
            problems.append("no_vision_modality")
        if args.expect_template_sha256 and result["props"]["template_sha256"] != args.expect_template_sha256:
            problems.append("template_not_the_model_files")
        image = synthetic_picture()
        jpeg = opencv_jpeg(image)
        result["picture"] = {"jpeg_bytes": len(jpeg), "sha256": hashlib.sha256(jpeg).hexdigest()}
        renders = {}
        for name, body in (("scene", scene_request_body(jpeg, chosen.request_model, chosen.chat_template_kwargs)),
                           ("chat", chat_request_body(jpeg, ChatQuestion(QUESTION), CHAT_MAX_TOKENS,
                                                      chosen.request_model, chosen.chat_template_kwargs))):
            request = json.loads(body)
            code, rendered = _get_json(args.port, "POST", "/apply-template",
                                       json.dumps({"messages": request["messages"]}).encode())
            prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
            system = request["messages"][0]["content"]
            text = request["messages"][1]["content"][1]["text"]
            renders[name] = {"status": code, "problem": chatml_problem(prompt, system, text)
                             if isinstance(prompt, str) else "no_prompt",
                             # what precedes the first turn (a BOS text, or nothing: then it is added as a token)
                             "prefix": prompt[:max(prompt.find("<|im_start|>"), 0)][:32] if isinstance(prompt, str)
                             else None,
                             "picture_markers": [m for m in ("<__media__>", "<image>") if isinstance(prompt, str)
                                                 and m in prompt]}
            if renders[name]["problem"] is not None:
                problems.append(f"template_{name}:{renders[name]['problem']}")
        result["template"] = renders
        transport = RecordingTransport(LoopbackTransport(args.port, REQUEST_TIMEOUT_S))
        stamper = FrameStamper("smoke", SystemClock())
        stamper.connect()
        scene = LlamaSceneRequest(transport, model=chosen.request_model, chat_template_kwargs=chosen.chat_template_kwargs)
        result["scene"] = []
        for index in range(args.repeats):
            frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
            job = AnalysisJob(job_id=f"smoke-{index}", purpose=JobPurpose.PERIODIC, frame=frame,
                              deadline_mono_ns=frame.ingest_mono_ns + JOB_TIMEOUT_MS * 1_000_000)
            begun = time.monotonic()
            outcome = scene(job, image)
            wall_ms = round((time.monotonic() - begun) * 1000, 1)
            item: dict[str, Any] = {"wall_ms": wall_ms, "kind": outcome.kind.value, **_usage(transport.last)}
            if outcome.text is None:
                item["problem"] = outcome.detail
                problems.append(f"scene_{index}:{outcome.detail}")
            else:
                try:
                    report = parse_scene_report(outcome.text)
                except SceneReportError as exc:
                    item["problem"] = str(exc)
                    problems.append(f"scene_{index}:report_rejected")
                else:
                    item["report"] = report.model_dump(mode="json")
                    if report.persons_visible != 0:
                        problems.append(f"scene_{index}:persons_visible_{report.persons_visible}")
            if wall_ms > JOB_TIMEOUT_MS:
                item["over_job_timeout"] = True
            result["scene"].append(item)
        chat = LlamaChatRequest(transport, model=chosen.request_model, chat_template_kwargs=chosen.chat_template_kwargs)
        result["chat"] = []
        for index in range(args.repeats):
            begun = time.monotonic()
            try:
                reply = chat(ChatQuestion(QUESTION), image, max_tokens=CHAT_MAX_TOKENS)
            except Exception as exc:  # noqa: BLE001 - a failed question is this check's result
                result["chat"].append({"problem": f"{type(exc).__name__}: {exc}"})
                problems.append(f"chat_{index}:{type(exc).__name__}")
                continue
            run_ms = round((time.monotonic() - begun) * 1000, 1)
            sample, why = (pace_sample(run_ms, reply.timing) if reply.timing is not None else (None, "no_timing"))
            mentions = picture_mentions(reply.answer)
            result["chat"].append({"run_ms": run_ms, "answer": reply.answer, "cut_short": reply.cut_short,
                                   "timing": None if reply.timing is None else vars(reply.timing),
                                   "pace_sample": None if sample is None else {"overhead_ms": round(sample[0], 1),
                                                                               "ms_per_token": round(sample[1], 2)},
                                   "no_sample_because": why, "mentions": mentions, **_usage(transport.last)})
            if reply.timing is None:
                problems.append(f"chat_{index}:no_timings")
            if not (mentions["red"] and mentions["blue"]):
                problems.append(f"chat_{index}:colours_not_named")
        result["memory"]["end_of_requests"] = _memory()
        return result
    finally:
        stopped = server.stop(10.0)
        time.sleep(0.5)
        result["stop"] = {"stopped": stopped, "state": server.status().state.value,
                          "returncode": server.status().returncode, "llama_servers_left": len(llama_servers()),
                          "port_open": port_in_use(args.port)}
        if not stopped or result["stop"]["llama_servers_left"] or result["stop"]["port_open"]:
            problems.append("unclean_stop")
        result["memory"]["after_stop"] = _memory()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-set", choices=sorted(SCENE_MODEL_SETS), required=True)
    parser.add_argument("--llama-server", type=Path, default=None,
                        help="the llama-server binary (default: the selected set's own pinned build)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--repeats", type=int, default=3, choices=range(1, MAX_REPEATS + 1))
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--post-load-release", action="store_true")
    parser.add_argument("--expect-template-sha256", help="the chat template's SHA-256 in the model file, if known")
    parser.add_argument("--out", type=Path, required=True, help="a new JSON file for the result")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists")
    result = run(args)
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["verdict"] = "passed" if not result["problems"] else "failed"
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"smoke": result["verdict"], "problems": result["problems"]}), flush=True)
    return 0 if result["verdict"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
