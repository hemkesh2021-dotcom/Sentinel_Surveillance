#!/usr/bin/env python3
"""A bounded, standalone reproduction of the LFM2.5-VL scene request's HTTP errors on this device's llama-server
(session 115, part 19). Operator tooling, not part of the runtime; diagnostic evidence only: it changes no verdict, no
criterion, no request and no admission, and it is not a resource measurement.

    scene_grammar_repro.py run --model-set lfm2.5-vl --post-load-release --out DIR/repro.json

It starts one llama-server through the runtime's own LlamaServerProcess, as scene_smoke.py does (the model set's files,
the profiled flags with the prompt cache off, L4T's libcuda, the readiness checks), after the runtime's MemFree
precheck and only if no other llama-server runs. With ``--post-load-release`` the model files are released from the
page cache after the load settles (D58). Then, for ROUNDS rounds, three requests about scene_smoke's synthetic picture
(labelled shapes on white; nobody, no fire):

- **basic:** the runtime's chat request (chat_request_body) with a fixed short question: image input and decoding with
  no output constraint at all;
- **response_format:** the runtime's scene request (scene_request_body) without its constraint member (the
  ``grammar`` since part 20) and nothing else changed: the request every accepted profile measured (D47 before part 17);
- **json_schema:** part 17's request (bc8fc91): that body with the top-level ``json_schema`` field in the grammar's
  place, the request step4cand-lfm25-20261027T145229Z sent, kept here as the reproduction of its failure.

The last two differ in that one member only; the result records both bodies' SHA-256 and that check. (Until part 20
the third case was the runtime's own request; scenegram-20261027T160238Z ran it so, with the field.) Each request is
posted on the loopback with http.client, as the runtime's transport does, but the response body is read whatever the
status: the runtime's transport and the step-4 workload keep only an error's status, which is why
step4cand-lfm25-20261027T145229Z could not say why its 150 scene requests failed.

For each request the result keeps: the HTTP status, or the transport's fixed label; for an error, the envelope's type
(one of b8932's fixed strings, else "other") and a category from b8932's known messages, and for the sampler's grammar
prefill, whether the rejected piece begins the chat template's generation prompt; for an answer, the finish reason, the
usage and timings counts, and the strict parser's outcome (valid, with the report's enumerations and lengths, or the
rejection's reason with the known field paths and error categories). Never a prompt, the picture, an answer, a report's
text or a server's message (only its SHA-256 and length).

What it can establish: whether this server with this model answers an image request at all, answers the request
without the field, and rejects the request with it, with the server's own reason category. What it cannot: whether a
grammar was applied to an answer (the server does not report it), scene accuracy, memory or timing under the full
workload, or anything about other builds, models, flags or templates.

Run it under the existing guard, from the worktree, with the runtime's Python (it has OpenCV):

    env PYTHONPATH=src ~/sentinel-surveillance/.venv/bin/python benchmarks/runner/step5_guard.py --out DIR \\
      --from-launch --duration-s 300 -- env LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \\
      PYTHONPATH=src ~/onvif_env/bin/python benchmarks/runner/scene_grammar_repro.py run --model-set lfm2.5-vl \\
      --post-load-release --out DIR/repro.json

Exit status 0 means the comparison completed (every request got a status or a transport label) and the server was
stopped cleanly; the outcome is a finding, never a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import socket
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chat_preempt_check import rejection_details, report_facts, server_build  # noqa: E402
from llama_errors import error_facts  # noqa: E402
from scene_smoke import (  # noqa: E402
    DEFAULT_BINARY,
    JOB_TIMEOUT_MS,
    MIN_FREE_BYTES,
    PORT,
    REQUEST_TIMEOUT_S,
    SETTLE_S,
    llama_servers,
    synthetic_picture,
)
from sentinel.demo_runtime import memfree_problem  # noqa: E402
from sentinel.memory_policy import release_component_files  # noqa: E402
from sentinel.scene.chat import CHAT_MAX_TOKENS, ChatQuestion, chat_request_body  # noqa: E402
from sentinel.scene.completion import (  # noqa: E402
    SceneCompletionError,
    parse_scene_completion,
    scene_completion_content,
    scene_completion_finish_reason,
    scene_report_schema,
)
from sentinel.scene.llama_server import COMPLETIONS_PATH, LOOPBACK_HOST, MAX_RESPONSE_BYTES, opencv_jpeg, \
    scene_request_body  # noqa: E402
from sentinel.scene.models import SCENE_MODEL_SETS, scene_model_set  # noqa: E402
from sentinel.scene.report import SceneReportError  # noqa: E402
from sentinel.scene.server import LlamaServerProcess, port_in_use  # noqa: E402
from sentinel.tracking.probe import read_meminfo  # noqa: E402

CASES = ("basic", "response_format", "json_schema")
MAX_ROUNDS = 5
QUESTION = "Describe this picture in one short sentence."
VMSTAT = Path("/proc/vmstat")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def case_bodies(jpeg: bytes, model: str, chat_template_kwargs: Mapping[str, object] | None = None) -> dict[str, bytes]:
    """The three requests: the runtime's chat body; the runtime's scene body without its constraint member (since
    part 20 the ``grammar``; the request every accepted profile measured); and part 17's request (bc8fc91), that body
    with the top-level ``json_schema`` field in the grammar's place, kept as the reproduction of the failure. Each
    carries the set's template arguments, when it has any (session 116)."""
    request = json.loads(scene_request_body(jpeg, model, chat_template_kwargs))
    without_field = {key: value for key, value in request.items() if key != "grammar"}
    with_field = {key: (scene_report_schema() if key == "grammar" else value) for key, value in request.items()}
    with_field = {("json_schema" if key == "grammar" else key): value for key, value in with_field.items()}
    return {"basic": chat_request_body(jpeg, ChatQuestion(QUESTION), CHAT_MAX_TOKENS, model, chat_template_kwargs),
            "response_format": json.dumps(without_field).encode("utf-8"),  # scene_request_body's serialisation
            "json_schema": json.dumps(with_field).encode("utf-8")}


def body_difference(without_field: bytes, with_field: bytes) -> dict[str, Any]:
    """Which top-level members differ between the two scene bodies: ``["json_schema"]`` when only the field does."""
    a, b = json.loads(without_field), json.loads(with_field)
    differing = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    return {"differing_members": differing, "only_json_schema": differing == ["json_schema"],
            "json_schema_in_response_format": "json_schema" not in a and "json_schema" in b}


def answer_facts(payload: Any, scene: bool) -> dict[str, Any]:
    """A 200 answer: the finish reason, the usage and timings counts, and for a scene case the strict parser's
    outcome; never the text."""
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}
    timings = payload.get("timings") if isinstance(payload, Mapping) else None
    timings = timings if isinstance(timings, Mapping) else {}
    facts: dict[str, Any] = {
        "finish_reason": scene_completion_finish_reason(payload),
        "usage": {key: usage[key] if type(usage.get(key)) is int and usage[key] >= 0 else None
                  for key in ("prompt_tokens", "completion_tokens")},
        "timings": {key: timings.get(key) if isinstance(timings.get(key), (int, float)) else None
                    for key in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")},
    }
    try:
        content = scene_completion_content(payload)
    except SceneCompletionError as exc:
        facts["content_chars"] = None
        facts["validation"] = {"valid": False, "reason": exc.reason}
        return facts
    facts["content_chars"] = len(content)
    if not scene:
        facts["validation"] = "not_applicable"
        return facts
    try:
        parse_scene_completion(payload)
    except SceneReportError:
        facts["validation"] = {"valid": False, "reason": "invalid_report", "details": rejection_details(content)}
    else:
        facts["validation"] = report_facts(content)
    return facts


def post(port: int, body: bytes, timeout_s: float = REQUEST_TIMEOUT_S) -> tuple[int | None, bytes, str | None]:
    """(status, body, transport label): the body whatever the status, bounded as the runtime bounds it."""
    conn = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=timeout_s)
    try:
        conn.request("POST", COMPLETIONS_PATH, body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read(MAX_RESPONSE_BYTES + 1), None
    except (TimeoutError, socket.timeout):
        return None, b"", "request_timed_out"
    except (OSError, http.client.HTTPException) as exc:
        return None, b"", f"request_failed:{type(exc).__name__}"
    finally:
        conn.close()


def one_request(name: str, status: int | None, data: bytes, transport: str | None, wall_ms: float) -> dict[str, Any]:
    item: dict[str, Any] = {"case": name, "wall_ms": wall_ms, "status": status, "transport": transport,
                            "response_bytes": len(data), "error": None, "answer": None}
    if transport is not None:
        return item
    if len(data) > MAX_RESPONSE_BYTES:
        item["transport"] = "response_too_large"
        return item
    try:
        payload = json.loads(data)
    except ValueError:
        item["transport"] = "response_not_json"
        return item
    if status == 200:
        item["answer"] = answer_facts(payload, scene=name != "basic")
        if wall_ms > JOB_TIMEOUT_MS:
            item["over_job_timeout"] = True
    else:
        item["error"] = error_facts(payload)
    return item


def comparison(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Per case, the statuses, error categories, finish reasons and validity seen, and one fixed outcome label."""
    per_case: dict[str, Any] = {}
    for name in CASES:
        items = [item for item in requests if item["case"] == name]
        errors = [item["error"] for item in items if item["error"] is not None]
        answers = [item["answer"] for item in items if item["answer"] is not None]
        per_case[name] = {
            "requests": len(items),
            "statuses": sorted({str(item["status"]) for item in items}),
            "transport": sorted({item["transport"] for item in items if item["transport"] is not None}),
            "error_categories": sorted({error["category"] for error in errors}),
            "grammar_prefill_rejected": sum(error["grammar_prefill_rejected"] for error in errors),
            "finish_reasons": sorted({answer["finish_reason"] for answer in answers}),
            "valid_reports": sum(isinstance(answer["validation"], Mapping) and answer["validation"].get("valid")
                                 is True for answer in answers),
        }
    answered = {name: per_case[name]["requests"] > 0 and per_case[name]["statuses"] == ["200"] for name in CASES}
    with_field = per_case["json_schema"]
    if not answered["basic"]:
        outcome = "basic_request_failed"
    elif all(answered.values()):
        outcome = "all_answered"
    elif answered["response_format"] and with_field["statuses"] == ["400"] and \
            with_field["error_categories"] == ["sampler_init_failed"] and \
            with_field["grammar_prefill_rejected"] == with_field["requests"]:
        outcome = "field_rejected_at_sampler_init"
    elif answered["response_format"] and "200" not in with_field["statuses"]:
        outcome = "field_rejected_otherwise"
    else:
        outcome = "mixed"
    return {"cases": per_case, "outcome": outcome}


def read_vmstat(path: Path = VMSTAT) -> dict[str, int]:
    """The page-cache drop counter since boot (D37: a drop is confirmed by this counter, never by an answer)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values = dict(line.split() for line in lines if line.count(" ") == 1)
    return {key: int(values[key]) for key in ("drop_pagecache",) if values.get(key, "").isdigit()}


def _memory(meminfo: Callable[[], dict[str, int] | None]) -> dict[str, int | None]:
    info = meminfo() or {}
    return {key: info.get(key) for key in ("MemTotal", "MemFree", "MemAvailable")}


def run(args: argparse.Namespace, *, server_factory: Callable[..., Any] = LlamaServerProcess,
        servers: Callable[[], list[int]] = llama_servers, in_use: Callable[[int], bool] = port_in_use,
        meminfo: Callable[[], dict[str, int] | None] = read_meminfo, picture: Callable[[], Any] = synthetic_picture,
        encode: Callable[[Any], bytes] = opencv_jpeg, send: Callable[..., Any] = post,
        build: Callable[[int], str | None] = server_build, vmstat: Callable[[], dict[str, int]] = read_vmstat,
        sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    chosen = scene_model_set(args.model_set)
    model, mmproj = chosen.model_path, chosen.mmproj_path
    binary = args.llama_server or chosen.llama_server  # the set's own pinned build (session 116)
    result: dict[str, Any] = {"model_set": chosen.key, "request_model": chosen.request_model,
                              "scene_request_sha256": chosen.scene_request_sha256,
                              "files": {"model": model.name, "mmproj": mmproj.name}, "binary": str(binary),
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "rounds": args.rounds, "memory": {"before": _memory(meminfo)}, "vmstat": vmstat(),
                              "problems": [], "requests": []}
    problems: list[str] = result["problems"]
    if servers():
        problems.append("another_llama_server_running")
        return result
    if in_use(args.port):
        problems.append("port_in_use")
        return result
    problem, memory = memfree_problem(meminfo, MIN_FREE_BYTES)
    result["memory"]["precheck"] = memory
    if problem is not None:
        problems.append(f"memfree_precheck:{problem}")
        return result
    server = server_factory(binary, model, mmproj, args.port, extra_flags=chosen.server_flags)
    started = time.monotonic()
    status = server.start(args.ready_timeout_s)
    result["readiness"] = {"state": status.state.value, "problem": status.problem, "layers": status.layers,
                           "vision_on_gpu": status.vision_on_gpu, "ready_s": round(time.monotonic() - started, 2)}
    try:
        if status.state.value != "ready":
            problems.append(f"not_ready:{status.problem}")
            return result
        result["memory"]["ready"] = _memory(meminfo)
        if args.post_load_release:
            sleep(SETTLE_S)
            record = release_component_files("scene", {"llm": model, "mmproj": mmproj})  # judged by memory_policy
            result["release"] = {"problem": record["problem"],
                                 "results": {role: item.get("result") for role, item in record["files"].items()}}
            if not record["verified"]:
                problems.append(f"release_failed:{record['problem']}")
            result["memory"]["released"] = _memory(meminfo)
        result["build_info"] = build(args.port)
        jpeg = encode(picture())
        bodies = case_bodies(jpeg, chosen.request_model, chosen.chat_template_kwargs)
        result["picture"] = {"jpeg_bytes": len(jpeg), "sha256": _sha256(jpeg)}
        result["bodies"] = {name: {"bytes": len(body), "sha256": _sha256(body)} for name, body in bodies.items()}
        request = json.loads(scene_request_body(jpeg, chosen.request_model, chosen.chat_template_kwargs))
        result["bodies"]["json_schema"]["is_part_17s_request"] = json.loads(bodies["json_schema"]) == {
            **{key: value for key, value in request.items() if key != "grammar"}, "json_schema": scene_report_schema()}
        result["bodies"]["response_format"]["is_scene_request_body_without_grammar"] = \
            json.loads(bodies["response_format"]) == {k: v for k, v in request.items() if k != "grammar"}
        result["scene_bodies_differ"] = body_difference(bodies["response_format"], bodies["json_schema"])
        if not result["scene_bodies_differ"]["only_json_schema"]:
            problems.append("scene_bodies_differ_beyond_the_field")
            return result
        for round_number in range(1, args.rounds + 1):
            for name in CASES:
                begun = time.monotonic()
                status_code, data, transport = send(args.port, bodies[name])
                wall_ms = round((time.monotonic() - begun) * 1000, 1)
                item = one_request(name, status_code, data, transport, wall_ms)
                item["round"] = round_number
                result["requests"].append(item)
                if item["transport"] is not None:
                    problems.append(f"{name}_{round_number}:{item['transport']}")
        result["comparison"] = comparison(result["requests"])
        result["memory"]["end_of_requests"] = _memory(meminfo)
        return result
    finally:
        stopped = server.stop(10.0)
        sleep(0.5)
        result["stop"] = {"stopped": stopped, "state": server.status().state.value,
                          "returncode": server.status().returncode, "llama_servers_left": len(servers()),
                          "port_open": in_use(args.port)}
        if not stopped or result["stop"]["llama_servers_left"] or result["stop"]["port_open"]:
            problems.append("unclean_stop")
        result["memory"]["after_stop"] = _memory(meminfo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="one server load and the bounded comparison")
    runner.add_argument("--model-set", choices=sorted(SCENE_MODEL_SETS), required=True)
    runner.add_argument("--llama-server", type=Path, default=None,
                        help="the llama-server binary (default: the selected set's own pinned build)")
    runner.add_argument("--port", type=int, default=PORT)
    runner.add_argument("--rounds", type=int, default=3, choices=range(1, MAX_ROUNDS + 1))
    runner.add_argument("--ready-timeout-s", type=float, default=180.0)
    runner.add_argument("--post-load-release", action="store_true")
    runner.add_argument("--out", type=Path, required=True, help="a new JSON file for the result")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists")
    result = run(args)
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    expected = args.rounds * len(CASES)
    result["verdict"] = "complete" if not result["problems"] and len(result["requests"]) == expected else "incomplete"
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"repro": result["verdict"], "outcome": (result.get("comparison") or {}).get("outcome"),
                      "problems": result["problems"]}), flush=True)
    return 0 if result["verdict"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
