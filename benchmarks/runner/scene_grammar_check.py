#!/usr/bin/env python3
"""A short guarded verification that the scene request's grammar constrains actual generation on this device's
llama-server (session 115, part 20). Operator tooling, not part of the runtime; diagnostic evidence only: it changes no
verdict, no criterion, no request and no admission, and it is not a resource measurement.

    scene_grammar_check.py run --model-set lfm2.5-vl --post-load-release --out DIR/check.json

It starts one llama-server as scene_grammar_repro.py does (the runtime's own LlamaServerProcess, the set's files, the
profiled flags with the prompt cache off, L4T's libcuda, the readiness checks, the runtime's MemFree precheck, no other
llama-server; with ``--post-load-release`` D58's release after the settle). Then, for ROUNDS rounds, three requests
about the chat preemption check's synthetic picture (twelve labelled shapes in eight colours on white; nobody, no fire:
the picture whose description ran to more than five observations four of four times without a grammar, in
cp-20261027T112950Z):

- **chat:** the runtime's chat request (chat_request_body) with a fixed short question: ordinary chat, which carries no
  grammar, must keep answering;
- **unconstrained:** the runtime's scene request (scene_request_body) with its top-level ``grammar`` member removed and
  nothing else changed: the control, the same picture and prompt without the grammar;
- **grammar:** the runtime's scene request as it is sent since part 20, byte for byte scene_request_body's.

The control and the grammar request differ in that member only (checked and recorded). Each request is posted and read
as scene_grammar_repro.py posts and reads it: the status, or a transport label; for an error the envelope's fixed type
and category, never the message; for an answer the finish reason, the usage and timings counts, and the strict parser's
outcome (valid with the report's enumerations and lengths, or the rejection's reason with the known field paths and
error categories); for a chat answer only its length and whether it parses as a report. Never a prompt, the picture, an
answer, a report's text or a server's message.

One outcome label: ``enforced`` when every grammar request was answered, finished ``stop`` and was strict-valid, and
the control was rejected for more than five observations at least once (the constraint was seen to bind);
``enforced_control_not_exercised`` when the grammar requests all passed but the control never exceeded the limit (then
this picture did not show the constraint binding); ``grammar_requests_failed`` (a grammar request not answered),
``grammar_reports_rejected`` (answered, but a report the parser rejects: the rejection's reason says whether the
grammar failed to bind, ``observations: too_long``, or the token budget ended it, ``truncated``), ``chat_failed`` (a
chat request not answered), else ``mixed``. The verdict ``complete`` means every request got a status and the server
stopped cleanly, never a pass.

What this establishes: that on this build with this model the request's grammar reaches the sampler and bounds the
observations where the same prompt without it does not, that the request is answered (no sampler-initialisation
error), and that chat is answered beside it. What it does not: scene accuracy, memory or timing under the full workload,
anything about the real clip, other builds, flags or templates, or admission (D46).

Run it under the existing guard, from the worktree, with the runtime's Python (it has OpenCV):

    env PYTHONPATH=src ~/sentinel-surveillance/.venv/bin/python benchmarks/runner/step5_guard.py --out DIR \\
      --from-launch --duration-s 300 -- env LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \\
      PYTHONPATH=src ~/onvif_env/bin/python benchmarks/runner/scene_grammar_check.py run --model-set lfm2.5-vl \\
      --post-load-release --out DIR/check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chat_preempt_check import server_build, synthetic_picture  # noqa: E402
from scene_grammar_repro import _memory, one_request, post, read_vmstat  # noqa: E402
from scene_smoke import DEFAULT_BINARY, MIN_FREE_BYTES, PORT, SETTLE_S, llama_servers  # noqa: E402
from sentinel.demo_runtime import memfree_problem  # noqa: E402
from sentinel.memory_policy import release_component_files  # noqa: E402
from sentinel.scene.chat import CHAT_MAX_TOKENS, ChatQuestion, chat_request_body  # noqa: E402
from sentinel.scene.completion import SCENE_REPORT_GRAMMAR_SHA256, scene_grammar_problem  # noqa: E402
from sentinel.scene.llama_server import opencv_jpeg, scene_request_body  # noqa: E402
from sentinel.scene.models import SCENE_MODEL_SETS, scene_model_set  # noqa: E402
from sentinel.scene.report import SceneReportError, parse_scene_report  # noqa: E402
from sentinel.scene.server import LlamaServerProcess, port_in_use  # noqa: E402
from sentinel.tracking.probe import read_meminfo  # noqa: E402

CASES = ("chat", "unconstrained", "grammar")
MAX_ROUNDS = 5
QUESTION = "Which shapes and colours do you see in this picture, and is there any text?"
OBSERVATIONS_OVER_LIMIT = {"loc": "observations", "type": "too_long"}  # the parser's rejection of a sixth observation


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def case_bodies(jpeg: bytes, model: str) -> dict[str, bytes]:
    """The three requests: the runtime's chat body, the runtime's scene body without its ``grammar`` member, and the
    runtime's scene body as sent (scene_request_body's bytes)."""
    with_grammar = scene_request_body(jpeg, model)
    request = json.loads(with_grammar)
    del request["grammar"]
    return {"chat": chat_request_body(jpeg, ChatQuestion(QUESTION), CHAT_MAX_TOKENS, model),
            "unconstrained": json.dumps(request).encode("utf-8"), "grammar": with_grammar}


def body_difference(unconstrained: bytes, with_grammar: bytes) -> dict[str, Any]:
    a, b = json.loads(unconstrained), json.loads(with_grammar)
    differing = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    grammar = b.get("grammar")
    return {"differing_members": differing, "only_grammar": differing == ["grammar"],
            "grammar_is_pinned": isinstance(grammar, str) and _sha256(grammar.encode("utf-8")) ==
            SCENE_REPORT_GRAMMAR_SHA256}


def chat_facts(item: dict[str, Any]) -> dict[str, Any]:
    """Whether a chat answer parses as a scene report (it should not be constrained to one); its length is kept by
    one_request, its text never."""
    answer = item.get("answer")
    if not isinstance(answer, Mapping) or answer.get("content_chars") is None:
        return {"parses_as_report": None}
    return {"parses_as_report": answer.get("parses_as_report")}


def comparison(requests: list[dict[str, Any]]) -> dict[str, Any]:
    per_case: dict[str, Any] = {}
    for name in CASES:
        items = [item for item in requests if item["case"] == name]
        answers = [item["answer"] for item in items if item["answer"] is not None]
        errors = [item["error"] for item in items if item["error"] is not None]
        validations = [answer["validation"] for answer in answers if isinstance(answer.get("validation"), Mapping)]
        rejections = [v for v in validations if v.get("valid") is not True]
        per_case[name] = {
            "requests": len(items),
            "statuses": sorted({str(item["status"]) for item in items}),
            "transport": sorted({item["transport"] for item in items if item["transport"] is not None}),
            "error_categories": sorted({error["category"] for error in errors}),
            "finish_reasons": sorted({answer["finish_reason"] for answer in answers}),
            "valid_reports": sum(v.get("valid") is True for v in validations),
            "rejected_reports": len(rejections),
            "rejection_reasons": sorted({str(v.get("reason")) for v in rejections}),
            "observations_over_limit": sum(OBSERVATIONS_OVER_LIMIT in (v.get("details") or []) for v in rejections),
            "observations_max": max((v.get("observations") for v in validations if v.get("valid") is True),
                                    default=None),
        }
    answered = {name: per_case[name]["requests"] > 0 and per_case[name]["statuses"] == ["200"] and
                not per_case[name]["transport"] for name in CASES}
    grammar = per_case["grammar"]
    grammar_ok = answered["grammar"] and grammar["finish_reasons"] == ["stop"] and \
        grammar["valid_reports"] == grammar["requests"]
    if not answered["chat"]:
        outcome = "chat_failed"
    elif not answered["grammar"]:
        outcome = "grammar_requests_failed"
    elif grammar["rejected_reports"] or grammar["finish_reasons"] != ["stop"]:
        outcome = "grammar_reports_rejected"
    elif grammar_ok and answered["unconstrained"] and per_case["unconstrained"]["observations_over_limit"] > 0:
        outcome = "enforced"
    elif grammar_ok and answered["unconstrained"]:
        outcome = "enforced_control_not_exercised"
    else:
        outcome = "mixed"  # a control not answered, or a grammar answer without a finish reason or validation
    return {"cases": per_case, "outcome": outcome}


def run(args: argparse.Namespace, *, server_factory: Callable[..., Any] = LlamaServerProcess,
        servers: Callable[[], list[int]] = llama_servers, in_use: Callable[[int], bool] = port_in_use,
        meminfo: Callable[[], dict[str, int] | None] = read_meminfo, picture: Callable[[], Any] = synthetic_picture,
        encode: Callable[[Any], bytes] = opencv_jpeg, send: Callable[..., Any] = post,
        build: Callable[[int], str | None] = server_build, vmstat: Callable[[], dict[str, int]] = read_vmstat,
        sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    chosen = scene_model_set(args.model_set)
    model, mmproj = chosen.model_path, chosen.mmproj_path
    grammar_problem = scene_grammar_problem()  # before the fingerprint, which refuses a stale grammar by raising
    result: dict[str, Any] = {"model_set": chosen.key, "request_model": chosen.request_model,
                              "scene_request_sha256": None if grammar_problem else chosen.scene_request_sha256,
                              "grammar_problem": grammar_problem,
                              "files": {"model": model.name, "mmproj": mmproj.name}, "binary": str(args.llama_server),
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "rounds": args.rounds, "memory": {"before": _memory(meminfo)}, "vmstat": vmstat(),
                              "problems": [], "requests": []}
    problems: list[str] = result["problems"]
    if result["grammar_problem"] is not None:
        problems.append(f"grammar:{result['grammar_problem']}")
        return result
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
    server = server_factory(args.llama_server, model, mmproj, args.port)
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
        bodies = case_bodies(jpeg, chosen.request_model)
        result["picture"] = {"jpeg_bytes": len(jpeg), "sha256": _sha256(jpeg)}
        result["bodies"] = {name: {"bytes": len(body), "sha256": _sha256(body)} for name, body in bodies.items()}
        result["bodies"]["grammar"]["is_scene_request_body"] = bodies["grammar"] == scene_request_body(
            jpeg, chosen.request_model)
        result["scene_bodies_differ"] = body_difference(bodies["unconstrained"], bodies["grammar"])
        if not (result["scene_bodies_differ"]["only_grammar"] and result["scene_bodies_differ"]["grammar_is_pinned"]):
            problems.append("scene_bodies_differ_beyond_the_pinned_grammar")
            return result
        for round_number in range(1, args.rounds + 1):
            for name in CASES:
                begun = time.monotonic()
                status_code, data, transport = send(args.port, bodies[name])
                wall_ms = round((time.monotonic() - begun) * 1000, 1)
                item = one_request("basic" if name == "chat" else name, status_code, data, transport, wall_ms)
                item["case"], item["round"] = name, round_number
                if name == "chat" and item["answer"] is not None and item["transport"] is None:
                    item["answer"]["parses_as_report"] = _parses_as_report(data)
                    item["answer"]["validation"] = "not_applicable"
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


def _parses_as_report(data: bytes) -> bool | None:
    """Whether a chat answer's text is a scene report the strict parser accepts; the text is never kept."""
    try:
        content = json.loads(data)["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    try:
        parse_scene_report(content)
    except SceneReportError:  # a ValueError too, so it is told apart from an unreadable answer above
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="one server load and the bounded verification")
    runner.add_argument("--model-set", choices=sorted(SCENE_MODEL_SETS), required=True)
    runner.add_argument("--llama-server", type=Path, default=DEFAULT_BINARY)
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
    print(json.dumps({"check": result["verdict"], "outcome": (result.get("comparison") or {}).get("outcome"),
                      "problems": result["problems"]}), flush=True)
    return 0 if result["verdict"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
