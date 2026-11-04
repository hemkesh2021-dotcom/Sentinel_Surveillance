"""Session 115, part 11: the controlled chat preemption check (benchmarks/runner/chat_preempt_check.py).

The check runs against a fake llama-server (tests/support/fake_llama_server.py) with the runtime's own scene worker,
chat service, requests, transports and the 1.5 s grace; only the minimum interval between questions is shortened here.
- **Passing:** a trial passes only when its request was still decoding when the scene job came, and the grace then
  ended it: cancelled with the scene job's reason, the request aborted (the server saw the connection close), and the
  scene job started within D80's 2000 ms with a valid report.
- **Not exercised:** an answer that ends by itself, within the grace or before the job could be submitted, is NOT
  EXERCISED, never passed.
- **Failing:** a bad scene job fails.
- **No text:** the result holds no answer or slot text.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fake_llama_server import ANSWER, BUILD, GENERATED, LFM3_BUILD, REPORT, TEMPLATE, FakeLlamaServer, \
    FakeServerProcess, free_port

from sentinel.scene.server import ServerState, ServerStatus

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
LONG, SHORT = "Describe everything", "one sentence"  # words of alpha_load's long and short questions


@pytest.fixture(scope="module")
def cp():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield importlib.import_module("chat_preempt_check")
    finally:
        sys.path.remove(str(RUNNERS))


def checked(cp, monkeypatch, tmp_path: Path, *, long_s: float, short_s: float = 0.3, trials: int = 1,
            scene_text: str | None = None, prompt_s: float = 0.0, **fake_options: Any
            ) -> tuple[dict[str, Any], FakeLlamaServer]:
    import sentinel.scene.chat as chat

    monkeypatch.setattr(chat, "CHAT_MIN_INTERVAL_S", 0.3)  # chat's one question every 4 s, shortened for the test
    monkeypatch.setattr(cp, "CHAT_MIN_INTERVAL_S", 0.3)
    port = free_port()
    fake = FakeLlamaServer(port, decode_s={LONG: long_s, SHORT: short_s}, scene_text=scene_text, prompt_s=prompt_s,
                           **fake_options)
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent/llama-server"), port=port,
                              trials=trials, withdrawals=0, cancellations=0, baselines=0, ready_timeout_s=5.0,
                              post_load_release=False,
                              out=tmp_path / "chatpreempt.json")
    result = cp.run(args, server_factory=FakeServerProcess(fake), environ={}, servers=lambda: [],
                    meminfo=lambda: None, picture=lambda: "a synthetic picture",
                    encode=lambda image: b"\xff\xd8x\xff\xd9")
    result["verdict"] = cp.overall(result)
    return result, fake


def test_the_limits_are_the_existing_ones(cp) -> None:
    sys.path.insert(0, str(RUNNERS))
    try:
        alpha_check, alpha_load = importlib.import_module("alpha_check"), importlib.import_module("alpha_load")
    finally:
        sys.path.remove(str(RUNNERS))
    from sentinel.scene.analyzer import SIDE_GRACE_S

    assert cp.MAX_CHAT_DELAY_MS == alpha_check.MAX_CHAT_DELAY_MS and cp.GRACE_MS == alpha_check.GRACE_MS
    assert cp.GRACE_MS == round(SIDE_GRACE_S * 1000) == 1500 and cp.PREEMPTED == alpha_check.CANCELLED_FOR_SCENE
    assert cp.QUESTIONS == alpha_load.QUESTIONS and LONG in cp.QUESTIONS["long"] and SHORT in cp.QUESTIONS["short"]
    with pytest.raises(SystemExit):
        cp.main(["run", "--model-set", "lfm2.5-vl", "--trials", "4", "--out", "/nonexistent/x.json"])  # at most three


def test_a_request_still_decoding_when_the_grace_ends_is_ended_for_the_scene_job(cp, monkeypatch, tmp_path) -> None:
    result, fake = checked(cp, monkeypatch, tmp_path, long_s=4.0, prompt_s=0.4)  # a prompt phase, nothing decoded
    control, (trial,) = result["control"], result["trials"]
    assert control["verdict"] == "AS EXPECTED" and control["question_end"]["state"] == "answered", control
    assert trial["verdict"] == "PASS", trial
    assert trial["in_flight_at_submission"] is True and trial["slots_at_submission"]["processing"] == 1
    assert trial["slots_at_submission"]["n_decoded"] >= 1  # submitted once decoding had begun, not during the prompt
    assert trial["submitted_after_request_began_ms"] >= 400.0
    assert (trial["question_end"]["state"], trial["question_end"]["reason"]) == ("cancelled", cp.PREEMPTED)
    assert trial["request"]["outcome"] == "aborted"
    assert (trial["counters"]["side_preempted"], trial["counters"]["side_preempted_after_grace"]) == (1, 1)
    assert 1500.0 <= trial["scene"]["start_delay_ms"] <= 2000.0, trial["scene"]
    assert 1500.0 <= trial["scene"]["waited_behind_question_ms"] <= 2000.0 and trial["scene"]["valid"] is True
    assert fake.closed_by_client == 1  # the abort reached the server while it was decoding
    assert result["problems"] == [] and result["health"]["slots"]["processing"] == 0
    assert result["verdict"] == "passed" and result["stop"]["stopped"] is True
    text = json.dumps(result)
    assert ANSWER not in text and GENERATED not in text  # metadata only: no answer and no slot text


def test_an_answer_that_ends_by_itself_within_the_grace_is_not_exercised(cp, monkeypatch, tmp_path) -> None:
    result, fake = checked(cp, monkeypatch, tmp_path, long_s=0.8)
    (trial,) = result["trials"]
    assert trial["in_flight_at_submission"] is True and trial["question_end"]["state"] == "answered"
    assert trial["verdict"] == "NOT EXERCISED" and trial["counters"]["side_finished_in_grace"] == 1
    assert "side_preempted" not in trial["counters"] and fake.closed_by_client == 0
    assert result["verdict"] == "not_exercised" and cp.EXIT[result["verdict"]] == 3


def test_an_answer_that_ends_before_the_scene_job_is_submitted_is_not_exercised(cp, monkeypatch, tmp_path) -> None:
    result, _ = checked(cp, monkeypatch, tmp_path, long_s=0.0)
    (trial,) = result["trials"]
    assert trial["in_flight_at_submission"] is False and trial["question_end"]["state"] == "answered"
    assert trial["verdict"] == "NOT EXERCISED" and trial["scene"]["waited_behind_question_ms"] is None
    assert result["verdict"] == "not_exercised"


def test_a_scene_job_without_a_valid_report_fails_whatever_happened_to_the_question(cp, monkeypatch, tmp_path) -> None:
    result, _ = checked(cp, monkeypatch, tmp_path, long_s=4.0, scene_text="{}")
    (trial,) = result["trials"]
    assert trial["question_end"]["state"] == "cancelled"  # the preemption happened ...
    assert trial["verdict"] == "FAIL" and "scene_report_report_rejected" in trial["reasons"]  # ... the scene job failed
    assert result["control"]["verdict"] == "NOT AS EXPECTED" and result["verdict"] == "failed"


def test_three_bounded_trials_are_each_reported(cp, monkeypatch, tmp_path) -> None:
    result, fake = checked(cp, monkeypatch, tmp_path, long_s=4.0, trials=3)
    assert [t["name"] for t in result["trials"]] == ["trial-1", "trial-2", "trial-3"]
    assert [t["verdict"] for t in result["trials"]] == ["PASS", "PASS", "PASS"] and fake.closed_by_client == 3


def test_slots_debug_refuses_before_anything_starts(cp, tmp_path) -> None:
    def never(*args: Any) -> Any:
        raise AssertionError("no server may start")

    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent"), port=free_port(), trials=1, withdrawals=0, cancellations=0, baselines=0,
                              ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "x.json")
    result = cp.run(args, server_factory=never, environ={cp.SLOTS_DEBUG: "1"}, servers=lambda: [])
    assert result["problems"] == ["slots_debug_set"] and cp.overall(result) == "failed"
    assert cp.run(args, server_factory=never, environ={}, servers=lambda: [4242])["problems"] == [
        "another_llama_server_running"]


def item(**changes: Any) -> dict[str, Any]:
    """An exercised trial that passes, changed in one place."""
    base: dict[str, Any] = {
        "name": "trial-1", "in_flight_at_submission": True, "request": {"outcome": "aborted"},
        "question_end": {"state": "cancelled", "reason": "a scene job is waiting: monitoring comes first"},
        "scene": {"outcome": "completed", "valid": True, "start_delay_ms": 1520.0, "waited_behind_question_ms": 1519.0,
                  "completed_after_submission_ms": 3900.0},
        "counters": {"side_preempted": 1, "side_preempted_after_grace": 1, "side_grace_started": 1}}
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        target = base
        for key in parents:
            target = target[key]
        target[leaf] = value
    return base


@pytest.mark.parametrize(("changes", "verdict", "reason"), [
    ({}, "PASS", "ended by the scene job after the grace"),
    ({"scene__start_delay_ms": 2000.1}, "FAIL", "scene_start_delay_ms_over_2000"),
    ({"scene__waited_behind_question_ms": 2400.0}, "FAIL", "scene_waited_behind_question_ms_over_2000"),
    ({"scene__completed_after_submission_ms": 8001.0}, "FAIL", "scene_job_past_its_deadline"),
    ({"scene__outcome": "timeout"}, "FAIL", "scene_job_timeout"),
    ({"question_end__reason": "no answer in time"}, "FAIL", "cancelled_for_another_reason"),
    ({"request__outcome": "returned"}, "FAIL", "request_returned"),
    ({"counters": {"side_preempted": 1}}, "FAIL", "no_preemption_after_the_grace_counted"),
    ({"question_end__state": "answered", "counters": {"side_finished_in_grace": 1}}, "NOT EXERCISED",
     "the answer ended by itself within the grace"),
    ({"question_end__state": "answered", "counters": {}}, "FAIL", "answered without ending within the grace"),
    ({"in_flight_at_submission": False}, "NOT EXERCISED", "not in flight"),
    ({"question_end__state": "error"}, "FAIL", "question_error"),
])
def test_the_trial_judging(cp, changes, verdict, reason) -> None:
    got, reasons = cp.judge_trial(item(**changes))
    assert got == verdict and any(reason in r for r in reasons), (got, reasons)


def test_the_reading_judges_the_guard_and_never_passes_without_both_results(cp, monkeypatch, tmp_path) -> None:
    result, _ = checked(cp, monkeypatch, tmp_path, long_s=4.0)
    (tmp_path / "chatpreempt.json").write_text(json.dumps(result))
    guard = {"status": "child_exited", "child": {"returncode": 0}, "stop": {"forced": False},
             "guard": {"trigger": None, "peak_pressure_bytes": 3_100_000_000}}
    (tmp_path / "result.json").write_text(json.dumps(guard))
    lines, code = cp.read(tmp_path)
    assert code == 0 and lines[-1].startswith("chat preemption check: passed")
    assert lines[0].startswith("PASS the guard") and lines[1].startswith("PASS memory")
    assert lines[2].startswith("AS EXPECTED control") and lines[3].startswith("PASS trial-1")
    assert all(ANSWER not in line and GENERATED not in line for line in lines)
    (tmp_path / "result.json").write_text(json.dumps({**guard, "guard": {"trigger": "pressure", "peak_pressure_bytes":
                                                                         4_800_000_000}}))
    lines, code = cp.read(tmp_path)
    assert code == 1 and lines[-1].startswith("chat preemption check: failed")
    (tmp_path / "result.json").unlink()
    lines, code = cp.read(tmp_path)
    assert (code, lines) == (1, ["MISSING the guard's result, result.json (FileNotFoundError): nothing is judged"])


GUARD_STOP = {  # cp-20261026T191803Z's guard result in shape and numbers: MemFree's floor reached during the load
    "status": "guard_stop", "child": {"returncode": -2, "ran_s": 3.22},
    "stop": {"requested_by": "guard_stop", "requested_at_s": 3.017, "forced": False},
    "guard": {"peak_pressure_bytes": 2_934_214_656, "min_mem_free_bytes": 879_345_664,
              "trigger": {"stop": "free_or_available_stop", "conditions": [
                  {"stop": "free_or_available_stop", "condition": "MemFree < 1073741824", "field": "MemFree",
                   "comparison": "<", "threshold": 1_073_741_824, "value": 1_004_228_608}],
                  "sample": {"MemFree": 1_004_228_608, "MemAvailable": 5_076_869_120}}},
    "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0}}


def test_a_guard_stop_before_the_check_wrote_its_result_is_read_from_the_guard(cp, tmp_path) -> None:
    (tmp_path / "result.json").write_text(json.dumps(GUARD_STOP))  # no chatpreempt.json: the check was stopped
    lines, code = cp.read(tmp_path)
    assert code == 1 and len(lines) == 3, lines
    assert lines[0] == ("FAIL the guard: status guard_stop, trigger free_or_available_stop: MemFree < 1073741824 "
                        "(value 1004228608), stop requested by guard_stop 3.017 s after launch, returncode -2, "
                        "llama-server after the check's exit 0, left 0")
    assert lines[1].startswith("FAIL memory: the guard stopped it on free_or_available_stop: MemFree < 1073741824")
    assert lines[2] == ("MISSING the check's result, chatpreempt.json (FileNotFoundError): not written; nothing else "
                        "is judged")
    assert not any(line.startswith("chat preemption check") for line in lines)  # never a verdict without both


def test_a_refused_launch_is_read_as_nothing_started(cp, tmp_path) -> None:
    (tmp_path / "result.json").write_text(json.dumps({"status": "refused:initial_headroom_refused"}))
    assert cp.read(tmp_path) == ([
        "REFUSED the guard: refused:initial_headroom_refused, before launch: nothing was started and nothing is "
        "judged"], 1)


def test_an_interrupt_while_the_server_loads_stops_the_checks_own_server(cp, tmp_path) -> None:
    class InterruptedStart(FakeServerProcess):
        """The guard's SIGINT arriving while the check waits for its server to be ready."""

        def start(self, ready_timeout_s: float) -> Any:
            self.fake.start()
            raise KeyboardInterrupt

    port = free_port()
    fake = FakeLlamaServer(port, decode_s={})
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent"), port=port, trials=1, withdrawals=0, cancellations=0, baselines=0,
                              ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "x.json")
    with pytest.raises(KeyboardInterrupt):  # the interrupt still ends the check, without a result
        cp.run(args, server_factory=InterruptedStart(fake), environ={}, servers=lambda: [], meminfo=lambda: None)
    assert not cp.port_in_use(port)  # ... after the check stopped its own server


# ---------------------------------------------------------------- session 115, part 15: attribution and rejections


def _post(port: int, body: dict[str, Any]) -> None:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        conn.getresponse().read()
    finally:
        conn.close()


def test_the_fake_keeps_the_previous_tasks_counters_through_the_next_prompt(cp) -> None:
    """As llama-server b8932: a new task's id and n_predict at once, the old counters until its prompt is done."""
    import threading
    import time

    port = free_port()
    fake = FakeLlamaServer(port, decode_s={LONG: 1.0}, prompt_s=0.6, scene_tokens=194)
    fake.start()
    try:
        _post(port, {"max_tokens": 200, "response_format": {"type": "json_schema"}, "messages": []})  # a scene job
        after_scene = cp.slot_reading(port)
        assert after_scene == {"slots": 1, "processing": False, "id_task": 1, "n_predict": 200, "n_decoded": 194,
                               "n_remain": 6}
        chat = threading.Thread(target=_post, args=(port, {"max_tokens": 120, "messages": [
            {"role": "user", "content": [{"type": "text", "text": LONG}]}]}))
        chat.start()
        time.sleep(0.3)
        in_prompt = cp.slot_reading(port)
        assert in_prompt == {"slots": 1, "processing": True, "id_task": 2, "n_predict": 120, "n_decoded": 194,
                             "n_remain": 6}  # the chat task, with the scene task's counters
        time.sleep(0.6)
        decoding = cp.slot_reading(port)
        assert decoding["id_task"] == 2 and 1 <= decoding["n_decoded"] < 120
        assert decoding["n_decoded"] + decoding["n_remain"] == 120
        chat.join(5)
    finally:
        fake.stop()


def reading(task: int | None, decoded: int, remain: int, *, n_predict: int = 120, processing: bool = True
            ) -> dict[str, Any]:
    return {"slots": 1, "processing": processing, "id_task": task, "n_predict": n_predict, "n_decoded": decoded,
            "n_remain": remain}


@pytest.mark.parametrize(("baseline", "readings", "result"), [
    # the previous scene task's 194/6 through the chat task's prompt, then the reset and the first tokens
    (reading(7, 194, 6, n_predict=200, processing=False),
     [reading(8, 194, 6), reading(8, 194, 6), reading(8, 0, 6), reading(8, 1, 119)], "decoding_observed"),
    # stale counters alone, even rising-looking pairs never come from them: they stay constant
    (reading(7, 194, 6, n_predict=200, processing=False), [reading(8, 194, 6)] * 6, None),
    # a previous chat task's counters sum to the chat cap: arithmetic alone would accept them
    (reading(7, 50, 70, processing=False), [reading(8, 50, 70)] * 6, None),
    # the slot still shows the task it held before the question
    (reading(7, 3, 117, processing=False), [reading(7, 3, 117), reading(7, 4, 116)], None),
    # a rise, but the counters do not match the task's n_predict, or its n_predict is not the chat's cap
    (reading(7, 194, 6, n_predict=200, processing=False), [reading(8, 1, 6), reading(8, 2, 6)], None),
    (reading(7, 194, 6, n_predict=200, processing=False),
     [reading(8, 1, 199, n_predict=200), reading(8, 2, 198, n_predict=200)], None),
    # ambiguous: a second new task, no task identity, an unreadable /slots, a busy slot before the question
    (reading(7, 194, 6, n_predict=200, processing=False), [reading(8, 0, 6), reading(9, 1, 119)],
     "ambiguous:another_new_task"),
    (reading(None, 194, 6, n_predict=200, processing=False), [reading(None, 0, 6), reading(None, 1, 119)],
     "ambiguous:no_task_identity"),
    (reading(7, 194, 6, n_predict=200, processing=False), [None], "ambiguous:slots_unreadable"),
    (reading(7, 194, 6, n_predict=200, processing=True), [reading(8, 0, 6), reading(8, 1, 119)],
     "ambiguous:no_idle_slot_before_the_question"),
])
def test_the_decoding_rule_counts_only_the_current_tasks_rising_counters(cp, baseline, readings, result) -> None:
    watch = cp.DecodingWatch(baseline, cp.VALIDATED_BUILD, 120)
    observed = [watch.feed(r) for r in readings]
    assert watch.result == result, (observed, watch.record())
    assert any(observed) == (result == "decoding_observed")


def test_the_decoding_rule_holds_only_for_the_validated_build(cp) -> None:
    for build in (None, "b8933-0123456", "b8932-98dc1418f"):
        watch = cp.DecodingWatch(reading(7, 194, 6, n_predict=200, processing=False), build, 120)
        assert not any(watch.feed(r) for r in (reading(8, 0, 6), reading(8, 1, 119), reading(8, 2, 118)))
        assert watch.result == "ambiguous:unvalidated_build"
    assert cp.VALIDATED_BUILD == BUILD


def test_stale_scene_counters_through_a_long_prompt_do_not_start_the_trial(cp, monkeypatch, tmp_path) -> None:
    """The device's case (cp-20261027T100655Z): 194/6 left by the scene job, then a prompt longer than the grace."""
    result, fake = checked(cp, monkeypatch, tmp_path, long_s=4.0, prompt_s=1.6, scene_tokens=194)
    (trial,) = result["trials"]
    decoding = trial["decoding"]
    assert decoding["result"] == "decoding_observed" and decoding["task"] != decoding["baseline_task"]
    assert decoding["n_decoded_first"] == 194  # stale counters were seen, and did not count
    assert trial["submitted_after_request_began_ms"] >= 1600.0  # only after the prompt, once decoding was observed
    assert trial["verdict"] == "PASS" and fake.closed_by_client == 1
    assert result["server_build"] == BUILD


def test_a_request_ending_at_its_prompts_end_is_not_exercised_despite_stale_counters(cp, monkeypatch,
                                                                                     tmp_path) -> None:
    result, _ = checked(cp, monkeypatch, tmp_path, long_s=0.0, prompt_s=1.0, scene_tokens=194)
    (trial,) = result["trials"]
    assert trial["in_flight_at_submission"] is False
    assert trial["decoding"]["result"] == "ended_before_decoding_was_observed"
    assert trial["verdict"] == "NOT EXERCISED" and result["verdict"] == "not_exercised"


@pytest.mark.parametrize(("options", "why"), [({"build": "b9999-0123456"}, "ambiguous:unvalidated_build"),
                                              ({"slot_task_ids": False}, "ambiguous:no_task_identity")])
def test_an_unattributable_slot_is_not_exercised(cp, monkeypatch, tmp_path, options, why) -> None:
    result, _ = checked(cp, monkeypatch, tmp_path, long_s=4.0, prompt_s=0.4, **options)
    (trial,) = result["trials"]
    assert trial["decoding"]["result"] == why and trial["in_flight_at_submission"] is False
    assert trial["verdict"] == "NOT EXERCISED" and any(why in reason for reason in trial["reasons"])
    assert result["control"]["verdict"] == "AS EXPECTED" and result["verdict"] == "not_exercised"


SECRET_KEY, SECRET_VALUE = "PRIVATE_KEY_NAME", "private value words"
REPORT_OK = REPORT  # a report the parser accepts


@pytest.mark.parametrize(("text", "details"), [
    (json.dumps({**REPORT_OK, SECRET_KEY: SECRET_VALUE}), [{"loc": "<unexpected>", "type": "extra_forbidden"}]),
    (json.dumps({**REPORT_OK, "observations": [SECRET_VALUE * 5]}),
     [{"loc": "observations.0", "type": "string_too_long"}]),
    (json.dumps({**REPORT_OK, "observations": ["a"] * 6}), [{"loc": "observations", "type": "too_long"}]),
    (json.dumps({**REPORT_OK, "summary": SECRET_VALUE * 9}), [{"loc": "summary", "type": "string_too_long"}]),
    (json.dumps({**REPORT_OK, "threat": SECRET_VALUE}), [{"loc": "threat", "type": "enum"}]),
    (json.dumps({key: value for key, value in REPORT_OK.items() if key != "uncertainty"}),
     [{"loc": "uncertainty", "type": "missing"}]),
    (SECRET_VALUE, [{"loc": "<report>", "type": "json_invalid"}]),
    ("{" + " " * 5000 + "}", [{"loc": "<report>", "type": "report_too_long"}]),
])
def test_a_rejected_report_keeps_its_field_paths_and_error_categories_only(cp, text, details) -> None:
    from sentinel.scene.report import SceneReportError, parse_scene_report

    with pytest.raises(SceneReportError):
        parse_scene_report(text)  # the production parser rejects it ...
    facts = cp.report_facts(text)  # ... and the check records why, safely
    assert facts["valid"] is False and facts["rejection"] == details
    assert SECRET_KEY not in json.dumps(facts) and SECRET_VALUE not in json.dumps(facts)
    fenced = cp.report_facts("```json\n" + json.dumps(REPORT_OK) + "\n```")  # the parser's own fence rule
    assert fenced["valid"] is True and "rejection" not in fenced


def test_a_rejected_scene_report_is_explained_without_its_text(cp, monkeypatch, tmp_path) -> None:
    text = json.dumps({**REPORT_OK, "observations": [SECRET_VALUE * 5], SECRET_KEY: SECRET_VALUE})
    result, _ = checked(cp, monkeypatch, tmp_path, long_s=4.0, prompt_s=0.4, scene_text=text, scene_tokens=150)
    (trial,) = result["trials"]
    scene = trial["scene"]
    assert trial["verdict"] == "FAIL" and "scene_report_report_rejected" in trial["reasons"]
    assert (scene["finish_reason"], scene["completion_tokens"]) == ("stop", 150)
    assert {"loc": "<unexpected>", "type": "extra_forbidden"} in scene["rejection"]
    assert {"loc": "observations.0", "type": "string_too_long"} in scene["rejection"]
    (tmp_path / "chatpreempt.json").write_text(json.dumps(result))
    guard = {"status": "child_exited", "child": {"returncode": 1}, "stop": {"forced": False},
             "guard": {"trigger": None, "peak_pressure_bytes": 3_000_000_000}}
    (tmp_path / "result.json").write_text(json.dumps(guard))
    lines, code = cp.read(tmp_path)
    assert code == 1 and any("extra_forbidden" in line and '"scene_tokens": 150' in line for line in lines)
    everything = json.dumps(result) + "\n".join(lines)
    for private in (SECRET_KEY, SECRET_VALUE, ANSWER, GENERATED, TEMPLATE, "/synthetic/model.gguf"):
        assert private not in everything


def test_the_reading_before_the_question_waits_for_the_slot_to_be_released(cp, monkeypatch) -> None:
    busy, idle = reading(7, 194, 6, n_predict=200), reading(7, 194, 6, n_predict=200, processing=False)
    sequence = iter([busy, busy, idle])
    monkeypatch.setattr(cp, "slot_reading", lambda port: next(sequence))
    assert cp.idle_slot(0) == idle  # released just after the response: waited for
    monkeypatch.setattr(cp, "slot_reading", lambda port: busy)
    assert cp.idle_slot(0, timeout_s=0.05) == busy  # still processing: returned, and the watch calls it ambiguous
    assert cp.DecodingWatch(busy, cp.VALIDATED_BUILD, 120).result == "ambiguous:no_idle_slot_before_the_question"


# ---------------------------------------------------------------- session 116, part 11: three things apart, on LFM3's build


LFM3_SOURCE = Path.home() / "llama.cpp-lfm3/tools/server"
needs_pinned_build = pytest.mark.skipif(not LFM3_SOURCE.is_dir(), reason="only on the device: the pinned build's source")


@needs_pinned_build
def test_the_pinned_builds_source_says_what_the_rule_and_the_fake_follow(cp) -> None:
    context = (LFM3_SOURCE / "server-context.cpp").read_text()
    queue = (LFM3_SOURCE / "server-queue.cpp").read_text()
    common = (LFM3_SOURCE / "server-common.h").read_text()
    reset = context[context.index("task_prev = std::move(task);"):context.index("llama_set_sampler(ctx_tgt, id, nullptr);")]
    assert "stats = {};" in reset and "n_predict_max = -1;" in reset  # an idle slot reads 0 decoded and -1 remaining
    assert "return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;" in context
    assert "slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;" in context
    assert 'const auto & ptask = task ? task : task_prev;' in context  # a finished task's id stays on the slot
    assert "constexpr int HTTP_POLLING_SECONDS = 1;" in context  # a closed connection is noticed on this poll
    assert "queue_results.recv_with_timeout(id_tasks, polling_interval_seconds)" in queue
    assert "if (should_stop()) {" in queue
    assert "queue_tasks.defer(std::move(task));" in context  # a request for a busy slot waits for it
    assert "rd.wait_for_all(req.should_stop)" in context  # the non-streaming path chat and scene requests take
    assert "if (slot.task && slot.task->id == task.id_target) {" in context and "slot.release();" in context
    cancel_format = 'SRV_WRN("cancel task, id_task = %d\\n", id_task);'
    assert cancel_format in queue
    assert 'SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\\n", prompt.n_tokens(), truncated);' in context
    assert 'LOG_INF("slot %12.*s: id %2d | task %d | " fmt' in common
    assert cp.LOG_CANCEL.search("srv         stop: cancel task, id_task = 77\n")[1] == "77"
    release = cp.LOG_RELEASE.search("slot      release: id  0 | task 77 | stop processing: n_tokens = 412, truncated = 0\n")
    assert (release[1], release[2], release[3]) == ("77", "412", "0")


def test_the_lfm3_set_launches_with_its_log_level_and_cap_and_is_validated_for_its_own_build(cp) -> None:
    from sentinel.scene.models import scene_model_set

    chosen = scene_model_set("lfm3-vl")
    assert chosen.server_flags == ("--log-verbosity", "4", "--image-max-tokens", "256")
    assert chosen.llama_server == Path.home() / "llama.cpp-lfm3/build/bin/llama-server"
    assert cp.VALIDATED_LFM3_BUILD == "b11938-19cce6919" and LFM3_BUILD == cp.VALIDATED_LFM3_BUILD
    assert "b9999-0000000" not in cp.VALIDATED_BUILDS


def test_decoding_on_lfm3s_build_needs_two_rising_readings_of_a_new_task(cp) -> None:
    idle = {"slots": 1, "processing": False, "id_task": 40, "n_predict": 200, "n_decoded": 0, "n_remain": -1}
    watch = cp.DecodingWatch(idle, LFM3_BUILD, 120)
    assert watch.result is None
    assert watch.feed(idle) is False  # the idle slot's reset counters
    prompt = {**idle, "processing": True, "id_task": 41, "n_predict": 120, "n_decoded": 0, "n_remain": 120}
    assert watch.feed(prompt) is False and watch.feed(prompt) is False and watch.task == 41  # 0 and the cap: no decoding yet
    assert watch.feed({**prompt, "n_decoded": 1, "n_remain": 119}) is True
    assert watch.record()["n_decoded"] == 1 and watch.result == "decoding_observed"


def test_another_build_is_never_attributed(cp) -> None:
    idle = {"slots": 1, "processing": False, "id_task": 40}
    assert cp.DecodingWatch(idle, "b11938-19cce6910", 120).result == "ambiguous:unvalidated_build"


def full(cp, monkeypatch, tmp_path: Path, *, long_s: float = 3.0, trials: int = 1, withdrawals: int = 1,
         cancellations: int = 2, baselines: int = 1, poll: float = 0.25, prompt_s: float = 0.4, scene_s: float = 0.4,
         process: Any = None, **fake_options: Any) -> tuple[dict[str, Any], FakeLlamaServer]:
    import sentinel.scene.chat as chat

    for module in (chat, cp):
        monkeypatch.setattr(module, "CHAT_MIN_INTERVAL_S", 0.2)
        monkeypatch.setattr(module, "CHAT_START_S", 0.8)
    monkeypatch.setattr(cp, "TAIL_S", 0.3)
    port = free_port()
    fake = FakeLlamaServer(port, decode_s={LONG: long_s, SHORT: 0.1}, build=LFM3_BUILD, close_poll_s=poll, prompt_s=prompt_s,
                           scene_prompt_s=0.05, scene_s=scene_s, **fake_options)
    args = argparse.Namespace(model_set="lfm3-vl", llama_server=Path("/nonexistent/llama-server"), port=port, trials=trials,
                              withdrawals=withdrawals, cancellations=cancellations, baselines=baselines,
                              ready_timeout_s=5.0, post_load_release=False, out=tmp_path / "chatpreempt.json")
    made = FakeServerProcess(fake) if process is None else process(fake)
    result = cp.run(args, server_factory=made, environ={}, servers=lambda: [], meminfo=lambda: None,
                    picture=lambda: "a synthetic picture", encode=lambda image: b"\xff\xd8x\xff\xd9", load=lambda: 500)
    result["verdict"] = cp.overall(result)
    return result, fake


def test_the_three_things_are_exercised_and_read_apart(cp, monkeypatch, tmp_path) -> None:
    result, fake = full(cp, monkeypatch, tmp_path)
    assert result["verdict"] == "passed" and result["problems"] == []
    assert result["server_build"] == LFM3_BUILD
    assert result["server_flags"] == ["--log-verbosity", "4", "--image-max-tokens", "256"]
    assert result["baselines"][0]["verdict"] == "AS EXPECTED"
    assert result["baselines"][0]["scene"]["timings"]["predicted_n"] > 0
    assert result["control"]["verdict"] == result["control_after"]["verdict"] == "AS EXPECTED"
    # withdrawal: it never reached the server
    withdrawn = result["withdrawal"][0]
    assert withdrawn["verdict"] == "PASS" and withdrawn["question_end"]["state"] == "busy"
    assert withdrawn["request_began"] is False and withdrawn["chat_cap_task_seen"] is False
    assert withdrawn["counters"]["side_cancelled"] == 1 and withdrawn["scene_jobs"] >= 2
    # cancellation, during the prompt and during decoding: the server's own lines, and /slots left the task
    during_prompt, during_decoding = result["cancellation"]
    assert (during_prompt["phase"], during_decoding["phase"]) == ("prompt", "decoding")
    for item in result["cancellation"]:
        server = item["server"]
        assert item["verdict"] == "PASS" and item["question_end"]["reason"] == cp.ABORT_REASON
        assert item["request"]["outcome"] == "aborted" and item["counters"]["side_aborted"] == 1
        assert server["log_available"] and server["cancel_seen_ms"] is not None and server["release_seen_ms"] is not None
        assert server["still_on_task_at_the_end"] is False
        low, high = server["released_between_ms"]
        assert 0.0 <= low <= high and server["gpu_load"]["samples"] > 0
    assert during_prompt["server"]["n_decoded_at_abort"] == 0
    # generation outlasted the client's abort until the server's poll, and that shows
    assert during_decoding["server"]["n_decoded_last_seen"] >= during_decoding["server"]["n_decoded_at_abort"] >= 1
    assert during_decoding["server"]["n_decoded_last_seen"] < 120
    # resume: the scene job's task followed on the server, its decoding was seen, and its report was valid
    trial = result["trials"][0]
    assert trial["verdict"] == "PASS" and trial["scene"]["valid"] is True
    server = trial["server"]
    assert server["cancel_seen_ms"] is not None and server["next_task_first_seen_ms"] is not None
    assert server["next_task_is_chat_cap"] is False and server["next_task_decoding_seen_ms"] is not None
    assert trial["scene"]["timings"]["predicted_n"] > 0
    assert fake.completed["chat"] == 2  # the two controls alone: every other question was withdrawn or ended
    text = json.dumps(result)
    for private in (ANSWER, GENERATED, TEMPLATE):
        assert private not in text


def test_the_reading_keeps_the_three_apart_and_says_what_timing_means(cp, monkeypatch, tmp_path) -> None:
    result, _ = full(cp, monkeypatch, tmp_path)
    (tmp_path / "chatpreempt.json").write_text(json.dumps(result))
    (tmp_path / "result.json").write_text(json.dumps({
        "status": "child_exited", "child": {"returncode": 0}, "stop": {"forced": False},
        "guard": {"trigger": None, "peak_pressure_bytes": 4_400_000_000}}))
    lines, code = cp.read(tmp_path)
    text = "\n".join(lines)
    assert code == 0
    for heading in ("--- waiting-question withdrawal: passed", "--- in-flight cancellation: passed",
                    "--- resume after preemption: passed"):
        assert heading in text
    assert "AS EXPECTED baseline-1" in text and "control after everything" in text
    assert all(limit in text for limit in cp.TIMING_LIMITS)
    assert "marks no requirement satisfied" in text and "does not show that GPU work stopped" in text
    assert "cancel-1 (during its prompt)" in text and "cancel-2 (during its decoding)" in text


def test_a_server_whose_log_is_not_read_cannot_pass_a_cancellation(cp, monkeypatch, tmp_path) -> None:
    class NoLog:
        def __init__(self, fake) -> None:
            self.inner = FakeServerProcess(fake)

        def __call__(self, *args, **kwargs):
            self.inner(*args, **kwargs)
            return self

        def __getattr__(self, name):
            if name == "attach_log":
                raise AttributeError(name)
            return getattr(self.inner, name)

    result, _ = full(cp, monkeypatch, tmp_path, process=NoLog, withdrawals=0, cancellations=1)
    assert result["cancellation"][0]["verdict"] == "NOT EXERCISED"
    assert "a client abort alone proves nothing" in result["cancellation"][0]["reasons"][0]
    assert result["verdict"] == "not_exercised"


def test_a_short_answer_that_ends_before_the_abort_is_not_exercised(cp, monkeypatch, tmp_path) -> None:
    result, _ = full(cp, monkeypatch, tmp_path, long_s=0.0, prompt_s=0.0, withdrawals=0, cancellations=1)
    assert result["cancellation"][0]["verdict"] == "NOT EXERCISED" and result["verdict"] != "passed"


CANCEL = {"abort_returned": True, "decoding": {"result": "decoding_observed"}, "request": {"outcome": "aborted"},
          "question_end": {"state": "cancelled", "reason": "the check ended this question"},
          "server": {"task": 7, "log_available": True, "cancel_seen_ms": 410.0, "release_seen_ms": 440.0,
                     "still_on_task_at_the_end": False}}


@pytest.mark.parametrize("change, verdict, words", [
    ({}, "PASS", "reported cancelling"),
    ({"abort_returned": False}, "NOT EXERCISED", "ended before the abort"),
    ({"question_end": {"state": "answered", "reason": None}}, "NOT EXERCISED", "ended before the abort"),
    ({"decoding": {"result": "ambiguous:unvalidated_build"}}, "NOT EXERCISED", "could not be attributed"),
    ({"server": {**CANCEL["server"], "log_available": False}}, "NOT EXERCISED", "proves nothing"),
    ({"server": {**CANCEL["server"], "cancel_seen_ms": None}}, "NOT EXERCISED", "no cancellation of this task"),
    ({"server": {**CANCEL["server"], "task": None}}, "NOT EXERCISED", "before the task was seen"),
    ({"server": {**CANCEL["server"], "release_seen_ms": None}}, "FAIL", "no release"),
    ({"server": {**CANCEL["server"], "still_on_task_at_the_end": True}}, "FAIL", "still on the task"),
    ({"question_end": {"state": "failed", "reason": None}}, "FAIL", "question_failed"),
    ({"question_end": {"state": "cancelled", "reason": "memory"}}, "FAIL", "another_reason"),
    ({"request": {"outcome": "returned"}}, "FAIL", "request_returned"),
])
def test_how_a_cancellation_is_judged(cp, change, verdict, words) -> None:
    got, reasons = cp.judge_cancellation({**CANCEL, **change})
    assert got == verdict and any(words in reason for reason in reasons)


WITHDRAWN = {"request_began": False, "chat_cap_task_seen": False, "counters": {"side_cancelled": 1},
             "question_end": {"state": "busy", "reason": "x"},
             "scenes": [{"outcome": "completed", "valid": True, "start_delay_ms": 4000.0}]}


@pytest.mark.parametrize("change, verdict, words", [
    ({}, "PASS", "no request began"),
    ({"request_began": True, "question_end": {"state": "answered"}}, "NOT EXERCISED", "a gap let"),
    ({"request_began": True}, "FAIL", "after its request began"),
    ({"chat_cap_task_seen": True}, "FAIL", "chat's cap"),
    ({"counters": {}}, "FAIL", "no_withdrawal_counted"),
    ({"question_end": {"state": "failed"}}, "FAIL", "question_failed"),
    ({"scenes": [{"outcome": "error", "valid": True, "start_delay_ms": 1.0}]}, "FAIL", "scene_job_error"),
    ({"scenes": [{"outcome": "completed", "valid": False, "problem": "report_rejected", "start_delay_ms": 1.0}]},
     "FAIL", "report_rejected"),
    ({"scenes": [{"outcome": "completed", "valid": True, "start_delay_ms": None}]}, "FAIL", "never_started"),
])
def test_how_a_withdrawal_is_judged(cp, change, verdict, words) -> None:
    got, reasons = cp.judge_withdrawal({**WITHDRAWN, **change})
    assert got == verdict and any(words in reason for reason in reasons)


@pytest.mark.parametrize("sections, verdict", [
    ({"withdrawal": ["PASS"], "cancellation": ["PASS"], "trials": ["PASS"]}, "passed"),
    ({"withdrawal": ["PASS"], "cancellation": ["NOT EXERCISED"], "trials": ["PASS"]}, "not_exercised"),
    ({"withdrawal": ["NOT EXERCISED"], "cancellation": ["PASS"], "trials": ["PASS"]}, "not_exercised"),
    ({"withdrawal": ["PASS"], "cancellation": ["PASS"], "trials": ["NOT EXERCISED"]}, "not_exercised"),
    ({"withdrawal": ["PASS"], "cancellation": ["PASS", "FAIL"], "trials": ["PASS"]}, "failed"),
    ({"withdrawal": ["FAIL"], "cancellation": ["PASS"], "trials": ["PASS"]}, "failed"),
    ({"trials": ["PASS"]}, "passed"),
])
def test_every_section_asked_for_must_pass(cp, sections, verdict) -> None:
    result = {"problems": [], "control": {"verdict": "AS EXPECTED"}, "control_after": {"verdict": "AS EXPECTED"},
              **{name: [{"verdict": v} for v in values] for name, values in sections.items()}}
    assert cp.overall(result) == verdict


def test_the_second_control_decides_too(cp) -> None:
    result = {"problems": [], "control": {"verdict": "AS EXPECTED"}, "control_after": {"verdict": "NOT AS EXPECTED"},
              "trials": [{"verdict": "PASS"}]}
    assert cp.overall(result) == "failed"
    result["control_after"] = {"verdict": "AS EXPECTED"}
    result["baselines"] = [{"verdict": "NOT AS EXPECTED"}]
    assert cp.overall(result) == "failed"


def timed(sent_ms: float, back_ms: float, *, processing: bool, task: int, decoded: int = 0, predict: int = 120) -> dict:
    return {"sent": int(sent_ms * 1e6), "received": int(back_ms * 1e6), "slots": 1, "processing": processing, "id_task": task,
            "n_predict": predict, "n_decoded": decoded, "n_remain": predict - decoded}


def test_server_timing_gives_an_interval_and_never_an_instant(cp) -> None:
    readings = [timed(900, 910, processing=True, task=7, decoded=20),
                timed(950, 990, processing=True, task=7, decoded=22),
                timed(1400, 1430, processing=True, task=7, decoded=35),
                timed(1500, 1900, processing=True, task=8, decoded=0, predict=200),  # a slow reading, on the next task
                timed(1950, 1960, processing=True, task=8, decoded=1, predict=200),
                timed(2000, 2010, processing=True, task=8, decoded=2, predict=200)]
    loads = [(int(900e6), 600), (int(1000e6), 600), (int(1600e6), 0), (int(1700e6), 400)]
    events = [{"kind": "cancel", "task": 7, "at_ns": int(1300e6)}, {"kind": "release", "task": 7, "n_tokens": 400, "at_ns": int(1440e6)},
              {"kind": "cancel", "task": 3, "at_ns": int(100e6)}]
    got = cp.server_timing(readings, loads, events, task=7, abort_ns=int(1000e6), attached=True)
    assert got["n_decoded_at_abort"] == 22 and got["n_decoded_last_seen"] == 35 and got["tokens_seen_after_abort"] == 13
    assert got["released_between_ms"] == [400.0, 900.0]  # the last reading on it was sent at +400; the next returned at +900
    assert got["cancel_seen_ms"] == 300.0 and got["release_seen_ms"] == 440.0 and got["released_n_tokens"] == 400
    assert got["next_task_first_seen_ms"] == 900.0 and got["next_task_is_chat_cap"] is False
    assert got["next_task_decoding_seen_ms"] == 960.0
    assert got["reading_ms"]["max"] == 400.0 and got["still_on_task_at_the_end"] is False
    assert got["gpu_load"] == {"samples": 4, "before_abort_p50": 600, "first_zero_after_abort_ms": 600.0, "max_after_that": 400}


def test_server_timing_says_when_the_slot_never_left(cp) -> None:
    got = cp.server_timing([timed(10, 20, processing=True, task=7, decoded=3)], [], [], task=7, abort_ns=int(5e6), attached=True)
    assert got["still_on_task_at_the_end"] is True and "released_between_ms" not in got and got["gpu_load"] is None


def test_the_log_watch_keeps_only_its_two_patterns(cp, tmp_path) -> None:
    watch = cp.ServerLogWatch()
    private = "a prompt's private words, an image description"
    watch.feed(f"slot update_slots: id 0 | task 9 | prompt: {private}\n")
    watch.feed("srv         stop: cancel task, id_task = 9\n")
    watch.feed("slot      release: id  0 | task 9 | stop processing: n_tokens = 412, truncated = 0\n")
    events = watch.events()
    assert [(e["kind"], e["task"]) for e in events] == [("cancel", 9), ("release", 9)] and events[1]["n_tokens"] == 412
    assert private not in json.dumps(events) and watch.attached is False


def test_the_log_watch_reads_a_real_processs_output_and_leaves_it_for_the_runtime(cp) -> None:
    watch = cp.ServerLogWatch()
    process = watch.popen([sys.executable, "-c", "print('srv stop: cancel task, id_task = 12'); print('other')"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = list(iter(lambda: process.stdout.readline(65_536), b""))
    process.wait()
    process.stdout.close()
    assert watch.attached and [e["task"] for e in watch.events()] == [12]
    assert lines == [b"srv stop: cancel task, id_task = 12\n", b"other\n"]  # what LlamaServerProcess reads, unchanged


def test_the_real_server_is_given_the_watchs_popen(cp, monkeypatch, tmp_path) -> None:
    given = {}

    class Real:
        def __init__(self, binary, model, mmproj, port, *, extra_flags=(), popen=None) -> None:
            given.update(extra_flags=extra_flags, popen=popen)

        def start(self, timeout):
            raise KeyboardInterrupt

        def stop(self, grace):
            return True

        def status(self):
            return ServerStatus(ServerState.STOPPED, None, 0, None, False)

    monkeypatch.setattr(cp, "LlamaServerProcess", Real)
    args = argparse.Namespace(model_set="lfm3-vl", llama_server=Path("/nonexistent"), port=free_port(), trials=1, withdrawals=0,
                              cancellations=0, baselines=0, ready_timeout_s=1.0, post_load_release=False, out=tmp_path / "x.json")
    with pytest.raises(KeyboardInterrupt):
        cp.run(args, environ={}, servers=lambda: [], meminfo=lambda: None)
    assert given["extra_flags"] == ("--log-verbosity", "4", "--image-max-tokens", "256")
    assert given["popen"] is not None and given["popen"].__self__.__class__.__name__ == "ServerLogWatch"
