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
import sys
from pathlib import Path
from typing import Any

import pytest
from fake_llama_server import ANSWER, BUILD, GENERATED, REPORT, TEMPLATE, FakeLlamaServer, FakeServerProcess, \
    free_port

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
                              trials=trials, ready_timeout_s=5.0, post_load_release=False,
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

    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent"), port=free_port(), trials=1,
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
    args = argparse.Namespace(model_set="lfm2.5-vl", llama_server=Path("/nonexistent"), port=port, trials=1,
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
