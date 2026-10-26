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
from fake_llama_server import ANSWER, GENERATED, FakeLlamaServer, FakeServerProcess, free_port

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
            scene_text: str | None = None, prompt_s: float = 0.0) -> tuple[dict[str, Any], FakeLlamaServer]:
    import sentinel.scene.chat as chat

    monkeypatch.setattr(chat, "CHAT_MIN_INTERVAL_S", 0.3)  # chat's one question every 4 s, shortened for the test
    monkeypatch.setattr(cp, "CHAT_MIN_INTERVAL_S", 0.3)
    port = free_port()
    fake = FakeLlamaServer(port, decode_s={LONG: long_s, SHORT: short_s}, scene_text=scene_text, prompt_s=prompt_s)
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
