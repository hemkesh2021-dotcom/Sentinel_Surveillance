"""An operator block's verdict on a guarded run (session 81; benchmarks/runner/guard_verdict.py), against a verbatim
copy of the device run trk-20261016T100149Z's sealed tv/result.json (SHA-256 6a7af72a6aaca52a4988ada6b4005b4f6c2736
00a54dd6c8ee1842d9e2d7efaa; every text in it a fixed label or a UTC time) and the cases the investigation proposed:
complete, a guard stop after T0, a runtime that exited on its own, readiness unmet, a refusal before launch, a
missing cue, a forced stop, a cleanup failure, and a missing, unreadable or malformed result."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
SEALED = Path(__file__).resolve().parents[1] / "support/fixtures/trk-20261016T100149Z-tv-result.json"
CUES = ("t0", "k1", "a", "e2", "k2", "stop")  # the diagnostic session's cues, as its t-V passed them


@pytest.fixture
def verdict(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("guard_verdict")


@pytest.fixture
def sealed() -> dict:
    return json.loads(SEALED.read_text())


def complete(sealed: dict) -> dict:
    """The sealed run as it would have been had it completed: the duration's stop, no trigger, every cue printed."""
    result = copy.deepcopy(sealed)
    result.update(status="duration_stop", stop={"requested_by": "duration_elapsed", "requested_at_s": 670.3,
                                                "exited_after_stop_s": 15.0, "forced": False})
    result["guard"]["trigger"] = None
    result["cues"] = [{"name": c["name"], "at_s": c["at_s"], "printed_after_launch_s": 70.2 + c["at_s"],
                       "printed_mono_s": 7776.9 + c["at_s"]} for c in result["parameters"]["cues"]]
    return result


def lines(verdict, result, guard_exit=0, cues=CUES, require_ready="tracking_diagnostic") -> dict[str, str]:
    return dict(verdict.verdict(result, guard_exit, cues, require_ready))


def reasons(judged: dict[str, str]) -> list[str]:
    assert judged["verdict"].startswith("incomplete:"), judged
    return judged["verdict"][len("incomplete:"):].split(";")


def test_the_fixture_is_the_sealed_result(sealed) -> None:
    assert sealed["status"] == "guard_stop" and sealed["readiness"]["met"] is True
    assert sealed["child"]["returncode"] == 0 and sealed["cleanup_clear"] is True
    assert sealed["guard"]["trigger"]["stop"] == "free_or_available_stop"


def test_the_sealed_guard_stop_is_incomplete_whatever_exit_the_block_kept(verdict, sealed) -> None:
    """The device run: readiness met, the runtime's clean exit 0 and cleanup clear, yet the guard stopped it 18.7 s
    after T0 and printed only the first cue. t-V wrote TV_exit=0 for it; the verdict is incomplete with the guard's
    exit 1 and stays incomplete if a block lost that exit and passed 0."""
    judged = lines(verdict, sealed, guard_exit=1)
    assert judged["status"] == "guard_stop" and judged["returncode"] == "0" and judged["readiness"] == "met"
    assert judged["cues_printed"] == "1/6" and judged["cleanup"] == "clear"
    assert reasons(judged) == ["guard_exit_1", "status_guard_stop", "memory_guard_stop", "cue_not_printed:k1",
                               "cue_not_printed:a", "cue_not_printed:e2", "cue_not_printed:k2", "cue_not_printed:stop"]
    assert reasons(lines(verdict, sealed, guard_exit=0))[0] == "status_guard_stop"


def test_only_a_completed_ready_clean_run_with_every_cue_printed_is_complete(verdict, sealed) -> None:
    judged = lines(verdict, complete(sealed))
    assert judged == {"status": "duration_stop", "returncode": "0", "readiness": "met", "cues_printed": "6/6",
                      "cleanup": "clear", "verdict": "complete"}
    assert reasons(lines(verdict, complete(sealed), guard_exit=1)) == ["guard_exit_1"]  # the guard's exit counts too


@pytest.mark.parametrize("case, expected", [
    ("child_exited", ["status_child_exited", "cue_not_printed:stop"]),  # the guard's exit 0 covers it; never complete
    ("runtime_exit", ["runtime_exit_1"]),
    ("forced", ["status_forced_stop", "forced_stop"]),
    ("cleanup", ["status_cleanup_failed", "cleanup_not_clear"]),
    ("cleanup_flag", ["cleanup_not_clear"]),
    ("missing_cue", ["cue_not_printed:stop"]),
    ("other_cues", ["cues_not_as_expected", "cue_not_printed:k1"]),
    ("readiness_unmet", ["status_readiness_refused",
                         "readiness_unmet:detector_problem:diagnostic_unsupported (callback_order)"]),
    ("readiness_missing", ["readiness_missing"]),
    ("refused", ["status_refused:initial_headroom_refused", "runtime_exit_none", "guard_record_missing",
                 "stop_record_missing", "readiness_missing", "cue_not_printed:t0", "cleanup_not_clear"]),
    ("status_text", ["status_unrecognized"]),
])
def test_every_other_outcome_is_incomplete_with_its_reasons(verdict, sealed, case, expected) -> None:
    result = complete(sealed)
    if case == "child_exited":
        result.update(status="child_exited", stop={"requested_by": None, "requested_at_s": None,
                                                   "exited_after_stop_s": None, "forced": False})
        result["cues"][-1].update(printed_after_launch_s=None, printed_mono_s=None)
    elif case == "runtime_exit":
        result["child"]["returncode"] = 1
    elif case == "forced":
        result.update(status="forced_stop")
        result["stop"]["forced"] = True
    elif case == "cleanup":
        result.update(status="cleanup_failed", cleanup_clear=False)
        result["leftovers"]["llama_server_left"] = 1
    elif case == "cleanup_flag":
        result["cleanup_clear"] = False  # an orphan the guard stopped: the leftovers clear, the flag not
    elif case == "missing_cue":
        result["cues"][-1].update(printed_after_launch_s=None, printed_mono_s=None)
    elif case == "other_cues":
        result["parameters"]["cues"] = result["parameters"]["cues"][:1] + result["parameters"]["cues"][2:]
        result["cues"] = [c for c in result["cues"] if c["name"] != "k1"]
    elif case == "readiness_unmet":
        result.update(status="readiness_refused", readiness={
            "required": "tracking_diagnostic", "met": False, "starting_after_s": 55.0,
            "reason": "detector_problem:diagnostic_unsupported (callback_order)"})
        result["stop"]["requested_by"] = "readiness_refused"
        result["cues"] = [{**c, "printed_after_launch_s": None, "printed_mono_s": None} for c in result["cues"]]
        expected = expected + [f"cue_not_printed:{name}" for name in CUES]
    elif case == "readiness_missing":
        result.pop("readiness")  # a guard that required nothing
    elif case == "refused":  # refused before launch: the guard's result holds the parameters and the sample only
        result = {"schema_version": 1, "mode": "step5_guard", "status": "refused:initial_headroom_refused",
                  "parameters": result["parameters"], "limits": result["limits"],
                  "preflight_sample": result["baseline"]}
        expected = expected[:6] + [f"cue_not_printed:{name}" for name in CUES[1:]] + expected[6:]
    elif case == "status_text":
        result["status"] = "Jane stopped it"  # a text that is not one of the guard's labels is never printed
    judged = lines(verdict, result)
    assert reasons(judged) == expected
    assert "Jane" not in "".join(judged.values())


@pytest.mark.parametrize("content, label", [(None, "missing"), ("{not json", "unreadable"), ("[]", "malformed"),
                                             ('{"mode": "other"}', "malformed"), (b"\xff\xfe", "unreadable")])
def test_a_missing_unreadable_or_malformed_result_is_incomplete(verdict, tmp_path, content, label) -> None:
    path = tmp_path / "result.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content)
    judged = dict(verdict.verdict(verdict.load(path), 0, CUES, "tracking_diagnostic"))
    assert judged["status"] == label and judged["verdict"] == f"incomplete:result_{label}"
    assert judged["readiness"] == "missing" and judged["cues_printed"] == "0/6" and judged["cleanup"] == "not_clear"


def test_the_cli_prints_prefixed_lines_and_exits_by_the_verdict(verdict, sealed, tmp_path, capsys) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(sealed))
    argv = [str(path), "--guard-exit", "1", "--cues", ",".join(CUES), "--require-ready", "tracking_diagnostic",
            "--prefix", "TV_"]
    assert verdict.main(argv) == 1
    printed = capsys.readouterr().out.splitlines()
    assert printed[:5] == ["TV_status=guard_stop", "TV_returncode=0", "TV_readiness=met", "TV_cues_printed=1/6",
                           "TV_cleanup=clear"]
    assert printed[5].startswith("TV_verdict=incomplete:guard_exit_1;status_guard_stop;memory_guard_stop;")
    path.write_text(json.dumps(complete(sealed)))
    assert verdict.main([str(path), "--guard-exit", "0", "--cues", ",".join(CUES), "--require-ready",
                         "tracking_diagnostic"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "verdict=complete"
    assert verdict.main([str(path), "--guard-exit", "0", "--cues", ",".join(CUES)]) == 0  # nothing required
    assert "readiness=not_required" in capsys.readouterr().out
    for bad in (["--prefix", "tv_"], ["--cues", "t0,t0"], ["--cues", "T0"], ["--require-ready", "detector"]):
        args = [str(path), "--guard-exit", "0", "--cues", "t0", *bad]
        with pytest.raises(SystemExit) as stopped:
            verdict.main(args)
        assert stopped.value.code == 2
    capsys.readouterr()


def test_its_vocabularies_are_the_guards(verdict, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(RUNNERS))
    guard, evidence = importlib.import_module("step5_guard"), importlib.import_module("face_evidence")
    assert verdict.READINESS == guard.READINESS
    statuses = next(option for option in evidence.GUARD_STATUS.options if hasattr(option, "values")).values
    assert set(verdict.STATUSES) == set(statuses)
