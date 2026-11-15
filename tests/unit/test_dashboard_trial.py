"""Session 109: the one-hour live-dashboard trial (validation only, before F2 validates; the maintainer's authorization):
the reading alpha_check.py trial, in five separately reported parts, and the guard's one-hour bound.

The run is made by the real runtime (DemoRuntime with a threaded face worker in its validation run, the threaded scene
worker and chat in its validation run, and the live view with a viewer pulling pictures as the dashboard's player does;
chat_runs.build_run) over a simulated hour at one frame a second. The operator's own questions leave no record but the
status page's counters, so the chat load and the viewer's log are removed, as the trial has neither. The account comes
through face_account's own prompts (format 11) and the privacy counts from the real face_counts.sh. Each part is shown
failing on its own evidence while the others still pass; the integration reading is unchanged.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from chat_runs import FCI_CAPTURES, build_run
from face_runs import bash_counts

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
TRIAL_CAPTURES = ((60.0, "c1"), (600.0, "c2"), (1200.0, "c3"), (1800.0, "c4"), (2400.0, "c5"), (3000.0, "c6"),
                  (3585.0, "end"))  # the trial package's --status-at offsets
ACCOUNT = {"stayed_seated": "no", "worked_as_usual": "yes", "looked_at_camera_on_purpose": "no",
           "other_faces_in_view": "no", "notes": ""}  # the operator left and came back
PARTS = ("execution", "live view", "chat", "recognition behaviour", "memory")


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield type("Mods", (), {"check": importlib.import_module("alpha_check"),
                                "load": importlib.import_module("alpha_load"),
                                "guard": importlib.import_module("step5_guard"),
                                "account": importlib.import_module("face_account"),
                                "evidence": importlib.import_module("face_evidence")})
    finally:
        sys.path.remove(str(RUNNERS))


def trial_run(mods, root: Path, *, viewer: bool = True, **kwargs) -> Path:
    """An hour as the trial records it: no chat load and no viewer log (the operator's own browser and questions)."""
    run = build_run(mods, root, face=True, live=True, viewer=viewer, duration=3600.0, fps=1.0,
                    captures_at=TRIAL_CAPTURES, **kwargs)
    for name in ("chat-load.jsonl", "viewer.jsonl"):
        (run / name).unlink()
    return run


def finish(mods, run: Path) -> Path:
    """As the package's blocks do after the launch: the provenance moved in, the account (hv-D), the counts (hv-E)."""
    (run / "provenance.txt").write_text("pin_commit=0000000000000000000000000000000000000000\n")
    answers = iter([ACCOUNT[f.key] for f in mods.account.FIELDS_V11] + ["SAVE"])
    assert mods.account.enter(run, ask=lambda prompt: next(answers), say=lambda text: None,
                              account_format_number=11) == 0
    for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                           ("face_counts_f2", "secret-counts-identity.txt")):
        done = bash_counts(function, str(run))
        assert done.returncode == 0, done.stderr
        (run / name).write_text(done.stdout)
    return run


@pytest.fixture(scope="module")
def made(mods, tmp_path_factory) -> Path:
    return finish(mods, trial_run(mods, tmp_path_factory.mktemp("trial")))


def copy_of(made: Path, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(made, target)
    return target


def text(part) -> str:
    return "\n".join(part.lines)


def line_for(body: str, start: str) -> str:
    return next(line for line in body.splitlines() if line.split(" ", 1)[1].startswith(start))


def oks(parts) -> dict[str, bool]:
    return {name: part.ok for name, part in parts.items()}


def test_an_hour_from_the_real_runtime_passes_each_part_and_says_what_it_does_not_show(mods, made, capsys) -> None:
    parts = mods.check.read_integration(made, trial=True)
    assert tuple(parts) == PARTS and all(part.ok for part in parts.values()), {k: text(v) for k, v in parts.items()}
    assert not any("MISSING" in text(part) or "FAIL" in text(part) for part in parts.values())
    execution = text(parts["execution"])
    assert line_for(execution, "the run: 3600 s from readiness, the face_chat readiness").startswith("PASS")
    assert line_for(execution, "chat offered as a validation run (not admitted); the live view on").startswith("PASS")
    assert line_for(execution, "every capture:").count('"live_view": true') == len(TRIAL_CAPTURES)
    assert line_for(execution, "identity records:").startswith("PASS")  # an hour's records within the runtime's cap
    assert line_for(execution, "monitoring at each capture").startswith("DESCRIPTIVE")
    live = text(parts["live view"])
    assert line_for(live, "the live view sent pictures to a viewer").startswith("PASS")
    assert line_for(live, "the live view's encoder stopped with the runtime").startswith("PASS")
    chat = text(parts["chat"])
    assert line_for(chat, "your questions: at least one accepted and answered").startswith("PASS")
    assert line_for(chat, "chat-induced scene delay").startswith("PASS")
    recognition = text(parts["recognition behaviour"])
    assert [line.split(" ", 1)[0] for line in parts["recognition behaviour"].lines].count("PASS") == 1  # only that
    assert mods.check.RECOGNITION_SCOPE in recognition and '"stayed_seated": "no"' in recognition
    assert line_for(recognition, "identity states at each capture").startswith("DESCRIPTIVE")
    assert mods.check.LAUNCH_FLOOR_NOTE in text(parts["memory"])
    assert mods.check.main(["trial", str(made)]) == 0
    out = capsys.readouterr().out
    for name in PARTS:
        assert f"=== {name}\n" in out and f"trial {name}: passed" in out
    assert out.rstrip().endswith(f"alpha dashboard trial: passed ({mods.check.TRIAL_SCOPE})")
    assert "not F2 and not the combined run" in mods.check.TRIAL_SCOPE and "no answer's correctness" in \
        mods.check.TRIAL_SCOPE


def test_the_trials_files_fit_the_privacy_profiles(mods, made) -> None:
    counts = (made / "secret-counts-identity.txt").read_text().splitlines()
    assert {line.split()[0] for line in counts} >= {f"status-{name}.json" for _, name in TRIAL_CAPTURES}
    assert counts and all(line.endswith("=0") for line in counts)  # c4-c6 declared; nothing nonconforming
    assert not (made / "chat-load.jsonl").exists() and not (made / "viewer.jsonl").exists()


def test_the_trial_and_the_integration_readings_never_stand_for_each_other(mods, made, tmp_path) -> None:
    # the integration reading of the hour: its 480 s, no live view and its chat load are not this run
    integration = mods.check.read_integration(made)
    assert oks(integration) == {"execution": False, "chat completion": False, "recognition behaviour": True,
                                "memory": True}
    # the trial reading of an eight-minute integration run without the live view
    fci = build_run(mods, tmp_path, face=True, live=False, duration=480.0, captures_at=FCI_CAPTURES)
    (fci / "chat-load.jsonl").unlink()
    parts = mods.check.read_integration(finish(mods, fci), trial=True)
    assert oks(parts)["execution"] is False and oks(parts)["live view"] is False
    assert line_for(text(parts["execution"]), "the run: 3600 s").startswith("FAIL")


def test_without_a_viewer_the_live_view_part_alone_is_not_passed(mods, tmp_path) -> None:
    parts = mods.check.read_integration(finish(mods, trial_run(mods, tmp_path, viewer=False)), trial=True)
    assert oks(parts) == {"execution": True, "live view": False, "chat": True, "recognition behaviour": True,
                          "memory": True}
    assert line_for(text(parts["live view"]), "the live view sent pictures").startswith("FAIL")


def test_without_a_question_the_chat_part_alone_is_not_passed(mods, made, tmp_path) -> None:
    run = copy_of(made, tmp_path)
    for _, name in TRIAL_CAPTURES:  # as the counters read when nobody asked: none counted, no scene job waited
        path = run / f"status-{name}.json"
        status = json.loads(path.read_text())
        status["components"]["chat"]["counters"] = {}
        status["components"]["chat"]["scene_waits_behind_chat"] = {
            "n": 0, "median_ms": None, "max_ms": None, "last_ms": None, "max_ever_ms": None, "grace_ms": 1500,
            "run_ms": {"behind_side": {"n": 0, "median_ms": None, "max_ms": None},
                       "others": {"n": 0, "median_ms": None, "max_ms": None}}}
        path.write_text(json.dumps(status))
    parts = mods.check.read_integration(run, trial=True)
    assert oks(parts) == {"execution": True, "live view": True, "chat": False, "recognition behaviour": True,
                          "memory": True}
    assert line_for(text(parts["chat"]), "your questions").startswith("MISSING")


def test_memory_over_the_hour_is_judged_alone_against_the_existing_limits(mods, tmp_path) -> None:
    run = finish(mods, trial_run(mods, tmp_path, pressure=lambda s: 5_100_000_000 if s > 2000 else 4_100_000_000))
    parts = mods.check.read_integration(run, trial=True)
    assert oks(parts) == {"execution": True, "live view": True, "chat": True, "recognition behaviour": True,
                          "memory": False}
    assert line_for(text(parts["memory"]), "memory:").startswith("FAIL")


def test_the_guard_takes_the_hour_and_its_offsets(mods) -> None:
    assert mods.guard.MAX_DURATION_S == mods.check.TRIAL_DURATION_S == 3600.0
    assert mods.guard.MAX_READY_TIMEOUT_S == 900.0
    for at, name in TRIAL_CAPTURES:
        assert mods.guard._status_at(f"{at:g}:{name}") == (at, name)
    assert set(mods.evidence.STATUS_NAMES) >= {name for _, name in TRIAL_CAPTURES}


def test_a_capture_without_the_live_view_fails_execution_alone(mods, made, tmp_path) -> None:
    run = copy_of(made, tmp_path)
    path = run / "status-c4.json"
    status = json.loads(path.read_text())
    status["preview"]["enabled"] = False  # as a capture would read with the live view off half-way through the hour
    path.write_text(json.dumps(status))
    parts = mods.check.read_integration(run, trial=True)
    assert oks(parts) == {"execution": False, "live view": True, "chat": True, "recognition behaviour": True,
                          "memory": True}
    assert line_for(text(parts["execution"]), "every capture:").startswith("FAIL")


def test_questions_asked_but_none_answered_fail_the_chat_part_alone(mods, made, tmp_path) -> None:
    run = copy_of(made, tmp_path)
    path = run / "status-end.json"  # the last capture: every question was cancelled, none answered
    status = json.loads(path.read_text())
    status["components"]["chat"]["counters"] = {"accepted": 3, "cancelled": 3}
    path.write_text(json.dumps(status))
    parts = mods.check.read_integration(run, trial=True)
    assert oks(parts) == {"execution": True, "live view": True, "chat": False, "recognition behaviour": True,
                          "memory": True}
    assert line_for(text(parts["chat"]), "your questions").startswith('FAIL')
    assert '{"accepted": 3, "answered": 0}' in line_for(text(parts["chat"]), "your questions")
