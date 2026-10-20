"""Session 103: the validation-only face + chat integration run (fci): the guard's face_chat readiness and the reading
alpha_check.py integration, in four separately reported parts.

The run directories are made by the real runtime (DemoRuntime with a threaded face worker in its validation run, the
threaded scene worker and chat in its validation run; test_alpha_check.build_run's face mode), the account through
face_account's own prompts (format 11) and the privacy counts by the real face_counts.sh. Each part is shown failing
on its own evidence while the others still pass: an integration pass never stands for recognition accuracy or F2.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from chat_runs import FACE_BOX, FCI_CAPTURES, build_run
from face_runs import bash_counts, real_face_startup

from sentinel.scene.chat import CHAT_ADMISSION, VALIDATION_NOTE

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
ACCOUNT = {"stayed_seated": "yes", "worked_as_usual": "yes", "looked_at_camera_on_purpose": "no",
           "other_faces_in_view": "no", "notes": ""}
PARTS = ("execution", "chat completion", "recognition behaviour", "memory")


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield type("Mods", (), {"check": importlib.import_module("alpha_check"),
                                "load": importlib.import_module("alpha_load"),
                                "guard": importlib.import_module("step5_guard"),
                                "verdict": importlib.import_module("guard_verdict"),
                                "account": importlib.import_module("face_account"),
                                "evidence": importlib.import_module("face_evidence")})
    finally:
        sys.path.remove(str(RUNNERS))


def finish(mods, run: Path, account: dict[str, str] | None = ACCOUNT, account_format: int = 11) -> Path:
    """As the package's blocks do after the launch: the provenance moved in, the account (iv-D), the counts (iv-E)."""
    (run / "provenance.txt").write_text("pin_commit=0000000000000000000000000000000000000000\n")
    if account is not None:
        fields = {10: mods.account.FIELDS_V10, 11: mods.account.FIELDS_V11}[account_format]
        answers = iter([account[f.key] for f in fields] + ["SAVE"])
        assert mods.account.enter(run, ask=lambda prompt: next(answers), say=lambda text: None,
                                  account_format_number=account_format) == 0
    for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                           ("face_counts_f2", "secret-counts-identity.txt")):
        done = bash_counts(function, str(run))
        assert done.returncode == 0, done.stderr
        (run / name).write_text(done.stdout)
    return run


def fci_run(mods, root: Path, **kwargs) -> Path:
    return build_run(mods, root, face=True, live=False, duration=480.0, captures_at=FCI_CAPTURES, **kwargs)


@pytest.fixture(scope="module")
def made(mods, tmp_path_factory) -> Path:
    return finish(mods, fci_run(mods, tmp_path_factory.mktemp("fci")))


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


# ---------------------------------------------------------------- the guard's face_chat readiness


def test_face_chat_readiness_needs_the_plain_face_run_the_ready_scene_server_and_chat_as_a_validation_run(
        mods, tmp_path, accepted_scene) -> None:
    problem = mods.guard.readiness_problem
    (tmp_path / "plain").mkdir()
    (tmp_path / "landmarks").mkdir()
    plain = real_face_startup(tmp_path / "plain", accepted_scene)  # assemble()'s own face fields
    marked = real_face_startup(tmp_path / "landmarks", accepted_scene, landmarks=True)
    assert plain["face"] == mods.guard.FACE_CHAT_FACE and plain["face_problem"] is None
    chat = {"state": "available", "reason": f"ask about the picture the camera is showing now ({VALIDATION_NOTE})"}
    assert chat == mods.guard.CHAT_VALIDATION_AVAILABLE and chat["reason"] == mods.check.VALIDATION_REASON
    assert CHAT_ADMISSION.admitted is False and CHAT_ADMISSION.validation_allowed is True
    ready = {"detector_problem": None, "scene_problem": None, "face": plain["face"], "face_problem": None,
             "scene_server": {"state": "ready", "problem": None, "layers": "17/17", "vision_on_gpu": True},
             "chat": chat}
    assert problem("face_chat", ready) is None
    for change, reason in [
        ({"detector_problem": "memfree_below_minimum"}, "detector_problem:memfree_below_minimum"),
        ({"face_problem": "load_failed"}, "face_problem:load_failed"),
        ({"scene_problem": "server_not_ready"}, "scene_problem:server_not_ready"),
        ({"face": None}, "face_missing"),
        ({"face": marked["face"]}, "face_not_the_plain_validation_run"),  # a diagnostic marker: not this run
        ({"face": {"validation_run": False, "identities_enrolled": 1}}, "face_not_the_plain_validation_run"),
        ({"face": {"validation_run": True, "identities_enrolled": 0}}, "face_not_the_plain_validation_run"),
        ({"scene_server": {**ready["scene_server"], "layers": "12/17"}}, "scene_server_not_ready"),
        ({"scene_server": {**ready["scene_server"], "vision_on_gpu": False}}, "scene_server_not_ready"),
        ({"scene_server": None}, "scene_server_not_ready"),
        ({"chat": None}, "chat_missing"),
        ({"chat": {"state": "paused", "reason": "memory pressure"}}, "chat_not_available_as_validation_run:paused"),
        ({"chat": {"state": "available", "reason": "ask about the picture the camera is showing now"}},
         "chat_not_available_as_validation_run:available"),  # admitted chat is not this run
        ({"chat": {"state": "unavailable", "reason": CHAT_ADMISSION.reason}},
         "chat_not_available_as_validation_run:unavailable"),
    ]:
        assert problem("face_chat", {**ready, **change}) == reason, change
        assert mods.verdict.REASON.fullmatch(reason)  # guard_verdict prints it as a fixed label
    assert problem("face_chat", None) == "startup_missing"
    assert mods.guard.READINESS == mods.verdict.READINESS == ("tracking_diagnostic", "landmark_diagnostic", "face_chat")
    assert mods.evidence.READINESS.values == frozenset(mods.guard.READINESS)  # result.json's readiness fits the profile


# ---------------------------------------------------------------- the reading, part by part


def test_an_integration_run_from_the_real_runtime_passes_each_part_and_says_what_it_does_not_show(
        mods, made, capsys) -> None:
    parts = mods.check.read_integration(made)
    assert tuple(parts) == PARTS and all(part.ok for part in parts.values()), {k: text(v) for k, v in parts.items()}
    assert not any("MISSING" in text(part) or "FAIL" in text(part) for part in parts.values())
    recognition = text(parts["recognition behaviour"])
    assert line_for(recognition, "face executed alongside").startswith("PASS")
    assert [line.split(" ", 1)[0] for line in parts["recognition behaviour"].lines].count("PASS") == 1  # only that
    assert mods.check.RECOGNITION_SCOPE in recognition and "never F2" in recognition
    assert '"unresolved->known": 1' in line_for(recognition, "identity transitions")  # described, never judged
    assert mods.check.LAUNCH_FLOOR_NOTE in text(parts["memory"])
    chat = text(parts["chat completion"])
    assert '"questions_cancelled_for_a_scene_job": 1' in line_for(chat, "cancellation verified")
    assert line_for(chat, "real answers").startswith("PASS")
    assert mods.check.main(["integration", str(made)]) == 0
    out = capsys.readouterr().out
    for name in PARTS:
        assert f"=== {name}\n" in out and f"integration {name}: passed" in out
    assert "integration recognition behaviour: passed (only that face executed alongside is judged; no recognition " \
           "accuracy)" in out
    assert out.rstrip().endswith(f"alpha integration check: passed ({mods.check.INTEGRATION_SCOPE})")
    assert "not F2 and not the combined run" in mods.check.INTEGRATION_SCOPE


def test_the_records_and_files_fit_the_privacy_profiles_and_hold_no_answer_text(mods, made) -> None:
    assert "A chair stands by the window." not in "".join(p.read_text() for p in made.iterdir() if p.is_file())
    counts = (made / "secret-counts-identity.txt").read_text()
    assert counts and all(line.endswith("=0") for line in counts.splitlines())  # the real face_privacy, every file


def test_recognition_behaviour_never_judges_what_was_recognized(mods, tmp_path) -> None:
    # The face matches nobody enrolled: no KNOWN at all. Face still executed alongside, so every part passes.
    from sentinel.identity.association import FaceObservation
    from sentinel.identity.legacy_deepface import FaceRun

    def nobody(frame, image):
        return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=(0.0, 0.1, 0.99, 0.0)),),
                       1, 0)

    run = finish(mods, fci_run(mods, tmp_path, analyze=nobody))
    parts = mods.check.read_integration(run)
    assert all(oks(parts).values()), {k: text(v) for k, v in parts.items()}
    body = text(parts["recognition behaviour"])
    assert "->known" not in line_for(body, "identity transitions") and '"votes_unknown"' in body


def test_a_face_worker_that_fails_fails_recognition_behaviour_alone(mods, tmp_path) -> None:
    def failing(frame, image):
        raise RuntimeError("synthetic analysis failure")

    run = finish(mods, fci_run(mods, tmp_path, analyze=failing))
    parts = mods.check.read_integration(run)
    assert oks(parts) == {"execution": True, "chat completion": True, "recognition behaviour": False, "memory": True}
    assert line_for(text(parts["recognition behaviour"]), "face executed alongside").startswith("FAIL")


@pytest.mark.parametrize(("pressure", "label"), [(lambda s: 5_100_000_000 if s > 200 else 4_000_000_000, "FAIL"),
                                                 (lambda s: 4_500_000_000, "PASS")])
def test_memory_is_judged_alone_against_the_existing_limits(mods, tmp_path, pressure, label) -> None:
    parts = mods.check.read_integration(finish(mods, fci_run(mods, tmp_path, pressure=pressure)))
    assert line_for(text(parts["memory"]), "memory:").startswith(label)
    assert oks(parts) == {"execution": True, "chat completion": True, "recognition behaviour": True,
                          "memory": label == "PASS"}


def test_no_chat_load_fails_chat_completion_alone(mods, made, tmp_path) -> None:
    run = copy_of(made, tmp_path)
    (run / "chat-load.jsonl").unlink()
    parts = mods.check.read_integration(run)
    assert oks(parts) == {"execution": True, "chat completion": False, "recognition behaviour": True, "memory": True}
    assert line_for(text(parts["chat completion"]), "real answers").startswith("MISSING")


@pytest.mark.parametrize("change", ["unmet", "duration", "live_view", "face_marker", "account_format", "no_account"])
def test_execution_needs_the_decided_run_its_readiness_and_the_account(mods, made, tmp_path, change) -> None:
    run = copy_of(made, tmp_path)
    result = json.loads((run / "result.json").read_text())
    if change == "unmet":
        result["readiness"] = {"required": "face_chat", "met": False, "reason": "chat_missing", "starting_after_s": 1}
    elif change == "duration":
        result["parameters"]["duration_s"] = 600.0
    (run / "result.json").write_text(json.dumps(result))
    if change in ("live_view", "face_marker"):
        lines = (run / "run.jsonl").read_text().splitlines()
        first = json.loads(lines[0])
        if change == "live_view":
            first["startup"]["live_preview"] = {"max_fps": 2}
        else:
            first["startup"]["face"] = {**first["startup"]["face"], "landmark_diagnostic": {"version": 1}}
        (run / "run.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
        result["run_output"] = mods.guard.read_run_output(run / "run.jsonl")
        (run / "result.json").write_text(json.dumps(result))
    if change in ("account_format", "no_account"):
        (run / "account.txt").unlink()
        (run / "provenance.txt").write_text("pin_commit=0000000000000000000000000000000000000000\n")
        if change == "account_format":
            finish(mods, run, account_format=10)
    parts = mods.check.read_integration(run)
    assert oks(parts)["execution"] is False and oks(parts)["chat completion"] is True and oks(parts)["memory"] is True
    assert mods.check.main(["integration", str(run)]) == 1
