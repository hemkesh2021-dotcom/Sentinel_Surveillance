"""`track probe` and `run` keep stdout for their JSON; library output goes to stderr (checklist step 3, session 21).

On the device, the model load wrote Ultralytics' and TensorRT's diagnostics to
stdout, ahead of the track probe's JSON. These tests run the real CLI in a
child process (cli_stdout_child.py) with fake devices whose load and first
track call write to stdout through print, a pre-bound logging handler,
os.write and libc's buffered stdio. No camera, GPU or model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sentinel.cli import _json_stdout

CHILD = Path(__file__).with_name("cli_stdout_child.py")
SRC = Path(__file__).resolve().parents[2] / "src"
WAYS = ("print", "logging", "oswrite", "native")


def run_child(scenario: str, work: Path) -> tuple[subprocess.CompletedProcess[str], str, dict]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTINEL_")}
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run([sys.executable, str(CHILD), scenario, str(work)], capture_output=True, text=True,
                          timeout=60, env=env, check=False)
    lines = done.stdout.splitlines(keepends=True)
    # The child's own output around the command: flushed before the switch, written after the restore.
    assert lines[:2] == ["BEFORE-print\n", "BEFORE-native\n"], done.stdout + done.stderr
    assert sorted(line.strip() for line in lines[-3:]) == ["AFTER-native", "AFTER-oswrite", "AFTER-print"]
    checks = json.loads((work / "checks.json").read_text())
    return done, "".join(lines[2:-3]), checks


def assert_library_output_on_stderr(done: subprocess.CompletedProcess[str], *tags: str) -> None:
    for tag in tags:
        for way in WAYS:
            marker = f"NOISE-{way}-{tag}"
            assert marker in done.stderr and marker not in done.stdout, marker


def assert_restored(checks: dict) -> None:
    # Descriptor 1 is the original stdout again and the duplicate is closed.
    assert checks["after"] == checks["before"]


def test_track_probe_stdout_is_one_json_document_with_library_output_on_stderr(tmp_path: Path) -> None:
    done, payload, checks = run_child("track", tmp_path)
    assert done.returncode == 0, done.stderr
    summary = json.loads(payload)
    assert summary["probe"] == "track" and summary["status"] == "frames_received"
    assert summary["tracking"]["processed"] > 0 and summary["tracking"]["failed"] == 0
    assert_library_output_on_stderr(done, "load", "track")
    assert_restored(checks) and checks["raised"] is None


def test_run_stdout_is_json_lines_only_with_library_output_on_stderr(tmp_path: Path) -> None:
    done, payload, checks = run_child("run", tmp_path)
    assert done.returncode == 0, done.stderr
    lines = [json.loads(line) for line in payload.splitlines()]
    assert lines[0]["run"] == "starting" and lines[0]["startup"]["detector_problem"] is None
    assert lines[-1]["run"] == "stopped" and lines[-1]["shutdown"]["all_stopped"] is True
    assert_library_output_on_stderr(done, "load", "track")
    assert_restored(checks) and checks["raised"] is None


def test_a_memfree_refusal_is_still_json_on_stdout_with_exit_code_1(tmp_path: Path) -> None:
    done, payload, checks = run_child("track_memfree", tmp_path)
    assert done.returncode == 1
    refusal = json.loads(payload)
    assert refusal["status"] == "refused" and refusal["reason"] == "memfree_below_minimum"
    assert "NOISE" not in done.stdout + done.stderr  # refused before the model load
    assert_restored(checks)


def test_a_load_failure_keeps_its_stderr_label_and_exit_code_1(tmp_path: Path) -> None:
    done, payload, checks = run_child("track_load_failed", tmp_path)
    assert done.returncode == 1 and payload == ""
    assert "track probe: load_failed (RuntimeError)\n" in done.stderr
    assert_library_output_on_stderr(done, "load")
    assert_restored(checks)


@pytest.mark.parametrize(
    ("scenario", "code", "raised"),
    [("track_raises", 3, "RuntimeError"), ("track_interrupted", 3, "KeyboardInterrupt"), ("run_interrupted", 130, None)],
)
def test_an_exception_or_interrupt_restores_stdout_and_closes_the_duplicate(
    tmp_path: Path, scenario: str, code: int, raised: str | None
) -> None:
    done, payload, checks = run_child(scenario, tmp_path)
    assert done.returncode == code and checks["raised"] == raised
    assert payload == ""
    if scenario == "run_interrupted":
        assert "run: interrupted during startup\n" in done.stderr
    assert_library_output_on_stderr(done, "load")
    assert_restored(checks)


def test_the_preview_keeps_stdout_one_json_document_and_never_prints_its_token(tmp_path: Path) -> None:
    done, payload, checks = run_child("track_preview", tmp_path)
    assert done.returncode == 0, done.stderr
    summary = json.loads(payload)
    preview, viewer = summary["preview"], checks["preview"]
    assert viewer["status"] == "200" and len(viewer["parts"]) == 2
    assert all(part.startswith("frame-") for part in viewer["parts"])
    assert preview["frames_sent"] >= 2 and preview["closed"] and preview["ended_by"] == "duration"
    assert summary["track_boxes"]["tracks"] and viewer["port_closed_after"]
    assert viewer["token"] not in done.stdout and viewer["token"] not in done.stderr
    assert_library_output_on_stderr(done, "load", "track")
    assert_restored(checks) and checks["raised"] is None


def test_ctrl_c_ends_a_preview_run_early_with_its_json_summary_and_the_port_closed(tmp_path: Path) -> None:
    done, payload, checks = run_child("track_previewsigint", tmp_path)
    assert done.returncode == 130, done.stderr
    summary = json.loads(payload)
    assert summary["preview"]["ended_by"] == "signal" and summary["preview"]["closed"]
    assert summary["seconds"] < 10 and summary["worker"]["stopped"] and summary["tracking"]["processed"] > 0
    assert checks["preview"]["port_closed_after"] and checks["raised"] is None
    assert checks["preview"]["token"] not in done.stdout + done.stderr
    assert_restored(checks)


def test_without_a_real_stdout_descriptor_nothing_is_redirected(capsys: pytest.CaptureFixture[str]) -> None:
    before = os.fstat(1)
    with _json_stdout() as out:
        assert out is sys.stdout
        print("in-process")
    after = os.fstat(1)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert capsys.readouterr().out == "in-process\n"
