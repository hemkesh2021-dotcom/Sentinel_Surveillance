"""Session 36: the 3TC account helper (benchmarks/runner/operator_account.py).

Scripted answers stand in for the operator. The helper must write 3TC-5's
format so 3TC-6 reads it unchanged, accept unknown, reject malformed answers,
show incomplete and out-of-order items before saving, invent nothing and never
change a locked account.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
# 3TC-5's template as written in docs/IMPLEMENTATION_STATUS.md (session 33).
TEMPLATE = """\
stopwatch_started_at_prompt=unknown   # yes / no: you started it when NOW appeared
out_of_view_1=unknown   # stopwatch m:ss from which you were sure the camera could not see you, after 0:00
started_walking_in=unknown   # m:ss when you started walking back in (schedule 3:00)
in_view_from=unknown   # m:ss from which you were sure the camera could see you
in_view_until=unknown   # m:ss when you stood up to leave (schedule 4:30); you were surely visible until then
out_of_view_2=unknown   # m:ss from which you were sure the camera could not see you again
returned_at=unknown   # m:ss when you came back into view (schedule 7:00)
anyone_else_entered_view=unknown   # anyone, or a pet, as far as you know: yes / no / unknown
finished_shown_on_return=unknown   # FINISHED was on the screen when you came back: yes / no / unknown
person_like_objects_or_reflections_noticed=unknown   # yes / no / unknown; describe in notes
changes_noticed_on_return=unknown   # lighting, objects, screens or camera: yes / no / unknown; describe in notes
followed_schedule=unknown   # yes / roughly / no
notes=
"""
COMPLETE = ["yes", "0:41", "3:00", "3:04", "4:30", "4:36", "7:01", "no", "yes", "no", "no", "yes", "seated at the Mac"]
TIMES = ["out_of_view_1", "started_walking_in", "in_view_from", "in_view_until", "out_of_view_2", "returned_at"]


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("operator_account")


def make_run(path: Path) -> Path:
    """A finished, unlocked run directory; b.json holds a count that must never be shown."""
    path.mkdir(exist_ok=True)
    (path / "provenance.txt").write_text("utc=2026-10-05T19:03:20Z\nexit_b=0\nprobe_procs_after=0\n")
    (path / "marks.txt").write_text("departure_prompt_utc=2026-10-05T19:36:29.478Z\nend_utc=2026-10-05T19:42:36.793Z\n")
    (path / "b.json").write_text('{"persons": {"frames_with_persons": 4242}}')
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return make_run(tmp_path / "run")


def kv_3tc6(text: str) -> dict[str, str]:
    """3TC-6's reader, copied from the block."""
    out = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def ordered_3tc6(obs: dict[str, str]) -> bool:
    """3TC-6's order test on m:ss readings."""
    def sw(k: str) -> int | None:
        try:
            mm, ss = obs.get(k, "").split(":")
            mm, ss = int(mm), int(ss)
        except ValueError:
            return None
        return mm * 60 + ss if mm >= 0 and 0 <= ss < 60 else None
    tm = {k: sw(k) for k in TIMES}
    return all(v is not None for v in tm.values()) and tm["out_of_view_1"] < tm["started_walking_in"] <= \
        tm["in_view_from"] < tm["in_view_until"] <= tm["out_of_view_2"] < tm["returned_at"]


class Operator:
    """Answers prompts from a script; running out of answers is an EOF, as at a closed terminal."""

    def __init__(self, answers: Iterable[str], on_ask: Callable[[str], None] | None = None) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.said: list[str] = []
        self.on_ask = on_ask

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.on_ask is not None:
            self.on_ask(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def say(self, text: str) -> None:
        self.said.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.said)


def test_a_complete_account_is_written_in_the_template_format(account: Any, run_dir: Path) -> None:
    op = Operator([*COMPLETE, "SAVE"])
    assert account.enter(run_dir, op.ask, op.say) == 0
    written = (run_dir / "observations.txt").read_text()
    values = kv_3tc6(written)
    assert list(values) == list(kv_3tc6(TEMPLATE))  # the same keys in the same order
    assert list(values.values()) == COMPLETE and ordered_3tc6(values)
    # A comment header, then each template line with the typed value in place of its default.
    expected = [line.replace("=unknown", "=" + value, 1) if not line.startswith("notes=") else "notes=" + value
                for line, value in zip(TEMPLATE.splitlines(), COMPLETE)]
    assert written.splitlines()[0].startswith("#") and written.splitlines()[1:] == expected
    assert "Incomplete (unknown): none" in op.text
    assert "Complete and in order: yes" in op.text
    assert "4242" not in op.text  # no detection count is read or shown


def test_unknown_is_accepted_and_incomplete_items_are_shown_before_saving(account: Any, run_dir: Path) -> None:
    answers = ["unknown", "u", "3:00", "UNKNOWN", "4:30", "unknown", "7:00", "unknown", "yes", "u", "no", "roughly", ""]
    saved_when: list[int] = []
    op = Operator([*answers, "SAVE"], on_ask=lambda p: saved_when.append(len(op.said)) if "SAVE" in p else None)
    assert account.enter(run_dir, op.ask, op.say) == 0
    values = kv_3tc6((run_dir / "observations.txt").read_text())
    assert [values[k] for k in TIMES] == ["unknown", "3:00", "unknown", "4:30", "unknown", "7:00"]
    assert values["stopwatch_started_at_prompt"] == "unknown" and values["notes"] == ""
    shown_before_save = "\n".join(op.said[:saved_when[0]])
    assert ("Incomplete (unknown): stopwatch_started_at_prompt, out_of_view_1, in_view_from, out_of_view_2, "
            "anyone_else_entered_view, person_like_objects_or_reflections_noticed") in shown_before_save
    assert "Complete and in order: no" in shown_before_save
    assert not ordered_3tc6(values)  # 3TC-6 will not treat this account as complete

    # All unknown: nothing is filled in, not even the schedule times.
    blank = make_run(run_dir.parent / "blank")
    op = Operator([*["unknown"] * 12, "", "SAVE"])
    assert account.enter(blank, op.ask, op.say) == 0
    values = kv_3tc6((blank / "observations.txt").read_text())
    assert set(list(values.values())[:12]) == {"unknown"} and values["notes"] == ""
    assert "Complete and in order: no" in op.text


def test_malformed_answers_are_rejected_and_asked_again(account: Any, run_dir: Path) -> None:
    answers = ["maybe", "Y", "yes",                     # yes/no item
               "3:5", "3:60", "yes", "", "0:41",          # a time: one-digit and 60 seconds, a word, nothing
               "3:00", "3:04", "4:30", "4:36", "7:01",
               "no", "yes", "no", "no",
               "mostly", "yes",                           # yes / roughly / no
               "lamp # on", "lamp on"]                    # # would cut the note short in 3TC-6
    op = Operator([*answers, "SAVE"])
    assert account.enter(run_dir, op.ask, op.say) == 0
    values = kv_3tc6((run_dir / "observations.txt").read_text())
    assert values["stopwatch_started_at_prompt"] == "yes" and values["out_of_view_1"] == "0:41"
    assert values["followed_schedule"] == "yes" and values["notes"] == "lamp on"
    rejections = [line for line in op.said if "not accepted" in line]
    assert len(rejections) == 8
    assert sum("two-digit seconds" in line for line in rejections) == 4
    assert any("# is not allowed" in line for line in rejections)


def test_times_out_of_order_are_shown_and_can_be_re_entered(account: Any, run_dir: Path) -> None:
    answers = list(COMPLETE)
    answers[2] = "0:41"  # started_walking_in equal to out_of_view_1: it must be strictly after
    answers[4] = "3:02"  # in_view_until before in_view_from (3:04)
    op = Operator([*answers, "5", "4:30", "3", "3:00", "SAVE"])
    assert account.enter(run_dir, op.ask, op.say) == 0
    first_review = op.text.split("Your account (not saved yet):")[1]
    assert "in_view_until 3:02 should be after in_view_from 3:04" in first_review
    assert "started_walking_in 0:41 should be after out_of_view_1 0:41" in first_review
    assert "Complete and in order: no" in first_review
    values = kv_3tc6((run_dir / "observations.txt").read_text())
    assert values["in_view_until"] == "4:30" and values["started_walking_in"] == "3:00" and ordered_3tc6(values)
    # Equal readings are fine where 3TC-6 allows them (walking in -> in view; standing up -> out of view).
    same = list(COMPLETE)
    same[2:6] = ["3:04", "3:04", "4:30", "4:30"]
    assert account.order_problems(dict(zip(TIMES, same[1:7]))) == []


def test_quitting_or_a_closed_terminal_writes_nothing(account: Any, run_dir: Path) -> None:
    for answers in ([*COMPLETE, "QUIT"], COMPLETE[:5]):  # QUIT at the review; EOF part-way
        op = Operator(answers)
        assert account.enter(run_dir, op.ask, op.say) == 1 and "Nothing written" in op.text
    assert sorted(p.name for p in run_dir.iterdir()) == ["b.json", "marks.txt", "provenance.txt"]


def test_a_locked_account_is_never_changed(account: Any, run_dir: Path) -> None:
    path = run_dir / "observations.txt"
    path.write_text(TEMPLATE)
    os.utime(path, (1_000_000_000, 1_000_000_000))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with (run_dir / "provenance.txt").open("a") as handle:
        handle.write(f"observations_sha256={digest}\n")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    op = Operator(["REPLACE", *COMPLETE, "SAVE"])
    assert account.enter(run_dir, op.ask, op.say) == 2
    assert op.prompts == [] and "locked" in op.text
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before

    op = Operator([])
    assert account.check(run_dir, op.say) == 0
    assert "Locked: yes; the file matches its recorded SHA-256" in op.text
    assert "Incomplete (unknown): stopwatch_started_at_prompt, out_of_view_1" in op.text
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    path.write_text(TEMPLATE + "late=edit\n")
    op = Operator([])
    account.check(run_dir, op.say)
    assert "THE FILE DIFFERS FROM ITS RECORDED SHA-256" in op.text


def test_a_lock_or_an_edit_made_during_entry_stops_the_save(account: Any, run_dir: Path) -> None:
    def lock(prompt: str) -> None:
        if "SAVE" in prompt:
            with (run_dir / "provenance.txt").open("a") as handle:
                handle.write("observations_sha256=" + "0" * 64 + "\n")

    op = Operator([*COMPLETE, "SAVE"], on_ask=lock)
    assert account.enter(run_dir, op.ask, op.say) == 2
    assert not (run_dir / "observations.txt").exists() and "locked meanwhile" in op.text

    other = make_run(run_dir.parent / "other")
    path = other / "observations.txt"
    path.write_text(TEMPLATE)

    def edit(prompt: str) -> None:
        if "SAVE" in prompt:
            path.write_text(TEMPLATE.replace("notes=", "notes=edited elsewhere"))

    op = Operator(["REPLACE", *COMPLETE, "SAVE"], on_ask=edit)
    assert account.enter(other, op.ask, op.say) == 2
    assert "changed meanwhile" in op.text and path.read_text().endswith("notes=edited elsewhere\n")


def test_an_unlocked_account_is_replaced_only_on_request(account: Any, run_dir: Path) -> None:
    path = run_dir / "observations.txt"
    path.write_text(TEMPLATE)
    op = Operator([""])
    assert account.enter(run_dir, op.ask, op.say) == 1
    assert path.read_text() == TEMPLATE and "An unlocked account already exists" in op.text
    op = Operator(["REPLACE", *COMPLETE, "SAVE"])
    assert account.enter(run_dir, op.ask, op.say) == 0
    assert list(kv_3tc6(path.read_text()).values()) == COMPLETE


def test_entry_waits_for_the_end_of_the_run(account: Any, run_dir: Path, tmp_path: Path) -> None:
    (run_dir / "marks.txt").write_text("departure_prompt_utc=2026-10-05T19:36:29.478Z\n")
    op = Operator(COMPLETE)
    assert account.enter(run_dir, op.ask, op.say) == 2 and "has not finished" in op.text and op.prompts == []
    (run_dir / "provenance.txt").unlink()
    (run_dir / "marks.txt").write_text("end_utc=2026-10-05T19:42:36.793Z\n")
    op = Operator(COMPLETE)
    assert account.enter(run_dir, op.ask, op.say) == 2 and "provenance.txt is missing" in op.text
    assert not (run_dir / "observations.txt").exists()


def test_check_names_values_3tc6_would_not_accept(account: Any, run_dir: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    # Run 1's account as written (session 34): a word for a time and free text for a yes / no item.
    text = (TEMPLATE.replace("stopwatch_started_at_prompt=unknown", "stopwatch_started_at_prompt=yes")
            .replace("out_of_view_1=unknown", "out_of_view_1=yes")
            .replace("changes_noticed_on_return=unknown", "changes_noticed_on_return=lightin, night time")
            .replace("anyone_else_entered_view=unknown", "anyone_else_entered_view=No"))
    (run_dir / "observations.txt").write_text(text)
    assert account.main(["check", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "out_of_view_1=yes (write m:ss" in out
    assert "changes_noticed_on_return=lightin, night time (answer yes / no / unknown)" in out
    assert "anyone_else_entered_view=No (3TC-6 reads the exact text; the accepted form is no)" in out
    assert "Locked: no" in out and "Complete and in order: no" in out
    assert account.main(["check", str(make_run(run_dir.parent / "empty"))]) == 1
