"""Session 106: the grounding check G in the validation-only integration run (the maintainer's approval of session
105's proposal, judged by the maintainer, with three clarifications).

- alpha_load asks G1-G4 once each, at the first regular slot at or after A0 + offset (A0: when the first accepted
  question's POST returned), keeps their answers' text only in a new 0600 G file, records every other outcome without
  retrying, and writes the slots it never reached when it ends (a real SIGTERM included);
- grounding.py writes the staging (refusing a launch unless S1 and S2 are yes) and is the operator's local judging
  interface: it shows answers only on a terminal, only after the account and S5 are saved, and saves categories and
  metadata only; the labels follow the clarifications (uncertainty apart; G3's printed words against an unlabelled
  bottle contradicted, a brand guessed from its look unverifiable);
- the evidence profiles, the real privacy counts and the reading: no answer's text anywhere but the G file, the counts
  covering it, the reading's G section descriptive and never part of a verdict.

Session 108 (format 2 of the staging and the judgments, after session 107's clarifications): the person question
excludes brands and objects; G3 records whether a brand is named and whether it is said to be printed apart, each
allowing unsure; S3 asks about a printed brand name; uncertainty and the operator's unsure are never consistent with
the staging; format 1's records (session 107's sealed files, verbatim) read exactly as sealed.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from chat_runs import FCI_CAPTURES, build_run
from face_runs import bash_counts

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
SEALED_G = Path(__file__).resolve().parents[1] / "support" / "fixtures" / "fcg-20261020T154003Z-g"
ACCOUNT = {"stayed_seated": "yes", "worked_as_usual": "yes", "looked_at_camera_on_purpose": "no",
           "other_faces_in_view": "no", "notes": ""}
G_ANSWERS = {  # synthetic, never a real answer
    "G1": {"answer": "Yes, a water bottle stands on the desk."},
    "G2": {"answer": "No cosmetic products are visible; a notebook lies near it."},
    "G3": {"answer": "The bottle appears to say ZEPHYRA SPRING on its label, and the label continues with a line of",
           "cut": True},
    "G4": {"answer": "One picture cannot show what happened a minute earlier."},
}


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(RUNNERS))
    try:
        yield SimpleNamespace(load=importlib.import_module("alpha_load"), g=importlib.import_module("grounding"),
                              check=importlib.import_module("alpha_check"),
                              guard=importlib.import_module("step5_guard"),
                              account=importlib.import_module("face_account"),
                              evidence=importlib.import_module("face_evidence"))
    finally:
        sys.path.remove(str(RUNNERS))


# ---------------------------------------------------------------- alpha_load: the G slots


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def scripted(clock: Clock, *, refuse_first: int = 1, post_s: float = 0.25, outcome=None,
             gone_after: float | None = None):
    """A page on the fake clock: the first ``refuse_first`` questions refused as starting; each POST takes ``post_s``;
    ``outcome(question)`` gives a question's final document; after ``gone_after`` the page is gone."""

    class Page:
        def __init__(self, port: int) -> None:
            self.posts: list[tuple[float, dict]] = []
            self.jobs: dict[str, str] = {}
            Page.last = self

        def refresh_token(self) -> bool:
            return True

        def json(self, method: str, path: str, document: dict | None = None) -> tuple[int, dict]:
            if gone_after is not None and clock.now >= gone_after:
                raise OSError("gone")
            if method == "GET":
                question = self.jobs[path.rsplit("/", 1)[1]]
                final = (outcome or (lambda q: None))(question) or {"state": "answered", "answer": "Synthetic."}
                return 200, {"frame": {"stream_epoch": 1, "frame_seq": 7, "ingest_utc": "2026-10-20T11:10:03Z",
                                       "age_ms_when_asked": 40}, "age_ms_when_answered": 2200, "historical": True,
                             "answer_ms": 2160, "cut_short": False, **final}
            self.posts.append((clock.now, document))
            clock.now += post_s
            if len(self.posts) <= refuse_first:
                return 503, {"state": "unavailable", "reason": "the runtime is starting"}
            job = f"{len(self.posts):016x}"
            self.jobs[job] = document["question"]
            return 202, {"job": job, "state": "sending"}

    return Page


def run_load(mods, monkeypatch, tmp_path, page, clock, *, grounding: bool = True, every_s: float = 6.0):
    monkeypatch.setattr(mods.load, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    monkeypatch.setattr(mods.load, "Page", page)
    monkeypatch.setattr(mods.load, "_wait_for", lambda p, wait_s: True)
    monkeypatch.setattr(mods.load, "GONE_S", 0.0)
    out, g_file = tmp_path / "chat-load.jsonl", tmp_path / "grounding.jsonl"
    code = mods.load.chat(out, 18090, every_s=every_s, wait_s=5.0, grounding=g_file if grounding else None)
    load = [json.loads(line) for line in out.read_text().splitlines()]
    g = [json.loads(line) for line in g_file.read_text().splitlines()] if grounding else None
    return code, load, g, g_file


def test_g_is_asked_once_each_at_the_first_slot_at_or_after_a0_plus_its_offset(mods, monkeypatch, tmp_path) -> None:
    clock = Clock()
    page = scripted(clock, gone_after=100.0 + 420.0)
    code, load, g, g_file = run_load(mods, monkeypatch, tmp_path, page, clock)
    assert code == 0
    posts = page.last.posts
    # A0 is when the first accepted POST returned: the second question, asked at 106.0, returned at 106.25
    a0 = 106.25
    assert {r["a0_mono_s"] for r in g} == {a0}
    asked = {r["g"]: r["asked_mono_s"] for r in g}
    for gid, offset, question in mods.load.GROUNDING:
        expected = min(t for t, _ in posts if t >= a0 + offset)  # the first regular slot at or after it
        assert asked[gid] == expected and asked[gid] - a0 >= offset
        assert [d for t, d in posts if d["question"] == question] == [{"question": question, "history": []}]
    # the cadence and every other slot unchanged: one POST every 6 s; the other questions alternate as before
    times = [t for t, _ in posts]
    assert all(abs(b - a - 6.0) < 1e-9 for a, b in zip(times, times[1:]))
    kinds = ["short" if i % 2 == 0 else "long" for i in range(len(posts))]
    for (t, d), kind in zip(posts, kinds):
        if d["question"] not in {q for _, _, q in mods.load.GROUNDING}:
            assert d["question"] == mods.load.QUESTIONS[kind]
    assert {r["g"]: r["replaced_kind"] for r in g} == {r["g"]: kinds[times.index(r["asked_mono_s"])] for r in g}
    # OUT.jsonl: the G slots as kind grounding with their ID and no text; G.jsonl: 0600, the text, exactly its fields
    g_rows = [r for r in load if r.get("kind") == "grounding"]
    assert [r["g"] for r in g_rows] == ["G1", "G2", "G3", "G4"] and all("answer" not in r for r in load)
    assert "Synthetic." not in (tmp_path / "chat-load.jsonl").read_text()
    assert (g_file.stat().st_mode & 0o777) == 0o600
    assert [r["answer"] for r in g] == ["Synthetic."] * 4
    assert set(g[0]) == {"g", "question", "offset_s", "a0_mono_s", "asked_mono_s", "asked_after_a0_s", "replaced_kind",
                         "http_status", "state", "reason", "stream_epoch", "frame_seq", "ingest_utc",
                         "age_ms_when_asked", "age_ms_when_answered", "historical", "answer_ms", "answer_chars",
                         "cut_short", "answer"}


def test_every_outcome_is_kept_once_and_nothing_is_retried_or_extended(mods, monkeypatch, tmp_path) -> None:
    clock = Clock()
    questions = {gid: q for gid, _, q in mods.load.GROUNDING}
    outcomes = {questions["G1"]: {"state": "cancelled", "reason": "a scene job is waiting: monitoring comes first",
                                  "answer_ms": 1610},
                questions["G2"]: {"state": "answered", "answer": "A" * 590, "cut_short": True},
                questions["G3"]: {"state": "busy", "reason": "the scene model stayed busy; monitoring comes first"}}
    page = scripted(clock, outcome=outcomes.get, gone_after=100.0 + 300.0)
    # the page goes away before G4's slot: G4 is written not asked, and asking does not go on past the run
    _, load, g, _ = run_load(mods, monkeypatch, tmp_path, page, clock)
    by = {r["g"]: r for r in g}
    assert (by["G1"]["state"], by["G1"]["reason"], by["G1"]["answer"]) == (
        "cancelled", "a scene job is waiting: monitoring comes first", None)
    assert (by["G2"]["state"], by["G2"]["cut_short"], len(by["G2"]["answer"])) == ("answered", True, 590)
    assert (by["G3"]["state"], by["G3"]["answer"]) == ("busy", None)
    assert (by["G4"]["state"], by["G4"]["reason"], by["G4"]["asked_mono_s"]) == (
        "not_asked", "the page went away before this slot", None)
    asked = [d["question"] for _, d in page.last.posts]
    assert all(asked.count(q) == (0 if gid == "G4" else 1) for gid, q in questions.items())  # never retried
    assert max(t for t, _ in page.last.posts) < 100.0 + 300.0  # no asking beyond the run
    assert len(g) == 4


def test_a_g_slot_without_a_response_is_no_response_and_never_asked_again(mods, monkeypatch, tmp_path) -> None:
    clock = Clock()
    base = scripted(clock, gone_after=100.0 + 200.0)
    question = mods.load.GROUNDING[0][2]

    class Silent(base):
        def json(self, method, path, document=None):
            if method == "POST" and document["question"] == question:
                self.posts.append((clock.now, document))
                raise OSError("no response")
            return super().json(method, path, document)

    _, _, g, _ = run_load(mods, monkeypatch, tmp_path, Silent, clock)
    by = {r["g"]: r for r in g}
    assert (by["G1"]["state"], by["G1"]["reason"]) == ("no_response", "the page did not answer")
    assert [d["question"] for _, d in Silent.last.posts].count(question) == 1


def test_without_grounding_the_load_is_unchanged_and_asks_no_g_question(mods, monkeypatch, tmp_path) -> None:
    clock = Clock()
    page = scripted(clock, gone_after=100.0 + 420.0)
    _, load, _, g_file = run_load(mods, monkeypatch, tmp_path, page, clock, grounding=False)
    assert not g_file.exists()
    assert {d["question"] for _, d in page.last.posts} == set(mods.load.QUESTIONS.values())
    assert {r["kind"] for r in load if "kind" in r} == {"short", "long"} and all("g" not in r for r in load)


def test_an_existing_g_file_is_refused_and_nothing_is_asked(mods, tmp_path, capsys) -> None:
    g_file = tmp_path / "grounding.jsonl"
    g_file.write_text("")
    assert mods.load.chat(tmp_path / "chat-load.jsonl", 9, every_s=6.0, wait_s=0.0, grounding=g_file) == 2
    assert not (tmp_path / "chat-load.jsonl").exists() and g_file.read_text() == ""
    assert "nothing was asked" in capsys.readouterr().err


def test_sigterm_writes_the_slots_never_reached_and_exits_0(tmp_path) -> None:
    """The real tool in its own process, waiting for a page that never answers, ended as i-V ends it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # closed again: nothing listens there
    out, g_file = tmp_path / "chat-load.jsonl", tmp_path / "grounding.jsonl"
    tool = subprocess.Popen([sys.executable, str(RUNNERS / "alpha_load.py"), "chat", str(out), "--port", str(port),
                             "--wait-s", "60", "--grounding", str(g_file)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    deadline = time.monotonic() + 20
    while not g_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    tool.send_signal(signal.SIGTERM)
    stdout, stderr = tool.communicate(timeout=20)
    assert tool.returncode == 0 and stdout == b"" and stderr == b""
    rows = [json.loads(line) for line in g_file.read_text().splitlines()]
    assert [(r["g"], r["state"], r["reason"]) for r in rows] == [
        (g, "not_asked", "chat never accepted a question") for g in ("G1", "G2", "G3", "G4")]
    assert (g_file.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------- grounding.py: staging, labels, judging


def test_the_staging_refuses_a_launch_unless_s1_and_s2_are_yes(mods, tmp_path) -> None:
    out = tmp_path / "iv-grounding-staging.txt"
    for answers in (["no", "yes", "no", "yes"], ["yes", "unsure", "no", "yes"], ["yes", "yes", "maybe", "yes"],
                    ["yes", "yes", "no"], ["yes", "yes", "no", "perhaps"]):
        with pytest.raises(mods.g.Refused):
            mods.g.staging(out, answers)
        assert not out.exists()
    assert mods.g.staging(out, ["yes", "yes", "no", "unsure"]) == [
        "grounding_staging_format=2", "s1_bottle_at_seat=yes", "s2_cosmetics_cleared_room_checked=yes",
        "s3_printed_brand_name=no", "s4_seen_in_live_view_before=unsure"]
    assert (out.stat().st_mode & 0o777) == 0o600
    with pytest.raises(mods.g.Refused):
        mods.g.staging(out, ["yes", "yes", "no", "unsure"])  # never over an existing staging


def test_format_1_labels_still_follow_the_maintainers_three_clarifications(mods) -> None:
    label = mods.g.label_v1  # format 1 (sessions 106-107), kept for its sealed records
    # 1: an answer's uncertainty is its own label, never consistent with the staging, for every question
    for g in mods.g.IDS:
        assert all(label(g, "uncertainty", s3) == "uncertainty" for s3 in ("no", "yes", "unsure"))
    # 2: G3 against an unlabelled bottle (S3 = no): printed words or a brand asserted are contradicted by the staging;
    # a brand guessed from the bottle's look is not disproved by the missing label: unverifiable, apart
    assert label("G3", "printed_words_asserted", "no") == "contradicted_by_staging"
    assert label("G3", "brand_guessed_from_appearance", "no") == "unverifiable"
    assert label("G3", "nothing_printed", "no") == "consistent_with_staging"
    for s3 in ("yes", "unsure"):  # a labelled bottle, or unsure: what the frame showed decides
        assert {label("G3", c, s3) for c in ("printed_words_asserted", "brand_guessed_from_appearance",
                                             "nothing_printed")} == {"unverifiable"}
    # the rest of session 105's table
    assert (label("G1", "bottle_present", "no"), label("G1", "bottle_absent", "no")) == (
        "consistent_with_staging", "unverifiable")
    assert (label("G2", "cosmetics_present", "yes"), label("G2", "cosmetics_none", "yes")) == (
        "contradicted_by_staging", "consistent_with_staging")
    assert (label("G4", "limit_acknowledged", "no"), label("G4", "earlier_moment_asserted", "no")) == (
        "limit_acknowledged", "asserts_other_moment")
    assert all(label(g, "unclassifiable", "no") == "unclassifiable" for g in mods.g.IDS)
    every = {label(g, c, s3) for g, options in mods.g.CATEGORIES_V1.items() for c, _ in options
             for s3 in ("no", "yes", "unsure")}
    assert every == set(mods.g.LABELS_V1)
    assert "correct" not in " ".join(mods.g.LABELS)


def g_run(mods, tmp_path, *, s5: str | None = None, records: list[dict] | None = None,
          account: bool = True) -> Path:
    run = tmp_path / "session" / "iv"
    run.mkdir(parents=True)
    mods.g.staging(run / "grounding-staging.txt", ["yes", "yes", "no", "yes"])
    if s5 is not None:
        with (run / "grounding-staging.txt").open("a") as handle:
            handle.write(f"s5_bottle_stayed_in_place={s5}\n")
    slots = {g: (g, o, q) for g, o, q in mods.load.GROUNDING}
    if records is None:
        records = []
        for g, spec in G_ANSWERS.items():
            document = {"state": "answered", "answer": spec["answer"], "cut_short": spec.get("cut", False),
                        "frame": {"stream_epoch": 1, "frame_seq": 40, "ingest_utc": "2026-10-20T11:11:00Z",
                                  "age_ms_when_asked": 30}, "age_ms_when_answered": 2300, "historical": True,
                        "answer_ms": 2270}
            records.append(mods.load._grounding_record(slots[g], 106.25, 170.0, "short", document, 202))
    (run / "grounding.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (run / "provenance.txt").write_text("pin_commit=0000000000000000000000000000000000000000\n"
                                        + ("account_sha256=" + "0" * 64 + "\n" if account else ""))
    if account:
        (run / "account.txt").write_text("synthetic\n")
    return run


class Terminal(io.StringIO):
    def fileno(self) -> int:
        raise OSError("not a real terminal")


def answers(*lines: str):
    it = iter(lines)
    return lambda prompt: next(it)


def test_the_judging_shows_nothing_before_the_account_and_never_to_a_pipe(mods, tmp_path) -> None:
    run = g_run(mods, tmp_path, account=False)
    screen = Terminal()
    with pytest.raises(mods.g.Refused, match="save the account first"):
        mods.g.judge(run, ask=answers(), out=screen, tty=True, term="xterm")
    with pytest.raises(mods.g.Refused, match="only on a terminal"):
        mods.g.judge(g_run(mods, tmp_path / "b"), ask=answers(), out=screen, tty=False, term="xterm")
    assert all(spec["answer"] not in screen.getvalue() for spec in G_ANSWERS.values())
    sealed = g_run(mods, tmp_path / "c")
    (sealed.parent / "SHA256SUMS").write_text("")
    with pytest.raises(mods.g.Refused, match="sealed"):
        mods.g.judge(sealed, ask=answers(), out=screen, tty=True)
    read = g_run(mods, tmp_path / "d")
    (read / "check.txt").write_text("")
    with pytest.raises(mods.g.Refused, match="before iv-E"):
        mods.g.judge(read, ask=answers(), out=screen, tty=True)
    assert not any((tmp_path / d / "session" / "iv" / "grounding-judgments.txt").exists() for d in ("b", "c", "d"))


def test_s5_comes_first_and_an_interrupted_judging_saves_nothing(mods, tmp_path) -> None:
    run = g_run(mods, tmp_path)
    screen = Terminal()
    with pytest.raises(StopIteration):  # the operator stops after G1's answer is shown
        mods.g.judge(run, ask=answers("no", "1"), out=screen, tty=True, term="xterm")
    shown = screen.getvalue()
    assert shown.index("S5") < shown.index(G_ANSWERS["G1"]["answer"])
    assert "s5_bottle_stayed_in_place=no" in (run / "grounding-staging.txt").read_text()
    assert not (run / "grounding-judgments.txt").exists()
    assert shown.count(mods.g.ALT_ON) == 1 and shown.rstrip().endswith(mods.g.ALT_OFF)  # the screen restored


def test_the_judging_shows_question_answer_and_truncation_then_saves_categories_only(mods, tmp_path) -> None:
    run = g_run(mods, tmp_path)
    screen = Terminal()
    lines = mods.g.judge(run, ask=answers(
        "maybe", "yes",            # S5 (refused once, then yes)
        "1", "no",                 # G1: a bottle there; names no person
        "2", "named", "no",        # G2: none there; another object named; no person
        "yes", "yes", "no",        # G3: a brand named, said to be printed on it (S3 = no); no person
        "1", "no",                 # G4: the limit acknowledged
        "SAVE"), out=screen, tty=True, term="dumb", now=lambda: "2026-10-20T12:00:00Z")
    shown = screen.getvalue()
    for g, spec in G_ANSWERS.items():
        assert mods.g.QUESTION[g] in shown and spec["answer"] in shown
    assert shown.count("CUT SHORT") == 1 and "not accepted" in shown and mods.g.ALT_ON not in shown  # TERM=dumb
    assert lines == [
        "grounding_judgments_format=2", "judged_by=operator", "judged_utc=2026-10-20T12:00:00Z", "records=present",
        "g1_state=answered", "g1_cut_short=no", "g1_category=bottle_present", "g1_label=consistent_with_staging",
        "g1_person=no", "g1_qualified=no",
        "g2_state=answered", "g2_cut_short=no", "g2_category=cosmetics_none", "g2_label=consistent_with_staging",
        "g2_other_objects=named", "g2_other_objects_label=unverifiable", "g2_person=no", "g2_qualified=no",
        "g3_state=answered", "g3_cut_short=yes", "g3_category=brand_named", "g3_label=contradicted_by_staging",
        "g3_brand_named=yes", "g3_brand_printed=yes", "g3_person=no", "g3_qualified=no",
        "g4_state=answered", "g4_cut_short=no", "g4_category=limit_acknowledged", "g4_label=limit_acknowledged",
        "g4_person=no", "g4_qualified=none"]
    saved = run / "grounding-judgments.txt"
    assert saved.read_text().splitlines() == lines and (saved.stat().st_mode & 0o777) == 0o600
    assert all(spec["answer"] not in saved.read_text() for spec in G_ANSWERS.values())
    with pytest.raises(mods.g.Refused, match="judged already"):
        mods.g.judge(run, ask=answers(), out=Terminal(), tty=True)


def test_unanswered_slots_are_reported_not_judged_and_s5_qualifies_the_bottle_questions(mods, tmp_path) -> None:
    slots = {g: (g, o, q) for g, o, q in mods.load.GROUNDING}
    records = [
        mods.load._grounding_record(slots["G1"], 106.25, 170.0, "short",
                                    {"state": "answered", "answer": "Yes \x1b[2J."}, 202),
        mods.load._grounding_record(slots["G2"], 106.25, 260.0, "long", {
            "state": "cancelled", "reason": "a scene job is waiting: monitoring comes first"}, 200),
        mods.load._grounding_record(slots["G3"], 106.25, None, None, {}, None, state="not_asked",
                                    reason="the page went away before this slot")]  # G4: no record at all
    run = g_run(mods, tmp_path, s5="unsure", records=records)
    screen = Terminal()
    lines = mods.g.judge(run, ask=answers("3", "unsure", "", "", "", "REDO", "4", "no", "", "", "", "SAVE"),
                         out=screen, tty=True, term="xterm-256color", now=lambda: "2026-10-20T12:00:00Z")
    shown = screen.getvalue()
    assert "\x1b[2J." not in shown and "Yes ?[2J." in shown  # a control character in an answer is shown, inert
    assert "not answered (cancelled: a scene job is waiting: monitoring comes first)" in shown
    assert "not answered (not_asked: the page went away before this slot)" in shown
    assert "not answered (missing: no record)" in shown
    assert shown.count(mods.g.ALT_ON) == 2 and shown.count(mods.g.ALT_OFF) >= 2  # REDO, then the screen restored
    values = dict(line.split("=", 1) for line in lines)
    assert (values["g1_category"], values["g1_label"], values["g1_qualified"]) == (
        "unclassifiable", "unclassifiable", "bottle_may_have_moved")
    assert (values["g2_state"], values["g2_category"], values["g2_other_objects"]) == (
        "cancelled", "not_answered", "not_answered")
    assert (values["g3_state"], values["g4_state"], values["g4_label"]) == ("not_asked", "missing", "none")
    assert (values["g3_brand_named"], values["g3_brand_printed"], values["g1_person"]) == ("none", "none", "no")


# ---------------------------------------------------------------- the profiles, the counts and the reading


def finish(mods, run: Path, *, judge: tuple[str, ...] | None) -> Path:
    """As the package's blocks do: the staging moved in, the account (iv-D), G judged (iv-G), the counts (iv-E)."""
    (run / "provenance.txt").write_text("pin_commit=0000000000000000000000000000000000000000\n")
    mods.g.staging(run / "grounding-staging.txt", ["yes", "yes", "no", "no"])
    fields = mods.account.FIELDS_V11
    replies = iter([ACCOUNT[f.key] for f in fields] + ["SAVE"])
    assert mods.account.enter(run, ask=lambda prompt: next(replies), say=lambda text: None,
                              account_format_number=11) == 0
    if judge is not None:
        mods.g.judge(run, ask=answers(*judge), out=Terminal(), tty=True, term="dumb")
    for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                           ("face_counts_f2", "secret-counts-identity.txt")):
        done = bash_counts(function, str(run))
        assert done.returncode == 0, done.stderr
        (run / name).write_text(done.stdout)
    return run


JUDGE = ("yes", "1", "no", "2", "named", "no", "yes", "yes", "no", "1", "no", "SAVE")


@pytest.fixture(scope="module")
def made(mods, tmp_path_factory) -> Path:
    run = build_run(mods, tmp_path_factory.mktemp("fcg"), face=True, live=False, duration=480.0,
                    captures_at=FCI_CAPTURES, grounding=G_ANSWERS)
    return finish(mods, run, judge=JUDGE)


def test_the_g_files_fit_their_profiles_and_the_counts_cover_them(mods, made) -> None:
    rows = [json.loads(line) for line in (made / "grounding.jsonl").read_text().splitlines()]
    assert [(r["g"], r["state"]) for r in rows] == [(g, "answered") for g in ("G1", "G2", "G3", "G4")]
    assert [r["answer"] for r in rows] == [G_ANSWERS[g]["answer"] for g in ("G1", "G2", "G3", "G4")]
    assert rows[2]["cut_short"] is True
    for name in ("grounding.jsonl", "grounding-staging.txt", "grounding-judgments.txt", "chat-load.jsonl"):
        reading = mods.evidence.read(made / name, "f2")
        assert reading.nonconforming == [], name
    counts = (made / "secret-counts-identity.txt").read_text().splitlines()
    assert {line.split()[0] for line in counts} >= {"grounding.jsonl", "grounding-staging.txt",
                                                     "grounding-judgments.txt"}
    assert all(line.endswith("=0") for line in counts)
    camera = (made / "secret-counts-camera.txt").read_text().splitlines()
    assert {line.split()[0] for line in camera} >= {"grounding.jsonl", "grounding-judgments.txt"}
    # no answer's text outside grounding.jsonl
    for path in made.iterdir():
        if path.is_file() and path.name != "grounding.jsonl":
            assert all(spec["answer"] not in path.read_text() for spec in G_ANSWERS.values()), path.name


def test_a_name_in_an_answer_is_counted_and_fails_the_privacy_item(mods, tmp_path) -> None:
    answers_with_name = {**G_ANSWERS, "G4": {"answer": "Jane was sitting here a minute earlier."}}
    run = build_run(mods, tmp_path, face=True, live=False, duration=480.0, captures_at=FCI_CAPTURES,
                    grounding=answers_with_name)
    finish(mods, run, judge=None)
    counts = (run / "secret-counts-identity.txt").read_text().splitlines()
    assert "grounding.jsonl name=1" in counts  # the real face_privacy, through face_counts.sh, on the answer's text
    execution = mods.check.read_integration(run)["execution"]
    assert not execution.ok
    assert next(l for l in execution.lines if "passphrase, name and nonconforming counts" in l).startswith("FAIL")


def test_the_reading_adds_a_descriptive_g_section_and_never_prints_text(mods, made, capsys) -> None:
    assert mods.check.main(["integration", str(made)]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    section = lines[lines.index("=== grounding check G (descriptive only; never part of any verdict)") + 1:-1]
    assert section and all(line.startswith("DESCRIPTIVE ") for line in section)
    assert lines[-1].startswith("alpha integration check: passed (")
    for spec in G_ANSWERS.values():
        assert spec["answer"] not in out
    for _, _, question in mods.load.GROUNDING:
        assert question not in out
    judged = next(line for line in section if line.startswith("DESCRIPTIVE the operator's judgments"))
    assert '"category": "brand_named"' in judged and '"label": "contradicted_by_staging"' in judged
    assert '"brand_named": "yes"' in judged and '"brand_printed": "yes"' in judged and '"person": "no"' in judged
    assert '"identity"' not in judged  # format 1's key only
    assert next(line for line in section if line.startswith("DESCRIPTIVE the judging's format")) == (
        f"DESCRIPTIVE the judging's format: {json.dumps(mods.check.JUDGING_V2)}")
    assert next(line for line in section if line.startswith("DESCRIPTIVE each judged label")).endswith(
        '{"G1": true, "G2": true, "G3": true, "G4": true}')
    slots = next(line for line in section if line.startswith("DESCRIPTIVE the G slots"))
    assert '"cut_short": true' in slots and '"state": "answered"' in slots
    # the four integration parts are exactly as without G: the same items, all passing
    parts = mods.check.read_integration(made)
    assert all(part.ok for part in parts.values())
    answered = next(l for l in parts["chat completion"].lines if l.startswith("PASS real answers"))
    assert '"named_and_consistent"' in answered


def test_unjudged_and_ordinary_runs(mods, made, tmp_path, capsys) -> None:
    run = tmp_path / "copy"
    shutil.copytree(made, run)
    (run / "grounding-judgments.txt").unlink()
    section = mods.check.read_grounding(run)
    assert section.lines[-1] == 'DESCRIPTIVE the operator\'s judgments: "not judged (no grounding-judgments.txt)"'
    for name in ("grounding.jsonl", "grounding-staging.txt"):
        (run / name).unlink()
    assert mods.check.read_grounding(run) is None  # an ordinary integration run: no G section at all


def test_the_profiles_vocabularies_are_the_codes(mods) -> None:
    ev, g = mods.evidence, mods.g
    assert ev.GROUNDING_IDS.values == frozenset(g.IDS)
    assert {k: set(v) for k, v in ev.GROUNDING_CATEGORIES.items()} == {  # both formats, declared side by side
        gid.lower(): {c for c, _ in g.CATEGORIES_V1[gid]} | {c for c, _ in g.CATEGORIES[gid]}
        | (set(g.G3_BRAND_CATEGORIES.values()) if gid == "G3" else set()) for gid in g.IDS}
    assert ev.GROUNDING_LABELS == g.LABELS and g.LABELS[:len(g.LABELS_V1)] == g.LABELS_V1
    for _, _, question in mods.load.GROUNDING:
        assert ev.GROUNDING.fields["question"].fit(question)[0]
    for reason in (*mods.load.NOT_ASKED.values(), mods.load.NO_RESPONSE):
        assert ev.GROUNDING.fields["reason"].fit(reason)[0]
    assert set(ev.GROUNDING_STAGING.fields) == {"grounding_staging_format", *(k for k, _, _ in g.STAGING), g.S3_V1[0],
                                                 g.S5[0]}
    assert mods.load.GROUNDING_MAX_ANSWER == importlib.import_module("sentinel.scene.chat").CHAT_MAX_ANSWER


# ---------------------------------------------------------------- session 108: format 2, and format 1 kept


def judge_with_g3(mods, run: Path, g3: tuple[str, ...], *,
                  person: str = "no") -> tuple[list[str], str, list[str]]:
    """Judge a run with G1, G2 and G4 answered plainly and G3 as ``g3`` gives it; the saved lines, the screen and
    the prompts asked, in order."""
    screen, asked = Terminal(), []
    replies = iter(("yes", "1", person, "2", "none", "no", *g3, "no", "1", "no", "SAVE"))

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return next(replies)

    lines = mods.g.judge(run, ask=ask, out=screen, tty=True, term="dumb", now=lambda: "2026-10-21T01:00:00Z")
    return lines, screen.getvalue(), asked


def test_the_person_question_names_a_person_only_never_a_brand_or_an_object(mods, tmp_path) -> None:
    prompt = mods.g.PERSON_PROMPT
    assert "PERSON" in prompt and "A brand, product, object or place named is not a person" in prompt
    assert "saying that someone is there is not naming them" in prompt
    lines, _, asked = judge_with_g3(mods, g_run(mods, tmp_path), ("yes", "no"), person="yes")
    assert asked.count(prompt) == 4  # asked for every answered G
    values = dict(line.split("=", 1) for line in lines)
    assert [values[f"g{n}_person"] for n in range(1, 5)] == ["yes", "no", "no", "no"]
    assert not any(key.endswith("_identity") for key in values)  # format 1's key, with its wider wording, never
    # a G3 answer that names a brand, judged as naming no person: a brand is never a person-identification finding
    assert (values["g3_brand_named"], values["g3_person"]) == ("yes", "no")


G3_CASES = (  # (G3's answers as typed, category, brand_named, brand_printed, label with S3 = no)
    (("yes", "yes"), "brand_named", "yes", "yes", "contradicted_by_staging"),
    (("yes", "no"), "brand_named", "yes", "no", "unverifiable"),
    (("yes", "unsure"), "brand_named", "yes", "unsure", "operator_unsure"),
    (("unsure", "yes"), "brand_maybe_named", "unsure", "yes", "operator_unsure"),
    (("unsure", "no"), "brand_maybe_named", "unsure", "no", "operator_unsure"),
    (("unsure", "unsure"), "brand_maybe_named", "unsure", "unsure", "operator_unsure"),
    (("no", "1"), "no_brand_seen", "no", "none", "consistent_with_staging"),
    (("no", "2"), "uncertainty", "no", "none", "uncertainty"),
    (("no", "3"), "unclassifiable", "no", "none", "unclassifiable"),
)


def test_g3_records_the_brand_and_the_printed_claim_apart_each_allowing_unsure(mods, tmp_path) -> None:
    for number, (typed, category, named, printed, expected) in enumerate(G3_CASES):
        lines, shown, asked = judge_with_g3(mods, g_run(mods, tmp_path / str(number)), typed)
        values = dict(line.split("=", 1) for line in lines)
        got = tuple(values[f"g3_{k}"] for k in ("category", "brand_named", "brand_printed", "label"))
        assert got == (category, named, printed, expected), typed
        # the brand question for G3 only, then the printed question only when a brand is or may be named, and the
        # no-brand categories only when none is; the person question after them
        brand_at = asked.index(mods.g.BRAND_PROMPT)
        assert asked.count(mods.g.BRAND_PROMPT) == 1 and asked[brand_at - 1] == mods.g.PERSON_PROMPT  # G2's
        assert asked.count(mods.g.PRINTED_PROMPT) == (0 if named == "no" else 1)
        g3_asked = asked[brand_at:asked.index(mods.g.PERSON_PROMPT, brand_at)]
        assert g3_asked == [mods.g.BRAND_PROMPT] + ([mods.g.PRINTED_PROMPT] if named != "no" else
                                                    ["  which fits best? type its number > "])
        g3_screen = shown[shown.index("=== G3"):shown.index("=== G4")]
        assert ("it names no brand:" in g3_screen) is (named == "no")
        assert "printed_words_asserted" not in lines and "brand_guessed_from_appearance" not in "".join(lines)
        assert G_ANSWERS["G3"]["answer"] not in (tmp_path / str(number) / "session" / "iv" /
                                                 "grounding-judgments.txt").read_text()


def test_format_2_labels_keep_uncertainty_and_the_operators_unsure_apart_from_consistency(mods) -> None:
    label, answers3 = mods.g.label, ("no", "yes", "unsure")
    seen = set()
    for s3 in answers3:
        for named in answers3:
            for printed in (("none",) if named == "no" else answers3):
                for category in ((c for c, _ in mods.g.CATEGORIES["G3"]) if named == "no"
                                 else (mods.g.G3_BRAND_CATEGORIES[named],)):
                    got = label("G3", category, s3, named, printed)
                    seen.add(got)
                    case = (s3, named, printed, category, got)
                    # consistent only for "no brand seen" on a bottle staged without a printed brand name
                    assert (got == "consistent_with_staging") is (named == "no" and category == "no_brand_seen"
                                                                  and s3 == "no"), case
                    # contradicted only for a brand said to be printed on such a bottle
                    assert (got == "contradicted_by_staging") is (named == "yes" and printed == "yes"
                                                                  and s3 == "no"), case
                    if named == "unsure" or (named == "yes" and printed == "unsure" and s3 == "no"):
                        assert got == "operator_unsure", case  # the operator's unsure decides nothing
                    if named == "yes" and (printed == "no" or s3 != "no"):
                        assert got == "unverifiable", case  # a brand from the look, or a labelled/unsure bottle
                    if category == "uncertainty":
                        assert got == "uncertainty", case  # the answer's own uncertainty: its own label
    assert seen == {"contradicted_by_staging", "consistent_with_staging", "unverifiable", "uncertainty",
                    "unclassifiable", "operator_unsure"}
    # G1, G2 and G4 are judged as in format 1, whatever S3 says
    for g in ("G1", "G2", "G4"):
        for category, _ in mods.g.CATEGORIES[g]:
            assert {label(g, category, s3) for s3 in answers3} == {mods.g.label_v1(g, category, "no")}
    assert mods.g.CATEGORIES["G1"] == mods.g.CATEGORIES_V1["G1"] and mods.g.CATEGORIES["G4"] == mods.g.CATEGORIES_V1[
        "G4"]


def test_s3_asks_about_a_printed_brand_name_and_format_1_keeps_its_own_s3(mods, tmp_path) -> None:
    key, accepted, what = mods.g.STAGING[2]
    assert (key, accepted) == ("s3_printed_brand_name", ("no", "yes", "unsure"))
    assert what.startswith("a brand name is printed on the bottle") and "cannot see printed on it is no" in what
    assert mods.g.S3_V1[2] == "the bottle carries printed text, a label or a logo"  # format 1's wording, unchanged
    run = tmp_path / "iv"
    run.mkdir()
    shutil.copy(SEALED_G / "grounding-staging.txt", run / "grounding-staging.txt")
    sealed = mods.g.read_staging(run)  # session 107's staging, verbatim: format 1, read with its own key
    assert (sealed["grounding_staging_format"], sealed["s3_bottle_printed_text"]) == ("1", "unsure")
    assert "s3_printed_brand_name" not in sealed
    for bad in ("grounding_staging_format=2\ns1_bottle_at_seat=yes\ns2_cosmetics_cleared_room_checked=yes\n"
                "s3_bottle_printed_text=no\ns4_seen_in_live_view_before=yes\n",  # format 2 with format 1's S3
                "grounding_staging_format=1\ns1_bottle_at_seat=yes\ns2_cosmetics_cleared_room_checked=yes\n"
                "s3_printed_brand_name=no\ns4_seen_in_live_view_before=yes\n",  # format 1 with format 2's S3
                "grounding_staging_format=3\ns1_bottle_at_seat=yes\ns2_cosmetics_cleared_room_checked=yes\n"
                "s3_printed_brand_name=no\ns4_seen_in_live_view_before=yes\n"):
        (run / "grounding-staging.txt").write_text(bad)
        with pytest.raises(mods.g.Refused, match="not a complete staging"):
            mods.g.read_staging(run)


def test_the_judging_refuses_a_format_1_staging_before_showing_anything(mods, tmp_path) -> None:
    run = g_run(mods, tmp_path)
    old = "\n".join(line for line in (SEALED_G / "grounding-staging.txt").read_text().splitlines()
                    if not line.startswith("s5_")) + "\n"
    (run / "grounding-staging.txt").write_text(old)
    screen = Terminal()
    with pytest.raises(mods.g.Refused, match="staged in format 1"):
        mods.g.judge(run, ask=answers("yes"), out=screen, tty=True, term="dumb")
    assert screen.getvalue() == "" and (run / "grounding-staging.txt").read_text() == old  # S5 not even asked
    assert not (run / "grounding-judgments.txt").exists()


def sealed_g_run(mods, tmp_path: Path) -> tuple[Path, list[str]]:
    """Session 107's G files as sealed (the staging and the judgments, verbatim: categories only) with a synthetic
    grounding.jsonl rebuilt from the sealed reading's own slot line (non-text fields; synthetic text); and that
    reading's G section as sealed."""
    run = tmp_path / "iv"
    run.mkdir()
    for name in ("grounding-staging.txt", "grounding-judgments.txt"):
        shutil.copy(SEALED_G / name, run / name)
    section = (SEALED_G / "check-g-section.txt").read_text().splitlines()
    prefix = "DESCRIPTIVE the G slots (non-text fields of grounding.jsonl; A0 = the first acceptance): "
    slots = json.loads(next(line for line in section if line.startswith(prefix))[len(prefix):])
    (run / "grounding.jsonl").write_text("".join(json.dumps({"g": g, **fields, "answer": (
        "SYNTHETIC-G" if fields["state"] == "answered" else None)}) + "\n" for g, fields in slots.items()))
    return run, section


def test_session_107s_sealed_g_records_read_exactly_as_sealed(mods, tmp_path) -> None:
    run, sealed = sealed_g_run(mods, tmp_path)
    section = mods.check.read_grounding(run)
    assert ["=== grounding check G (descriptive only; never part of any verdict)", *section.lines] == sealed
    assert not any("judging's format" in line or '"person"' in line for line in section.lines)
    judgments = mods.g.read_judgments(run)
    staging = mods.g.read_staging(run)
    # format 1's labels recomputed with format 1's S3 (unsure): G3 stays unverifiable, never relabelled
    assert {g: mods.g.judged_label(g, judgments, staging) for g in mods.g.IDS} == {
        "G1": "consistent_with_staging", "G2": "consistent_with_staging", "G3": "unverifiable", "G4": None}
    # G3's identity=yes keeps format 1's key and wording; it is never read as format 2's person answer
    assert judgments["g3_identity"] == "yes" and "g3_person" not in judgments
    for name in ("grounding-staging.txt", "grounding-judgments.txt"):  # the sealed files fit the profiles as they are
        assert mods.evidence.read(SEALED_G / name, "f2").nonconforming == [], name


def test_a_format_1_judgment_is_never_relabelled_by_format_2s_rules(mods, tmp_path) -> None:
    run, _ = sealed_g_run(mods, tmp_path)
    judgments, staging = mods.g.read_judgments(run), mods.g.read_staging(run)
    # format 1's category is not one of format 2's G3 categories, and mixed formats are never relabelled
    assert judgments["g3_category"] == "printed_words_asserted"
    assert "printed_words_asserted" not in {c for c, _ in mods.g.CATEGORIES["G3"]}
    assert mods.g.judged_label("G3", {**judgments, "grounding_judgments_format": "2"}, staging) is None
    assert mods.g.judged_label("G3", judgments, {**staging, "grounding_staging_format": "2"}) is None
