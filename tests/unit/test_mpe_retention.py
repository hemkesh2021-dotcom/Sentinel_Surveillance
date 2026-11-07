"""The multi-pose comparison's outcome precedence (decision 12) and its retention and withdrawal controls (decisions 10
and 11), on synthetic records only (session 116, part 23)."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from face_runs import RUNNERS

sys.path.insert(0, str(RUNNERS))
import mpe_manifest  # noqa: E402
import mpe_outcome  # noqa: E402
import mpe_retention  # noqa: E402

PIN = "a" * 40
PROTO = "p" * 64
FROZEN = {"format": "mpe-frozen-1", "digest": "d" * 64, "prototypes_digest": PROTO,
          "windows": [["bs:lens", 1], ["b4:left", 2], ["b5:right", 3]], "candidate": "candidate-mp-b-20261101T090000Z"}
S30, S60 = 30_000_000_000, 60_000_000_000


def hold(name, spot, kind, base, cand, *, length=None, at=("none", "none"), excluded=False) -> dict:
    length = length or (S30 if kind == "facing" else S60)
    return {"hold": name, "spot": spot, "kind": kind, "excluded": excluded, "not_as_cued": False, "length_ns": length,
            "known_ns": {"baseline": round(base * length), "candidate": round(cand * length)},
            "at_mark": {"baseline": at[0], "candidate": at[1]}}


def em_compare(*, natural=(0.20, 0.45), facing=(0.90, 0.90)) -> dict:
    holds = []
    for spot in ("s", "4", "5"):
        holds += [hold(f"m{spot}f{n}", spot, "facing", *facing, at=("acquired", "acquired")) for n in (1, 2, 3)]
    for spot in ("4", "5"):
        holds += [hold(f"m{spot}n{n}", spot, "natural", *natural) for n in (1, 2, 3)]
    conditions = [{"spot": s, "kind": k, "inconclusive": False} for s, k in mpe_outcome_conditions()]
    return {"format": "mpe-compare-1", "session": "E_M", "replay_valid": True, "inconclusive": [], "holds": holds,
            "conditions": conditions, "frozen_prototypes_digest": PROTO}


def mpe_outcome_conditions():
    return (("s", "facing"), ("4", "facing"), ("4", "natural"), ("5", "facing"), ("5", "natural"))


def ep_compare(*, match=0, known=0, labels=(), exposure=None, replay_valid=True) -> dict:
    holds = [hold(f"p{spot}n{n}", spot, "natural", 0.0, 0.0) for spot in ("4", "5") for n in (1, 2, 3)]
    return {"format": "mpe-compare-1", "session": "E_P", "replay_valid": replay_valid, "inconclusive": list(labels),
            "holds": holds, "conditions": [{"spot": s, "kind": "natural", "inconclusive": False} for s in "45"],
            "participant": {"compared": 500, "candidate_match_votes": match, "baseline_match_votes": 0,
                            "candidate_known_entries": known if replay_valid else None},
            "exposure": exposure if exposure is not None else {"bs:lens": 17, "b4:left": 9, "b5:right": 4},
            "frozen_prototypes_digest": PROTO}


def ev(kind: str, compare: dict | None = None, state: str = "compared") -> dict:
    name = {"E_M": "mp-em-20261102T090000Z", "E_P": "mp-ep-20261102T100000Z"}[kind]
    return {"kind": kind, "name": name, "state": state, "compare": compare, "seal_sha256": "s" * 64}


# --- decision 12: the order of precedence ------------------------------------------------------------------------------

def test_every_criterion_met_is_supported_and_nothing_else() -> None:
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), ev("E_P", ep_compare()))
    assert out["outcome"] == "SUPPORTED" and out["reasons"] == ["every_criterion_met"]
    assert all(c["met"] for c in out["criteria"].values())


def test_a_participant_match_rejects_whatever_the_other_evidence_or_the_replay_shows() -> None:
    broken = ep_compare(match=3, labels=("life_records_missing", "replay_does_not_reproduce_the_recorded_history"),
                        replay_valid=False, exposure={"bs:lens": 0})
    for em in (ev("E_M", state="not_run"), ev("E_M", state="execution_not_passed"), ev("E_M", em_compare())):
        out = mpe_outcome.evaluate(FROZEN, em, ev("E_P", broken))
        assert out["outcome"] == "REJECTED" and out["reasons"] == ["e_p_candidate_match_on_the_participant"]
    # a candidate KNOWN alone (no vote counted) also rejects
    assert mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), ev("E_P", ep_compare(known=1)))["outcome"] == \
        "REJECTED"


def test_a_match_is_rejected_only_on_readable_evidence_of_the_frozen_candidate_and_the_participant_alone() -> None:
    em = ev("E_M", em_compare())
    others = mpe_outcome.evaluate(FROZEN, em, ev("E_P", ep_compare(match=2, labels=("account_others_in_view_not_no",))))
    assert others["outcome"] == "INCONCLUSIVE"
    assert "e_p_candidate_match_not_attributable_to_the_frozen_candidate_and_the_participant" in others["reasons"]
    wrong = ep_compare(match=2)
    wrong["frozen_prototypes_digest"] = "q" * 64
    out = mpe_outcome.evaluate(FROZEN, em, ev("E_P", wrong))
    assert out["outcome"] == "INCONCLUSIVE" and "e_p_compared_with_another_candidate" in out["reasons"]
    for state in ("not_run", "not_compared", "execution_not_passed", "seal_not_verified", "not_sealed",
                  "comparison_missing", "compared_more_than_once"):
        out = mpe_outcome.evaluate(FROZEN, em, ev("E_P", state=state))
        assert out["outcome"] == "INCONCLUSIVE" and f"e_p_{state}" in out["reasons"]


@pytest.mark.parametrize("change,reason", [
    (lambda em, ep, fr: None, None),
    (lambda em, ep, fr: em.update(inconclusive=["life_records_incomplete"], replay_valid=False),
     "e_m:life_records_incomplete"),
    (lambda em, ep, fr: em["conditions"][2].update(inconclusive=True), "e_m:fewer_than_two_valid_holds_4_natural"),
    (lambda em, ep, fr: ep["conditions"][1].update(inconclusive=True), "e_p:fewer_than_two_valid_holds_5_natural"),
    (lambda em, ep, fr: ep["exposure"].update({"b4:left": 0}), "e_p:participant_not_seen_in_window_b4:left"),
    (lambda em, ep, fr: ep["exposure"].update({"b5:right": "excluded_not_as_cued"}),
     "e_p:participant_not_seen_in_window_b5:right"),
    (lambda em, ep, fr: ep["exposure"].pop("bs:lens"), "e_p:participant_not_seen_in_window_bs:lens"),
    (lambda em, ep, fr: ep.update(inconclusive=["compared_faces_without_a_shadow_decision"]),
     "e_p:compared_faces_without_a_shadow_decision"),
    (lambda em, ep, fr: fr.update(format="other"), "frozen_record_missing"),
    (lambda em, ep, fr: em.update(frozen_prototypes_digest="q" * 64), "e_m_compared_with_another_candidate"),
    (lambda em, ep, fr: [h.update(known_ns=None) for h in em["holds"]], "e_m:replay_numbers_withheld"),
])
def test_missing_invalid_or_incomplete_evidence_is_inconclusive_before_any_criterion(change, reason) -> None:
    # every case would fail a criterion too (no natural gain): INCONCLUSIVE comes first
    em, ep, fr = em_compare(natural=(0.2, 0.2)), ep_compare(), dict(FROZEN)
    change(em, ep, fr)
    out = mpe_outcome.evaluate(fr, ev("E_M", em), ev("E_P", ep))
    if reason is None:
        assert out["outcome"] == "NOT_SUPPORTED"
    else:
        assert out["outcome"] == "INCONCLUSIVE" and reason in out["reasons"] and "criteria" not in out


def test_the_gain_bar_is_twenty_points_at_both_marks_in_integer_nanoseconds() -> None:
    em = em_compare(natural=(0.20, 0.40))  # exactly +20 points
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em), ev("E_P", ep_compare()))
    assert out["outcome"] == "SUPPORTED"
    em["holds"][-1]["known_ns"]["candidate"] -= 1  # one nanosecond short at 5.0 m
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em), ev("E_P", ep_compare()))
    assert out["outcome"] == "NOT_SUPPORTED" and out["reasons"] == ["natural_gain_5_0m"]
    assert out["criteria"]["natural_gain_4_0m"]["met"]


def test_a_facing_acquisition_lost_or_a_pooled_facing_loss_over_five_points_is_not_supported() -> None:
    em = em_compare()
    em["holds"][4]["at_mark"] = {"baseline": "acquired", "candidate": "not_at_mark"}
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em), ev("E_P", ep_compare()))
    assert out["outcome"] == "NOT_SUPPORTED" and out["reasons"] == ["facing_acquisitions_lost"]
    em = em_compare(facing=(0.90, 0.85))  # exactly -5 points: within the limit
    assert mpe_outcome.evaluate(FROZEN, ev("E_M", em), ev("E_P", ep_compare()))["outcome"] == "SUPPORTED"
    em["holds"][0]["known_ns"]["candidate"] -= 1
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em), ev("E_P", ep_compare()))
    assert out["outcome"] == "NOT_SUPPORTED" and out["reasons"] == ["pooled_facing"]
    # an excluded hold counts in no criterion
    em = em_compare()
    em["holds"][4].update(excluded=True, at_mark={"baseline": "acquired", "candidate": "none"})
    assert mpe_outcome.evaluate(FROZEN, ev("E_M", em), ev("E_P", ep_compare()))["outcome"] == "SUPPORTED"


def test_the_record_holds_labels_and_e_m_numbers_and_names_the_git_date_and_no_adoption() -> None:
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), {**ev("E_P", ep_compare()),
                                                                  "launch_utc": "2026-11-02T10:05:00Z"})
    lines = mpe_outcome.record_lines(out, PIN, datetime(2026, 11, 3, tzinfo=timezone.utc))
    # 7 days after the launch plus 2 h (a bound on the run's end): never earlier than 7 days after collection
    assert "outcome=SUPPORTED" in lines and "e_p_counts_enter_git_not_before=2026-11-09T12:05Z" in lines
    out["sessions"]["E_P"]["launch_utc"] = None  # no launch recorded: a day after the session's start, later still
    assert "e_p_counts_enter_git_not_before=2026-11-10T12:00Z" in mpe_outcome.record_lines(out, PIN, datetime.now())
    assert "criterion_natural_gain_4_0m=met (+25.0 points; bar >= +20)" in lines
    assert any(line.startswith("adoption=not_authorized_by_this_outcome") for line in lines)
    assert not any("similarity" in line or "idn-" in line or "compared" in line.split("=")[0] for line in lines)


# --- the sessions as the package leaves them ---------------------------------------------------------------------------

def make_session(sessions: Path, data: Path | None, code: str, ts: str, *, pin: str = PIN, sealed: bool = True,
                 prov: tuple[str, ...] = ("shell_pid=1", "shell_start=1", "shell_boot_id=none")) -> Path:
    folder = sessions / f"mp-{code}-{ts}"
    folder.mkdir(mode=0o700, parents=True)
    (folder / "provenance.txt").write_text("\n".join((f"package_pin={pin}", *prov)) + "\n")
    (folder / "run.jsonl").write_text('{"synthetic": true}\n')
    if data is not None:
        (data / folder.name / "mv").mkdir(parents=True)
        (data / folder.name / "mv" / "state.db").write_bytes(b"synthetic")
    if sealed:
        seal(folder)
    return folder


def seal(folder: Path) -> None:
    import hashlib
    rows = sorted("./" + str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (folder / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((folder / r[2:]).read_bytes()).hexdigest()}  {r}\n"
                                               for r in rows))


def compared_session(sessions: Path, code: str, ts: str, record: dict, *, passed: bool = True) -> Path:
    folder = make_session(sessions, None, code, ts, sealed=False,
                          prov=("MPE_exit=0" if passed else "MPE_exit=1", "MX_started=yes", "MX_shown=yes"))
    (folder / "mv").mkdir()
    (folder / "mv" / "check.txt").write_text("PASS something\nmpe-execution: passed\n" if passed else
                                             "FAIL something\nmpe-execution: not passed (1 item)\n")
    (folder / "compare.json").write_text(json.dumps(record))
    seal(folder)
    return folder


def test_the_outcome_reads_only_sealed_compared_sessions_whose_seal_and_execution_check_hold(tmp_path) -> None:
    sessions = tmp_path / "sessions"
    em = compared_session(sessions, "em", "20261102T090000Z", em_compare())
    ep = compared_session(sessions, "ep", "20261102T100000Z", ep_compare())
    make_session(sessions, None, "ep", "20261102T095000Z")  # an earlier E_P that stopped before its comparison
    make_session(sessions, None, "em", "20261102T080000Z", pin="b" * 40)  # another package's
    assert mpe_outcome.evidence(sessions, PIN, "E_M")["state"] == "compared"
    assert mpe_outcome.evidence(sessions, PIN, "E_P")["name"] == ep.name
    (em / "compare.json").chmod(0o600)
    (em / "compare.json").write_text(json.dumps(em_compare(natural=(0.2, 0.9))))  # edited after the seal
    assert mpe_outcome.evidence(sessions, PIN, "E_M")["state"] == "seal_not_verified"
    (em / "compare.json").write_text(json.dumps(em_compare()))
    (em / "extra.txt").write_text("added after the seal")
    assert mpe_outcome.evidence(sessions, PIN, "E_M")["state"] == "seal_not_verified"
    failed = compared_session(tmp_path / "s2", "ep", "20261102T100000Z", ep_compare(match=4), passed=False)
    assert mpe_outcome.evidence(tmp_path / "s2", PIN, "E_P")["state"] == "execution_not_passed"
    # not passed: nothing of it is read, so its match votes reject nothing
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), mpe_outcome.evidence(tmp_path / "s2", PIN, "E_P"))
    assert out["outcome"] == "INCONCLUSIVE" and failed.exists()
    compared_session(tmp_path / "s2", "ep", "20261102T110000Z", ep_compare())
    assert mpe_outcome.evidence(tmp_path / "s2", PIN, "E_P")["state"] == "compared_more_than_once"


def test_the_outcome_record_is_written_once_and_needs_both_sessions_or_the_typed_end(tmp_path, capsys) -> None:
    sessions = tmp_path / "sessions"
    frozen = tmp_path / "candidate.frozen.json"
    frozen.write_text(json.dumps(FROZEN))
    out = tmp_path / "mp-outcome-aaaaaaa.txt"
    compared_session(sessions, "em", "20261102T090000Z", em_compare())
    args = ["record", "--sessions", str(sessions), "--pin", PIN, "--frozen", str(frozen), "--out", str(out)]
    assert mpe_outcome.main(args) == 1 and not out.exists()
    assert "both evaluation sessions are not compared" in capsys.readouterr().out
    assert mpe_outcome.main([*args, "--end"]) == 0
    text = out.read_text()
    assert "outcome=INCONCLUSIVE" in text and "reason=e_p_not_run" in text and oct(out.stat().st_mode)[-3:] == "400"
    compared_session(sessions, "ep", "20261102T100000Z", ep_compare(match=1))
    assert mpe_outcome.main(args) == 1 and out.read_text() == text
    assert "recorded already" in capsys.readouterr().out
    out.unlink()
    assert mpe_outcome.main(args) == 0 and "outcome=REJECTED" in out.read_text()


# --- decision 11: deadlines from collection ----------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path):
    sessions, data = tmp_path / "runs", tmp_path / "data"
    root, identity = data / "mpe-aaaaaaa", data / "identity"
    for d in (sessions, root, identity):
        d.mkdir(parents=True, mode=0o700)
    (identity / "gallery.sealed").write_bytes(b"the original enrollment (synthetic)")
    return type("Home", (), {"sessions": sessions, "data": data, "root": root, "identity": identity})


def at(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def frozen_candidate(root: Path, b: str) -> Path:
    manifest = mpe_manifest.Manifest(root)
    candidate = root / f"candidate-mp-b-{b}"
    manifest.reserve(candidate, "candidate", "gallery", "test: the build")
    candidate.mkdir(mode=0o700)
    (candidate / "gallery.sealed").write_bytes(b"synthetic candidate")
    (candidate / "inbox").mkdir(mode=0o700)
    mpe_manifest.claim(manifest, candidate, writer_exited=True)
    record = root / mpe_manifest.FROZEN_NAME
    record.write_text(json.dumps({**FROZEN, "candidate": candidate.name}))
    manifest.add(record, "output", "test: the build")
    return candidate


def deadlines(home) -> dict[str, datetime]:
    return {f"{i.what}:{i.path.name}": i.deadline for i in mpe_retention.items(home.root, home.sessions, home.data, PIN)}


def test_every_deadline_runs_from_the_sessions_own_start(home) -> None:
    b = make_session(home.sessions, home.data, "b", "20261101T090000Z")
    em = make_session(home.sessions, home.data, "em", "20261102T090000Z")
    ep = make_session(home.sessions, home.data, "ep", "20261102T100000Z")
    frozen_candidate(home.root, "20261101T090000Z")
    (home.root / "raw-mp-b-20261101T090000Z").mkdir()
    got = deadlines(home)
    assert got[f"B_session_folder:{b.name}"] == got[f"B_data_folder:{b.name}"] == at("20261101T090000Z") + timedelta(90)
    assert got[f"E_M_session_folder:{em.name}"] == at("20261102T090000Z") + timedelta(90)
    assert got[f"E_P_session_folder:{ep.name}"] == got[f"E_P_data_folder:{ep.name}"] == \
        at("20261102T100000Z") + timedelta(30)
    assert got["candidate:candidate-mp-b-20261101T090000Z"] == got["frozen_record:candidate.frozen.json"] == \
        at("20261101T090000Z") + timedelta(30)
    assert got["raw_capture:raw-mp-b-20261101T090000Z"] == at("20261101T090000Z") + timedelta(hours=24)
    lines, ok = mpe_retention.check(home.root, home.sessions, home.data, PIN, now=at("20261101T120000Z"))
    assert ok and "retention: every deadline met" in lines
    lines, ok = mpe_retention.check(home.root, home.sessions, home.data, PIN, now=at("20261102T090000Z"))
    assert not ok and any("raw_capture raw-mp-b-20261101T090000Z" in line and "OVERDUE" in line for line in lines)


def test_the_close_out_deletes_only_what_is_due_logs_it_outside_and_never_another_packages(home) -> None:
    keep_b = make_session(home.sessions, home.data, "b", "20261001T090000Z")
    old_ep = make_session(home.sessions, home.data, "ep", "20261002T100000Z")
    unsealed_ep = make_session(home.sessions, home.data, "ep", "20261003T100000Z", sealed=False)  # abandoned
    other = make_session(home.sessions, home.data, "ep", "20261002T110000Z", pin="b" * 40)
    outside = home.data / "outside.txt"
    outside.write_text("never deleted")
    (old_ep / "link").symlink_to(outside)
    report = mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity,
                                     now=at("20261108T000000Z"), processes=[])
    assert not old_ep.exists() and not (home.data / old_ep.name).exists() and not unsealed_ep.exists()
    assert keep_b.exists() and (home.data / keep_b.name).exists() and other.exists() and outside.read_text()
    assert (home.identity / "gallery.sealed").exists() and report["left"] == []
    rows = mpe_retention.log_rows(home.root)
    deleted = [r for r in rows if r[0] == "deleted"]
    assert [Path(r[1]).name for r in deleted].count(old_ep.name) == 2 and all(r[3] == "deadline" for r in deleted)
    # the data folder went first: its session folder (the attribution) last
    names = [(Path(r[1]).parent.name, Path(r[1]).name) for r in deleted if Path(r[1]).name == old_ep.name]
    assert names == [("data", old_ep.name), ("runs", old_ep.name)]
    lines, ok = mpe_retention.check(home.root, home.sessions, home.data, PIN, now=at("20261108T000000Z"))
    assert ok and any("not this package's" in line and other.name in line for line in lines)


def test_a_deletion_stopped_part_way_is_finished_from_the_log(home) -> None:
    ep = make_session(home.sessions, home.data, "ep", "20261002T100000Z")
    (ep / "provenance.txt").unlink()  # its attribution went before the rest (a crash part-way)
    mpe_retention.log(home.root, "deleting", str(ep), "E_P_session_folder", "deadline", "x", "y")
    assert [i.what for i in mpe_retention.items(home.root, home.sessions, home.data, PIN)] == \
        ["E_P_session_folder", "E_P_data_folder"]
    mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity, now=at("20261108T000000Z"),
                            processes=[])
    assert not ep.exists() and not (home.data / ep.name).exists()


def test_the_candidate_goes_at_thirty_days_after_b_or_when_the_validation_is_recorded_finished(home) -> None:
    candidate = frozen_candidate(home.root, "20261101T090000Z")
    report = mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity,
                                     now=at("20261120T000000Z"), processes=[])
    assert candidate.exists() and report["deleted"] == [] and mpe_retention.closed(home.root) == []
    report = mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity,
                                     now=at("20261120T000000Z"), processes=[], validation_finished=True)
    assert not candidate.exists() and not (home.root / mpe_manifest.FROZEN_NAME).exists() and report["left"] == []
    assert mpe_retention.closed(home.root) == ["candidate_deleted_validation_finished"]
    manifest = mpe_manifest.Manifest(home.root).entries()
    assert all(e.deleted for e in manifest.values())  # the manifest keeps the proof of each deletion
    assert sorted(p.name for p in home.root.iterdir()) == ["MANIFEST.tsv", "RETENTION.tsv"]


def test_the_candidate_deadline_closes_the_experiment_and_an_edited_record_is_left(home) -> None:
    candidate = frozen_candidate(home.root, "20261001T090000Z")
    record = home.root / mpe_manifest.FROZEN_NAME
    record.write_text(record.read_text() + " ")  # no longer the file the build listed
    report = mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity,
                                     now=at("20261108T000000Z"), processes=[])
    assert not candidate.exists() and record.exists() and any("left_changed" in x for x in report["left"])
    assert mpe_retention.closed(home.root) == []  # not closed while its record remains: mp-R keeps refusing (overdue)
    lines, ok = mpe_retention.check(home.root, home.sessions, home.data, PIN, now=at("20261108T000000Z"))
    assert not ok


def test_nothing_is_deleted_while_a_session_shell_is_open_or_a_runtime_runs(home) -> None:
    old = make_session(home.sessions, home.data, "ep", "20261002T100000Z")
    stat_fields = Path("/proc/self/stat").read_text()
    start = stat_fields[stat_fields.rindex(")") + 2:].split()[19]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    make_session(home.sessions, home.data, "em", "20261107T100000Z", sealed=False,
                 prov=(f"shell_pid={os.getpid()}", f"shell_start={start}", f"shell_boot_id={boot}"))
    with pytest.raises(mpe_retention.Refused, match="still open in its shell"):
        mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity, now=at("20261108T000000Z"),
                                processes=[])
    with pytest.raises(mpe_retention.Refused, match="runtime"):
        mpe_retention.withdraw(home.root, home.sessions, home.data, PIN, home.identity, "participant",
                               date(2026, 11, 7), today=date(2026, 11, 8),
                               processes=[[b"python", b"-m", b"sentinel.cli", b"run", b"x.yaml"]])
    assert old.exists() and mpe_retention.log_rows(home.root) == []


def test_the_participants_withdrawal_deletes_every_e_p_folder_at_once_and_marks_the_outcome(home) -> None:
    b = make_session(home.sessions, home.data, "b", "20261101T090000Z")
    em = make_session(home.sessions, home.data, "em", "20261102T090000Z")
    eps = [make_session(home.sessions, home.data, "ep", ts) for ts in ("20261102T100000Z", "20261102T110000Z")]
    candidate = frozen_candidate(home.root, "20261101T090000Z")
    (home.sessions / "mp-outcome-aaaaaaa.txt").write_text("outcome=SUPPORTED\n")
    report = mpe_retention.withdraw(home.root, home.sessions, home.data, PIN, home.identity, "participant",
                                    date(2026, 11, 5), today=date(2026, 11, 8), processes=[])
    assert not any(e.exists() or (home.data / e.name).exists() for e in eps)
    assert b.exists() and em.exists() and candidate.exists() and report["within_two_days"] is False
    rows = mpe_retention.log_rows(home.root)
    assert ["withdrawal", "participant", "2026-11-05"] == rows[0][:3] and rows[0][4] == "within_2_days=no"
    assert any(r[0] == "outcome_marked_withdrawn" for r in rows)
    assert mpe_retention.closed(home.root) == ["participant_withdrawal"]


def test_the_maintainers_withdrawal_deletes_the_candidate_and_every_session_but_never_the_original(home) -> None:
    folders = [make_session(home.sessions, home.data, c, ts) for c, ts in
               (("b", "20261101T090000Z"), ("em", "20261102T090000Z"), ("ep", "20261102T100000Z"))]
    other = make_session(home.sessions, home.data, "b", "20261101T080000Z", pin="b" * 40)
    frozen_candidate(home.root, "20261101T090000Z")
    original = home.data / "original"
    original.mkdir()
    (original / "gallery.sealed").write_bytes(b"synthetic original")
    mpe_manifest.make_working_copy(mpe_manifest.Manifest(home.root), original,
                                   home.root / "wc-mp-ep-20261102T100000Z", "test: an abandoned session")
    report = mpe_retention.withdraw(home.root, home.sessions, home.data, PIN, original, "maintainer",
                                    date(2026, 11, 8), today=date(2026, 11, 8), processes=[])
    assert report["left"] == [] and report["within_two_days"] is True
    assert not any(f.exists() or (home.data / f.name).exists() for f in folders)
    assert other.exists() and (original / "gallery.sealed").read_bytes() == b"synthetic original"
    assert sorted(p.name for p in home.root.iterdir()) == ["MANIFEST.tsv", "RETENTION.tsv"]
    assert mpe_retention.closed(home.root)[-1] == "maintainer_withdrawal"
    with pytest.raises(mpe_retention.Refused, match="future"):
        mpe_retention.withdraw(home.root, home.sessions, home.data, PIN, original, "maintainer", date(2026, 11, 9),
                               today=date(2026, 11, 8), processes=[])


def test_a_folder_reached_through_a_link_or_not_named_as_a_session_is_never_deleted(home, tmp_path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    (home.sessions / "mp-ep-20261002T100000Z").symlink_to(target)
    assert mpe_retention.items(home.root, home.sessions, home.data, PIN) == []
    with pytest.raises(mpe_retention.Refused):
        mpe_retention._delete_tree(home.sessions / "mp-ep-20261002T100000Z", home.sessions, ())
    with pytest.raises(mpe_retention.Refused):
        mpe_retention._delete_tree(home.identity, home.data, ())
    (home.data / "mp-ep-20261002T100000Z").mkdir()
    with pytest.raises(mpe_retention.Refused):
        mpe_retention._delete_tree(home.data / "mp-ep-20261002T100000Z", home.data, (home.data / "mp-ep-20261002T100000Z",))
    assert (target / "keep.txt").read_text() == "keep"


def test_retiring_a_candidate_needs_a_record_that_names_one(home) -> None:
    (home.root / mpe_manifest.FROZEN_NAME).write_text(json.dumps({"format": "mpe-frozen-1"}))
    with pytest.raises(mpe_manifest.ManifestError, match="names no candidate"):
        mpe_manifest.retire_candidate(mpe_manifest.Manifest(home.root))


def test_the_cli_check_exits_nonzero_only_when_a_deadline_has_passed(home, capsys) -> None:
    make_session(home.sessions, home.data, "ep", "20200102T100000Z")
    args = ["--root", str(home.root), "--sessions", str(home.sessions), "--data", str(home.data), "--pin", PIN]
    assert mpe_retention.main(["check", *args]) == 1
    assert "OVERDUE" in capsys.readouterr().out
    assert mpe_retention.main(["closed", "--root", str(home.root)]) == 1
    assert mpe_retention.main(["close-out", *args, "--identity-dir", str(home.identity)]) == 0
    assert mpe_retention.main(["check", *args]) == 0


def test_e_m_never_runs_on_bs_local_day(tmp_path, capsys, monkeypatch) -> None:
    frozen = tmp_path / "candidate.frozen.json"
    b = datetime.now(timezone.utc).replace(microsecond=0)
    frozen.write_text(json.dumps({**FROZEN, "candidate": f"candidate-mp-b-{b:%Y%m%dT%H%M%SZ}"}))
    assert mpe_outcome.main(["day-check", "--frozen", str(frozen)]) == 1  # B today: refused
    earlier = b - timedelta(days=1)
    frozen.write_text(json.dumps({**FROZEN, "candidate": f"candidate-mp-b-{earlier:%Y%m%dT%H%M%SZ}"}))
    assert mpe_outcome.main(["day-check", "--frozen", str(frozen)]) == 0
    frozen.write_text(json.dumps({**FROZEN, "candidate": "candidate-elsewhere"}))
    assert mpe_outcome.main(["day-check", "--frozen", str(frozen)]) == 1  # unknown B: refused
    assert "b_day=unknown" in capsys.readouterr().out
    # the local calendar day decides: 23:30 the day before in this time zone is another day
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    import time
    time.tzset()
    try:
        assert mpe_outcome.b_day({"candidate": "candidate-mp-b-20261107T190000Z"}) == date(2026, 11, 8)
        assert mpe_outcome.b_day({"candidate": "candidate-mp-b-20261107T180000Z"}) == date(2026, 11, 7)
    finally:
        monkeypatch.undo()
        time.tzset()


def passed_uncompared(sessions: Path, ts: str, *, votes: int, others: str = "no", digest: str = PROTO) -> Path:
    """An E_P session that passed its execution check and was sealed before its comparison (mp-C after a lost shell)."""
    folder = make_session(sessions, None, "ep", ts, sealed=False, prov=("MPE_exit=0",))
    run = folder / "mv"
    run.mkdir()
    (run / "check.txt").write_text("PASS something\nmpe-execution: passed\n")
    (run / "mpe-account.txt").write_text(f"format=mpe-account-1\nsession=E_P\nothers_in_view={others}\nnot_as_cued=none\n")
    person = lambda decision: {"label": "unknown", "vote": "unknown", "track_id": 1, "shadow": {"decision": decision}}
    rows = [{"run": "starting", "startup": {"face": {"shadow": True, "shadow_digest": digest, "lives": True}}}]
    rows += [{"identity": "result", "outcome": "applied", "persons": [person("match" if i < votes else "unknown")]}
             for i in range(5)]
    rows += [{"identity": "result", "outcome": "stale", "persons": [person("match")]}]  # never applied: not counted
    (run / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    seal(folder)
    return folder


def test_a_participant_match_in_a_session_sealed_before_its_comparison_still_rejects(tmp_path) -> None:
    sessions = tmp_path / "sessions"
    passed_uncompared(sessions, "20261102T100000Z", votes=2)
    compared_session(sessions, "ep", "20261102T110000Z", ep_compare())  # a clean later E_P never hides it
    ep = mpe_outcome.evidence(sessions, PIN, "E_P")
    assert ep["state"] == "compared" and ep["passed_uncompared"][0]["candidate_match_votes"] == 2
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), ep)
    assert out["outcome"] == "REJECTED" and out["reasons"][0] == "e_p_candidate_match_on_the_participant"
    alone = tmp_path / "alone"
    passed_uncompared(alone, "20261102T100000Z", votes=1)
    assert mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), mpe_outcome.evidence(alone, PIN, "E_P"))[
        "outcome"] == "REJECTED"
    for name, kwargs in (("others", {"others": "yes"}), ("digest", {"digest": "q" * 64})):
        other = tmp_path / name
        passed_uncompared(other, "20261102T100000Z", votes=3, **kwargs)
        out = mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), mpe_outcome.evidence(other, PIN, "E_P"))
        assert out["outcome"] == "INCONCLUSIVE" and any("not_attributable" in r for r in out["reasons"])
    none = tmp_path / "none"
    passed_uncompared(none, "20261102T100000Z", votes=0)
    out = mpe_outcome.evaluate(FROZEN, ev("E_M", em_compare()), mpe_outcome.evidence(none, PIN, "E_P"))
    assert out["outcome"] == "INCONCLUSIVE" and out["reasons"] == ["e_p_not_compared"]


def test_no_outcome_is_recorded_while_an_evaluation_session_is_unsealed(tmp_path, capsys) -> None:
    sessions = tmp_path / "sessions"
    frozen = tmp_path / "candidate.frozen.json"
    frozen.write_text(json.dumps(FROZEN))
    compared_session(sessions, "em", "20261102T090000Z", em_compare())
    open_ep = compared_session(sessions, "ep", "20261102T100000Z", ep_compare(match=2))
    (open_ep / "SHA256SUMS").unlink()  # its shell was lost before mp-H
    out = tmp_path / "mp-outcome-aaaaaaa.txt"
    args = ["record", "--sessions", str(sessions), "--pin", PIN, "--frozen", str(frozen), "--out", str(out), "--end"]
    assert mpe_outcome.main(args) == 1 and not out.exists() and "is not sealed" in capsys.readouterr().out
    seal(open_ep)
    assert mpe_outcome.main(args) == 0 and "outcome=REJECTED" in out.read_text()


def test_the_candidates_deletion_closes_the_experiment_even_without_its_log_row(home) -> None:
    frozen_candidate(home.root, "20261101T090000Z")
    assert mpe_retention.closed(home.root) == []
    mpe_manifest.retire_candidate(mpe_manifest.Manifest(home.root))  # a power cut before the log's row
    assert mpe_retention.closed(home.root) == ["candidate_deleted"]
    (home.root / "MANIFEST.tsv").write_text("garbage\n")
    assert mpe_retention.closed(home.root) == ["manifest_unreadable"]


def test_a_recovery_that_stops_never_keeps_the_overdue_folders(home) -> None:
    old = make_session(home.sessions, home.data, "ep", "20261002T100000Z")
    (home.root / "MANIFEST.tsv").write_text("not an event\n")
    report = mpe_retention.close_out(home.root, home.sessions, home.data, PIN, home.identity,
                                     now=at("20261108T000000Z"), processes=[])
    assert not old.exists() and any(x.startswith("recovery stopped: ManifestError") for x in report["left"])
    report = mpe_retention.withdraw(home.root, home.sessions, home.data, PIN, home.identity, "maintainer",
                                    date(2026, 11, 8), today=date(2026, 11, 8), processes=[])
    assert mpe_retention.closed(home.root)[0] == "maintainer_withdrawal" and report["left"]
