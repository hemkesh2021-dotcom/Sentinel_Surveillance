"""Where the time went before a track became KNOWN (session 116, part 28; measurement only).

The offline reading (benchmarks/runner/identity_latency.py) on synthetic records whose times are known exactly, and on
the records the real runtime writes; and the page-read timing (sentinel.page_reads), which exists only in
--identity-retention runs. Nothing here validates a latency: these tests show the phases are taken from the right
timestamps, add up, never join two lives, and never print an identity ID or a similarity value.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: the tests' own modules, by name

import identity_latency  # noqa: E402
from identity_latency import FIRST, LIFE_RECORDS, REACQUIRED, SEGMENTS, analyse  # noqa: E402

from sentinel.config import parse_config  # noqa: E402
from sentinel.media.clock import NS_PER_SECOND  # noqa: E402
from sentinel.page_reads import (  # noqa: E402
    ASSOCIATION,
    DONE_MAX,
    IDENTITY,
    PENDING_MAX,
    UNREAD_AFTER_NS,
    PageReadLatency,
    page_read_record,
)
from test_face_runtime import NOW, FaceRun, Loop, analyze_with, vec  # noqa: E402

MS = 1_000_000
WHO, OTHER = "idn-00000000000a", "idn-00000000000b"


def s(seconds: float) -> int:
    return round(seconds * NS_PER_SECOND)


def tick(at: float, *tracks: int, label: str = "offered", epoch: int = 1) -> dict:
    return {"identity": "tick", "schema": 8, "stream_epoch": epoch, "frame_seq": round(at * 20),
            "frame_mono_ns": s(at), "tick": label, "tracks": list(tracks)}


def result(at: float, track: int, label: str, *, applied: float | None = None, processing_ms: float = 400.0,
           who: str | None = WHO, epoch: int = 1, outcome: str = "applied") -> dict:
    """One track's result on the frame at ``at``; ``label`` as the runtime writes it (match, unknown, no_face, ...)."""
    vote = {"match": "match", "unknown": "unknown"}.get(label, "none")
    ownership = {"no_face": "no_face", "ambiguous": "ambiguous", "not_analysed": "not_analysed"}.get(label, "assigned")
    return {"identity": "result", "schema": 8, "stream_epoch": epoch, "frame_seq": round(at * 20),
            "frame_mono_ns": s(at), "applied_mono_ns": s(at + 0.6 if applied is None else applied), "outcome": outcome,
            "faces": 1, "processing_ms": processing_ms, "error": None,
            "persons": [{"track_id": track, "ownership": ownership, "vote": vote,
                         "identity_id": who if vote == "match" else None, "label": label, "similarity": 0.8123}]}


def transition(at: float, track: int, frm: str, to: str, *, reason: str = "2 consistent matches",
               epoch: int = 1) -> dict:
    return {"identity": "transition", "schema": 8, "stream_epoch": epoch, "track_id": track, "from": frm, "to": to,
            "identity_id": WHO if to == "known" else None, "basis": "fresh" if to == "known" else None,
            "last_vote_age_ms": 600, "reason": reason, "mono_ns": s(at)}


def life(at: float, track: int, since: float, event: str = "began", epoch: int = 1) -> dict:
    return {"identity": "life", "schema": 8, "stream_epoch": epoch, "track_id": track, "event": event,
            "since_mono_ns": s(since), "mono_ns": s(at)}


def page_read(changed: float, published: float, read: float, track: int = 7, change: str = "identity",
              to: str = "known") -> dict:
    return {"identity": "page_read", "schema": 8, "stream_epoch": 1, "track_id": track, "watched": change, "to": to,
            "changed_mono_ns": s(changed), "published_mono_ns": s(published), "read_mono_ns": s(read)}


def read(*records: dict) -> dict:
    return analyse(json.dumps(record) for record in records)


def first_life(*, with_life: bool = True, extra: tuple[dict, ...] = ()) -> list[dict]:
    """Track 7: first measured at 10.0 s, first face tick 10.4, no face at 10.4, a non-matching face at 11.4, matches
    at 12.4 and 13.4 (applied 14.0 after 400 ms of processing), KNOWN at 14.001."""
    return [
        *([life(10.05, 7, 10.0)] if with_life else []),
        tick(10.4, 7), result(10.4, 7, "no_face"),
        tick(11.4, 7), result(11.4, 7, "unknown"),
        tick(12.4, 7), result(12.4, 7, "match"),
        tick(13.4, 7), result(13.4, 7, "match", applied=14.0, processing_ms=400.0),
        transition(14.001, 7, "uncertain", "known"),
        *extra,
    ]


# ---------------------------------------------------------------- the reading: phases from the records' own times


def test_every_phase_of_a_first_acquisition_comes_from_its_own_timestamps_and_they_add_up() -> None:
    document = read(*first_life(extra=(page_read(14.001, 14.3, 15.9),)))
    (row,) = document["rows"]
    assert (row["kind"], row["life_source"], row["track_id"], row["stream_epoch"]) == (FIRST, LIFE_RECORDS, 7, 1)
    assert row["phases_ms"] == {
        "tracking": 400.0,  # 10.0 -> the first tick with the track, 10.4
        "eligible_face": 1000.0,  # 10.4 -> the first assigned face's frame, 11.4
        "not_matching": 1000.0,  # 11.4 -> the first confirming vote's frame, 12.4
        "confirmation": 1000.0,  # 12.4 -> 13.4
        "queue_and_apply": 200.0,  # applied 14.0 - frame 13.4 - 400 ms of processing
        "inference": 400.0,
        "apply_to_state": 1.0,
    }
    assert row["since_start_ms"] == 4001.0 == sum(row["phases_ms"].values())  # life start 10.0 -> KNOWN 14.001
    assert row["adds_up"] is True and row["ordered"] is True
    assert row["results_before_a_face"] == {"no_face": 1} and row["results_before_confirming"] == {"unknown": 1}
    assert row["ui_ms"] == {"status_publish": 299.0, "page_read": 1600.0}
    assert document["acquisitions"] == {"total": 1, "complete": 1, "incomplete": {}, "not_adding_up": 0}
    summary = document["summary"][FIRST]
    assert summary["rows"] == 1 and summary["phases_ms"]["tracking"]["median"] == 400.0
    assert summary["ui_ms"]["page_read"] == {"n": 1, "n_unavailable": 0, "min": 1600.0, "median": 1600.0,
                                             "p90": 1600.0, "max": 1600.0}


def test_without_life_records_a_row_is_a_segment_and_tracking_is_unavailable_not_guessed() -> None:
    document = read(*first_life(with_life=False))
    (row,) = document["rows"]
    assert row["life_source"] == SEGMENTS == document["run"]["life_source"]
    assert row["phases_ms"]["tracking"] is None and row["tracking_unavailable"] == "no_life_records"
    assert row["phases_ms"]["eligible_face"] == 1000.0 and row["since_start_ms"] == 3601.0  # from the first tick
    assert row["adds_up"] is True
    assert row["ui_ms"] is None and row["ui_unavailable"] == "no_page_read_records"
    assert document["summary"][FIRST]["phases_ms"]["tracking"] == {"n": 0, "n_unavailable": 1}
    assert any("not a proven continuous life" in note for note in document["notes"])


def test_a_segment_ends_where_the_number_was_cleared_and_where_its_ticks_stop() -> None:
    earlier = [tick(1.0, 7), result(1.0, 7, "unknown"), tick(2.0, 7), result(2.0, 7, "unknown")]
    cleared = read(*earlier, transition(3.0, 7, "unknown", "cleared", reason="track no longer current"),
                   *first_life(with_life=False))
    gap = read(*earlier, *first_life(with_life=False))  # no clear recorded, but its ticks stop for 8.4 s
    for document in (cleared, gap):
        (row,) = document["rows"]
        assert row["since_start_ms"] == 3601.0  # from 10.4: nothing of the earlier stretch is joined to it
        assert row["results_before_a_face"] == {"no_face": 1} and row["results_before_confirming"] == {"unknown": 1}
    joined = read(tick(6.4, 7), result(6.4, 7, "unknown"), *first_life(with_life=False))  # 4 s: within the gap limit
    assert joined["rows"][0]["since_start_ms"] == 7601.0


def test_a_number_reused_on_a_second_life_is_never_joined_to_the_first() -> None:
    """Track 7 was KNOWN on an earlier life; the number comes back within a second on a new life (a life record says
    so). The new life's row starts at its own first measurement and is a first acquisition, not a reacquisition."""
    old = [life(1.05, 7, 1.0), tick(1.4, 7), result(1.4, 7, "match"), tick(2.4, 7), result(2.4, 7, "match"),
           transition(3.001, 7, "uncertain", "known"), life(9.9, 7, 1.0, "ended")]
    document = read(*old, *first_life())
    first, second = document["rows"]
    assert (first["kind"], second["kind"]) == (FIRST, FIRST)
    assert second["since_start_ms"] == 4001.0 and second["phases_ms"]["tracking"] == 400.0
    assert second["results_before_confirming"] == {"unknown": 1}  # none of the old life's matches


def test_a_late_result_of_the_old_life_is_not_a_confirming_vote() -> None:
    """A result the raw rule rejected as from before the track's expiry (before_expiry) confirms nothing."""
    records = first_life()
    records.insert(3, result(9.0, 7, "before_expiry", applied=10.9))
    (row,) = read(*records)["rows"]
    assert row["phases_ms"]["not_matching"] == 1000.0 and row["adds_up"] is True
    assert "before_expiry" not in row["results_before_a_face"]


def test_an_acquisition_whose_confirming_votes_the_records_do_not_hold_is_counted_incomplete_never_guessed() -> None:
    missing = read(life(10.05, 7, 10.0), tick(10.4, 7), result(10.4, 7, "match"),
                   transition(14.0, 7, "uncertain", "known"))  # one match only: the other is not in the records
    two_people = read(life(10.05, 7, 10.0), tick(10.4, 7), result(10.4, 7, "match"), tick(11.4, 7),
                      result(11.4, 7, "match", who=OTHER), transition(12.0, 7, "uncertain", "known"))
    no_count = read(*first_life()[:-1], transition(14.001, 7, "uncertain", "known", reason="something else"))
    no_life = read(life(0.5, 3, 0.4), *first_life(with_life=False))  # life records exist, but none for track 7
    for document, reason in ((missing, "confirming_votes_not_found"), (two_people, "confirming_votes_not_found"),
                             (no_count, "reason_without_a_count"), (no_life, "life_record_missing")):
        assert document["acquisitions"] == {"total": 1, "complete": 0, "incomplete": {reason: 1}, "not_adding_up": 0}
        assert document["rows"][0]["incomplete"] == reason and "phases_ms" not in document["rows"][0]
        assert document["summary"][FIRST]["rows"] == 0


def test_a_reacquisition_is_timed_from_the_loss_without_the_first_three_phases() -> None:
    document = read(*first_life(), tick(14.4, 7), result(14.4, 7, "unknown", applied=15.0),
                    transition(15.001, 7, "known", "uncertain", reason="contradictory face evidence"),
                    tick(15.4, 7), result(15.4, 7, "no_face", applied=16.0),
                    tick(16.4, 7), result(16.4, 7, "match", applied=17.0),
                    tick(17.4, 7), result(17.4, 7, "match", applied=18.2, processing_ms=500.0),
                    transition(18.202, 7, "uncertain", "known"))
    first, again = document["rows"]
    assert (first["kind"], again["kind"]) == (FIRST, REACQUIRED)
    assert again["phases_ms"] == {"wait_for_match": 1399.0,  # the loss at 15.001 -> the first confirming frame, 16.4
                                  "confirmation": 1000.0, "queue_and_apply": 300.0, "inference": 500.0,
                                  "apply_to_state": 2.0}
    assert again["since_loss_ms"] == 3201.0 == sum(again["phases_ms"].values()) and again["adds_up"] is True
    assert again["results_since_loss"] == {"no_face": 1}
    assert document["summary"][REACQUIRED]["rows"] == 1 and document["summary"][FIRST]["rows"] == 1


def test_times_out_of_order_are_reported_and_left_out_of_the_summary() -> None:
    records = first_life()
    records[-1] = transition(13.9, 7, "uncertain", "known")  # before its own confirming result was applied (14.0)
    document = read(*records)
    assert document["rows"][0]["ordered"] is False and document["rows"][0]["adds_up"] is False
    assert document["acquisitions"]["not_adding_up"] == 1 and document["summary"][FIRST]["rows"] == 0


def test_the_run_wide_numbers_come_from_every_applied_result_and_offered_tick() -> None:
    document = read(*first_life(), tick(14.4, label="skipped_no_person"),
                    result(15.0, 7, "match", outcome="rejected_age"))
    results = document["results"]
    assert results["outcomes"] == {"applied": 4, "rejected_age": 1}
    assert results["ticks"] == {"offered": 4, "skipped_no_person": 1}
    assert results["inference_ms"]["n"] == 4 and results["inference_ms"]["median"] == 400.0
    assert results["queue_and_apply_ms"]["max"] == 200.0
    assert results["offered_tick_interval_ms"] == {"n": 3, "n_unavailable": 0, "min": 1000.0, "median": 1000.0,
                                                   "p90": 1000.0, "max": 1000.0}


def test_a_change_no_page_read_says_so_and_an_association_before_is_reported_only_from_its_records() -> None:
    unread = read(*first_life(extra=(page_read(99.0, 99.2, 99.9, track=3),)))  # the run has page reads, not this one
    assert unread["rows"][0]["ui_ms"] is None and unread["rows"][0]["ui_unavailable"] == "not_read"
    assert unread["run"]["page_read_records"] == {"identity:known": 1}
    assert "association_before" not in unread["rows"][0]
    retained = {"identity": "association", "schema": 8, "stream_epoch": 1, "track_id": 7, "from": "current",
                "to": "retained", "identity_id": WHO, "reason": "no_current_match", "last_match_age_ms": 1000,
                "contradictions": 1, "mono_ns": s(11.0)}
    records = first_life()
    records.insert(4, retained)
    assert read(*records)["rows"][0]["association_before"] == "retained"


def test_the_whole_output_carries_no_identity_id_and_no_similarity(tmp_path, capsys) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("not json\n" + "".join(json.dumps(r) + "\n" for r in [
        {"run": "starting", "startup": {}}, {"updated_utc": "2026-11-09T18:28:23+00:00", "state": "running"},
        *first_life()]))
    assert identity_latency.main([str(path)]) == 0
    text = capsys.readouterr().out
    document = json.loads(text)
    assert "idn-" not in text and "similarity" not in text and "0.8123" not in text
    assert document["run"]["unreadable_lines"] == 1 and document["run"]["status_lines"] == 1
    assert document["rows"][0]["after_status_utc"] == "2026-11-09T18:28:23+00:00"  # the status line written before it
    assert identity_latency.main([str(path), "--summary"]) == 0
    assert "rows" not in json.loads(capsys.readouterr().out)
    assert identity_latency.main([str(tmp_path / "absent.jsonl")]) == 2
    assert "unreadable" in capsys.readouterr().err


# ---------------------------------------------------------------- the page-read timing (sentinel.page_reads)


def watch(reads: PageReadLatency, at: int, track: int = 1, to: str = "known") -> None:
    reads.changed(1, track, IDENTITY, to, at, watch=True, holds=True)


def test_a_change_is_read_only_by_a_page_request_after_the_status_that_holds_it() -> None:
    reads = PageReadLatency()
    watch(reads, 100)
    reads.read(150)  # a page read before any status was published: it read the old status
    assert reads.drain() == [] and reads.numbers()["pending"] == 1
    reads.published(200)
    reads.read(199)  # a request that read before that publication
    assert reads.drain() == []
    reads.read(260)
    assert reads.drain() == [{"stream_epoch": 1, "track_id": 1, "watched": "identity", "to": "known",
                              "changed_mono_ns": 100, "published_mono_ns": 200, "read_mono_ns": 260}]
    reads.read(300)  # once only
    assert reads.drain() == []
    assert reads.numbers() == {"pending": 0, "watched": 1, "completed": 1, "superseded": 0, "unread": 0,
                               "dropped_pending": 0, "dropped_done": 0}
    assert page_read_record({"stream_epoch": 1}, 8) == {"identity": "page_read", "schema": 8, "stream_epoch": 1}


def test_one_read_completes_every_published_change_and_no_reads_mean_no_records() -> None:
    reads = PageReadLatency()
    for track in (1, 2, 3):
        watch(reads, 100 + track, track)
    reads.published(200)
    reads.changed(1, 4, ASSOCIATION, "current", 210, watch=True, holds=True)  # after that publication
    assert reads.drain() == []
    reads.read(250)
    done = reads.drain()
    assert [(item["track_id"], item["watched"]) for item in done] == [(1, "identity"), (2, "identity"), (3, "identity")]
    assert reads.numbers()["pending"] == 1  # the association waits for the next status


def test_a_change_undone_before_any_status_held_it_is_superseded_not_timed() -> None:
    reads = PageReadLatency()
    watch(reads, 100)
    reads.changed(1, 1, IDENTITY, "uncertain", 120, watch=False, holds=False)  # no status ever showed it KNOWN
    reads.changed(1, 1, IDENTITY, "known", 130, watch=False, holds=True)  # known -> known (its basis): nothing undone
    reads.published(200)
    reads.read(210)
    assert reads.drain() == [] and reads.numbers()["superseded"] == 1
    watch(reads, 300)
    reads.published(310)
    reads.changed(1, 1, IDENTITY, "unknown", 320, watch=False, holds=False)  # after a status held it: still timed
    reads.changed(1, 2, IDENTITY, "unknown", 320, watch=False, holds=False)  # another track's change: nothing of it
    reads.read(330)
    assert [item["changed_mono_ns"] for item in reads.drain()] == [300]
    with pytest.raises(ValueError):
        reads.changed(1, 1, "other", "known", 1, watch=True, holds=True)


def test_everything_is_bounded_and_what_is_dropped_is_counted() -> None:
    reads = PageReadLatency()
    for track in range(PENDING_MAX + 3):
        watch(reads, 100 + track, track)
    assert reads.numbers()["pending"] == PENDING_MAX and reads.numbers()["dropped_pending"] == 3
    reads.published(1000)
    reads.read(1000 + UNREAD_AFTER_NS + 1)  # no page was polling: not a latency
    assert reads.drain() == [] and reads.numbers()["unread"] == PENDING_MAX and reads.numbers()["pending"] == 0
    for n in range(DONE_MAX + 5):
        watch(reads, 5000 + n)
        reads.published(5000 + n)
        reads.read(5000 + n)
    assert len(reads.drain()) == DONE_MAX and reads.numbers()["dropped_done"] == 5


def test_changes_reads_and_drains_from_several_threads_lose_and_duplicate_nothing() -> None:
    reads = PageReadLatency()
    stop = threading.Event()
    seen: list[dict] = []

    def page() -> None:
        at = 0
        while not stop.is_set():
            at += 1
            reads.read(10_000_000 + at)

    readers = [threading.Thread(target=page) for _ in range(3)]
    for thread in readers:
        thread.start()
    for n in range(400):
        watch(reads, n, n)
        reads.published(n)
        seen += reads.drain()
    stop.set()
    for thread in readers:
        thread.join()
    reads.read(20_000_000)
    seen += reads.drain()
    numbers = reads.numbers()
    assert len(seen) + numbers["dropped_pending"] + numbers["dropped_done"] == 400
    assert len({item["track_id"] for item in seen}) == len(seen)


# ---------------------------------------------------------------- the real runtime's records, read by the reading


def retention_loop(tmp_path, analyze=analyze_with(vec(0.97, 0.1, 0.0, 0.05))) -> Loop:
    policy = parse_config({"config_version": 1, "camera": {"id": "cam-1"}}).identity.retention_policy()
    return Loop(tmp_path, analyze=analyze, retention=policy, perf_counter=lambda: 0.0)  # no real time in the records


def test_a_retention_runs_own_records_are_read_end_to_end_with_the_page_read(tmp_path, monkeypatch) -> None:
    import face_evidence

    seen = {"match": False}
    matching = analyze_with(vec(0.97, 0.1, 0.0, 0.05))
    loop = retention_loop(tmp_path, lambda frame, image: matching(frame, image) if seen["match"] else FaceRun((), 0, 0))
    for _ in range(20):  # the person is tracked, no face found
        loop.frame()
    seen["match"] = True
    for _ in range(31):  # two matching ticks: KNOWN
        loop.frame()
    assert loop.runtime.snapshot()["live"]["identity"]["known"] == 1
    assert loop.runtime.snapshot() is loop.runtime.snapshot()  # the plain status read is not timed...
    assert [r for r in loop.runtime.drain_identity_records() if r["identity"] == "page_read"] == []
    before = loop.runtime.snapshot()["components"]["face"]["page_reads"]
    assert before["watched"] == 2 and before["pending"] == 2 and before["completed"] == 0  # KNOWN and the association
    loop.clock.advance(1.5)
    loop.runtime.page_snapshot()  # ... a page's own read is
    loop.frame()
    records = loop.runtime.drain_identity_records()
    reads = [r for r in records if r["identity"] == "page_read"]
    assert sorted((r["watched"], r["to"]) for r in reads) == [("association", "current"), ("identity", "known")]
    # the fake clock stands still within a step: published in the step that changed it, read 1.5 s and more later
    assert all(r["published_mono_ns"] == r["changed_mono_ns"] for r in reads)
    assert len({r["read_mono_ns"] for r in reads}) == 1
    assert all(r["read_mono_ns"] - r["published_mono_ns"] >= 1_500_000_000 for r in reads)
    assert all("idn-" not in json.dumps(r) for r in reads)
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reads))
    assert face_evidence.read(out / "run.jsonl", "f2").nonconforming == []
    loop.close()


def test_the_runtimes_records_give_a_row_that_adds_up_from_the_life_start(tmp_path) -> None:
    seen = {"match": False}
    matching = analyze_with(vec(0.97, 0.1, 0.0, 0.05))
    loop = retention_loop(tmp_path, lambda frame, image: matching(frame, image) if seen["match"] else FaceRun((), 0, 0))
    records: list[dict] = []
    for _ in range(35):  # three face ticks without a face (2.3 s at 15 frames per second)
        loop.frame()
    seen["match"] = True
    for _ in range(31):
        loop.frame()
        records += loop.runtime.drain_identity_records()
    loop.clock.advance(0.7)
    loop.runtime.page_snapshot()
    loop.frame()
    records += loop.runtime.drain_identity_records()
    document = analyse(json.dumps(r) for r in records)
    assert document["run"]["life_source"] == LIFE_RECORDS
    (row,) = document["rows"]
    assert row["kind"] == FIRST and row["adds_up"] is True and row["ordered"] is True
    phases = row["phases_ms"]
    assert phases["inference"] == 0.0 and phases["tracking"] is not None  # the fake timer; the life record's start
    assert 900 <= phases["confirmation"] <= 1100  # two votes, one face tick apart
    assert phases["eligible_face"] >= 2900 and row["results_before_a_face"] == {"no_face": 3}
    assert row["ui_ms"]["status_publish"] == 0.0 and row["ui_ms"]["page_read"] >= 700.0
    assert row["association_before"] == "none"
    assert "idn-" not in json.dumps(document)
    loop.close()


def test_without_track_retention_none_of_it_exists(tmp_path) -> None:
    loop = Loop(tmp_path)
    for _ in range(31):
        loop.frame()
    assert loop.runtime.page_snapshot() is loop.runtime.snapshot()
    loop.frame()
    assert "page_reads" not in loop.runtime.snapshot()["components"]["face"]
    assert {r["identity"] for r in loop.runtime.drain_identity_records()} == {"tick", "result", "transition"}
    loop.close()


def test_the_status_server_times_only_a_pages_own_read(tmp_path) -> None:
    from sentinel.status_page import StatusServer

    calls = {"plain": 0, "page": 0}

    def plain() -> dict:
        calls["plain"] += 1
        return {"runtime": {"state": "starting", "camera_id": "cam-1"}}

    def page() -> dict:
        calls["page"] += 1
        return plain()

    server = StatusServer(0, plain, tmp_path / "absent.db", now=lambda: NOW, page_snapshot=page)
    try:
        server.status()
        assert calls == {"plain": 1, "page": 0}  # /status.json
        server.status(for_page=True)
        server.status(with_picture=True, for_page=True)
        assert calls["page"] == 2
        without = StatusServer(0, plain, tmp_path / "absent.db", now=lambda: NOW)
        try:
            assert without.status(for_page=True)["runtime"]["state"] == "starting"  # as before, with no timed path
        finally:
            without.stop(1.0)
    finally:
        server.stop(1.0)
