"""The V2-25 device check's readings (face_check.py) and the F2 account helper (face_account.py).

F2's evidence is produced by the real runtime: DemoRuntime, EdgeCore and a real FaceWorker run a scripted ten
minutes on a fake clock; the identity records come from drain_identity_records(), the status lines from the CLI's
status_line(), the captures from the status page's build_status(), the startup's face fields from assemble() and its
face release from post_load_release(), and the guard summary from step5_guard.read_run_output(). Only the camera,
the detector and DeepFace are fakes, with synthetic embeddings. F1's evidence comes from the real `sentinel identity`
commands with a real sealed gallery. So a change in the runtime's output schema fails these tests (unlike session
47's hand-written layers value).

The privacy counts come from face_counts.sh's functions, run by bash as the package's blocks run them (only the
interpreter differs), on that evidence: initials and short names against the evidence's own schema text, injected
leaks, malformed evidence and unexpected fields. The offline F1 recheck, the maintainer's acceptance and the F1 gate
for F2 are exercised on a run directory built as F1f and FH left face-20261007T215900Z.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import shutil
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from face_runs import (  # the shared real-runtime helpers (tests/support)
    BOX,
    COMPAT,
    ENROLLED,
    F2_PROVENANCE,
    FACE,
    FACE_BOX,
    GALLERY,
    GATE_LINES,
    LOOPBACK,
    NOW,
    REPO,
    REPORT,
    RUNNERS,
    TERMS,
    Capture,
    Image,
    Tracker,
    bash_counts,
    real_face_startup,
    segment,
    startup_line,
    vec,
)

from sentinel import demo_runtime
from sentinel.adapters import FACE_ADAPTER_ID, FaceAdmission, FaceAdmissionStatus
from sentinel.cli import main, status_line
from sentinel.config import IdentityConfig, parse_config
from sentinel.contracts import FrameKey, NormalizedBox, PixelFormat, StreamIdentity
from sentinel.demo_runtime import (
    DemoRuntime,
    Devices,
    FaceOptions,
    FaceRuntime,
    MemoryOps,
    RunOptions,
    SceneRuntime,
    assemble,
    file_facts,
    post_load_release,
)
from sentinel.identity.association import FaceObservation
from sentinel.identity.gallery import (
    CONSENT_SCOPE,
    FaceCompatibility,
    GalleryDocument,
    GalleryIdentity,
    IdentityStore,
    unit,
)
from sentinel.identity.legacy_deepface import EnrollmentFace, FaceRun, static_compatibility_fields
from sentinel.identity.vault import SECRET_ENV, Sealer, Secret, VaultError
from sentinel.identity.worker import TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING, FaceWorker
from sentinel.incidents.service import IncidentService
from sentinel.media.capture import CapturedFrame, CaptureState, CaptureStatus, LatestFrame
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.memory_policy import RELEASE_FILE_ROLES
from sentinel.jobs import WorkerOutcome
from sentinel.runtime import (
    IDENTITY_DIAGNOSTIC_SCHEMA,
    FaceDiagnostics,
    FaceResultRecord,
    IdentityTransition,
    tick_record,
)
from sentinel.scene.analyzer import RecentImages, ThreadedSceneAnalyzer
from sentinel.status_page import build_status, read_store
from sentinel.storage.database import Database
from sentinel.tracking.tracker import PersonTracker, RawTrack

# What the operator did (seconds after the T0 cue), and the account they would give.
SCRIPT = ((0, 12, "facing", 3), (12, 120, "absent", None), (120, 240, "facing", 1), (240, 360, "away", 1),
          (360, 480, "absent", None), (480, 601, "facing", 2))
ACCOUNT = {"stopwatch_started_at_t0": "yes", "out_of_view_1": "0:12", "seated_facing_1": "2:00",
           "turned_away": "4:00", "face_hidden_while_turned": "yes", "stood_up": "6:00", "out_of_view_2": "6:01",
           "seated_facing_2": "8:00", "stayed_until_stop": "yes", "other_faces_in_view": "no",
           "notes": "a note that is never printed"}
CAPTURES = ((60.0, "e1"), (180.0, "k1"), (300.0, "a"), (420.0, "e2"), (540.0, "k2"), (590.0, "end"))


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(check=importlib.import_module("face_check"), account=importlib.import_module("face_account"),
                           guard=importlib.import_module("step5_guard"), evidence=importlib.import_module("face_evidence"),
                           privacy=importlib.import_module("face_privacy"), recheck=importlib.import_module("face_recheck"))


# ---------------------------------------------------------------- the scripted live run


def simulate(root: Path, startup: dict, *, script=SCRIPT, away_face=None, fps: float = 4.0,
             duration: float = 600.0, combined: bool = False, analyze=None, comparator=None,
             captures=CAPTURES, detection: str = "deepface", shadow=None, experiment_capture=None,
             preview: bool = False) -> Path:
    """Run the real runtime through ``script`` and write the F2 directory as the package's blocks would.

    ``combined`` (D78): with the live view, incident snapshots and chat in a validation run as well, each used during
    the run (pictures made, a question asked now and then), and the status captures as the plain /status.json serves
    them (the live view's state included). ``preview`` (D95's dashboard check): the live view alone, likewise."""
    out = root / "f2"
    out.mkdir()
    clock = FakeClock(utc=NOW)
    config = parse_config({"config_version": 1, "camera": {"id": "cam-1"}})
    slot, capture, tracker = LatestFrame(), Capture(clock), Tracker()
    t0 = clock.monotonic_ns()

    analyze_override = analyze

    def analyze(frame, image):
        state, _ = segment(script, (frame.ingest_mono_ns - t0) / 1e9)
        embedding = FACE if state == "facing" else away_face if state == "away" else None
        if embedding is None:
            return FaceRun((), 0, 1)  # DeepFace's no-face fallback, dropped
        return FaceRun((FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=embedding),), 1, 0)

    if analyze_override is not None:
        analyze = analyze_override(t0)
    worker = FaceWorker(analyze, clock, comparator=comparator)
    images = RecentImages()
    scene = SceneRuntime(ThreadedSceneAnalyzer(lambda job, image: WorkerOutcome.completed(job.job_id, REPORT), images,
                                               revision="1+lfm2-vl-1.6b-q4_0"), images)
    db = Database.open(root / "sentinel.db")
    extra: dict[str, Any] = {}
    store = None
    if combined:
        from sentinel.incidents.snapshots import SnapshotStore, SnapshotWriter
        from sentinel.media.jpeg import EncodedJpeg
        from sentinel.scene.chat import CHAT_ADMISSION

        store = SnapshotStore(db, root / "snapshots", clock, encode=lambda *a, **k: EncodedJpeg(b"\xff\xd8s\xff\xd9", 8, 8),
                              disk_usage=lambda path: SimpleNamespace(total=10**11, free=5 * 10**10))
        opened = store.open()
        extra = dict(live_preview=True, encode=lambda *a, **k: EncodedJpeg(b"\xff\xd8p\xff\xd9", 8, 8),
                     snapshots=SnapshotWriter(store, clock), chat=True, chat_admission=CHAT_ADMISSION,
                     chat_validation=True, chat_run=lambda question, image, abort=None: ("A chair.", False),
                     pressure=lambda: 2_500_000_000)
    elif preview:
        from sentinel.media.jpeg import EncodedJpeg

        extra = dict(live_preview=True, encode=lambda *a, **k: EncodedJpeg(b"\xff\xd8p\xff\xd9", 8, 8))
    runtime = DemoRuntime(config, clock, capture=capture, slot=slot,
                          incidents=IncidentService(db, clock, incidents=config.incidents,
                                                    notifications=config.notifications),
                          tracker=PersonTracker(tracker), scene=scene,
                          face=FaceRuntime(worker, 1, True, comparator is not None, detection, shadow,
                                           experiment_capture),
                          enrollment=GALLERY.enrollment(), status_refresh_s=0.0, **extra)
    runtime.start()
    if combined:  # what assemble adds to the starting line for these flags, from the same objects
        startup = {**startup, "startup": {**startup["startup"], "live_preview": runtime.preview.bounds(),
                                          "snapshots": {**opened, "policy": store.policy.describe()},
                                          "chat": runtime.chat.availability()}}
    elif preview:
        startup = {**startup, "startup": {**startup["startup"], "live_preview": runtime.preview.bounds()}}
    lines = [json.dumps(startup)]
    pending, captures = list(captures), []
    next_status = 0.0
    for n in range(1, int(duration * fps) + 1):
        t = n / fps
        state, track = segment(script, t)
        tracker.next = [] if state == "absent" else [RawTrack(track, *BOX, 0.9)]
        clock.advance(ns=t0 + round(n * 1e9 / fps) - clock.monotonic_ns())
        frame = capture.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        slot.publish(CapturedFrame(frame, Image()))
        runtime.step(timeout_s=0)
        deadline = time.monotonic() + 5
        while worker.status()["busy"] or worker.status()["waiting"]:
            assert time.monotonic() < deadline
            time.sleep(0.0005)
        runtime.step(timeout_s=0)
        lines += [json.dumps(record) for record in runtime.drain_identity_records()]
        if t >= next_status:
            lines.append(json.dumps(status_line(runtime.snapshot())))
            next_status += 30.0
        if preview and n % int(fps * 15) == 0:  # a viewer: a picture made every 15 s
            runtime.preview.view(with_picture=True)
        if combined and n % int(fps * 15) == 0:  # a picture made and a question asked every 15 s
            runtime.preview.view(with_picture=True)
            from sentinel.scene.chat import ChatQuestion
            runtime.chat.ask(ChatQuestion("What is visible?"))
        while pending and t >= pending[0][0]:
            at, name = pending.pop(0)
            status = build_status(runtime.snapshot(), read_store(root / "sentinel.db"), clock.utc_now(),
                                  runtime.preview.view() if combined or preview else None)
            (out / f"status-{name}.json").write_text(json.dumps(status))
            captures.append({"name": name, "at_s": at, "status": "saved", "listeners": LOOPBACK})
    shutdown = runtime.shutdown()
    lines += [json.dumps(record) for record in runtime.drain_identity_records()]
    lines.append(json.dumps({"run": "stopped", "shutdown": shutdown, "database_closed": True,
                             "status": status_line(runtime.snapshot())}))
    db.close()
    (out / "run.jsonl").write_text("\n".join(lines) + "\n")
    return out


def finish(mods, out: Path, *, cue_delay_s: float = 0.1, account: dict[str, str] | None = ACCOUNT,
           counts: dict[str, int] | None = None, cues: dict[str, float] | None = None, cued_format: int = 3,
           duration_s: float = 600.0, captures=CAPTURES, provenance: str = F2_PROVENANCE) -> None:
    """result.json as the guard writes it, then the account and the counts, in the package's order (the counts by
    face_counts.sh unless ``counts`` fakes them); ``provenance`` is what the launch block wrote before the account."""
    ready = 1.0  # FakeClock's default start (seconds), the runtime's T0
    result = {
        "schema_version": 1, "mode": "step5_guard", "status": "duration_stop",
        "parameters": {"duration_s": duration_s, "cues": [{"at_s": 0.0, "name": "t0"}]},
        "child": {"returncode": 0, "ready_after_s": 70.0, "ready_mono_s": ready},
        "guard": {"trigger": None, "peak_pressure_bytes": 1, "min_mem_free_bytes": 2, "min_mem_available_bytes": 3},
        "stop": {"forced": False}, "cleanup_clear": True,
        "leftovers": {"llama_server_after_exit": 0, "llama_server_left": 0, "runtime_group_left": False,
                      "listeners_after": []},
        "captures": json.loads((out / "captures.json").read_text()) if (out / "captures.json").exists() else
        [{"name": name, "at_s": at, "status": "saved", "listeners": LOOPBACK} for at, name in captures],
        "cues": [{"name": name, "at_s": float(at), "printed_after_launch_s": 70.0 + at + cue_delay_s,
                  "printed_mono_s": ready + at + cue_delay_s} for name, at in (cues or {"t0": 0.0}).items()],
        "run_output": mods.guard.read_run_output(out / "run.jsonl"),
    }
    (out / "result.json").write_text(json.dumps(result))
    (out / "provenance.txt").write_text(provenance)
    if account is not None:
        save_account(mods, out, account, cued_format)
    if counts is None:  # F2e's counts, by the real workflow
        for function, name in (("face_counts_camera", "secret-counts-camera.txt"),
                               ("face_counts_f2", "secret-counts-identity.txt")):
            done = bash_counts(function, str(out))
            assert done.returncode == 0, done.stderr
            (out / name).write_text(done.stdout)
        return
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in mods.check.NOT_COUNTED)
    (out / "secret-counts-camera.txt").write_text("".join(f"{n} userinfo={counts.get('userinfo', 0)}\n" for n in names))
    (out / "secret-counts-identity.txt").write_text("".join(
        f"{n} passphrase={counts.get('passphrase', 0)}\n{n} name={counts.get('name', 0)}\n"
        f"{n} nonconforming={counts.get('nonconforming', 0)}\n" for n in names))


def save_account(mods, out: Path, account: dict[str, str], cued_format: int = 3) -> None:
    """A format-1 account (the face-0f7855e package's) as that helper saved and locked it; a format-2 account through
    the current helper's prompts (an unknown reading's uncertainty is not asked); a cued account (format 3, or 4 when
    asked: the same keys) through the helper's prompts."""
    if cued_format in (5, 6, 7, 8):  # the natural check's (D86), the comparison's (D87) or D91's, by prompts
        fields = {5: mods.account.FIELDS_V5, 6: mods.account.FIELDS_V6, 7: mods.account.FIELDS_V7,
                  8: mods.account.FIELDS_V8}[cued_format]
        answers = iter([account[f.key] for f in fields] + ["SAVE"])
        assert mods.account.enter(out, ask=lambda prompt: next(answers), say=lambda text: None,
                                  account_format_number=cued_format) == 0
        return
    if any(key.endswith("_within_20_s") for key in account):  # format 3 (D79) or 4 (D85), through the prompts
        fields = {3: mods.account.FIELDS_V3, 4: mods.account.FIELDS_V4}[cued_format]
        answers = iter([account[f.key] for f in fields] + ["SAVE"])
        assert mods.account.enter(out, ask=lambda prompt: next(answers), say=lambda text: None,
                                  account_format_number=cued_format) == 0
        return
    if "stopwatch_started_at_t0" in account:
        data = mods.account.render(account, mods.account.FIELDS_V1, mods.account.HEADER_V1).encode()
        (out / "account.txt").write_bytes(data)
        with (out / "provenance.txt").open("a") as provenance:
            provenance.write(f"account_sha256={hashlib.sha256(data).hexdigest()}\n")
        return
    answers = iter([account[f.key] for f in mods.account.FIELDS
                    if not (f.key in mods.account.PAIRED and account[mods.account.PAIRED[f.key]] == "unknown")]
                   + ["SAVE"])
    assert mods.account.enter(out, ask=lambda prompt: next(answers), say=lambda text: None) == 0


@pytest.fixture
def startup(tmp_path, accepted_scene) -> dict:
    root = tmp_path / "startup"
    root.mkdir()
    return startup_line(real_face_startup(root, accepted_scene))


@pytest.fixture
def f1_list(tmp_path) -> Path:
    f1 = tmp_path / "f1"
    f1.mkdir()
    (f1 / "list.json").write_text(json.dumps({"identity": "list", "identities": [
        {"identity_id": ENROLLED, "prototypes": 3, "consent_date": "2026-10-08", "enrolled_utc": NOW.isoformat()}]}))
    return f1


def reading(mods, out, f1):
    result = mods.check.read_f2(out, f1)
    return result, "\n".join(result.lines)


def line_for(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.split(" ", 1)[1].startswith(start))


# ---------------------------------------------------------------- F2


@pytest.mark.parametrize("change", ["startup", "records"])
def test_f2_never_reads_a_continuity_run_by_its_plain_rule(mods, tmp_path, startup, f1_list, change) -> None:
    """D97: F2's retention item encodes the plain rule (cleared when the older confirming vote is 30 s old), so a run
    with the continuity rule is never judged by it: its startup's face entry differs from F2's, and its break records
    are unknown identity records to every existing reading."""
    out = simulate(tmp_path, startup)

    def edit(lines):
        if change == "startup":
            for line in lines:
                if line.get("run") == "starting":
                    line["startup"]["face"]["continuity"] = {"max_gap_s": 0.5, "overlap_iou": 0.3,
                                                             "min_step_iou": 0.4, "retention_s": 120.0}
            return lines
        return lines + [{"identity": "break", "schema": 3, "stream_epoch": 1, "track_id": 1, "cause": "jump",
                         "mono_ns": 2_000_000_000}]

    _mutate(out, edit)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert not result.ok
    item = "face admitted for this validation run" if change == "startup" else "identity records"
    assert line_for(text, item).startswith("FAIL"), text


def test_the_scripted_live_run_is_validated_from_the_runtimes_own_output(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert "MISSING" not in text and "FAIL" not in text and "NOT_EXERCISED" not in text
    retention = line_for(text, "A: a known identity is retained")
    assert retention.startswith("PASS") and '"cleared_to": "unresolved"' in retention
    assert "a note that is never printed" not in text
    assert "DESCRIPTIVE scheduling rate" in text and "DESCRIPTIVE time to known" in text


def test_a_combined_run_is_read_field_by_field_and_its_f2_reading_is_unchanged(mods, tmp_path, startup, f1_list) -> None:
    """D78: face recognition with the live view, snapshots and a chat validation run: every evidence file fits its
    profile (nothing nonconforming, so the privacy counts read it as strictly as a face-only run) and the F2 reading
    is the same as without them."""
    out = simulate(tmp_path, startup, combined=True)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert "MISSING" not in text and "FAIL" not in text and "NOT_EXERCISED" not in text
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            read = mods.evidence.read(path, "f2")
            assert read.nonconforming == [], (path.name, read.nonconforming[:5])
    status = json.loads((out / "status-k1.json").read_text())
    assert status["preview"]["state"] == "live" and status["components"]["chat"]["validation_run"] is True
    assert status["components"]["chat"]["counters"]["accepted"] >= 1
    assert status["components"]["preview"]["counters"]["encodes"] >= 1
    assert status["components"]["snapshots"]["state"] == "running"
    stopped = json.loads((out / "run.jsonl").read_text().splitlines()[-1])["shutdown"]["stopped"]
    assert stopped["live_view"] is True and stopped["snapshots"] is True
    starting = json.loads((out / "run.jsonl").read_text().splitlines()[0])["startup"]
    assert starting["chat"]["state"] == "available" and "validation run" in starting["chat"]["reason"]


def test_an_undeclared_combined_field_is_nonconforming(mods, tmp_path, startup) -> None:
    """The additions are explicit fields, not a bypass: an unknown key or value under them is still counted."""
    out = simulate(tmp_path, startup, combined=True, duration=70.0)  # the first capture is at 60 s
    status = json.loads((out / "status-e1.json").read_text())
    status["components"]["chat"]["counters"]["asked_someone"] = 1
    status["components"]["chat"]["reason"] = "Alice asked"
    status["preview"]["state"] = "glowing"
    (out / "status-e1.json").write_text(json.dumps(status))
    read = mods.evidence.read(out / "status-e1.json", "f2")
    assert len(read.nonconforming) == 3, read.nonconforming


def test_the_dashboards_recognition_rows_are_declared_exactly(mods, tmp_path, startup) -> None:
    """D95: the status's per-person rows (and the rule's numbers) fit the profile as written by the real runtime;
    an identity ID, a similarity, an undeclared state or basis, or free text in a reason is still counted."""
    out = simulate(tmp_path, startup)
    status = json.loads((out / "status-k1.json").read_text())  # the known segment: one person, recognized
    assert status["live"]["faces"] and status["components"]["face"]["policy"]["confirmations"] == 2
    assert mods.evidence.read(out / "status-k1.json", "f2").nonconforming == []
    row = status["live"]["faces"][0]
    status["live"]["faces"] = [{**row, "identity_id": "idn-00000000000a"}, {**row, "similarity": 0.81},
                               {**row, "state": "stranger"}, {**row, "basis": "remembered"},
                               {**row, "reason": "Jane looked away"}]
    (out / "status-k1.json").write_text(json.dumps(status))
    read = mods.evidence.read(out / "status-k1.json", "f2")
    assert len(read.nonconforming) == 5, read.nonconforming


def test_a_face_still_visible_while_turned_away_does_not_exercise_retention(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, away_face=FACE)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "A: a known identity is retained").startswith("NOT_EXERCISED")


def test_retention_is_not_judged_when_the_account_says_the_face_was_visible(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account={**ACCOUNT, "face_hidden_while_turned": "no"})
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "A: a known identity is retained").startswith("NOT_EXERCISED")


def test_a_tracker_reusing_the_id_after_the_return_carries_nothing_over(mods, tmp_path, startup, f1_list) -> None:
    script = SCRIPT[:-1] + ((480, 601, "facing", 1),)
    out = simulate(tmp_path, startup, script=script)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text


def test_the_account_is_required_locked_and_unchanged(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account=None)
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "account saved and locked").startswith("FAIL")
    assert "completion" not in text  # nothing about identity is shown without an account
    (tmp_path / "b").mkdir()
    out2 = simulate(tmp_path / "b", startup)
    finish(mods, out2)
    (out2 / "account.txt").write_text((out2 / "account.txt").read_text().replace("turned_away=4:00", "turned_away=4:10"))
    result, text = reading(mods, out2, f1_list)
    assert not result.ok and "differs from its recorded SHA-256" in text


@pytest.mark.parametrize(("change", "item", "label"), [
    (lambda a: {**a, "out_of_view_1": "unknown"}, "account complete and in order", "FAIL"),
    (lambda a: {**a, "other_faces_in_view": "yes"}, "stopwatch started at the T0 cue", "FAIL"),
    (lambda a: {**a, "stood_up": "4:40", "out_of_view_2": "4:41"}, "judged windows from the account", "FAIL"),
])
def test_account_problems_fail_the_reading(mods, tmp_path, startup, f1_list, change, item, label) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, account=change(ACCOUNT))
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, item).startswith(label)


def _nth_result(lines: list[dict], n: int, change) -> list[dict]:
    indices = [i for i, line in enumerate(lines) if line.get("identity") == "result"]
    return [change(line) if i == indices[n] else line for i, line in enumerate(lines)]


def _mutate(out: Path, edit) -> None:
    lines = [json.loads(line) for line in (out / "run.jsonl").read_text().splitlines()]
    lines = edit(lines)
    (out / "run.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


@pytest.mark.parametrize(("edit", "item", "label"), [
    # a field missing from one result: the schema changed (a validation run writes schema 3, D84 and D86)
    (lambda ls: _nth_result(ls, 0, lambda l: {k: v for k, v in l.items() if k != "frame_mono_ns"}),
     "identity records: schema 3", "FAIL"),
    # an unknown schema version
    (lambda ls: [dict(l, schema=4) if l.get("identity") == "tick" else l for l in ls], "identity records", "FAIL"),
    # one record kind in schema 1 among schema 3 records: a run's records share one schema
    (lambda ls: [dict(l, schema=1) if l.get("identity") == "tick" else l for l in ls], "identity records", "FAIL"),
    # most results lost: completion below 0.95
    (lambda ls: [l for n, l in enumerate(ls) if not (l.get("identity") == "result" and n % 3)], "completion", "FAIL"),
    # one processing error
    (lambda ls: _nth_result(ls, 4, lambda l: dict(l, outcome="failed", persons=[])), "completion", "FAIL"),
    # another identity known on the return
    (lambda ls: [dict(l, identity_id="idn-00000000000b") if l.get("identity") == "transition"
                 and l["to"] == "known" and l["track_id"] == 2 else l for l in ls], "only the enrolled", "FAIL"),
    # no offered tick and no result while facing in K2 (the person was not seen)
    (lambda ls: [l for l in ls if not (l.get("identity") in ("tick", "result") and l.get("frame_mono_ns", 0) > 495e9)],
     "K2 sampling", "FAIL"),
])
def test_runtime_output_deviations_fail_the_reading(mods, tmp_path, startup, f1_list, edit, item, label) -> None:
    out = simulate(tmp_path, startup)
    _mutate(out, edit)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, item).startswith(label), text


def test_a_known_change_without_two_matches_or_on_two_tracks_fails(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)

    def inject(lines):
        start = next(l for l in lines if l.get("identity") == "transition" and l["to"] == "known" and l["track_id"] == 1)
        fake = dict(start, track_id=9, mono_ns=start["mono_ns"] + 5_000_000_000)
        return lines + [fake]

    _mutate(out, inject)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert line_for(text, "one identity is never known on two tracks").startswith("FAIL")
    assert line_for(text, "every change to known follows").startswith("FAIL") and not result.ok


def test_t0_secrets_and_startup_problems_fail_the_reading(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, cue_delay_s=3.0, counts={"name": 1})
    result, text = reading(mods, out, f1_list)
    assert line_for(text, "T0: the t0 cue").startswith("FAIL")
    assert line_for(text, "passphrase, name and nonconforming counts").startswith("FAIL")
    broken = dict(startup, startup={**startup["startup"], "face": {"validation_run": False, "identities_enrolled": 1}})
    (tmp_path / "x").mkdir()
    out2 = simulate(tmp_path / "x", broken, duration=10.0)
    finish(mods, out2)
    _, text = reading(mods, out2, f1_list)
    assert line_for(text, "face admitted for this validation run").startswith("FAIL")


def test_the_checker_matches_the_runtimes_schema_and_values(mods) -> None:
    check = mods.check
    frame = SimpleNamespace(stream_epoch=0, frame_seq=1, ingest_mono_ns=5)
    assert set(tick_record(frame, [], TICK_OFFERED)) == check.TICK_KEYS
    record = FaceResultRecord(0, 1, 5, 6, "applied", 1, ((1, "assigned", "match", ENROLLED, "match"),)).record()
    assert set(record) == check.RESULT_KEYS and set(record["persons"][0]) == check.PERSON_KEYS
    assert set(IdentityTransition(0, 1, "unresolved", "known", ENROLLED, "fresh", 0, "r", 5).record()) == \
        check.TRANSITION_KEYS
    assert check.TICKS == {TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING}
    defaults = IdentityConfig()
    assert (check.VOTE_TTL_S, check.FRESH_S, check.CONFIRMATIONS) == (
        defaults.vote_ttl_s, defaults.face_result_max_age_s, defaults.confirmations)
    assert check.MAX_IDENTITY_RECORDS == demo_runtime.MAX_IDENTITY_RECORDS
    assert set(check.FACE_RELEASES) == set(RELEASE_FILE_ROLES["face"])
    assert check.MIN_COMPLETION == 0.95 and check.SCHEMA == 1
    # D84: schema 2, the validation runs' records
    numbers = FaceDiagnostics((0.1, 0.2, 0.3, 0.9), 0.8, (40, 50), 0.95, 0.75)
    diagnostic = FaceResultRecord(0, 1, 5, 6, "applied", 1, ((1, "assigned", "match", ENROLLED, "match"),),
                                  diagnostics=(numbers,)).record(IDENTITY_DIAGNOSTIC_SCHEMA)
    assert (check.DIAGNOSTIC_SCHEMA, check.SCHEMAS) == (IDENTITY_DIAGNOSTIC_SCHEMA, (1, 2, 3, 4, 5, 6))
    # D88: schema 5, the candidate's records, and its constants
    from sentinel.identity.legacy_deepface import DETECTION_FULL_SCALE, FULL_SCALE_FLOOR, CandidateDetection
    from sentinel.runtime import IDENTITY_CANDIDATE_SCHEMA, detection_numbers
    assert (check.CANDIDATE_SCHEMA, check.DETECTION_FULL_SCALE, check.RECORDING_FLOOR) == (
        IDENTITY_CANDIDATE_SCHEMA, DETECTION_FULL_SCALE, FULL_SCALE_FLOOR)
    detection = detection_numbers(CandidateDetection((0.62, FACE_BOX), 47.25, 861.0),
                                  SimpleNamespace(native_width=640))
    assert set(detection) == check.DETECTION_KEYS and check._detection_fits(detection, 1, "applied")
    candidate = FaceResultRecord(0, 1, 5, 6, "applied", 1, ((1, "assigned", "match", ENROLLED, "match"),),
                                 diagnostics=(numbers,), detection=detection).record(IDENTITY_CANDIDATE_SCHEMA)
    assert check._record_problem("result", candidate) is None and "full_scale" not in candidate
    # D90: schema 6, its keys, bounds and the prototype positions
    from sentinel.identity.legacy_deepface import FaceGeometry
    from sentinel.identity.matching import MAX_PROTOTYPES
    from sentinel.runtime import IDENTITY_GEOMETRY_SCHEMA, RAW_BOUNDS, face_geometry_numbers
    geometry = face_geometry_numbers(FaceObservation(frame=FrameKey(camera_id="c", boot_id="b", run_id="r",
                                                                    stream_epoch=0, frame_seq=1), box=FACE_BOX,
                                                     quality=0.9), FaceGeometry((-2.0, 104.0, 58.0, 62.0),
                                                                                (15.0, 128.0), (38.0, 129.0)), 640, 480)
    assert set(geometry) == check.GEOMETRY_KEYS and set(geometry["eyes"]) == check.EYES_KEYS
    six = FaceResultRecord(0, 1, 5, 6, "applied", 1, ((1, "assigned", "match", ENROLLED, "match"),),
                           diagnostics=(dataclasses.replace(numbers, prototype=3, geometry=geometry),),
                           detection=detection).record(IDENTITY_GEOMETRY_SCHEMA)
    assert set(six["persons"][0]) == check.GEOMETRY_PERSON_KEYS and check._record_problem("result", six) is None
    assert (check.GEOMETRY_SCHEMA, check.RAW_BOUNDS, check.MAX_PROTOTYPES) == (
        IDENTITY_GEOMETRY_SCHEMA, RAW_BOUNDS, MAX_PROTOTYPES)
    assert check.CANDIDATE_SCHEMAS == (IDENTITY_CANDIDATE_SCHEMA, IDENTITY_GEOMETRY_SCHEMA)
    # D91: the capture's record, its refusals and minimum, and the experiment's markers
    from sentinel.identity import experiment
    assert check.CAPTURE_REFUSALS == experiment.CAPTURE_REFUSALS == mods.evidence.CAPTURE_REFUSALS
    assert check.CAPTURE_MIN_PROTOTYPES == experiment.CAPTURE_MIN_PROTOTYPES
    assert check.EXPERIMENT_CUES[6] == mods.account.COMPARISON_CUE_AT_S
    assert check.EXPERIMENT_CUES[8] == mods.account.NON_ENROLLED_CUE_AT_S
    from sentinel.identity.legacy_deepface import YUNET_THRESHOLD
    from sentinel.runtime import FULL_SCALE_COUNTED, IDENTITY_FULL_SCALE_SCHEMA, full_scale_numbers
    assert check.FULL_SCALE_SCHEMA == IDENTITY_FULL_SCALE_SCHEMA and FULL_SCALE_COUNTED == YUNET_THRESHOLD
    numbers = full_scale_numbers(SimpleNamespace(candidates=((0.95, FACE_BOX),), ms=3.0),
                                 SimpleNamespace(key=None, native_width=640), (), None)
    assert set(numbers) == check.FULL_SCALE_KEYS and check._full_scale_fits(numbers)
    assert set(diagnostic["persons"][0]) == check.DIAGNOSTIC_PERSON_KEYS
    assert set(diagnostic["persons"][0]["face"]) == check.FACE_KEYS
    assert check.MATCH_THRESHOLD == defaults.match_threshold
    assert set(tick_record(frame, [], TICK_OFFERED, IDENTITY_DIAGNOSTIC_SCHEMA)) == check.TICK_KEYS
    # D87: schema 4's comparison records, their kinds and dispositions
    from sentinel.identity.comparison import Disposition, Kind
    from sentinel.runtime import IDENTITY_COMPARISON_SCHEMA, ComparisonRecord
    comparison = ComparisonRecord(0, 1, 5, 1, "both", "completed", 0.93, 58, 0.8, 812.0, None).record()
    assert set(comparison) == check.COMPARISON_KEYS and check.COMPARISON_SCHEMA == IDENTITY_COMPARISON_SCHEMA
    assert set(check.COMPARISON_KINDS) == {k.value for k in Kind}
    assert set(check.COMPARISON_DISPOSITIONS) == {d.value for d in Disposition}
    assert check._comparison_problem(comparison) is None
    assert set(mods.evidence.COMPARISON_KINDS) == set(check.COMPARISON_KINDS)
    assert set(mods.evidence.COMPARISON_DISPOSITIONS) == set(check.COMPARISON_DISPOSITIONS)


# ---------------------------------------------------------------- D84: schema 2's diagnostic numbers


def _value(text: str, start: str) -> Any:
    line = line_for(text, start)
    return json.loads(line[line.index(": {") + 2:])


def test_a_validation_runs_diagnostics_are_described_per_window_and_never_judged(mods, tmp_path, startup,
                                                                                  f1_list) -> None:
    """D84: the scripted validation run's records are schema 2. The reading judges what it always judged and adds,
    per judged window, the runtime's own numbers: tick continuity, the people's boxes and detector confidence, the
    faces' size and score, and each vote's similarity."""
    out = simulate(tmp_path, startup)
    finish(mods, out)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert line_for(text, "identity records: schema 3").startswith("PASS")
    continuity = _value(text, "diagnostics: person continuity")
    assert continuity["E1"]["offered"] == 0 and continuity["E1"]["track_ids"] == []
    assert continuity["E1"]["longest_skipped_run"] == continuity["E1"]["skipped_no_person"] > 0
    assert continuity["A"] == {"offered": 90, "skipped_no_person": 0, "longest_skipped_run": 0, "track_ids": [1],
                               "cleared_then_offered": [], "known_entries": 0}
    assert continuity["K2"]["known_entries"] == 0  # known at 481.1 s, before the window opens at 495 s
    full = _value(text, "diagnostics: the full-scale face pass")
    assert full["K1"] == {"results": 90, "with_pass": 0, "production_face": 90, "full_scale_face": 0,
                          "full_scale_only": 0, "best_score_where_production_found_none": None,
                          "full_scale_w_px": None, "ms": None}  # the scripted adapter has no full-scale pass
    people = _value(text, "diagnostics: people on applied results")
    assert people["E1"] is None and people["E2"] is None
    assert people["K1"] == {"n": 90, "box_median": [0.156, 0.208, 0.312, 0.833], "confidence": [0.9, 0.9, 0.9]}
    faces = _value(text, "diagnostics: assigned faces")
    assert faces["K1"] == {"faces": 90, "people": 90, "w_px": [58, 58, 58], "h_px": [62, 62, 62],
                           "score": [0.9, 0.9, 0.9]}
    assert faces["A"] == {"faces": 0, "people": 90, "w_px": None, "h_px": None, "score": None}  # turned away
    similarity = _value(text, "diagnostics: similarity to the closest enrolled identity")
    assert similarity["K2"] == {"match_votes": 90, "match": [0.993, 0.993, 0.993], "unknown_votes": 0,
                                "unknown": None}
    assert similarity["A"] == {"match_votes": 0, "match": None, "unknown_votes": 0, "unknown": None}


def test_the_same_run_in_schema_1_reads_exactly_as_before(mods, tmp_path, startup, f1_list) -> None:
    """Every saved attempt is schema 1 and reads as it always did: the same run without the diagnostic numbers gives
    the same lines, except the schema's number and the diagnostics, which only schema 2 has."""
    out = simulate(tmp_path, startup)
    finish(mods, out)
    _, with_numbers = reading(mods, out, f1_list)

    def schema_1(lines: list[dict]) -> list[dict]:
        plain = []
        for line in lines:
            if "identity" in line:
                line = {k: v for k, v in line.items() if k != "full_scale"} | {"schema": 1}
                if line["identity"] == "result":
                    line["persons"] = [{k: v for k, v in p.items() if k in mods.check.PERSON_KEYS}
                                       for p in line["persons"]]
            plain.append(line)
        return plain

    _mutate(out, schema_1)
    result, without = reading(mods, out, f1_list)
    assert result.ok, without
    assert without.splitlines() == [line.replace("identity records: schema 3", "identity records: schema 1")
                                    for line in with_numbers.splitlines()
                                    if not line.startswith("DESCRIPTIVE diagnostics: ")]


def _first_match(lines: list[dict], change) -> list[dict]:
    """The first result person that voted match, changed."""
    indices = [i for i, l in enumerate(lines) if l.get("identity") == "result" and l["persons"]
               and l["persons"][0]["label"] == "match"]
    first = lines[indices[0]]
    return [dict(l, persons=[change(dict(l["persons"][0]))] + l["persons"][1:]) if l is first else l for l in lines]


NO_FACE = {"ownership": "no_face", "vote": "none", "identity_id": None, "label": "no_face", "face": None,
           "similarity": None}


@pytest.mark.parametrize(("change", "fits"), [
    (lambda p: p, True),  # unchanged: the control
    (lambda p: dict(p, **NO_FACE), True),  # a consistent no-face entry
    (lambda p: dict(p, vote="unknown", identity_id=None, label="unknown", similarity=0.7), True),  # rounded up to it
    (lambda p: dict(p, **{**NO_FACE, "face": {"w_px": 40, "h_px": 50, "score": 0.95}}), False),  # nobody's face
    (lambda p: dict(p, **{**NO_FACE, "similarity": 0.4}), False),  # a similarity where none was computed
    (lambda p: dict(p, face=None), False),  # no face where one was assigned
    (lambda p: dict(p, similarity=None), False),  # a vote without its similarity
    (lambda p: dict(p, similarity=0.5), False),  # a match below the threshold
    (lambda p: dict(p, vote="unknown", identity_id=None, label="unknown", similarity=0.9), False),  # unknown above
    (lambda p: dict(p, box=[0.5, 0.2, 0.4, 0.8]), False),  # x1 after x2
    (lambda p: dict(p, box=[0.1, 0.2, 0.3]), False),  # three corners
    (lambda p: dict(p, confidence=None), False),  # a detection without its confidence
    (lambda p: dict(p, face=dict(p["face"], w_px=40.5)), False),  # pixels as a fraction
    (lambda p: dict(p, face=dict(p["face"], extra=1)), False),  # an undeclared number
    (lambda p: dict(p, similarity="0.9"), False),  # a number as text
])
def test_diagnostic_numbers_must_fit_their_entrys_labels(mods, tmp_path, startup, f1_list, change, fits) -> None:
    out = simulate(tmp_path, startup, duration=20.0)  # too short to pass as a whole: only the records are read here
    _mutate(out, lambda ls: _first_match(ls, change))
    finish(mods, out, counts={})
    _, text = reading(mods, out, f1_list)
    problems = _value(text, "identity records: schema 3")["problems"]
    if fits:
        assert problems == [], problems
    else:
        assert len(problems) == 1 and problems[0].endswith("diagnostic numbers do not fit its labels"), problems


def test_schema_2_numbers_are_schema_text_and_anything_else_is_nonconforming(mods, tmp_path, startup) -> None:
    """The privacy profile declares the numbers as numbers: a name or text under them is counted, never passed."""
    out = simulate(tmp_path, startup, duration=20.0)
    assert mods.evidence.read(out / "run.jsonl", "f2").nonconforming == []
    _mutate(out, lambda ls: _first_match(ls, lambda p: dict(p, face=dict(p["face"], who="Jane"), similarity="high")))
    read = mods.evidence.read(out / "run.jsonl", "f2")
    assert len(read.nonconforming) == 2 and any(text == "Jane" for _, text in read.content), read.nonconforming


# ---------------------------------------------------------------- F2 account helper


def test_the_account_waits_for_the_run_saves_once_and_locks(mods, tmp_path) -> None:
    run = tmp_path / "f2"
    run.mkdir()
    (run / "provenance.txt").write_text("part=f2\n")
    said: list[str] = []
    assert mods.account.enter(run, ask=lambda p: "", say=said.append) == 2 and "not finished" in said[-1]
    (run / "result.json").write_text("{}")
    # delay, its uncertainty, out_1 and its uncertainty, seated_1 unknown (its uncertainty is not asked), turned and
    # its uncertainty, hidden, stood ..., then re-entering seated_1 (item 5) asks its uncertainty again
    answers = iter(["2", "1", "0:12", "2", "unknown", "4:00", "1", "yes", "6:00", "1", "6:01", "3", "8:00", "1",
                    "yes", "no", "", "5", "1:58", "4", "SAVE"])
    assert mods.account.enter(run, ask=lambda p: next(answers), say=said.append) == 0
    assert any("seated_facing_1_uncertainty_s: unknown, because seated_facing_1 is unknown" in x for x in said)
    saved = (run / "account.txt").read_text()
    assert saved.startswith(mods.account.HEADER + "\n") and "unknown" not in saved
    assert "seated_facing_1=1:58   #" in saved and "seated_facing_1_uncertainty_s=4   #" in saved
    assert "stopwatch_start_delay_s=2   #" in saved and "stopwatch_start_delay_uncertainty_s=1   #" in saved
    locked = (run / "provenance.txt").read_text().splitlines()[-1]
    assert locked == "account_sha256=" + hashlib.sha256(saved.encode()).hexdigest()
    assert mods.account.enter(run, ask=lambda p: "", say=said.append) == 2  # never changed once saved
    said.clear()
    assert mods.account.check(run, say=said.append) == 0 and said[-1].startswith("Locked: yes; the file matches")
    assert said[0].endswith("format 2:")


def test_the_account_review_reports_order_and_unknowns(mods) -> None:
    values = {**ACCOUNT, "seated_facing_1": "5:00", "stood_up": "unknown"}
    review = mods.account.review(values, mods.account.FIELDS_V1)  # a format-1 account, read as before
    assert "stood_up" in review[0] and "turned_away 4:00 should be after seated_facing_1 5:00" in review[2]
    assert review[-1] == "Complete and in order: no"


# ---------------------------------------------------------------- F2 timing: account format 2 (D66)

# The scripted operator's account in format 2: started at the cue (0, give or take 1 s), every reading within 1 s.
ACCOUNT_V2 = {"stopwatch_start_delay_s": "0", "stopwatch_start_delay_uncertainty_s": "1",
              "out_of_view_1": "0:12", "out_of_view_1_uncertainty_s": "1",
              "seated_facing_1": "2:00", "seated_facing_1_uncertainty_s": "1",
              "turned_away": "4:00", "turned_away_uncertainty_s": "1", "face_hidden_while_turned": "yes",
              "stood_up": "6:00", "stood_up_uncertainty_s": "1",
              "out_of_view_2": "6:01", "out_of_view_2_uncertainty_s": "1",
              "seated_facing_2": "8:00", "seated_facing_2_uncertainty_s": "1",
              "stayed_until_stop": "yes", "other_faces_in_view": "no", "notes": "a note that is never printed"}
# The proposal's examples (session 50, part 9): a stopwatch started 2 s late, readings 2 s behind T0.
LATE = {**ACCOUNT_V2, "stopwatch_start_delay_s": "2", "out_of_view_1": "0:18", "seated_facing_1": "1:58",
        "turned_away": "3:58", "stood_up": "5:58", "out_of_view_2": "6:18", "seated_facing_2": "7:58"}
# The face-0f7855e helper's own rendering of ACCOUNT (generated with that pin's face_account.render): format 1, pinned.
LEGACY_TEXT = (
    "# F2 account entered with benchmarks/runner/face_account.py; values as typed\n"
    "stopwatch_started_at_t0=yes   # yes / no: you started the stopwatch when the T0 cue appeared\n"
    "out_of_view_1=0:12   # m:ss from which you were sure the camera could not see you (schedule: by 0:20)\n"
    "seated_facing_1=2:00   # m:ss from which you sat at the Mac with your face toward the camera (schedule 2:00)\n"
    "turned_away=4:00   # m:ss from which your face was turned away from the camera, still seated (schedule 4:00)\n"
    "face_hidden_while_turned=yes   # yes / no: the camera could not see your face at all while turned away\n"
    "stood_up=6:00   # m:ss when you stood up to leave (schedule 6:00)\n"
    "out_of_view_2=6:01   # m:ss from which you were sure the camera could not see you again\n"
    "seated_facing_2=8:00   # m:ss from which you sat facing the camera again (schedule 8:00)\n"
    "stayed_until_stop=yes   # yes / no: you stayed seated facing the camera until the stop cue (10:00)\n"
    "other_faces_in_view=no   # yes / no: anyone else, or a photo or screen showing a face, was in the camera's view "
    "at any time\n"
    "notes=a note that is never printed\n")


def windows_of(mods, account: dict[str, str], fmt: int = 2):
    bounds = mods.check._bounds(account, fmt)
    return bounds, None if bounds is None else mods.check._windows(bounds, 600.0)


def test_format_1_is_kept_read_only_and_renders_as_the_old_helper_did(mods, tmp_path) -> None:
    assert mods.account.render(ACCOUNT, mods.account.FIELDS_V1, mods.account.HEADER_V1) == LEGACY_TEXT
    assert mods.account.account_format(LEGACY_TEXT)[0] == 1
    assert mods.account.account_format(mods.account.render(ACCOUNT_V2))[0] == 2
    assert mods.account.account_format("# something else\nx=1\n") is None
    bounds, windows = windows_of(mods, ACCOUNT, fmt=1)  # each reading taken as exact, as it always was
    assert bounds == {"out_of_view_1": (12, 12), "seated_facing_1": (120, 120), "turned_away": (240, 240),
                      "stood_up": (360, 360), "out_of_view_2": (361, 361), "seated_facing_2": (480, 480)}
    assert windows == {"E1": (27.0, 105.0), "K1": (135.0, 225.0), "A": (255.0, 345.0), "E2": (376.0, 465.0),
                       "K2": (495.0, 585.0)}
    run = tmp_path / "f2"
    run.mkdir()
    (run / "provenance.txt").write_text("part=f2\n")
    save_account(mods, run, ACCOUNT)
    said: list[str] = []
    assert mods.account.check(run, say=said.append) == 0
    assert said[0].endswith("format 1:") and said[-2] == "Complete and in order: yes" and said[-1].startswith("Locked: yes")


def test_the_proposals_examples_align_to_t0_and_window_from_the_far_edges(mods) -> None:
    bounds, windows = windows_of(mods, LATE)  # delay 2 +/- 1: [1, 3] s
    assert bounds["seated_facing_1"] == (118, 122) and bounds["turned_away"] == (238, 242)
    assert windows == {"E1": (37.0, 103.0), "K1": (137.0, 223.0), "A": (257.0, 343.0), "E2": (397.0, 463.0),
                       "K2": (497.0, 585.0)}
    bounds, windows = windows_of(mods, {**LATE, "out_of_view_2": "6:24", "out_of_view_2_uncertainty_s": "5"})
    assert bounds["out_of_view_2"] == (380, 392) and windows["E2"] == (407.0, 463.0)  # 56 s
    _, windows = windows_of(mods, {**LATE, "out_of_view_2": "6:24", "out_of_view_2_uncertainty_s": "5",
                                   "seated_facing_2": "7:45", "seated_facing_2_uncertainty_s": "5"})
    assert windows["E2"] == (407.0, 446.0) and windows["K2"] == (488.0, 585.0)  # E2 39 s: too short
    worst = {**ACCOUNT_V2, "stopwatch_start_delay_s": "0", "stopwatch_start_delay_uncertainty_s": "5",
             "out_of_view_1": "0:20", "out_of_view_2": "6:20",
             **{mods.account.uncertainty_key(k): "5" for k in mods.account.TIMES}}
    _, windows = windows_of(mods, worst)  # the delay clamped at 0: [0, 5]
    assert {k: b - a for k, (a, b) in windows.items()} == {"E1": 55.0, "K1": 75.0, "A": 75.0, "E2": 55.0, "K2": 80.0}


@pytest.mark.parametrize(("delay", "uncertainty", "label"), [
    ("0", "5", "PASS"), ("4", "1", "PASS"), ("2", "3", "PASS"),  # the upper end exactly 5 s
    ("4", "2", "FAIL"), ("5", "1", "FAIL"), ("1", "5", "FAIL"), ("7", "1", "FAIL"),  # over 5 s
    ("unknown", "unknown", "FAIL"), ("2", "unknown", "FAIL"),
])
def test_the_stopwatch_delays_upper_end_must_be_at_most_5_s(mods, tmp_path, startup, f1_list, delay, uncertainty,
                                                            label) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, account={**ACCOUNT_V2, "stopwatch_start_delay_s": delay,
                               "stopwatch_start_delay_uncertainty_s": uncertainty})
    result, text = reading(mods, out, f1_list)
    item = line_for(text, "stopwatch start delay known, its upper end (delay + uncertainty) at most 5 s")
    assert item.startswith(label) and not result.ok, item
    assert "stopwatch started at the T0 cue" not in text  # format 1's item is not used for format 2
    windows = line_for(text, "judged windows from the account's intervals")
    assert windows.startswith("MISSING") == (delay == "unknown" or uncertainty == "unknown"), windows


@pytest.mark.parametrize("change", [
    {"turned_away": "unknown"},  # its uncertainty then is unknown without being asked
    {"turned_away_uncertainty_s": "unknown"},  # a reading the operator cannot bound within 5 s
    {"stopwatch_start_delay_s": "unknown"},
])
def test_unknown_readings_uncertainties_and_delays_leave_no_windows(mods, tmp_path, startup, f1_list, change) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    account = {**ACCOUNT_V2, **change}
    if account["turned_away"] == "unknown":
        account["turned_away_uncertainty_s"] = "unknown"
    if account["stopwatch_start_delay_s"] == "unknown":
        account["stopwatch_start_delay_uncertainty_s"] = "unknown"
    finish(mods, out, account=account)
    saved = (out / "account.txt").read_text()
    assert all(f"{key}=unknown   #" in saved for key, value in account.items() if value == "unknown")
    result, text = reading(mods, out, f1_list)
    assert line_for(text, "account complete and in order").startswith("FAIL")
    assert line_for(text, "judged windows from the account's intervals").startswith("MISSING")
    assert "completion" not in text and "aligned to T0" not in text


def test_invalid_uncertainties_and_delays_are_refused_at_entry(mods, tmp_path) -> None:
    run = tmp_path / "f2"
    run.mkdir()
    (run / "provenance.txt").write_text("part=f2\n")
    (run / "result.json").write_text("{}")
    said: list[str] = []
    answers = iter(["-1", "60", "1.5", "+2", "02",  # the delay: refused four times, then 2 (written as 2)
                    "0", "6", "-1", "2.5", "x", "+2", "±2", "1",  # its uncertainty: refused seven times
                    "0:12", "1", "2:00", "1", "4:00", "1", "yes", "6:00", "1", "6:01", "1", "8:00", "1",
                    "yes", "no", "", "SAVE"])
    assert mods.account.enter(run, ask=lambda p: next(answers), say=said.append) == 0
    refused = [x for x in said if x.startswith("  not accepted")]
    assert len(refused) == 11
    assert sum("whole seconds from 0 to 59" in x for x in refused) == 4
    assert sum("whole seconds from 1 to 5" in x for x in refused) == 7
    saved = mods.account.parse((run / "account.txt").read_text())
    assert saved["stopwatch_start_delay_s"] == "2" and saved["stopwatch_start_delay_uncertainty_s"] == "1"


def test_a_stored_uncertainty_out_of_range_is_not_an_accepted_value(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, account=None)
    data = mods.account.render({**ACCOUNT_V2, "turned_away_uncertainty_s": "7"}).encode()  # never written by enter
    (out / "account.txt").write_bytes(data)
    with (out / "provenance.txt").open("a") as provenance:
        provenance.write(f"account_sha256={hashlib.sha256(data).hexdigest()}\n")
    result, text = reading(mods, out, f1_list)
    assert "turned_away_uncertainty_s=7" in line_for(text, "account complete and in order")
    assert line_for(text, "judged windows from the account's intervals").startswith("MISSING")


@pytest.mark.parametrize(("change", "item", "label"), [
    ({"turned_away": "1:59"}, "account complete and in order", "FAIL"),  # readings out of order
    ({"seated_facing_1": "2:00", "turned_away": "2:10", "seated_facing_1_uncertainty_s": "5",
      "turned_away_uncertainty_s": "5"}, "judged windows from the account's intervals", "FAIL"),  # intervals overlap
    ({"stood_up": "6:01", "out_of_view_2": "6:01"}, "account complete and in order", "PASS"),  # equal: allowed
    ({"seated_facing_1": "2:42"}, "judged windows from the account's intervals", "PASS"),  # K1 exactly 45 s
    ({"seated_facing_1": "2:43"}, "judged windows from the account's intervals", "FAIL"),  # K1 44 s
])
def test_order_overlap_and_the_45_s_minimum(mods, tmp_path, startup, f1_list, change, item, label) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out, account={**ACCOUNT_V2, **change})
    _, text = reading(mods, out, f1_list)
    assert line_for(text, item).startswith(label), line_for(text, item)
    if change.get("seated_facing_1") in ("2:42", "2:43"):  # K1 = hi(2:42 + 1 + 1) + 15 .. lo(4:00 - 1 + 0) - 15
        assert ('"K1": [179.0, 224.0]' if change["seated_facing_1"] == "2:42" else '"K1": [180.0, 224.0]') in text


def test_the_scripted_run_validates_with_a_format_2_account(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account=ACCOUNT_V2)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert line_for(text, "stopwatch start delay known").startswith("PASS")
    assert '"turned_away": ["4:00", 1, 239, 242]' in text  # the aligned interval, shown
    assert line_for(text, "A: a known identity is retained").startswith("PASS")
    assert "DESCRIPTIVE A: match votes on the retained track inside the turn interval" in text


def test_a_2_s_late_start_is_aligned_and_validates(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account={**LATE, "out_of_view_1": "0:10", "out_of_view_2": "5:59"})
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert '"seated_facing_1": ["1:58", 1, 118, 122]' in text and '"stopwatch_start_delay_s": [1, 3]' in text
    assert '"K1": [137.0, 223.0]' in text


def test_old_format_accounts_keep_their_original_reading(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out)  # ACCOUNT: format 1
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert line_for(text, "stopwatch started at the T0 cue").startswith("PASS")
    assert line_for(text, "judged windows from the account (washout 15 s)").startswith("PASS")
    assert '"K1": [135.0, 225.0]' in text
    assert "aligned to T0" not in text and "inside the turn interval" not in text
    assert "DESCRIPTIVE A: new matching evidence on the retained track after the turn (match votes)" in text


# ---------------------------------------------------------------- format 3 (D79, PROPOSED): cued, no stopwatch

FORMAT3_CUES = {"t0": 0.0, "k1": 120.0, "a": 240.0, "e2": 360.0, "k2": 480.0}


def account_v3(mods, **changes: str) -> dict[str, str]:
    values = {**{mods.account.within_key(k): "yes" for k in mods.account.TIMES}, "face_hidden_while_turned": "yes",
              "stayed_until_stop": "yes", "other_faces_in_view": "no", "notes": "a note that is never printed"}
    return {**values, **changes}


def test_the_scripted_run_validates_with_a_cued_format_3_account(mods, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup)
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES)
    result, text = reading(mods, out, f1_list)
    assert result.ok, text
    assert "MISSING" not in text and "FAIL" not in text and "NOT_EXERCISED" not in text
    windows = line_for(text, "judged windows from the account's intervals")
    # [cue + allowance + 15, next cue - 15]: 70, 70, 85, 70 and 70 s, every one at least 45 s.
    assert ('"E1": [35.1, 105.1]' in windows and '"K1": [155.1, 225.1]' in windows and '"A": [260.1, 345.1]' in windows
            and '"E2": [395.1, 465.1]' in windows and '"K2": [515.1, 585.0]' in windows), windows
    assert line_for(text, "every movement cue printed within 0.5 s").startswith("PASS")
    assert "DESCRIPTIVE account movements from the guard's cues" in text
    assert line_for(text, "A: a known identity is retained").startswith("PASS")


@pytest.mark.parametrize("answer", ["no", "unknown"])
def test_a_cued_movement_not_done_within_its_allowance_leaves_no_windows(mods, tmp_path, startup, f1_list,
                                                                          answer) -> None:
    out = simulate(tmp_path, startup, duration=70.0)
    finish(mods, out, account=account_v3(mods, **{mods.account.within_key("turned_away"): answer}), cues=FORMAT3_CUES)
    result, text = reading(mods, out, f1_list)
    assert not result.ok
    assert line_for(text, "every movement cue printed within 0.5 s").startswith("FAIL")
    assert line_for(text, "judged windows from the account's intervals").startswith("MISSING")


@pytest.mark.parametrize("change", ["missing", "late"])
def test_a_movement_cue_missing_or_late_leaves_no_windows(mods, tmp_path, startup, f1_list, change) -> None:
    cues = dict(FORMAT3_CUES)
    if change == "missing":
        del cues["a"]
    else:
        cues["a"] += 0.6  # beyond the 0.5 s tolerance, as the T0 cue's
    out = simulate(tmp_path, startup, duration=70.0)
    finish(mods, out, account=account_v3(mods), cues=cues)
    result, text = reading(mods, out, f1_list)
    assert not result.ok
    assert line_for(text, "every movement cue printed within 0.5 s").startswith("FAIL")
    assert line_for(text, "judged windows from the account's intervals").startswith("MISSING")


def test_a_cued_account_is_schema_text_to_the_privacy_profile_and_entered_only_after_the_run(mods, tmp_path) -> None:
    run = tmp_path / "f2"
    run.mkdir()
    (run / "provenance.txt").write_text(F2_PROVENANCE)
    assert mods.account.enter(run, ask=lambda prompt: "SAVE", say=lambda text: None, account_format_number=3) == 2
    (run / "result.json").write_text("{}")
    save_account(mods, run, account_v3(mods))
    text = (run / "account.txt").read_text()
    assert text.splitlines()[0] == mods.account.HEADER_V3 and "turned_away_within_5_s=yes" in text
    read = mods.evidence.read(run / "account.txt", "f2")
    assert read.nonconforming == [] and read.content == [(f"line {len(mods.account.FIELDS_V3) + 1}: notes",
                                                          "a note that is never printed")]


def test_both_formats_are_schema_text_to_the_privacy_profile(mods, tmp_path) -> None:
    for text, notes_line in ((LEGACY_TEXT, 12), (mods.account.render(ACCOUNT_V2), 19)):
        path = tmp_path / "account.txt"
        path.write_text(text)
        read = mods.evidence.read(path, "f2")
        assert read.nonconforming == [] and read.content == [(f"line {notes_line}: notes",
                                                              "a note that is never printed")], read.content
    path.write_text(mods.account.render({**ACCOUNT_V2, "turned_away_uncertainty_s": "7"}))
    read = mods.evidence.read(path, "f2")  # an answer as typed outside its field: content, as before
    assert read.nonconforming == [] and ("line 9: turned_away_uncertainty_s", "7") in read.content
    path.write_text("# another header\n" + mods.account.render(ACCOUNT_V2).split("\n", 1)[1])
    assert mods.evidence.read(path, "f2").nonconforming == ["line 1: header"]


# ---------------------------------------------------------------- D85: format 4, the 4:00 head turn stated


def test_format_4_is_format_3_with_the_4_00_posture_stated(mods) -> None:
    """The same keys in the same order; only the turn's and the face-hidden question's wording differ, and they ask for
    what the 4:00 cue asks: a head turn within 5 s and not before the cue, seated, torso upright where it was, the
    chair not moved until 6:00, no face visible, not even in profile."""
    v3, v4 = mods.account.FIELDS_V3, mods.account.FIELDS_V4
    assert [f.key for f in v4] == [f.key for f in v3] and [f.kind for f in v4] == [f.kind for f in v3]
    changed = {f4.key for f3, f4 in zip(v3, v4) if f3 != f4}
    assert changed == {"turned_away_within_5_s", "face_hidden_while_turned"}
    comment = {f.key: f.comment for f in v4}
    for words in ("within 5 s after the 4:00 cue", "not before it", "turned your head away", "until the 6:00 cue",
                  "stayed seated", "torso upright", "chair not moved"):
        assert words in comment["turned_away_within_5_s"], words
    for words in ("until the 6:00 cue", "could not see your face at all", "not even in profile"):
        assert words in comment["face_hidden_while_turned"], words
    assert mods.account.FORMATS[mods.account.HEADER_V4] == (4, v4) and mods.account.CUED_FORMATS == (3, 4)
    assert mods.account.FORMATS[mods.account.HEADER_V3] == (3, v3)  # format 3 unchanged: saved accounts read as before


@pytest.mark.parametrize(("change", "item", "label"), [
    ({}, None, None),  # every answer yes: the same reading as format 3
    ({"face_hidden_while_turned": "no"}, "A: a known identity is retained", "NOT_EXERCISED"),
    ({"turned_away_within_5_s": "unknown"}, "judged windows from the account's intervals", "MISSING"),
])
def test_a_format_4_account_is_read_exactly_as_format_3(mods, tmp_path, startup, f1_list, change, item, label) -> None:
    run = simulate(tmp_path, startup)
    twin = tmp_path / "twin"
    shutil.copytree(run, twin)
    finish(mods, run, account=account_v3(mods, **change), cues=FORMAT3_CUES)
    finish(mods, twin, account=account_v3(mods, **change), cues=FORMAT3_CUES, cued_format=4)
    assert (twin / "account.txt").read_text().splitlines()[0] == mods.account.HEADER_V4
    three, (result, four) = reading(mods, run, f1_list)[1], reading(mods, twin, f1_list)
    assert four == three
    if item is None:
        assert result.ok, four
    else:
        assert not result.ok and line_for(four, item).startswith(label)


def test_a_format_4_account_is_schema_text_and_the_helper_offers_it(mods, tmp_path) -> None:
    run = tmp_path / "f2"
    run.mkdir()
    (run / "provenance.txt").write_text(F2_PROVENANCE)
    assert mods.account.main(["enter", str(run), "--format", "4"]) == 2  # accepted; refused only as the run is unfinished
    (run / "result.json").write_text("{}")
    save_account(mods, run, account_v3(mods), cued_format=4)
    read = mods.evidence.read(run / "account.txt", "f2")
    assert read.nonconforming == [] and read.content == [(f"line {len(mods.account.FIELDS_V4) + 1}: notes",
                                                          "a note that is never printed")]
    with pytest.raises(SystemExit):
        mods.account.main(["enter", str(run), "--format", "9"])  # 6 is the comparison's (D87), 7 and 8 D91's


# ---------------------------------------------------------------- the turn and the departure as intervals

def synthetic(mods, *, votes=(), known=(), ticks=(), end: float = 600.0):
    """A Timeline from hand-made records (seconds after T0): votes (t, track, vote), applied 1 s after their frame;
    known (track, start, [(t, basis), ...], end, end_to); ticks (t, tracks) offered."""
    ns = lambda t: round(t * 1e9)  # noqa: E731
    results = [{"frame_mono_ns": ns(t), "applied_mono_ns": ns(t + 1.0), "outcome": "applied", "stream_epoch": 1,
                "persons": [{"track_id": track, "vote": vote, "identity_id": ENROLLED if vote == "match" else None}]}
               for t, track, vote in votes]
    transitions = []
    for track, start, bases, stop, end_to in known:
        transitions += [{"mono_ns": ns(t), "stream_epoch": 1, "track_id": track, "from": "known" if n else "unknown",
                         "to": "known", "identity_id": ENROLLED, "basis": basis, "reason": "votes"}
                        for n, (t, basis) in enumerate([(start, "fresh"), *bases])]
        if stop is not None:
            transitions.append({"mono_ns": ns(stop), "stream_epoch": 1, "track_id": track, "from": "known",
                                "to": end_to, "identity_id": None, "basis": None, "reason": "votes"})
    records = {"tick": [{"frame_mono_ns": ns(t), "tick": "offered", "tracks": list(tracks)} for t, tracks in ticks],
               "result": results, "transition": transitions}
    return mods.check.Timeline(records, 0, end)


RETAINED_TICKS = tuple((float(t), (1,)) for t in range(200, 420))


def retention(mods, account: dict[str, str], fmt: int, line) -> tuple[str, list[str]]:
    bounds, windows = windows_of(mods, account, fmt)
    r = mods.check.Reading()
    mods.check._retention(r, line, windows, ENROLLED, account, bounds, fmt)
    item = next(x for x in r.lines if "A: a known identity is retained" in x)
    return item, r.lines


def test_retention_needs_the_identity_known_throughout_the_turn_interval(mods) -> None:
    # turned_away 4:00 +/- 1 with the delay 0 +/- 1: the turn lies in [239, 242]
    held = synthetic(mods, votes=[(236.0, 1, "match"), (237.0, 1, "match"), (238.0, 1, "match"), (239.0, 1, "match")],
                     known=[(1, 238.0, [(242.0, "retained")], 268.0, "unresolved")], ticks=RETAINED_TICKS)
    item, lines = retention(mods, ACCOUNT_V2, 2, held)
    assert item.startswith("PASS"), lines
    assert "DESCRIPTIVE A: match votes on the retained track inside the turn interval: 1" in lines
    # a match inside the turn interval: retention is timed from it (newest 240.5 + 3; older 239 + 30)
    later = synthetic(mods, votes=[(236.0, 1, "match"), (237.0, 1, "match"), (239.0, 1, "match"),
                                   (240.5, 1, "match")],
                      known=[(1, 238.0, [(243.5, "retained")], 269.0, "unresolved")], ticks=RETAINED_TICKS)
    item, lines = retention(mods, ACCOUNT_V2, 2, later)
    assert item.startswith("PASS") and "DESCRIPTIVE A: match votes on the retained track inside the turn interval: 2" \
        in lines, lines
    # a contradicting vote inside the turn interval ends KNOWN at 241, within [239, 242]: known in part only
    cut = synthetic(mods, votes=[(236.0, 1, "match"), (237.0, 1, "match"), (240.0, 1, "unknown")],
                    known=[(1, 238.0, [], 241.0, "uncertain")], ticks=RETAINED_TICKS)
    item, _ = retention(mods, ACCOUNT_V2, 2, cut)
    assert item.startswith("NOT_EXERCISED") and '"known_throughout_it": 0' in item and '"known_in_part_of_it": 1' in item
    item, _ = retention(mods, ACCOUNT, 1, cut)  # format 1, the turn exactly at 240: the contradicting vote ends it
    assert item.startswith("NOT_EXERCISED") and "a contradicting vote ended it" in item
    # a second track known in part of the turn interval: which track was held at the turn is ambiguous
    two = synthetic(mods, votes=[(236.0, 1, "match"), (237.0, 1, "match"), (238.0, 1, "match"), (239.0, 1, "match")],
                    known=[(1, 238.0, [(242.0, "retained")], 268.0, "unresolved"),
                           (2, 240.5, [], 250.0, "unresolved")], ticks=RETAINED_TICKS)
    item, _ = retention(mods, ACCOUNT_V2, 2, two)
    assert item.startswith("NOT_EXERCISED") and '"known_in_part_of_it": 1' in item
    item, _ = retention(mods, ACCOUNT, 1, two)  # format 1: at 240 exactly, only track 1 was known
    assert item.startswith("PASS")


def departure(mods, account: dict[str, str], fmt: int, line) -> list[str]:
    bounds, windows = windows_of(mods, account, fmt)
    r = mods.check.Reading()
    mods.check._return(r, line, windows, bounds, fmt)
    return r.lines


# out_of_view_2 at 6:20 +/- 5 (delay 0 +/- 1): the departure lies in [375, 386]; seated again at 6:40 +/- 1
DEPARTING = {**ACCOUNT_V2, "out_of_view_2": "6:20", "out_of_view_2_uncertainty_s": "5", "seated_facing_2": "6:40"}
DEPARTING_V1 = {**ACCOUNT, "out_of_view_2": "6:20", "seated_facing_2": "6:40"}  # the same readings, as exact


def test_an_identity_known_inside_the_departure_interval_counts_as_known_before_leaving(mods) -> None:
    line = synthetic(mods, known=[(3, 381.0, [], 405.0, "unresolved")])  # known from 381, inside [375, 386]
    carried = departure(mods, DEPARTING, 2, line)[0]
    assert carried.startswith("FAIL every identity known before the latest possible leaving time") and "381" in carried
    assert departure(mods, DEPARTING_V1, 1, line)[0].startswith("PASS every identity known before leaving")


def test_k2_knowledge_from_a_vote_inside_the_departure_interval_is_carried_over(mods) -> None:
    line = synthetic(mods, votes=[(384.0, 4, "match"), (395.0, 4, "match")],  # 384 lies in [375, 386]
                     known=[(4, 410.0, [], None, None)])
    late = departure(mods, DEPARTING, 2, line)[1]
    assert late.startswith("FAIL K2: nothing carried over: known again only from votes after the latest possible"), late
    assert departure(mods, DEPARTING_V1, 1, line)[1].startswith("PASS K2: nothing carried over")


# ---------------------------------------------------------------- F1


PHOTOS = tuple(f"Jane Doe {n}.jpg" for n in range(8))


class Pipeline:
    def load(self) -> None:
        pass

    def compatibility(self) -> FaceCompatibility:
        return COMPAT

    def enrollment_photo(self, path: Path):
        kind = path.read_text()
        if kind == "none":
            return None, "no_face"
        return EnrollmentFace(vec(1.0, 0.05 * int(kind)), 0.9, 120), "face_found"


def run_f1(mods, tmp_path, monkeypatch, capsys, *, kinds=("1", "2", "3"), break_step=None) -> tuple[Path, Path]:
    sealer = Sealer(kdf_n=2**14)
    try:
        sealer.check()
    except VaultError as exc:
        pytest.skip(f"the system Python's cryptography is not usable here ({exc.label})")
    identity_dir = tmp_path / "identity"
    store = IdentityStore(identity_dir, sealer)
    store.ensure_directory()
    folder = store.inbox / "face-20261008T090000Z"
    folder.mkdir(mode=0o700)
    for name, kind in zip(PHOTOS, kinds):
        (folder / name).write_text(kind)
    f1 = tmp_path / "f1"
    f1.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("config_version: 1\ncamera: {id: cam-1}\n")
    provenance = [f"photos_supplied={len(kinds)}", "consent_date=2026-10-08", f"inbox_folder_name={folder.name}",
                  "gallery_before_dryrun=absent"]

    def identity(*args):
        monkeypatch.setenv(SECRET_ENV, "a long synthetic passphrase")
        code = main(["identity", *args[:1], str(config), *args[1:], "--identity-dir", str(identity_dir)],
                     face_backend=lambda w: Pipeline(), identity_store=lambda d: store)
        return code, capsys.readouterr().out

    common = (str(folder), "--consent-confirmed", "--consent-date", "2026-10-08")
    code, out = identity("enroll", *common, "--dry-run")
    (f1 / "dryrun.json").write_text(out)
    provenance += [f"dryrun_exit={code}", f"gallery_after_dryrun={'present' if store.exists() else 'absent'}"]
    if break_step == "dryrun_writes":
        provenance[-1] = "gallery_after_dryrun=present"
    code, out = identity("enroll", *common)
    (f1 / "enroll.json").write_text(out)
    provenance.append(f"enroll_exit={code}")
    code, out = identity("list")
    (f1 / "list.json").write_text(out)
    provenance.append(f"list_exit={code}")
    if break_step == "name_found":
        provenance.append("operator_note=Jane D.")  # a name where nothing may hold one
    (f1 / "provenance.txt").write_text("\n".join(provenance) + "\n")
    done = bash_counts("face_counts_f1", str(f1), str(identity_dir / "audit.jsonl"), photos=PHOTOS[:len(kinds)])
    assert done.returncode == 0, done.stderr
    (f1 / "secret-counts-identity.txt").write_text(done.stdout)
    return f1, identity_dir


def test_a_consented_enrollment_is_validated(mods, tmp_path, monkeypatch, capsys) -> None:
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys)
    result = mods.check.read_f1(f1, identity_dir)
    text = "\n".join(result.lines)
    assert result.ok, text
    assert "Jane" not in text and "Jane" not in "".join(p.read_text() for p in f1.iterdir())


@pytest.mark.parametrize(("break_step", "item"), [
    ("dryrun_writes", "dry run"),
    ("name_found", "passphrase, name and photo-file-name counts"),
])
def test_f1_deviations_fail(mods, tmp_path, monkeypatch, capsys, break_step, item) -> None:
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys, break_step=break_step)
    result = mods.check.read_f1(f1, identity_dir)
    assert not result.ok and line_for("\n".join(result.lines), item).startswith("FAIL")


def test_two_usable_photos_fail_this_package_and_three_pass(mods, tmp_path, monkeypatch, capsys) -> None:
    """The package needs 3 usable photos (maintainer, 2026-10-08), although the enrollment API accepts 2."""
    from sentinel.identity.enroll import MIN_PHOTOS

    assert (MIN_PHOTOS, mods.check.MIN_USABLE_PHOTOS) == (2, 3)
    two = tmp_path / "two"
    two.mkdir()
    f1, identity_dir = run_f1(mods, two, monkeypatch, capsys, kinds=("1", "2", "none"))
    assert json.loads((f1 / "enroll.json").read_text())["verified"] is True  # the API itself accepted 2
    result = mods.check.read_f1(f1, identity_dir)
    text = "\n".join(result.lines)
    assert not result.ok
    assert line_for(text, "dry run").startswith("FAIL") and line_for(text, "enrollment verified").startswith("FAIL")
    assert '"accepted": 2' in line_for(text, "dry run")
    others = [line for line in text.splitlines()[:-1]
              if not line.split(" ", 1)[1].startswith(("dry run", "enrollment verified"))]
    assert all(line.startswith("PASS") for line in others), text  # every other condition holds
    for kinds in (("1", "2", "3"), ("1", "2", "3", "none")):
        root = tmp_path / f"three-{len(kinds)}"
        root.mkdir()
        f1, identity_dir = run_f1(mods, root, monkeypatch, capsys, kinds=kinds)
        result = mods.check.read_f1(f1, identity_dir)
        assert result.ok, "\n".join(result.lines)


def test_f1_fails_if_a_supplied_copy_was_left_or_storage_is_not_private(mods, tmp_path, monkeypatch, capsys) -> None:
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys, kinds=("1", "2", "3", "none"))
    text = "\n".join(mods.check.read_f1(f1, identity_dir).lines)
    assert line_for(text, "dry run").startswith("PASS")  # 3 usable of 4
    (identity_dir / "gallery.sealed").chmod(0o644)
    (identity_dir / "inbox" / "stray.jpg").write_text("x")
    text = "\n".join(mods.check.read_f1(f1, identity_dir).lines)
    assert line_for(text, "private storage").startswith("FAIL")


# ================================================================ privacy counts (face_counts.sh, face_privacy, face_evidence)

F1_FILES = ("dryrun.json", "enroll.json", "list.json", "provenance.txt", "audit.jsonl")
TRT = "[10/08/2026-09:00:01] [TRT] [I] Loaded engine size: 13 MiB"
TRT_W = ("[10/08/2026-09:00:01] [TRT] [W] Using an engine plan file across different models of devices is not "
         "recommended and is likely to affect performance or even cause errors.")
WARNING = ("/home/user/onvif_env/lib/python3.10/site-packages/torch/cuda/__init__.py:827: UserWarning: Can't "
           "initialize NVML")


def parse(stdout: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for line in stdout.splitlines():
        label, pair = line.rsplit(" ", 1)
        key, value = pair.split("=")
        out.setdefault(label, {})[key] = int(value)
    return out


def stdin_of(terms: dict[str, list[str]]) -> bytes:
    return b"".join(f"{kind}={value}\0".encode() for kind, values in terms.items() for value in values)


def counts(mods, command: str, directory: Path, terms: dict[str, list[str]], **kwargs) -> dict[str, dict[str, int]]:
    """face_privacy's own API, for sweeps; the shell workflow is exercised separately."""
    return parse("\n".join(mods.privacy.run(command, directory, stdin_of(terms), **kwargs)))


def explain(*args: str, terms: dict[str, list[str]]) -> list[str]:
    done = subprocess.run([sys.executable, str(RUNNERS / "face_privacy.py"), *args, "--explain"],
                          input=stdin_of(terms), capture_output=True, timeout=60, check=True)
    return done.stdout.decode().splitlines()


@pytest.fixture
def f1_evidence(mods, tmp_path, monkeypatch, capsys):
    """Two real enrollments: all photos accepted, and one rejected (its reason label is no_face)."""
    out = []
    for name, kinds in (("plain", ("1", "2", "3")), ("rejected", ("1", "2", "3", "none"))):
        root = tmp_path / name
        root.mkdir()
        out.append(run_f1(mods, root, monkeypatch, capsys, kinds=kinds))
    return out


@pytest.fixture
def f2_evidence(mods, tmp_path, startup):
    out = simulate(tmp_path, startup)
    (out / "run.err").write_text("\n".join((WARNING, '  warnings.warn("Can\'t initialize NVML")', TRT, TRT_W)) + "\n")
    (out / "guard.jsonl").write_text(json.dumps({"MemAvailable": 1, "MemFree": 2, "pressure_bytes": 3,
                                                 "pswpin": 0, "pswpout": 0, "t_mono": 1.5}) + "\n")
    finish(mods, out)
    return out


# ---------------------------------------------------------------- the defect and its correction, F1


def test_no_initial_or_short_name_matches_f1_schema_text(mods, f1_evidence) -> None:
    """Every letter (an initial) and two-letter names drawn from the schema's own words count 0 and every file fits
    its profile, on the real enrollment's evidence, while the old substring search flagged nearly every letter."""
    for f1, identity_dir in f1_evidence:
        audit = identity_dir / "audit.jsonl"
        for name in [*string.ascii_letters, "No", "Id", "Px", "Ed", "Al", "Dry"[:2]]:
            found = counts(mods, "f1", f1, {"name": ["Jane", name]}, audit=audit, keys=("name",))
            assert sorted(found) == sorted(F1_FILES)
            assert all(v == {"name": 0, "nonconforming": 0} for v in found.values()), (name, found)
        texts = [p.read_text().casefold() for p in (*(f1 / n for n in F1_FILES[:4]), audit)]
        flagged = [c for c in string.ascii_lowercase if any(c in t for t in texts)]
        assert len(flagged) >= 20  # grep -ciF's view: an initial "leaked" almost whatever it was


def test_the_f1_workflow_counts_through_bash_and_validates(mods, f1_evidence) -> None:
    f1, identity_dir = f1_evidence[1]
    done = bash_counts("face_counts_f1", str(f1), str(identity_dir / "audit.jsonl"), photos=PHOTOS[:4],
                       env={**TERMS, "N2": "K"})
    assert done.returncode == 0 and done.stderr == ""
    found = parse(done.stdout)
    assert found == {n: {"passphrase": 0, "name": 0, "filenames": 0, "nonconforming": 0} for n in F1_FILES}
    assert all(TERMS[k] not in done.stdout for k in ("N1", "SENTINEL_IDENTITY_PASSPHRASE")) and "Jane" not in done.stdout
    (f1 / "secret-counts-identity.txt").write_text(done.stdout)
    assert mods.check.read_f1(f1, identity_dir).ok


def _copy(f1: Path, identity_dir: Path, tmp: Path) -> tuple[Path, Path]:
    f1c, identity = tmp / "f1", tmp / "identity"
    shutil.copytree(f1, f1c)
    identity.mkdir()
    shutil.copy2(identity_dir / "audit.jsonl", identity / "audit.jsonl")
    return f1c, identity / "audit.jsonl"


def _edit_json(path: Path, change) -> None:
    item = json.loads(path.read_text())
    change(item)
    path.write_text(json.dumps(item))


LEAKS = {  # what an initial "D" (or the long name, passphrase, file name) looks like where it does not belong
    "a value holding the initial": ("enroll.json", lambda p: _edit_json(p, lambda d: d["results"][0].update(
        reason="D"))),
    "an unexpected key": ("dryrun.json", lambda p: _edit_json(p, lambda d: d.update({"D": 1}))),
    "a file-name-like value": ("provenance.txt", lambda p: p.write_text(
        p.read_text().replace("inbox_folder_name=face-20261008T090000Z", "inbox_folder_name=D_selfies"))),
    "an extra file": ("notes.txt", lambda p: p.write_text("photos by D.\n")),
    "a duplicated key": ("list.json", lambda p: p.write_text(p.read_text().replace('{"identity": "list"',
                                                                                   '{"identity": "D", "identity": "list"'))),
    "an appended audit line": ("audit.jsonl", lambda p: p.write_text(
        p.read_text() + '{"action":"enroll","identity_id":"D","utc":"2026-10-08T09:00:00+00:00"}\n')),
}


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_an_initial_out_of_place_and_unexpected_content_are_counted(mods, f1_evidence, tmp_path, leak) -> None:
    f1, identity_dir = f1_evidence[0]
    f1c, audit = _copy(f1, identity_dir, tmp_path)
    name, change = LEAKS[leak]
    change(audit if name == "audit.jsonl" else f1c / name)
    done = bash_counts("face_counts_f1", str(f1c), str(audit), photos=PHOTOS[:3])
    assert done.returncode == 0, done.stderr
    found = parse(done.stdout)[name]
    assert found["name"] >= 1 and found["nonconforming"] >= 1, found
    where = explain("f1", str(f1c), "--audit", str(audit), "--keys", "name", terms={"name": ["Jane", "D"]})
    assert any(x.startswith(f"{name} name: ") for x in where) and not any('"D"' in x or "Jane" in x for x in where)
    (f1c / "secret-counts-identity.txt").write_text(done.stdout)
    text = "\n".join(mods.check.read_f1(f1c, audit.parent).lines)
    assert next(x for x in text.splitlines() if "photo-file-name counts" in x).startswith("FAIL")


def test_longer_names_passphrases_and_file_names_keep_their_byte_searches(mods, f1_evidence, tmp_path) -> None:
    f1, identity_dir = f1_evidence[0]
    f1c, audit = _copy(f1, identity_dir, tmp_path)
    secret = 'pass"phrase-with-a-quote'
    raw = (f1c / "enroll.json").read_text()[:-1]
    (f1c / "enroll.json").write_text(raw[:-1] + ', "x1": "JaneDoe", "x2": "Zo\\u00eb", "x3": '
                                     + json.dumps(secret) + ', "x4": "Jane Doe 1.jpg"}\n')
    found = counts(mods, "f1", f1c, {"passphrase": [secret], "name": ["Jane", "Zoë"], "filename": list(PHOTOS[:3])},
                   audit=audit)["enroll.json"]
    assert found["name"] >= 2  # a substring of a value, and a non-ASCII name only visible decoded
    assert found["passphrase"] == 1  # only visible decoded: its quote is escaped in the bytes
    assert found["filenames"] == 1 and found["nonconforming"] == 4
    quality = counts(mods, "f1", f1, {"name": ["Ali"]}, audit=audit, keys=("name",))  # "Ali" is inside "quality"
    assert quality["dryrun.json"]["name"] >= 1  # unchanged: a longer name still fails closed on such a collision


MALFORMED = {
    "truncated JSON": lambda p: p.write_text(p.read_text()[:40]),
    "a wrong type": lambda p: _edit_json(p, lambda d: d.update(photos=str(d["photos"]))),
    "a value outside its vocabulary": lambda p: _edit_json(p, lambda d: d["results"][0].update(result="maybe")),
    "a root that is not an object": lambda p: p.write_text("[1, 2]\n"),
    "bytes that are not UTF-8": lambda p: p.write_bytes(p.read_bytes() + b"\xff\xfe"),
}


@pytest.mark.parametrize("kind", sorted(MALFORMED))
def test_malformed_evidence_fails_closed(mods, f1_evidence, tmp_path, kind) -> None:
    f1, identity_dir = f1_evidence[0]
    f1c, audit = _copy(f1, identity_dir, tmp_path)
    MALFORMED[kind](f1c / "dryrun.json")
    done = bash_counts("face_counts_f1", str(f1c), str(audit), photos=PHOTOS[:3])
    assert done.returncode == 0 and parse(done.stdout)["dryrun.json"]["nonconforming"] >= 1
    (f1c / "secret-counts-identity.txt").write_text(done.stdout)
    assert not mods.check.read_f1(f1c, audit.parent).ok


@pytest.mark.parametrize(("env", "photos", "stop"), [
    ({k: v for k, v in TERMS.items() if k != "SENTINEL_IDENTITY_PASSPHRASE"}, PHOTOS[:3], "no passphrase"),
    ({**TERMS, "N1": "", "N2": ""}, PHOTOS[:3], "no name"),
    (TERMS, (), "no filename"),
    ({**TERMS, "N1": "1984"}, PHOTOS[:3], "a name without a letter"),
])
def test_missing_terms_refuse_with_no_counts(f1_evidence, env, photos, stop) -> None:
    f1, identity_dir = f1_evidence[0]
    done = bash_counts("face_counts_f1", str(f1), str(identity_dir / "audit.jsonl"), env=env, photos=photos)
    assert (done.returncode, done.stdout) == (2, "") and stop in done.stderr


@pytest.mark.parametrize(("stdin", "stop"), [
    (b"passphrase=a\0passphrase=b\0name=Jane\0filename=x\0", "more than one passphrase"),
    (b"password=a\0name=Jane\0filename=x\0", "unknown kind"),
    (b"passphrase=a\0name=Jane\0filename=x", "NUL-terminated"),
    (b"passphrase=\xff\0name=Jane\0filename=x\0", "not UTF-8"),
    (b"passphrase=a\0name=Jane\0filename=x\0url=rtsp://h/x\0", "no userinfo"),
    (b"passphrase=a\0name=Jane\0filename=x\0url=rtsp://u:p@h/x\0", "does not use"),
])
def test_malformed_terms_refuse(mods, f1_evidence, stdin, stop) -> None:
    f1, identity_dir = f1_evidence[0]
    with pytest.raises(mods.privacy.Refused, match=stop):
        mods.privacy.run("f1", f1, stdin, audit=identity_dir / "audit.jsonl")


# ---------------------------------------------------------------- F2 and FH


def test_f2_schema_text_never_counts_and_free_text_always_can(mods, f2_evidence) -> None:
    files = sorted(p.name for p in f2_evidence.iterdir() if p.name not in mods.evidence.NOT_COUNTED)
    assert {"run.jsonl", "run.err", "result.json", "guard.jsonl", "account.txt", "provenance.txt",
            "status-e1.json"} <= set(files)
    for letter in string.ascii_lowercase:
        found = counts(mods, "f2", f2_evidence, {"name": ["Jane", letter]}, keys=("name",))
        assert sorted(found) == files and all(v["nonconforming"] == 0 for v in found.values()), found
        hits = {name: v["name"] for name, v in found.items() if v["name"]}
        if letter != "a":
            assert hits == {}, (letter, hits)  # TRT's [I] and [W], "Can't", every key, label and template: schema
    where = {x.split(": ", 1)[1] for x in explain("f2", str(f2_evidence), "--keys", "name",
                                                   terms={"name": ["Jane", "a"]}) if " name: " in x}
    # "A synthetic report." (the scene model's summary) and the account's notes are free text: content
    assert where == {"live.scene_report.summary (content)", "line 12: notes (content)"}, where


def test_the_f2_workflow_counts_through_bash_and_validates(mods, f2_evidence, f1_list) -> None:
    files = {p.name for p in f2_evidence.iterdir() if p.name not in mods.evidence.NOT_COUNTED}
    for name, keys in (("secret-counts-camera.txt", {"userinfo"}),
                       ("secret-counts-identity.txt", {"passphrase", "name", "nonconforming"})):
        found = parse((f2_evidence / name).read_text())
        assert set(found) == files and all(set(v) == keys and not any(v.values()) for v in found.values()), found
    result = mods.check.read_f2(f2_evidence, f1_list)
    assert result.ok, "\n".join(result.lines)


def test_a_participants_name_is_searched_only_when_given(mods, f2_evidence) -> None:
    """D91: x-N's hidden prompt sets N3 (a non-enrolled participant's first name); face_counts_f2 searches it with the
    operator's names, and without it every count is as before."""
    account = f2_evidence / "account.txt"
    account.write_text(account.read_text().replace("a note that is never printed", "Rosalind sat here"))
    before = parse(bash_counts("face_counts_f2", str(f2_evidence)).stdout)
    assert before["account.txt"]["name"] == 0
    found = bash_counts("face_counts_f2", str(f2_evidence), env={**TERMS, "N3": "Rosalind"})
    assert found.returncode == 0, found.stderr
    after = parse(found.stdout)
    assert after["account.txt"]["name"] == 1
    assert {f: v for f, v in after.items() if f != "account.txt"} == {f: v for f, v in before.items()
                                                                     if f != "account.txt"}


F2_LEAKS = {
    "an unexpected status field": ("status-k1.json", lambda p: _edit_json(p, lambda d: d["live"].update(operator="D"))),
    "a non-JSON stdout line": ("run.jsonl", lambda p: p.write_text(p.read_text() + "hello D\n")),
    "an unexpected capture name": ("result.json", lambda p: _edit_json(
        p, lambda d: d["captures"][0].update(name="D"))),
    "an extra file": ("extra.txt", lambda p: p.write_text("D\n")),
    "a provenance line": ("provenance.txt", lambda p: p.write_text(p.read_text() + "operator=D\n")),
}


@pytest.mark.parametrize("leak", sorted(F2_LEAKS))
def test_f2_unexpected_fields_and_lines_are_counted(mods, f2_evidence, f1_list, leak) -> None:
    name, change = F2_LEAKS[leak]
    change(f2_evidence / name)
    done = bash_counts("face_counts_f2", str(f2_evidence))
    found = parse(done.stdout)[name]
    assert found["name"] >= 1 and found["nonconforming"] >= 1, found
    (f2_evidence / "secret-counts-identity.txt").write_text(done.stdout)
    camera = bash_counts("face_counts_camera", str(f2_evidence))
    (f2_evidence / "secret-counts-camera.txt").write_text(camera.stdout)
    text = "\n".join(mods.check.read_f2(f2_evidence, f1_list).lines)
    assert next(x for x in text.splitlines() if "passphrase, name and nonconforming counts" in x).startswith("FAIL")


def test_f2_free_text_and_secrets_in_logs_are_counted(f2_evidence) -> None:
    lines = (f2_evidence / "account.txt").read_text().replace("notes=a note that is never printed", "notes=asked D")
    (f2_evidence / "account.txt").write_text(lines)
    with (f2_evidence / "run.err").open("a") as err:
        err.write(f"open failed: {TERMS['SENTINEL_RTSP_URL']}\n[10/08/2026-09:00:02] [TRT] [E] tensor named D\n")
    identity = parse(bash_counts("face_counts_f2", str(f2_evidence)).stdout)
    assert identity["account.txt"] == {"passphrase": 0, "name": 1, "nonconforming": 0}  # notes: content, not structure
    assert identity["run.err"] == {"passphrase": 0, "name": 1, "nonconforming": 0}  # a log message: content
    camera = parse(bash_counts("face_counts_camera", str(f2_evidence)).stdout)
    assert camera["run.err"] == {"userinfo": 1} and sum(v["userinfo"] for v in camera.values()) == 1


def test_fh_counts_the_session_files_and_needs_a_url(tmp_path) -> None:
    session = tmp_path / "session"
    (session / "f1").mkdir(parents=True)
    (session / "face.yaml").write_text("camera: {id: cam-1}\n")
    (session / "validate.txt").write_text("ok\n")
    (session / "camera-probe.json").write_text(json.dumps({"note": "cam-user:cam-secret-9"}))
    (session / "f1" / "dryrun.json").write_text("{}")
    (session / "secret-counts-top.txt").write_text("")
    done = bash_counts("face_counts_top", str(session))
    assert parse(done.stdout) == {"camera-probe.json": {"userinfo": 1}, "face.yaml": {"userinfo": 0},
                                  "validate.txt": {"userinfo": 0}, "f1/dryrun.json": {"userinfo": 0}}
    assert bash_counts("face_counts_top", str(session), env={k: v for k, v in TERMS.items()
                                                             if k != "SENTINEL_RTSP_URL"}).stdout == ""


def test_terms_reach_the_counter_on_stdin_only(f1_evidence, f2_evidence, tmp_path) -> None:
    """No process gets a term as an argument (FH included); the counter gets them on stdin."""
    recorder = tmp_path / "recorder.py"
    record = tmp_path / "record.jsonl"
    recorder.write_text("import json, sys\n"
                        f"open({str(record)!r}, 'a').write(json.dumps({{'argv': sys.argv, "
                        "'stdin': sys.stdin.buffer.read().decode()}) + '\\n')\n")
    f1, identity_dir = f1_evidence[0]
    session = tmp_path / "session"
    session.mkdir()
    for function, args in (("face_counts_f1", (str(f1), str(identity_dir / "audit.jsonl"))),
                           ("face_counts_f2", (str(f2_evidence),)), ("face_counts_camera", (str(f2_evidence),)),
                           ("face_counts_top", (str(session),)),
                           ("face_recheck_f1", (str(f1), str(identity_dir), str(tmp_path / "r"), "0" * 40))):
        done = bash_counts(function, *args, photos=PHOTOS[:3], prelude=f'FACE_PY=("$TEST_PY" {recorder});')
        assert done.returncode == 0, done.stderr
    calls = [json.loads(x) for x in record.read_text().splitlines()]
    assert len(calls) == 5
    secrets = [TERMS["N1"], TERMS["SENTINEL_IDENTITY_PASSPHRASE"], "cam-secret-9", *PHOTOS[:3]]
    for call in calls:
        assert not any(s in arg for s in secrets for arg in call["argv"]), call["argv"]
    assert "passphrase=a long synthetic passphrase\0name=Jane\0name=D\0filename=Jane Doe 0.jpg\0" in calls[0]["stdin"]
    assert calls[3]["stdin"] == f"url={TERMS['SENTINEL_RTSP_URL']}\0" and calls[4]["stdin"] == "name=Jane\0name=D\0"


# ---------------------------------------------------------------- the profiles follow the pinned code


def test_the_vocabularies_are_the_codes(mods) -> None:
    from operator_check import MEMORY_KEYS

    from sentinel.demo_runtime import MAX_PENDING_SIGNALS, PENDING_DURABILITY, RuntimeState
    from sentinel.identity import state
    from sentinel.identity.association import Ownership
    from sentinel.identity.worker import TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING
    from sentinel.incidents.service import IncidentStatus
    from sentinel.live_state import Capability, Occupancy, SceneStatus
    from sentinel.media.capture import CaptureState
    from sentinel.media.health import VideoState
    from sentinel.memory_policy import CANDIDATE_POLICY, DEFAULT_POLICY, RELEASE_FILE_ROLES, RELEASE_RESULTS
    from sentinel.rules.scene_hazard import Severity
    from sentinel.scene.report import Threat, Uncertainty
    from sentinel.scene.server import ServerState

    fe = mods.evidence
    assert set(fe.MEMORY_KEYS) == set(MEMORY_KEYS)
    for vocab, enum in ((fe.CAPABILITY, Capability), (fe.OCCUPANCY, Occupancy), (fe.SCENE_STATUS, SceneStatus),
                        (fe.VIDEO, VideoState), (fe.CAPTURE, CaptureState), (fe.SERVER, ServerState),
                        (fe.RUNTIME, RuntimeState), (fe.BASIS, state.Basis), (fe.SEVERITY, Severity),
                        (fe.INCIDENT_STATUS, IncidentStatus), (fe.THREAT, Threat), (fe.UNCERTAINTY, Uncertainty)):
        assert vocab.values == {e.value for e in enum}, enum
    assert set(fe.IDENTITY_STATES) == {e.value for e in state.IdentityState}
    assert set(fe.OWNERSHIP) == {e.value for e in Ownership}
    assert set(fe.VOTE_NOTES) == set(state._NOTE_LABELS)
    assert set(fe.VOTE_LABELS) == set(state._NOTE_LABELS.values()) | {state.BEFORE_BREAK}  # D97
    from sentinel.identity.continuity import NOT_CARRIED, Cause, ContinuityPolicy
    assert set(fe.CONTINUITY_CAUSES) == {c.value for c in Cause} and set(fe.NOT_CARRIED) == set(NOT_CARRIED.values())
    assert set(fe.CONTINUITY_POLICY) == set(ContinuityPolicy().numbers())
    from sentinel import features, status_page
    from sentinel.live_preview import PAUSED, PAUSED_REASON
    from sentinel.scene.chat import OPERATOR_OFF
    assert fe.FEATURES == features.FEATURES and set(fe.FEATURE_RESULTS) == set(features.OUTCOMES)  # D99
    assert set(fe.FEATURE_ACTUAL) == set(status_page.FEATURE_ACTUAL)
    assert PAUSED in fe.PREVIEW_STATE.values and fe.PREVIEW_REASON.fit(PAUSED_REASON) == (True, [])
    assert fe.CHAT_REASON.fit(OPERATOR_OFF)[0] and "snapshots_off" in status_page.SNAPSHOT_REASONS
    assert fe.TICKS.values == {TICK_OFFERED, TICK_SKIPPED_NO_PERSON, TICK_STOPPING}
    assert fe.RELEASE_RESULTS.values == set(RELEASE_RESULTS)
    assert fe.RELEASE_COMPONENTS.values == set(RELEASE_FILE_ROLES)
    assert fe.RELEASE_ROLES.values == {r for roles in RELEASE_FILE_ROLES.values() for r in roles}
    assert {DEFAULT_POLICY.thp, CANDIDATE_POLICY.thp} == fe.THP.values
    assert {DEFAULT_POLICY.model_file_release, CANDIDATE_POLICY.model_file_release} == fe.RELEASE.values
    assert fe.PENDING_DURABILITY.fit(PENDING_DURABILITY) == (True, [str(MAX_PENDING_SIGNALS)])


def test_the_runtimes_fixed_phrases_are_templates_and_their_variable_parts_content(mods) -> None:
    from sentinel.demo_runtime import degradation

    snapshot = {"runtime": {"state": "running"}, "live": {"video": "stale"},
                "components": {"capture": {"ready": False, "state": "waiting", "problem": "read_timeout"},
                               "detector": {"state": "unavailable", "problem": "engine_missing"},
                               "scene": {"state": "unavailable", "problem": "server_failed"},
                               "face": {"state": "unavailable", "problem": "gallery_locked"},
                               "incidents": {"pending_signals": 2, "problem": "database_locked", "signals_dropped": 1},
                               "notifications": {"channels": {"telegram": {"state": "unavailable", "problem": "http_5"}},
                                                 "worker": {"state": "stopped", "problem": "lease"}}}}
    phrases = degradation(snapshot)
    assert len(phrases) == 10
    for phrase in phrases:
        assert mods.evidence.DEGRADED.fit(phrase)[0], phrase
    assert mods.evidence.DEGRADED.fit("capture waiting (read_timeout)") == (True, ["waiting", "read_timeout"])
    assert mods.evidence.TRANSITION_REASON.fit("2 consistent matches") == (True, ["2"])
    assert mods.evidence.TRANSITION_REASON.fit("matched by D") == (False, [])
    assert mods.evidence.INCIDENT_TITLE.fit("Person in restricted zone 'whole-view'") == (True, ["whole-view"])
    assert mods.evidence.OCCUPANCY_REASON.fit("nobody detected on fresh video") == (True, [])


# ================================================================ the offline F1 recheck, its acceptance, the F1 gate

PIN = "0123456789abcdef0123456789abcdef01234567"
FH_MANIFEST = "find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"


def seal(directory: Path) -> None:
    """The run's SHA256SUMS, by FH's own command line."""
    subprocess.run(["bash", "-c", FH_MANIFEST], cwd=directory, check=True)


def tree(*roots: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for root in roots
            for p in ([root] if root.is_file() else sorted(root.rglob("*"))) if p.is_file()}


@pytest.fixture
def face_run(mods, tmp_path, monkeypatch, capsys):
    """A run as F1f and FH left face-20261007T215900Z: the old substring counts flagged the family initial in every
    file (name=1, no nonconforming key), so F1_exit=1 and f1/check.txt says not validated; then its SHA256SUMS."""
    f1, identity_dir = run_f1(mods, tmp_path, monkeypatch, capsys)
    run = tmp_path / "runs" / "face-20261008T090000Z"
    run.mkdir(parents=True)
    shutil.move(f1, run / "f1")
    files = sorted(mods.check._f1_files(run / "f1"))
    (run / "f1" / "secret-counts-identity.txt").write_text(
        "".join(f"{n} passphrase=0\n{n} name=1\n{n} filenames=0\n" for n in files))
    original = mods.check.read_f1(run / "f1", identity_dir)
    assert not original.ok
    (run / "f1" / "check.txt").write_text("\n".join([*original.lines, "face-f1: not validated"]) + "\n")
    (run / "face.yaml").write_text("config_version: 1\ncamera: {id: cam-1}\n")
    (run / "validate.txt").write_text("config ok\n")
    (run / "secret-counts-top.txt").write_text("")
    (run / "provenance.txt").write_text(f"pin_commit={PIN}\nFB_exit=0\nF1_exit=1\n")
    seal(run)
    return SimpleNamespace(run=run, identity=identity_dir, root=tmp_path / "recheck")


def recheck(face_run, name: str = "20261008T100000Z", env: dict[str, str] | None = None):
    out = face_run.root / f"{face_run.run.name}-f1-{PIN[:7]}-{name}"
    return out, bash_counts("face_recheck_f1", str(face_run.run), str(face_run.identity), str(out), PIN, env=env)


def accept(mods, face_run, out: Path, typed: str | None = None) -> tuple[int, str]:
    said: list[str] = []
    identity = mods.check._enrolled_id(face_run.run / "f1")
    line = typed if typed is not None else f"ACCEPT {face_run.run.name} {identity}"
    code = mods.recheck.accept(out, face_run.run, face_run.identity, ask=io.StringIO(line + "\n"), say=said.append)
    return code, "\n".join(said)


def test_the_offline_recheck_validates_and_changes_nothing_of_the_run(mods, face_run) -> None:
    before = tree(face_run.run, face_run.identity / "audit.jsonl")
    out, done = recheck(face_run)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "face-f1-recheck: validated" in done.stdout and "Jane" not in done.stdout
    assert tree(face_run.run, face_run.identity / "audit.jsonl") == before  # the run and the audit log unchanged
    assert sorted(p.name for p in out.iterdir()) == ["README.txt", "SHA256SUMS", "check.txt", "provenance.txt",
                                                      "secret-counts-names.txt"]
    assert oct(out.stat().st_mode & 0o777) == "0o700" and all(oct(p.stat().st_mode & 0o777) == "0o600"
                                                                for p in out.iterdir())
    provenance = mods.check._key_values(out / "provenance.txt")
    assert provenance["names"] == "recounted" and provenance["passphrase_counts"] == "carried_forward"
    assert provenance["filename_counts"] == "carried_forward" and provenance["carried_source_in_run_manifest"] == "yes"
    assert provenance["run_manifest_check"] == "verified" and provenance["pin_commit"] == PIN
    check = (out / "check.txt").read_text()
    carried = next(x for x in check.splitlines() if "carried forward" in x)
    assert carried.startswith("PASS") and '"rescanned": false' in carried
    assert '"F1_exit": "1"' in check and "F1_exit=1" in (out / "README.txt").read_text()
    assert not any("Jane" in p.read_text() for p in out.iterdir())
    assert mods.check._manifest(out) == ({p: hashlib.sha256((out / p).read_bytes()).hexdigest()
                                          for p in sorted(x.name for x in out.iterdir()) if p != "SHA256SUMS"}, [])


def _changed_run_file(face_run) -> None:
    with (face_run.run / "face.yaml").open("a") as handle:
        handle.write("# edited after the run\n")


def _added_run_file(face_run) -> None:
    (face_run.run / "f1" / "extra.json").write_text("{}")


def _carried_passphrase(face_run) -> None:
    counts = face_run.run / "f1" / "secret-counts-identity.txt"
    counts.write_text(counts.read_text().replace("list.json passphrase=0", "list.json passphrase=1"))
    seal(face_run.run)


def _original_item_differs(face_run) -> None:
    check = face_run.run / "f1" / "check.txt"
    check.write_text(check.read_text().replace("PASS each identity command", "FAIL each identity command", 1))
    seal(face_run.run)


@pytest.mark.parametrize(("change", "item"), [
    (_changed_run_file, "the run's evidence unchanged"),
    (_added_run_file, "the run's evidence unchanged"),
    (_carried_passphrase, "passphrase and photo-file-name counts carried forward"),
    (_original_item_differs, "every original F1 item other than the privacy counts"),
    (None, "names counted again"),  # a first name that collides with schema text: fails closed
])
def test_the_recheck_fails_closed(mods, face_run, change, item) -> None:
    if change is not None:
        change(face_run)
    out, done = recheck(face_run, env={**TERMS, "N1": "Ali"} if change is None else None)
    assert done.returncode == 1 and "face-f1-recheck: not validated" in done.stdout
    assert next(x for x in (out / "check.txt").read_text().splitlines() if item in x).startswith("FAIL")
    assert accept(mods, face_run, out)[0] == 1 and not (out / "acceptance.txt").exists()


def test_acceptance_needs_the_exact_typed_line_and_is_written_once(mods, face_run) -> None:
    out, done = recheck(face_run)
    assert done.returncode == 0
    code, said = accept(mods, face_run, out, typed="ACCEPT")
    assert code == 1 and "nothing was written" in said and not (out / "acceptance.txt").exists()
    code, said = accept(mods, face_run, out)
    assert code == 0, said
    acceptance = out / "acceptance.txt"
    assert oct(acceptance.stat().st_mode & 0o777) == "0o600"
    fields = mods.check._key_values(acceptance)
    assert fields["run"] == face_run.run.name and fields["accepted_reading"] == "f1-recheck"
    assert fields["recheck_check_sha256"] == hashlib.sha256((out / "check.txt").read_bytes()).hexdigest()
    assert accept(mods, face_run, out)[0] == 1  # never overwritten


def test_the_gate_lets_f2_build_only_on_an_accepted_recheck_that_still_matches(mods, face_run, tmp_path) -> None:
    gate = mods.check.f1_gate
    assert gate(face_run.run, face_run.identity, face_run.root) == (False, ["f1_gate=refused",
                                                                         "f1_gate_reason=no_accepted_recheck"])
    out, _ = recheck(face_run)
    assert gate(face_run.run, face_run.identity, face_run.root)[1][-1] == "f1_gate_reason=no_accepted_recheck"
    assert accept(mods, face_run, out)[0] == 0
    ok, lines = gate(face_run.run, face_run.identity, face_run.root)
    assert ok and lines[:3] == ["f1_gate=passed", f"run={face_run.run.name}", "f1_basis=accepted_recheck"]
    provenance = tmp_path / "provenance.txt"  # F2c writes these lines into F2's provenance, which is counted
    provenance.write_text(F2_PROVENANCE + "\n".join(lines) + "\n")
    assert mods.evidence.read(provenance, "f2").nonconforming == []
    cli = subprocess.run([sys.executable, str(RUNNERS / "face_check.py"), "gate", str(face_run.run), "--identity-dir",
                          str(face_run.identity), "--recheck-root", str(face_run.root)], capture_output=True, text=True)
    assert cli.returncode == 0 and cli.stdout.splitlines() == lines

    def refused(reason: str) -> None:
        assert gate(face_run.run, face_run.identity, face_run.root) == (False, ["f1_gate=refused",
                                                                             f"f1_gate_reason={reason}"])

    audit = face_run.identity / "audit.jsonl"
    saved = audit.read_bytes()
    audit.write_bytes(saved + b'{"action":"revoke","identities_left":0,"identity_id":"idn-000000000001",'
                              b'"utc":"2026-10-08T11:00:00+00:00"}\n')
    refused("recheck_no_longer_validates")  # the identity's audit log changed after the acceptance
    audit.write_bytes(saved)
    check = out / "check.txt"
    kept = check.read_bytes()
    check.write_bytes(kept + b"\n")
    refused("recheck_changed")
    check.write_bytes(kept)
    acceptance = out / "acceptance.txt"
    text = acceptance.read_text()
    acceptance.write_text(text.replace("confirmation=ACCEPT", "confirmation=accept"))
    refused("acceptance_does_not_match")
    acceptance.write_text(text)
    other, _ = recheck(face_run, name="20261008T110000Z")
    shutil.copy2(acceptance, other / "acceptance.txt")
    refused("several_accepted_rechecks")
    shutil.rmtree(other)
    (face_run.identity / "gallery.sealed").rename(face_run.identity / "gallery.moved")
    refused("gallery_missing")
    (face_run.identity / "gallery.moved").rename(face_run.identity / "gallery.sealed")
    _changed_run_file(face_run)
    refused("run_evidence_changed")


def test_the_gate_accepts_a_run_whose_own_f1_validated(mods, face_run) -> None:
    f1 = face_run.run / "f1"
    done = bash_counts("face_counts_f1", str(f1), str(face_run.identity / "audit.jsonl"), photos=PHOTOS[:3])
    (f1 / "secret-counts-identity.txt").write_text(done.stdout)
    reading = mods.check.read_f1(f1, face_run.identity)
    assert reading.ok
    (f1 / "check.txt").write_text("\n".join([*reading.lines, "face-f1: validated"]) + "\n")
    (face_run.run / "provenance.txt").write_text(f"pin_commit={PIN}\nFB_exit=0\nF1_exit=0\n")
    seal(face_run.run)
    ok, lines = mods.check.f1_gate(face_run.run, face_run.identity, None)
    assert ok and "f1_basis=original" in lines


def test_the_f2_reading_requires_the_f1_basis_when_given(mods, face_run, tmp_path, startup, f1_list) -> None:
    out = simulate(tmp_path, startup, duration=10.0)
    finish(mods, out)
    gate = (face_run.run, face_run.identity, face_run.root)
    first = mods.check.read_f2(out, face_run.run / "f1", gate).lines[0]
    assert first.startswith("FAIL F1 basis for this F2") and "no_accepted_recheck" in first
    recheck_dir, _ = recheck(face_run)
    assert accept(mods, face_run, recheck_dir)[0] == 0
    assert mods.check.read_f2(out, face_run.run / "f1", gate).lines[0].startswith("PASS F1 basis for this F2")
    assert mods.check.read_f2(out, f1_list, gate).lines[0].startswith("FAIL F1 basis")  # another run's F1
    bad = subprocess.run([sys.executable, str(RUNNERS / "face_check.py"), "f2", str(out), "--f1", str(f1_list),
                          "--run", str(face_run.run)], capture_output=True, text=True)
    assert bad.returncode == 2 and "--run and --identity-dir go together" in bad.stderr


# ---------------------------------------------------------------- D86: the natural seated check (descriptive)

NATURAL_SCRIPT = ((0, 210, "facing", 1), (210, 300, "away", 1), (300, 391, "facing", 1))
NATURAL_CUES = {"t0": 0.0, "look": 120.0, "turn": 210.0, "n2": 300.0, "stop": 385.0}


def account_v5(mods, **changes: str) -> dict[str, str]:
    values = {f.key: "yes" for f in mods.account.FIELDS_V5[:4]}
    return {**values, "other_faces_in_view": "no", "notes": "a note that is never printed", **changes}


def test_the_natural_check_reads_execution_then_describes_each_phase(mods, tmp_path, startup, f1_list) -> None:
    """D86: the run, the account and the counts are checked; each phase between the cues is then described with the
    same diagnostic lines as F2's windows (the seated person stays on one track through the turn here). Nothing is
    validated, and the F2 reading refuses this account."""
    out = simulate(tmp_path, startup, script=NATURAL_SCRIPT, duration=390.0)
    twin = tmp_path / "twin"
    shutil.copytree(out, twin)
    finish(mods, out, account=account_v5(mods), cues=NATURAL_CUES, cued_format=5, duration_s=390.0)
    reading = mods.check.read_natural(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    assert line_for(text, "identity records: schema 3").startswith("PASS")
    phases = _value(text, "phases from the guard's cues")
    assert phases == {"N1": [15.1, 105.1], "LOOK": [145.1, 195.1], "TURN": [235.1, 285.1], "N2": [325.1, 375.0]}
    continuity, faces = _value(text, "diagnostics: person continuity"), _value(text, "diagnostics: assigned faces")
    assert continuity["TURN"]["track_ids"] == [1] and continuity["TURN"]["cleared_then_offered"] == []
    assert faces["TURN"]["faces"] == 0 and faces["LOOK"]["faces"] == faces["LOOK"]["people"]
    assert "DESCRIPTIVE diagnostics: the full-scale face pass" in text and "never printed" not in text
    assert mods.evidence.read(out / "account.txt", "f2").nonconforming == []
    _, f2_text = reading_f2 = mods.check.read_f2(out, f1_list), "\n".join(mods.check.read_f2(out, f1_list).lines)
    assert not reading_f2[0].ok and line_for(f2_text, "account complete and in order").startswith("FAIL")
    finish(mods, twin, account=account_v5(mods, swivelled_away_within_10_s="unknown"), cues=NATURAL_CUES,
           cued_format=5, duration_s=390.0)
    declared = mods.check.read_natural(twin, f1_list)
    assert not declared.ok and line_for("\n".join(declared.lines), "every cue printed").startswith("FAIL")
    assert not any(line.startswith("DESCRIPTIVE diagnostics") for line in declared.lines)  # no phases without it


# ---------------------------------------------------------------- D87: the natural recognition comparison


COMPARISON_CUES = {"t0": 0.0, "stop": 290.0}
COMPARISON_CAPTURES = ((60.0, "c1"), (150.0, "c2"), (240.0, "c3"), (290.0, "end"))
NOBODY = vec(0.0, 0.1, 0.99)


def account_v6(mods, **changes: str) -> dict[str, str]:
    return {"stayed_seated": "yes", "worked_as_usual": "yes", "looked_at_camera_on_purpose": "no",
            "other_faces_in_view": "no", "notes": "a note that is never printed", **changes}


def comparison_run(mods, tmp_path, accepted_scene, name: str = "cmp") -> Path:
    """The real runtime with a real comparator for 300 s: seated throughout; production finds the face in alternate
    10 s blocks; the full-scale pass finds it always. The comparison's embedding matches, does not match, or fails,
    by the frame's sequence number."""
    from sentinel.identity.association import AssociationGeometry
    from sentinel.identity.comparison import FullScaleComparator
    from sentinel.identity.legacy_deepface import FaceBackendError, FullScale, FullScaleRow

    root = tmp_path / name
    root.mkdir()
    start_root = root / "startup"
    start_root.mkdir()
    startup = startup_line(real_face_startup(start_root, accepted_scene, comparison=True))
    row = FullScaleRow(121.0, 106.0, 58.0, 62.0, (135.0, 125.0), (160.0, 126.0), 0.93)
    full = FullScale(((0.93, FACE_BOX),), 48.0, (row,))

    def analyze_from(t0):
        def analyze(frame, image):
            block = int((frame.ingest_mono_ns - t0) / 1e9 // 10)
            faces = (FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.9, embedding=FACE),) if block % 2 else ()
            return FaceRun(faces, len(faces), 1 - len(faces), full)
        return analyze

    def embed(frame, image, candidate):
        if frame.frame_seq % 7 == 0:
            raise FaceBackendError("empty_crop")
        return NOBODY if frame.frame_seq % 3 == 0 else FACE

    comparator = FullScaleComparator(embed, AssociationGeometry())
    out = simulate(root, startup, script=((0, 301, "facing", 1),), duration=300.0, analyze=analyze_from,
                   comparator=comparator, captures=COMPARISON_CAPTURES)
    return out


@pytest.fixture
def comparison_out(mods, tmp_path, accepted_scene):
    out = comparison_run(mods, tmp_path, accepted_scene)
    pristine = tmp_path / "pristine"
    shutil.copytree(out, pristine)
    finish(mods, out, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES)
    return SimpleNamespace(out=out, pristine=pristine)


def _comparison_text(mods, out, f1) -> tuple[Any, str]:
    reading = mods.check.read_comparison(out, f1)
    return reading, "\n".join(reading.lines)


def _described(text: str, start: str) -> tuple[str, Any]:
    """(the dependency label, the value) of the DESCRIPTIVE line whose name contains ``start``: the label is always
    right after DESCRIPTIVE, the value is the JSON after the name."""
    line = next(line for line in text.splitlines() if line.startswith("DESCRIPTIVE [") and start in line)
    label = line[len("DESCRIPTIVE ["):line.index("] ")]
    at = line.find(": ")
    while at != -1:
        try:
            return label, json.loads(line[at + 2:])
        except ValueError:
            at = line.find(": ", at + 1)
    raise AssertionError(f"no value in {line!r}")


def test_the_comparison_reading_checks_execution_then_describes_with_dependencies(mods, comparison_out,
                                                                                  f1_list) -> None:
    reading, text = _comparison_text(mods, comparison_out.out, f1_list)
    assert reading.ok, text
    assert line_for(text, "identity records: schema 4").startswith("PASS")
    assert line_for(text, "face admitted for this validation run with the full-scale comparison").startswith("PASS")
    assert line_for(text, "account in format 6").startswith("PASS") and "never printed" not in text
    records = [json.loads(x) for x in (comparison_out.out / "run.jsonl").read_text().splitlines()]
    comparisons = [x for x in records if x.get("identity") == "comparison"]
    assert comparisons and {x["schema"] for x in records if "identity" in x} == {4}
    label, accounting = _described(text, "comparison accounting")
    assert label == "depends on no account answer"
    assert sum(sum(v.values()) for v in accounting["run"].values()) == len(comparisons)
    label, similarity = _described(text, "full-scale-only faces in W")
    assert label == "SUPPORTED by other_faces_in_view=no"
    completed = [c for c in comparisons if c["kind"] == "full_scale_only" and c["disposition"] == "completed"]
    assert similarity["completed"] >= 1 and similarity["at_or_above"] + similarity["below"] == similarity["completed"]
    assert similarity["below"] >= 1 and similarity["failed"] >= 1  # failures are counted apart, never as nonmatches
    assert {c["similarity"] for c in completed} <= {0.993, 0.0}  # FACE and NOBODY against the enrolled vec(1.0)
    label, controls = _described(text, "controls in W")
    assert label.startswith("SUPPORTED") and controls["pairs"] + controls["unpaired"] >= 0
    label, cost = _described(text, "production and the comparison's cost")
    assert label == "depends on no account answer" and cost["comparison_ms"] is not None
    assert _described(text, "sampling in W")[0] == "depends on no account answer"
    assert _described(text, "diagnostics: person continuity")[0] == (
        "SUPPORTED by stayed_seated=yes, worked_as_usual=yes, looked_at_camera_on_purpose=no")
    assert mods.evidence.read(comparison_out.out / "run.jsonl", "f2").nonconforming == []
    assert mods.evidence.read(comparison_out.out / "account.txt", "f2").nonconforming == []
    for name in ("c1", "c2", "c3", "end"):
        assert mods.evidence.read(comparison_out.out / f"status-{name}.json", "f2").nonconforming == []


@pytest.mark.parametrize(("change", "context", "attribution"), [
    ({"worked_as_usual": "unknown"}, "QUALIFIED: worked_as_usual=unknown", "SUPPORTED by other_faces_in_view=no"),
    ({"looked_at_camera_on_purpose": "yes"}, "CONTRADICTED by looked_at_camera_on_purpose=yes",
     "SUPPORTED by other_faces_in_view=no"),
    ({"other_faces_in_view": "unknown"}, "SUPPORTED by stayed_seated=yes, worked_as_usual=yes, "
     "looked_at_camera_on_purpose=no", "QUALIFIED: other_faces_in_view=unknown"),
    ({"other_faces_in_view": "yes"}, "SUPPORTED by stayed_seated=yes, worked_as_usual=yes, "
     "looked_at_camera_on_purpose=no", "CONTRADICTED by other_faces_in_view=yes"),
])
def test_an_uncertain_answer_qualifies_only_the_lines_that_depend_on_it(mods, comparison_out, f1_list, change, context,
                                                                       attribution) -> None:
    baseline = _comparison_text(mods, comparison_out.out, f1_list)[1]
    finish(mods, comparison_out.pristine, account=account_v6(mods, **change), cues=COMPARISON_CUES, cued_format=6,
           duration_s=300.0, captures=COMPARISON_CAPTURES, counts={})
    reading, text = _comparison_text(mods, comparison_out.pristine, f1_list)
    assert reading.ok, text  # an unknown or unexpected answer is accepted: the run's execution is unaffected
    assert _described(text, "diagnostics: person continuity")[0] == context
    assert _described(text, "full-scale-only faces in W")[0] == attribution
    assert _described(text, "controls in W")[0] == attribution
    for start in ("comparison accounting", "production and the comparison's cost", "sampling in W",
                  "monitoring at the last capture"):  # never suppressed or relabelled by an answer
        assert _described(text, start) == _described(baseline, start)
    assert _described(text, "full-scale-only faces in W")[1] == _described(baseline, "full-scale-only faces in W")[1]


def test_nothing_is_described_before_a_locked_format_6_account(mods, comparison_out, f1_list) -> None:
    finish(mods, comparison_out.pristine, account=None, cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading, text = _comparison_text(mods, comparison_out.pristine, f1_list)
    assert not reading.ok and not any(line.startswith("DESCRIPTIVE") for line in reading.lines)
    assert line_for(text, "account saved and locked").startswith("FAIL")


@pytest.mark.parametrize("change", [
    lambda c: c.update(similarity=0.5) if c["disposition"].startswith("skipped") else None,
    lambda c: c.update(similarity=None) if c["disposition"] == "completed" else None,
    lambda c: c.update(ms=None) if c["disposition"] == "completed" else None,
    lambda c: c.update(error="empty_crop") if c["disposition"] == "completed" else None,
    lambda c: c.update(error="a free text error") if c["disposition"] == "failed" else None,
    lambda c: c.update(score=0.85),
    lambda c: c.update(kind="other"),
    lambda c: c.update(schema=3),
    lambda c: c.update(embedding=[0.1]),
])
def test_comparison_records_must_fit_their_disposition(mods, comparison_out, f1_list, change) -> None:
    def edit(lines):
        for item in lines:
            if item.get("identity") == "comparison":
                change(item)
        return lines

    _mutate(comparison_out.pristine, edit)
    finish(mods, comparison_out.pristine, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6,
           duration_s=300.0, captures=COMPARISON_CAPTURES, counts={})
    reading, text = _comparison_text(mods, comparison_out.pristine, f1_list)
    assert not reading.ok and line_for(text, "identity records: schema 4").startswith("FAIL")
    assert not any(line.startswith("DESCRIPTIVE") for line in reading.lines)


def test_format_6_accepts_unknown_answers_and_is_schema_text(mods, tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "provenance.txt").write_text(F2_PROVENANCE)
    (out / "result.json").write_text("{}")
    save_account(mods, out, account_v6(mods, stayed_seated="unknown"), 6)
    values = mods.account.parse((out / "account.txt").read_text())
    review = mods.account.review_separate(values)
    assert review[0].endswith(": stayed_seated") and review[-1].endswith(": yes")
    assert mods.evidence.read(out / "account.txt", "f2").nonconforming == []
    assert [f.key for f in mods.account.FIELDS_V6] == ["stayed_seated", "worked_as_usual",
                                                       "looked_at_camera_on_purpose", "other_faces_in_view", "notes"]
    assert mods.account.main(["enter", str(tmp_path / "x"), "--format", "6"]) == 2  # not a run directory
    assert mods.account.COMPARISON_CUE_AT_S == {"t0": 0, "stop": 290} and mods.account.COMPARISON_DURATION_S == 300


# ---------------------------------------------------------------- D88: the full-scale detection candidate's reading


def as_schema_5(out: Path) -> None:
    """Rewrite a schema-6 run's identity records as the candidate wrote them before D90 (9460548): schema 5, without
    each person's prototype, geometry and shadow, every other field as it was."""
    lines = []
    for line in (out / "run.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item.get("identity") in ("tick", "result", "transition"):
            assert item["schema"] == 6
            item["schema"] = 5
            for person in item.get("persons", ()):
                for key in ("prototype", "geometry", "shadow"):
                    del person[key]
        lines.append(json.dumps(item))
    (out / "run.jsonl").write_text("\n".join(lines) + "\n")


def candidate_run(mods, tmp_path, accepted_scene, name: str = "cand", schema: int = 6) -> Path:
    """The real runtime with the candidate selected for 300 s, seated throughout: the candidate finds the face except
    in every fourth 10 s block (then the best detection under 0.9 is at 0.84, or there is none in alternate such
    blocks); the face found matches, except on frames whose sequence number is a multiple of 5. Its records are
    schema 6 (D90), or, with ``schema=5``, as the candidate wrote them before D90."""
    from sentinel.identity.legacy_deepface import CandidateDetection, FaceGeometry

    root = tmp_path / name
    root.mkdir()
    start_root = root / "startup"
    start_root.mkdir()
    startup = startup_line(real_face_startup(start_root, accepted_scene, candidate=True))

    def analyze_from(t0):
        def analyze(frame, image):
            block = int((frame.ingest_mono_ns - t0) / 1e9 // 10)
            if block % 4 == 3:
                below = (0.84, NormalizedBox(x1=0.2, y1=0.2, x2=0.28, y2=0.3)) if block % 8 == 3 else None
                return FaceRun((), 0, 0, detection=CandidateDetection(below, 47.5, None))
            embedding = NOBODY if frame.frame_seq % 5 == 0 else FACE
            face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.92, embedding=embedding)
            edge = -2.5 if frame.frame_seq % 3 == 0 else 121.6  # a third of the faces reach past the left edge
            geometry = FaceGeometry((edge, 105.6, 57.6, 62.4), (edge + 17.0, 128.0), (edge + 40.0, 129.0))
            return FaceRun((face,), 1, 0, detection=CandidateDetection(None, 48.0, 861.0), geometry=(geometry,))
        return analyze

    out = simulate(root, startup, script=((0, 301, "facing", 1),), duration=300.0, analyze=analyze_from,
                   captures=COMPARISON_CAPTURES, detection="full_scale")
    if schema == 5:
        as_schema_5(out)
    return out


@pytest.fixture(params=[5, 6], ids=["schema5", "schema6"])
def candidate_out(mods, tmp_path, accepted_scene, request):
    out = candidate_run(mods, tmp_path, accepted_scene, schema=request.param)
    pristine = tmp_path / "pristine"
    shutil.copytree(out, pristine)
    finish(mods, out, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES)
    return SimpleNamespace(out=out, pristine=pristine, schema=request.param)


def _candidate_text(mods, out, f1) -> tuple[Any, str]:
    reading = mods.check.read_candidate(out, f1)
    return reading, "\n".join(reading.lines)


def test_the_candidate_reading_checks_execution_then_describes_production_numbers(mods, candidate_out,
                                                                                 f1_list) -> None:
    reading, text = _candidate_text(mods, candidate_out.out, f1_list)
    assert reading.ok, text
    assert line_for(text, f"identity records: schema {candidate_out.schema}").startswith("PASS")
    assert line_for(text, "face admitted for this validation run with the full-scale detection candidate").startswith(
        "PASS")
    assert line_for(text, "account in format 6").startswith("PASS") and "never printed" not in text
    records = [json.loads(x) for x in (candidate_out.out / "run.jsonl").read_text().splitlines()]
    assert {x["schema"] for x in records if "identity" in x} == {candidate_out.schema}
    geometry = [x for x in text.splitlines() if "the assigned faces' geometry" in x]
    if candidate_out.schema == 5:  # read exactly as before D90: no geometry line
        assert geometry == []
    else:
        label, (window,) = _described(text, "diagnostics: the assigned faces' geometry")[0], \
            _described(text, "diagnostics: the assigned faces' geometry")[1].values()
        assert label.startswith("SUPPORTED")
        match = window["match"]
        assert match["faces"] == match["raw_box"] > 0 and match["past_edge"]["left"]["n"] > 0
        assert match["past_edge"]["left"]["px"] == [2.5, 2.5, 2.5] and match["past_edge"]["right"]["n"] == 0
        assert match["roll_deg"][1] == round(math.degrees(math.atan2(1.0, 23.0)), 1) and match["prototype"] == {
            "0": match["faces"]}
    starting = next(x for x in records if x.get("run") == "starting")
    assert starting["startup"]["face"] == {"validation_run": True, "identities_enrolled": 1, "detection": "full_scale"}
    results = [x for x in records if x.get("identity") == "result" and x["outcome"] == "applied"]
    label, cost = _described(text, "production with the candidate")
    assert label == "depends on no account answer"
    assert cost["detect_ms"] == [47.5, 48.0, 48.0] and cost["embed_ms"] == [861.0, 861.0, 861.0]
    assert cost["results"] == {"applied": len(results)} and cost["errors"] == {}
    assert cost["worker_at_last_capture"]["fallback_dropped"] == 0
    label, people = _described(text, "detection and votes per person")
    assert label == ("SUPPORTED by stayed_seated=yes, worked_as_usual=yes, looked_at_camera_on_purpose=no, "
                     "other_faces_in_view=no")
    w = mods.check._comparison_window(COMPARISON_CUES)
    t0 = 1_000_000_000
    in_w = [x for x in results if w[0] <= (x["frame_mono_ns"] - t0 - 100_000_000) / 1e9 < w[1]]
    assert people["one_person"] == len(in_w) and people["more_people_apart"] == 0
    assert people["face_assigned"] == sum(1 for x in in_w if x["faces"])
    assert people["votes"]["match"] + people["votes"]["unknown"] == people["face_assigned"]
    assert people["votes"]["no_face"] == people["one_person"] - people["face_assigned"]
    without = people["without_a_face"]
    assert without["n"] == people["votes"]["no_face"] and without["best_below_0.9"] == [0.84, 0.84, 0.84]
    assert 0 < without["no_detection_at_the_floor"] < without["n"]
    label, identity = _described(text, "identity time")
    assert label.startswith("SUPPORTED") and 0 < identity["known_s"] <= identity["window_s"]
    assert identity["entries_into_known"] >= 1 and identity["window_s"] == round(w[1] - w[0], 1)
    assert "full-scale" not in text.split("account answers", 1)[1].split("diagnostics:")[0]  # no shadow pass lines
    assert mods.evidence.read(candidate_out.out / "run.jsonl", "f2").nonconforming == []
    for name in ("c1", "c2", "c3", "end"):
        assert mods.evidence.read(candidate_out.out / f"status-{name}.json", "f2").nonconforming == []
    assert mods.check.main(["candidate", str(candidate_out.out), "--f1", str(f1_list)]) == 0


def test_a_candidate_run_is_not_read_as_the_comparison_and_the_comparison_not_as_the_candidate(
        mods, candidate_out, f1_list, tmp_path, accepted_scene) -> None:
    reading, text = _comparison_text(mods, candidate_out.out, f1_list)
    assert not reading.ok and line_for(text, "identity records: schema 4").startswith("FAIL")
    assert line_for(text, "face admitted for this validation run with the full-scale comparison").startswith("FAIL")
    comparison = comparison_run(mods, tmp_path, accepted_scene, name="other")
    finish(mods, comparison, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading, text = _candidate_text(mods, comparison, f1_list)
    assert not reading.ok and line_for(text, "identity records: schema 5").startswith("FAIL")
    assert line_for(text, "face admitted for this validation run with the full-scale detection").startswith("FAIL")
    assert not any(line.startswith("DESCRIPTIVE") for line in reading.lines)


@pytest.mark.parametrize("change", [
    lambda r: r["detection"].update(best_below_score=0.95) if r["detection"] else None,  # a face, not under 0.9
    lambda r: r["detection"].update(best_below_score=0.2) if r["detection"] else None,  # under the recording floor
    lambda r: r["detection"].update(best_below_w_px=None) if r["detection"]["best_below_score"] else None,
    lambda r: r["detection"].update(embed_ms=None) if r["faces"] else None,  # a face kept but never embedded
    lambda r: r["detection"].update(detect_ms=None),
    lambda r: r["detection"].update(embedding=[0.1]),
    lambda r: r.update(detection=None),
    lambda r: r.update(full_scale=None),  # schema 5 has no full-scale pass
    lambda r: r.pop("detection"),
    lambda r: r.update(schema=3),
])
def test_schema_5_results_must_carry_fitting_candidate_numbers(mods, candidate_out, f1_list, change) -> None:
    def edit(lines):
        for item in lines:
            if item.get("identity") == "result" and item["outcome"] == "applied":
                change(item)
        return lines

    _mutate(candidate_out.pristine, edit)
    finish(mods, candidate_out.pristine, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6,
           duration_s=300.0, captures=COMPARISON_CAPTURES, counts={})
    reading, text = _candidate_text(mods, candidate_out.pristine, f1_list)
    assert not reading.ok and line_for(text, "identity records: schema").startswith("FAIL")
    assert not any(line.startswith("DESCRIPTIVE") for line in reading.lines)


def _first_assigned(change):
    def edit(lines):
        for item in lines:
            if item.get("identity") == "result" and item["outcome"] == "applied" and item["persons"] \
                    and item["persons"][0]["face"] is not None:
                change(item["persons"][0])
                break
        return lines
    return edit


@pytest.mark.parametrize("change", [
    lambda p: p.update(prototype=8),  # beyond MAX_PROTOTYPES
    lambda p: p.update(prototype=-1),
    lambda p: p.update(prototype=None),  # a similarity was computed: its position must be there
    lambda p: p.update(prototype=0.5),
    lambda p: p.update(geometry=None),  # an assigned face always has its geometry
    lambda p: p["geometry"].update(raw_box=[-1.5, 0.2, 0.3, 0.4]),  # outside RAW_BOUNDS
    lambda p: p["geometry"].update(raw_box=[0.3, 0.2, 0.1, 0.4]),  # x1 after x2
    lambda p: p["geometry"].update(raw_box=None),  # the overhang without the raw box
    lambda p: p["geometry"].update(overhang_px=[-1.0, 0.0, 0.0, 0.0]),
    lambda p: p["geometry"].update(overhang_px=[0.0, 0.0, 0.0]),
    lambda p: p["geometry"].update(box=[0.1, 0.2, 1.2, 0.4]),  # the corrected box is inside the frame
    lambda p: p["geometry"].update(eyes={"roll_deg": 181.0, "spacing_px": 20.0}),
    lambda p: p["geometry"].update(eyes={"roll_deg": 1.0, "spacing_px": -2.0}),
    lambda p: p["geometry"].update(eyes={"roll_deg": 1.0, "spacing_px": 2.0, "nose": [1, 2]}),
    lambda p: p["geometry"].update(image="..."),
    lambda p: p.update(shadow={"similarity": 1.5, "prototype": 0}),
    lambda p: p.update(shadow={"similarity": 0.7, "prototype": 8}),
    lambda p: p.update(shadow={"similarity": 0.7}),
    lambda p: p.update(similarity=None, label="low_quality", vote="none", identity_id=None, prototype=None,
                       shadow={"similarity": 0.7, "prototype": 0}),  # a shadow where nothing was compared
])
def test_schema_6_numbers_must_fit_their_bounds_and_labels(mods, tmp_path, accepted_scene, f1_list, change) -> None:
    out = candidate_run(mods, tmp_path, accepted_scene)
    _mutate(out, _first_assigned(change))
    finish(mods, out, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading, text = _candidate_text(mods, out, f1_list)
    assert not reading.ok and line_for(text, "identity records: schema").startswith("FAIL")
    assert "geometry, prototype or shadow numbers do not fit" in line_for(text, "identity records: schema") or \
        "do not fit its labels" in line_for(text, "identity records: schema")
    assert not any(line.startswith("DESCRIPTIVE") for line in reading.lines)


def test_schema_6_evidence_is_numbers_only_to_the_privacy_profile(mods, tmp_path, accepted_scene) -> None:
    out = candidate_run(mods, tmp_path, accepted_scene)
    assert mods.evidence.read(out / "run.jsonl", "f2").nonconforming == []
    _mutate(out, _first_assigned(lambda p: p["geometry"].update(label="Jane")))
    assert len(mods.evidence.read(out / "run.jsonl", "f2").nonconforming) == 1


def test_nothing_is_described_before_the_candidates_locked_account(mods, candidate_out, f1_list) -> None:
    finish(mods, candidate_out.pristine, account=None, cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading, text = _candidate_text(mods, candidate_out.pristine, f1_list)
    assert not reading.ok and not any(line.startswith("DESCRIPTIVE") for line in reading.lines)
    assert line_for(text, "account saved and locked").startswith("FAIL")


def test_one_denominator_a_result_with_two_people_is_counted_apart(mods, comparison_out, f1_list) -> None:
    """D88's correction: in the comparison check's reading the faces line counted people (207) and the full-scale line
    results (206), because one result had two people. The per-person line counts only results with one person, so
    D86's result-level numbers and the person-level face and vote describe the same people."""
    window = mods.check._comparison_window(COMPARISON_CUES)

    def edit(lines):
        for item in lines:
            if item.get("identity") == "result" and item["outcome"] == "applied" and item["persons"][0]["face"]:
                t = (item["frame_mono_ns"] - 1_100_000_000) / 1e9
                if window[0] + 20 <= t < window[1]:  # one result inside W gains a second, faceless person
                    item["persons"].append({**item["persons"][0], "track_id": 7, "ownership": "no_face",
                                            "vote": "none", "identity_id": None, "label": "no_face", "face": None,
                                            "similarity": None, "box": [0.6, 0.3, 0.9, 0.9], "confidence": 0.45})
                    break
        return lines

    before = _described(_comparison_text(mods, comparison_out.out, f1_list)[1], "detection and votes per person")[1]
    _mutate(comparison_out.pristine, edit)
    finish(mods, comparison_out.pristine, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6,
           duration_s=300.0, captures=COMPARISON_CAPTURES, counts={})
    reading, text = _comparison_text(mods, comparison_out.pristine, f1_list)
    assert reading.ok, text
    after = _described(text, "detection and votes per person")[1]
    assert (after["applied_results"], after["one_person"], after["more_people_apart"]) == (
        before["applied_results"], before["one_person"] - 1, 1)
    assert after["face_assigned"] == before["face_assigned"] - 1  # that result's person left out, never mixed in
    assert after["full_scale_pass"]["results"] == after["one_person"]
    assert sum(after["full_scale_pass"][k] for k in ("both", "production_only", "full_scale_only", "neither")) == \
        after["one_person"]
    assert after["full_scale_pass"]["both"] + after["full_scale_pass"]["production_only"] == after["face_assigned"]
    people = _described(text, "diagnostics: assigned faces per window")[1]["W"]["people"]
    results = _described(text, "the full-scale face pass per window")[1]["W"]["results"]
    assert people == results + 1  # the old lines keep their own denominators, now named apart
    _, cost = _described(text, "production and the comparison's cost")
    assert cost["offered_at_last_capture"] >= cost["yielded_to_comparison"]  # the same capture's two counts


# ---------------------------------------------------------------- D89: F2 for the full-scale detection candidate


def candidate_f2_run(mods, tmp_path, accepted_scene, *, name: str = "cf2", script=SCRIPT, away_face=None,
                     marker: str = "candidate", detection: str = "full_scale", duration: float = 600.0) -> Path:
    """The F2 script run by the real runtime with the candidate's faces (D88's FaceRuns, with its own numbers):
    ``marker`` names the startup line's face entry (``candidate``, ``production`` or ``comparison``) and ``detection``
    the FaceRuntime's, which sets the records' schema (``full_scale``: 5; ``deepface``: 3)."""
    from sentinel.identity.legacy_deepface import CandidateDetection

    root = tmp_path / name
    root.mkdir()
    start_root = root / "startup"
    start_root.mkdir()
    startup = startup_line(real_face_startup(start_root, accepted_scene, candidate=marker == "candidate",
                                             comparison=marker == "comparison"))

    def analyze_from(t0):
        def analyze(frame, image):
            state, _ = segment(script, (frame.ingest_mono_ns - t0) / 1e9)
            embedding = FACE if state == "facing" else away_face if state == "away" else None
            if embedding is None:  # no face at 0.9: never embedded, and no stand-in to drop
                return FaceRun((), 0, 0, detection=CandidateDetection(None, 47.0, None))
            face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.92, embedding=embedding)
            return FaceRun((face,), 1, 0, detection=CandidateDetection(None, 48.0, 861.0))
        return analyze

    return simulate(root, startup, script=script, analyze=analyze_from, detection=detection, duration=duration)


def judged(text: str) -> list[tuple[str, str]]:
    """(verdict, item name) of every judged line; the value is the JSON after the name."""
    out = []
    for line in text.splitlines():
        label, rest = line.split(" ", 1)
        if label == "DESCRIPTIVE":
            continue
        for split in (i for i in range(len(rest)) if rest.startswith(": ", i)):
            try:
                json.loads(rest[split + 2:])
            except ValueError:
                continue
            out.append((label, rest[:split]))
            break
    return out


@pytest.mark.parametrize("schema", [5, 6])
def test_a_candidate_f2_run_is_judged_item_for_item_as_a_production_run(mods, tmp_path, startup, f1_list,
                                                                        accepted_scene, schema) -> None:
    """D89: the same script and account, read once with production's faces (schema 3) and once with the candidate's
    (schema 5, as before D90, or 6): every judged item, its verdict and its criterion are the same; only the three
    items that say which detection produced the faces name the candidate."""
    production = simulate(tmp_path, startup)
    finish(mods, production, account=account_v3(mods), cues=FORMAT3_CUES)
    candidate = candidate_f2_run(mods, tmp_path, accepted_scene)
    if schema == 5:
        as_schema_5(candidate)
    finish(mods, candidate, account=account_v3(mods), cues=FORMAT3_CUES)
    p_result, p_text = reading(mods, production, f1_list)
    c_result, c_text = reading(mods, candidate, f1_list)
    assert p_result.ok, p_text
    assert c_result.ok, c_text
    assert "MISSING" not in c_text and "FAIL" not in c_text and "NOT_EXERCISED" not in c_text
    renamed = {
        "face admitted for this validation run with the full-scale detection candidate (D88): one enrolled identity, "
        "no problem": "face admitted for this validation run: one enrolled identity, no problem",
        "each status capture: face available (validation run, 1 enrolled), scene available, video fresh, no channel, "
        "no delivery row; the candidate's detection": "each status capture: face available (validation run, 1 "
        "enrolled), scene available, video fresh, no channel, no delivery row",
        f"identity records: schema {schema}, complete, none dropped": "identity records: schema 3, complete, none "
        "dropped"}
    assert [(label, renamed.get(name, name)) for label, name in judged(c_text)] == judged(p_text)
    assert line_for(c_text, "A: a known identity is retained").startswith("PASS")
    records = [json.loads(x) for x in (candidate / "run.jsonl").read_text().splitlines()]
    assert {x["schema"] for x in records if "identity" in x} == {schema}
    assert ("the assigned faces' geometry" in c_text) == (schema == 6)  # descriptive, never judged
    assert json.loads(line_for(c_text, "each status capture").split(": ", 1)[1].split("; the candidate's detection: ",
                                                                                         1)[1])[0]["detection"] \
        == "full_scale"


@pytest.mark.parametrize("marker, detection", [("production", "full_scale"), ("candidate", "deepface")])
def test_the_candidates_marker_and_its_records_schema_must_agree(mods, tmp_path, f1_list, accepted_scene, marker,
                                                                 detection) -> None:
    out = candidate_f2_run(mods, tmp_path, accepted_scene, marker=marker, detection=detection, duration=70.0)
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, counts={}, captures=((60.0, "e1"),))
    result, text = reading(mods, out, f1_list)
    assert not result.ok
    assert line_for(text, "face admitted for this validation run").startswith("PASS")
    assert line_for(text, "identity records: schema").startswith("FAIL")
    assert "records with the startup's face" in line_for(text, "identity records: schema")


def test_a_candidate_capture_without_the_candidates_marker_fails(mods, tmp_path, f1_list, accepted_scene) -> None:
    out = candidate_f2_run(mods, tmp_path, accepted_scene, duration=70.0)
    status = json.loads((out / "status-e1.json").read_text())
    del status["components"]["face"]["detection"]
    (out / "status-e1.json").write_text(json.dumps(status))
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, counts={})
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "each status capture").startswith("MISSING")


def test_a_comparison_run_is_still_not_an_f2_run(mods, tmp_path, f1_list, accepted_scene) -> None:
    out = candidate_f2_run(mods, tmp_path, accepted_scene, marker="comparison", detection="deepface", duration=70.0)
    finish(mods, out, account=account_v3(mods), cues=FORMAT3_CUES, counts={})
    result, text = reading(mods, out, f1_list)
    assert not result.ok and line_for(text, "face admitted for this validation run: one").startswith("FAIL")


@pytest.mark.parametrize("case", ["face_visible_while_turned", "face_not_hidden_in_account", "track_lost_while_away"])
def test_the_candidates_retention_is_never_passed_unless_exercised(mods, tmp_path, f1_list, accepted_scene,
                                                                   case) -> None:
    """D89 changes no retention rule: a face still matching while turned, an account saying it was visible, or a
    person lost to the detector while away before the identity's votes expired (the identity cleared with its track,
    never counted as retained) leaves retention NOT_EXERCISED."""
    script = SCRIPT if case != "track_lost_while_away" else (
        SCRIPT[:3] + ((240, 255, "away", 1), (255, 360, "absent", None)) + SCRIPT[4:])
    out = candidate_f2_run(mods, tmp_path, accepted_scene, script=script,
                           away_face=FACE if case == "face_visible_while_turned" else None)
    account = account_v3(mods, **({"face_hidden_while_turned": "no"} if case == "face_not_hidden_in_account" else {}))
    finish(mods, out, account=account, cues=FORMAT3_CUES, counts={})
    result, text = reading(mods, out, f1_list)
    retention = line_for(text, "A: a known identity is retained")
    assert not result.ok and retention.startswith("NOT_EXERCISED"), retention
    if case == "track_lost_while_away":
        transitions = [json.loads(x) for x in (out / "run.jsonl").read_text().splitlines()
                       if json.loads(x).get("identity") == "transition"]
        assert any(t["from"] == "known" and t["to"] == "cleared" for t in transitions)
        assert '"ended": "cleared"' in retention


# ---------------------------------------------------------------- D91: the camera-side enrollment experiment


class WorkStore:
    """A working gallery's directory, in memory (the capture writes it once; nothing is sealed here)."""

    def __init__(self) -> None:
        self.document, self.audits = None, []

    def exists(self) -> bool:
        return self.document is not None

    def ensure_directory(self) -> None:
        pass

    def save(self, document, secret) -> None:
        self.document = document

    def load(self, secret):
        return self.document

    def audit(self, action, identity_id=None, **counts) -> None:
        self.audits.append((action, identity_id, counts))


DRIFT = vec(0.62, 0.78, 0.0, 0.05)  # a face of the enrolled person that the photo prototype does not reach (< 0.70)
NON_ENROLLED_CUES = {"t0": 0.0, "look": 150.0, "stop": 290.0}


def account_v7(**changes: str) -> dict[str, str]:
    return {"consented_to_capture": "yes", "stayed_seated": "yes", "worked_as_usual": "yes",
            "other_faces_in_view": "no", "notes": "a note that is never printed", **changes}


def account_v8(**changes: str) -> dict[str, str]:
    return {"participant_not_enrolled": "yes", "participant_consented": "yes", "participant_stayed_seated": "yes",
            "participant_looked_at_cue": "yes", "only_participant_in_view": "yes",
            "notes": "a note that is never printed", **changes}


def experiment_run(mods, tmp_path, accepted_scene, *, kind: str, faces, name: str = "exp", work=None):
    """The real runtime with the candidate for 300 s, seated throughout, as a capture run (``kind`` "capture") or with
    a working gallery compared read only ("shadow"); ``faces(t)`` gives each analysed frame's embedding."""
    from sentinel.identity.experiment import Capture, Shadow
    from sentinel.identity.legacy_deepface import CandidateDetection

    root = tmp_path / name
    root.mkdir()
    start_root = root / "startup"
    start_root.mkdir()
    startup = startup_line(real_face_startup(start_root, accepted_scene, candidate=True, experiment=kind))

    def analyze_from(t0):
        def analyze(frame, image):
            face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.92,
                                   embedding=faces((frame.ingest_mono_ns - t0) / 1e9))
            return FaceRun((face,), 1, 0, detection=CandidateDetection(None, 48.0, 861.0))
        return analyze

    store = WorkStore()
    shadow = capture = None
    if kind == "shadow":
        shadow = Shadow((work or GALLERY).enrollment(), 1)
    else:
        capture = Capture(store, Secret("a long synthetic passphrase"), GALLERY, "2026-10-14", 0.70)
    out = simulate(root, startup, script=((0, 301, "facing", 1),), duration=300.0, analyze=analyze_from,
                   captures=COMPARISON_CAPTURES, detection="full_scale", shadow=shadow, experiment_capture=capture)
    return out, store


def test_a_capture_reading_says_whether_its_working_gallery_may_be_used(mods, tmp_path, accepted_scene,
                                                                         f1_list) -> None:
    out, store = experiment_run(mods, tmp_path, accepted_scene, kind="capture",
                                faces=lambda t: FACE if int(t) % 3 else DRIFT)
    finish(mods, out, account=account_v7(), cues=COMPARISON_CUES, cued_format=7, duration_s=300.0,
           captures=COMPARISON_CAPTURES)
    reading = mods.check.read_capture(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    assert line_for(text, "the working gallery: written, read back and verified").startswith("PASS")
    assert line_for(text, "the capture's conditions").startswith("PASS")
    label, numbers = _described(text, "the capture's numbers")
    assert label.startswith("SUPPORTED") and numbers["prototypes"] == 8 and len(numbers["similarity_to_original"]) == 8
    (identity,) = store.document.identities
    assert identity.identity_id == ENROLLED and len(identity.prototypes) == 8  # the working gallery, never the original
    for path in sorted(out.iterdir()):  # the capture's record and account fit the privacy profile
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            assert mods.evidence.read(path, "f2").nonconforming == [], path.name
    assert mods.check.main(["capture", str(out), "--f1", str(f1_list)]) == 0


@pytest.mark.parametrize("change", [{"other_faces_in_view": "yes"}, {"consented_to_capture": "unknown"},
                                    {"stayed_seated": "no"}])
def test_a_capture_with_another_face_no_consent_or_a_move_is_not_usable(mods, tmp_path, accepted_scene, f1_list,
                                                                        change) -> None:
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="capture", faces=lambda t: FACE)
    finish(mods, out, account=account_v7(**change), cues=COMPARISON_CUES, cued_format=7, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading = mods.check.read_capture(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and line_for(text, "the capture's conditions").startswith("FAIL")
    assert line_for(text, "the working gallery: written").startswith("PASS")  # written, but not to be used


def test_a_refused_capture_is_read_as_not_usable(mods, tmp_path, accepted_scene, f1_list) -> None:
    out, store = experiment_run(mods, tmp_path, accepted_scene, kind="capture",
                                faces=lambda t: FACE if 60 <= t <= 72 or t < 30 else NOBODY)  # mostly someone else
    finish(mods, out, account=account_v7(), cues=COMPARISON_CUES, cued_format=7, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading = mods.check.read_capture(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and store.document is None
    assert '"refused": "inconsistent_capture"' in line_for(text, "the working gallery: written")


# What the enrollment experiment's -r2 launch blocks write into the run's provenance before the account: the launch
# lines, the framing line (x-C's fixed line; x-G's or x-N's answer, which the block accepts only as none, changed or
# unsure) and the F1 gate's lines for an accepted recheck (GATE_LINES, in face_runs).
FRAMING_LINES = ("framing_baseline=this_capture", "framing_since_capture=none", "framing_since_capture=changed",
                 "framing_since_capture=unsure")


def block_provenance(framing: str) -> str:
    return F2_PROVENANCE + framing + "\n" + GATE_LINES


@pytest.mark.parametrize("framing", FRAMING_LINES)
def test_the_experiment_blocks_framing_lines_fit_the_run_provenance_profile(mods, tmp_path, framing) -> None:
    provenance = tmp_path / "provenance.txt"
    provenance.write_text(block_provenance(framing) + "account_sha256=" + "0" * 64 + "\n")
    reading = mods.evidence.read(provenance, "f2")
    assert reading.nonconforming == [] and reading.content == []  # schema text, nothing searched as content


@pytest.mark.parametrize(("framing", "where"), [
    ("framing_baseline=changed", "line 5: framing_baseline"),  # x-C writes only this_capture
    ("framing_baseline=This_capture", "line 5: framing_baseline"),
    ("framing_baseline=this_capture ", "line 5: framing_baseline"),
    ("framing_baseline=", "line 5: framing_baseline"),
    ("framing_since_capture=this_capture", "line 5: framing_since_capture"),
    ("framing_since_capture=maybe", "line 5: framing_since_capture"),  # x-G and x-N stop before writing these
    ("framing_since_capture=NONE", "line 5: framing_since_capture"),
    ("framing_since_capture=", "line 5: framing_since_capture"),
    ("framing=fixed_from_capture_changed_sessions_apart", "line 5"),  # the decisions file's line, not a run's
    ("framing_note=none", "line 5"),
    ("lighting=dusk", "line 5: lighting"),  # the evaluation blocks accept only daylight, lamps, both or unsure
    ("lighting=Daylight", "line 5: lighting"),
    ("capture_gate=refused", "line 5: capture_gate"),  # a refused gate never launches
    ("capture_basis=accepted", "line 5: capture_basis"),
    ("capture_session=exp-20261013", "line 5: capture_session"),
    ("capture_recheck=exp-20261013T103725Z-f1-0a7c2ce-20261013T121352Z", "line 5: capture_recheck"),
    ("capture_accepted_utc=yesterday", "line 5: capture_accepted_utc"),
])
def test_any_other_framing_line_is_nonconforming_and_its_text_is_content(mods, tmp_path, framing, where) -> None:
    provenance = tmp_path / "provenance.txt"
    provenance.write_text(block_provenance(framing))
    reading = mods.evidence.read(provenance, "f2")
    assert reading.nonconforming == [where] and (where, framing) in reading.content


def test_framing_lines_fit_only_the_runs_provenance_not_f1s(mods, tmp_path) -> None:
    provenance = tmp_path / "provenance.txt"
    provenance.write_text("consent_date=2026-10-08\nframing_baseline=this_capture\n")
    assert mods.evidence.read(provenance, "f1").nonconforming == ["line 2"]


@pytest.mark.parametrize(("framing", "name", "nonconforming"), [
    ("framing_since_capture=none", 0, 0), ("framing_baseline=this_capture", 0, 0),
    ("framing_since_capture=D", 1, 1),  # the family initial typed as the answer: counted, as content
    ("framing_baseline=Jane", 1, 1),
])
def test_the_privacy_counts_still_find_a_name_written_as_a_framing_answer(tmp_path, framing, name,
                                                                         nonconforming) -> None:
    run = tmp_path / "xg"
    run.mkdir()
    (run / "provenance.txt").write_text(block_provenance(framing))
    done = bash_counts("face_counts_f2", str(run))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["provenance.txt passphrase=0", f"provenance.txt name={name}",
                                        f"provenance.txt nonconforming={nonconforming}"]


@pytest.mark.parametrize(("framing", "usable"), [("framing_baseline=this_capture", True),
                                                 ("framing_baseline=changed", False)])
def test_a_capture_is_read_with_the_provenance_its_block_writes(mods, tmp_path, accepted_scene, f1_list, framing,
                                                                usable) -> None:
    """exp-20261013T103725Z: the -r2 capture block wrote framing_baseline=this_capture, which c3a37f0's profile did not
    declare, so its reading failed the privacy counts on "provenance.txt nonconforming" alone. The line fits now;
    another value of it fails exactly as that run's reading did."""
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="capture",
                            faces=lambda t: FACE if int(t) % 3 else DRIFT)
    finish(mods, out, account=account_v7(), cues=COMPARISON_CUES, cued_format=7, duration_s=300.0,
           captures=COMPARISON_CAPTURES, provenance=block_provenance(framing))
    reading = mods.check.read_capture(out, f1_list)
    text = "\n".join(reading.lines)
    counts = line_for(text, "passphrase, name and nonconforming counts")
    assert line_for(text, "the working gallery: written").startswith("PASS")
    if usable:
        assert reading.ok and counts.startswith("PASS"), text
    else:
        assert not reading.ok and counts.startswith("FAIL") and '"nonzero": ["provenance.txt nonconforming"]' in counts
        assert [x for x in reading.lines if not x.startswith(("PASS ", "DESCRIPTIVE "))] == [counts]


@pytest.mark.parametrize(("framing", "complete"), [("framing_since_capture=none", True),
                                                   ("framing_since_capture=unsure", True),
                                                   ("framing_since_capture=maybe", False)])
def test_an_evaluation_session_is_read_with_the_provenance_its_block_writes(mods, tmp_path, accepted_scene, f1_list,
                                                                            framing, complete) -> None:
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="shadow", faces=lambda t: FACE)
    finish(mods, out, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES, provenance=block_provenance(framing))
    reading = mods.check.read_experiment(out, f1_list)
    counts = line_for("\n".join(reading.lines), "passphrase, name and nonconforming counts")
    assert reading.ok == complete and counts.startswith("PASS" if complete else "FAIL")


# ---------------------------------------------------------------- D91: the capture's offline recheck and its acceptance

SESSION = "exp-20261014T080000Z"


@pytest.fixture
def basis(mods, face_run):
    """The F1 basis the experiment builds on: a run whose own F1 did not validate and its accepted corrected reading."""
    out, done = recheck(face_run)
    assert done.returncode == 0 and accept(mods, face_run, out)[0] == 0
    return SimpleNamespace(f1=face_run.run / "f1", gate=(face_run.run, face_run.identity, face_run.root),
                           identity=face_run.identity, root=face_run.root)


def capture_session(mods, tmp_path, accepted_scene, basis, *, account=None, before: str = "absent",
                    tamper=None) -> SimpleNamespace:
    """A session as x-B (twice), x-F, x-C, x-CE at c3a37f0 and x-H left exp-20261013T103725Z: the capture block's
    framing line counted nonconforming by c3a37f0's profile, so its reading said not usable; then sealed. ``tamper``
    changes the capture's files before its reading and the seal."""
    session = tmp_path / "runs" / SESSION
    session.mkdir(parents=True)
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="capture",
                            faces=lambda t: FACE if int(t) % 3 else DRIFT)
    finish(mods, out, account=account or account_v7(), cues=COMPARISON_CUES, cued_format=7, duration_s=300.0,
           captures=COMPARISON_CAPTURES, provenance=block_provenance("framing_baseline=this_capture"))
    capture = Path(shutil.move(out, session / "xc"))
    counts = capture / "secret-counts-identity.txt"
    counts.write_text(counts.read_text().replace("provenance.txt nonconforming=0", "provenance.txt nonconforming=1"))
    if tamper is not None:
        tamper(capture)
    original = mods.check.read_capture(capture, basis.f1, basis.gate)
    assert not original.ok
    (capture / "check.txt").write_text("\n".join([*original.lines, "face-capture: the working gallery is not usable"])
                                       + "\n")
    working = tmp_path / "identity-experiment"
    working.mkdir()
    (working / "gallery.sealed").write_bytes(b"a sealed working gallery")
    digest = hashlib.sha256(b"a sealed working gallery").hexdigest()
    (session / "provenance.txt").write_text(
        f"pin_commit={PIN}\nXB_exit=0\nXB_exit=1\nworking_gallery_before={before}\nXF_exit=0\n"
        f"working_gallery_after={digest}\nXC_exit=0\nXC_check_exit=1\noriginal_enrollment_unchanged=yes\n"
        f"working_copy_removed=yes\nworking_gallery_at_end={digest}\n")
    seal(session)
    return SimpleNamespace(session=session, working=working, out=basis.root / f"{SESSION}-xc-{PIN[:7]}-20261014T090000Z")


def capture_recheck(cs, basis, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return bash_counts("face_recheck_capture", str(cs.session), str(basis.f1), str(basis.gate[0]),
                       str(basis.identity), str(basis.root), str(cs.working), str(cs.out), PIN, env=env)


def accept_capture(mods, cs, basis, typed: str | None = None) -> tuple[int, str]:
    said: list[str] = []
    line = typed if typed is not None else f"ACCEPT {SESSION} capture"
    code = mods.recheck.accept_capture(cs.out, cs.session, basis.f1, *basis.gate, cs.working,
                                       ask=io.StringIO(line + "\n"), say=said.append)
    return code, "\n".join(said)


def test_the_capture_recheck_reads_the_sealed_capture_again_and_changes_nothing(mods, tmp_path, accepted_scene,
                                                                                basis) -> None:
    cs = capture_session(mods, tmp_path, accepted_scene, basis)
    before = tree(cs.session, cs.working, basis.identity)
    done = capture_recheck(cs, basis)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "face-capture-recheck: validated" in done.stdout and "Jane" not in done.stdout
    assert tree(cs.session, cs.working, basis.identity) == before  # the session, both galleries' files unchanged
    assert sorted(p.name for p in cs.out.iterdir()) == ["README.txt", "SHA256SUMS", "check.txt", "provenance.txt",
                                                         "secret-counts-names.txt"]
    assert oct(cs.out.stat().st_mode & 0o777) == "0o700" and all(oct(p.stat().st_mode & 0o777) == "0o600"
                                                                   for p in cs.out.iterdir())
    check = (cs.out / "check.txt").read_text()
    assert all(x.startswith(("PASS ", "DESCRIPTIVE ")) for x in check.splitlines()[:-1])
    exits = line_for(check, "the session's own result")
    assert '"XB_exit": ["0", "1"]' in exits and '"XC_check_exit": ["1"]' in exits and "not usable" in exits
    assert '"identical": true' in line_for(check, "every original capture item")
    for carried in ("passphrase counts carried forward", "camera userinfo counts carried forward"):
        assert '"rescanned": false' in line_for(check, carried)
    names = (cs.out / "secret-counts-names.txt").read_text()
    assert "provenance.txt nonconforming=0" in names and "=1" not in names
    provenance = mods.check._key_values(cs.out / "provenance.txt")
    assert provenance["session_manifest_check"] == "verified" and provenance["pin_commit"] == PIN
    assert "XC_check_exit=1" in (cs.out / "README.txt").read_text()
    assert not any("Jane" in p.read_text() for p in cs.out.iterdir())
    assert mods.check._manifest(cs.out)[1] == []
    code, said = accept_capture(mods, cs, basis, typed="ACCEPT")
    assert code == 1 and "nothing was written" in said and not (cs.out / "acceptance.txt").exists()
    code, said = accept_capture(mods, cs, basis)
    assert code == 0, said
    fields = mods.check._key_values(cs.out / "acceptance.txt")
    assert fields["accepted_reading"] == "capture-recheck" and fields["session"] == SESSION
    assert fields["working_gallery_sha256"] == hashlib.sha256(b"a sealed working gallery").hexdigest()
    assert fields["recheck_check_sha256"] == hashlib.sha256((cs.out / "check.txt").read_bytes()).hexdigest()
    assert oct((cs.out / "acceptance.txt").stat().st_mode & 0o777) == "0o600"
    assert accept_capture(mods, cs, basis)[0] == 1  # never overwritten
    assert tree(cs.session, cs.working, basis.identity) == before


def _gallery_replaced(cs) -> None:
    (cs.working / "gallery.sealed").write_bytes(b"another gallery")


def _session_file_changed(cs) -> None:
    with (cs.session / "xc" / "run.err").open("a") as handle:
        handle.write("edited after the seal\n")


def _original_item_differs(cs) -> None:  # the session's own reading says something this code does not reproduce
    check = cs.session / "xc" / "check.txt"
    check.write_text(check.read_text().replace("PASS clean shutdown", "FAIL clean shutdown", 1))
    (cs.session / "SHA256SUMS").unlink()
    seal(cs.session)


@pytest.mark.parametrize(("case", "item"), [
    ("other_face", "every original capture item other than the privacy counts"),  # a real failure stays one
    ("name_in_notes", "names counted again"),  # a name in the capture's evidence is still found
    ("passphrase", "passphrase counts carried forward"),
    ("gallery_before", "the working gallery is the one this capture wrote"),
    ("gallery_replaced", "the working gallery is the one this capture wrote"),
    ("session_changed", "the session's evidence unchanged"),
    ("original_differs", "every original capture item other than the privacy counts"),
])
def test_the_capture_recheck_fails_closed(mods, tmp_path, accepted_scene, basis, case, item) -> None:
    def passphrase(capture: Path) -> None:
        counts = capture / "secret-counts-identity.txt"
        counts.write_text(counts.read_text().replace("run.jsonl passphrase=0", "run.jsonl passphrase=1"))

    account = {"other_face": account_v7(other_faces_in_view="yes"),
               "name_in_notes": account_v7(notes="Jane stayed at the desk")}.get(case)
    cs = capture_session(mods, tmp_path, accepted_scene, basis, account=account,
                         before="present" if case == "gallery_before" else "absent",
                         tamper=passphrase if case == "passphrase" else None)
    {"gallery_replaced": _gallery_replaced, "session_changed": _session_file_changed,
     "original_differs": _original_item_differs}.get(case, lambda cs: None)(cs)
    done = capture_recheck(cs, basis)
    assert done.returncode == 1 and "face-capture-recheck: not validated" in done.stdout
    assert line_for((cs.out / "check.txt").read_text(), item).startswith("FAIL")
    assert accept_capture(mods, cs, basis)[0] == 1 and not (cs.out / "acceptance.txt").exists()


def test_an_acceptance_refuses_a_changed_recheck_or_gallery(mods, tmp_path, accepted_scene, basis) -> None:
    cs = capture_session(mods, tmp_path, accepted_scene, basis)
    assert capture_recheck(cs, basis).returncode == 0
    check = cs.out / "check.txt"
    kept = check.read_bytes()
    check.write_bytes(kept + b"\n")
    code, said = accept_capture(mods, cs, basis)
    assert code == 1 and "changed since it was written" in said
    check.write_bytes(kept)
    provenance = cs.out / "provenance.txt"  # edited and resealed: the reading no longer validates
    kept_provenance, kept_sums = provenance.read_bytes(), (cs.out / "SHA256SUMS").read_bytes()
    provenance.write_bytes(kept_provenance.replace(b"userinfo_counts=carried_forward", b"userinfo_counts=recounted"))
    (cs.out / "SHA256SUMS").unlink()
    seal(cs.out)
    code, said = accept_capture(mods, cs, basis)
    assert code == 1 and "no longer validates" in said
    provenance.write_bytes(kept_provenance)
    (cs.out / "SHA256SUMS").write_bytes(kept_sums)
    _gallery_replaced(cs)
    code, said = accept_capture(mods, cs, basis)
    assert code == 1 and "no longer validates" in said and not (cs.out / "acceptance.txt").exists()


def test_the_capture_recheck_refuses_a_wrong_name_an_unsealed_session_or_a_second_directory(
        mods, tmp_path, accepted_scene, basis) -> None:
    cs = capture_session(mods, tmp_path, accepted_scene, basis)
    wrong = SimpleNamespace(**{**vars(cs), "out": basis.root / f"{SESSION}-f1-{PIN[:7]}-20261014T090000Z"})
    done = capture_recheck(wrong, basis)
    assert done.returncode == 2 and "<session>-xc-<pin>-<UTC>" in done.stderr and not wrong.out.exists()
    assert capture_recheck(cs, basis).returncode == 0
    done = capture_recheck(cs, basis)
    assert done.returncode == 2 and "exists" in done.stderr
    (cs.session / "SHA256SUMS").unlink()
    other = SimpleNamespace(**{**vars(cs), "out": basis.root / f"{SESSION}-xc-{PIN[:7]}-20261014T100000Z"})
    done = capture_recheck(other, basis)
    assert done.returncode == 2 and "no SHA256SUMS" in done.stderr and not other.out.exists()


# ---------------------------------------------------------------- D91: the capture gate and the evaluation launch

def launch_provenance(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """experiment_launch.sh's function, run by bash as the evaluation blocks run it."""
    return bash_counts("experiment_launch_provenance", *args, env=env,
                       prelude="source benchmarks/runner/experiment_launch.sh;")


BOOT = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"


@pytest.mark.parametrize(("framing", "lighting", "f1", "capture", "message"), [
    ("maybe", "daylight", "f1_gate=passed", "capture_gate=passed", "framing answer"),
    ("none", "dusk", "f1_gate=passed", "capture_gate=passed", "room light"),
    ("none", "", "f1_gate=passed", "capture_gate=passed", "room light"),
    ("none", "lamps", "f1_gate=refused", "capture_gate=passed", "F1 gate"),
    ("none", "lamps", "f1_gate=passed", "capture_gate=refused\ncapture_gate_reason=no_accepted_recheck", "capture gate"),
])
def test_the_launch_provenance_is_refused_with_nothing_written_for_what_the_profile_refuses(framing, lighting, f1,
                                                                                         capture, message) -> None:
    done = launch_provenance(PIN, BOOT, framing, lighting, f1, capture)
    assert done.returncode == 2 and done.stdout == "" and message in done.stderr


def accepted_capture(mods, tmp_path, accepted_scene, basis) -> SimpleNamespace:
    cs = capture_session(mods, tmp_path, accepted_scene, basis)
    assert capture_recheck(cs, basis).returncode == 0 and accept_capture(mods, cs, basis)[0] == 0
    return cs


def test_the_capture_gate_passes_only_on_an_accepted_recheck_that_still_matches(mods, tmp_path, accepted_scene,
                                                                                basis) -> None:
    cs = capture_session(mods, tmp_path, accepted_scene, basis)

    def gate() -> tuple[bool, list[str]]:
        return mods.check.capture_gate(cs.session, cs.working, basis.f1, basis.gate)

    def refused(reason: str) -> None:
        assert gate() == (False, ["capture_gate=refused", f"capture_gate_reason={reason}"])

    refused("no_accepted_recheck")
    assert capture_recheck(cs, basis).returncode == 0
    refused("no_accepted_recheck")  # a recheck alone is not the maintainer's decision
    assert accept_capture(mods, cs, basis)[0] == 0
    ok, lines = gate()
    assert ok and lines == ["capture_gate=passed", f"capture_session={SESSION}", "capture_basis=accepted_recheck",
                            f"capture_recheck={cs.out.name}",
                            "capture_accepted_utc=" + mods.check._key_values(cs.out / "acceptance.txt")["accepted_utc"]]
    cli = subprocess.run([sys.executable, str(RUNNERS / "face_check.py"), "capture-gate", str(cs.session),
                          "--working-gallery", str(cs.working), "--f1", str(basis.f1), "--run", str(basis.gate[0]),
                          "--identity-dir", str(basis.identity), "--recheck-root", str(basis.root)],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert cli.returncode == 0 and cli.stdout.splitlines() == lines
    acceptance = cs.out / "acceptance.txt"
    text = acceptance.read_text()
    acceptance.write_text(text.replace("confirmation=ACCEPT", "confirmation=accept"))
    refused("acceptance_does_not_match")
    acceptance.write_text(text)
    check = cs.out / "check.txt"
    kept = check.read_bytes()
    check.write_bytes(kept + b"\n")
    refused("recheck_changed")
    check.write_bytes(kept)
    other = basis.root / f"{SESSION}-xc-{PIN[:7]}-20261014T100000Z"
    shutil.copytree(cs.out, other)
    refused("several_accepted_rechecks")
    shutil.rmtree(other)
    gallery = (cs.working / "gallery.sealed").read_bytes()
    _gallery_replaced(cs)
    refused("working_gallery_not_the_captures")
    (cs.working / "gallery.sealed").write_bytes(gallery)
    names = cs.out / "secret-counts-names.txt"  # resealed: the recheck no longer validates, the acceptance no longer matches
    kept_names, kept_sums = names.read_bytes(), (cs.out / "SHA256SUMS").read_bytes()
    names.write_bytes(kept_names.replace(b"account.txt name=0", b"account.txt name=1"))
    (cs.out / "SHA256SUMS").unlink()
    seal(cs.out)
    (cs.out / "SHA256SUMS").write_bytes(b"".join(x + b"\n" for x in (cs.out / "SHA256SUMS").read_bytes().splitlines()
                                                 if b"acceptance.txt" not in x))
    refused("recheck_no_longer_validates")
    names.write_bytes(kept_names)
    (cs.out / "SHA256SUMS").write_bytes(kept_sums)
    assert gate()[0]
    _session_file_changed(cs)
    refused("capture_evidence_changed")


@pytest.mark.parametrize(("verdict", "passed"), [("face-capture: the working gallery may be used in the experiment "
                                                   "(it validates nothing)", True),
                                                  ("face-capture: the working gallery is not usable", False)])
def test_the_capture_gate_takes_a_capture_whose_own_reading_said_usable(mods, tmp_path, accepted_scene, basis, verdict,
                                                                         passed) -> None:
    cs = capture_session(mods, tmp_path, accepted_scene, basis)
    provenance = cs.session / "provenance.txt"
    provenance.write_text(provenance.read_text().replace("XC_check_exit=1", "XC_check_exit=0"))
    check = cs.session / "xc" / "check.txt"
    check.write_text("".join(f"{x}\n" for x in [*check.read_text().splitlines()[:-1], verdict]))
    (cs.session / "SHA256SUMS").unlink()
    seal(cs.session)
    ok, lines = mods.check.capture_gate(cs.session, cs.working, basis.f1, basis.gate)
    assert (ok, lines[-1]) == ((True, "capture_basis=original") if passed else
                               (False, "capture_gate_reason=capture_check_not_usable"))


@pytest.mark.parametrize(("kind", "framing", "lighting"), [("genuine", "none", "lamps"),
                                                           ("genuine", "changed", "daylight"),
                                                           ("non-enrolled", "none", "both")])
def test_an_evaluation_sessions_launch_provenance_passes_the_privacy_and_reading_path(mods, tmp_path, accepted_scene,
                                                                                      basis, kind, framing,
                                                                                      lighting) -> None:
    """x-G's and x-N's run provenance as the package writes it (experiment_launch.sh, with the real F1 and capture
    gates' lines on an accepted capture recheck), then the account and face_counts.sh's counts, read by face_check
    experiment with the F1 basis: execution complete, every file conforming, the recorded conditions shown."""
    cs = accepted_capture(mods, tmp_path, accepted_scene, basis)
    f1_ok, f1_lines = mods.check.f1_gate(*basis.gate)
    capture_ok, capture_lines = mods.check.capture_gate(cs.session, cs.working, basis.f1, basis.gate)
    assert f1_ok and capture_ok
    done = launch_provenance(PIN, BOOT, framing, lighting, "\n".join(f1_lines), "\n".join(capture_lines))
    assert done.returncode == 0, done.stderr
    assert [x.split("=", 1)[0] for x in done.stdout.splitlines()] == [
        "f2_launch_utc", "boot_id", "commit", "pin_commit", "framing_since_capture", "lighting",
        *[x.split("=", 1)[0] for x in f1_lines + capture_lines]]
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="shadow", name="eval", faces=lambda t: FACE)
    genuine = kind == "genuine"
    finish(mods, out, account=account_v6(mods) if genuine else account_v8(),
           cues=COMPARISON_CUES if genuine else NON_ENROLLED_CUES, cued_format=6 if genuine else 8, duration_s=300.0,
           captures=COMPARISON_CAPTURES, provenance=done.stdout)
    if not genuine:  # x-NE counts the participant's name too
        counted = bash_counts("face_counts_f2", str(out), env={**TERMS, "N3": "Ola"})
        assert counted.returncode == 0, counted.stderr
        (out / "secret-counts-identity.txt").write_text(counted.stdout)
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            assert mods.evidence.read(path, "f2").nonconforming == [], path.name
    reading = mods.check.read_experiment(out, basis.f1, basis.gate)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    assert line_for(text, "passphrase, name and nonconforming counts").startswith("PASS")
    assert line_for(text, "F1 basis for this run").startswith("PASS")
    shown = line_for(text, "the run's conditions as its launch recorded them")
    conditions = json.loads(shown[shown.index(": {") + 2:])
    assert conditions == {"framing_since_capture": framing, "lighting": lighting, "capture_session": SESSION,
                          "capture_basis": "accepted_recheck", "capture_recheck": cs.out.name}


def test_a_genuine_session_is_compared_face_by_face_and_replayed_under_each_gallery(mods, tmp_path, accepted_scene,
                                                                                   f1_list) -> None:
    from sentinel.identity.gallery import GalleryDocument, GalleryIdentity
    work = GalleryDocument(compatibility=COMPAT, identities=(GalleryIdentity(
        identity_id=ENROLLED, enrolled_utc=NOW, consent_date="2026-10-14", consent_scope=CONSENT_SCOPE,
        prototypes=(DRIFT, FACE)),))
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="shadow", work=work,
                            faces=lambda t: DRIFT if 100 <= t < 160 else FACE)
    finish(mods, out, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES)
    reading = mods.check.read_experiment(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    label, paired = _described(text, "paired comparison on the same faces")
    assert label.startswith("SUPPORTED")
    assert paired["faces"] == paired["with_working"] > 0
    assert paired["working_only"] == paired["at_or_above"]["working"] - paired["at_or_above"]["original"] > 0
    assert paired["original_only"] == 0 and paired["at_or_above"]["either"] == paired["at_or_above"]["working"]
    _, replay = _described(text, "identity time replayed")
    assert replay["replay_original"]["known_s"] == replay["runtime_original"]["known_s"]  # the replay is the rule
    assert replay["replay_working"]["known_s"] > replay["replay_original"]["known_s"]
    assert replay["replay_either"]["known_s"] == replay["replay_working"]["known_s"]
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            assert mods.evidence.read(path, "f2").nonconforming == [], path.name
    assert mods.check.main(["experiment", str(out), "--f1", str(f1_list)]) == 0


def test_a_non_enrolled_session_counts_any_face_at_the_threshold_with_either_gallery(mods, tmp_path, accepted_scene,
                                                                                    f1_list) -> None:
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="shadow", faces=lambda t: NOBODY)
    result = json.loads((out / "run.jsonl").read_text().splitlines()[0])
    assert result["startup"]["face"]["shadow"] is True
    cues = NON_ENROLLED_CUES
    finish(mods, out, account=account_v8(), cues=cues, cued_format=8, duration_s=300.0,
           captures=COMPARISON_CAPTURES)
    reading = mods.check.read_experiment(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    kind = line_for(text, "session kind and account answers")
    assert json.loads(kind.split(": ", 1)[1])["kind"] == "non-enrolled"
    _, paired = _described(text, "paired comparison on the same faces")
    assert paired["faces"] > 0 and paired["at_or_above"] == {"original": 0, "working": 0, "either": 0}
    assert _described(text, "the same, within 30 s of the look cue")[1]["faces"] > 0
    _, replay = _described(text, "identity time replayed")
    assert all(replay[k]["known_s"] == 0.0 for k in ("replay_original", "replay_working", "replay_either"))


def test_an_experiment_session_needs_its_marker_its_shadows_and_its_cues(mods, tmp_path, accepted_scene,
                                                                         f1_list) -> None:
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="shadow", faces=lambda t: FACE)

    def drop_one_shadow(lines):
        for item in lines:
            if item.get("identity") == "result" and item["outcome"] == "applied" and item["persons"] \
                    and item["persons"][0]["similarity"] is not None:
                item["persons"][0]["shadow"] = None
                break
        return lines

    _mutate(out, drop_one_shadow)
    finish(mods, out, account=account_v8(), cues={"t0": 0.0, "stop": 290.0}, cued_format=8, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading = mods.check.read_experiment(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and not any(line.startswith("DESCRIPTIVE paired") for line in reading.lines)
    assert line_for(text, "identity records: schema 6").startswith("FAIL")
    assert line_for(text, "the run's 300 s and every cue").startswith("FAIL")  # format 8's look cue is missing
    capture_run, _ = experiment_run(mods, tmp_path, accepted_scene, kind="capture", faces=lambda t: FACE, name="cap")
    finish(mods, capture_run, account=account_v6(mods), cues=COMPARISON_CUES, cued_format=6, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading = mods.check.read_experiment(capture_run, f1_list)  # a capture run is not an evaluation session
    assert not reading.ok and line_for("\n".join(reading.lines),
                                       "face admitted for this validation run").startswith("FAIL")
    assert not mods.check.read_f2(capture_run, f1_list).ok  # nor an F2 run


@pytest.mark.parametrize("change", [
    lambda c: c.update(written=True, refused="too_few_faces"),
    lambda c: c.update(verified=True, written=False, refused="too_few_faces"),
    lambda c: c.update(refused="something_else"),
    lambda c: c.update(prototypes=9),
    lambda c: c.update(similarity_to_original=[0.5]),
    lambda c: c.update(faces_in_span=10_000),
    lambda c: c.update(schema=5),
    lambda c: c.update(embedding=[0.1]),
])
def test_the_capture_record_must_fit_its_outcome(mods, tmp_path, accepted_scene, f1_list, change) -> None:
    out, _ = experiment_run(mods, tmp_path, accepted_scene, kind="capture", faces=lambda t: FACE)

    def edit(lines):
        for item in lines:
            if item.get("identity") == "capture":
                change(item)
        return lines

    _mutate(out, edit)
    finish(mods, out, account=account_v7(), cues=COMPARISON_CUES, cued_format=7, duration_s=300.0,
           captures=COMPARISON_CAPTURES, counts={})
    reading = mods.check.read_capture(out, f1_list)
    assert not reading.ok and line_for("\n".join(reading.lines), "identity records: schema").startswith("FAIL")


def test_the_replay_is_the_runtimes_rule_on_runs_with_contradictions_and_lost_tracks(mods, tmp_path, accepted_scene,
                                                                                      startup) -> None:
    """The replay used for the working gallery reproduces, with the original's labels, the KNOWN time the runtime
    itself recorded: on the F2 script (absences, a turn, a track lost and a new one) and on a seated run with
    intermittent unknown votes."""
    runs = [(simulate(tmp_path, startup), 600.0)]
    runs.append((candidate_run(mods, tmp_path, accepted_scene, name="intermittent"), 300.0))
    for out, duration in runs:
        records, problems, _, _ = mods.check.read_records(out / "run.jsonl")
        assert not problems
        line = mods.check.Timeline(records, 1_000_000_000, duration)
        for window in ((15.0, 275.0), (0.0, duration), (100.0, 400.0)):
            runtime = mods.check._identity_time(line, window)
            replay = mods.check._replay_known(line, window, lambda p: p["label"] if p["label"] in ("match", "unknown")
                                              else None)
            assert replay["known_s"] == runtime["known_s"], (out, window, replay, runtime)
            assert replay["entries_into_known"] == runtime["entries_into_known"], (out, window)


# ---------------------------------------------------------------- D95: the live dashboard check (descriptive)

DASHBOARD_CUE_TIMES = {"t0": 0.0, "look": 90.0, "turn": 150.0, "leave": 210.0, "back": 270.0, "pause": 360.0,
                       "resume": 400.0, "stop": 465.0}
DASHBOARD_SCRIPT = ((0, 210, "facing", 1), (210, 270, "absent", None), (270, 481, "facing", 2))


def _plain(text: str, start: str) -> Any:
    """The JSON value of the unlabelled DESCRIPTIVE line whose name starts with ``start``."""
    line = next(line for line in text.splitlines() if line.startswith(f"DESCRIPTIVE {start}"))
    return json.loads(line[line.index(": {") + 2:])


def dashboard_run(mods, tmp_path, accepted_scene, *, name: str = "dash") -> Path:
    """The real runtime as d-L runs it: face recognition (production detection, a validation run) with the live view
    and no scene server, 480 s, the package's captures; the person leaves at 3:30 and comes back as a new track."""
    root = tmp_path / name
    root.mkdir()
    (root / "startup").mkdir()
    startup = startup_line(real_face_startup(root / "startup", accepted_scene))
    del startup["startup"]["scene_server"]
    return simulate(root, startup, script=DASHBOARD_SCRIPT, duration=480.0, preview=True,
                    captures=tuple(mods.check.DASHBOARD_CAPTURES))


def test_the_dashboard_reading_judges_execution_and_privacy_and_describes_recognition(mods, tmp_path, accepted_scene,
                                                                                     f1_list) -> None:
    out = dashboard_run(mods, tmp_path, accepted_scene)
    finish(mods, out, account=None, cues=DASHBOARD_CUE_TIMES, duration_s=480.0,
           captures=tuple(mods.check.DASHBOARD_CAPTURES), provenance=F2_PROVENANCE + GATE_LINES)
    for path in sorted(out.iterdir()):  # every evidence file fits its profile: the counts read it strictly
        if path.is_file() and path.name not in mods.check.NOT_COUNTED:
            assert mods.evidence.read(path, "f2").nonconforming == [], path.name
    reading = mods.check.read_dashboard(out, f1_list)
    text = "\n".join(reading.lines)
    assert reading.ok, text
    assert not [line for line in reading.lines if line.startswith(("FAIL", "MISSING"))]
    captures = _plain(text, "each capture as the page showed it")
    assert set(captures) == {"n1", "look", "turn", "e2", "n2", "end"}
    assert captures["n1"]["rows"][0][:3] == [1, "known", "fresh"] and captures["e2"]["rows"] == []
    assert captures["n2"]["rows"][0][:2] == [2, "known"]  # the returning person is a new track, recognized anew
    assert captures["n1"]["live_view"] == "live"
    segments = _plain(text, "per cue interval")
    assert segments["away"]["known_s"] == 0.0 or segments["away"]["ended_to"]  # nothing KNOWN carried into the absence
    assert segments["back at the seat"]["entries_into_known"] == 1
    assert mods.check.main(["dashboard", str(out), "--f1", str(f1_list)]) == 0
    assert "similarity" not in text and ENROLLED not in "\n".join(line for line in reading.lines
                                                                   if line.startswith("DESCRIPTIVE each capture"))


def test_the_dashboard_reading_needs_the_live_view_its_captures_and_its_cues(mods, tmp_path, accepted_scene,
                                                                            f1_list) -> None:
    out = dashboard_run(mods, tmp_path, accepted_scene)
    (out / "status-turn.json").unlink()
    finish(mods, out, account=None, cues={k: v for k, v in DASHBOARD_CUE_TIMES.items() if k != "pause"},
           duration_s=480.0, captures=tuple(mods.check.DASHBOARD_CAPTURES), counts={})
    reading = mods.check.read_dashboard(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok
    assert line_for(text, "the run's 480 s and every cue").startswith("FAIL")
    assert line_for(text, "the page's status at each capture").startswith("FAIL")
    assert mods.check.main(["dashboard", str(out), "--f1", str(f1_list)]) == 1


def test_a_run_with_the_scene_server_or_without_the_live_view_is_not_this_check(mods, tmp_path, accepted_scene,
                                                                                f1_list) -> None:
    root = tmp_path / "scene"
    root.mkdir()
    (root / "startup").mkdir()
    startup = startup_line(real_face_startup(root / "startup", accepted_scene))  # with the scene server, no live view
    out = simulate(root, startup, script=DASHBOARD_SCRIPT, duration=480.0,
                   captures=tuple(mods.check.DASHBOARD_CAPTURES))
    finish(mods, out, account=None, cues=DASHBOARD_CUE_TIMES, duration_s=480.0,
           captures=tuple(mods.check.DASHBOARD_CAPTURES), counts={})
    reading = mods.check.read_dashboard(out, f1_list)
    text = "\n".join(reading.lines)
    assert not reading.ok and line_for(text, "face admitted for this validation run with production detection") \
        .startswith("FAIL")
