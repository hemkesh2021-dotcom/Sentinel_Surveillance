"""The Review 1 rehearsal's reading of its saved evidence (benchmarks/runner/rehearsal_check.py), on portable fixtures.

The fixture is what the package's blocks would leave behind, made with the real parts wherever the device run would
use them: the incident service records a zone observation's incident, the snapshot store saves its frame, and
``sentinel incident ack`` acknowledges it between the two runs. The guard's files and the status captures are written
as step5_guard writes them. Every judged item is shown to pass on complete evidence and to fail, or be missing, when
its evidence says otherwise.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from status_fixtures import PICTURE

from sentinel.cli import main as sentinel
from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.contracts import PixelFormat, TrackKey
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import signal_from_zone
from sentinel.incidents.snapshots import SNAPSHOT_DIR, SnapshotJob, SnapshotStore
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import EncodedJpeg
from sentinel.rules.scene_hazard import Severity
from sentinel.rules.zones import ZoneObservation, ZonePhase
from sentinel.scene.chat import CHAT_ADMISSION
from sentinel.storage.database import SCHEMA_VERSION, Database

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
T0 = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
LOOP = {"port": 18090, "family": "tcp", "scope": "loopback"}
CHAT = {"state": "unavailable", "reason": CHAT_ADMISSION.reason, "counters": {}}
PROVENANCE = """utc=2026-10-20T08:58:00Z
commit=0123456789abcdef0123456789abcdef01234567
pin_commit=0123456789abcdef0123456789abcdef01234567
tracked_changes=0
shell_thp_enabled=1
data_dir_entries_before=0
RHa_exit=0
camera_probe_exit=0
display_manager_before=active
display_manager=inactive v1_procs=0 llama_server_procs=0
RHd_exit=0
RHe_exit=0
RHf_exit=0
"""


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("rehearsal_check")


def utc(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def ack(data: Path, incident: str, at: datetime, **options: str) -> dict[str, Any]:
    out = io.StringIO()
    extra = [part for key, value in options.items() for part in (f"--{key}", value)]
    with contextlib.redirect_stdout(out):
        assert sentinel(["incident", "ack", "--data-dir", str(data), incident, *extra],
                        clock=FakeClock(utc=at)) == 0
    return json.loads(out.getvalue())


def guard_result(name: str, launched: datetime, exited: datetime, captures: tuple[str, ...]) -> dict[str, Any]:
    return {"status": "duration_stop", "cleanup_clear": True, "run_error_label": None,
            "child": {"returncode": 0, "launched_utc": utc(launched), "exited_utc": utc(exited)},
            "guard": {"trigger": None, "peak_pressure_bytes": 1, "min_mem_free_bytes": 2, "min_mem_available_bytes": 3},
            "stop": {"forced": False},
            "captures": [{"name": capture, "status": "ok", "listeners": [dict(LOOP)]} for capture in captures]}


def starting(**changes: Any) -> str:
    startup = {"memory_policy": {"thp": "workload_disabled", "model_file_release": "post_load"},
               "snapshots": {"saved_files": 0, "policy": {}}, "detector_problem": None, "scene_problem": None,
               "notifier_problems": {}, "live_preview": {"max_width": 640},
               "chat": {"state": "unavailable", "reason": CHAT_ADMISSION.reason}, **changes}
    return json.dumps({"run": "starting", "startup": startup}) + "\n" + json.dumps({"run": "stopped"}) + "\n"


def status(incident: str, state: str, *, encodes: int = 12, saved: int = 1) -> dict[str, Any]:
    return {"runtime": {"state": "running"},
            "preview": {"enabled": True, "state": "live", "reason": "the newest frame", "picture": None, "problem": None},
            "components": {"preview": {"problem": None, "counters": {"encodes": encodes}}, "chat": json.loads(json.dumps(CHAT)),
                           "snapshots": {"state": "running", "problem": None, "store": {"saved": saved}}},
            "incidents": {"recent": [{"incident_id": incident, "status": state}]}}


@pytest.fixture
def rehearsal(tmp_path: Path, mod) -> SimpleNamespace:
    """A complete, passing rehearsal: RUN_DIR as the blocks leave it, and the run's data directory."""
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir(mode=0o700)
    run.mkdir()
    clock = FakeClock(utc=T0)
    db = Database.open(data / "sentinel.db")
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    clock.advance(1.0)
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    observation = ZoneObservation(
        observation_id="whole-view-1.entered", episode_id="whole-view-1", kind="zone.restricted_entry",
        phase=ZonePhase.ENTERED, zone_id="whole-view", zone_revision="rev-1", severity=Severity.WARNING,
        track=TrackKey(camera_id="cam-1", boot_id=frame.boot_id, run_id=frame.run_id, stream_epoch=frame.stream_epoch,
                       track_id=3),
        first_source=frame.key, last_source=frame.key, first_mono_ns=frame.ingest_mono_ns,
        last_mono_ns=frame.ingest_mono_ns, observed_utc=frame.ingest_utc, duration_ms=1000,
        reason="track 3 stayed in the zone for 1.0 s (at least 1.0 s)")
    service = IncidentService(db, clock, incidents=IncidentsConfig(), notifications=NotificationsConfig())
    incident = service.record(signal_from_zone(observation)).incident_id
    store = SnapshotStore(db, data / SNAPSHOT_DIR, clock, None, encode=lambda *_, **__: EncodedJpeg(PICTURE, 8, 8),
                          disk_usage=lambda _: SimpleNamespace(total=10**11, free=5 * 10**10))
    store.open()
    store.save(SnapshotJob(incident_id=incident, observation_id=observation.observation_id, key=frame.key,
                           frame=frame, image=object(), missing=None))
    store.close()
    db.close()
    # RHd: run 1; RHe: between the runs, read only, then the acknowledgments; RHf: run 2; RHg: the decode.
    (run / "run1").mkdir()
    (run / "run1" / "result.json").write_text(json.dumps(
        guard_result("run1", T0 - timedelta(minutes=1), T0 + timedelta(minutes=5), ("live1", "run1"))))
    (run / "run1" / "run.jsonl").write_text(starting())
    (run / "run1" / "status-live1.json").write_text(json.dumps(status(incident, "open")))
    (run / "run1" / "status-run1.json").write_text(json.dumps(status(incident, "open")))
    assert mod.main(["first-incident", "--data-dir", str(data)]) == 0
    (run / "incident-id.txt").write_text(incident + "\n")
    (run / "snapshots-after-run1.json").write_text(json.dumps(mod.snapshot_rows(data), indent=2))
    first = ack(data, incident, T0 + timedelta(minutes=6), actor="maintainer", note="review 1 rehearsal")
    (run / "ack-1.json").write_text(json.dumps(first))
    (run / "ack-2.json").write_text(json.dumps(ack(data, incident, T0 + timedelta(minutes=6, seconds=5))))
    (run / "run2").mkdir()
    (run / "run2" / "result.json").write_text(json.dumps(
        guard_result("run2", T0 + timedelta(minutes=7), T0 + timedelta(minutes=12), ("live2", "run2"))))
    (run / "run2" / "run.jsonl").write_text(starting())
    (run / "run2" / "status-live2.json").write_text(json.dumps(status(incident, "acknowledged", saved=0)))
    (run / "run2" / "status-run2.json").write_text(json.dumps(status(incident, "acknowledged", saved=0)))
    file_name = mod.snapshot_rows(data)[0]
    (run / "decoded.json").write_text(json.dumps({sorted((data / SNAPSHOT_DIR).iterdir())[0].name: [8, 8]}))
    (run / "provenance.txt").write_text(PROVENANCE)
    return SimpleNamespace(run=run, data=data, incident=incident, ack=first, row=file_name)


def lines(mod, rehearsal: SimpleNamespace) -> tuple[bool, list[str]]:
    reading = mod.read(rehearsal.run, rehearsal.data)
    return reading.ok, reading.lines


def edit(path: Path, change: Any) -> None:
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_a_complete_rehearsal_passes_every_item(mod, rehearsal: SimpleNamespace, capsys) -> None:
    detail = rehearsal.run / "incident-detail.json"
    code = mod.main(["check", str(rehearsal.run), "--data-dir", str(rehearsal.data), "--detail-out", str(detail)])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and out[-1] == "review1-rehearsal: passed", "\n".join(out)
    judged = [line for line in out[:-1] if not line.startswith("DESCRIPTIVE ")]
    assert judged and all(line.startswith("PASS ") for line in judged)
    assert len(judged) == 30  # every predeclared item, each judged once
    assert sum(line.startswith("DESCRIPTIVE ") for line in out) == 5  # memory twice, counters twice, row counts
    assert json.loads(detail.read_text())["acknowledgment"]["note"] == "review 1 rehearsal"
    assert rehearsal.ack["outcome"] == "acknowledged" and rehearsal.ack["actor"] == "operator:maintainer"


def test_the_helper_commands_read_the_database_only(mod, rehearsal: SimpleNamespace, capsys, tmp_path: Path) -> None:
    before = (rehearsal.data / "sentinel.db").read_bytes()
    assert mod.main(["first-incident", "--data-dir", str(rehearsal.data)]) == 0
    assert capsys.readouterr().out.strip() == rehearsal.incident
    assert mod.main(["snapshot-rows", "--data-dir", str(rehearsal.data)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["state"] for row in rows] == ["saved"] and rows[0]["incident_id"] == rehearsal.incident
    assert (rehearsal.data / "sentinel.db").read_bytes() == before
    empty = tmp_path / "empty"
    empty.mkdir()
    Database.open(empty / "sentinel.db").close()
    assert mod.main(["first-incident", "--data-dir", str(empty)]) == 1
    assert capsys.readouterr().err.strip() == "no incident in this database"


def _set_provenance(key: str, value: str) -> Any:
    def change(r: SimpleNamespace) -> None:
        text = (r.run / "provenance.txt").read_text()
        (r.run / "provenance.txt").write_text(text.replace(f"{key}=0\n", f"{key}={value}\n"))
    return change


def _capture(run: str, name: str, change: Any) -> Any:
    return lambda r: edit(r.run / run / f"status-{name}.json", change)


def _result(run: str, change: Any) -> Any:
    return lambda r: edit(r.run / run / "result.json", change)


def _sql(statement: str) -> Any:
    def change(r: SimpleNamespace) -> None:
        connection = sqlite3.connect(r.data / "sentinel.db")
        with connection:
            connection.execute(statement)
        connection.close()
    return change


def _tamper(r: SimpleNamespace) -> None:
    picture = sorted((r.data / SNAPSHOT_DIR).iterdir())[0]
    picture.write_bytes(picture.read_bytes() + b"\x00")


def _late_ack(r: SimpleNamespace) -> None:  # run 2 was launched before the acknowledgment
    edit(r.run / "run2" / "result.json",
         lambda d: d["child"].update(launched_utc=utc(T0 + timedelta(minutes=5, seconds=30))))


CASES = {
    "a block failed": (_set_provenance("RHe_exit", "1"), "blocks RHa-RHf"),
    "the data directory was not fresh": (_set_provenance("data_dir_entries_before", "4"), "blocks RHa-RHf"),
    "run 1 was stopped by the memory guard": (_result("run1", lambda d: d["guard"].update(trigger="mem_free")),
                                              "run1: stopped at its duration"),
    "the scene server listened": (
        _result("run2", lambda d: d["captures"][0]["listeners"].append({"port": 18081, "scope": "loopback"})),
        "run2: at each status capture only the status page listened"),
    "a listener on every address": (_result("run1", lambda d: d["captures"][1]["listeners"][0].update(scope="any")),
                                    "run1: at each status capture only the status page listened"),
    "the scene model was started": (lambda r: (r.run / "run1" / "run.jsonl").write_text(
        starting(scene_server={"state": "ready"})), "run1: the detector, live view and snapshots on"),
    "chat was admitted": (lambda r: (r.run / "run2" / "run.jsonl").write_text(
        starting(chat={"state": "available", "reason": "x"})), "run2: the detector, live view and snapshots on"),
    "the live view was stale": (_capture("run2", "live2", lambda d: d["preview"].update(state="stale")),
                                "live2: the live view was on and live"),
    "no picture was encoded": (_capture("run1", "live1", lambda d: d["components"]["preview"]["counters"].update(
        encodes=0)), "live1: the page had been viewed"),
    "chat asked a model": (_capture("run2", "run2", lambda d: d["components"]["chat"]["counters"].update(
        accepted=1, answered=1)), "chat stayed unavailable"),
    "an older schema": (_sql(f"PRAGMA user_version = {SCHEMA_VERSION - 1}"), "the database has the current schema"),
    "the snapshot writer had a problem": (_capture("run1", "run1", lambda d: d["components"]["snapshots"].update(
        problem="disk_full")), "run1: the snapshot writer was running"),
    "the snapshot is of another frame": (_sql("UPDATE incident_snapshots SET frame_seq = frame_seq + 1"),
                                         "its frame is the opening observation's source frame"),
    "the snapshot file changed": (_tamper, "the page's own check"),
    "it did not decode to its size": (lambda r: (r.run / "decoded.json").write_text(json.dumps(
        {sorted((r.data / SNAPSHOT_DIR).iterdir())[0].name: [8, 9]})), "it decodes"),
    "a stray file in the snapshot folder": (lambda r: (r.data / SNAPSHOT_DIR / "notes.txt").write_text("x"),
                                            "private and tidy"),
    "the snapshot record changed after the restart": (lambda r: edit(
        r.run / "snapshots-after-run1.json", lambda d: d[0].update(sha256="0" * 64)), "unchanged across the restart"),
    "another incident was acknowledged": (lambda r: (r.run / "incident-id.txt").write_text("inc-other\n"),
                                          "the acknowledged incident is the first one"),
    "acknowledged while run 2 was running": (_late_ack, "while monitoring was stopped"),
    "the repeat acknowledged again": (lambda r: edit(r.run / "ack-2.json", lambda d: d.update(outcome="acknowledged")),
                                      "a repeat changed nothing"),
    "run 2 showed it open": (_capture("run2", "live2", lambda d: d["incidents"]["recent"][0].update(status="open")),
                             "the status page listed it as open in run 1 and acknowledged in run 2"),
    "a credential in a saved file": (lambda r: (r.run / "run1" / "run.err").write_text(
        "rtsp://admin:hunter2@192.0.2.10:554/stream\n"), "no URL with userinfo"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_item_fails_when_its_evidence_says_otherwise(mod, rehearsal: SimpleNamespace, case: str) -> None:
    change, item = CASES[case]
    change(rehearsal)
    ok, out = lines(mod, rehearsal)
    flagged = [line for line in out if line.split(" ", 1)[1].startswith(item) and not line.startswith("PASS ")]
    assert not ok and flagged, "\n".join(out)
    assert all(line.startswith(("PASS ", "DESCRIPTIVE ")) for line in out if line not in flagged
               and not line.split(" ", 1)[1].startswith(item)) or case in {
        "the snapshot is of another frame", "the snapshot file changed", "an older schema",
        "another incident was acknowledged"}  # these also fail the items that rest on the same evidence


@pytest.mark.parametrize("name", ["ack-2.json", "decoded.json", "incident-id.txt", "run2/status-live2.json",
                                  "run1/run.jsonl", "provenance.txt", "snapshots-after-run1.json"])
def test_missing_evidence_never_passes(mod, rehearsal: SimpleNamespace, name: str) -> None:
    (rehearsal.run / name).unlink()
    ok, out = lines(mod, rehearsal)
    assert not ok and any(line.startswith(("MISSING ", "FAIL ")) for line in out)


def test_without_a_database_nothing_about_it_passes(mod, rehearsal: SimpleNamespace) -> None:
    for path in rehearsal.data.glob("sentinel.db*"):
        path.unlink()
    ok, out = lines(mod, rehearsal)
    assert not ok and "MISSING the run's database is readable: \"missing\"" in out
    assert not any(line.startswith("PASS the first incident has a saved snapshot") for line in out)
