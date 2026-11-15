"""D70: incident snapshots (the store, its writer thread, retention, and the incident page's view and download).

Real SQLite, real files in a temporary directory, real FrameRefs on a FakeClock; images are labelled stand-ins and
the encoder is a fake whose JPEG names the image it was given. No camera, model or cv2.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import sqlite3
import stat
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentinel.config import IncidentsConfig, NotificationsConfig
from sentinel.contracts import PixelFormat
from sentinel.incidents.service import IncidentService
from sentinel.incidents.signals import IncidentSignal, SignalPhase
from sentinel.incidents.snapshots import (
    SNAPSHOT_DIR,
    SnapshotDirectoryError,
    SnapshotJob,
    SnapshotPolicy,
    SnapshotStore,
    SnapshotWriter,
)
from sentinel.media.clock import NS_PER_SECOND, FakeClock
from sentinel.media.frames import FrameStamper
from sentinel.media.jpeg import SOI, EncodedJpeg, EncodeError, jpeg_comments
from sentinel.rules.scene_hazard import Severity
from sentinel.status_page import StatusServer, read_incident, read_snapshot, render_incident_html
from sentinel.storage.database import Database

T0 = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
APP0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
EOI = b"\xff\xd9"
ROOMY = SimpleNamespace(total=100_000_000_000, free=50_000_000_000)


class Image:
    def __init__(self, label: str) -> None:
        self.label = label
        self.shape = (480, 640, 3)


def fake_encode(image: Image, pixel_format: PixelFormat, *, max_width: int, max_height: int,
                quality: int) -> EncodedJpeg:
    return EncodedJpeg(SOI + APP0 + image.label.encode() + EOI, min(640, max_width), min(480, max_height))


def sized_encode(size: int):
    def encode(image: Image, pixel_format: PixelFormat, **kwargs: Any) -> EncodedJpeg:
        body = image.label.encode().ljust(size - len(SOI + APP0 + EOI) - 150, b"\0")  # ~150 B of comment is added
        return EncodedJpeg(SOI + APP0 + body + EOI, 640, 480)
    return encode


def signal(zone: str) -> IncidentSignal:
    return IncidentSignal(
        observation_id=f"zone-{zone}.entered", episode_id=f"zone-{zone}", camera_id="cam-1",
        kind="zone.restricted_entry", phase=SignalPhase.ENTERED, zone_id=zone, rule_revision="rev1",
        severity=Severity.WARNING, title=f"Person in restricted zone {zone!r}", observed_utc=T0, boot_id="boot-1",
        observed_mono_ns=NS_PER_SECOND, reason="test", payload="{}",
    )


class Rig:
    """A database with incidents, a snapshot store on a FakeClock, and real frames with labelled images."""

    def __init__(self, tmp_path: Path, *, policy: SnapshotPolicy | None = None, encode=fake_encode,
                 usage: Any = ROOMY) -> None:
        self.tmp_path = tmp_path
        self.clock = FakeClock(utc=T0)
        self.db = Database.open(tmp_path / "sentinel.db")
        self.incidents = IncidentService(self.db, self.clock, incidents=IncidentsConfig(),
                                         notifications=NotificationsConfig())
        self.usage = usage
        self.dir = tmp_path / SNAPSHOT_DIR
        self.store = SnapshotStore(self.db, self.dir, self.clock, policy, encode=encode,
                                   disk_usage=lambda path: self.usage)
        self.stamper = FrameStamper("cam-1", self.clock)
        self.stamper.connect()
        self.zones = 0

    def incident(self) -> str:
        self.zones += 1
        return self.incidents.record(signal(f"z{self.zones}")).incident_id  # type: ignore[return-value]

    def frame(self, dt: float = 1 / 15):
        self.clock.advance(dt)
        frame = self.stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
        return frame, Image(f"e{frame.stream_epoch}-f{frame.frame_seq}")

    def job(self, *, image: bool = True, **changes: Any) -> SnapshotJob:
        frame, picture = self.frame()
        fields = {"incident_id": self.incident(), "observation_id": f"zone-z{self.zones}.entered", "key": frame.key,
                  "frame": frame, "image": picture if image else None, "missing": None if image else "frame_not_retained"}
        return SnapshotJob(**(fields | changes))

    def rows(self) -> list[dict[str, Any]]:
        cursor = self.db.connection.execute("SELECT * FROM incident_snapshots ORDER BY rowid")
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def files(self) -> list[str]:
        return sorted(os.listdir(self.dir))


@pytest.fixture
def rig(tmp_path: Path):
    rig = Rig(tmp_path)
    rig.store.open()
    yield rig
    rig.store.close()
    rig.db.close()


# ---------------------------------------------------------------- the store


def test_a_saved_snapshot_is_its_own_frame_with_identity_time_and_checksum(rig: Rig) -> None:
    job = rig.job()
    rig.frame()  # a newer frame exists; the job names its own
    assert rig.store.save(job) == "saved"
    (row,) = rig.rows()
    frame = job.frame
    assert row | {"recorded_utc": None} == {
        "incident_id": job.incident_id, "observation_id": job.observation_id, "state": "saved", "reason": None,
        "camera_id": "cam-1", "boot_id": frame.boot_id, "run_id": frame.run_id, "stream_epoch": 1, "frame_seq": 0,
        "ingest_utc": "2026-10-09T12:00:00.066667Z", "source_time_quality": "none", "source_pts": None,
        "width": 640, "height": 480, "file_name": f"snap-{job.incident_id}.jpg", "bytes": row["bytes"],
        "sha256": row["sha256"], "recorded_utc": None, "pruned_utc": None}
    data = (rig.dir / row["file_name"]).read_bytes()
    assert b"e1-f0" in data and b"e1-f1" not in data
    assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
    (comment,) = jpeg_comments(data)
    assert comment.startswith(f"Sentinel incident {job.incident_id}; camera cam-1; stream epoch 1 frame 0; ingest "
                              "2026-10-09T12:00:00.066667Z (when the device received the frame; source time quality none)")
    assert stat.S_IMODE(os.stat(rig.dir / row["file_name"]).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(rig.dir).st_mode) == 0o700
    assert rig.files() == [row["file_name"]]  # no temporary file is left


def test_a_job_without_its_image_is_recorded_missing_and_nothing_is_written(rig: Rig) -> None:
    job = rig.job(image=False)
    assert rig.store.save(job) == "missing"
    (row,) = rig.rows()
    assert (row["state"], row["reason"], row["frame_seq"], row["file_name"], row["sha256"]) == (
        "missing", "frame_not_retained", job.key.frame_seq, None, None)
    assert row["ingest_utc"] == "2026-10-09T12:00:00.066667Z"  # the frame's record was still known
    assert rig.files() == []
    unknown_record = rig.job(image=False, frame=None)
    assert rig.store.save(unknown_record) == "missing" and rig.rows()[1]["ingest_utc"] is None


def test_another_frame_is_never_saved_in_its_place(rig: Rig) -> None:
    job = rig.job()
    other, _ = rig.frame()
    assert rig.store.save(SnapshotJob(job.incident_id, job.observation_id, job.key, other, job.image)) == "missing"
    assert rig.rows()[0]["reason"] == "frame_identity_mismatch" and rig.files() == []


def test_a_repeat_is_a_duplicate_and_changes_nothing(rig: Rig) -> None:
    job = rig.job()
    assert rig.store.save(job) == "saved"
    before = (rig.rows(), rig.files())
    assert rig.store.save(job) == "duplicate" and rig.store.save(SnapshotJob(
        job.incident_id, "other", job.key, None, None, "frame_not_retained")) == "duplicate"
    assert (rig.rows(), rig.files()) == before


@pytest.mark.parametrize(("encode", "reason"), [
    (lambda image, fmt, **kw: (_ for _ in ()).throw(EncodeError("unsupported_pixel_format")), "unsupported_pixel_format"),
    (lambda image, fmt, **kw: (_ for _ in ()).throw(RuntimeError("rtsp://admin:hunter2@192.0.2.10")),
     "encode_failed:RuntimeError"),
    (lambda image, fmt, **kw: EncodedJpeg(b"GIF89a", 1, 1), "not_a_jpeg"),
    (sized_encode(2_000_200), "too_large"),
])
def test_an_encoding_failure_is_recorded_by_label_and_leaves_no_file(tmp_path: Path, encode, reason: str) -> None:
    rig = Rig(tmp_path, encode=encode)
    rig.store.open()
    assert rig.store.save(rig.job()) == "missing"
    assert rig.rows()[0]["reason"] == reason and rig.files() == []
    assert "hunter2" not in repr(rig.rows()) + repr(rig.store.status())


def test_nothing_is_saved_below_the_free_space_reserve(tmp_path: Path) -> None:
    rig = Rig(tmp_path, usage=SimpleNamespace(total=100_000_000_000, free=10_000_000_100))  # reserve: 10 GB (10 %)
    rig.store.open()
    assert rig.store.save(rig.job()) == "missing"
    assert rig.rows()[0]["reason"] == "free_space_reserve" and rig.files() == []
    assert rig.store.status()["problem"] == "free_space_reserve"
    rig.usage = ROOMY
    assert rig.store.save(rig.job()) == "saved" and rig.store.status()["problem"] is None
    small = Rig(tmp_path / "small", usage=SimpleNamespace(total=5_000_000_000, free=1_000_000_100))
    small.store.open()  # 10 % of 5 GB is below 1 GB: the 1 GB floor applies
    assert small.store.save(small.job()) == "missing" and small.rows()[0]["reason"] == "free_space_reserve"


def test_a_write_failure_leaves_no_file_and_no_saved_row(rig: Rig, monkeypatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "replace", refuse)
    assert rig.store.save(rig.job()) == "missing"
    assert rig.rows()[0]["reason"] == "write_failed:PermissionError" and rig.files() == []
    assert rig.store.status()["problem"] == "write_failed:PermissionError"


def test_retention_deletes_by_age_and_oldest_first_for_the_quota_and_count(tmp_path: Path) -> None:
    policy = SnapshotPolicy(quota_bytes=3_000, max_count=3, max_bytes=1_000)
    rig = Rig(tmp_path, policy=policy, encode=sized_encode(900))
    rig.store.open()
    jobs = [rig.job() for _ in range(5)]
    for job in jobs:
        rig.clock.advance(1.0)
        assert rig.store.save(job) == "saved"
    rows = rig.rows()
    assert [r["state"] for r in rows] == ["pruned", "pruned", "saved", "saved", "saved"]
    assert {r["reason"] for r in rows[:2]} == {"retention_quota"} and all(r["pruned_utc"] for r in rows[:2])
    assert rig.files() == sorted(r["file_name"] for r in rows[2:])
    assert sum(r["bytes"] for r in rows[2:]) <= 3_000
    rig.clock.advance(7 * 86400.0 - 0.5)  # saved 2 s, 1 s and 0 s ago: the first two kept are now over 7 days old
    assert rig.store.prune() == 2
    rows = rig.rows()
    assert [r["state"] for r in rows] == ["pruned", "pruned", "pruned", "pruned", "saved"]
    assert [r["reason"] for r in rows[2:4]] == ["retention_age", "retention_age"]
    assert rig.files() == [rows[4]["file_name"]]
    assert rows[0]["frame_seq"] == jobs[0].key.frame_seq  # the record of a pruned snapshot is kept


def test_retention_never_deletes_outside_the_directory(rig: Rig) -> None:
    job = rig.job()
    assert rig.store.save(job) == "saved"
    victim = rig.tmp_path / "keep.txt"
    victim.write_text("not a snapshot")
    rig.db.connection.execute("UPDATE incident_snapshots SET file_name = '../keep.txt'")  # a tampered row
    rig.clock.advance(8 * 86400.0)
    assert rig.store.prune() == 1
    assert victim.read_text() == "not a snapshot" and rig.rows()[0]["state"] == "pruned"


def test_startup_removes_interrupted_writes_and_unrecorded_files_only(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.store.open()
    job = rig.job()
    assert rig.store.save(job) == "saved"
    rig.store.close()
    (rig.dir / ".snap-inc-x.0123abcd.tmp").write_bytes(b"half")
    (rig.dir / "snap-inc-orphan.jpg").write_bytes(b"renamed, never recorded")
    (rig.dir / "notes.txt").write_text("not ours")
    again = SnapshotStore(rig.db, rig.dir, rig.clock, encode=fake_encode, disk_usage=lambda path: ROOMY)
    assert again.open() == {"removed_temporary": 1, "removed_unrecorded": 1, "unexpected_files": 1, "pruned": 0}
    assert rig.files() == sorted(["notes.txt", f"snap-{job.incident_id}.jpg"])
    again.close()


def test_the_directory_must_be_a_private_directory(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rig.dir.symlink_to(elsewhere)
    with pytest.raises(SnapshotDirectoryError, match="not_a_directory"):
        rig.store.open()
    rig.dir.unlink()
    rig.dir.write_text("a file")
    with pytest.raises(SnapshotDirectoryError, match="not_a_directory"):
        rig.store.open()
    rig.dir.unlink()
    rig.dir.mkdir(mode=0o755)
    os.chmod(rig.dir, 0o755)
    rig.store.open()
    assert stat.S_IMODE(os.stat(rig.dir).st_mode) == 0o700
    rig.store.close()


def test_the_policy_is_validated() -> None:
    assert SnapshotPolicy().describe()["retention_s"] == 7 * 86400.0
    for bad in ({"retention_s": 0}, {"quota_bytes": 0}, {"max_bytes": 10, "quota_bytes": 5},
                {"reserve_fraction": 0.9}, {"jpeg_quality": 101}):
        with pytest.raises(ValueError):
            SnapshotPolicy(**bad)


# ---------------------------------------------------------------- the writer


def test_the_writer_never_blocks_the_caller_and_bounds_the_images_it_holds(rig: Rig) -> None:
    entered, release = threading.Event(), threading.Event()

    def slow(image: Image, fmt: PixelFormat, **kwargs: Any) -> EncodedJpeg:
        entered.set()
        release.wait(5)
        return fake_encode(image, fmt, **kwargs)

    rig.store._encode = slow  # type: ignore[attr-defined]
    writer = SnapshotWriter(rig.store, rig.clock)
    writer.start()
    jobs = [rig.job() for _ in range(6)]
    started = time.perf_counter()
    assert writer.submit(jobs[0]) and entered.wait(5)
    assert all(writer.submit(job) for job in jobs[1:])
    assert time.perf_counter() - started < 0.5  # queued without waiting for the slow save
    assert writer.status()["queued"] == 5
    release.set()
    assert writer.stop(5.0)
    states = [(r["state"], r["reason"]) for r in rig.rows()]
    assert states == [("saved", None)] * 4 + [("missing", "queue_full")] * 2  # 4 images held at most
    assert writer.status()["writer"]["queued"] == 6 and writer.status()["state"] == "stopped"


def test_jobs_of_one_frame_count_once_toward_the_bound() -> None:
    clock = FakeClock()
    store = SimpleNamespace(policy=SnapshotPolicy(), saved=[], close=lambda: None)
    hold = threading.Event()
    store.save = lambda job: hold.wait(5) and store.saved.append(job) or "saved"
    writer = SnapshotWriter(store, clock, max_held_frames=1)  # type: ignore[arg-type]
    writer.start()
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    frame = stamper.stamp(native_width=640, native_height=480, pixel_format=PixelFormat.BGR)
    image = Image("shared")
    for n in range(3):
        assert writer.submit(SnapshotJob(f"inc-{n}", f"obs-{n}", frame.key, frame, image))
    hold.set()
    assert writer.stop(5.0)
    assert [job.image for job in store.saved] == [image, image, image]  # the same frame: no job lost its image


def test_stop_saves_what_is_queued_then_refuses_more(rig: Rig) -> None:
    writer = SnapshotWriter(rig.store, rig.clock)
    assert not writer.submit(rig.job())  # not started
    writer.start()
    jobs = [rig.job() for _ in range(3)]
    for job in jobs:
        assert writer.submit(job)
    assert writer.stop(5.0)
    assert [r["state"] for r in rig.rows()] == ["saved"] * 3
    assert not writer.submit(rig.job()) and writer.status()["writer"]["not_queued_stopped"] == 2


def test_an_error_does_not_end_the_writer(rig: Rig) -> None:
    writer = SnapshotWriter(rig.store, rig.clock)
    real = rig.store.save
    calls = {"n": 0}

    def flaky(job: SnapshotJob) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return real(job)

    rig.store.save = flaky  # type: ignore[method-assign]
    writer.start()
    writer.submit(rig.job())
    second = rig.job()
    writer.submit(second)
    deadline = time.monotonic() + 5
    while not rig.rows() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.stop(5.0)
    assert [r["incident_id"] for r in rig.rows()] == [second.incident_id]
    assert writer.status()["writer"]["errors"] == 1


# ---------------------------------------------------------------- the incident page


def page_get(server: StatusServer, path: str, *, host: str | None = None, method: str = "GET"):
    address, port = server.address
    conn = http.client.HTTPConnection(address, port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{port}")
        conn.endheaders()
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.fixture
def served(rig: Rig):
    server = StatusServer(0, lambda: {"runtime": {"state": "running"}}, rig.tmp_path / "sentinel.db")
    server.start()
    yield server
    assert server.stop(2.0)


def test_the_incident_page_shows_and_offers_the_saved_snapshot(rig: Rig, served: StatusServer) -> None:
    job = rig.job()
    rig.store.save(job)
    row = rig.rows()[0]
    data = (rig.dir / row["file_name"]).read_bytes()
    code, headers, body = page_get(served, f"/incidents/{job.incident_id}")
    page = body.decode()
    assert code == 200 and "img-src 'self'" in headers["Content-Security-Policy"]
    assert f'<img src="/incidents/{job.incident_id}/snapshot.jpg" width="640" height="480"' in page
    assert f'<a class=button href="/incidents/{job.incident_id}/snapshot/download" download>' in page
    assert "stream epoch 1, frame 0" in page and "2026-10-09T12:00:00.066667Z" in page
    assert "not when the camera captured it" in page and row["sha256"] in page
    assert "never a later one" in page and "kept for 7 days" in page
    code, headers, body = page_get(served, f"/incidents/{job.incident_id}/snapshot.jpg")
    assert (code, headers["Content-Type"], body) == (200, "image/jpeg", data)
    assert headers["Content-Disposition"] == f'inline; filename="sentinel-{job.incident_id}.jpg"'
    assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"
    code, headers, body = page_get(served, f"/incidents/{job.incident_id}/snapshot/download")
    assert (code, body) == (200, data)
    assert headers["Content-Disposition"] == f'attachment; filename="sentinel-{job.incident_id}.jpg"'
    detail = read_incident(rig.tmp_path / "sentinel.db", job.incident_id)["snapshot"]
    assert detail["file"] == "verified" and "_file" not in detail and detail["frame"] == {"stream_epoch": 1,
                                                                                           "frame_seq": 0}


def test_missing_pruned_and_none_are_stated_and_never_substituted(rig: Rig, served: StatusServer) -> None:
    missing, saved = rig.job(image=False), rig.job()
    rig.store.save(missing)
    rig.store.save(saved)
    none = rig.incident()
    page = page_get(served, f"/incidents/{missing.incident_id}")[2].decode()
    assert ("missing: the frame it came from was no longer held when the incident was recorded; no other frame was "
            "substituted.") in page and "<img" not in page
    assert page_get(served, f"/incidents/{missing.incident_id}/snapshot.jpg")[:3:2] == (404, b"no snapshot\n")
    page = page_get(served, f"/incidents/{none}")[2].decode()
    assert "none recorded for this incident: snapshots are saved only by sentinel run --snapshots" in page
    assert page_get(served, f"/incidents/{none}/snapshot/download")[0] == 404
    rig.clock.advance(8 * 86400.0)
    rig.store.prune()
    page = page_get(served, f"/incidents/{saved.incident_id}")[2].decode()
    assert "deleted by retention (older than the retention period) at 2026-10-17T12:00:" in page and "<img" not in page
    assert page_get(served, f"/incidents/{saved.incident_id}/snapshot.jpg")[0] == 404


def test_a_file_that_does_not_match_its_record_is_never_served(rig: Rig, served: StatusServer) -> None:
    job = rig.job()
    rig.store.save(job)
    path = rig.dir / rig.rows()[0]["file_name"]
    original = path.read_bytes()
    url = f"/incidents/{job.incident_id}/snapshot.jpg"
    refused = (409, b"snapshot file missing or not matching its record\n")
    path.write_bytes(original[:-1] + b"\0")  # same size, other bytes
    assert page_get(served, url)[:3:2] == refused
    assert "Saved, but its file is missing or does not match its record" in page_get(
        served, f"/incidents/{job.incident_id}")[2].decode()
    secret = rig.tmp_path / "secret.jpg"
    secret.write_bytes(original)
    path.unlink()
    path.symlink_to(secret)  # the right bytes behind a link: still refused
    assert page_get(served, url)[:3:2] == refused
    path.unlink()
    os.mkfifo(path)  # a FIFO is refused at once, not waited on
    started = time.monotonic()
    assert page_get(served, url)[:3:2] == refused and time.monotonic() - started < 2
    path.unlink()
    assert page_get(served, url)[:3:2] == refused
    path.write_bytes(original)
    assert page_get(served, url)[0] == 200


def test_a_tampered_row_cannot_point_outside_its_own_file(rig: Rig, served: StatusServer) -> None:
    job, other = rig.job(), rig.job()
    rig.store.save(job)
    rig.store.save(other)
    theirs = (rig.dir / f"snap-{other.incident_id}.jpg").read_bytes()
    outside = rig.tmp_path / "outside.jpg"
    outside.write_bytes(theirs)
    # Each name below holds exactly the bytes the tampered row records: only the name rule refuses them.
    for name in (f"snap-{other.incident_id}.jpg", "../outside.jpg", str(outside)):
        rig.db.connection.execute(
            "UPDATE incident_snapshots SET file_name = ?, bytes = ?, sha256 = ? WHERE incident_id = ?",
            (name, len(theirs), hashlib.sha256(theirs).hexdigest(), job.incident_id))
        assert read_snapshot(rig.tmp_path / "sentinel.db", job.incident_id) == ("not_verified", None)
        assert page_get(served, f"/incidents/{job.incident_id}/snapshot.jpg")[0] == 409
    assert read_snapshot(rig.tmp_path / "sentinel.db", other.incident_id) == ("ok", theirs)


@pytest.mark.parametrize("path", [
    "/incidents/../snapshot.jpg", "/incidents/inc-1/snapshot.png", "/incidents/inc-1/snapshot.jpg/x",
    "/incidents/%2e%2e/snapshot.jpg", "/incidents/inc.1/snapshot.jpg", "/incidents/inc-1/snapshot",
    "/incidents/inc-1/snapshot/download/", "/incidents/inc-1/snapshot.jpg.json", "/snapshots/snap-inc-1.jpg",
])
def test_snapshot_paths_are_validated_and_never_echoed(served: StatusServer, path: str) -> None:
    assert page_get(served, path)[:3:2] == (404, b"not found\n")


def test_snapshot_requests_need_get_and_the_loopback_host(rig: Rig, served: StatusServer) -> None:
    job = rig.job()
    rig.store.save(job)
    url = f"/incidents/{job.incident_id}/snapshot.jpg"
    assert page_get(served, url, method="POST")[0] == 405
    assert page_get(served, url, method="DELETE")[0] == 405
    assert page_get(served, url, host="evil.example")[0] == 421
    assert served.address[0] == "127.0.0.1"


def test_a_database_from_before_snapshots_reads_none(tmp_path: Path, monkeypatch) -> None:
    from sentinel.storage import database

    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:2])
        patch.setattr(database, "SCHEMA_VERSION", 2)
        db = Database.open(tmp_path / "sentinel.db")
        service = IncidentService(db, FakeClock(utc=T0), incidents=IncidentsConfig(),
                                  notifications=NotificationsConfig())
        incident_id = service.record(signal("old")).incident_id
        db.close()
    detail = read_incident(tmp_path / "sentinel.db", incident_id)  # the page never migrates
    assert detail["snapshot"] == {"state": "none", "text": detail["snapshot"]["text"]}
    assert "none recorded for this incident" in render_incident_html({**detail, "generated_utc": "x"})
    assert read_snapshot(tmp_path / "sentinel.db", incident_id) == ("none", None)
    db = Database.open(tmp_path / "sentinel.db")  # the next writer migrates it to schema 3
    assert db.connection.execute("PRAGMA user_version").fetchone()[0] == 3
    db.close()
