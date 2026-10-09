#!/usr/bin/env python3
"""Review 1 rehearsal (R1): the predeclared reading of its saved evidence, from the files the package's blocks wrote.

    rehearsal_check.py check RUN_DIR --data-dir DATA_DIR [--detail-out PATH]
    rehearsal_check.py first-incident --data-dir DATA_DIR
    rehearsal_check.py snapshot-rows --data-dir DATA_DIR

``first-incident`` prints the ID of the run's first incident (the one the rehearsal acknowledges) and
``snapshot-rows`` the incident snapshot records as JSON (taken between the runs); both read the database only.

RUN_DIR is the rehearsal's evidence folder: ``run1/`` and ``run2/`` as step5_guard wrote them (``result.json``,
``run.jsonl`` and the status captures ``status-live1``/``status-run1`` and ``status-live2``/``status-run2``), and
the blocks' own files: ``provenance.txt``, ``incident-id.txt``, ``snapshots-after-run1.json``, ``ack-1.json``,
``ack-2.json`` and ``decoded.json`` (the device's OpenCV reading of each snapshot file). DATA_DIR is the run's fresh
data directory, read only: its database and ``snapshots/``.

Each item prints PASS, FAIL or MISSING, then ``review1-rehearsal: passed`` or ``not passed``; exit status 0 only when
every item passed. Missing evidence never passes. DESCRIPTIVE lines (device-wide memory, counters) are reported and
never judged: nothing is compared with a run without these features, so they measure no feature's cost. Only
``--detail-out`` is written: the first incident's detail as its page reads it. No secret is read.

What it does not judge: anything only a browser shows. The page's look, its in-place updates, the NOT CURRENT marking
while the tunnel is paused (DV), the download and the chat panel's wording are the operator's observations
(OBSERVATIONS.md), recorded as supplied and never inferred from this reading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import step5_check as s5

from sentinel.incidents.snapshots import FILE_NAME
from sentinel.media.jpeg import jpeg_comments
from sentinel.rules.zones import ZoneObservation
from sentinel.scene.chat import CHAT_ADMISSION
from sentinel.status_page import read_incident, read_snapshot
from sentinel.storage.database import SCHEMA_VERSION, connect_reader

MISSING = s5.MISSING
DATABASE = "sentinel.db"
STATUS_PORT = 18090  # the status page; the scene server's 18081 must not listen in this rehearsal
RUNS = {"run1": ("live1", "run1"), "run2": ("live2", "run2")}
CLEAN = {"status": "duration_stop", "returncode": 0, "trigger": None, "forced": False, "cleanup_clear": True,
         "run_error_label": None}
STARTED = {"detector_problem": None, "live_preview": True, "snapshots": True, "scene": False, "face": False,
           "chat": {"state": "unavailable", "reason": CHAT_ADMISSION.reason}}
# provenance.txt lines the blocks write; every block's exit must be 0 before the reading is made.
PROVENANCE = {"tracked_changes": "0", "data_dir_entries_before": "0", "RHa_exit": "0", "camera_probe_exit": "0",
              "display_manager": "inactive", "v1_procs": "0", "llama_server_procs": "0", "RHd_exit": "0",
              "RHe_exit": "0", "RHf_exit": "0"}
SNAPSHOT_KEYS = ("state", "stream_epoch", "frame_seq", "sha256", "recorded_utc")
GUARD_RESOLUTION = timedelta(seconds=1)  # step5_guard records launch and exit times in whole seconds, truncated
USERINFO = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s@]+@")


def _utc(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo is not None else None


def _starting(path: Path) -> Any:
    """The runtime's own ``starting`` line in run.jsonl (the guard's summary leaves out the live view and chat)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return MISSING
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("run") == "starting":
            startup = item.get("startup")
            return startup if isinstance(startup, dict) else MISSING
    return MISSING


def _provenance(path: Path) -> Any:
    try:
        text = path.read_text()
    except OSError:
        return MISSING
    found: dict[str, str] = {}
    for line in text.splitlines():  # the PSc-style line holds three key=value pairs
        for pair in line.split():
            key, sep, value = pair.partition("=")
            if sep:
                found[key] = value
    return found


def _recent(status: Any) -> dict[str, Any]:
    rows = s5._get(status, "incidents", "recent")
    return {row.get("incident_id"): row.get("status") for row in s5._list(rows) if isinstance(row, dict)}


def _runs(r: s5.Reading, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    results, statuses = {}, {}
    for run, names in RUNS.items():
        result = s5._load(run_dir / run / "result.json")
        results[run] = result
        seen = {"status": s5._get(result, "status"), "returncode": s5._get(result, "child", "returncode"),
                "trigger": s5._get(result, "guard", "trigger"), "forced": s5._get(result, "stop", "forced"),
                "cleanup_clear": s5._get(result, "cleanup_clear"),
                "run_error_label": s5._get(result, "run_error_label")}
        r.item(f"{run}: stopped at its duration, exit 0, no guard or forced stop, cleanup clear", seen, seen == CLEAN)
        captures = s5._get(result, "captures")
        groups = ([c.get("listeners") if isinstance(c, dict) else None for c in captures]
                  if isinstance(captures, list) else MISSING)
        r.item(f"{run}: at each status capture only the status page listened, on loopback (no scene server)", groups,
               isinstance(groups, list) and len(groups) == len(names)
               and all(isinstance(group, list) and group for group in groups)
               and all(isinstance(item, dict) and item.get("scope") == "loopback" and item.get("port") == STATUS_PORT
                       for group in groups for item in group))
        start = _starting(run_dir / run / "run.jsonl")
        started = MISSING if start is MISSING else {
            "detector_problem": start.get("detector_problem", MISSING),
            "live_preview": isinstance(start.get("live_preview"), dict),
            "snapshots": isinstance(start.get("snapshots"), dict),
            "scene": "scene_server" in start or "scene_memory_before" in start,
            "face": "face" in start,
            "chat": start.get("chat", MISSING),
        }
        r.item(f"{run}: the detector, live view and snapshots on; chat unavailable as not admitted; no scene model or "
               "face recognition", started, started == STARTED)
        r.describe(f"{run}: device-wide memory while it ran (no baseline; not a measurement of any feature)",
                   {key: s5._get(result, "guard", key) for key in s5.GUARD_FIGURES})
        for name in names:
            statuses[name] = s5._load(run_dir / run / f"status-{name}.json")
    return results, statuses


def _live_view(r: s5.Reading, statuses: dict[str, Any]) -> None:
    for name in ("live1", "live2"):
        view = s5._get(statuses[name], "preview")
        seen = {key: s5._get(view, key) for key in ("enabled", "state", "picture", "problem")}
        r.item(f"{name}: the live view was on and live, with no problem (status.json never carries a picture)", seen,
               seen == {"enabled": True, "state": "live", "picture": None, "problem": None})
        r.describe(f"{name}: live view counters", s5._get(statuses[name], "components", "preview", "counters"))
    encodes = s5._get(statuses["live1"], "components", "preview", "counters", "encodes")
    r.item("live1: the page had been viewed, and at least one picture encoded, by its capture", encodes,
           isinstance(encodes, int) and encodes >= 1)


def _chat(r: s5.Reading, statuses: dict[str, Any]) -> None:
    seen = {}
    for name, status in statuses.items():
        part = s5._get(status, "components", "chat")
        counters = s5._get(part, "counters")
        seen[name] = {"state": s5._get(part, "state"), "reason": s5._get(part, "reason"),
                      "asked_the_model": (MISSING if not isinstance(counters, dict)
                                          else counters.get("accepted", 0) + counters.get("answered", 0))}
    r.item("chat stayed unavailable, saying it is not admitted, and never asked a model", seen,
           all(v == {"state": "unavailable", "reason": CHAT_ADMISSION.reason, "asked_the_model": 0}
               for v in seen.values()))


SNAPSHOT_COLUMNS = ("incident_id", "observation_id", "state", "reason", "stream_epoch", "frame_seq", "ingest_utc",
                    "source_time_quality", "width", "height", "bytes", "sha256", "recorded_utc")


def first_incident(data: Path) -> str | None:
    """The run's first incident, by creation time; None if there is none. Read only."""
    c = connect_reader(data / DATABASE)
    try:
        row = c.execute("SELECT incident_id FROM incidents ORDER BY created_utc, rowid LIMIT 1").fetchone()
    finally:
        c.close()
    return row[0] if row else None


def snapshot_rows(data: Path) -> list[dict[str, Any]]:
    """Every incident snapshot record, oldest first. Read only."""
    c = connect_reader(data / DATABASE)
    try:
        rows = c.execute(f"SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM incident_snapshots ORDER BY rowid").fetchall()
    finally:
        c.close()
    return [dict(zip(SNAPSHOT_COLUMNS, row)) for row in rows]


def _database(r: s5.Reading, data: Path) -> tuple[Any, Any, Any, int]:
    """The first incident, its snapshot row and opening observation; and the number of saved snapshot rows."""
    try:
        c = connect_reader(data / DATABASE)
    except Exception:  # noqa: BLE001 - a missing or unreadable database is missing evidence
        r.item("the run's database is readable", MISSING, False)
        return MISSING, None, None, 0
    try:
        version = c.execute("PRAGMA user_version").fetchone()[0]
        r.item("the database has the current schema (created fresh by run 1)", version, version == SCHEMA_VERSION)
        first = c.execute("SELECT incident_id FROM incidents ORDER BY created_utc, rowid LIMIT 1").fetchone()
        incident = first[0] if first else MISSING  # as first_incident()
        names = [d[0] for d in c.execute("SELECT * FROM incident_snapshots LIMIT 0").description]
        found = c.execute("SELECT * FROM incident_snapshots WHERE incident_id = ?", (incident,)).fetchone()
        row = dict(zip(names, found)) if found else None
        entered = c.execute("SELECT payload FROM incident_evidence WHERE incident_id = ? AND phase = 'entered' "
                            "ORDER BY evidence_seq LIMIT 1", (incident,)).fetchone()
        saved = c.execute("SELECT count(*) FROM incident_snapshots WHERE state = 'saved'").fetchone()[0]
        r.describe("incidents, and snapshot rows by state", {
            "incidents": c.execute("SELECT count(*) FROM incidents").fetchone()[0],
            **{state: n for state, n in c.execute("SELECT state, count(*) FROM incident_snapshots GROUP BY state")}})
    finally:
        c.close()
    try:
        observation = ZoneObservation.model_validate_json(entered[0]) if entered else None
    except ValueError:
        observation = None
    return incident, row, observation, saved


def _snapshot(r: s5.Reading, run_dir: Path, data: Path, statuses: dict[str, Any], incident: Any, row: Any,
              observation: Any, saved: int) -> None:
    writer = s5._get(statuses["run1"], "components", "snapshots")
    store_saved = s5._get(writer, "store", "saved")
    r.item("run1: the snapshot writer was running, with no problem, and had saved a snapshot",
           {"state": s5._get(writer, "state"), "problem": s5._get(writer, "problem"), "saved": store_saved},
           s5._get(writer, "state") == "running" and s5._get(writer, "problem") is None
           and isinstance(store_saved, int) and store_saved >= 1)
    r.item("the first incident has a saved snapshot", {
        "incident_id": incident, "state": row["state"] if row else MISSING, "reason": row["reason"] if row else None},
           bool(row) and row["state"] == "saved")
    source = observation.last_source if observation is not None else None
    same = bool(row and source and (row["camera_id"], row["boot_id"], row["run_id"], row["stream_epoch"],
                                    row["frame_seq"]) == (source.camera_id, source.boot_id, source.run_id,
                                                          source.stream_epoch, source.frame_seq))
    r.item("its frame is the opening observation's source frame (camera, boot, run, stream epoch, frame)",
           {"identity": "same" if same else "different" if row and source else MISSING,
            "stream_epoch": row["stream_epoch"] if row else None, "frame_seq": row["frame_seq"] if row else None},
           same)
    when = _utc(row["ingest_utc"]) if row else None
    r.item("its time is that frame's ingest time, with its source time quality",
           {"ingest_utc": row["ingest_utc"] if row else MISSING,
            "source_time_quality": row["source_time_quality"] if row else MISSING},
           bool(row and observation and when == observation.observed_utc
                and row["source_time_quality"] in ("none", "stream_relative")))
    state, blob = read_snapshot(data / DATABASE, incident) if isinstance(incident, str) else ("none", None)
    r.item("the page's own check: the file matches its record (derived name, size, SHA-256)", state, state == "ok")
    comments = jpeg_comments(blob or b"")
    r.item("its JPEG comment names the incident and the frame", comments[:1], bool(
        row and comments and comments[0].startswith(f"Sentinel incident {incident}; camera {row['camera_id']}; "
                                                    f"stream epoch {row['stream_epoch']} frame {row['frame_seq']};")))
    decoded = s5._load(run_dir / "decoded.json")
    size = decoded.get(row["file_name"], MISSING) if isinstance(decoded, dict) and row else MISSING
    r.item("it decodes (the device's OpenCV) to the recorded size", size,
           bool(row) and size == [row["height"], row["width"]])
    folder = data / "snapshots"
    try:
        names = sorted(os.listdir(folder))
        modes: Any = {"folder": oct(stat.S_IMODE(folder.stat().st_mode)),
                      "files": sorted({oct(stat.S_IMODE((folder / n).stat().st_mode)) for n in names})}
    except OSError:
        names, modes = [], MISSING
    r.item("private and tidy: folder 0700, files 0600, one snapshot file per saved row, nothing else",
           modes if modes is MISSING else {**modes, "file_count": len(names), "saved_rows": saved},
           modes == {"folder": "0o700", "files": ["0o600"]} and all(FILE_NAME.fullmatch(n) for n in names)
           and len(names) == saved)
    before = s5._load(run_dir / "snapshots-after-run1.json")
    kept = next((item for item in s5._list(before) if isinstance(item, dict) and item.get("incident_id") == incident),
                None)
    r.item("unchanged across the restart (against the record taken between the runs)",
           {key: kept.get(key) for key in SNAPSHOT_KEYS} if kept else MISSING,
           bool(kept and row) and all(kept.get(key) == row[key] for key in SNAPSHOT_KEYS))


def _acknowledgment(r: s5.Reading, run_dir: Path, results: dict[str, Any], statuses: dict[str, Any],
                    incident: Any) -> dict[str, Any]:
    try:
        named = (run_dir / "incident-id.txt").read_text().strip()
    except OSError:
        named = MISSING
    r.item("the acknowledged incident is the first one, the one with the snapshot", named,
           isinstance(incident, str) and named == incident)
    first, repeat = s5._load(run_dir / "ack-1.json"), s5._load(run_dir / "ack-2.json")
    r.item("acknowledged with sentinel incident ack",
           {key: s5._get(first, key) for key in ("outcome", "incident_id", "acknowledged_utc", "actor")},
           s5._get(first, "outcome") == "acknowledged" and s5._get(first, "incident_id") == incident
           and _utc(s5._get(first, "acknowledged_utc")) is not None)
    stopped = _utc(s5._get(results["run1"], "child", "exited_utc"))
    restarted = _utc(s5._get(results["run2"], "child", "launched_utc"))
    at = _utc(s5._get(first, "acknowledged_utc"))
    # The acknowledgment succeeded only because no runtime held the database's writer lock (D68); its time is then
    # checked against the guard's records, which are whole seconds: no earlier than run 1's exit second, and before
    # the end of run 2's launch second.
    r.item("while monitoring was stopped: between run 1's exit and run 2's launch (to the guard's whole seconds)", {
        "run1_exited_utc": s5._get(results["run1"], "child", "exited_utc"),
        "acknowledged_utc": s5._get(first, "acknowledged_utc"),
        "run2_launched_utc": s5._get(results["run2"], "child", "launched_utc")},
           None not in (stopped, at, restarted)
           and stopped <= at < restarted + GUARD_RESOLUTION)  # type: ignore[operator]
    r.item("a repeat changed nothing and reported the first acknowledgment",
           {key: s5._get(repeat, key) for key in ("outcome", "acknowledged_utc")},
           s5._get(repeat, "outcome") == "already_acknowledged"
           and s5._get(repeat, "acknowledged_utc") == s5._get(first, "acknowledged_utc"))
    listed = {name: _recent(statuses[name]).get(incident, MISSING) if statuses[name] is not MISSING else MISSING
              for name in ("run1", "live2", "run2")}
    r.item("the status page listed it as open in run 1 and acknowledged in run 2", listed,
           listed == {"run1": "open", "live2": "acknowledged", "run2": "acknowledged"})
    return first if isinstance(first, dict) else {}


def _detail(r: s5.Reading, data: Path, incident: Any, first: dict[str, Any], detail_out: Path | None) -> None:
    detail = read_incident(data / DATABASE, incident) if isinstance(incident, str) else {"state": "no_incident"}
    if detail_out is not None:
        detail_out.write_text(json.dumps(detail, indent=2))
    seen = detail.get("acknowledgment") or {}
    r.item("after run 2 the stored incident is acknowledged, with the first acknowledgment's time and actor",
           {key: seen.get(key) for key in ("state", "at_utc", "by")},
           detail.get("state") == "available" and seen.get("state") == "acknowledged"
           and seen.get("at_utc") == first.get("acknowledged_utc") and seen.get("by") == first.get("actor"))
    moves = [(t.get("from_status"), t.get("to_status"))
             for t in s5._list((detail.get("transitions") or {}).get("shown")) if isinstance(t, dict)]
    r.item("its lifecycle: opened, acknowledged once, still unresolved", moves,
           moves.count(("open", "acknowledged")) == 1
           and (detail.get("incident") or {}).get("status") == "acknowledged")
    r.item("the incident detail reports the snapshot file verified", (detail.get("snapshot") or {}).get("file"),
           (detail.get("snapshot") or {}).get("file") == "verified")


def read(run_dir: Path, data: Path, *, detail_out: Path | None = None) -> s5.Reading:
    r = s5.Reading()
    found = _provenance(run_dir / "provenance.txt")
    seen = MISSING if found is MISSING else {key: found.get(key, MISSING) for key in PROVENANCE}
    r.item("blocks RHa-RHf: pinned code, a fresh data directory, the display manager stopped, no v1 or llama-server "
           "process, every exit 0", seen, seen == PROVENANCE)
    results, statuses = _runs(r, run_dir)
    _live_view(r, statuses)
    _chat(r, statuses)
    incident, row, observation, saved = _database(r, data)
    _snapshot(r, run_dir, data, statuses, incident, row, observation, saved)
    first = _acknowledgment(r, run_dir, results, statuses, incident)
    _detail(r, data, incident, first, detail_out)
    counts = {str(p.relative_to(run_dir)): len(USERINFO.findall(p.read_text(errors="replace")))
              for p in sorted(run_dir.rglob("*")) if p.is_file()}
    r.item("no URL with userinfo in any saved file", sum(counts.values()), all(v == 0 for v in counts.values()))
    return r


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="read the saved evidence")
    check.add_argument("run_dir", type=Path)
    check.add_argument("--data-dir", type=Path, required=True)
    check.add_argument("--detail-out", type=Path)
    for name in ("first-incident", "snapshot-rows"):
        commands.add_parser(name).add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "first-incident":
        incident = first_incident(args.data_dir)
        if incident is None:
            print("no incident in this database", file=sys.stderr)
            return 1
        print(incident)
        return 0
    if args.command == "snapshot-rows":
        print(json.dumps(snapshot_rows(args.data_dir), indent=2))
        return 0
    reading = read(args.run_dir, args.data_dir, detail_out=args.detail_out)
    for line in reading.lines:
        print(line)
    print(f"review1-rehearsal: {'passed' if reading.ok else 'not passed'}")
    return 0 if reading.ok else 1


if __name__ == "__main__":
    sys.exit(main())
