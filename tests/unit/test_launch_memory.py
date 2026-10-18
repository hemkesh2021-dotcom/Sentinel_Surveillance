"""Memory around the operator's page-cache drop and before a guarded launch (D103, session 87;
benchmarks/runner/launch_memory.py), against a verbatim copy of the ten-minute diagnostic trk-20261017T103831Z's sealed
tv/result.json (SHA-256 57578d19e802aff212749d70382e74db5ca5d75307b860b4cbcda5b9e94e9e67; every text in it a fixed label
or a UTC time), the run the launch floor is derived from: the derivation and its refusals, the floor's boundary, the
preparation's evidence (a failed sync or drop, a missing or invalid record, another boot, records out of order),
memory changing between the preparation and the launch, the records never overwritten, and the report."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
REFERENCE = Path(__file__).resolve().parents[1] / "support/fixtures/trk-20261017T103831Z-tv-result.json"
BOOT = "0f1e2d3c-4b5a-4968-8776-a5b4c3d2e1f0"  # synthetic boot IDs
OTHER_BOOT = "1a2b3c4d-5e6f-4a0b-9c8d-7e6f5a4b3c2d"
FLOOR = 5_569_835_008
GIB = 1 << 30


@pytest.fixture
def lm(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("launch_memory")


@pytest.fixture
def reference() -> dict:
    return json.loads(REFERENCE.read_text())


class Host:
    """A stand-in for /proc: meminfo in bytes (as the guard's parser returns it), the boot ID and both clocks."""

    def __init__(self, memfree: int = 6_900_000_000, cached: int = 300_000_000, boot: str = BOOT) -> None:
        self.values = {"MemTotal": 7_990_001_664, "MemFree": memfree, "MemAvailable": 7_090_000_000,
                       "Cached": cached, "Buffers": 40_000_000, "AnonPages": 230_000_000}
        self.boot = boot
        self.t = 600.0

    def meminfo(self) -> dict[str, int]:
        return dict(self.values)

    def boot_id(self) -> str:
        return self.boot

    def monotonic(self) -> float:
        return self.t

    def boottime(self) -> float:
        return self.t + 0.25  # CLOCK_BOOTTIME runs beside CLOCK_MONOTONIC (equal without a suspend; offset here)

    def utc(self) -> str:
        return f"2026-10-19T10:{int(self.t // 60) % 60:02d}:{self.t % 60:06.3f}Z"


GOOD = ["session_utc=2026-10-19T10:00:00Z", "FB_exit=0", "FC_exit=0", "camera_probe_exit=0", "FF_exit=0",
        "FM_attempt=1 utc=2026-10-19T10:08:00Z", "FM_sync_exit=0", "FM_drop_exit=0", "FM_exit=0",
        "FO_observation=hm-20261019T100900Z-prepared-f2c-20261019t100000z",
        "FO_memory_record=memory-prepared-20261019T100900Z.json", "FO_prepared=complete", "FO_exit=0"]
PREPARED = "memory-prepared-20261019T100900Z.json"


def session(lm, tmp_path: Path, host: Host, provenance: list[str] | None = None, *, before: int = 5_400_000_000,
            after: int = 6_900_000_000, prepared: int = 6_880_000_000, skip: tuple[str, ...] = ()) -> Path:
    """A session folder as f-M and f-O leave it: the provenance and the three records, written by the real snapshot at
    t = 480, 482 and 540 s on the stand-in's clocks; the host's clock then stands at 600 s (the launch check)."""
    w = tmp_path / "f2c-20261019T100000Z"
    w.mkdir()
    (w / "provenance.txt").write_text("\n".join(GOOD if provenance is None else provenance) + "\n")
    for label, name, t, memfree in (("before_drop", "memory-before-drop.json", 480.0, before),
                                    ("after_drop", "memory-after-drop.json", 482.0, after),
                                    ("prepared", PREPARED, 540.0, prepared)):
        if label in skip:
            continue
        host.t, keep = t, host.values["MemFree"]
        host.values["MemFree"] = memfree
        assert lm.main(["snapshot", "--label", label, "--out", str(w / name)], host=host) == 0
        host.values["MemFree"] = keep
    host.t = 600.0
    return w


def run_check(lm, w: Path, host: Host, capsys, reference: Path = REFERENCE) -> tuple[int, dict[str, str], dict | None]:
    out = w / "memory-launch-check.json"
    code = lm.main(["check", "--session-dir", str(w), "--reference-result", str(reference), "--out", str(out),
                    "--prefix", "FV_"], host=host)
    printed = capsys.readouterr().out.splitlines()  # the session's snapshots printed first; the check's lines last
    lines = dict(line.split("=", 1) for line in printed if line.startswith("FV_"))
    assert list(lines) == ["FV_launch_check", "FV_launch_memfree", "FV_launch_floor", "FV_launch_margin",
                           "FV_launch_since_drop_s", "FV_launch_since_prepared_s"]
    record = json.loads(out.read_text()) if out.exists() else None
    return code, lines, record


# ---------------------------------------------------------------- the floor and its derivation


def test_the_fixture_is_the_sealed_reference_result(lm, reference) -> None:
    import hashlib
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == lm.REFERENCE_RESULT_SHA256
    assert reference["status"] == "duration_stop" and reference["guard"]["trigger"] is None
    assert reference["baseline"]["MemFree"] == 5_745_963_008
    assert reference["guard"]["min_mem_free_bytes"] == 1_249_869_824
    assert reference["limits"]["mem_free_floor_bytes"] == GIB


def test_the_floor_is_the_guards_floor_plus_the_reference_runs_largest_consumption(lm, reference) -> None:
    """5,569,835,008 = 1,073,741,824 + (5,745,963,008 - 1,249,869,824): the guard's floor (unchanged, read from the
    guard's own module) plus how far MemFree fell from that run's launch baseline at its lowest; equally, that run's
    launch baseline less its own margin to the floor (176,128,000 B). A whole number of kB, so /proc/meminfo can sit
    on it."""
    floor = lm.derivation(reference)
    assert floor["bytes"] == lm.LAUNCH_FLOOR_BYTES == FLOOR
    assert floor["guard_free_floor_bytes"] == lm.oc.FREE_FLOOR == GIB
    assert floor["reference_consumption_bytes"] == 4_496_093_184
    assert FLOOR == 5_745_963_008 - (1_249_869_824 - GIB) and FLOOR % 1024 == 0 and FLOOR // 1024 == 5_439_292
    assert lm.load_reference(REFERENCE) == floor
    text = lm.derivation_text(floor)
    for part in ("5,569,835,008 B", "1,073,741,824 B", "4,496,093,184 B", "5,745,963,008 B", "1,249,869,824 B",
                 "trk-20261017T103831Z", "an empirical minimum from one successful ten-minute diagnostic",
                 "no safety reserve is added", "guarantees nothing"):
        assert part in text
    assert lm.derivation_text() == text  # the constants say the same as the sealed result


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="guard_stop"),  # a run that did not complete is no basis
    lambda r: r["guard"].update(trigger={"stop": "free_or_available_stop"}),
    lambda r: r["child"].update(returncode=1),
    lambda r: r["limits"].update(mem_free_floor_bytes=2 * GIB),  # the guard's floor is not the one in force
    lambda r: r["baseline"].update(MemFree=5_745_963_009),  # the numbers no longer give the floor
    lambda r: r["guard"].update(min_mem_free_bytes="1249869824"),
    lambda r: r.pop("baseline"),
    lambda r: r.update(mode="profile"),
])
def test_the_derivation_refuses_another_run_or_numbers_that_do_not_reproduce_the_floor(lm, reference, change) -> None:
    changed = copy.deepcopy(reference)
    change(changed)
    with pytest.raises(lm.Refused, match="reference_result_invalid"):
        lm.derivation(changed)


def test_the_reference_file_must_be_the_sealed_one(lm, reference, tmp_path, capsys) -> None:
    altered = tmp_path / "result.json"
    altered.write_text(json.dumps(reference))  # the same values, re-serialised: another file, refused by its hash
    with pytest.raises(lm.Refused, match="reference_result_differs"):
        lm.load_reference(altered)
    with pytest.raises(lm.Refused, match="reference_result_missing"):
        lm.load_reference(tmp_path / "absent.json")
    assert lm.main(["derive", "--reference-result", str(altered)]) == 2
    assert lm.main(["derive", "--reference-result", str(REFERENCE)]) == 0
    assert "5,569,835,008 B" in capsys.readouterr().out


# ---------------------------------------------------------------- the launch check


@pytest.mark.parametrize("memfree, code, verdict", [
    (FLOOR + 1, 0, "passed"),
    (FLOOR, 0, "passed"),  # at the floor passes: the guard itself stops only below its floor
    (FLOOR - 1, 1, "refused:below_floor"),
    (5_439_291 * 1024, 1, "refused:below_floor"),  # one kB under, as /proc/meminfo would show it
])
def test_the_floors_boundary(lm, tmp_path, capsys, memfree, code, verdict) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    host.values["MemFree"] = memfree
    got, lines, record = run_check(lm, w, host, capsys)
    assert got == code and lines["FV_launch_check"] == verdict
    assert lines["FV_launch_margin"] == str(memfree - FLOOR) and lines["FV_launch_floor"] == str(FLOOR)
    assert record["margin_bytes"] == memfree - FLOOR and record["verdict"] == verdict.partition(":")[0]


def test_a_complete_preparation_passes_and_records_the_margin_the_elapsed_times_and_the_changes(
        lm, tmp_path, capsys) -> None:
    host = Host(memfree=6_850_000_000)
    w = session(lm, tmp_path, host)
    code, lines, record = run_check(lm, w, host, capsys)
    assert code == 0 and lines == {
        "FV_launch_check": "passed", "FV_launch_memfree": "6850000000", "FV_launch_floor": str(FLOOR),
        "FV_launch_margin": str(6_850_000_000 - FLOOR), "FV_launch_since_drop_s": "118.0",
        "FV_launch_since_prepared_s": "60.0"}
    assert record["condition"] == "page_cache_dropped_once" and record["reason"] is None
    assert record["floor"]["bytes"] == FLOOR and record["floor"]["reference_session"] == "trk-20261017T103831Z"
    assert record["floor"]["basis"].startswith("an empirical minimum from one successful ten-minute diagnostic")
    assert record["since_drop_s"] == 118.0 and record["since_prepared_s"] == 60.0  # boot time, never judged
    assert record["change_since_drop"]["MemFree"] == 6_850_000_000 - 6_900_000_000
    assert record["change_since_prepared"]["MemFree"] == 6_850_000_000 - 6_880_000_000
    assert record["records"] == {"before_drop": "memory-before-drop.json", "after_drop": "memory-after-drop.json",
                                 "prepared": PREPARED}
    assert record["now"]["boot_id"] == BOOT and record["now"]["monotonic_s"] == 600.0


def test_memory_changing_between_the_preparation_and_the_launch_is_judged_at_the_launch(lm, tmp_path, capsys) -> None:
    """Above the floor after the drop and at f-O, below it at the launch check: refused. The change is recorded."""
    host = Host(memfree=FLOOR - 50_000_000)
    w = session(lm, tmp_path, host, after=FLOOR + 900_000_000, prepared=FLOOR + 400_000_000)
    code, lines, record = run_check(lm, w, host, capsys)
    assert code == 1 and lines["FV_launch_check"] == "refused:below_floor"
    assert record["change_since_prepared"]["MemFree"] == -450_000_000
    assert record["change_since_drop"]["MemFree"] == -950_000_000


def test_a_preparation_below_the_floor_with_the_launch_above_it_passes(lm, tmp_path, capsys) -> None:
    """The reverse: only the reading immediately before the launch is judged; the earlier records are described."""
    host = Host(memfree=FLOOR + 10_000_000)
    w = session(lm, tmp_path, host, after=FLOOR - 20_000_000, prepared=FLOOR - 5_000_000)
    code, lines, _ = run_check(lm, w, host, capsys)
    assert code == 0 and lines["FV_launch_check"] == "passed"


def _without(lines: list[str], prefix: str) -> list[str]:
    return [line for line in lines if not line.startswith(prefix)]


def _replace(lines: list[str], old: str, new: str) -> list[str]:
    assert old in lines
    return [new if line == old else line for line in lines]


@pytest.mark.parametrize("provenance, reason", [
    (_without(GOOD, "FM_"), "drop_not_attempted"),
    (GOOD[:6] + ["FM_attempt=1 utc=2026-10-19T10:08:30Z"] + GOOD[6:], "drop_attempted_more_than_once"),
    (_replace(GOOD, "FM_sync_exit=0", "FM_sync_exit=1"), "sync_failed"),  # a failed sync: the drop never ran
    (_without(GOOD, "FM_sync_exit="), "sync_failed"),
    (_replace(GOOD, "FM_drop_exit=0", "FM_drop_exit=1"), "drop_failed"),  # sudo or sysctl failed
    (_replace(GOOD, "FM_drop_exit=0", "FM_drop_exit=not_run"), "drop_failed"),
    (_without(GOOD, "FM_drop_exit="), "drop_failed"),
    (_replace(GOOD, "FM_exit=0", "FM_exit=1"), "drop_block_failed"),  # e.g. the record after the drop failed
    (_without(GOOD, "FM_exit="), "drop_block_failed"),
    (_without(GOOD, "FF_exit="), "ff_not_passed_before_drop"),
    (GOOD[:4] + GOOD[5:9] + ["FF_exit=0"] + GOOD[9:], "ff_not_passed_before_drop"),  # f-F passed only after the drop
    (_replace(GOOD, "FO_exit=0", "FO_exit=1"), "fo_not_passed_after_drop"),  # f-O stopped: not prepared
    (GOOD[:5] + GOOD[9:] + GOOD[5:9], "fo_not_passed_after_drop"),  # f-O before the drop
    (_without(GOOD, "FO_memory_record="), "prepared_record_not_named"),
    (_replace(GOOD, f"FO_memory_record={PREPARED}", "FO_memory_record=../memory.json"), "prepared_record_not_named"),
])
def test_the_preparation_evidence_must_show_one_successful_drop_between_f_f_and_f_o(
        lm, tmp_path, capsys, provenance, reason) -> None:
    host = Host()
    w = session(lm, tmp_path, host, provenance)
    code, lines, record = run_check(lm, w, host, capsys)
    assert code == 2 and lines["FV_launch_check"] == f"refused:evidence:{reason}"
    assert record["verdict"] == "refused" and record["reason"] == f"evidence:{reason}"
    assert record["now"]["meminfo"]["MemFree"] == 6_900_000_000  # the memory is still recorded, never judged then
    assert lines["FV_launch_margin"] == str(6_900_000_000 - FLOOR)


def test_an_unreadable_provenance_is_refused(lm, tmp_path, capsys) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    (w / "provenance.txt").unlink()
    assert run_check(lm, w, host, capsys)[1]["FV_launch_check"] == "refused:evidence:provenance_unreadable"


@pytest.mark.parametrize("damage, reason", [
    (lambda w: (w / "memory-before-drop.json").unlink(), "before_drop_record_missing"),
    (lambda w: (w / "memory-after-drop.json").unlink(), "after_drop_record_missing"),
    (lambda w: (w / PREPARED).unlink(), "prepared_record_missing"),
    (lambda w: (w / "memory-after-drop.json").write_text("{not json\n"), "after_drop_record_invalid"),
    (lambda w: (w / "memory-after-drop.json").write_text((w / "memory-before-drop.json").read_text()),
     "after_drop_record_invalid"),  # the record before the drop under the name of the one after it
    (lambda w: _edit(w / PREPARED, lambda r: r["meminfo"].pop("MemFree")), "prepared_record_invalid"),
    (lambda w: _edit(w / "memory-before-drop.json", lambda r: r.update(schema="other/1")),
     "before_drop_record_invalid"),
    (lambda w: _edit(w / "memory-after-drop.json", lambda r: r.update(boot_id=OTHER_BOOT)), "boot_changed"),
    (lambda w: _edit(w / "memory-after-drop.json", lambda r: r.update(boottime_s=470.0)), "records_out_of_order"),
    (lambda w: _edit(w / PREPARED, lambda r: r.update(boottime_s=700.0)), "records_out_of_order"),  # after "now"
])
def test_missing_or_invalid_observations_are_refused(lm, tmp_path, capsys, damage, reason) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    damage(w)
    code, lines, record = run_check(lm, w, host, capsys)
    assert code == 2 and lines["FV_launch_check"] == f"refused:evidence:{reason}" and record["verdict"] == "refused"


def _edit(path: Path, change) -> None:
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def test_a_reference_that_is_missing_or_not_the_sealed_file_is_refused(lm, tmp_path, capsys, reference) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    code, lines, _ = run_check(lm, w, host, capsys, reference=tmp_path / "absent.json")
    assert code == 2 and lines["FV_launch_check"] == "refused:evidence:reference_result_missing"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(reference))
    (w / "memory-launch-check.json").unlink()
    assert run_check(lm, w, host, capsys, reference=other)[1]["FV_launch_check"] == \
        "refused:evidence:reference_result_differs"


def test_unreadable_memory_now_is_refused_and_recorded(lm, tmp_path, capsys) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    del host.values["MemFree"]
    code, lines, record = run_check(lm, w, host, capsys)
    assert code == 2 and lines["FV_launch_check"] == "refused:evidence:meminfo_unreadable"
    assert lines["FV_launch_memfree"] == "none" and lines["FV_launch_margin"] == "none" and record["now"] is None


def test_the_check_runs_once_and_never_overwrites_its_record(lm, tmp_path, capsys) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    assert run_check(lm, w, host, capsys)[0] == 0
    kept = (w / "memory-launch-check.json").read_bytes()
    host.values["MemFree"] = FLOOR - 1
    code, lines, _ = run_check(lm, w, host, capsys)
    assert code == 2 and lines["FV_launch_check"] == "refused:evidence:launch_record_exists"
    assert (w / "memory-launch-check.json").read_bytes() == kept


# ---------------------------------------------------------------- snapshots


def test_a_snapshot_is_a_new_file_and_a_reading_without_memfree_writes_nothing(lm, tmp_path, capsys) -> None:
    host = Host()
    out = tmp_path / "memory-before-drop.json"
    assert lm.main(["snapshot", "--label", "before_drop", "--out", str(out)], host=host) == 0
    record = lm.load_record(out, "before_drop")
    assert record["meminfo"]["MemFree"] == 6_900_000_000 and record["boot_id"] == BOOT
    kept = out.read_bytes()
    assert lm.main(["snapshot", "--label", "before_drop", "--out", str(out)], host=host) == 2
    assert out.read_bytes() == kept
    del host.values["MemFree"]
    after = tmp_path / "memory-after-drop.json"
    assert lm.main(["snapshot", "--label", "after_drop", "--out", str(after)], host=host) == 2
    assert not after.exists()
    assert "nothing was written" in capsys.readouterr().err


def test_a_snapshot_prints_the_change_since_an_earlier_record(lm, tmp_path, capsys) -> None:
    host = Host(memfree=5_400_000_000, cached=1_600_000_000)
    before = tmp_path / "memory-before-drop.json"
    assert lm.main(["snapshot", "--label", "before_drop", "--out", str(before)], host=host) == 0
    host.values.update(MemFree=6_900_000_000, Cached=120_000_000)
    host.t += 2
    after = tmp_path / "memory-after-drop.json"
    assert lm.main(["snapshot", "--label", "after_drop", "--out", str(after), "--since", str(before)], host=host) == 0
    printed = capsys.readouterr().out
    assert "change since memory-before-drop.json: MemFree +1,500,000,000 B" in printed
    assert "Cached -1,480,000,000 B" in printed


# ---------------------------------------------------------------- the report


def _result(memfree: int, t_mono: float, status: str = "duration_stop") -> dict:
    return {"mode": "step5_guard", "status": status, "baseline": {"MemFree": memfree, "t_mono": t_mono},
            "guard": {"min_mem_free_bytes": 1_400_000_000, "trigger": None}}


def test_the_report_describes_every_state_and_the_margin_at_the_launch(lm, tmp_path, capsys) -> None:
    host = Host(memfree=6_850_000_000)
    w = session(lm, tmp_path, host)
    assert run_check(lm, w, host, capsys)[0] == 0
    result = w / "result.json"
    result.write_text(json.dumps(_result(6_840_000_000, 600.75)))
    assert lm.main(["report", "--session-dir", str(w), "--result", str(result), "--out", str(w / "report.txt")]) == 0
    lines = (w / "report.txt").read_text().splitlines()
    assert lines[0].startswith("CONDITION page_cache_dropped_once (D103)") and "report this session apart" in lines[0]
    assert lines[1].startswith("launch floor 5,569,835,008 B of MemFree")
    assert any(line.startswith("memory before the drop: MemFree 5,400,000,000 B") for line in lines)
    assert any(line.startswith("change across the drop: MemFree +1,500,000,000 B") for line in lines)
    assert any(line.startswith("memory at f-O's prepared record: MemFree 6,880,000,000 B") for line in lines)
    assert any(line.startswith("memory at the launch check: MemFree 6,850,000,000 B") and "+1,280,164,992 B" in line
               and "(passed)" in line and "since the drop 118.0 s" in line for line in lines)
    launch = next(line for line in lines if line.startswith("memory at the launch (the guard's baseline sample)"))
    assert "MemFree 6,840,000,000 B, +1,270,164,992 B against the launch floor" in launch
    assert "0.750 s after the launch check, MemFree -10,000,000 B since it" in launch
    assert any(line.startswith("the run's lowest MemFree (the guard's): 1,400,000,000 B, +326,258,176 B")
               for line in lines)


def test_the_report_marks_what_is_missing_and_never_overwrites(lm, tmp_path, capsys) -> None:
    host = Host()
    w = session(lm, tmp_path, host, skip=("prepared",))
    out = w / "report.txt"
    assert lm.main(["report", "--session-dir", str(w), "--result", str(w / "absent.json"), "--out", str(out)]) == 0
    text = out.read_text()
    assert "memory at f-O's prepared record: MISSING" in text and "memory at the launch check: MISSING" in text
    assert "memory at the launch (the guard's baseline sample): MISSING" in text
    assert lm.main(["report", "--session-dir", str(w), "--out", str(out)]) == 2


def test_the_check_shows_the_derivation_and_the_margin_to_the_operator(lm, tmp_path, capsys) -> None:
    host = Host()
    w = session(lm, tmp_path, host)
    host.values["MemFree"] = FLOOR - 1
    capsys.readouterr()
    assert lm.main(["check", "--session-dir", str(w), "--reference-result", str(REFERENCE), "--out",
                    str(w / "memory-launch-check.json"), "--prefix", "FV_"], host=host) == 1
    err = capsys.readouterr().err
    assert ("launch floor 5,569,835,008 B of MemFree = the guard's MemFree floor 1,073,741,824 B + "
            "trk-20261017T103831Z's largest MemFree consumption from its launch, 4,496,093,184 B") in err
    assert "MemFree now 5,569,835,007 B: -1 B against the launch floor" in err and "refused:below_floor" in err


def test_the_tool_runs_no_command_and_writes_no_setting() -> None:
    """D37: the operator drops the cache; the runner never runs a command (sudo included) or writes under /proc/sys.
    Judged on the code, not the text: the docstring and the report's condition line name the operator's command."""
    import ast
    tree = ast.parse((RUNNERS / "launch_memory.py").read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names} | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"subprocess", "shutil", "pty", "multiprocessing"}
    called = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {name for name in called if name.startswith(("system", "exec", "spawn", "popen", "fork", "kill"))}
    texts = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not [text for text in texts if "/proc/sys/vm" in text or text.startswith("/etc/sysctl")]
