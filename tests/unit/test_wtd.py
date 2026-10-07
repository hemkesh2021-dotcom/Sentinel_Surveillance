"""MA1-WTD (D57, opt-in diagnostic): transparent huge pages disabled in the workload process alone.

Covers the workload's own prctl and its verification (refused before the CUDA driver, any model library or any load
unless verified), its clip reopen times, the profiler's flag, command, read-only THP_enabled checkpoints and event
relay, the D57 net-step rule (rebounds, transients, loop-coincident steps, unresolved windows), the opportunity and
growth measures, the predeclared reading (supported, contrary, inconclusive) and the summary record. Portable: kernel
files are synthetic; one test calls the real prctl in a child process and is skipped where the kernel lacks it.
"""

from __future__ import annotations

import csv
import errno
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
MIB = 1 << 20
SNAPSHOT = {
    "t_mono": 1.0, "kernel": {"release": "5.15.148-tegra", "version": "#1 SMP PREEMPT"}, "base_page_bytes": 4096,
    "thp_pmd_bytes": 2 * MIB, "hugetlb_default_bytes": 2 * MIB, "enabled": "always", "defrag": "madvise",
    "shmem_enabled": "never", "use_zero_page": 1,
    "khugepaged": {"defrag": 1, "scan_sleep_millisecs": 10000, "alloc_sleep_millisecs": 60000, "pages_to_scan": 4096,
                   "max_ptes_none": 511, "max_ptes_swap": 64, "max_ptes_shared": 256},
    "config": {"HZ": 250, "TRANSPARENT_HUGEPAGE": "y", "TRANSPARENT_HUGEPAGE_ALWAYS": "y",
               "TRANSPARENT_HUGEPAGE_MADVISE": "n", "READ_ONLY_THP_FOR_FS": "n"},
    "read_ms": 3.0,
}


@pytest.fixture
def p(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("demo_profile")


@pytest.fixture
def w(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("demo_workload")


# ---------------------------------------------------------------- the workload's own prctl


def proc_files(status: str | None = "Name:\tpython\nTHP_enabled:\t0\n",
               rollup: str | None = "Rss:\t100 kB\nAnonHugePages:\t0 kB\n"):
    files = {"/proc/self/status": status, "/proc/self/smaps_rollup": rollup}

    def opener(path, mode):
        assert mode == "rb"  # read-only
        if files.get(path) is None:
            raise FileNotFoundError(path)
        return io.BytesIO(files[path].encode())

    return opener


def fake_prctl(set_result=(0, 0), get_result=(1, 0)):
    calls = []

    def call(option, arg2):
        calls.append((option, arg2))
        return set_result if option == 41 else get_result

    return call, calls


def test_a_verified_disable_sets_the_flag_once_and_reads_it_back_three_ways(w) -> None:
    prctl, calls = fake_prctl()
    record = w.disable_thp_for_this_process(prctl=prctl, modules={}, opener=proc_files(), clock=lambda: 12.5)
    assert calls == [(41, 1), (42, 0)]  # PR_SET_THP_DISABLE 1, then PR_GET_THP_DISABLE
    assert record == {"t_mono": 12.5, "requested": True, "model_modules_loaded": [], "set_rc": 0, "set_errno": None,
                      "get_value": 1, "thp_enabled": 0, "anon_huge_pages_bytes": 0, "verified": True, "reason": None}


@pytest.mark.parametrize(("case", "reason"), [
    ("set_fails", "set_failed"),
    ("get_reads_0", "get_mismatch"),
    ("no_status_line", "status_unavailable"),
    ("status_reads_1", "status_mismatch"),
    ("status_oversized", "status_unavailable"),
    ("status_unreadable", "status_unavailable"),
    ("no_rollup", "anon_huge_pages_unavailable"),
    ("rollup_garbled", "anon_huge_pages_unavailable"),
])
def test_any_failed_check_leaves_the_disable_unverified_with_its_reason(w, case, reason) -> None:
    prctl, _ = fake_prctl(set_result=(-1, errno.EINVAL) if case == "set_fails" else (0, 0),
                          get_result=(0, 0) if case == "get_reads_0" else (1, 0))
    status = {"no_status_line": "Name:\tpython\n", "status_reads_1": "THP_enabled:\t1\n",
              "status_oversized": "THP_enabled:\t0\n" + "x" * (20 << 10), "status_unreadable": None}.get(
        case, "THP_enabled:\t0\n")
    rollup = {"no_rollup": None, "rollup_garbled": "AnonHugePages:\tmany\n"}.get(case, "AnonHugePages:\t2048 kB\n")
    record = w.disable_thp_for_this_process(prctl=prctl, modules={}, opener=proc_files(status, rollup))
    assert record["verified"] is False and record["reason"] == reason
    if case == "set_fails":
        assert (record["set_rc"], record["set_errno"]) == (-1, "EINVAL")
    if case == "status_reads_1":  # never inferred from the call's result
        assert record["thp_enabled"] == 1 and record["get_value"] == 1


def test_a_loaded_model_library_refuses_without_setting_anything(w) -> None:
    prctl, calls = fake_prctl()
    record = w.disable_thp_for_this_process(prctl=prctl, modules={"json": 1, "torch": 1, "numpy": 1},
                                            opener=proc_files())
    assert calls == [] and record["verified"] is False and record["reason"] == "model_modules_loaded"
    assert record["model_modules_loaded"] == ["numpy", "torch"]


def test_without_prctl_the_disable_is_unverified(w, monkeypatch) -> None:
    monkeypatch.setattr(w, "libc_prctl", lambda: None)
    record = w.disable_thp_for_this_process(modules={}, opener=proc_files())
    assert (record["verified"], record["reason"], record["set_rc"]) == (False, "prctl_unavailable", None)


def test_the_libc_binding_passes_zero_for_the_unused_arguments(w) -> None:
    seen = []

    class Function:
        def __call__(self, *args):
            seen.append(args)
            return 0

    class Library:
        prctl = Function()

    call = w.libc_prctl(lambda name, use_errno: Library() if name is None and use_errno else None)
    assert call(41, 1) == (0, 0) and seen == [(41, 1, 0, 0, 0)]
    assert len(Library.prctl.argtypes) == 5
    assert w.libc_prctl(lambda name, use_errno: SimpleNamespace()) is None  # no prctl symbol


def run_main(w, monkeypatch, argv, *, verified=True):
    order, events = [], []
    monkeypatch.setattr(w, "event", lambda kind, /, **fields: events.append((kind, fields)))

    def disable():
        order.append("thp_disable")
        return {"verified": verified, "reason": None if verified else "status_mismatch", "t_mono": 1.0}

    monkeypatch.setattr(w, "disable_thp_for_this_process", disable)
    monkeypatch.setattr(w, "check_cuda_driver", lambda: order.append("cuda_driver") or False)
    monkeypatch.setattr(w, "load_detector", lambda engine: order.append("detector") or None)
    return w.main(argv), order, events


def test_the_check_comes_before_the_cuda_driver_and_any_model_and_refuses_unless_verified(w, monkeypatch) -> None:
    flags = ["--port", "1", "--engine", "e", "--post-load-release", "--memory-attribution", "--thp-disable"]
    code, order, events = run_main(w, monkeypatch, flags, verified=False)
    assert (code, order) == (5, ["thp_disable"])  # nothing loads after an unverified check
    assert [name for name, _ in events] == ["thp_disable", "fatal"]
    code, order, events = run_main(w, monkeypatch, flags)
    assert order[:2] == ["thp_disable", "cuda_driver"] and code == 3  # (the fake driver check fails next)
    assert events[0] == ("thp_disable", {"verified": True, "reason": None, "t_mono": 1.0})


def test_without_the_flag_the_workload_never_touches_thp(w, monkeypatch) -> None:
    code, order, events = run_main(w, monkeypatch, ["--port", "1", "--engine", "e", "--memory-attribution"])
    assert order == ["cuda_driver"] and code == 3 and "thp_disable" not in [name for name, _ in events]


def test_the_flag_is_defined_only_with_the_post_load_release(w, monkeypatch) -> None:
    # D57 required --memory-attribution; D58 relaxed that to the post-load release both MA1-WTD and the candidate use.
    for argv in (["--port", "1", "--engine", "e", "--thp-disable"],
                 ["--port", "1", "--engine", "e", "--memory-attribution", "--thp-disable"],
                 ["--port", "1", "--scene-only", "--thp-disable"]):
        with pytest.raises(SystemExit):
            run_main(w, monkeypatch, argv)


def test_the_model_module_lists_agree(p, w) -> None:
    assert p.WTD_MODEL_MODULES == w.WTD_MODEL_MODULES and (w.PR_SET_THP_DISABLE, w.PR_GET_THP_DISABLE) == (41, 42)


def test_reopen_times_are_reported_only_with_the_flag(w, monkeypatch) -> None:
    class Frames:
        def __init__(self, clip, fps):
            self.loops, self.loop_t_mono = 2, [101.25, 161.32]

        def read(self):
            return object()

    class Model:
        def track(self, frame, **kwargs):
            return []

    stats_events = []
    monkeypatch.setattr(w, "Frames", Frames)
    monkeypatch.setattr(w, "scene_attempt", lambda port, frame, stats: {})
    monkeypatch.setattr(w, "face_attempt", lambda deepface, frame, stats: None)
    monkeypatch.setattr(w, "event", lambda kind, /, **fields: kind == "workload_stats" and stats_events.append(fields))
    for flag in (False, True):
        args = SimpleNamespace(clip="c.mp4", fps=30.0, face_hz=5.0, scene_interval_s=1.0, port=1, warmup_s=0.05,
                               steady_s=0.1, **({"thp_disable": True} if flag else {}))
        w.run_workload(Model(), None, args, w.Stats(), SimpleNamespace(phase="face_settle"))
    plain, disabled = stats_events
    assert "clip_loop_t_mono" not in plain and plain["clip_loops"] == 2  # existing telemetry unchanged
    assert disabled["clip_loop_t_mono"] == [101.25, 161.32]


@pytest.mark.skipif(not sys.platform.startswith("linux") or not os.path.exists("/proc/self/status")
                    or "THP_enabled:" not in Path("/proc/self/status").read_text(),
                    reason="needs a Linux kernel that reports THP_enabled")
def test_the_real_prctl_in_a_child_process_is_verified_and_inherited_but_never_reaches_its_parent(w) -> None:
    """The real call, in a throwaway child only: verified, inherited by the child's own child, absent here."""
    before = Path("/proc/self/status").read_text()
    script = (
        "import json, subprocess, sys; sys.path.insert(0, sys.argv[1]); import demo_workload as w\n"
        "record = w.disable_thp_for_this_process(modules={})\n"
        "grandchild = subprocess.run([sys.executable, '-c', \"print([l for l in open('/proc/self/status') "
        "if l.startswith('THP_enabled')][0].split()[1])\"], capture_output=True, text=True).stdout.strip()\n"
        "print(json.dumps({**record, 'grandchild': grandchild}))\n"
    )
    done = subprocess.run([sys.executable, "-c", script, str(RUNNERS)], capture_output=True, text=True, timeout=60,
                          check=True)
    record = json.loads(done.stdout)
    assert record["verified"] is True and (record["set_rc"], record["get_value"], record["thp_enabled"]) == (0, 1, 0)
    assert record["grandchild"] == "0"  # the flag passes to a child's children (fork and exec)
    after = Path("/proc/self/status").read_text()
    assert [l for l in after.splitlines() if l.startswith("THP_enabled")] == [
        l for l in before.splitlines() if l.startswith("THP_enabled")]  # this test process is unaffected


# ---------------------------------------------------------------- the profiler's flag, command and checkpoints


def test_the_profiler_flag_is_off_by_default_and_needs_thp_observation(p, monkeypatch) -> None:
    assert p.parse_args([]).workload_thp_disable is False
    monkeypatch.setattr(p, "scan_processes", lambda: [])
    monkeypatch.setattr(p, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(p, "port_in_use", lambda port: False)
    ma1 = ["--post-load-release", "--memory-attribution"]
    problems, _ = p.preconditions(p.parse_args([*ma1, "--workload-thp-disable"]))
    assert any("only with --thp-observation" in problem for problem in problems)
    problems, _ = p.preconditions(p.parse_args([*ma1, "--thp-observation", "--workload-thp-disable"]))
    assert not any("--workload-thp-disable" in problem for problem in problems)


def test_only_the_workload_command_gains_the_disable_flag(p, monkeypatch, tmp_path) -> None:
    launched = []
    monkeypatch.setattr(p.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))
    ma1thp = ["--clip", "c.mp4", "--post-load-release", "--memory-attribution", "--thp-observation"]
    p.start_workload(p.parse_args(ma1thp), tmp_path)
    p.start_workload(p.parse_args([*ma1thp, "--workload-thp-disable"]), tmp_path)
    assert launched[1] == [*launched[0], "--thp-disable"]  # nothing else in the workload's command changes
    assert "--thp-disable" not in launched[0]


@pytest.mark.parametrize(("text", "value"), [
    ("Name:\tx\nTHP_enabled:\t1\n", 1), ("THP_enabled:\t0\n", 0), ("Name:\tx\n", None), ("THP_enabled:\tmaybe\n", None),
    (None, None), ("THP_enabled:\t1\n" + "x" * (20 << 10), None),
])
def test_thp_enabled_is_read_only_and_unreadable_is_none(p, text, value) -> None:
    def opener(path, mode):
        assert mode == "rb" and str(path) == "/proc/7/status"
        if text is None:
            raise PermissionError(path)
        return io.BytesIO(text.encode())

    assert p.read_thp_enabled(7, opener=opener) == value


def test_each_checkpoint_needs_every_listed_process_read_as_expected(p) -> None:
    states = {11: 1, 22: 1, 33: 0}
    read = lambda pid: states.get(pid)  # noqa: E731
    pids = {"profiler": 11, "llama": 22, "workload": 33}
    assert [p.wtd_scope(name, pids, read=read, clock=lambda: 5.0)["ok"] for name in p.WTD_CHECKPOINTS] == [True] * 4
    before = p.wtd_scope("before_launch", pids, read=read, clock=lambda: 5.0)
    assert before == {"checkpoint": "before_launch", "t_mono": 5.0, "thp_enabled": {"profiler": 1},
                      "expected": {"profiler": 1}, "ok": True}
    states[22] = 0  # llama-server inheriting the flag would fail every later checkpoint
    assert p.wtd_scope("llama_ready", pids, read=read)["ok"] is False
    states[22], states[33] = 1, 1  # the workload not disabled
    assert p.wtd_scope("steady_end", pids, read=read)["ok"] is False
    unread = p.wtd_scope("workload_verified", {"profiler": 11, "llama": 22, "workload": None}, read=read)
    assert unread["ok"] is False and unread["thp_enabled"]["workload"] is None  # missing is never as expected


def test_the_relay_keeps_the_check_as_numbers_and_labels_and_calls_the_checkpoints(p, tmp_path) -> None:
    lines = [
        p.EVENT_PREFIX + json.dumps({"event": "thp_disable", "t_mono": 9.5, "requested": True,
                                     "model_modules_loaded": ["torch", "/home/someone/x"], "set_rc": 0,
                                     "set_errno": "not an errno", "get_value": 1, "thp_enabled": 0,
                                     "anon_huge_pages_bytes": 0, "verified": True, "reason": "free text",
                                     "path": "/home/someone"}) + "\n",
        p.EVENT_PREFIX + json.dumps({"event": "steady_boundary", "edge": "start", "boundary_t_mono": 10.0}) + "\n",
        p.EVENT_PREFIX + json.dumps({"event": "steady_boundary", "edge": "end", "boundary_t_mono": 20.0}) + "\n",
    ]
    proc = SimpleNamespace(stdout=io.StringIO("".join(lines)))
    events = p.Events(tmp_path / "events.jsonl", 0.0)
    called = []
    p.pump_workload(proc, tmp_path, events, lambda name, source="": None, sanitized=True, on_wtd=called.append)
    events.close()
    saved = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    check = next(record for record in saved if record["event"] == "thp_disable")
    assert check["t_mono"] == 9.5 and check["model_modules_loaded"] == ["torch"] and check["verified"] is True
    assert check["set_errno"] is None and check["reason"] is None and "path" not in check
    assert called == ["workload_verified", "steady_end"]
    assert "/home/someone" not in (tmp_path / "events.jsonl").read_text()


# ---------------------------------------------------------------- the net-step rule (D57)


def rows_from(levels: list[int], start: float = 100.0) -> list[dict]:
    return [{"t": round(start + 0.2 * i, 3), "phase": "steady", "work_rss": level} for i, level in enumerate(levels)]


def classes(p, levels: list[int], window=(100.0, 200.0)) -> list[tuple[str, int | None]]:
    return [(item["class"], item["net_bytes"]) for item in p.net_steps(rows_from(levels), window)]


BASE = 2_000 * MIB


def test_a_held_rise_is_a_net_step_of_its_net_size(p) -> None:
    assert classes(p, [BASE] * 10 + [BASE + 3 * MIB] * 10) == [("net_step", 3 * MIB)]


def test_a_dip_and_its_rebound_within_a_second_is_a_rebound_not_growth(p) -> None:
    levels = [BASE] * 10 + [BASE - 2 * MIB] + [BASE - 16384] * 10  # MA1-THP's clip-loop excursions, one row long
    assert classes(p, levels) == [("rebound", -16384)]
    longer = [BASE] * 10 + [BASE - 2 * MIB] * 4 + [BASE] * 10  # four rows down: still inside the 1.0 s window
    assert classes(p, longer) == [("rebound", 0)]


def test_a_dip_longer_than_the_window_is_counted_as_its_rise(p) -> None:
    levels = [BASE] * 10 + [BASE - 2 * MIB] * 5 + [BASE] * 10  # five rows down: the window no longer reaches back
    assert classes(p, levels) == [("net_step", 2 * MIB)]


def test_a_rise_undone_within_a_second_is_a_transient(p) -> None:
    assert classes(p, [BASE] * 10 + [BASE + 2 * MIB] * 3 + [BASE] * 10) == [("transient", 0)]


def test_steps_close_together_each_count_their_own_net_change(p) -> None:
    levels = [BASE] * 10 + [BASE + 2 * MIB] + [BASE + 5 * MIB] * 10
    assert classes(p, levels) == [("net_step", 2 * MIB), ("net_step", 3 * MIB)]
    two_rows = [BASE] * 10 + [BASE + MIB // 2] + [BASE + MIB // 2 + 4 * MIB] * 10  # a beat across two rows
    assert classes(p, two_rows) == [("net_step", 4 * MIB)]


def test_a_rise_below_the_step_threshold_is_not_a_candidate(p) -> None:
    assert classes(p, [BASE] * 10 + [BASE + MIB - 1] * 10) == []


def test_an_incomplete_window_is_unresolved_never_growth(p) -> None:
    assert classes(p, [BASE] * 10 + [BASE + 3 * MIB] * 3) == [("unresolved", None)]  # the run ends
    gap = rows_from([BASE] * 10 + [BASE + 3 * MIB] * 10)
    gap[12]["work_rss"] = None
    assert [item["class"] for item in p.net_steps(gap, (100.0, 200.0))] == ["unresolved"]
    assert classes(p, [BASE] * 3 + [BASE + 3 * MIB] * 10) == [("unresolved", None)]  # too close to the start


def test_only_candidates_inside_the_window_count_but_their_levels_may_reach_past_it(p) -> None:
    levels = [BASE] * 10 + [BASE + 3 * MIB] * 10  # the rise is between the rows at 101.8 s and 102.0 s
    assert classes(p, levels, window=(101.8, 102.0)) == [("net_step", 3 * MIB)]  # its levels come from outside
    assert classes(p, levels, window=(102.0, 104.0)) == []  # its first row is outside the window


# ---------------------------------------------------------------- the reading


def wtd_run(p, tmp_path: Path, *, rss=(), ahp_heap=(), collapse=(), scans=(), headroom=130 * MIB, verified=True,
            scope=None, before_load=True, startup_ahp=0, loops=(), loops_count=None, after=SNAPSHOT, maps=True,
            category="heap", end=170.0, rss_gap=None) -> dict:
    """A synthetic MA1-WTD run: memory.csv rows every 0.2 s from 100 s, smaps every 5 s, the workload's check at
    100.5 s, the detector load at 101 s, steady 110 s to ``end``. rss/ahp_heap/collapse/scans: ((t, amount), ...)."""
    def total(changes, t):
        return sum(amount for at, amount in changes if t >= at - 1e-9)

    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    columns = [*p.Sampler.COLUMNS, *p.THP_COLUMNS]
    times = [round(100 + 0.2 * i, 3) for i in range(int(round((end + 5 - 100) / 0.2)) + 1)]
    with (run / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for t in times:
            row = dict.fromkeys(columns, "")
            row.update(t_mono=f"{t:.3f}", phase="steady", mem_total=8_000_000_000, mem_available=4_000_000_000,
                       mem_free=2_000_000_000, swap_total=0, swap_free=0,
                       work_rss="" if rss_gap and rss_gap[0] <= t <= rss_gap[1] else 2_000_000_000 + total(rss, t),
                       llama_rss=1_000_000_000, anon_pages=1_800_000_000, anon_huge_pages=400 * MIB,
                       thp_t_mono=f"{t + 0.001:.3f}", thp_read_us=80.0, thp_collapse_alloc=100 + total(collapse, t),
                       khugepaged_pages_collapsed=100 + total(collapse, t), khugepaged_full_scans=3 + total(scans, t),
                       thp_split_page=0, thp_split_pmd=0)
            writer.writerow(row)
    events = [{"t_mono": 100.5, "event": "thp_disable", "requested": True, "model_modules_loaded": [], "set_rc": 0,
               "set_errno": None, "get_value": 1, "thp_enabled": 0, "anon_huge_pages_bytes": startup_ahp,
               "verified": verified, "reason": None if verified else "status_mismatch"},
              {"t_mono": 101.0 if before_load else 100.0, "event": "phase", "name": "detector_load"}]
    for name in p.WTD_CHECKPOINTS:
        expected = p.WTD_EXPECTED[name]
        values = dict(expected) if scope is None else scope.get(name, dict(expected))
        events.append({"t_mono": 100.0, "event": "wtd_scope", "checkpoint": name, "thp_enabled": values,
                       "expected": expected, "ok": values == expected})
    t = 100.0
    while t <= end + 5 + 1e-9:
        rss_now = 600 * MIB + total(rss, t)
        heap = {"vmas": 7, "size_bytes": 600 * MIB + headroom, "rss_bytes": rss_now, "private_dirty_bytes": 0,
                "anonymous_bytes": 0, "anon_huge_pages_bytes": total(ahp_heap, t)}
        categories = {name: {"vmas": 1, "size_bytes": MIB, "rss_bytes": MIB, "private_dirty_bytes": 0,
                             "anonymous_bytes": 0, "anon_huge_pages_bytes": 0} for name in p.MAPS_CATEGORIES}
        if category != "heap":  # put the steps' Rss in another category instead
            categories[category] = {**categories[category], "rss_bytes": MIB + total(rss, t)}
            heap["rss_bytes"] = 600 * MIB
        if maps:
            events.append({"t_mono": round(t, 3), "event": "memattr_maps", "phase": "steady", "status": "observed",
                           "categories": {**categories, "heap": heap}})
        t += 5.0
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    manifest = {"thp_settings": {"before": SNAPSHOT, "after": after}}
    stats = {"clip_loops": len(loops) if loops_count is None else loops_count, "clip_loop_t_mono": list(loops),
             "detector": {"unique_fps": 15.0, "latency_ms": {"p50": 44.0}}, "face": {"achieved_hz": 1.0},
             "scene": {"completed": 15}}
    return p.wtd_report(run, "complete", p.load_samples(run / "memory.csv"), manifest, (110.0, end), stats)


def reading(report: dict) -> tuple[str, str | None]:
    return report["outcome"]["label"], report["outcome"]["reason"]


def test_no_net_growth_with_comparable_opportunity_is_supported(p, tmp_path) -> None:
    report = wtd_run(p, tmp_path, collapse=((130.0, 8),))
    assert reading(report) == ("supported", None)
    assert report["intervention"]["verified"] is True and report["intervention"]["effect"]["max_bytes"] == 0
    opportunity = report["growth"]["opportunity"]
    assert opportunity["heap_nonresident_at_first_steady_reading_bytes"] == 130 * MIB
    assert opportunity["required_headroom_bytes"] == p.WTD_BASELINE["steady_heap_rss_rise_bytes"] == 115_912_704
    assert (opportunity["headroom_met"], opportunity["khugepaged_active"], opportunity["comparable"]) == (True, True, True)
    assert report["performance"]["detector"]["unique_fps"] == 15.0  # latency and throughput beside the reading


def test_a_net_heap_step_without_any_workload_huge_page_is_contrary(p, tmp_path) -> None:
    report = wtd_run(p, tmp_path, rss=((130.0, 5 * MIB),), collapse=((130.0, 3),))
    assert reading(report) == ("contrary", "net_growth_without_workload_huge_pages")
    steps = report["growth"]["net_steps"]
    assert (steps["anonymous"], steps["anonymous_bytes"], steps["by_category"]["heap"]) == (1, 5 * MIB, 1)
    assert steps["with_counted_collapse"] == 1  # reported, not a requirement: collapses here are other processes'


def test_contrary_needs_no_opportunity(p, tmp_path) -> None:
    report = wtd_run(p, tmp_path, rss=((130.0, 5 * MIB),), headroom=10 * MIB)
    assert reading(report) == ("contrary", "net_growth_without_workload_huge_pages")


@pytest.mark.parametrize("case", ["rebound", "transient", "file_step", "loop_coincident"])
def test_excursions_file_steps_and_loop_coincident_steps_are_never_growth(p, tmp_path, case) -> None:
    kwargs = {
        "rebound": {"rss": ((129.8, -2 * MIB), (130.0, 2 * MIB))},
        "transient": {"rss": ((130.0, 2 * MIB), (130.4, -2 * MIB))},
        "file_step": {"rss": ((130.0, 3 * MIB),), "category": "file"},
        "loop_coincident": {"rss": ((130.0, 2 * MIB),), "loops": (129.3, 160.0)},
    }[case]
    report = wtd_run(p, tmp_path, collapse=((140.0, 2),), **kwargs)
    assert reading(report) == ("supported", None)
    steps = report["growth"]["net_steps"]
    assert steps["anonymous"] == 0
    if case == "loop_coincident":
        assert (steps["loop_coincident"], steps["loop_coincident_bytes"]) == (1, 2 * MIB)
        assert report["clip_loops"] == {"count": 2, "times_recorded": 2, "complete": True}
    if case in ("rebound", "transient"):
        assert steps["by_class"][case] == 1


@pytest.mark.parametrize(("kwargs", "reason"), [
    ({"verified": False}, "intervention_not_verified"),
    ({"before_load": False}, "intervention_not_verified"),
    ({"scope": {"llama_ready": {"profiler": 1, "llama": 0}}}, "intervention_not_verified"),
    ({"scope": {"steady_end": {"profiler": 1, "llama": 1, "workload": None}}}, "intervention_not_verified"),
    ({"ahp_heap": ((140.0, 2 * MIB),)}, "intervention_not_verified"),
    ({"after": {**SNAPSHOT, "enabled": "madvise"}}, "thp_settings_changed"),
    ({"maps": False}, "intervention_not_verified"),
    ({"rss": ((130.0, 3 * MIB),), "loops": (150.0,), "loops_count": 2}, "clip_loops_unrecorded"),
    ({"rss_gap": (130.4, 130.6), "rss": ((130.0, 3 * MIB),)}, "net_steps_unresolved"),  # a row after it is missing
    ({"headroom": 100 * MIB, "collapse": ((130.0, 8),)}, "no_comparable_opportunity"),
    ({"collapse": ()}, "no_comparable_opportunity"),
])
def test_each_inconclusive_reason_follows_the_predeclared_order(p, tmp_path, kwargs, reason) -> None:
    kwargs = {"collapse": ((130.0, 8),), **kwargs}
    assert reading(wtd_run(p, tmp_path, **kwargs)) == ("inconclusive", reason)


def test_a_full_scan_counts_as_khugepaged_activity_and_two_as_coverage(p, tmp_path) -> None:
    report = wtd_run(p, tmp_path, collapse=(), scans=((120.0, 1), (150.0, 1)))
    assert reading(report) == ("supported", None)
    opportunity = report["growth"]["opportunity"]
    assert opportunity["full_scan_increments"] == [120.0, 150.0] and opportunity["scan_coverage"] == "complete_pass_in_steady"


def test_huge_pages_made_before_the_check_may_stay_but_never_grow(p, tmp_path) -> None:
    kept = wtd_run(p, tmp_path, startup_ahp=4 * MIB, ahp_heap=((100.0, 4 * MIB),), collapse=((130.0, 8),))
    assert reading(kept) == ("supported", None) and kept["intervention"]["effect"]["reference_bytes"] == 4 * MIB
    grown = wtd_run(p, tmp_path, startup_ahp=4 * MIB, ahp_heap=((100.0, 4 * MIB), (140.0, 2 * MIB)),
                    collapse=((130.0, 8),))
    assert reading(grown) == ("inconclusive", "intervention_not_verified")
    assert grown["intervention"]["effect"]["verified"] is False


def test_the_net_step_report_reads_a_saved_run_and_writes_nothing(p, tmp_path) -> None:
    wtd_run(p, tmp_path, rss=((129.8, -2 * MIB), (130.0, 2 * MIB), (140.0, 4 * MIB)), collapse=((140.0, 2),))
    run = tmp_path / "run"
    (run / "manifest.json").write_text(json.dumps({"run_id": "demo-profile-x", "parameters": {"steady_s": 60.0}}))
    with (run / "events.jsonl").open("a") as handle:
        for edge, t in (("start", 110.0), ("end", 170.0)):
            handle.write(json.dumps({"t_mono": t, "event": "steady_boundary", "edge": edge, "boundary_t_mono": t}) + "\n")
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    report = p.net_step_report(run)
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before
    steps = report["growth"]["net_steps"]
    assert steps["by_class"] == {"net_step": 1, "rebound": 1, "transient": 0, "unresolved": 0}
    assert steps["loops_recorded"] is None  # earlier runs recorded no reopen times: nothing is set apart
    assert p.main(["--net-steps", str(run)]) == 0


# ---------------------------------------------------------------- summary


def write_summary_run(p, run: Path, *, wtd: bool) -> None:
    parameters = {"face_hz": 1.0, "scene_interval_s": 4.0, "steady_s": 600.0, "post_load_release": True,
                  "release_settle_s": 5.0, "memory_attribution": True, "maps_interval_s": 5.0,
                  "thp_observation": True, **({"workload_thp_disable": True} if wtd else {})}
    manifest = {"run_id": "demo-profile-20991231T000000Z", "input": {"synthetic": False, "fps": 15.0},
                "parameters": parameters, "llama_server": {"cache_ram_mib": 0, "flags": []},
                "repository": {"commit": "a" * 40, "tracked_changes": False},
                "thp_settings": {"before": SNAPSHOT, "after": SNAPSHOT}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    columns = [*p.Sampler.COLUMNS, *p.THP_COLUMNS]
    with (run / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for i in range(3651):
            t = round(10 + i / 5, 3)
            row = dict.fromkeys(columns, "")
            row.update(t_mono=f"{t:.3f}", phase="warmup" if t < 130 else "steady", mem_total=8_000_000_000,
                       mem_available=4_000_000_000, mem_free=2_000_000_000, cached=1, swap_total=0, swap_free=0,
                       pswpin=0, pswpout=0, work_rss=2_000_000_000, anon_pages=1_800_000_000,
                       anon_huge_pages=300 * MIB, thp_t_mono=f"{t:.3f}", thp_read_us=90.0,
                       thp_collapse_alloc=100 + (7 if t >= 300 else 0),
                       khugepaged_pages_collapsed=100 + (7 if t >= 300 else 0), khugepaged_full_scans=2)
            writer.writerow(row)
    (run / "tegrastats.log").write_text("")
    events = [{"t_mono": 11.0, "event": "thp_disable", "requested": True, "model_modules_loaded": [], "set_rc": 0,
               "set_errno": None, "get_value": 1, "thp_enabled": 0, "anon_huge_pages_bytes": 0, "verified": True,
               "reason": None} if wtd else None,
              {"t_mono": 12.0, "event": "phase", "name": "detector_load"},
              {"t_mono": 130.0, "event": "steady_boundary", "edge": "start", "boundary_t_mono": 130.0},
              {"t_mono": 730.0, "event": "steady_boundary", "edge": "end", "boundary_t_mono": 730.0},
              {"t_mono": 731.0, "event": "workload_stats", "clip_loops": 0, "clip_loop_t_mono": [],
               "detector": {"unique_fps": 15.0}, "face": {}, "scene": {}},
              {"t_mono": 900.0, "event": "run_end", "status": "complete"}]
    events = [e for e in events if e]
    for name in p.WTD_CHECKPOINTS if wtd else ():
        events.append({"t_mono": 10.0, "event": "wtd_scope", "checkpoint": name, "thp_enabled": p.WTD_EXPECTED[name],
                       "expected": p.WTD_EXPECTED[name], "ok": True})
    for t in range(10, 736, 5):
        heap = {"size_bytes": 800 * MIB, "rss_bytes": 600 * MIB, "private_dirty_bytes": 0, "anonymous_bytes": 0,
                "anon_huge_pages_bytes": 0}
        events.append({"t_mono": float(t), "event": "memattr_maps", "phase": "steady", "status": "observed",
                       "read_ms": 2.0, "parse_ms": 1.0, "vmas": 10,
                       "categories": {**{name: {"rss_bytes": MIB, "private_dirty_bytes": 0, "anonymous_bytes": 0,
                                                "anon_huge_pages_bytes": 0} for name in p.MAPS_CATEGORIES},
                                      "heap": heap}})
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "llama-server.log").write_text("t_mono=2.000 prompt cache is disabled\n")


def test_the_summary_writes_the_wtd_record_beside_the_unchanged_thp_record_only_with_the_flag(p, tmp_path) -> None:
    run = tmp_path / "wtd-run"
    run.mkdir()
    write_summary_run(p, run, wtd=True)
    text = p.summarize(run)
    report = json.loads((run / "wtd.json").read_text())
    written = json.loads((run / "profile.json").read_text())
    assert written["workload_thp_disable"] == "wtd.json" and "THP disabled in the workload" in written["instrumentation"]
    assert written["thp_observation"] == "thp.json"  # D56's record and reading are still written, unchanged
    assert json.loads((run / "thp.json").read_text())["outcome"]["rule"] == p.THP_RULE
    assert reading(report) == ("supported", None) and report["intervention"]["verified"] is True
    assert "Workload THP disable (MA1-WTD, D57" in text and "reading: supported" in text
    assert any("non-reproduction is not a fix" in item for item in report["cannot_establish"])
    plain = tmp_path / "thp-run"
    plain.mkdir()
    write_summary_run(p, plain, wtd=False)
    text = p.summarize(plain)
    assert not (plain / "wtd.json").exists() and "MA1-WTD" not in text
    assert "workload_thp_disable" not in json.loads((plain / "profile.json").read_text())
