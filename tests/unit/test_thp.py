"""MA1-THP (D56, opt-in diagnostic): read-only transparent-huge-page observation beside MA1's readings.

Covers the smaps AnonHugePages aggregate, the bounded counter and settings readers, the sampler's extra columns,
the predeclared per-step and per-run reading (supported, unsupported, inconclusive) with its sampling brackets,
splits and unrelated processes, and the summary record. Portable: every kernel file is a synthetic one under
tmp_path; nothing reads or writes a real THP setting, and no workload, model or GPU is involved.
"""

from __future__ import annotations

import csv
import gzip
import importlib
import json
import os
import subprocess
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


def smaps(*mappings: tuple[str, str, str, dict[str, int]]) -> str:
    """Synthetic /proc/<pid>/smaps: (range, perms, path, {field: kB})."""
    lines = []
    for span, perms, path, fields in mappings:
        lines.append(f"{span} {perms} 00000000 00:00 0 {path}".rstrip())
        lines += [f"{name}: {value} kB" for name, value in fields.items()]
    return "\n".join(lines) + "\n"


FULL = {"Rss": 8, "Pss": 8, "Private_Dirty": 8, "Anonymous": 8, "Swap": 0}


# ---------------------------------------------------------------- flag and command


def test_thp_observation_is_off_by_default_and_defined_only_with_ma1(p, monkeypatch) -> None:
    assert p.parse_args([]).thp_observation is False
    monkeypatch.setattr(p, "scan_processes", lambda: [])
    monkeypatch.setattr(p, "command_output", lambda argv, env=None: "inactive")
    monkeypatch.setattr(p, "port_in_use", lambda port: False)
    for flags in (["--thp-observation"], ["--thp-observation", "--post-load-release"]):
        problems, _ = p.preconditions(p.parse_args(flags))
        assert any("only with --memory-attribution" in problem for problem in problems)
    problems, _ = p.preconditions(p.parse_args(["--thp-observation", "--memory-attribution", "--post-load-release"]))
    assert not any("--thp-observation" in problem for problem in problems)


def test_the_workload_command_is_unchanged_by_thp_observation(p, monkeypatch, tmp_path) -> None:
    launched = []
    monkeypatch.setattr(p.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))
    ma1 = ["--clip", "c.mp4", "--post-load-release", "--memory-attribution"]
    p.start_workload(p.parse_args(ma1), tmp_path)
    p.start_workload(p.parse_args([*ma1, "--thp-observation"]), tmp_path)
    assert launched[0] == launched[1] and "--thp-observation" not in launched[1]  # the profiler alone reads THP


# ---------------------------------------------------------------- smaps aggregate


def test_ma1s_default_aggregate_is_unchanged_and_thp_adds_anon_huge_pages_per_category(p) -> None:
    text = smaps(("0400000-0600000", "rw-p", "[heap]", {**FULL, "Rss": 2048, "AnonHugePages": 2048}),
                 ("0600000-0800000", "rw-p", "[heap]", {**FULL, "AnonHugePages": 0}),
                 ("7f00000-7f01000", "rw-p", "", {**FULL, "AnonHugePages": 0}),
                 ("8000000-8001000", "r-xp", "/usr/lib/libx.so", {**FULL, "AnonHugePages": 0}))
    default = p.parse_smaps(text)
    assert set(default["categories"]["heap"]) == {"vmas", "size_bytes", *p.SMAPS_FIELDS.values()}
    observed = p.parse_smaps(text, p.SMAPS_THP_FIELDS)
    heap = observed["categories"]["heap"]
    assert heap["anon_huge_pages_bytes"] == 2 * MIB and heap["rss_bytes"] == (2048 + 8) * 1024
    assert observed["categories"]["anon_other"]["anon_huge_pages_bytes"] == 0
    assert {k: v for k, v in heap.items() if k != "anon_huge_pages_bytes"} == default["categories"]["heap"]
    assert "libx" not in json.dumps(observed)


def test_a_mapping_without_anon_huge_pages_makes_only_its_category_unavailable(p) -> None:
    text = smaps(("0400000-0600000", "rw-p", "[heap]", {**FULL, "AnonHugePages": 2048}),
                 ("0600000-0800000", "rw-p", "[heap]", FULL),  # this kernel line is missing
                 ("7f00000-7f01000", "rw-p", "", {**FULL, "AnonHugePages": 4}))
    categories = p.parse_smaps(text, p.SMAPS_THP_FIELDS)["categories"]
    assert categories["heap"]["anon_huge_pages_bytes"] is None  # unavailable, never a partial sum or zero
    assert categories["heap"]["rss_bytes"] == 16 * 1024 and categories["anon_other"]["anon_huge_pages_bytes"] == 4096


def test_the_smaps_reader_passes_the_thp_fields_and_keeps_its_bounds(p, tmp_path) -> None:
    path = tmp_path / "smaps"
    path.write_text(smaps(("0400000-0600000", "rw-p", "[heap]", {**FULL, "AnonHugePages": 2048})))
    opener = lambda name, mode: open(path, mode)  # noqa: E731 - any pid reads the synthetic file
    record = p.read_maps_aggregate(1, opener=opener, fields=p.SMAPS_THP_FIELDS)
    assert record["status"] == "observed" and record["categories"]["heap"]["anon_huge_pages_bytes"] == 2 * MIB
    assert "anon_huge_pages_bytes" not in p.read_maps_aggregate(1, opener=opener)["categories"]["heap"]


# ---------------------------------------------------------------- counters


def kernel_tree(tmp_path: Path, *, vmstat: str | None = None, khugepaged: dict[str, str] | None = None) -> SimpleNamespace:
    root = tmp_path / "thp"
    (root / "khugepaged").mkdir(parents=True)
    for name, value in (khugepaged if khugepaged is not None else {"pages_collapsed": "1166\n", "full_scans": "14\n"}).items():
        (root / "khugepaged" / name).write_text(value)
    stat = tmp_path / "vmstat"
    stat.write_text(vmstat if vmstat is not None else
                    "nr_free_pages 100\nthp_fault_alloc 4671\nthp_collapse_alloc 1166\nthp_collapse_alloc_failed 0\n"
                    "thp_split_page 0\nthp_split_pmd 3513\nthp_deferred_split_page 424\npswpin 0\n")
    return SimpleNamespace(root=root, vmstat=stat)


def test_counters_are_read_with_their_time_and_cost_and_absent_ones_stay_unavailable(p, tmp_path) -> None:
    tree = kernel_tree(tmp_path)
    ticks = iter([10.0, 10.000_25])
    reading = p.read_thp_counters(vmstat=tree.vmstat, sysfs=tree.root, clock=lambda: 1234.5678,
                                  timer=lambda: next(ticks))
    assert reading["thp_t_mono"] == 1234.568 and reading["thp_read_us"] == 250.0
    assert (reading["thp_fault_alloc"], reading["thp_collapse_alloc"], reading["thp_split_pmd"]) == (4671, 1166, 3513)
    assert reading["thp_collapse_alloc_failed"] == 0 and reading["thp_split_page"] == 0  # read as 0, kept as 0
    assert reading["thp_fault_fallback"] is None and reading["thp_split_page_failed"] is None  # absent: unavailable
    assert (reading["khugepaged_pages_collapsed"], reading["khugepaged_full_scans"]) == (1166, 14)
    assert set(reading) == {"thp_t_mono", "thp_read_us", *p.THP_COUNTER_COLUMNS} and "nr_free_pages" not in reading


@pytest.mark.parametrize("case", ["missing", "oversized", "malformed", "no_khugepaged"])
def test_an_unreadable_counter_source_is_unavailable_never_zero(p, tmp_path, case) -> None:
    tree = kernel_tree(
        tmp_path,
        vmstat={"oversized": "thp_fault_alloc 1\n" + "x 1\n" * 20_000, "malformed": "thp_fault_alloc -3\nthp_split_pmd 1.5\n"
                "thp_collapse_alloc 7 extra\n"}.get(case),
        khugepaged={"no_khugepaged": {}, "malformed": {"pages_collapsed": "abc", "full_scans": ""}}.get(case))
    if case == "missing":
        tree.vmstat.unlink()
    reading = p.read_thp_counters(vmstat=tree.vmstat, sysfs=tree.root)
    vmstat = [reading[key] for key in p.THP_VMSTAT_KEYS]
    if case == "no_khugepaged":  # vmstat readable: only the khugepaged counters are unavailable
        assert reading["thp_collapse_alloc"] == 1166
    else:
        assert all(value is None for value in vmstat)
    if case in ("no_khugepaged", "malformed"):
        assert reading["khugepaged_pages_collapsed"] is None and reading["khugepaged_full_scans"] is None
    else:
        assert reading["khugepaged_pages_collapsed"] == 1166
    assert type(reading["thp_read_us"]) is float


# ---------------------------------------------------------------- settings


def settings_tree(tmp_path: Path, **overrides: str) -> Path:
    root = tmp_path / "settings"
    (root / "khugepaged").mkdir(parents=True)
    values = {"enabled": "[always] madvise never\n", "defrag": "always defer defer+madvise [madvise] never\n",
              "shmem_enabled": "always within_size advise [never] deny force\n", "use_zero_page": "1\n",
              "hpage_pmd_size": "2097152\n", "khugepaged/defrag": "1\n", "khugepaged/scan_sleep_millisecs": "10000\n",
              "khugepaged/alloc_sleep_millisecs": "60000\n", "khugepaged/pages_to_scan": "4096\n",
              "khugepaged/max_ptes_none": "511\n", "khugepaged/max_ptes_swap": "64\n",
              "khugepaged/max_ptes_shared": "256\n", **overrides}
    for name, text in values.items():
        if text is not None:
            (root / name).write_text(text)
    return root


def config_gz(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(text)
    return path


CONFIG = ("CONFIG_HZ_250=y\nCONFIG_HZ=250\nCONFIG_TRANSPARENT_HUGEPAGE=y\nCONFIG_TRANSPARENT_HUGEPAGE_ALWAYS=y\n"
          "# CONFIG_TRANSPARENT_HUGEPAGE_MADVISE is not set\n# CONFIG_READ_ONLY_THP_FOR_FS is not set\n")


def test_the_settings_snapshot_records_kernel_sizes_choices_khugepaged_and_config(p, tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000 kB\nHugepagesize: 2048 kB\n")
    snapshot = p.thp_settings_snapshot(
        sysfs=settings_tree(tmp_path), config=config_gz(tmp_path, CONFIG), meminfo=meminfo, page_size=lambda: 4096,
        uname=lambda: SimpleNamespace(release="5.15.148-tegra", version="#1 SMP PREEMPT"), clock=lambda: 7.0)
    expected = {key: value for key, value in SNAPSHOT.items() if key not in ("read_ms", "t_mono")}
    assert {key: value for key, value in snapshot.items() if key not in ("read_ms", "t_mono")} == expected
    assert snapshot["t_mono"] == 7.0 and snapshot["read_ms"] >= 0


def test_unreadable_or_unrecognised_settings_are_unavailable_not_defaults(p, tmp_path) -> None:
    root = settings_tree(tmp_path, **{"enabled": "always madvise never\n", "khugepaged/max_ptes_none": "-1\n",
                                      "defrag": None})
    snapshot = p.thp_settings_snapshot(sysfs=root, config=tmp_path / "absent.gz", meminfo=tmp_path / "absent",
                                       page_size=lambda: (_ for _ in ()).throw(ValueError("no sysconf")))
    assert snapshot["enabled"] is None and snapshot["defrag"] is None  # no bracketed choice / no file
    assert snapshot["khugepaged"]["max_ptes_none"] is None and snapshot["khugepaged"]["pages_to_scan"] == 4096
    assert snapshot["config"] is None and snapshot["hugetlb_default_bytes"] is None
    assert snapshot["base_page_bytes"] is None
    big = config_gz(tmp_path, "#" * (p.THP_CONFIG_LIMIT_BYTES + 10))
    assert p.read_kernel_config(big) is None  # bounded decompression
    garbled = tmp_path / "garbled.gz"
    garbled.write_bytes(b"not gzip")
    assert p.read_kernel_config(garbled) is None


def test_settings_changes_compare_only_values_read_on_both_sides(p) -> None:
    after = json.loads(json.dumps(SNAPSHOT))
    after["khugepaged"]["max_ptes_none"] = 0
    after["config"] = None
    changed, unverified = p.thp_settings_changes(SNAPSHOT, after)
    assert changed == ["khugepaged.max_ptes_none"]
    assert unverified == [f"config.{key}" for key in p.THP_CONFIG_KEYS]
    assert p.thp_settings_changes(SNAPSHOT, None) == ([], list(p.THP_SETTING_KEYS))  # missing: never "unchanged"
    assert p.thp_settings_changes(SNAPSHOT, SNAPSHOT) == ([], [])


# ---------------------------------------------------------------- sampler


def test_the_sampler_adds_thp_columns_after_every_existing_column_in_the_same_row(p, tmp_path, monkeypatch) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000000 kB\nMemFree: 4000000 kB\nMemAvailable: 6000000 kB\nCached: 1 kB\n"
                       "SwapTotal: 0 kB\nSwapFree: 0 kB\nAnonPages: 100 kB\nAnonHugePages: 315392 kB\n")
    read_meminfo = p.read_meminfo
    monkeypatch.setattr(p, "read_meminfo", lambda keys=p.MEMINFO_KEYS: read_meminfo(meminfo, keys))
    monkeypatch.setattr(p, "read_swap_counters", lambda: {"pswpin": 0, "pswpout": 0})
    monkeypatch.setattr(p.time, "monotonic", lambda: 50.0)
    reading = {"thp_t_mono": 50.001, "thp_read_us": 87.5, **dict.fromkeys(p.THP_COUNTER_COLUMNS),
               "thp_collapse_alloc": 1166, "khugepaged_pages_collapsed": 1166, "thp_split_page": 0}
    path = tmp_path / "memory.csv"
    sampler = p.Sampler(path, lambda: pytest.fail("unexpected floor"), thp=True, read_thp=lambda: reading)
    monkeypatch.setattr(sampler._halt, "wait", lambda timeout: sampler._halt.set())
    sampler.run()
    with path.open(newline="") as handle:
        header, row = list(csv.reader(handle))
    assert header == [*p.Sampler.COLUMNS, *p.THP_COLUMNS] and len(row) == len(header)
    assert header[:len(p.Sampler.COLUMNS)] == p.Sampler.COLUMNS  # every existing column keeps its position
    (sample,) = p.load_samples(path)
    assert sample["anon_huge_pages"] == 315392 * 1024 and sample["anon_pages"] == 100 * 1024
    assert sample["thp_collapse_alloc"] == 1166 and sample["thp_split_page"] == 0
    assert sample["thp_fault_alloc"] is None and dict(zip(header, row))["thp_fault_alloc"] == ""  # blank, not 0
    assert sample["thp_t_mono"] == 50.001 and sample["thp_read_us"] == 87.5


def test_without_the_flag_the_sampler_reads_and_writes_exactly_as_before(p, tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(p, "read_meminfo", lambda: calls.append("meminfo") or {
        "MemTotal": 1, "MemAvailable": 1, "SwapTotal": 0, "SwapFree": 0})
    monkeypatch.setattr(p, "read_swap_counters", lambda: {})
    path = tmp_path / "memory.csv"
    sampler = p.Sampler(path, lambda: None, read_thp=lambda: pytest.fail("THP read without the flag"))
    monkeypatch.setattr(sampler._halt, "wait", lambda timeout: sampler._halt.set())
    sampler.run()
    header = path.read_text().splitlines()[0].split(",")
    assert header == p.Sampler.COLUMNS and calls == ["meminfo"]
    assert "anon_huge_pages" not in p.load_samples(path)[0]


# ---------------------------------------------------------------- the predeclared reading


def run_dir(p, tmp_path: Path, *, rss=(), collapse=(), ahp=(), fault=(), split=(), llama=(), counters=True,
            maps_ahp=True, after=SNAPSHOT, end=160.0, maps_every=5.0) -> dict:
    """A synthetic MA1-THP run: memory.csv rows every 0.2 s from 100 s, smaps readings every ``maps_every`` s.

    Each of rss/collapse/ahp/fault/split/llama is ((t, amount), ...): the value rises by ``amount`` at the row (or
    reading) at time t and stays. Warm-up is 100-110 s and the steady interval 110 s to ``end``."""
    def total(changes, t):
        return sum(amount for at, amount in changes if t >= at - 1e-9)

    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    columns = [*p.Sampler.COLUMNS, *p.THP_COLUMNS]
    times = [round(100 + 0.2 * i, 3) for i in range(int(round((end - 100) / 0.2)) + 1)]
    with (run / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for t in times:
            row = dict.fromkeys(columns, "")
            row.update(t_mono=f"{t:.3f}", phase="warmup" if t < 110 else "steady", mem_total=8_000_000_000,
                       mem_available=4_000_000_000, mem_free=2_000_000_000, swap_total=0, swap_free=0,
                       work_rss=2_000_000_000 + total(rss, t), llama_rss=1_000_000_000 + total(llama, t),
                       anon_pages=1_800_000_000 + total(rss, t) + total(llama, t))
            if counters:
                row.update(anon_huge_pages=300 * MIB + total(ahp, t), thp_t_mono=f"{t + 0.001:.3f}", thp_read_us=80.0,
                           thp_fault_alloc=50 + total(fault, t), thp_collapse_alloc=100 + total(collapse, t),
                           khugepaged_pages_collapsed=100 + total(collapse, t), thp_split_page=0,
                           thp_split_pmd=10 + total(split, t), khugepaged_full_scans=3)
            writer.writerow(row)
    events = []
    t = 100.0
    while t <= end + 1e-9:
        heap = {"rss_bytes": 600 * MIB + total(rss, t), "private_dirty_bytes": 0, "anonymous_bytes": 0}
        if maps_ahp:
            heap["anon_huge_pages_bytes"] = 100 * MIB + total(ahp, t)
        categories = {name: {"rss_bytes": MIB, "private_dirty_bytes": 0, "anonymous_bytes": 0,
                             **({"anon_huge_pages_bytes": 0} if maps_ahp else {})} for name in p.MAPS_CATEGORIES}
        events.append({"t_mono": round(t, 3), "event": "memattr_maps", "phase": "steady", "status": "observed",
                       "categories": {**categories, "heap": heap}})
        t += maps_every
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    manifest = {"thp_settings": {"before": SNAPSHOT, "after": after}}
    return p.thp_report(run, "complete", p.load_samples(run / "memory.csv"), manifest, (100.0, 110.0), (110.0, end))


def test_a_step_with_a_counted_collapse_and_a_local_huge_page_rise_is_supported(p, tmp_path) -> None:
    report = run_dir(p, tmp_path, rss=((120.0, 7 * MIB),), collapse=((120.0, 4),), ahp=((120.0, 8 * MIB),))
    (step,) = report["steps"]["listed"]
    assert step["outcome"] == "supported" and step["reasons"] == [] and step["collapse_counted"] is True
    assert step["maps_bracket"]["step_category"] == "heap" and step["maps_bracket"]["local_ahp_delta"] == 8 * MIB
    assert step["counter_bracket"]["deltas"]["khugepaged_pages_collapsed"] == 4
    assert (step["counter_bracket"]["t0"], step["counter_bracket"]["t1"]) == (119.6, 120.2)
    assert report["outcome"]["label"] == "supported" and report["outcome"]["reason"] is None
    assert report["steps"]["bytes_by_outcome"]["supported"] == 7 * MIB
    assert report["sources"]["steady_counter_readings"] > 0 and report["sources"]["steady_maps_heap_ahp_observed"] > 0
    assert report["settings"]["before_recorded"] is True and report["settings"]["changed"] == []


def test_a_step_with_neither_signal_is_unsupported(p, tmp_path) -> None:
    report = run_dir(p, tmp_path, rss=((120.0, 7 * MIB),))
    (step,) = report["steps"]["listed"]
    assert (step["outcome"], step["collapse_counted"], step["maps_bracket"]["local_ahp_delta"]) == ("unsupported", False, 0)
    assert report["outcome"]["label"] == "unsupported"


@pytest.mark.parametrize(("case", "reasons"), [
    ("collapse_elsewhere", ["collapse_counted_without_local_huge_page_rise"]),
    ("collapse_with_split", ["collapse_counted_without_local_huge_page_rise", "split_counted_in_maps_bracket"]),
    ("fault_time_huge_page", ["local_huge_page_rise_without_counted_collapse"]),
    ("counters_unavailable", ["collapse_counters_unavailable"]),
    ("maps_without_anon_huge_pages", ["process_maps_unavailable"]),
])
def test_a_step_with_one_signal_or_an_unavailable_one_is_inconclusive(p, tmp_path, case, reasons) -> None:
    kwargs = {"collapse_elsewhere": {"collapse": ((120.0, 2),)},
              "collapse_with_split": {"collapse": ((120.0, 2),), "split": ((117.0, 1),)},
              "fault_time_huge_page": {"fault": ((120.0, 1),), "ahp": ((120.0, 2 * MIB),)},
              "counters_unavailable": {"counters": False, "ahp": ((120.0, 2 * MIB),)},
              "maps_without_anon_huge_pages": {"collapse": ((120.0, 2),), "maps_ahp": False}}[case]
    report = run_dir(p, tmp_path, rss=((120.0, 3 * MIB),), **kwargs)
    (step,) = report["steps"]["listed"]
    assert step["outcome"] == "inconclusive" and step["reasons"] == reasons
    assert report["outcome"] == {"label": "inconclusive", "reason": "only_inconclusive_steps", "rule": p.THP_RULE}
    if case == "counters_unavailable":  # never read as "no collapse"
        assert step["collapse_counted"] is None and report["sources"]["steady_counter_readings"] == 0
        assert report["windows"]["steady"]["counters"]["thp_collapse_alloc"]["delta"] is None


@pytest.mark.parametrize(("collapse_at", "outcome"), [  # the step: work_rss rises between rows 119.8 and 120.0
    (119.8, "supported"),  # first seen at the step's first row: one interval early, inside the tolerance
    (120.0, "supported"),  # at its second row
    (120.2, "supported"),  # one interval late: inside
    (119.6, "inconclusive"),  # two intervals early: outside, so the local rise has no counted collapse
    (120.4, "inconclusive"),  # two intervals late: outside
])
def test_the_counter_bracket_tolerates_exactly_one_interval_each_side(p, tmp_path, collapse_at, outcome) -> None:
    report = run_dir(p, tmp_path, rss=((120.0, 3 * MIB),), collapse=((collapse_at, 1),), ahp=((120.0, 2 * MIB),))
    (step,) = report["steps"]["listed"]
    assert step["outcome"] == outcome


def test_no_count_of_collapses_in_a_bracket_is_a_rule(p, tmp_path) -> None:
    report = run_dir(p, tmp_path, rss=((120.0, 13 * MIB),), collapse=((120.0, 20),), ahp=((120.0, 40 * MIB),))
    assert report["steps"]["listed"][0]["outcome"] == "supported" and report["outcome"]["label"] == "supported"


@pytest.mark.parametrize(("case", "label", "reason"), [
    ("mixed", "inconclusive", "mixed"),
    ("no_steps", "inconclusive", "no_steady_steps_not_reproduced"),
    ("warmup_step_only", "inconclusive", "no_steady_steps_not_reproduced"),
    ("warmup_allocation_and_steady_fill", "supported", None),
    ("settings_changed", "inconclusive", "thp_settings_changed"),
    ("settings_after_missing", "supported", None),
    ("supported_and_inconclusive", "supported", None),
])
def test_the_run_reading_follows_the_predeclared_rule(p, tmp_path, case, label, reason) -> None:
    supported = {"rss": ((120.0, 3 * MIB),), "collapse": ((120.0, 1),), "ahp": ((120.0, 2 * MIB),)}
    if case == "mixed":
        kwargs = {"rss": ((120.0, 3 * MIB), (140.0, 3 * MIB)), "collapse": ((120.0, 1),), "ahp": ((120.0, 2 * MIB),)}
    elif case == "no_steps":
        kwargs = {"collapse": ((120.0, 1),)}  # collapses without any RSS step: nothing to attribute
    elif case == "settings_changed":
        after = json.loads(json.dumps(SNAPSHOT))
        after["enabled"] = "madvise"
        kwargs = {**supported, "after": after}
    elif case == "settings_after_missing":
        kwargs = {**supported, "after": None}
    elif case == "warmup_step_only":  # a warm-up step is labelled but never counted
        kwargs = {"rss": ((105.0, 3 * MIB),), "collapse": ((105.0, 1),), "ahp": ((105.0, 2 * MIB),)}
    elif case == "warmup_allocation_and_steady_fill":  # a warm-up allocation step does not make the reading mixed
        kwargs = {"rss": ((105.0, 60 * MIB), (120.0, 3 * MIB)), "collapse": ((120.0, 1),), "ahp": ((120.0, 2 * MIB),)}
    else:
        kwargs = {"rss": ((120.0, 3 * MIB), (140.0, 3 * MIB)), "collapse": ((120.0, 1), (140.0, 1)),
                  "ahp": ((120.0, 2 * MIB),)}
    report = run_dir(p, tmp_path, **kwargs)
    assert (report["outcome"]["label"], report["outcome"]["reason"]) == (label, reason)
    if case == "settings_after_missing":  # not comparable, and recorded as such for the exit status to see
        assert report["settings"]["after_recorded"] is False and report["settings"]["changed"] == []
    if case == "no_steps":
        assert report["steps"]["count"] == 0 and report["collapses"]["readings"] == 1
    if case == "warmup_allocation_and_steady_fill":
        warm, steady = report["steps"]["listed"]
        assert (warm["window"], warm["outcome"], steady["window"]) == ("warmup", "unsupported", "steady")
        assert report["steps"]["warmup_by_outcome"]["unsupported"] == 1 and report["steps"]["steady_count"] == 1
        assert report["steps"]["by_outcome"] == {"supported": 1, "unsupported": 0, "inconclusive": 0}


def test_collapse_readings_record_which_recorded_rss_rose_beside_them(p, tmp_path) -> None:
    report = run_dir(p, tmp_path, rss=((120.0, 3 * MIB),), llama=((130.0, 4 * MIB),),
                     collapse=((120.0, 1), (130.0, 1), (140.0, 1)), ahp=((120.0, 2 * MIB),))
    listed = report["collapses"]["listed"]
    assert [(c["t_mono"], c["rss"]) for c in listed] == [
        (120.0, "workload_rss_rose"), (130.0, "llama_rss_rose"), (140.0, "no_recorded_rss_rise")]
    assert report["collapses"]["by_rss"]["llama_rss_rose"] == 1
    assert report["collapses"]["gaps_s"] == {"n": 2, "min": 10.0, "p50": 10.0, "max": 10.0}  # descriptive only
    assert report["steps"]["count"] == 1  # llama-server's rise is not a workload step


def test_steps_sharing_one_smaps_bracket_are_marked(p, tmp_path) -> None:
    report = run_dir(p, tmp_path, rss=((121.0, 2 * MIB), (123.0, 2 * MIB)), collapse=((121.0, 1), (123.0, 1)),
                     ahp=((121.0, 2 * MIB),))
    assert [s["steps_in_maps_bracket"] for s in report["steps"]["listed"]] == [2, 2]
    assert [s["outcome"] for s in report["steps"]["listed"]] == ["supported", "supported"]


# ---------------------------------------------------------------- summary


def write_summary_run(p, run: Path, *, thp: bool) -> None:
    parameters = {"face_hz": 1.0, "scene_interval_s": 4.0, "steady_s": 600.0, "post_load_release": True,
                  "release_settle_s": 5.0, "memory_attribution": True, "maps_interval_s": 5.0,
                  **({"thp_observation": True} if thp else {})}
    manifest = {"run_id": "demo-profile-20991231T000000Z", "input": {"synthetic": False, "fps": 15.0},
                "parameters": parameters, "llama_server": {"cache_ram_mib": 0, "flags": []},
                "repository": {"commit": "a" * 40, "tracked_changes": False},
                **({"thp_settings": {"before": SNAPSHOT, "after": SNAPSHOT}} if thp else {})}
    (run / "manifest.json").write_text(json.dumps(manifest))
    columns = [*p.Sampler.COLUMNS, *(p.THP_COLUMNS if thp else ())]
    with (run / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for i in range(3601):
            t = round(10 + i / 5, 3)
            grown = t >= 300.0
            row = dict.fromkeys(columns, "")
            row.update(t_mono=f"{t:.3f}", phase="warmup" if t < 130 else "steady", mem_total=8_000_000_000,
                       mem_available=4_000_000_000, mem_free=2_000_000_000, cached=1, swap_total=0, swap_free=0,
                       pswpin=0, pswpout=0, work_rss=2_000_000_000 + (12 * MIB if grown else 0),
                       anon_pages=1_800_000_000 + (12 * MIB if grown else 0))
            if thp:
                row.update(anon_huge_pages=300 * MIB + (14 * MIB if grown else 0), thp_t_mono=f"{t:.3f}",
                           thp_read_us=90.0, thp_collapse_alloc=100 + (7 if grown else 0),
                           khugepaged_pages_collapsed=100 + (7 if grown else 0), khugepaged_full_scans=2)
            writer.writerow(row)
    (run / "tegrastats.log").write_text("")
    events = [{"t_mono": 130.0, "event": "steady_boundary", "edge": "start", "boundary_t_mono": 130.0},
              {"t_mono": 730.0, "event": "steady_boundary", "edge": "end", "boundary_t_mono": 730.0},
              {"t_mono": 900.0, "event": "run_end", "status": "complete"}]
    for t in range(10, 731, 5):
        heap = {"rss_bytes": 600 * MIB + (12 * MIB if t >= 300 else 0), "private_dirty_bytes": 0, "anonymous_bytes": 0}
        if thp:
            heap["anon_huge_pages_bytes"] = 100 * MIB + (14 * MIB if t >= 300 else 0)
        events.append({"t_mono": float(t), "event": "memattr_maps", "phase": "steady", "status": "observed",
                       "read_ms": 2.0, "parse_ms": 1.0, "vmas": 10,
                       "categories": {**{name: {"rss_bytes": MIB, "private_dirty_bytes": 0, "anonymous_bytes": 0}
                                         for name in p.MAPS_CATEGORIES}, "heap": heap}})
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "llama-server.log").write_text("t_mono=2.000 prompt cache is disabled\n")


def test_the_summary_writes_the_thp_record_only_with_the_flag(p, tmp_path) -> None:
    run = tmp_path / "thp-run"
    run.mkdir()
    write_summary_run(p, run, thp=True)
    text = p.summarize(run)
    report = json.loads((run / "thp.json").read_text())
    written = json.loads((run / "profile.json").read_text())
    assert written["thp_observation"] == "thp.json" and "THP observation (D56), read-only" in written["instrumentation"]
    assert written["criteria_id"] == "step4plr-combined-cache-off-v2"
    attribution = json.loads((run / "memattr.json").read_text())  # MA1's own record, beside it
    assert [s["t_mono"] for s in attribution["steps"]["listed"]] == [s["t_mono"] for s in report["steps"]["listed"]]
    (step,) = report["steps"]["listed"]
    assert step["outcome"] == "supported" and report["outcome"]["label"] == "supported"
    steady = report["windows"]["steady"]
    assert steady["counters"]["khugepaged_pages_collapsed"]["delta"] == 7
    assert steady["maps"]["heap"]["anon_huge_pages_bytes"]["delta"] == 14 * MIB
    assert steady["counters"]["anon_huge_pages"]["t_first"] == 130.0
    assert "THP observation (D56; read-only" in text and "reading: supported" in text
    assert any("non-reproduction is not a fix" in item for item in report["cannot_establish"])
    plain = tmp_path / "ma1-run"
    plain.mkdir()
    write_summary_run(p, plain, thp=False)
    text = p.summarize(plain)
    assert not (plain / "thp.json").exists() and "THP observation" not in text
    assert "thp_observation" not in json.loads((plain / "profile.json").read_text())


def test_the_real_kernel_files_are_only_read(p) -> None:
    """On this host, when they exist: the readers return numbers or None and never write (opened read-only)."""
    opened = []

    def opener(path, mode):
        opened.append(mode)
        return open(path, mode)

    reading = p.read_thp_counters(opener=opener)
    snapshot = p.thp_settings_snapshot(opener=opener)
    assert all(mode == "rb" for mode in opened)
    assert all(value is None or type(value) is int for key, value in reading.items() if key in p.THP_COUNTER_COLUMNS)
    assert snapshot["base_page_bytes"] in (None, os.sysconf("SC_PAGE_SIZE"))
