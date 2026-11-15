from __future__ import annotations

import importlib
import hashlib
import io
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
COMMIT = "a" * 40
BOOT = "test-boot"
SERVICES = (
    "ActiveState=inactive\nUnitFileState=enabled\nId=ollama.service\nLoadState=loaded\n\n"
    "Id=gdm.service\nNames=gdm.service display-manager.service\nActiveState=inactive\n"
    "UnitFileState=disabled\nLoadState=loaded\n"
)


@pytest.fixture
def operator(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("operator_check")


class FakeChild:
    def __init__(self, backend, argv, *, duration=0.0, returncode=0, output=b"", stubborn=False, descendants=False,
                 errors=b""):
        self.backend = backend
        self.argv = argv
        self.errors = errors
        self.end = backend.now + duration
        self.returncode = returncode
        self.chunks = [output]
        self.stubborn = stubborn
        self.descendants = descendants
        self.killed = False
        self.closed = False
        self.signals = []

    def poll(self):
        return self.returncode if self.killed or self.backend.now >= self.end else None

    def read(self):
        return self.chunks.pop(0) if self.chunks else b""

    def error_output(self):
        return self.errors

    def exists(self):
        return self.stubborn or (not self.killed and (self.descendants or self.poll() is None))

    def signal(self, signum):
        if self.exists():
            self.signals.append(signum)
            if signum == signal.SIGKILL and not self.stubborn:
                self.killed = True
                self.returncode = -9

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self):
        self.now = 0.0
        self.children = []
        self.services = SERVICES
        self.journal = json.dumps({"MESSAGE": "NvMapMemAlloc secret://user:password@camera failed"}).encode()
        self.overrides = {}
        self.context_overrides = {}
        self.hardware_plan = {}
        self.telemetry = lambda: {}

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def sample(self):
        return {
            "t_mono": self.now, "MemTotal": 8_000_000_000, "MemAvailable": 7_000_000_000,
            "MemFree": 6_000_000_000, "pswpin": 0, "pswpout": 0, **self.telemetry(),
        }

    def context(self):
        return {
            "boot_id": BOOT, "root": False, "port_18081_in_use": False, "cached_assets_available": True,
            "process_inspection_unavailable": False,
            "workloads": {key: {"count": 0, "pids": []} for key in
                          ("model_servers", "desktop", "dev_tools", "python_unclassified", "media_or_gpu_tools")},
            **self.context_overrides,
        }

    def spawn(self, argv, env, capture_stderr=False):
        self.captured = [*getattr(self, "captured", []), capture_stderr]
        if "systemctl" in argv[0]:
            plan = {"output": self.services.encode(), **self.overrides.get("services", {})}
        elif "journalctl" in argv[0]:
            plan = {"output": self.journal, **self.overrides.get("journal", {})}
        elif argv[0] == "git":
            plan = {"output": COMMIT.encode(), **self.overrides.get("git", {})}
        else:
            if "--api" in argv:
                api = argv[argv.index("--api") + 1]
                plan = {"duration": 0.4, "output": json.dumps({
                    "api": api, "status": "bounded_smoke_complete", "allocated_bytes": 256 * (1 << 20),
                    "cap_bytes": 256 * (1 << 20), "chunk_bytes": 32 * (1 << 20), "cleanup_clear": True,
                }).encode()}
            else:
                plan = {"duration": 0.4, "output": b"secret raw child text not persisted"}
            plan.update(self.hardware_plan)
        child = FakeChild(self, argv, **plan)
        self.children.append(child)
        return child


def hardware_children(backend):
    return [child for child in backend.children if child.argv[0] != "git"
            and "systemctl" not in child.argv[0] and "journalctl" not in child.argv[0]]


def write_check9(tmp_path, operator, backend=None):
    output = tmp_path / "sentinel-operator-fake"
    output.mkdir(mode=0o700)
    result = operator.execute("check9", backend or FakeBackend(), output)
    path = output / "result.json"
    path.write_text(json.dumps(result))
    path.chmod(0o600)
    return path


def test_default_inspection_only_reads_and_reports_unknown_acceptance(operator, tmp_path):
    backend = FakeBackend()
    result = operator.execute(None, backend, tmp_path)
    assert hardware_children(backend) == []
    assert all(child.closed for child in backend.children)
    assert result["check9"]["status"] == result["u21"]["status"] == "PENDING"
    assert result["inspection"]["services"]["ollama.service"] == {
        "ActiveState": "inactive", "UnitFileState": "enabled", "LoadState": "loaded",
    }
    assert result["inspection"]["services"]["display-manager.service"]["ActiveState"] == "inactive"
    assert result["inspection"]["kernel"]["nvmap_candidates"] == 1
    assert result["inspection"]["workload_refusals"] == []
    assert "password" not in json.dumps(result) and "secret" not in json.dumps(result)
    assert not (tmp_path / "guard.jsonl").exists()


@pytest.mark.parametrize("condition", [
    "root", "port_18081_in_use", "process_inspection_unavailable", "cached_assets_missing",
    "model_servers", "desktop", "dev_tools", "python_unclassified", "media_or_gpu_tools",
    "ollama_active", "display_manager_active", "services_unavailable", "journal_unavailable", "commit_unavailable",
])
def test_prerequisite_refusal_starts_no_gpu_or_models(operator, tmp_path, condition):
    backend = FakeBackend()
    if condition in ("root", "port_18081_in_use", "process_inspection_unavailable"):
        backend.context_overrides[condition] = True
    elif condition == "cached_assets_missing":
        backend.context_overrides["cached_assets_available"] = False
    elif condition in backend.context()["workloads"]:
        workloads = backend.context()["workloads"]
        workloads[condition] = {"count": 1, "pids": [1234]}
        backend.context_overrides["workloads"] = workloads
    elif condition == "ollama_active":
        backend.services = SERVICES.replace("ActiveState=inactive", "ActiveState=active", 1)
    elif condition == "display_manager_active":
        backend.services = SERVICES.replace("Names=gdm.service display-manager.service\nActiveState=inactive",
                                            "Names=gdm.service display-manager.service\nActiveState=active")
    elif condition == "services_unavailable":
        backend.overrides["services"] = {"returncode": 1}
    elif condition == "journal_unavailable":
        backend.overrides["journal"] = {"returncode": 1}
    else:
        backend.overrides["git"] = {"output": b"unknown"}
    result = operator.execute("check9", backend, tmp_path)
    assert result["check9"]["status"] == "refused"
    assert result["u21"]["status"] == "PENDING"
    assert hardware_children(backend) == []


@pytest.mark.parametrize("update", [
    {"MemAvailable": 6_000_000_000}, {"MemFree": 3_499_999_999},
    {"MemAvailable": 3_999_999_999}, {"MemFree": None}, {"pswpout": None},
    {"MemTotal": 0}, {"MemAvailable": 9_000_000_000}, {"MemFree": True},
])
def test_baseline_refuses_low_missing_or_invalid_values(operator, tmp_path, update):
    backend = FakeBackend()
    backend.telemetry = lambda: update
    result = operator.execute("check9", backend, tmp_path)
    assert result["check9"]["status"] == "refused"
    assert hardware_children(backend) == []
    for key, value in update.items():
        if type(value) is not int or value < 0:
            assert result["inspection"]["memory"][key] is None


def test_bounded_check9_is_separate_from_u21_and_not_full_u18_acceptance(operator, tmp_path):
    backend = FakeBackend()
    result = operator.execute("check9", backend, tmp_path)
    assert result["check9"]["status"] == "bounded_smoke_complete"
    assert result["check9"]["server_comparison"] == result["check9"]["u18_acceptance"] == "PENDING"
    assert result["u21"]["status"] == "PENDING"
    children = hardware_children(backend)
    assert len(children) == 2
    assert all("--cap-mib" in child.argv and "--chunk-mib" in child.argv for child in children)
    assert result["check9"]["device"]["allocated_bytes"] == 256 * (1 << 20)


@pytest.mark.parametrize("failure", ["child_failed", "timeout", "cleanup_failed", "malformed", "output_limit"])
def test_probe_failure_prevents_dependent_managed_probe(operator, tmp_path, failure):
    backend = FakeBackend()
    backend.hardware_plan = {
        "child_failed": {"returncode": 1},
        "timeout": {"duration": 1000},
        "cleanup_failed": {"duration": 1000, "stubborn": True},
        "malformed": {"output": b"secret invalid output"},
        "output_limit": {"output": b"x" * (operator.OUTPUT_LIMIT + 1)},
    }[failure]
    result = operator.execute("check9", backend, tmp_path)
    assert result["check9"]["status"] == "inconclusive"
    assert len(hardware_children(backend)) == 1
    assert "managed" not in result["check9"] and result["u21"]["status"] == "PENDING"
    assert "secret" not in json.dumps(result)
    if failure != "malformed":
        assert result["check9"]["device"]["status"] == failure


@pytest.mark.parametrize("field,value,reason", [
    ("MemAvailable", 3_000_000_000, "sampled_pressure_stop"),
    ("MemFree", (1 << 30) - 1, "free_or_available_stop"),
    ("pswpout", 1, "swap_counter_change"), ("MemFree", None, "telemetry_unavailable"),
])
def test_live_guard_aborts_owned_probe_and_stops_next_phase(operator, tmp_path, field, value, reason):
    backend = FakeBackend()
    backend.hardware_plan = {"duration": 100}
    backend.telemetry = lambda: {field: value} if hardware_children(backend) else {}
    result = operator.execute("check9", backend, tmp_path)
    assert result["check9"]["device"]["status"] == reason
    child = hardware_children(backend)[0]
    assert child.signals == [signal.SIGTERM, signal.SIGKILL] and child.closed
    assert "managed" not in result["check9"]
    if field == "MemAvailable":
        assert result["check9"]["sampled_peak_pressure_bytes"] == 5_000_000_000
    trigger = result["check9"]["guard_trigger"]
    assert trigger["stop"] == reason and trigger["guard_jsonl_line"] == result["check9"]["samples"]
    assert field in str(trigger["conditions"][0].get("field", trigger["conditions"][0].get("fields")))


def reference_memory_problem(sample, baseline=None, projected=0):
    """The guard's decision before session 39, kept verbatim to prove the refactor changed nothing."""
    if any(type(sample.get(key)) is not int or sample[key] < 0 for key in ("MemTotal", "MemAvailable", "MemFree",
                                                                          "pswpin", "pswpout")):
        return "telemetry_unavailable"
    total = sample["MemTotal"]
    if total <= 0 or sample["MemAvailable"] > total or sample["MemFree"] > total:
        return "telemetry_invalid"
    if total - sample["MemAvailable"] + projected >= 4_800_000_000:
        return "sampled_pressure_stop"
    if sample["MemFree"] - projected < (1 << 30) or sample["MemAvailable"] - projected < (2 << 30):
        return "free_or_available_stop"
    if baseline and any(sample[key] != baseline[key] for key in ("pswpin", "pswpout")):
        return "swap_counter_change"
    return None


def test_guard_decisions_and_thresholds_are_unchanged_on_a_boundary_grid(operator):
    assert (operator.PRESSURE_STOP, operator.FREE_FLOOR, operator.AVAILABLE_FLOOR) == (4_800_000_000, 1 << 30, 2 << 30)
    baseline = {"MemTotal": 8_000_000_000, "MemAvailable": 7_000_000_000, "MemFree": 6_000_000_000,
                "pswpin": 0, "pswpout": 0}
    total = 8_000_000_000
    frees = [None, -1, 0, (1 << 30) - 1, 1 << 30, 3_000_000_000, total + 1]
    availables = [(2 << 30) - 1, 2 << 30, total - 4_800_000_001, total - 4_800_000_000, total - 4_799_999_999, total + 1]
    count = 0
    for free in frees:
        for available in availables:
            for swap in (0, 1):
                for projected in (0, 32 << 20):
                    sample = {"MemTotal": total, "MemFree": free, "MemAvailable": available, "pswpin": 0,
                              "pswpout": swap}
                    for base in (None, baseline):
                        expected = reference_memory_problem(sample, base, projected)
                        assert operator.memory_problem(sample, base, projected) == expected
                        met = operator.memory_conditions(sample, base, projected)
                        assert (met[0]["stop"] if met else None) == expected
                        count += 1
    assert count == len(frees) * len(availables) * 2 * 2 * 2


@pytest.mark.parametrize("update,conditions", [
    ({"MemFree": 1_067_401_216, "MemAvailable": 3_682_267_136},  # step 4 run 1's stopping sample
     [("free_or_available_stop", "MemFree", "<", 1 << 30, 1_067_401_216)]),
    ({"MemAvailable": (2 << 30) - 1},
     [("sampled_pressure_stop", "MemTotal - MemAvailable", ">=", 4_800_000_000, 8_000_000_000 - (2 << 30) + 1),
      ("free_or_available_stop", "MemAvailable", "<", 2 << 30, (2 << 30) - 1)]),
    ({"MemTotal": 4_000_000_000, "MemAvailable": (2 << 30) - 1, "MemFree": (1 << 30) - 1},
     [("free_or_available_stop", "MemFree", "<", 1 << 30, (1 << 30) - 1),
      ("free_or_available_stop", "MemAvailable", "<", 2 << 30, (2 << 30) - 1)]),
    ({"pswpin": 3}, [("swap_counter_change", "pswpin", "!=", 0, 3)]),
])
def test_the_guard_trigger_names_every_condition_met_with_its_threshold_and_value(operator, update, conditions):
    backend = FakeBackend()
    lines = []
    guard = operator.PressureGuard(backend.sample(), emit=lines.append)
    for _ in range(3):
        backend.now += 0.2
        assert guard.observe(backend.sample()) is None
    backend.now += 0.2
    stopping = {**backend.sample(), **update}
    stop = guard.observe(stopping)
    trigger = guard.trigger
    assert stop == trigger["stop"] == conditions[0][0]
    assert [(c["stop"], c["field"], c["comparison"], c["threshold"], c["value"]) for c in trigger["conditions"]] == conditions
    assert all(c["condition"] == f"{c['field']} {c['comparison']} {c['threshold']}" for c in trigger["conditions"])
    assert trigger["t_mono"] == stopping["t_mono"] and trigger["guard_jsonl_line"] == 4 == len(lines)
    assert trigger["sample"] == lines[-1] == operator.safe_sample(stopping)
    backend.now += 0.2
    guard.observe({**backend.sample(), "MemAvailable": 1})
    assert guard.trigger is trigger  # the first stopping sample is kept


@pytest.mark.parametrize("sample_update,stop,condition", [
    ({"t_mono": 0.801}, "sampling_gap", "t_mono interval > 0.5"),
    ({"t_mono": 0.1}, "sampling_gap", "t_mono interval < 0"),
    ({"t_mono": float("nan")}, "clock_unavailable", "t_mono missing or not finite"),
    ({"MemFree": None, "pswpout": True}, "telemetry_unavailable", "required field missing or invalid"),
])
def test_clock_and_telemetry_stops_are_attributed_too(operator, sample_update, stop, condition):
    backend = FakeBackend()
    guard = operator.PressureGuard(backend.sample())
    backend.now = 0.2
    assert guard.observe(backend.sample()) is None
    assert guard.observe({**backend.sample(), **sample_update}) == stop
    assert guard.trigger["stop"] == stop and guard.trigger["conditions"][0]["condition"] == condition
    if stop == "telemetry_unavailable":
        assert guard.trigger["conditions"][0]["fields"] == ["MemFree", "pswpout"]
    if stop == "sampling_gap":
        assert guard.trigger["conditions"][0]["value"] == round(sample_update["t_mono"] - 0.2, 6)


def test_missing_or_invalid_attribution_fields_are_recorded_unavailable_and_never_stop_the_guard(operator):
    assert {"AnonPages", "Mapped", "Active(file)", "Inactive(file)"} <= set(operator.MEMORY_KEYS)
    backend = FakeBackend()
    lines = []
    guard = operator.PressureGuard(backend.sample(), emit=lines.append)
    for update in ({}, {"AnonPages": -1, "Mapped": True, "Active(file)": 1.5, "Inactive(file)": "7"},
                   {"AnonPages": 1024, "Mapped": 2048, "Active(file)": 0, "Inactive(file)": 4096}):
        backend.now += 0.2
        assert guard.observe({**backend.sample(), **update}) is None
    assert guard.trigger is None
    for line in lines[:2]:
        assert all(line[key] is None for key in ("AnonPages", "Mapped", "Active(file)", "Inactive(file)"))
    assert [lines[2][key] for key in ("AnonPages", "Mapped", "Active(file)", "Inactive(file)")] == [1024, 2048, 0, 4096]
    assert lines[2]["pressure_bytes"] == 1_000_000_000  # the guard metric is unchanged


def test_stderr_is_captured_bounded_only_when_asked_and_never_otherwise(operator):
    backend = FakeBackend()
    backend.overrides["journal"] = {"returncode": 1, "errors": b"x" * 10_000}
    runner = operator.ProcessRunner(backend)
    captured = runner.run(["/usr/bin/journalctl"], 1.0, capture_stderr=True)
    plain = runner.run(["/usr/bin/journalctl"], 1.0)
    assert backend.captured == [True, False]
    assert captured.status == plain.status == "child_failed" and captured.returncode == 1
    assert captured.errors == b"x" * operator.ERROR_LIMIT and plain.errors == b""
    assert "errors" not in captured.diagnostic()


def test_a_real_child_stderr_goes_to_an_unlinked_file_only_when_captured(operator):
    script = "import sys; sys.stdout.write('out'); sys.stderr.write('e' * 10000); sys.exit(3)"
    runner = operator.ProcessRunner(operator.SystemBackend())
    captured = runner.run([sys.executable, "-c", script], 10.0, capture_stderr=True)
    plain = runner.run([sys.executable, "-c", script], 10.0)
    assert (captured.status, captured.returncode, captured.output) == ("child_failed", 3, b"out")
    assert captured.errors == b"e" * operator.ERROR_LIMIT and captured.cleanup_clear
    assert (plain.status, plain.returncode, plain.output, plain.errors) == ("child_failed", 3, b"out", b"")


def test_sampling_gap_and_unavailable_clock_are_fail_closed(operator):
    backend = FakeBackend()
    guard = operator.PressureGuard(backend.sample())
    assert guard.observe(backend.sample()) is None
    backend.now = 0.501
    assert guard.observe(backend.sample()) == "sampling_gap"
    assert guard.observe({**backend.sample(), "t_mono": float("nan")}) == "clock_unavailable"


def test_timeout_and_descendant_cleanup_target_only_created_children(operator):
    backend = FakeBackend()
    backend.hardware_plan = {"duration": 100}
    unrelated = FakeChild(backend, ["unrelated"], duration=1000)
    child = operator.ProcessRunner(backend).run(["fake-hardware"], 0.4, guard=operator.PressureGuard(backend.sample()))
    assert child.status == "timeout" and child.cleanup_clear
    assert unrelated.signals == [] and unrelated.exists()
    assert backend.now < 11
    backend.hardware_plan = {"descendants": True}
    child = operator.ProcessRunner(backend).run(["parent-exits-first"], 1.0)
    assert child.status == "completed" and child.cleanup_clear
    assert backend.children[-1].signals == [signal.SIGTERM, signal.SIGKILL]


def test_interrupt_and_provider_error_cleanup_owned_processes(operator):
    backend = FakeBackend()
    backend.hardware_plan = {"duration": 100}
    runner = operator.ProcessRunner(backend, interrupted=lambda: backend.now >= 0.2)
    child = runner.run(["fake"], 1, guard=operator.PressureGuard(backend.sample()))
    assert child.status == "interrupted" and child.cleanup_clear
    backend = FakeBackend()
    guard = operator.PressureGuard(backend.sample())
    backend.sample = lambda: (_ for _ in ()).throw(RuntimeError("private failure"))
    backend.hardware_plan = {"duration": 100}
    child = operator.ProcessRunner(backend).run(["fake"], 1, guard=guard)
    assert child.status == "process_error" and child.cleanup_clear
    assert b"private" not in child.output


@pytest.mark.parametrize("case", [
    "missing", "unconfirmed", "wrong_boot", "wrong_revision", "failed", "symlink", "public_mode",
    "missing_phase", "bad_cleanup",
])
def test_u21_requires_review_and_valid_private_same_boot_check9_report(operator, tmp_path, case):
    backend = FakeBackend()
    report_path = None if case == "missing" else write_check9(tmp_path, operator)
    if report_path and case in ("wrong_boot", "wrong_revision", "failed", "missing_phase", "bad_cleanup"):
        previous = json.loads(report_path.read_text())
        if case == "failed":
            previous["check9"]["status"] = "inconclusive"
        elif case == "missing_phase":
            previous["check9"].pop("managed")
        elif case == "bad_cleanup":
            previous["check9"]["device"]["cleanup_clear"] = False
        else:
            previous["inspection"]["boot_id" if case == "wrong_boot" else "repository_commit"] = "wrong"
        report_path.write_text(json.dumps(previous))
    if case == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(report_path)
        report_path = alias
    if case == "public_mode":
        report_path.chmod(0o644)
    result = operator.execute("u21", backend, tmp_path, check9_report=report_path, confirm_u21=case != "unconfirmed")
    assert result["u21"]["status"] == "refused"
    assert hardware_children(backend) == []


def test_u21_reuses_synthetic_profiler_without_cache_or_service_changes(operator, tmp_path):
    path = write_check9(tmp_path, operator)
    backend = FakeBackend()
    result = operator.execute("u21", backend, tmp_path, check9_report=path, confirm_u21=True)
    child = hardware_children(backend)[0]
    assert result["u21"]["status"] == "completed" and result["hardware_acceptance"] == "PENDING"
    assert result["check9"]["status"] == "PENDING"
    assert "--no-evict" in child.argv and "--sanitized-logs" in child.argv and "--clip" not in child.argv
    assert child.argv[child.argv.index("--steady-s") + 1] == "120"
    assert "secret" not in json.dumps(result)
    assert backend.now < 376


def test_kernel_unknown_and_duplicate_candidate_lines_do_not_prove_oom(operator):
    result = operator.ChildResult("completed", 0, b"", True)
    assert operator.kernel_counts(result)["oom_candidates"] is None
    result.output = b'{"MESSAGE":"killed process"}\n{"MESSAGE":"killed process"}'
    assert operator.kernel_counts(result)["oom_candidates"] == 2
    result.output = b'{"MESSAGE":"secret"}\ninvalid'
    assert operator.kernel_counts(result)["status"] == "unavailable"


def test_sanitized_server_and_workload_logs_never_retain_raw_text(operator, tmp_path):
    profile = operator.profile
    private = "secret://user:password@camera/ private model description"
    output = io.BytesIO()
    server = SimpleNamespace(stdout=io.BytesIO((
        private + "\noffloaded 17/17 layers to GPU " + private +
        "\nCLIP using CUDA0 " + private + "\nCUDA0 buffer size = 10.50 MiB " + private + "\n"
    ).encode()))
    profile.pump_llama_diagnostics(server, output)
    assert output.getvalue() == (
        b"offloaded 17/17 layers to GPU\nCLIP using CUDA0\ndiagnostic: buffer size = 10.50 MiB\n"
    )
    lines = [
        private, "@@EVENT malformed " + private,
        "@@EVENT " + json.dumps({"event": "workload_stats", "scene": {"valid_reports": 1, "summary": private}}),
        "@@EVENT " + json.dumps({"event": "phase", "name": private}),
        "@@EVENT " + json.dumps({"event": "unexpected", "detail": private}),
    ]
    events = []
    sink = SimpleNamespace(add=lambda *args, **fields: events.append(fields))
    proc = SimpleNamespace(stdout=io.StringIO("\n".join(lines) + "\n"))
    profile.pump_workload(proc, tmp_path, sink, lambda *args, **kwargs: pytest.fail("unsafe phase"), sanitized=True)
    assert events == [{"scene": {"valid_reports": 1}}]
    assert (tmp_path / "workload.log").read_text() == ""
    assert "secret" not in json.dumps(events)


def test_cli_defaults_to_inspection_and_writes_private_results_only(operator, tmp_path, monkeypatch, capsys):
    output = tmp_path / "sentinel-operator-main"
    output.mkdir(mode=0o700)
    def mkdtemp(**kwargs):
        assert kwargs["dir"] == Path("/tmp")
        return str(output)

    monkeypatch.setattr(operator.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(operator.signal, "signal", lambda *args: None)
    previous = operator.os.umask(0o077)
    backend = FakeBackend()
    try:
        assert operator.main([], backend=backend) == 0
    finally:
        operator.os.umask(previous)
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "inspection" and hardware_children(backend) == []
    assert result["sampled_stop_is_guaranteed_cap"] is False
    path = output / "result.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == result
    assert sorted(path.name for path in output.iterdir()) == ["result.json"]


@pytest.mark.parametrize("failure", ["timeout", "cleanup_failed", "child_failed", "pressure_stop"])
def test_u21_timeout_refusal_and_cleanup_stay_separate_from_check9(operator, tmp_path, failure):
    path = write_check9(tmp_path, operator)
    backend = FakeBackend()
    backend.hardware_plan = {
        "timeout": {"duration": 1000}, "cleanup_failed": {"duration": 1000, "stubborn": True},
        "child_failed": {"returncode": 1}, "pressure_stop": {"duration": 1000},
    }[failure]
    if failure == "pressure_stop":
        backend.telemetry = lambda: {"MemAvailable": 3_000_000_000} if hardware_children(backend) else {}
    result = operator.execute("u21", backend, tmp_path, check9_report=path, confirm_u21=True)
    expected = "sampled_pressure_stop" if failure == "pressure_stop" else failure
    assert result["u21"]["status"] == expected
    assert result["check9"]["status"] == "PENDING"
    assert len(hardware_children(backend)) == 1
    assert backend.now < 377


def test_preexisting_interruption_starts_no_subprocesses(operator):
    backend = FakeBackend()
    result = operator.ProcessRunner(backend, interrupted=lambda: True).run(["fake"], 1)
    assert result.status == "interrupted" and backend.children == []


def test_pipe_setup_failure_still_cleans_up_the_owned_process(operator, monkeypatch):
    backend = FakeBackend()
    child = backend.spawn(["fake"], None)
    child.end = 1000
    monkeypatch.setattr(backend, "spawn", lambda argv, env: child)

    def failed_read():
        raise OSError("private setup detail")

    monkeypatch.setattr(child, "read", failed_read)
    result = operator.ProcessRunner(backend).run(["fake"], 1)
    assert result.status == "process_error" and result.cleanup_clear
    assert child.closed and child.signals == [signal.SIGTERM, signal.SIGKILL]
    assert "private" not in json.dumps(result.diagnostic())


def test_swap_change_between_probes_prevents_the_dependent_process(operator, tmp_path):
    backend = FakeBackend()
    backend.telemetry = lambda: {"pswpout": 1} if (
        hardware_children(backend) and hardware_children(backend)[0].poll() is not None
    ) else {}
    result = operator.execute("check9", backend, tmp_path)
    assert result["check9"]["status"] in ("refused", "inconclusive")
    assert "managed" not in result["check9"]
    assert len(hardware_children(backend)) == 1


@pytest.fixture
def pva_identity(operator, tmp_path, monkeypatch):
    fragment = tmp_path / "unit"
    daemon = tmp_path / "daemon"
    checksums = tmp_path / "md5sums"
    fragment.write_bytes(b"synthetic packaged unit")
    daemon.write_bytes(b"synthetic packaged launcher")
    checksums.write_text("\n".join(
        hashlib.md5(path.read_bytes()).hexdigest() + "  " + str(path).lstrip("/") for path in (fragment, daemon)
    ))
    for name, path in (("PVA_FRAGMENT", fragment), ("PVA_DAEMON", daemon), ("PVA_CHECKSUMS", checksums)):
        monkeypatch.setattr(operator, name, str(path))
    original_stat = Path.stat

    def root_stat(path, **kwargs):
        info = original_stat(path, **kwargs)
        if path in (fragment, daemon, checksums):
            return SimpleNamespace(st_uid=0, st_mode=0o100644, st_size=info.st_size)
        return info

    monkeypatch.setattr(Path, "stat", root_stat)
    metadata = {
        "Id": operator.PVA_UNIT, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
        "MainPID": "4214", "ControlGroup": operator.PVA_CGROUP, "FragmentPath": str(fragment),
        "DropInPaths": "", "User": "", "ExecStart": "{ path=" + str(daemon) + " ; argv[]=private synthetic argument ; }",
    }
    calls = []
    failures = {}

    def run(argv, timeout):
        calls.append((argv, timeout))
        if "systemctl" in argv[0]:
            name = "service"
            output = "\n".join(key + "=" + value for key, value in metadata.items()).encode()
        elif "--show" in argv:
            name, output = "installed", b"install ok installed\n"
        else:
            name, output = "owned", ("pva-allow-2: " + str(fragment) + "\npva-allow-2: " + str(daemon) + "\n").encode()
        return operator.ChildResult(failures.get(name, "completed"), 0, output, True)

    return SimpleNamespace(runner=SimpleNamespace(run=run), metadata=metadata, calls=calls,
                           failures=failures, daemon=daemon, checksums=checksums)


def test_pva_identity_requires_installed_package_launch_metadata_and_checksums(operator, pva_identity):
    assert operator.verified_pva_main_pid(pva_identity.runner) == 4214
    assert len(pva_identity.calls) == 3 and all(timeout == 3 for _, timeout in pva_identity.calls)
    assert all(argv[0] in ("/usr/bin/systemctl", "/usr/bin/dpkg-query") for argv, _ in pva_identity.calls)


@pytest.mark.parametrize("field,value", [
    ("Id", "other.service"), ("LoadState", "not-found"), ("ActiveState", "inactive"),
    ("SubState", "dead"), ("MainPID", "0"), ("ControlGroup", "/user.slice/nvidia-pva-allowd.service"),
    ("FragmentPath", "/tmp/untrusted.service"), ("DropInPaths", "/etc/override.conf"),
    ("User", "maintainer"), ("ExecStart", "{ path=/tmp/other.py ; argv[]=hidden ; }"),
])
def test_unverified_pva_metadata_is_not_exempt(operator, pva_identity, field, value):
    pva_identity.metadata[field] = value
    assert operator.verified_pva_main_pid(pva_identity.runner) is None


@pytest.mark.parametrize("failure", ["service", "installed", "owned", "changed_launcher", "bad_checksum_record"])
def test_pva_identity_failures_keep_python_unclassified(operator, pva_identity, failure):
    if failure == "changed_launcher":
        pva_identity.daemon.write_bytes(b"changed launcher")
    elif failure == "bad_checksum_record":
        pva_identity.checksums.write_text("malformed")
    else:
        pva_identity.failures[failure] = "timeout"
    assert operator.verified_pva_main_pid(pva_identity.runner) is None


class FakeProc:
    def __init__(self, pid, cgroup, uid=0):
        self.name = str(pid)
        self.cgroup = cgroup
        self.uid = uid

    def stat(self):
        return SimpleNamespace(st_uid=self.uid)

    def __truediv__(self, name):
        assert name in ("comm", "cgroup")
        return SimpleNamespace(read_text=lambda: "python3\n" if name == "comm" else self.cgroup)


@pytest.mark.parametrize("cgroup,uid,expected", [
    ("0::/system.slice/nvidia-pva-allowd.service\n", 0, True),
    ("0::/system.slice/nvidia-pva-allowd.service\n", 1000, False),
    ("0::/system.slice/nvidia-pva-allowd.service/child\n", 0, False),
    ("0::/system.slice/other.service\n", 0, False),
])
def test_only_exact_root_pva_cgroup_is_a_candidate(operator, cgroup, uid, expected):
    assert operator.pva_cgroup_member(FakeProc(4214, cgroup, uid)) is expected


@pytest.mark.parametrize("verified_pid,unknown_present", [(4214, False), (4214, True), (None, False), (9999, False)])
def test_context_exempts_only_verified_pva_main_pid_and_preserves_memory_and_other_refusals(
    operator, monkeypatch, tmp_path, verified_pid, unknown_present,
):
    entries = [FakeProc(4214, "0::" + operator.PVA_CGROUP + "\n")]
    if unknown_present:
        entries.append(FakeProc(4215, "0::/system.slice/unrelated.service\n"))
    asset = tmp_path / "asset"
    asset.write_text("synthetic asset")
    real_path = Path

    def fake_path(value):
        if str(value) == "/proc":
            return SimpleNamespace(iterdir=lambda: iter(entries))
        if str(value) == "/proc/sys/kernel/random/boot_id":
            return SimpleNamespace(read_text=lambda: BOOT)
        if str(value) == operator.profile.L4T_LIBCUDA:
            return asset
        return real_path(value)

    monkeypatch.setattr(operator, "Path", fake_path)
    monkeypatch.setattr(operator.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(operator, "verified_pva_main_pid", lambda runner: verified_pid)
    monkeypatch.setattr(operator.profile, "parse_args", lambda argv: SimpleNamespace(python=asset, llama=asset))
    monkeypatch.setattr(operator.profile, "model_files", lambda args: {})
    monkeypatch.setattr(operator.profile, "port_in_use", lambda port: False)
    context = operator.SystemBackend().context()
    expected_unknown = int(verified_pid != 4214) + int(unknown_present)
    assert context["workloads"]["python_unclassified"]["count"] == expected_unknown
    assert bool(context["known_system_services"]) is (verified_pid == 4214)
    if verified_pid == 4214:
        assert context["known_system_services"][operator.PVA_UNIT]["pids"] == [4214]
    backend = FakeBackend()
    backend.context_overrides = context
    result = operator.execute("check9", backend, tmp_path)
    assert result["inspection"]["memory"]["pressure_bytes"] == 1_000_000_000
    assert ("python_unclassified" in result["inspection"]["workload_refusals"]) is bool(expected_unknown)
    if expected_unknown:
        assert hardware_children(backend) == []
    assert "private synthetic argument" not in json.dumps(result)


def all_argv_text(backend):
    return " ".join(" ".join(child.argv) for child in backend.children)


@pytest.mark.parametrize("case", ["missing_report", "unconfirmed", "no_arm", "wrong_boot", "failed_check9"])
def test_s1_requires_arm_confirmation_and_a_valid_same_boot_check9(operator, tmp_path, case):
    backend = FakeBackend()
    path = None if case == "missing_report" else write_check9(tmp_path, operator)
    if case in ("wrong_boot", "failed_check9"):
        previous = json.loads(path.read_text())
        if case == "wrong_boot":
            previous["inspection"]["boot_id"] = "previous-boot"
        else:
            previous["check9"]["status"] = "inconclusive"
        path.write_text(json.dumps(previous))
    result = operator.execute("s1", backend, tmp_path, check9_report=path, confirm_s1=case != "unconfirmed",
                              s1_arm=None if case == "no_arm" else "a")
    assert result["s1"]["status"] == "refused"
    assert hardware_children(backend) == []
    assert result["u21"]["status"] == result["check9"]["status"] == "PENDING"


def test_s1_arms_differ_only_in_cache_ram_and_reuse_the_u21_guard(operator, tmp_path):
    path = write_check9(tmp_path, operator)
    argv = {}
    for arm in ("a", "b"):
        backend = FakeBackend()
        output = tmp_path / ("out-" + arm)
        output.mkdir()
        result = operator.execute("s1", backend, output, check9_report=path, confirm_s1=True, s1_arm=arm)
        (child,) = hardware_children(backend)
        argv[arm] = child.argv[child.argv.index("--out") + 2:]
        assert result["s1"]["status"] == "completed" and result["hardware_acceptance"] == "PENDING"
        assert result["s1"]["arm"] == arm and result["s1"]["operator_prerequisites_confirmed"] is True
        assert result["s1"]["llama_cache_ram_mib"] == {"a": None, "b": 0}[arm]
        assert result["s1"]["check9_report_dir"] == "sentinel-operator-fake"
        assert result["s1"]["profile"] == {"status": "unavailable"}  # the fake child writes no profile
        assert result["u21"]["status"] == "PENDING" and "post_exit" in result["s1"]
        assert (output / "guard.jsonl").exists() and backend.now < 376
        assert not any(word in all_argv_text(backend) for word in ("drop_caches", "sysctl", "sudo", "--clip"))
        assert "secret" not in json.dumps(result)
    assert argv["b"] == argv["a"] + ["--llama-cache-ram", "0"]
    for flag in ("--scene-only", "--no-evict", "--sanitized-logs"):
        assert flag in argv["a"]
    assert argv["a"][argv["a"].index("--steady-s") + 1] == "180"
    assert argv["a"][argv["a"].index("--min-free-gb") + 1] == "3.5"


@pytest.mark.parametrize("failure", ["pressure_stop", "free_floor", "timeout", "cleanup_failed"])
def test_s1_guard_stops_timeouts_and_cleanup_match_u21(operator, tmp_path, failure):
    path = write_check9(tmp_path, operator)
    backend = FakeBackend()
    backend.hardware_plan = {"timeout": {"duration": 1000}, "cleanup_failed": {"duration": 1000, "stubborn": True}}.get(
        failure, {"duration": 1000})
    if failure in ("pressure_stop", "free_floor"):
        update = {"MemAvailable": 3_000_000_000} if failure == "pressure_stop" else {"MemFree": (1 << 30) - 1}
        backend.telemetry = lambda: update if hardware_children(backend) else {}
    result = operator.execute("s1", backend, tmp_path, check9_report=path, confirm_s1=True, s1_arm="b")
    expected = {"pressure_stop": "sampled_pressure_stop", "free_floor": "free_or_available_stop"}.get(failure, failure)
    assert result["s1"]["status"] == expected
    assert len(hardware_children(backend)) == 1
    assert backend.now < 377


def test_s1_result_excerpt_keeps_numbers_and_fixed_labels_only(operator, tmp_path):
    assert operator.s1_profile_excerpt(tmp_path) == {"status": "unavailable"}
    run = tmp_path / "demo-profile-20261004T000000Z"
    run.mkdir()
    (run / "profile.json").write_text(json.dumps({
        "status": "aborted: MemFree is 1.00 GB private detail",
        "steady_trend": {"seconds": 180.0, "used": {"first": 1, "last": 2, "slope_bytes_per_min": 3, "n": 900}},
        "prompt_cache": {"startup": {"enabled": True, "limit_mib": 8192}, "state_updates": 53,
                         "last": {"prompts": 53, "size_mib": 331.5}, "note": "private"},
        "scene_progress_last": {"phase": "steady", "requests": 53, "utc": "2026-10-04T00:00:00Z",
                                "source": "workload", "errors": {"http": 0, "HTTP 500": 1}},
        "unload": {"workload_residual_bytes": -5},
        "components": {"scene": {"cold_load_seconds": 4.1, "input_limit": {"image": "480x360 private"}}},
        "workload_steady": {"scene": {"completed": 45, "accuracy": "not evaluated", "latency_ms": {"p50": 900.0}}},
        "provenance": {"llama_server": {"cache_ram_mib": 0, "version": ["private build text"]}},
    }))
    excerpt = operator.s1_profile_excerpt(tmp_path)
    assert excerpt == {
        "status": "aborted",
        "steady_trend": {"seconds": 180.0, "used": {"first": 1, "last": 2, "slope_bytes_per_min": 3, "n": 900}},
        "prompt_cache": {"startup": {"enabled": True, "limit_mib": 8192}, "state_updates": 53,
                         "last": {"prompts": 53, "size_mib": 331.5}},
        "scene_progress_last": {"phase": "steady", "requests": 53, "errors": {"http": 0}},
        "unload": {"workload_residual_bytes": -5},
        "scene_component": {"cold_load_seconds": 4.1, "input_limit": {}},
        "workload_steady_scene": {"completed": 45, "latency_ms": {"p50": 900.0}},
        "llama_cache_ram_mib": 0,
    }
    (tmp_path / "demo-profile-second").mkdir()
    (tmp_path / "demo-profile-second" / "profile.json").write_text("{}")
    assert operator.s1_profile_excerpt(tmp_path) == {"status": "unavailable"}


def test_latest_check9_report_uses_the_newest_check9_and_fails_closed(operator, tmp_path, monkeypatch):
    import os

    older = write_check9(tmp_path, operator)
    newer_dir = tmp_path / "sentinel-operator-newer"
    newer_dir.mkdir(mode=0o700)
    failed = json.loads(older.read_text())
    failed["check9"]["status"] = "inconclusive"
    newer = newer_dir / "result.json"
    newer.write_text(json.dumps(failed))
    newer.chmod(0o600)
    unrelated_dir = tmp_path / "sentinel-operator-inspection"
    unrelated_dir.mkdir(mode=0o700)
    (unrelated_dir / "result.json").write_text(json.dumps({"mode": "inspection"}))
    (tmp_path / "sentinel-operator-garbage").mkdir()
    (tmp_path / "sentinel-operator-garbage" / "result.json").write_text("not json")
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))
    os.utime(unrelated_dir / "result.json", ns=(3, 3))
    assert operator.latest_check9_report(tmp_path) == newer
    backend = FakeBackend()
    result = operator.execute("s1", backend, tmp_path, check9_report=newer, confirm_s1=True, s1_arm="a")
    assert result["s1"]["status"] == "refused" and "check9_report_mismatch" in result["s1"]["refusals"]
    assert hardware_children(backend) == []  # an older successful Check 9 is not used instead
    os.utime(older, ns=(4, 4))
    assert operator.latest_check9_report(tmp_path) == older
    assert operator.latest_check9_report(tmp_path / "empty") is None


def test_cli_s1_uses_latest_check9_and_records_the_declared_cache_drop(operator, tmp_path, monkeypatch, capsys):
    path = write_check9(tmp_path, operator)
    output = tmp_path / "sentinel-operator-main"
    output.mkdir(mode=0o700)
    monkeypatch.setattr(operator.tempfile, "mkdtemp", lambda **kwargs: str(output))
    monkeypatch.setattr(operator.signal, "signal", lambda *args: None)
    monkeypatch.setattr(operator, "OUTPUT_ROOT", tmp_path)
    previous = operator.os.umask(0o077)
    backend = FakeBackend()
    try:
        code = operator.main(["--execute-workload", "s1", "--s1-arm", "b", "--latest-check9-report",
                              "--confirm-s1-prerequisites", "--operator-dropped-caches"], backend=backend)
    finally:
        operator.os.umask(previous)
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["s1"]["status"] == "completed"
    assert result["s1"]["check9_report_dir"] == path.parent.name
    assert result["preparation"]["drop_caches"] == "operator_declared"
    assert "never run or verified by this workflow" in result["preparation"]["procedure"]
    assert not any(word in all_argv_text(backend) for word in ("drop_caches", "sysctl", "sudo"))
    with pytest.raises(SystemExit):
        operator.main(["--execute-workload", "s1", "--latest-check9-report", "--check9-report", str(path)],
                      backend=FakeBackend())


def test_cache_drop_is_not_declared_by_default(operator, tmp_path):
    result = operator.execute(None, FakeBackend(), tmp_path)
    assert result["preparation"]["drop_caches"] == "not_declared"
    assert result["s1"]["status"] == "PENDING"
