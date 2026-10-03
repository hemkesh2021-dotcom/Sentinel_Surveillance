from __future__ import annotations

import importlib
import io
import json
import signal
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
    def __init__(self, backend, argv, *, duration=0.0, returncode=0, output=b"", stubborn=False, descendants=False):
        self.backend = backend
        self.argv = argv
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

    def spawn(self, argv, env):
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
