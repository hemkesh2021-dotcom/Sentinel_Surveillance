"""D-1: the scene server process: loopback-only launch, D27 placement guard, bounded stop (no real server)."""

from __future__ import annotations

import importlib.util
import io
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from sentinel.scene.llama_server import llama_server_command
from sentinel.scene.server import LlamaServerProcess, ServerState, loopback_problem

ROOT = Path(__file__).resolve().parents[2]
L4T = "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"
GOOD_OUTPUT = [
    b"load_tensors: offloaded 17/17 layers to GPU\n",
    b"srv  load_model: a request body with a private prompt\n",
    b"clip_model_loader: CLIP using CUDA0 backend\n",
]


class FakeProcess:
    def __init__(self, lines: list[bytes], *, exit_code: int | None = None, ignores_terminate: bool = False) -> None:
        self.stdout = io.BytesIO(b"".join(lines))
        self.pid = 4242
        self.returncode = exit_code
        self.ignores_terminate = ignores_terminate
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("TERM")
        if not self.ignores_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.signals.append("KILL")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout or 0)
        return self.returncode


class FakePopen:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> FakeProcess:
        self.calls.append((argv, kwargs))
        return self.process


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in ("llama-server", "model.gguf", "mmproj.gguf")}
    for path in paths.values():
        path.write_bytes(b"x")
    return paths


def server(files: dict[str, Path], popen: FakePopen, *, healthy_after: int = 2, libcuda: list[str] | None = None,
           in_use: bool = False, environ: dict[str, str] | None = None, command=llama_server_command,
           libraries: list[str] | None = None) -> LlamaServerProcess:
    polls = {"n": 0}
    clock = Clock()

    def health(port: int) -> bool:
        polls["n"] += 1
        return polls["n"] > healthy_after

    return LlamaServerProcess(
        files["llama-server"], files["model.gguf"], files["mmproj.gguf"], 18081,
        environ=environ if environ is not None else {"PATH": "/usr/bin"},
        command=command, popen=popen, health=health, in_use=lambda port: in_use,
        libcuda=lambda pid: [L4T] if libcuda is None else libcuda,
        libraries=lambda pid: [str(files["llama-server"].parent / "libllama.so.0.0.8932")] if libraries is None else libraries,
        sleep=clock.sleep, monotonic=clock.monotonic,
    )


# ---------------------------------------------------------------- loopback only


def test_the_demo_command_passes_the_loopback_check() -> None:
    argv = llama_server_command(Path("/bin/llama-server"), Path("m"), Path("p"), 18081)
    assert loopback_problem(argv) is None


@pytest.mark.parametrize(
    ("argv", "label"),
    [
        (["llama-server", "--host", "0.0.0.0", "--port", "18081"], "non_loopback_bind"),
        (["llama-server", "--host=0.0.0.0"], "non_loopback_bind"),
        (["llama-server", "--host", "::"], "non_loopback_bind"),
        (["llama-server", "--host", "192.168.1.20"], "non_loopback_bind"),
        (["llama-server", "--host", "localhost"], "non_loopback_bind"),  # resolution is not trusted
        (["llama-server", "--host", "/tmp/llama.sock"], "non_loopback_bind"),  # a Unix socket
        (["llama-server", "--host", "127.0.0.1", "--host", "0.0.0.0"], "non_loopback_bind"),
        (["llama-server", "--host"], "non_loopback_bind"),
        (["llama-server", "--port", "18081"], "host_not_set"),  # the server's default is not relied on
    ],
)
def test_a_non_loopback_or_missing_bind_is_refused(argv: list[str], label: str) -> None:
    assert loopback_problem(argv) == label


def test_a_non_loopback_command_is_refused_before_anything_is_spawned(files) -> None:
    popen = FakePopen(FakeProcess(GOOD_OUTPUT))

    def all_interfaces(binary, model, mmproj, port):
        argv = llama_server_command(binary, model, mmproj, port)
        argv[argv.index("--host") + 1] = "0.0.0.0"
        return argv

    status = server(files, popen, command=all_interfaces).start(30.0)
    assert (status.state, status.problem) == (ServerState.FAILED, "non_loopback_bind")
    assert popen.calls == []


def test_the_spawned_server_binds_loopback_without_llama_overrides_and_with_cache_off(files) -> None:
    popen = FakePopen(FakeProcess(GOOD_OUTPUT))
    environ = {"PATH": "/usr/bin", "LLAMA_ARG_HOST": "0.0.0.0", "LLAMA_ARG_PORT": "80", "SENTINEL_RTSP_URL": "x"}
    status = server(files, popen, environ=environ).start(30.0)
    assert status.state is ServerState.READY
    (argv, kwargs), = popen.calls
    assert loopback_problem(argv) is None
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--cache-ram") + 1] == "0"  # D41
    env = kwargs["env"]
    assert not [key for key in env if key.startswith("LLAMA_ARG_")]
    assert env["LD_PRELOAD"] == L4T
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["start_new_session"] is True


def test_a_port_that_already_answers_is_refused(files) -> None:
    popen = FakePopen(FakeProcess(GOOD_OUTPUT))
    status = server(files, popen, in_use=True).start(30.0)
    assert (status.state, status.problem) == (ServerState.FAILED, "port_in_use")
    assert popen.calls == []


def test_missing_model_files_are_refused_by_label(files) -> None:
    files["mmproj.gguf"].unlink()
    popen = FakePopen(FakeProcess(GOOD_OUTPUT))
    status = server(files, popen).start(30.0)
    assert (status.state, status.problem) == (ServerState.FAILED, "scene_mmproj_missing")
    assert popen.calls == []


def test_every_llama_server_launch_in_the_benchmark_profiler_binds_loopback(tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("profile_launch", ROOT / "benchmarks/runner/demo_profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv) or FakeProcess([]))
    for cache_ram in (None, 0):
        for sanitized in (False, True):
            args = Namespace(llama=Path("llama-server"), model=Path("m"), mmproj=Path("p"), port=18081,
                             llama_cache_ram=cache_ram, sanitized_logs=sanitized)
            _, log = profile.start_llama(args, tmp_path)
            log.close()
    assert len(launched) == 4
    assert all(loopback_problem(argv) is None for argv in launched)


# ---------------------------------------------------------------- D27 placement guard


def test_a_healthy_fully_offloaded_server_is_ready_and_keeps_no_output_text(files) -> None:
    popen = FakePopen(FakeProcess(GOOD_OUTPUT))
    process = server(files, popen)
    status = process.start(30.0)
    assert (status.state, status.problem, status.layers, status.vision_on_gpu) == (ServerState.READY, None, "17/17", True)
    assert "private" not in repr(process) and "private" not in repr(status)


@pytest.mark.parametrize(
    ("lines", "libcuda", "label"),
    [
        ([b"offloaded 16/17 layers to GPU\n", b"CLIP using CUDA0\n"], None, "not_fully_offloaded"),
        ([b"CLIP using CUDA0\n"], None, "not_fully_offloaded"),
        ([b"offloaded 17/17 layers to GPU\n"], None, "vision_encoder_not_on_gpu"),
        (GOOD_OUTPUT, ["/usr/lib/aarch64-linux-gnu/libcuda.so.1"], "libcuda_not_l4t"),
        (GOOD_OUTPUT, [], "libcuda_not_l4t"),
    ],
)
def test_no_silent_cpu_fallback_the_server_is_stopped(files, lines, libcuda, label) -> None:
    process_fake = FakeProcess(lines)
    status = server(files, FakePopen(process_fake), libcuda=libcuda).start(30.0)
    assert (status.state, status.problem) == (ServerState.FAILED, label)
    assert process_fake.signals == ["TERM"]


def test_a_server_that_exits_during_load_or_never_gets_ready_fails(files) -> None:
    exited = FakeProcess([], exit_code=1)
    assert server(files, FakePopen(exited), healthy_after=10**9).start(30.0).problem == "exited_during_load:1"
    slow = FakeProcess([])
    status = server(files, FakePopen(slow), healthy_after=10**9).start(5.0)
    assert (status.state, status.problem) == (ServerState.FAILED, "not_ready_in_time")
    assert slow.signals == ["TERM"]


def test_a_ready_server_that_exits_reports_it(files) -> None:
    fake = FakeProcess(GOOD_OUTPUT)
    process = server(files, FakePopen(fake))
    assert process.start(30.0).state is ServerState.READY
    fake.returncode = 134
    status = process.status()
    assert (status.state, status.problem, status.returncode) == (ServerState.EXITED, "exited:134", 134)


def test_stop_is_bounded_and_kills_a_server_that_ignores_sigterm(files) -> None:
    fake = FakeProcess(GOOD_OUTPUT, ignores_terminate=True)
    process = server(files, FakePopen(fake))
    process.start(30.0)
    assert process.stop(0.01) is True
    assert fake.signals == ["TERM", "KILL"]
    assert process.status().state is ServerState.STOPPED
    assert fake.stdout.closed


@pytest.mark.parametrize("mapped", [[], ["/usr/local/lib/libggml-cuda.so.0.10.0"], ["/tmp/other/libllama.so.0"]])
def test_libraries_from_outside_the_checked_build_are_refused(files, mapped) -> None:
    fake = FakeProcess(GOOD_OUTPUT)
    status = server(files, FakePopen(fake), libraries=mapped).start(30.0)
    assert (status.state, status.problem) == (ServerState.FAILED, "libraries_not_profiled")
    assert fake.signals == ["TERM"]


# ---------------------------------------------------------------- session 116, part 2: the LFM3 build's log verbosity
#
# Representative output of the pinned LFM3 build (llama.cpp 19cce69), built from its format strings: since 67b2b7f2f
# ("logs : reduce") the library's INFO lines are TRACE (4), so at the default verbosity 3 the server prints only its
# own INFO lines and neither placement marker; lfm3check-20261028T101443Z was refused exactly so (layers None, vision
# False). At verbosity 4 the two markers appear with the usual text.

PINNED_DEFAULT_OUTPUT = [
    b"log_info: verbosity = 3 (adjust with the `-lv N` CLI arg)\n",
    b"device_info:\n",
    b"  - CUDA0   : Orin (7620 MiB, 6500 MiB free)\n",
    b"main: server is listening on http://127.0.0.1:18081\n",
]
PINNED_VERBOSE_OUTPUT = [
    b"log_info: verbosity = 4 (adjust with the `-lv N` CLI arg)\n",
    b"device_info:\n",
    b"  - CUDA0   : Orin (7620 MiB, 6500 MiB free)\n",
    b"srv          load_model: local path '/synthetic/model.gguf'\n",
    b"load_tensors: offloaded 25/25 layers to GPU\n",
    b"clip_ctx: CLIP using CUDA0 backend\n",
    b"srv          load_model: [mtmd] estimated worst-case memory usage of mmproj is 210.00 MiB (took 3.10 ms)\n",
    b"main: server is listening on http://127.0.0.1:18081\n",
]


def test_the_pinned_builds_default_output_lacks_both_markers_and_is_still_refused(files) -> None:
    """Missing evidence is not readiness: the run's refusal is kept for this output."""
    process_fake = FakeProcess(PINNED_DEFAULT_OUTPUT)
    status = server(files, FakePopen(process_fake)).start(30.0)
    assert (status.state, status.problem, status.layers, status.vision_on_gpu) == (
        ServerState.FAILED, "not_fully_offloaded", None, False)
    assert process_fake.signals == ["TERM"]


def test_the_pinned_builds_verbose_output_is_read_as_full_gpu_placement(files) -> None:
    status = server(files, FakePopen(FakeProcess(PINNED_VERBOSE_OUTPUT))).start(30.0)
    assert (status.state, status.problem, status.layers, status.vision_on_gpu) == (ServerState.READY, None, "25/25", True)


@pytest.mark.parametrize(("replace", "label"), [
    ((b"load_tensors: offloaded 25/25 layers to GPU\n", b"load_tensors: offloaded 20/25 layers to GPU\n"),
     "not_fully_offloaded"),
    ((b"load_tensors: offloaded 25/25 layers to GPU\n", b"load_tensors: offloaded 0/25 layers to GPU\n"),
     "not_fully_offloaded"),
    ((b"clip_ctx: CLIP using CUDA0 backend\n", b"clip_ctx: CLIP using CPU backend\n"), "vision_encoder_not_on_gpu"),
    ((b"clip_ctx: CLIP using CUDA0 backend\n", b""), "vision_encoder_not_on_gpu"),
])
def test_partial_or_cpu_placement_in_the_verbose_output_is_still_refused(files, replace, label) -> None:
    old, new = replace
    lines = [new if line == old else line for line in PINNED_VERBOSE_OUTPUT]
    process_fake = FakeProcess(lines)
    status = server(files, FakePopen(process_fake)).start(30.0)
    assert (status.state, status.problem) == (ServerState.FAILED, label)
    assert process_fake.signals == ["TERM"]


def test_the_sets_own_launch_flags_follow_the_profiled_ones_and_keep_loopback(files) -> None:
    from sentinel.scene.models import SCENE_MODEL_SETS

    lfm3 = SCENE_MODEL_SETS["lfm3-vl"]
    assert lfm3.server_flags == ("--log-verbosity", "4")
    assert SCENE_MODEL_SETS["lfm2.5-vl"].server_flags == () == SCENE_MODEL_SETS["lfm2-vl"].server_flags
    popen = FakePopen(FakeProcess(PINNED_VERBOSE_OUTPUT))
    process = LlamaServerProcess(
        files["llama-server"], files["model.gguf"], files["mmproj.gguf"], 18081, environ={"PATH": "/usr/bin"},
        popen=popen, health=lambda port: True, in_use=lambda port: False, libcuda=lambda pid: [L4T],
        libraries=lambda pid: [str(files["llama-server"].parent / "libllama.so.0.7.0")], sleep=lambda s: None,
        extra_flags=lfm3.server_flags)
    assert process.start(30.0).state is ServerState.READY
    (argv, _), = popen.calls
    assert argv[-2:] == ["--log-verbosity", "4"] and loopback_problem(argv) is None
    assert argv[:-2] == llama_server_command(files["llama-server"], files["model.gguf"], files["mmproj.gguf"], 18081)
    assert llama_server_command(files["llama-server"], files["model.gguf"], files["mmproj.gguf"], 18081,
                                lfm3.server_flags) == argv


@pytest.mark.parametrize("value", [True, -1, 6, "4", 4.0])
def test_only_a_real_verbosity_level_makes_flags(value) -> None:
    from sentinel.scene.llama_server import log_verbosity_flags

    assert log_verbosity_flags(None) == () and log_verbosity_flags(4) == ("--log-verbosity", "4")
    with pytest.raises(ValueError):
        log_verbosity_flags(value)


def test_the_cli_launches_the_selected_sets_flags(files) -> None:
    from sentinel import cli
    from sentinel.demo_runtime import SceneOptions

    for key, flags in (("lfm3-vl", ("--log-verbosity", "4")), ("lfm2.5-vl", ())):
        options = SceneOptions(files["llama-server"], files["model.gguf"], files["mmproj.gguf"], model_set=key)
        assert cli._scene_server(options, 18081)._extra_flags == flags


def test_the_profiler_passes_the_same_verbosity_only_when_given(tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("profile_verbosity", ROOT / "benchmarks/runner/demo_profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    assert "--log-verbosity" not in profile.llama_flags(profile.parse_args([]))
    flags = profile.llama_flags(profile.parse_args(["--llama-log-verbosity", "4", "--llama-cache-ram", "0"]))
    assert flags[-4:] == ["--cache-ram", "0", "--log-verbosity", "4"]
    launched = []
    monkeypatch.setattr(profile.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv) or FakeProcess([]))
    args = Namespace(llama=Path("llama-server"), model=Path("m"), mmproj=Path("p"), port=18081, llama_cache_ram=0,
                     llama_log_verbosity=4, sanitized_logs=True)
    _, log = profile.start_llama(args, tmp_path)
    log.close()
    assert launched[0][-2:] == ["--log-verbosity", "4"] and loopback_problem(launched[0]) is None
