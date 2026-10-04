"""The demo scene server process (D-1; V2-26 demo form, full acceptance pending).

``sentinel run`` starts llama-server only through LlamaServerProcess, which
owns the child from start to stop:

- **Loopback only (D42).** The argv comes from ``llama_server_command()`` and
  is checked again by ``loopback_problem()`` before anything is spawned: every
  ``--host`` must be 127.0.0.1 (llama-server takes a host ending in ``.sock``
  as a Unix socket), and there must be one. The environment comes from
  ``llama_server_environment()``, which removes every ``LLAMA_ARG_*`` variable.
  A port that already answers on 127.0.0.1 is refused, so the client can never
  talk to some other server.
- **D27 guard.** The server counts as ready only after ``/health`` answers 200
  and its output shows a full offload (every layer and the vision encoder on
  CUDA0) with only L4T's libcuda mapped into the child. There is no silent CPU
  fallback: otherwise the server is stopped and scene analysis is unavailable.
- **No raw output kept.** A pump thread reads the child's output so it never
  blocks on a full pipe, and keeps only fixed placement markers. Server text
  can contain request details; it is never stored or shown.
- **Bounded stop.** SIGTERM, then SIGKILL after a grace period; the pipe and
  pump thread are always released.

Problems are fixed labels. Portable: nothing here imports a model library.
"""

from __future__ import annotations

import http.client
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..inference.legacy_ultralytics import L4T_LIBCUDA_DIR
from .llama_server import LOOPBACK_HOST, llama_server_command, llama_server_environment

HEALTH_PATH = "/health"
_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_VISION_ON_GPU = "CLIP using CUDA0"


class ServerState(str, Enum):
    NOT_STARTED = "not_started"
    STARTING = "starting"
    READY = "ready"  # healthy, fully offloaded, L4T libcuda only
    FAILED = "failed"  # refused or did not become ready; see problem
    EXITED = "exited"  # it was ready and then exited by itself
    STOPPED = "stopped"


@dataclass(frozen=True)
class ServerStatus:
    state: ServerState
    problem: str | None
    returncode: int | None
    layers: str | None  # "17/17" once the offload line was seen
    vision_on_gpu: bool


class ServerRefused(Exception):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def loopback_problem(argv: Sequence[str]) -> str | None:
    """None if ``argv`` makes llama-server listen on 127.0.0.1 only, else a label."""
    hosts = []
    for index, arg in enumerate(argv):
        if arg.startswith("--host="):
            hosts.append(arg.partition("=")[2])
        elif arg == "--host":
            hosts.append(argv[index + 1] if index + 1 < len(argv) else "")
    if not hosts:
        return "host_not_set"
    if any(host != LOOPBACK_HOST for host in hosts):
        return "non_loopback_bind"
    return None


def port_in_use(port: int, *, timeout_s: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((LOOPBACK_HOST, port)) == 0


def health_ok(port: int, *, timeout_s: float = 2.0) -> bool:
    """HTTP 200 from /health on 127.0.0.1; llama-server answers 503 while loading."""
    conn = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=timeout_s)
    try:
        conn.request("GET", HEALTH_PATH)
        response = conn.getresponse()
        response.read(4096)
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def mapped_libcuda(pid: int) -> list[str]:
    try:
        text = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return sorted({line.split()[-1] for line in text.splitlines() if "libcuda.so" in line})


class LlamaServerProcess:
    def __init__(
        self,
        binary: Path,
        model: Path,
        mmproj: Path,
        port: int,
        *,
        environ: Mapping[str, str] | None = None,
        command: Callable[[Path, Path, Path, int], list[str]] = llama_server_command,
        popen: Callable[..., Any] = subprocess.Popen,
        health: Callable[[int], bool] = health_ok,
        in_use: Callable[[int], bool] = port_in_use,
        libcuda: Callable[[int], list[str]] = mapped_libcuda,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._files = {"llama_server_binary": binary, "scene_model": model, "scene_mmproj": mmproj}
        self._port = port
        self._environ = os.environ if environ is None else environ
        self._command = command
        self._popen = popen
        self._health = health
        self._in_use = in_use
        self._libcuda = libcuda
        self._sleep = sleep
        self._monotonic = monotonic
        self._proc: Any = None
        self._pump: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = ServerState.NOT_STARTED
        self._problem: str | None = None
        self._layers: tuple[int, int] | None = None
        self._vision = False

    @property
    def port(self) -> int:
        return self._port

    def __repr__(self) -> str:
        return f"LlamaServerProcess(port={self._port}, state={self._state.value})"

    def status(self) -> ServerStatus:
        proc = self._proc
        returncode = proc.poll() if proc is not None else None
        with self._lock:
            if self._state is ServerState.READY and returncode is not None:
                self._state, self._problem = ServerState.EXITED, f"exited:{returncode}"
            layers = None if self._layers is None else f"{self._layers[0]}/{self._layers[1]}"
            return ServerStatus(self._state, self._problem, returncode, layers, self._vision)

    def start(self, ready_timeout_s: float) -> ServerStatus:
        """Spawn, wait for health and check placement; returns READY or FAILED (stopped)."""
        if self._state is not ServerState.NOT_STARTED:
            raise RuntimeError("the scene server can be started once")
        try:
            self._spawn()
            self._await_ready(ready_timeout_s)
        except ServerRefused as exc:
            self._fail(exc.label)
            return self.status()
        with self._lock:
            self._state = ServerState.READY
        return self.status()

    def _spawn(self) -> None:
        for label, path in self._files.items():
            if not Path(path).is_file():
                raise ServerRefused(f"{label}_missing")
        argv = self._command(*self._files.values(), self._port)
        problem = loopback_problem(argv)
        if problem is not None:
            raise ServerRefused(problem)
        env = llama_server_environment(self._environ)
        if any(key.startswith("LLAMA_ARG_") for key in env):
            raise ServerRefused("llama_arg_override")
        if self._in_use(self._port):
            raise ServerRefused("port_in_use")
        with self._lock:
            self._state = ServerState.STARTING
        try:
            self._proc = self._popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
        except OSError as exc:
            raise ServerRefused(f"spawn_failed:{type(exc).__name__}") from None
        self._pump = threading.Thread(target=self._read_output, name="sentinel-scene-server-log", daemon=True)
        self._pump.start()

    def _await_ready(self, timeout_s: float) -> None:
        deadline = self._monotonic() + timeout_s
        while not self._health(self._port):
            if self._proc.poll() is not None:
                raise ServerRefused(f"exited_during_load:{self._proc.poll()}")
            if self._monotonic() >= deadline:
                raise ServerRefused("not_ready_in_time")
            self._sleep(0.25)
        # The placement lines precede the listening socket; give the pump a moment to read them.
        marker_deadline = self._monotonic() + 5.0
        while not self._placement_complete() and self._monotonic() < marker_deadline:
            self._sleep(0.1)
        with self._lock:
            layers, vision = self._layers, self._vision
        if layers is None or layers[0] != layers[1] or layers[1] == 0:
            raise ServerRefused("not_fully_offloaded")
        if not vision:
            raise ServerRefused("vision_encoder_not_on_gpu")
        libcuda = self._libcuda(self._proc.pid)
        if not libcuda or not all(path.startswith(L4T_LIBCUDA_DIR) for path in libcuda):
            raise ServerRefused("libcuda_not_l4t")

    def _placement_complete(self) -> bool:
        with self._lock:
            return self._layers is not None and self._vision

    def _read_output(self) -> None:
        stream = self._proc.stdout
        try:
            for raw in iter(lambda: stream.readline(65_536), b""):
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                offload = _OFFLOAD.search(text)
                with self._lock:
                    if offload:
                        self._layers = (int(offload[1]), int(offload[2]))
                    if _VISION_ON_GPU in text:
                        self._vision = True
        except (OSError, ValueError):
            return

    def _fail(self, label: str) -> None:
        self.stop(10.0)
        with self._lock:
            self._state, self._problem = ServerState.FAILED, label

    def stop(self, grace_s: float) -> bool:
        """Terminate the child (then kill it); False if it was still running afterwards."""
        proc, stopped = self._proc, True
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    stopped = False
        if proc is not None and proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if self._pump is not None:
            self._pump.join(2.0)
        with self._lock:
            if self._state in (ServerState.STARTING, ServerState.READY, ServerState.EXITED):
                self._state = ServerState.STOPPED
        return stopped
