#!/usr/bin/env python3
"""Checklist step 5 (D59): run ``sentinel run`` for a bounded time under the existing memory guard, keeping evidence.

Operator tooling, not part of the runtime. The command after ``--`` starts in its own session, with stdout written to
``run.jsonl`` and stderr to ``run.err`` in ``--out``: a new directory, normally under ~/sentinel-runs so that a
reboot keeps it. operator_check's PressureGuard samples every 0.2 s with its limits unchanged: pressure 4.8 GB,
MemFree 1 GiB, MemAvailable 2 GiB, any swap-counter change, or a sampling gap over 0.5 s.

The runtime is stopped the way Ctrl-C stops it: SIGINT to the runtime process alone. The runtime then stops its own
llama-server, which runs in its own session. That happens when:
- ``--duration-s`` has passed since the runtime's ``"run": "starting"`` line (or since launch, with
  ``--from-launch``);
- the guard meets a condition;
- the runtime is not ready within ``--ready-timeout-s``; or
- this process gets SIGINT or SIGTERM.

If the runtime has not exited ``--stop-grace-s`` after that, its process group gets SIGTERM and then SIGKILL. A
llama-server that this run started and that is left afterwards is stopped and reported; none may run before launch.

Which processes are this run's is never read from a name or a port. Before launch the guard asks the kernel to hand it
the processes its runtime leaves behind (``PR_SET_CHILD_SUBREAPER``; refused before launch as
``ownership_unavailable`` without it), so that after the runtime's exit everything the run started is this process's
own descendant, whatever session it started. Only such a descendant named llama-server is signalled, through a handle
on the process itself after its number and start time are read again, so a later process with the same number is
never signalled. A llama-server of this user that is not a descendant (somebody else's, started during the run) is
counted as ``leftovers.llama_server_unresolved``, left running and never signalled, and the run is then not reported
clear. Any other process this run started that is still running afterwards is reported with the runtime's group
(``leftovers.runtime_group_left``), in whatever session it runs, and is not signalled here. The runtime's group is
signalled only while the runtime itself has not been collected: until then its number, which is the group's, cannot
pass to another process. If this guard itself fails with an error, it first stops what it started in the same way,
then raises the error. The guard is meant to run as a process of its own that starts nothing but the runtime: what a
process it already had before the launch leaves behind afterwards would be taken for this run's.

With ``--step4-headroom`` it also refuses before launch unless operator_check's step-4 admission headroom holds
(pressure below 2.0 GB, MemFree at least 3.5 GB, MemAvailable at least 4.0 GB; ``baseline_problem``, unchanged).

At each ``--cue-at SECONDS:NAME:TEXT`` offset (counted like the duration) it prints TEXT with a bell to its own
stderr, the operator's terminal, and records when: the guard's monotonic clock is CLOCK_MONOTONIC, the runtime's
too, so ``child.ready_mono_s`` and each cue's ``printed_mono_s`` place the runtime's records relative to the cue
the operator's stopwatch started from. Cue texts are fixed protocol text, never a secret.

With ``--require-ready tracking_diagnostic`` (session 80) the ``starting`` line alone is not readiness: its startup
must report no detector problem and the tracking diagnostic's marker with the tracker's settings complete (the
detector loaded with the diagnostic's hooks). Otherwise the runtime is stopped at once (``readiness_refused``; the
reason, a fixed label, under ``readiness`` in ``result.json``), nothing counts from readiness, no cue is printed and
no status is saved; a runtime that exits before its ``starting`` line leaves the requirement unmet
(``no_starting_line``). The exit status is then 1: a clean process exit alone never implies the diagnostic was ready.
``--require-ready landmark_diagnostic`` (session 90) works the same way for the landmark diagnostic: no detector
problem, no face problem, and the marker ``face.landmark_diagnostic`` holding exactly the pinned values below.
``--require-ready face_chat`` (session 103, the validation-only face + chat integration run) works the same way: no
detector, face or scene problem; face exactly ``FACE_CHAT_FACE`` (a validation run of the normal path with one
enrolled identity, no candidate, comparison, experiment, continuity rule or diagnostic marker); the scene server
ready (17/17 layers, or the count ``--scene-layers N/N`` names, and the vision encoder on the GPU); and chat
``available`` with exactly its validation-run reason (``CHAT_VALIDATION_AVAILABLE``; chat paused by its memory gate,
unavailable or admitted is not this run). ``--scene-layers`` (session 116, part 5) is for a scene model set whose
model has another layer count (LFM3-VL-2B: 25/25); it must be fully offloaded, is recorded under ``parameters`` and is
what the readings then require; without it nothing is recorded and 17/17 is required, as before.

At each ``--status-at SECONDS:NAME`` offset it saves ``status-NAME.json`` from the loopback status page, and it
classifies the listeners on 18090 and 18081 as loopback, any or other. ``result.json`` holds numbers and fixed labels
only. No stream URL, token, chat ID, address or environment value is read or written; the command's own output goes
to ``run.jsonl`` and ``run.err`` unchanged.

Exit status 0 means the supervision itself was clean: the runtime exited on its own, or exited 0 after the duration's
stop, with no guard stop, no forced stop, no leftover process and no listener left. Whether the runtime did what a
step-5 part expects is judged from ``result.json`` by the operator block (``guard_verdict.py`` since session 81: the
exit alone also covers a runtime that exited on its own before the duration).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import operator_check as oc

SAMPLE_S = oc.SAMPLE_S
WATCHED_PORTS = (18090, 18081)  # the status page and the scene server, both loopback-only by design
STATUS_URL = "http://127.0.0.1:18090/status.json"
MAX_STATUS_BYTES = 1_048_576
STATUS_TIMEOUT_S = 2.0
MAX_DURATION_S = 3600.0  # session 109: the one-hour dashboard trial (900 s before it); offsets bounded alike
MAX_READY_TIMEOUT_S = 900.0  # the readiness wait keeps its earlier bound
# sentinel run's shutdown is bounded at about 66 s with scene analysis (capture 16 s, scene server 10 s plus 5 s to
# kill, scene worker 21 s, outbox 12 s, status page 2 s); the grace leaves room above that.
STOP_GRACE_S = 90.0
KILL_GRACE_S = 5.0
READY_TIMEOUT_S = 300.0
READINESS = ("tracking_diagnostic", "landmark_diagnostic", "face_chat",  # --require-ready values
             "face_windows", "face_chat_windows",  # route B (session 116, part 17): main_single validation runs
             "face_windows_capture_raw", "face_windows_shadow")  # the multi-pose comparison's B and E runs
ROUTE_B_READINESS = ("face_windows", "face_chat_windows", "face_windows_capture_raw", "face_windows_shadow")
# What the landmark diagnostic's startup marker must be before T0 (sentinel.identity.landmarks.marker(); kept equal by a
# test; this script imports nothing from the runtime).
LANDMARK_MARKER = {"version": 1, "score": 0.9, "min_eye_spacing_px": 4.0, "persons_max": 4}
# What the face + chat integration run's starting line must show before T0 (session 103; kept equal to the runtime's by
# a test; this script imports nothing from the runtime): face recognition on the normal path, nothing else, and chat
# as a validation run, available now.
FACE_CHAT_FACE = {"validation_run": True, "identities_enrolled": 1}
# Route B (session 116, part 17; kept equal to the runtime's by a test): a main_single validation run's face entry, with
# the head-window stage as pinned (identity.window_stage.WindowSettings().numbers()), and its stream entry's fixed parts.
WINDOWS_FACE = {**FACE_CHAT_FACE, "detection": "head_windows",
                "stage": {"kind": "head_windows", "side": 640, "margin": 0.5, "max_windows": 4, "head_fraction": 0.4,
                          "duplicate_iou": 0.3, "context": 0.5, "explained_share": 0.5}}
# The multi-pose comparison (kept equal to the runtime's by a test): B's raw capture, and E's shadow comparison with
# its working gallery's prototypes digest (given as --shadow-digest, the frozen candidate's) and its track lives.
WINDOWS_CAPTURE_RAW_FACE = {**WINDOWS_FACE, "capture": True, "capture_raw": True}
SHADOW_DIGEST = re.compile(r"[0-9a-f]{64}")
STREAM_MAIN_SINGLE, WORKING_SIZE, BYTETRACK_LOST_TRACK_FRAMES = "main_single", [640, 480], 30
CHAT_VALIDATION_AVAILABLE = {
    "state": "available",
    "reason": ("ask about the picture the camera is showing now (validation run: chat is not admitted; its answers "
               "and their effect on monitoring are being checked)")}
SCENE_LAYERS = "17/17"  # LFM2-VL and LFM2.5-VL; another set's count is passed with --scene-layers (session 116)
# What the tracking diagnostic's startup marker must hold before T0: the tracker's settings as the runtime records
# them under face.tracking_diagnostic.tracker (legacy_ultralytics.diagnostic_settings; equal to
# sentinel.tracking.diagnostic.SETTING_KEYS, kept so by a test; this script imports nothing from the runtime).
TRACKING_SETTING_KEYS = ("conf", "match_thresh", "track_high_thresh", "track_low_thresh", "new_track_thresh",
                         "track_buffer", "max_time_lost", "fuse_score", "unconfirmed_thresh", "shadow_conf",
                         "shadow_max_nms", "max_det", "iou")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- read-only host facts


def _scope_v4(hex_address: str) -> str:
    octets = bytes.fromhex(hex_address)[::-1]
    if octets == b"\0\0\0\0":
        return "any"
    return "loopback" if octets[0] == 127 else "other"


def _scope_v6(hex_address: str) -> str:
    raw = bytes.fromhex(hex_address)
    words = b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4))  # four host-order 32-bit words
    if words == bytes(16):
        return "any"
    if words == bytes(15) + b"\1":
        return "loopback"
    if words[:12] == bytes(10) + b"\xff\xff":  # an IPv4-mapped address
        return "any" if words[12:] == b"\0\0\0\0" else ("loopback" if words[12] == 127 else "other")
    return "other"


def listeners(root: Path = Path("/proc/net"), ports: tuple[int, ...] = WATCHED_PORTS) -> list[dict[str, Any]]:
    """Listening TCP sockets on ``ports``: port and scope (loopback, any, other) only, never the address."""
    found = []
    for name, scope in (("tcp", _scope_v4), ("tcp6", _scope_v6)):
        try:
            lines = (root / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":  # 0A: LISTEN
                continue
            address, _, port_hex = fields[1].partition(":")
            try:
                port = int(port_hex, 16)
                if port in ports:
                    found.append({"port": port, "family": name, "scope": scope(address)})
            except ValueError:
                continue
    return sorted(found, key=lambda item: (item["port"], item["family"], item["scope"]))


def process_ids(comm: str, proc: Path = Path("/proc"), uid: int | None = None) -> list[int]:
    """This user's processes whose name is ``comm``."""
    uid = os.getuid() if uid is None else uid
    found = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid == uid and (path / "comm").read_text().strip() == comm:
                found.append(int(path.name))
        except OSError:
            continue
    return sorted(found)


# ---------------------------------------------------------------- which processes are this run's

PR_SET_CHILD_SUBREAPER, PR_GET_CHILD_SUBREAPER = 36, 37  # <linux/prctl.h>, Linux 3.4


def process_info(pid: int, proc: Path = Path("/proc"), uid: int | None = None) -> dict[str, Any] | None:
    """One process of this user from ``/proc/<pid>/stat``: name, state, parent, group and start time (clock ticks
    since boot). The number with the start time is an identity that no later process has. None once it has gone."""
    uid = os.getuid() if uid is None else uid
    path = proc / str(pid)
    try:
        if path.stat().st_uid != uid:
            return None
        text = (path / "stat").read_text(errors="replace")
        fields = text[text.rindex(")") + 2:].split()
        return {"pid": pid, "comm": text[text.index("(") + 1:text.rindex(")")], "state": fields[0],
                "ppid": int(fields[1]), "pgrp": int(fields[2]), "start": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def process_table(proc: Path = Path("/proc"), uid: int | None = None) -> dict[int, dict[str, Any]]:
    """This user's processes by number."""
    found: dict[int, dict[str, Any]] = {}
    try:
        names = [path.name for path in proc.iterdir()]
    except OSError:
        return found
    for name in names:
        if name.isdigit():
            info = process_info(int(name), proc, uid)
            if info is not None:
                found[info["pid"]] = info
    return found


def still_running(info: dict[str, Any], proc: Path = Path("/proc")) -> bool:
    """Whether the process still runs. ``stat`` shows the state of its first thread alone: that one may have ended
    (Z) while others still hold the memory and the ports, so an ended first thread counts as gone only once no other
    thread is left."""
    if info["state"] != "Z":
        return True
    try:
        return len(os.listdir(proc / str(info["pid"]) / "task")) > 1
    except OSError:
        return False


def descendants(table: dict[int, dict[str, Any]], root: int, skip=frozenset()) -> list[dict[str, Any]]:
    """The processes of ``table`` below ``root``, ended ones not yet collected included, parents before children. A
    link counts only when the child did not start before its parent, so a number that passed to a later process never
    joins two families; ``skip`` holds (number, start time) identities left out with everything below them."""
    children: dict[int, list[dict[str, Any]]] = {}
    for info in table.values():
        children.setdefault(info["ppid"], []).append(info)
    found, stack, seen = [], [root], {root}
    while stack:
        parent = stack.pop()
        if parent not in table:
            continue
        for info in sorted(children.get(parent, ()), key=lambda item: item["pid"]):
            if info["pid"] in seen or (info["pid"], info["start"]) in skip or info["start"] < table[parent]["start"]:
                continue
            seen.add(info["pid"])
            found.append(info)
            stack.append(info["pid"])
    return found


def child_subreaper(enable: bool | None = None) -> bool:
    """Whether this process is handed the processes its descendants leave behind; sets it first when ``enable`` is
    given. Raises OSError when the kernel or libc cannot do it."""
    import ctypes

    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.restype = ctypes.c_int
    if enable is not None:
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        if prctl(PR_SET_CHILD_SUBREAPER, 1 if enable else 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    value = ctypes.c_int(-1)
    prctl.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)] + [ctypes.c_ulong] * 3
    if prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(value), 0, 0, 0) != 0 or value.value not in (0, 1):
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER")
    return value.value == 1


def send_to_process(info: dict[str, Any], signum: int, proc: Path = Path("/proc"), own_pid: int | None = None,
                    open_handle=None, send_handle=None, kill=os.kill) -> bool:
    """Signal exactly the process ``info`` names (number and start time). The handle is taken first and the identity
    read again after it, so the handle is on that process or the signal is not sent; a later process with the same
    number is never signalled. Without process handles (before Linux 5.3) only this process's own child is signalled:
    its number cannot pass on before this process collects it. False when nothing was sent."""
    open_handle = getattr(os, "pidfd_open", None) if open_handle is None else open_handle
    send_handle = getattr(signal, "pidfd_send_signal", None) if send_handle is None else send_handle
    own_pid = os.getpid() if own_pid is None else own_pid
    handle = None
    if open_handle is not None and send_handle is not None:
        try:
            handle = open_handle(info["pid"])
        except ProcessLookupError:
            return False
        except OSError:
            handle = None
    try:
        now = process_info(info["pid"], proc)
        if now is None or now["start"] != info["start"] or not still_running(now, proc):
            return False
        if handle is not None:
            send_handle(handle, signum)
        elif now["ppid"] == own_pid:
            kill(info["pid"], signum)
        else:
            return False
        return True
    except OSError:
        return False
    finally:
        if handle is not None:
            os.close(handle)


class Ownership:
    """What this run may signal besides the runtime it started: this process's descendants that were not there before
    the launch. With the kernel handing orphans to this process, that is everything the run started and left."""

    def __init__(self, proc: Path = Path("/proc"), pid: int | None = None, subreaper=child_subreaper,
                 send=send_to_process, collect=os.waitpid) -> None:
        self.proc, self.pid = proc, os.getpid() if pid is None else pid
        self._subreaper, self._send, self._collect = subreaper, send, collect
        self._restore: bool | None = None
        self._before: frozenset = frozenset()
        self.leader: int | None = None  # the runtime: collected by its own Popen, never here
        self.active = False

    def begin(self) -> str | None:
        """Before launch. A fixed label when ownership cannot be established (then nothing may be launched)."""
        try:
            was = self._subreaper()
            if not self._subreaper(True):
                return "ownership_unavailable"
        except (OSError, AttributeError, TypeError, ImportError):
            return "ownership_unavailable"
        self._restore = was
        table = process_table(self.proc)
        if self.pid not in table:
            self.end()
            return "ownership_unavailable"
        self._before = frozenset((info["pid"], info["start"]) for info in descendants(table, self.pid))
        self.active = True
        return None

    def owned(self) -> list[dict[str, Any]]:
        """The running processes this run owns now (the runtime too, while it runs). Those that have ended and were
        handed to this process are collected, so that neither their numbers nor their group stay taken."""
        if not self.active:
            return []
        found = []
        for info in descendants(process_table(self.proc), self.pid, self._before):
            if still_running(info, self.proc):
                found.append(info)
            elif info["ppid"] == self.pid and info["pid"] != self.leader:
                try:
                    done = self._collect(info["pid"], os.WNOHANG)
                except OSError:
                    continue
                if isinstance(done, tuple) and done[0] == 0:  # not collectable yet: it has not wholly ended
                    found.append(info)
        return found

    def not_owned(self, comm: str) -> list[dict[str, Any]]:
        """This user's running processes named ``comm`` that are not shown to be this run's: one reading of /proc, so
        a process is never counted both as owned and as not."""
        table = process_table(self.proc)
        mine = {(info["pid"], info["start"]) for info in descendants(table, self.pid, self._before)} if self.active else set()
        return [info for _, info in sorted(table.items()) if info["comm"] == comm and still_running(info, self.proc)
                and info["pid"] != self.pid and (info["pid"], info["start"]) not in mine]

    def signal(self, info: dict[str, Any], signum: int) -> bool:
        if not any(item["pid"] == info["pid"] and item["start"] == info["start"] for item in self.owned()):
            return False  # not shown to be this run's now: not signalled
        return self._send(info, signum, self.proc, self.pid)

    def end(self) -> None:
        self.active = False
        if self._restore is not None:
            try:
                self._subreaper(self._restore)
            except (OSError, AttributeError, TypeError, ImportError):
                pass
            self._restore = None


def thp_enabled_self() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("THP_enabled:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def fetch_status(url: str = STATUS_URL) -> tuple[str, bytes]:
    """(label, body): ``saved`` with at most MAX_STATUS_BYTES, or an error class; never raises."""
    try:
        with urllib.request.urlopen(url, timeout=STATUS_TIMEOUT_S) as response:  # loopback only
            body = response.read(MAX_STATUS_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"unavailable:{type(exc).__name__}", b""
    if len(body) > MAX_STATUS_BYTES:
        return "too_large", b""
    try:
        json.loads(body)
    except ValueError:
        return "not_json", b""
    return "saved", body


class Child:
    def __init__(self, argv: list[str], stdout: Path, stderr: Path) -> None:
        with open(stdout, "wb") as out, open(stderr, "wb") as err:
            self.process = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                            start_new_session=True)
        self.pid = self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def signal(self, signum: int) -> None:
        if self.process.poll() is None:
            try:
                os.kill(self.pid, signum)
            except ProcessLookupError:
                pass

    def signal_group(self, signum: int) -> bool:
        """Signal the runtime's whole group; False once nothing in it is left, and once the runtime has been
        collected: from then on its number, which is the group's, may pass to a process that is not this run's."""
        if self.process.returncode is not None:
            return False
        try:
            os.killpg(self.pid, signum)
            return True
        except ProcessLookupError:
            return False

    def group_alive(self) -> bool:
        """Whether a process of the runtime's group is still running. Read from /proc and nothing is collected, so
        that the runtime keeps its number (and with it the group's) until the loop has seen its exit."""
        return any(info["pgrp"] == self.pid and still_running(info) for info in process_table().values())


class SystemBackend:
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    utc = staticmethod(utc_now)
    listeners = staticmethod(listeners)
    fetch = staticmethod(fetch_status)
    thp_self = staticmethod(thp_enabled_self)

    def __init__(self) -> None:
        self._oc = oc.SystemBackend()
        self._ownership = Ownership()

    def sample(self) -> dict:
        return self._oc.sample()

    def spawn(self, argv: list[str], stdout: Path, stderr: Path) -> Child:
        child = Child(argv, stdout, stderr)
        self._ownership.leader = child.pid
        return child

    @staticmethod
    def llama_pids() -> list[int]:
        """Every llama-server of this user, by name: for the refusal before launch and for the count of those that
        are not this run's. Nothing is ever signalled from this list."""
        return process_ids("llama-server")

    def own(self) -> str | None:
        """Before launch: have the kernel hand this process what the runtime leaves behind. A label if it cannot."""
        return self._ownership.begin()

    def owned(self) -> list[dict[str, Any]]:
        """The running processes this run started (number, start time, name), read now."""
        return self._ownership.owned()

    def signal(self, info: dict[str, Any], signum: int) -> bool:
        """Signal one process of ``owned()``, if it is still that process; False when nothing was sent."""
        return self._ownership.signal(info, signum)

    def unresolved_servers(self) -> list[dict[str, Any]]:
        """This user's running llama-servers that are not shown to be this run's. Reported, never signalled."""
        return self._ownership.not_owned("llama-server")

    def disown(self) -> None:
        self._ownership.end()


# ---------------------------------------------------------------- the runtime's own output


def _starting_summary(startup: dict) -> dict[str, Any]:
    """The policy and component evidence from the runtime's ``starting`` line: numbers and fixed labels."""
    releases = {}
    for component, record in (startup.get("releases") or {}).items():
        files = (record or {}).get("files") or {}
        releases[component] = {role: (item or {}).get("result") for role, item in files.items()}
    disable = startup.get("thp_disable") or {}
    server = startup.get("scene_server")
    return {
        "memory_policy": startup.get("memory_policy"),
        "thp_scope": startup.get("thp_scope"),
        "thp_disable": {key: disable.get(key) for key in ("verified", "reason", "set_rc", "thp_enabled")}
        if disable else None,
        "releases": releases,
        "scene_server": {key: server.get(key) for key in ("state", "problem", "layers", "vision_on_gpu")}
        if isinstance(server, dict) else None,
        "scene_problem": startup.get("scene_problem"),
        "detector_problem": startup.get("detector_problem"),
        "face": startup.get("face"),  # sentinel run --face: validation_run and identities_enrolled (a count)
        **({"stream": startup.get("stream")} if "stream" in startup else {}),  # route B: only main_single runs have it
        "face_problem": startup.get("face_problem"),
        "face_memory_before": startup.get("face_memory_before"),
        "notifier_problems": startup.get("notifier_problems"),
        "scene_memory_before": startup.get("scene_memory_before"),
        "detector_memory_before": startup.get("detector_memory_before"),
    }


def readiness_problem(required: str | None, startup: Any, scene_layers: str = SCENE_LAYERS,
                      shadow_digest: str | None = None) -> str | None:
    """None when ``startup`` (the runtime's ``starting`` line's ``startup``) meets ``required``, else a fixed label
    naming what is missing; None without a requirement. ``tracking_diagnostic``: no ``detector_problem`` and the
    marker ``face.tracking_diagnostic.tracker`` a mapping with every TRACKING_SETTING_KEYS entry a number
    (``fuse_score`` a flag). The device run trk-20261016T032337Z started with ``detector_problem`` set and the
    marker's ``tracker`` null, and its cues ran against no detector (session 80)."""
    if required is None:
        return None
    if required not in READINESS:
        return "unknown_requirement"
    if not isinstance(startup, dict):
        return "startup_missing"
    problem = startup.get("detector_problem")
    if problem is not None:
        return f"detector_problem:{problem}"
    face = startup.get("face")
    route = stream_route(startup)  # route B: each requirement names the profile it reads; another is refused
    if route != ("main_single" if required in ROUTE_B_READINESS else "sub"):
        return f"stream_profile_{route}"
    if required in ROUTE_B_READINESS:  # route B: the face on head windows, the stream as configured
        if startup.get("face_problem") is not None:
            return f"face_problem:{startup.get('face_problem')}"
        expected = {"face_windows_capture_raw": WINDOWS_CAPTURE_RAW_FACE,
                    "face_windows_shadow": {**WINDOWS_FACE, "shadow": True, "shadow_digest": shadow_digest,
                                            "lives": True}}.get(required, WINDOWS_FACE)
        if required == "face_windows_shadow" and isinstance(face, dict) and face.get("shadow_digest") != shadow_digest \
                and {k: v for k, v in face.items() if k != "shadow_digest"} == \
                {k: v for k, v in expected.items() if k != "shadow_digest"}:
            return "shadow_digest_not_the_frozen_candidate"  # another working gallery: stopped before T0
        if face != expected:
            if not isinstance(face, dict):
                return "face_missing"
            return {"face_windows_capture_raw": "face_not_the_raw_capture_run",
                    "face_windows_shadow": "face_not_the_shadow_run"}.get(required,
                                                                         "face_not_the_windows_validation_run")
        problem = _stream_problem(startup.get("stream"))
        if problem is not None:
            return problem
        if required in ("face_windows", "face_windows_capture_raw", "face_windows_shadow"):
            return None
        face = FACE_CHAT_FACE  # the rest exactly as face_chat
        required = "face_chat"
    if required == "face_chat":  # session 103: face, the scene server and chat as a validation run, all ready together
        for name in ("face_problem", "scene_problem"):
            if startup.get(name) is not None:
                return f"{name}:{startup.get(name)}"
        if face != FACE_CHAT_FACE:
            return "face_not_the_plain_validation_run" if isinstance(face, dict) else "face_missing"
        server = startup.get("scene_server")
        if not (isinstance(server, dict) and server.get("state") == "ready" and server.get("problem") is None
                and server.get("layers") == scene_layers and server.get("vision_on_gpu") is True):
            return "scene_server_not_ready"
        chat = startup.get("chat")
        if not isinstance(chat, dict):
            return "chat_missing"
        if chat != CHAT_VALIDATION_AVAILABLE:
            return f"chat_not_available_as_validation_run:{chat.get('state')}"
        return None
    if required == "landmark_diagnostic":  # session 90: the face loaded, and the marker exactly as pinned
        if startup.get("face_problem") is not None:
            return f"face_problem:{startup.get('face_problem')}"
        marker = face.get("landmark_diagnostic") if isinstance(face, dict) else None
        if not isinstance(marker, dict):
            return "marker_missing"
        return None if marker == LANDMARK_MARKER else "marker_not_pinned"
    marker = face.get("tracking_diagnostic") if isinstance(face, dict) else None
    if not isinstance(marker, dict):
        return "marker_missing"
    tracker = marker.get("tracker")
    if not isinstance(tracker, dict):
        return "tracker_settings_missing"
    for key in TRACKING_SETTING_KEYS:
        value = tracker.get(key)
        if key == "fuse_score":
            if not isinstance(value, bool):
                return f"tracker_settings_incomplete:{key}"
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"tracker_settings_incomplete:{key}"
    return None


def stream_route(startup: dict) -> str:
    """Which stream profile a starting line shows: ``sub`` (no stream entry and no head-window face), ``main_single``
    (both), or ``inconsistent`` (one without the other)."""
    stream, face = startup.get("stream"), startup.get("face")
    windows = isinstance(face, dict) and face.get("detection") == "head_windows"
    main = isinstance(stream, dict) and stream.get("profile") == STREAM_MAIN_SINGLE
    if "stream" not in startup and not windows:
        return "sub"
    return "main_single" if main and windows else "inconsistent"


def _stream_problem(stream: Any) -> str | None:
    """Route B's stream entry before T0: the profile, the decoder threads (1 to 4), the working copy and its resize as
    configured, and the tracker's frame-counted settings recorded with no rate measured yet."""
    if not isinstance(stream, dict):
        return "stream_missing"
    threads = stream.get("decode_threads")
    if type(threads) is not int or not 1 <= threads <= 4:
        return "stream_decode_threads"
    if stream.get("working") != WORKING_SIZE or stream.get("resize") != "area":
        return "stream_working_copy"
    counted = stream.get("tracker_frame_counted")
    frames = counted.get("frames") if isinstance(counted, dict) else None
    if not (isinstance(frames, dict) and frames.get("bytetrack_lost_track") == BYTETRACK_LOST_TRACK_FRAMES
            and counted.get("processed_fps") is None and counted.get("seconds") is None):
        return "stream_tracker_frame_counted"
    return None


def read_run_output(path: Path) -> dict[str, Any]:
    """Line counts (D48: every stdout line is JSON), the starting summary and the stopped line."""
    lines = json_lines = 0
    starting = stopped = None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            item = json.loads(line)
        except ValueError:
            continue
        json_lines += 1
        if isinstance(item, dict) and item.get("run") == "starting" and starting is None:
            starting = _starting_summary(item.get("startup") or {})
        elif isinstance(item, dict) and item.get("run") == "stopped":
            shutdown = item.get("shutdown") or {}
            stopped = {"all_stopped": shutdown.get("all_stopped"), "stopped": shutdown.get("stopped"),
                       "database_closed": item.get("database_closed"),
                       "signals_not_recorded": shutdown.get("signals_not_recorded")}
    return {"lines": lines, "json_lines": json_lines, "non_json_lines": lines - json_lines,
            "starting": starting, "stopped": stopped}


def last_error_label(path: Path) -> str | None:
    """The runtime's last ``run: <label>`` line on stderr (a refusal names its reason there), else None."""
    try:
        lines = [line for line in path.read_text(errors="replace").splitlines() if line.startswith("run: ")]
    except OSError:
        return None
    return lines[-1][len("run: "):][:400] if lines else None


# ---------------------------------------------------------------- supervision


def preflight(backend, out: Path, step4_headroom: bool = False) -> str | None:
    if out.exists():
        return "out_exists"
    if backend.thp_self() != 1:
        return "thp_enabled_not_1"
    if backend.llama_pids():
        return "llama_server_running"
    if backend.listeners():
        return "port_in_use"
    sample = backend.sample()
    if oc.memory_problem(sample):
        return "memory_guard_already_met"
    if step4_headroom and oc.baseline_problem(sample):  # step 4's admission headroom, its values unchanged
        return "initial_headroom_refused"
    return None


def owned_servers(backend) -> list[dict[str, Any]]:
    return [info for info in backend.owned() if info["comm"] == "llama-server"]


def stop_owned_servers(backend) -> tuple[int, list[dict[str, Any]]]:
    """SIGTERM to the llama-servers this run owns, SIGKILL to those left KILL_GRACE_S later, and KILL_GRACE_S for
    those to go: (how many there were, those still running). One handed to this process meanwhile gets the signal of
    the stage it appears in. No other process is signalled, whatever its name."""
    seen: set[tuple[int, int]] = set()
    for signum in (signal.SIGTERM, signal.SIGKILL):
        sent: set[tuple[int, int]] = set()
        deadline = None
        while True:
            servers = owned_servers(backend)
            for info in servers:
                key = (info["pid"], info["start"])
                seen.add(key)
                if key not in sent:
                    sent.add(key)
                    backend.signal(info, signum)
            deadline = backend.clock() + KILL_GRACE_S if deadline is None else deadline
            if not servers or backend.clock() >= deadline:
                break
            backend.sleep(0.1)
    return len(seen), owned_servers(backend)


def stop_after_failure(child, backend, stop_grace_s: float) -> None:
    """The guard itself failed: what it started must not go on without it. The same order and the same waits as a
    stop: SIGINT to the runtime, its group after the grace, then the llama-servers this run owns."""
    child.signal(signal.SIGINT)
    deadline = backend.clock() + stop_grace_s
    while child.poll() is None and backend.clock() < deadline:
        backend.sleep(0.1)
    if child.poll() is None:
        child.signal_group(signal.SIGTERM)
        deadline = backend.clock() + KILL_GRACE_S
        while child.group_alive() and backend.clock() < deadline:
            backend.sleep(0.1)
        child.signal_group(signal.SIGKILL)
        deadline = backend.clock() + KILL_GRACE_S
        while child.poll() is None and backend.clock() < deadline:
            backend.sleep(0.1)
    stop_owned_servers(backend)


def supervise(argv: list[str], out: Path, *, backend=None, **options) -> dict:
    """One guarded run (the options are ``_supervise``'s). Ownership of what the run starts is set up before the
    launch and given back afterwards, whatever happens in between."""
    backend = backend or SystemBackend()
    try:
        return _supervise(argv, out, backend=backend, **options)
    finally:
        backend.disown()


def _supervise(argv: list[str], out: Path, *, duration_s: float, from_launch: bool = False,
               ready_timeout_s: float = READY_TIMEOUT_S, stop_grace_s: float = STOP_GRACE_S,
               status_at: tuple[tuple[float, str], ...] = (), step4_headroom: bool = False, backend=None,
               interrupted=lambda: False, cues: tuple[tuple[float, str, str], ...] = (),
               say=lambda text: (sys.stderr.write(text + "\n"), sys.stderr.flush()),
               require_ready: str | None = None, scene_layers: str | None = None,
               shadow_digest: str | None = None) -> dict:
    result: dict[str, Any] = {"schema_version": 1, "mode": "step5_guard", "status": None,
                              "parameters": {"duration_s": duration_s, "from_launch": from_launch,
                                             "ready_timeout_s": ready_timeout_s, "stop_grace_s": stop_grace_s,
                                             "step4_headroom": step4_headroom,
                                             "status_at": [{"at_s": at, "name": name} for at, name in status_at],
                                             "cues": [{"at_s": at, "name": name} for at, name, _ in cues],
                                             "require_ready": require_ready,
                                             **({} if scene_layers is None else {"scene_layers": scene_layers}),
                                             **({} if shadow_digest is None else {"shadow_digest": shadow_digest})},
                              "limits": {"pressure_stop_bytes": oc.PRESSURE_STOP, "mem_free_floor_bytes": oc.FREE_FLOOR,
                                         "mem_available_floor_bytes": oc.AVAILABLE_FLOOR,
                                         "swap_counters": "unchanged", "max_sample_gap_s": 0.5}}
    refused = preflight(backend, out, step4_headroom) or backend.own()  # no launch without ownership of its leftovers
    if refused:
        result["status"] = f"refused:{refused}"
        result["preflight_sample"] = oc.safe_sample(backend.sample())
        if refused != "out_exists":  # keep the refusal as evidence; never write into an existing directory
            out.mkdir(mode=0o700, parents=False)
        return result
    out.mkdir(mode=0o700, parents=False)
    baseline = backend.sample()
    result["baseline"] = oc.safe_sample(baseline)
    guard_file = open(out / "guard.jsonl", "w")
    guard = oc.PressureGuard(baseline, emit=lambda reading: guard_file.write(json.dumps(reading) + "\n"))
    stdout, stderr = out / "run.jsonl", out / "run.err"
    launch = backend.clock()
    result["child"] = {"launched_utc": backend.utc()}
    child = backend.spawn(argv, stdout, stderr)
    ready_at = None
    not_ready = False  # the starting line came, but the readiness requirement was not met: stop, no cue, no capture
    read_offset = 0
    stop = {"requested_by": None, "requested_at_s": None, "exited_after_stop_s": None, "forced": False}
    pending = sorted(status_at)
    cue_queue = sorted(cues)
    printed: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    workers: list[threading.Thread] = []
    returncode = None
    gaps, previous, min_available = [], None, None

    def capture(at: float, name: str, taken: float) -> None:  # off the sampling loop: a slow page never stalls it
        label, body = backend.fetch()
        if label == "saved":
            (out / f"status-{name}.json").write_bytes(body)
        captures.append({"name": name, "at_s": at, "taken_after_launch_s": taken, "status": label,
                         "bytes": len(body), "listeners": backend.listeners()})
    try:
        while True:
            now = backend.clock()
            sample = backend.sample()
            guard.observe(sample)
            available = sample.get("MemAvailable")
            if type(available) is int:
                min_available = available if min_available is None else min(min_available, available)
            stamp = sample.get("t_mono")
            if isinstance(stamp, (int, float)):
                if previous is not None:
                    gaps.append(stamp - previous)
                previous = stamp
            if ready_at is None:
                try:
                    with open(stdout, "rb") as handle:
                        handle.seek(read_offset)
                        chunk = handle.read()
                except OSError:
                    chunk = b""
                complete = chunk[:chunk.rfind(b"\n") + 1]
                read_offset += len(complete)
                for line in complete.splitlines():
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict) and item.get("run") == "starting":
                        problem = readiness_problem(require_ready, item.get("startup"),
                                                    SCENE_LAYERS if scene_layers is None else scene_layers,
                                                    shadow_digest)
                        if require_ready is not None:
                            result["readiness"] = {"required": require_ready, "met": problem is None,
                                                   "reason": problem, "starting_after_s": round(now - launch, 3)}
                        if problem is None:
                            ready_at = now
                            result["child"]["ready_after_s"] = round(now - launch, 3)
                            result["child"]["ready_mono_s"] = round(now, 6)
                        else:
                            not_ready = True
                        break
            returncode = child.poll()
            if returncode is not None:
                break
            base = launch if from_launch else ready_at
            if stop["requested_by"] is None:
                reason = None
                if guard.trigger is not None:
                    reason = "guard_stop"
                elif interrupted():
                    reason = "operator_interrupt"
                elif not_ready:
                    reason = "readiness_refused"
                elif ready_at is None and not from_launch and now - launch >= ready_timeout_s:
                    reason = "ready_timeout"
                elif base is not None and now - base >= duration_s:
                    reason = "duration_elapsed"
                if reason:
                    child.signal(signal.SIGINT)  # sentinel run's Ctrl-C path: a bounded, clean shutdown
                    stop.update(requested_by=reason, requested_at_s=round(now - launch, 3))
            elif not stop["forced"] and now - launch - stop["requested_at_s"] >= stop_grace_s:
                stop["forced"] = True
                child.signal_group(signal.SIGTERM)
                deadline = backend.clock() + KILL_GRACE_S
                while child.group_alive() and backend.clock() < deadline:
                    backend.sleep(0.1)
                child.signal_group(signal.SIGKILL)
            while cue_queue and base is not None and stop["requested_by"] is None and now - base >= cue_queue[0][0]:
                at, name, text = cue_queue.pop(0)
                elapsed = int(now - base)
                say(f"\a[{elapsed // 60}:{elapsed % 60:02d}] {text}")
                printed.append({"name": name, "at_s": at, "printed_after_launch_s": round(now - launch, 3),
                                "printed_mono_s": round(now, 6)})
            while pending and base is not None and stop["requested_by"] is None and now - base >= pending[0][0]:
                at, name = pending.pop(0)
                worker = threading.Thread(target=capture, args=(at, name, round(now - launch, 3)), daemon=True)
                worker.start()
                workers.append(worker)
            backend.sleep(SAMPLE_S)
    except BaseException:
        stop_after_failure(child, backend, stop_grace_s)
        raise
    finally:
        guard_file.close()
        for worker in workers:
            worker.join(STATUS_TIMEOUT_S + 1.0)
    ended = backend.clock()
    if require_ready is not None and "readiness" not in result:  # no starting line: never ready, whatever the exit
        result["readiness"] = {"required": require_ready, "met": False, "reason": "no_starting_line",
                               "starting_after_s": None}
    if stop["requested_at_s"] is not None:
        stop["exited_after_stop_s"] = round(ended - launch - stop["requested_at_s"], 3)
    result["child"].update(returncode=returncode, exited_utc=backend.utc(), ran_s=round(ended - launch, 3))
    for at, name in pending:
        captures.append({"name": name, "at_s": at, "status": "not_reached", "bytes": 0, "listeners": None})
    for at, name, _ in cue_queue:
        printed.append({"name": name, "at_s": at, "printed_after_launch_s": None, "printed_mono_s": None})
    orphans, left = stop_owned_servers(backend)  # this run's own, by descent; never by name
    others = [info for info in backend.owned() if info["comm"] != "llama-server"]
    # a llama-server of this user that is not shown to be this run's: counted, left running, never signalled
    unresolved = backend.unresolved_servers()
    group_left = child.group_alive() or bool(others)
    after = backend.listeners()
    result.update(
        stop=stop,
        guard={"samples": guard.samples, "largest_gap_s": round(max(gaps), 3) if gaps else None,
               "peak_pressure_bytes": guard.peak, "min_mem_free_bytes": guard.minimum_free,
               "min_mem_available_bytes": min_available, "trigger": guard.trigger,
               "trigger_after_stop_request": bool(guard.trigger and stop["requested_by"] not in (None, "guard_stop"))},
        captures=captures,
        cues=printed,
        run_output=read_run_output(stdout),
        run_error_label=last_error_label(stderr),
        leftovers={"llama_server_after_exit": orphans, "llama_server_left": len(left),
                   "llama_server_unresolved": len(unresolved),
                   "runtime_group_left": group_left, "listeners_after": after},
    )
    clear = not left and not group_left and not after
    if not clear:
        status = "cleanup_failed"
    elif stop["forced"]:
        status = "forced_stop"
    elif stop["requested_by"] in ("guard_stop", "ready_timeout", "operator_interrupt", "readiness_refused"):
        status = stop["requested_by"]
    elif stop["requested_by"] == "duration_elapsed":
        status = "duration_stop"
    else:
        status = "child_exited"
    result["status"] = status
    result["cleanup_clear"] = clear and not orphans and not unresolved
    return result


def _status_at(text: str) -> tuple[float, str]:
    seconds, sep, name = text.partition(":")
    if not sep or not name.replace("-", "").isalnum() or len(name) > 32:
        raise argparse.ArgumentTypeError("use SECONDS:NAME with a short alphanumeric name")
    value = float(seconds)
    if not 0 <= value <= MAX_DURATION_S:
        raise argparse.ArgumentTypeError(f"SECONDS must be within 0-{MAX_DURATION_S:g}")
    return value, name


def _cue_at(text: str) -> tuple[float, str, str]:
    seconds, sep, rest = text.partition(":")
    name, sep2, words = rest.partition(":")
    if not (sep and sep2) or not name.replace("-", "").isalnum() or len(name) > 32:
        raise argparse.ArgumentTypeError("use SECONDS:NAME:TEXT with a short alphanumeric name")
    if not words or len(words) > 160 or not all(32 <= ord(ch) < 127 for ch in words):
        raise argparse.ArgumentTypeError("TEXT: 1-160 printable ASCII characters")
    try:
        value = float(seconds)
    except ValueError:
        raise argparse.ArgumentTypeError("SECONDS must be a number") from None
    if not 0 <= value <= MAX_DURATION_S:
        raise argparse.ArgumentTypeError(f"SECONDS must be within 0-{MAX_DURATION_S:g}")
    return value, name, words


def _layers(text: str) -> str:
    """A fully offloaded count, N/N with N from 1 to 9999: the readiness never asks for a partial offload."""
    loaded, sep, total = text.partition("/")
    if not (sep and loaded.isdigit() and loaded == total and len(loaded) <= 4 and int(loaded) > 0
            and loaded == str(int(loaded))):
        raise argparse.ArgumentTypeError("use N/N, the fully offloaded layer count (for example 25/25)")
    return text


def _seconds(limit: float):
    def parse(text: str) -> float:
        value = float(text)
        if not 0 < value <= limit:
            raise argparse.ArgumentTypeError(f"must be within (0, {limit:g}]")
        return value
    return parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a new directory for this part's evidence")
    parser.add_argument("--duration-s", type=_seconds(MAX_DURATION_S), required=True)
    parser.add_argument("--from-launch", action="store_true",
                        help="count the duration and the status offsets from launch instead of from readiness")
    parser.add_argument("--ready-timeout-s", type=_seconds(MAX_READY_TIMEOUT_S), default=READY_TIMEOUT_S)
    parser.add_argument("--stop-grace-s", type=_seconds(300.0), default=STOP_GRACE_S)
    parser.add_argument("--status-at", type=_status_at, action="append", default=[], metavar="SECONDS:NAME")
    parser.add_argument("--cue-at", type=_cue_at, action="append", default=[], metavar="SECONDS:NAME:TEXT",
                        help="print TEXT to the operator's terminal at this offset (counted like the duration)")
    parser.add_argument("--step4-headroom", action="store_true",
                        help="also refuse before launch unless step 4's admission headroom holds (operator_check: "
                             "pressure below 2.0 GB, MemFree at least 3.5 GB, MemAvailable at least 4.0 GB)")
    parser.add_argument("--require-ready", choices=READINESS, default=None, metavar="REQUIREMENT",
                        help="readiness needs more than the starting line: tracking_diagnostic = no detector problem "
                             "and the diagnostic's marker with the tracker's settings; landmark_diagnostic = no "
                             "detector or face problem and the landmark marker as pinned; otherwise the runtime is "
                             "stopped at once, no cue is printed and the exit status is 1")
    parser.add_argument("--scene-layers", type=_layers, default=None, metavar="N/N",
                        help="face_chat: the scene server's fully offloaded layer count for the selected model set "
                             f"(default {SCENE_LAYERS}); recorded under parameters and required by the readings")
    parser.add_argument("--shadow-digest", default=None, metavar="SHA256",
                        help="face_windows_shadow (required with it, refused otherwise): the frozen candidate's "
                             "prototypes digest; a run comparing another working gallery stops before T0")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the sentinel run command")
    args = parser.parse_args(argv)
    if (args.require_ready == "face_windows_shadow") != (args.shadow_digest is not None):
        parser.error("--shadow-digest goes with --require-ready face_windows_shadow, and only with it")
    if args.shadow_digest is not None and not SHADOW_DIGEST.fullmatch(args.shadow_digest):
        parser.error("--shadow-digest is 64 lowercase hexadecimal characters")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    flag = {"set": False}

    def interrupt(*_):
        flag["set"] = True

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = supervise(command, args.out, duration_s=args.duration_s, from_launch=args.from_launch,
                           ready_timeout_s=args.ready_timeout_s, stop_grace_s=args.stop_grace_s,
                           status_at=tuple(args.status_at), step4_headroom=args.step4_headroom,
                           interrupted=lambda: flag["set"], cues=tuple(args.cue_at),
                           require_ready=args.require_ready, scene_layers=args.scene_layers,
                           shadow_digest=args.shadow_digest)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    text = json.dumps(result, indent=2)
    if result["status"] != "refused:out_exists" and args.out.is_dir():
        (args.out / "result.json").write_text(text + "\n")
    print(text)
    readiness = result.get("readiness")
    clean = result.get("cleanup_clear") is True and (
        result["status"] == "child_exited"
        or (result["status"] == "duration_stop" and (result.get("child") or {}).get("returncode") == 0)
    ) and (readiness is None or readiness.get("met") is True)  # a clean exit alone never implies readiness
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
