"""Process memory policies shared by ``sentinel run`` and the profiler's workload (D58).

Two policies. A resource profile records the ones it was measured with, and ``sentinel run`` compares them with
its own before scene admission (``demo_runtime.profile_mismatch``):

- **THP** (``thp``): ``system`` keeps the system's transparent-huge-page setting. ``workload_disabled`` has the
  process that loads the detector (the runtime, or the profiler's workload) disable transparent huge pages for
  itself alone with ``prctl(PR_SET_THP_DISABLE, 1)``, after the scene server is spawned and before CUDA's cuInit or
  any model library is imported. ``disable_thp_for_this_process`` verifies it three ways: the call's return value,
  ``PR_GET_THP_DISABLE`` and /proc/self/status ``THP_enabled``. The kernel copies the flag to the children a process
  starts afterwards (session 43's check on this device's kernel), so a scene server must be spawned before it:
  ``LlamaServerProcess`` refuses a later spawn when asked to keep the system setting.
- **Model-file release** (``model_file_release``): ``none``, or ``post_load``: after each component's load and a
  settle of RELEASE_SETTLE_S, that component's model files (RELEASE_FILE_ROLES) are released from the page cache
  with ``posix_fadvise(DONTNEED)``. Every call must return 0; otherwise the policy was not applied, and the run or
  the startup stops (``release_problem``).

Standard library only: importing this module imports no model library, CUDA or OpenCV. Nothing here writes a
system-wide setting, another process's flag or a file.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# ---------------------------------------------------------------- the two policies

THP_SYSTEM = "system"
THP_WORKLOAD_DISABLED = "workload_disabled"
RELEASE_NONE = "none"
RELEASE_POST_LOAD = "post_load"
THP_POLICIES = (THP_SYSTEM, THP_WORKLOAD_DISABLED)
RELEASE_POLICIES = (RELEASE_NONE, RELEASE_POST_LOAD)


@dataclass(frozen=True)
class MemoryPolicy:
    """The memory configuration a resource profile was measured with and ``sentinel run`` applies (D58)."""

    thp: str = THP_SYSTEM
    model_file_release: str = RELEASE_NONE

    def __post_init__(self) -> None:
        if self.thp not in THP_POLICIES:
            raise ValueError(f"unknown THP policy {self.thp!r}")
        if self.model_file_release not in RELEASE_POLICIES:
            raise ValueError(f"unknown model-file release policy {self.model_file_release!r}")

    def describe(self) -> str:
        return f"THP {self.thp}, model-file release {self.model_file_release}"

    def labels(self) -> dict[str, str]:
        return {"thp": self.thp, "model_file_release": self.model_file_release}


DEFAULT_POLICY = MemoryPolicy()  # what every earlier profile and the default runtime use
CANDIDATE_POLICY = MemoryPolicy(THP_WORKLOAD_DISABLED, RELEASE_POST_LOAD)  # D58


def policy_from_flags(*, workload_thp_disable: bool, post_load_release: bool) -> MemoryPolicy:
    """The policy that a profiler run's or ``sentinel run``'s two opt-in flags select."""
    return MemoryPolicy(THP_WORKLOAD_DISABLED if workload_thp_disable else THP_SYSTEM,
                        RELEASE_POST_LOAD if post_load_release else RELEASE_NONE)


# ---------------------------------------------------------------- THP: this process alone (D57's call, shared)

PR_SET_THP_DISABLE = 41  # <linux/prctl.h>; the installed kernel's uapi header has the same values (session 43)
PR_GET_THP_DISABLE = 42
MODEL_MODULES = ("numpy", "cv2", "torch", "ultralytics", "tensorflow", "keras", "deepface")  # none may be loaded yet
PROC_READ_LIMIT_BYTES = 16 << 10  # /proc/<pid>/status and smaps_rollup are about 1.5 KB each
THP_DISABLE_REASONS = ("model_modules_loaded", "prctl_unavailable", "set_failed", "get_mismatch", "status_unavailable",
                       "status_mismatch", "anon_huge_pages_unavailable")


def libc_prctl(load: Callable = ctypes.CDLL) -> Callable[[int, int], tuple[int, int]] | None:
    """prctl(2) through libc as ``call(option, arg2) -> (return value, errno)`` with arg3-arg5 0, or None without it."""
    try:
        function = load(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    function.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    function.restype = ctypes.c_int

    def call(option: int, arg2: int) -> tuple[int, int]:
        ctypes.set_errno(0)
        value = function(option, arg2, 0, 0, 0)
        return value, ctypes.get_errno()

    return call


def _proc_field(path: str, name: str, opener=open) -> str | None:
    """One ``name:`` line's value from a small /proc file, or None when unreadable, oversized or absent."""
    try:
        with opener(path, "rb") as handle:
            data = handle.read(PROC_READ_LIMIT_BYTES + 1)
    except OSError:
        return None
    if len(data) > PROC_READ_LIMIT_BYTES:
        return None
    for line in data.decode("ascii", "replace").splitlines():
        key, _, value = line.partition(":")
        if key == name:
            return value.strip()
    return None


def _kilobytes(value: str | None) -> int | None:
    parts = (value or "").split()
    return int(parts[0]) * 1024 if len(parts) == 2 and parts[0].isascii() and parts[0].isdigit() and parts[1] == "kB" \
        else None


def disable_thp_for_this_process(*, prctl: Callable[[int, int], tuple[int, int]] | None = None,
                                 binding: Callable[[], Callable[[int, int], tuple[int, int]] | None] | None = None,
                                 modules: Mapping[str, object] | None = None, opener=open,
                                 clock: Callable[[], float] = time.monotonic) -> dict[str, object]:
    """``PR_SET_THP_DISABLE`` for this process alone, then verified, with numbers and fixed labels only (D57, D58).

    Refused (nothing set) when a model library is already imported. ``verified`` needs the call to return 0,
    ``PR_GET_THP_DISABLE`` to return 1, /proc/self/status ``THP_enabled`` to read 0 and this process's AnonHugePages
    (smaps_rollup; huge pages made before the call stay) to be readable; ``reason`` names the first check that
    failed. ``prctl`` is the call to use; without it, ``binding`` (default ``libc_prctl``) provides it. The kernel
    flag is inherited by this process's later children; nothing system-wide is touched."""
    loaded = [name for name in MODEL_MODULES if name in (sys.modules if modules is None else modules)]
    record: dict[str, object] = {"t_mono": round(clock(), 3), "requested": True, "model_modules_loaded": loaded,
                                 "set_rc": None, "set_errno": None, "get_value": None, "thp_enabled": None,
                                 "anon_huge_pages_bytes": None, "verified": False, "reason": None}
    if loaded:
        record["reason"] = "model_modules_loaded"
        return record
    call = prctl if prctl is not None else (binding or libc_prctl)()
    if call is None:
        record["reason"] = "prctl_unavailable"
        return record
    rc, error = call(PR_SET_THP_DISABLE, 1)
    record.update(set_rc=rc, set_errno=errno.errorcode.get(error) if error else None)
    record["get_value"] = call(PR_GET_THP_DISABLE, 0)[0]
    status = _proc_field("/proc/self/status", "THP_enabled", opener)
    record["thp_enabled"] = int(status) if status in ("0", "1") else None
    record["anon_huge_pages_bytes"] = _kilobytes(_proc_field("/proc/self/smaps_rollup", "AnonHugePages", opener))
    checks = (("set_failed", rc == 0), ("get_mismatch", record["get_value"] == 1),
              ("status_unavailable", record["thp_enabled"] is not None), ("status_mismatch", record["thp_enabled"] == 0),
              ("anon_huge_pages_unavailable", record["anon_huge_pages_bytes"] is not None))
    record["reason"] = next((reason for reason, passed in checks if not passed), None)
    record["verified"] = record["reason"] is None
    return record


def read_thp_enabled(pid: int | str | None, *, opener=open) -> int | None:
    """``THP_enabled`` (0 or 1) from /proc/<pid>/status, read-only; None when unreadable, unparsed or without a pid."""
    if pid is None:
        return None
    value = _proc_field(f"/proc/{pid}/status", "THP_enabled", opener)
    return int(value) if value in ("0", "1") else None


def read_anon_huge_pages(pid: int | str | None, *, opener=open) -> int | None:
    """A process's AnonHugePages in bytes from /proc/<pid>/smaps_rollup, read-only; None when unavailable."""
    if pid is None:
        return None
    return _kilobytes(_proc_field(f"/proc/{pid}/smaps_rollup", "AnonHugePages", opener))


# ---------------------------------------------------------------- post-load model-file release (MR1's call, shared)

FADVISE_CALL = "posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)"
RELEASE_RESULTS = ("returned_0", "returned_error", "open_failed", "unsupported")
# Each component's model files, released after its load and settle, in this order. The profiler labels its files the
# same way (demo_profile.model_files); sentinel run has the scene and detector components (no face model is wired).
RELEASE_FILE_ROLES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "scene": ("llm", "mmproj"),
    "detector": ("engine",),
    "face": ("facenet512_weights.h5", "face_detection_yunet_2023mar.onnx"),
})
RELEASE_SETTLE_S = 15.0  # after each load, before its release: step 4's --settle-s (the profiler's settle phases)


def release_file_cache(path: Path, *, advise=None, clock=time.monotonic) -> dict[str, object]:
    """The release call on one whole file, with its exact outcome (MR1; D54, D58).

    ``returncode`` is posix_fadvise's own return value (0, or the error number it reported), or
    open()'s errno when the file could not be opened. A 0 return means the kernel accepted the advice,
    not that pages were dropped: mapped, dirty or locked pages stay. Needs no root; nothing is unmapped.
    """
    advise = advise or getattr(os, "posix_fadvise", None)
    record: dict[str, object] = {"name": Path(path).name, "bytes": None, "call": FADVISE_CALL,
                                 "result": None, "returncode": None, "error": None}
    started = clock()
    if advise is None:
        record["result"] = "unsupported"
    else:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as exc:
            record.update(result="open_failed", returncode=exc.errno, error=errno.errorcode.get(exc.errno, "unknown"))
        else:
            try:
                record["bytes"] = os.fstat(fd).st_size
                advise(fd, 0, 0, getattr(os, "POSIX_FADV_DONTNEED", 4))
                record.update(result="returned_0", returncode=0)
            except OSError as exc:
                record.update(result="returned_error", returncode=exc.errno,
                              error=errno.errorcode.get(exc.errno, "unknown"))
            finally:
                os.close(fd)
    record["elapsed_s"] = round(clock() - started, 6)
    return record


def release_problem(component: str, records: Mapping[str, Any]) -> str | None:
    """None when ``records`` hold exactly this component's eligible files and every call returned 0; else a label.

    The same check for the profiler's candidate run and for ``sentinel run`` (D58): ``unknown_component``,
    ``files_not_eligible`` (a file missing or extra), or ``<file role>:<result>`` for the first call that did not
    return 0, in RELEASE_FILE_ROLES order. Missing is never met."""
    roles = RELEASE_FILE_ROLES.get(component)
    if roles is None:
        return "unknown_component"
    if not isinstance(records, Mapping) or set(records) != set(roles):
        return "files_not_eligible"
    for role in roles:
        item = records[role]
        result = item.get("result") if isinstance(item, Mapping) else None
        if result != "returned_0":
            return f"{role}:{result if result in RELEASE_RESULTS else 'unknown'}"
    return None


def release_component_files(component: str, files: Mapping[str, Path], *,
                            release: Callable[[Path], dict[str, object]] | None = None) -> dict[str, object]:
    """Release each of ``component``'s eligible files in RELEASE_FILE_ROLES order and judge the calls.

    ``files`` maps each role to its path; anything else is refused before any call (``files_not_eligible``)."""
    roles = RELEASE_FILE_ROLES.get(component)
    if roles is None or set(files) != set(roles):
        problem = "unknown_component" if roles is None else "files_not_eligible"
        return {"component": component, "files": {}, "verified": False, "problem": problem}
    release = release or release_file_cache
    records = {role: release(Path(files[role])) for role in roles}
    problem = release_problem(component, records)
    return {"component": component, "files": records, "verified": problem is None, "problem": problem}
