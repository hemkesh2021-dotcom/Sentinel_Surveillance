#!/usr/bin/env python3
"""One build step of the personal-beta package under the memory monitor (lib/guarded_run.py).

guarded-build.sh starts this after checking the package and setting the installation's own cache and temporary
locations. The build scripts are not changed by being run here: `build-engine.sh` and `build-llama.sh` are started
exactly as you would start them yourself, with the monitor watching the device's memory beside them.

  run    --package DIR --step engine|llama [--jobs N] [--min-start-free-bytes N] [--check]
  stop   --package DIR      from a second shell: a clean stop of the step that is running
  status --package DIR      what the last guarded build wrote

What `run` does, in this order, and nothing else:
 1. Start conditions. Not root; no other guarded build; no model server, Sentinel runtime or build of this user
    running; the step's output not there yet; memory (read last). While one fails nothing is started. Nothing is
    ever stopped, closed or freed for you: closing a desktop and releasing the page cache are your own steps
    (QUICKSTART.md, 'Building on this device').
 2. You type BUILD.
 3. The step's own script under the monitor, in a session of its own. The monitor samples memory five times a second
    and stops the step (SIGTERM to every process the step owns, SIGKILL to those left after 20 s) at its limits:
      engine  below 1 GiB MemFree, below 2 GiB MemAvailable, or after one hour
      llama   below 2 GiB MemAvailable, or after three hours
    These are stops, not requirements: they say when a build is ended, not what a build needs.
 4. RESULT.txt, the step's log, the memory samples and the monitor's result in <data dir>/builds/<step>-<UTC>/.

The engine step also asks for 5 GiB of MemFree before it starts (5,368,709,120 B; --min-start-free-bytes chooses
another figure, not below the script's own 1.5 GB). That figure is a chosen preparation condition. It is not a
measured minimum: the least the export needs has not been measured on any device.

A stop (Ctrl-C once here, or `stop` from a second shell) is sent to the step's first process; the monitor then ends
everything the step owns. A terminal that goes away does not end a started step: the result is in the files. Only
processes that this run's monitor started are ever signalled, each identified by its number and start time (and, when
the step's first process is gone, by working inside this installation). It never uses sudo.

Standard library only. The supervision is the installation rehearsal's (the monitor is that file, unchanged).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta  # noqa: E402  the package's own bookkeeping: where things are, and its refusal type

GIB = 1024 ** 3
STEPS = {
    "engine": {"script": "build-engine.sh", "output": ("engine", "yolov8n.engine"), "stdin_text": "BUILD\n",
               "min_free_bytes": GIB, "min_available_bytes": 2 * GIB, "timeout_s": 3600,
               "start_free_bytes": 5 * GIB, "script_min_free_bytes": 1500000000,
               "what": "the detector engine export"},
    "llama": {"script": "build-llama.sh", "output": ("llama.cpp", "build", "bin", "llama-server"), "stdin_text": None,
              "min_free_bytes": 0, "min_available_bytes": 2 * GIB, "timeout_s": 10800,
              "start_free_bytes": 0, "script_min_free_bytes": 0,
              "what": "the llama.cpp build"},
}
TERM_GRACE_S, KILL_GRACE_S = 20.0, 10.0
INTERPRETER = re.compile(r"^(python[0-9.]*|bash|sh|dash)$")
OPTIONS_WITH_VALUE = ("-W", "-X", "-o", "-O", "--rcfile", "--init-file")
BUILD_SCRIPTS = ("build-engine.sh", "build-llama.sh", "guarded_run.py")  # another front end is known by the lock
GRAPHICAL = ("Xorg", "Xwayland", "Xtigervnc", "Xvnc", "gnome-shell", "xfce4-session", "xfwm4", "gdm-x-session",
             "gdm-wayland-ses", "lightdm", "sddm", "kwin_x11", "kwin_wayland", "plasmashell")
EXIT = {"built": 0, "refused": 10, "active": 11, "stopped_by_monitor": 20, "failed": 21, "stopped_by_operator": 22,
        "refused_at_monitor_start": 23, "monitor_failed": 24, "leftovers": 5, "error": 50}
FLAG = {"stop": False}
LOG = {"path": None}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(text: str, stamp: bool = True) -> None:
    line = f"{utc()} {text}" if stamp else text
    if LOG["path"] is not None:
        try:
            with open(LOG["path"], "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
    try:
        print(line, flush=True)
    except (OSError, ValueError):
        pass  # the terminal went away; the files keep everything


def gib(count: int | None) -> str:
    return "?" if count is None else f"{count} B ({count / GIB:.2f} GiB)"


# ---------------------------------------------------------------- processes, by exact identity

def proc(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        return {"pid": pid, "comm": text[text.index("(") + 1:text.rindex(")")], "state": fields[0],
                "ppid": int(fields[1]), "sid": int(fields[3]), "start": int(fields[19]),
                "uid": os.stat(f"/proc/{pid}").st_uid}
    except (OSError, ValueError, IndexError):
        return None


def procs() -> list[dict]:
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            info = proc(int(entry))
            if info is not None and info["state"] != "Z":
                found.append(info)
    return found


def argv_of(pid: int) -> list[str]:
    """The arguments as separate elements, never one joined line: a shell whose command text merely mentions a
    name is not that program."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return [part.decode("utf-8", "replace") for part in parts]


def started_as(argv: list[str]) -> tuple[str, str]:
    """('script', name) for the script an interpreter runs, ('module', name) for the module after -m, else ('', '')."""
    if not argv or not INTERPRETER.match(os.path.basename(argv[0])):
        return "", ""
    skip = False
    for index, arg in enumerate(argv[1:], 1):
        if skip:
            skip = False
            continue
        if arg == "-m":
            return "module", argv[index + 1] if index + 1 < len(argv) else ""
        if arg.startswith("-m") and not arg.startswith("--"):
            return "module", arg[2:]
        if arg == "-c":
            return "", ""  # a command text, not a script
        if arg in OPTIONS_WITH_VALUE:
            skip = True
            continue
        if arg.startswith("-"):
            continue
        return "script", os.path.basename(arg)
    return "", ""


def competing() -> list[str]:
    """This user's processes that a build must not run beside: a model server, a Sentinel runtime, another build
    or monitor. Found by process name or by argument position; nothing is signalled."""
    me, uid, found = os.getpid(), os.getuid(), []
    for item in procs():
        if item["pid"] == me or item["uid"] != uid:
            continue
        kind, name = started_as(argv_of(item["pid"]))
        if item["comm"] == "llama-server":
            found.append(f"{item['pid']}(model server llama-server)")
        elif kind == "module" and name == "sentinel.cli":
            found.append(f"{item['pid']}(Sentinel runtime)")
        elif kind == "script" and name in BUILD_SCRIPTS:
            found.append(f"{item['pid']}({name})")
    return found


def send(info: dict, signum: int) -> bool:
    """Signal exactly that process (number and start time), through a handle on it; never a later one."""
    try:
        handle = os.pidfd_open(info["pid"])
    except OSError:
        return False
    try:
        now = proc(info["pid"])
        if now is None or now["start"] != info["start"]:
            return False
        signal.pidfd_send_signal(handle, signum)
        return True
    except OSError:
        return False
    finally:
        os.close(handle)


def inside_installation(pid: int, roots: tuple[str, ...]) -> bool:
    for link in ("cwd", "exe"):
        try:
            target = os.readlink(f"/proc/{pid}/{link}")
        except OSError:
            continue
        if any(target == root or target.startswith(root + "/") for root in roots):
            return True
    return any(part == root or part.startswith(root + "/") for part in argv_of(pid) for root in roots)


def session_members(leader: dict, roots: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """Processes of the session a monitor of this front end started: (verified, not verified). While the session's
    first process is there with its start time, the session is that one. When it is gone, a member must also work
    inside this installation; anything else is reported and never signalled."""
    first = proc(leader["pid"])
    intact = first is not None and first["start"] == leader["start"]
    verified, doubtful = [], []
    for info in procs():
        if info["sid"] != leader["pid"] or info["start"] < leader["start"]:
            continue
        (verified if intact or inside_installation(info["pid"], roots) else doubtful).append(info)
    return verified, doubtful


def cleanup_session(leader: dict, roots: tuple[str, ...]) -> dict:
    """Bounded stop of a monitor's session when its monitor is gone: SIGTERM, grace, SIGKILL, grace, report."""
    report = {"signalled": [], "left": [], "not_verified": []}
    for signum, grace in ((signal.SIGTERM, TERM_GRACE_S), (signal.SIGKILL, KILL_GRACE_S)):
        members, doubtful = session_members(leader, roots)
        report["not_verified"] = [f"{p['pid']}({p['comm']})" for p in doubtful]
        if not members:
            return report
        say(f"cleanup: {signal.Signals(signum).name} to {len(members)} process(es) of session {leader['pid']}: "
            + " ".join(f"{p['pid']}({p['comm']})" for p in members[:12]))
        for info in members:
            if send(info, signum):
                report["signalled"].append(f"{signal.Signals(signum).name}:{info['pid']}({info['comm']})")
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and session_members(leader, roots)[0]:
            time.sleep(0.1)
    report["left"] = [f"{p['pid']}({p['comm']})" for p in session_members(leader, roots)[0]]
    return report


# ---------------------------------------------------------------- where the records are

class Place:
    """The installation this front end works for, and its builds folder in the tester's data directory."""

    def __init__(self, package: str) -> None:
        self.paths = beta.resolved_paths(Path(package))
        self.package = Path(self.paths["PKG"])
        self.prefix = Path(self.paths["PREFIX"])
        self.data = Path(self.paths["DATA_DIR"])
        self.builds = self.data / "builds"
        self.roots = (str(self.prefix), str(self.data))

    def prepare(self) -> None:
        if not self.data.is_dir():
            raise beta.Stop(f"the data directory recorded at installation is not there: {self.data}")
        if self.builds.is_symlink() or (os.path.lexists(self.builds) and not self.builds.is_dir()):
            raise beta.Stop(f"{self.builds} is a link or is not a directory: nothing is written through it. Remove it "
                            "yourself (what it points at is left alone), then run this again")
        if not os.path.lexists(self.builds):
            os.mkdir(self.builds, 0o700)
            os.chmod(self.builds, 0o700)

    def lock(self) -> int | None:
        handle = os.open(self.builds / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(handle)
            return None
        return handle

    def state(self) -> dict:
        try:
            return json.loads((self.builds / "state.json").read_text())
        except (OSError, ValueError):
            return {}

    def save_state(self, **values) -> None:
        state = self.state()
        state.update(values)
        temporary = self.builds / f".state.json.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(state, indent=1) + "\n")
        os.replace(temporary, self.builds / "state.json")

    def output(self, step: str) -> Path:
        return Path(self.paths["ASSETS"]).joinpath(*STEPS[step]["output"])


def stop_requested(place: Place) -> bool:
    return FLAG["stop"] or (place.builds / "STOP-REQUEST").exists()


def consume_stop(place: Place) -> None:
    FLAG["stop"] = False
    try:
        os.unlink(place.builds / "STOP-REQUEST")
    except OSError:
        pass


def on_stop_signal(signum, _frame) -> None:
    FLAG["stop"] = True


def on_ignored_signal(signum, _frame) -> None:
    pass  # a lost terminal or a stray key does not end the run; handlers are not inherited by what is started


# ---------------------------------------------------------------- start conditions

def meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree"):
                values[key] = int(rest.split()[0]) * 1024
    return values


def memory_rows(step: str, start_free: int) -> list[tuple[str, bool, str]]:
    limits, mem = STEPS[step], meminfo()
    rows = []
    if step == "engine":
        rows.append(("memfree_at_least_the_start_figure", mem.get("MemFree", 0) >= start_free,
                     f"MemFree {gib(mem.get('MemFree'))}; start figure in use {gib(start_free)}: a chosen preparation "
                     f"condition, not a measured minimum (the script's own minimum is {limits['script_min_free_bytes']} B; "
                     f"the monitor stops the build below {limits['min_free_bytes']} B)"))
    rows.append(("memavailable_at_least_the_monitor_stop", mem.get("MemAvailable", 0) >= limits["min_available_bytes"],
                 f"MemAvailable {gib(mem.get('MemAvailable'))}; the monitor stops the build below "
                 f"{limits['min_available_bytes']} B"))
    return rows


def conditions(place: Place, step: str, start_free: int) -> tuple[list[tuple[str, bool, str]], list[str]]:
    rows: list[tuple[str, bool, str]] = []
    info: list[str] = []
    rows.append(("not_root", os.getuid() != 0, f"uid={os.getuid()}"))
    missing = [name for name in (STEPS[step]["script"], "lib/guarded_run.py") if not (place.package / name).is_file()]
    rows.append(("package_has_the_step_and_the_monitor", not missing, f"missing: {missing or 'none'}"))
    python = Path(place.paths["RT_PY"])
    if step == "engine":
        rows.append(("python_environment_present", os.access(python, os.X_OK), f"{python} (bash setup-env.sh makes it)"))
    output = place.output(step)
    if step == "engine":
        rows.append(("no_engine_yet", not os.path.lexists(output),
                     f"{output}" + (" EXISTS: it is not replaced; move it away yourself to build again" if os.path.lexists(output) else " absent")))
    found = competing()
    rows.append(("no_model_server_runtime_or_other_build", not found, f"running: {found or 'none'}"))
    table = []
    for item in procs():
        try:
            rss = int(Path(f"/proc/{item['pid']}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue
        table.append((rss, item["pid"], item["comm"]))
    table.sort(reverse=True)
    graphical = [f"{comm}:{pid}:{rss // 1048576}MiB" for rss, pid, comm in table if comm in GRAPHICAL]
    info.append("graphical_processes_running " + (" ".join(graphical) if graphical else "none")
                + " (written down, not required to be gone: closing a desktop is your own step)")
    info.append("largest_resident_processes " + " ".join(f"{comm}:{pid}:{rss // 1048576}MiB" for rss, pid, comm in table[:10]))
    try:
        info.append("loadavg " + Path("/proc/loadavg").read_text().strip())
    except OSError:
        pass
    rows += memory_rows(step, start_free)  # memory last
    info.append("meminfo " + " ".join(f"{key}={value}" for key, value in meminfo().items()))
    return rows, info


def show(rows, info) -> list[str]:
    lines = [f"{'ok     ' if ok else 'REFUSED'} {name}: {detail}" for name, ok, detail in rows]
    lines += [f"info    {line}" for line in info]
    for line in lines:
        say(line, stamp=False)
    return lines


# ---------------------------------------------------------------- one step under the monitor

def run_step(place: Place, step: str, record: Path, arguments: list[str]) -> dict:
    limits = STEPS[step]
    command = [sys.executable, "-I", "-B", str(place.package / "lib" / "guarded_run.py"), "--name", step,
               "--out", str(record), "--min-available-bytes", str(limits["min_available_bytes"]),
               "--min-free-bytes", str(limits["min_free_bytes"]), "--timeout", str(limits["timeout_s"]),
               "--term-grace", str(TERM_GRACE_S), "--kill-grace", str(KILL_GRACE_S)]
    if limits["stdin_text"] is not None:
        command += ["--stdin-text", limits["stdin_text"]]
    command += ["--", "bash", str(place.package / limits["script"]), *arguments]
    say(f"{step}: starting under the monitor (stop below {limits['min_free_bytes']} B MemFree"
        f"{' (not watched)' if not limits['min_free_bytes'] else ''}, below {limits['min_available_bytes']} B MemAvailable, "
        f"after {limits['timeout_s']} s)")
    t0 = time.monotonic()
    with open(record / "monitor.txt", "wb") as out:
        monitor = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=str(place.package))
    place.save_state(record=str(record), step=step, monitor=proc(monitor.pid), leader=None)
    leader, stopped_at, last_note, offset = None, None, time.monotonic(), 0
    log_path = record / f"{step}.log"

    def relay(start: int) -> int:
        """The step's own output, as it is written, to this terminal (the log file keeps all of it)."""
        try:
            with open(log_path, "rb") as handle:
                handle.seek(start)
                data = handle.read()
        except OSError:
            return start
        cut = data.rfind(b"\n") + 1
        if cut:
            try:
                sys.stdout.write(data[:cut].decode("utf-8", "replace"))
                sys.stdout.flush()
            except (OSError, ValueError):
                pass  # the terminal went away; the log file keeps everything
        return start + cut

    try:
        while monitor.poll() is None:
            if leader is None:
                for item in procs():
                    if item["ppid"] == monitor.pid and item["sid"] == item["pid"]:
                        leader = item
                        place.save_state(leader=leader)
                        break
            if stopped_at is None and leader is not None and stop_requested(place):
                delivered = send(leader, signal.SIGTERM)
                stopped_at = utc()
                consume_stop(place)
                say(f"{step}: stop requested: SIGTERM to the step's first process {leader['pid']}({leader['comm']}) "
                    f"{'sent' if delivered else 'not sent (it had ended)'}; the monitor ends the rest")
            offset = relay(offset)
            if time.monotonic() - last_note >= 60:
                last_note = time.monotonic()
                mem = meminfo()
                say(f"{step}: running {time.monotonic() - t0:.0f} s; MemFree {mem.get('MemFree')} B, "
                    f"MemAvailable {mem.get('MemAvailable')} B")
            time.sleep(0.25)
    finally:
        if monitor.poll() is None:
            monitor.wait()
    relay(offset)
    result = None
    try:
        result = json.loads((record / f"{step}.result.json").read_text())
    except (OSError, ValueError):
        pass
    backstop = None
    if result is None and leader is not None:
        say(f"{step}: the monitor ended (status {monitor.returncode}) without its result; ending its session's processes")
        backstop = cleanup_session(leader, place.roots)
    place.save_state(monitor=None, leader=None, ended_utc=utc())  # the step is over: nothing of it is left to stop
    stop_line = ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("STOP:") or line.startswith("\aSTOP:"):
                    stop_line = line.strip().lstrip("\a")
    except OSError:
        pass
    return {"monitor_status": monitor.returncode, "result": result, "operator_stop_utc": stopped_at,
            "backstop": backstop, "stop_line": stop_line}


def classify(run: dict, output: Path) -> tuple[str, str]:
    result = run["result"]
    if result is None:
        return "monitor_failed", f"monitor status {run['monitor_status']}"
    if not result.get("started"):
        return "refused_at_monitor_start", str(result.get("reason"))
    if result.get("stopped_by_monitor"):
        return "stopped_by_monitor", str(result["stopped_by_monitor"])
    if result.get("exit_code") == 0 and output.is_file():
        return "built", "status 0 and the output is there"
    if run["operator_stop_utc"]:
        return "stopped_by_operator", f"stop at {run['operator_stop_utc']}; status {result.get('exit_code')}"
    return "failed", f"status {result.get('exit_code')}" + ("" if result.get("exit_code") else "; the output was not written")


def file_line(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"{path} bytes={path.stat().st_size} sha256={digest.hexdigest()}"


# ---------------------------------------------------------------- modes

def mode_run(place: Place, args: argparse.Namespace) -> int:
    step, limits = args.step, STEPS[args.step]
    arguments: list[str] = []
    if args.jobs is not None:
        if step != "llama" or not 1 <= args.jobs <= 8:
            raise beta.Stop("--jobs N (1 to 8) belongs to the llama step")
        arguments = ["--jobs", str(args.jobs)]
    start_free = limits["start_free_bytes"]
    if args.min_start_free_bytes is not None:
        if step != "engine":
            raise beta.Stop("--min-start-free-bytes belongs to the engine step")
        if args.min_start_free_bytes < limits["script_min_free_bytes"]:
            raise beta.Stop(f"--min-start-free-bytes may not be below build-engine.sh's own minimum "
                            f"({limits['script_min_free_bytes']} B): the script would refuse")
        start_free = args.min_start_free_bytes
    place.prepare()
    held = place.lock()
    if held is None:
        say("STOP: a guarded build is already running for this installation. Nothing was started", stamp=False)
        return EXIT["active"]
    say(f"Guarded build: {limits['what']} ({limits['script']}), package {place.package}", stamp=False)
    rows, info = conditions(place, step, start_free)
    lines = show(rows, info)
    refused = [name for name, ok, _ in rows if not ok]
    if args.check:
        say(f"CHECK={'would_start' if not refused else 'would_refuse:' + ','.join(refused)}; nothing was started", stamp=False)
        return 0 if not refused else EXIT["refused"]

    def refuse(names: list[str], shown: list[str]) -> int:
        path = place.builds / f"refused-{step}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.txt"
        path.write_text(f"utc={utc()} step={step}\n" + "\n".join(shown) + "\n")
        say(f"REFUSED before anything started: {', '.join(names)}. Nothing ran. Measurements: {path}", stamp=False)
        return EXIT["refused"]

    if refused:
        return refuse(refused, lines)
    say(f"type BUILD to start {limits['what']} under the monitor now, or press Enter to stop: ", stamp=False)
    try:
        answer = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        answer = ""
    if answer != "BUILD":
        say("STOP: nothing was started", stamp=False)
        return EXIT["refused"]
    again = memory_rows(step, start_free)  # read again, last: typing takes time
    if [name for name, ok, _ in again if not ok]:
        return refuse([name for name, ok, _ in again if not ok],
                      lines + [f"REFUSED {name}: {detail}" for name, ok, detail in again if not ok])

    record = place.builds / f"{step}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    os.mkdir(record, 0o700)
    LOG["path"] = record / "frontend.log"
    (record / "conditions.txt").write_text(f"utc={utc()} step={step} package={place.package}\n" + "\n".join(
        lines + [f"{'ok     ' if ok else 'REFUSED'} {name} (read again after BUILD): {detail}" for name, ok, detail in again]) + "\n")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_stop_signal)
    for signum in (signal.SIGHUP, signal.SIGQUIT, signal.SIGTSTP):
        signal.signal(signum, on_ignored_signal)
    consume_stop(place)  # a request left from an earlier run is not this run's
    started, output = utc(), place.output(step)
    run, error = None, ""
    try:
        run = run_step(place, step, record, arguments)
        outcome, reason = classify(run, output)
    except Exception:  # whatever went wrong in this front end is written down with the rest
        error = traceback.format_exc()
        say("front-end error:\n" + error)
        outcome, reason = "error", error.strip().splitlines()[-1]
    result = (run or {}).get("result") or {}
    leftovers = [f"{p['pid']}({p['comm']})" for p in result.get("leftover_processes", [])
                 if proc(p["pid"]) and proc(p["pid"])["start"] == p["start"]]
    if run and run.get("backstop"):
        leftovers += run["backstop"]["left"]
    try:
        package_name = (place.package / "VERSION").read_text("utf-8").splitlines()[0]
    except (OSError, IndexError):
        package_name = "unknown"
    lines = [f"guarded_build step={step} script={limits['script']} package={package_name}",
             f"started_utc={started} ended_utc={utc()}",
             f"OUTCOME={outcome}", f"reason={reason}",
             f"monitor_stops=memfree_below_{limits['min_free_bytes']}_B memavailable_below_{limits['min_available_bytes']}_B "
             f"timeout_{limits['timeout_s']}_s"]
    if step == "engine":
        lines.append(f"start_figure_bytes={start_free} (the preparation figure in use: chosen, not a measured minimum)")
    if result.get("started"):
        lines += [f"elapsed_s={result.get('elapsed_s')} exit_code={result.get('exit_code')} "
                  f"stopped_by_monitor={result.get('stopped_by_monitor')}",
                  f"memfree_at_start_bytes={result['mem_at_start'].get('MemFree')} lowest_memfree_bytes={result.get('lowest_mem_free_bytes')}",
                  f"memavailable_at_start_bytes={result['mem_at_start'].get('MemAvailable')} "
                  f"lowest_memavailable_bytes={result.get('lowest_mem_available_bytes')}",
                  f"swapfree_at_start_bytes={result['mem_at_start'].get('SwapFree')} lowest_swapfree_bytes={result.get('lowest_swap_free_bytes')}",
                  f"samples={result.get('samples')} longest_gap_s={result.get('max_sampling_gap_s')}"]
    elif result:
        lines.append(f"not_started mem_at_start={result.get('mem_at_start')}")
    if run:
        lines.append(f"script_stop_line={run['stop_line']}")
        if run["operator_stop_utc"]:
            lines.append(f"operator_stop_utc={run['operator_stop_utc']}")
    lines.append(f"output={file_line(output)}" if output.is_file() else "output=none")
    lines += [f"leftover_processes_of_this_run={leftovers or 'none'}",
              "these_figures=whole-device readings on this device for this one run: not a requirement, not a profile",
              f"records={record}"]
    if outcome == "built":
        lines.append("next=" + ("bash record-artifacts.sh" if step == "engine" else "bash guarded-build.sh engine"))
    (record / "RESULT.txt").write_text("\n".join(lines) + "\n")
    say("RESULT:", stamp=False)
    for line in lines:
        say("  " + line, stamp=False)
    os.close(held)
    if leftovers:
        return EXIT["leftovers"]
    return EXIT[outcome]


def mode_stop(place: Place) -> int:
    place.prepare()
    held = place.lock()
    if held is None:  # a front end is active: it acts on the request itself
        (place.builds / "STOP-REQUEST").touch()
        print("stop requested. The running guarded build ends its step; waiting for it (up to 60 s)")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and (place.builds / "STOP-REQUEST").exists():
            time.sleep(0.5)
        print("the request was taken" if not (place.builds / "STOP-REQUEST").exists() else "the request was not taken yet")
        return 0
    try:  # no front end: was something of an interrupted run left under its monitor, or without it?
        state = place.state()
        monitor, leader, record = state.get("monitor"), state.get("leader"), state.get("record")
        if record and state.get("step") and (Path(record) / f"{state['step']}.result.json").exists():
            leader = None  # that step's monitor finished and wrote its result
        if not leader:
            print("no guarded build is running and no step's process was recorded: nothing to stop")
            return 0
        alive = monitor and proc(monitor["pid"]) and proc(monitor["pid"])["start"] == monitor["start"]
        if alive:
            first = proc(leader["pid"])
            if first and first["start"] == leader["start"] and first["ppid"] == monitor["pid"]:
                print(f"no front end is active, but the monitor {monitor['pid']} still watches the step: SIGTERM to "
                      f"its first process {leader['pid']}; the monitor ends the rest and writes its result")
                send(first, signal.SIGTERM)
            else:
                print(f"the monitor {monitor['pid']} is still there and is finishing by itself; nothing to signal")
            return 0
        report = cleanup_session(leader, place.roots)
        print(f"no front end and no monitor are active. Processes of the recorded session {leader['pid']}: signalled "
              f"{report['signalled'] or 'none'}; left {report['left'] or 'none'}; not verified and not touched "
              f"{report['not_verified'] or 'none'}")
        place.save_state(monitor=None, leader=None)
        return 0 if not report["left"] else EXIT["leftovers"]
    finally:
        os.close(held)


def mode_status(place: Place) -> int:
    place.prepare()
    held = place.lock()
    print(f"guarded build running: {'no' if held is not None else 'yes'}")
    if held is not None:
        os.close(held)
    record = place.state().get("record")
    print(f"latest records: {record or 'none'}")
    if record:
        try:
            print((Path(record) / "RESULT.txt").read_text(), end="")
        except OSError:
            print("(no RESULT.txt there yet)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One build step under the memory monitor (see the module docstring).")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--package", required=True)
    run.add_argument("--step", required=True, choices=sorted(STEPS))
    run.add_argument("--jobs", type=int, default=None)
    run.add_argument("--min-start-free-bytes", type=int, default=None)
    run.add_argument("--check", action="store_true")
    for name in ("stop", "status"):
        commands.add_parser(name).add_argument("--package", required=True)
    args = parser.parse_args(argv)
    try:
        place = Place(args.package)
        if args.command == "run":
            return mode_run(place, args)
        return mode_stop(place) if args.command == "stop" else mode_status(place)
    except beta.Stop as stop:
        print(f"STOP: {stop}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
