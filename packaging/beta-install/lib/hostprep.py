#!/usr/bin/env python3
"""Sentinel personal beta: pre-launch preparation (the accepted launcher's helper, with its advice texts made general).

Identifies, and only on the operator's typed request gracefully closes, the current user's optional desktop and
development processes; reports whether any of them still runs. Subcommands:

  scan  --out FILE     read only: list what runs now and write the closing plan (JSON) to FILE (a new file)
  close --plan FILE --record FILE
                       SIGTERM once to each planned target that is still the same process (pid, start time, uid and
                       kernel name, checked again through a pidfd), then wait up to --wait-s for the targets and the
                       VNC server to exit; never SIGKILL, never a second signal
  check --record FILE [--session S]
                       read only: whether any optional process of the listed kinds still runs, and the display
                       manager's state

What is identified (the current user's processes only; another user's process is never a target):
  development  'claude' and 'codex' by kernel name; VS Code's server processes by their executable under
               ~/.vscode-server/ (one is a target when its parent is not such a process; the processes under a
               target are expected to end with it and are listed, never signalled)
  desktop      'update-manager' (held, not closed, while a package operation runs) and the VNC desktop's session
               leader ('xfce4-session' whose parent is 'vncserver')
  watched      the VNC server ('Xtigervnc' or 'Xvnc') and 'vncserver': never signalled; they must be gone after
               the session ends
Never: a signal by name or pattern, any other node or python process, a process of another user, the shell this
runs in or any of its ancestors, a service stopped, disabled or started, a crontab read or changed. Reads
/proc/<pid>/stat, status and cgroup and, for the user's own processes, the exe link (compared, never written out);
no command line or environment. Per-process RssAnon values overlap and are never added together.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "sentinel-host-prep/1"
PROC = Path("/proc")
PF_KTHREAD = 0x00200000  # include/linux/sched.h: a kernel thread
NAME_OTHER = re.compile(r"[^A-Za-z0-9._:+@()/\-]")
UNIT = re.compile(r"[A-Za-z0-9@:._\\-]{1,96}\.(?:service|scope)")
BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DM_STATES = frozenset({"active", "inactive", "failed", "activating", "deactivating", "reloading", "unknown"})
DEVELOPMENT = ("claude", "codex")
UPDATER = "update-manager"
SESSION_LEADER = "xfce4-session"
VNC_WRAPPER = "vncserver"
VNC_SERVERS = ("Xtigervnc", "Xvnc")
PACKAGE_OPERATIONS = ("apt", "apt-get", "aptitude", "dpkg", "aptd", "unattended-upgr", "synaptic")
DEVELOPMENT_LABELS = ("node", "npm", "npx")  # described by check, never judged or signalled
ROLE_TEXT = {"claude": "Claude Code", "codex": "Codex", "vscode_server": "VS Code server",
             "update_manager": "update-manager", "vnc_session": "VNC desktop session"}


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_text(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def sanitize(comm: str) -> str:
    words = comm.strip().split()
    return (NAME_OTHER.sub("_", words[0])[:15] if words else "") or "_"


def cgroup_unit(text: str) -> str | None:
    path = None
    for line in text.splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, cgroup = rest.partition(":")
        if hierarchy == "0" and controllers == "":
            path = cgroup
            break
        if controllers == "name=systemd":
            path = cgroup
    last = (path or "").rstrip("/").rsplit("/", 1)[-1]
    return last if UNIT.fullmatch(last) else None


def vscode_root() -> str:
    return str(Path.home() / ".vscode-server") + os.sep


def read_proc(pid: int, uid: int, vs_root: str) -> dict | None:
    """One process, or None (exited, unreadable or a kernel thread)."""
    base = PROC / str(pid)
    try:
        text = read_text(base / "stat")
        opened, closed = text.find("("), text.rfind(")")
        if opened < 0 or closed < opened:
            return None
        rest = text[closed + 2:].split()
        comm, state, ppid, flags, start = text[opened + 1:closed], rest[0], int(rest[1]), int(rest[6]), int(rest[19])
    except (OSError, ValueError, IndexError):
        return None
    if flags & PF_KTHREAD:
        return None
    proc = {"pid": pid, "ppid": ppid, "start": start, "state": state, "comm": comm, "name": sanitize(comm),
            "uid": None, "rss_anon_bytes": None, "unit": None, "vscode": False}
    try:
        for line in read_text(base / "status").splitlines():
            key, _, value = line.partition(":")
            parts = value.split()
            if key == "Uid" and parts and parts[0].isdigit():
                proc["uid"] = int(parts[0])
            elif key == "RssAnon" and len(parts) == 2 and parts[0].isdigit():
                proc["rss_anon_bytes"] = int(parts[0]) * 1024
    except OSError:
        pass
    try:
        proc["unit"] = cgroup_unit(read_text(base / "cgroup"))
    except OSError:
        pass
    if proc["uid"] == uid and state != "Z":
        try:
            proc["vscode"] = os.readlink(base / "exe").startswith(vs_root)
        except OSError:
            pass
    return proc


def snapshot(uid: int) -> dict[int, dict]:
    vs_root = vscode_root()
    procs = {}
    for entry in os.listdir(PROC):
        if entry.isdigit():
            proc = read_proc(int(entry), uid, vs_root)
            if proc is not None:
                procs[proc["pid"]] = proc
    return procs


def boot_id() -> str:
    try:
        value = read_text(PROC / "sys/kernel/random/boot_id").strip()
    except OSError:
        return "unreadable"
    return value if BOOT_ID.fullmatch(value) else "unrecognized"


def display_manager_state() -> str:
    exe = next((p for p in ("/usr/bin/systemctl", "/bin/systemctl") if os.path.exists(p)), None)
    if exe is None:
        return "unavailable"
    try:
        done = subprocess.run([exe, "is-active", "display-manager"], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False,
                              env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    word = done.stdout.decode("ascii", "replace").strip().split("\n")[0].strip()
    return word if word in DM_STATES else "unrecognized"


def mb(value) -> str:
    return "-" if value is None else f"{value / 1e6:,.1f}"


def entry(proc: dict, group: str = "", role: str = "") -> dict:
    keep = ("pid", "ppid", "start", "uid", "comm", "name", "unit", "rss_anon_bytes")
    return {**{k: proc[k] for k in keep}, **({"group": group, "role": role} if role else {})}


def same(current: dict | None, planned: dict) -> bool:
    """Still the planned process: present, not a zombie, same start time, uid and kernel name."""
    return (current is not None and current["state"] != "Z" and current["start"] == planned["start"]
            and current["uid"] == planned["uid"] and current["comm"] == planned["comm"])


def alive(planned: dict, uid: int) -> dict | None:
    current = read_proc(planned["pid"], uid, vscode_root())
    return current if same(current, planned) else None


def ancestors(procs: dict[int, dict]) -> list[dict]:
    chain, pid, seen = [], os.getppid(), set()
    while pid > 1 and pid in procs and pid not in seen:
        seen.add(pid)
        chain.append(procs[pid])
        pid = procs[pid]["ppid"]
    return chain


def classify(procs: dict[int, dict], uid: int) -> dict:
    live = {pid: p for pid, p in procs.items() if p["state"] != "Z"}
    mine = {pid: p for pid, p in live.items() if p["uid"] == uid and pid != os.getpid()}

    def parent(p):
        return live.get(p["ppid"]) or {}

    package_ops = [entry(p) for p in sorted(live.values(), key=lambda p: p["pid"]) if p["comm"] in PACKAGE_OPERATIONS]
    by_pid = lambda p: p["pid"]  # noqa: E731
    targets, held = [], []
    for p in sorted((p for p in mine.values() if p["comm"] in DEVELOPMENT), key=by_pid):
        targets.append(entry(p, "development", p["comm"]))
    for p in sorted((p for p in mine.values() if p["vscode"] and not parent(p).get("vscode")), key=by_pid):
        targets.append(entry(p, "development", "vscode_server"))
    for p in sorted((p for p in mine.values() if p["comm"] == UPDATER), key=by_pid):
        (held if package_ops else targets).append(entry(p, "desktop", "update_manager"))
    leaders = [p for p in mine.values() if p["comm"] == SESSION_LEADER and parent(p).get("comm") == VNC_WRAPPER]
    for p in sorted(leaders, key=by_pid):
        targets.append(entry(p, "desktop", "vnc_session"))
    watched = [entry(p) for p in sorted(mine.values(), key=by_pid)
               if p["comm"] == VNC_WRAPPER or (p["comm"] in VNC_SERVERS and parent(p).get("comm") == VNC_WRAPPER)]
    children: dict[int, list[int]] = {}
    for p in live.values():
        children.setdefault(p["ppid"], []).append(p["pid"])
    target_pids = {t["pid"] for t in targets}
    watched_pids = {w["pid"] for w in watched}
    under: dict[int, list[dict]] = {}
    covered = set()
    for t in targets:
        found, todo = [], list(children.get(t["pid"], []))
        while todo:
            pid = todo.pop()
            if pid in covered or pid in target_pids:
                continue
            covered.add(pid)
            found.append(entry(live[pid]))
            todo.extend(children.get(pid, []))
        under[str(t["pid"])] = sorted(found, key=by_pid)
    chain = ancestors(procs)
    units = {t["unit"] for t in targets if t["role"] == "vnc_session" and t["unit"]}
    others = [entry(p) for p in sorted(mine.values(), key=by_pid) if p["unit"] in units
              and p["pid"] not in target_pids | watched_pids | covered | {a["pid"] for a in chain}]
    inside = [entry(a) for a in chain if a["pid"] in target_pids | covered | watched_pids]
    return {"targets": targets, "held": held, "package_operations": package_ops, "watched": watched,
            "under": under, "unit_others": others, "inside": inside}


def names(entries: list[dict], limit: int = 8) -> str:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["name"]] = counts.get(e["name"], 0) + 1
    text = ", ".join(f"{n}" + (f" x{c}" if c > 1 else "") for n, c in list(counts.items())[:limit])
    return text + (", ..." if len(counts) > limit else "")


def listing(plan: dict) -> list[str]:
    lines = [f"optional processes of uid {plan['uid']} now (PIDs read now; RssAnon per process, overlapping, "
             "never a total):"]
    if plan["targets"]:
        lines.append("will be closed on CLOSE (SIGTERM once each, in this order; never SIGKILL):")
        for n, t in enumerate(plan["targets"], 1):
            sub = plan["under"].get(str(t["pid"]), [])
            lines.append(f"  {n}. {t['group']:<11} {ROLE_TEXT[t['role']]:<20} {t['name']:<15} pid {t['pid']:>7}  "
                         f"RssAnon {mb(t['rss_anon_bytes']):>7} MB  unit {t['unit'] or '-'}"
                         + (f"  (+{len(sub)} under it: {names(sub)})" if sub else ""))
    else:
        lines.append("will be closed: nothing (none of the listed kinds runs)")
    for h in plan["held"]:
        ops = ", ".join(f"{o['name']} pid {o['pid']}" for o in plan["package_operations"])
        lines.append(f"held, NOT closed: update-manager pid {h['pid']}: a package operation runs ({ops}); "
                     "let it finish, then run start.sh again")
    if plan["watched"]:
        lines.append("expected to end with the VNC session (never signalled): "
                     + ", ".join(f"{w['name']} pid {w['pid']}" for w in plan["watched"]))
    if plan["unit_others"]:
        lines.append(f"others of yours in the session's unit (never signalled; listed only): "
                     f"{len(plan['unit_others'])}: {names(plan['unit_others'], 12)}")
    return lines


def explanation(plan: dict) -> list[str]:
    roles = {t["role"] for t in plan["targets"]}
    lines = ["", "What CLOSE does (read before typing; nothing is uninstalled, disabled or reconfigured):"]
    if roles & {"claude", "codex"}:
        lines.append("- each Claude Code or Codex session listed ends (a Claude Code conversation can be resumed later "
                     "with 'claude --continue' in the same directory)")
    if "vscode_server" in roles:
        lines.append("- VS Code's server on the Jetson ends: a VS Code window still connected to the Jetson disconnects "
                     "and unsaved changes in it may be lost: save them first")
    if "update_manager" in roles:
        lines.append("- update-manager's window closes; nothing is installed, removed or cancelled (it is never closed "
                     "while a package operation runs); this package does not restart it")
    if "vnc_session" in roles:
        lines += ["- the VNC desktop session is asked to log out: its window manager, panel, desktop and every "
                  "application open in it close, a VNC viewer connected to it disconnects, and unsaved work in it may "
                  "be lost: save it first, or press Enter now. The VNC server is expected to exit by itself; it is "
                  "never signalled. No service is stopped or disabled; the crontab is not changed.",
                  "  Restore it after the run by starting your VNC server again the way you normally do."]
    if roles & {"claude", "codex", "vscode_server"}:
        lines.append("- restore VS Code and Claude Code after the run by reconnecting as usual")
    return lines


def write_new(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def cmd_scan(args) -> int:
    uid = os.getuid()
    procs = snapshot(uid)
    plan = {"schema": SCHEMA, "action": "scan", "utc": utc(), "boot_id": boot_id(), "uid": uid,
            **classify(procs, uid)}
    for line in listing(plan):
        print(line)
    if plan["inside"]:
        print("STOP: this shell runs inside a process that would be closed ("
              + ", ".join(f"{a['name']} pid {a['pid']}" for a in plan["inside"])
              + "): open a plain SSH session (not a VS Code terminal) and run start.sh again; "
              "no plan was written and nothing was signalled")
        return 1
    if plan["watched"] and not any(t["role"] == "vnc_session" for t in plan["targets"]):
        print("STOP: a VNC server runs without a session this package can close; no plan was written and "
              "nothing was signalled: send back the lines above")
        return 1
    try:
        write_new(args.out, json.dumps(plan, indent=1) + "\n")
    except OSError as exc:
        print(f"STOP: the plan could not be written ({type(exc).__name__}); nothing was signalled")
        return 1
    if not plan["targets"]:
        print(f"plan={args.out.name} targets=0")
        if plan["held"]:
            print("STOP: nothing can be closed now (update-manager is held while a package operation runs): let it "
                  "finish, then run start.sh again")
            return 1
        return 0
    for line in explanation(plan):
        print(line)
    print(f"plan={args.out.name} targets={len(plan['targets'])}")
    return 3


def load_plan(path: Path, uid: int) -> tuple[dict | None, str]:
    try:
        raw = path.read_bytes()
        plan = json.loads(raw)
    except (OSError, ValueError) as exc:
        return None, f"plan unreadable ({type(exc).__name__})"
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("action") != "scan":
        return None, "not a scan plan"
    if plan.get("uid") != uid:
        return None, "plan made for another user"
    if plan.get("boot_id") != boot_id():
        return None, "plan made in another boot"
    for key in ("targets", "held", "watched", "under", "unit_others"):
        if not isinstance(plan.get(key), (list, dict)):
            return None, f"plan lacks {key}"
    plan["_sha256"] = hashlib.sha256(raw).hexdigest()
    return plan, ""


def cmd_close(args) -> int:
    uid = os.getuid()
    plan, problem = load_plan(args.plan, uid)
    if plan is None:
        print(f"STOP: {problem}; nothing was signalled")
        return 2
    procs = snapshot(uid)
    protected = {t["pid"] for t in plan["targets"]} | {e["pid"] for sub in plan["under"].values() for e in sub} \
        | {w["pid"] for w in plan["watched"]}
    inside = [a for a in ancestors(procs) if a["pid"] in protected]
    if inside:
        print("STOP: this shell runs inside a process that would be closed ("
              + ", ".join(f"{a['name']} pid {a['pid']}" for a in inside) + "); nothing was signalled")
        return 2
    lines = [f"schema={SCHEMA}", "action=close", f"utc_start={utc()}", f"boot_id={plan['boot_id']}",
             f"plan={args.plan.name}", f"plan_sha256={plan['_sha256']}", f"wait_s={args.wait_s}"]
    t0 = time.monotonic()
    pidfds: dict[int, int] = {}
    problems: list[str] = []
    for t in plan["targets"]:
        label = f"pid={t['pid']} start={t['start']} name={t['name']} role={t['role']}"
        current = procs.get(t["pid"])
        if not same(current, t):  # the planned process is gone; another may hold its pid now (never signalled)
            result = "changed_not_signalled" if current is not None and current["state"] != "Z" else "gone_before_signal"
        else:
            try:
                fd = os.pidfd_open(t["pid"])
            except ProcessLookupError:
                result = "gone_before_signal"
            except OSError as exc:
                result = f"not_signalled:{type(exc).__name__}"
            else:
                current = read_proc(t["pid"], uid, vscode_root())
                if not same(current, t):
                    os.close(fd)
                    result = "changed_not_signalled" if current is not None and current["state"] != "Z" \
                        else "gone_before_signal"
                else:
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                    except ProcessLookupError:
                        os.close(fd)
                        result = "gone_before_signal"
                    except OSError as exc:
                        os.close(fd)
                        result = f"not_signalled:{type(exc).__name__}"
                    else:
                        pidfds[t["pid"]] = fd
                        result = f"sigterm_sent_at_s={time.monotonic() - t0:.2f}"
        if result.startswith("not_signalled"):
            problems.append(f"not_signalled:{t['name']}")
        lines.append(f"signal {label} result={result}")
        print(f"{ROLE_TEXT[t['role']]} ({t['name']} pid {t['pid']}): {result}")
    signalled = [t for t in plan["targets"] if t["pid"] in pidfds]
    exited: dict[int, float] = {}
    deadline = t0 + args.wait_s
    print(f"waiting up to {args.wait_s:g} s for them and the VNC server to exit (nothing more is signalled)...")
    while True:
        for t in signalled:
            if t["pid"] not in exited and alive(t, uid) is None:
                exited[t["pid"]] = time.monotonic() - t0
        pending = [t for t in signalled if t["pid"] not in exited]
        watched_alive = [w for w in plan["watched"] if alive(w, uid) is not None]
        if (not pending and not watched_alive) or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    for fd in pidfds.values():
        os.close(fd)
    for t in signalled:
        if t["pid"] in exited:
            lines.append(f"exited pid={t['pid']} name={t['name']} after_s={exited[t['pid']]:.1f}")
            print(f"exited: {t['name']} pid {t['pid']} after {exited[t['pid']]:.1f} s")
    still = [t for t in signalled if t["pid"] not in exited] + [w for w in plan["watched"] if alive(w, uid)]
    held = [h for h in plan["held"] if alive(h, uid)]
    for s in still:
        lines.append(f"still_running pid={s['pid']} name={s['name']}")
        problems.append(f"still_running:{s['name']}")
    for h in held:
        lines.append(f"held_running pid={h['pid']} name={h['name']}")
        problems.append(f"held:{h['name']}")
    left = [cur for e in [x for sub in plan["under"].values() for x in sub] + plan["unit_others"]
            if (cur := alive(e, uid)) is not None]
    for cur in left:
        lines.append(f"left pid={cur['pid']} name={cur['name']} rss_anon_bytes={cur['rss_anon_bytes']}")
    if still:
        print("STILL RUNNING after the wait (nothing more was signalled): "
              + ", ".join(f"{s['name']} pid {s['pid']}" for s in still))
    if held:
        print("held, not closed: " + ", ".join(f"{h['name']} pid {h['pid']}" for h in held))
    if left:
        print("left running (listed only, never signalled; RssAnon per process): "
              + ", ".join(f"{c['name']} pid {c['pid']} {mb(c['rss_anon_bytes'])} MB" for c in left))
    verdict = "complete" if not problems else "incomplete:" + ";".join(problems)
    lines += [f"utc_end={utc()}", f"PC_result={verdict}"]
    print(f"PC_result={verdict}")
    try:
        write_new(args.record, "\n".join(lines) + "\n")
    except OSError as exc:
        print(f"the record could not be written ({type(exc).__name__})")
        return 1
    return 0 if not problems else 1


def cmd_check(args) -> int:
    uid = os.getuid()
    procs = snapshot(uid)
    mine = [p for p in sorted(procs.values(), key=lambda p: p["pid"])
            if p["uid"] == uid and p["state"] != "Z" and p["pid"] != os.getpid()]
    running = [p for p in mine if p["comm"] in DEVELOPMENT or p["vscode"] or p["comm"] == UPDATER
               or p["comm"] == SESSION_LEADER or p["comm"] in VNC_SERVERS or p["comm"] == VNC_WRAPPER]
    labelled = [p for p in mine if p["comm"] in DEVELOPMENT_LABELS and not p["vscode"]]
    dm = display_manager_state()
    lines = [f"schema={SCHEMA}", "action=check", f"utc={utc()}", f"boot_id={boot_id()}",
             f"session={args.session or '-'}"]
    lines.append(f"display_manager={dm}")
    for p in running:
        lines.append(f"running pid={p['pid']} name={p['name']} unit={p['unit'] or '-'} "
                     f"rss_anon_bytes={p['rss_anon_bytes']}")
    for p in labelled:
        lines.append(f"described pid={p['pid']} name={p['name']} unit={p['unit'] or '-'} "
                     f"rss_anon_bytes={p['rss_anon_bytes']}")
    reasons = ([("running:" + ",".join(sorted({p['name'] for p in running})))] if running else []) \
        + ([] if dm == "inactive" else [f"display_manager_{dm}"])
    verdict = "complete" if not reasons else "incomplete:" + ";".join(reasons)
    lines.append(f"PO_prepared={verdict}")
    print(f"display-manager: {dm}")
    print("optional processes still running: "
          + (", ".join(f"{p['name']} pid {p['pid']}" for p in running) or "none"))
    if labelled:
        print("other development-labelled names of yours (described, not judged): "
              + ", ".join(f"{p['name']} pid {p['pid']} {mb(p['rss_anon_bytes'])} MB" for p in labelled))
    print(f"PO_prepared={verdict}")
    try:
        write_new(args.record, "\n".join(lines) + "\n")
    except OSError as exc:
        print(f"STOP: the record could not be written ({type(exc).__name__})")
        return 2
    return 0 if not reasons else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-launch preparation (see the module docstring).")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--out", type=Path, required=True)
    close = sub.add_parser("close")
    close.add_argument("--plan", type=Path, required=True)
    close.add_argument("--record", type=Path, required=True)
    close.add_argument("--wait-s", type=float, default=60.0)
    check = sub.add_parser("check")
    check.add_argument("--record", type=Path, required=True)
    check.add_argument("--session", default="")
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        print("STOP: run as your own user, without sudo", file=sys.stderr)
        return 2
    if args.command == "close" and not 1 <= args.wait_s <= 120:
        print("STOP: --wait-s must be 1-120", file=sys.stderr)
        return 2
    return {"scan": cmd_scan, "close": cmd_close, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
