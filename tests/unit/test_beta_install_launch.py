"""The personal-beta package's launcher (start.sh, check.sh, stop.sh), driven as a tester would, with stand-ins.

Nothing real is started: the runtime, the memory guard, sudo, systemctl, the process scan and llama-server are
stand-ins made by this file; the camera address and the passphrase are invented strings. The launcher scripts
themselves are the package's own, with two substitutions that the shipped package does not have: the memory reading
comes from a fixture file (one line of start.sh) and the asset manifest lists small synthetic files. The package is
then sealed again, so every integrity check still runs.

Covered: the offline check passes without credentials, sudo, the camera or a model, and each thing that is missing
or refused stops it with what to do; nothing opens the camera unless START is typed; the address and the passphrase
are asked for hidden, reach the runtime only through its environment, and are in no file and not on the terminal;
a clean stop from the window and from stop.sh; the enrollment is checked before and after.

A pass here says the launcher's own logic behaves as written with stand-ins. It is not a launch.
"""

from __future__ import annotations

import hashlib
import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from beta_install_fixture import World, said

URL = "rtsp://tester:s3cret-pw@10.9.8.7:554/main-stream-XYZ"
PASSPHRASE = "an-invented-Passphrase-987"
COMMIT = "19cce6919992fdffd92e0ac152caaf1e2b8f254e"
ADMITTED = "\n".join((
    "stream_profile=main_single", "notification_channels=0", "thresholds=unchanged", "detector_engine=the_pinned_engine",
    "scene_admission=admitted", "face_admission_without_validation_flag=refused", "face_admission_validation_run=allowed",
    "identity_retention_rule=proposed_defaults", "identity_retention_without_validation_flag=refused", "display_labels=0",
    "chat_admitted=False", "chat_validation_run_allowed=True", "admission_check=as_expected")) + "\n"
REFUSED = ("detector_engine=refused\nscene_admission=refused\nadmission_check=failed:detector_engine,scene_admission\n")

INTERPRETER = r'''#!/bin/bash
# Stand-in for <prefix>/venv/bin/python: the runtime, the guard and the admission check never run for real here.
if [ "$1" = benchmarks/runner/step5_guard.py ]; then shift; exec "$REAL_PYTHON" -B "$FIXTURE/standin/step5_guard.py" "$@"; fi
if [ "$1" = -B ] && [[ "$2" == */lib/admission_check.py ]]; then
  printf '%s\n' "${@:3}" > "$FIXTURE/admission.args"; cat "$FIXTURE/admission.out"; exit "$(cat "$FIXTURE/admission.rc")"
fi
if [ "$1" = -m ] && [ "$2" = sentinel.cli ] && [ "$3" = config ]; then exec "$REAL_PYTHON" -B "$@"; fi
if [ "$1" = -m ] && [ "$2" = sentinel.cli ] && [ "$3" = run ]; then exec "$REAL_PYTHON" -B "$FIXTURE/standin/runtime.py" "$@"; fi
echo "stand-in interpreter: unexpected use: $*" >&2; exit 97
'''
GUARD = r'''"""Stand-in for the memory guard: runs the command after "--", forwards one SIGINT to it, writes result.json."""
import json, os, signal, subprocess, sys
args = sys.argv[1:]
out = args[args.index("--out") + 1]
command = args[args.index("--") + 1:]
os.makedirs(out, mode=0o700)
json.dump({"guard_args": args[:args.index("--")]}, open(os.path.join(os.environ["FIXTURE"], "guard.json"), "w"))
interrupted = {"yes": False}
with open(os.path.join(out, "run.jsonl"), "wb") as stdout, open(os.path.join(out, "run.err"), "wb") as stderr:
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL)
    def stop(*_):
        interrupted["yes"] = True
        child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT, stop)
    code = child.wait()
status = "operator_interrupt" if interrupted["yes"] else "duration_stop"
json.dump({"status": status, "child": {"returncode": code}, "readiness": {"met": True, "reason": None},
           "guard": {"trigger": None, "peak_pressure_bytes": 1}, "cleanup_clear": True, "run_error_label": None},
          open(os.path.join(out, "result.json"), "w"))
sys.exit(1 if interrupted["yes"] else 0)
'''
RUNTIME = r'''"""Stand-in for `sentinel run`: no camera, no model. Records what it was given (digests of secrets, never the secrets)."""
import hashlib, json, os, signal, sys, time
def digest(name):
    value = os.environ.get(name)
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()
url = os.environ.get("SENTINEL_RTSP_URL")
record = {"argv": sys.argv[1:], "url_sha256": digest("SENTINEL_RTSP_URL"), "passphrase_sha256": digest("SENTINEL_IDENTITY_PASSPHRASE"),
          "deepface_home": os.environ.get("DEEPFACE_HOME"), "ld_preload": os.environ.get("LD_PRELOAD"),
          "telegram": bool(os.environ.get("SENTINEL_TELEGRAM_BOT_TOKEN")), "cwd": os.getcwd(),
          "hup_ignored": signal.getsignal(signal.SIGHUP) == signal.SIG_IGN}
name = "runtime.json" if url else "runtime-check.json"
json.dump(record, open(os.path.join(os.environ["FIXTURE"], name), "w"))
if not url:
    print("run: rtsp_url_missing", file=sys.stderr)
    sys.exit(1)
stop = {"now": False}
signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
print(json.dumps({"run": "starting", "startup": {"face": {"validation_run": True, "identities_enrolled": 1, "detection": "head_windows"},
                                                 "scene_server": {"state": "ready", "layers": "25/25", "vision_on_gpu": True},
                                                 "chat": {"state": "available", "reason": "a stand-in"}, "live_preview": {},
                                                 "feature_controls": {"face": True}}}), flush=True)
while not stop["now"]:
    time.sleep(0.05)
print(json.dumps({"run": "stopped"}), flush=True)
'''
PYTHON3 = r'''#!/bin/bash
# Stand-in for python3 on the PATH: the process scan never looks at, or closes, a real process here.
for argument in "$@"; do
  if [[ "$argument" == */lib/hostprep.py ]]; then
    echo "stand-in scan: nothing real is listed or closed ($*)" | sed 's/ -I -B .*hostprep.py//'
    case " $* " in *" scan "*) exit "$(cat "$FIXTURE/scan.rc")" ;; *" close "*) echo closed >> "$FIXTURE/calls.log"; exit 0 ;; *) exit 0 ;; esac
  fi
done
exec "$REAL_SYSTEM_PYTHON" "$@"
'''


def sealed_again(package: Path) -> None:
    lines = []
    for line in (package / "SHA256SUMS").read_text().splitlines():
        name = line.partition("  ")[2]
        lines.append(f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n")
    (package / "SHA256SUMS").write_text("".join(lines))


class Launch(World):
    """A synthetic device with the package installed and every stand-in in place: ready for check.sh and start.sh."""

    def __init__(self, tmp: Path) -> None:
        super().__init__(tmp)
        start = (self.package / "start.sh").read_text()
        assert start.count("MEMINFO=/proc/meminfo\n") == 1
        (self.package / "start.sh").write_text(start.replace("MEMINFO=/proc/meminfo\n", f'MEMINFO="{tmp}/meminfo"\n'))
        files = {"scene_model": ("models/lfm3-vl/model.gguf", b"model"), "scene_mmproj": ("models/lfm3-vl/mmproj.gguf", b"mmproj"),
                 "face_weights": ("deepface-home/.deepface/weights/weights.h5", b"weights")}
        assets = [{"id": key, "kind": "download", "path": path, "url": "https://example.invalid/unused", "size": len(content),
                   "sha256": hashlib.sha256(content).hexdigest(), "how": "Run: bash fetch-assets.sh"}
                  for key, (path, content) in files.items()]
        assets += [{"id": "llama_server", "kind": "build", "path": "llama.cpp/build/bin/llama-server", "commit": COMMIT,
                    "how": "Run: bash build-llama.sh"},
                   {"id": "detector_engine", "kind": "build", "path": "engine/yolov8n.engine", "how": "Run: bash build-engine.sh"}]
        (self.package / "manifest/assets.json").write_text(json.dumps({"schema": "sentinel-beta-assets/1", "assets": assets}))
        sealed_again(self.package)
        installed = self.install()
        assert installed.returncode == 0, said(installed)
        for path, content in files.values():
            target = self.prefix / "assets" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (self.prefix / "assets/engine").mkdir(parents=True)
        (self.prefix / "assets/engine/yolov8n.engine").write_bytes(b"engine")
        self.script(self.prefix / "assets/llama.cpp/build/bin/llama-server",
                    '#!/bin/bash\necho "version: 0.7.0-dev (build 11938, commit $(cat "$FIXTURE/llama.commit"))"\n')
        self.script(self.prefix / "venv/bin/python", INTERPRETER)
        (tmp / "standin").mkdir()
        (tmp / "standin/step5_guard.py").write_text(GUARD)
        (tmp / "standin/runtime.py").write_text(RUNTIME)
        real_system_python = os.path.realpath(self.bin / "python3")
        (self.bin / "python3").unlink()
        self.script(self.bin / "python3", PYTHON3)
        self.command("sudo", 'echo "sudo $*" >> "$FIXTURE/calls.log"; exit 0')
        self.command("systemctl", 'echo "systemctl $*" >> "$FIXTURE/calls.log"; [ "$1" = is-active ] && cat "$FIXTURE/desktop"; exit 0')
        self.command("pgrep", "exit 1")
        self.command("ss", "exit 0")
        self.command("sync", "exit 0")
        for name, text in (("meminfo", "MemTotal: 7802736 kB\nMemFree: 6300000 kB\nMemAvailable: 6600000 kB\n"),
                           ("admission.out", ADMITTED), ("admission.rc", "0\n"), ("scan.rc", "0\n"), ("desktop", "inactive\n"),
                           ("llama.commit", COMMIT[:9] + "\n"), ("calls.log", "")):
            self.fixture(name, text)
        (self.data / "identity/gallery.sealed").write_bytes(b"a placeholder, not an enrollment")
        (self.data / "identity/gallery.sealed").chmod(0o600)
        (self.data / "identity/audit.jsonl").write_text("{}\n")
        real = {tool: os.path.realpath(self.bin / tool) for tool in ("bash",)}
        self.env = {**self.env, "PATH": f"{self.bin}:/usr/bin:/bin", "REAL_PYTHON": sys.executable,
                    "REAL_SYSTEM_PYTHON": real_system_python, "TERM": "xterm", "LANG": "C.UTF-8", "BASH": real["bash"]}

    def script(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)

    def check(self, **env: str) -> subprocess.CompletedProcess:
        return self.run("check.sh", package=self.installed, **env)

    def observations(self) -> list[Path]:
        return sorted((self.data / "observe").glob("obs-*"))

    def calls(self) -> list[str]:
        return (self.tmp / "calls.log").read_text().splitlines()

    def secrets_in_files(self) -> list[str]:
        found = []
        for path in self.tmp.rglob("*"):
            if path.is_file() and not path.is_symlink():
                data = path.read_bytes()
                if any(text.encode() in data for text in (URL, PASSPHRASE, "s3cret-pw")):
                    found.append(str(path.relative_to(self.tmp)))
        return found


class Terminal:
    """start.sh on a pseudo-terminal, as in an SSH shell: prompts are read, answers are typed."""

    def __init__(self, launch: Launch, *args: str) -> None:
        self.master, slave = pty.openpty()
        self.process = subprocess.Popen(["bash", str(launch.installed / "start.sh"), *args], stdin=slave, stdout=slave,
                                        stderr=slave, env=launch.env, cwd=launch.tmp, start_new_session=True)
        os.close(slave)
        self.seen = b""
        self.at = 0

    def pump(self, wait: float = 0.1) -> bool:
        ready, _, _ = select.select([self.master], [], [], wait)
        if not ready:
            return False
        try:
            data = os.read(self.master, 65536)
        except OSError:
            return False
        self.seen += data
        return bool(data)

    def wait_for(self, pattern: str, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        expression = re.compile(pattern.encode())
        while time.monotonic() < deadline:
            self.pump()
            match = expression.search(self.seen, self.at)
            if match:
                self.at = match.end()
                return True
            if self.process.poll() is not None:
                while self.pump(0.05):
                    pass
                match = expression.search(self.seen, self.at)
                if match:
                    self.at = match.end()
                return bool(match)
        return False

    def type(self, text: str) -> None:
        os.write(self.master, text.encode() + b"\n")

    def interrupt(self) -> None:
        """Ctrl-C in the window: one SIGINT to everything the launcher started in its terminal."""
        os.killpg(self.process.pid, signal.SIGINT)

    def finish(self, timeout: float = 60.0) -> int | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.process.poll() is None:
            self.pump()
        while self.pump(0.05):
            pass
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            raise AssertionError(f"the launcher did not end:\n{self.text()[-1500:]}")
        os.close(self.master)
        return self.process.returncode

    def text(self) -> str:
        return self.seen.decode("utf-8", "replace")


@pytest.fixture
def launch(tmp_path: Path) -> Launch:
    status = Path("/proc/self/status")
    if not status.exists() or not re.search(r"^THP_enabled:\s*1$", status.read_text(), re.M):
        pytest.skip("the launcher requires transparent huge pages enabled for the shell (Linux)")
    return Launch(tmp_path)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- the offline check

def test_the_offline_check_passes_without_credentials_sudo_camera_or_model(launch: Launch) -> None:
    result = launch.check()
    assert result.returncode == 0, said(result)
    assert "admission_check=as_expected" in result.stdout and "command_line_check=run: rtsp_url_missing" in result.stdout
    assert "BETA_CHECK=passed (no credentials asked, no sudo, the camera was not opened, no model was loaded)" in result.stdout
    assert re.search(r"^package=sentinel-personal-beta-949da67-r0-worktree$", result.stdout, re.M)
    assert launch.observations() == [] and not (launch.tmp / "runtime.json").exists()  # nothing was launched or recorded
    assert not any(call.startswith("sudo") for call in launch.calls())
    seen = json.loads((launch.tmp / "runtime-check.json").read_text())
    assert seen["url_sha256"] is None and seen["passphrase_sha256"] == sha("check-only-not-a-passphrase")
    assert seen["cwd"] == str(launch.installed / "app") and seen["ld_preload"] == "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"
    assert seen["deepface_home"] == str(launch.prefix / "assets/deepface-home")
    argv = seen["argv"]
    for flag in ("--scene", "--face", "--face-validation", "--identity-retention", "--chat", "--chat-validation",
                 "--live-preview", "--feature-controls", "--workload-thp-disable", "--post-load-release"):
        assert flag in argv, flag
    assert "--snapshots" not in argv  # snapshots stay off
    assert argv[argv.index("--scene-model-set") + 1] == "lfm3-vl"
    assert argv[argv.index("--engine") + 1] == str(launch.prefix / "assets/engine/yolov8n.engine")
    assert argv[argv.index("--llama-server") + 1] == str(launch.prefix / "assets/llama.cpp/build/bin/llama-server")
    assert argv[argv.index("--scene-model") + 1] == str(launch.prefix / "assets/models/lfm3-vl/model.gguf")
    assert argv[argv.index("--identity-dir") + 1] == str(launch.data / "identity")
    assert argv[argv.index("run") + 1] == str(launch.data / "config/live.yaml")
    given = (launch.tmp / "admission.args").read_text().split("\n")  # the admission check is given this installation's files
    assert given[0] == str(launch.data / "config/live.yaml") and given[1] == str(launch.data / "identity")
    assert given[given.index("--engine") + 1] == str(launch.prefix / "assets/engine/yolov8n.engine")
    assert given[given.index("--weights-dir") + 1] == str(launch.prefix / "assets/deepface-home/.deepface/weights")


def missing(launch: Launch, case: str) -> dict[str, str]:
    if case == "no_environment":
        (launch.prefix / "venv/bin/python").unlink()
    elif case == "no_model":
        (launch.prefix / "assets/models/lfm3-vl/model.gguf").unlink()
    elif case == "other_model":
        (launch.prefix / "assets/models/lfm3-vl/model.gguf").write_bytes(b"MODEL")
    elif case == "no_engine":
        (launch.prefix / "assets/engine/yolov8n.engine").unlink()
    elif case == "other_llama_build":
        launch.fixture("llama.commit", "0123456789\n")
    elif case == "no_enrollment":
        (launch.data / "identity/gallery.sealed").unlink()
    elif case == "not_admitted":
        launch.fixture("admission.out", REFUSED)
        launch.fixture("admission.rc", "1\n")
    elif case == "telegram":
        return {"SENTINEL_TELEGRAM_BOT_TOKEN": "x"}
    elif case == "edited_package":
        (launch.installed / "lib/features.py").write_text("print('edited')\n")
    elif case == "stray_module":
        (launch.installed / "app/src/sentinel/extra.py").write_text("")
    elif case == "no_configuration":
        (launch.data / "config/live.yaml").unlink()
    elif case == "face_adapter_removed":
        path = launch.data / "config/live.yaml"
        path.write_text("".join(line for line in path.read_text().splitlines(True) if "legacy-deepface" not in line))
    elif case == "notifications_added":
        path = launch.data / "config/live.yaml"
        path.write_text(path.read_text() + "this_is_not_a_setting: true\n")
    else:
        raise AssertionError(case)
    return {}


@pytest.mark.parametrize("case,expected", [
    ("no_environment", "the Python environment is not there"),
    ("no_model", "asset scene_model: MISSING"),
    ("other_model", "asset scene_model: WRONG CHECKSUM"),
    ("no_engine", "asset detector_engine: MISSING"),
    ("other_llama_build", "is not the llama.cpp build this package needs (it reports commit '0123456789'"),
    ("no_enrollment", "there is no enrollment at"),
    ("not_admitted", "an admission decision is not as this package expects"),
    ("telegram", "unset the Telegram variables first (external notifications stay off)"),
    ("edited_package", "differ from their checksums"),
    ("stray_module", "files that are not part of it: app/src/sentinel/extra.py"),
    ("no_configuration", "your configuration is not at"),
    ("face_adapter_removed", "the face adapter is not as expected"),
    ("notifications_added", "the configuration did not validate"),
])
def test_whatever_is_missing_or_refused_stops_the_check_and_the_launch_with_what_to_do(
        launch: Launch, case: str, expected: str) -> None:
    env = missing(launch, case)
    result = launch.check(**env)
    assert result.returncode == 1 and expected in said(result), said(result)
    assert "BETA_CHECK=passed" not in result.stdout
    if case not in ("edited_package", "stray_module"):  # those stop before the launcher's own closing line
        assert "The camera was not opened." in result.stdout
    if case == "no_enrollment":
        assert "nobody else's gallery comes with this package" in said(result)
    if case == "not_admitted":
        assert "detector_engine=refused" in result.stdout and "nothing is bypassed here" in said(result)
        assert "What still blocks another device" in said(result)
    terminal = Terminal(launch)  # the launch stops at the same place, before any prompt
    assert terminal.finish() == 1
    assert expected in terminal.text() and "type START" not in terminal.text()
    assert launch.observations() == [] and not (launch.tmp / "runtime.json").exists()


# ---------------------------------------------------------------- the launch

def to_the_start_prompt(terminal: Terminal) -> None:
    assert terminal.wait_for(r"What opens the camera : only typing START"), terminal.text()
    assert terminal.wait_for(r"camera MAIN-stream URL \(hidden, never saved\): "), terminal.text()
    terminal.type(URL)
    assert terminal.wait_for(r"gallery passphrase \(hidden, never saved\): "), terminal.text()
    terminal.type(PASSPHRASE)
    assert terminal.wait_for(r"THE NEXT STEP OPENS THE CAMERA"), terminal.text()
    assert terminal.wait_for(r"type START to open the camera now, or press Enter to stop without opening it: ")


@pytest.mark.parametrize("answer", ["", "start", "yes", "START "])
def test_the_camera_is_not_opened_unless_start_is_typed(launch: Launch, answer: str) -> None:
    terminal = Terminal(launch)
    to_the_start_prompt(terminal)
    terminal.type(answer)
    assert terminal.finish() == 1
    text = terminal.text()
    assert "STOP: you did not type START" in text and "The camera was not opened." in text
    assert not (launch.tmp / "runtime.json").exists() and not (launch.tmp / "guard.json").exists()
    assert launch.secrets_in_files() == [] and URL not in text and PASSPHRASE not in text and "s3cret-pw" not in text


def test_start_launches_once_and_the_secrets_reach_only_the_runtimes_environment(launch: Launch) -> None:
    launch.fixture("desktop", "active\n")  # a device with a desktop: PREPARE is offered and typed
    before = hashlib.sha256((launch.data / "identity/gallery.sealed").read_bytes()).hexdigest()
    terminal = Terminal(launch)
    assert terminal.wait_for(r"type PREPARE to do that now"), terminal.text()
    terminal.type("PREPARE")
    to_the_start_prompt(terminal)
    assert not (launch.tmp / "runtime.json").exists()  # still nothing started: START has not been typed
    terminal.type("START")
    assert terminal.wait_for(r"Start nothing else on this device until OBS_exit"), terminal.text()
    assert terminal.wait_for(r"Features this run started with:", 30), terminal.text()
    assert (launch.tmp / "runtime.json").exists()
    terminal.interrupt()
    assert terminal.finish() == 0, terminal.text()
    text = terminal.text()
    assert re.search(r"Observation ended: status=operator_interrupt .* camera_opened=yes clean=yes", text), text
    assert "ended_by=you (Ctrl-C or stop.sh)" in text and "enrollment=unchanged" in text and "OBS_exit=0" in text
    seen = json.loads((launch.tmp / "runtime.json").read_text())
    assert seen["url_sha256"] == sha(URL) and seen["passphrase_sha256"] == sha(PASSPHRASE)  # given, through the environment
    assert not any(URL in argument or PASSPHRASE in argument for argument in seen["argv"])
    assert seen["hup_ignored"] and not seen["telegram"]
    guard = json.loads((launch.tmp / "guard.json").read_text())["guard_args"]
    assert guard[guard.index("--duration-s") + 1] == "3600" and guard[guard.index("--require-ready") + 1] == "face_chat_windows"
    assert "--step4-headroom" in guard and guard[guard.index("--scene-layers") + 1] == "25/25"
    # Nowhere else: not on the terminal (the prompts are hidden), not in any file the run left.
    assert URL not in text and PASSPHRASE not in text and "s3cret-pw" not in text
    assert launch.secrets_in_files() == []
    (observation,) = launch.observations()
    assert observation.stat().st_mode & 0o777 == 0o700
    record = (observation / "launch.txt").read_text()
    assert "kind=live_observation_validates_nothing" in record and "package=sentinel-personal-beta-949da67" in record
    assert "ended_by=you" in record and "enrollment=unchanged" in record
    assert hashlib.sha256((launch.data / "identity/gallery.sealed").read_bytes()).hexdigest() == before
    calls = launch.calls()
    assert calls.index("sudo systemctl stop display-manager") < calls.index("sudo sysctl -w vm.drop_caches=1")
    assert calls[-1] == "sudo systemctl start display-manager"  # the desktop comes back when the run has ended
    assert not (observation / "guard.pid").exists()


@pytest.mark.parametrize("url,passphrase,expected", [
    ("http://tester:s3cret-pw@10.9.8.7/main", PASSPHRASE, "that is not an rtsp:// URL (it is not shown or saved)"),
    (URL, "", "the passphrase is empty"),
])
def test_a_wrong_address_or_an_empty_passphrase_stops_before_the_start_prompt_and_is_not_shown(
        launch: Launch, url: str, passphrase: str, expected: str) -> None:
    terminal = Terminal(launch)
    assert terminal.wait_for(r"camera MAIN-stream URL \(hidden, never saved\): ")
    terminal.type(url)
    assert terminal.wait_for(r"gallery passphrase \(hidden, never saved\): ")
    terminal.type(passphrase)
    assert terminal.finish() == 1
    text = terminal.text()
    assert expected in text and "The camera was not opened." in text and "type START" not in text
    assert "s3cret-pw" not in text and launch.secrets_in_files() == [] and not (launch.tmp / "runtime.json").exists()


def test_declining_close_or_prepare_stops_without_closing_preparing_or_opening_anything(launch: Launch) -> None:
    launch.fixture("scan.rc", "3\n")  # optional processes run
    terminal = Terminal(launch)
    assert terminal.wait_for(r"type CLOSE to close the processes listed")
    terminal.type("")
    assert terminal.finish() == 1 and "nothing was closed: the run needs them closed" in terminal.text()
    assert "closed" not in launch.calls()
    launch.fixture("scan.rc", "0\n")
    launch.fixture("desktop", "active\n")
    terminal = Terminal(launch)
    assert terminal.wait_for(r"type PREPARE to do that now")
    terminal.type("")
    assert terminal.finish() == 1
    assert "not prepared; the desktop and the page cache were left as they were" in terminal.text()
    assert not any(call.startswith("sudo") for call in launch.calls())
    assert not (launch.tmp / "runtime.json").exists()


def test_a_shell_inside_the_desktop_is_told_to_use_ssh_before_prepare_would_stop_it(launch: Launch) -> None:
    launch.fixture("desktop", "active\n")
    launch.env["XDG_SESSION_TYPE"] = "wayland"
    terminal = Terminal(launch)
    assert terminal.finish() == 1
    assert "this shell runs inside the desktop that PREPARE would stop" in terminal.text()
    assert not any(call.startswith("sudo") for call in launch.calls())


def test_stop_sh_signals_nothing_when_nothing_runs_and_stops_a_running_observation_cleanly(launch: Launch) -> None:
    idle = launch.run("stop.sh", package=launch.installed)
    assert idle.returncode == 0 and "No observation folder exists; nothing was signalled." in idle.stdout
    terminal = Terminal(launch)
    to_the_start_prompt(terminal)
    terminal.type("START")
    assert terminal.wait_for(r"Features this run started with:", 30), terminal.text()
    launch.command("pgrep", 'case "$*" in *"--data-dir"*) exit 1 ;; esac; exit 1')
    stopped = launch.run("stop.sh", package=launch.installed)
    assert stopped.returncode == 0, said(stopped)
    assert "waiting for the clean stop" in stopped.stdout and "The guard has stopped." in stopped.stdout
    assert terminal.finish() == 0, terminal.text()
    assert "clean=yes" in terminal.text() and "OBS_exit=0" in terminal.text()
    again = launch.run("stop.sh", package=launch.installed)
    assert again.returncode == 0 and "No observation is running; nothing was signalled." in again.stdout
