"""The personal-beta package's launcher (start.sh, check.sh, stop.sh), driven as a tester would, with stand-ins.

Nothing real is started: the runtime, the memory guard, sudo, systemctl, the process scan and llama-server are
stand-ins made by this file; the camera address and the passphrase are invented strings. The launcher scripts
themselves are the package's own, with three substitutions that the shipped package does not have: the memory reading
comes from a fixture file (one line of start.sh), whether the L4T driver library is there is asked of the synthetic
device instead of the machine the tests run on (one condition of start.sh; only a Jetson has that file), and the
asset manifest lists small synthetic files. The package is then sealed again, so every integrity check still runs.

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
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from beta_install_fixture import World, said, tree

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
# the installed versions, as the environment's metadata gives them (lib/artifacts.py asks; nothing is imported)
if [ "$1" = -I ] && [ "$2" = -B ] && [ "$3" = -c ]; then cat "$FIXTURE/environment.json"; exit 0; fi
# build-engine.sh: its check of the export's packages, then the export itself (no GPU, no Ultralytics: a stand-in file)
if [ "$1" = -B ] && [ "$2" = - ]; then cat > /dev/null; exit 0; fi
if [ "$1" = - ]; then
  cat > /dev/null; printf '%s\n' "$2" "$3" > "$FIXTURE/export.args"
  [ "$(cat "$FIXTURE/export.rc")" = 0 ] || exit 1
  printf 'a stand-in engine exported with %s' "$3" > yolov8n.engine; exit 0
fi
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
GIT = r'''#!/bin/bash
# Stand-in for git in build-llama.sh: no network, no clone; it answers what the script and its note ask.
case " $* " in
  *" clone "*) mkdir -p "${@: -1}/.git" ;;
  *" rev-parse HEAD "*) cat "$FIXTURE/llama.head" ;;
  *" status --porcelain "*) cat "$FIXTURE/llama.status" ;;
esac
exit 0
'''
CMAKE = r'''#!/bin/bash
# Stand-in for cmake: "configures" by writing the options it was given, "builds" small files. Nothing is compiled.
if [ "$1" = --version ]; then echo "cmake version 3.28.3"; exit 0; fi
if [ "$1" = -S ]; then
  mkdir -p "$4"; : > "$4/CMakeCache.txt"
  for option in "${@:5}"; do option="${option#-D}"; echo "${option%%=*}:STRING=${option#*=}" >> "$4/CMakeCache.txt"; done
  [ ! -s "$FIXTURE/cmake.cache.extra" ] || cat "$FIXTURE/cmake.cache.extra" >> "$4/CMakeCache.txt"
  printf '%s\n' "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++" "CMAKE_C_COMPILER:FILEPATH=$FIXTURE/home/toolchain/bin/cc" \
    "CMAKE_C_FLAGS:STRING=" "CMAKE_CXX_FLAGS:STRING=" "CMAKE_CUDA_FLAGS:STRING=" >> "$4/CMakeCache.txt"
  exit 0
fi
if [ "$1" = --build ]; then
  mkdir -p "$2/bin"
  echo "${@:3}" > "$FIXTURE/build.args"
  left=""; for name in llama-server $(cat "$FIXTURE/build.libraries"); do [ ! -e "$2/bin/$name" ] || left="$left $name"; done
  echo "left in place when the build command ran:$left" > "$FIXTURE/build.before"
  printf '#!/bin/bash\necho "version: 0.7.0-dev (build 11938, commit $(cat "$FIXTURE/llama.commit"))"\n# %s\n' "$(cat "$FIXTURE/build.salt")" > "$2/bin/llama-server"
  chmod 755 "$2/bin/llama-server"
  for name in $(cat "$FIXTURE/build.libraries"); do printf 'library %s %s' "$name" "$(cat "$FIXTURE/build.salt")" > "$2/bin/$name"; done
  exit 0
fi
exit 0
'''
LIBRARIES = ["libggml-base.so.0.27.0", "libggml-cuda.so.0.27.0", "libllama.so.0.7.0"]
CMAKE_OPTIONS = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=87",
                 "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc"]
EXPORT = {"format": "engine", "imgsz": 640, "batch": 1, "half": False, "int8": False, "dynamic": False, "simplify": True,
          "nms": False, "device": 0}
ENVIRONMENT = {"python": "3.10.12", "packages": {
    "torch": "2.9.1", "torchvision": "0.24.1", "ultralytics": "8.4.25", "tensorrt": "10.3.0", "onnx": "1.20.1",
    "onnxslim": "0.1.90", "onnxruntime-gpu": "1.23.0", "onnxruntime": "1.23.2", "tensorflow": "2.21.0",
    "tf_keras": "2.21.0", "deepface": "0.0.99", "opencv-python": "4.13.0.92", "numpy": "2.2.6"}}
ROUTE_ADMITTED = "\n".join((
    "stream_profile=main_single", "notification_channels=0", "thresholds=unchanged", "default_admission_detector_engine=refused",
    "default_admission_scene=refused", "detector_engine=the_recorded_engine_validation_run_only",
    "scene_admission=files_as_recorded_validation_run_only", "face_admission_without_validation_flag=refused",
    "face_admission_validation_run=allowed", "identity_retention_rule=proposed_defaults",
    "identity_retention_without_validation_flag=refused", "display_labels=0", "chat_admitted=False",
    "chat_validation_run_allowed=True", "resource_validation=pending", "admission_check=as_expected_for_device_artifacts")) + "\n"

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
        for script in ("start.sh", "build-engine.sh"):  # the memory reading of both comes from the fixture file
            text = (self.package / script).read_text()
            assert text.count("MEMINFO=/proc/meminfo\n") == 1
            (self.package / script).write_text(text.replace("MEMINFO=/proc/meminfo\n", f'MEMINFO="{tmp}/meminfo"\n'))
        # Whether the L4T driver library is there is asked of the synthetic device (the fact World made), not of the
        # machine the tests run on: only a Jetson has that file. What the runtime is given to preload stays the path.
        start = (self.package / "start.sh").read_text()
        assert start.count('[ -e "$L4T_LIBCUDA" ]') == 1
        (self.package / "start.sh").write_text(
            start.replace('[ -e "$L4T_LIBCUDA" ]', '[ -e "${SENTINEL_BETA_FACTS_ROOT:-}$L4T_LIBCUDA" ]'))
        files = {"scene_model": ("models/lfm3-vl/model.gguf", b"model"), "scene_mmproj": ("models/lfm3-vl/mmproj.gguf", b"mmproj"),
                 "face_weights": ("deepface-home/.deepface/weights/weights.h5", b"weights"),
                 "face_detector": ("deepface-home/.deepface/weights/detector.onnx", b"detector"),
                 "detector_weights": ("engine/yolov8n.pt", b"detector weights")}
        (tmp / "upstream").mkdir()  # what "upstream" serves: the same small files, from a local directory
        for path, content in files.values():
            (tmp / "upstream" / Path(path).name).write_bytes(content)
        assets = [{"id": key, "kind": "download", "path": path, "url": (tmp / "upstream" / Path(path).name).as_uri(),
                   "revision": "a-synthetic-revision", "size": len(content),
                   "sha256": hashlib.sha256(content).hexdigest(), "how": "Run: bash fetch-assets.sh"}
                  for key, (path, content) in files.items()]
        assets += [{"id": "llama_server", "kind": "build", "path": "llama.cpp/build/bin/llama-server", "commit": COMMIT,
                    "repo": "https://example.invalid/llama.cpp", "cmake": CMAKE_OPTIONS, "cmake_target": "llama-server",
                    "libraries": LIBRARIES, "how": "Run: bash build-llama.sh"},
                   {"id": "detector_engine", "kind": "build", "path": "engine/yolov8n.engine", "export": EXPORT,
                    "tools": {"ultralytics": "8.4.25", "tensorrt": "10.3.0"}, "how": "Run: bash build-engine.sh"}]
        (self.package / "manifest/assets.json").write_text(json.dumps({"schema": "sentinel-beta-assets/1", "assets": assets}))
        sealed_again(self.package)
        installed = self.install()
        assert installed.returncode == 0, said(installed)
        owned = subprocess.run([str(self.bin / "python3"), "-I", "-B", str(self.installed / "lib/beta.py"), "own", "--prefix",
                                str(self.prefix), "assets"], env=self.env, capture_output=True, text=True, timeout=60)
        assert owned.returncode == 0, said(owned)  # recorded as the installer's before it is made, as the scripts do
        for path, content in files.values():
            target = self.prefix / "assets" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (self.prefix / "assets/engine").mkdir(parents=True, exist_ok=True)
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
                           ("llama.commit", COMMIT[:9] + "\n"), ("calls.log", ""),
                           # the stand-in build steps and the environment's installed versions (the device-artifacts route)
                           ("environment.json", json.dumps(ENVIRONMENT)), ("export.rc", "0\n"), ("llama.head", COMMIT + "\n"),
                           ("llama.status", ""), ("build.salt", "first build\n"), ("build.libraries", " ".join(LIBRARIES)),
                           ("cmake.cache.extra", "")):
            self.fixture(name, text)
        self.script(self.bin / "git", GIT)
        self.script(self.bin / "cmake", CMAKE)
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

    def check(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return self.run("check.sh", *args, package=self.installed, **env)

    # ---- the device-artifacts route: the package's own build steps (with stand-ins), then its record ----

    @property
    def record(self) -> Path:
        return self.data / "config/device-artifacts.json"

    def build_llama(self, *args: str) -> subprocess.CompletedProcess:
        """build-llama.sh as a tester runs it; git and cmake are stand-ins, so nothing is fetched or compiled."""
        if not (self.prefix / "assets/llama.cpp/.git").is_dir():
            shutil.rmtree(self.prefix / "assets/llama.cpp")  # the fixture's ready-made binary: not a clone of the script's
        return self.run("build-llama.sh", *args, package=self.installed)

    def build_engine(self) -> subprocess.CompletedProcess:
        (self.prefix / "assets/engine/yolov8n.engine").unlink(missing_ok=True)
        return self.run("build-engine.sh", package=self.installed, stdin="BUILD\n")

    def record_artifacts(self, *args: str) -> subprocess.CompletedProcess:
        return self.run("record-artifacts.sh", *args, package=self.installed)

    def recorded(self) -> Path:
        """Both build steps run, then the record written: a device ready for the opt-in route."""
        for step in (self.build_llama(), self.build_engine(), self.record_artifacts()):
            assert step.returncode == 0, said(step)
        self.fixture("admission.out", ROUTE_ADMITTED)
        return self.record

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
    assert re.search(r"^package=sentinel-personal-beta-949da67-mod[0-9a-f]{7}-r0-worktree$", result.stdout, re.M)
    assert re.search(r"^application=the accepted code with 5 file\(s\) changed or added after acceptance \(below\); "
                     r"this is NOT the unchanged accepted build$", result.stdout, re.M)
    assert "BETA_CHECK_ROUTE" not in result.stdout and "artifact_verification" not in result.stdout  # the default path
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
    assert "--device-artifacts" not in argv  # the route is never taken unless the tester asks for it
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
    assert "--device-artifacts" not in given


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
    elif case == "no_l4t_library":
        (launch.root / "usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1").unlink()
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
    ("no_l4t_library", "the L4T CUDA driver library is not at /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"),
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
        assert "The default admission admits only the reference device's own files" in said(result)
        assert "the opt-in device-artifacts route, QUICKSTART.md, section 1" in said(result)
    launch.env.update(env)
    terminal = Terminal(launch)  # the launch stops at the same place, before any prompt
    assert terminal.finish() == 1
    assert expected in terminal.text() and "type START" not in terminal.text()
    assert launch.observations() == [] and not (launch.tmp / "runtime.json").exists()


# ---------------------------------------------------------------- the launch

def to_the_start_prompt(terminal: Terminal, prepare: bool = False) -> None:
    assert terminal.wait_for(r"What opens the camera : only typing START"), terminal.text()
    if prepare:
        assert terminal.wait_for(r"type PREPARE to do that now"), terminal.text()
        terminal.type("PREPARE")
    assert terminal.wait_for(r"camera MAIN-stream URL \(hidden, never saved\): "), terminal.text()
    terminal.type(URL)
    assert terminal.wait_for(r"gallery passphrase \(hidden, never saved\): "), terminal.text()
    terminal.type(PASSPHRASE)
    assert terminal.wait_for(r"THE NEXT STEP OPENS THE CAMERA"), terminal.text()
    assert terminal.wait_for(r"type START to open the camera now, or press Enter to stop without opening it: ")


@pytest.mark.parametrize("answer", ["", "start", "yes", "START now", "STARTED"])
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
    to_the_start_prompt(terminal, prepare=True)
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
    desktop = (observation / "desktop.txt").read_text()
    assert desktop.index("stopped_by_launcher") < desktop.index("restart=ok")  # recorded as it happened
    assert (observation / "after-guard.txt").read_text() == "RESIDUE=none\n" and "runtime_after_guard=none" in text
    recorded = (observation / "processes.txt").read_text().split()  # what descended from this run's guard, by number
    assert recorded and recorded[0].isdigit() and recorded[1].isdigit()


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
    stopped = launch.run("stop.sh", package=launch.installed)
    assert stopped.returncode == 0, said(stopped)
    assert "waiting for the clean stop" in stopped.stdout and "The guard has stopped." in stopped.stdout
    assert terminal.finish() == 0, terminal.text()
    assert "clean=yes" in terminal.text() and "OBS_exit=0" in terminal.text()
    again = launch.run("stop.sh", package=launch.installed)
    assert again.returncode == 0 and "No observation is running; nothing was signalled." in again.stdout


def test_a_desktop_that_does_not_restart_is_recorded_and_stop_sh_still_says_what_to_do(launch: Launch) -> None:
    """Review finding IR9: the failed restart was only printed; after a lost window nothing said the desktop was down."""
    launch.fixture("desktop", "active\n")
    launch.command("sudo", 'echo "sudo $*" >> "$FIXTURE/calls.log"; [ "$*" = "systemctl start display-manager" ] && exit 1; exit 0')
    terminal = Terminal(launch)
    to_the_start_prompt(terminal, prepare=True)
    terminal.type("START")
    assert terminal.wait_for(r"Features this run started with:", 30), terminal.text()
    launch.fixture("desktop", "inactive\n")  # PREPARE stopped it
    terminal.interrupt()
    assert terminal.finish() == 0, terminal.text()
    assert "The desktop did NOT restart. Run: sudo systemctl start display-manager" in terminal.text()
    (observation,) = launch.observations()
    assert "restart=FAILED" in (observation / "desktop.txt").read_text()
    later = launch.run("stop.sh", package=launch.installed)  # from another shell, with nothing left running
    assert later.returncode == 0, said(later)
    assert "No observation is running; nothing was signalled." in later.stdout
    assert "its restart is not recorded as done (it failed)" in later.stdout
    assert "sudo systemctl start display-manager" in later.stdout
    launch.fixture("desktop", "active\n")  # once it is back, the advice is not repeated
    assert "sudo systemctl start display-manager" not in launch.run("stop.sh", package=launch.installed).stdout


# ---------------------------------------------------------------- the opt-in device-artifacts route
#
# The package's own build steps run here with stand-ins for git, cmake and the export (nothing is fetched, compiled
# or loaded), then record-artifacts.sh writes the record and check.sh / start.sh are given it. What is established:
# the scripts' own logic. No real build, no engine, no model; a pass says nothing about any device's resources.

NVCC = Path("/usr/local/cuda-12.6/bin/nvcc")  # build-llama.sh requires this path to exist; it is never run as a compiler


@pytest.fixture
def routed(launch: Launch) -> Launch:
    if not os.access(NVCC, os.X_OK):
        pytest.skip("build-llama.sh stops where the CUDA 12.6 compiler's path does not exist")
    return launch


def artifacts(launch: Launch, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(launch.bin / "python3"), "-I", "-B", str(launch.installed / "lib/artifacts.py"), *args],
                          env=launch.env, capture_output=True, text=True, timeout=120)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edited(record: Path, change) -> None:
    document = json.loads(record.read_text())
    change(document)
    record.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


def test_the_build_steps_note_what_they_built_and_the_record_holds_only_what_was_observed(routed: Launch) -> None:
    launch = routed
    built = launch.build_llama()
    assert built.returncode == 0 and "Noted what this step built" in built.stdout, said(built)
    binary = launch.prefix / "assets/llama.cpp/build/bin/llama-server"
    note = json.loads((launch.prefix / "assets/provenance/llama_server.json").read_text())
    assert (note["observed"], note["step"], note["commit"], note["cmake"], note["target"]) == (
        "built_by_package_step", "build-llama.sh", COMMIT, CMAKE_OPTIONS, "llama-server")
    assert note["repository"] == "https://example.invalid/llama.cpp" and note["reports"].startswith("version: 0.7.0-dev")
    assert sorted(note["outputs"]) == sorted(["llama-server", *LIBRARIES])
    assert note["outputs"]["llama-server"] == {"sha256": digest(binary), "bytes": binary.stat().st_size}
    assert note["tools"]["cmake"] == "cmake version 3.28.3"
    # compilers by name: a path can lie under the tester's home (this one does), and a record holds no such path
    assert (note["tools"]["cmake_cuda_compiler"], note["tools"]["cmake_c_compiler"]) == ("nvcc", "cc")
    assert sorted(note) == ["cmake", "commit", "observed", "outputs", "package", "reports", "repository", "step", "target",
                            "tools", "utc"]
    assert (launch.tmp / "build.before").read_text() == "left in place when the build command ran:\n"
    exported = launch.build_engine()
    assert exported.returncode == 0 and "Noted what this step built" in exported.stdout, said(exported)
    weights, settings = (launch.tmp / "export.args").read_text().splitlines()
    assert weights == str(launch.prefix / "assets/engine/yolov8n.pt") and json.loads(settings) == EXPORT  # as given to the export
    engine = launch.prefix / "assets/engine/yolov8n.engine"
    note = json.loads((launch.prefix / "assets/provenance/detector_engine.json").read_text())
    assert (note["observed"], note["weights_sha256"], note["export"]) == (
        "built_by_package_step", hashlib.sha256(b"detector weights").hexdigest(), EXPORT)
    assert note["tools"]["ultralytics"] == "8.4.25" and note["tools"]["tensorrt"] == "10.3.0"
    assert note["outputs"] == {"yolov8n.engine": {"sha256": digest(engine), "bytes": engine.stat().st_size}}

    recorded = launch.record_artifacts()
    assert recorded.returncode == 0, said(recorded)
    assert f"device_artifacts_record={launch.record} (written)" in recorded.stdout
    assert "resource_validation=pending" in recorded.stdout and "platform_preflight=verified_combination" in recorded.stdout
    assert "TEST FIXTURE: this record's platform facts came from a test fixture, not from a device; no run uses it" in recorded.stdout
    assert "no run on this device has been measured or judged" in recorded.stdout
    assert f"Next: bash {launch.installed}/check.sh --device-artifacts '{launch.record}'" in recorded.stdout
    assert launch.record.stat().st_mode & 0o777 == 0o600
    text = launch.record.read_text()
    document = json.loads(text)
    assert sorted(document) == ["artifacts", "created_utc", "package", "platform", "purpose", "resource_validation", "schema"]
    assert document["resource_validation"] == "pending" and "it is not a resource profile and measures nothing" in document["purpose"]
    for word in ("steady", "peak", "fps", "profile_id", "criteria", str(launch.prefix), str(launch.data), str(launch.home)):
        assert word not in text.replace("resource profile", ""), word  # no measurement and no path of the installation
    seen = {role: item["provenance"] for role, item in document["artifacts"].items()}
    assert seen["llama_server"]["observed"] == "built_by_package_step" and seen["llama_server"]["commit"] == COMMIT
    assert seen["llama_server"]["cmake"] == CMAKE_OPTIONS
    assert sorted(seen["llama_server"]) == ["cmake", "commit", "observed", "package", "reports", "repository", "step",
                                            "target", "tools", "utc"]  # the schema's keys, nothing else of the note
    assert sorted(seen["detector_engine"]) == ["export", "observed", "package", "step", "tools", "utc", "weights_sha256"]
    assert seen["detector_engine"]["observed"] == "built_by_package_step" and seen["detector_engine"]["export"] == EXPORT
    assert document["artifacts"]["llama_server"]["file"]["sha256"] == digest(binary)
    assert [item["name"] for item in document["artifacts"]["llama_server"]["libraries"]] == sorted(LIBRARIES)
    for role in ("scene_model", "scene_mmproj", "face_model_weights", "face_detector_weights"):
        # the fixture put these files in place: they equal the pinned checksum, and no download was observed
        assert seen[role]["observed"] == "verified_against_pinned_checksum", role
        assert seen[role]["statement"] == "this file equals the pinned SHA-256; this package did not observe its download"
    platform = document["platform"]
    assert (platform["l4t"], platform["os"], platform["tensorrt"], platform["python"]) == (
        "36.4.7", "ubuntu 24.04", "10.3.0.30-1+cuda12.5", "3.10.12")
    assert platform["packages"] == ENVIRONMENT["packages"] and platform["preflight"] == "verified_combination"
    assert platform["test_fixture"] == "facts were read from a test fixture, not from a device"

    # the runtime's own, stricter reader takes the record the package wrote, and says the same about it
    from sentinel import device_artifacts

    read = device_artifacts.load(launch.record)
    assert read.files["detector_engine"].sha256 == digest(engine)
    assert read.startup_entry(scene=True)["resource_validation"] == "pending"
    assert device_artifacts.PURPOSE == document["purpose"] and device_artifacts.SCHEMA == document["schema"]
    assert device_artifacts.llama_problem(read, model_set="lfm3-vl", library_names=tuple(LIBRARIES)) == (
        "the llama.cpp build is not https://github.com/ggml-org/llama.cpp at the pinned commit "
        "19cce6919992fdffd92e0ac152caaf1e2b8f254e")  # read, and judged by the runtime's own pins (the fixture's repository)
    # ... and the runtime never runs on a record whose platform facts came from a fixture, as this one's did
    assert device_artifacts.engine_problem(read) == (
        "the record's platform facts were read from a test fixture, not from a device: no run uses it")

    # a download is recorded as this package's own only when fetch-assets.sh made it
    model = launch.prefix / "assets/models/lfm3-vl/model.gguf"
    model.unlink()
    fetched = launch.run("fetch-assets.sh", "--only", "scene_model", package=launch.installed)
    assert fetched.returncode == 0 and "asset scene_model: ok" in fetched.stdout, said(fetched)
    launch.record.rename(launch.record.with_suffix(".old"))
    assert launch.record_artifacts().returncode == 0
    seen = {role: item["provenance"] for role, item in json.loads(launch.record.read_text())["artifacts"].items()}
    assert seen["scene_model"]["observed"] == "downloaded_by_package_step" and seen["scene_model"]["step"] == "fetch-assets.sh"
    assert seen["scene_model"]["url"] == (launch.tmp / "upstream/model.gguf").as_uri()
    assert seen["scene_mmproj"]["observed"] == "verified_against_pinned_checksum"
    # ... and only for the very copy it wrote: the same bytes put there again are a file whose download nobody observed
    copy = model.read_bytes()
    model.unlink()
    model.write_bytes(copy)
    os.utime(model, (model.stat().st_atime, model.stat().st_mtime + 120))
    launch.record.rename(launch.record.with_suffix(".older"))
    assert launch.record_artifacts().returncode == 0
    seen = json.loads(launch.record.read_text())["artifacts"]["scene_model"]["provenance"]
    assert seen["observed"] == "verified_against_pinned_checksum" and "step" not in seen


@pytest.mark.parametrize("case", ["never_built_by_the_steps", "binary_replaced", "library_replaced", "engine_replaced"])
def test_a_built_file_that_the_packages_steps_did_not_make_is_inspected_only_and_nothing_is_recorded(
        routed: Launch, case: str) -> None:
    launch = routed
    binary = launch.prefix / "assets/llama.cpp/build/bin/llama-server"
    if case == "never_built_by_the_steps":  # a build and an engine that were simply put there
        for name in LIBRARIES:
            (binary.parent / name).write_bytes(b"a library from somewhere")
    else:
        assert launch.build_llama().returncode == 0 and launch.build_engine().returncode == 0
        if case == "binary_replaced":
            binary.write_text(binary.read_text() + "# rebuilt by hand\n")
        elif case == "library_replaced":
            (binary.parent / LIBRARIES[1]).write_bytes(b"another build's library")
        else:
            (launch.prefix / "assets/engine/yolov8n.engine").write_bytes(b"an engine from somewhere")
    result = launch.record_artifacts()
    assert result.returncode == 1 and not launch.record.exists(), said(result)
    llama = case != "engine_replaced"
    engine = case in ("never_built_by_the_steps", "engine_replaced")
    assert ("artifact llama_server: llama-server SHA-256 " + digest(binary) + " [inspected_only]" in result.stdout) is llama
    assert ("NOT USABLE FOR A LAUNCH: llama_server: its build was not observed by this package's step. A checksum does "
            "not show how a file was built." in result.stdout) is llama
    assert ("NOT USABLE FOR A LAUNCH: detector_engine: its build was not observed" in result.stdout) is engine
    assert "nothing was recorded: a record of these files could not be used for a launch" in result.stderr


@pytest.mark.parametrize(("case", "told"), [
    ("another_commit", "the llama.cpp source tree is not at the pinned commit"),
    ("local_changes", "the llama.cpp source tree has local changes"),
    ("other_options", "the build tree was not configured with -DGGML_CUDA=ON (it has 'OFF')"),
    ("library_not_built", "the build ended without"),
    ("no_marker", "was not started by build-llama.sh for the pinned commit and options"),
    ("marker_of_other_options", "was not started by build-llama.sh for the pinned commit and options"),
    ("extra_compiler_flags", "the build tree was configured with extra compiler flags (CMAKE_CXX_FLAGS=-O1 -march=native)"),
    ("other_export_settings", "the export settings given are not the asset manifest's"),
    ("other_weights", "are not the pinned file (WRONG CHECKSUM)"),
])
def test_a_step_notes_nothing_for_a_tree_or_an_export_that_is_not_the_pinned_one(routed: Launch, case: str, told: str) -> None:
    launch = routed
    assert launch.build_llama().returncode == 0 and launch.build_engine().returncode == 0
    step, extra = "build-llama", []
    note = launch.prefix / "assets/provenance/llama_server.json"
    cache = launch.prefix / "assets/llama.cpp/build/CMakeCache.txt"
    if case == "another_commit":
        launch.fixture("llama.head", "0" * 40 + "\n")
    elif case == "local_changes":
        launch.fixture("llama.status", " M src/llama.cpp\n")
    elif case == "other_options":
        cache.write_text(cache.read_text().replace("GGML_CUDA:STRING=ON", "GGML_CUDA:STRING=OFF"))
    elif case == "library_not_built":
        (launch.prefix / "assets/llama.cpp/build/bin" / LIBRARIES[0]).unlink()
    elif case == "no_marker":
        (cache.parent / ".sentinel-beta-build").unlink()
    elif case == "marker_of_other_options":
        (cache.parent / ".sentinel-beta-build").write_text(f"{COMMIT} -DCMAKE_BUILD_TYPE=Debug\n")
    elif case == "extra_compiler_flags":
        cache.write_text(cache.read_text().replace("CMAKE_CXX_FLAGS:STRING=", "CMAKE_CXX_FLAGS:STRING=-O1 -march=native"))
    else:
        step, note = "build-engine", launch.prefix / "assets/provenance/detector_engine.json"
        extra = ["--export", json.dumps({**EXPORT, "imgsz": 1280} if case == "other_export_settings" else EXPORT)]
        if case == "other_weights":
            (launch.prefix / "assets/engine/yolov8n.pt").write_bytes(b"DETECTOR WEIGHTS")
    note.unlink()
    result = artifacts(launch, "note", "--package", str(launch.installed), "--step", step, *extra)
    assert result.returncode == 1 and told in result.stderr and "nothing was noted" in result.stderr, said(result)
    assert not note.exists()


def test_the_build_script_builds_only_in_a_tree_it_started_and_links_again_what_it_then_notes(routed: Launch) -> None:
    launch = routed
    source = launch.prefix / "assets/llama.cpp"
    binary = source / "build/bin/llama-server"
    assert launch.build_llama().returncode == 0
    first = json.loads((launch.prefix / "assets/provenance/llama_server.json").read_text())["outputs"]["llama-server"]["sha256"]
    # a binary from somewhere else put in the script's own tree, and the script run again: it is not noted as built.
    # The script removes what an earlier run left, so the build command makes the file again and that one is hashed
    binary.write_text("#!/bin/bash\necho 'version: 0.7.0-dev (build 11938, commit 19cce6919)'\n# built elsewhere\n")
    foreign = digest(binary)
    again = launch.build_llama()
    assert again.returncode == 0, said(again)
    assert (launch.tmp / "build.before").read_text() == "left in place when the build command ran:\n"
    noted = json.loads((launch.prefix / "assets/provenance/llama_server.json").read_text())["outputs"]["llama-server"]["sha256"]
    assert noted == digest(binary) == first and noted != foreign
    # compiler or linker flags from the shell would make another build: refused before anything is configured
    flagged = launch.run("build-llama.sh", package=launch.installed, CXXFLAGS="-O1")
    assert flagged.returncode == 1 and "CXXFLAGS is set in this shell: unset it first" in flagged.stderr, said(flagged)
    # a build tree that the script did not start (configured by hand, perhaps with other settings) is never built on
    (source / "build/.sentinel-beta-build").unlink()
    (launch.prefix / "assets/provenance/llama_server.json").unlink()
    by_hand = launch.build_llama()
    assert by_hand.returncode == 1 and "exists and was not started by this script for this commit and these options" in by_hand.stderr
    assert not (launch.prefix / "assets/provenance/llama_server.json").exists()
    shutil.rmtree(source / "build")
    (source / "build").mkdir()
    (source / "build/.sentinel-beta-build").write_text(f"{'0' * 40} {' '.join(CMAKE_OPTIONS)}\n")  # another commit's tree
    assert launch.build_llama().returncode == 1 and not (launch.prefix / "assets/provenance/llama_server.json").exists()


def test_the_build_uses_four_jobs_unless_the_tester_asks_for_another_number(routed: Launch) -> None:
    """Found by the installation rehearsal: the script compiled with four jobs and offered no way to use fewer on a
    device that is doing something else. The target is still built by its name, and the note is the same."""
    launch = routed
    assert launch.build_llama().returncode == 0
    assert (launch.tmp / "build.args").read_text().split() == ["--target", "llama-server", "-j", "4"]
    two = launch.build_llama("--jobs", "2")
    assert two.returncode == 0 and "Building llama-server with 2 parallel job(s)" in two.stdout, said(two)
    assert (launch.tmp / "build.args").read_text().split() == ["--target", "llama-server", "-j", "2"]
    assert json.loads((launch.prefix / "assets/provenance/llama_server.json").read_text())["observed"] == "built_by_package_step"
    (launch.tmp / "build.args").unlink()
    for bad in (["--jobs"], ["--jobs", "0"], ["--jobs", "9"], ["--jobs", "two"], ["--jobs", "2 4"], ["--jobs", ""]):
        refused = launch.build_llama(*bad)
        assert refused.returncode == 1 and "--jobs needs a number from 1 to 8" in refused.stderr, (bad, said(refused))
    unknown = launch.build_llama("--fast")
    assert unknown.returncode == 2 and "usage: bash build-llama.sh [--jobs N] [--configure-only]" in unknown.stdout
    assert not (launch.tmp / "build.args").exists()  # nothing was built for any of them
    configured = launch.build_llama("--jobs", "1", "--configure-only")
    assert configured.returncode == 0 and "Configured only, as asked; nothing was compiled." in configured.stdout
    assert not (launch.tmp / "build.args").exists()


def test_the_record_and_its_check_use_the_interpreter_and_binding_the_installation_was_made_with(routed: Launch) -> None:
    """A tester whose Python 3.10 is not on the PATH names it at installation; a later shell without that variable
    must get the same preflight answer (the installation's own record of it), not a refusal."""
    launch = routed
    custom = launch.tmp / "custom/python3.10"
    launch.script(custom, '#!/bin/bash\ncase "$*" in *version_info*) echo 3.10 ;; esac; exit 0\n')
    binding = launch.tmp / "custom/dist-packages"
    (binding / "tensorrt").mkdir(parents=True)
    (binding / "tensorrt/__init__.py").write_text("")
    named = launch.install(SENTINEL_BETA_PYTHON310=str(custom), SENTINEL_BETA_TENSORRT_PYTHON=str(binding))
    assert named.returncode == 0, said(named)
    shutil.rmtree(launch.root / "usr/lib/python3.10/dist-packages")  # the binding is nowhere but where the tester named it
    plain = launch.run("preflight.sh", package=launch.installed)  # a later shell, without the two variables
    assert plain.returncode == 1 and "no TensorRT Python binding for Python 3.10" in plain.stdout
    # refreshing the package from that later shell uses what the installation was made with (found by the rehearsal)
    again = launch.install()
    assert again.returncode == 0 and "PREFLIGHT: this is the verified combination." in again.stdout, said(again)
    conf = (launch.prefix / "install.conf").read_text()
    assert f"PYTHON310={custom}\n" in conf and f"TENSORRT_PYTHON={binding}\n" in conf
    elsewhere = launch.tmp / "another/dist-packages"  # and what is named anew still wins
    (elsewhere / "tensorrt").mkdir(parents=True)
    (elsewhere / "tensorrt/__init__.py").write_text("")
    assert launch.install(SENTINEL_BETA_TENSORRT_PYTHON=str(elsewhere)).returncode == 0
    assert f"TENSORRT_PYTHON={elsewhere}\n" in (launch.prefix / "install.conf").read_text()
    assert launch.install(SENTINEL_BETA_TENSORRT_PYTHON=str(binding)).returncode == 0
    record = launch.recorded()
    assert json.loads(record.read_text())["platform"]["preflight"] == "verified_combination"
    checked = launch.check("--device-artifacts", str(record))
    assert checked.returncode == 0 and "artifact_verification=passed" in checked.stdout, said(checked)


def test_the_engine_export_is_refused_before_anything_loads_when_free_memory_is_low(routed: Launch) -> None:
    """Found by the installation rehearsal: with the page cache full the export crashed in its first GPU allocation
    with an error that names NVML. The script now reads free memory first and says what to do; "available" memory
    does not count."""
    launch = routed
    launch.fixture("meminfo", "MemTotal: 7802736 kB\nMemFree: 820000 kB\nMemAvailable: 5400000 kB\n")
    refused = launch.build_engine()
    assert refused.returncode == 1, said(refused)
    assert "MemFree is 839680000 B, below the 1500000000 B needed before the detector is loaded on the GPU" in refused.stderr
    assert "only \"available\" (page cache) does not count" in refused.stderr and "Nothing was built" in refused.stderr
    assert "sync; sudo sysctl -w vm.drop_caches=1" in refused.stderr and "type BUILD" not in said(refused)
    assert not (launch.tmp / "export.args").exists()  # the export was never started
    assert not (launch.prefix / "assets/engine/yolov8n.engine").exists()
    assert not (launch.prefix / "assets/provenance/detector_engine.json").exists()
    assert not any(call.startswith("sudo") for call in launch.calls())  # the script frees nothing itself
    launch.fixture("meminfo", "MemTotal: 7802736 kB\nMemAvailable: 5400000 kB\n")
    unreadable = launch.build_engine()
    assert unreadable.returncode == 1 and "free memory could not be read" in unreadable.stderr
    launch.fixture("meminfo", "MemTotal: 7802736 kB\nMemFree: 1464844 kB\nMemAvailable: 5400000 kB\n")  # 1,500,000,256 B
    built = launch.build_engine()
    assert built.returncode == 0 and "MemFree is 1500000256 B (at least 1500000000 B is required" in built.stdout, said(built)
    script = (launch.installed / "build-engine.sh").read_text()
    assert script.index("MIN_FREE_BYTES=1500000000") < script.index("type BUILD to build the engine now")
    assert "The least the export needs was NOT measured" in script and "nothing here frees memory for you" in script
    # what the reference device showed is stated as that device's, with its three attempts, and no duration is promised
    assert "completed in 187 s" in script and "ended the build after 136 s" in script and "a few minutes" not in script
    assert "no duration is promised" in script and "not a figure known to be enough" in script


def test_an_export_that_fails_leaves_no_engine_note_and_the_script_says_so(routed: Launch) -> None:
    launch = routed
    launch.fixture("export.rc", "1\n")
    result = launch.build_engine()
    assert result.returncode == 1 and "the export failed" in said(result), said(result)
    assert "MemFree was 6451200000 B before it and is 6451200000 B now" in said(result)
    assert "the GPU could not get memory" in said(result) and "Nothing was noted, and no engine is in place" in said(result)
    assert not (launch.prefix / "assets/provenance/detector_engine.json").exists()
    assert not (launch.prefix / "assets/engine/yolov8n.engine").exists()


def test_a_record_is_never_replaced_and_an_unverified_platform_is_recorded_only_on_the_testers_word(routed: Launch) -> None:
    launch = routed
    record = launch.recorded()
    first = record.read_bytes()
    time.sleep(1.1)  # another second: only the record's own time could differ
    again = launch.record_artifacts()
    assert again.returncode == 0 and "(already there and unchanged; kept)" in again.stdout and record.read_bytes() == first
    launch.fixture("build.salt", "second build\n")  # a rebuild: other bytes
    assert launch.build_llama().returncode == 0
    refused = launch.record_artifacts()
    assert refused.returncode == 1 and "it is not replaced. Move it away yourself" in refused.stderr, said(refused)
    assert record.read_bytes() == first
    record.rename(record.with_suffix(".old"))
    launch.fact("etc/os-release", 'ID=ubuntu\nVERSION_ID="22.04"\n')  # stock JetPack 6: never verified
    unverified = launch.record_artifacts()
    assert unverified.returncode == 1 and not record.exists(), said(unverified)
    assert "NOT VERIFIED userland: Ubuntu 22.04" in unverified.stdout
    assert "this platform was NOT verified (above); nothing was recorded" in unverified.stderr
    assert "record-artifacts.sh --accept-unverified-platform" in unverified.stderr
    accepted = launch.record_artifacts("--accept-unverified-platform")
    assert accepted.returncode == 0 and "platform_preflight=not_verified_acknowledged" in accepted.stdout, said(accepted)
    assert json.loads(record.read_text())["platform"]["preflight"] == "not_verified_acknowledged"
    checked = launch.check("--device-artifacts", str(record))
    assert checked.returncode == 0, said(checked)
    assert ("platform_preflight=not_verified_acknowledged (NOT VERIFIED: nothing was ever run on this combination; "
            "acknowledged when the record was made)") in checked.stdout
    assert "platform not_verified_acknowledged_test_fixture)" in checked.stdout  # the route's closing line says it too
    launch.fact("etc/os-release", 'ID=debian\nVERSION_ID="12"\n')  # a platform the preflight fails is never recorded
    record.rename(record.with_suffix(".older"))
    failed = launch.record_artifacts("--accept-unverified-platform")
    assert failed.returncode == 1 and "the preflight failed on this device (above); nothing was recorded" in failed.stderr
    assert not record.exists()


def test_the_check_with_a_record_passes_and_says_what_it_did_and_did_not_establish(routed: Launch) -> None:
    launch = routed
    record = launch.recorded()
    result = launch.check("--device-artifacts", str(record))
    assert result.returncode == 0, said(result)
    out = result.stdout
    for role in ("detector_engine", "llama_server", "scene_model", "scene_mmproj", "face_model_weights", "face_detector_weights"):
        assert f"artifact {role}: ok (" in out, role
    assert f"device_artifacts_record={record}" in out and f"device_artifacts_record_sha256={digest(record)}" in out
    assert "artifact_verification=passed (every file equals the record, whole, by SHA-256)" in out
    assert "resource_validation=pending" in out
    assert ("resource_validation_note=no run on this device has been measured or judged. The accepted resource profile, "
            "the launch gates and the guard's limits come from the reference device: here they are limits, not evidence "
            "about this device") in out
    assert "TEST FIXTURE: this record's platform facts came from a test fixture, not from a device" in out
    assert "platform_facts=test_fixture" in out.splitlines()
    assert "admission_check=as_expected_for_device_artifacts" in out and "command_line_check=run: rtsp_url_missing" in out
    route = next(line for line in out.splitlines() if line.startswith("BETA_CHECK_ROUTE="))
    assert "opt-in, experimental validation runs only" in route and "its resources are NOT validated" in route
    assert "resource_validation=pending" in route and route.endswith("platform verified_combination_test_fixture)")
    assert out.rstrip().endswith("BETA_CHECK=passed (no credentials asked, no sudo, the camera was not opened, no model was loaded)")
    for word in ("validated=yes", "resource_validation=passed", "admitted_by_profile"):
        assert word not in out
    given = (launch.tmp / "admission.args").read_text().split("\n")  # the runtime's decisions are asked for that record
    assert given[given.index("--device-artifacts") + 1] == str(record)
    argv = json.loads((launch.tmp / "runtime-check.json").read_text())["argv"]
    assert argv[argv.index("--device-artifacts") + 1] == str(record)
    for flag in ("--face", "--face-validation", "--chat-validation", "--workload-thp-disable", "--post-load-release"):
        assert flag in argv, flag
    assert "--snapshots" not in argv and launch.observations() == [] and not (launch.tmp / "runtime.json").exists()
    assert not any(call.startswith("sudo") for call in launch.calls())
    relative = launch.run("check.sh", "--device-artifacts", os.path.relpath(record, launch.tmp), package=launch.installed)
    assert relative.returncode == 0 and f"device_artifacts_record={record}" in relative.stdout  # a relative path is this shell's
    # the default check on the same device is exactly the default: no record is read, and none is passed on
    launch.fixture("admission.out", ADMITTED)
    plain = launch.check()
    assert plain.returncode == 0 and "device_artifacts" not in plain.stdout and "BETA_CHECK_ROUTE" not in plain.stdout
    assert "--device-artifacts" not in (launch.tmp / "admission.args").read_text()
    assert "--device-artifacts" not in json.loads((launch.tmp / "runtime-check.json").read_text())["argv"]
    usage = launch.check("--device-artifacts")
    assert usage.returncode == 2 and "usage: bash start.sh [--check] [--device-artifacts FILE]" in usage.stdout


def differs(launch: Launch, case: str) -> str:
    record = launch.record
    binary = launch.prefix / "assets/llama.cpp/build/bin/llama-server"
    if case == "engine_rebuilt":
        (launch.prefix / "assets/engine/yolov8n.engine").write_bytes(b"another engine, built later")
    elif case == "library_missing":
        (binary.parent / LIBRARIES[2]).unlink()
    elif case == "binary_changed":
        binary.write_text(binary.read_text() + "# changed\n")
    elif case == "model_copied_again":
        model = launch.prefix / "assets/models/lfm3-vl/model.gguf"
        os.utime(model, (model.stat().st_atime, model.stat().st_mtime + 3600))
    elif case == "record_claims_a_validation":
        edited(record, lambda d: d.update(resource_validation="passed"))
    elif case == "record_carries_a_measurement":
        edited(record, lambda d: d.update(resource_profile={"steady_max_bytes": 4_397_879_296}))
    elif case == "provenance_carries_a_claim":
        edited(record, lambda d: d["artifacts"]["detector_engine"]["provenance"].update(resource_validation="validated"))
    elif case == "platform_carries_a_claim":
        edited(record, lambda d: d["platform"].update(resource_validated=True))
    elif case == "record_of_mistyped_export_settings":
        edited(record, lambda d: d["artifacts"]["detector_engine"]["provenance"]["export"].update(batch=True, device=False))
    elif case == "record_with_an_impossible_time":
        edited(record, lambda d: d.update(created_utc="9999-99-99T99:99:99Z"))
    elif case == "record_of_an_inspected_build":
        edited(record, lambda d: d["artifacts"]["llama_server"].update(provenance={
            "observed": "inspected_only",
            "statement": ("only this file's checksum was read; how it was built was not observed by this package's "
                          "build step, so nothing is recorded about its source or settings")}))
    elif case == "record_of_another_commit":
        edited(record, lambda d: d["artifacts"]["llama_server"]["provenance"].update(commit="0" * 40))
    elif case == "record_of_other_export_settings":
        edited(record, lambda d: d["artifacts"]["detector_engine"]["provenance"]["export"].update(imgsz=1280))
    elif case == "record_of_other_tools":
        edited(record, lambda d: d["artifacts"]["detector_engine"]["provenance"]["tools"].update(ultralytics="8.4.26"))
    elif case == "record_not_json":
        record.write_text("not a record\n")
    elif case == "record_missing":
        record.unlink()
    elif case == "tensorrt_upgraded":
        launch.dpkg({"cuda-toolkit-12-6": "12.6.11-1", "libnvinfer10": "10.7.0.23-1+cuda12.6", "libcudnn9-cuda-12": "9.3.0.75-1"})
    elif case == "tensorrt_patched":
        launch.dpkg({"cuda-toolkit-12-6": "12.6.11-1", "libnvinfer10": "10.3.0.31-1+cuda12.5", "libcudnn9-cuda-12": "9.3.0.75-1"})
    elif case == "package_upgraded":
        launch.fixture("environment.json", json.dumps({**ENVIRONMENT, "packages": {**ENVIRONMENT["packages"], "torch": "2.9.2"}}))
    elif case == "userland_changed":
        launch.fact("etc/os-release", 'ID=ubuntu\nVERSION_ID="22.04"\n')
    elif case == "refused_by_the_runtime":
        launch.fixture("admission.out", "detector_engine=refused\nadmission_check=failed:detector_engine\n")
        launch.fixture("admission.rc", "1\n")
    else:
        raise AssertionError(case)
    return str(record)


@pytest.mark.parametrize(("case", "expected", "launch_too"), [
    ("engine_rebuilt", "REFUSED: detector_engine: ", True),
    ("library_missing", "libllama.so.0.7.0 is MISSING", False),
    ("binary_changed", "has CHANGED since it was recorded", False),
    ("model_copied_again", "has the recorded content but another modification time", False),
    ("record_claims_a_validation", 'the record\'s resource_validation is not "pending": a record identifies files and validates nothing', True),
    ("record_carries_a_measurement", "the record's top level does not hold exactly", False),
    ("provenance_carries_a_claim", "the record's artifacts.detector_engine.provenance does not hold exactly", False),
    ("platform_carries_a_claim", "the record's platform does not hold exactly its versions, packages and preflight", False),
    ("record_of_mistyped_export_settings", "REFUSED: detector_engine: it was not exported from the pinned weights with the pinned settings", False),
    ("record_with_an_impossible_time", "the record's created_utc or package is not as the schema requires", False),
    ("record_of_an_inspected_build", "REFUSED: llama_server: its build was not observed by this package's step", False),
    ("record_of_another_commit", "REFUSED: llama_server: it was not built from the pinned repository and commit with the pinned options", False),
    ("record_of_other_export_settings", "REFUSED: detector_engine: it was not exported from the pinned weights with the pinned settings", False),
    ("record_of_other_tools", "REFUSED: detector_engine: it was not exported with the pinned tool versions", False),
    ("record_not_json", "the record is not a JSON document", False),
    ("record_missing", "there is no device-artifacts record at", True),
    ("tensorrt_upgraded", "REFUSED: platform: the preflight fails on this device now", False),
    ("tensorrt_patched", 'REFUSED: platform: tensorrt is "10.3.0.31-1+cuda12.5" now; the record has "10.3.0.30-1+cuda12.5"', False),
    ("package_upgraded", "REFUSED: platform: packages is ", False),
    ("userland_changed", "REFUSED: platform: the preflight now reads 'not_verified_acknowledged', the record has 'verified_combination'", False),
    ("refused_by_the_runtime", "an admission decision is not as the device-artifacts route expects", True),
])
def test_whatever_is_not_as_the_record_says_stops_the_check_and_the_launch(
        routed: Launch, case: str, expected: str, launch_too: bool) -> None:
    launch = routed
    launch.recorded()
    record = differs(launch, case)
    result = launch.check("--device-artifacts", record)
    assert result.returncode == 1 and expected in said(result), said(result)
    assert "BETA_CHECK=passed" not in result.stdout and "BETA_CHECK_ROUTE" not in result.stdout
    assert "The camera was not opened." in result.stdout
    assert ("Nothing is bypassed" in result.stdout) is (case != "record_missing")  # a missing record: how to make one
    assert "artifact_verification=passed" not in result.stdout
    if "REFUSED: " in expected:
        assert "artifact_verification=FAILED" in result.stdout and "never edit a record or a file's time by hand" in result.stdout
    if case != "refused_by_the_runtime":
        assert not (launch.tmp / "admission.args").exists()  # stopped before the runtime's decisions were even asked for
    if launch_too:
        terminal = Terminal(launch, "--device-artifacts", record)  # the launch stops at the same place, before any prompt
        assert terminal.finish() == 1
        assert expected in terminal.text() and "type START" not in terminal.text()
    assert launch.observations() == [] and not (launch.tmp / "runtime.json").exists()


def test_a_launch_with_a_record_keeps_every_gate_and_says_that_nothing_was_validated(routed: Launch) -> None:
    launch = routed
    record = launch.recorded()
    script = (launch.installed / "start.sh").read_text()  # the gates and limits are the accepted launcher's, as they were
    for line in ("DURATION_S=3600  # the guard's own maximum", "FLOOR=$((5647138816 + HELD_PICTURES_BYTES))",
                 "IN_USE_LIMIT=$((1327566848 - HELD_PICTURES_BYTES))", '[ "$A" = START ] || stop_now "you did not type START"',
                 '--require-ready face_chat_windows --scene-layers 25/25 --duration-s "$DURATION_S" --ready-timeout-s 300'):
        assert line in script, line
    declined = Terminal(launch, "--device-artifacts", str(record))
    to_the_start_prompt(declined)
    declined.type("")  # Enter: with a record too, only START opens the camera
    assert declined.finish() == 1 and "STOP: you did not type START" in declined.text()
    assert not (launch.tmp / "runtime.json").exists() and not (launch.tmp / "guard.json").exists()
    for folder in launch.observations():
        assert not (folder / "device-artifacts.json").exists()  # nothing of the route was written before START
        shutil.rmtree(folder)
    time.sleep(1.1)  # observation folders are named by the second

    terminal = Terminal(launch, "--device-artifacts", str(record))
    to_the_start_prompt(terminal)
    before = terminal.text()
    assert "DEVICE-ARTIFACTS ROUTE (opt-in, experimental)" in before and "NOT validated" in before
    assert "Resource" in before and "validation is PENDING: nothing was measured on this device." in before
    assert "here they are limits, not evidence." in before and "Platform      : TEST FIXTURE: the platform facts did not come from this device (the verified combination)" in before
    assert "DEVICE-ARTIFACTS ROUTE: this device's own files, as recorded; this device's resources are NOT validated (pending)." in before
    assert not (launch.tmp / "runtime.json").exists()  # still nothing started
    terminal.type("START")
    assert terminal.wait_for(r"Features this run started with:", 30), terminal.text()
    terminal.interrupt()
    assert terminal.finish() == 0, terminal.text()
    text = terminal.text()
    assert re.search(r"Observation ended: status=operator_interrupt .* camera_opened=yes clean=yes", text), text
    assert ("route=device artifacts (experimental validation run on this device's own files; resource_validation=pending: "
            "this run measured and validated nothing)") in text and "OBS_exit=0" in text
    (observation,) = launch.observations()
    kept = observation / "device-artifacts.json"
    assert kept.read_bytes() == record.read_bytes() and kept.stat().st_mode & 0o777 == 0o600
    seen = json.loads((launch.tmp / "runtime.json").read_text())
    argv = seen["argv"]
    assert argv[argv.index("--device-artifacts") + 1] == str(kept)  # the run reads its own copy: the bytes that were checked
    assert argv.count("--device-artifacts") == 1
    for flag in ("--scene", "--face", "--face-validation", "--identity-retention", "--chat", "--chat-validation",
                 "--live-preview", "--feature-controls", "--workload-thp-disable", "--post-load-release"):
        assert flag in argv, flag
    assert "--snapshots" not in argv and not seen["telegram"]  # snapshots and notifications stay off
    assert seen["url_sha256"] == sha(URL) and seen["passphrase_sha256"] == sha(PASSPHRASE)
    guard = json.loads((launch.tmp / "guard.json").read_text())["guard_args"]  # the guard is started exactly as without a record
    assert guard[guard.index("--duration-s") + 1] == "3600" and guard[guard.index("--require-ready") + 1] == "face_chat_windows"
    assert "--step4-headroom" in guard and guard[guard.index("--scene-layers") + 1] == "25/25"
    assert guard[guard.index("--ready-timeout-s") + 1] == "300" and "--device-artifacts" not in guard
    note = (observation / "launch.txt").read_text()
    for line in ("kind=live_observation_validates_nothing", "route=device_artifacts_experimental_validation_run",
                 f"device_artifacts_record_sha256={digest(record)}", "artifact_verification=passed",
                 "resource_validation=pending", "platform_preflight=verified_combination_test_fixture"):
        assert line in note.splitlines(), line
    assert launch.secrets_in_files() == [] and URL not in text and PASSPHRASE not in text


# ---------------------------------------------------------------- the installation's own cache and temporary locations
#
# What a script starts is seen through a recorder put in front of each stand-in: the recorder writes down the
# locations it was started with before the stand-in does anything, which is what the real program would have before
# its first import. Synthetic: no pip, no download, no build, no model, no camera.

LOCAL = {"PIP_CACHE_DIR": "cache/pip", "XDG_CACHE_HOME": "cache/xdg-cache", "XDG_CONFIG_HOME": "cache/xdg-config",
         "XDG_DATA_HOME": "cache/xdg-data", "HF_HOME": "cache/hf", "TORCH_HOME": "cache/torch",
         "YOLO_CONFIG_DIR": "cache/yolo", "DEEPFACE_HOME": "cache/deepface-home", "CUDA_CACHE_PATH": "cache/cuda",
         "KERAS_HOME": "cache/keras", "MPLCONFIGDIR": "cache/matplotlib", "TMPDIR": "tmp"}
LOCAL_REMOVED = ("PYTHONPYCACHEPREFIX", "TEMP", "TMP", "HF_HUB_CACHE")


def recorded_starts(launch: Launch, path: Path, label: str) -> None:
    original = path.with_name(path.name + ".stand-in")
    path.rename(original)
    names = " ".join((*LOCAL, *LOCAL_REMOVED, "HOME"))
    launch.script(path, f"""#!/bin/bash
{{ printf 'START {label} %s\\n' "$*"; for name in {names}; do printf '%s=%s\\n' "$name" "${{!name-<unset>}}"; done; }} >> "$FIXTURE/starts.txt"
exec "{original}" "$@"
""")


def starts(launch: Launch) -> list[tuple[str, str, dict[str, str]]]:
    found: list[tuple[str, str, dict[str, str]]] = []
    for line in (launch.tmp / "starts.txt").read_text().splitlines():
        if line.startswith("START "):
            _, label, *rest = line.split(" ", 2)
            found.append((label, rest[0] if rest else "", {}))
        else:
            name, _, value = line.partition("=")
            found[-1][2][name] = value
    return found


def set_in_the_shell(launch: Launch) -> dict[str, str]:
    return {name: str(launch.tmp / "elsewhere" / name.lower()) for name in (*LOCAL, *LOCAL_REMOVED)}


def expect_the_installations_own(launch: Launch, entries, face_weights: Path | None = None) -> None:
    assert entries
    for label, arguments, seen in entries:
        runtime = arguments.startswith("-m sentinel.cli ")
        for variable, relative in LOCAL.items():
            own = launch.prefix / relative
            if variable == "DEEPFACE_HOME" and runtime:  # the weights' own place, named by the launcher itself
                own = face_weights or launch.prefix / "assets/deepface-home"
            assert seen[variable] == str(own), (label, arguments[:60], variable)
        for variable in LOCAL_REMOVED:
            assert seen[variable] == "<unset>", (label, arguments[:60], variable)
        assert seen["HOME"] == str(launch.home), label


def test_everything_the_scripts_start_gets_the_installations_own_locations_before_it_imports_anything(routed: Launch) -> None:
    launch = routed
    recorded_starts(launch, launch.prefix / "venv/bin/python", "interpreter")
    recorded_starts(launch, launch.bin / "git", "git")
    recorded_starts(launch, launch.bin / "cmake", "cmake")
    shell = set_in_the_shell(launch)  # every one of them points somewhere else: none of it may be used
    shutil.rmtree(launch.prefix / "assets/llama.cpp")
    (launch.prefix / "assets/engine/yolov8n.engine").unlink()
    for step in (launch.run("build-llama.sh", package=launch.installed, **shell),
                 launch.run("build-engine.sh", package=launch.installed, stdin="BUILD\n", **shell),
                 launch.run("record-artifacts.sh", package=launch.installed, **shell)):
        assert step.returncode == 0, said(step)
        assert (f"note: set in your shell, not used by this installation (it keeps its own under {launch.prefix}): "
                + " ".join((*LOCAL, *LOCAL_REMOVED))) in step.stderr
    launch.fixture("admission.out", ROUTE_ADMITTED)
    checked = launch.check("--device-artifacts", str(launch.record), **shell)
    assert checked.returncode == 0, said(checked)
    entries = starts(launch)
    expect_the_installations_own(launch, entries)
    kinds = [(label, arguments) for label, arguments, _ in entries]
    assert any(label == "git" and arguments.startswith("clone ") for label, arguments in kinds)
    assert any(label == "cmake" and arguments.startswith("--build ") for label, arguments in kinds)
    for shape in ("-B -", f"- {launch.prefix / 'assets/engine/yolov8n.pt'} ", "-I -B -c ", "-m sentinel.cli config validate ",
                  f"-B {launch.installed / 'lib/admission_check.py'} ", "-m sentinel.cli run "):
        assert any(label == "interpreter" and (arguments + " ").startswith(shape) for label, arguments in kinds), shape
    assert not (launch.tmp / "elsewhere").exists()  # nothing was made or written where the shell pointed
    assert list(launch.home.iterdir()) == []  # and nothing under the home directory
    # the check says where things are, and what it did not use
    for variable, relative in LOCAL.items():
        assert f"local_env {variable}={launch.prefix / relative}" in checked.stdout.splitlines(), variable
    assert ("local_env_set_in_your_shell_and_not_used=" + " ".join((*LOCAL, *LOCAL_REMOVED))) in checked.stdout.splitlines()
    assert "local_env LOCAL_ENV_COUNT" not in checked.stdout
    # the check's own temporary directory was inside the installation, and is gone
    argv = json.loads((launch.tmp / "runtime-check.json").read_text())["argv"]
    assert argv[argv.index("--data-dir") + 1].startswith(str(launch.prefix / "tmp") + "/")
    assert list((launch.prefix / "tmp").iterdir()) == []
    plain = launch.check("--device-artifacts", str(launch.record))  # nothing set in the shell: nothing to name
    assert plain.returncode == 0 and "local_env_set_in_your_shell_and_not_used=none" in plain.stdout.splitlines()
    assert "note: set in your shell" not in plain.stderr


def test_a_launch_gives_the_guard_and_the_runtime_the_installations_own_locations(launch: Launch) -> None:
    recorded_starts(launch, launch.prefix / "venv/bin/python", "interpreter")
    launch.env = {**launch.env, **set_in_the_shell(launch)}
    terminal = Terminal(launch)
    to_the_start_prompt(terminal)
    terminal.type("START")
    assert terminal.wait_for(r"Features this run started with:", 30), terminal.text()
    terminal.interrupt()
    assert terminal.finish() == 0, terminal.text()
    entries = starts(launch)
    expect_the_installations_own(launch, entries)
    arguments = [arguments for _, arguments, _ in entries]
    assert any(text.startswith("benchmarks/runner/step5_guard.py --step4-headroom ") for text in arguments)  # the guard
    runs = [text for text in arguments if text.startswith("-m sentinel.cli run ")]
    assert len(runs) == 1 and "--data-dir " + str(launch.data / "observe") in runs[0]  # the one launch, in the data directory
    assert json.loads((launch.tmp / "runtime.json").read_text())["deepface_home"] == str(launch.prefix / "assets/deepface-home")
    assert not (launch.tmp / "elsewhere").exists() and launch.secrets_in_files() == []
    assert "set in your shell, not used by this installation" in terminal.text()


def test_the_face_weights_stay_where_the_settings_put_them_and_nothing_is_moved_or_fetched(launch: Launch) -> None:
    """A cache default for DEEPFACE_HOME must not move the admitted weights: the launch and the enrollment name the
    weights' own place (the installation's, or the tester's site.conf setting), whatever the shell holds."""
    recorded_starts(launch, launch.prefix / "venv/bin/python", "interpreter")
    weights = launch.prefix / "assets/deepface-home"
    before = tree(weights)
    shell = {"DEEPFACE_HOME": str(launch.tmp / "elsewhere/deepface")}
    checked = launch.check(**shell)
    assert checked.returncode == 0, said(checked)
    assert json.loads((launch.tmp / "runtime-check.json").read_text())["deepface_home"] == str(weights)
    custom = launch.tmp / "my weights home"
    shutil.copytree(weights, custom)
    configured = launch.run("configure.sh", "--set", f"DEEPFACE_HOME={custom}", package=launch.installed)
    assert configured.returncode == 0, said(configured)
    checked = launch.check(**shell)
    assert checked.returncode == 0, said(checked)
    assert json.loads((launch.tmp / "runtime-check.json").read_text())["deepface_home"] == str(custom)
    expect_the_installations_own(launch, starts(launch)[-3:], face_weights=custom)
    admission = (launch.tmp / "admission.args").read_text().splitlines()
    assert admission[admission.index("--weights-dir") + 1] == str(custom / ".deepface/weights")
    assert tree(weights) == before and tree(custom) == before  # neither copy was touched, and nothing was downloaded
    assert not (launch.tmp / "elsewhere").exists()
    assert list((launch.prefix / "cache/deepface-home").iterdir()) == []  # the cache place holds no weights


GUARDED_DRIVER = r"""import json, sys
sys.path.insert(0, sys.argv[1])
import guarded_build as front_end
front_end.STEPS["engine"].update(json.loads(sys.argv[2]))
sys.exit(front_end.main(sys.argv[3:]))
"""


def test_an_engine_built_under_the_monitor_is_noted_and_recorded_as_the_scripts_own_build(routed: Launch) -> None:
    """guarded-build's front end and the real monitor around the package's real build-engine.sh (the interpreter is the
    stand-in: no GPU, no export). The memory figures of the front end are lowered from outside, because this test
    must not depend on how much memory the test device has free; the script's own check reads the fixture."""
    launch = routed
    assert launch.build_llama().returncode == 0
    (launch.prefix / "assets/engine/yolov8n.engine").unlink()
    lowered = {"start_free_bytes": 0, "script_min_free_bytes": 0, "min_free_bytes": 0, "min_available_bytes": 64 * 1024 * 1024}
    result = subprocess.run([launch.env["REAL_SYSTEM_PYTHON"], "-I", "-B", "-c", GUARDED_DRIVER, str(launch.installed / "lib"),
                             json.dumps(lowered), "run", "--package", str(launch.installed), "--step", "engine"],
                            env=launch.env, input="BUILD\n", capture_output=True, text=True, timeout=120, cwd=launch.tmp)
    assert result.returncode == 0, said(result)
    assert "Noted what this step built" in result.stdout  # the script's own lines, relayed
    (record_folder,) = sorted((launch.data / "builds").glob("engine-*"))
    monitor = json.loads((record_folder / "engine.result.json").read_text())
    assert monitor["command"] == ["bash", str(launch.installed / "build-engine.sh")] and monitor["exit_code"] == 0
    assert monitor["thresholds"]["timeout_s"] == 3600.0  # the hour, as packaged
    weights, settings = (launch.tmp / "export.args").read_text().splitlines()
    assert weights == str(launch.prefix / "assets/engine/yolov8n.pt") and json.loads(settings) == EXPORT  # the same call
    engine = launch.prefix / "assets/engine/yolov8n.engine"
    note = json.loads((launch.prefix / "assets/provenance/detector_engine.json").read_text())
    assert (note["observed"], note["step"], note["export"]) == ("built_by_package_step", "build-engine.sh", EXPORT)
    assert note["outputs"] == {"yolov8n.engine": {"sha256": digest(engine), "bytes": engine.stat().st_size}}
    assert f"output={engine} bytes={engine.stat().st_size} sha256={digest(engine)}" in (record_folder / "RESULT.txt").read_text()
    recorded = launch.record_artifacts()
    assert recorded.returncode == 0, said(recorded)
    document = json.loads(launch.record.read_text())
    provenance = document["artifacts"]["detector_engine"]["provenance"]
    assert provenance["observed"] == "built_by_package_step" and provenance["export"] == EXPORT
    assert document["resource_validation"] == "pending"
    verified = artifacts(launch, "verify", "--package", str(launch.installed), "--record", str(launch.record))
    assert verified.returncode == 0 and "artifact_verification=passed" in verified.stdout, said(verified)
    assert "resource_validation=pending" in verified.stdout
    launch.fixture("admission.out", ROUTE_ADMITTED)
    checked = launch.check("--device-artifacts", str(launch.record))
    assert checked.returncode == 0 and "BETA_CHECK_ROUTE=device_artifacts" in checked.stdout, said(checked)
