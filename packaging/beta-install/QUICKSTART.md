# Sentinel personal beta: installing on your own Jetson

> **READ THIS FIRST.** On any Jetson but the reference device, the default `check.sh` and `start.sh` are refused:
> the default admission admits only the exact files it was measured with on the reference device, and those files
> are not in this package. Your own files can be used only through the **opt-in device-artifacts route**
> (`--device-artifacts FILE`), for **experimental validation runs**. That route checks that your files are the ones
> your own record names. It validates **nothing** about your device's memory use, speed or stability, and it has
> **never been run on another device**. Section 1 says exactly what it does and does not do. Nothing in this package
> works around admission, and you must not either (no copied private files, no changed file times, no edited record).

Private, for the maintainer's testers only; not for redistribution. It validates and admits nothing: face recognition,
identity retention and chat run as experimental validation-run features, exactly as in the build it was made from
(code `949da67`). Snapshots and external notifications are off. Testing with more than one person is deferred.

**Nothing private comes with it:** no gallery or enrollment, no camera address or credential, no detector engine, no
llama.cpp build, no device-artifacts record, no observation data. You download the pinned model files, build the
two device-specific files and record your own artifacts on your own device.

**Four application files are not the accepted build's.** The package's name carries a modification mark (`-mod…`)
for that reason, and `VERSION` names each of them with the accepted and the packaged SHA-256, and every file of the
package itself that changed since revision 5. Everything else under `app/` equals `949da67`.

- The memory guard (`app/benchmarks/runner/step5_guard.py`): when a run ends it signals only the processes that run
  started, and no longer every `llama-server` of your user. Its memory limits, readiness checks and deadlines are
  the accepted ones.
- The device-artifacts route: `app/src/sentinel/device_artifacts.py` (a new file) and its wiring in
  `app/src/sentinel/demo_runtime.py` and `app/src/sentinel/cli.py`. Without `--device-artifacts` the runtime decides
  exactly as the accepted build does.

**What has run where.** Revision 5 of this package ran for real on the reference device, without a camera: the
installation, the Python environment, the downloads, both builds, the record of the built files, its verification
and the route's admission decision (section 14 has the details). This revision differs from revision 5 in the files
`VERSION` lists; **those changes were verified offline only**, with temporary directories and stand-ins. No guarded
run with a camera or a model has used the changed guard, and no launch has ever been made from this package.

The steps, in order: prerequisites (2) → install (3) → environment and downloads (4) → the two builds (5) → your
enrollment (6) → settings (7) → record and check (8) → launch (9). Sections 10 to 14: limits, uninstall, which
steps touch a camera or face data, the bug report, and what was verified.

## 1. Your own device: what the default refuses, and the opt-in route

### What the default admission refuses

The accepted code compares three kinds of file with records made on the reference device. That path is unchanged.

| What | How the default identifies it | What you will see with your own file |
|---|---|---|
| Detector engine (`yolov8n.engine`) | SHA-256 of the reference device's engine | `detector_engine=refused` (`engine_hash_mismatch`): a TensorRT engine built on your device is another file |
| llama.cpp build (`llama-server` and its libraries) | SHA-256 of the reference device's own build; one large library by size and build time | `scene_admission=refused`: a build made on your device, or in another directory, has other bytes |
| Scene model, its projector, the face weights | name, size and **modification time** of the reference device's copies | refused even though `fetch-assets.sh` proved your copies equal the pinned SHA-256: a download has another time |

### The opt-in device-artifacts route (experimental validation runs only)

`record-artifacts.sh` writes a record of your own files. `check.sh --device-artifacts FILE` and
`start.sh --device-artifacts FILE` then identify your files by that record instead of by the reference device's.
Nothing uses the route unless you type the option.

Checked every time, before anything starts:

- **The record itself**: its format, and that it claims nothing but `pending` for resource validation.
- **How each built file was made.** The llama.cpp build must be `build-llama.sh`'s own, at the pinned commit with the
  pinned options; the engine must be `build-engine.sh`'s own, from the pinned `yolov8n.pt` with the pinned settings
  and tool versions. Each script notes what it built when it finishes. A file made any other way is "inspected
  only": its checksum is known, how it was built is not, no record is written for it and nothing launches with it.
  Running a build script through `guarded-build.sh` (section 5) is running that script: the note is the same.
- **The downloaded files**: the model, its projector and the two face weight files must equal the pinned SHA-256.
- **Every file, whole**, against the record's SHA-256, size and modification time. A missing, changed, rebuilt or
  re-copied file is refused: run its step again, move the old record away and record again.
- **The platform** against the record: the device, L4T, CUDA, TensorRT, cuDNN, the kernel, the Python packages.
- **Everything the default requires besides file identity**, unchanged: your configuration's thresholds, stream
  profile and adapters; face recognition, retention and chat only as validation-run features; no snapshots and no
  notifications. The runtime itself refuses a record outside a `--face-validation` run.

What the route does **not** do:

- **It does not validate your device.** A record identifies files. No run on your device has been measured or
  judged against anything. The accepted resource profile your configuration names was measured on the reference
  device with the reference device's files: here it says which settings are run, not how your device behaves under
  them. Device resource validation is **pending**, and every check, launch and run record says so.
- **The launch gates and the guard's limits are limits, not evidence** (section 10). Their numbers come from the
  reference device. Passing them does not show that your device has the same headroom. A guard stop on your device
  is a result to report, not something to work around.
- **It does not verify your platform** (section 2), and it does not change the default admission.
- **It has never been run on another device.** Expect to be the first, and report what you see (section 13).

## 2. Platform and prerequisites

The only combination this build has run on (the reference device):

- NVIDIA Jetson Orin Nano developer kit, 8 GB
- L4T R36.4.7 (JetPack 6.2.1 packages), CUDA 12.6, TensorRT 10.3.0, cuDNN 9
- Python 3.10
- an Ubuntu **24.04** userland (the reference device was upgraded; NVIDIA ships JetPack 6 on Ubuntu 22.04)

`preflight.sh` passes only on that combination. On stock JetPack 6.1 to 6.2.1 (L4T R36.4.x, Ubuntu 22.04), which is
what you most likely have, it says **NOT VERIFIED** and `install.sh` goes on only with
`--accept-unverified-platform`: it may work, nobody has tried. Anything else fails with what to do.

You install the system prerequisites yourself; the package never uses sudo for that and installs no system package:

| Needed | On stock JetPack 6 |
|---|---|
| CUDA 12.6, TensorRT 10.3, cuDNN 9, the L4T CUDA driver library | `sudo apt install nvidia-jetpack` |
| TensorRT's Python binding for Python 3.10 | `sudo apt install python3-libnvinfer` (part of JetPack), or the extraction route below |
| Python 3.10 with venv | `sudo apt install python3.10 python3.10-venv` |
| cuDSS (`libcudss.so.0`), which the pinned torch build loads | `sudo apt install libcudss0-cuda-12` from NVIDIA's CUDA apt repository (Ultralytics' NVIDIA Jetson guide gives the repository steps) |
| OpenBLAS | `sudo apt install libopenblas0` |
| `python3-cryptography` for `/usr/bin/python3` (seals the enrollment) | `sudo apt install python3-cryptography` |
| git, cmake, a C++ compiler | `sudo apt install git cmake build-essential` |

### The TensorRT Python binding

`setup-env.sh` links NVIDIA's TensorRT Python binding (the `tensorrt` module for Python 3.10, TensorRT 10.3.0) into
the environment. The package does not contain, download or build one, and none is borrowed from anybody's device.
The preflight looks for it in `/usr/lib/python3.10/dist-packages`, or in the directory you name with
`SENTINEL_BETA_TENSORRT_PYTHON` (the directory that holds the `tensorrt` folder). There are two ways to supply it.

**Installed** (stock JetPack 6: Ubuntu 22.04, whose system Python is 3.10): `sudo apt install python3-libnvinfer`.
Expected to be the ordinary case. **Not verified**: it has never been tried with this package.

**Extracted, not installed** (the route used on the reference device, whose system Python is 3.12, so that the
package cannot be installed there). It downloads NVIDIA's own package from the apt repository your JetPack already
uses and unpacks it into a directory of your own. No sudo, nothing is installed:

```bash
mkdir -p ~/sentinel-prereq && cd ~/sentinel-prereq
apt-get download python3-libnvinfer=10.3.0.30-1+cuda12.5
sha256sum python3-libnvinfer_10.3.0.30-1+cuda12.5_arm64.deb
dpkg-deb -x python3-libnvinfer_10.3.0.30-1+cuda12.5_arm64.deb python3-libnvinfer
export SENTINEL_BETA_TENSORRT_PYTHON=~/sentinel-prereq/python3-libnvinfer/usr/lib/python3.10/dist-packages
```

Then run `preflight.sh` and `install.sh` (section 3) from that same shell. `install.sh` records the directory and
uses it again when you install again; keep the directory, the environment links to it.

| What was used on the reference device | |
|---|---|
| Source | `https://repo.download.nvidia.com/jetson/common`, suite `r36.4`, component `main` |
| Package | `python3-libnvinfer` 10.3.0.30-1+cuda12.5, arm64 (`pool/main/t/tensorrt/python3-libnvinfer_10.3.0.30-1+cuda12.5_arm64.deb`) |
| Size and SHA-256 | 649,382 B, `d2b7d6e9755e7dc470c1bdb2a5564c0916989ae6a4b4063a1cbfc2510b644a6d` |

Compare the `sha256sum` line with that value. The package is NVIDIA's build for Python 3.10 and for the TensorRT
libraries of exactly that version (`libnvinfer10` 10.3.0.30). **Extracting it does not make a platform supported:**
it worked on one device, where those system libraries are installed. If your `libnvinfer10` is another 10.3.0
build, the matching `python3-libnvinfer` version is the one to try; that was not verified.

## 3. Install

Check the archive against the checksum file that came with it, extract it anywhere, then, as your own user (never
with sudo):

```bash
sha256sum -c sentinel-personal-beta-949da67-mod79c18d4-r6.tar.gz.sha256
tar -xzf sentinel-personal-beta-949da67-mod79c18d4-r6.tar.gz && cd sentinel-personal-beta-949da67-mod79c18d4-r6
bash preflight.sh        # reads facts only; changes nothing
bash install.sh          # defaults: ~/sentinel-beta (the installation) and ~/sentinel-beta-data (your data)
```

`bash install.sh --prefix DIR --data-dir DIR` chooses other directories. Installation copies the package and makes
the data directory; it uses no network, opens no camera, loads no model, installs no service and starts nothing, now
or at boot. Installing again refreshes the package and keeps your configuration, enrollment and run records. If an
installation is interrupted, run `install.sh` again: it completes it. Every script checks the installed package
against its own checksums before it does anything; never edit its files.

### Where the installation keeps its files

| Place | What | Made by |
|---|---|---|
| `<prefix>/package/` | this package | `install.sh` |
| `<prefix>/venv/` | the Python environment, with the packages' bytecode | `setup-env.sh` |
| `<prefix>/assets/` | the model files, the llama.cpp build, the detector engine, the build notes | `fetch-assets.sh`, the builds |
| `<prefix>/cache/` | the libraries' own caches and settings: pip's downloads, Ultralytics' settings, Matplotlib's font list, Keras' configuration, CUDA, torch and Hugging Face caches | every script that needs one |
| `<prefix>/tmp/` | temporary files of the scripts and of what they start | every script that needs one |
| `<data dir>/config/` | your settings and your device-artifacts record | `install.sh`, you |
| `<data dir>/identity/` | your enrollment | you (section 6) |
| `<data dir>/observe/` | one folder of records per run | `start.sh` |
| `<data dir>/builds/` | one folder of records per guarded build | `guarded-build.sh` |

The scripts give these variables a directory inside the installation, for themselves and for everything they start,
before anything is imported: `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `HF_HOME`,
`TORCH_HOME`, `YOLO_CONFIG_DIR`, `DEEPFACE_HOME`, `CUDA_CACHE_PATH`, `KERAS_HOME`, `MPLCONFIGDIR` and `TMPDIR`.
`HOME` and your shell's own files are not changed, and nothing is written under your home directory except the two
directories you chose.

- **A value you have set in your shell is not used.** The script replaces it for its own run and names it once
  (`note: set in your shell, not used by this installation …`). A few variables that would move one of these places
  again are removed for the run (`PYTHONPYCACHEPREFIX`, `TEMP`, `TMP`, `HF_HUB_CACHE` and the like).
  `bash check.sh` prints the places in effect (`local_env …` lines), and `bash configure.sh` shows the two folders.
- **The face weights are not a cache.** They stay where the asset manifest puts them (`<prefix>/assets/deepface-home`,
  or your `DEEPFACE_HOME` setting in `site.conf`): the enrollment and the launch name that place themselves, and it
  is the place your record describes. The `DEEPFACE_HOME` above only keeps an incidental import from making folders
  under your home directory.
- **`cache/` and `tmp/` hold nothing you need to keep.** In the rehearsal on the reference device pip's downloads
  came to about 0.9 GB and the other libraries' entries to less than 1 MB. You may empty both while nothing from the
  installation is running; they are made again. A link or a foreign folder in their place is refused, not followed.
- **Settings under `~/.config` are not read by these scripts** (a per-user pip or git configuration there, for
  example): the programs they start look in the installation's own, empty settings folder instead. Files directly
  under your home directory and environment variables such as a proxy's are untouched.
- **Not covered:** programs you start yourself outside these scripts, and what the camera or your browser keep.

This part is new in this revision and was verified offline only (section 14).

## 4. The Python environment and the pinned downloads

From the installed package (`cd ~/sentinel-beta/package`), in this order:

```bash
bash setup-env.sh        # a new Python 3.10 environment with the pinned packages (downloads about 1.5 GB)
bash fetch-assets.sh     # the model files (about 1.35 GB), each checked against its SHA-256
```

Every Python package is pinned to one version and to its files' SHA-256 (`manifest/requirements-pypi.txt`,
`manifest/requirements-jetson.txt`); every model file has its upstream source, revision, licence and SHA-256 in
`manifest/assets.json`. A file that does not match is never used. `setup-env.sh` ends by importing the GPU and face
libraries once; no model is loaded. On the reference device, with its network, the two took about eight and seven
minutes; yours will differ.

## 5. Building on this device

Two files are built where they run: `llama-server` (llama.cpp at the pinned commit) and the detector's TensorRT
engine. Remember section 1: the default admission refuses both; only the opt-in route can use them, and only when
the package's own scripts built them. Each script notes what it built when it finishes (under
`assets/provenance/`); if a build is interrupted, run it again rather than finishing it by hand.

### What was observed, and what was not measured

All of this is from **one device**, the reference device, with revision 5. These are whole-device readings: other
programs and the page cache move them too, so they are not a build's own demand.

| Build | Start | What happened |
|---|---|---|
| llama.cpp, two jobs | MemAvailable 5.80 GB, a desktop running | completed in 53 minutes; MemAvailable never below 4.66 GB |
| Engine, 2026-11-13 | MemFree 0.08 to 0.84 GB (4.6 to 5.5 GB "available") | failed in its first GPU allocations |
| Engine, 2026-11-14 | MemFree 5.12 GB, a desktop and other programs running | the monitor ended it after 136 s, when MemFree crossed below 1 GiB; no allocation error |
| Engine, 2026-11-15 | MemFree 6.95 GB, no desktop session, after a page-cache release | **completed in 187 s**; lowest MemFree 2.62 GB, lowest MemAvailable 3.51 GB, no swap used |

Three different things, which must not be read as one:

1. **The monitor's stops** say when a build is ended, for the device's safety: the engine step below 1 GiB of
   MemFree, below 2 GiB of MemAvailable, or after one hour; the llama.cpp step below 2 GiB of MemAvailable or after
   three hours. They are not what a build needs.
2. **The 5 GiB preparation figure** (5,368,709,120 B of MemFree) is what `guarded-build.sh engine` asks for before
   it starts. It was **chosen**, for the rehearsal's last attempt. It is not a measurement.
   `--min-start-free-bytes N` chooses another figure.
3. **The measurements** are the table above: one success, started with 6.95 GB free, during which MemFree fell by
   4.33 GB. The same fall from exactly 5 GiB would have ended just below the 1 GiB stop. That is arithmetic on one
   run of one device, not a requirement.

**The least memory the export needs has not been measured**, on any device. Not every 8 GB Jetson needs the same
preparation, and no completion time is promised: the export has taken 187 s once.

### Prepare memory yourself

On a Jetson the GPU's memory is taken out of free memory, and memory that is only "available" (page cache) does not
count for it; after the downloads and the llama.cpp build the cache is usually full. **Preparing memory is your own
step, done by hand.** Nothing in this package stops a desktop, closes a program, releases memory or uses sudo for
you during a build.

1. Save your work. Close what you do not need on the Jetson (an editor's remote session, a browser).
2. Work from a plain SSH shell on another machine, not from a terminal inside a desktop you are about to close.
3. Look first: `bash guarded-build.sh --check engine` starts nothing and prints MemFree, the largest processes and
   the graphical processes that are running.
4. **If you decide to close a graphical desktop for the build**, how depends on how yours is started, and closing it
   ends that session: unsaved work in it is lost.
   - A session run by a display manager: `sudo systemctl stop display-manager` (nothing is disabled; it comes back
     with `sudo systemctl start display-manager`, and by itself at the next boot).
   - A remote desktop (VNC or similar) is often started another way, and **stopping the display manager does not end
     it**. End it with the tool that started it (for a TigerVNC server started as `vncserver :N`, that is
     `vncserver -kill :N`), and start it afterwards the way you normally do.
   - Run the check of step 3 again: `graphical_processes_running` shows what is still there.
5. Release the page cache once, as your own command: `sync; sudo sysctl -w vm.drop_caches=1`
6. Read nothing large afterwards; start the build straight away.

A fresh boot is another way to free memory, but whatever starts at boot (a desktop, a remote desktop) is then
running again.

### Run the builds under the memory monitor

```bash
bash guarded-build.sh --check llama --jobs 2    # the start conditions only; starts nothing
bash guarded-build.sh llama --jobs 2            # asks BUILD
bash guarded-build.sh --check engine
bash guarded-build.sh engine                    # asks BUILD; uses the GPU
```

`guarded-build.sh` starts the build script itself, unchanged, and a monitor beside it that reads the device's
memory five times a second. It refuses to start (status 10, nothing started) while a model server, a Sentinel
runtime or another build of yours is running, while the engine is already there, or while memory is below the
figures above; nothing is stopped or freed for you. When the monitor stops a step it ends every process that step
started, and only those. Each run keeps its records in `~/sentinel-beta-data/builds/<step>-<UTC time>/`:
`RESULT.txt`, the step's log, the memory samples (`.mem.csv`) and the monitor's result.

| Exit status | Meaning |
|---|---|
| 0 | built; `RESULT.txt` reads `OUTCOME=built` |
| 10 | refused before anything started (the line says which condition) |
| 20 | the monitor stopped the step at a memory stop or at its time limit |
| 21 | the build script failed by itself (its `STOP:` line is repeated in the result) |
| 22 | stopped by you |
| 23 / 24 / 5 | the monitor refused at its own start / the monitor failed / something of the step was still running |
| 11 | another guarded build of this installation is running; nothing was started |

The build scripts can also be run by themselves (`bash build-llama.sh --jobs 2`, `bash build-engine.sh`): the same
recipe, without the monitor. `build-llama.sh` builds only in a build tree it started itself, refuses compiler or
linker flags set in your shell (`CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, `CUDAFLAGS`, `LDFLAGS`), and links the binaries
again each time it runs, so run it once and record afterwards. `build-engine.sh` refuses to start when MemFree is
below 1.5 GB: that is a minimum for trying, not a figure known to be enough.

### Inspect, stop, recover

```bash
bash guarded-build.sh --status     # is one running; the last result
bash guarded-build.sh --stop       # from a second SSH shell: ends the running step cleanly
```

Ctrl-C **once** in the build's own window is the same clean stop. Either way the step's first process is asked to
end and the monitor ends the rest.

- **After a success:** bring back what you closed (your own commands, as in step 4), then go on with section 6.
- **After a refusal (10):** nothing was started. Fix what the line names and run the same command again.
- **After a monitor stop (20):** nothing of the step is left running, and no engine was written (an intermediate
  `yolov8n.onnx` may remain; it is made anew). Free more memory and run it again. The package never tries again by
  itself, and its stops are not to be worked around: a stop on your device is a result to report (section 13).
- **After a failure (21):** read the `STOP:` line and the log in the run's folder. `build-llama.sh` continues the
  build tree it started. If `assets/engine/yolov8n.engine` exists although the run did not end with
  `OUTCOME=built`, move that file away yourself before building again: an engine without its build note is never
  recorded or launched.
- **After a lost SSH connection:** a started step goes on without the terminal. Log in again and run `--status`.
  `guarded build running: yes`: wait for it, or `--stop`. Bring a desktop back only when it has ended: starting one
  takes memory from a running build.
- **If you cannot log in at all:** nothing was disabled, so a restart of the device brings back whatever starts at
  boot. A restart during a build ends that build; run the step again afterwards.

## 6. Your own enrollment

**This step collects face data: your own.** No camera is used.

Recognition needs exactly one enrolled person: you. Nobody's enrollment comes with this package, and none is made
for you. Enrol **only yourself**: the build stores a consent flag, a date and one fixed sentence ("enrollment on this
device for the bounded Sentinel demo; revocable at any time"), and has no consent form or record for another person.
No name is stored anywhere.

1. Make one folder for your photos (the installer made `identity/` and `inbox/`, private to you):
   `mkdir -m 700 ~/sentinel-beta-data/identity/inbox/me`
2. **Copy** 3 to 8 photos of yourself into it: one face per photo, facing the camera, in your usual light, JPEG or
   PNG. From another machine: `scp -p photos/* <user>@<jetson>:sentinel-beta-data/identity/inbox/me/`
3. Try it first; this writes and deletes nothing and reports each photo:
   `bash identity.sh enroll me --consent-confirmed --consent-date YYYY-MM-DD --dry-run`
4. Enrol. `--consent-confirmed` is your own statement that you consent; the date is stored with the enrollment:
   `bash identity.sh enroll me --consent-confirmed --consent-date YYYY-MM-DD`
   It asks for a new gallery passphrase twice, hidden (at least 12 characters). The passphrase is never stored: you
   type it at every launch, and if you lose it the enrollment must be deleted and made again. The copies in the
   inbox folder are deleted once the enrollment is written.
5. `bash identity.sh list` shows one enrolled identity (an opaque id and the consent date).

At most 8 photos; at least 2 must pass (one clear face each, and they must agree with each other). The face model
runs on the CPU for a few seconds per photo. To withdraw: `bash identity.sh list`, then
`bash identity.sh revoke idn-…`; to delete everything: `bash identity.sh purge --yes`. The enrollment records the
versions of the face libraries: if the Python environment is ever rebuilt with other versions, enrol again.

## 7. Settings

`bash configure.sh` shows every path in use and whether it is there. Your settings are in
`~/sentinel-beta-data/config/` and are never overwritten: `live.yaml` (you may change the camera's id; `check.sh`
refuses a file whose thresholds, stream profile, notifications, labels or adapters differ from the accepted ones) and
`site.conf` (`bash configure.sh --set ENGINE=/abs/path` and the like, if an asset is somewhere else). Neither holds
the camera address or the passphrase, and nothing in them is executed.

## 8. Record your files, then check (no credentials, no sudo, no camera, no model)

After the builds of section 5:

```bash
bash record-artifacts.sh
```

It reads each file whole (about 1.6 GB, a minute or two), changes none, and writes
`~/sentinel-beta-data/config/device-artifacts.json`: each file's name, size, modification time and SHA-256, what this
package's steps observed about it, and your platform's versions. It prints the record's path. A record that is
already there is never replaced: after a rebuild, move the old one away and record again. On a platform the
preflight calls NOT VERIFIED it records only with `--accept-unverified-platform`, and the record says so. The record
does not need your enrollment; the check does.

```bash
bash check.sh --device-artifacts ~/sentinel-beta-data/config/device-artifacts.json
```

It ends with `BETA_CHECK=passed`, or with one `STOP:` line that says what is missing and what to do. Without an
enrollment it stops at `there is no enrollment`: that is a refusal, not a pass, whatever the lines above it say.
With the route it also prints `artifact_verification=passed`, `resource_validation=pending` and a
`BETA_CHECK_ROUTE=` line: your files equal your record, and nothing about your device's resources was validated.
Plain `bash check.sh` is the default admission: on a device other than the reference device it stops at "an
admission decision is not as this package expects" (section 1).

## 9. Launch, dashboard, stop

**This is the only step that opens a camera, and only after you type START.** It runs under the memory guard.

On your workstation, first (keep it open; on a Mac put `caffeinate -i` in front):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18090:127.0.0.1:18090 <user>@<jetson>
```

On the Jetson, in a plain SSH shell (not a desktop terminal, not a VS Code terminal):

```bash
bash ~/sentinel-beta/package/start.sh --device-artifacts ~/sentinel-beta-data/config/device-artifacts.json
```

(Plain `start.sh`, without the option, is the default admission and launches on the reference device only.) It first
checks every file against your record again, then asks, in order: **CLOSE** (only if Claude Code, Codex, VS Code server, update-manager or a VNC desktop run; each is
listed first), **PREPARE** (only if needed: stops the desktop until the run ends and drops the page cache; sudo asks
your password), the camera's **main-stream URL** (hidden, never saved), the **gallery passphrase** (hidden, never
saved), and last **START**. **Only typing START opens the camera.** Pressing Enter at any prompt stops without
opening it. Nothing is closed, stopped or released unless you type the word that asks for it. The models load first
(1-2 minutes); a run lasts at most 60 minutes after that and stops by itself. Start nothing else on the Jetson
until `OBS_exit`.

- **Dashboard (your browser):** http://127.0.0.1:18090/dashboard on the workstation, through the tunnel above. It
  shows the live view and the experimental face labels; the page is served on the Jetson's 127.0.0.1 only.
- **Feature-control key** (to pause or resume face, scene, chat or the live view in Settings → Features), from
  another SSH shell once the run is ready: `cat "$(ls -d ~/sentinel-beta-data/observe/obs-*/ | sort | tail -n 1)db/control-key"`
- **Clean stop:** Ctrl-C **once** in the Jetson window, then wait for `OBS_exit` (up to about 2 minutes; press no
  other key). Then close the tunnel with Ctrl-C.
- **If the window was lost:** `bash ~/sentinel-beta/package/stop.sh`. It asks only that observation's own guard or
  runtime to stop, and always ends with what is recorded about the desktop.
- **If the desktop did not come back:** `sudo systemctl start display-manager`.

A Ctrl-C or the 60-minute stop that ends cleanly reads `clean=yes` and `OBS_exit=0`. A guard stop reads
`status=guard_stop`, `ended_by=the guard`, `OBS_exit=1`: the memory guard ended the run safely. With the route, the
run's folder also keeps the record it used (`device-artifacts.json`) and `launch.txt` reads
`resource_validation=pending`: a run that ends cleanly is an observation, not a validation of your device.

## 10. Operating limits (the accepted build's, unchanged)

- **Memory.** Launch needs MemFree ≥ 5,680,693,248 B, MemAvailable ≥ 4,000,000,000 B and memory in use below
  1,294,012,416 B (numbers from one run on the reference device). The guard stops the run at memory pressure
  ≥ 4,800,000,000 B, MemFree < 1 GiB, MemAvailable < 2 GiB, any swap activity or a sampling gap over 0.5 s.
  On your device these are the same limits, kept as limits: they were derived from the reference device's
  measurements, and nothing shows yet that a run on your device stays inside them.
- **Time.** At most 60 minutes per run after readiness; readiness within 300 s.
- **Scope.** One camera (its main stream), one enrolled person, one person in view. Recognition is context only.
- **Chat.** Advisory answers about one frame; one question at a time, at most 300 characters. Monitoring comes first.
- **Other programs.** When a run ends, the memory guard stops a `llama-server` only if that run started it (it
  descends from the guard; checked by process number and start time). Another `llama-server` of your user is counted
  (`llama_server_unresolved` in `guard/result.json`), named in a NOTE by the launcher, left running and never
  signalled; the run then reads `clean=no`. The launcher still refuses to start while one runs, and its memory use
  can trigger the guard's stop: do not run another one during an observation.
- **Privacy.** The page is on the Jetson's 127.0.0.1 only. Each run keeps records (numbers per face result, no
  pictures) in `~/sentinel-beta-data/observe/obs-<UTC time>/`; `guard/run.err` holds the runtime's own messages,
  which can name the camera's host and port. Delete a folder when you no longer need it; never share footage, the
  enrollment or credentials.

## 11. Uninstall and recovery

```bash
bash ~/sentinel-beta/package/uninstall.sh            # asks UNINSTALL; --keep-assets keeps the downloaded models and builds
```

| | Removed | Kept |
|---|---|---|
| `uninstall.sh` | the package, the Python environment, the assets, `cache/`, `tmp/`, the installer's two record files; then the prefix itself if nothing else is in it | your data directory, and anything in the prefix that the installer did not make |
| `uninstall.sh --keep-assets` | the same, without the assets | the assets (the downloads and both builds), and a small `uninstall.sh` in the prefix to remove them later |

It removes only what the installer's own record names, and refuses while anything from the installation runs.
**Your data directory is never removed**: configuration, the device-artifacts record, enrollment, run records and
build records stay until you delete them yourself (`bash identity.sh purge --yes` deletes the enrollment first). If
you kept the assets, or an uninstall was interrupted, finish with `bash ~/sentinel-beta/uninstall.sh`. No sudo;
nothing outside the two directories is touched. The directory of an extracted TensorRT binding (section 2) and the
extracted archive are yours to delete.

Recovery, in one place: an interrupted `install.sh` or `uninstall.sh` is completed by running it again; an
interrupted download or environment setup by running that script again; builds as in section 5; a lost launch
window with `stop.sh` (section 9).

## 12. Which steps touch a camera or face data

| Step | Camera | Face data |
|---|---|---|
| `preflight.sh`, `install.sh`, `setup-env.sh`, `fetch-assets.sh`, `guarded-build.sh`, the build scripts, `record-artifacts.sh`, `configure.sh`, `uninstall.sh` | no | none (the two face *model* files are downloaded and hashed; they describe nobody) |
| `identity.sh enroll … --dry-run` | no | reads your photos; stores nothing |
| `identity.sh enroll …` | no | reads your photos, writes the sealed enrollment, deletes the copies in the inbox |
| `identity.sh list`, `revoke`, `purge` | no | reads or changes the sealed enrollment |
| `check.sh` | no | none: it only looks whether an enrollment is there; nothing is opened |
| `start.sh`, before START | no | none |
| `start.sh`, after you type START | **yes: the camera's main stream** | **yes**: faces in view are compared with your enrollment; per-face numbers are recorded, no pictures |
| The dashboard, during a run | shows the live view, to your workstation through your SSH tunnel only | shows the experimental labels |
| `stop.sh` | no (it ends a run) | none |

Nothing is enrolled, opened or sent anywhere automatically. Snapshots and external notifications stay off.

## 13. Bug report template

Keep it free of the camera address, passwords, the passphrase, the control key, names and pictures.

```text
When (UTC):
Package: (first line of ~/sentinel-beta/package/VERSION)
Device: (the first five lines of `bash preflight.sh`)
Step: preflight / install / setup-env / fetch-assets / guarded-build llama / guarded-build engine / enrollment / record-artifacts / check / launch (which prompt) / dashboard / stop / uninstall
What I did:
What I expected:
What happened (the exact STOP line, or the last lines printed):
For a guarded build: RESULT.txt of that run (~/sentinel-beta-data/builds/<step>-<UTC time>/), and how you prepared memory:
For a launch: the "Observation ended", "ended_by", "runtime_after_guard", "enrollment", "route" and "OBS_exit" lines:
For the device-artifacts route: everything `check.sh --device-artifacts FILE` printed (checksums and versions; no paths outside your home, no secrets):
Anything else running on the Jetson at the time:
```

## 14. What was and was not verified

**With revision 5, for real, on the reference device, without a camera** (the installation rehearsal of 2026-11-13
to 2026-11-15): the preflight with an extracted TensorRT binding; installation, reinstallation and uninstall in
separate directories; the Python environment built from the pinned files; every model file downloaded from upstream
and equal to its SHA-256; the llama.cpp build (two jobs, 53 minutes); the engine export, completed once (section 5),
under the monitor that is now `lib/guarded_run.py`; the device-artifacts record made from those builds, its
verification, and the route's admission decision for that record. In that rehearsal the caches and temporary files
were kept inside its own directory by a file of the rehearsal's, not by the package.

**This revision's own changes, offline only** (temporary directories, synthetic files, stand-ins; the package's
tests): the installation's own cache, settings and temporary locations in every script; `guarded-build.sh` and its
front end, with the monitor unchanged; uninstall removing `cache/` and `tmp/`; this guide. **Nothing of this revision
has run a download, a build, an export, a model or a camera on any device.**

**Not verified:** anything on another device; stock JetPack 6 on Ubuntu 22.04 (JetPack's `python3-libnvinfer`
installed with apt, cuDSS from NVIDIA's repository); the complete `check.sh` on real builds (the rehearsal has no
enrollment, so it stopped there as it should); an enrollment made with this package; any launch from this package,
with or without the route; the changed memory guard in a guarded run with a camera or a model (its ownership of
processes is covered by synthetic tests and by tests with small stand-in processes only); the least memory and the
time the builds need; **any device's resources but the reference device's**: no resource profile, memory level,
frame rate or stability figure exists for another Jetson. Tests with stand-ins are not an installation on a clean
Jetson.
