# Sentinel personal beta: installing on your own Jetson

> **THIS PACKAGE CANNOT LAUNCH ON ANOTHER JETSON YET.**
> You can install it, build everything and enrol yourself, but `check.sh` will stop at the admission step and
> `start.sh` will not open the camera. The accepted build admits only the exact files it was measured with on the
> reference device, and those files are not in this package. Section 1 says exactly what is refused and why. Nothing
> in this package works around that, and you must not either (no copied private files, no changed file times).

Private, for the maintainer's testers only; not for redistribution. It validates and admits nothing: face recognition,
identity retention and chat run as experimental validation-run features, exactly as in the build it was made from
(code `949da67`). Snapshots and external notifications are off. Testing with more than one person is deferred.

## 1. What still blocks another device

The accepted code compares three kinds of file with records made on the reference device. No option changes this.

| What | How the accepted build identifies it | What you will see with your own file |
|---|---|---|
| Detector engine (`yolov8n.engine`) | SHA-256 of the reference device's engine | `detector_engine=refused` (`engine_hash_mismatch`): a TensorRT engine built on your device is another file. Without it there is no person detection |
| llama.cpp build (`llama-server` and its libraries) | SHA-256 of the reference device's own build; one large library by size and build time | `scene_admission=refused`: a build made on your device, or in another directory, has other bytes. No scene analysis and no chat |
| Scene model, its projector, the face weights | name, size and **modification time** of the reference device's copies | refused even though `fetch-assets.sh` proved your copies equal the pinned SHA-256: a download has another time |

What it takes to lift this is the maintainer's decision and a separate change to the admission rules (for example
admitting a detector engine and a llama.cpp build made on the tester's own device by the pinned procedure, and
identifying the model files by checksum). Until then: install, build, enrol, run `check.sh`, and send the maintainer
what it prints.

Also not verified anywhere but on the reference device: see section 2.

## 2. Platform

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
| TensorRT's Python binding for Python 3.10 | `sudo apt install python3-libnvinfer` (part of JetPack) |
| Python 3.10 with venv | `sudo apt install python3.10 python3.10-venv` |
| cuDSS (`libcudss.so.0`), which the pinned torch build loads | `sudo apt install libcudss0-cuda-12` from NVIDIA's CUDA apt repository (Ultralytics' NVIDIA Jetson guide gives the repository steps) |
| OpenBLAS | `sudo apt install libopenblas0` |
| `python3-cryptography` for `/usr/bin/python3` (seals the enrollment) | `sudo apt install python3-cryptography` |
| git, cmake, a C++ compiler | `sudo apt install git cmake build-essential` |

## 3. Install

Extract the archive anywhere, then, as your own user (never with sudo):

```bash
tar -xzf sentinel-personal-beta-949da67-r1.tar.gz && cd sentinel-personal-beta-949da67-r1
bash preflight.sh        # reads facts only; changes nothing
bash install.sh          # defaults: ~/sentinel-beta (the installation) and ~/sentinel-beta-data (your data)
```

`bash install.sh --prefix DIR --data-dir DIR` chooses other directories. Installation copies the package and makes
the data directory; it uses no network, opens no camera, loads no model, installs no service and starts nothing, now
or at boot. Installing again refreshes the package and keeps your configuration, enrollment and run records. If an
installation is interrupted, run `install.sh` again: it completes it.

Then, from the installed package (`cd ~/sentinel-beta/package`), in this order:

```bash
bash setup-env.sh        # a new Python 3.10 environment with the pinned packages (downloads about 1.5 GB)
bash fetch-assets.sh     # the model files (about 1.35 GB), each checked against its SHA-256
bash build-llama.sh      # llama.cpp at the pinned commit, built on this device (about 45 minutes)
bash build-engine.sh     # the detector's TensorRT engine, built on this device (asks BUILD; uses the GPU)
```

Every Python package is pinned to one version and to its files' SHA-256 (`manifest/requirements-pypi.txt`,
`manifest/requirements-jetson.txt`); every model file has its upstream source, revision, licence and SHA-256 in
`manifest/assets.json`. A file that does not match is never used. Remember section 1: the last two builds are
documented and reproducible, and the accepted build still refuses their results.

## 4. Your own enrollment

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
runs on the CPU for a few seconds per photo; no camera is used. To withdraw: `bash identity.sh list`, then
`bash identity.sh revoke idn-…`; to delete everything: `bash identity.sh purge --yes`. The enrollment records the
versions of the face libraries: if the Python environment is ever rebuilt with other versions, enrol again.

## 5. Settings

`bash configure.sh` shows every path in use and whether it is there. Your settings are in
`~/sentinel-beta-data/config/` and are never overwritten: `live.yaml` (you may change the camera's id; `check.sh`
refuses a file whose thresholds, stream profile, notifications, labels or adapters differ from the accepted ones) and
`site.conf` (`bash configure.sh --set ENGINE=/abs/path` and the like, if an asset is somewhere else). Neither holds
the camera address or the passphrase, and nothing in them is executed.

## 6. Check (no credentials, no sudo, no camera, no model)

```bash
bash check.sh
```

It ends with `BETA_CHECK=passed`, or with one `STOP:` line that says what is missing and what to do. On a device
other than the reference device it stops at "an admission decision is not as this package expects" (section 1).

## 7. Launch, dashboard, stop (for when section 1 no longer applies)

On your workstation, first (keep it open; on a Mac put `caffeinate -i` in front):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18090:127.0.0.1:18090 <user>@<jetson>
```

On the Jetson, in a plain SSH shell (not a desktop terminal, not a VS Code terminal):

```bash
bash ~/sentinel-beta/package/start.sh
```

It asks, in order: **CLOSE** (only if Claude Code, Codex, VS Code server, update-manager or a VNC desktop run; each is
listed first), **PREPARE** (only if needed: stops the desktop until the run ends and drops the page cache; sudo asks
your password), the camera's **main-stream URL** (hidden, never saved), the **gallery passphrase** (hidden, never
saved), and last **START**. **Only typing START opens the camera.** Pressing Enter at any prompt stops without
opening it. The models load first (1-2 minutes); a run lasts at most 60 minutes after that and stops by itself.
Start nothing else on the Jetson until `OBS_exit`.

- **Dashboard:** http://127.0.0.1:18090/dashboard on the workstation.
- **Feature-control key** (to pause or resume face, scene, chat or the live view in Settings → Features), from
  another SSH shell once the run is ready: `cat "$(ls -d ~/sentinel-beta-data/observe/obs-*/ | sort | tail -n 1)db/control-key"`
- **Clean stop:** Ctrl-C **once** in the Jetson window, then wait for `OBS_exit` (up to about 2 minutes; press no
  other key). Then close the tunnel with Ctrl-C.
- **If the window was lost:** `bash ~/sentinel-beta/package/stop.sh`. It asks only that observation's own guard or
  runtime to stop, and always ends with what is recorded about the desktop.
- **If the desktop did not come back:** `sudo systemctl start display-manager`.

A Ctrl-C or the 60-minute stop that ends cleanly reads `clean=yes` and `OBS_exit=0`. A guard stop reads
`status=guard_stop`, `ended_by=the guard`, `OBS_exit=1`: the memory guard ended the run safely.

## 8. Operating limits (the accepted build's, unchanged)

- **Memory.** Launch needs MemFree ≥ 5,680,693,248 B, MemAvailable ≥ 4,000,000,000 B and memory in use below
  1,294,012,416 B (numbers from one run on the reference device). The guard stops the run at memory pressure
  ≥ 4,800,000,000 B, MemFree < 1 GiB, MemAvailable < 2 GiB, any swap activity or a sampling gap over 0.5 s.
- **Time.** At most 60 minutes per run after readiness; readiness within 300 s.
- **Scope.** One camera (its main stream), one enrolled person, one person in view. Recognition is context only.
- **Chat.** Advisory answers about one frame; one question at a time, at most 300 characters. Monitoring comes first.
- **Other programs.** The memory guard stops every `llama-server` of your user when a run ends, and the launcher
  refuses to start while one runs: do not run another one during an observation.
- **Privacy.** The page is on the Jetson's 127.0.0.1 only. Each run keeps records (numbers per face result, no
  pictures) in `~/sentinel-beta-data/observe/obs-<UTC time>/`; `guard/run.err` holds the runtime's own messages,
  which can name the camera's host and port. Delete a folder when you no longer need it; never share footage, the
  enrollment or credentials.

## 9. Uninstall

```bash
bash ~/sentinel-beta/package/uninstall.sh            # asks UNINSTALL; --keep-assets keeps the downloaded models and builds
```

It removes only what the installer made (the package, its Python environment, its assets, its two record files) and
refuses while anything from the installation runs. **Your data directory is never removed**: configuration,
enrollment and run records stay until you delete them yourself (`bash identity.sh purge --yes` deletes the
enrollment first). If you kept the assets, or an uninstall was interrupted, finish with
`bash ~/sentinel-beta/uninstall.sh`. No sudo; nothing outside the two directories is touched.

## 10. Bug report template

Keep it free of the camera address, passwords, the passphrase, the control key, names and pictures.

```text
When (UTC):
Package: (first line of ~/sentinel-beta/package/VERSION)
Device: (the first five lines of `bash preflight.sh`)
Step: preflight / install / setup-env / fetch-assets / build-llama / build-engine / enrollment / check / launch (which prompt) / dashboard / stop / uninstall
What I did:
What I expected:
What happened (the exact STOP line, or the last lines printed):
For a launch: the "Observation ended", "ended_by", "runtime_after_guard", "enrollment" and "OBS_exit" lines:
Anything else running on the Jetson at the time:
```

## 11. What was and was not verified

Verified on the reference device, without a camera or a model: the preflight; installation, reinstallation and
uninstall in separate directories; the Python environment built from the pinned files (every package at its pinned
version, the GPU and face libraries import, torch sees CUDA); every model file downloaded from upstream and equal to
its SHA-256; the offline check against the reference device's own engine, llama.cpp build and model files. The
scripts' logic is covered by the package's tests with synthetic fixtures.

**Not verified:** anything on another device; stock JetPack 6 on Ubuntu 22.04 (JetPack's `python3-libnvinfer`
binding linked into the environment, cuDSS from NVIDIA's repository); `build-llama.sh` beyond its configure step;
`build-engine.sh` (it has never run); an enrollment made with this package; any launch from this package. Tests with
stand-ins are not an installation on a clean Jetson.
