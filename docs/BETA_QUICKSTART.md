# Sentinel personal beta: quick start

> **2026-11-15: this page describes the maintainer's own device and private launcher, as written on 2026-11-11.** For
> installing on another Jetson, use the tester package and its guide instead:
> [`packaging/beta-install/QUICKSTART.md`](../packaging/beta-install/QUICKSTART.md). That package is experimental and
> validates nothing; the statements below about what cannot be reproduced refer to this private launcher.

Private, local use only. This is the build the maintainer accepted for personal use on 2026-11-11: code `949da67` on
branch `v2-ui-refresh`, run by the private launcher `live-observe-949da67`. It **validates and admits nothing**: face
recognition, identity retention and chat run as experimental validation-run features, and their status labels are
unchanged (see "Status labels" below). Testing with more than one person is deferred.

## 0. Who can follow this guide

It works only on the maintainer's Jetson, from the maintainer's own account. Read this before handing it to anyone:

- **The launcher is outside Git.** It lives in `~/sentinel-prep/live-observe-949da67/` (mode 0700), and every path in
  it is relative to the maintainer's home: the worktree, both Python environments, the detector engine, the models, the
  LFM3 llama.cpp build and the sealed enrollment. Another account, or another device, cannot run it.
- **There is no reproducible device installation.** The runtime uses an existing Python environment built for v1 that
  holds the device libraries (TensorRT, CUDA, OpenCV, DeepFace); the README says so, and nothing here changes it.
- **A launch needs three things that are the maintainer's own:** sudo on the Jetson (to stop the desktop and drop the
  page cache), the camera's main-stream RTSP URL (it carries the camera's credentials), and the gallery passphrase.
- **Recognition knows one enrolled person only** (the original enrollment, read only). Anyone else is shown as not
  recognized. Enrolling another person needs their recorded consent and is outside this beta.

So another tester can only use this beta with the maintainer running the launch for them, or after the device setup is
made reproducible (not done).

## 1. Prerequisites

| Where | What |
|---|---|
| Jetson | Jetson Orin Nano, JetPack 6 (L4T 36.4.7), the worktree `~/sentinel-worktrees/v2-ui` at a commit whose code equals `949da67` (later commits may change records only), `~/sentinel-surveillance/.venv` (the guard), `~/onvif_env` (the runtime), `~/yolov8n.engine`, `~/models/lfm3-vl/`, `~/llama.cpp-lfm3/build/bin/llama-server`, `~/sentinel-data/identity/gallery.sealed` |
| Mac (or any SSH client) | SSH to the Jetson; over the USB-C link the Jetson is `192.168.55.1`. `caffeinate` is macOS only: drop it elsewhere and keep the machine awake |
| You | Sudo on the Jetson, the camera's main-stream RTSP URL, the gallery passphrase. Type them only at the launcher's hidden prompts; never write them in a file, a chat or a bug report |

## 2. Setup and configuration

Nothing is installed or edited. The launcher's files are fixed, and its first check compares them with its own
checksums; editing any of them (including its configuration) makes it refuse.

- **Configuration:** `live.yaml` in the launcher folder: camera `cam-1`, the main stream only (`main_single`), the LFM3
  scene adapter (capped profile) and the face adapter (validation runs only).
- **What a launch runs:** `sentinel run` under the memory guard (`step5_guard.py`), with face recognition
  (head windows, validation run), identity retention (opt-in, PROPOSED values), the live view, LFM3 scene analysis,
  chat (validation run) and feature controls. No snapshots and no notifications: the launcher refuses to start if the
  Telegram variables are set.
- **The pin:** the launcher runs only while the worktree's code (`benchmarks src tests config pyproject.toml`) equals
  `949da67` and is committed. A code change needs a new launcher with a new pin; never edit the pin or the checksums.

## 3. Check (no camera, no credentials, no sudo)

On the Jetson:

```bash
bash ~/sentinel-prep/live-observe-949da67/start.sh --check
```

It must end with `OBS_CHECK=passed`. It checks the files, the pin, the admission decisions and the launch command up to
the camera URL. A `memfree_bytes` below `launch_floor_bytes` is normal here: the launch offers PREPARE.

## 4. Mac tunnel (start it first, keep it open)

```bash
caffeinate -i ssh -N -o ExitOnForwardFailure=yes -L 18090:127.0.0.1:18090 <user>@192.168.55.1
```

The page listens on the Jetson's 127.0.0.1 only; this forwards it to the Mac. The window shows nothing while it works.

## 5. Launch: typing START opens the camera

In a second Mac Terminal window, open a plain SSH shell (not a VS Code terminal) with `ssh <user>@192.168.55.1`, then:

```bash
bash ~/sentinel-prep/live-observe-949da67/start.sh
```

It asks, in order:

1. **CLOSE** (only if optional processes run): ends Claude Code, Codex and VS Code server sessions on the Jetson,
   update-manager and a VNC desktop, each listed first. Save your work before typing it.
2. **PREPARE** (only if needed): stops the desktop (it restarts at the end) and drops the page cache once; sudo may
   ask for your password.
3. The camera's **main-stream URL** (hidden, never saved).
4. The **gallery passphrase** (hidden, never saved).
5. **START.** **Only typing START opens the camera.** Pressing Enter at any prompt stops without opening it.

The models load first (1-2 minutes), then the camera opens. A run lasts at most 60 minutes after the models are
ready, then stops by itself. **Start nothing else on the Jetson (Claude Code included) until `OBS_exit`:** extra memory
can trigger the guard's stop.

## 6. Dashboard

Open **http://127.0.0.1:18090/dashboard** on the Mac. The top bar says whether the page is current: it updates every
2 s and reads NOT CURRENT within 5 s if updates stop. The live picture comes first, recognition beside it, then
incidents, Ask (chat), health and alerts; Appearance and Features open one Settings window from the sidebar. In
Ask, the ⓘ beside the heading explains what may be asked and where answers come from.

## 7. Feature-control key

Pausing or resuming face, scene, chat or the live view (Settings → Features) needs this run's control key. Once the
run is ready, from another SSH shell on the Jetson:

```bash
cat "$(ls -d ~/sentinel-data/observe/obs-*/ | sort | tail -n 1)db/control-key"
```

The key is deleted at the stop. The controls never start, load or admit anything.

## 8. Clean stop

1. Press **Ctrl-C once** in the Jetson window. Nothing more appears until `OBS_exit` (up to about 2 minutes); press
   no other key.
2. Sudo may ask for your password to restart the desktop.
3. Close the Mac tunnel with Ctrl-C.

A Ctrl-C or the 60-minute stop that ends cleanly reads `clean=yes` and `OBS_exit=0`. A guard stop reads
`status=guard_stop`, `clean=no` and `OBS_exit=1`: the guard ended the run safely; it is not a defect by itself.

## 9. Recovery

| What you see | What to do |
|---|---|
| The launch window was lost | `bash ~/sentinel-prep/live-observe-949da67/stop.sh` (signals only this observation's guard or runtime; with nothing running it says so and signals nothing) |
| The desktop did not come back | `sudo systemctl start display-manager` |
| `STOP: the code differs from 949da67…` | The worktree's code changed after the pin. Use the launcher pinned to the current code; do not edit the pin |
| `STOP: a llama-server is running` or `a Sentinel runtime is running` | Run `stop.sh`; it prints the exact command for anything it may not stop itself |
| `STOP: port 18090 or 18081 is already in use` | Something else holds the port: stop it, or wait for an earlier run to finish |
| `STOP: transparent huge pages are disabled for this shell` | Open a new plain SSH shell and rerun |
| `MemFree … is below the launch floor` or `memory in use … is at or above …` | Close what is running on the Jetson, rerun `start.sh` (it offers CLOSE and PREPARE) |
| `runtime_after_guard=still_running` | Run `stop.sh`; the desktop stays stopped until the runtime has gone |
| `original_enrollment=CHANGED` | Stop using the beta and report it: the enrollment must never change |
| VS Code, Claude Code or VNC after CLOSE | Reconnect from the Mac as usual (`claude --continue` in the same directory resumes a session); the scan prints the command that restores VNC |

## 10. Operating limits (unchanged)

- **Memory.** Launch needs MemFree ≥ 5,680,693,248 B, MemAvailable ≥ 4,000,000,000 B and memory in use below
  1,294,012,416 B. The guard stops the run at memory pressure (MemTotal − MemAvailable) ≥ 4,800,000,000 B, MemFree
  < 1 GiB, MemAvailable < 2 GiB, any swap activity, or a sampling gap over 0.5 s. Whole-device targets of
  5,000,000,000 B steady and 5,400,000,000 B peak are acceptance targets, not measurements.
- **Time.** At most 60 minutes per run after readiness; readiness must come within 300 s.
- **Scope.** One camera (its main stream), one enrolled person, one person in view at a time (multi-person testing
  deferred). What has been observed is ordinary single-person use in the room (the maintainer's report); the 4-5 m
  doorway target is not validated. Recognition is context only: it never grants or denies anything.
- **Chat.** Advisory answers about one frame; it cannot acknowledge incidents, change rules, control the camera, run
  commands or identify anyone. One question at a time, at most 300 characters, one every 4 s; automatic refresh every
  15-120 s (off until you start it); Zoom and Scan hold at most 2 pictures for 120 s, in memory only; the conversation
  keeps 12 exchanges. Monitoring comes first: chat pauses or is cancelled under memory pressure or a scene check.
- **Privacy.** The page is on the Jetson's 127.0.0.1 only. Each run keeps records (numbers per face result, no
  pictures) in `~/sentinel-data/observe/obs-<UTC time>/`; delete a folder when you no longer need it. Never share
  footage, screenshots of the live picture, the gallery or credentials.
- **Not tried:** Safari's and Firefox's handling of the newest dashboard changes, and a real phone; only headless
  Chromium ran the offline browser checks.

## 11. Status labels (unchanged by this beta)

| Feature | Status |
|---|---|
| Face recognition | experimental: `PENDING_VALIDATION`, runs only as a validation run |
| Identity retention | opt-in experimental mode; every value PROPOSED |
| Chat | not admitted; runs as an authorized validation run, and the page says so |
| Scene analysis (capped LFM3) | passes the runtime's admission check against the D107 profile, which measured a recorded replay without camera, chat or page; this live configuration is outside what it measured, and the guard is the protection |
| Feature controls | opt-in; not validated |

## 12. Bug report template

Copy, fill in, and keep it free of credentials, the control key, names and pictures.

```text
When (UTC):
Launcher: live-observe-949da67      Commit: (git -C ~/sentinel-worktrees/v2-ui rev-parse --short HEAD)
Observation folder: obs-YYYYMMDDTHHMMSSZ (or "not launched")
Step: check / tunnel / launch prompt (which) / dashboard (which panel) / stop / recovery
What I did:
What I expected:
What happened (exact text of any STOP line):
Last lines of the Jetson window (Observation ended / runtime_after_guard / original_enrollment / OBS_exit):
Browser and device used for the dashboard:
Anything else running on the Jetson at the time:
```
