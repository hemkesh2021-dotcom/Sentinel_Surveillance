<div align="center">

```
███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     
███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     
███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

**Sentinel v2: edge monitoring for one camera on an NVIDIA Jetson**
*Person detection and tracking · zone rules · incidents and alerts · a local dashboard ·
face recognition and scene analysis under validation*

![Version](https://img.shields.io/badge/version-2.0.0--beta.1-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/status-experimental_pre--release-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jetson](https://img.shields.io/badge/NVIDIA-Jetson_Orin_Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

---

> **EXPERIMENTAL PRE-RELEASE: Sentinel v2.0.0-beta.1.** The beta name is a release designation, not a validation
> result: no feature's status, threshold or admission changed for it. It adds an installation package for testers
> with their own Jetson ([Install on your own Jetson](#install-on-your-own-jetson-tester-package-experimental)) and
> carries the work since alpha.2. It is published so that the work can be reviewed and tried. It is **not
> production-accepted**, and **feature admission is a separate decision**: face recognition and chat are not admitted
> for normal use. This beta is released from `master`; the published v1 and alpha tags remain available.
>
> **The installation package validates nothing.** One revision of it completed a camera-free build rehearsal on the
> reference device; its later changes were checked offline only; a complete launch from the package, and
> any other Jetson, are unverified. The [release notes](docs/RELEASE_NOTES_v2.0.0-beta.1.md) say exactly what ran.
>
> **The alpha's goal is not met yet.** The October alpha was set to centre on working face recognition and real local
> chat with the vision-language model. Neither is done:
> - face recognition's formal validation (F2) has never passed, and there is no false-acceptance data;
> - chat is not admitted;
> - the two have run together only in validation-only runs: two eight-minute integration checks (20 October) and a
>   one-hour live-dashboard trial with the operator's own questions (21 October). All passed, but they judged no
>   recognition accuracy and no answer's correctness. The second integration check asked four grounding questions,
>   judged by the maintainer against the staged room (two consistent, one unverifiable, one cancelled and untested).
>   None is the combined validation, which still needs a passing F2, and they admit nothing.
>
> Face recognition is shown only as an experimental demonstration in guarded validation runs. See the
> [alpha notes](docs/ALPHA_CANDIDATE_NOTES.md).
>
> - **Known limitations:** [Status and limitations](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Status-and-limitations)
>   (wiki) and [`docs/ALPHA_CANDIDATE_NOTES.md`](docs/ALPHA_CANDIDATE_NOTES.md)
> - **Release:** [v2.0.0-beta.1](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases/tag/v2.0.0-beta.1)
>   ([release notes](docs/RELEASE_NOTES_v2.0.0-beta.1.md)); the earlier
>   [v2.0.0-alpha.2](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases/tag/v2.0.0-alpha.2)
>   ([release notes](docs/RELEASE_NOTES_v2.0.0-alpha.2.md)) and
>   [v2.0.0-alpha.1](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases/tag/v2.0.0-alpha.1)
>   ([release notes](docs/RELEASE_NOTES_v2.0.0-alpha.1.md))
> - **Installation problems:** the [installation issue form](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/issues/new?template=installation-issue.md)
>   (public: no credentials, photos, footage or whole logs)
> - **Wiki:** [github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki)
> - **Evidence for every claim, session by session:** [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
> - **Legacy v1 (the original prototype):** [`docs/LEGACY_V1.md`](docs/LEGACY_V1.md)

## Overview

Sentinel watches one camera from an NVIDIA Jetson Orin Nano. It detects and tracks people, applies zone rules, opens
incidents in a local SQLite database and delivers alerts through a restart-safe outbox. A dashboard on the device's
loopback address shows the live picture, the incidents with their snapshots and, in validation runs, face recognition
per tracked person. Face recognition (DeepFace with Facenet512 and YuNet) and scene analysis with a local
vision-language model (LFM3-VL-2B through llama.cpp's `llama-server`) are optional lanes. They run only when the
configuration lists their adapters as admitted, or in guarded validation runs.

v2 is a rewrite of the v1 prototype as a tested package (`src/sentinel/`, command `sentinel`). Every observation
carries its frame's source time, the boot and the stream epoch. Stale or mismatched results never drive a current
decision. A face is associated with at most one person, and an unknown face is never called a stranger. Fire or smoke
from the scene model alone is never a critical alarm. Empty, fresh, stale and offline states are published
explicitly.

## Architecture

```
camera (RTSP; URL from SENTINEL_RTSP_URL, never a file)
  │
  ▼
capture: one stream, frames stamped with stream epoch, sequence and ingest time
  │
  ├─► person detector (YOLOv8n TensorRT engine) ─► tracker (ByteTrack) ─► zone rules
  │                                                                          │
  │                                                                          ▼
  │                                       incidents (SQLite, one writer) ─► alert outbox ─► Telegram
  │                                                                          │
  ├─► face worker (optional, 1 Hz; sealed gallery) ──► identity state ──────┤
  ├─► scene lane (optional; llama-server on 127.0.0.1) ─► scene reports ─────┤
  │                                                                          ▼
  └────────────────────────────────────────────► status page and dashboard on 127.0.0.1:18090
                                                  (live view, snapshots, chat and feature controls opt-in)
```

| Area | Package path |
|---|---|
| Capture, frame identity, freshness | `src/sentinel/media/` |
| Person detection and tracking | `src/sentinel/inference/`, `src/sentinel/tracking/` |
| Zone and schedule rules, scene hazards | `src/sentinel/rules/` |
| Incidents, snapshots, the database | `src/sentinel/incidents/`, `src/sentinel/storage/` |
| Alert outbox and Telegram | `src/sentinel/alerts/` |
| Face association, sealed gallery, matching | `src/sentinel/identity/` |
| Scene analysis, llama-server, chat | `src/sentinel/scene/` |
| Status page and dashboard | `src/sentinel/status_page.py`, `src/sentinel/web/` |
| The runtime and the command line | `src/sentinel/demo_runtime.py`, `src/sentinel/runtime.py`, `src/sentinel/cli.py` |
| Operator check tooling (memory guard, readings) | `benchmarks/runner/` |

Device-only libraries (TensorRT, CUDA, DeepFace, OpenCV) are imported only when a device lane starts, so the package,
its configuration checks and its tests run on any machine.

## Feature status

| Feature | Implemented, portable-tested | On the device | Admitted for normal use |
|---|---|---|---|
| Core monitoring: capture, detection, tracking, zone rules, incidents, alert outbox | yes | one guarded check passed within its bounded scope, with one real Telegram send | core path; no production acceptance |
| Incident page and acknowledgment (while monitoring is stopped) | yes | checked | core path |
| Live view and incident snapshots | yes | checked | opt-in flags |
| Dashboard (live picture, recognition panel, incidents, appearance) | yes | checked in validation runs, including a one-hour trial with face recognition, scene analysis and chat; Safari observed by the operator | opt-in flags |
| Face recognition | yes | described in validation runs, **not accepted**: no F2 has validated (three attempts on the normal path since alpha.1); in ordinary seated use recognition is intermittent (a single non-matching vote ends it; it lapses when no face is detected for longer than the 30 s vote lifetime) | **no** (`PENDING_VALIDATION`) |
| Validation-only diagnostics (tracking, landmarks, a launch-memory check for operator packages) | yes | each ran in device sessions, descriptive only | not features; guarded validation runs only |
| Identity continuity rule (proposed) | yes | ran in one session; not validated | **no** (validation runs only) |
| Feature controls (pause and resume lanes) | yes | partly exercised; not validated | opt-in flag |
| Chat about the live frame (local VLM) | yes | runtime checks passed in two earlier runs, and with face recognition in two validation-only integration runs and a one-hour dashboard trial (answers judged for form only, or by the operator's report) | **no**: admission, factual accuracy and combined face/VLM validation pending |
| Scene analysis (local VLM) | yes | earlier scene checks; loaded beside face recognition in its validation runs, the integration runs and the one-hour trial, not validated together | needs an admitted adapter |

The [known limitations](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Status-and-limitations) list
everything still unresolved; none of it is waived by this pre-release. The table above and the wiki were last revised
for alpha.2. No admission status has changed since; what was added is in the
[beta.1 release notes](docs/RELEASE_NOTES_v2.0.0-beta.1.md) and, session by session, in the status record.

## Getting started

### Portable development (CI runs on every push)

Python 3.10 or newer. No camera, GPU or model is needed. A pre-release tag is placed only on a commit whose CI passed
(Dashboard checks; v2 portable checks on Python 3.10 and 3.12). On the commit the release's archive was built from,
`5a1238a`, CI ran the whole suite under both versions: 4,224 passed, 78 skipped in each
([Dashboard checks](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/actions/runs/43871694511),
[v2 portable checks](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/actions/runs/43871694570)). Locally on
the device, not CI: the full offline suite passed once under Python 3.10 at `1fac8d6` (3,931 passed, 7 skipped); the
commits after it ran the test files they affect.

```bash
git clone --branch v2.0.0-beta.1 https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance.git
cd Sentinel_Surveillance
python -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest                              # browser tests run when a Chromium-family browser is found
sentinel config validate config/default.yaml  # checks a configuration file
sentinel --version                            # sentinel 2.0.0b1
```

### Install on your own Jetson (tester package, experimental)

The pre-release carries one archive, revision 9 of the package, and its `.sha256` file
([releases](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases)). It installs under your own
account, without sudo, downloads the pinned Python packages and model files from their upstream sources, and builds
llama.cpp and the detector engine on your device. Its guide is
[`packaging/beta-install/QUICKSTART.md`](packaging/beta-install/QUICKSTART.md) (also inside the archive, with a
`README.md`).

| | |
|---|---|
| Archive | `sentinel-personal-beta-949da67-mod960fcc3-r9.tar.gz` |
| Size, files | 722,378 B; 126 files |
| SHA-256 | `3b5dde56d1926d6be5391bf8a49c313ca952f8693452c56ebf334bf3929f2d1c` |
| Built from | commit `5a1238a10d764b54a184969734715298338bf43d`, by `packaging/build_archive.py` |

```bash
sha256sum -c sentinel-personal-beta-949da67-mod960fcc3-r9.tar.gz.sha256   # then extract it and read README.md and QUICKSTART.md
```

Before you start:

- **What ran, and what did not.** Revision 5 of the package completed a camera-free build rehearsal on the reference
  device (installation, environment, downloads, both builds, the record of the built files and its verification).
  The packaging changes of revision 6 and the documents, licence files and version string of revisions 7 and 8
  were checked offline only. Revision 9, this archive, differs from revision 8 in one file of the application, the
  dashboard's script (the Ask panel's conversation while a window is resized), which was checked in a test browser
  only. The archive itself was installed, checked and uninstalled in a scratch folder on the reference device,
  without network, camera, model or build. **A complete launch from the package has never been made, and no other
  Jetson has run it.**
- **The default check and launch refuse on your device.** They admit only the reference device's own files. Your
  files are used only through the opt-in `--device-artifacts` route, for experimental validation runs; it validates
  nothing about your device's memory, speed or stability (resource validation is pending).
- **Stock JetPack is unverified.** The reference device runs L4T R36.4.7 with an Ubuntu 24.04 userland. On stock
  JetPack 6 the installer goes on only with `--accept-unverified-platform`.
- **You supply your own** camera, configuration, enrollment (photos of yourself, sealed with your passphrase) and
  device artifacts. Nobody's gallery, credentials, models, engine or recordings are in the archive.
- **Face data and the camera.** Only `identity.sh enroll` reads your photos, and only `start.sh`, after you type
  START, opens the camera. The guide's section 12 lists every step.
- **Licences.** The archive holds Sentinel's own files (MIT). What it downloads keeps its own licences, among them
  AGPL-3.0 and the LFM Open License v1.0:
  [`THIRD_PARTY_NOTICES.md`](packaging/beta-install/THIRD_PARTY_NOTICES.md).

Problems go to the [installation issue form](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/issues/new?template=installation-issue.md).

### On the device (as the recorded device checks ran it)

**Apart from the tester package above, there is no device installation guide.** What follows is only what the
recorded checks used on the reference device; anything not listed here is not documented.

- **Hardware and system:** one Jetson Orin Nano with JetPack 6 (L4T 36.4.7) and one RTSP camera, reached from a
  workstation over SSH.
- **Python with the device libraries:** the checks ran the package from the repository with an existing Python
  environment that already held the device libraries (TensorRT, Ultralytics, OpenCV and, for face recognition,
  DeepFace). That environment was built for v1; how to build one for v2 is not documented.
- **Detector engine:** a profiled `yolov8n.engine` (TensorRT). The engine used was built for v1 (see
  [`BUILD.md`](BUILD.md), legacy); a v2 build procedure is not documented.
- **The command shape** (core monitoring, with the dashboard's live view):

  ```bash
  read -rs SENTINEL_RTSP_URL && export SENTINEL_RTSP_URL   # typed, never stored in a file
  LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONPATH=src <device-python> -m sentinel.cli \
    run <config>.yaml --data-dir <data-dir> --engine <path>/yolov8n.engine --live-preview --snapshots
  ```

  `config/default.yaml` is the core-only starting point (`sentinel config validate` checks a file); the device checks
  used their own configuration files, kept with their private evidence and not in the repository.

  The checks ran under the memory guard (`benchmarks/runner/step5_guard.py`) and added
  `--workload-thp-disable --post-load-release`. `sentinel run --help` lists every option. Face recognition, scene
  analysis and chat need admitted adapters in the configuration; the device checks used them only through guarded
  validation packages.
- **Opening the dashboard:** the page listens on 127.0.0.1 only. From the workstation:
  `ssh -N -L 18090:127.0.0.1:18090 <user>@<device>`, then open `http://127.0.0.1:18090/dashboard`.

More detail: [Getting started](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Getting-started) and
[Dashboard and controls](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Dashboard-and-controls).

## Data and privacy

- Credentials (the camera URL, the Telegram token, the gallery passphrase) come from the environment and are never
  written to the repository or the database. Configuration files must hold none.
- The face gallery is sealed with a passphrase, needs recorded consent to enroll, and can be revoked and purged
  (`sentinel identity`). Face embeddings are biometric data.
- No page, log or record shows a person's name. Face recognition is context only: it never grants or denies anything.
- Never commit galleries, photos, footage, run evidence or credentials.

## Repository layout

| Path | What it is |
|---|---|
| `src/sentinel/`, `tests/`, `config/`, `pyproject.toml` | Sentinel v2 |
| `benchmarks/runner/` | v2 operator check tooling (memory guard, readings of saved evidence) |
| `packaging/` | the tester installation package (`beta-install/`) and the tool that builds its archive |
| `docs/` | v2 status record, requirements trace, alpha notes, release notes, legacy v1 README |
| `surveillance4_1.py`, `dashboard.py`, `build_face_db.py`, `start_sentinel.sh`, `requirements.txt`, `.env.example`, `BUILD.md`, `SETUP_GUIDE.txt`, `tests/test_dashboard.py` | legacy v1, unchanged; see [`docs/LEGACY_V1.md`](docs/LEGACY_V1.md) |

## Maintainer

V2 maintainer: Hemkesh ([@hemkesh2021-dotcom](https://github.com/hemkesh2021-dotcom)).

Historical v1 contributor credits are kept in [`docs/LEGACY_V1.md`](docs/LEGACY_V1.md#authors).

## License

Sentinel's own code is under the MIT License: see [`LICENSE`](LICENSE). Third-party software and model files are not
part of this repository and keep their own licences; the tester package's
[`THIRD_PARTY_NOTICES.md`](packaging/beta-install/THIRD_PARTY_NOTICES.md) lists the ones it downloads.

This project is for personal / educational use. Do not deploy in public spaces without complying with local privacy
laws — face embeddings are **biometric data** (see DPDP / GDPR / BIPA).

> Note: YOLOv8 / Ultralytics is **AGPL-3.0**. Any closed-source or commercial deployment requires either open-sourcing
> the derived work or an Ultralytics commercial license.
