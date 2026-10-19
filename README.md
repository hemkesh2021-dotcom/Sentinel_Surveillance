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

![Version](https://img.shields.io/badge/version-2.0.0--alpha.1-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/status-experimental_alpha-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jetson](https://img.shields.io/badge/NVIDIA-Jetson_Orin_Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

---

> **EXPERIMENTAL ALPHA: Sentinel v2.0.0-alpha.1 is an experimental pre-release.** It is published so that the work
> can be reviewed and tried. It is **not production-accepted**, and **feature admission is a separate decision**:
> face recognition and chat are not admitted for normal use, and nothing on this branch is merged into `master`.
>
> **The alpha's goal is not met yet.** The October alpha was set to centre on working face recognition and real local
> chat with the vision-language model. Neither is done:
> - face recognition's formal validation (F2) has never passed, and there is no false-acceptance data;
> - chat is not admitted;
> - the two have never run together.
>
> Face recognition is shown only as an experimental demonstration in guarded validation runs. Later work (validation
> diagnostics and their device sessions, through 19 October) is unreleased: the version is still `2.0.0a1`. See the
> [alpha notes](docs/ALPHA_CANDIDATE_NOTES.md).
>
> - **Known limitations:** [Status and limitations](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Status-and-limitations)
>   (wiki) and [`docs/ALPHA_CANDIDATE_NOTES.md`](docs/ALPHA_CANDIDATE_NOTES.md)
> - **Release:** [v2.0.0-alpha.1](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases/tag/v2.0.0-alpha.1)
>   ([release notes](docs/RELEASE_NOTES_v2.0.0-alpha.1.md))
> - **Wiki:** [github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki)
> - **Evidence for every claim, session by session:** [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
> - **Legacy v1 (the original prototype):** [`docs/LEGACY_V1.md`](docs/LEGACY_V1.md)

## Overview

Sentinel watches one camera from an NVIDIA Jetson Orin Nano. It detects and tracks people, applies zone rules, opens
incidents in a local SQLite database and delivers alerts through a restart-safe outbox. A dashboard on the device's
loopback address shows the live picture, the incidents with their snapshots and, in validation runs, face recognition
per tracked person. Face recognition (DeepFace with Facenet512 and YuNet) and scene analysis with a local
vision-language model (LFM2-VL through llama.cpp's `llama-server`) are optional lanes. They run only when the
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
| Dashboard (live picture, recognition panel, incidents, appearance) | yes | checked in validation runs; Safari observed by the operator | opt-in flags |
| Face recognition | yes | described in validation runs, **not accepted**: no F2 has validated (three attempts on the normal path since the pre-release); in ordinary seated use recognition is intermittent (a single non-matching vote ends it; it lapses when no face is detected for longer than the 30 s vote lifetime) | **no** (`PENDING_VALIDATION`) |
| Validation-only diagnostics (tracking, landmarks, a launch-memory check for operator packages) | yes | each ran in device sessions, descriptive only | not features; guarded validation runs only |
| Identity continuity rule (proposed) | yes | ran in one session; not validated | **no** (validation runs only) |
| Feature controls (pause and resume lanes) | yes | partly exercised; not validated | opt-in flag |
| Chat about the live frame (local VLM) | yes | runtime checks passed in two earlier runs | **no**: admission, factual accuracy and combined face/VLM validation pending |
| Scene analysis (local VLM) | yes | earlier scene checks; not with face recognition | needs an admitted adapter |

The [known limitations](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Status-and-limitations) list
everything still unresolved; none of it is waived by this pre-release.

## Getting started

### Portable development (CI runs on every push)

Python 3.10 or newer. No camera, GPU or model is needed. The unreleased commits after `65df7f3` have not been pushed,
so they have not run in CI.

```bash
git clone --branch v2.0.0-alpha.1 https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance.git
cd Sentinel_Surveillance
python -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest                              # browser tests run when a Chromium-family browser is found
sentinel config validate config/default.yaml  # checks a configuration file
sentinel --version                            # sentinel 2.0.0a1
```

### On the device (as the recorded device checks ran it)

**There is no complete, reproducible device installation guide yet.** What follows is only what the recorded checks
used; anything not listed here is not documented.

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
| `docs/` | v2 status record, requirements trace, alpha notes, release notes, legacy v1 README |
| `surveillance4_1.py`, `dashboard.py`, `build_face_db.py`, `start_sentinel.sh`, `requirements.txt`, `.env.example`, `BUILD.md`, `SETUP_GUIDE.txt`, `tests/test_dashboard.py` | legacy v1, unchanged; see [`docs/LEGACY_V1.md`](docs/LEGACY_V1.md) |

## Maintainer

V2 maintainer: Hemkesh ([@hemkesh2021-dotcom](https://github.com/hemkesh2021-dotcom)).

Historical v1 contributor credits are kept in [`docs/LEGACY_V1.md`](docs/LEGACY_V1.md#authors).

## License

This project is for personal / educational use. Do not deploy in public spaces without complying with local privacy
laws — face embeddings are **biometric data** (see DPDP / GDPR / BIPA).

> Note: YOLOv8 / Ultralytics is **AGPL-3.0**. Any closed-source or commercial deployment requires either open-sourcing
> the derived work or an Ultralytics commercial license.
