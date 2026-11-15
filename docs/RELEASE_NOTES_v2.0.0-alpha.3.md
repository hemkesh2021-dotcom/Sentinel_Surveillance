# Sentinel v2.0.0-alpha.3 — Experimental Alpha (tester installation package)

Git tag `v2.0.0-alpha.3` · Python package `sentinel-surveillance` `2.0.0a3` · November 2026 · branch
`alpha-candidate-2026-11`

**An experimental pre-release that adds an installation package for testers with their own Jetson.** Like alpha.1
and alpha.2, it is **not production-accepted**, and it makes **no admission decision**: face recognition stays
`PENDING_VALIDATION`, chat is not admitted, and no threshold, gallery, identity rule, admission or memory-guard limit
was changed for it. Nothing is merged into `master`. Evidence for every claim is in
[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), by session (113 to 116); the alpha.2 notes
([`RELEASE_NOTES_v2.0.0-alpha.2.md`](RELEASE_NOTES_v2.0.0-alpha.2.md)) describe the earlier pre-release unchanged.

## The tester installation package

| | |
|---|---|
| Release assets | `sentinel-personal-beta-949da67-mod191a193-r7.tar.gz` and `sentinel-personal-beta-949da67-mod191a193-r7.tar.gz.sha256` |
| Size, files | 720,936 B; 126 files |
| SHA-256 | `51a717d4b92823afa033d9fb5ecf022be43c7dac01417ba8a4f2d2f9a130dcde` |
| Built from | commit `8ef4d027af92a99987d671bcbf88cb7ad369fc04`, by `packaging/build_archive.py` (the same commit gives the same bytes) |
| Guide | [`packaging/beta-install/QUICKSTART.md`](../packaging/beta-install/QUICKSTART.md); entry page [`README.md`](../packaging/beta-install/README.md); both are in the archive |

It installs under the tester's own account without sudo: a preflight, the installation, a Python environment from
hash-pinned requirements, the model files from their upstream sources (each checked against its SHA-256), llama.cpp
and the detector engine built on the device under a memory monitor, the tester's own enrollment, a record of the
device's own files, an offline check, a manual launch, a clean stop and an uninstall. The archive holds Sentinel's
own files only (text): no model, engine, gallery, enrollment, credential, recording or observation data.

### What ran, and what did not

- **Revision 5 of the package completed a camera-free build rehearsal on the reference device** (one Jetson Orin
  Nano 8 GB; 13 to 15 November): the installation, the environment, the downloads, the llama.cpp build (53 minutes
  with two jobs), the detector engine export, the device-artifacts record, its verification, and the opt-in route's
  admission decision for that record. The export completed once, in 187 s, started with 6.95 GB of free memory after
  a manual page-cache release and with no desktop session; two earlier attempts ended without an engine (one failed
  with too little free memory, one was stopped by the monitor at its 1 GiB limit). In the rehearsal `check.sh`
  stopped at the missing enrollment, as it must: that is a refusal, not a passing check.
- **Revision 6's packaging changes were checked offline only:** the installation's own cache, settings and temporary
  locations, and `guarded-build.sh` with the rehearsal's memory monitor.
- **Revision 7, this asset, differs from revision 6 in documents only** (the guide's wording, an entry page, the
  licence and notice files) and in the application's version string. Checked offline only: the package's tests,
  the archive's own checks, and an installation, check and uninstall of the real archive in a scratch folder on the
  reference device, without network, camera, model or build.
- **Unverified:** a complete launch from the package (none has ever been made); the complete check on real builds;
  an enrollment made with the package; any other Jetson; the least memory and the time the builds need.
- **Stock JetPack 6 on Ubuntu 22.04 is unverified.** The reference device runs L4T R36.4.7 with an Ubuntu 24.04
  userland and an extracted TensorRT Python binding. On stock JetPack the preflight says NOT VERIFIED and the
  installer goes on only with the explicit `--accept-unverified-platform`.

### What a tester needs to know

- **The default check and launch refuse on another device.** The default admission admits only the reference
  device's own engine, llama.cpp build and model-file copies. A tester's own files are used only through the opt-in
  `--device-artifacts FILE` route, for experimental validation runs. The route identifies files; it validates nothing
  about the device. **Device resource validation is pending**, and the launch gates and the guard's limits are the
  reference device's limits, not evidence about another device.
- **Testers supply their own** configuration (the camera's address is typed at launch and never saved), enrollment
  (photos of themselves, sealed with their own passphrase) and device artifacts (the two builds and their record).
- **Face data and the camera.** `identity.sh enroll` reads the tester's own photos and writes the sealed enrollment.
  `start.sh` opens the camera, and compares faces in view with that enrollment, only after the tester types START.
  No other step touches either; the guide's section 12 lists every step.
- **Launch, dashboard, stop, recovery.** `start.sh --device-artifacts FILE` from a plain SSH shell; the dashboard at
  `http://127.0.0.1:18090/dashboard` through `ssh -N -o ExitOnForwardFailure=yes -L 18090:127.0.0.1:18090 <user>@<jetson>`;
  Ctrl-C once, or `stop.sh`, for a clean stop. Memory preparation before the engine build is the tester's own manual
  step. Recovery after an interrupted step is in the guide's sections 5, 9 and 11.
- **Licences.** The archive's files are under the repository's MIT licence (`LICENSE` is in the archive). What its
  scripts download keeps its own licences and is not in the archive: among them AGPL-3.0 (Ultralytics and the
  detector weights), the LFM Open License v1.0 (the scene model, with a condition on commercial use) and NVIDIA's
  licence agreements. [`THIRD_PARTY_NOTICES.md`](../packaging/beta-install/THIRD_PARTY_NOTICES.md) names each with
  its source.
- **Problems:** the [installation issue form](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/issues/new?template=installation-issue.md).
  The tracker is public: no credentials, camera addresses, photos, footage, enrollments or whole logs.

## What else is new since alpha.2 (`27d6308`, 21 October)

- **Scene analysis and chat use LFM3-VL-2B by default,** through a pinned llama.cpp build, with scene reports
  constrained by a grammar and an image-token cap on the model server's launch (it leaves the submitted pictures as
  they are and shrinks the vision warm-up). In a guarded integration run on the reference device, execution, memory
  and face execution passed; chat completion, and so the run's overall reading, stay not passed, because no
  cancellation occurred to be judged. A separate controlled preemption check passed, as separate evidence. Neither
  admits anything, and chat's status is unchanged.
- **Recognition at a distance was examined, validation only:** tooling and guarded sessions for stream timing, the
  door range and a main-stream recognition route for validation runs, and a multi-pose enrollment comparison in
  which two of four predeclared criteria were not met (its candidate was not adopted). No threshold changed, and the
  default recognition path is unchanged.
- **Dashboard:** a Settings dialog, an Ask panel (questions about the live picture and about what was recorded,
  follow-ups, zoom, scan, a conversation that scrolls in its own list, an information key), and identity
  continuity on a followed track as an opt-in experimental mode with proposed values (`--identity-retention`).
- **Review fixes** with regression tests (session 116, parts 29 and 32): chat's held pictures and pinned frames end
  at their stated time, a waiting question ends when chat is turned off, one zoom picture at a time.
- **The memory guard of the installation package** signals only the processes its own run started. It is covered by
  synthetic and stand-in tests; no guarded run with a camera or a model has used it.
- **Version** `2.0.0a3` (`sentinel --version` prints it; nothing else reads it).

## Still pending (unchanged by this pre-release)

- **Face:** F2 has never validated; there is no false-acceptance data; recognition accuracy is not assessed; testing
  with more than one person is deferred. The maintainer's satisfactory single-person observation of 11 November is
  recorded as reported, not as a validation.
- **Chat:** not admitted; answer correctness is not assessed.
- **The installation package:** everything listed under "Unverified" above, and device resource validation, for
  which no procedure exists yet.
- **From the earlier notes:** whatever the alpha.1 and alpha.2 notes list as open stays open unless the status
  record says otherwise. The wiki was last revised for alpha.2.

## Checks

This tag is placed only on a commit whose GitHub CI passed: Dashboard checks, and v2 portable checks on Python 3.10
and 3.12 (the portable job now fetches the branch's history, which the package's tests compare against). Locally on
the device, not CI: the full offline suite once at `1fac8d6` (3,931 passed, 7 skipped; Python 3.10.14); at the
release preparation only the package's focused test files (330 passed, none skipped, under Python 3.10.14, and the same files, 330 passed, in a fresh clone of the package commit under Python 3.12.3).
