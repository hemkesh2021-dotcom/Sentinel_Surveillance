# Sentinel v2.0.0-beta.1 — Experimental Beta (tester installation package)

Git tag `v2.0.0-beta.1` · Python package `sentinel-surveillance` `2.0.0b1` · November 2026 · branch
`master`

**An experimental pre-release that adds an installation package for testers with their own Jetson. The beta name
is a release designation, not a new validation result.** Like alpha.1 and alpha.2, it is **not
production-accepted**, and it makes **no admission decision**: face recognition stays
`PENDING_VALIDATION`, chat is not admitted, and no threshold, gallery, identity rule, admission or memory-guard limit
was changed for it. This beta is released from `master`, preserving its existing community and traffic-reporting
files. Evidence for every claim is in
[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), by session (113 to 116); the alpha.2 notes
([`RELEASE_NOTES_v2.0.0-alpha.2.md`](RELEASE_NOTES_v2.0.0-alpha.2.md)) describe the earlier pre-release unchanged.

## The tester installation package

| | |
|---|---|
| Release assets | `sentinel-personal-beta-949da67-mod960fcc3-r9.tar.gz` and `sentinel-personal-beta-949da67-mod960fcc3-r9.tar.gz.sha256` (revision 9 of the package) |
| Size, files | 722,378 B; 126 files |
| SHA-256 | `3b5dde56d1926d6be5391bf8a49c313ca952f8693452c56ebf334bf3929f2d1c` |
| Built from | commit `5a1238a10d764b54a184969734715298338bf43d`, by `packaging/build_archive.py` (the same commit gives the same bytes). The tag is on a later commit that changes documents only: no file that goes into the archive differs between the two |
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
- **Revisions 7 and 8 differ from revision 6 in documents only** (the guide's wording, an entry page, the licence
  and notice files) and in the application's version string. Neither was published: revision 7 was prepared under
  another release name, and revision 8 was replaced by revision 9 before the release. Checked offline only.
- **Revision 9 is this asset.** It differs from revision 8 in one file of the application, the dashboard's script:
  the Ask panel's conversation no longer stops following the newest exchange when its list is resized (a window
  being resized) in a browser that does not keep the view in place itself. The guide and the entry page say so;
  no script, helper, pin or step of the package changed. The correction was checked in a test browser only
  (headless Chromium with its own scroll anchoring switched off for the list, synthetic data), on the reference
  device and in CI. It has not run in a browser that has no scroll anchoring, nor with a camera or a model.
- **Checked on revision 9's real archive, offline only:** its checksum file, its members and its own `SHA256SUMS`;
  that every file in it is the package commit's; that three builds from that commit, under Python 3.10 and 3.12, are
  the same bytes; against revision 8, the same 126 paths with five files differing (the script, the guide, the entry
  page, `VERSION` and `SHA256SUMS`); and an installation, check and uninstall in a scratch folder on the reference
  device, without network, camera, model or build. There `check.sh` stopped at the missing Python environment, as it
  must: a refusal, not a passing check.
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
  follow-ups, zoom, scan, a conversation that scrolls in its own list and keeps following the newest exchange while
  that list is resized, an information key), and identity continuity on a followed track as an opt-in experimental
  mode with proposed values (`--identity-retention`).
- **Review fixes** with regression tests (session 116, parts 29 and 32): chat's held pictures and pinned frames end
  at their stated time, a waiting question ends when chat is turned off, one zoom picture at a time.
- **The memory guard of the installation package** signals only the processes its own run started. It is covered by
  synthetic and stand-in tests; no guarded run with a camera or a model has used it.
- **Version** `2.0.0b1` (`sentinel --version` prints it; nothing else reads it). No feature's status changed with
  the name.

## Still pending (unchanged by this pre-release)

- **Face:** F2 has never validated; there is no false-acceptance data; recognition accuracy is not assessed; testing
  with more than one person is deferred. The maintainer's satisfactory single-person observation of 11 November is
  recorded as reported, not as a validation.
- **Chat:** not admitted; answer correctness is not assessed.
- **The installation package:** everything listed under "Unverified" above, and device resource validation, for
  which no procedure exists yet.
- **From the earlier notes:** whatever the alpha.1 and alpha.2 notes list as open stays open unless the status
  record says otherwise. At publication the wiki still described an earlier alpha; the [current Wiki](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki) was updated for beta.1 on 17 November.

## Checks

This tag is placed only on a commit whose GitHub CI passed: Dashboard checks, and v2 portable checks on Python 3.10
and 3.12 (the portable job fetches the branch's history, which the package's tests compare against).

- **CI on the commit the archive was built from (`5a1238a`):** Dashboard checks passed
  ([run 43871694511](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/actions/runs/43871694511)); v2 portable
  checks passed ([run 43871694570](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/actions/runs/43871694570)):
  the whole suite, 4,224 passed and 78 skipped under Python 3.10.22, the same under Python 3.12.16, and the
  configuration check. The 78 skipped tests need what GitHub's runner does not have: a Jetson's CUDA compiler path
  (45), the reference device's own builds, model files and sensor files (24), or packages and llama.cpp tools that
  the portable job does not install (9). The tagged commit differs from that commit in documents only.
- **Locally on the reference device, not CI:** the full offline suite once at `1fac8d6` (3,931 passed, 7 skipped;
  Python 3.10.14); for the dashboard's correction the browser and packaging test files it affects, under Python
  3.10.14 and 3.12.3 (the conversation's module 37 passed, the package's seven files 333 passed, under each); and
  the checks of the real archive listed above.
