# Sentinel personal beta: installation package for testers

This archive installs Sentinel v2 on your own NVIDIA Jetson, for experimental testing. It belongs to the
experimental pre-release **v2.0.0-beta.1** of Sentinel (application version `2.0.0b1`):
<https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases>

**The beta name is a release designation, not a validation result.**

**Start with [`QUICKSTART.md`](QUICKSTART.md).** It is the installation guide, in the order to follow. This page is
only the summary.

## What you should know first

- **Experimental, and it validates nothing.** Face recognition, identity retention and chat run only as
  experimental validation-run features. Nothing here is accepted for production, and no feature is admitted for
  normal use.
- **The default check and launch refuse on your device.** The default admission accepts only the exact files of
  the reference device, and those are not in this archive. Your own files can be used only through the opt-in
  device-artifacts route (`--device-artifacts FILE`), for experimental validation runs. `QUICKSTART.md`, section 1,
  says what that route does and does not do.
- **What has actually been run:**
  - Revision 5 of this package completed a camera-free build rehearsal on the reference device (one Jetson Orin
    Nano 8 GB): installation, the Python environment, the downloads, the llama.cpp build, the detector engine
    export, the record of those files and its verification.
  - Revision 6's packaging changes (the installation's own cache locations, `guarded-build.sh`), revision 7's
    documents, licence and notice files, and this revision's corrections (the release name and version string,
    how two documents name the licence file) were checked **offline only**, with tests, temporary directories and
    stand-ins.
  - **Never run:** a complete launch from this package; the complete check on real builds; anything on another
    Jetson.
- **Stock JetPack is unverified.** The reference device runs L4T R36.4.7 with an Ubuntu 24.04 userland. On stock
  JetPack 6 (Ubuntu 22.04) `preflight.sh` says NOT VERIFIED and `install.sh` goes on only if you pass
  `--accept-unverified-platform`. It may work; nobody has tried.
- **Device resource validation is pending.** The launch gates and the memory guard's limits come from the
  reference device. Passing them shows nothing about your device.

## What is in the archive, and what is not

| In it | Not in it: you supply or make these yourself |
|---|---|
| Scripts and helpers, the pinned manifests, `QUICKSTART.md` | Your configuration (the camera address is typed at launch and never saved) |
| The application source under `app/` | Your enrollment: photos of yourself, sealed with your own passphrase |
| `VERSION` (exactly what this package is) and `SHA256SUMS` | The model files (downloaded from upstream by `fetch-assets.sh`) |
| `LICENSE` and `THIRD_PARTY_NOTICES.md` | The llama.cpp build and the detector engine (built on your device), and the record of them |

No gallery, enrollment, credential, model, engine, device-artifacts record, recording or observation data of
anybody comes with it.

## The steps, and which of them touch a camera or face data

| Step | Command | Camera | Face data |
|---|---|---|---|
| Check the device | `bash preflight.sh` | no | none |
| Install | `bash install.sh` | no | none |
| Python environment, model downloads | `bash setup-env.sh`, `bash fetch-assets.sh` | no | none (two face *model* files are downloaded; they describe nobody) |
| The two builds, under a memory monitor | `bash guarded-build.sh llama --jobs 2`, `bash guarded-build.sh engine` | no | none |
| Your enrollment | `bash identity.sh enroll me --consent-confirmed --consent-date YYYY-MM-DD` | no | **yes: reads your photos and writes your sealed enrollment** |
| Record your files, then check | `bash record-artifacts.sh`, `bash check.sh --device-artifacts FILE` | no | none (the check only looks whether an enrollment exists) |
| Launch | `bash start.sh --device-artifacts FILE` | **yes, only after you type START** | **yes: faces in view are compared with your enrollment** |
| Stop | Ctrl-C once, or `bash stop.sh` | ends the run | none |
| Remove | `bash uninstall.sh` | no | none (your data directory is kept) |

Memory preparation before the engine build (closing a desktop, releasing the page cache) is your own manual step;
no script does it for you. The dashboard is reached through an SSH tunnel
(`ssh -N -o ExitOnForwardFailure=yes -L 18090:127.0.0.1:18090 <user>@<jetson>`) at
`http://127.0.0.1:18090/dashboard`. Details, recovery after an interrupted step and the operating limits are in
`QUICKSTART.md`.

## Licence

Sentinel's own files in this archive are under the MIT License: see `LICENSE`, the repository's own licence file,
at the top of the archive. The Python packages and
model files the scripts download, and the software you build, are **not** part of this archive and keep their own
licences, among them AGPL-3.0 (Ultralytics and the detector weights), the LFM Open License v1.0 (the scene model)
and NVIDIA's licence agreements. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists each with its source.

Face embeddings are biometric data. Enrol only yourself, and do not point a camera at people or places where
local privacy law does not allow it.

## Reporting a problem

Open an installation issue:
<https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/issues/new?template=installation-issue.md>

The template asks for the first line of `VERSION`, the first lines of `bash preflight.sh`, the step and the exact
`STOP:` line. **The issue tracker is public.** Do not post the camera address, passwords, the gallery passphrase,
the control key, photos, footage, your enrollment, or whole log files: `guard/run.err` in a run's folder can name
your camera's host.
