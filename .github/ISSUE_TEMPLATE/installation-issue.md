---
name: Installation issue (tester package)
about: A problem installing, checking or starting Sentinel from the personal-beta tester package on a Jetson
title: "[install] "
labels: []
---

<!--
This tracker is PUBLIC. Leave out:
  - the camera's address, user name or password, and any URL that contains them
  - the gallery passphrase, the feature-control key, tokens
  - photos, footage, screenshots that show people, and your enrollment
  - whole log files or run folders (guard/run.err can name your camera's host)
Paste single lines, and write ~ for your home directory.
-->

**Package** (the first line of `VERSION`, from the extracted archive or `~/sentinel-beta/package/VERSION`):

**Platform** (the `ok`, `NOT VERIFIED` and `FAIL` lines of `bash preflight.sh`; say so if you used
`--accept-unverified-platform`):

**Step** (preflight / install / setup-env / fetch-assets / guarded-build llama / guarded-build engine / enrollment /
record-artifacts / check / launch, and at which prompt / dashboard / stop / uninstall):

**What you did, and what you expected:**

**What happened** (the exact `STOP:` line, or the last few lines printed):

**Only where it applies:**

- Guarded build: the `OUTCOME=` line and the memory lines of that run's `RESULT.txt`, and how you prepared memory.
- Launch: the `Observation ended`, `ended_by`, `runtime_after_guard`, `enrollment`, `route` and `OBS_exit` lines.
- Device-artifacts route: the lines `check.sh --device-artifacts FILE` printed (checksums and versions).

**Anything else running on the Jetson at the time:**
