# Sentinel v2.0.0-alpha.1 — Experimental Alpha

Git tag `v2.0.0-alpha.1` · Python package `sentinel-surveillance` `2.0.0a1` · 14 October 2026 · branch
`alpha-candidate-2026-10`

> **Note (added 2026-10-19, session 94; updated 2026-10-20, session 106).** These notes describe the pre-release as
> published at `b33c2f3` and are not revised. Later work on `v2-ui-refresh` (sessions 75–106: validation diagnostics,
> review fixes, a validation-only face + chat integration run, the grounding check's tooling, their device sessions and
> records) is unreleased and summarized in [`ALPHA_CANDIDATE_NOTES.md`](ALPHA_CANDIDATE_NOTES.md). No F2 has validated
> since, face recognition stays `PENDING_VALIDATION`, chat is not admitted, and the face-plus-chat alpha goal is not met.

**An experimental pre-release, for review and trial.** It is **not production-accepted**, and **feature admission is a
separate decision** that this pre-release does not make: face recognition stays `PENDING_VALIDATION`, chat is not
admitted, and no threshold, gallery or admission was changed for it. Nothing is merged into `master`, whose v1 code
and README are unchanged. Evidence for every claim is in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), by
session.

## What it contains (implemented and portable-tested)

- **Core monitoring:** RTSP capture with frame identity (stream epoch, sequence, ingest time) and explicit freshness
  states; YOLOv8n (TensorRT) person detection with ByteTrack tracking; zone and schedule rules; incidents in SQLite
  (one writer); a restart-safe alert outbox with Telegram delivery.
- **Incidents:** a read-only incident page with snapshots, and acknowledgment through `sentinel incident ack` while
  monitoring is stopped.
- **The dashboard** on 127.0.0.1:18090: the live picture, recognition per tracked person in validation runs, incidents
  with filters, light and dark themes with seven palettes, browser-only viewing preferences, and a NOT CURRENT state
  whenever updates stop.
- **Face recognition** (DeepFace, Facenet512, YuNet) against a sealed, consented gallery, in guarded validation runs
  only, with diagnostic options (a full-scale comparison, a full-scale detection candidate, a camera-side enrollment
  experiment) and a proposed identity continuity rule.
- **Advisory chat** about the live frame through the local scene model, bounded and refused unless admitted;
  **feature controls** that pause and resume the optional lanes with the run's own control key.
- **Operator check tooling** (`benchmarks/runner/`): a memory guard and predeclared readings of saved evidence.

## What has been checked on the device (each within a bounded scope)

One Jetson Orin Nano (JetPack 6, L4T 36.4.7) with one camera: core monitoring in a guarded run (with one real
Telegram send); the incident page and acknowledgment; the live view and snapshots; the dashboard with face recognition
in validation runs (execution and privacy passed, recognition described, not accepted); two chat runs whose runtime
and browser checks passed. Safari was observed by the operator; the browser tests use headless Chromium.

## Known limitations (all unresolved; none waived by this pre-release)

- **Intermittent face recognition failure, cause unresolved.** With the same matching code and photo enrollment, the
  enrolled person was matched on few compared faces in four daytime sessions of 12-14 October (36 of 185 in the
  13:15 dashboard session of 14 October) and on most in others (204 of 207 at 15:18, 171 of 176 at 17:05 the same day,
  and the evening and night sessions). No implementation defect has been established; lighting is not established as
  the cause.
- **Face acceptance pending.** The formal validation (F2) has never passed; no session with consenting non-enrolled
  participants has run, so there is no false-acceptance data.
- **The identity continuity rule and the feature controls are not fully validated on the device.** The rule ran in one
  session; the controls' face session ran with its switch items not exercised, and their scene session has not run.
- **Chat:** its earlier runtime checks passed; **admission, factual accuracy and combined face/VLM validation remain
  pending** (one unverified operator report of a wrong answer).
- **Face recognition with the scene model (VLM):** never run together.
- **Memory:** the whole-device targets (5,000,000,000 bytes steady, 5,400,000,000 bytes peak) are acceptance targets;
  the full configuration (face, chat, scene and the live view together) has not been measured.
- **Device setup:** no reproducible device installation guide; the device checks used a Python environment and a
  detector engine built for v1.
- **Not started for this alpha:** a two-hour soak, a real API with sessions and screens, clips, more cameras; the
  Review 1 rehearsal is incomplete. PIR/LiDAR and semantic-model experiments stay deferred.

## Since the branch was first published (`35f1a3a`)

- Three tests that depended on scheduling or wall-clock time are now deterministic, with their coverage kept (no bound
  widened, nothing skipped, no runtime change).
- The README is about v2; the v1 README moved, unchanged, to [`LEGACY_V1.md`](LEGACY_V1.md).
- The package version is `2.0.0a1`; this file; the wiki.

## Upgrading from v1

v2 is a separate package beside the unchanged v1 files. It does not read v1's `face_db.pkl` or `.env`: enrollment is
`sentinel identity enroll` with recorded consent, and credentials come from the environment (`SENTINEL_RTSP_URL`,
`SENTINEL_IDENTITY_PASSPHRASE`, the Telegram variables). See the README's getting-started section.
