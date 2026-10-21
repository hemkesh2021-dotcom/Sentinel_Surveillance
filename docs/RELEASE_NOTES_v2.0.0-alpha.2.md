# Sentinel v2.0.0-alpha.2 — Experimental Alpha (review checkpoint)

Git tag `v2.0.0-alpha.2` · Python package `sentinel-surveillance` `2.0.0a2` · October 2026 · branch
`alpha-candidate-2026-10`

**An experimental pre-release for the 24 October review checkpoint.** Like alpha.1, it is **not
production-accepted**, and it makes **no admission decision**: face recognition stays `PENDING_VALIDATION`, chat is
not admitted, and no threshold, gallery, identity rule, admission or memory-guard limit was changed for it. Nothing is
merged into `master`. Evidence for every claim is in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), by
session; the alpha.1 notes ([`RELEASE_NOTES_v2.0.0-alpha.1.md`](RELEASE_NOTES_v2.0.0-alpha.1.md)) describe the earlier
pre-release unchanged.

## What is new since alpha.1 (`b33c2f3`, 14 October)

- **Face recognition and local VLM chat run together in guarded validation runs.** A `face_chat` readiness before
  T0, and readings that report execution, the live view, chat, recognition behaviour and memory separately
  (`alpha_check.py integration` and `trial`).
- **On the device, validation only** (one Jetson Orin Nano, one camera; each a single session after a one-time
  page-cache drop, reported apart):
  - **two eight-minute face + chat runs** (20 October) passed their four parts. The second asked four grounding
    questions, judged by the maintainer against the staged room: two consistent, one unverifiable, one cancelled and
    untested;
  - **a one-hour live-dashboard trial** (`dtr-20261021T065058Z`, 21 October, daylight) passed all five parts.
    Face recognition, scene analysis, the live view and the operator's own chat questions ran together for 3,600 s
    from readiness:
    - **live view:** 64,808 pictures sent to the dashboard;
    - **chat:** 8 questions accepted, 5 answered, 3 ended by periodic scene jobs after the 1.5 s grace;
    - **face:** 2,911 results applied, none failed;
    - **memory:** peak pressure 4,370,120,704 B and lowest MemFree 2,200,756,224 B, within the 5.0/5.4 GB targets.
      This is the first hour-long measurement of face, chat, scene and the live view together (no snapshots).
  - **The operator's report of the trial (user-reported, not measured):** smooth video, reliable recognition at the
    tested distance, and generally useful chat.
- **Review fixes** (review scope A, identity). Redaction of credentials in errors, gallery and audit consistency,
  purge, enrollment errors, and face votes counted only within a track's current life. These corrected defects; no
  threshold or criterion changed.
- **Validation-only tooling,** none of it changing production behaviour: a tracking diagnostic and a landmark
  diagnostic, a launch-memory check with a one-time page-cache drop for operator packages, F2's corrected retention
  reading, the grounding check, and the memory guard's one-hour bound.
- **Version** `2.0.0a2`.

## Still pending (unchanged by this pre-release)

- **Face:** F2 has never validated (three attempts on the normal path since alpha.1, none validated). There is no
  false-acceptance data, and recognition accuracy is not assessed. In ordinary seated use recognition has been
  intermittent.
- **Chat:** not admitted; answer correctness is not assessed (no answer's text was kept, apart from the four
  maintainer-judged grounding answers).
- **The combined face + chat validation** still needs a passing F2. The trial and the integration runs are not it.
- **Deferred until after the review:** recognition-distance and VLM chat improvements.
- **From alpha.1, still open:** no reproducible device installation guide, no two-hour soak, no real API or clips, the
  Review 1 rehearsal incomplete, other browsers than Safari and headless Chromium.

## Checks

This tag is placed only on a commit whose GitHub CI passed: Dashboard checks, and v2 portable checks on Python 3.10
and 3.12. Locally on the device, not CI: the full portable suite at `2004725` (2,778 passed, 2 skipped, under Python
3.10.14 and 3.12.3), and focused suites at the two later operator-tooling commits, `1231d7e` and `4e1d4f5`. After
`4e1d4f5` the code differs only in its version string.
