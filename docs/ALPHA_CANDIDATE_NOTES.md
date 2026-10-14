# Sentinel v2 — alpha-candidate development branch: notes

**This is a development branch, not a release.** Nothing here is tagged, merged or declared accepted. As of
2026-10-14 (session 68), the code is on branch `v2-ui-refresh`, which descends from `v2-beta`; it is published only if
and when the maintainer decides. `docs/IMPLEMENTATION_STATUS.md`
is the source of every result named here, by session; where the two differ, the status record governs.

Device: one Jetson Orin Nano (JetPack 6, L4T 36.4.7) with one camera, reached from a Mac over SSH. Portable tests run
on any machine without a camera, GPU or model; browser tests use headless Chromium when one is installed.

## What has been shown on the device (each within its own bounded scope)

| Feature | Evidence (status record) | Scope and limits |
|---|---|---|
| Core monitoring: capture, person detector and tracker, zone rules, incidents, alert outbox | Step 5, validated for its bounded test (session 50, D60), including one real Telegram send | A guarded run under the accepted demo memory profile (D59, 640×480 replay scope); no long-duration, accuracy or production acceptance |
| Incident detail page and acknowledgment while monitoring is stopped | Incident-page check passed (session 52) | Read-only loopback page over an SSH forward |
| Live view (the frames the runtime already reads) and incident snapshots | Saved-evidence checks passed (session 53); continuous delivery since D75 | Encoded only while viewed; bounded; Safari observed by the maintainer (user-reported) |
| The live dashboard with face recognition in a validation run (D95, D96) | The dashboard check `dash-20261013T181703Z` (session 68): execution and privacy passed; recognition described, not judged | Safari observed by the maintainer (user-reported); departure and return not exercised; **face recognition stays pending validation** |
| Advisory chat about the live frame, answered by the local scene model | Two passing device runs: `alpha-20261012T040450Z` (`f6ced5e`) and `alpha-20261013T084310Z` (`c3a37f0`), runtime and browser results | **Chat is not admitted.** Answers' accuracy is not evaluated (one unverified operator report of a wrong answer); never with face recognition running |

## Implemented and portable-tested, not validated on the device

- **Face recognition (V2-25 demo form): `PENDING_VALIDATION`.** It runs only in guarded validation runs. The formal
  validation (F2, with retention) has never passed. Descriptive device checks exist (sessions 56–62). The full-scale
  detection candidate (D88) is a validation-only option, not adopted. The camera-side enrollment experiment (D91):
  its capture was accepted through a corrected reading (session 66); the two genuine evaluation sessions (session 67)
  gave mixed results; sessions with non-enrolled participants, and so any false-acceptance data, are pending. The
  experimental working gallery is not adopted or merged.
- **The live dashboard (D95, D96, session 67):** recognition per tracked person (current match apart from a retained
  one, observation time, unknown never presented as a stranger, NOT CURRENT when updates stop), the sidebar (rail and
  phone drawer), the camera focus view, filters over the loaded incidents, theme, seven palettes and density saved in
  the browser, and `/dashboard` beside `/`. Headless Chromium 155 in the tests. The
  operator check `dash-face` ran on the device (session 68: execution and privacy passed, recognition described;
  Safari observed by the maintainer, user-reported).
- **Session 68:** the identity continuity rule (D97, PROPOSED, validation runs only: an identity is kept only while its track
  is followed without a break, up to 120 s, never on a track number alone); the Sentinel mark (D98); the Features panel
  (D99: face recognition, scene analysis, chat, the live view and snapshots paused and resumed at runtime with the
  run's own control key; core monitoring has no switch; nothing is loaded, admitted or restarted from the page); viewing
  preferences kept in the browser only (D100). Their two-session operator check `features` is prepared and has not run.
- Chat cancellation, the memory-pressure response and the scene-job grace (D77, D81); the combined face-and-chat
  reading (D80) and a grounding check (G, designed for the combined package, not implemented).

## Gates still open for the alpha

1. Face recognition: a passing F2 (the seated-away procedure is undecided), false-acceptance data from consenting
   non-enrolled participants, and the maintainer's decision on any detection or enrollment change. Daytime matching
   with the photo gallery was poor in four daytime sessions from 12 October and good in the daylight evaluation of 14
   October (204 of 207 faces; session 71), so it is not a general daytime failure (no implementation defect found);
   whether the dashboard's own detection loses matches on daylight frames is checked by `dash-path`, prepared and not run.
2. Chat: admission (operator-run resource and interference checks; the maintainer's decision).
3. Face and chat together: no combined run yet.
4. The feature controls on the device and in Safari (`features`, two sessions: the face session ran once with its switch
   items not exercised, session 69; the scene session pending), departure and return on the dashboard, and resource
   use with face and the live view together (the dashboard check ran; session 68).
5. The Review 1 rehearsal is incomplete (resumable with its own package).
6. Whole-device memory targets (5,000,000,000 B steady, 5,400,000,000 B peak) are acceptance targets. The two chat
   runs' readings passed their memory items (peaks 3,773,243,392 B and 3,721,973,760 B); the full alpha configuration
   (face, chat, scene and the live view together) has not been measured.
7. Not started for the alpha: a two-hour soak (V2-24), real API, sessions and screens (V2-17, V2-18), clips (V2-21).

## Data and privacy in this repository

- No credentials, tokens, keys, enrollment galleries, face images, embeddings, similarity values, photos, run evidence or
  the local-only planning documents are tracked; test fixtures use synthetic values and reserved documentation
  addresses. The publication scan (session 67) lists what was checked and how.
- The status record names device runs, packages and some machine identifiers (the device's user name, the USB tether's
  default address, boot IDs), as earlier published records already do; they hold no credential.
- Face recognition is context only: it never grants or denies anything, and no page, record or log shows a name.

## Running the portable checks

```
python -m pip install -e '.[dev]'
python -m pytest            # browser tests skip when no Chromium-family browser is found (SENTINEL_TEST_BROWSER)
sentinel config validate config/default.yaml
```

Known intermittent tests (pre-existing, timing-dependent): `test_alpha_check`'s load-tool timing test and
`test_face_runtime`'s shadow-run comparison of wall-clock `processing_ms`.
