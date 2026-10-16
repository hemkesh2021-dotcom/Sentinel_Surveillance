# Sentinel v2 — EXPERIMENTAL ALPHA (v2.0.0-alpha.1): notes

**EXPERIMENTAL ALPHA. v2.0.0-alpha.1 is an experimental pre-release, not a production release.** Production acceptance
and feature admission are separate decisions, and this pre-release makes neither: nothing is declared accepted, face
recognition stays `PENDING_VALIDATION`, chat is not admitted, and nothing is merged into `master` or `v2-beta`. The
branch `alpha-candidate-2026-10` (published 2026-10-14 from the local `v2-ui-refresh`, which descends from `v2-beta`)
carries the pre-release; the tag is `v2.0.0-alpha.1` and the package version `2.0.0a1` (release notes:
[`RELEASE_NOTES_v2.0.0-alpha.1.md`](RELEASE_NOTES_v2.0.0-alpha.1.md)). `docs/IMPLEMENTATION_STATUS.md` is the source of
every result named here, by session; where the two differ, the status record governs.

Device: one Jetson Orin Nano (JetPack 6, L4T 36.4.7) with one camera, reached from a Mac over SSH. Portable tests run
on any machine without a camera, GPU or model; browser tests use headless Chromium when one is installed.

## Status at a glance

- **Implemented and portable-tested:** everything named below. Portable tests show the code does what its tests state;
  they show nothing about the camera, the GPU, memory, accuracy or a browser other than headless Chromium.
- **Tested on the device:** only the checks in the next table, each within its own bounded scope. A device check that
  passed its execution and privacy items describes recognition; it does not accept it.
- **Pending, and known to be unresolved:**
  - **Intermittent face recognition failure, unresolved.** With the same matching code and photo enrollment, the enrolled person was
    matched on few compared faces in four daytime sessions of 12-14 October (the dashboard session at 13:15 on
    14 October: 36 of 185) and on most in others (204 of 207 at 15:18 and 171 of 176 at 17:05 the same day, and in the
    evening and night sessions). No implementation defect has been established and the cause is not known; lighting
    is not established as the cause. The detection path differed between these sessions: the 13:15 and 17:05 figures
    come from production detection, the normal recognition path the dashboard runs; the 15:18 figure comes from the
    full-scale detection candidate (a validation-only option, not adopted), so it is not evidence for the normal path
    (session 77).
  - **Face acceptance and the non-enrolled-person checks, pending.** The formal validation (F2) has never passed. Its
    attempt on the normal recognition path (15 October, under lamps) did not validate: retention while the face was
    hidden was not exercised, though recognition while facing the camera and clearing across both absences passed in
    that one session (session 78). No session with consenting non-enrolled participants has run, so there is no
    false-acceptance data. Face
    recognition stays `PENDING_VALIDATION` and runs only in guarded validation runs.
  - **The proposed identity continuity rule (D97) and the feature controls (D99), not fully validated on the device.**
    The rule ran in one device session without changing any vote; the controls' face session ran with its switch
    items not exercised, and their scene session has not run.
  - **Chat: its earlier runtime checks passed (two device runs); admission, factual accuracy and combined face/VLM
    validation remain pending** (one unverified operator report of a wrong answer). Chat is not admitted.
  - **Face recognition and the scene model (VLM) together: unvalidated.** No combined run has taken place.

## What has been shown on the device (each within its own bounded scope)

| Feature | Evidence (status record) | Scope and limits |
|---|---|---|
| Core monitoring: capture, person detector and tracker, zone rules, incidents, alert outbox | Step 5, validated for its bounded test (session 50, D60), including one real Telegram send | A guarded run under the accepted demo memory profile (D59, 640×480 replay scope); no long-duration, accuracy or production acceptance |
| Incident detail page and acknowledgment while monitoring is stopped | Incident-page check passed (session 52) | Read-only loopback page over an SSH forward |
| Live view (the frames the runtime already reads) and incident snapshots | Saved-evidence checks passed (session 53); continuous delivery since D75 | Encoded only while viewed; bounded; Safari observed by the maintainer (user-reported) |
| The live dashboard with face recognition in a validation run (D95, D96) | The dashboard check `dash-20261013T181703Z` (session 68): execution and privacy passed; recognition described, not judged | Safari observed by the maintainer (user-reported); departure and return not exercised; **face recognition stays pending validation** |
| The dashboard's detection path, compared on the same frames (production detection with the live view) | `dpath2-20261014T113113Z` (session 73): execution and privacy passed; 171 of 176 compared faces matched, outcome D1 for that session's late-afternoon light (launched 17:04 IST, reported apart) | Investigative, not acceptance; one control pair and three full-scale-only faces compared, too few to compare the detection paths; the live view sent no picture for at least 90 s of the run; **face recognition stays pending validation** |
| F2 on the normal recognition path (production detection, the original enrollment, D97 off) | `f2n-20261015T125427Z` (session 78): **not validated**. Recognition while seated facing the camera (K1, K2) and clearing across both absences (E1, E2) passed; retention with the face hidden (A) NOT_EXERCISED | One session, under lamps; not reliable recognition and not validated retention; **face recognition stays pending validation** |
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
  preferences kept in the browser only (D100). Their two-session operator check `features`: the face session ran with
  its switch items not exercised (session 69); the scene session has not run.
- Chat cancellation, the memory-pressure response and the scene-job grace (D77, D81); the combined face-and-chat
  reading (D80) and a grounding check (G, designed for the combined package, not implemented).

## Gates still open for the alpha

1. Face recognition: a passing F2 (its seated-away posture decided in session 77, below), false-acceptance data from
   consenting non-enrolled participants, and the maintainer's decision on any detection or enrollment change. Daytime
   matching with the photo gallery was poor in four daytime sessions from 12 October and good in the daylight
   evaluation of 14 October (204 of 207 faces on the full-scale detection candidate, not the normal path; session 71),
   so it is not a general daytime failure (no implementation defect established). Session 77 prepared F2 on the normal
   recognition path (production detection, D97 off; its still away posture, the face hidden behind a book or sheet
   with the body still, confirmed by the maintainer) and the non-enrolled sessions on the same path (the earlier plan
   measured the candidate); the combined run launches only after a validated F2. A passing F2 shows retention with the
   face hidden and the body still, not through a chair swivel; the non-enrolled sessions give per-person counts, never
   a false-acceptance rate. The dashboard's own detection matched 171 of 176 compared faces in late-afternoon daylight
   (`dpath2-20261014T113113Z`, 17:04 IST, outcome D1 for that light only; session 73), too few paired comparisons to
   judge another detection path, so no detection change. The failure seen at 13:15 on 14 October is unexplained.
   Session 78: F2 on the normal path ran once (`f2n-20261015T125427Z`, under lamps) and did not validate. A was not
   exercised: a non-matching face at the 4:00 cue ended the KNOWN identity, and the person's track changed twice while
   the face was hidden. A bounded rule for isolated non-matching faces is proposed for review, to be judged with the
   non-enrolled sessions' votes first; it is not implemented, and it would not have changed this F2.
   Session 79: an opt-in tracking diagnostic (`--tracking-diagnostic`, validation runs only) records per frame what the
   tracker was handed, published and lost, its own association costs and a shadow pass below the 0.4 cut, so a track
   break during the occlusion can be read from evidence; its package `tracking-diag-4e3c46c` is dry-run and never an
   F2 attempt; nothing changes with the flag off; hardware validation and its overhead are pending.
   Session 80: its one device session ran its cues against no detector (the load check expected the capture callback
   first in Ultralytics' callback list, where 8.4.25 keeps its own no-op); fixed at `732d82d`, and the diagnostic's
   readiness is now a prerequisite for T0 (the runtime refuses to start and the guard prints no cue without it); the
   package is regenerated as `tracking-diag-732d82d`; hardware validation and its overhead are still pending.
   Session 81: that package's device session met readiness, then the memory guard stopped it 18.7 s after T0, and its
   shadow pass failed on every frame (an inference-mode defect, fixed at `c530ff2` and tested on the installed
   libraries on the CPU); a launch block's success now comes from the guard's result, and a run with no completed shadow
   pass fails its reading. A short integration check (`tracking-integration-c530ff2`, at most 2 minutes) comes
   before any ten-minute session; hardware validation and the diagnostic's overhead are still pending.
   Session 82: that check ran once; the memory guard stopped it 3.0 s after T0 (MemFree under its floor) while the
   shadow pass completed on its 10 recorded frames. Same-boot observations showed far less anonymous memory resident
   once the development processes and the VNC desktop session were closed; a preparation supplement that closes them
   on request before launch (limits and criteria unchanged) is dry-run and pending.
   Session 83: with that preparation the check completed (short-run integration only: two minutes, seated; the lowest
   MemFree 503 MB above the guard's floor; every recorded frame's shadow pass completed). The ten-minute diagnostic
   package (`tracking-diag-c530ff2`, the same code, the preparation built in) is dry-run and pending; ten-minute
   stability, tracking behaviour and recognition are not established by the short check.
2. Chat: its earlier runtime checks passed; admission (operator-run resource and interference checks; the maintainer's
   decision) and factual accuracy (the grounding check G is designed, not implemented) remain pending.
3. Face and chat (the VLM) together: no combined run yet.
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
  addresses. The publication scans (sessions 67 and 73) list what was checked and how.
- The status record names device runs, packages and some machine identifiers (the device's user name, the USB tether's
  default address, boot IDs), as earlier published records already do; they hold no credential.
- Face recognition is context only: it never grants or denies anything, and no page, record or log shows a name.

## Running the portable checks

```
python -m pip install -e '.[dev]'
python -m pytest            # browser tests skip when no Chromium-family browser is found (SENTINEL_TEST_BROWSER)
sentinel config validate config/default.yaml
```

The two tests that used to fail intermittently (`test_alpha_check`'s load-tool window, anchored on a question the page
could refuse, and `test_face_runtime`'s shadow-run comparison, which included wall-clock `processing_ms`) are
deterministic since v2.0.0-alpha.1, with their coverage kept, and so is `test_live_player_browser`'s slow-viewer test
(its wall-clock lag bound, which failed on the Jetson and in CI, replaced by checks in the player's own protocol).
