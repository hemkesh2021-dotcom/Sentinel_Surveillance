# Sentinel v2 — EXPERIMENTAL ALPHA (v2.0.0-alpha.1 and the unreleased work after it): notes

**EXPERIMENTAL ALPHA. v2.0.0-alpha.1 is an experimental pre-release, not a production release.** Production acceptance
and feature admission are separate decisions, and this pre-release makes neither: nothing is declared accepted, face
recognition stays `PENDING_VALIDATION`, chat is not admitted, and nothing is merged into `master` or `v2-beta`. The
branch `alpha-candidate-2026-10` (published 2026-10-14 from the local `v2-ui-refresh`, which descends from `v2-beta`)
carries the pre-release; the tag is `v2.0.0-alpha.1` and the package version `2.0.0a1` (release notes:
[`RELEASE_NOTES_v2.0.0-alpha.1.md`](RELEASE_NOTES_v2.0.0-alpha.1.md)). `docs/IMPLEMENTATION_STATUS.md` is the source of
every result named here, by session; where the two differ, the status record governs.

**Since v2.0.0-alpha.1 (sessions 75–106; these notes updated 2026-10-20, session 106).**
- **Where the work is:** the local `v2-ui-refresh`, more than 30 commits past the tag. The code last changed at
  `2004725` (session 106: the grounding check G's operator tooling), after `aaca744` (session 103: the integration
  run's readiness and reading) and review scope A's fixes (sessions 96–101); later commits are records and documents.
- **Not published or released:** nothing pushed after `65df7f3` (14 October, a README change after the tag); the
  package version is still `2.0.0a1`; no new pre-release decided.
- **Unchanged since the tag:** face recognition stays `PENDING_VALIDATION`; chat is not admitted; no threshold,
  gallery, identity rule, admission or memory-guard limit has changed. The review fixes of sessions 96–101 corrected
  defects: redaction, gallery and audit consistency, purge, enrollment errors, and face votes counted only within a
  track's current life and recognition's current activation, as the identity rules already stated (session 97). None
  changed a threshold, a criterion or admission.

**The alpha's goal is not met.** On 12 October the maintainer set the October 20 alpha's centre as working face
recognition and real local chat with the vision-language model, as one plan. Neither part is done:
- no F2 has validated: since the tag, three attempts on the normal recognition path, one stopped by the memory guard
  and two not validated;
- there is no false-acceptance data;
- chat is not admitted;
- face recognition and the VLM have run together only once, in a validation-only integration run
  (`fci-20261020T110028Z`, 20 October, session 104). It passed, but it is not the combined run, which still needs a
  validated F2. It judged no answer's correctness and no recognition accuracy, and it admits nothing.

Under the recorded readiness rule, face is shown only as the experimental demonstration until an F2 actually validates.

**Readiness of the unreleased candidate (session 93): not ready to call the alpha complete.** No release-blocking code
defect is established: none is recorded open. Review scope A (identity) and a follow-up review of its fixes ran
(sessions 96–101: the findings fixed, two deferred with their reasons); the other proposed scopes have not run. The
blocking items:
1. the release documents, which this revision brings up to session 106;
2. no CI run on the unpushed commits (the last recorded CI result is at `b33c2f3`). The portable suite passed locally
   at `2004725` under Python 3.10.14 and 3.12.3 (session 106): local validation on the device, not CI;
3. the alpha goal above, a decision gate rather than a defect;
4. the full alpha configuration's memory, still unmeasured (the integration run had no live view or snapshots).

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
  - **Recognition in ordinary seated use is intermittent** (sessions 89 and 91; no code defect established). Looking
    at the Mac, with natural head turns and chair movement, KNOWN is lost in two ways:
    - when one non-matching vote follows matches: the documented two-vote rule, with the enrolled person's similarity
      running close to the threshold;
    - when production detection finds no face for longer than the 30 s vote lifetime allows.

    In the ten-minute ordinary-use diagnostic `lmk-20261019T105756Z`:
    - the identity was KNOWN 363 of 600 s;
    - production detection found the face on 22 % of results;
    - 232 of the 237 s not KNOWN followed vote expiry during such stretches;
    - a native-scale detection pass found a face on 258 of the 450 results without a production face; whether those
      faces would match is not recorded.
  - **Face acceptance and the non-enrolled-person checks, pending.** The formal validation (F2) has never passed. On
    the normal recognition path:
    - 15 October, under lamps (session 78): not validated. Retention while the face was hidden was not exercised,
      though recognition while facing the camera and clearing across both absences passed in that one session.
    - 18 October (session 86): stopped by the memory guard 54.6 s after T0, before any cue.
    - 18 October, after a one-time page-cache drop (session 88): completed but not validated. The locked account
      answered `unknown` to whether other faces were in view, and retention was again not exercised.

    No session with consenting non-enrolled participants has run, so there is no false-acceptance data. Face
    recognition stays `PENDING_VALIDATION` and runs only in guarded validation runs.
  - **The proposed identity continuity rule (D97) and the feature controls (D99), not fully validated on the device.**
    The rule ran in one device session without changing any vote; the controls' face session ran with its switch
    items not exercised, and their scene session has not run.
  - **Chat: its runtime checks passed (two chat device runs, and one validation-only integration run with face
    recognition); admission, factual accuracy and combined face/VLM validation remain pending.** No run has kept or
    read an answer's text, and there is one unverified operator report of a wrong answer. The grounding check G (four
    fixed questions judged by the maintainer against a staged room) is implemented and packaged, not run (session
    106). Chat is not admitted.
  - **Face recognition and the scene model (VLM) together: unvalidated.** One validation-only integration run passed
    its execution, chat-completion, face-execution and memory items (session 104); no combined run has taken place.

## What has been shown on the device (each within its own bounded scope)

| Feature | Evidence (status record) | Scope and limits |
|---|---|---|
| Core monitoring: capture, person detector and tracker, zone rules, incidents, alert outbox | Step 5, validated for its bounded test (session 50, D60), including one real Telegram send | A guarded run under the accepted demo memory profile (D59, 640×480 replay scope); no long-duration, accuracy or production acceptance |
| Incident detail page and acknowledgment while monitoring is stopped | Incident-page check passed (session 52) | Read-only loopback page over an SSH forward |
| Live view (the frames the runtime already reads) and incident snapshots | Saved-evidence checks passed (session 53); continuous delivery since D75 | Encoded only while viewed; bounded; Safari observed by the maintainer (user-reported) |
| The live dashboard with face recognition in a validation run (D95, D96) | The dashboard check `dash-20261013T181703Z` (session 68): execution and privacy passed; recognition described, not judged | Safari observed by the maintainer (user-reported); departure and return not exercised; **face recognition stays pending validation** |
| The dashboard's detection path, compared on the same frames (production detection with the live view) | `dpath2-20261014T113113Z` (session 73): execution and privacy passed; 171 of 176 compared faces matched, outcome D1 for that session's late-afternoon light (launched 17:04 IST, reported apart) | Investigative, not acceptance; one control pair and three full-scale-only faces compared, too few to compare the detection paths; the live view sent no picture for at least 90 s of the run; **face recognition stays pending validation** |
| F2 on the normal recognition path (production detection, the original enrollment, D97 off) | `f2n-20261015T125427Z` (session 78): **not validated**. Recognition while seated facing the camera (K1, K2) and clearing across both absences (E1, E2) passed; retention with the face hidden (A) NOT_EXERCISED | One session, under lamps; not reliable recognition and not validated retention; **face recognition stays pending validation** |
| Ten-minute tracking diagnostic (validation only, never an F2 attempt) | `trk-20261017T103831Z` (session 84): execution complete; one tracker number on the seated person through a face-hidden hold; the identity retained by its 30 s rule, then unresolved | One session, daylight; descriptive; a 1.9 s input stall cleared the identity once (re-earned in 3 s) |
| F2 on the normal path after a one-time page-cache drop (D103, a changed test condition, reported apart) | `f2c-20261018T155424Z` (session 88): completed, **not validated** (the account's `other_faces_in_view` answer `unknown`; A NOT_EXERCISED: one non-matching vote just after the 4:00 cue ended KNOWN). The attempt before it, `f2n-20261018T104038Z` (session 86), was stopped by the memory guard before any cue | One session, under lamps; K1 and K2 passed but unstable (7 of 62 and 6 of 58 votes did not match); **face recognition stays pending validation** |
| Landmark diagnostic in ordinary use (validation only, never an F2 attempt) | `lmk-20261019T105756Z` (session 91): execution complete; its predeclared outcome **inconclusive** (39 usable evaluation votes against 100 required, no misses among them, the reference band unstable); kept as the result, never extended or repeated | One session, daylight, after a one-time page-cache drop (reported apart); landmark numbers are proxies, never angles; the recognition figures above come from its identity records |
| Advisory chat about the live frame, answered by the local scene model | Two passing device runs: `alpha-20261012T040450Z` (`f6ced5e`) and `alpha-20261013T084310Z` (`c3a37f0`), runtime and browser results | **Chat is not admitted.** Answers' accuracy is not evaluated (one unverified operator report of a wrong answer); with face recognition running only in the integration run below |
| Face recognition and chat with the VLM together (validation only; never F2 or the combined run) | `fci-20261020T110028Z` (session 104): all four parts passed. Execution; chat completion (75 answers to the load tool's two fixed questions, 4 ended by scene jobs after the 1.5 s grace, the largest scene wait 1,502.5 ms); face executed alongside (442 results applied, none failed); memory (peak pressure 4,203,548,672 B, lowest MemFree 2,443,522,048 B) | Eight minutes, one session, daylight, after a one-time page-cache drop (reported apart); no live view or snapshots; answers judged for form only (their text never kept); recognition described, not judged; **face recognition stays pending validation and chat is not admitted** |

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
  reading (D80) and the grounding check G (session 106: implemented for the validation-only integration run, the
  answers judged by the maintainer on the device and never read by Claude; packaged as `fcg-2004725`, not run).
- **Validation-only diagnostics and operator tooling since the tag:**
  - the tracking diagnostic (`--tracking-diagnostic`, sessions 79–81);
  - F2's corrected retention reading (session 85);
  - the operator's launch-memory check (`benchmarks/runner/launch_memory.py`, D103, session 87): a one-time page-cache
    drop and an empirical launch floor with no reserve;
  - the landmark diagnostic (`--face-landmark-diagnostic`, session 90): derived landmark proxies of the native-scale
    face, associated one to one with the vote's own frame and person, read with a predeclared, association-only
    analysis;
  - the face + chat integration run's readiness (`--require-ready face_chat`) and reading (`alpha_check.py
    integration`, session 103), which reports execution, chat completion, recognition behaviour and memory apart;
  - the grounding check G (`alpha_load chat --grounding`, `benchmarks/runner/grounding.py`, session 106): four fixed
    questions once each, their answers' text kept only in a private sealed file and shown only in the maintainer's
    local judging; only categories and labels (contradicted by or consistent with the staging, unverifiable,
    uncertainty) reach the reading, descriptively.

  None changes production detection, votes, thresholds, enrollment or admission.
- **PROPOSED, not implemented (session 92):** a complete-coverage variant of the existing shadow comparison (D87).
  Its pinned budget was shown, on sealed timelines, unable to establish whether voting native-only faces would help;
  the variant would compare every such face and record its outcome, never as a vote.

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
   Session 84: that ten-minute diagnostic ran once and completed (`trk-20261017T103831Z`, daylight; a diagnostic, not
   an F2 attempt; the lowest MemFree 176 MB above the guard's floor). In this one session the tracker kept one number
   on the seated person through the face-hidden hold; the identity, KNOWN at the cue, was retained by its 30 s rule and
   then unresolved (no face was found during the hold); departure cleared it about 1 s after the last detection; a
   1.9 s stretch in which no frame reached the runtime cleared it once while the person sat (re-earned in 3 s; where
   the frames were held is unknown). F2's retention reading was found to read FAIL on real timing even when the
   runtime follows its rule (a reading defect; a correction is proposed, not made). No F2 has passed, so face remains
   the experimental demonstration.
   Session 85: that reading is corrected (`d754976`: the clear and the retained basis timed from the votes the runtime
   actually held; the criterion, tolerances and every other F2 item unchanged; no earlier F2 verdict changes), and F2 is
   re-pinned with the host preparation (`f2-normal-d754976`, dry-run, not yet run on the device). Face remains the
   experimental demonstration until an F2 actually validates.

   **Sessions 86–92:**
   - **Session 86:** that package's session was stopped by the memory guard before any cue.
   - **Session 87:** F2 re-packaged with an operator-controlled one-time page-cache drop and a launch-memory check
     (D103).
   - **Session 88:** the resulting session completed but did not validate (the account's `unknown` answer about
     other faces in view; A not exercised).
   - **Session 89:** a read-only investigation across 13 sealed sessions. Its strongest supported explanation for
     intermittent KNOWN loss in ordinary use: score variability around a thin margin to the 7-photo gallery, turned
     into exits by the single-vote contradiction rule, with head turn and drift contributing. Production's half-scale
     detection also misses many faces a native-scale pass finds. No code defect established.
   - **Session 90:** the landmark diagnostic.
   - **Session 91:** its ten-minute ordinary-use session (inconclusive by its predeclared rule; the identity figures
     above).
   - **Session 92:** whether native-scale detection would help is not established. The existing shadow comparison
     cannot settle it, and a complete-coverage variant is proposed, not implemented.

   No detection, threshold, identity rule, enrollment or admission changed. Face remains the experimental
   demonstration until an F2 actually validates.
2. Chat: its runtime checks passed; admission (operator-run resource and interference checks; the maintainer's
   decision) and factual accuracy remain pending. The integration run judged answers' form only; no run has kept an
   answer's text. The grounding check G is implemented and packaged, not run; it would judge four correlated answers,
   never a rate (session 106).
3. Face and chat (the VLM) together: one validation-only integration run passed (`fci-20261020T110028Z`, session 104:
   eight minutes, after a one-time page-cache drop, no live view). The combined run still needs a validated F2.
   Untested: answer content, the dashboard's presentation of answers with face running, and the full alpha
   configuration.
4. The feature controls on the device and in Safari (`features`, two sessions: the face session ran once with its switch
   items not exercised, session 69; the scene session pending), departure and return on the dashboard, and resource
   use with face and the live view together (the dashboard check ran; session 68).
5. The Review 1 rehearsal is incomplete (resumable with its own package).
6. Whole-device memory targets (5,000,000,000 B steady, 5,400,000,000 B peak) are acceptance targets. The two chat
   runs' readings passed their memory items (peaks 3,773,243,392 B and 3,721,973,760 B); the full alpha configuration
   (face, chat, scene and the live view together) has not been measured.

   Since the tag:
   - **Guard stops:** the memory guard stopped three face runs with the scene model loaded (18.7 s, 3.0 s and 54.6 s
     after T0). Lower launch headroom is the strongest supported explanation; these were policy stops at its 1 GiB
     floor, not allocation failures.
   - **The launch-memory check:** with the one-time page-cache drop and the launch check (D103), two runs completed.
     Their lowest MemFree was 2,480,082,944 B and 2,283,102,208 B; their peak pressure 4,217,470,976 B and
     4,389,052,416 B (face, the scene model loaded, no chat or live view).
   - **The floor's reference exceeded:** in the second run, one unattributed 11 s transient took consumption from
     launch 221 MB past the launch floor's reference run. The floor carries no reserve.
   - **Face, chat and the scene model together (session 104):** after the one-time drop and the launch check, one
     eight-minute integration run had peak pressure 4,203,548,672 B and lowest MemFree 2,443,522,048 B, within the
     limits; it had no live view or snapshots, so the full alpha configuration is still unmeasured.
8. Release readiness (session 93):
   - **CI:** no CI run on the unreleased commits. Local validation only: the portable suite passed at `2004725` under
     Python 3.10.14 and 3.12.3 on the device (2,778 passed, 2 skipped; session 106), which does not stand in for CI.
   - **Review:** scope A (identity) of the six proposed scopes ran, with a follow-up review of its fixes (sessions
     96–101: the findings fixed, two deferred with their reasons); scopes B–F have not run.
   - **Release documents:** brought up to session 106 in this revision.
7. Not started for the alpha: a two-hour soak (V2-24), real API, sessions and screens (V2-17, V2-18), clips (V2-21).

## Data and privacy in this repository

- No credentials, tokens, keys, enrollment galleries, face images, embeddings, similarity values, photos, run evidence or
  the local-only planning documents are tracked. The publication scans (sessions 67 and 73) list what was checked and
  how; session 93 scanned every file version reachable from the unreleased branch (counts only).
- **Test fixtures:** synthetic, using reserved documentation addresses and, in refusal and redaction tests, synthetic
  private-range addresses. The exceptions are three fixtures derived from sealed device runs (sessions 81, 85 and 87):
  they hold memory figures, times and fixed labels only, with no identity ID, similarity, box or image.
- The status record names device runs, packages and some machine identifiers (the device's user name, the USB tether's
  default address, boot IDs), as earlier published records already do; they hold no credential.
- Face recognition is context only: it never grants or denies anything, and no page, record or log shows a name.

## Running the portable checks

```
python -m pip install -e '.[dev]'
python -m pytest            # browser tests skip when no Chromium-family browser is found (SENTINEL_TEST_BROWSER)
sentinel config validate config/default.yaml
```

**Latest recorded results:**
- the full portable suite at `2004725` (the unreleased code): 2,778 passed, 0 failed and 2 skipped (`torch`, `cv2`
  absent), under the device's Python 3.10.14 `.venv` and then the isolated Python 3.12.3 environment, with
  `config/default.yaml` valid under both (session 106; local, not CI);
- the isolated 3.12.3 environment was set up in session 95 following `.github/workflows/v2-portable.yml` (the editable
  install with the pinned dependencies, warnings as errors; 2,578 passed at `df8d9bc` then);
- the last recorded CI result: at `b33c2f3`, on Python 3.10 and 3.12;
- not covered since:
  - CI itself on either Python (GitHub's x86_64 runners; earlier 3.12-only CI failures were timing-dependent tests);
  - the two skipped tests;
  - browsers other than headless Chromium.

The two tests that used to fail intermittently (`test_alpha_check`'s load-tool window, anchored on a question the page
could refuse, and `test_face_runtime`'s shadow-run comparison, which included wall-clock `processing_ms`) are
deterministic since v2.0.0-alpha.1, with their coverage kept, and so is `test_live_player_browser`'s slow-viewer test
(its wall-clock lag bound, which failed on the Jetson and in CI, replaced by checks in the player's own protocol).
