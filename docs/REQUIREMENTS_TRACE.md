# Requirements-to-evidence trace (Review 1)

As of 2026-10-09, code at `283c071`. Session 52 updated two rows (D-2, and "stale video never says Live") for the incident-page device check and for D69/D70 at `3fbc50f`; the counts in section 5 are unchanged, because D-2 is not a backlog package.

**Sources:**
- the v2 beta implementation guide, revised 23 September 2026 (a local file, not in this repository): chapter 20's backlog and its acceptance column, chapter 2's regressions, chapter 21's mandatory scenarios and chapter 22's release gates;
- `docs/IMPLEMENTATION_STATUS.md`, the source of every device result below, with its session numbers;
- the portable test suite (1596 tests passing at `283c071`).

Acceptance texts are paraphrased. Where this trace and the status record differ, the status record governs and this file should be corrected.

## Evidence classes

Each requirement gets one class, the highest that its **whole** text supports:

| Class | Meaning |
|---|---|
| **Pending** | Nothing implemented toward it |
| **Partially complete** | Some parts exist; others are missing. Most demo-form work (D23) is here. "(device)" means some part was verified on the device, "(portable)" that no part was |
| **Implemented** | Code for the whole requirement exists, without a behavioural test of it |
| **Portable-tested** | Passing portable tests (fakes and synthetic replays; no camera or GPU) cover the whole requirement as written |
| **Device-verified** | A recorded device run met the whole requirement within its stated, bounded scope. Device results are USER-SUPPLIED and were reviewed in the status record |

**Full acceptance** is a separate column. It is "met" only when the guide's acceptance holds in its full form. Under D23, a demo-form package counts as partial at most, whatever its tests or device runs. A count of tests is never a class.

## 1. Backlog (guide chapter 20): 56 packages, 100 planned person-days

C1–C6 (29 packages, 53 days) are the guide's alpha span; V2-24, the alpha soak report, closes it. ✱ marks the 14 backlog packages on the Oct 20 demo path (29 days).

| ID | Cycle / days | Acceptance (paraphrased) | Class | Evidence | Missing for full acceptance | Full |
|---|---|---|---|---|---|---|
| V2-01 ✱ | C1 / 2 | Sanitized inventory; B0/B1 workload and trace contract; provisional CPU budget; effort re-estimate | Partially complete (device) | Inventory checks 1, 3, 5, 6 (session 3); check 8, Check 9 smokes, S1 (sessions 5–13); memory diagnostics and the accepted bounded 640×480 replay profile (sessions 40–47, D59) | The B0 run, the trace contract, the CPU budget, the re-estimate, checks 2 and 9b/U18 | no |
| V2-02 ✱ | C1 / 2 | Invalid config fails clearly; epoch/TTL tests pass | Portable-tested | `test_config.py`, `test_frame_identity.py`, `test_evidence_applicability.py`, `test_clock.py` | none (portable acceptance) | met |
| V2-03 ✱ | C1 / 2 | Multi-person and zero-person regressions recorded | Portable-tested | `tests/replay/test_r1_face_identity.py`, `test_r2_zero_person_scenes.py`, `test_r3_stale_scene_results.py` (synthetic replays) | Real-clip replays wait for V2-07 | met (synthetic) |
| V2-04 | C1 / 1 | A clean laptop runs the non-GPU checks | Partially complete (portable) | CI passed at `32985c2` (the maintainer's report); `test_portable_imports.py` | A clean-laptop (macOS) run | no |
| V2-05 ✱ | C2 / 3 | No duplicate upstream session; bounded capture | Partially complete (device) | Demo-form capture adapter (D39). On the camera: one upstream session from this host, bounded open/read, an outage and reconnect (sessions 13, 20) | The relay as sole ingest, hardware decode (NVDEC), the session count with live view and recording, a restore point | no |
| V2-06 | C2 / 2 | Playback and overlay strategy documented | Pending | — | All | no |
| V2-07 | C2 / 1 | Session-separated splits and hashes | Pending | — | All | no |
| V2-08 | C2 / 1 | Restore procedure checked; undecided features listed | Pending | — | All | no |
| V2-09 ✱ | C3 / 3 | TensorRT parity, buffer ownership and timing report; cached batch-1 context | Partially complete (device) | The legacy parity adapter (Ultralytics engine + ByteTrack, D40). On the device: guard, load and stability (session 21), and 3T's detection conditions met under the declared current conditions (3TC run 3, session 38; D53) | A TensorRT adapter without torch, fixed buffers, the parity and timing reports | no |
| V2-10 ✱ | C3 / 2 | Occlusion/resize/reconnect fixtures pass | Partially complete (device) | `test_person_tracker.py` (epoch, occlusion and resize fixtures); tracks on the device (sessions 21–38) | Labelled replays (V2-07); ByteTrack with low-score boxes | no |
| V2-11 | C3 / 1 | Bounded traces/queues; frame-to-incident timing; durable acknowledgment; profiling overhead | Pending | Package not started. Related demo behaviour: D-1 acknowledges an observation only after its commit, with a bounded in-memory hand-off (tested) | The runtime/core contract, traces, timing, overhead | no |
| V2-12 | C3 / 1 | Stale or misaligned overlays disappear | Pending | The P1 preview (D49) is a diagnostic, not this package | All | no |
| V2-13 ✱ | C4 / 2 | Deterministic replay results, including midnight | Partially complete (device) | Restricted and dwell rules: `tests/replay/test_v2_13_zone_rules.py`, including midnight. A zone incident on the device (Step 5 part 2, session 50) | Directed crossing, hysteresis, calibration, revalidation after V2-10/V2-11 | no |
| V2-14 ✱ | C4 / 2 | A repeated source event produces one incident transition | Partially complete (device) | The acceptance sentence is portable-tested (`test_incidents.py`: the repeated-event and crash-retry tests). Incidents were recorded on the device (Step 5) | Chapter 10's other tables, WAL checkpoint monitoring, quotas, revalidation after V2-11 | no |
| V2-15 ✱ | C4 / 2 | Outage/restart retries; provider success checked | Partially complete (device) | `test_outbox.py` (restart, `ok=true`, 429, timeout, dead letter), `tests/replay/test_v2_15_delivery.py`. One real Telegram send on the device, exactly one message received (Step 5 part 2, session 50) | An authenticated retry action, quotas, a device outage/restart test, revalidation after V2-11 | no |
| V2-16 | C4 / 1 | Duplicate or correlated evidence cannot inflate severity | Partially complete (portable) | Severity is the maximum, never a sum (`test_incidents.py`); a repeated report counts once (`test_scene_hazard.py`) | The regression suite across all rules | no |
| V2-17 | C5 / 3 | REST/WS/media negative-access tests pass | Pending | — (D-2 is a loopback read-only stand-in, not a substitute) | All | no |
| V2-18 | C5 / 3 | Reconnect, empty and offline states correct on the Live and Incidents screens | Pending | — (D-2 stands in; see below) | All | no |
| V2-19 | C5 / 1 | A worker failure does not kill unrelated processes | Pending | — | All | no |
| V2-20 ✱ | C5 / 1 | Model/schema validation and revoke policy defined | Partially complete (device) | A sealed gallery with a compatibility record, revoke and purge (`test_identity_gallery.py`, `test_identity_vault.py`, `test_identity_enroll.py`). One consented F1 enrollment on the device (corrected reading accepted, session 50 part 5) | Key management beyond a typed passphrase, the screens (V2-27) | no |
| V2-21 | C6 / 2 | Pre/post clip coverage and discontinuities recorded | Pending | — | All | no |
| V2-22 | C6 / 2 | Quota, crash reconciliation and restore tests pass | Pending | — | All | no |
| V2-23 | C6 / 2 | A new install reaches its first incident without source edits | Pending | — | All | no |
| V2-24 | C6 / 1 | A two-hour alpha run with initial B1/B2 comparisons | Pending | — | All | no |
| V2-25 ✱ | C7 / 3 | One-to-one identity ownership; empty-DB/TTL tests; vectorized matching; accelerated-adapter report | Partially complete (device) | One-to-one, empty enrollment and TTL: `test_r1_face_identity.py`, `test_identity.py`, `test_face_outcome.py`. The interim adapter ran on the device in four F2 attempts, none validated; face admission `PENDING_VALIDATION` | Device validation, vectorized matching, the accelerated adapter and its quality/latency report | no |
| V2-26 ✱ | C7 / 2 | Cold/warm load, vision/prefill/generation, source-age and schema report at equal scene coverage | Partially complete (device) | The loopback adapter with schema validation (`test_scene_report.py`, `test_scene_analyzer.py`). S1 on the device; the scene server ran 10 minutes in Step 5 part 3 (D60) | The model comparison report | no |
| V2-27 | C7 / 1 | Consent and deletion verified end to end (screens) | Pending | CLI enrollment and revoke exist (V2-20); no screens | All | no |
| V2-28 ✱ | C7 / 1 | A late result only enriches its own incident | Portable-tested | `tests/replay/test_v2_28_enrichment.py`, `test_r3_stale_scene_results.py`, `test_incidents.py`, the runtime test of a late scene result; shown with its applicability on the incident page (D67) | Revalidation with the real scene adapter and after V2-11 (D23) | no |
| V2-29 | C8 / 3 | Full-AI deadlines; bounded pressure behaviour | Pending | — | All | no |
| V2-30 | C8 / 2 | The H3 pressure experiment recorded | Pending | — | All | no |
| V2-31 | C8 / 1 | Fault-specific recovery; restart-rate limits | Pending | — | All | no |
| V2-32 | C8 / 1 | Disabled models visible, never silently "healthy" | Pending | The D-2 page shows disabled and unavailable components with their reasons (portable tests); the package itself is not started | All | no |
| V2-33 – V2-36 | C9 / 6 | PTZ adapter, policy, privacy/zone validation, controls | Pending | — | All | no |
| V2-37 | C10 / 2 | PWA matrix; no sensitive cache | Pending | — | All | no |
| V2-38 | C10 / 2 | Signed webhook, bounded retry | Pending | — | All | no |
| V2-39 | C10 / 2 | Tested matching-code/schema recovery | Pending | Numbered migrations; an older schema is upgraded in place and a newer one refused (`test_incidents.py`) | The drill itself | no |
| V2-40 | C10 / 1 | License and quality status per enabled feature | Pending | — | All | no |
| V2-41 – V2-48 | C11–C12 / 14 | Held-out evaluation, 24 h soak, threat tests, external install, gate reruns, bundle, research package, final rehearsal | Pending | — | All | no |
| V2-49 ✱ | C1 / 2 | Unknown versions rejected; disabled adapters import no ML framework | Portable-tested | `test_adapters.py`, `test_portable_imports.py` | none (portable acceptance) | met |
| V2-50 | C3 / 2 | Camera quality lane replays pass | Pending | — | All | no |
| V2-51 | C4 / 2 | Temporal sequence rules pass | Pending | — | All | no |
| V2-52 | C5 / 2 | Authorized pagination, escaped queries, retention consistency | Pending | The incident detail view is not search | All | no |
| V2-53 | C6 / 2 | Auditable operator feedback | Pending | — | All | no |
| V2-54 | C8 / 2 | One heavy optional job at a time; timeout, unload, unknown-profile rejection | Partially complete (portable) | One job at a time, timeout and cancel in the scene lane (`test_scene_analyzer.py`); unaccepted profiles refused (`test_scene_admission.py`) | Unload (U19) | no |
| V2-55 | C9 / 2 | A fake encoder integrates without core edits | Pending | — | All | no |
| V2-56 | C10 / 2 | Feature gates join C11; per-feature disable and rollback | Pending | — | All | no |

**Demo-only rows** (not in the backlog; no planned days):

| ID | What | Class | Evidence | Open |
|---|---|---|---|---|
| D-1 | `sentinel run`: camera → detector → rules → incidents → outbox, with optional scene and face | Partially complete (device) | Step 5 validated for its bounded test (session 50, D60), with the original `S5g_exit=1` kept; F2 runs | No production, accuracy, long-duration or resource acceptance |
| D-2 | The loopback read-only status page, incident detail (D67), the stopped-runtime acknowledgment (D68), the opt-in live view (D69) and incident snapshots (D70) | Partially complete (device) | Served on the device in Step 5 (`/status.json` captures). The detail view and the acknowledgment were checked on the device (session 52: `incident-page-20261009T112754Z`, USER-SUPPLIED, 11 PASS). The live view and snapshots are portable-tested (`test_live_preview.py`, `test_incident_snapshots.py`, `test_demo_runtime.py`), and their device check passed its saved-evidence checks (session 53: `preview-snapshot-20261009T131931Z`, USER-SUPPLIED, 19 PASS) | The contradictory freshness display that check found is fixed portably (D71, session 53: a headless-browser regression); the corrected display on the device is PENDING; their CPU, memory and disk cost (not measured). They stand in for, and do not count toward, V2-18, V2-21 or V2-22 |

## 2. Chapter 2: the prototype's known failure modes

| Regression the guide requires | Class | Evidence | Gap |
|---|---|---|---|
| An empty scene clears departed tracks; a stub fire observation with zero people produces an incident | Portable-tested | `test_r2_zero_person_scenes.py` (minutes without people; a hazard with nobody present); `test_v2_14_incident_flow.py` (a VLM-only fire candidate opens an unconfirmed warning). On the device: empty intervals with zero person detections (3TC run 3, session 38) | The fire path has not run on the device |
| Two people, one visible face: the second never inherits the first identity | Portable-tested | `test_r1_face_identity.py` (known and unknown with one visible face; a face inside two overlapping people) | Not on the device (F2 not validated) |
| Unplugging the camera gives no new detections from the frozen last frame | Portable-tested | `test_demo_runtime.py` (no inference on an ended epoch; a stall withdraws people); `test_health.py`. One capture-probe outage on the device (session 20) | Not run with the detector on the device |
| A delayed result from an old event cannot modify a new incident | Portable-tested | `test_r3_stale_scene_results.py`, `test_v2_28_enrichment.py` | Not on the device |
| A calm zone entry alerts while the VLM is off | Device-verified (bounded) | Step 5 part 2: one incident and one Telegram alert with scene analysis off (session 50); `test_v2_13_zone_rules.py` | One run, one alert; no rate claim |
| A provider outage and a process restart preserve pending delivery | Portable-tested | `test_outbox.py` (an outage across a restart), `test_v2_15_delivery.py` | Not on the device |
| Starting and stopping Sentinel leaves unrelated sessions and the desktop unchanged | Partially complete (device) | `sentinel run` owns and stops its children; guarded device runs end with no leftovers (Step 5, F2) | The demo procedure itself stops the display manager (D29); supervision (V2-19) |
| A viewer cannot change settings; stale video never says Live | Partially complete (portable) | The D-2 page accepts GET only (others get 405, tested); fresh, stale and offline at 2 s and 10 s (`test_r2_zero_person_scenes.py`). The opt-in live view (D69) shows a picture only while its frame is live under the same rule, refuses a picture whose frame went stale, and marks a picture not current when the page stops refreshing (`test_live_preview.py`, `test_demo_runtime.py`). On the device (session 53, USER-SUPPLIED) it was live at both saved captures. D71 (session 53): every current claim on the page (`LIVE`, the healthy state, the picture, the page itself) lasts 5 s after the browser shows the page, by CSS alone, and only a successful refresh brings it back; a headless-browser regression covers a refresh that hangs, one the browser gives up, and recovery (`test_status_page_browser.py`), plus markup tests | **Not met on the device as checked:** a page that had stopped refreshing kept its green `LIVE` and `All components working` banners beside the picture marked `NOT CURRENT` (session 53). The corrected display on the device is PENDING; the regression ran in Chromium only. No users or roles (V2-17); V2-18's Live screen is not started |

## 3. Chapter 21: mandatory test scenarios

| Scenario | Class | Evidence | Gap |
|---|---|---|---|
| No people for minutes: empty-scene checks run, the UI shows empty | Portable-tested | `test_r2_zero_person_scenes.py`, `test_r3_stale_scene_results.py` (scene checks with nobody present); the runtime's empty-room test | Device: only short empty intervals (3TC run 3) |
| Known and unknown people overlap, one face visible: no identity transfer | Portable-tested | `test_r1_face_identity.py` | Device validation pending (F2) |
| Unplug, reconnect, PTS reset, repeated frames: no stale frame reprocessed | Portable-tested | `test_capture.py`, `test_frame_identity.py`, `test_opencv_source.py`, `test_demo_runtime.py` | Device: one capture-probe outage only |
| VLM malformed JSON, wrong event ID, timeout, very late answer: core continues | Portable-tested | `test_r3_stale_scene_results.py`, `test_scene_report.py`, `test_scene_analyzer.py` | Not induced on the device |
| Telegram HTTP error, `ok=false`, 429, timeout, success then worker crash | Portable-tested | `test_outbox.py` | Device: one success only |
| A core crash after the commit and before the acknowledgment: the retry deduplicates | Portable-tested | `test_incidents.py` (a child process killed after COMMIT), `test_demo_runtime.py` | Not on the device |
| A recorder crash during clip finalization; disk quota | Pending | — (V2-21, V2-22) | All |
| Pressure unloads models; monitoring stays above its declared minimum rate | Pending | — (V2-29) | All |
| Viewer mutations, unauthenticated media, expired sessions, WebSocket reuse | Pending | D-2 refuses every non-GET method (tested), but there are no sessions, media or WebSocket | All (V2-17) |
| PTZ loses target or network during movement | Pending | — | All |
| An upgrade fails at each boundary; a backup restore keeps schema and data | Pending | Migration upgrade and newer-schema refusal are tested | The drill (V2-39, V2-22) |

## 4. Chapter 22: release gates (proposed targets)

| Gate | Class | Evidence | Gap |
|---|---|---|---|
| Supported hardware profile | Partially complete (device) | Device inventory (V2-01); the camera's substream profile used throughout | Hardware decode not used (software decode, D24) |
| Core throughput ≥ 15 unique detected fps, 1080p core profile | Pending | Demo runs use the 640×480 substream; that is not the declared profile | A measurement in the declared profile |
| Compute optimization (chapter 32 comparison) | Pending | — | All |
| Freshness: stale within 2 s, offline within 10 s, no incident from expired frames | Portable-tested | `test_r2_zero_person_scenes.py` (stale at 2 s, offline at 10 s), `test_health.py`, `test_evidence_applicability.py` | Not measured on the device |
| Rule latency p95 ≤ 1 s | Pending | — | A measurement |
| External alerts p95 ≤ 5 s | Pending | One send (Step 5) | A distribution |
| Memory ≤ 5.0 GB steady, ≤ 5.4 GB peak | Partially complete (device) | The D58 candidate run passed the step-4 criteria for the bounded 640×480 replay profile, with a 4,297,113,600 B sampled peak (session 46, accepted under D59) | The full runtime, the declared profile, long duration |
| Durability: no loss of committed incident/outbox state across crash and restart | Portable-tested | Crash-after-commit, outbox restart, and acknowledgment across a restart (`test_incidents.py`, `test_outbox.py`, `test_incident_ack.py`) | No device crash test |
| Camera recovery: fresh frames within 30 s | Partially complete (device) | One capture-probe outage: frames resumed roughly 26 s after power was restored. This is an estimate from frame counts, not a measurement (session 20) | A measured check under `sentinel run` |
| Worker recovery: a failed optional worker cannot halt the core | Partially complete (portable) | A scene server that exits mid-run is shown unavailable while the core runs (`test_demo_runtime.py`) | Stable degraded mode within a retry budget (V2-29, V2-31) |
| Reliability: 24 h with fault injections | Pending | — | All |
| Detection quality: event recall and false alerts on staged events | Pending | 3TC run 3 is one bounded check (D53), not a quality evaluation | All |
| Identity: association regressions; held-out FAR/FRR | Partially complete (portable) | Association regressions pass (`test_r1_face_identity.py`) | Held-out curves; device validation (`PENDING_VALIDATION`) |
| Security/privacy | Partially complete (device) | Listeners loopback only and no credentials in saved output, in Step 5 and F2 captures; model endpoint loopback only (D44) | Authentication across REST, WS and playback (no API exists yet) |
| Recovery/update | Pending | — | All |
| Installation: two clean installs | Pending | — | All |
| Research: H3 by week 16 | Pending | — | All |
| Release claims labelled | Pending | — | All |

## 5. How much is complete? (a proposal; the maintainer decides)

The figures below come from section 1 alone. Package days are the guide's planned person-days, so they weight each package by planned effort, not by importance.

| Class (section 1) | All 100 days | Alpha span, C1–C6 (53 days) | Oct 20 demo scope (29 days) |
|---|---|---|---|
| Full acceptance met | 6 | 6 | 6 |
| Portable-tested, full acceptance pending (V2-28) | 1 | 0 | 1 |
| Partially complete (device) | 22 | 17 | 22 |
| Partially complete (portable) | 4 | 2 | 0 |
| Pending | 67 | 28 | 0 |

- **Measure A: full acceptance (the only acceptance-based figure).** The planned days of packages whose guide acceptance is met in full: **6 % of the backlog** (6/100), **11 % of the alpha span** (6/53) and **21 % of the demo scope** (6/29).
- **Measure B: started (an upper bound, not completion).** The days of packages with any evidence at all, counted as if they were finished, which D23 does not allow: **33 %** of the backlog (33/100) and **47 %** of the alpha span (25/53). Even this generous bound stays below 50 % for the backlog and for the alpha span.
- **The demo scope** has evidence in all 14 of its packages (29/29 days), but only 6 of those days are fully accepted. Any claim scoped to the demo should name that scope, and say that most of it is demo form.
- **Not used:** test counts, numbers of device runs, and Claude-hour estimates. None of these measures acceptance.

**For Review 1:** under the guide's acceptance and D23, "50 % complete" is not supported by any of these scopes. A defensible statement is per scope: for example, "fully accepted: 6 % of planned backlog days; every Oct 20 demo-path package has demo-form evidence, 22 of its 29 days with some part verified on the device". Any other weighting, such as partial credit for demo form, is the maintainer's decision. It should be fixed before it is applied and reported beside Measure A.
