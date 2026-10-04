# Sentinel v2 — implementation status

Last updated 2026-10-04, session 21 (Claude). **Checklist step 3 recorded in part** as USER-SUPPLIED MEASUREMENTS (boot `dbdbdc0c-5c27-469b-ac1e-280a2b1140c7`, commit `7dc7a04`, rebooted since step 2). The D27 guard refused without the preload (`exit_a=1`, `libcuda_not_l4t`). 60 s of tracking gave `exit_b=0`, 858 of 888 captured frames processed, no failures, upstream max 1 and no credentials in the output. **Detection correctness is PENDING:** the maintainer's observations of who was in view were not supplied, so the person counters are not evaluated, and track IDs are not counted as people. **Defect found and fixed (D48):** the model load wrote Ultralytics' and TensorRT's diagnostics to stdout, so `b.json` was not valid JSON (446 characters before the object). `sentinel track probe` and `sentinel run` now keep stdout for their JSON and send other stdout output to stderr while they run. 8 new subprocess tests; 736 portable tests pass. Device verification of the fix is PENDING. The caffeine record is corrected, and the TensorRT engine-plan warning is recorded as unresolved (U22). Session 20's summary: **Checklist step 2 recorded** as USER-SUPPLIED MEASUREMENTS: a camera power cut and restore during a 180 s capture probe gave `exit=0`, `frames_received`, `connects` 2, `epochs` 2, `stream_ends` 1, `open_failures` 3, upstream max 1 and no credentials in the output. This meets step 2's outage criteria. It closes V2-05's demo-form outage and read-timeout items; V2-05 stays partial (demo form, full acceptance pending). An earlier run with no physical outage is recorded as steady capture only. **Environment change (USER-SUPPLIED):** the maintainer installed the `caffeine` apt package. Claude saw no caffeine process afterwards; whether it runs during later steps is unverified. Step 3 instructions were tightened (docs only). Session 19's summary: **Checklist step 1 recorded** as USER-SUPPLIED OBSERVATIONS: with `YOLO_OFFLINE=true`, Ultralytics is offline and its events are disabled; `settings_sync` stays `True` (1b not performed). **Step 2 revised before its first run:** a 180 s window for a camera reboot and timed cut/restore prompts. Docs only; no camera, GPU or service was touched. Session 18's summary: **Review of `268e905` (D47), portable only:** one defect fixed. With `--sanitized-logs`, which step 4 always uses, error names the sanitizer does not recognise (such as `RemoteDisconnected`) were dropped, so criteria F and V1 could pass with real face or scene errors. F and V1 now gate on error totals counted before sanitizing. 728 portable tests pass. Step 4 is still not authorized to run, and no profile is accepted. Session 17's summary: **Step-4 criteria v2 and the guarded `step4` mode (D47), portable only:** `step4-combined-cache-off-v2` requires:
- every sample in the steady interval ≤ 5,000,000,000 B, with the time above 5.0 GB reported;
- a sampled run peak ≤ 5,400,000,000 B;
- monotonic steady boundaries with teardown excluded;
- unique-frame throughput and replay scheduling age;
- face and scene completion;
- verified prompt-cache-off evidence;
- kernel-log coverage proven per boot and run interval;
- identity hashing before the cache drop, rehashed at the end.

`operator_check.py --execute-workload step4` runs under the existing guard, thresholds and owned-process cleanup, with a 1,200 s deadline and an explicit same-boot Check 9 result. Checklist step 4 is now runnable shell, exercised against a stub. 723 portable tests pass. The run and any acceptance commit stay PENDING; no profile is accepted. Session 16's summary: **`--scene` admission fix (D46):** `sentinel run --scene` now needs an ACCEPTED combined profile in the existing resource-profile registry, measured with `--cache-ram 0`, whose recorded evidence passes the predeclared step-4 criteria and matches the selected runtime configuration. The check runs before the database opens or any scene process starts. The registry holds no accepted profile, so scene analysis cannot be admitted until a separate, maintainer-approved acceptance commit (D46). Core-only runs are unchanged. 37 new tests; 647 portable tests pass. The in-memory pending-write limitation is now on the status page. Session 15's summary: **D-2 loopback status page, demo form, portable part:** a stdlib, read-only page on 127.0.0.1 (with a Host check) showing component readiness, capture/reconnect state, rates, degradation, incidents, and alerts as queued, attempted, delivered or failed (D45). It is served by `sentinel run` on port 18090. 23 new tests; 610 portable tests pass. One numbered operator checklist (steps 1–5) replaces the separate V2-05 and V2-09/V2-10 blocks; every step is PENDING. **D-1 `sentinel run`, demo form, portable part:** wires capture → legacy detector/tracker → EdgeCore → zone/hazard rules → `IncidentService` → outbox/Telegram through their existing contracts (D43). It adds a llama-server process owner that refuses any non-loopback bind before spawning and applies the D27 full-offload guard (D44). Scene analysis stays off by default. 54 new tests; 587 portable tests pass. The device run is PENDING. Session 14's summary: **D41:** S1 met its predeclared criteria, so `--cache-ram 0` is adopted for the demo scene path (maintainer's conditional authorization); scope and limits are in the session 14 log. **V2-26 demo form, portable part:** a loopback-only llama-server launch spec (D42), the S1/check 8 request, a bounded threaded scene worker and the `scene_server` config; 35 new tests; 533 portable tests pass. Privacy item: Ultralytics analytics (operator action PENDING). Session 13's summary: The maintainer's headless runs at `e9af7f4` are recorded as USER-SUPPLIED MEASUREMENTS: two Check 9 smokes, three S1 arms (arm a twice; the first arm b was interrupted by a signal to the runner) and four capture probes. **S1 result:** with llama-server's default prompt cache, steady pressure rose +115 MB/min (reproduced: +117 MB/min); with `--cache-ram 0` it rose +3.9 MB/min, with the same 53/53 valid requests and no latency penalty. The timebox's first branch applies: Claude proposes adopting `--cache-ram 0` for the demo profile and D-1; the maintainer decides. Other memory work stays paused. Further U19 evidence: the first GPU model load of the boot left a one-time step of about 0.66 GB, in no sampled meminfo field; later cycles added none. **V2-05 demo form:** one upstream session, bounded fresh hand-off, 15 fps 640×480 decode at about 0.07 core-equivalents and credential-free output were observed on the camera (runs b and b2). Run a never connected ("No route to host"), which corrects the maintainer's summary, and no outage was tested (b3 never contacted the camera), so the outage and read-timeout checks stay PENDING. The `pts_quality: none` share (21.5 % and 15.4 %) is explained: OpenCV 4.13's `CAP_PROP_PTS` rounds the PTS to whole periods of the average frame rate and repeats the last value when a frame has none. This was reproduced on a synthetic file. The source now reads the unrounded `CAP_PROP_POS_MSEC`, and the probe names the cause of each `none` frame. **V2-09/V2-10 demo form, portable part:** a `PersonTracker` boundary (per-epoch reset, process-once ordering, epoch-unique IDs, v1's confirmation score, validated output) and the legacy `yolov8n.engine` + ByteTrack backend. The backend has the D27 guard, an engine SHA-256 pin and Ultralytics offline mode. Added: `sentinel track probe` and the registry entry with D33's provisional profile. 47 new tests; 498 portable tests pass. The detector device check is PENDING. Session 12's summary: the maintainer's first S1 attempt at `84f15ec` never reached either arm; the V2-05 demo-form capture adapter (D39) was implemented with 40 tests (451 portable tests).

## Position

| | |
|---|---|
| Branch | `v2-beta`, created from `master`. At session 11 start, HEAD was `f60652903ef6cf7656f6711e29b81c13e598cfb0`, and the local `origin/v2-beta` reference was the same commit (reflog: pushed 2026-10-03 21:31 +0530, after session 9). This session's commit is one ahead and was not pushed. No fetch or remote query ran; this is local repository evidence, not a freshly verified remote state. |
| CI | Maintainer report, 2026-09-29: GitHub Actions passed at `32985c2`: "v2 portable checks" (run #2) and "Dashboard checks" (run #5). Earlier runs at `6578ded` also passed. |
| Base commit | `2b2d639621e8c043cc58a126f47b1b8ab6c22135`, the commit the audit verified, confirmed as HEAD before starting |
| Session 1 commits | `ec6698d` CLAUDE.md · `b52920f` package skeleton and portable tests · `e9f959d` clock · `bc42248` frame identity · `3b47f5f` evidence/track applicability · `ee50275` config and CLI · `6a9e71d` CI workflow · `6578ded` status record |
| Session 2 commits | `4587021` PTS tolerance at ingest · `812b42c` V2-01 records · `e81db81` replay timelines · `86889f8` scene lane (R3) · `432dc69` live state and freshness (R2) · `66937ef` face association and identity (R1) · `48d6188` status record · `9593b64` adapter manifests (V2-49) · `32985c2` maintainer decisions D22–D24 |
| Session 3 work (2026-09-29) | Interrupted by a usage limit before anything was committed; committed in session 4 as the first commit below |
| Session 4 commits (2026-10-03) | See the session 4 slice log |
| Session 5 (2026-10-03) | `c865f14` check 8 and 1 Hz cadence · `45d9ba3` V2-28 enrichment · `b872c27` verification record. See the session 5 log. |
| Session 6 (Codex, 2026-10-03) | `805d837`: portable U19/U21 instrumentation, behavioral tests and status update, recorded before committing. See the session 6 log. |
| Session 7 (Codex, 2026-10-03) | `79093b7`: portable U20 requests/completions and synthetic regressions, counter/build investigation, guarded U21 proposal and pre-commit status update. See the session 7 log. |
| Session 8 (Codex, 2026-10-03) | `00f940f`: portable operator orchestration, bounded allocation smoke, sanitized profiler diagnostics, synthetic tests, reconciled evidence and pre-commit status update. See the session 8 log. |
| Session 9 (Codex, 2026-10-03) | `f606529`: verified PVA service classifier correction, synthetic regressions, operator checklist and pre-commit status record. |
| Session 10 (Claude, 2026-10-03) | Read-only review of Check 9, the short U21 and the package logs; proposed S1. No commit; recorded in session 11. See the session 10 log. |
| Session 11 (Claude, 2026-10-03) | `84f15ec`: D37/D38, E-1 and reboot records, E-2 calculations, S1 portable implementation and tests, status record. Pushed by the maintainer (local `origin/v2-beta` = `84f15ec` at session 12 start). See the session 11 log. |
| Session 12 (Claude, 2026-10-03) | `ed26dc9`: record of the maintainer's Check 9/S1 runs at `84f15ec` (S1 did not run). `e9af7f4`: V2-05 demo-form capture adapter, tests and status record. See the session 12 log. |
| Session 13 (Claude, 2026-10-04) | `f6f9d5b` record of the Check 9, S1 and capture runs at `e9af7f4`, S1 comparison and V2-05 evaluation (docs only) · `8836929` PTS follow-up · `cd1c201` V2-09/V2-10 demo form (D40). See the session 13 log. |
| Session 14 (Claude, 2026-10-04) | `ea95571`: S1 evaluation and D41 adoption, D42, V2-26 demo form (portable), Ultralytics privacy item. See the session 14 log. |
| Session 21 (Claude, 2026-10-04) | Step 3 recorded in part (USER-SUPPLIED; detection correctness PENDING); D48 stdout/JSON fix for `track probe` and `run` with 8 subprocess tests; caffeine observation corrected; U22 added; one local commit, not pushed. See the session 21 log. |
| Session 20 (Claude, 2026-10-04) | Step 2 recorded (USER-SUPPLIED); caffeine package install recorded; step 2a and step 3 instructions tightened (docs only); one local commit, not pushed. See the session 20 log. |
| Session 19 (Claude, 2026-10-04) | Step 1 recorded (USER-SUPPLIED); step 2 instructions revised (docs only); one local commit, not pushed. See the session 19 log. |
| Session 18 (Claude, 2026-10-04) | Review and portable verification of `268e905`; the F/V1 error-total fix (D47 amendment); one local commit, not pushed. See the session 18 log. |
| Session 17 (Claude, 2026-10-04) | `268e905`: step-4 criteria v2, the guarded `step4` and identity modes, startup identity checks and the runnable checklist (D47); not pushed. See the session 17 log. |
| Session 16 (Claude, 2026-10-04) | `7f06055`: the narrow `--scene` admission fix (D46), the acceptance procedure, and the pending-write limitation on the status page. See the session 16 log. |
| Session 15 (Claude, 2026-10-04) | `537557c` D-1 demo form (portable part); `7f5f9f3` D-2 (portable part) and the operator checklist; not pushed. See the session 15 log. |
| Working tree | At session 15 start (10:46 IST): HEAD `ea95571`, equal to the local `origin/v2-beta` (the maintainer pushed after session 14; no fetch ran). The tree was clean. The session-start snapshot had listed `BUILD.md` as modified (session 12's trailing blank line), but by 10:43:07 IST, before Claude's first command, `BUILD.md` matched HEAD again. No other agent process was running; the maintainer's VS Code session was. Claude did not touch `BUILD.md`. Earlier: at session 13 start: HEAD `e9af7f4`, two commits ahead of the local `origin/v2-beta` (`84f15ec`; nothing pushed since session 11). The only change was session 12's unrelated trailing blank line in `BUILD.md` (not Claude's), which stays uncommitted. |
| Known limitation (D-1, D43) | **Pending incident writes are buffered in memory, not in a crash-safe spool.** Rule observations waiting for a failed `record()` to be retried (at most 256) are lost if the process stops abruptly; overflow is counted as lost. The status page states this beside the pending count. A durable spool belongs to the runtime/core split (guide ch. 4, V2-11). |
| Local-only files | `AGENTS.md`, the v2 guide, the audit review and `docs/LOCAL_NOTES.md` (device-specific notes), excluded through `.git/info/exclude`; none included in session 6–11 increments. A fresh clone does not contain them, although CLAUDE.md names the guide and audit. |
| Selected package | **D48 stdout/JSON separation for `track probe` and `run`** (session 21): a narrow CLI fix for the defect checklist step 3 found; portable part done, device verification PENDING. Before it: **Step-4 criteria v2 and guarded mode** (session 17, D47; reviewed and corrected in session 18): portable part done; the run is PENDING (checklist step 4). Before it: **`--scene` admission fix** (session 16, D46): a narrow amendment to D-1, done portably; no device check of its own. Before it: **D-2 loopback status page, demo form** (session 15): portable part done, device check PENDING (checklist step 5). Before it in session 15: **D-1 `sentinel run`, demo form**, portable part done (`537557c`), device run PENDING. Before that: **V2-09/V2-10 demo form** (session 13): the legacy detector + ByteTrack parity adapter. Portable part complete (D40); device check PENDING; partial at most (D23). V2-05's demo form is evaluated (session 13 log). Its outage and read-timeout checks are now met too (USER-SUPPLIED, session 20 log). It stays partial under D23. S1 (D38) ran; adopting `--cache-ram 0` is the maintainer's decision. U20 and U19/U21 instrumentation remain portable-complete only; D-1 and hardware acceptance are not complete. V2-28 remains demo form, full acceptance pending. |
| Other branches | `origin/Yogeshvar425-patch-1` (teammate) is **not merged**: a single commit `6755796` that adds @Yogeshvar425 to `.github/CODEOWNERS` (merge base `c66ebde`). `origin/codex/github-audit-fixes-2026-09-19` is already in `master` via PR #4. |
| Effort | Per-package estimates are in the package table (given to the maintainer on 2026-09-29). The re-estimate of optimization effort still waits for V2-01's B0 run. |
| v1 on this device | Historically **not running** (maintainer, 2026-09-29); current state must be inspected, not inferred. This workflow never terminates existing v1 processes. |
| Waiting on the maintainer | **The session 15 operator checklist, steps 3–5** (PENDING). Step 1, the Ultralytics setting, is done (USER-SUPPLIED, session 19); its optional 1b `yolo settings sync=False` was not performed. Step 2, the V2-05 physical outage, is done (USER-SUPPLIED, session 20). Step 3, the V2-09/V2-10 device check, is recorded in part (USER-SUPPLIED, session 21): the guard and stability criteria are met, and detection correctness waits for the maintainer's observations of who was in view. Step 4 the confirming combined profile with `--llama-cache-ram 0` (needs approval and predeclared criteria). Step 5 is the first end-to-end alert and the status page, which is also V2-15's Telegram device check. Then come the decisions on scene admission for D-1 and the review of D43–D45. Still open from earlier: why S1 arm b #1 was interrupted; U18's runtime policy, U19 unload, U20 real-model/demo-exception evidence, the 30-minute U21 rerun (not approved) and consent/figure labels. D36 provisionally retains b8932; beta still requires a tested compatible schema-fix descendant. E-1 reported Ollama inactive and disabled (USER-SUPPLIED). The full empty-room-clip review and the deletion of review images are not confirmed. |

## Next concrete task

1. **Step 3 completion (maintainer).** Give the observations from the tracking run (`b_start_utc` 15:49:33Z to `b_end_utc` 15:50:44Z on 2026-10-04, which includes the 8.69 s model load): how many people were in view, roughly when they were visible, whether three were ever visible at once, and any person-like objects or reflections. Until then, detection correctness stays PENDING and step 4's prerequisite "step 3 passed" is not met. Device verification of D48 is also PENDING: step 5's `run.jsonl` must parse line by line, or the maintainer may choose a step 3 rerun in a new directory (not requested).
2. **Maintainer decisions before step 4:**
   - the D47 criteria (`step4-combined-cache-off-v2`) are approved for implementation (session 17). Session 18 amended F and V1 to gate on error totals (session 18 log); review that amendment. The step-4 **run** still needs the maintainer's explicit go-ahead, through checklist step 4 (4a–4f). Implementation and passing portable tests are not that go-ahead;
   - after it, if it passed, approve the separate acceptance commit (D46) that adds the step-4 profile to `RESOURCE_PROFILES`. That commit is the only way `--scene` can be admitted;
   - review D43–D47.
3. **Step 5 (maintainer, PENDING):** the first end-to-end alert and status-page check (5a core; 5b scene only after step 4 and admission). It is also V2-15's device check: one real Telegram send.
4. **Record the step results** as USER-SUPPLIED MEASUREMENTS, with each step's own boot ID and commit. Then the maintainer picks the next package. Claude's suggestion is the interim face adapter (V2-25 demo form, 1 Hz, D34), whose enrollment needs consented photos (unresolved), never `face_db.pkl`.
5. **Unchanged:** U19/U21 instrumentation is portable-complete only (Codex, session 6). Memory work stays paused under the timebox except the confirming run. V2-13's rest (directed crossing, hysteresis, D30) comes after the demo.

## Oct 20 demo milestone: plan and deviations from the guide order

Target (maintainer, 2026-09-29): a demoable end-to-end path on this Jetson by 2026-10-20: camera → detection → tracking → identity → scene/VLM → alert/outbox → dashboard. That is three weeks. In the guide's order it spans C2–C7 (about 12 cycles' worth of dependencies). It is feasible only as a **demo profile**: the v2 core logic (contracts, freshness, identity, scene lane, rules, durable incidents/outbox) plus interim adapters around the existing models. It is not the optimized B2 profile and establishes no gate.

| Week | Work | Status |
|---|---|---|
| 1 (to Oct 6) | V2-49; V2-13 zone rule; V2-14 SQLite incidents/outbox; V2-15 leased outbox + Telegram (mocked). All portable. | V2-49 done. Demo form done: V2-13 (restricted + dwell; crossing deferred), V2-14, V2-15 (mocked); these count as partial (D23). |
| 2 (to Oct 13) | Device adapters (D24): capture from the substream (profile A, D22) with `~/onvif_env`'s OpenCV/FFmpeg software decode, video only, stamped by FrameStamper; detector + ByteTrack via the existing `yolov8n.engine` as the *legacy parity adapter*; interim face adapter (existing DeepFace/Facenet512 on CPU) feeding v2 association; llama-server scene adapter; `sentinel run` loop around `EdgeCore` | Checks 3 and 5 done; U13 settled (D27); U17 settled (D28, D33: check 8 run 2026-10-03); capture demo form on the camera (session 13; outage check PENDING); detector + ByteTrack demo form portable (session 13; device check PENDING); S1 run, `--cache-ram 0` proposed; U18 open; U20, U21 need decisions |
| 3 (to Oct 20) | Loopback-only, read-only status page (stdlib HTTP server) showing LiveState, incidents and delivery outcomes; end-to-end rehearsal; demo script including camera loss and recovery | — |

**Deviations from the guide's order.** Accepted by the maintainer (D23) **on condition that every affected package is marked "demo form, full acceptance pending" in this file and is not counted done.** The package table applies that marking.

1. **Face (V2-25, C7) and VLM (V2-26, C7) come before C4–C6, via interim adapters** around the models v1 already uses: DeepFace/Facenet512 on CPU and llama-server with LFM2-VL-1.6B. Reason: the milestone requires identity and scene. The v2 association, identity and scene-lane logic is final; the adapters are labelled demo-only, uncalibrated and unbenchmarked. Enrollment will be re-created as validated non-executable data from consented photos, never by loading `face_db.pkl`.
2. **Portable C4 packages (V2-13/14/15) before the C2/C3 hardware packages.** Reason: C2/C3 are blocked on the PENDING hardware checks, while C4 is portable and on the milestone path. They must be revalidated once V2-10 and V2-11 exist.
3. **No go2rtc relay for the demo (V2-05 deferred).** The v2 runtime opens the substream itself as the only ingest. v1 is not running on the device, so this is the only upstream session. Installing go2rtc is a new binary dependency that needs a decision.
4. **Software decode for the demo (D24):** `~/onvif_env`'s OpenCV with its bundled FFmpeg. 640×480 at 15 fps is about 4.6 Mpx/s. GStreamer/NVDEC waits for V2-05 proper; revisit after Oct 20. The CPU cost has not been measured.
5. **Dashboard: a loopback-only, read-only status page instead of V2-17/V2-18** (FastAPI, auth, roles, PWA). Access is over an SSH port forward. This avoids new dependencies and does not expose an unauthenticated service. FastAPI is not installed anywhere; adding it is a dependency decision.
6. **Runtime environment for the demo (D24):** `~/onvif_env`, unchanged, with `PYTHONPATH=src`. Its CPU-only parts run as they are. GPU parts (detector engine and llama-server) need the L4T libcuda preload that `~/onvif_env/bin/activate` sets; D27 keeps it for the demo, with a guard. The portable package passes its suite with that environment's pydantic 2.12.5 (verified below). Nothing will be installed into it; if an adapter needs anything missing, a separate environment will be proposed first.
7. **Headless demo host (D29):** the display manager is stopped for the demo, as v1's launcher does, and the U17 profile is measured the same way.

## Package status

Estimates as given to the maintainer on 2026-09-29. V2-49 has since been done.

- **Claude h** is 0.35 h per remaining guide person-day (session pace so far is about 0.2–0.27 h), plus 0.25 h per expected device round trip, ×1.5 for UI work, rounded up to 0.5 h per package.
- **Maintainer Jetson h** is attended time at the device or camera. It excludes unattended runs and code review.

**Counting rule (D23):** a package marked **"demo form, full acceptance pending"** counts as partial at most, even when its demo form is complete. It becomes done only when its guide acceptance passes in the full form. For example: V2-05 with the relay and verified hardware decode, V2-09 with the TensorRT adapter and parity report, V2-13/14/15 revalidated after V2-10/V2-11.

| Package | Title | Status | Portable or Jetson | Oct 20 path | Claude h | Maintainer Jetson h | Depends on |
|---|---|---|---|---|---|---|---|
| V2-01 | Hardware and v1 timing/memory baseline | in progress | Jetson | yes (checks 3, 5, 8, the bounded Check 9 smokes and S1 done; 9b/U18 pending) | 1.5 | 3 | — |
| V2-02 | Config, frame/evidence contracts, fake clock | done | Portable | yes | 0 | 0 | — |
| V2-03 | Replay fixtures, first identity/empty-scene fixes | done | Portable | yes | 0 | 0 | — |
| V2-04 | Dev setup and CI skeleton | in progress | Portable | no | 0.5 | 0 | — |
| V2-05 | Relay ownership and hardware decode spike | partial: demo-form capture adapter done (session 12); on the camera, all demo-form criteria met (session 13; outage and read-timeout checks met in session 20, USER-SUPPLIED). Relay, NVDEC, the live-view/recording session count and a restore point remain | Jetson | yes: **demo form, full acceptance pending** | 2 | 2.5 | 01, 02 |
| V2-06 | Browser/codec/timestamp spike | not started | Jetson | no | 2 | 3 | 05 |
| V2-07 | Dataset consent, labels, split manifest | not started | Jetson (recording) | no | 1 | 3 | 03 |
| V2-08 | Gate B record, recoverable device baseline | not started | Jetson | no | 1 | 4 | 01, 05, 06 |
| V2-09 | TensorRT adapter and fixed buffers | partial: demo form (legacy parity adapter) portable part done (session 13); device check in part (session 21, USER-SUPPLIED: guard refusal and 60 s load/stability met; detection correctness PENDING) | Jetson | yes: **demo form, full acceptance pending** | 2.5 | 3 | 05, 08 |
| V2-10 | Tracker and coordinate parity | partial: demo form (tracker boundary, epoch/occlusion/resize fixtures) portable part done (session 13); device check in part (session 21, USER-SUPPLIED: tracks produced, no failures; detection correctness PENDING) | Jetson | yes: **demo form, full acceptance pending** | 1.5 | 1.5 | 07, 09 |
| V2-11 | Telemetry and runtime handoff contract | not started | Portable + device check | no | 1 | 0.5 | 02, 05 |
| V2-12 | Overlay rendering vs timed fixtures | not started | Portable + device check | no | 1 | 1 | 06, 10 |
| V2-13 | Zone, crossing and dwell rules | partial: demo form (restricted + dwell) done, directed crossing not started | Portable | yes: **demo form, full acceptance pending** | 1 | 0 | 10, 11 |
| V2-14 | Incident transaction, SQLite migrations | partial: demo form done | Portable | yes: **demo form, full acceptance pending** | 1 | 0 | 02, 11 |
| V2-15 | Leased outbox and Telegram adapter | partial: demo form done (mocked); device check pending | Portable + device check | yes: **demo form, full acceptance pending** | 1 | 0.5 | 14 |
| V2-16 | Rule evidence/correlation regression suite | partial | Portable | no | 0.5 | 0 | 13, 14 |
| V2-17 | Sessions, roles, API, media authorization | not started | Portable + device check | no (D-2 stands in for the demo; not a substitute) | 1.5 | 0.5 | 14, 15 |
| V2-18 | Live and Incidents web screens | not started | Portable + device check | no (D-2 stands in for the demo; not a substitute) | 2 | 1 | 12, 17 |
| V2-19 | Runtime service supervision | not started | Jetson | no | 1 | 1.5 | 09, 11 |
| V2-20 | Identity storage and enrollment contract | partial | Portable + device check | yes: **demo form, full acceptance pending** | 0.5 | 0.5 | 07, 10 |
| V2-21 | Compressed ring and event clips | not started | Jetson | no | 1.5 | 2 | 05, 14 |
| V2-22 | Retention and consistent backup | not started | Portable + device check | no | 1 | 0.5 | 14, 21 |
| V2-23 | Installer alpha and first-run flow | not started | Jetson | no | 1.5 | 3 | 17, 19 |
| V2-24 | Alpha replay/soak report | not started | Jetson | no | 1 | 3 | 16, 18, 22, 23 |
| V2-25 | Face association/alignment/runtime adapter | partial | Jetson | yes: **demo form, full acceptance pending** | 2 | 4 | 20, 24 |
| V2-26 | Small VLM vs existing model comparison | partial: demo form (loopback llama-server adapter, bounded worker) portable part done (session 14); device check with D-1 | Jetson | yes: **demo form, full acceptance pending** | 2 | 5 | 09, 24 |
| V2-27 | Enrollment/revoke screens | not started | Portable + device check | no | 1 | 0.5 | 20, 25 |
| V2-28 | Evidence enrichment isolation | partial: demo form done (scene and store sides) | Portable | yes: **demo form, full acceptance pending** | 0.5 | 0 | 14, 26 |
| V2-29 | Admission/degradation controller | not started | Jetson | no | 2.5 | 3 | 09, 25, 26 |
| V2-30 | H3 pressure experiment and analysis | not started | Jetson | no | 1.5 | 5 | 29 |
| V2-31 | Recovery actions, diagnostic redaction | not started | Portable + device check | no | 1 | 1 | 19, 29 |
| V2-32 | Capability/health feedback in UI | not started | Portable | no | 0.5 | 0 | 29, 31 |
| V2-33 | Actual-camera PTZ adapter | not started | Jetson | no | 1.5 | 2.5 | 08, 31 |
| V2-34 | PTZ policy, override, home state machine | not started | Jetson | no | 1.5 | 1.5 | 13, 33 |
| V2-35 | PTZ and privacy/zone validation | not started | Jetson | no | 1 | 1.5 | 25, 34 |
| V2-36 | PTZ controls and capability fallback | not started | Portable | no | 1 | 0 | 32, 34 |
| V2-37 | Responsive PWA and offline behaviour | not started | Portable + device check | no | 2 | 2 | 18, 27, 32 |
| V2-38 | Webhook and optional MQTT adapter | not started | Portable | no | 1 | 0 | 15, 17 |
| V2-39 | Upgrade/rollback and TLS drill | not started | Jetson | no | 1.5 | 3 | 22, 23, 31 |
| V2-40 | Frozen model/config/feature manifests | not started | Portable + device check | no | 1 | 0.5 | 25, 26, 35 |
| V2-41 | Held-out quality and scheduler evaluation | not started | Jetson | no | 1.5 | 10 | 07, 30, 40, 56 |
| V2-42 | 24-hour soak and fault campaign | not started | Jetson | no | 1.5 | 4 | 39, 40, 56 |
| V2-43 | Auth/media/update threat tests | not started | Portable + device check | no | 1 | 1 | 17, 38, 39 |
| V2-44 | External install/usability rehearsal | not started | Jetson | no | 1 | 3 | 37, 39, 40 |
| V2-45 | Fix and rerun failed mandatory gates | not started | Jetson | no | 2.5 | 4 | 41–44 |
| V2-46 | Candidate bundle and provenance | not started | Portable + device check | no | 1 | 0.5 | 45 |
| V2-47 | Research report, reproducibility package | not started | Portable | no | 1 | 0 | 30, 41, 45 |
| V2-48 | Final restore/demo/release rehearsal | not started | Jetson | no | 1 | 3 | 46, 47 |
| V2-49 | Versioned extension manifest, evidence validation | done | Portable | yes | 0 | 0 | 02 |
| V2-50 | Camera quality observation lane | not started | Jetson | no | 1.5 | 2 | 09, 49 |
| V2-51 | Bounded temporal sequence rules | not started | Portable | no | 1 | 0 | 13, 49 |
| V2-52 | SQLite incident filters and FTS search | not started | Portable + device check | no | 1 | 0.25 | 14, 17 |
| V2-53 | Review queue, versioned operator feedback | not started | Portable | no | 1.5 | 0 | 18, 22, 52 |
| V2-54 | Optional worker lifecycle, resource manifest | partial | Jetson | no | 1.5 | 1.5 | 29, 49 |
| V2-55 | Mock future adapter, embedding repository | not started | Portable | no | 1 | 0 | 49, 53, 54 |
| V2-56 | Upgrade feature acceptance, manifest freeze | not started | Jetson | no | 1.5 | 1.5 | 50–55 |
| D-1 | Demo runtime loop `sentinel run` (not in backlog) | partial: demo form, portable part done (session 15, D43, D44); device run PENDING | Jetson | yes (demo only) | 2 | 2.5 | demo parts of 05, 09, 10, 13–15, 20, 25, 26 |
| D-2 | Loopback read-only status page (not in backlog) | partial: demo form, portable part done (session 15, D45); device check PENDING (checklist step 5) | Portable + device check | yes (demo only) | 1 | 0.5 | 14, 15 |

| Totals | Claude h | Maintainer Jetson h |
|---|---|---|
| Remaining backlog | 68.5 | 94.25 |
| Demo-only rows D-1, D-2 | 3.0 | 3.0 |
| **Total** | **71.5** | **97.25** |
| Oct 20 demo scope only | 12.0 | 9.5 |

Notes on partial and in-progress rows:

- **In progress:**
  - V2-01: records and checks 1 (`nvpmodel`), 3, 5 and 6 are done (session 3); check 8 and the headless idle baseline (0.970 GB) by the maintainer on 2026-10-03; the bounded Check 9 smoke by the maintainer on 2026-10-03, on the boot before the reboot (session 10 log); two further Check 9 smokes and S1 at `e9af7f4` (session 13 log). Checks 2 (rest) and 9b/U18, the B0 run, trace contract, CPU budget and re-estimate are pending.
  - V2-04: GitHub Actions passed at `32985c2` (maintainer report). The clean-laptop (macOS) run is not done.
- **Done:** V2-03 with synthetic replays only; real-clip replay needs V2-07. V2-49's registry now holds one built-in adapter (`legacy-yolov8n-bytetrack`) and one known profile (`provisional-demo-20261003T085010Z`, D28/D33), recorded in session 13.
- **Partial:**
  - V2-13: restricted-zone and dwell rules in demo form (session 4). Directed line crossing and hysteresis are not done; revalidation after V2-10/V2-11 and calibration of persistence and gap values on labelled replays (V2-07) are pending.
  - V2-14: demo form done (session 4): schema v1 and migrations, the one-transaction record path, correlation, lifecycle and outbox rows. Not done: the guide's other tables (cameras, zones, policies, users, sessions, identities, audit events, clip manifests), WAL checkpoint monitoring, quotas, and revalidation after V2-11.
  - V2-15: demo form done against mocks (session 4). Not done: a real Telegram send from the device (the "device check", which needs the maintainer's bot and chat), the authenticated retry action in an API (only `retry_dead()` exists), outbox quotas, wiring into `sentinel run` (D-1), and revalidation after V2-11.
  - V2-16: only the scene-hazard correlation.
  - V2-20: validated in-memory enrollment only.
  - V2-25: the association and identity core is done; the adapter, alignment, vectorized matching and report are not.
  - V2-05: demo form (session 12, D39): single-reader capture worker, one-frame handoff, OpenCV/FFmpeg software-decode source (TCP, video only, bounded open/read) and `sentinel capture probe`. On the camera (USER-SUPPLIED, session 13): one upstream session from this host, 15 fps 640×480 decode at about 0.07 core-equivalents, sub-millisecond hand-off, open failures retried with backoff, no credentials in output. Not done: a real outage and the read-timeout path on the device, wiring into `sentinel run` (D-1), and everything in V2-05 proper (relay, NVDEC/GStreamer, main stream D26, connection count with live view and recording, restore point).
  - V2-09/V2-10: demo form, portable part (session 13, D40): `PersonTracker` boundary, legacy Ultralytics + ByteTrack backend with the D27 guard, engine pin and offline mode, `sentinel track probe`. Not done: the device check (guard, load, timing, throughput, real tracks), wiring into `sentinel run` (D-1), and V2-09/V2-10 proper (TensorRT without torch, fixed buffers, parity report, separated ByteTrack with low-score boxes, labelled occlusion/resize/reconnect replays from V2-07). Session 21: step 3 met the guard, load and stability criteria (USER-SUPPLIED); detection correctness against the maintainer's observations is PENDING, and the step-3 counters are counts, not accuracy.
  - V2-28: demo form done (session 5): the scene lane routes late evidence (session 2) and `IncidentService.annotate()` stores it on its own incident only (D35). Not done: wiring into `sentinel run` (D-1), display on the status page (D-2), and revalidation with the real scene adapter (V2-26) and after V2-11.
  - D-1: demo form, portable part (session 15, D43): `sentinel run` wiring, the scene server owner (D44), status snapshot, bounded shutdown, startup refusals. Not done: any run on the device (camera, detector, scene server, Telegram), the confirming combined profile (D41), and everything D-1 leaves to V2-11/V2-19/V2-29 (runtime/core split, supervision, admission control).
  - D-2: demo form, portable part (session 15, D45): the loopback-only, read-only page and `/status.json`, served by `sentinel run`. Not done: viewing it on the device over the SSH forward (checklist step 5); it stands in for, and does not replace, V2-17/V2-18 (no sessions, roles, API, media or live video).
  - V2-54: one job at a time, timeout and cancel exist in the scene lane; unload and memory are not done. **Open issue U19:** +0.983 GB remained after every check 8 process had stopped.

## Session 21 log (Claude, 2026-10-04)

**Scope.** The maintainer asked Claude to review checklist step 3 from USER-SUPPLIED MEASUREMENTS, to record the supported results with the unresolved correctness checks marked PENDING, and to implement a narrowly scoped fix for the extra stdout text. Claude ran no camera, GPU or hardware check. It did not rebuild the engine, change any threshold or start step 4. It read the step-3 artifacts in place and changed none of them. The session started at `7dc7a04` with a clean tree, on boot `dbdbdc0c-5c27-469b-ac1e-280a2b1140c7`, the boot step 3 ran on.

**USER-SUPPLIED MEASUREMENTS (maintainer; checklist step 3, `~/sentinel-runs/d1-checklist/3-detector/`).**

| Item | Value |
|---|---|
| Provenance | `utc` 2026-10-04T15:46:50Z; boot `dbdbdc0c…`, started 2026-10-04 20:28:01 IST; commit `7dc7a04bf242e05aae1bc29db89c37996bb51b8a`; `tracked_changes=0`; `display_manager=inactive`; `other_camera_clients=0`; `caffeine_procs=0`; MemFree 3,966,988,288 B |
| Operator note | Rebooted since step 2; SIGTERM sent to caffeine PID 2662 before the provenance (see the environment row) |
| 3a guard | `exit_a=1`; `a.txt` holds only `track probe: libcuda_not_l4t` |
| 3b run | `b_start_utc` 15:49:33Z, `exit_b=0`, `b_end_utc` 15:50:44Z (`b_end_utc` was added by the operator) |
| Frames | captured 888, delivered 858, replaced 29, discarded 1; captured 14.794 fps over 60.026 s |
| Tracking | processed 858, skipped 0, failed 0, failures and error types empty, epoch/failure resets 0, boxes dropped 0; processed 14.981 fps |
| Timing | backend p50/p95/max 37.456/38.165/118.701 ms; ingest-to-result p50/p95 38.466/39.896 ms; ingest interval p50/p95/max 51.3/102.3/219.1 ms |
| Persons (counts, not accuracy) | track IDs 5, confirmed 4, frames with persons 845, max per frame 3 |
| Capture | one connection, one epoch, no stream end or open failure; upstream observed: before 0, min 1, max 1 (59 samples) |
| Load | 8.69 s (includes importing torch and Ultralytics); MemFree 3,926,548,480 → 2,687,381,504 B, MemAvailable 5,403,049,984 → 4,646,977,536 B |
| Userinfo | exact match count 0 across `a.txt`, `b.json` and `b.err` (maintainer; not saved to `provenance.txt`) |

**Maintainer observations of people in view: not supplied.** The maintainer's reply returned the observation template with its placeholders unfilled, so the number of people, when they were visible, whether three were ever visible at once, and any person-like objects are unknown. **Detection correctness is PENDING.** No accuracy is inferred from the counters. Track IDs are tracker identities per epoch: one person who leaves and returns can get several, and a false positive can get one. 845 of 858 frames had at least one person box, and up to 3 boxes appeared at once. Neither is explained until the observations are given.

**Claude's checks and calculations (not measurements).**
- **Frame accounting.** The slot balances: 888 published = 858 delivered + 29 replaced + 1 discarded + 0 pending. The single discard is the pending frame the worker drops when its session ends (`media/capture.py`, the teardown `finally`). With `stream_ends` 0 and one epoch, it is the stop at the end of the run, not an outage. The tracker side balances too: 858 delivered = 858 consumed = 858 processed.
- **Rates.** 14.794 fps is captured frames over the whole 60.026 s window. 14.981 fps is 857 intervals over about 57.2 s, from the first to the last processed frame. The difference is the time before the first frame plus the tail.
- **Replaced frames.** 29 (3.3 %) were replaced although backend p95 (38 ms) is well under the about 67 ms frame interval. Bursty arrival (ingest p95 102 ms, max 219 ms) is the likely cause; this is an inference, not measured. "Few replaced" has no predeclared number.
- **Timing.** The 71 s from `b_start_utc` to `b_end_utc` holds the 8.69 s load, the 60.03 s probe and about 2 s of startup and shutdown.
- **Backend timing.** 37.5/38.2 ms is for information. Check 8's 45/56 ms was measured under different conditions, so the two are not compared.
- **Credentials.** Pattern counts only, no lines printed: none of the three files contains `://` or `@`.
- **`b.json`.** It holds 4 lines (446 characters) of library output, then one complete tracking-result object, then only whitespace. Claude decoded the object from offset 446 in memory, and the values match the maintainer's summary. The file was not modified.
- **`b.err`.** Two lines: torch's warning that NVML cannot be initialized. That is expected on Jetson, which has no NVML. Nothing else.

**Source of the extra stdout text (from the library code, read-only in `~/onvif_env`; no raw log shown).** Both writers run inside `LegacyUltralyticsTracker.load()`, because the TensorRT engine is deserialized by the warm-up `track()` calls:
1. Ultralytics 8.4.25 logs one line announcing the TensorRT load (`ultralytics/nn/backends/tensorrt.py:31`). Its `LOGGER` has a `StreamHandler(sys.stdout)` installed at import (`ultralytics/utils/__init__.py:469`). The level is INFO because `YOLO_VERBOSE` defaults to true; `TRACK_ARGS`' `verbose=False` does not govern this message.
2. TensorRT 10.3 writes three timestamped lines (two information lines and one warning) through the `trt.Logger(trt.Logger.INFO)` Ultralytics creates (`tensorrt.py:50`). The artifacts do not show whether it writes through Python's stdout or straight to descriptor 1, so the fix works at the descriptor level.

The text also contains the engine's absolute path, but no credentials. `sentinel run` loads the same backend before its first JSON line, so step 5's `run.jsonl` would have started with the same lines. `demo_workload` (step 4) reads only `@@EVENT`-prefixed lines and is not affected. `capture probe` loads no model.

**Against step 3's expected results.**

| Criterion | Result |
|---|---|
| 3-0/3-1: headless, MemFree ≥ 1.5e9, no other camera clients, no tracked changes | Met |
| a: `exit_a=1`, `libcuda_not_l4t` | **Met** |
| b: `exit_b=0`, `frames_received`, `failed` 0, empty failures and error types | **Met** |
| `processed` close to `captured`, few `replaced` | Met (858 of 888; 29 replaced; "few" not predeclared) |
| backend p50/p95 (information) | 37.456/38.165 ms |
| `track_ids` > 0 and `frames_with_persons` > 0 if someone walked through | Counters > 0; **whether they match who was in view is PENDING** |
| upstream observed max 1 | Met |
| MemFree before/after the load (information) | 3.927 → 2.687 GB |
| userinfo count 0 | Met (maintainer's exact count; Claude's pattern check) |
| 3-5 summary runs as written | **Not met** at `7dc7a04` (stdout defect); fixed by D48, device verification PENDING |

**Step 3 is recorded in part:** the guard, load and stability criteria are met, and detection correctness is PENDING. Step 4's prerequisite "step 3 passed" is not yet met.

**D48. CLI-owned stdout for machine-readable output (session 21 implementation decision, not yet reviewed).**
- `sentinel track probe` and `sentinel run` run inside `_json_stdout()` (`src/sentinel/cli.py`). It flushes Python's and libc's stdout buffers, duplicates descriptor 1 for the command's own JSON, and points descriptor 1 at stderr. On every exit (return, exception, `KeyboardInterrupt`), its `finally` blocks flush again, restore descriptor 1 and close the duplicate.
- **Process-wide while it lasts:** every thread, library, logging handler and inheriting child writing to descriptor 1 goes to stderr. Logging handlers are not modified. A handler bound to `sys.stdout` still writes to descriptor 1, which is why the switch is at the descriptor level.
- It applies only when `sys.stdout` and `sys.stderr` are descriptors 1 and 2. Otherwise, as with in-process tests that capture `sys.stdout`, nothing is redirected and the behaviour is unchanged.
- **Unchanged:** the JSON content, the refusal JSON on stdout, the stderr error labels and every exit code. `capture probe`, `config validate`, the adapter, thresholds and dependencies are untouched. The 3-5 summary block needs no change.
- **Not covered:** a thread that outlives the command and writes after the restore. `run` and `track probe` stop their threads before returning.

**Tests (`tests/unit/test_cli_stdout.py`, child `tests/unit/cli_stdout_child.py`).** The real CLI runs in a subprocess with fake capture and backend; no camera, GPU or model. The fake load and the first `track()` write to stdout through `print`, a logging handler bound to `sys.stdout` before the command, `os.write(1, …)` and libc's buffered `puts`. The child writes its own markers before the command (still buffered) and after it.
- `track probe`: stdout is the before-markers, exactly one JSON document, then the after-markers. All eight library markers are on stderr.
- `run`: every stdout line between the markers is JSON, from `starting` to `stopped`, and all library markers are on stderr.
- A MemFree refusal still prints its JSON on stdout with exit 1.
- A load failure keeps `track probe: load_failed (RuntimeError)` on stderr with exit 1 and no JSON.
- An unexpected exception and `KeyboardInterrupt` in `track probe` propagate as before. An interrupt during `run` startup still returns 130 with its label.
- In every case, descriptors 1 and 2 are the originals afterwards and the set of open descriptors is unchanged.
- In-process with captured `sys.stdout`, nothing is redirected.
- **Mutation check:** four temporary changes each made 7 of the 8 tests fail. They were: no redirect (the old behaviour), no libc flush, no restore, and the duplicate left open. The fix was restored byte for byte (`cmp`).

### Session 21 verification: exact commands and results

All portable, in the repository `.venv` (Python 3.10.14):
- At `7dc7a04`, before any change: `.venv/bin/python -m pytest -q -x` → 728 passed in 31.52 s.
- `.venv/bin/python -m pytest -q tests/unit/test_cli_stdout.py` → 8 passed in 5.65 s.
- Mutation check (above): each mutation gave 7 failed, 1 passed.
- `.venv/bin/python -m pytest -q tests/unit/test_cli_stdout.py tests/unit/test_track_probe.py tests/unit/test_demo_runtime.py tests/unit/test_scene_admission.py tests/unit/test_portable_imports.py` → 108 passed in 19.47 s.
- `.venv/bin/python -m pytest -q` → 736 passed in 36.55 s.
- `PYTHONPATH=src .venv/bin/python -m sentinel.cli config validate config/default.yaml` → `valid Sentinel configuration (version 1, camera cam-1)`, `no optional adapters configured: core monitoring only`, exit 0.
- **Not run:** any hardware, camera, GPU, model or llama-server; a step 3 rerun; the D48 device verification (PENDING: step 5's `run.jsonl`, or a step 3 rerun in a new directory if the maintainer chooses); step 4; CI; Python 3.12.

## Session 20 log (Claude, 2026-10-04)

**Scope.** The maintainer asked to record checklist step 2 as USER-SUPPLIED MEASUREMENTS and update only the acceptance items it supports. They also asked to record a package install and to prepare step 3. Docs only. Claude ran no camera, GPU or hardware check, changed no service, cleared no cache and removed no package. It started at `55ab3b2` with a clean tree on boot `201a195f-98a2-4cef-b6e3-3955f6f33f2b`, the same boot as steps 1 and 2.

**Session start (Claude's read-only checks, before the runs).** At 11:56Z, nothing matched `sentinel.cli`, `capture probe` or `track probe`, and no v1, llama-server or ffmpeg/ffprobe process was running. Only counts and process names were checked, never arguments. No step-2 directory existed. `display-manager` was inactive. The session-19 commit `55ab3b2` (docs only) is one ahead of the maintainer's last-known `85fb7e7`. Step 1's `provenance.txt` and `settings.txt` match the session 19 record.

**USER-SUPPLIED MEASUREMENTS (maintainer; checklist step 2).** Both runs were on boot `201a195f…`, at commit `55ab3b26fe918e5613897f891f5a37e0d5f40db1` with `tracked_changes=0` and `display_manager=inactive`. Both used `capture probe … --seconds 180`.

| | Run 1: `~/sentinel-runs/d1-checklist/2-capture-outage/` | Run 2: `…/2-capture-outage-power-20261004T141037Z/` |
|---|---|---|
| Physical outage | **None performed** (maintainer). The prompts printed, but the camera stayed powered. Steady capture only; **not** outage or recovery evidence. | **Camera power** cut at the `CUT` prompt and restored at the `RESTORE` prompt, each within about 2 s (operator confirmation, session 20) |
| Provenance `utc` | 13:45:52Z | 14:10:37Z |
| Prompt times (terminal prompts, not independently measured physical actions) | probe start 13:59:33Z, cut prompt 13:59:53Z, restore prompt 14:00:08Z | cut prompt 14:10:57Z, restore prompt 14:11:12Z; `probe_start_utc` not recorded |
| Client check | `other_camera_clients=0` | **Not run** (operator). See the upstream counts below. |
| `exit`, `status` | 0, `frames_received` | 0, `frames_received` |
| `connects` / `epochs` | 1 / 1 | 2 / 2 |
| `stream_ends` / `open_failures` | 0 / 0 | 1 / 3 |
| `worker.problem` (last recorded) | none | `open_failed` |
| Frames captured / fps over the run | 2,690 / 14.934 | 2,072 / 11.503 |
| Upstream connections (this host, `/proc/net/tcp`) | observed: before 0, min 1, max 1 (180 samples) | observed: before 0, min 0, max 1 (180 samples) |
| Ingest interval p50 / p95 / max | 52.2 / 147.1 / 399.8 ms | 50.8 / 148.7 / 5,020.1 ms |
| Hand-off age max | 0.68 ms | 0.681 ms |
| `pts_quality: none` | 1 (`missing`) | 2 (`missing`) |
| CPU (core-equivalents), max RSS | 0.079, 84,783,104 B | 0.066, 84,951,040 B |
| `userinfo_lines` | no line recorded; see Claude's check | 0, printed on screen and not saved to `provenance.txt` (operator) |

**Claude's checks and calculations (not measurements).**
- **Credentials.** Claude checked pattern counts only and printed no lines. Neither `c.json` contains `://` or `@`. Run 1's `c.err` is empty. Run 2's `c.err` has 8 lines and no `scheme://…@` userinfo. Every `@` in it is an OpenCV (7) or FFmpeg (1) log prefix. Both `c.err` files stay local.
- **Client check substitute (run 2).** `upstream before 0, max 1` means this host had no connection to the camera endpoint before the probe, and only one during it. As in session 13, clients on other hosts are invisible here. `min 0` matches the outage.
- **Outage length.** At run 1's rate (14.934 fps, same boot), 180.133 s would give about 2,690 frames. Run 2 is short by about 618, about 41 s without frames. Power was off for about 15 s (prompt to prompt), so frames came back roughly 26 s after the restore. That fits a camera reboot plus the 1/2/4/8/15 s backoff, and it left about 2 minutes of capture after recovery. It is approximate and assumes a steady rate otherwise. The `c.json` file times are consistent with the probe starting at about 14:10:37Z: it was written 180 s later.
- **The 5,020 ms ingest interval** is between consecutive frames of one epoch (`frame_seq` restarts each epoch). It is not the 41 s gap. The summary does not show its cause. It is not a step-2 criterion, but it is above the 2 s stale threshold. D-1's live state must mark capture stale during such a gap, and that comes with step 5.
- One `pts_quality: none` (`missing`) per epoch, as in run 1.

**Against step 2's expected results (current checklist).**

| Criterion | Run 2 | Result |
|---|---|---|
| `exit=0`, `status frames_received` | 0, `frames_received` | Met |
| Recovery: `connects` ≥ 2 and `epochs` ≥ 2 | 2 and 2 | **Met** |
| Read timeout: `stream_ends` ≥ 1 | 1 | **Met** (path exercised; the summary does not time the 5 s) |
| Reopen during the outage: `open_failures` ≥ 1 | 3 | Met |
| `last_problem` informational | `open_failed` | As expected after a power cut |
| `upstream observed max 1` | observed, max 1 | Met |
| `userinfo_lines=0` | 0 (operator, on screen) + Claude's pattern check | Met |
| 2a client check | not run; upstream before 0 / max 1 instead | Deviation recorded |

**Step 2 is done.** The session 13 V2-05 demo-form rows are now:
- "Reconnect after an outage": **met** (USER-SUPPLIED, run 2).
- "Read-timeout path": **met** (exercised).
- "Open failure and backoff": also seen during a real outage.

**Still open:**
- V2-05 full acceptance (D23): relay, NVDEC, the session count with live view and recording, and a restore point.
- The camera's boot time (not measured).
- An uplink-cut variant (not run).
- FFmpeg-stderr handling for D-1 (session 13).
- Checklist steps 3–5, and every other pending hardware gate.

**Environment change (USER-SUPPLIED; Claude's read-only observations).**
- **Maintainer:** "installed the caffeine package on the Jetson". Whether it is running is unverified.
- **apt history:** `apt install caffeine`, 2026-10-04 13:45:03–13:45:06Z (19:15 IST). It installed `caffeine` 2.9.12-1 plus three automatic dependencies: `python3-ewmh` 0.1.6-3, `python3-xlib` 0.33-2 and `gir1.2-ayatanaappindicator3-0.1` 0.5.93-1build3. Nothing was upgraded or removed, and no CUDA or L4T package was involved.
- **Autostart:** it adds `/etc/xdg/autostart/caffeine.desktop`, which starts only in a graphical session. `display-manager` was inactive.
- **No process seen:** after the runs, Claude saw no caffeine process (name-only check).
- **Timing:** the install came 49 s before run 1's provenance, so both step-2 runs ran with the package installed. Its running state during them was not observed.
- **Effect:** it is not a Sentinel dependency and nothing in the repository uses it. For steps 3–5, provenance now records a caffeine process count, so measurement conditions show whether it ran.
- **No action taken:** nothing was removed and no service was changed.

**Instruction changes (docs only).**
- The "before each step" block and 2a now refuse an existing step directory, so earlier results cannot be overwritten.
- The client check now prints a count instead of `pgrep -fa`, so process arguments (which can include a camera URL) are never shown.
- Step 3 changes:
  - The display-manager stop is only for when it is active.
  - The URL is re-entered in a separate hidden paste, because 2d unsets it. The URL check runs before the guard, so 3a needs it too.
  - 3a runs with `env -u LD_PRELOAD`, so an exported preload cannot mask the guard.
  - A numbers-only summary is added, and the userinfo check now covers every output file.
- `bash -n` passed on the new blocks. Nothing was run against the camera or GPU, and the portable suite was not rerun (docs only; 728 passed at `85fb7e7`).

## Session 19 log (Claude, 2026-10-04)

**Scope.** Maintainer request: record checklist step 1 as USER-SUPPLIED OBSERVATIONS, mark only step 1 complete, and prepare the step 2 instructions. Docs only. Claude ran no camera, GPU or hardware check, changed no service and cleared no cache. Started at `85fb7e7` with a clean tree.

**USER-SUPPLIED OBSERVATIONS (maintainer; checklist step 1).**

| | |
|---|---|
| UTC | 2026-10-04T10:02:46Z |
| Boot ID | `201a195f-98a2-4cef-b6e3-3955f6f33f2b` (the current boot when Claude recorded it) |
| Commit | `85fb7e70eac5d7ce3d335d3595881fea09fe096b`, `tracked_changes=0` |
| Display manager | inactive |
| `settings_sync` | `True` |
| With `YOLO_OFFLINE=true` | `ONLINE=False`, `events_enabled=False` |
| Adapter | sets `YOLO_OFFLINE` before importing Ultralytics |
| 1b (`sync=False`) | not performed |
| Artifacts (local) | `~/sentinel-runs/d1-checklist/1-ultralytics/provenance.txt`, `settings.txt` |

These match step 1's expected results, so step 1 is done. v1, `demo_workload.py` and check 8 do not set `YOLO_OFFLINE`. With `sync` still `True`, they send analytics when online (step 1's note); that stays the maintainer's choice (1b).

**Step 2 revised (from the code; not a measurement).**
- **The window.** A power cut makes the camera reboot, and its boot time is unmeasured. After a failed open, the worker waits 1, 2, 4, 8, then 15 s, and each failing open can take up to 10 s. Once the camera is up, the next successful open can therefore be about 25 s away, so session 15's 90 s window could end before recovery. It is now 180 s.
- **The timing.** The probe runs in the background while the shell prints `CUT` and `RESTORE` prompts (20 s and 35 s in) and records their UTC times. The old command held the terminal, so the times could not be written while it ran.
- **The credential.** Credential entry is its own paste (`read -rsp`).
- **The summary.** A numbers-only summary line is printed. The userinfo check now covers `c.json` too.
- **Expected results, from the code.** `epochs` counts only epochs whose frames reached the probe, so `epochs` ≥ 2 shows frames after the restore. `exit=0` alone does not show recovery, and `worker.problem` keeps the last problem after recovery.
- **The stop conditions** add: `epochs` < 2 is reported as not shown, or as inconclusive if the camera was still booting.
- **Checked.** Each block passed `bash -n`. 2c and 2d ran against a stub probe in a scratch directory, with a synthetic URL and shortened sleeps:
  - A good stub printed the summary line. The provenance file recorded `probe_start_utc`, `cut_prompt_utc`, `restore_prompt_utc` and `exit=0` in that order, then `userinfo_lines=0`. The URL variable was empty after 2d.
  - A failing stub that wrote the synthetic URL to stderr and exited 1 produced `exit=1`, `status no_frames … epochs 0` and `userinfo_lines=1`.
  - No real camera, URL or probe was used. Docs-only change: the portable suite was not rerun (728 passed at `85fb7e7`). `git diff --check` was clean.

## Session 18 log (Claude, 2026-10-04)

**Scope.** Maintainer request: review `268e905` (session 17) and complete its portable verification; fix only what that review requires. No new package, hardware, cache drop, service change or push. Started at `268e905` with a clean tree. The session 17 process had been left open in the same checkout. The maintainer confirmed it was closed; it was idle (about 6 s of CPU in 5 min, no tree changes) before this session edited anything.

**Checks before any change, at `268e905`:** `test_step4.py` 49 passed; `test_scene_admission.py` 60 passed; full suite 723 passed; `config validate` ok; `git diff --check 7f06055 268e905` clean.

**Review.** Read the criteria module, the workload and profiler changes, the step-4 orchestration (identity prerequisite and end check, post-run wait, kernel coverage, peak from both samplers, run validity) and the startup identity checks. One defect; the rest held.

- **Defect (fixed): sanitized error names could read as zero errors.** Step 4 runs the profiler with `--sanitized-logs`. `sanitize_diagnostic()` keeps a dict key only if it is allowlisted, ends in `Error`/`Exception`, or is `HTTP nnn`. The workload counts face and scene errors by exception class name, so names such as `RemoteDisconnected` and `IncompleteRead` (`http.client`, typical when llama-server drops a connection) or `OutOfMemory` were dropped, and the `errors` map arrived empty. F and V1 gated only on that map being empty. Reproduced with the real functions: face errors `{"OutOfMemory": 3}` and scene errors `RemoteDisconnected`/`IncompleteRead` were sanitized to `{}`, and F and V1 both returned `pass` while `transport_errors` was 2.
- **D47 amendment (session 18; not yet reviewed by the maintainer).**
  - The workload's face summary adds `error_count`, the total counted before sanitizing, and the sanitizer keeps it.
  - **F** requires an integer `error_count` equal to 0, as well as an empty `errors` map. Without the count, F is `unavailable`.
  - **V1** requires integer `client_timeouts`, `http_errors` and `transport_errors`, all 0, as well as an empty `errors` map. These totals were already counted before sanitizing. A missing one makes V1 `unavailable`.
  - A registry profile's `face_errors` is the workload's `error_count`. Thresholds, the criteria ID and every other criterion are unchanged.
- **Observed, not changed:**
  - `cache_evidence()` counts a missing steady update count as 0 even when there is no steady window. That cannot make a run eligible, because S is then not `pass`.
  - The unchanged guard stops at 4.8 GB pressure, so a run above 4.8 GB stops and is invalid (R) before M1's 5.0 GB or M2's 5.4 GB limit can fail it. This is the guard's existing, documented role.

| File | Change |
|---|---|
| `benchmarks/runner/demo_workload.py` | Face summary `error_count`. |
| `benchmarks/runner/demo_profile.py` | Sanitizer keeps `error_count`. |
| `benchmarks/runner/step4_criteria.py` | F and V1 gate on integer error totals; missing totals are `unavailable`. |
| `src/sentinel/adapters.py` | Comment: `face_errors` is the workload's `error_count`. |
| `tests/unit/test_step4.py` (+5) | Fixture with `error_count`; F fails on a count or is unavailable without one; V1 fails on `transport_errors` or is unavailable without `http_errors`; an end-to-end test where the real workload summary, sanitized by the real sanitizer, fails F and V1. All 5 fail against `268e905`'s criteria module (checked by swapping it in, then restored). |

### Session 18 verification: exact commands and results

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_step4.py
# 54 passed in 3.69s, exit 0 (49 at 268e905)
.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_scene_admission.py
# 60 passed in 0.61s, exit 0
.venv/bin/python -m pytest -q -p no:cacheprovider
# 728 passed in 32.17s, exit 0 (723 at 268e905)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
/usr/bin/python3 and ~/onvif_env/bin/python: compile() of demo_workload.py, step4_criteria.py, demo_profile.py
# 3.12.3 compile ok; 3.10.14 compile ok
git diff --check
# no output, exit 0
```

- **Authorization is separate from implementation.** The criteria are implemented and tested portably. Nothing in this session authorizes running step 4. Its prerequisites are unchanged: checklist steps 1 and 3 passed, and the maintainer explicitly approves the run and the D47 criteria, now including this amendment.
- **Not run:** the identity snapshot on real files, Check 9, step 4, any journal query, GPU, camera, model or Telegram work; CI. Step 4 and any acceptance commit stay PENDING, and no profile is accepted.

## Session 17 log (Claude, 2026-10-04)

**Scope.** Maintainer approval: implement, portably, the proposed `step4-combined-cache-off-v2` criteria and the guarded step4 mode, with corrections:
1. reliable monotonic steady boundaries;
2. a fail-fast shell flow that passes the exact Check 9 result;
3. a real 60 s post-run wait and bounded kernel queries with explicit coverage;
4. clip identity by its recorded hash, with identity hashing before the cache drop and its timing recorded;
5. cache evidence from the installed build and the running process, with completed work and both frame ages recorded.

Guard thresholds and cleanup stay unchanged. No V2-29, no hardware, cache eviction, service change, reboot or push. Started at `7f06055` with a clean tree.

**Clip identity (Claude, read-only).** The current `~/clips/two_people_doorway_60s_2026-09-29.mp4` has the SHA-256 recorded in the excluded local notes for check 8's input (`room_static`, formerly `one_person`, 1,770,670 B). It is the same bytes. The hash itself stays in the local notes and in local run records only.

- **D47. Step-4 criteria v2 and the guarded step4 mode** (maintainer approval, 2026-10-04; implementation details are session 17 decisions, not yet reviewed).
  - **Steady interval.** It starts at the workload's `steady_boundary` start and ends at its `steady_boundary` end. Both are `time.monotonic()` stamps. The end is stamped as soon as the steady loop ends, before any worker thread is joined, and the workload's summary is taken at that moment. An orchestrator `stop_boundary` (memory floor, abort, or the interruption time captured in the signal handler) ends it earlier. CSV phase labels are not used: rows after the end, including those still labelled `steady`, are excluded from every steady statistic (maximum, median, p95, time above target, swap deltas, trend, tegrastats, prompt-cache window). The interval reports its duration, coverage, largest gap and what ended it. Without a start or any end it is unavailable, and so is acceptance. The guard's own stop marks the run invalid (criterion R).
  - **Workload evidence.**
    - Unique detected frames per second, and the lowest complete 10 s window.
    - **Replay scheduling age**: from when a live camera would have delivered the frame to the result. It includes any backlog. This one is gated.
    - **Decode-to-result age**: actual ingest-to-result in the replay. Reported only.
    - Neither is camera-to-result (U3).
    - Face runs, rate and errors.
    - Scene attempts, completed requests, client timeouts (30 s), HTTP and transport errors, completions over 8 s, finish reasons, strict-valid and rejected counts by reason, and latency.
    - A SHA-256 fingerprint of the request the workload sends: prompts, image size, JPEG quality, token and temperature limits, model name, schema.
  - **Cache evidence.** "Disabled, verified" requires all five sources to agree:
    1. the manifest flag (`cache_ram_mib` 0);
    2. `--cache-ram 0` in the running server's `/proc/<pid>/cmdline`, captured once it was ready;
    3. the installed build's files containing the option text (llama-server also exits on an unknown option);
    4. the startup log line "prompt cache is disabled";
    5. zero cache-state updates in the steady interval, counted only when the startup line was captured with receipt times.

    Missing telemetry is "unverified", never zero updates. A running server without `--cache-ram` counts as "enabled".
  - **Identity.**
    - **The snapshot.** `operator_check.py --step4-identity` (read-only) runs **before** the operator's cache drop. It hashes the clip, the llama-server binary, every llama/ggml/mtmd library beside it (real files), the LLM, the projector, the engine and the face weights, and records per-file and total hashing time, the boot ID and the commit. The clip must equal its recorded hash, which is passed from the local notes and never committed. Only a summary is printed; the full result stays in the private `result.json`.
    - **At step-4 start.** The step4 mode refuses unless the snapshot is complete, from this boot and commit, finished before the Check 9 result, and every file still has its recorded name, size and mtime.
    - **At the end.** The profiler rehashes all of them, the build and the clip included. Identity is `verified` only if every end hash equals the snapshot.
    - **Before acceptance.** Checklist 4f rehashes again.
  - **Kernel evidence.**
    - **When.** After the owned process group's cleanup, the mode waits 60 s (interruptible, and recorded), then writes an end marker.
    - **What it queries.** Three bounded `journalctl` queries (10 s, 1 MiB each) for the **manifest's boot ID**, which must still be the current boot because the journal is volatile:
      1. kernel records from run start − 1 s to run end + 60 s;
      2. the run's own `sentinel-step4` markers, written with `logger` before the child starts and after the wait;
      3. journald's own messages in the interval.
    - **When coverage counts as observed.** Only if:
      - every query completed untruncated and parsed;
      - both markers are readable and bracket the interval, which proves the journal is readable and retained from start to end;
      - journald logged no loss (missed, suppressed, rate-limited, full, truncated, corrupt).
    - **Otherwise.** Coverage is `truncated`, `uncertain` or `unavailable`, and the OOM/NvMap counts are absent (lower bounds at most), never zero.
  - **Step-4 mode.**
    - **Admission.** `--execute-workload step4` requires:
      - the inspection to pass;
      - an **explicit** `--check9-report` (`--latest-check9-report` is refused) for a successful Check 9 at the same boot and commit;
      - the identity report;
      - `--step4-clip`, `--operator-dropped-caches` and `--confirm-step4-prerequisites`;
      - the guard's baseline headroom.
    - **The child.** `demo_profile.py --clip <clip> --no-evict --sanitized-logs --llama-cache-ram 0 --face-hz 1 --scene-interval-s 4 --baseline-s 30 --settle-s 15 --warmup-s 120 --steady-s 600 --llama-timeout-s 60 --load-timeout-s 90 --min-free-gb 3.5`.
    - **Guard and cleanup.** The unchanged `PressureGuard` stops at 4.8 GB pressure, 1 GiB MemFree, 2 GiB MemAvailable, any swap-counter change, or a sampling gap over 0.5 s. The child deadline is 1,200 s, and cleanup sends TERM then KILL to the owned process group.
    - **The report.** Every criterion's status, `eligible_for_maintainer_review`, `blocking` and `accepted: false`. Exit 0 only when eligible.
  - **Startup checks (`sentinel run --scene`), exactly:**
    - the registry entry's evidence (`accepted_profile_problem()`);
    - the server flags and the scene interval;
    - the request fingerprint against `SCENE_REQUEST_SHA256`;
    - **SHA-256** of the llama-server binary and of every profiled build library up to 32,000,000 B (about 20 MB in total);
    - **name, size and mtime only** for larger libraries (libggml-cuda, 200 MB) and for the model and projector files;
    - the detector engine's pin, through the entry, while the engine itself is hashed when the detector loads;
    - after the server is ready, every mapped llama/ggml/mtmd library must come from the binary's directory (`libraries_not_profiled` otherwise).

    **Limitation:** metadata checks do not detect a same-size replacement that keeps its mtime (a test demonstrates this). An accepted profile must carry `STARTUP_IDENTITY_LIMITATION`. The step-4 snapshot, the end-of-run hashes and the pre-acceptance rehash cover the measured run.
  - **Acceptance (amends D46).** A run that is eligible goes to the maintainer for review; nothing at runtime marks a profile accepted. The acceptance commit copies into a `ResourceProfile`:
    - the identity: run directory, commit, boot ID, flags, `cache_ram_mib` 0, interval, `llm`/`mmproj`/`llama_server`/`llama_libraries` with SHA-256, `engine_sha256` and `scene_request_sha256`;
    - the evidence fields: `cache_verdict`, `steady_status`, `steady_coverage`, `steady_max_bytes`, `steady_seconds_above_target`, `peak_bytes`, `steady_slope_bytes_per_min`, `unique_fps`, `min_window_fps`, `schedule_age_p95_ms`/`p99_ms`, `face_hz`, `face_errors`, `scene_attempts`, `scene_valid`, `scene_truncated`, `scene_errors`, `scene_over_deadline`, `kernel_coverage`, `oom_candidates`, `nvmap_candidates`, `identity_status`, `replay_clip_verified`, `gpu_guard_ok`;
    - `criteria_id`, `criteria_passed`;
    - `limitations` including the startup limitation.

    `accepted_profile_problem()` rechecks every threshold.

**Final criteria `step4-combined-cache-off-v2`.** Demo profile only: a 640×480 replay at 15 fps. It does not test the guide's 1080p core-throughput gate or any beta gate.

| # | Criterion | Requirement |
|---|---|---|
| R | Valid run | Guard `completed` with cleanup clear; explicit same-boot Check 9; drop declared; profile `complete`; headless; no dev tools; no tracked changes; commit equal to the identity's |
| S | Steady interval | `complete` from monotonic boundaries (600 s), coverage ≥ 0.95, no gap > 1 s |
| M1 | Steady pressure | Every steady sample ≤ 5,000,000,000 B; 0 s above (time above and its share reported) |
| M2 | Run peak | Sampled cold-load/runtime peak ≤ 5,400,000,000 B, the higher of the profiler's and the guard's samplers |
| M3 | Trend, swap | Steady slope ≤ +10,000,000 B/min; 0 pages swapped in during steady (unavailable if not sampled) |
| T1 | Throughput | Mean ≥ 14.5 unique detected frames/s; every complete 10 s window ≥ 13.5/s |
| T2 | Frame age | Replay scheduling age p95 ≤ 150 ms, p99 ≤ 250 ms; decode-to-result age reported |
| T3 | Processed/decoded | ≥ 0.99 (supplementary) |
| F | Face | ≥ 0.95 Hz achieved; 0 errors |
| V1 | Scene requests | ≥ 140 attempts; 0 HTTP, transport or client-timeout errors; 0 completions over 8 s |
| V2 | Scene completion | Every completion `stop` (0 truncated or other); strict-valid ≥ 0.95 of attempts; rejections by reason; structural validity only, not accuracy |
| G | GPU | All layers and the vision encoder on CUDA0, L4T libcuda only; workload `cuInit` 0 |
| C | Prompt cache | Flags exactly `--n-gpu-layers 999 --ctx-size 2048 --parallel 1 --cache-ram 0`; cache verdict `disabled_verified` |
| K | Kernel | Coverage `observed` (boot plus interval plus markers plus no journald loss); 0 OOM, 0 NvMap candidate lines |
| I | Identity | Snapshot before the drop; clip equals its recorded hash; unchanged at start; end hashes equal |

Any criterion that is not `pass` blocks eligibility. When R fails, the others become unavailable. Recorded but not gated: steady median and p95, per-process PSS, latencies, tegrastats, tokens, unload residue (U19).

| File | Change |
|---|---|
| `benchmarks/runner/step4_criteria.py` (new) | Constants, `steady_interval()`, `cache_evidence()`, `evaluate_profile()`, `evaluate()`. Standard library only. |
| `benchmarks/runner/demo_workload.py` | Steady boundaries; summary before joins; `stopping` labels for teardown; frame ages, windows, scene counters; request fingerprint. |
| `benchmarks/runner/demo_profile.py` | `stopping` phase; build files and option evidence; running command line; stop boundaries; end-of-run build and clip hashes; sanitizer keys; summary on the monotonic interval; cache and GPU evidence; profile-side criteria. |
| `benchmarks/runner/operator_check.py` | `--step4-identity`; `--execute-workload step4`; identity prerequisite; post-run wait; markers; bounded kernel queries; end-of-run identity check; criteria report. Guard and cleanup unchanged. |
| `src/sentinel/adapters.py` | v2 fields and checks; `STARTUP_IDENTITY_LIMITATION`; `STARTUP_HASH_LIMIT_BYTES`. |
| `src/sentinel/demo_runtime.py`, `scene/server.py`, `scene/llama_server.py` | Startup identity checks; mapped-library check; `request_fingerprint()` and `SCENE_REQUEST_SHA256`. |
| Tests | `test_step4.py` (+49); updates in `test_scene_admission.py`, `test_scene_server.py`, `test_demo_measurements.py`, and `conftest.py` (synthetic v2 fixture). |

**Tests and checks.**
- **New tests:**
  - teardown excluded;
  - time above target measured while p95 stays under it;
  - early stop truncates, and missing boundaries are unavailable;
  - each cache source required;
  - the workload ends steady before joins and excludes work finished in teardown, labelling it `stopping`;
  - error and window counters;
  - each criterion on its own;
  - an invalid run or uncertain evidence is never eligible;
  - the summary on the interval, and without an end;
  - the sanitizer;
  - identity snapshot, mismatch and missing file;
  - nine step-4 refusals before any process;
  - a completed run waits, then queries the recorded boot and interval;
  - six kernel coverage failures;
  - an interrupted wait;
  - a guard stop;
  - a file changed during the run;
  - CLI exit and hash privacy;
  - startup identity checks and the limitation demonstration.
- **Mutation sweep** (one-off): 22/22 caught after strengthening one test.
- **Checklist flow:** 4a–4f (without sudo) ran against a stub `operator_check` in a scratch repository. That found and fixed one defect: `set -e` is ignored in a subshell on the left of `||`. The blocks now use `( … ); echo "<block>_exit=$?"`. Every case behaved as intended (the table above). All checklist shell blocks pass `bash -n`.

### Session 17 verification: exact commands and results

```bash
.venv/bin/python -m pytest -q tests/unit/test_step4.py
# 49 passed, exit 0 (3 consecutive runs)
.venv/bin/python -m pytest -q
# 723 passed, exit 0 (647 before)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m py_compile benchmarks/runner/*.py   # system Python 3.12.3: ok
PYTHONDONTWRITEBYTECODE=1 ~/onvif_env/bin/python -m py_compile benchmarks/runner/demo_workload.py benchmarks/runner/step4_criteria.py   # 3.10: ok
/usr/bin/python3 benchmarks/runner/demo_profile.py --help; .venv/bin/python benchmarks/runner/operator_check.py --help   # both parse
git diff --check -- src tests docs benchmarks
# no output, exit 0
```

- **Not run:** the identity snapshot on the real files (it would fill the page cache), Check 9, step 4, any journal query, any GPU, camera or model work; CI. Console output for a full run is estimated at about 300 KB, under the runner's 1 MiB limit (not measured). Step 4 and any acceptance commit stay PENDING, and no profile is accepted.

## Session 16 log (Claude, 2026-10-04)

**Scope.** Maintainer instruction: implement the narrow `--scene` admission fix. Scene analysis stays off by default. `--scene` requires an explicitly accepted combined profile, measured with `--cache-ram 0` and matching the selected runtime configuration, through the existing profile/registry mechanism. The old provisional cache-enabled profile and missing, pending, failed or mismatched evidence are rejected before llama-server starts or scene resources load, with a sanitized reason. Core-only operation stays available. No real profile is marked accepted, no evidence is fabricated, and there is no bypass. Tests may use explicitly synthetic fixtures. Also: define how a real profile becomes accepted after step 4, and keep the in-memory pending-write limitation visible in this file and on the status page. No push, hardware, cache eviction, service change or further package. Started on `v2-beta` at `7f5f9f3`, two commits ahead of the local `origin/v2-beta` (`ea95571`), with a clean tree.

- **D46. Scene admission needs an accepted, matching combined profile** (maintainer instruction, 2026-10-04; the implementation details are session 16 decisions, not yet reviewed).
  - **Registry.** `sentinel.adapters.RESOURCE_PROFILES`, the existing registry, now holds `ResourceProfile` records instead of bare IDs: a status (`provisional`, `accepted`, `pending` or `failed`) and the evidence copied from the run's manifest and summary. It is a read-only mapping of frozen records, static in code. `KNOWN_RESOURCE_PROFILES` is derived from it (provisional or accepted), so the detector's admission is unchanged.
  - **Check 8's entry.** It is recorded from its manifest (run directory `demo-profile-20261003T085010Z`, commit `d85eb1e2…`, boot `2dfc802c…`, file facts and SHA-256) as `provisional` with the default prompt cache (`cache_ram_mib` None). Its memory figures are left unset rather than retyped from rounded values.
  - **The scene adapter's spec** (`llama-lfm2-vl-scene`) has `requires_accepted_profile`. `resolve()`, and therefore `sentinel config validate`, reports it unavailable unless `accepted_profile_problem()` passes. That needs, in order:
    1. a named profile in the registry, filed under its own ID;
    2. status `accepted` (a provisional, pending or failed profile is refused with its status named);
    3. `cache_ram_mib` 0;
    4. `criteria_id` equal to `step4-combined-cache-off-v1`;
    5. `criteria_passed` true;
    6. a `demo-profile-YYYYMMDDThhmmssZ` run directory, a 40-hex commit and a boot ID;
    7. a recorded GPU guard pass and prompt-cache-disabled record;
    8. run peak, steady p95 and steady slope present and within 5.4e9 B, 5.4e9 B and 1e7 B/min;
    9. zero scene request errors.
  - **Runtime match.** `sentinel run --scene` then requires the entry to match the selected runtime: the server flags (`--n-gpu-layers 999 --ctx-size 2048 --parallel 1 --cache-ram 0`), `scene.interval_s`, and the chosen `--scene-model`/`--scene-mmproj` by name, size and modification time from metadata, plus the pinned detector engine's SHA-256. The model files are not hashed at startup, because reading 1.26 GB would fill the page cache right before the MemFree precheck (U18); the entry keeps the run's SHA-256 for audit. The llama.cpp build is not checked at startup.
  - **Before any scene resource.** The check runs before the database opens, before the scene server factory or worker is created, and before the detector loads. The refusal is `run: scene_not_admitted: <reason>`, naming the missing prerequisite with profile IDs and file base names only.
  - **No bypass.** `sentinel run` has no option or configuration setting that skips or changes this check. The registry is a parameter of `assemble()`/`resolve()` only, so that tests can pass synthetic fixtures.
  - **How a real profile becomes accepted (procedure).** Only through a **separate, maintainer-approved commit**, after step 4 ran and the maintainer judged it against `step4-combined-cache-off-v1`. That commit adds one `ResourceProfile` entry with:
    - `status=ProfileStatus.ACCEPTED`;
    - `run_dir` (the step-4 directory under `~/sentinel-runs/`);
    - `commit` (the manifest's full `repository.commit`) and `boot_id` (the manifest's `boot_id`);
    - `llama_flags`, `cache_ram_mib=0` and `scene_interval_s` from the manifest;
    - `llm`, `mmproj` and `engine_sha256`, from the manifest's file facts and SHA-256;
    - `criteria_id`, `criteria_passed=True`, `gpu_guard_ok`, `prompt_cache_disabled`, `peak_bytes`, `steady_p95_bytes`, `steady_slope_bytes_per_min` and `scene_errors`, from the run's results;
    - a note citing the status-record section that records the run as a USER-SUPPLIED MEASUREMENT.

    **Amended by D47 (session 17):** the entry now carries the v2 fields (see the session 17 log), and `criteria_id` must be `step4-combined-cache-off-v2`.

    The same commit updates `test_the_runtime_registry_holds_no_synthetic_and_no_admissible_scene_profile`, whose second assertion states that no profile is admissible today. A run that failed may be recorded as `failed`. **Nothing at runtime creates, edits or accepts a profile**, and no runtime file, flag or database row can.
  - **Unchanged.** Scene analysis is off by default, and core-only runs need no scene profile. A scene manifest that names the provisional profile without `--scene` is harmless.

**Pending-write limitation visible.** The runtime snapshot's `components.incidents` now carries `pending_limit` 256 and a fixed `durability` text: "rule observations waiting to be recorded are held in memory only (at most 256); they are lost if the process stops abruptly: not a crash-safe spool". The status page always shows "Waiting to be recorded: N" and the limitation line, with a fixed fallback while the runtime is starting. This file shows it in the "Known limitation" row of "Position" and in D43.

| File | Change |
|---|---|
| `src/sentinel/adapters.py` | `ProfileStatus`, `FileFacts`, `ResourceProfile`, `RESOURCE_PROFILES`, step-4 constants, `known_profile_ids()`, `accepted_profile_problem()`; `AdapterSpec.requires_accepted_profile`; `resolve(profiles=...)`. |
| `src/sentinel/demo_runtime.py` | `file_facts()`, `profile_mismatch()`; `scene_admission(config, scene, profiles=...)`; `assemble(..., profiles=...)`; `PENDING_DURABILITY` in the snapshot. |
| `src/sentinel/status_page.py` | The pending count and limitation line. |
| `tests/unit/conftest.py` | The `accepted_scene` fixture: a SYNTHETIC accepted profile (`synthetic-test-cache-off`, a run directory dated 2099, an all-zero commit and boot ID) with matching temporary model files. It is passed explicitly, never registered. |
| `tests/unit/test_scene_admission.py` (+36) | See below. |
| `tests/unit/test_demo_runtime.py`, `test_llama_server.py`, `test_status_page.py` | Tests that relied on the provisional profile admitting scene now use the synthetic fixture. Session 14's registry test now asserts that the provisional profile is refused. One new status-page test (+1). |

**Tests (+37).**
- **Default off:** core runs, with no scene manifest or with a provisional one.
- **CLI refusal:** with `subprocess.Popen` patched to fail, the CLI `--scene` refusal starts no process and constructs no detector; the exact message is checked.
- **No bypass:** `run --help` offers no option matching allow, force, skip, unsafe, override, profile or admit.
- **Rejected evidence:** the provisional profile; 23 parametrized cases (pending, failed, provisional with cache 0, the default cache, 8192 MiB, wrong or absent criteria ID, a pass false or absent, a bad run directory including `../`, a short commit, no boot ID, guard false or absent, cache-disabled absent, peak absent or over, p95 over, slope absent or over, scene errors 1 or absent); an unknown or unnamed profile; a disabled or absent manifest; an entry filed under another ID.
- **Configuration mismatches:** flags, interval, another model file, a missing projector, a projector size change, a model mtime change, another engine.
- **Every rejected case** asserts that no scene server, scene worker or detector load happened, that no database was created, and that the reason contains no path.
- **Admission:** with the synthetic fixture, the server, worker and detector start in order and scene analysis is available.
- **Runtime registry:** no synthetic entry, no admissible scene profile, immutable mapping and records; the provisional profile still admits the detector, while `resolve()` explains the scene refusal.
- **Status page:** the limitation line shows with the runtime's text, in the JSON, and in the `starting` fallback.

**Mutation sweep (one-off; script not committed): 16/16 caught.** The mutations:
- the spec without the requirement;
- no cache check;
- pending/failed accepted;
- no pass required; no criteria ID check;
- no thresholds; no scene-error check;
- no boot ID required;
- no entry-ID cross-check;
- no flag, interval, mtime or engine match; no runtime match at all;
- `assemble()` ignoring the fixture registry;
- the limitation not shown.

### Session 16 verification: exact commands and results

```bash
.venv/bin/python -m pytest -q tests/unit/test_scene_admission.py
# 36 passed in 0.49 s, exit 0
.venv/bin/python -m pytest -q
# 647 passed, exit 0 (610 before)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
git diff --check -- src tests docs
# no output, exit 0
```

- **Not run:** any hardware, llama-server, camera or Telegram; any checklist step; CI; Python 3.12. Step 4 stays PENDING and not approved, and no profile is accepted.

## Session 15 log (Claude, 2026-10-04)

**Scope.** Maintainer instruction: implement D-1 `sentinel run` in its approved demo form, then D-2, a minimal loopback-only status page, one package at a time. Verify the reported V2-26 commit, the S1 decision and the portable test results first. Preserve `BUILD.md`. Do not push. Keep scene inference optional and off by default; keep `--cache-ram 0`; do not treat S1 as combined-runtime memory acceptance; do not implement V2-29 or face recognition/enrollment. Use portable tests and fakes only: no live capture, model, llama-server or Telegram. Also verify that every llama-server launch in the demo path binds 127.0.0.1 only, with a test that refuses a non-loopback bind. Prepare one numbered operator checklist.

**Starting state, verified.**
- HEAD `ea95571` ("Adopt --cache-ram 0 for the demo scene path (D41); add V2-26 demo form"); 11 files, as session 14 recorded. The local `origin/v2-beta` equals it.
- D41's record (session 14 log) matches the S1 evidence recorded in session 13. Arm b's steady pressure slope was +3.9 MB/min against arm a's +115 MB/min, with 46/46 valid requests in each arm.
- `.venv/bin/python -m pytest -q`: **533 passed** in 14.62 s, exit 0, matching session 14. `sentinel config validate config/default.yaml`: valid, exit 0.
- `BUILD.md`: see the "Working tree" row. There was no diff to preserve or discard.

### D-1 `sentinel run` (demo form, portable part; device run PENDING)

| File | Change |
|---|---|
| `src/sentinel/demo_runtime.py` (new) | `DemoRuntime`: the loop. `OutboxLoop`: runs `OutboxWorker.run_once()` in its own thread. `assemble()`: startup checks and loads. `snapshot()`: status of numbers and labels. `degradation()`: the reasons the runtime is degraded. |
| `src/sentinel/scene/server.py` (new) | `LlamaServerProcess` and `loopback_problem()` (D44). |
| `src/sentinel/runtime.py` | `EdgeCore(scene_problem=...)`. A scene adapter that was requested but failed reads `scene_analysis: unavailable` with the reason, not `disabled`. Behaviour is otherwise unchanged. |
| `src/sentinel/cli.py` | `sentinel run CONFIG --data-dir DIR --engine PATH [--min-free-gb 1.5] [--scene ...] [--status-interval-s 30]`. It prints one JSON line at start, a status line every interval and one at stop. Exit codes: 0 after a clean stop; 1 for a startup refusal (by label); 2 if a component did not stop in time; 130 for an interrupt during startup. |
| `tests/unit/test_demo_runtime.py` (+30), `tests/unit/test_scene_server.py` (+24) | See "Tests" below. |

- **D43. D-1 wiring (session 15 implementation decision, not yet reviewed).**
  - **Threads.** One capture thread (`CaptureWorker`, the only camera reader); the loop thread (`step()`: tracker, EdgeCore, rules, `record()`, `annotate()`); one scene worker (`ThreadedSceneAnalyzer`, only with scene analysis); one outbox thread. All share one database writer connection under the lock that D31 describes.
  - **Frame ownership.** The loop takes frames from the one-slot `LatestFrame`, so it never queues them. The tracker uses the image synchronously. `RecentImages` (4 images) keeps references only while scene analysis runs. A frame whose epoch already ended (`frame.stream != capture.connected`) skips inference.
  - **Detector state.** It comes from `PersonTracker`'s result: a FAILED frame makes the detector UNAVAILABLE for that step (occupancy UNKNOWN, D18), and the next PROCESSED frame makes it AVAILABLE again. A detector that failed to load stays UNAVAILABLE with its label.
  - **Incident hand-off (D31).** Zone observations and hazard candidates become signals that are acknowledged only after `record()` returns. A failure keeps them in order, and the retry waits at least 1 s. At most 256 wait; beyond that, new ones are dropped, counted and shown as lost. This is not a durable spool (guide ch. 4); a crash loses whatever is waiting. The waiting signals get one more attempt at shutdown.
  - **Enrichment (D35).** Every evidence item that names an incident goes to `annotate()`. A newly CREATED zone incident requests enrichment of its own source frame (one of the last 4 frames) while scene analysis runs.
  - **Startup order.** The stream URL is checked first, then scene admission (when `--scene` is given), then the database writer lock. Any of these refuses startup with a label before anything loads. Then come the notifiers, the scene server and the detector, in check 8's load order. MemFree prechecks guard the GPU loads: 3.0 GB before the scene server (as `demo_profile.py`) and 1.5 GB before the detector (as the track probe). They are provisional, not U18's policy. No page-cache eviction and no sudo. A model failure does not refuse startup: that component is unavailable, and core monitoring runs.
  - **Scene off by default.** It needs both `--scene` and an enabled `llama-lfm2-vl-scene` manifest in the configuration that the existing registry admits (V2-49, D28). `config/default.yaml` lists none. ~~The code does not block enabling it before the confirming combined profile~~ **Superseded by D46 (session 16):** `--scene` now needs an ACCEPTED, matching combined profile measured with `--cache-ram 0`, and D33's provisional profile is refused for scene analysis.
  - **Telegram** is used only if listed in `notifications.channels`. Without `SENTINEL_TELEGRAM_BOT_TOKEN`/`_CHAT_ID`, the channel is unavailable (`credentials_missing`) and its rows wait as queued.
  - **Shutdown.** SIGINT/SIGTERM call `request_stop()`. Bounds: capture `open + read + 1` s (16 s by default); a waiting scene job is dropped, then the scene server gets SIGTERM with a 10 s grace before SIGKILL; the scene worker gets the request timeout + 1 s; the outbox gets the Telegram timeout + 2 s. The database is closed only if the outbox thread stopped. An interrupted send's lease expires and is retried as ambiguous (D32).
  - **Not included.** No V2-29 admission/degradation controller, memory-pressure response, restart or re-admission; no face stage (`face_recognition: disabled`); no runtime/core process split (V2-11); no systemd unit (V2-19).
- **D44. llama-server launch owner (session 15 implementation decision, not yet reviewed).** `sentinel run` starts llama-server only through `LlamaServerProcess`. Before spawning, it requires the three model files to exist and runs `loopback_problem()` on the argv. That check refuses any `--host`/`--host=` value other than `127.0.0.1` (including `localhost`, `::`, all-interfaces and `.sock` Unix sockets, since b8932's `--host` takes those), and a missing `--host`. It also refuses a remaining `LLAMA_ARG_*` variable and a port that already answers on 127.0.0.1. The server is ready only when `/health` answers 200, its output shows `offloaded N/N layers to GPU`, `CLIP using CUDA0` appears, and only L4T's libcuda is mapped into the child (D27). Otherwise it is stopped and scene analysis is unavailable. Only those placement markers are read from its output; nothing else is kept. It runs in its own session, with stdin from `/dev/null`.

**Loopback verification (llama-server launches in the demo path).**
- `sentinel run` → `LlamaServerProcess`: the argv comes from `llama_server_command()` with `--host 127.0.0.1`, checked by `loopback_problem()` at launch, with `LLAMA_ARG_*` removed. Tests check the spawned argv/env and the refusal of a non-loopback command before any spawn.
- `benchmarks/runner/demo_profile.py` `start_llama()` (Check 8, S1, the confirming run, through `operator_check.py`): it passes `--host 127.0.0.1` literally. A new test launches it, with a fake `Popen`, in all four flag/log combinations; every argv passes `loopback_problem()`. It does not remove `LLAMA_ARG_*` from its environment. The command-line `--host` overrides `LLAMA_ARG_HOST` in b8932 (D42), so it stays loopback. The script is unchanged, so earlier measurements stay reproducible (D41).
- v1's `start_sentinel.sh` (read only; it is not in the v2 demo path) binds `--host 127.0.0.1` (existing source-scan test).
- `llama-server --help` in b8932 has only `--host` for the listen address (`common/arg.cpp`, read only).

**Tests (+54).** `test_scene_server.py`:
- the loopback check on the demo argv and on nine refused variants;
- refusal before spawn for a non-loopback command, a port in use or missing files;
- the spawned argv/env (loopback, `--cache-ram 0`, no `LLAMA_ARG_*`, the preload);
- the benchmark profiler's launches;
- placement failures: partial offload, no vision line, a libcuda that is not L4T's, or none;
- exit during load, not ready in time, exit after ready;
- SIGKILL after the grace;
- no output text kept.

`test_demo_runtime.py`:
- an incident and one delivered alert end to end, with no resend;
- an empty room reads EMPTY;
- a reconnect gives a new epoch, a tracker reset and the old episode ENDED;
- a frame from an ended epoch gets no inference;
- a stall withdraws people and occupancy;
- a late enrichment report annotates only its incident and never becomes current;
- a failing detector makes occupancy unknown, and it recovers;
- a detector that never loaded, and unavailable scene analysis, are shown with their reasons;
- `record()` failures are retried once a second;
- a lost commit acknowledgement is a duplicate on retry;
- overflow is counted as lost;
- missing credentials keep alerts queued;
- an ambiguous timeout and a notifier exception are retried, flagged and never counted as delivered;
- a dead worker's lease is retried as ambiguous;
- bounded shutdown of every owned component;
- a send outlasting its bound keeps the database open;
- waiting observations are recorded at shutdown;
- the real `CaptureWorker` reconnects (3+ connects) and stops cleanly (open = close);
- startup refusals: no URL (before any database or model), scene not admitted, a second writer;
- admitted scene ready; model failures unavailable; low MemFree skips GPU loads; an unexpected error stops the server and releases the lock;
- the scene server exiting mid-run and an outbox database error are shown;
- `sentinel run` stopped by SIGTERM (exit 0, handlers restored, no URL, credentials or token in stdout/stderr);
- a refusal by label.

**Mutation sweep (one-off; script not committed): 22/22 caught.** The mutations:
- acknowledgement before commit;
- inference on an ended epoch;
- a failed frame keeping the detector available;
- no annotation; no enrichment;
- retry without backoff;
- outbox not stopped; no final flush;
- unbounded overflow;
- scene without admission;
- no MemFree precheck;
- database closed under a running sender;
- unavailable scene shown as disabled;
- any host accepted; a missing host accepted;
- launch without the loopback check;
- `LLAMA_ARG_*` kept;
- port in use ignored;
- partial offload accepted;
- libcuda unchecked;
- a failed server left running;
- no SIGKILL after the grace.

### Session 15 verification (D-1): exact commands and results

```bash
.venv/bin/python -m pytest -q tests/unit/test_demo_runtime.py tests/unit/test_scene_server.py
# 54 passed in 6.0 s, exit 0; the 52 tests before the last two additions passed in 5 consecutive runs
.venv/bin/python -m pytest -q
# 587 passed, exit 0 (533 before)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
git diff --check -- src tests docs
# no output, exit 0
```

- **Not run:** `sentinel run` against the camera, detector, llama-server or Telegram; any GPU, MemFree or memory measurement; CI; Python 3.12. D-1's device run, the V2-05 outage check, the V2-09/V2-10 device check and the confirming combined profile are PENDING (see "Hardware checks PENDING").

### D-2 loopback status page (demo form, portable part; device check PENDING)

| File | Change |
|---|---|
| `src/sentinel/status_page.py` (new) | `StatusServer` (stdlib `ThreadingHTTPServer` on 127.0.0.1); `read_store()` for incidents and deliveries over a read-only connection; `build_status()` for the `/status.json` document; `render_html()`; `delivery_state()`. |
| `src/sentinel/cli.py` | `sentinel run --status-port` (default 18090; 0 = no page). The page is bound before any model loads, so a busy port refuses startup (`status_port_unavailable:OSError`). It shows `starting` while the models load. It is stopped as part of shutdown and counted in `all_stopped`. The startup line prints its loopback URL. |
| `tests/unit/test_status_page.py` (+20), `tests/unit/test_demo_runtime.py` (+2, and `--status-port 0` in the two earlier CLI tests) | See "Tests (D-2)". |

- **D45. Status page (D-2 demo form; session 15 implementation decision, not yet reviewed).**
  - **Binding.** 127.0.0.1 only. Any other host (including `localhost`, `::`, all-interfaces) raises before a socket opens, and the CLI has no host option.
  - **Host check.** Requests must carry `Host: 127.0.0.1:<port>` or `localhost:<port>`; anything else gets 421. That blocks DNS-rebinding reads from a browser.
  - **Read only.** GET `/` (HTML, `meta refresh` 2 s) and `/status.json` only. Other methods get 405 with `Allow: GET`, other paths 404.
  - **Headers.** `Cache-Control: no-store`, `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and a CSP of `default-src 'none'` (inline style only). `Server: sentinel-status`, with no Python version. No request logging.
  - **Database reads.** Each request opens its own read-only connection (`mode=ro`; a missing database is reported, never created) for one short read transaction, so it never takes the writer's lock or writes.
  - **Delivery states.** queued = pending with 0 attempts; attempted = at least one attempt, not confirmed (retrying, or `in flight` while leased); delivered = `sent` (Telegram returned `ok` with a message ID); failed = `dead` (needs an operator). Ambiguous rows are counted and labelled "may have been delivered".
  - **What it shows.** It is built only from the runtime snapshot (numbers and fixed labels) and stored rows: incident title, zone, severity, status and times; channel, message kind, attempts and the redacted `last_error`. Every value is HTML-escaped, including the scene model's summary. A snapshot older than 5 s while running is called out as "runtime status not updated". Rendering errors return a fixed 500 text.
  - **Limits.** It is not V2-17/V2-18: no authentication, roles, API, actions, media or live video. Access is over `ssh -L 18090:127.0.0.1:18090`.

**Tests (D-2).**
- Delivery-state mapping; a real store with one alert in each state, plus ambiguity and the redacted token in an error.
- Reads while the runtime holds the writer lock, with the file unchanged.
- A missing database is reported and not created.
- A stale snapshot is called out; the `starting` page.
- Binds other than 127.0.0.1 refused (5); listens on loopback; HTML and JSON content and headers.
- Six non-GET methods give 405; unknown paths and traversal give 404; five foreign Host headers give 421; `localhost:<port>` is accepted.
- Degradation is shown, and script text is escaped; a build failure shows no exception text; stop closes the socket.
- `sentinel run` serves the page mid-run. It shows `telegram unavailable (credentials_missing)` when a token is set without a chat ID, then stops with the page closed. No URL, password, host or token appears in the page, JSON, stdout or stderr.
- A busy status port refuses startup before any model load.

**Mutation sweep (one-off; script not committed): 16/16 caught.** The mutations:
- any bind host; binding all interfaces;
- an unchecked Host header;
- dead as attempted; attempted as queued; sent not delivered;
- ambiguity hidden;
- no escaping;
- write methods served;
- exception text shown;
- stale snapshot not shown; database failure not shown;
- no security headers;
- socket not closed;
- a reader that creates the database (caught after adding the no-create assertion);
- page not stopped at shutdown.

**Operator checklist.** Written into "Hardware checks PENDING" as steps 1–5, replacing the separate V2-05 and V2-09/V2-10 blocks. Every shell block passes `bash -n`, and both Python one-liners parse. The `printf` configuration lines, copied with their indentation, produce files that `sentinel config validate` accepts, with the scene adapter `enabled` when the manifest line is added. The pass criteria proposed for step 4 are a proposal for the maintainer to confirm before the run.

**Read-only observations while preparing step 1 (not the step itself).**
- `~/.config/Ultralytics/settings.json` still has `sync: true`.
- In the installed Ultralytics 8.4.25 (`utils/events.py`), `events.enabled` requires `ONLINE`, and `utils/__init__.py` `is_online()` returns false when `YOLO_OFFLINE=true`.
- Claude did not import Ultralytics or run `yolo`.
- The new modules import under `~/onvif_env` (Python 3.10.14, pydantic 2.12.5) without loading torch, cv2 or Ultralytics, and `sentinel run --help` parses there. Nothing was installed.

### Session 15 verification (D-2): exact commands and results

```bash
.venv/bin/python -m pytest -q tests/unit/test_status_page.py tests/unit/test_demo_runtime.py tests/unit/test_scene_server.py
# 77 passed in 12.7 s, exit 0; 3 consecutive runs
.venv/bin/python -m pytest -q
# 610 passed, exit 0 (587 after D-1, 533 at session start)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
PYTHONPATH=src ~/onvif_env/bin/python -c "import sentinel.demo_runtime, sentinel.status_page, sentinel.scene.server"
# imports ok (3.10.14, pydantic 2.12.5); torch, cv2 and ultralytics not imported
git diff --check -- src tests docs
# no output, exit 0
```

- **Not run:** the page on the device or over the SSH forward; any browser; any checklist step; CI; Python 3.12.

## Session 14 log (Claude, 2026-10-04)

**Scope.** Maintainer instruction: evaluate S1 against its predeclared criteria and, if met, adopt `--cache-ram 0` for the demo scene path; implement only the portable V2-26 demo increment; record the Ultralytics privacy item; verify llama-server is loopback-only. Started at `cd1c201`, five commits ahead of the local `origin/v2-beta` (`84f15ec`). `BUILD.md` still has only the unrelated trailing blank line, left uncommitted. No GPU, camera, cache drop, install, guard change or push.

### S1 evidence (USER-SUPPLIED, recorded in session 13) against its predeclared criteria

Runs: Check 9 `n51_c4_0`, arm a `l4ikjyet`, arm b `vzh8rtqp`, commit `e9af7f4`, boot `201a195f…`, headless, D37 declared. Earlier replicate: arm a `0n316rus` (Check 9 `5dbqfrm1`); arm b `1lsjvxhd` was interrupted in warm-up with no steady data.

| Criterion (session 11 "What S1 can establish", D38, session 12 timebox) | Evidence | Result |
|---|---|---|
| Same-boot, same-commit successful Check 9 before both arms; headless; D37 declared | Both arms cite `n51_c4_0`; 0 desktop/dev tools; display manager inactive. One chained command fits the timing (arm b baseline 62.6 s after arm a's post-exit sample, matching `sleep 60`) but was not recorded. | Met |
| Completion | Both `complete`, rc 0; steady 180.1 s each; 46 completed steady requests each (53 in all) | Met |
| Arm a's cache and memory grow per request | Cache 8 → 52 prompts, 42.482 → 280.178 MiB (0 duplicate, evicted, failed); pressure +7.54 MB and llama-server PSS +7.59 MB per completed steady request (+115.0 / +114.9 MB/min) | Met |
| Arm b stays flat | Cache disabled, 0 updates; +0.30 MB pressure and +0.29 MB PSS per request (+3.9 / +3.5 MB/min, about 3 % of arm a). "Flat" was never quantified in advance; this judges it comparatively. The residual slope is unexplained, and not known beyond 180 s. | Met (comparative) |
| Structure equal | 53/53 and 46/46 valid, all `stop`, 0 errors, 0 over 8 s; prompt tokens 325 both; completion mean 110.0 vs 107.8; latency p50/p95/max 2324.4/2497.2/2597.4 vs 2271.7/2382.9/2561.4 ms. Noise images: structure only. | Met |
| Within-arm comparison; arm-order and baseline limits | Baselines equal (1,916,297,216 vs 1,915,895,808 B; arm a residue −9,740,288 B). Arm b always ran second (not counterbalanced) and was not replicated; arm a replicated from a 1.26 GB baseline (slope within 1.8 %). Both arms ran after the boot's one-time ~0.66 GB step. Steady trends include exit rows (finding 4), equally in both arms. | Limits acknowledged; no contradiction |
| Cleanup | Both cleanup clear; post-exit pressure 1,901,981,696 / 1,917,984,768 B; residue −9.7 / +2.4 MB; 0 OOM/NvMap; swap 0 | Met |

### Decisions

- **D41. `--cache-ram 0` for the demo scene path** (maintainer authorization conditional on the criteria, 2026-10-04; evaluated as met above).
  - **Applies to:** the llama-server behind the v2 demo scene adapter (`llama_server_command()`, used by D-1). It is fixed in code, not configurable.
  - **Does not apply to:** `benchmarks/runner/demo_profile.py`, whose default stays "not passed" so earlier measurements remain reproducible (a confirming run passes `--llama-cache-ram 0` explicitly); v1's `start_sentinel.sh` (protected, unchanged); or the beta model choice (V2-26 proper).
  - **Does not establish:** combined-workload memory stability (D33's provisional profile was measured with the default cache and remains the admission record until a confirming run), long-run behaviour, U18, U19, U21, scene accuracy or beta acceptance.
- **D42. Loopback-only scene server (V2-26 demo form; session 14 implementation decision, not yet reviewed).**
  - **Binding.** llama-server is always launched with `--host 127.0.0.1`. On the b8932 command line this overrides `LLAMA_ARG_HOST`, and every `LLAMA_ARG_*` variable is removed from its environment.
  - **Client.** The client uses `http.client` to 127.0.0.1 only: no proxy, no redirects. `scene_server` has no host setting.
  - **Model flags.** The model flags are check 8's profiled ones (`--n-gpu-layers 999 --ctx-size 2048 --parallel 1`) plus D41, with the D27 preload and v1's unified-memory variable.
  - **Verified (source scan in a test):** no `0.0.0.0` anywhere in `src/`, `benchmarks/` or `config/default.yaml`; every `"--host"` in Python sources is followed by loopback. v1's launcher binds `127.0.0.1` (read only). b8932's default host is also 127.0.0.1 (`common/common.h`, read only).

### V2-26 demo form (portable part; device check comes with D-1)

| File | Change |
|---|---|
| `src/sentinel/scene/llama_server.py` (new) | Launch argv/environment (D41, D42). The S1/check 8 request: 480×360 JPEG at quality 60, the same system and user prompts, U20 `response_format`, 200 tokens, temperature 0.05, non-streaming. `LoopbackTransport`: bounded 1,000,000 B read, fixed labels (`request_timed_out`, `request_failed:<class>`, `http_<status>`, `response_too_large`). `LlamaSceneRequest`: encode (OpenCV, imported lazily) → request → U20 envelope check → `WorkerOutcome`. Timeouts become TIMEOUT outcomes; envelope rejections become `completion rejected: <reason>`; no model or server text is kept. |
| `src/sentinel/scene/analyzer.py` (new) | `RecentImages` (the last 4 frame images, by `FrameKey`). `ThreadedSceneAnalyzer`, a `SceneAnalyzer` for the existing `SceneLane`. `submit()` is non-blocking. One job runs and one waits; a job whose image is gone fails at once. Cancel drops a waiting job; a running request is not interrupted, and its outcome is delivered late so the lane annotates its incident only (D15, D35). Exceptions become that job's error, with class name only. The inbox is bounded at 8, and `drain()` runs on the runtime thread. `stop()` is bounded; `submit()` after stop raises, which the lane records as error evidence. |
| `src/sentinel/scene/completion.py` | `scene_completion_content()` split out of `parse_scene_completion()` (behaviour unchanged): the worker checks the envelope, and the lane parses the report strictly. |
| `src/sentinel/config.py`, `config/default.yaml` | New `scene_server` section: `port` 18081 (1024–65535), `request_timeout_s` 20 (at least `scene.job_timeout_s`). `host` and cache settings are refused as unknown settings. |
| `src/sentinel/adapters.py` | Registry entry `llama-lfm2-vl-scene` (scene analyzer) with the provisional profile. |
| Tests (+35) | `test_llama_server.py` (+20), `test_scene_analyzer.py` (+13), `test_config.py` (+1), `test_scene_completion.py` (+1). Covered: argv/env and parity with `demo_profile.LLAMA_FLAGS` and `demo_workload`'s prompts and request; the loopback source scan; transport host, bounds and failures; envelope rejections without text; timeout and encode failure; registry admission. Worker: non-blocking submit, missing image, cancel waiting/running, replacement, exceptions, inbox bound, bounded stop. Through `EdgeCore`: an on-time report becomes current; a late result annotates `inc-7` only; a superseded-epoch result is not current; an error replaces the previous verdict. |

### Privacy item: Ultralytics analytics (from session 13)

With `sync: true` in this device's Ultralytics settings, v1, `demo_workload.py` and check 8 sent Ultralytics usage analytics (task, mode and environment metadata, no images) whenever online. v2's legacy adapter forces `YOLO_OFFLINE=true` (D40). **Operator action, PENDING until the maintainer confirms it ran:** `~/onvif_env/bin/yolo settings sync=False`. Claude did not run it. The benchmark's `urllib` client also honours proxy environment variables; v2's adapter does not use it.

### Session 14 verification: exact commands and results

```bash
.venv/bin/python -m pytest -q tests/unit/test_scene_analyzer.py tests/unit/test_llama_server.py tests/unit/test_scene_completion.py
# 75 passed, exit 0 (before the config and completion additions); test_scene_analyzer.py 13 passed in 8 consecutive runs
.venv/bin/python -m pytest -q
# 533 passed in 14.61s, exit 0 (498 before)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
git diff --check -- src tests config docs
# no output, exit 0
```

- **Mutation sweep (one-off; not committed): 17/17 caught.** Mutations: all-interfaces host; no `--cache-ram 0`; `LLAMA_ARG_*` kept; client host `localhost`; no response bound; connection not closed; timeout as error; envelope skipped; temperature changed; cancel not dropping a waiting job; missing image not refused; worker exceptions escaping; inbox unbounded; submit after stop; image store unbounded; replacement silent; config cross-check removed.
- **Not run:** any llama-server, GPU, camera or OpenCV encoding; CI; Python 3.12. The PTS fix does not retroactively validate the session 13 capture runs. V2-05 outage and V2-09/V2-10 device checks stay PENDING.

## Session 13 log (Claude, 2026-10-04)

**Scope and preservation.** Maintainer instruction:
1. Record the headless Check 9, both S1 arms and the V2-05 camera checks as USER-SUPPLIED MEASUREMENTS.
2. Compare the S1 arms and apply the memory timebox.
3. Evaluate V2-05 against its acceptance and explain the `pts_quality` "none" counts.
4. Continue with V2-09/V2-10 (demo form).

Started on `v2-beta` at `e9af7f4`, two commits ahead of the local `origin/v2-beta` (`84f15ec`). The working tree held only session 12's unrelated `BUILD.md` blank line. No fetch or push, branch change, installation, workload, cache drop, service change, camera or GPU access by Claude.

Claude read:
- the six new sanitized `result.json` files;
- the interrupted run's `summary.txt`, `events.jsonl` and the tail of its `guard.jsonl`;
- the four capture JSON files and their stderr files, with addresses masked by `sed`;
- `ps` start times.

`diff -rq` showed every copy in `~/sentinel-runs/operator/` identical to its `/tmp` original.

### USER-SUPPLIED MEASUREMENTS (maintainer, 2026-10-04 IST, boot `201a195f…`, repository `e9af7f4`)

**Operator runs.** Common to all six runs:
- `e9af7f4`, on the boot that started 22:27:14 IST;
- the D37 drop declared (`preparation.drop_caches: operator_declared`);
- inspection found 0 desktop, dev-tool, model-server, unclassified-Python and media/GPU-tool processes; the display manager inactive; NVIDIA PVA the only known system service; no refusals;
- 0 OOM and 0 NvMap candidate lines; swap counters at 0.

They form two sequences. **The maintainer's report names only the second.** The first is recorded because its directories are preserved.

| Run (directory) | Finished (UTC / IST) | Result |
|---|---|---|
| Check 9 #3 (`5dbqfrm1`) | 19:29:40 / 00:59:40 | `bounded_smoke_complete`: baseline pressure 1,258,913,792 B, sampled peak 1,282,379,776 B, post-exit 1,259,933,696 B |
| S1 arm a #1 (`0n316rus`) | 19:34:25 / 01:04:25 | `completed` (table below) |
| S1 arm b #1 (`1lsjvxhd`) | 19:36:13 / 01:06:13 | **`interrupted`** in warm-up, after 2 valid requests. `operator_check.py` itself received SIGINT, SIGTERM or SIGHUP (which one is not recorded) and stopped the child with SIGTERM. Cleanup clear, port free, no post-exit sample. Startup logged the prompt cache as disabled. The cause is not recorded; per `ps`, a new SSH login followed at 01:07:04 IST and a tmux client attached at 01:07:27. |
| Check 9 #4 (`n51_c4_0`) | 19:38:08 / 01:08:08 | `bounded_smoke_complete`: baseline 1,921,302,528 B, sampled peak 1,942,663,168 B, post-exit 1,925,824,512 B |
| S1 arm a #2 (`l4ikjyet`) | 19:42:52 / 01:12:52 | `completed` |
| S1 arm b #2 (`vzh8rtqp`) | 19:48:37 / 01:18:37 | `completed` |

Each Check 9 allocated 268,435,456 B per API in 33,554,432 B chunks (device and managed; rc 0, cleanup clear).

**S1 arms.** Common conditions:
- a 180 s steady phase, one scene request every 4 s;
- a distinct synthetic noise image per request;
- the GPU check passed: 17/17 layers and the vision encoder on CUDA0, L4T libcuda only.

Bytes unless stated:

| | Arm a #2: default cache | Arm b #2: `--cache-ram 0` | Arm a #1 (first sequence) |
|---|---|---|---|
| Check 9 used | `n51_c4_0` | `n51_c4_0` | `5dbqfrm1` |
| Baseline pressure | 1,916,297,216 | 1,915,895,808 | 1,259,565,056 |
| Prompt cache at startup | enabled, 8192 MiB | disabled | enabled, 8192 MiB |
| Cache, steady first → last | 8 prompts, 42.482 MiB → 52 prompts, 280.178 MiB; 0 duplicates, evictions, failures | no updates | 8, 43.033 MiB → 52, 278.864 MiB; 0 / 0 / 0 |
| Pressure, steady first → last | 3,068,239,872 → 3,415,216,128 | 3,054,936,064 → 3,068,571,648 | 2,952,343,552 → 3,303,878,656 |
| Pressure slope (B/min) | +115,040,663 | +3,855,411 | +117,103,238 |
| MemFree slope | −115,338,056 | −4,413,870 | −117,731,380 |
| Cached slope | +200,687 | +437,399 | +532,286 |
| llama-server PSS, first → last | 570,285,056 → 919,215,104 | 515,972,096 → 529,212,416 | 570,555,392 → 919,249,920 |
| llama-server PSS slope | +114,939,122 | +3,510,888 | +115,116,080 |
| Workload PSS slope | +427,991 | +85,942 | +85,690 |
| Requests (all phases) | 53 / 53 valid, all `stop`, 0 errors | 53 / 53 valid, all `stop`, 0 errors | 53 / 53 valid, all `stop`, 0 errors |
| Steady scene completed / valid | 46 / 46 | 46 / 46 | 46 / 46 |
| Latency p50 / p95 / max (ms) | 2,324.4 / 2,497.2 / 2,597.4 | 2,271.7 / 2,382.9 / 2,561.4 | 2,235.4 / 2,421.1 / 2,468.2 |
| Over the 8 s D16 timeout | 0 | 0 | 0 |
| Completion tokens (mean); prompt tokens | 110.0; 325 | 107.8; 325 | 106.7; 325 |
| Valid summaries / observations at limit | 2 / 2 | 1 / 1 | 0 / 1 |
| Cold load; load delta | 3.36 s; +849,825,792 | 4.11 s; +891,461,632 | 3.6 s; +1,400,885,248 |
| Sampled peak pressure; min MemFree | 3,415,240,704; 2,759,700,480 | 3,070,148,608; 3,100,798,976 | 3,304,927,232; 2,867,605,504 |
| Unload residual vs baseline | −9,740,288 | +2,371,584 | +656,420,864 |
| Post-exit pressure | 1,901,981,696 | 1,917,984,768 | 1,905,340,416 |

**Capture probes** (V2-05 demo form). Common to all runs:
- `sentinel capture probe` at `e9af7f4`, with `config/default.yaml`'s capture settings;
- substream URL from `SENTINEL_RTSP_URL`;
- times from file modification times;
- no display-manager process started between the S1 inspections and Claude's `ps`.

| Run | Time (IST) | Conditions | Result |
|---|---|---|---|
| a (60 s) | about 01:20–01:21 | before any dev tool started | **`no_frames`**: 0 connects, 6 open failures, `problem: open_failed`, 0 frames. Upstream connections 0 in all 60 samples. CPU 0.994 s; max RSS 77,955,072 B. Stderr has six FFmpeg errors `Connection to tcp://<camera>:554 … failed: No route to host`, at 3.1, 6.2, 9.3, 16.4, 27.4 and 45.5 s: the camera was unreachable from this host for the whole run. |
| b (90 s, outage planned) | about 01:21–01:23 | before any dev tool started; no outage occurred (maintainer) | `frames_received`: 1 connect, 0 open failures, 0 stream ends. Frames captured 1,278, delivered 1,277, replaced 0, discarded 1. 14.191 captured fps; 1 epoch; 640×480 BGR. PTS quality none 274, stream_relative 1,003. Ingest interval p50 52.559 / p95 102.787 / max 410.249 ms (1,276 intervals); hand-off age p50 0.467 / p95 0.510 / max 0.641 ms. CPU 5.991 s = 0.067 core-equivalents; max RSS 87,945,216 B. Upstream connections before 0, min 1, max 1 (90 samples). |
| b2 (150 s) | about 01:28–01:30 | VS Code server and Claude Code running (started 01:25:46 and 01:25:59); no outage occurred (maintainer) | `frames_received`: 1 connect, 0 open failures, 0 stream ends. Frames captured 2,242, delivered 2,241, replaced 0, discarded 1. 14.932 fps; 1 epoch; 640×480 BGR. PTS none 344, stream_relative 1,897. Ingest p50 52.382 / p95 103.078 / max 376.221 ms; hand-off p50 0.292 / p95 0.502 / max 1.016 ms. CPU 10.160 s = 0.068 core-equivalents; max RSS 88,096,768 B. Upstream before 0, min 1, max 1 (150 samples). |
| b3 (simulated outage) | 01:47 | the maintainer planned a 15 s `iptables` cut on the Jetson about 25 s in | **Never contacted the camera.** Stderr is `capture probe: rtsp_url_missing` and the JSON file is empty: `SENTINEL_RTSP_URL` was not set in that shell. The maintainer has deferred the outage test. |

**Credentials.** The maintainer reports `userinfo_lines=0` for every stderr file. Claude's addition: `b.err` and `b2.err` are empty. `a.err` has 12 lines: six FFmpeg connection errors that carry the camera's **host and port** (no user information or path) and six OpenCV warnings.

**Correction to the maintainer's summary.** Run a did not connect: 0 connects and 0 epochs. Only b and b2 had one connect and one epoch, with maximum ingest gaps of 0.410 s and 0.376 s.

### S1 comparison and the timebox (Claude's calculations, not measurements)

- **Equal conditions.** Arms a #2 and b #2 used the same Check 9, commit and boot, both headless. Baselines were equal (1,916,297,216 vs 1,915,895,808 B), and both ran the same 53 requests.
- **`--cache-ram 0` removed about 97 % of the steady growth.**
  - Steady pressure: +346,976,256 B in arm a, +13,635,584 B in arm b.
  - Arm b's pressure slope is 3.4 % of arm a's. Its llama-server PSS slope is 3.1 % (+3,510,888 vs +114,939,122 B/min).
- **Arm a reproduced.** Arm a #1 ran on a different baseline. Its pressure slope was 1.8 % higher, with the same cache growth per prompt (5.36 vs 5.40 MiB).
- **Per prompt, arm a #2.**
  - Logged cache: +237.696 MiB over 44 prompts, so 5.40 MiB (5,664,598 B) per prompt.
  - llama-server PSS: +348,930,048 B, about 7.93 MB per prompt. That is 1.40× the logged cache size; the extra 40 % is unexplained (allocator overhead is one untested candidate).
  - E-2's estimate for Check 8's mixed workload was 6.32 MB per request.
- **Same work, no slower.**
  - Completion, validity and finish counts are identical, and completion tokens are close (110.0 vs 107.8).
  - Arm b's latency p50 is 52.7 ms lower and p95 114.3 ms lower (one run each; not significant).
  - The images are noise, so this shows structural equivalence only, not scene accuracy.
- **Left in arm b:** +3.5 MB/min of llama-server PSS over 3 minutes. Whether it continues, settles or is noise is unknown from 180 s.
- **Timebox outcome.** The first branch applies: `--cache-ram 0` removes the growth. **Claude proposes adding `--cache-ram 0` to llama-server in the demo profile and in D-1**; the maintainer decides, since D38 made adoption a separate decision.
  - Why: Check 8's steady ramp (+100 MB/min, E-2) coincides with llama-server PSS growth, and S1 shows that growth is the prompt cache.
  - Under b8932's default 8192 MiB limit, the cache keeps growing well past the device's 5.4 GB ceiling. Check 8 reached 5.350 GB after 10 minutes.
  - Cost: no host-RAM prompt reuse between scene requests. With a new image on every request, S1 measured no latency penalty.
  - If adopted, a full demo-profile run with `--llama-cache-ram 0` must replace D33's provisional profile. Claude's projection, not a measurement: steady would stay near Check 8's 4.42 GB steady start instead of ramping. That run is PENDING and not approved.
  - **Other memory work stays paused until after Oct 20:** U18 policy, U19 unload and the 30-minute U21 rerun.

### U19 observation from the same runs (Claude's calculations; not investigated further under the timebox)

- **First load of the boot.** The first llama-server load of this boot (arm a #1) had a load delta of +1,400,885,248 B and left +656,420,864 B (the profile's residual vs baseline).
  - Later loads had deltas of +801,611,776 (b #1), +849,825,792 (a #2) and +891,461,632 B (b #2).
  - Of those, a #2 and b #2 left −9,740,288 and +2,371,584 B; b #1 recorded no post-exit sample.
- **Where it went.** Compare Check 9 #3's and #4's baselines, both taken after a D37 drop:
  - pressure +662,388,736 B and MemFree −662,048,768 B;
  - but Cached (−45,056), Shmem (+40,960), SUnreclaim (−724,992), KReclaimable (−270,336), CmaFree (0) and Unevictable (0) barely moved.
  - So about 0.66 GB was consumed in none of the sampled fields. It survived the cache drops and did not grow over three further load/unload cycles (one interrupted).
- **Earlier evidence.** The short U21 (session 10) left a similar +658,501,632 B.
- **Narrowed, not explained.** On this boot, the residue behaves like a one-time step at the boot's first large GPU load, not per-cycle accumulation. Its owner is untested (for example, driver or NvMap retention). V2-54's unload acceptance still cannot pass on this evidence.

### V2-05 demo form against its acceptance (Claude's evaluation)

Guide acceptance: "No duplicate upstream session; bounded capture". The demo form is D39. Full V2-05 stays open under D23: relay, NVDEC, connection count with live view and recording, and a restore point.

| Criterion (demo form) | Evidence | Status |
|---|---|---|
| One upstream session from this host | b, b2: established connections 0 before, exactly 1 in all 240 samples | **Met for this host.** Other camera clients are invisible here; the relay-based count is V2-05 proper. |
| Bounded capture, fresh hand-off | b, b2: replaced 0; discarded 1 (the frame pending at stop); hand-off age max ≤ 1.016 ms; max RSS about 88 MB. Portable tests cover the slow-consumer bound. | **Met** with the probe's fast consumer. A slow consumer on the device comes with D-1. |
| Decode on the camera: 640×480 BGR, about 15 fps, one epoch | b2: 14.932 fps over 150 s, one epoch. b: 14.191 fps including connection setup, about 74 frames short of the 15.01 fps source rate over 90 s; the probe does not report time to first frame. | **Met** |
| CPU cost of software decode (D24) | 0.067 and 0.068 core-equivalents for the whole probe process, one run each | **Measured** (USER-SUPPLIED): about 7 % of one core at `decode_threads` 1 |
| Max ingest gap below the 2 s stale threshold | 0.410 s and 0.376 s | **Met** in these runs |
| Open failure and backoff on the device | a: 6 bounded attempts in 60 s, failing at 3.1, 6.2, 9.3, 16.4, 27.4 and 45.5 s; consistent with 1/2/4/8/15 s waits plus about 1–3 s per failed attempt; clean stop | **Observed** (camera unreachable) |
| Reconnect after an outage: new epoch, frames after restore, still at most 1 session | no outage in b or b2; b3 never connected | **Not established (PENDING)** |
| Read-timeout path (5 s) on the device | not exercised | **PENDING** |
| No credentials in output | 0 userinfo lines; the JSON has no host. FFmpeg's stderr carries the camera host and port on connection failures. | **Met.** D-1 must keep FFmpeg's stderr out of shared or persistent logs, or redact the host. |
| Source PTS | 21.5 % (b) and 15.4 % (b2) of frames stamped `none` | Explained below; diagnostic only |

V2-05 stays **partial (demo form, full acceptance pending)**. The outage and read-timeout checks are PENDING.

### Why `pts_quality: none` was 274 of 1,277 and 344 of 2,241 (Claude: source, synthetic files and arithmetic; no camera)

- **The rule.** `FrameStamper` stamps `none` when a frame has no PTS, or its PTS is not strictly greater than the previous frame's in the epoch (D21). `OpenCvSource` read the PTS from OpenCV's `CAP_PROP_PTS`.
- **What `CAP_PROP_PTS` is in OpenCV 4.13.0.** Source: `modules/videoio/src/cap_ffmpeg_impl.hpp` at tag 4.13.0, read by Claude.
  - It is not a stream timestamp. It is the frame's PTS rescaled to whole periods of the stream's estimated **average** frame rate, and rounded (`av_rescale_q(picture_pts, time_base, 1/avg_frame_rate)`).
  - It is **not updated when a decoded frame has no PTS**, so the previous value repeats.
  - When FFmpeg has no average rate (0/0), every frame reads −2⁶³.
  - Session 12's code comment ("a missing PTS is reported as negative") was therefore wrong for this build.
- **Why this camera triggers it.**
  - The substream averages 15.01 fps while FFmpeg reports `tbr 20` (V2-01 inventory). Ingest intervals have p50 ≈ 52 ms and p95 ≈ 103 ms.
  - Both fit frames timestamped on a 50 ms grid with some slots skipped.
  - Rounded to periods of about 66.7 ms, two frames 50 ms apart regularly land on the same integer, and the second is stamped `none`. The share depends on frame timing and phase, so it varies between connections.
- **Reproduced without the camera** (`~/onvif_env`, OpenCV 4.13.0, files in the session scratchpad). One H.264 stream: 120 frames on a 50 ms grid, every fourth slot skipped.
  - As MP4 (average rate 800/53 ≈ 15.09 fps): `CAP_PROP_PTS` failed to increase on **13 of 120 frames (10.8 %)**, while `CAP_PROP_POS_MSEC` increased on every frame.
  - The same frames as MKV (average rate 20/1): 0 of 120.
  - An MPEG-TS file without an average rate: −2⁶³ on every frame.
- **Not established:** the camera's actual RTP timestamps, the average rate FFmpeg estimated on each connection, and how many `none` frames lacked a PTS rather than repeating a rounded one.
- **Impact.** None on decisions: every age, TTL and freshness check uses ingest time (D21), and nothing reads `source_pts`. But the value was not a usable stream timestamp, which V2-06's timestamp contract will need.
- **Follow-up:** see "PTS follow-up" below.

### PTS follow-up (V2-05 demo form; device effect PENDING)

| File | Change |
|---|---|
| `src/sentinel/media/opencv_source.py` | `source_pts` is now stream time in **microseconds** from `CAP_PROP_POS_MSEC`, the same PTS unrounded. OpenCV reports 0 ms for a frame without PTS, so 0 counts as a real value only on the first frame after each open and is `None` afterwards; negative and non-finite values stay `None`. The module docstring records why `CAP_PROP_PTS` is unsuitable, and that FFmpeg's stderr can name the camera host and port. |
| `src/sentinel/media/capture.py` | `DecodedFrame` documents that `source_pts` is in the source's own unit and advisory (D21). |
| `src/sentinel/media/probe.py` | The summary adds `pts_none_reasons` (`missing`, `repeated`, `backwards`, and `unattributed` when the probe did not see the frame the stamper compared with) and `pts_step_ms` (p50/p95/max of the PTS step between consecutive consumed frames). |
| `tests/unit/test_opencv_source.py` (+2 tests, +1 case), `tests/unit/test_capture.py` (+1) | The fake `cv2` now answers only `CAP_PROP_POS_MSEC`. Covered: microsecond values with sub-frame steps (1,400,000 / 1,450,000 / 1,500,000 / 1,600,033); 0 accepted only first after each open; NaN, ±inf, −1 and −2⁶³ give `None`. Probe: one stream with each `none` cause, including a replaced frame, and the step percentiles. |

**Verification with real OpenCV (Claude, `~/onvif_env`, OpenCV 4.13.0, no camera):** reading the scratchpad files through `OpenCvSource` → `FrameStamper` → `ProbeStats`:
- 50 ms-grid MP4 (average 15.09 fps; 13/120 repeats with `CAP_PROP_PTS`): `none` 0/120, `pts_step_ms` p50 50.0 / p95 100.0 / max 100.0.
- The same frames as MKV: `none` 0/120, steps p50 50 / p95 100.
- 15 fps constant-rate MPEG-TS: `none` 0/120, steps 66.667.

What the next camera run will show (`pts_none_reasons`, `pts_step_ms`) is PENDING. If the camera's own timestamps repeat or regress, `none` will stay above 0 with the cause named.

### V2-09/V2-10 demo form: legacy detector + ByteTrack parity adapter (portable part; device check PENDING)

Per D24 and the week-2 plan, the demo form runs v1's `yolov8n.engine` through Ultralytics `track()` with ByteTrack and v1's arguments, as the *legacy parity adapter*. It is neither V2-09 proper (a TensorRT adapter without torch, with fixed buffers and a cached context) nor V2-10 proper (ByteTrack separated from Ultralytics). The implementation decision is D40. `surveillance4_1.py`'s `update_tracks` and the Ultralytics 8.4.25 tracking code in `~/onvif_env` were read only, for parity; nothing was copied or installed.

| File | Change |
|---|---|
| `src/sentinel/tracking/tracker.py` (new, portable) | **`TrackerBackend`** protocol (`track(image)` returns `RawTrack`s in image pixels; `reset()`) and **`PersonTracker`**, which turns backend output into `TrackObservation`s for `EdgeCore`. It resets the backend on every new epoch, run, boot or camera. It processes each frame at most once, in order: repeats and frames from older epochs are skipped without inference. It publishes detections only, never predictions. It keeps track IDs unique within an epoch: after a mid-epoch reset, IDs are offset past every one already used, so identity votes cannot pass between people. Confirmation is v1's score (+1 per detection, cap 4, −1 per frame missed; CONFIRMED from 2 until the score reaches 0). Output is validated: non-finite values, confidence outside [0, 1], unordered boxes, and negative, non-integer or duplicate IDs fail the frame. Boxes are clipped, sub-pixel boxes dropped, at most 64 kept (highest confidence). An image that is not the frame's native size fails. Failures are labels plus an exception class name, followed by a backend reset. |
| `src/sentinel/inference/legacy_ultralytics.py` (new, device adapter) | **`LegacyUltralyticsTracker`**. `load()` runs in this order: (1) the D27 guard through ctypes (`libcuda.so.1` mapped only from L4T's directory, `cuInit` 0), before torch or TensorRT can touch CUDA; (2) the engine's SHA-256 must equal check 6's `08370639…`; (3) `YOLO_OFFLINE=true`, then torch/Ultralytics/numpy are imported; (4) `torch.cuda.is_available()`; (5) `YOLO(engine, task="detect")` and three warm-up frames with v1's `TRACK_ARGS`; (6) a tracker reset. `track()` returns only tracked boxes, since Ultralytics returns untracked detections with `id` None, which v1 also skips. `reset()` resets Ultralytics' ByteTrack, clearing tracks and restarting IDs. |
| `src/sentinel/tracking/probe.py` (new), `src/sentinel/media/probe.py`, `src/sentinel/cli.py` | **`sentinel track probe CONFIG --engine PATH [--seconds 1–300] [--min-free-gb 0.5–7, default 1.5]`**. It checks the URL, refuses below the MemFree minimum (a provisional probe guard, not a U18 policy), loads the backend with timing and memory before and after, then runs the capture probe's loop with a `PersonTracker` as consumer. The new `consumer` hook in `run_probe` makes a slow detector show up as replaced frames. The summary adds processed/skipped/failed frames, processed fps, failure labels and error types, resets, dropped boxes, backend ms and ingest-to-result age (p50/p95/max), frames with people, maximum people per frame, and track and confirmed-track ID counts. Numbers only: no boxes, images, engine path, URL or host. Exit 0 only if frames were processed with no failures and the worker stopped. |
| `src/sentinel/adapters.py` | Registry entry **`legacy-yolov8n-bytetrack`** (detector; input `frame`, output `person.track`) and known profile **`provisional-demo-20261003T085010Z`** (check 8, D28/D33). Resolving and loading the manifest imports the module only, not torch or Ultralytics. |
| `tests/unit/test_person_tracker.py` (+20), `tests/unit/test_legacy_ultralytics.py` (+17), `tests/unit/test_track_probe.py` (+6) | **43 new tests.** Tracker: normalized native-image boxes and timing, clipping, v1's confirmation sequence, an occlusion fixture (hidden three frames, returns CONFIRMED, nothing published while hidden), reconnect and geometry-change fixtures (reset, IDs and confirmation do not carry over, boxes use the new size), process-once ordering, failure labels without text, IDs never repeated after a failure reset, failed reset blocks tracking, eight malformed outputs, size mismatch, the 64-box bound, and `EdgeCore` integration (tentative reads EMPTY; a failed frame reads UNKNOWN "person detector unavailable"; the person expires after 1 s). Backend with fake torch/YOLO/numpy: v1 arguments and warm-up, then reset; `YOLO_OFFLINE=true` before import; `TRACK_ARGS` equal to `demo_workload.TRACK_ARGS` (what check 8 profiled); the guard runs before hashing or importing; hash mismatch and missing engine; missing runtime; no CUDA; load errors without text; tracked boxes only; inconsistent output refused; reset; the D27 guard matrix (L4T only, the 535 library, both, cuInit failure, none, missing library); registry admission only with the provisional profile. Probe: meminfo parsing, outcome/track/timing counts, CLI success, slow detector giving replaced frames, refusals (MemFree, load failures, missing URL, bad options) loading nothing and opening no camera, and a non-zero exit on tracking failures. |

**What this establishes.** Portable behaviour only: the v2 tracking rules around a backend (epochs, ordering, ID uniqueness, confirmation, validation, failure handling) and the guarded, offline, engine-pinned load path, against fakes. The engine on disk still hashes to `08370639…` (Claude, `sha256sum`, read-only). **Not established:** the D27 guard and the load on the device, real ByteTrack behaviour (occlusion, ID switches), detector timing and throughput on camera frames, coordinate parity with v1 on real frames, and memory after load. These are the device check in "Hardware checks PENDING". The replay fixtures for V2-10 proper (occlusion, resize, reconnect on labelled clips) need V2-07. No consumer runs the tracker yet: `sentinel run` (D-1) will call `process()` on each `CapturedFrame`, call `set_detector(UNAVAILABLE)` for a failed frame and pass `persons` to `EdgeCore.on_frame()`.

**Observation (Claude, read-only):** Ultralytics' settings on this device have `sync: true`. When online, Ultralytics then sends usage analytics during prediction (task, mode and environment metadata, no images). v1, `demo_workload.py` and check 8 ran that way. The v2 adapter forces `YOLO_OFFLINE=true`. `demo_workload.py` is unchanged, so its measurement conditions stay comparable.

### Session 13 verification: exact commands and results

PTS follow-up (repository `.venv`, Python 3.10.14, pytest 9.1.1; nothing installed):

```bash
.venv/bin/python -m pytest -q tests/unit/test_capture.py tests/unit/test_opencv_source.py
# 43 passed in 2.90s, exit 0
.venv/bin/python -m pytest -q
# 455 passed in 10.91s, exit 0 (451 before)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
env -u SENTINEL_RTSP_URL PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli capture probe config/default.yaml --seconds 2
# capture probe: rtsp_url_missing; exit 1 (no camera contacted)
git diff --check -- src tests docs
# no output, exit 0
```

- **Mutation sweep (one-off; script not committed): 8/8 caught.** Mutations: read `CAP_PROP_PTS` again; accept 0 after the first frame; no first-frame reset on reopen; milliseconds instead of microseconds; probe keeps the last PTS across a skipped frame; steps measured from the last known PTS instead of the previous frame; backwards counted as repeated; probe never updates its last PTS. Originals were restored by file copy and compared with `cmp`.
- **OpenCV source reading (Claude):** `cap_ffmpeg_impl.hpp` from OpenCV's public repository at tag 4.13.0, fetched into the scratchpad; the installed wheel is 4.13.0.92.
- **Not run:** any camera or GPU access, CI, Python 3.12.

V2-09/V2-10 demo form:

```bash
.venv/bin/python -m pytest -q tests/unit/test_person_tracker.py tests/unit/test_legacy_ultralytics.py tests/unit/test_track_probe.py
# 43 passed, exit 0
.venv/bin/python -m pytest -q
# 498 passed in 14.20s, exit 0 (455 before this slice); includes the blocked-import check of every module
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
env -u SENTINEL_RTSP_URL PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli track probe config/default.yaml --engine ~/yolov8n.engine --seconds 2
# track probe: rtsp_url_missing; exit 1 (no camera contacted, no model loaded)
sha256sum ~/yolov8n.engine
# 08370639f961d2c67148c19562718ef80527c7085e88d2d923176180f1b98637 (14,486,949 B, dated 2026-04-18), equal to check 6
git diff --check -- src tests docs
# no output, exit 0
```

- **Mutation sweep (one-off; script not committed): 18/19 caught.**
  - Tracker: no reset on a new epoch; no ID offset after a failure; repeats processed; older epochs processed; no score decay; no clipping; duplicate IDs accepted; overflow not by confidence; size check off; failed reset ignored; confirmation at 1.
  - Backend: no reset after warm-up; `YOLO_OFFLINE` not set; hash not checked; guard accepts any L4T mapping; no CUDA check; untracked detections returned.
  - Probe: skipped frames counted as processed.
  - The one miss, removing the explicit confidence range check, is behaviour-equivalent: `TrackObservation`'s `UnitInterval` rejects the value with a `ValueError`, so the frame still fails as `invalid_output`.
  - Originals were restored by file copy and compared with `cmp`.
- **Not run:** the default importer (`numpy`, `torch`, Ultralytics) and the D27 guard on the device, any GPU or camera access, CI, Python 3.12.

## Session 12 log (Claude, 2026-10-03)

**Scope and preservation.** Maintainer instruction:
1. Read-only, review the Check 9 and S1 results run headless at `84f15ec`, preserve them and check them against the start times of the dev tools.
2. Record them as USER-SUPPLIED MEASUREMENTS and compare the arms.
3. Apply the memory timebox.
4. Start V2-05 (capture adapter, demo form).

Started on `v2-beta` at `84f15ecae9d75fc6adfe48000eab0cfbb9cb3cf2`, clean. The local `origin/v2-beta` reference was the same commit (pushed by the maintainer). No fetch or push, branch change, installation, workload, cache drop, service change, camera or GPU access.

Claude read:
- the three sanitized `result.json` files and one `guard.jsonl`;
- process start times (`ps`);
- the maintainer's `s1` tmux pane scrollback, filtered to command lines and status fields, with URLs and tokens masked.

`cp -a --update=none /tmp/sentinel-operator-* ~/sentinel-runs/operator/` ran with rc 0. The three directories were already there, and `cmp` showed them byte-identical.

### USER-SUPPLIED MEASUREMENTS (maintainer, 2026-10-03, boot `201a195f…` started 22:27:14 IST, repository `84f15ec`)

All three runs declared the D37 drop (`preparation.drop_caches: operator_declared`). The scrollback shows `sync && sudo -n sysctl -w vm.drop_caches=1` before each. Each kernel window had 0 OOM and 0 NvMap candidate lines. The swap counters stayed at 0.

| Run (directory) | Finished (UTC / IST) | Result | Conditions |
|---|---|---|---|
| Check 9 #1 (`09o0piel`) | 17:23:05 / 22:53:05 | `refused`: `initial_headroom_refused`, `desktop`, `dev_tools`, `service_not_known_idle:display-manager.service` | Running: 5 desktop processes, including a remote-desktop session, and 7 dev-tool processes. Display manager active. Pressure 2,942,177,280 B; MemFree 4,556,402,688 B; MemAvailable 5,047,832,576 B. |
| S1 arm a (`isj3mojf`) | 17:23:44 / 22:53:44 | `refused`: the same four, plus `check9_report_mismatch` (the newest Check 9 was the refused `09o0piel`). No child started. | Same conditions. The result does not name the arm, because a refusal returns before the arm is recorded; the scrollback shows `--s1-arm a`. The chained command stopped at this non-zero exit, so **arm b never started**. |
| Check 9 #2 (`dlo7otnl`) | 17:25:23 / 22:55:23 | `bounded_smoke_complete`. The device and managed probes each allocated 268,435,456 B in 33,554,432 B chunks; rc 0, cleanup clear. | Headless: the maintainer first stopped the display manager, the remote-desktop session and the VS Code server. Inspection found 0 desktop, 0 dev-tool and 0 model-server processes, the display manager inactive and no refusals. Baseline pressure 898,416,640 B. Sampled peak 1,227,427,840 B (7 samples over 1.01 s). Min MemFree 6,648,827,904 B. Post-exit (about 5 s) pressure 1,208,434,688 B. |

After Check 9 #2, the scrollback shows the copy to `~/sentinel-runs/operator/` and `sudo systemctl start display-manager`. No S1 command follows. No other `/tmp/sentinel-operator-*`, `demo-profile-*` or profiler output was written on this boot.

### Dev-tool timing (Claude, from `ps` start times; IST)

| Time | Event |
|---|---|
| 22:52:50 | tmux session `s1` started |
| 22:53:05 | Check 9 #1 (refused) |
| 22:53:44 | S1 arm a refused |
| 22:55:18–22:55:23 | Check 9 #2 |
| 22:56:23 | `gdm3` (display manager) started |
| 22:57:06 | VS Code server started |
| 22:57:16 | Claude Code (this session) started |

The 7 dev-tool processes reported at 22:53 were gone by 22:55:16, according to Check 9 #2's inspection. The current VS Code server and Claude Code started 1 min 43 s and 1 min 53 s after Check 9 #2 finished. **Check 9 #2 is not contaminated.** No S1 arm produced measurements, so no arm can be marked either contaminated or clean.

### S1 comparison and the memory timebox

- **No comparison is possible.** Neither arm reached the workload: arm a was refused at admission, and arm b never started. Neither arm has within-arm slopes, prompt-cache entries, request counts, validity, latency or unload residue. The H2 question is **untested**: does `--cache-ram 0` remove llama-server's per-request growth?
- **Timebox (maintainer, 2026-10-03).** If `--cache-ram 0` removes the growth, propose adopting it for the demo profile (the maintainer decides). If not, record the open question for V2-29/V2-30 and stop memory work until after Oct 20. Without S1 data, neither branch applies. Claude did no further memory work. **The maintainer chooses** between:
  - re-running S1 once, with the command in "Hardware checks PENDING". It starts with Check 9, because HEAD has moved past `84f15ec`.
  - closing the timebox now, with the question left open.
- **Open question for V2-29/V2-30 (recorded either way until S1 runs).** Check 8's steady ramp coincides with llama-server PSS growth (+104 MB/min, E-2). Two things are unknown: whether the growth is llama-server's host prompt cache (b8932 default 8192 MiB), and whether `--cache-ram 0` removes it. V2-29's admission controller and V2-30's pressure experiment must not assume a flat llama-server footprint.
- **Claude's calculation (not a measurement).** About 5 s after exit, Check 9 #2's pressure was 310,018,048 B above its baseline; on the previous boot the residue was 126,685,184 B. Not investigated further (timebox).
- **Tooling note.** A refused S1 result does not record the requested arm, so the arm had to be read from the scrollback. Not changed, because memory tooling is paused.

### V2-05 demo form: capture adapter (portable part; device check PENDING)

The maintainer's instruction was to start V2-05 in demo form under the standing rules. Per D22, D24 and deviation 3, the demo form means:
- the v2 runtime itself opens the substream, as the only ingest;
- software decode with `~/onvif_env`'s OpenCV and its bundled FFmpeg (OpenCV 4.13.0: FFmpeg yes, GStreamer no);
- video only.

The relay, NVDEC/GStreamer decode and the main stream (D26) remain V2-05 proper. The implementation decision is D39. The v1 capture (`surveillance4_1.py` `FrameReader`) and `~/onvif_env/src/nano_surveillance/capture.py` were read only for comparison (URLs masked); neither was copied.

| File | Change |
|---|---|
| `src/sentinel/media/capture.py` (new, portable) | **`CaptureWorker`** is the single reader of a `VideoSource`. Each successful open is a new stream epoch from `FrameStamper`. Frames are stamped when decode returns. On a disconnect it withdraws `connected`, drops the epoch's undelivered frame and closes the source, on every path and only from its own thread. Reconnect waits start at 1 s, double up to 15 s while connections deliver nothing, and reset after a connection delivers frames. A frame-size change or an invalid frame ends the connection, so the new geometry starts a new epoch. **`LatestFrame`** is the single-slot handoff: the newest frame replaces an undelivered one, each frame is delivered at most once, and the counters satisfy published = delivered + replaced + discarded + pending. Problems are fixed labels or exception class names, never exception text. The thread has `start()`/`stop(timeout)`; stop costs at most one bounded source call and also interrupts a reconnect wait. A worker bug is reported as `failed`. |
| `src/sentinel/media/opencv_source.py` (new, device adapter) | **`OpenCvSource`**: `cv2.VideoCapture(target, CAP_FFMPEG, [open timeout, read timeout, decode threads])`, with `cv2` imported on first open. RTSP targets get `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp\|allowed_media_types;video`. Frames must be HxWx3 `uint8`, reported as BGR. A negative or non-finite PTS becomes `None`. The URL comes only from `SENTINEL_RTSP_URL` and must be `rtsp://` or `rtsps://` with a host. It never appears in errors, status, `repr` or output. A failed open releases the capture. |
| `src/sentinel/media/probe.py` (new, portable) | The `sentinel capture probe` summary, numbers and fixed labels only: worker counters, captured/delivered/replaced/discarded frames, captured fps, epochs, sizes, PTS quality, ingest cadence (consecutive frames only) and hand-off age p50/p95/max, process CPU s and core-equivalents, max RSS, and **this host's established TCP connections to the camera endpoint** (from `/proc/net/tcp{,6}`; IP-literal hosts only) before and during the run. |
| `src/sentinel/config.py`, `config/default.yaml` | New `capture` section (proposed values): `open_timeout_s` 10, `read_timeout_s` 5, `reconnect_initial_s` 1, `reconnect_max_s` 15 (≥ initial), `decode_threads` 1 (1–4). The section has no URL setting, and unknown keys are refused. |
| `src/sentinel/cli.py` | `sentinel capture probe CONFIG [--seconds 1–300]`: prints the JSON summary and exits 0 only if frames arrived and the worker stopped. Without `SENTINEL_RTSP_URL` it prints `capture probe: rtsp_url_missing` and exits 1. |
| `tests/unit/test_capture.py` (+20), `tests/unit/test_opencv_source.py` (+19), `tests/unit/test_config.py` (+1) | 40 new tests. Worker and slot: epochs and sequence per connection, ingest from the injected clock, the old epoch withdrawn and its frame dropped on reconnect, a frame taken before a reconnect is `NOT_LIVE`, the backoff sequence 1/2/4/5/5/1/1/2 (cap 5), a half-failed open still closed, problems without exception text, geometry change and invalid frames, stop mid-connection, slow consumer, take/close wake-up, thread ownership and bounded stop, stop interrupting the wait, a worker bug. Probe: `/proc/net` decoding including byte order, TIME_WAIT and IPv4-mapped IPv6; percentiles; cadence over skipped frames; summary without the host; CLI wiring and refusal. Adapter with a fake `cv2`: the parameters and TCP/video-only options, no RTSP options for files, PTS handling, pixel-layout refusal, release on failed open and reopen, missing OpenCV, URL source and validation without echo. Config: defaults and bounds. |

**What this establishes.** Portable behaviour only: bounded fresh-frame delivery (at most one waiting image: 921,600 B at 640×480 BGR, plus the frame the consumer holds), epoch and freshness semantics on reconnect, and credential-free diagnostics. **Not established:** decode on the actual camera, the read timeout, the CPU cost of software decode (the D24 question), whether FFmpeg really sets up no audio track, behaviour through camera outages, and the single-upstream-session count on the device. These are the device check in "Hardware checks PENDING". No consumer uses the worker yet: `sentinel run` (D-1) will pass `CapturedFrame.frame` and `worker.connected` to `EdgeCore`. D11's buffer reference stays deferred, because the image travels beside `FrameRef` in `CapturedFrame`.

### Session 12 verification: exact commands and results

```bash
.venv/bin/python -m pytest -q tests/unit/test_capture.py tests/unit/test_opencv_source.py tests/unit/test_config.py
# 53 passed in 2.82s, exit 0
.venv/bin/python -m pytest -q
# 451 passed in 10.69s, exit 0 (411 before this session)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
env -u SENTINEL_RTSP_URL PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli capture probe config/default.yaml --seconds 2
# capture probe: rtsp_url_missing; exit 1 (no camera contacted)
git diff --check -- src tests config docs
# no output, exit 0
```

- **Mutation sweep (one-off; script not committed): 24/24 caught.** The first pass caught 22. Its two misses, a half-failed open left unclosed and stop not closing the slot (hidden by a 5 s wait), led to one new test and one timing assertion. The mutations were:
  - Slot: replaced not counted; take keeps the frame; close does not wake waiters.
  - Disconnect and stop: no discard on disconnect; stream kept after disconnect; stop not closing the slot; stop counted as a stream end.
  - Backoff: never doubles; not capped; not reset after frames.
  - Problems and frames: exception text kept; free-text `SourceError`; size change ignored; source not closed after a failed open.
  - Adapter: no RTSP options; audio allowed; no read timeout; negative PTS kept; failed open not released; any URL scheme.
  - Probe: big-endian addresses; any TCP state; cadence across skipped frames.
  - Config: maximum below initial allowed.

  Originals were restored by file copy, and the full suite then passed.
- **Real OpenCV, no camera (Claude, `~/onvif_env` Python 3.10.14, OpenCV 4.13.0 FFmpeg backend).** A synthetic file in the session scratchpad: 45 frames of random 640×480 noise, MJPG at 15 fps. `OpenCvSource` read 45 BGR `uint8` frames 640×480, with PTS 0…44 strictly increasing. `CAP_PROP_N_THREADS` read back 1. Both timeout properties read back 0, because this build reports no value for them, so they were checked by behaviour instead. Worker and probe over the file for 1.2 s: 2 connections, 2 epochs, 90 frames captured and delivered, 0 replaced, state `stopped`.
- **Open timeout, loopback only.** A local listener accepted the RTSP connection and never replied. With `open_timeout_s` 2.0, `open_failed` came after 2.15 s; with 4.0, after 4.01 s. During each open, `established_connections` counted exactly 1 connection to the endpoint. OpenCV's two warnings ("Stream timeout triggered", "can't be used to capture by name") contained neither the URL nor its credentials.
- **Not run:** any camera or GPU access, CI, Python 3.12, the read-timeout path and anything on the actual substream. Mac/replay-style and loopback results do not establish camera decode, CPU, memory or throughput.

## Session 11 log (Claude, 2026-10-03)

**Scope and preservation.** Maintainer instruction: record the decisions and new evidence, compute E-2 from Check 8's `memory.csv` (numeric fields only), implement S1's portable parts with tests and commit. Don't run hardware. Started on `v2-beta` at `f606529`, clean, equal to the local `origin/v2-beta` reference. No fetch or push, branch change, installation, rebuild, model/server/GPU launch, cache drop or service change. No secrets, clips, enrollment data, review images or raw logs were opened. Read: E-1's sanitized `result.json`, Check 8's `memory.csv` numeric columns, the local llama.cpp b8932 source (prompt-cache log formats only) and the previous session's transcript (to recover the session 10 wording). Protected v1 files and local-only files stay out of the commit.

### Decisions

D37 (standing operator cache drop for measurement runs only) and D38 (S1 approved: portable parts now, operator runs both arms) are recorded under Decisions.

### USER-SUPPLIED MEASUREMENTS (maintainer, 2026-10-03)

- **Reboot.** Completed; the new boot started 2026-10-03 22:27:14 IST. No reboot is pending. The failed units are identical to the pre-reboot list kept outside Git (details: `docs/LOCAL_NOTES.md`). The boot ID changed, so Check 9 must be re-run before any U21 or S1 run. Claude read only the boot ID and `uptime -s`, which agree.
- **E-1, read-only inspection before the reboot** (finished 16:53 UTC, repository `f606529`). Output is preserved at `~/sentinel-runs/operator/sentinel-operator-6xae55qo/`; Claude read its sanitized `result.json`, and the values below match it.
  - Kernel: 0 OOM and 0 NvMap candidate lines in the latest 1,000 records of that boot. These are candidate lines, not unique events.
  - Pressure 3,201,232,896 B with no model servers and 7 dev-tool processes running: 1,932,963,840 B above U21's 1,268,269,056 B headless baseline.
  - MemFree 1,690,787,840 B, MemAvailable 4,788,776,960 B, Cached 3,122,421,760 B, CmaFree 12,406,784 B with no GPU workloads; swap counters pswpin 500 / pswpout 644.
  - Ollama inactive and disabled; display manager inactive. Refusals were `initial_headroom_refused` and `dev_tools` only.
  - All `/tmp` operator directories are preserved in `~/sentinel-runs/operator/`.

### E-2: Check 8's `memory.csv` (Claude's calculations, not measurements)

Run `demo-profile-20261003T085010Z`. Pressure is `MemTotal − MemAvailable` in decimal bytes. PSS is sampled at 1 Hz; each value is the nearest PSS sample in the phase.

| Point | t_mono (s) | Pressure (B) | llama-server PSS (B) | Workload PSS (B) |
|---|---|---|---|---|
| Warm-up start | 3247.861 | 3,888,345,088 | 328,325,120 | 1,761,650,688 |
| Steady start | 3367.964 | 4,424,753,152 | 711,694,336 | 1,836,224,512 |
| Steady end, before the workload exits | 3972.644 | 5,349,761,024 | 1,660,114,944 | 1,832,843,264 |
| Last row labelled steady (workload exiting) | 3973.644 | 5,254,385,664 | 1,659,998,208 | 1,754,603,520 |
| Unload, workload: start / end | 3973.845 / 3989.053 | 4,134,608,896 / 2,020,614,144 | 1,685,941,248 / 1,685,941,248 | — |
| Unload, llama-server: start / end | 3989.253 / 4004.063 | 1,887,854,592 / 1,888,575,488 | — | — |

- **Steady, 604.9 s, exit rows excluded.** Pressure +925,007,872 B. llama-server PSS +948,420,608 B. Workload PSS −3,381,248 B.
- **Least-squares slopes.** Pressure +100,052,669 B/min; llama-server PSS +104,052,523 B/min; workload PSS −477,281 B/min; Cached −17,996,202 B/min. In the first 7 minutes, pressure rose +117,563,113 B/min and llama-server PSS +121,173,335 B/min. Warm-up: pressure +143,135,800 B/min, llama-server PSS +131,084,440 B/min.
- **Per request.** About 6.32 MB of llama-server PSS per completed steady scene request, using session 5's recorded 150.
- **Flattening.** Growth slowed in minutes 7–9: llama-server PSS 1,626,046,464 → 1,660,116,992 B while MemFree was about 0.18–0.19 GB and 168 pages were swapped out. The reason is unknown; b8932 evicts only at its 8192 MiB limit.
- **What this shows.** Check 8's steady ramp coincides in time and size with llama-server PSS growth, while the workload stayed flat. This supports H2 (state retained per request inside llama-server) over H3 for that ramp. It does not identify the mechanism; S1 tests it.
- **Correction.** Session 5's "llama-server PSS (about 1.66–1.69 GB) stayed flat" describes only the end values; it was not flat in steady.
- **Open anomaly (U19).** llama-server PSS was 1,685,941,248 B while it ran alone after the workload exited, yet stopping it lowered pressure by only 132,759,552 B. The RSS/PSS and whole-device views account differently and are never added; the cause is unknown.
- **Contamination at the end of steady (finding 4).** The last row labelled steady was taken while the workload was exiting.

### S1 portable implementation

| Path | Change |
|---|---|
| `benchmarks/runner/demo_workload.py` | **`--scene-only`**: scene requests only, with no detector, face model, torch import or CUDA driver call in the workload process. **`SyntheticScenes`** generates each request's 640×480 noise image from `shake_256(seed ‖ index)` (stdlib): byte-identical across processes and arms, distinct per request, not retained after use; the same resize/JPEG path as before. **`SceneProgress`** emits a `scene_progress` event after every request (both modes) with cumulative, never-reset counters since the workload started, fixed keys and fixed labels: requests, completed, valid/invalid, finish reasons (4), rejection reasons (6), errors (`http`, `timeout`, `other`), this request's latency and tokens, the phase and images issued. No exception text or model output. `--engine` is required only without `--scene-only`; `--scene-only --clip` is refused. |
| `benchmarks/runner/demo_profile.py` | **`--llama-cache-ram MIB`** (≥ 0) adds `--cache-ram MIB` to llama-server only when given; flags and `cache_ram_mib` are recorded in provenance, and `None` means the server default. `--scene-only` passes through, needs only the LLM/mmproj files and refuses `--clip`. The sanitized llama-server pump also saves llama-server's numeric prompt-cache lines (startup limit or disabled; `cache state` prompts, MiB and limits), stamped with receipt `time.monotonic()`, plus fixed labels for duplicate-skip, eviction and allocation failure. The summary and `profile.json` add `prompt_cache` (totals, steady-window counts, first/last), `steady_trend` (first/last/least-squares slope of pressure, MemFree, Cached and both PSS) and `scene_progress_last`. The sanitizer admits `scene_progress` and its keys. |
| `benchmarks/runner/operator_check.py` | **`--execute-workload s1 --s1-arm a|b`**: U21's prerequisites (successful private same-boot/revision Check 9, operator attestation `--confirm-s1-prerequisites`) with the same admission, guard, 360 s deadline and owned-group cleanup. Arm b differs from arm a only by `--llama-cache-ram 0`. The result adds a numbers-only excerpt of the profile (status label, trend, prompt cache, last progress, unload, scene load and steady scene counters). **`--latest-check9-report`** selects the newest Check 9 result and fails closed if that one failed. **`--operator-dropped-caches`** records the D37 declaration in `preparation`; the runner never drops caches or calls sudo. |
| `tests/unit/test_demo_scene.py` (+5) | Cross-process determinism (pinned SHA-256 of image 0) and per-request distinctness. A scene-only `main()` sends a new image per request, loads no models and never touches CUDA; workload/clip CLI refusals. Progress is cumulative across the steady reset, keeps fixed keys and holds totals without `workload_stats`, as after a guard stop; no private text. |
| `tests/unit/test_demo_measurements.py` (+11) | `--cache-ram` appears only when given (none, 0, 512); unlimited/malformed values are refused. Scene-only file set and clip refusal. b8932 log formats are captured as numbers only (no addresses or private text). Sanitized and raw-log cache summaries, including the steady window. Scene-only summary trend, cache and progress. The sanitized pump keeps `scene_progress`. |
| `tests/unit/test_operator_check.py` (+14) | S1 refusals start no child: missing report, unconfirmed, no arm, wrong boot, failed Check 9. The arms differ only by the cache flag, with no clip, sudo, sysctl or drop. Pressure, MemFree, timeout and cleanup stops match U21. The excerpt keeps numbers only. Latest-Check-9 selection fails closed. CLI wiring, mutually exclusive report options and the default `not_declared`. |

**What S1 can establish.** S1 can show whether, with distinct images, arm a's prompt-cache entries and llama-server PSS/pressure grow per request while arm b stays flat. It also checks that completion counts, validity, finish reasons, latency and tokens match between the arms; this is structural only, since images are noise. It cannot establish scene accuracy, long-run behaviour, the Check 8 workload mix, unload/U19 cause or GPU allocation attribution. Adopting `--cache-ram 0` (including in D-1) is a separate decision. The steady trend includes rows taken while the workload exits (finding 4, not fixed here); compare the arms under equal conditions. Arm b always runs after arm a, so its baseline includes any residue from a, after a 60 s pause and a D37 drop. Compare within-arm deltas, slopes and cache counts, not absolute levels. If a's residue pushes pressure to 2.0 GB or more, b refuses at admission; that refusal is itself U19 evidence. The deadline is unchanged at 360 s for about 295 s nominal per arm.

### Verification: exact commands and results

Repository `.venv` (Python 3.10.14, pytest 9.1.1); nothing installed.

```bash
.venv/bin/python -m pytest tests/unit/test_operator_check.py tests/unit/test_demo_measurements.py tests/unit/test_demo_scene.py tests/unit/test_bounded_gpu_probe.py
# 162 passed in 1.63s, exit 0
.venv/bin/python -m pytest
# 411 passed in 8.06s, exit 0 (381 before this session)
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only; exit 0
git diff --check
# no output, exit 0
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m py_compile benchmarks/runner/operator_check.py benchmarks/runner/demo_profile.py
# exit 0 (system Python 3.12.3, which runs these two on the device); --help of both parses the new options
```

- **Failures along the way.** The first focused run failed 1 of 14 in `test_demo_scene.py`: `SceneProgress` bound `event` as a default argument at class definition, bypassing a patched emitter. The emitter is now resolved when it is called. A later run failed 1 of 35 in `test_demo_measurements.py` and exposed a **real defect**: the sanitizer kept the `event` key but nulled the unlisted value `scene_progress`, so sanitized runs would have saved unnamed events. `scene_progress` is now an allowed fixed string. No assertion was relaxed.
- **Mutation sweep (one-off; script not committed): 21/21 caught.** Mutations: arm b without the flag; no `--scene-only`; S1 skipping the Check 9 check; no confirmation check; oldest Check 9 chosen; strings leaking into the excerpt; drop always declared; 120 s steady; images not advancing; progress not emitted on errors; scene-only calling the CUDA driver; no phase change in progress; non-fixed finish keys; a 4-image cycle; the cache flag always passed; `scene_progress` not allowed; the steady window ignored; scene-only requiring detector files; `--scene-only` not passed to the workload; raw text in cache event lines; `-1` accepted. Originals were restored byte for byte (`cmp`).
- **CPU-only smoke in `~/onvif_env`** (OpenCV/numpy; no model, GPU or network; run twice from the scratch directory with `PYTHONDONTWRITEBYTECODE=1`; nothing installed or written there). Six scene-only images resized and JPEG-encoded at quality 60 gave 6 distinct JPEGs of 61,995–62,202 B, identical across the two processes. Arrays were writable, and neither torch nor TensorFlow was imported. This is not a hardware or model result.

These are portable results, not CI, hardware, model-quality or memory acceptance. **PENDING:** Check 9 at this commit on the current boot; S1 arms a and b (`docs/U21_SHORT_VALIDATION.md`); U18 runtime policy; U19 unload/residue; U20 real-model and demo-exception evidence; the 30-minute U21 rerun (not approved); post-workload kernel candidate counts; file equivalence, consent and figure labels.

## Session 10 log (Claude, 2026-10-03): read-only review of Check 9 and the short U21

**Scope.** Read-only review and handoff reconciliation. `v2-beta` was at `f60652903ef6cf7656f6711e29b81c13e598cfb0`, clean. The local `origin/v2-beta` reference was the same commit (reflog: pushed 21:31 +0530, after session 9). Claude changed no code and ran no tests, workload, cache drop, service change, installation or reboot. It opened no secrets, clips, enrollment data, review images or local notes. Read: both sanitized `result.json`, `guard.jsonl`, and the profiler's `memory.csv`, `events.jsonl` and `profile.json`; byte-identical copies are kept outside Git. Recorded in session 11 with the maintainer's approval.

### USER-SUPPLIED MEASUREMENTS (maintainer, 2026-10-03, previous boot, repository f606529)
- **Preparation.** Desktop stopped (not disabled); remote dev processes ended; Ollama inactive and disabled; NVIDIA PVA allowlist service left running, classified as a known system service, its memory included in pressure. Inspection then refused only on MemFree (2.458 GB; MemAvailable 6.843 GB; pressure 1.147 GB).
- **One-time deviation, since superseded by D37.** With explicit approval, the maintainer ran `sync && sudo sysctl -w vm.drop_caches=1` once. The procedure then said not to drop caches, and this was not ongoing authorization. Inspection then had no refusals (MemFree 6.664 GB, MemAvailable 6.845 GB, pressure 1.145 GB). No further eviction occurred. Input was synthetic only.
- **Check 9** (15:05 UTC): `bounded_smoke_complete`. The device and managed probes each allocated 268,435,456 B in 33,554,432 B chunks, sequentially; rc 0, cleanup clear. Sampled peak pressure 1,284,661,248 B; min MemFree 6,474,420,224 B; swap unchanged. It does not test allocation beyond MemFree, server comparison or U18.
- **U21 short pilot** (15:08–15:11 UTC): `free_or_available_stop`, rc 1, cleanup clear. 1,114 guard samples over 224.101 s, max gap 0.270 s. Sampled peak pressure 4,270,002,176 B; min MemFree 1,069,268,992 B; MemAvailable at stop 3,720,007,680 B; swap unchanged; GPU guard passed. Profile `interrupted (SIGTERM)`, with no workload counters and no unload phase. Post-exit (~7 s) pressure 1,926,770,688 B vs baseline 1,268,269,056 B.

### Claude's calculations from those files (not measurements)
- **The stop.** The MemFree floor stopped the run, not pressure (530 MB under its stop) or MemAvailable (1.57 GB over its floor). MemFree fell 5,421,363,200 B = pressure +3,001,733,120 B + Cached +2,410,127,360 B; the page cache refilled during the loads after the drop. Margin over the floor: 383,733,760 B at warm-up start, 57,528,320 B at steady start. This neither shows nor rules out a leak.
- **Phases and deltas.**
  - Phase starts (s): baseline 0.98, llama 15.98, detector 35.98, face 59.33, warm-up 90.33, steady 120.33, stop 224.10. That is 103.77 of the 120 steady seconds.
  - Load deltas, including runtime overhead: scene +1,382,899,712 B, detector +805,322,752 B, face +452,481,024 B; warm-up +299,294,720 B.
  - Steady: pressure +52,338,688 B (non-monotonic), llama-server PSS +35,010,560 B, workload PSS +4,796,416 B, torch reserved constant at 23,068,672 B. Not extrapolated.
- **Unexplained residues.** About 0.67 GB of consumed MemFree was in no sampled field at the post-exit sample. CmaFree fell 219,897,856 → 8,192 B during the face load and was 158,253,056 B after exit. Check 9's +126,685,184 B residue persisted into the U21 baseline. Causes unknown.
- **Completed work and cadence cannot be verified.** Counters were emitted only on normal completion; session 11 adds per-request `scene_progress` events.

### Package-log reconciliation
- One upgrade ran 2026-10-03 08:10–08:15 UTC on that boot: 387 upgrades, no installs or removals, all configured.
- No L4T, kernel, bootloader, CUDA, cuDNN, TensorRT or GPU-driver package changed, and none of the D27 packages. Only the NVIDIA container-toolkit packages changed (1.19.1 → 1.20.1).
- Check 8, Check 9 and U21 all ran after the upgrade, on the same boot. Earlier rotated logs were not reviewed.
- Restart details: `docs/LOCAL_NOTES.md`. The reboot happened in session 11.

### Source findings (no code changed in session 10)
1. **Prompt cache.** llama-server runs without `--cache-ram`. The b8932 source defaults the host prompt cache to 8192 MiB (above MemTotal) and saves state plus an image copy for each request with a new image. Not confirmed at runtime (H2); E-2 and S1 follow up.
2. **Synthetic input.** U21's 16-frame input revisits about 4 scene images, so it can neither reproduce nor rule out Check 8's ramp. S1 sends a distinct image per request.
3. **Lost counters.** Workload counters are lost on interruption. S1 adds per-request progress events.
4. **Shutdown rows.** Samples taken during shutdown keep the steady label. Not fixed.
5. **Kernel checks.** Kernel OOM/NvMap candidates are counted only before the workload. Not changed.
6. **One status for two floors.** `free_or_available_stop` covers both memory floors, and the MemFree floor also applies during loads. This is a U18 question; no threshold changed.
7. **Torch allocator.** Torch's allocator held 23 MB while the detector added 805 MB.

### PENDING
U21's full steady window and attribution; the 30-minute rerun (not approved); U18 policy, including MemFree-floor semantics; Check 9b; U19 unload/residue; U20 real-model quality and demo-exception evidence (b8932 provisional); S1 run (approved in session 11, D38); post-U21 kernel candidate counts; file equivalence, consent and figure labels. Check 9 success and this short U21 grant no hardware acceptance.

## Session 9 log (Codex, 2026-10-03)

**Scope/preservation.** Authorized only the NVIDIA PVA service identity/classifier correction and a headless preparation checklist. Initial `git status --short --branch`, `git diff --stat` and cached diff were clean on `v2-beta` at `00f940f8cf05095ff7b8667a405f3a2fcb00ceb0`, ahead six of the local origin reference; no remote verification/fetch/push. No hardware validation, model/server/stream/GPU launch, installation, rebuild, service stop/disable, cache change, private media/face/secret access or rename occurred. NVIDIA PVA is left running. Only four permitted task files change; local-only and protected v1 files are untouched. No next package is started.

**Identity evidence, not hardware acceptance.** USER-SUPPLIED DIAGNOSTIC: PID 813's cgroup was `0::/system.slice/nvidia-pva-allowd.service`. Codex then inspected only read-only systemd/cgroup/package metadata with explicit 3–5 s timeouts (1 s kill grace):
- `systemctl show` identity fields report MainPID 813, that exact ControlGroup, root service user (empty User), loaded/active/running, no drop-ins, fragment `/etc/systemd/system/nvidia-pva-allowd.service`, description "service for managing PVA allowlists" and enabled state. `/proc/813` is root-owned and its cgroup matches. No Ollama/display-manager state was checked or inferred here.
- `dpkg-query` confirms installed `pva-allow-2` version **2.0.5**, architecture `all`, NVIDIA PVA SDK support maintainer; it owns both that unit and `/opt/nvidia/pva-allow-2/bin/nvidiaPvaAllowd.py`. Systemd's effective `ExecStart.path` is that daemon. Only the allowlisted executable path was extracted from captured launch metadata; arguments/environment and raw journals were never displayed or saved.
- Unit MD5 `2ba17cb53ea09d37ac5981ac28eda31e` and daemon MD5 `e5d5a04e8c40df662e8687df03c4f1ab` match the installed package's checksum records. This verifies installed package-file identity, not a cryptographic vendor attestation or proof of GPU/PVA inactivity.
- Limitations: sandboxed bus access was denied; allowed read-only metadata checks were repeated outside the sandbox. A proposed raw unit dump was rejected by auto-review because it could expose arguments/settings; it did not execute, and the safer selected-field extraction replaced it. Direct `/proc/813/exe` lookup is permission-denied for this root process; noninteractive `sudo -n readlink` reported that a password is required. **No live kernel executable symlink identity is claimed.** Classification uses authoritative systemd MainPID/cgroup plus the checked packaged launcher/unit, without requiring runtime sudo or reading process argv/environment. The system Python symlink resolves to `/usr/bin/python3.12`, owned by `python3.12-minimal`; that is installed interpreter metadata, not proof of PID 813's kernel image.

**Correction and tests.** No existing narrower service allowlist mechanism was found. `benchmarks/runner/operator_check.py` now checks only a root-owned Python process in the exact PVA cgroup against fresh active/running systemd MainPID, expected unit/launch paths, no drop-ins, installed package ownership and matching checksums of root-owned non-group/world-writable files. Identity subprocesses reuse `ProcessRunner` with 3 s deadlines/owned cleanup; output and file reads stay bounded. Errors, missing metadata, changed launcher/unit, unknown Python, different PID, different cgroup and non-root candidates fail closed. No PID is hard-coded, no generic Python whitelist is introduced. `known_system_services` reports only fixed labels/PIDs separately from interfering workloads. **PVA memory is not subtracted from MemTotal - MemAvailable**, and all existing admission checks/thresholds/guard settings are unchanged. Recognition is not proof of GPU idleness. `tests/unit/test_operator_check.py` adds 24 synthetic cases (80 total), including unrelated Python refusal/no workload launch, wrong MainPID, wrong metadata/package/checksum, timeouts, root/cgroup checks, unchanged device pressure and no argument leakage. `docs/U21_SHORT_VALIDATION.md` explains the exception and one explicitly operator-only reversible desktop checklist; no shutdown/restoration command ran.

**Portable verification (repo `.venv`, Python 3.10.14/pytest 9.1.1; status before commit):**

```bash
.venv/bin/python -m pytest tests/unit/test_operator_check.py tests/unit/test_demo_measurements.py tests/unit/test_bounded_gpu_probe.py
# final: 123 passed in 1.00s, exit 0 (earlier focused pass: 123 in 1.08s)
.venv/bin/python -m pytest
# 381 passed in 7.69s, exit 0
.venv/bin/sentinel config validate config/default.yaml
# valid Sentinel configuration (version 1, camera cam-1); core monitoring only, exit 0
git diff --check
# no output, exit 0
```

These are portable synthetic/configuration results, not fresh CI or hardware acceptance. Next operator action remains the default read-only inspection and returning its complete sanitized JSON; preparation/restoration commands are in the procedure. Check 9/U18, U19 unload, U20 real-model/demo exception, both U21 validations and outstanding maintainer decisions remain **PENDING**. No service shutdown, broader exception, threshold relaxation or following implementation is approved.

## Session 8 log (Codex, 2026-10-03)

**Scope and preservation.** Authorized increment: reconcile permitted evidence and provide one default-read-only operator entry point, with separately opted-in bounded Check 9/U21 diagnostics, portable refusal/timeout/cleanup tests and status before committing. Started on `v2-beta` at `79093b7120455d361cd5d70a71f8002d137f35e3`; initial status, working diff and cached diff were clean, ahead five of the **local** origin reference. No remote query/fetch/push, branch change, installation, llama.cpp rebuild, hardware inspection/workload, stream/model/GPU launch or service/cache-state change occurred. Only fake subprocesses/telemetry were used in orchestration tests. Protected v1 files and local-only files are untouched; secrets, private media, review images, face data and local notes were not opened/copied/deleted. No next package is started.

### Evidence reconciliation

- **USER-SUPPLIED FILENAME LISTING, not a measurement or media review:** the maintainer reports `empty_room_2026-09-29_1613.mp4` and `two_people_doorway_60s_2026-09-29.mp4`; `room_static_60s_2026-09-29.mp4` was not listed. Codex did not repeat that listing or inspect the files. **Do not rename anything.**
- Permitted Git history: `c865f14` records Claude's earlier `one_person` → `room_static` rename and unchanged SHA-256. `79093b7` records Codex's later requested `room_static` → `two_people_doorway` attempt failing before `mv` because the exact source was absent. No tracked successful second rename/equivalence record was found. The new listing is consistent with a rename elsewhere, but does not prove it happened or that the listed file has the earlier run's bytes. **File equivalence remains UNRESOLVED**, not a pending instruction to rename. Historical source frame/duration/hash records are not freshly verified facts about the newly listed file.
- Current Ollama stopped/disabled state is **not confirmed** (inactive and disabled are different). Full review of the empty-room clip and deletion of review images are **not confirmed**. Preserve all existing review images; no deletion is requested. Session 7's qualitative occupied-clip example review remains separately attributed, not full empty-clip review or 9,000-frame ground truth. Third-party consent and the early figure label remain unspecified; private clips stay local-only and excluded from datasets/enrollment without consent.
- U20 portable completion/schema work is complete. Real-model quality, shorter-description results, exact demo-exception evidence and hardware acceptance remain **PENDING**. D36 retains b8932 provisionally only; beta still requires a tested compatible schema-fix descendant. No threshold tuning, schema relaxation, rebuild or approval is inferred.

### What changed

| File | Scoped change |
|---|---|
| `benchmarks/runner/operator_check.py` | Standard-library entry point; default read-only inspection with 3 s subprocess deadlines, bounded output and fixed-label sanitization. Reports meminfo/swap, comm-only workload categories, service active/enabled states, boot/revision and current-boot kernel candidate counts. Separate Check 9/U21 result objects, prerequisite refusal, explicit execution options, streamed 0.2 s guard, deadlines and owned-process-group cleanup. |
| `benchmarks/runner/gpu_alloc_probe.py` | Replaces the old allocation-to-failure approach with at most 256 MiB per API in 32 MiB chunks, projected headroom checks, abort on unexpected errors and finally-based release. CUDA is loaded only in the explicit child workload, never on portable import. The parent supplies the 20 s deadline. |
| `benchmarks/runner/demo_profile.py` | Reused by the synthetic short pilot with `--no-evict --sanitized-logs`; sanitized mode drops arbitrary worker/server output and top-process names, retaining bounded known telemetry and fixed numeric GPU-placement/buffer diagnostics. Existing legacy mode is retained. |
| `tests/unit/test_operator_check.py` | 56 cases with injected fake subprocesses/telemetry: default read-only, privacy, service aliases, prerequisite/baseline refusal, API dependency failure, output bounds, swap/clock gaps, pressure overshoot, timeout, interruption, setup/provider errors, owned-parent/descendant cleanup, unrelated-process preservation, prior-report admission and separate U21 outcomes. |
| `tests/unit/test_bounded_gpu_probe.py` | 19 fake-allocator cases: explicit budgets, both APIs, baseline/projected-memory refusal before allocations, error abort and release of every acquired handle even after write/telemetry failures. |
| `tests/unit/test_demo_measurements.py` | Import-block regression additionally covers both new diagnostic modules without ML imports or hardware library initialization. Uses normal module registration for dataclass imports. |
| `docs/U21_SHORT_VALIDATION.md` | Replaces the inline proposed controller with the single tested entry point, conditions, memory definitions, sampling limitations, process ownership, privacy and separate execution modes. |
| `docs/IMPLEMENTATION_STATUS.md` | This pre-commit evidence/result update; removes unsafe current Check 8/9 instructions and supersedes the rename request without fabricating media/service/hardware verification. |

**What the checks mean.** Read-only kernel inspection resolves only availability/candidate counts in the latest 1,000 current-boot records, not unique OOM events or the full historical v1 incident. Check 9's opt-in bounded smoke tests whether each allocation API can initialize, allocate/write/release a small budget under admitted conditions. It deliberately does **not** answer allocations beyond MemFree, induce cache pressure, compare llama-server UMA settings or settle U18; those questions need a separately reviewed non-OOM plan/conservative-load decision. Device-phase failure blocks managed phase; neither mode automatically starts U21. U21 additionally requires a successful private same-boot/revision Check 9 report and explicit confirmation of reviewed U18/model-exception/workload/run prerequisites. The 120 s synthetic steady pilot investigates whether the historical trend recurs under different, recorded conditions; it cannot prove a linear leak, scene quality, long-run acceptance or unload. The 30-minute rerun remains separately PENDING.

**Metric and guard.** Whole-device pressure is `MemTotal - MemAvailable`, integer bytes (`/proc/meminfo` kB ×1024), sampled with `time.monotonic()` within an awake boot. Admission refuses pressure ≥2,000,000,000 B, MemFree <3,500,000,000 B or MemAvailable <4,000,000,000 B. During work, 0.2 s sampling stops on observed pressure ≥4,800,000,000 B, MemFree <1 GiB, MemAvailable <2 GiB, any swap-counter change, invalid/missing telemetry or a >0.5 s/backwards/nonfinite clock reading. **4.8 GB is a sampled stop threshold, not a guaranteed cap**: native/context allocations, guard overhead, transients between samples and shutdown may overshoot. Nominal 600 MB distance to the 5.4 GB target is not a proven safety bound. Guard overhead is in absolute pressure and the child baseline but remains unmeasured. Torch's ≤1 Hz process allocator current/lifetime peaks are separate, do not establish whole-device GPU usage and do not prove unload.

**Conditions, duration and cleanup.** Initial inspection takes seconds; each subprocess is bounded at 3 s with 0.2 s grace per signal. Workloads require normal-user plain SSH/headless operation with no existing v1, desktop/dev tools, model servers, unclassified Python or competing media/GPU work, cached assets, free port 18081, available process/service/kernel/revision inspection and conservative memory headroom. Comm-only inspection cannot prove all GPU users absent; operator review remains required. Check 9 uses up to two 20 s child deadlines; U21 reuses 15 s baseline/settling, 30 s warm-up and 120 s steady with 60/90 s readiness/load limits and a 360 s parent/descendant deadline (normally about 4–5 min). Cleanup TERM/KILL targets only this run's newly owned process group, with at most 5 s per signal for workloads, including descendants after parent exit; lingering groups fail dependent phases. Uninterruptible tasks are not guaranteed to disappear. A 5 s post-exit sample is not the pending U19 reclamation check. No service stop/disable, unrelated termination, cache-fill/eviction/drop, private replay or long run is included.

**Output and next action.** Mode-0700 `/tmp/sentinel-operator-*` directories and mode-0600 sanitized JSON/numeric diagnostics remain outside Git, irrespective of `TMPDIR`. No raw kernel/model text, credentials, private media or face templates are saved by this workflow. Run only the initial command in the next-task section and return its JSON; then identify actual conditions for the next separately authorized short step. Future hardware output must be labelled **USER-SUPPLIED MEASUREMENT** with conditions/limitations; this session receives no new hardware measurement. Check 9/U18, U19 unload, U20 real-model/demo exception, U21 short/long acceptance and maintainer decisions all remain **PENDING**.

### Verification: exact commands and results

Repository `.venv`, Python 3.10.14, pytest 9.1.1; nothing installed. Status is updated before committing this scoped increment.

```bash
.venv/bin/python -m pytest tests/unit/test_operator_check.py tests/unit/test_bounded_gpu_probe.py tests/unit/test_demo_measurements.py tests/unit/test_demo_scene.py
# final: 108 passed in 1.05s, exit 0
.venv/bin/python -m pytest
# 357 passed in 7.49s, exit 0
.venv/bin/sentinel config validate config/default.yaml
# config/default.yaml: valid Sentinel configuration (version 1, camera cam-1)
# no optional adapters configured: core monitoring only; exit 0
git diff --check
# no output, exit 0
```

Earlier focused iterations honestly reported **5 failed, 74 passed in 0.91 s** (a test incorrectly expected absent update keys to become unavailable despite valid baseline values), then **1 failed, 106 passed in 1.14 s** (the synthetic import harness did not register modules in `sys.modules` before executing a dataclass-bearing module). Corrected test harness/assertions, without relaxing admission/guards; subsequent 107-case focused run passed in 0.99 s, and the added setup-cleanup regression yields the final 108/357 results above. No configured lint/type checker was added or run. These are portable behavioral/configuration results, not fresh remote/CI, inspection, model quality, memory, U18 or hardware acceptance.

## Session 7 log (Codex, 2026-10-03)

**Scope and preservation.** Maintainer authorization: portable U20 schema-constrained generation with unchanged strict parsing, investigation of length-limit/counter evidence without opening private media, provisional retention of the current demo build, and preparation only of a guarded U21 procedure. Started on `v2-beta` at `805d837e78409a9958749a403707a644b1bc66bf`; `git status --short --branch`, `git diff --stat` and `git diff --cached --stat` showed a clean tree, ahead four of the local origin reference. No fetch/push, branch change, dependency installation, rebuild, model/server/camera/GPU run or service change occurred. No media contents, face data, secrets or local notes were opened/copied. A later explicit request authorized a filename-only rename attempt; the exact source was absent at the suggested location, so nothing changed (details below). No new USER-SUPPLIED MEASUREMENT was received; the maintainer supplied a qualitative clip review. Local source/build metadata inspection is not hardware verification.

### Portable implementation

| Path | Change and evidence |
|---|---|
| `src/sentinel/scene/completion.py` | Generate llama-server `response_format: json_schema` directly from `SceneReport.model_json_schema()`; fail closed on server errors, malformed/multiple choices, missing/non-stop completion, refusals/tool calls and non-text messages. `finish_reason: length` is a distinct `truncated` rejection even when text happens to parse. A normal stop still passes through the unchanged `parse_scene_report()`; no clipping, JSON repair, limit relaxation or scene-accuracy inference. |
| `benchmarks/runner/demo_workload.py` | Request up to three short observations (aim <48 characters) and one complete summary sentence (aim <100); schema bounds remain five observations, 80/160 characters. Keep 200 output tokens, image size/quality, cadence and detector thresholds unchanged. Non-streaming schema requests; reject oversized/truncated HTTP bodies rather than silently accepting a clipped prefix. No unchecked fallback when validation dependencies are unavailable. Add fixed-label completion/rejection counts and valid-field-at-limit counters, not raw text or histories. |
| `benchmarks/runner/demo_profile.py` | Display the new counters when present; older run summaries show `unavailable`, not fabricated zero. Explicitly separate structural validity/character bounds from accuracy and semantic completeness. |
| `tests/unit/test_scene_completion.py` (42 cases) | Unchanged generated schema; fresh request schemas; exact-boundary/fenced complete content preserved; token truncation rejected even with valid JSON; missing/unsupported completion, malformed envelopes, refusals/server errors; strict type/size/extra-field/JSON rejection; bounded safe diagnostic labels; structurally valid but inaccurate synthetic report. |
| `tests/unit/test_demo_scene.py` (9 cases) | Fake HTTP/JPEG request wiring with no real network/ML; short prompt/schema/token limits; partial/oversized response rejection; aggregate-only rejection/boundary telemetry; absent validator fails before a request; new/legacy summary rendering; synthetic looped decodes and repeated latest-frame face calls establish counter semantics without opening footage. |
| `docs/U21_SHORT_VALIDATION.md` | Guarded operator-only short pilot with explicit metrics, reserves, timeouts, owned-group cleanup and stop conditions. Python syntax checked via AST only; **not executed**. |

Only bounded scalar/fixed-label counters are new retained scene telemetry. Response reads are bounded at 1,000,001 bytes (one extra byte detects overflow); report content remains capped at 4,096 characters by the unchanged parser. No raw report/image is stored. Existing run-duration latency/token lists are unchanged; this is not a new retention/quota policy. The portable completion helper is available for the future real adapter, but no live adapter, D-1 wiring, manifest admission or full V2-26 acceptance is implemented here.

### U20: what the 22/24 boundary cases establish

Source review: the tracked workload sends/reads model content, then parses it without slicing `summary` or observations to 160/80. Its old 1,000,000-byte HTTP read was a body bound, not a field-character clipping rule; this increment explicitly rejects overflow. The local llama.cpp revision's `common/json-schema-to-grammar.cpp` builds a bounded repetition followed by a closing quote for `maxLength`, not an application-side `summary[:160]`. `tools/server/server-task.cpp` reports `length` unless the stop is EOS/stop-word; schema closure and token-budget termination are different conditions. Its `server-common.cpp` accepts the `json_schema` wrapper used here.

The recorded constrained reproduction had 24/24 strict-valid reports, 121–183 completion tokens (below the 200-token budget) and 22 summaries at 160, reportedly mid-word. This supports **generation constrained at a field boundary** rather than application clipping in the tracked path or exhausting the configured token budget. It does **not** establish the 22 requests' actual finish reasons: constrained-response finish metadata/raw reproduction code was not archived in tracked evidence and private diagnostics were not read. Complete JSON exactly at the limit can have `finish_reason: stop`; a mid-word natural-language ending can still be semantically unfinished. Neither length alone nor strict-valid JSON proves semantic completeness. The historical claim that the grammar "cuts them off" is qualified below, not promoted to proof of transport truncation. Application clipping in an unarchived reproduction helper cannot be conclusively excluded.

Separately, the historical unconstrained run explicitly recorded seven `finish_reason: length` cases at 200 tokens; these are token-budget truncations. They must not be confused with the 22 constrained field-boundary cases. New output distinguishes `truncated`, structurally invalid stopped content, incomplete/unsupported envelopes and strict-valid boundary hits. Synthetic tests verify this distinction; no new real-model rate is claimed. The short prompt's effect on grammar-boundary hits, language completeness and latency is **PENDING**. Do not add punctuation/length heuristics, relax the schema or tune pass rates in lieu of actual completion metadata and maintainer review.

### Demo build record and exception evidence (D36)

- Retain **b8932 / `98dc1418ea0491d62948f712ed534ece3b773564`**, dated 2026-04-25, provisionally for the demo. Read-only `git -C /home/villain8001/llama.cpp rev-parse HEAD` and `git ... log -1 --format='%H %cs %s'` confirm that source revision. Existing `build/common/build-info.cpp` records b8932, abbreviated commit `98dc1418e`, GNU 13.3.0, Linux aarch64. No binary was executed or rebuilt; generated build metadata does not freshly prove the executable's identity.
- Historically tested pair: `LFM2-VL-1.6B-Q4_0.gguf`, SHA-256 `ce0d4b122d328d14390ef160785da3a51a527f96844f392a04cb2db96f134e5d`; `mmproj-LFM2-VL-1.6B-Q8_0.gguf`, SHA-256 `65ec437db88d65fff93f472d00c145e09880769ac67fedff5cd1c0f8d8301d87`. These are the session 3/5 recorded hashes, not newly rehashed assets. GGUF-embedded chat template, no template override in the tracked launcher; the historical reproduction logged **`peg-native` on all 48 requests**, not the specialized LFM2 handler. The GGUF hash pins its embedded metadata; template text/separate hash was not archived or newly inspected.
- b8932 predates [llama.cpp PR #24377](https://github.com/ggml-org/llama.cpp/pull/24377), merged 2026-06-10, fixing LFM2/LFM2.5 handlers ignoring `json_schema`. The inspected old `is_lfm2_template()` selects that path from tool-list markers. Historical `peg-native` enforcement is evidence for this exact pairing only, not other templates or builds. **Beta still requires a pinned, tested compatible descendant satisfying the guide/audit and actual model/template tests.** No rebuild/revision selection is approved here.
- Demo-exception acceptance remains **PENDING**: operator-confirm the actual binary revision/hash, model/projector hashes and embedded template/handler; show schema enforcement with approved synthetic/sanitized scenes including adversarial extra/type/length requests; verify strict rejection of incomplete/error responses and report finish reasons/tokens/boundary counts with the shorter prompt; maintainer-review description completeness and labelled scene accuracy separately (including empty/edge/occlusion cases); repeat full GPU placement, memory/headroom and latency evidence under recorded conditions. 24/24 valid structure alone grants neither accuracy nor hardware admission. A changed template/binary invalidates this provisional evidence.

### two_people_doorway_60s counter investigation — no media opened

The counter/read-loop implementations are unchanged between the recorded Check 8 source `d85eb1e` and session 7 start, apart from cadence/instrumentation additions. Findings from permitted code and historical records:

1. **9,000/9,000** means 9,000 successful detector `track()` calls and 9,000 decode/delivery events in the reset 600 s steady counters, at about 15/s. `frames_with_person` increments once per call if its returned `boxes` is nonempty, regardless of number/identity of people. It is an inference-result count, not human-labelled true-positive frames. There is no copied previous `results` in the benchmark's detector loop; each counted call invokes inference on the newly delivered latest frame. Catch-up decodes can be skipped for inference, although the recorded equal totals show none in that aggregate.
2. The source **reopens the same clip at EOF**. It was recorded as 901 frames; 9,000 events cannot be 9,000 unique clip-frame positions. The `clip_loops=11` counter covers the whole workload (warm-up plus steady), not a steady-only count. No source frame index/PTS/hash or per-stage delivery identity was logged, so exact distinct-image counts and loop-by-loop reuse cannot be reconstructed. The same source content is replayed. Tracker state also persists across calls/rewinds; there is no tracker reset at EOF.
3. The inspected installed Ultralytics tracker callback uses the current inference boxes in `tracker.update()`, then returns matched/current track boxes; ByteTrack predicts positions for association, updates matched tracks from detections and marks unmatched tracks lost. Returned boxes can be Kalman-smoothed track state, not raw detection coordinates. This code path does not export lost-only extrapolations as a fresh person frame; when no tracks return, the callback can leave current raw detection results. It is therefore a mixture of current detection/post-tracking results, **not verified ground truth or independent fresh evidence per original source frame**. Historical per-frame tensors/observed-versus-predicted flags and exact tracker dependency identity were not archived in tracked evidence; those are not fabricated from this code inspection.
4. **82 of 671 face runs** means 82 successful `DeepFace.represent()` calls with at least one returned `face_confidence > 0`. Multiple faces still increment once; 671 counts successful calls, with errors counted separately. Each call computes a new representation; this runner does not reuse a cached face verdict. The independent face thread can process the **same latest frame repeatedly**, has no source novelty/age check and does not count unique frames/faces/people. No enrollment or identity match occurred, so 82 is not 82 identified people or calibrated face recall. Worker calls can straddle the steady counter reset; source provenance is insufficient to resolve that boundary precisely.
5. Initially the tracked record contained only **Claude's** review, conflicting with the maintainer's empty-room description. **USER-SUPPLIED REVIEW, 2026-10-03 (session 7):** the maintainer corrected that description after personally reviewing raw frames at 5/30/55 s and Claude's annotated frame_0040/frame_0450/frame_0534, doorway zoom and contact sheet. They confirm their own moving presence, a second real doorway person later in the clip, the early face box on themselves, and **no empty-room period**. They judge detections largely correct. This resolves the presence dispute, but is a qualitative review of examples, not ground truth for 9,000 deliveries or calibrated recall. It cannot serve as an empty-room negative/false-alert clip; an occupied clip can still contain individual spurious boxes (Claude's chair interpretation remains a separate observation).
6. The reply leaves the third party's consent and the early bottom-right figure's person/photo/reflection label as bracketed alternatives. No identity or consent is inferred; Codex requested clarification without names. The maintainer's follow-up still leaves those alternatives unresolved but explicitly permits **local memory/latency profiling**; this does not authorize hardware execution in this turn or the pending short/30-minute runs. The clip contains a third party and **must remain local-only, excluded from datasets and enrollment until that person's consent is established**. It also has a burned-in timestamp/logo, untrusted image text that must never control policy. No raw frames, annotations, identities, face data or private-path/hash details from local notes enter this commit. No detection threshold is changed.
7. Judge the recorded VLM answers against the supplied description: its continuous-person-presence label conflicts with the historical constrained reproduction's **four zero-person answers out of 24**. Those reported answers are presence misses relative to that operator description, **not a new measured accuracy rate**. Exact counts for the 17 one-person answers cannot be adjudicated without frame-specific labels/visibility and the unresolved early figure; threat/fire and prose accuracy cannot be determined from aggregate counts. Structural 24/24 validity remains separate. No new model/image comparison was run.
8. At session 7 end, the requested filename was `two_people_doorway_60s_2026-09-29.mp4` (requested alias for `room_static_60s_2026-09-29.mp4`, earlier `one_person_2026-09-29_1606.mp4`), but the private rename was pending. With explicit filename-only permission, Codex attempted exact-filename metadata checks and a no-overwrite/no-symlink rename (8 s outer timeout, 3 s `mv` timeout if reached). Result: source absent, exit 1 before `mv`; **nothing changed**, no directory listing/media opening/copying. Session 8's user-supplied listing supersedes the pending request: **do not rename anything**; no successful second rename or file equivalence is established by tracked history.

### U21 proposal — still PENDING

Treat the historical +0.12 GB/min window and pressure-associated flattening as a trend to investigate, not a proven linear leak or GPU allocation attribution. Stable RSS/PSS and release on process exit suggest an investigation direction but do not distinguish torch, TensorRT/native buffers, driver/kernel memory, cache accounting or other services. Allocator and device metrics must remain separate.

`docs/U21_SHORT_VALIDATION.md` proposes a **synthetic-input** 120 s steady pilot, 0.2 s device guard, 1 Hz allocator telemetry, 4.8 decimal GB immediate pressure stop, 1 GiB MemFree/2 GiB MemAvailable floors, abort on new swap activity/missing samples/>0.5 s gaps, and a 360 s total child/descendant timeout. Only its newly owned process group is terminated then killed after bounded grace; no existing service is stopped. Headroom reserves accommodate but do not bound unobserved transients; guard overhead is included in absolute pressure and remains unmeasured. It deliberately disables file-cache eviction and is not an equal cold-cache/private-replay repeat. Typical duration 4–5 min; maximum 360 s plus 10 s cleanup and 5 s post-exit reading. This is a proposed safer short step, not approval or a 30-minute run. Check 9/U18 load policy, runtime/dependency availability, services, model/template evidence and separate operator authorization precede it. A short plateau would still not establish long-run acceptance or unload.

### Verification: exact commands and results

Repository `.venv`, Python 3.10.14, pytest 9.1.1; no dependencies installed. Status updated before committing this verified increment.

```bash
.venv/bin/python -m pytest tests/unit/test_scene_completion.py tests/unit/test_demo_scene.py tests/unit/test_scene_report.py tests/unit/test_demo_measurements.py tests/unit/test_portable_imports.py
# final: 88 passed in 1.61s, exit 0
.venv/bin/python -m pytest
# final: 282 passed in 7.08s, exit 0
.venv/bin/sentinel config validate config/default.yaml
# valid version 1, camera cam-1; no optional adapters configured; exit 0
git diff --check
# exit 0, no whitespace errors
.venv/bin/python - <<'PY'
import ast
from pathlib import Path
path = Path('docs/U21_SHORT_VALIDATION.md')
recipe = path.read_text().split('```bash\n', 1)[1].split('\n```', 1)[0]
body = recipe.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
ast.parse(body, filename=str(path))
print('U21 operator Python: syntax valid; no hardware code executed')
PY
# syntax valid, exit 0; parses only, never executes the proposed operator program
```

Initial focused result: 1 failed, 83 passed in 1.58 s. The test used equality to check whether a raw response was retained; its empty response equalled an unrelated empty aggregate dictionary. It now checks object identity. The corrected then-84-case suite passed in 1.56 s; HTTP-body regressions brought it to 86 passed in 1.56 s (then-full 280 passed in 7.08 s). Two new/legacy summary regressions brought the final focused count to 88 and full count to 282 above. `src/sentinel/scene/report.py` has no diff; existing strict-parser/freshness/scene-lane tests pass. These are portable results, not fresh CI, real-model, scene-accuracy, memory or hardware acceptance. Check 9/U18, U19 unload, U20 hardware/build exception, both U21 runs, detailed clip labels/consent and service decisions remain PENDING. **No following implementation package is started.**

## Session 6 log (Codex, 2026-10-03)

Scope authorized by the maintainer: portable U19/U21 measurement instrumentation only, then its tests, status update and commit. Started on `v2-beta` at `b872c27` with a clean working tree. No branch switch, fetch, push, model/server launch, camera/GPU probe, dependency installation, service change or private-media access ran. Local-only files and protected v1 files are outside the commit.

### What changed

| Path | Purpose |
|---|---|
| `benchmarks/runner/demo_profile.py` | Sample `Shmem`, `Unevictable`, `Mlocked`, `SUnreclaim`, `KReclaimable` and `CmaFree`; convert Linux `kB` explicitly as 1024 bytes; omit absent/invalid readings. CSV blanks and columns missing from older files load as `None`, never fabricated zero. Summary event loading streams the JSON-lines file and retains at most six latest summary records, excluding allocator history. |
| `benchmarks/runner/demo_workload.py` | `AllocatorSampler` with injected provider, clock and event sink; lazy runtime provider reads `torch.cuda.memory_stats(0)`. Emits allocated/reserved current and lifetime-peak bytes, phase, sample-start timestamp and availability. One sampling thread, at most 1 Hz, no catch-up burst or sample-history collection; stopped in `finally` on success or face/workload failure. Missing/invalid values stay `None`; failures expose the exception class only. |
| `tests/unit/test_demo_measurements.py` | 24 portable cases: ML/hardware-blocked imports, all six field conversions, missing/invalid values versus measured zero, new CSV round trips and legacy loading; injected allocator values/peaks, UTC steps, device-0 provider wiring, preservation of source timestamp through the orchestrator, cadence, discarded provider snapshots, stop behavior, and summary accounting independent of allocator values. Only synthetic temporary fixtures and fake providers are used. |

**Clock, units and bounds.** CSV and allocator `t_mono` are sample-start `time.monotonic()` seconds (millisecond serialization precision), comparable within the manifest's boot; on Linux this clock excludes suspend (D5). The allocator source timestamp survives the orchestrator relay rather than being replaced by receive time. Meminfo `kB` means KiB (1024 bytes); stored memory readings are integer bytes; displayed GB remains decimal. Meminfo streams every 0.2 s; allocator snapshots stream at most every 1 s from detector-settle through the workload, without a retained snapshot history. Raw files grow with the configured run duration; existing offline CSV/tegrastats analysis still loads their rows. The new allocator history is not retained by summary loading. No retention/quota policy was added.

**Accounting limits.** Torch readings describe this process's torch CUDA allocator on device 0 only. They exclude llama-server, TensorRT/native allocations outside torch and other device allocations; allocator bytes, RSS/PSS, tegrastats and `MemTotal - MemAvailable` are distinct views, never added together. Peaks cover allocator lifetime and are not reset at the steady phase. Sampling does not synchronize CUDA, empty caches or alter allocation policy. Allocator values cannot establish total GPU/device usage or prove unload; the sampler ends while its workload process still exists. Sampling overhead, real torch/Jetson behavior and U19/U21 attribution/reclamation require operator measurements and remain PENDING.

### Verification: exact commands and results

Repository `.venv`, Python 3.10.14, pytest 9.1.1; no packages installed. The status record is updated before the task commit.

```bash
.venv/bin/python -m pytest tests/unit/test_demo_measurements.py  # final: 24 passed in 0.49s, exit 0
.venv/bin/python -m pytest                                     # final: 231 passed in 6.86s, exit 0
.venv/bin/sentinel config validate config/default.yaml         # valid version 1, camera cam-1; no optional adapters configured; exit 0
```

The initial focused run was 1 failed, 21 passed in 0.56 s: a test incorrectly assumed the existing FakeClock starts at zero; it now asserts the injected clock's actual start. The corrected 22-case run passed in 0.48 s and the then-full 229-case suite passed in 6.99 s. Two further provider/relay regressions were added before the final successful commands above. These are portable results, not hardware acceptance or fresh CI results.

**Hardware and decisions remain PENDING.** Check 9/U18, U19 residue/unload, U20 constrained scene output and llama.cpp build, U21's 30-minute rerun, clip interpretation and the Ollama stop decision are unchanged in approval state. Session 5's check 8 remains the historical maintainer measurement, with its conditions and limitations; no new USER-SUPPLIED MEASUREMENT was provided in session 6. The next operator action is the decisions/check 9 recipe above; no following implementation package is authorized by this increment.

## Session 5 log (2026-10-03)

### Check 8: demo resource profile (maintainer's measurement)

Run `demo-profile-20261003T085010Z`, run by the maintainer on 2026-10-03 from 08:50 to 09:04 UTC. Headless (display manager inactive, no desktop processes, no dev tools), after `drop_caches`, with MemFree at 6.83 GB at the baseline. Repository at `d85eb1e` with no tracked changes; GPU guard (D27) passed: 17/17 layers and the vision encoder on CUDA0, only L4T's libcuda mapped, `cuInit` 0 in the workload. Input: the historical `one_person`/`room_static` replay, looped 11 times (Claude recorded unchanged SHA-256 for that earlier rename). The newly listed `two_people_doorway_60s_2026-09-29.mp4` is a requested alias whose equivalence remains unresolved; these run facts are not newly verified for it. Output stays in `~/sentinel-runs/<run id>/`. One cold load and one combined run: a provisional-demo profile (D28, D33), not a benchmark or gate result. Memory is whole-device `MemTotal − MemAvailable` in decimal bytes.

| Item | Result |
|---|---|
| Baseline | 0.906 GB (median of 30 s) |
| Cold loads, one after another | scene (llama-server) 4.1 s, +1.728 GB settled · detector 12.72 s, +0.791 GB · face 15.75 s, +0.453 GB |
| Run peak (includes cold loads) | **5.350 GB**: 50 MB under the 5.4 GB ceiling |
| Steady phase (600 s) | median **5.036 GB**, p95 **5.347 GB**: above the 5.0 GB target (but see "memory ramp" below) |
| Swap | at most 0.79 MB used; 0 pages swapped in and 168 out during the steady phase |
| Per process (not additive) | llama-server peak RSS 3.296 GB, PSS 1.686 GB · workload peak RSS 2.007 GB, PSS 1.837 GB |
| Detector | 9,000 of 9,000 source frames at 15.0 fps; latency p50 45.4 ms, p95 56.4, max 76.3; a person on 9,000 frames, at most 3 |
| Face | 671 runs, **1.12 Hz achieved against 2.0 Hz configured**; p50 896 ms, p95 922 ms; faces in 82 runs; no errors |
| Scene | 150 completed; p50 3.63 s, p95 4.32 s, max 4.47 s; none over the 8 s D16 timeout; **5 valid, 145 invalid** reports; mean 305 prompt and 163.6 completion tokens |
| tegrastats, steady | CPU mean 27.2 % over all cores (busiest 30.6 %); GR3D mean 82.1 %, max 99 %; maximum temperature 72.8 °C (tj, GPU); mean VDD_IN 17.65 W, VDD_CPU_GPU_CV 7.74 W; RAM max 5.187 GB (tegrastats accounting) |
| Unload | after the workload exits: 0.612 GB below the level before it started · after llama-server stops: **+0.983 GB above the baseline** (U19) |

**Idle baseline (maintainer's measurement).** The earlier run `demo-profile-20261003T084514Z` (08:45 UTC, same boot, headless, no `drop_caches`) completed its 30 s baseline phase and then refused to start the GPU loads, as designed: MemFree was 2.72 GB, below the 3.0 GB precheck (U18), with MemAvailable 6.99 GB and Cached 4.06 GB. Its baseline, **0.970 GB headless idle**, is the idle baseline. Its manifest lists the largest resident processes.

**Run input.** The maintainer initially described an empty room, conflicting with Claude's review below. In session 7 the maintainer personally reviewed examples and corrected this: real people, no empty period, requested `two_people_doorway_60s_2026-09-29.mp4`. Third-party consent/one early figure remain unspecified; this is not an empty negative or a consented dataset. Replay-delivery and face-call counters are not per-unique-frame ground truth. Historical memory/latency remain under their original conditions; no new measurement ran.

### Claude's follow-up diagnostics (session 5; observations, not measurements)

Run with VS Code and Claude Code open, on the same boot. Model output, frames and annotations were written only under the run directory (modes 0700/0600), never into the repository. llama-server ran twice, bound to 127.0.0.1, and was stopped each time (no process left, port free). Memory and latency from these runs are not recorded as measurements.

1. **Detector and face on the clip** (`annotated/`). Same engine and v1's `track()` arguments on all 901 frames; DeepFace/YuNet on every 10th frame.
   - A person was detected on 901 of 901 frames (one box on 626, two on 271, three on 4), with confidences from 0.408 to 0.951 (median 0.83). It was the same 901/901 without the tracker.
   - Faces (confidence > 0) appeared on 12 of 91 sampled frames (13 %), consistent with the run's 82 of 671 runs (12 %).
   - Claude viewed three annotated frames, a 16-frame contact sheet and an enlarged crop. A person is in the foreground for most of the clip, and a second, small, distant person in another room is visible in the far doorway for the rest. The camera view changes once.
   - One false positive was found: a chair back at 0.47 (frame 534), next to two real people.
   - **Historical Claude interpretation, qualified in session 7:** the maintainer now personally confirms real people/no empty period and requests the new descriptive filename. The 9,000/9,000 counter still does not establish per-frame accuracy; remaining labels/consent are PENDING. No threshold tuning follows from these counters.
2. **Scene reports** (`vlm_repro/`). The run stored no model text (by design: `demo_workload.py` records counts only). Claude reproduced the requests with the same model files, flags, preload, image shape, prompt and temperature on 24 clip frames (every 38th): **1 of 24 valid** (the run had 5 of 150).
   - All 24 were well-formed JSON objects with the right types; 7 were inside a ```` ```json ```` fence, which the parser accepts. The cause is **not** syntax strictness or the types in the schema.
   - 16 of 24 had a `summary` of 168–364 characters (limit 160).
   - 7 of 24 stopped at `max_tokens` 200 (`finish_reason: length`) before reaching `summary`, after five long observations of up to 95 characters (limit 80). In the run, 14 of 179 requests reached 200 tokens.
   - **Cause: the prompt asks for character limits that the 1.6B model does not follow**, and the schema's bounds are only checked after generation. The schema's limits are reasonable.
   - **Proposed fix, tested in the historical reproduction:** the same prompt plus the SceneReport JSON Schema as llama-server's `response_format`. The recorded build b8932 applied the grammar bounds: **24 of 24 strict-valid**; 121–183 completion tokens; p50 2.9 s, max 3.7 s. 22 of 24 summaries reportedly ended at exactly 160 characters, mid-word. Session 7 qualifies this as field-boundary generation evidence, not proof of token-budget/transport truncation or semantic completeness; constrained finish reasons were not recorded in tracked evidence. See U20/session 7.
   - **Person counts.** What the run's 5 valid reports said is unknown, because the run did not record report content. In the reproduction, the one valid unconstrained report said 1. With the schema, reports said 1 on 17 frames, 2 on 3 and 0 on 4. Claude interpreted the four zero frames (418, 798, 836, 874) as edge/distant-person misses. Session 7's maintainer-confirmed continuous-person-presence description conflicts with those four zeros; exact per-frame counts and remaining labels are not fully adjudicated. Occupancy never comes from the VLM (D18).
3. **Memory ramp (from the run's `memory.csv`).**
   - Used memory rose by about 0.12 GB per minute, from 4.24 GB at the start of warm-up to 5.30 GB in minute 9 of the 12-minute window. It flattened at 5.32–5.35 GB only when MemFree reached its floor (0.175 GB) and pages began to be swapped out.
   - Over the same period, workload RSS (2.00 → 1.79 GB) and llama-server PSS (about 1.66–1.69 GB) stayed flat. **Corrected in session 11 (E-2):** llama-server PSS was not flat. It rose 711,694,336 → 1,660,114,944 B during steady (about +104 MB/min) while workload PSS stayed flat; 1.66–1.69 GB were its end values.
   - When the workload exited, used memory fell by about 3.2 GB (5.25 → 2.02 GB), far more than its cold loads (+1.24 GB). Claude hypothesized GPU/native allocations not represented by process RSS. Session 7: this correlation is an investigation lead, not attribution to torch/TensorRT/Ultralytics or proof of a linear leak; driver/kernel/cache accounting and other services are not ruled out.
   - **The steady median and p95 describe a ramp, not a plateau**, and the 5.35 GB peak may be where free memory ran out rather than what the workload needs (U21).

### V2-28 demo form (incident side): what was added (`45d9ba3`)

| Path | Purpose |
|---|---|
| `src/sentinel/storage/database.py` | Migration 2: `incident_annotations` (evidence ID unique, incident, kind, status, producer and revision, applicability on arrival, source time, recorded time, payload ≤ 8,192 characters). Version 1 databases upgrade in place. |
| `src/sentinel/incidents/service.py` | `IncidentService.annotate(evidence, applicability)` and `annotations(incident_id)` (D35) |
| `tests/unit/test_incidents.py` (+6) | Late fire report on A: A's status, severity, title and times unchanged, no new incident, no outbox row, B untouched, and an operator holding A's old revision must reload; a repeat after a restart is a duplicate; evidence naming no incident is not stored; unknown incident, another camera and oversized payload refused; resolved incident stays resolved under a timeout and a late result; v1 → v2 upgrade keeps incidents |
| `tests/replay/test_v2_28_enrichment.py` (1) | R3's late-fire timeline into a real store, with `inc-A` and `inc-B` opened by zone signals at their timeline times: periodic reports are not stored, A gets `job-2.timeout` (current) and `job-2.observed.late` (expired), B gets nothing, nothing escalates or notifies, and replaying everything after a restart changes nothing. B0 v1 sends a fire alert at 11.5 s on the same answers. |

**V2-28 mutation sweep (one-off; script not committed): 12/12** in the first round. Mutations: annotate the newest unresolved incident instead of its own; no deduplication; an observed fire escalates severity; a late result reopens a resolved incident; annotation notifies; no camera check; no revision bump; periodic evidence attached to the newest incident; applicability not stored; nothing written; unknown incident raises; no size bound. The last one may have been caught by the table's `CHECK` rather than the outcome check; the bound holds either way.

### Session 5 verification: exact commands and results

All ran on 2026-10-03 on the Jetson from `~/sentinel-surveillance`. Nothing was installed into or written to `~/onvif_env` (`find -newer` on a marker: 0 files).

```bash
.venv/bin/python -m pytest                                  # 207 passed in 6.5s (Python 3.10.14, pydantic 2.13.5)
.venv/bin/sentinel config validate config/default.yaml      # valid, exit 0
/usr/bin/python3 -m py_compile benchmarks/runner/*.py       # ok; demo_profile.py --help shows --face-hz
```

- Each session 5 commit passes on its own (archive loop): `c865f14` 200 · `45d9ba3` 207 passed.
- Clean archive of `45d9ba3` on Python 3.12.3 (fresh venv): 207 passed.
- Inside `~/onvif_env` (`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`, database in the session scratchpad; pydantic 2.12.5, SQLite 3.45.1): a zone signal recorded, then a timeout annotation: `annotated`, then `duplicate`, status `open`.
- The detector, face and VLM diagnostics above, with their limits.

## Session 4 slice log (2026-10-03)

Commits on `v2-beta` after `32985c2`, all local until the maintainer pushes them:

| Commit | Content |
|---|---|
| `9ec10cf` | Session 3's record completed (this file) and the U17/U18 benchmark scripts. Also, outside the repository: the excluded local notes gained the remote-desktop observation, and the maintainer's Claude Code auto-mode settings now state that the repository is public (maintainer request). |
| `b8102b9` | V2-13 demo form: restricted-zone and dwell rules (D30) |
| `a30445c` | V2-14 demo form: SQLite incidents, observations, evidence, transitions and outbox (D31) |
| `1c69490` | V2-15 demo form: leased outbox worker and Telegram adapter, mocked (D32) |
| (this commit) | Session 4 verification record and next task |

### Session 4 verification: exact commands and results

All ran on 2026-10-03 on the Jetson from `~/sentinel-surveillance`. Nothing was installed into or written to `~/onvif_env`. No hardware check ran in session 4.

```bash
.venv/bin/python -m pytest                                  # 200 passed in 6.2s (Python 3.10.14, pydantic 2.13.5)
.venv/bin/sentinel config validate config/default.yaml      # valid, exit 0
/usr/bin/python3 -m py_compile benchmarks/runner/*.py       # ok; demo_profile.py and gpu_alloc_probe.py --help ok
```

- **Every session 4 commit passes its own tests** (session 1's archive loop over `32985c2..HEAD`): `9ec10cf` 123 · `b8102b9` 151 · `a30445c` 169 · `1c69490` 200 passed.
- **Clean archive of `1c69490` on Python 3.12.3** (fresh venv): 200 passed; `sentinel config validate` valid. v1 "Dashboard checks" in the same archive (Flask 3.0.3, requests 2.32.3, python-dotenv 1.0.1): `Ran 4 tests ... OK`.
- **pydantic 2.12.5** (scratch venv on `/usr/local/bin/python3.10`, as in session 2): 200 passed.
- **Inside `~/onvif_env`** (`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`, cwd and database in the session scratchpad; pydantic 2.12.5, SQLite 3.45.1): imported all 36 `sentinel` modules. Then FakeClock frames, an `EdgeCore` zone entry at 23:00 Kolkata, `IncidentService.record()` and an `OutboxWorker` pass with a fake notifier: `['entered'] -> ['created']`, `[(1, 'sent')]`. `find ~/onvif_env -newer <script>` found no changed files.
- Mutation sweeps: V2-13 22/22, V2-14 20/21 plus one equivalent mutant, V2-15 23/23 (details in each package's section above).

### V2-15 demo form: what was added

| Path | Purpose |
|---|---|
| `src/sentinel/alerts/outbox.py` | `OutboxWorker`: `lease()`, `complete()`, `run_once()`, `retry_dead()`; `render()` (plain text with incident ID); `DeliveryResult`, `Notifier` protocol |
| `src/sentinel/alerts/telegram.py` | `TelegramNotifier` (`urllib`, JSON `sendMessage`, response contract above), `from_environment()` |
| `src/sentinel/storage/database.py` | `outbox.budget_start_utc`, so an operator retry restores the budget without rewriting `created_utc`. Schema v1 was amended before any database existed outside tests; nothing from it was pushed or deployed. |
| `src/sentinel/config.py`, `config/default.yaml` | Delivery settings in `notifications` (the policy revision covers only `channels` and `min_severity`); `request_timeout_s` at most half of `lease_s` |
| `tests/unit/test_outbox.py` (29) | Telegram: success contract, 502, 503 without body, `ok: false` 400, 401, 403, 429 with and without a valid `retry_after`, non-JSON 200, `ok` without ID, timeouts, reset, refused, a URL containing the token; credentials from the environment, hidden in `repr`. Worker: sent with provider ID; message text; backoff with jitter; `retry_after`; ambiguous timeout; dead letter and operator retry; **crash after send** then restart, lease expiry and an ambiguous retry; lost lease; attempt and age bounds; severity order and starvation; independent channels; outage across a restart; a notifier exception; redaction of stored errors |
| `tests/replay/test_v2_15_delivery.py` (2), `b0_v1_snapshot.py` | Same provider script for v2 and B0 v1 (`V1AlertWorker`, L131-153): an outage across a restart is delivered by v2 and lost by v1; a 401 is dead-lettered by v2 with "needs operator action" and counted as sent by v1 |

**V2-15 mutation sweep (one-off; script not committed): 23/23.** The first round caught 22. The survivor (an operator retry that keeps the exhausted attempt count) was a test gap, fixed before committing. Mutations: `ok: true` not required; HTTP status ignored; `retry_after` ignored; 401 retried forever; timeout not ambiguous; error detail shows the URL; stored errors not redacted; expired leases never retried; live leases stolen; re-lease not ambiguous; a lost lease still writes; no backoff; no jitter; no attempt bound; no age bound; dead rows retried automatically; severity order ignored; starvation ignored; one channel blocks all; notifier exception escapes; operator retry keeps the old budget; message without incident ID; missing `message_id` accepted.

### V2-14 demo form: what was added

| Path | Purpose |
|---|---|
| `src/sentinel/storage/database.py` | `Database.open()`: writer lock file, WAL, `synchronous=FULL`, busy timeout, foreign keys, migrations in `user_version`, `write()`/`read()` transactions; `connect_reader()` read-only. Schema v1: `incidents`, `observations`, `incident_evidence` (payload ≤ 8,192 characters), `incident_transitions`, `outbox`, `delivery_attempts`, with indexes on camera/time, status, correlation and due outbox rows |
| `src/sentinel/incidents/signals.py` | `IncidentSignal` and converters from `ZoneObservation` and `HazardCandidate` (VLM-only: warning, "unconfirmed") |
| `src/sentinel/incidents/service.py` | `IncidentService.record()` (the single transaction), `transition()` (lifecycle with expected revision), `incident()` |
| `src/sentinel/config.py`, `config/default.yaml` | `incidents.merge_window_s`; `notifications.channels` (empty by default) and `min_severity` |
| `tests/unit/test_incidents.py` (16) | Migration and pragmas; single writer and read-only readers; reopen and newer-schema refusal; repeated source event; **crash after COMMIT** (subprocess `os._exit`) then retry; **crash inside the transaction** (subprocess) leaves nothing; failure at the last step rolls everything back; merge window, linking, resolved incidents, reboot; maximum severity and one notification per rise; ENDED on its own incident; channels off by default; policy revision; operator transitions; payload bound |
| `tests/replay/test_v2_14_incident_flow.py` (2) | Zone episodes from a replay (including a tracker ID switch) become one incident and replaying every observation changes nothing; a VLM-only fire candidate opens an unconfirmed warning |

**V2-14 mutation sweep (one-off; script not committed): 20/21, plus one equivalent mutant.** Removing `PRAGMA synchronous = FULL` is not observable here, because this SQLite build (3.45.1) already defaults to FULL. Setting it to NORMAL is caught. Caught: no deduplication; merge window ignored; never merge; joins across boots; resolved incidents joined; latest severity wins; operator revision unchecked; resolved can reopen; ENDED resolves the incident; ENDED goes to the newest incident; joined episodes notify; channels on by default; minimum severity ignored; no writer lock; schema version not recorded; newer schema accepted; failed transaction committed; hazard escalated to critical; no link to the previous incident; no WAL. Not covered: `BEGIN IMMEDIATE` versus a deferred `BEGIN` (no concurrent writer exists to observe it).

### V2-13 demo form: what was added

| Path | Purpose |
|---|---|
| `src/sentinel/rules/geometry.py` | Normalized polygons (3–32 points, no self-intersection, non-zero area), even-odd containment, `bottom_center`/`center` anchors |
| `src/sentinel/rules/schedule.py` | Daily `[start, end)` windows in an IANA timezone (`zoneinfo`, standard library), windows across midnight; fixed offsets and abbreviations such as `IST` rejected |
| `src/sentinel/rules/zones.py` | `ZoneRule`/`ZoneRules` and the `ZoneObservation` contract (ENTERED/ENDED, observation and episode IDs, zone revision, first/last source frames, monotonic duration, reason) |
| `src/sentinel/config.py`, `config/default.yaml`, `cli.py` | `zones:` list (unique IDs; located errors for polygons, schedules, unknown keys); `sentinel config validate` lists each zone |
| `src/sentinel/runtime.py` | `EdgeCore` evaluates zones on every step from current tracks only; `CoreOutput.zones`; `diagnostics()["zone_episodes"]` |
| `tests/replay/test_v2_13_zone_rules.py` (15), `zone_harness.py` | Midnight (one episode across 00:00; ends at 06:00 with "outside the zone's schedule"; persistence counted from 22:00:00), daytime gating, calm entry with scene analysis off, known person (KNOWN `alice`) still observed, stall (ends at track expiry), disconnect (ends: no fresh video), reconnect (new episode), persistence and gap tolerance, anchor, dwell, tentative tracks, several zones and disabled zones, IDs |
| `tests/unit/test_zones.py` (13) | Polygon validation and containment, midnight and DST (New York spring forward, London fall back), input validation, located config errors, revisions, predictions, sparse evaluation, new-epoch reason, wall-clock steps, exact gap boundary, CLI listing |
| `tests/replay/b0_v1_snapshot.py` | `v1_intruder_alert()`: v1's only person alert (L42, L414, L461-462, L587, L616-633) needs an unmatched face, a medium/high VLM threat and restricted hours, with no zones |

**V2-13 regressions recorded (B0 v1 on the same situation):** a calm intruder with no VLM verdict or a "none" threat at 23:00 gives a v2 ENTERED observation, and v1 does not alert. A recognised person in a restricted zone at 23:00 gives a v2 ENTERED observation, and v1 does not alert, even with a "high" threat. v1 has no zones, so where the person stands never matters to it.

**V2-13 mutation sweep (one-off; script not committed): 22/22.** The first round caught 19. Three survivors were test gaps, fixed before committing: a prediction for a never-detected track, a presence split when no evaluation happens during the gap, and the exact gap boundary. Mutations: predictions count; schedule ignored; midnight window as "and"; inclusive window end; fixed UTC offset instead of zone rules; box centre instead of anchor; in-zone gap never restarts; gap never ends a presence; inclusive gap end; vanished tracks keep their episode; repeated ENTERED; persistence ignored; tentative tracks count; ENDED for presences never entered; episode ID without epoch and frame; persistence on wall-clock time; outage reason lost; self-intersection accepted; disabled zones run; core skips zones; schedule read at evaluation time; revision ignores settings.

## Session 3 slice log (2026-09-29; committed 2026-10-03)

Session 3 hit a usage limit before it committed or finished this record. Its results come from that session's transcript and were re-checked against the files where possible.

### What was added

| Path | Purpose |
|---|---|
| `benchmarks/runner/demo_profile.py` | Check 8 (U17 option a, D28). Runs llama-server with v1's flags (D27 preload and full-offload guard), then `demo_workload.py` in `~/onvif_env`. Loads components one after another with settle periods, then a 120 s warm-up and a 600 s steady phase. Samples `/proc/meminfo`, per-process RSS/HWM/PSS and swap counters every 0.2 s, and tegrastats every 1 s. Writes a provisional `profile.json` and `summary.txt` under `~/sentinel-runs/<run id>/` (outside the repository). Refuses to start with a desktop, VS Code, Claude Code, v1 or another llama-server running, or with MemFree below 3.0 GB. Memory is whole-device `MemTotal − MemAvailable` in decimal bytes; per-process and tegrastats views are never added together. System Python, standard library only. |
| `benchmarks/runner/demo_workload.py` | The in-process workload: legacy `yolov8n.engine` through Ultralytics `track()` with ByteTrack and v1's arguments at the source rate (latest-frame semantics); DeepFace Facenet512 + YuNet on the CPU at 2 Hz; one scene request at a time to llama-server every 4 s with v1's image shape and the v2 `SceneReport` prompt, validated by `parse_scene_report`. Reports counts and timings only; model output text is never written. |
| `benchmarks/runner/gpu_alloc_probe.py` (session 4) | Historical Check 9 allocation-until-failure/cap probe; **superseded in session 8** by a bounded 256 MiB API smoke with projected-memory checks and release. The old failure-seeking approach must not be run. |

**Secrets (re-checked in session 4 by reading both scripts in full).** Neither script reads the camera URL, `.env`, tokens or face data. Process command lines are scanned only to detect v1 and are never printed; only command names are recorded. The full environment is passed to child processes, but no environment value is printed or stored. Model output text is not recorded. Errors are recorded as exception class names or HTTP status codes. Run output stays in `~/sentinel-runs/`, outside the repository, with mode 0700.

### Session 3 verification (Claude, on this device, with a desktop running)

- Device checks 1, 3a, 3b, 5 and 6: results in the V2-01 inventory.
- `demo_profile.py` smoke runs, all with `--allow-desktop --allow-dev-tools` and shortened phases, so **none is a profile**. A full orchestration run with a stand-in llama-server exited cleanly and produced its summary. A 20 s workload run on the replay clip (then named `one_person`) held 15 fps on the detector (p95 58 ms); face analysis took about 1 s per run on the CPU, so it reached about 1 Hz rather than the 2 Hz target. The real llama-server could not load alongside it, because MemFree stayed below what it needs (U18). These smoke runs caught and fixed three script bugs.
- No demo run and no profile measurement exist yet.

## Session 2 slice log (2026-09-29)

### What was added

| Path | Purpose |
|---|---|
| `src/sentinel/media/frames.py` | Non-increasing or unset source PTS stamped `source_time_quality=none` (D21) |
| `src/sentinel/replay.py` | Timeline format, loader with line-numbered errors, and `ReplayDriver` (FakeClock + FrameStamper) (D20) |
| `src/sentinel/redaction.py` | Shared redaction of URL userinfo, bot tokens and `token=`/`password=` style values, and bounded single-line messages. The config error messages now use it. |
| `src/sentinel/jobs.py` | ch. 27 `AnalysisJob` (source FrameRef, purpose, incident, deadline from ingest) and `WorkerOutcome` (claimed job ID, bounded untrusted text, redacted detail) |
| `src/sentinel/scene/report.py` | Strict `SceneReport` schema for VLM output and `parse_scene_report()`, whose errors never echo model text |
| `src/sentinel/scene/lane.py` | `SceneLane` scheduler (D15, D16) behind a `SceneAnalyzer` adapter protocol (`submit`/`cancel`/`revision`) |
| `src/sentinel/scene/state.py` | `CurrentScene` (the only path to current scene state) and `route_evidence()` (annotation only to the evidence's own incident) |
| `src/sentinel/media/health.py` | `FreshnessMonitor`: starting / fresh / stale / offline, with `live` only when fresh (D14) |
| `src/sentinel/tracking/tracks.py` | `TrackTable`: current tracks only, 1 s expiry after the last detection, duplicate/out-of-order/not-live frames ignored, 64-track bound |
| `src/sentinel/rules/scene_hazard.py` | `SceneHazardRule` and the `HazardCandidate` contract (D17) |
| `src/sentinel/identity/` | `association.py` (one-to-one ownership), `matching.py` (validated enrollment, cosine, runner-up), `state.py` (derived identity states) (D19) |
| `src/sentinel/live_state.py` | `LiveState` published on every step: video state and age, detector and face capability, occupancy with reason, people with identity, scene status and report |
| `src/sentinel/runtime.py` | `EdgeCore`: portable composition of all of the above; `on_frame`, `tick`, `on_scene_outcome`, `request_enrichment`, `diagnostics()`. Runs with no scene analyzer (scene analysis published as disabled). |
| `src/sentinel/adapters.py` (V2-49, `9593b64`) | ch. 27 `AdapterManifest` (strict config section), static `BUILTIN_ADAPTERS` registry (empty), `resolve()` (no imports), `load()` (import failures leave only that adapter unavailable), `evidence_from_adapter()` / `output_problem()` (strict parse; kind, producer and revision checked against the manifest; enabled adapters only) (D25) |
| `src/sentinel/cli.py`, `config.py` (V2-49) | `adapters:` list in config (duplicate IDs and unknown contract versions are config errors; typos inside list items get a suggestion); `sentinel config validate` prints each adapter's state and reason |
| `config/default.yaml` | New `scene`, `hazard` and `identity` sections, with values labelled as proposed or placeholders |
| `tests/replay/` | `test_timeline.py`, `test_r3_stale_scene_results.py` (11), `test_r2_zero_person_scenes.py` (5), `test_r1_face_identity.py` (9); harnesses, builder, fixture and the B0 v1 snapshot |
| `tests/unit/` | New `test_jobs.py`, `test_scene_report.py`, `test_health.py`, `test_scene_hazard.py`, `test_identity.py`, `test_adapters.py` (8); config tests extended; `test_portable_imports.py` adds an ML-blocked adapter resolve/load check |

### Regressions recorded (V2-03 acceptance)

| Regression | Guide/audit source | v2 result | B0 v1 snapshot on the same input |
|---|---|---|---|
| **R3** late scene answer for incident A (after deadline and TTL), with B opened meanwhile | Audit 3; ch. 2 row 4; ch. 21 | Timeout evidence at the deadline; the late fire report is EXPIRED, annotates only A and never becomes current | Fire alert and "high" threat from the stale answer |
| R3 answer after the deadline but within TTL | D15 | Late: expired, annotates only its incident | Alerts |
| R3 answer after a reconnect | Audit 3 | SUPERSEDED_EPOCH, not current | Alerts |
| R3 malformed JSON; the string `"true"` for a boolean | ch. 13, 21 | `error` evidence replaces the previous verdict | Malformed → "none" verdict; `"true"` → fire alert |
| R3 wrong job ID; worker timeout/error; submit failure | ch. 21, 27 | `error`/`timeout` evidence; credentials redacted; slot freed | Keeps the earlier verdict (fire) |
| R3 enrichment replaced/skipped; older-frame enrichment; outage | ch. 12, 27 | `scene.job_skipped` evidence for its own incident; an older frame never overwrites newer state; an outage clears the scene and it never revives | — |
| **R2** three minutes with nobody present | ch. 2 row 1; ch. 21 | EMPTY with fresh video on every step; the departed track goes 1 s after its last detection; a scene check about every 4 s throughout | Dashboard keeps showing 1 person; no scene checks while empty |
| R2 stall → stale at 2 s → offline at 10 s → reconnect, with a frozen reader | ch. 2 row 3; ch. 22 | Exact thresholds; occupancy UNKNOWN from 1 s without detector frames; frozen frames never reprocessed; pre-stall scene evidence never current again | Keeps "seeing" the person through stall and outage |
| R2 detector unavailable | ch. 6 | Occupancy UNKNOWN "person detector unavailable"; scene checks continue | — |
| R2 smoke with nobody present; positive → stall → positive; reconnect | Audit 4; ch. 1, 9 | One WARNING candidate per episode with `people_count=0` and no primary signal; no confirmation across a stall, gap or epoch | — |
| **R1** one face inside two overlapping people | Audit 1; ch. 21 | Both UNRESOLVED ("face ownership ambiguous") | Both named Alice |
| R1 known + faceless person, only one face visible | ch. 2 row 2; ch. 21 | Alice KNOWN after 2 matches; other UNRESOLVED ("no face visible") | Both named Alice (full-frame fallback) |
| R1 separation; reconnect; vote expiry; near-twin match | ch. 11 | Face attaches to exactly one person; a new epoch starts UNRESOLVED; KNOWN lapses after 30 s without faces; ambiguous match gives no vote | — |
| R1 empty enrollment; back-facing person | Audit 2 | UNRESOLVED, never a stranger | Every face and every faceless track become "Stranger" |
| R1 unknown / contradiction | ch. 11 | Low quality: no vote; consistent non-matches: UNKNOWN; contradiction: UNCERTAIN, re-derived by new consistent evidence | — |

### Session 2 verification: exact commands and results

All ran on 2026-09-29 on the Jetson from `~/sentinel-surveillance`. Nothing was installed into or written to `~/onvif_env`.

```bash
.venv/bin/python -m pytest                                  # 114 passed in 2.20s (Python 3.10.14, pydantic 2.13.5)
.venv/bin/sentinel config validate config/default.yaml      # valid, exit 0
```

**Every session 2 commit passes its own tests** (the loop from session 1, over `6578ded..HEAD`): `4587021` 48 · `812b42c` 48 · `e81db81` 60 · `86889f8` 87 · `432dc69` 100 · `66937ef` 114 passed. `48d6188` is documentation only.

**V2-49 (`9593b64`):** 123 passed on Python 3.10.14 with pydantic 2.13.5 in the working tree before committing, and 123 with pydantic 2.12.5 in the scratch venv described below. After committing, a clean archive of HEAD on Python 3.12.3 gave 123 passed, `sentinel config validate` valid ("no optional adapters configured: core monitoring only"), and the v1 dashboard tests OK.

**Clean archive of HEAD on Python 3.12.3** (fresh venv, as in session 1): 114 passed; `sentinel config validate` valid. **v1 "Dashboard checks"** in the same archive (Flask 3.0.3, requests 2.32.3, python-dotenv 1.0.1): `Ran 4 tests ... OK`.

**`~/onvif_env` compatibility.** That environment has pydantic 2.12.5 and no pytest, and must not be changed. So:

```bash
# scratch venv matching its interpreter and pydantic version
/usr/local/bin/python3.10 -m venv "$SCRATCH/venv-pyd2125"
"$SCRATCH/venv-pyd2125/bin/python" -m pip install pydantic==2.12.5 PyYAML==6.0.3 pytest==9.1.1
PYTHONPATH=src "$SCRATCH/venv-pyd2125/bin/python" -m pytest -p no:cacheprovider   # 114 passed
# smoke run inside ~/onvif_env itself: imports all 24 sentinel modules, loads the
# default config, drives EdgeCore with FakeClock through one frame and a 2 s stall
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/onvif_env/bin/python <script>           # "fresh empty" then "stale"
```

**Mutation sweeps (one-off; scripts not committed).** Each rule was broken in turn in a scratch copy of `src/` on `PYTHONPATH`. The suite failed every time, each through the test aimed at that rule:

- **R3, 15/15:** late result keeps full TTL; non-current evidence becomes scene state; held evidence never dropped; no timeout at the deadline; claimed job ID trusted; skipped jobs become scene state; older frame overwrites newer state; late evidence annotates nothing; failure keeps previous verdict; periodic checks need people; waiting enrichment never skipped; replaced enrichment silently dropped; submit exception uncaught; report accepts extra fields; worker detail unredacted.
- **R2, 15/15:** stale stream still live; stale boundary off by one; offline never reached; reconnect without a frame counted fresh; duplicate frames reprocessed; tracks never expire; empty scene not published; detector recency ignored; detector status ignored; scene lane only with people; hazard not reset on a missing view; hazard ignores epoch change; hazard critical; hazard emits every report; hazard not reset by failure.
- **V2-49, 16/16:** unknown contract version accepted; duplicate IDs accepted; resolve imports the module; disabled adapters enabled; implementation version ignored; declared kinds unchecked; unknown profile admitted; notifiers need a profile; load failure propagates; disabled adapter's output accepted; undeclared output kind accepted; foreign producer accepted; revision mismatch accepted; evidence contract version 2 accepted; core requires a scene analyzer; list-item suggestions lost. The R1 and R2 sweeps were re-run after the `runtime.py` change: 11/11 and 15/15.
- **R1, 11/11:** first candidate wins; person-side uniqueness only; head region ignored; inside fraction ignored; empty-enrollment reason; single vote establishes KNOWN; low-quality faces vote; contradiction ignored; votes never expire; identity bookkeeping kept after tracks end; runner-up margin ignored.

Four first-round misses were test gaps and were fixed before committing: the gap-and-reconnect hazard test confirmed before the stall; the inside-fraction face had its centre outside the box; there were no near-twin or faceless empty-enrollment cases; and identity bookkeeping was not observable (now `EdgeCore.diagnostics()`). One further miss is deliberate: redaction happens both in `WorkerOutcome` and in the lane, so removing one layer is not observable end to end. `test_jobs.py` now checks the first layer directly.

## Earlier slice log

### Session 1 (C1 foundation): what was added

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package `sentinel-surveillance` (import `sentinel`) with a `src/` layout. Pinned portable dependencies pydantic 2.13.5 and PyYAML 6.0.3, plus a `dev` extra (pytest 9.1.1). Provides the `sentinel` script and pytest settings (`tests/unit` only, importlib mode, warnings are errors). |
| `src/sentinel/media/clock.py` | Injectable `Clock`, `SystemClock`, `FakeClock` (advance / UTC step / reboot), boot-scoped `MonoInstant`, `read_boot_id()` |
| `src/sentinel/media/frames.py` | `FrameStamper` mints identity (random run ID per stamper, epoch counter per connect, frame counter per epoch); `frame_novelty()` classifies each frame for a consumer as new, new epoch, duplicate, out-of-order or not live |
| `src/sentinel/contracts.py` | `FrameRef`, `FrameKey`, `StreamIdentity`, `ResizeTransform`, `TrackObservation`/`TrackKey`, versioned `Evidence`, `Applicability`, `stream_relation()` |
| `src/sentinel/config.py` | Strict `SentinelConfig`, `load_config()`, and `ConfigError` with one located line per problem. The YAML loader rejects duplicate keys. |
| `src/sentinel/cli.py` | `sentinel config validate <path>` only (guide ch. 18); exit 0 valid, 1 invalid, 2 usage |
| `config/default.yaml` | Guide ch. 6 starting targets: stale 2 s, offline 10 s, track expiry 1 s |
| `tests/unit/` | 47 behavioral tests (mapped below) |
| `.github/workflows/v2-portable.yml` | Separate portable job on Python 3.10 and 3.12; `ci.yml` untouched |

Semantics now enforced (guide ch. 6, audit finding 3, D3):

- **Frame identity** is `(camera_id, boot_id, run_id, stream_epoch, frame_seq)`. Every part is assigned at ingest; none comes from a clock or from source PTS. Clock steps, PTS resets and runtime restarts therefore cannot recycle an identity.
- **Epochs:** each `connect()` increments `stream_epoch` within its run. Frames and evidence from an earlier epoch are `SUPERSEDED_EPOCH`; from another run of the same boot, `OTHER_RUN`; from another boot, `OTHER_BOOT`. Tracker IDs are scoped to their epoch (`TrackKey`).
- **Boots:** monotonic readings carry `boot_id`, and arithmetic across boots raises.
- **Source-age expiry:** evidence ages from its source frame's ingest time. `Evidence.observed_on(frame, …)` takes no completion time, so a late result arrives already aged. The validity window is `[ingest, ingest + ttl)`. Tracks expire one `track_expiry` after their last real detection; labelled predictions do not refresh them.
- **Current decisions:** `applicability(live_stream, now)` returns `CURRENT` only for the live epoch within TTL. Otherwise it reports why: `NO_LIVE_STREAM` during an outage, `SUPERSEDED_EPOCH`, `OTHER_RUN`, `OTHER_BOOT`, `OTHER_CAMERA`, `EXPIRED`, or `FUTURE` (clock or replay misuse).
- **History:** `may_annotate(incident_id)` lets late evidence enrich only the incident it was requested for.
- **Failures:** evidence with status `unknown`, `unavailable`, `timeout` or `error` carries no value or confidence, so an old verdict cannot stand in. `observed` requires a value. A confidence must state its kind; a similarity in [−1, 1] is not a probability.
- **Wall clock:** UTC is display and audit only. Naive or non-UTC datetimes (e.g. +05:30) are rejected, and UTC steps never change ages.
- **Config:** strict types (a YAML `on` or the string `"2"` is rejected), unknown keys with a "did you mean" suggestion, finite positive durations, offline > stale, duplicate YAML keys reported with line and column, and a clear unsupported-version message. URL credentials are redacted from messages, and every problem is reported in one run.

| Acceptance property | Tests |
|---|---|
| Invalid config gives clear errors | `test_config.py` (12) |
| Frame identity | `test_frame_identity.py`: PTS reset, frozen buffer, out-of-order, transform round trip and fit, unambiguous times |
| Boot/epoch isolation | `test_frame_identity.py`: reconnect increments the epoch; a restart in the same boot with identical clock readings, restarted counters and a backwards wall-clock step still yields a new identity; reboot. `test_evidence_applicability.py`: outage and reconnect, previous boot. `test_track_observations.py`: epoch-scoped track IDs. |
| Source-age expiry | `test_evidence_applicability.py` (result completing after its TTL, exact TTL boundary, late annotation, future evidence); `test_track_observations.py` (predictions don't keep a track alive) |
| Wall-clock changes | `test_clock.py` (UTC steps leave monotonic time alone); `test_evidence_applicability.py` (steps neither extend nor shorten validity); `test_frame_identity.py` (backwards step during restart) |
| Portable without Jetson/camera | `test_portable_imports.py` imports every module file of the package with cv2, gi, tensorrt, pycuda, cuda, torch, tensorflow, ultralytics, deepface, onnxruntime and jtop *blocked*. It fails if any module is skipped. |

### Session 1: verification commands and results

All commands ran on 2026-09-29 on the Jetson (aarch64), from `~/sentinel-surveillance` unless noted. `~/onvif_env` was not used or modified in session 1.

**Setup (V2-04).** `.venv/` was already git-ignored.

```bash
/usr/local/bin/python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
```

Resulting `pip freeze --exclude-editable`: annotated-types 0.8.0, exceptiongroup 1.3.1, iniconfig 2.3.0, packaging 26.3, pluggy 1.6.0, pydantic 2.13.5, pydantic_core 2.46.5, Pygments 2.21.0, pytest 9.1.1, PyYAML 6.0.3, tomli 2.4.1, typing-inspection 0.4.4, typing_extensions 4.16.0.

**Each increment on its own:**

```bash
.venv/bin/python -m pytest tests/unit/test_portable_imports.py       # 1 passed
.venv/bin/python -m pytest tests/unit/test_clock.py                  # 6 passed
.venv/bin/python -m pytest tests/unit/test_frame_identity.py         # 10 passed
.venv/bin/python -m pytest tests/unit/test_evidence_applicability.py # 16 passed
.venv/bin/python -m pytest tests/unit/test_track_observations.py     # 2 passed
.venv/bin/python -m pytest tests/unit/test_config.py                 # 12 passed
.venv/bin/python -m pytest                                           # 47 passed in 0.56s (Python 3.10.14)
.venv/bin/sentinel config validate config/default.yaml
# config/default.yaml: valid Sentinel configuration (version 1, camera cam-1)   exit 0
```

A manual invalid-config check: `sentinel config validate bad.yaml` on a file with version 2, an RTSP URL as camera ID, a misspelled key, offline < stale and `.nan` exited 1 with:

```text
bad.yaml: invalid configuration (5 problems)
  - config_version: unsupported version 2; this build reads version 1
  - camera.id: String should match pattern '^[A-Za-z0-9][A-Za-z0-9._:-]*$' (got 'rtsp://<redacted>@192.0.2.10/live')
  - freshness.offline_after_s: must be greater than stale_after_s (1.0 <= 2.0)
  - freshness.track_expiry_s: Input should be a finite number (got nan)
  - freshness.stale_afer_s: unknown setting (did you mean 'stale_after_s'?)
```

**Every commit passes its own tests.** Each commit's tree was extracted and tested with the dev venv. `PYTHONPATH` takes precedence over the editable install's `.pth` entry, so the extracted package is the one tested.

```bash
for sha in $(git rev-list --reverse master..v2-beta); do
  d=$(mktemp -d); git archive "$sha" | tar -x -C "$d"
  [ -f "$d/pyproject.toml" ] && (cd "$d" && PYTHONPATH="$d/src" ~/sentinel-surveillance/.venv/bin/python -m pytest -q -p no:cacheprovider | tail -1)
done
```

| Commit | Result |
|---|---|
| `ec6698d` CLAUDE.md | documentation only |
| `b52920f` skeleton | 1 passed |
| `e9f959d` clock | 7 passed |
| `bc42248` frame identity | 17 passed |
| `3b47f5f` evidence/track | 35 passed |
| `ee50275` config and CLI | 47 passed; `python -m sentinel.cli config validate config/default.yaml` valid |
| `6a9e71d` CI workflow | 47 passed; CLI valid |

The branch was rebuilt once before this record, while still unpushed. In the first build, the import test's sanity check named a module that only exists from the frame-identity commit, so the skeleton and clock commits failed on their own. The test now derives the expected modules from the package files.

**Clean archive of HEAD on Python 3.12.3** (fresh venv, no local-only files), run in a session scratch directory; `mktemp -d` is equivalent:

```bash
C=$(mktemp -d); git archive HEAD | tar -x -C "$C"; cd "$C"
/usr/bin/python3.12 -m venv .venv312 && .venv312/bin/python -m pip install -e '.[dev]'
.venv312/bin/python -m pytest -p no:cacheprovider                    # 47 passed in 0.67s
.venv312/bin/sentinel config validate config/default.yaml            # valid, exit 0
```

**The existing v1 "Dashboard checks" job is unaffected.** This was run in the same archive, so `load_dotenv()` in `dashboard.py` could not pick up a `.env` from a parent directory of the checkout.

```bash
/usr/bin/python3.12 -m venv .venv-v1ci
.venv-v1ci/bin/python -m pip install Flask==3.0.3 requests==2.32.3 python-dotenv==1.0.1
.venv-v1ci/bin/python -m unittest discover -s tests                  # Ran 4 tests ... OK
```

unittest discovery ignores `tests/unit/` because it has no `__init__.py`. This also establishes that the four v1 dashboard tests pass at `2b2d639`, which the audit had not rerun. The workflow itself uses Python 3.11; this run used 3.12.3.

**Mutation sweep (one-off; the script was not committed).** Each rule below was broken in turn in a scratch copy of `src/`, with that copy on `PYTHONPATH`. The suite failed every time, **16/16**, each through the test aimed at that rule: TTL boundary made inclusive; superseded epochs ignored; boot mismatch ignored; run ID ignored; one run ID shared by all stampers; epoch not incremented on connect; future evidence accepted; predictions refresh track age; failures may carry verdicts; `frame_seq` not advanced; duplicates treated as new; non-UTC offsets accepted; UTC steps move monotonic time; duplicate YAML keys allowed; credentials echoed; offline ≤ stale allowed.

## Not run or not established (sessions 1–6)

- GitHub Actions results are reported only through `32985c2` (maintainer report: both workflows passed). The local origin reference now includes sessions 3/4 at `d85eb1e`; no fresh remote or CI state was queried, so later CI results are unverified.
- There was no run on a clean laptop (x86-64 or macOS); all checks ran on this Jetson's aarch64 userspace.
- No lint or type check is configured yet (guide ch. 21 lists both); deferred to keep the slice small.
- Session 3 device checks and the maintainer's session 5 check 8 exist as separately attributed observations/measurements below. Session 6 adds portable instrumentation/tests only and establishes no hardware, Gate B or beta-readiness result.
- No v2 production camera/GPU adapter or `sentinel run` exists yet. Demo-form adapters exist for capture (session 12, run on the camera in session 13) and the legacy detector + ByteTrack (session 13, not yet run on the device); the GPU benchmark scripts are separate. `EdgeCore`, the incident store and the outbox are exercised by synthetic replays, mocks and FakeClock smoke runs (CPU). No real Telegram message has been sent by v2. The other ch. 18 CLI commands, including `sentinel replay`, were intentionally not added yet.
- The replay regressions use synthetic timelines, not the maintainer's clips. They record the behaviour of the v2 components and of a documented reference model of v1 (D12), not of the running v1 process.
- The mutation sweeps are one-off checks whose scripts are not committed.
- Checks 3a/3b were run by Claude in session 3 (inventory below). Check 8 was run by the maintainer (session 5 log): one run on an unlabelled clip, a provisional profile and not a benchmark; its steady phase was still ramping (U21). Session 8 portably tests the replacement bounded Check 9 orchestration. The maintainer's bounded Check 9 smoke and short U21 pilot (2026-10-03, previous boot) are USER-SUPPLIED MEASUREMENTS in the session 10 log; neither settles beyond-MemFree/U18 or long-run memory. S1 (session 11) is implemented portably. Session 12's attempt never reached an arm; the maintainer's S1 runs at `e9af7f4` are USER-SUPPLIED MEASUREMENTS in the session 13 log (scene-only, 180 s steady, noise images; not the full workload or a long run). Claude's session 5 diagnostics (detector, face and VLM reproductions) are observations with dev tools running, not measurements.

## Decisions

- **D1.** The branch is `v2-beta`. Guide ch. 3 suggested `codex/v2-beta`; a tool-neutral name was used.
- **D2.** Typed config and contracts use pydantic v2 in strict, frozen, extra-forbidding mode; the planned FastAPI API (ch. 15) uses the same library. PyYAML is loaded through a SafeLoader subclass. Direct dependencies are pinned exactly; there is no hashed lockfile yet.
- **D3. Run-scoped stream epochs** (maintainer decision, 2026-09-29).
  - Epochs are connection-based: each `connect()` increments `stream_epoch` within an ingest run, as guide ch. 6 specifies.
  - Identity also carries `run_id`, a random UUID4 per `FrameStamper`. The runtime keeps one stamper per camera, so in practice this is one per process; per stamper is the stricter form of the suggested per-process ID.
  - No part of identity comes from a clock. Wall-clock steps, repeated monotonic readings and runtime restarts in the same boot therefore cannot recycle `(boot, run, epoch, seq)`.
  - Epochs are ordered only within a run; another run of the same boot is `OTHER_RUN`, never current.
  - This adds `run_id` to the guide's FrameRef fields, and to `StreamIdentity`, `FrameKey` and `TrackKey`. It replaces an earlier, uncommitted proposal that derived epochs from monotonic time.
- **D4.** The boot ID comes from `/proc/sys/kernel/random/boot_id`. Hosts without it (macOS/Windows development) use one random ID per process, which is stricter.
- **D5.** Monotonic time is `time.monotonic_ns()`, which is CLOCK_MONOTONIC on Linux and excludes suspend. Revisit only if the device ever suspends.
- **D6.** Validity windows are half-open: `CURRENT` while age < TTL, `EXPIRED` at age = TTL.
- **D7.** `FUTURE` (evidence newer than `now`, or from an epoch newer than the live one) means "not current"; it does not raise an exception.
- **D8.** Config lives at `config/default.yaml` per guide ch. 5. `~/onvif_env`'s `nano_surveillance/config.py` was not reused: its `Section(**raw)` unpacking surfaces unknown keys as Python `TypeError`s and it does no type or range checks. Its idea of one YAML section per concern was kept.
- **D9.** `pyproject.toml` is the portable profile only; Jetson/ML packages will get a separate, device-verified profile. v1's `requirements.txt` is unchanged. Maintainer decision (2026-09-29): **do not change runtime dependencies**; the run ID uses the standard library.
- **D10.** The dev interpreter is `/usr/local/bin/python3.10` (3.10.14, the family `~/onvif_env` uses). CI adds 3.12, the device's system Python.
- **D11.** FrameRef's optional buffer reference is deferred to V2-05/V2-09; adding it is backward compatible.
- **D12. B0 reference model** (maintainer decision; resolves U9). Replay regressions show v1's failing behaviour with a small documented reference model in tests. It extracts only the relevant v1 logic, cites `surveillance4_1.py` at `2b2d639` with line ranges, and is labelled a v1 behaviour snapshot, not production code.
- **D13. Planning documents** (maintainer decision). CLAUDE.md is committed; the v2 guide and audit review stay local-only through `.git/info/exclude`.
D14–D21 are implementation decisions made in session 2 within the guide's rules; the maintainer has not reviewed them yet.

- **D14. Liveness (resolves U1).** Only a FRESH stream is live for decisions: `FreshnessMonitor.assess()` returns the connected stream as `live` only while its newest frame is younger than `stale_after_s`. Stale, offline, disconnected, or reconnected with no frame yet all give `live=None`, so current scene, track and hazard state are withdrawn during a stall. Held scene evidence that is found non-current is dropped and never revives, even if the same epoch resumes within its TTL.
- **D15. Late and skipped scene jobs.** A job's deadline is `source ingest + job_timeout_s`, which includes queue wait and source age. An outcome after the deadline is *late*. Its evidence is valid only up to the deadline, so it is already expired and can only annotate its own incident. A job that never ran (deadline passed while waiting, or replaced by a newer enrichment request) gives `unavailable` evidence of kind `scene.job_skipped`, which is never scene state. Evidence IDs are `<job>.<status>[.late]`.
- **D16. Scene lane defaults (proposed, to measure in V2-26):** periodic interval 4 s (v1's value, measured start-to-start), job timeout 8 s, report TTL 10 s. Config rejects a TTL below the timeout. One job in flight, one replaceable enrichment request, 8 timed-out jobs remembered for late results.
- **D17. Fire/smoke from the VLM alone** is a `scene.fire_smoke_candidate` capped at `warning` (`primary_signal=false`). It needs 2 distinct consecutive positive reports from one epoch within 12 s, fires once per episode, and resets on any negative, failed or missing report, stall, gap or epoch change. A critical fire alert needs a separately evaluated primary fire signal, which does not exist.
- **D18. Occupancy.** EMPTY only when video is fresh, the detector is available and it processed a frame within `track_expiry_s`. Otherwise UNKNOWN with a reason ("no fresh video (stale)", "person detector unavailable", "detector has not processed a recent frame"). This closes the window in which expired tracks on a stalled but not yet stale stream would read as an empty room.
- **D19. Identity.** Association accepts a face–person pair only when each is the other's sole candidate: face centre in the top 40 % of the person box, at least 60 % of the face inside it. Identity is derived on every read from the track's own votes (last 5, TTL 30 s): KNOWN or UNKNOWN after 2 consistent votes, UNCERTAIN on contradiction, else UNRESOLVED. Cosine threshold 0.5 and margin 0.05 are **placeholders until V2-25 calibration**. There is no "stranger" state. An empty enrollment is UNRESOLVED ("no identities enrolled"). Identity is context only; no rule reads it yet.
- **D20. Replay format.** JSON-lines timelines (`src/sentinel/replay.py`) with a header, `connect`/`disconnect`, `frame` runs (people, synthetic faces and embeddings, optional PTS), `tick`, `result` and `incident` events. Fixtures live in `tests/replay/fixtures/`; long variants are built inline with `tests/replay/timeline_builder.py` on a 66 ms grid. The B0 v1 snapshot (D12) is `tests/replay/b0_v1_snapshot.py`, imported through pytest's `pythonpath`. `tests/replay/` has no `__init__.py`, so the v1 unittest job ignores it.
- **D21. Source PTS.** A PTS that does not increase over the previous frame's in the epoch is kept but stamped `source_time_quality=none` (see V2-01 inventory).
- **D22. Perception profile for the demo** (maintainer decision, 2026-09-29; resolves U12 for the demo). Profile A: substream only, 640×480 at 15 fps, for the Oct 20 demo. The 20–25 fps target, the restatement of the ≥15 fps gate and the beta profile (A, B or D) stay open until after the demo.
- **D23. Demo deviations accepted** (maintainer decision, 2026-09-29), on condition that every affected package is marked "demo form, full acceptance pending" in this file and is not counted done. See the package table and its counting rule.
- **D24. Demo runtime environment and decode** (maintainer decision, 2026-09-29; resolves U14 for the demo). The demo runs in `~/onvif_env` with `PYTHONPATH=src` and decodes with its OpenCV/FFmpeg in software (640×480 at 15 fps is about 4.6 Mpx/s, cheap on CPU). GStreamer/NVDEC waits for V2-05 proper; revisit after Oct 20.
- **D25. Adapter manifests (V2-49, session 2 implementation decision; not yet reviewed).** Configuration names adapter IDs only; implementations come from a static registry in code. Unknown contract versions and duplicate IDs are configuration errors. Other failures leave only that adapter unavailable, with a reason shown by `sentinel config validate`. Model adapters (scene, detector, face) need a measured resource profile; notifiers do not. No profiles exist yet (U17).
- **D26. Post-demo stream path** (maintainer decision, 2026-09-29; narrows U12). After Oct 20, perception moves to the main stream (`subtype=0`, H.265 Main 2304×1296 at 20 fps, maintainer measurement) with NVDEC hardware decode, in V2-05 proper. The demo stays on profile A (D22). The restated frame-rate gate and the decision on a second (substream) session remain open.
- **D27. CUDA driver library (resolves U13)** (maintainer decision, 2026-09-29). Option 1, "preload now, restore later":
  - No system change before Oct 20. The demo's GPU processes (the Python runtime and llama-server) run with `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`.
  - A guard refuses GPU adapters unless the mapped `libcuda` is L4T's, `cuInit` returns 0 and llama-server logs a full offload (every layer and the vision encoder on CUDA0). There is no silent CPU fallback. `benchmarks/runner/` already applies this guard; `sentinel run` (D-1) must too.
  - **After Oct 20**, behind a V2-08 restore point (known-good backup first): restore the Jetson `nvidia-cuda-dev` (6.2.1+b38 from the r36.4 repository) and remove or pin Ubuntu's CUDA 12.0 packages (`nvidia-cuda-toolkit`, `libcudart12`, `libnvidia-ml-dev`, `libnvidia-compute-535`), then re-run checks 3a/3b without the preload. Until then, the preload is required, not optional.
- **D28. Admitting demo adapters (resolves U17)** (maintainer decision, 2026-09-29). Option (a): one measured run of the three demo model components together (check 8) gives provisional profiles labelled `provisional-demo`. They are recorded in this file and the adapter registry's known profiles, admit the adapters for the demo only, and are not a benchmark, Gate B record or beta-gate result.
- **D29. Headless demo host** (maintainer decision, 2026-09-29). The demo runs with the display manager and any remote-desktop session stopped, as v1's launcher does. Check 8 is measured the same way and refuses to start otherwise (or records `--allow-desktop`).
- **D30. Zone rules (V2-13 demo form; session 4 implementation decision, not yet reviewed).**
  - A track is in a zone on a frame when it was really detected on that frame (predictions never count), it is CONFIRMED, the zone's anchor point of its box (default bottom-centre) lies in the polygon, and the frame's UTC ingest time is inside the schedule.
  - One continuous presence of one track in one zone is an episode. It becomes an ENTERED observation once in-zone detections span `min_duration_s` (entry persistence for `restricted`, the dwell threshold for `dwell`), at most once per episode. It ENDS (only if it was ENTERED) when the track has not been in the zone for longer than `gap_tolerance_s`, the track expires, the video is not fresh, or the stream reconnects. A new epoch always starts a new episode.
  - Durations and gaps use source ingest times on the monotonic clock (half-open: a gap of exactly `gap_tolerance_s` continues the presence). **Schedules are the one wall-clock input**: membership is decided from the frame's UTC ingest time in the zone's IANA timezone, so daylight-saving changes follow the zone rules and a wall-clock step can move a frame in or out of a window. Windows are daily `[start, end)`; an end not after the start spans midnight.
  - Rules read neither identity nor scene verdicts. Observations carry no identity; identity can only be added as context by the incident service. Observation IDs are `<episode>.<phase>`, with the episode ID hashed from zone, zone revision, frame identity and track ID, for V2-14 deduplication.
  - Starting values (proposed; calibrate on labelled replays in V2-07): `min_duration_s` 1.0, `gap_tolerance_s` 1.0, severity `warning`. No hysteresis margin yet: the gap tolerance absorbs anchor jitter at the boundary. Up to 16 zones of up to 32 points each.
- **D31. Incidents and outbox (V2-14 demo form; session 4 implementation decision, not yet reviewed).**
  - One SQLite file per device, written only by the runtime process. The writer takes an exclusive lock file; one connection is shared by its threads under a lock, so incident recording and the outbox worker never write concurrently, and no transaction spans network I/O. WAL, `synchronous=FULL`, busy timeout 5 s, foreign keys on. Migrations are numbered scripts recorded in `PRAGMA user_version`; a newer schema is refused. Readers use read-only connections.
  - `record()` is the guide's single transaction: deduplicate `observation_id`, create or join the incident (optimistic revision check), append evidence and the opening transition, insert outbox rows `UNIQUE(incident, channel, policy_revision, message_kind)`. The runtime acknowledges an observation only after `record()` returns.
  - **Correlation:** an ENTERED observation joins the latest unresolved incident with the same camera, rule kind and zone if that incident's latest observation is from the same boot and at most `incidents.merge_window_s` (proposed 120 s) earlier on the monotonic clock. Otherwise it opens a new incident, linked to the previous one with that key. Across a reboot it never joins. ENDED is evidence on the incident its episode joined and never changes status.
  - **Severity** is the maximum seen, never a sum. A rise is notified once per level (`escalated-<severity>`). Rule observations open incidents directly (status `open`); the VLM-only hazard opens at most a `warning`, titled unconfirmed.
  - **Notifications are off by default** (`notifications.channels: []`), with `min_severity: warning`. The policy revision is a hash of that section, so a changed policy may notify an incident again.
- **D32. Delivery (V2-15 demo form; session 4 implementation decision, not yet reviewed).**
  - Delivery is **at least once**. A worker leases due rows in one transaction, sends outside it, and records the outcome only while it still holds the lease; a late outcome is logged as `late:` and changes nothing. When a lease expires (the worker crashed or stalled), the next lease marks the row **ambiguous** and the abandoned attempt `abandoned`. Messages carry the incident ID and say that a repeat is the same incident.
  - **Telegram success** is HTTP 200 and `ok: true` with an integer `message_id`. 429 waits at least `retry_after`. 400, 401, 403 and 404 are permanent: the row goes dead with "needs operator action". 5xx and network failures are retried. Timeouts, resets after sending, an unreadable 200 body and `ok: true` without a message ID are retried as ambiguous.
  - Backoff is `min(backoff_max_s, backoff_base_s · 2^(attempts−1))` with jitter in [½, 1) of that, and never shorter than `retry_after`. Rows go dead after `max_attempts` (8) or `max_age_s` (6 h) from their budget start; an operator retry restores the budget. Rows are ordered by severity, but a row due for longer than `starvation_s` (300 s) goes first. Each channel is independent, and rows for a channel without a configured notifier stay untouched.
  - Lease and retry times are UTC, because they must survive a restart. A wall-clock step can only cause an early or late retry.
  - **Credentials:** `SENTINEL_TELEGRAM_BOT_TOKEN` and `SENTINEL_TELEGRAM_CHAT_ID` come from the runtime's environment, never the config file. Without them the channel is unavailable and its rows wait. Errors are built from status codes, Telegram's redacted `description` and exception class names, never from URLs or raw exception text, and are redacted again before storage. Messages are plain text: no images, footage or identity data.

- **D33. Check 8 admitted as the provisional demo profile (settles U17)** (maintainer decision, 2026-10-03). Run `demo-profile-20261003T085010Z` is the `provisional-demo` profile (D28) for the scene, detector and face demo adapters, **with its exceedance recorded; it is not a gate.** Recorded exceedance: steady median 5.036 GB and p95 5.347 GB are above the 5.0 GB target (by 36 MB and 347 MB). The run peak, 5.350 GB, is 50 MB under the 5.4 GB ceiling. Claude's note (U21): the steady phase was still ramping, so the margin to the ceiling is not established. The cheapest reductions, by expected effect for the cost (none measured):
  1. **Attribute and stop the steady ramp (U21):** up to about 1.1 GB between the settled loads (3.88 GB) and the end of the steady phase. It needs one instrumented check 8 re-run (torch allocator statistics in the workload, plus more `/proc/meminfo` fields). If torch's caching allocator is the cause, a cap such as `torch.cuda.set_per_process_memory_fraction` needs no new dependency.
  2. **llama-server context and batch (flags only):** `--ctx-size` 2048 → 1024 (a request uses at most 505 tokens: 305 prompt + 200 completion) and `--batch-size`/`--ubatch-size` 512 → 256. These shrink the 24 MiB KV and 136 MiB compute buffers. Estimate: tens of MB, at most about 0.1 GB. The full-offload check and latency must be repeated.
  3. **Detector without torch (V2-09):** the detector's +0.79 GB cold load is mostly torch's CUDA context and Ultralytics, for a 14.5 MB engine that needs 18.9 MB of device memory. A TensorRT-only adapter could remove much of it (unmeasured). Cost: V2-09 (2.5 h Claude, 3 h maintainer) plus a CUDA memory API without torch: `ctypes` to `libcudart`, or a new dependency (a decision).
  4. **Face without TensorFlow (V2-25, later):** +0.45 GB. ONNX Runtime is already in `~/onvif_env`; this needs a converted model and a parity check.
- **D34. Demo face cadence 1 Hz** (maintainer decision, 2026-10-03). Check 8 measured the CPU face stage at p50 896 ms and p95 922 ms per run. At 2 Hz configured it reached 1.12 Hz, meaning it ran back to back and the configured rate was never met. 1 Hz is achievable with about 78 ms margin at p95; with two consistent votes (D19), an identity can settle about 2 s after a face is visible. The face thread is still busy about 90 % of the time at 1 Hz, so its CPU share drops only a little from check 8's. `demo_profile.py` and `demo_workload.py` now default to 1 Hz, and the D-1 face adapter uses 1 Hz.

- **D35. Enrichment annotations (V2-28 demo form; session 5 implementation decision, not yet reviewed).** Worker evidence that names an incident (a scene report, timeout, error or skipped job from an enrichment request) is appended to that incident's annotations in one transaction, whatever its age and the incident's status, including resolved or dismissed incidents. The incident's status, severity, title, observation times and outbox never change; only `revision` and `updated_utc` move, as for an ENDED note, so a screen showing the incident without the new evidence gets a revision conflict and reloads. The evidence ID is the deduplication key: lane job IDs carry a random per-lane prefix, so they do not repeat across restarts. Evidence that names no incident (periodic checks) is never attached to an incident by time or proximity. Unknown incidents, evidence about another camera's frames and payloads over 8,192 characters are refused and nothing is stored. Applicability on arrival (current, expired, superseded epoch, ...) is stored for display only. `sentinel run` (D-1) passes every `CoreOutput.evidence` item with `routing.annotates` to `annotate(evidence, routing.applicability)`, and requests enrichment with the incident ID that `record()` returned.
- **D36. Portable U20 and provisional demo build retention** (maintainer instruction, 2026-10-03; Codex session 7). Implement schema-constrained requests with the existing strict parser, shorter descriptions, explicit incomplete-response rejection and synthetic tests. Retain current llama.cpp b8932 provisionally for the demo, **without rebuilding now**. Record the exact revision/model/template and unresolved schema-fix requirement; demo-exception acceptance evidence is PENDING. Beta still needs a tested compatible revision satisfying the guide/audit. No hardware execution, clip label, threshold tuning, 30-minute rerun or Ollama/service change is approved by this instruction.

- **D37. Operator cache drop before measurement runs** (maintainer decision, 2026-10-03; supersedes the session 10 one-time deviation and the procedure's "do not drop caches"). `sync && sudo sysctl -w vm.drop_caches=1` is a standing, documented preparation step for **measurement runs only** (Check 9, U21, S1 and their successors). It is operator-run, never by Sentinel code or the runner (which never calls sudo), and **logged in each result**: the operator passes `--operator-dropped-caches`, and `result.json` records `preparation.drop_caches: operator_declared` (`not_declared` otherwise; a declaration, not verified). It is **never part of the Sentinel runtime**: D-1 must not depend on it, and U18's runtime policy stays open. Status records must state for every measurement whether it was used.
- **D38. S1 prompt-cache A/B approved** (maintainer decision, 2026-10-03). Claude implements the portable parts, with tests, and commits without running hardware. The parts: scene-only mode, a deterministic distinct image per request, per-request cumulative counters, optional `--llama-cache-ram`, prompt-cache log-line capture, and an opt-in operator mode reusing U21's guard. The maintainer runs both arms: **a** with llama-server's default cache and **b** with `--cache-ram 0`, after a same-boot Check 9 at the same commit, with one headless command. Adopting `--cache-ram 0` for the demo or D-1 is a separate decision after the results. No threshold, guard, model, flag of the tracked demo profile or 30-minute run changes.
- **D39. Demo capture (V2-05 demo form; session 12 implementation decision, not yet reviewed).**
  - **One reader.** `CaptureWorker` is the only code that reads the camera, in its own thread. Each successful open starts a new stream epoch (D3). A frame's ingest time is when decode returns. The consumer passes `worker.connected` to `EdgeCore`, which is `None` between connections.
  - **Bounded handoff.** `LatestFrame` keeps only the newest undelivered frame, so a slow consumer skips frames (counted as `replaced`) instead of queueing them. At most one decoded image waits (921,600 B at 640×480 BGR), plus the one the consumer holds. The epoch's undelivered frame is dropped on disconnect.
  - **Failures.** When a read returns no frame within `read_timeout_s` (5 s), or errors, the source is closed; reopening starts a new epoch. Opening is bounded by `open_timeout_s` (10 s). Reconnect waits start at 1 s and double up to 15 s while connections deliver nothing, resetting after a connection delivers frames. A change in frame size ends the connection, because boxes and tracks assume one geometry per epoch. Freshness (D14) still marks video stale at 2 s and offline at 10 s, independently of reconnects.
  - **Source PTS (amended in session 13).** `source_pts` is stream time in microseconds from `CAP_PROP_POS_MSEC`, not `CAP_PROP_PTS`, which OpenCV 4.13 rounds to whole average-frame periods and holds when a frame has no PTS. 0 is a real value only on the first frame after an open. The value is advisory (D21).
  - **FFmpeg settings.** The source passes TCP transport and video-only RTSP setup (`allowed_media_types;video`) through `OPENCV_FFMPEG_CAPTURE_OPTIONS`, which is process-wide and set before each RTSP open. Timeouts and decode threads go through `VideoCapture` parameters. `decode_threads` is 1, because FFmpeg frame threading delays each frame by up to threads − 1 frames; its CPU cost is unmeasured. FFmpeg's own error log stays at OpenCV's default.
  - **Credentials.** The URL comes only from `SENTINEL_RTSP_URL` (`rtsp://` or `rtsps://`). It never appears in config, errors, status, `repr` or probe output. Problems are fixed labels or exception class names.
  - **Unchanged for V2-05 proper.** The relay as the sole ingest owner, NVDEC/GStreamer, the H.265 main stream (D26) and a `FrameRef` buffer reference (D11) are all left for V2-05 proper.
- **D40. Legacy detector + tracker (V2-09/V2-10 demo form; session 13 implementation decision, not yet reviewed).**
  - **Boundary.** A `TrackerBackend` gets the native image (capture does not resize) and returns tracked boxes in its pixels; `PersonTracker` owns every v2 rule. Ultralytics maps boxes back to the original image itself, so the letterbox to 640×640 stays inside the backend. The V2-09 proper adapter records its own transform.
  - **Epochs and order.** The backend is reset for every new epoch, run, boot or camera; each frame is processed at most once and in order; only real detections are published. Track IDs are unique within an epoch, including after a reset following a failure (offset past the highest ID used).
  - **Confirmation.** v1's score (`surveillance4_1.py@2b2d639` lines 418–437): +1 per detection up to `confirm_detections + 2`, −1 per processed frame without the track; CONFIRMED from reaching `confirm_detections` (2) until the score falls to 0. v1's `PERSIST_FRAMES` display hold is not kept: TrackTable's 1 s expiry governs. Occupancy and zone rules count only CONFIRMED tracks (existing behaviour).
  - **Failures.** A backend error, invalid output or failed reset makes the frame FAILED with a label, and the backend is reset. `sentinel run` (D-1) marks the detector UNAVAILABLE for that frame, so occupancy is UNKNOWN, never EMPTY (D18).
  - **Legacy backend.** v1's arguments exactly (`conf` 0.4, `classes` [0], `persist`, `bytetrack.yaml`, `half` ineffective on the FP32 engine), equal to what check 8 profiled. Boxes below 0.4 never reach ByteTrack, as in v1; V2-10 proper should pass low-score boxes to ByteTrack's second stage. The engine is pinned by SHA-256 to check 6's file. The D27 guard runs through ctypes before any import that can reach CUDA. `YOLO_OFFLINE=true` stops Ultralytics' analytics and online checks.
  - **Admission.** Registry adapter `legacy-yolov8n-bytetrack` with known profile `provisional-demo-20261003T085010Z` (D28, D33). The track probe's 1.5 GB MemFree minimum is a provisional probe guard; U18's runtime policy stays open.

## Unresolved decisions and semantics

- **U1.** Resolved by D14.
- **U2. Multi-frame evidence.** `Evidence` has one source frame. VLM results citing several frames (ch. 13) need a rule for which frame's time governs age. Proposal: the oldest input frame.
- **U3. Capture time.** All ages are ingest-based. `SourceTimeQuality.CAPTURE_SYNCED` exists but nothing produces it; mapping to capture time is a Gate B / V2-06 decision.
- **U4. Per-rule TTLs.** Scene reports now use `scene.evidence_ttl_s` (D16). Rule evidence TTLs remain for V2-13/V2-16.
- **U5. PTZ pose.** The pose or view generation (ch. 16 and 27) is not in `FrameRef`/`StreamIdentity`. Decide in C9 whether a pose change supersedes evidence the way an epoch change does.
- **U6. Versioning.** Only `Evidence` carries `contract_version: 1`. Decide before the runtime↔core handoff (V2-11) whether every serialized contract carries a version, or whether the V2-49 adapter manifest version suffices.
- **U7. Evidence size.** Partly settled: scene reports are bounded (≤ 4,096 characters of input; ≤ 5 observations of ≤ 80 characters; summary ≤ 160 characters), and worker output over 8,192 characters is an error. There is still no general bound on `Evidence.value`.
- **U8. Confidence kinds.** `none`, `detector_score`, `similarity` and `calibrated_probability` are a proposal.
- **U9.** Resolved by D12.
- **U10. Memory units.** Decimal whole-device memory targets (5.0 / 5.4 GB) still need the kickoff confirmation asked for in guide ch. 26.
- **U11. Live-stream announcement.** Consumers outside the runtime (core, UI) must learn the live `StreamIdentity`, including `run_id`, through the V2-11 handoff. Until then, runtime evidence is not current for them. This is the safe default, but the handoff must carry it.
- **U12.** Resolved for the demo by D22; the post-demo direction is D26 (main stream + NVDEC). The restated gate, the 20–25 fps target and whether the substream stays as a second session remain open until after Oct 20. Original question: **Perception stream profile and frame-rate gate.** The 20–25 fps stretch target exceeds the measured 15 fps substream, and the ≥15 fps gate names a 1080p input. Options A–D and a recommendation are under "Conflict" in the V2-01 inventory. **Maintainer decision needed** before V2-05 fixes the ingest profile.
- **U13.** Resolved by D27 (preload now; restore the Jetson packages after Oct 20 behind a V2-08 restore point). Original question: **Shadowing CUDA driver library.** `libnvidia-compute-535` hides L4T's `libcuda.so.1`. Keep the `LD_PRELOAD` workaround, or remove the package or fix the loader order (a system change)? Nothing was changed.
- **U14.** Resolved for the demo by D24; revisit after Oct 20. Original question: **Runtime environment for hardware adapters.** No existing interpreter has both GStreamer bindings and TensorRT. Options (a)–(c) are in the V2-01 inventory; decide when V2-05 starts.
- **U15. Identity calibration.** D19's thresholds are placeholders. Calibrate with consented, session-separated identities (V2-07/V2-25) before any identity is shown as more than context.
- **U16. Face stage cadence.** `EdgeCore.on_frame(..., faces=None)` means the face stage did not run on that frame. Which frames get face analysis, and whether it runs asynchronously in a worker, is V2-25/V2-29 work. The interim demo adapter will run it on sampled frames.
- **U17.** Resolved by D28 and D33 (check 8 run by the maintainer on 2026-10-03, admitted with its exceedance recorded). Original question: **Admitting demo adapters without measured resource profiles.** V2-49 makes every model adapter unavailable until its resource profile is known, and no profile has been measured. Decide in D-1 how the demo admits the interim face and scene adapters. Options: record a provisional, clearly labelled "demo-unmeasured" profile from a first measured cold load, or run them outside the manifest path for the demo only.
- **U18. Free memory at GPU load** (new, session 3). On this device, GPU allocations failed whenever they exceeded **MemFree**, although MemAvailable was over 4 GB (inventory below). llama.cpp's own fit check uses MemAvailable, so it does not catch this. How does `sentinel run` (D-1) make GPU loads reliable? Options: (a) evict the model files' page cache with `posix_fadvise(DONTNEED)` (no root) and refuse to load below a MemFree threshold, as `demo_profile.py` does (3.0 GB default); (b) drop caches system-wide before loading (root; a system action); (c) load GPU components first, right after boot. Also open: whether allocations made after start-up (llama.cpp compute buffers, larger images) can fail the same way once the page cache refills. **Check 9** settles whether this depends on unified memory (`cudaMallocManaged`) or also affects `cudaMalloc`; decide after it. **Session 10/11:** the bounded Check 9 smoke passed for both APIs at 256 MiB, which says nothing about allocation beyond MemFree. The short U21 then stopped on the MemFree floor, because page cache refilled during the loads after a cache drop (session 10 log). D37 permits an operator cache drop before measurement runs only; it is not a runtime answer, and option (b) remains undecided for D-1.
- **U19. Memory left after unload (V2-54, open issue; hardware acceptance PENDING).** After check 8's workload exited and llama-server stopped, used memory was 1.889 GB: **+0.983 GB above the 0.906 GB baseline**, while Cached stayed +2.2 GB above it (0.41 → 2.61 GB). The workload's exit took used memory to 0.612 GB *below* its level with llama-server alone (2.633 → 2.021 GB). Stopping llama-server then freed only 0.13 GB, although its load had added 1.73 GB. Candidate explanations, none tested: page cache that `MemAvailable` does not credit, shared memory, NvMap/CMA pages kept by the driver, or unreclaimable slab. Check 8 recorded none of `Shmem`, `Unevictable`, `Mlocked`, `SUnreclaim`, `KReclaimable` or `CmaFree`; session 6 now instruments them portably. V2-54's unload acceptance cannot pass until the residue is explained. Proposed operator follow-up remains PENDING, with no execution approval from session 6: sample after 60 s, drop the page cache only (`echo 1`), sample again, then compare a second load/unload cycle for accumulation. **Session 10/11:** the short U21 left +658,501,632 B pressure about 7 s after exit, of which about 0.67 GB of consumed MemFree is in no sampled field. In Check 8 (E-2), llama-server PSS was 1,685,941,248 B while it ran alone, yet stopping it lowered pressure by only 132,759,552 B. Causes remain unknown. S1 records unload residues for both arms, but S1 is not the U19 reclamation check. **Session 13:** on the current boot, the first llama-server load left +656,420,864 B (about 0.66 GB of MemFree in no sampled field, surviving D37 drops), and three later cycles added none (within ±12 MB). This fits a one-time step at the boot's first large GPU load rather than per-cycle accumulation; the owner is untested.
- **U20. Portable constrained-output increment complete; hardware/model acceptance PENDING (D36).** The request schema comes from `SceneReport`; strict parsing/bounds remain unchanged. Descriptions request brevity; `length`, incomplete/error envelopes and invalid JSON are rejected. Fixed-label rejection/finish and valid-field-at-limit counts distinguish structure from accuracy. See session 7 for the 22/24 classification limits, exact b8932 revision/model/template pairing and required demo-exception evidence. Short-prompt/model validation and actual completion/semantic review remain PENDING. Retain b8932 provisionally, do not rebuild now; beta still needs the tested compatible schema-fix descendant. Real adapter/D-1 integration remains unimplemented; no schema relaxation/repair is accepted.
- **U21. Observed memory trend, hardware acceptance PENDING.** The historical +0.12 GB/min window is not a proven linear leak or GPU attribution. Session 6 adds allocator current/lifetime peaks and six meminfo fields; these views cannot establish total device usage or unload on their own. Session 8 replaces session 7's inline proposal with the tested `operator_check.py` entry point documented in `docs/U21_SHORT_VALIDATION.md`: conservative 0.2 s guard, synthetic 120 s steady pilot, explicit prerequisites, bounded timeout and owned-process cleanup. The 4.8 GB sampled stop is not a guaranteed cap. No short pilot or 30-minute rerun ran; each needs separate approval after load/service/model prerequisites. Equal-input repeated longer runs, sampler/guard overhead, transient headroom, attribution and reclamation remain PENDING; diagnostic guards are not an approved production allocation policy. **Session 10/11:** the short pilot (maintainer, previous boot) stopped on the MemFree floor after 103.77 of 120 steady seconds. Its 16-frame input could not reproduce the ramp. E-2 shows Check 8's steady ramp coincides with llama-server PSS growth (+104 MB/min) while workload PSS was flat. This supports H2 (per-request prompt-cache state; b8932 defaults to 8192 MiB) without confirming it. S1 (D38) tests H2 with distinct images per request. **Session 12:** S1 did not run. Arm a was refused at admission and arm b never started (session 12 log). **Session 13:** S1 ran at `e9af7f4`. With the default prompt cache, steady pressure rose +115,040,663 B/min (reproduced: +117,103,238); with `--cache-ram 0`, +3,855,411 B/min, with identical request counts and validity. H2 is supported for the scene-only workload: llama-server's host-RAM prompt cache causes the ramp. Adopting `--cache-ram 0` is proposed (maintainer decides). Long-run behaviour with the cache off and the full workload mix remain unmeasured.
- **U22. TensorRT engine-plan warning (new, session 21; unresolved).** Deserializing `~/yolov8n.engine` in checklist step 3 (boot `dbdbdc0c…`, TensorRT 10.3) logged TensorRT's warning that using an engine plan file across different device models is not recommended and may affect performance or cause errors. The SHA-256 matched the pin and step 3 had no failures, but neither shows that the plan was built for this device and software stack; the engine's build device and TensorRT version are not recorded. No rebuild in session 21 (maintainer instruction). A rebuild is a separate decision, and D33's profile and the parity reference apply to the pinned file only.

## V2-01 inventory so far

Nothing below is a performance, GPU-placement or memory measurement. Each row says who observed it and how.

### Camera substream (measured by the maintainer, 2026-09-29, ffmpeg/ffprobe on stream copies)

| Property | Value |
|---|---|
| Profile | `subtype=1` substream |
| Video | H.264 Main, 640×480, yuv420p |
| Frame rate | Average 15.01 fps (54060/3601); ffmpeg reports `tbr 20` |
| Bitrate | About 236 kbit/s with motion, about 38 kbit/s for a static scene |
| Audio | AAC-LC, 16 kHz mono. **v2 must drop audio** at ingest (video-only depay/decode; audio is neither decoded nor stored). |
| Timestamps | RTSP timestamps start at 0 (stream-relative). The first packet has unset timestamps and DTS is non-monotonic at stream start, reproduced on 2 of 2 recordings. |
| Main stream (`subtype=0`) | Measured by the maintainer, 2026-09-29, ffprobe: HEVC (H.265) Main, 2304×1296, 20 fps, AAC-LC audio. Bitrate and keyframe spacing PENDING (check 2). |
| Keyframe interval | **PENDING** for both profiles; command below |

Consequences already implemented: `4587021` makes `FrameStamper` keep a missing, repeated or backwards PTS for diagnostics but stamp it `source_time_quality=none`. Identity and ages already came only from ingest (receive) time, so nothing trusts such a PTS as stream time. Regression: `test_unset_and_non_increasing_pts_at_stream_start_fall_back_to_receive_time`.

Consequences for later packages: the capture adapter (V2-05) selects only the video stream; v1's 640×480 `cv2.resize` is a no-op in size on this profile (still a copy); a 640×480 source gives a 640×640 letterboxed detector input 25 % padding **if** the engine is 640×640, which the pending binding check settles (audit "engine shape").

**Replay clips.** Private recordings and review images stay outside Git and are preserved unopened. Historical ffprobe record for the `one_person`/`room_static` source: 60.0 s, 901 frames, H.264 Main 640×480 at 54060/3601 fps with an AAC-LC 16 kHz mono track; historical `empty_room`: 455.6 s, 6,832 frames, H.264 Main 640×480 at 204960/13667 (≈15.00) fps, video only. Session 8's **USER-SUPPLIED FILENAME LISTING** reports `empty_room_2026-09-29_1613.mp4` and `two_people_doorway_60s_2026-09-29.mp4`, not `room_static_60s_2026-09-29.mp4`. No rename is performed/requested now; equivalence of the listed occupied file to the historical run source remains **UNRESOLVED**. The maintainer's prior occupied-clip example review is separate from full empty-room-clip review, which is **not confirmed**. **Third-party consent is unconfirmed: local-only, no dataset/enrollment use without it.** One early figure's label remains unresolved. Codex did not open/probe/list these recordings or review images. Consent and split manifests belong to V2-07; tests that use recordings require an approved manifest, not merely a directory environment variable.

**Live runs** read the camera URL from `SENTINEL_RTSP_URL`, which the maintainer exports; Sentinel code and commands never print or log it.

### Conflict: perception frame-rate target versus the substream (U12; decided for the demo by D22)

Guide ch. 22 sets **≥15 unique detected frames/s sustained "in the declared 1080p-input core profile"**, with **20–25 fps as a stretch target**. The measured substream is **640×480 at 15.01 fps**, so:

1. The stretch target cannot be reached from this profile: there are only ~15 unique frames/s to detect.
2. The 15 fps minimum equals the source rate, so any dropped frame fails it. It needs restating as a fraction of source frames for this profile.
3. The substream is not the "1080p-input" profile the gate names. The main stream is not yet measured.

Options (for the maintainer; nothing is chosen yet):

- **A. Substream only (640×480 at 15 fps) for the beta core profile.** Restate the gate as "every unique source frame is detected: ≥ 98 % of source frames at ≥ 14.7 fps sustained, p95 frame age within budget", with 20–25 fps not applicable. Cheapest decode and memory, and it matches the detector input. Risk: small faces at distance reduce identity quality (measure in V2-25).
- **B. Raise the substream frame rate in the camera's encoder settings** (if the camera offers 20/25 fps at 640×480). Keeps option A's costs and restores the stretch target. Needs the camera's encoder options (PENDING) and a re-measured bitrate.
- **C. Main stream for perception** (resolution and rate PENDING). Meets the "1080p-input" wording, and 20–25 fps may be possible. Costs: larger decode and buffers, probably H.265 (the `~/onvif_env` restructure hints at H.265 on the main stream), more letterbox/resize work, and browser codec risk.
- **D. Two profiles:** substream for detection and tracking, main stream for face crops, clips and live view. Best identity detail, but two upstream sessions, which guide ch. 7 requires to be counted and approved, plus cross-stream timestamp mapping.

**Decision (D22):** option A for the Oct 20 demo; everything else stays open until after the demo.

**Recommendation as proposed:** A for the Oct 20 demo and the first Gate B work, because it is the only measured profile and is enough for detection and tracking. Measure the main stream and the camera's encoder options (PENDING commands) before choosing the beta profile between A, B and D. Record the chosen profile and restated gate as a decision; changing it later is a documented gate adjustment (ch. 22) made before held-out evaluation.

### v1 runtime environment `~/onvif_env` (read-only diagnostics by Claude, 2026-09-29)

Run with `PYTHONDONTWRITEBYTECODE=1` so no files were written into the environment; nothing was installed or changed. Config files were not read.

| Item | Observed |
|---|---|
| `pyvenv.cfg` | `home = /usr/local/bin`, `include-system-site-packages = false`, `version = 3.10.14` |
| Interpreter | CPython 3.10.14 (GCC 13.3.0, built 2026-03-23), base prefix `/usr/local` |
| Extra `sys.path` | `site-packages/system-packages.pth` adds `/usr/lib/python3/dist-packages`, which holds the **system Python 3.12** packages. So `import gi` finds the 3.12 build and fails ("cannot import name '_gi'"). The flag above therefore does not isolate this venv. |
| Packages | numpy 2.2.6, torch 2.9.1 (built with CUDA 12.6, cuDNN 9.3, aarch64 wheel), torchvision 0.24.1, tensorrt 10.3.0, ultralytics 8.4.25, tensorflow 2.21.0, deepface 0.0.99, onnxruntime 1.23.2, requests 2.32.5, Flask 3.1.3, PyYAML 6.0.3, pydantic 2.12.5; pycuda absent |
| OpenCV | cv2 4.13.0 from the **pip wheels** `opencv-python` **and** `opencv-python-headless` 4.13.0.92, both installed. Build info: **GStreamer NO**, FFMPEG YES (bundled avcodec 59.37.100 / avformat 59.27.100), v4l2 YES, CUDA devices 0, baseline NEON FP16. |
| v1 capture backend | Follows from the build: `cv2.VideoCapture(RTSP_URL)` can only use the bundled FFmpeg, which decodes H.264 **on the CPU**. There is no NVDEC path in this environment. This is established from the build, not from a runtime trace. |
| GPU from this venv | Without a preload, `torch.cuda.is_available()` is **False** ("No CUDA GPUs are available") and building a TensorRT `Builder` **aborts the process** ("terminate called without an active exception"). With `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`, torch reports `Orin`, capability (8, 7), and runs a tensor op; the TensorRT builder is created (`platform_has_fast_fp16` True). |

**Why: a desktop-GPU CUDA driver library shadows the Jetson one.** The Ubuntu package `libnvidia-compute-535` (535.309.01-0ubuntu0.24.04.1) installs `/usr/lib/aarch64-linux-gnu/libcuda.so.1`, which the loader finds before L4T's `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1` (package `nvidia-l4t-cuda`). Measured with `ctypes`: the 535 library reports driver API 12020 and `cuInit` returns 100 (`CUDA_ERROR_NO_DEVICE`), while the L4T library initialises, reports driver API 12060 and counts 1 device. `nvidia-smi` fails with "Driver/library version mismatch" for the same reason. apt history mentions the package on 2026-03-23 (`apt upgrade -y`) and 2026-06-13 (aptdaemon, a desktop updater); which transaction installed it was not determined.

- `~/onvif_env/bin/activate` works around this by exporting `LD_PRELOAD` of the L4T library. **Invoking `~/onvif_env/bin/python` directly, without `activate`, gets no GPU.** `start_sentinel.sh` does that, so v1's GPU use depends on the environment of the shell that ran the launcher. `llama-server` resolves `libcuda.so.1` the same way. Check 3 (run by Claude in session 3, below) confirmed both: without the preload, the TensorRT process crashes and llama-server silently runs on the CPU.
- For v2 live GPU runs: `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONPATH=src ~/onvif_env/bin/python …`.
- **How the package arrived (session 3, apt history in the release-upgrade logs):** during the 22.04 → 24.04 release upgrade on **2026-02-20**, apt replaced the Jetson `nvidia-cuda-dev` 6.2.1+b38 with Ubuntu's same-named package 12.0.146 (noble/multiverse). That pulled in `nvidia-cuda-toolkit` 12.0, `libcudart12`, `libnvidia-ml-dev` and `libnvidia-compute-535` as automatic dependencies. `nvidia-cuda-dev` and `nvidia-cuda` are marked manually installed.
- Handled by D27: preload for the demo; restore the Jetson packages after Oct 20 behind a restore point. Nothing was changed.

**Consequence for V2-05/V2-09 (U14).** No existing interpreter has both halves of the planned media path. `~/onvif_env` (3.10) has TensorRT 10.3 but no GStreamer in OpenCV and no importable `gi`. The system Python 3.12 imports `gi` with GStreamer 1.24.2 (`python3-gi` 3.48.2, `nvidia-l4t-gstreamer` 36.4.7 installed) but has no TensorRT bindings (not checked for a 3.12 wheel). **Decision for the demo (D24): option (c), software decode in `~/onvif_env`; the rest waits for V2-05 proper.** Options as recorded: (a) decode in a `gst-launch-1.0` subprocess (`nvv4l2decoder ! nvvidconv ! BGRx ! fdsink`) that any interpreter reads from a pipe, with no new Python dependencies; (b) a new, separate v2 runtime environment, never `~/onvif_env`; (c) interim CPU decode through `~/onvif_env`'s OpenCV for the Oct 20 demo only, labelled as such. Per the maintainer's rules, any new environment or dependency is proposed first.

### Device checks run by Claude on this device (2026-09-29, session 3)

The maintainer asked for these. They ran with v1 stopped, a desktop and a remote-desktop session running, and VS Code and Claude Code open. llama-server bound 127.0.0.1:18081 only and was stopped after every run; afterwards no llama-server process remained and the port was free. Nothing read or printed a secret. Each is one run: an observation, not a benchmark.

| Check | Result |
|---|---|
| 1 `nvpmodel -q` | `NV Power Mode: MAXN_SUPER`, mode 2 (no sudo needed). The `gst-inspect` lines wait for V2-05 proper (D24). |
| 3a, no preload | Maps `/usr/lib/aarch64-linux-gnu/libcuda.so.535.309.01`; `cuInit` returns 100; TensorRT logs "CUDA initialization failure with error: 100"; the process **segfaults (exit 139)**. |
| 3a, L4T preload | Maps `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1.1`; `cuInit` 0; the engine deserializes; exit 0. |
| 3b, no preload | Ready in 4.1 s and maps the 535 library. Logs "failed to initialize CUDA: no CUDA-capable device is detected" and "no usable GPU found, --gpu-layers option will be ignored": the model runs **entirely on the CPU, silently**. VmHWM 2,088,868 kB. |
| 3b, L4T preload | When it loads: 17/17 layers and the vision encoder (`CLIP using CUDA0`) on CUDA0; CUDA0 buffers: model 661.25 MiB, KV 24.00 MiB, compute 136.00 MiB, vision compute 30.31 MiB. Ready (HTTP 200 from `/health`) after 14.1 s; VmHWM 1,858,860 kB. It loaded only when MemFree was high enough (GPU memory, below). Session 2's version of this check counted any HTTP reply as ready, even during loading; it now waits for `/health`. |
| 5 detector engine | Ultralytics 8.4.25 metadata (exported 2026-04-18): task `detect`, imgsz 640×640, batch 1, stride 32; export arguments not half, int8 or dynamic, so the engine is **FP32** and v1's `half=True` has no effect on it. Output `output0` (1, 84, 8400) FP32, linear; NMS runs outside the engine. 1 optimization profile; engine device memory 18,944,000 B; TensorRT 10.3.0. TensorRT warns "Using an engine plan file across different models of devices"; not investigated. |
| 6 SHA-256 | See below |
| llama.cpp build | b8932 (`98dc1418e`) |

```text
08370639f961d2c67148c19562718ef80527c7085e88d2d923176180f1b98637  yolov8n.engine
ce0d4b122d328d14390ef160785da3a51a527f96844f392a04cb2db96f134e5d  LFM2-VL-1.6B-Q4_0.gguf
65ec437db88d65fff93f472d00c145e09880769ac67fedff5cd1c0f8d8301d87  mmproj-LFM2-VL-1.6B-Q8_0.gguf
4c3b66c0a6bf25d5ef876edf118a96deebc15de04d58f4e280cdcfda708b7932  LFM2-VL-1.6B-Q8_0.gguf
0a82498edc354b50247fee78081c8954ae7f4deee9068f8464a5ee774e82118a  LFM2-VL-1.6B-F16.gguf
b637bfa6060be2bc7503ec23ba48b407843d08c2ca83f52be206ea8563ccbae2  mmproj-LFM2-VL-1.6B-F16.gguf
```

**GPU memory (new finding, U18).** With the preload, llama-server's GPU allocations failed with `NvMapMemAllocInternalTagged: ... error 12` and "cudaMalloc failed: out of memory" whenever they exceeded **MemFree**, although MemAvailable was over 4 GB (mostly page cache):

| MemFree before loading | Outcome |
|---|---|
| 0.09 GB (MemAvailable 4.52 GB) | Failed at the first CUDA0 buffer (661.25 MiB model) |
| not recorded (first run of the check script) | Model loaded; failed at the 538.02 MiB vision projector buffer |
| about 1.70 GB (U17 smoke run) | Failed at the 538.02 MiB vision projector buffer |
| 3.08 GB, after evicting the model files' page cache with `posix_fadvise(DONTNEED)` | Loaded fully |

llama.cpp's own fit check reported 3.7–4.3 GB free each time, so it did not catch this.

### Check 3b and GPU memory: the maintainer's questions (answered 2026-10-03, session 4)

1. **Was `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` set in check 3b?** Yes, in every llama-server run of session 3: check 3b without and with the preload, both re-runs and the U17 smoke runs. It mirrors v1's launcher, which sets it. In this build (b8932) the variable makes `ggml_cuda_device_malloc` call `cudaMallocManaged` instead of `cudaMalloc` (`ggml/src/ggml-cuda/ggml-cuda.cu`); the error text says "cudaMalloc failed" either way. On an integrated GPU such as Orin, llama.cpp reports MemAvailable as free device memory whether or not the variable is set, which is why it showed 3.7–4.3 GB free while allocations failed.
2. **Do allocations beyond MemFree fail with and without it?** **With it: yes**, as far as session 3's load failures suggest, not a general allocator proof. Both loads starting at or below about 1.7 GB MemFree failed; the load at 3.08 GB succeeded. **Without it: not tested.** Session 8 explicitly excludes the historical cache-fill/failure-seeking recipe and server comparison. The replacement bounded API smoke does not cross MemFree and cannot resolve this uncertainty. A separately reviewed safe plan and conservative U18/D-1 load policy remain PENDING; no production policy is approved by portable smoke tests.
3. **Could the wrong libcuda have contributed to the original v1 OOM?** No record of that OOM (time, process or message) exists in the repository, the local notes or the session records. The answer below therefore comes from mechanism, not from evidence about the event:
   - **Not as the direct cause of a GPU allocation failure.** A process that loads the 535 library never creates a CUDA context (`cuInit` 100). TensorRT crashes (exit 139) rather than running out of memory, and llama-server allocates no GPU memory. A CUDA or NvMap out-of-memory error must therefore come from a process that had L4T's library.
   - **Possibly, indirectly.** If v1 was launched without `activate`, llama-server ran on the CPU. That puts full inference load on the CPU and slows v1's loop, and it changes where the memory is. Its effect on whole-device memory was not measured. VmHWM was 2.09 GB on the CPU against 1.86 GB on the GPU, but these are not comparable, because NvMap GPU buffers need not appear in a process's RSS.
   - **A more likely mechanism for a GPU-allocation OOM does not involve the libcuda at all:** allocations beyond MemFree fail even when page cache could be reclaimed (U18). MemFree sits far below MemAvailable after a desktop session or after reading the model files, which is the normal state of this device.
   - Check 9c (PENDING) searches the retained kernel log for OOM kills and NvMap failures, so the original event can be classified as a kernel OOM kill or a GPU allocation failure.

### Other software facts (read-only, 2026-09-29)

| Item | Observed |
|---|---|
| CUDA toolkit | `/usr/local/cuda-12.6` (`version.json`: CUDA SDK 12.6.11; `libcudart.so.12.6.68`) |
| Release upgrade | The 22.04 → 24.04 release upgrade ran on **2026-02-20** (release-upgrader `main.log` and apt history timestamps). Session 1 dated it 2026-04-18, which is only the timestamp of the log directory under `/var/log/dist-upgrade/`; corrected in session 3. This is how the device reached Ubuntu 24.04. |
| NVIDIA apt sources | `repo.download.nvidia.com/jetson/{common,t234}` at **r36.4**. There is also a generic `cuda-ubuntu2404-arm64` CUDA repository, which is not the Jetson repository. Installing from it could replace L4T CUDA components; treat it as a risk to check before any apt operation. |
| Media tools | ffmpeg/ffprobe 6.1.1 (Ubuntu), `gstreamer1.0-tools`, `-plugins-good`, `-plugins-bad` and `-libav` 1.24.x installed |
| Environment change, 2026-10-04 (USER-SUPPLIED; session 20) | The maintainer ran `apt install caffeine` (apt history 13:45:03–13:45:06Z): `caffeine` 2.9.12-1 plus the automatic dependencies `python3-ewmh`, `python3-xlib` and `gir1.2-ayatanaappindicator3-0.1`; nothing upgraded or removed; no CUDA/L4T package. It adds an XDG autostart entry, used only by graphical sessions. Claude saw no process afterwards; the running state is otherwise unverified. Checklist provenance records `caffeine_procs` from step 3 on. **Session 21 (USER-SUPPLIED, maintainer):** after the reboot before step 3, a process was observed with PID 2662, comm `caffeine`, cgroup `/system.slice/cron.service`; the maintainer sent it SIGTERM before step 3's provenance (`caffeine_procs=0`). How it started was not established; it is not attributed to the desktop autostart. |
| Model files | `~/yolov8n.engine` (14,486,949 B, dated 2026-04-18); launcher paths `~/models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf` and `mmproj-LFM2-VL-1.6B-Q8_0.gguf`; SHA-256 in the session 3 checks below |

### Device snapshot (read-only, session 1)

| Item | Observed on 2026-09-29 |
|---|---|
| L4T | `R36 (release), REVISION: 4.7` (built 2025-09-18); kernel `5.15.148-tegra`, aarch64 |
| Packages | `nvidia-l4t-core 36.4.7-20250918154033`, `libnvinfer10 10.3.0.30-1+cuda12.5`; the `nvidia-jetpack` meta-package is **not installed** |
| Userspace | **Ubuntu 24.04.4 LTS**, although JetPack 6 is documented on Ubuntu 22.04. How the device got there must be recorded before any dependency or runtime change. |
| JetPack mapping | The guide's example (JetPack 6.2.2 = L4T 36.5) does **not** match this device. Look up L4T 36.4.7 in NVIDIA's release notes; it was not inferred here. |
| Pythons | `/usr/bin/python3` 3.12.3 (system); `/usr/bin/python3.10` 3.10.12 (no `ensurepip`); `/usr/local/bin/python3.10` 3.10.14 (`~/onvif_env` and the dev venv) |
| Memory | `free -b`: total 7,990,001,664 B, available 5,130,117,120 B at one idle instant with v1 stopped. A snapshot, not a baseline. |
| Disk | `/` is 116 G with 29 G free (74 % used) |
| v1 stack | Not running during the session: no tmux server and no engine, dashboard or llama-server process |

### v1 files outside the repository (read-only; none edited)

| File | Compared with repo at `2b2d639` |
|---|---|
| `~/surveillance4_1.py` | Same code; only trailing whitespace at EOF differs. `start_sentinel.sh` launches this copy. |
| `~/build_face_db.py` | Same; only the final newline differs |
| `~/dashboard.py` | Older revision than the repo file; `start_sentinel.sh` launches this copy. Details are in the local note. |
| `~/dashboard_1.py` | No repo counterpart; historical dashboard revision, not launched by `start_sentinel.sh` |
| `~/onvif_env` restructure | Not a git repository. Contains `src/nano_surveillance/` (capture, classifiers, cli, config, detector, onvif_utils, pipeline, recognizer, tracker), `configs/nano.yaml`, `configs/nano-cloud.yaml`, `scripts/` and `requirements-{base,jetson,cloud-vlm}.txt`. Its pipeline decodes the main stream with `rtph265depay ! avdec_h265` (CPU H.265), while `.env.example` uses `subtype=1`; these are unverified codec hints for Gate B. It references another environment, `~/onvif_env2`. Never copy its configs into the repository. |

No v1 code was migrated in this slice.

## Hardware checks PENDING (for the maintainer to run; none of these results exist yet)

Checks 1 (`nvpmodel` only), 3, 5 and 6 were run by Claude in session 3 (inventory above). Check 4 was merged into 3b. Check 8 was run by the maintainer on 2026-10-03 (session 5 log). Bounded Check 9 smokes ran on the previous boot (session 10 log) and on the current boot at `84f15ec` and `e9af7f4`. S1 ran at `e9af7f4`, and the V2-05 steady capture ran on the camera (session 13 log). These are historical conditions, not a verification of the current service state. Session 8 superseded the old Check 8/9 service-stop, cache-fill, failure-seeking, server-launch and raw-journal recipes; none is authorized. **Session 15 replaces the separate V2-05 outage and V2-09/V2-10 blocks with one numbered checklist (below).** It covers the Ultralytics setting, the outage check, the detector check, the confirming profile (`--llama-cache-ram 0`, still subject to the maintainer's approval) and the first end-to-end alert with the status page. Checks 2 and 7 remain separately PENDING. New operator results need USER-SUPPLIED MEASUREMENT attribution and their conditions, including whether D37's cache drop was used.

### Operator checklist (session 15): steps 1–2 done (sessions 19–20, USER-SUPPLIED); step 3 recorded in part (session 21); steps 4–5 PENDING

Claude ran none of these steps. Run them in order; each step names its prerequisites, and later steps depend on earlier ones. Results are USER-SUPPLIED MEASUREMENTS or observations; record the real boot ID and commit from each step's `provenance.txt`, never the commit this list was written at.

**Rules for every step.**
- Authenticate with `sudo -v` on its own line first. Each command that needs sudo is then its own line; none is combined with other commands or placed in a script.
- No firewall rule is added or changed (no iptables, nft or ufw). Outages are physical: camera power or its network uplink.
- Secrets are entered with `read -rs` into the current shell only, never written to a file, and unset afterwards.
- Artifacts stay in `~/sentinel-runs/d1-checklist/<step>/` (mode 0700) and are never committed. Return the files each step lists, plus `provenance.txt`.

**Before each step (same shell).** Since session 20, the block refuses an existing step directory, so earlier results are never overwritten. A rerun uses a new directory name, as step 2's run 2 did. The client check prints counts only, never process arguments, which can contain a camera URL.

```bash
cd /home/villain8001/sentinel-surveillance && set -o pipefail
STEP=3-detector          # change per step: 4-confirming-profile, 5-end-to-end
D=~/sentinel-runs/d1-checklist/$STEP
if [ -e "$D" ]; then echo "STOP: $D already exists; do not continue"; D=/nonexistent-stop; else
  mkdir -p -m 700 "$D"
  { echo "utc=$(date -u +%FT%TZ)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"; echo "boot_started=$(uptime -s)"
    echo "commit=$(git rev-parse HEAD)"; echo "tracked_changes=$(git status --porcelain --untracked-files=no | wc -l)"
    echo "display_manager=$(systemctl is-active display-manager)"
    echo "other_camera_clients=$(pgrep -fc 'surveillance4_1|llama-server|ffmpeg|ffprobe|sentinel\.cli')"
    echo "caffeine_procs=$(pgrep -fc '[c]affeine')"
    echo "memfree_bytes=$(awk '/^MemFree:/{printf "%d", $2*1024}' /proc/meminfo)"; } | tee "$D/provenance.txt"
fi
```
After a `STOP`, later blocks fail without writing, because `/nonexistent-stop` does not exist.

**1. Ultralytics analytics and offline setting (inspection; about 1 min; no GPU, camera or sudo).**
- *Prerequisites:* none.
- *Commands:*
  ```bash
  python3 -c "import json,pathlib; p=pathlib.Path.home()/'.config/Ultralytics/settings.json'; print('settings_sync', json.loads(p.read_text()).get('sync'))" | tee "$D/settings.txt"
  YOLO_OFFLINE=true ~/onvif_env/bin/python -c "from ultralytics.utils import ONLINE, SETTINGS; from ultralytics.utils.events import events; print('v2_effective online', ONLINE, 'settings_sync', SETTINGS['sync'], 'events_enabled', events.enabled)" | tee -a "$D/settings.txt"
  grep -n 'YOLO_OFFLINE' src/sentinel/inference/legacy_ultralytics.py | tee -a "$D/settings.txt"
  ```
- *Expected:*
  - Line 1: `settings_sync True`, the device file as of session 13, or `False` if 1b already ran.
  - Line 2: `v2_effective online False … events_enabled False`. Ultralytics 8.4.25 enables events only when `ONLINE` is true, and `YOLO_OFFLINE=true` makes `is_online()` return false without a DNS lookup. v2's legacy adapter sets the variable before importing Ultralytics, so `sentinel run` and `sentinel track probe` send no analytics, whatever the settings file says.
  - Line 3: the assignment in `load()`.
  - v1, `demo_workload.py` and check 8 do not set the variable, so they send analytics when online while `sync` is true.
- *1b (optional, the maintainer's choice; it changes a user settings file, not the repository):* `YOLO_OFFLINE=true ~/onvif_env/bin/yolo settings sync=False`. Then run line 1 again: it should print `settings_sync False`.
- *Stop if* line 2 prints `events_enabled True`. Do not run steps 3–5; report it.
- *Artifacts:* `$D/settings.txt`.
- **Done (USER-SUPPLIED OBSERVATIONS, maintainer, 2026-10-04T10:02:46Z; recorded in session 19).** Boot `201a195f-98a2-4cef-b6e3-3955f6f33f2b`, commit `85fb7e70eac5d7ce3d335d3595881fea09fe096b`, `tracked_changes=0`, `display_manager=inactive`. `settings_sync=True`. With `YOLO_OFFLINE=true`: `ONLINE=False`, `events_enabled=False`. The adapter sets `YOLO_OFFLINE` before importing Ultralytics. 1b (`sync=False`) was **not** performed. Artifacts: `~/sentinel-runs/d1-checklist/1-ultralytics/provenance.txt` and `settings.txt` (local).

**2. V2-05 attended physical outage and recovery (about 4 min attended; no GPU, no sudo). Revised in session 19: 180 s window, timed prompts.**
- *Prerequisites:*
  - Step 1 done (session 19).
  - v1 and every other camera client on this host stopped (checked in 2a).
  - Headless preferred (note the conditions either way).
  - The camera's power plug (or its network uplink) within reach, with the terminal visible from there.
- *Why it changed (session 19, from the code, not a measurement):* a power cut makes the camera reboot, and its boot time on this device has not been measured. After a failed open, the worker waits 1, 2, 4, 8, then 15 s (`reconnect_max_s`) between attempts, and each failing open can take up to 10 s (`open_timeout_s`). Once the camera is up again, the next successful open can therefore be up to about 25 s away. Session 15's 90 s window could end before frames returned. The old command also held the terminal, so the cut and restore times could not be written while it ran. The probe now runs in the background and the shell prints timed prompts and records their times.
- *2a. Artifact directory, provenance and the client check (paste as one block):*
  ```bash
  cd /home/villain8001/sentinel-surveillance && set -o pipefail
  STEP=2-capture-outage; D=~/sentinel-runs/d1-checklist/$STEP
  if [ -e "$D" ]; then echo "STOP: $D already exists; do not continue"; D=/nonexistent-stop; else
    mkdir -p -m 700 "$D"
    { echo "utc=$(date -u +%FT%TZ)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"; echo "boot_started=$(uptime -s)"
      echo "commit=$(git rev-parse HEAD)"; echo "tracked_changes=$(git status --porcelain --untracked-files=no | wc -l)"
      echo "display_manager=$(systemctl is-active display-manager)"; } | tee "$D/provenance.txt"
    echo "other_camera_clients=$(pgrep -fc 'surveillance4_1|llama-server|ffmpeg|ffprobe|sentinel\.cli')" | tee -a "$D/provenance.txt"
  fi
  ```
  Expected: `tracked_changes=0` and `other_camera_clients=0`. Stop on `STOP` or a count above 0. (Session 20 replaced `pgrep -fa`, which prints process arguments, and added the existing-directory refusal.)
- *2b. Credential entry (paste this line alone, then type or paste the camera URL and press Enter; nothing is echoed and nothing goes into the shell history):*
  ```bash
  read -rsp 'Camera URL (hidden): ' SENTINEL_RTSP_URL; echo; export SENTINEL_RTSP_URL; [ -n "$SENTINEL_RTSP_URL" ] && echo "url set" || echo "URL EMPTY: stop"
  ```
  Expected: `url set`. Never paste 2b together with other lines: the next pasted line would be read as the URL.
- *2c. The timed run (paste as one block; about 3 min 20 s):*
  ```bash
  ( PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli capture probe config/default.yaml --seconds 180 \
      >"$D/c.json" 2>"$D/c.err"; echo "exit=$?" >> "$D/provenance.txt" ) &
  echo "probe_start_utc=$(date -u +%FT%TZ)" | tee -a "$D/provenance.txt"
  sleep 20; echo "cut_prompt_utc=$(date -u +%FT%TZ)" >> "$D/provenance.txt"; printf '\a\n>>> CUT CAMERA POWER NOW (keep it off until the next prompt)\n'
  sleep 15; echo "restore_prompt_utc=$(date -u +%FT%TZ)" >> "$D/provenance.txt"; printf '\a\n>>> RESTORE CAMERA POWER NOW\n'
  echo "Waiting for the probe to finish (about 2.5 min more)..."; wait; tail -n 1 "$D/provenance.txt"
  ```
  - At the `CUT` prompt (about 20 s in), unplug the camera's power at once. At the `RESTORE` prompt (15 s later), plug it back in. The recorded times are the prompt times; the physical action follows within a second or two.
  - Leave the terminal alone until the `exit=` line appears. The probe stops itself after 180 s, plus up to 16 s to close an open attempt.
  - Ctrl-C during the prompts stops only the prompts, not the probe: then run `wait`, and note in `provenance.txt` that the times are approximate.
- *2d. Summary and the userinfo check (paste as one block; prints numbers only, never the URL):*
  ```bash
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); w=r["worker"]; u=r["upstream_connections"]; print("status", r["status"], "| connects", w["connects"], "| epochs", r["epochs"], "| stream_ends", w["stream_ends"], "| open_failures", w["open_failures"], "| last_problem", w["problem"], "| frames", r["frames"]["captured"], "| upstream", u.get("status"), "max", u.get("max"))' "$D/c.json"
  U="${SENTINEL_RTSP_URL#*://}"; U="${U%%@*}"
  printf 'userinfo_lines=%s lines=%s\n' "$(cat "$D/c.err" "$D/c.json" | grep -cF -- "$U")" "$(cat "$D/c.err" "$D/c.json" | wc -l)" | tee -a "$D/provenance.txt"
  unset U SENTINEL_RTSP_URL
  ```
- *Expected:*
  - `exit=0` and `status frames_received`. These alone show only that frames arrived at some point, not recovery.
  - **Recovery:** `connects` ≥ 2 and `epochs` ≥ 2. `epochs` counts only epochs whose frames reached the probe, so 2 or more means frames arrived after the restore.
  - **Read timeout:** `stream_ends` ≥ 1. The stream ends about 5 s after the cut (`read_timeout_s`).
  - **Reopen during the outage:** `open_failures` ≥ 1 is expected with a power cut.
  - `last_problem` keeps the last recorded problem (for example `no_frame` or an open failure). Recovery does not clear it, so it is not an error by itself.
  - `upstream observed max 1`. If it shows `unavailable` (the URL uses a hostname, not an IP address), the session count is not observed: record that.
  - `userinfo_lines=0`.
- *Stop if:*
  - `URL EMPTY`, or 2a lists another camera client.
  - `status no_frames`: check the URL and the camera, and do not continue to steps 3 or 5.
  - `epochs` < 2: recovery is not shown. If the camera was still booting when the probe ended, record the result as inconclusive (camera boot longer than the window), not as a failure. Do not rerun silently: report it. A rerun with the network uplink cut instead of power needs no camera reboot.
  - `userinfo_lines` > 0: do not share `c.err` or `c.json`; report only the counts.
  - The connections max is above 1: another client is connected.
- *Artifacts:* `$D/c.json` and `$D/provenance.txt` (return them); `$D/c.err` stays local.
- **Done (USER-SUPPLIED MEASUREMENTS, maintainer, 2026-10-04; recorded in session 20).** Boot `201a195f-98a2-4cef-b6e3-3955f6f33f2b`, commit `55ab3b26fe918e5613897f891f5a37e0d5f40db1`, `tracked_changes=0`, headless.
  - Run 1 (`2-capture-outage/`): no physical outage. Steady capture only, not outage evidence.
  - Run 2 (`2-capture-outage-power-20261004T141037Z/`): camera power cut and restored within about 2 s of the prompts (cut prompt 14:10:57Z, restore prompt 14:11:12Z; prompt times, not measured actions). Results: `exit=0`, `status frames_received`, `connects` 2, `epochs` 2, `stream_ends` 1, `open_failures` 3, frames 2,072, upstream observed max 1, `userinfo_lines=0`.
  - All step-2 criteria are met. The 2a client check was not run for run 2; upstream before 0 / max 1 stands in for it.
  - Details and Claude's calculations are in the session 20 log.

**3. V2-09/V2-10 guard refusal and tracking (about 3 min attended; GPU; a functional check, not a measurement, so no D37 drop). Tightened in session 20.**
- *Prerequisites:*
  - Step 1 printed `events_enabled False` (done). Step 2 is done.
  - Headless (D29).
  - v1, llama-server and other camera clients stopped.
  - MemFree at least 1,500,000,000 B, the probe's default `--min-free-gb 1.5`. The MemFree check runs **before** the D27 guard, so below it even 3a prints a `memfree_below_minimum` refusal instead of the guard label. Claude saw 965,459,968 B at 14:26Z on boot `201a195f…`. The checklist's remedy is a reboot (the maintainer's decision), not a cache drop or a lower threshold.
  - Someone available to walk through the camera view during 3b.
- *3-0. Pre-check (paste as one block; writes nothing):*
  ```bash
  printf 'display_manager=%s memfree_bytes=%s other_camera_clients=%s caffeine_procs=%s\n' "$(systemctl is-active display-manager)" "$(awk '/^MemFree:/{printf "%d", $2*1024}' /proc/meminfo)" "$(pgrep -fc 'surveillance4_1|llama-server|ffmpeg|ffprobe|sentinel\.cli')" "$(pgrep -fc '[c]affeine')"
  ```
  - Continue only with `display_manager=inactive`, `memfree_bytes` ≥ 1500000000 and `other_camera_clients=0`.
  - If the display manager is active, run `sudo -v`, then `sudo systemctl stop display-manager`, each on its own line, and repeat 3-0.
  - If MemFree is low, stop. After a reboot (the maintainer's decision), make the system headless as above and repeat 3-0.
  - `caffeine_procs` is recorded for the conditions. Removing the package or stopping the process is the maintainer's choice and is not part of this step.
- *3-1. Directory and provenance:* the "Before each step" block above, with `STEP=3-detector`. Expected: `tracked_changes=0`, `other_camera_clients=0`, `memfree_bytes` ≥ 1500000000. Stop on `STOP`.
- *3-2. Credential entry (paste this line alone, then the URL, then Enter). Step 2's 2d unset the URL, and the probe checks it before the guard, so 3a needs it too:*
  ```bash
  read -rsp 'Camera URL (hidden): ' SENTINEL_RTSP_URL; echo; export SENTINEL_RTSP_URL; [ -n "$SENTINEL_RTSP_URL" ] && echo "url set" || echo "URL EMPTY: stop"
  ```
- *3-3. a) Guard refusal without the preload (paste as one block; a few seconds). `env -u LD_PRELOAD` makes sure an exported preload cannot mask the guard:*
  ```bash
  env -u LD_PRELOAD PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli track probe config/default.yaml \
    --engine ~/yolov8n.engine --seconds 10 >"$D/a.txt" 2>&1; echo "exit_a=$?" | tee -a "$D/provenance.txt"
  grep -m1 '^track probe:' "$D/a.txt" || echo "NO REFUSAL LINE: stop and report"
  ```
  Expected: `exit_a=1` and `track probe: libcuda_not_l4t`.
- *3-4. b) 60 s of tracking with the L4T preload (paste as one block; about 70–90 s with the model load). Walk through the view for part of it, and note roughly when:*
  ```bash
  echo "b_start_utc=$(date -u +%FT%TZ)" | tee -a "$D/provenance.txt"; echo "Running 60 s plus model load; walk through the view now."
  LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli \
    track probe config/default.yaml --engine ~/yolov8n.engine --seconds 60 >"$D/b.json" 2>"$D/b.err"
  echo "exit_b=$?" | tee -a "$D/provenance.txt"
  ```
- *3-5. Summary and the userinfo check (paste as one block; numbers only, never the URL):*
  ```bash
  python3 -c 'if 1:
      import json,sys
      r=json.load(open(sys.argv[1]))
      if r.get("status")=="refused": print("REFUSED", r.get("reason"), "memfree", r["memory_before"]["MemFree"], "min", r["min_free_bytes"]); sys.exit()
      t=r["tracking"]; p=r["persons"]; b=r["backend_ms"]; f=r["frames"]; u=r["upstream_connections"]; l=r["load"]
      g=lambda m: None if m is None else round(m["MemFree"]/1e9, 3)
      print("status", r["status"], "| captured", f["captured"], "replaced", f["replaced"], "| processed", t["processed"], "skipped", t["skipped"], "failed", t["failed"])
      print("failures", t["failures"], "| error_types", t["error_types"], "| backend_ms p50", b.get("p50"), "p95", b.get("p95"))
      print("track_ids", p["track_ids"], "confirmed", p["confirmed_track_ids"], "| frames_with_persons", p["frames_with_persons"], "max_per_frame", p["max_per_frame"])
      print("upstream", u.get("status"), "max", u.get("max"), "| load_s", l["seconds"], "| MemFree GB before", g(l["memory_before"]), "after", g(l["memory_after"]))' "$D/b.json"
  U="${SENTINEL_RTSP_URL#*://}"; U="${U%%@*}"
  printf 'userinfo_lines=%s lines=%s\n' "$(cat "$D/a.txt" "$D/b.json" "$D/b.err" | grep -cF -- "$U")" "$(cat "$D/a.txt" "$D/b.json" "$D/b.err" | wc -l)" | tee -a "$D/provenance.txt"
  unset U SENTINEL_RTSP_URL
  ```
- *Expected:*
  - a: `exit_a=1`, `track probe: libcuda_not_l4t`.
  - b:
    - `exit_b=0`, `status frames_received`, `failed 0`, empty `failures` and `error_types`.
    - `processed` close to `captured`, with few `replaced`.
    - `backend_ms` p50/p95 for information (check 8: 45/56 ms).
    - `track_ids` > 0 and `frames_with_persons` > 0 if someone walked through.
    - `upstream observed max 1`.
    - MemFree before and after the load, for information.
    - `userinfo_lines=0`.
- *Stop if:*
  - 3-0 or 3-1 fails its conditions, or `URL EMPTY`.
  - a prints anything other than `libcuda_not_l4t`, or `NO REFUSAL LINE`. The guard may be broken (or MemFree refused first), so do not run b, 4 or 5; report it.
  - b prints `REFUSED memfree_below_minimum`. Record it; the maintainer decides on a reboot, then step 3 runs again in a new directory.
  - b has `failed` > 0 or non-empty `failures`/`error_types`. Record them and stop.
  - `track_ids` 0 although someone walked through. Report it; do not rerun silently.
  - `userinfo_lines` > 0. Do not share `a.txt`, `b.json` or `b.err`; report only the counts.
- *Artifacts:* return `$D/provenance.txt`, the 3-3 and 3-5 output lines, and `$D/a.txt` and `$D/b.json` if `userinfo_lines=0`. `$D/b.err` stays local. Also give the conditions: when someone was in view, whether anything else ran, and whether the system was rebooted since step 2.
- **Recorded in part (USER-SUPPLIED MEASUREMENTS, maintainer, 2026-10-04; recorded in session 21).** Boot `dbdbdc0c-5c27-469b-ac1e-280a2b1140c7` (rebooted since step 2), commit `7dc7a04bf242e05aae1bc29db89c37996bb51b8a`, `tracked_changes=0`, headless, `other_camera_clients=0`, `caffeine_procs=0`, MemFree 3,966,988,288 B.
  - a: `exit_a=1`, `track probe: libcuda_not_l4t`. Met.
  - b: `exit_b=0`, `frames_received`, 888 captured, 858 processed, 29 replaced, 1 discarded, 0 skipped, 0 failed, no failures or error types, backend p50/p95 37.456/38.165 ms, load 8.69 s, upstream observed max 1, userinfo count 0. Met.
  - Persons: `track_ids` 5, `confirmed_track_ids` 4, `frames_with_persons` 845, `max_per_frame` 3. **Detection correctness is PENDING:** the maintainer's observations of who was in view were not supplied. These are counts, not accuracy, and track IDs are not people.
  - **Deviation:** 3-5's summary could not parse `b.json`, which had 446 characters of library output before the JSON object. The maintainer recovered the object; the original files are unchanged. Fixed in session 21 (D48). The userinfo count was not saved to `provenance.txt`.
  - Details and Claude's calculations are in the session 21 log.

**4. Confirming combined profile, guarded (`step4-combined-cache-off-v2`, D47; a measurement run: about 25 min, of which about 3 min attended; GPU; sudo only for headless and the D37 drop).**
- *Prerequisites:*
  - Steps 1 and 3 passed.
  - **The maintainer approved this run and the D47 criteria before it starts.**
  - A plain SSH session inside `tmux new -s step4`. `operator_check` stops on SIGHUP, so a dropped SSH must not end the run.
  - v1, any llama-server and Ollama stopped.
  - No tracked changes.
  - **VS Code Remote and Claude Code closed before 4d**: the inspection and the profiler refuse dev tools, and `--allow-*` is never used.
  - No reboot between 4b and 4d. The identity, Check 9 and kernel evidence are per boot, and the journal is volatile.
- *4a. Variables and a unique artifact directory (current shell):*
  ```bash
  cd /home/villain8001/sentinel-surveillance && set -o pipefail
  RUN="step4-$(date -u +%Y%m%dT%H%M%SZ)"
  D="$HOME/sentinel-runs/d1-checklist/$RUN"
  CLIP="$HOME/clips/two_people_doorway_60s_2026-09-29.mp4"
  mkdir -p -m 700 "$HOME/sentinel-runs/d1-checklist" && mkdir -m 700 "$D" && echo "artifacts: $D"
  ```
  Stop if `mkdir` reports an error.
- *4b. Identity snapshot, before the cache drop (read-only hashing of about 1.5 GB; no sudo). It stops at the first failure:*
  ```bash
  ( set -euo pipefail
    { echo "utc=$(date -u +%FT%TZ)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"; echo "boot_started=$(uptime -s)"
      echo "commit=$(git rev-parse HEAD)"; echo "tracked_changes=$(git status --porcelain --untracked-files=no | wc -l)"
      echo "display_manager=$(systemctl is-active display-manager || true)"; } > "$D/provenance.txt"
    [ "$(git status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "tracked changes present"; exit 1; }
    CLIP_SHA="$(grep -F 'room_static_60s_2026-09-29.mp4' docs/LOCAL_NOTES.md | grep -oE '[0-9a-f]{64}' || true)"
    [ "${#CLIP_SHA}" -eq 64 ] || { echo "recorded clip hash not found in docs/LOCAL_NOTES.md"; exit 1; }
    .venv/bin/python benchmarks/runner/operator_check.py --step4-identity --step4-clip "$CLIP" \
      --step4-clip-sha256 "$CLIP_SHA" | tee "$D/identity.json"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "complete"; print(r["result_file"])' \
      "$D/identity.json" > "$D/identity.path"
    cp "$(cat "$D/identity.path")" "$D/identity-result.json"
    echo "identity_finished_utc=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finished_utc"])' "$D/identity.json")" >> "$D/provenance.txt"
    echo "4b passed"
  ); echo "4b_exit=$?" | tee -a "$D/provenance.txt"
  ```
  - *Expected:* `"status": "complete"`, `"clip_matches_recorded": true`, the file count and `hash_seconds_total`.
  - The terminal shows a summary only. The full result, with every SHA-256 including the clip's, stays in `identity-result.json` and is never committed.
  - *Stop unless* it prints `4b passed` and `4b_exit=0`. A clip mismatch, a missing file or tracked changes stop it at the failing command. (Never put `|| …` after these blocks: bash ignores `set -e` inside a subshell on the left of `||`.)
- *4c. Headless and the D37 drop (current shell; each sudo command on its own line; only after `4b passed`):*
  ```bash
  sudo -v
  sudo systemctl stop display-manager
  sync
  sudo sysctl -w vm.drop_caches=1
  echo "d37_drop_utc=$(date -u +%FT%TZ)" | tee -a "$D/provenance.txt"
  ```
- *4d. Same-boot Check 9, then step 4 with that exact Check 9 result. It stops at the first failure, and the exit status is preserved:*
  ```bash
  ( set -euo pipefail
    .venv/bin/python benchmarks/runner/operator_check.py --execute-workload check9 --operator-dropped-caches \
      | tee "$D/check9.json"
    CHECK9="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["check9"]["status"] == "bounded_smoke_complete"; print(r["result_file"])' "$D/check9.json")"
    echo "check9_result=$CHECK9" >> "$D/provenance.txt"
    timeout -s TERM --kill-after=30s 1440s .venv/bin/python benchmarks/runner/operator_check.py \
      --execute-workload step4 --check9-report "$CHECK9" --identity-report "$(cat "$D/identity.path")" \
      --step4-clip "$CLIP" --confirm-step4-prerequisites --operator-dropped-caches | tee "$D/step4.json"
    echo "4d passed: eligible for maintainer review, not accepted"
  ); echo "4d_exit=$?" | tee -a "$D/provenance.txt"
  ```
  - *Expected:*
    - Check 9 is `bounded_smoke_complete`.
    - Step 4 runs about 17 min, with a 1,200 s child deadline, then waits 60 s and queries the journal; the outer `timeout` bounds everything at 24 min.
    - `step4.json` has `criteria` with every criterion `pass`, `eligible_for_maintainer_review: true` and `accepted: false`, and `4d_exit=0`.
  - *Stop if:*
    - Check 9 fails: do not run step 4 against any other report.
    - Step 4 is `refused`: fix the named refusal; a changed identity or a new boot means starting again at 4a.
    - A guard stop (`sampled_pressure_stop`, `free_or_available_stop`, `swap_counter_change`, `sampling_gap`, `timeout`) or any `fail`/`unavailable` criterion: record it as not eligible. No threshold, guard or flag is changed, and nothing is retried silently.
- *4e. Keep the evidence (always, even after a failure):*
  ```bash
  for f in check9 step4; do
    [ -s "$D/$f.json" ] && python3 -c 'import json,os,sys; print(os.path.dirname(json.load(open(sys.argv[1]))["result_file"]))' "$D/$f.json" || true
  done | while read -r dir; do cp -a "$dir" "$D/"; done; ls -l "$D"
  sudo systemctl start display-manager   # only if the desktop was active before 4c
  ```
- *4f. Rehash before any acceptance commit (after the run; not part of the measurement):*
  ```bash
  ( set -euo pipefail
    CLIP_SHA="$(grep -F 'room_static_60s_2026-09-29.mp4' docs/LOCAL_NOTES.md | grep -oE '[0-9a-f]{64}' || true)"
    .venv/bin/python benchmarks/runner/operator_check.py --step4-identity --step4-clip "$CLIP" \
      --step4-clip-sha256 "$CLIP_SHA" | tee "$D/identity-rehash.json"
    NEW="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result_file"])' "$D/identity-rehash.json")"
    python3 -c 'import json,sys; a,b=(json.load(open(p))["step4_identity"]["files"] for p in sys.argv[1:3]); c=sorted(k for k in set(a)|set(b) if (a.get(k) or {}).get("sha256")!=(b.get(k) or {}).get("sha256")); print("rehash equal" if not c else "CHANGED: %s" % c); sys.exit(1 if c else 0)' \
      "$D/identity-result.json" "$NEW"
  )
  ```
  Expected: `rehash equal`. Anything else blocks the acceptance commit.
- *Artifacts:* everything is under `$D`: `provenance.txt`, `identity.json` (a summary), `identity-result.json` (local only: it holds the clip's hash), `check9.json`, `step4.json`, and the copied `sentinel-operator-*` directories with `guard.jsonl` and the `demo-profile-*` run. Return `provenance.txt`, `check9.json` and `step4.json`.
- *Afterwards (D46, D47):* a pass makes the run eligible for the maintainer's review, nothing more. A profile becomes accepted only through a separate, maintainer-approved commit that adds a `ResourceProfile` with the D47 fields copied from this run.

**5. First end-to-end alert and status page (about 15 min attended; GPU; Telegram).**
- *Prerequisites:*
  - Steps 1–3 passed.
  - Headless, with v1 and other llama-servers stopped.
  - MemFree ≥ 1.5 GB (≥ 3.0 GB with 5b).
  - The maintainer's bot token and chat ID.
  - **5b only** after step 4 passed and the maintainer admits scene analysis for D-1.
  - VS Code may stay open: this is not a measurement.
- *Configuration, outside the repository:*
  ```bash
  mkdir -p -m 700 ~/sentinel-config ~/sentinel-data
  printf '%s\n' 'config_version: 1' 'camera: {id: cam-1}' \
    'zones: [{zone_id: whole-view, rule: restricted, polygon: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], min_duration_s: 1.0, severity: warning}]' \
    'notifications: {channels: [telegram]}' > ~/sentinel-config/demo.yaml
  PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli config validate ~/sentinel-config/demo.yaml | tee "$D/validate.txt"
  read -rs SENTINEL_TELEGRAM_BOT_TOKEN; export SENTINEL_TELEGRAM_BOT_TOKEN
  read -rs SENTINEL_TELEGRAM_CHAT_ID; export SENTINEL_TELEGRAM_CHAT_ID
  ```
- *5a, the core path (scene analysis off, the default):*
  ```bash
  LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli \
    run ~/sentinel-config/demo.yaml --data-dir ~/sentinel-data/demo --engine ~/yolov8n.engine \
    2>"$D/run.err" | tee "$D/run.jsonl"; echo "exit=$?" | tee -a "$D/provenance.txt"
  ```
  While it runs, in a second terminal on the Jetson:
  ```bash
  ss -ltn | grep -E ':(18090|18081)\b' | tee "$D/listen.txt"   # expect 127.0.0.1:18090 only (and 127.0.0.1:18081 with 5b)
  curl -s http://127.0.0.1:18090/status.json > "$D/status-idle.json"
  ```
  On the laptop, run `ssh -N -L 18090:127.0.0.1:18090 villain8001@<jetson-address>` and open `http://127.0.0.1:18090/`.

  The sequence:
  1. One minute with nobody in view. The page shows only expected reasons (or "All components working"), occupancy `empty`, about 15 captured fps and processed fps close to it.
  2. Walk into view and stay at least 3 s. One incident opens, and one Telegram message arrives carrying its incident ID. Under alert delivery, the row moves queued → attempted/in flight → delivered, with delivered 1. Run `curl -s http://127.0.0.1:18090/status.json > "$D/status-alert.json"`.
  3. *(optional)* Unplug the camera's power for about 15 s. The page shows `capture waiting` and video `stale`/`offline` as degraded. After the restore it shows reconnects 1 and fresh video.
  4. Ctrl-C. The last line is `"run": "stopped"` with `all_stopped` true and `database_closed` true, and `exit=0`.

  Then:
  ```bash
  U="${SENTINEL_RTSP_URL#*://}"; U="${U%%@*}"
  for f in "$D"/run.err "$D"/run.jsonl "$D"/status-*.json; do printf '%s userinfo=%s token=%s\n' "$(basename "$f")" \
    "$(grep -cF -- "$U" "$f")" "$(grep -cF -- "$SENTINEL_TELEGRAM_BOT_TOKEN" "$f")"; done | tee -a "$D/provenance.txt"; unset U
  unset SENTINEL_TELEGRAM_BOT_TOKEN SENTINEL_TELEGRAM_CHAT_ID
  ```
- *5b, with scene analysis:* **only after the maintainer-approved acceptance commit (D46) is checked out.** Before it, `--scene` is refused with `scene_not_admitted: …`; D33's provisional profile is refused too. Set `PROFILE` to the profile ID that commit added, append the manifest line below to `demo.yaml`, validate it again, and add `--scene` to the run command. The defaults are `~/llama.cpp/build/bin/llama-server` and the Q4_0 model with the Q8_0 projector, which must be the profiled files: same name, size and modification time. `--cache-ram 0` and `--host 127.0.0.1` are fixed in code.
  ```bash
  PROFILE="${PROFILE:?set PROFILE to the accepted profile ID from the acceptance commit}"
  printf '%s\n' 'adapters: [{adapter_id: llama-lfm2-vl-scene, contract_version: 1, implementation_revision: "1", enabled: true, input_kinds: [frame], output_kinds: [scene.report], model_revision: lfm2-vl-1.6b-q4_0, resource_profile_id: '"$PROFILE"', timeout_ms: 8000}]' >> ~/sentinel-config/demo.yaml
  PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli config validate ~/sentinel-config/demo.yaml | tee -a "$D/validate.txt"   # expect: adapter llama-lfm2-vl-scene: enabled (enabled)
  ```
  Expected: the startup line shows `scene_server` ready, `layers` 17/17 and `vision_on_gpu` true. The page shows scene analysis available, with reports about every 4 s. `listen.txt` shows `127.0.0.1:18081`.
- *Stop if:*
  - Startup is refused: the label is on `run.err`'s last line.
  - The detector is unavailable at startup (step 3 should have caught this).
  - Any listener is on an address other than 127.0.0.1: press Ctrl-C at once and report.
  - A delivery row is `failed`: record its `last_error` from the page (redacted) and check the bot and chat.
  - A row stays `attempted` for over 2 min.
  - With 5b, `scene_server` failed: record the label. There is no CPU fallback, and scene analysis stays unavailable.
  - Any userinfo or token count is above 0: share nothing but the counts.
- *Artifacts:* `$D/run.jsonl`, `$D/status-idle.json`, `$D/status-alert.json`, `$D/listen.txt`, `$D/validate.txt` and `$D/provenance.txt`; `$D/run.err` stays local. The database `~/sentinel-data/demo/sentinel.db` (incidents, outbox, delivery attempts; no images) stays on the device.

### Other pending hardware checks (outside this checklist)

```bash
# 2. Rest of check 2: frame rate, bitrate and keyframe spacing of both profiles, 60 s of
#    video packets each (no decode). The main stream's codec, size and rate are recorded.
#    Needs SENTINEL_RTSP_URL (substream) exported; prints variable names, never URLs.
MAIN_URL="${SENTINEL_RTSP_URL/subtype=1/subtype=0}"
for URL_VAR in MAIN_URL SENTINEL_RTSP_URL; do
  echo "== $URL_VAR"
  ffprobe -v error -rtsp_transport tcp -select_streams v:0 -read_intervals %+60 \
    -show_entries packet=pts_time,size,flags -of csv=p=0 "${!URL_VAR}" 2>/dev/null \
  | awk -F, '{n++; b+=$2; if(n==1)t0=$1; t1=$1; if($3~/K/){k++; if(kp!="")g=g" "sprintf("%.2f",$1-kp); kp=$1}}
      END {d=t1-t0; printf "packets=%d span_s=%.2f fps=%.2f kbit_s=%.0f keyframes=%d gop_s=%s\n", n, d, (n-1)/d, b*8/d/1000, k, g}'
done
unset MAIN_URL
#    Also note from the camera's web/app settings: model, firmware, and the frame-rate
#    choices offered for each profile (after the demo).

# 7. DEFERRED with V2-05 proper (D24); not needed for the Oct 20 demo.
gst-inspect-1.0 nvv4l2decoder | sed -n '1,25p'
gst-inspect-1.0 nvvidconv | sed -n '1,25p'
#    Hardware-decode smoke test on the substream, 30 s (in a second terminal: tegrastats --interval 1000).
timeout -s INT 30 gst-launch-1.0 -e rtspsrc location="$SENTINEL_RTSP_URL" protocols=tcp latency=200 \
  ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw,format=BGRx' \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false -v 2>&1 \
  | sed -E 's#rtsp://[^[:space:]]+#rtsp://<redacted>#g' | grep -E 'last-message|ERROR|WARN' | tail -4
```

Still open from the guide's V2-01 acceptance, after the commands above: the JetPack release that corresponds to L4T 36.4.7 (NVIDIA release notes); PTZ capability response (C9, may stay deferred); B0 run with unique-frame throughput, stage timings, per-process/thread CPU, `MemTotal − MemAvailable`, tegrastats, PSS, clocks, temperature, headless versus desktop (needs the V2-11 trace contract first); B0/B1 workload manifest, provisional CPU budget and re-estimated effort; a known-good backup and restore point before any runtime change (also required by D27 before the post-demo package restore).
