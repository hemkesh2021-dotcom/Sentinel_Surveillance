# Sentinel v2 — implementation status

Last updated 2026-10-03, session 5: check 8 (U17 demo profile) recorded as the maintainer's measurement and admitted as provisional with its exceedance (D33); the demo face cadence set to 1 Hz (D34); Claude's follow-up diagnostics of the detector, face and scene results; open issues U19–U21. Session 4: session 3's record completed (V2-01 device checks run by Claude on 2026-09-29, the maintainer's decisions D26–D29, the U17 demo-profile script whose run is PENDING as check 8), answers on check 3b and GPU memory (U18, check 9 PENDING), and V2-13, V2-14 and V2-15 in demo form (D30–D32). Requirements come from the v2 beta implementation guide (V2-01…V2-56 backlog), and corrections and regression cases from the 23 September audit review. Both documents are local-only (see D13).

## Position

| | |
|---|---|
| Branch | `v2-beta`, created from `master`. **Pushed by the maintainer**; `origin/v2-beta` was at `32985c2` when session 3 started. Session 3 commits are local until the maintainer pushes them. |
| CI | Maintainer report, 2026-09-29: GitHub Actions passed at `32985c2`: "v2 portable checks" (run #2) and "Dashboard checks" (run #5). Earlier runs at `6578ded` also passed. |
| Base commit | `2b2d639621e8c043cc58a126f47b1b8ab6c22135`, the commit the audit verified, confirmed as HEAD before starting |
| Session 1 commits | `ec6698d` CLAUDE.md · `b52920f` package skeleton and portable tests · `e9f959d` clock · `bc42248` frame identity · `3b47f5f` evidence/track applicability · `ee50275` config and CLI · `6a9e71d` CI workflow · `6578ded` status record |
| Session 2 commits | `4587021` PTS tolerance at ingest · `812b42c` V2-01 records · `e81db81` replay timelines · `86889f8` scene lane (R3) · `432dc69` live state and freshness (R2) · `66937ef` face association and identity (R1) · `48d6188` status record · `9593b64` adapter manifests (V2-49) · `32985c2` maintainer decisions D22–D24 |
| Session 3 work (2026-09-29) | Interrupted by a usage limit before anything was committed; committed in session 4 as the first commit below |
| Session 4 commits (2026-10-03) | See the session 4 slice log |
| Session 5 (2026-10-03) | See the session 5 log |
| Working tree | Clean after the session 4 commits, apart from ignored environments/build output and local-only files excluded through `.git/info/exclude` |
| Local-only files | The v2 guide, the audit review and `docs/LOCAL_NOTES.md` (device-specific notes). A fresh clone does not contain them, although CLAUDE.md names the first two. |
| Selected package | V2-28 incident side (session 5). Then the week-2 device adapters and D-1. V2-01's checks 1 (`nvpmodel` only), 3, 5 and 6 were run by Claude in session 3; check 8 was run by the maintainer on 2026-10-03 (D33); check 9 (U18) is PENDING. |
| Other branches | `origin/Yogeshvar425-patch-1` (teammate) is **not merged**: a single commit `6755796` that adds @Yogeshvar425 to `.github/CODEOWNERS` (merge base `c66ebde`). `origin/codex/github-audit-fixes-2026-09-19` is already in `master` via PR #4. |
| Effort | Per-package estimates are in the package table (given to the maintainer on 2026-09-29). The re-estimate of optimization effort still waits for V2-01's B0 run. |
| v1 on this device | **Not running** (maintainer, 2026-09-29). Checks must not assume v1 processes exist. |
| Waiting on the maintainer | **Check 9** (U18: GPU allocations beyond MemFree, with and without unified memory; about 5 minutes; needed before D-1 fixes its memory precheck). Decisions on **U20** (constrained scene output) and **U21** (re-run check 8 with attribution), and confirmation of the replay clip's content (session 5 log). Then the rest of check 2 and the V2-04 clean-laptop (macOS) run. |

## Next concrete task

1. **V2-28, demo form (incident side):** late scene evidence and enrichment annotate only their own incident in the store (append evidence via `IncidentService`, never a new incident or a status change). Small; the scene side is done.
2. **Week 2 device adapters:** record check 8's provisional demo profiles (D33) in the adapter registry's known profiles; settle U18 once check 9 is back. Then capture through `~/onvif_env`'s OpenCV/FFmpeg (D24); the legacy engine plus ByteTrack; interim face adapter at 1 Hz (D34) and llama-server adapter (constrained output if U20 is accepted); and `sentinel run` (D-1) with the D27 GPU guard and the U18 memory precheck. `sentinel run` hands every zone observation and hazard candidate to `IncidentService.record()` and acknowledges it only after that returns (D31). It runs `OutboxWorker` in its own thread with `TelegramNotifier.from_environment(timeout_s=notifications.request_timeout_s)` only if `telegram` is listed (D32).
3. **V2-15 device check (maintainer):** one real Telegram send from the device with the maintainer's bot, after D-1 wiring.
4. **V2-13, rest (after the demo):** directed line crossing and a decision on hysteresis (D30).

## Oct 20 demo milestone: plan and deviations from the guide order

Target (maintainer, 2026-09-29): a demoable end-to-end path on this Jetson by 2026-10-20: camera → detection → tracking → identity → scene/VLM → alert/outbox → dashboard. That is three weeks. In the guide's order it spans C2–C7 (about 12 cycles' worth of dependencies). It is feasible only as a **demo profile**: the v2 core logic (contracts, freshness, identity, scene lane, rules, durable incidents/outbox) plus interim adapters around the existing models. It is not the optimized B2 profile and establishes no gate.

| Week | Work | Status |
|---|---|---|
| 1 (to Oct 6) | V2-49; V2-13 zone rule; V2-14 SQLite incidents/outbox; V2-15 leased outbox + Telegram (mocked). All portable. | V2-49 done. Demo form done: V2-13 (restricted + dwell; crossing deferred), V2-14, V2-15 (mocked); these count as partial (D23). |
| 2 (to Oct 13) | Device adapters (D24): capture from the substream (profile A, D22) with `~/onvif_env`'s OpenCV/FFmpeg software decode, video only, stamped by FrameStamper; detector + ByteTrack via the existing `yolov8n.engine` as the *legacy parity adapter*; interim face adapter (existing DeepFace/Facenet512 on CPU) feeding v2 association; llama-server scene adapter; `sentinel run` loop around `EdgeCore` | Checks 3 and 5 done; U13 settled (D27); U17 settled (D28, D33: check 8 run 2026-10-03); U18 open (check 9); U20, U21 need decisions |
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
| V2-01 | Hardware and v1 timing/memory baseline | in progress | Jetson | yes (checks 3, 5, 8 done; 9 pending) | 1.5 | 3 | — |
| V2-02 | Config, frame/evidence contracts, fake clock | done | Portable | yes | 0 | 0 | — |
| V2-03 | Replay fixtures, first identity/empty-scene fixes | done | Portable | yes | 0 | 0 | — |
| V2-04 | Dev setup and CI skeleton | in progress | Portable | no | 0.5 | 0 | — |
| V2-05 | Relay ownership and hardware decode spike | not started | Jetson | yes: **demo form, full acceptance pending** | 2 | 2.5 | 01, 02 |
| V2-06 | Browser/codec/timestamp spike | not started | Jetson | no | 2 | 3 | 05 |
| V2-07 | Dataset consent, labels, split manifest | not started | Jetson (recording) | no | 1 | 3 | 03 |
| V2-08 | Gate B record, recoverable device baseline | not started | Jetson | no | 1 | 4 | 01, 05, 06 |
| V2-09 | TensorRT adapter and fixed buffers | not started | Jetson | yes: **demo form, full acceptance pending** | 2.5 | 3 | 05, 08 |
| V2-10 | Tracker and coordinate parity | not started | Jetson | yes: **demo form, full acceptance pending** | 1.5 | 1.5 | 07, 09 |
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
| V2-26 | Small VLM vs existing model comparison | not started | Jetson | yes: **demo form, full acceptance pending** | 2 | 5 | 09, 24 |
| V2-27 | Enrollment/revoke screens | not started | Portable + device check | no | 1 | 0.5 | 20, 25 |
| V2-28 | Evidence enrichment isolation | partial | Portable | yes: **demo form, full acceptance pending** | 0.5 | 0 | 14, 26 |
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
| D-1 | Demo runtime loop `sentinel run` (not in backlog) | not started | Jetson | yes (demo only) | 2 | 2.5 | demo parts of 05, 09, 10, 13–15, 20, 25, 26 |
| D-2 | Loopback read-only status page (not in backlog) | not started | Portable + device check | yes (demo only) | 1 | 0.5 | 14, 15 |

| Totals | Claude h | Maintainer Jetson h |
|---|---|---|
| Remaining backlog | 68.5 | 94.25 |
| Demo-only rows D-1, D-2 | 3.0 | 3.0 |
| **Total** | **71.5** | **97.25** |
| Oct 20 demo scope only | 12.0 | 9.5 |

Notes on partial and in-progress rows:

- **In progress:**
  - V2-01: records and checks 1 (`nvpmodel`), 3, 5 and 6 are done (session 3); check 8 and the headless idle baseline (0.970 GB) by the maintainer on 2026-10-03. Checks 2 (rest) and 9, the B0 run, trace contract, CPU budget and re-estimate are pending.
  - V2-04: GitHub Actions passed at `32985c2` (maintainer report). The clean-laptop (macOS) run is not done.
- **Done:** V2-03 with synthetic replays only; real-clip replay needs V2-07. V2-49's registry is empty until real adapters land, and its unknown-profile rule makes every model adapter unavailable until check 8's provisional profiles are recorded (D28).
- **Partial:**
  - V2-13: restricted-zone and dwell rules in demo form (session 4). Directed line crossing and hysteresis are not done; revalidation after V2-10/V2-11 and calibration of persistence and gap values on labelled replays (V2-07) are pending.
  - V2-14: demo form done (session 4): schema v1 and migrations, the one-transaction record path, correlation, lifecycle and outbox rows. Not done: the guide's other tables (cameras, zones, policies, users, sessions, identities, audit events, clip manifests), WAL checkpoint monitoring, quotas, and revalidation after V2-11.
  - V2-15: demo form done against mocks (session 4). Not done: a real Telegram send from the device (the "device check", which needs the maintainer's bot and chat), the authenticated retry action in an API (only `retry_dead()` exists), outbox quotas, wiring into `sentinel run` (D-1), and revalidation after V2-11.
  - V2-16: only the scene-hazard correlation.
  - V2-20: validated in-memory enrollment only.
  - V2-25: the association and identity core is done; the adapter, alignment, vectorized matching and report are not.
  - V2-28: the scene side is done; the incident side waits for V2-14.
  - V2-54: one job at a time, timeout and cancel exist in the scene lane; unload and memory are not done. **Open issue U19:** +0.983 GB remained after every check 8 process had stopped.

## Session 5 log (2026-10-03)

### Check 8: demo resource profile (maintainer's measurement)

Run `demo-profile-20261003T085010Z`, run by the maintainer on 2026-10-03 from 08:50 to 09:04 UTC. Headless (display manager inactive, no desktop processes, no dev tools), after `drop_caches`, with MemFree at 6.83 GB at the baseline. Repository at `d85eb1e` with no tracked changes; GPU guard (D27) passed: 17/17 layers and the vision encoder on CUDA0, only L4T's libcuda mapped, `cuInit` 0 in the workload. Input: the replay clip now named `room_static_60s_2026-09-29.mp4` (formerly `one_person_2026-09-29_1606.mp4`; Claude re-checked that its SHA-256 is unchanged), looped 11 times. Output stays in `~/sentinel-runs/<run id>/`. One cold load and one combined run: a provisional-demo profile (D28, D33), not a benchmark or gate result. Memory is whole-device `MemTotal − MemAvailable` in decimal bytes.

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

**Run input.** The maintainer's note says the clip shows only the room with no person, and it was renamed for that reason. Claude's check below finds people in every frame. The maintainer is asked to confirm. Either way the clip is unlabelled, so the detection and face counts are not ground truth. The memory and latency numbers stand.

### Claude's follow-up diagnostics (session 5; observations, not measurements)

Run with VS Code and Claude Code open, on the same boot. Model output, frames and annotations were written only under the run directory (modes 0700/0600), never into the repository. llama-server ran twice, bound to 127.0.0.1, and was stopped each time (no process left, port free). Memory and latency from these runs are not recorded as measurements.

1. **Detector and face on the clip** (`annotated/`). Same engine and v1's `track()` arguments on all 901 frames; DeepFace/YuNet on every 10th frame.
   - A person was detected on 901 of 901 frames (one box on 626, two on 271, three on 4), with confidences from 0.408 to 0.951 (median 0.83). It was the same 901/901 without the tracker.
   - Faces (confidence > 0) appeared on 12 of 91 sampled frames (13 %), consistent with the run's 82 of 671 runs (12 %).
   - Claude viewed three annotated frames, a 16-frame contact sheet and an enlarged crop. A person is in the foreground for most of the clip, and a second, small, distant person in another room is visible in the far doorway for the rest. The camera view changes once.
   - One false positive was found: a chair back at 0.47 (frame 534), next to two real people.
   - **Conclusion:** the 9,000/9,000 frames reflect mostly true detections, and the clip is not an empty room. If the maintainer agrees, the name `room_static_60s` is misleading.
2. **Scene reports** (`vlm_repro/`). The run stored no model text (by design: `demo_workload.py` records counts only). Claude reproduced the requests with the same model files, flags, preload, image shape, prompt and temperature on 24 clip frames (every 38th): **1 of 24 valid** (the run had 5 of 150).
   - All 24 were well-formed JSON objects with the right types; 7 were inside a ```` ```json ```` fence, which the parser accepts. The cause is **not** syntax strictness or the types in the schema.
   - 16 of 24 had a `summary` of 168–364 characters (limit 160).
   - 7 of 24 stopped at `max_tokens` 200 (`finish_reason: length`) before reaching `summary`, after five long observations of up to 95 characters (limit 80). In the run, 14 of 179 requests reached 200 tokens.
   - **Cause: the prompt asks for character limits that the 1.6B model does not follow**, and the schema's bounds are only checked after generation. The schema's limits are reasonable.
   - **Proposed fix, tested:** the same prompt plus the SceneReport JSON Schema as llama-server's `response_format`. llama.cpp compiles it to a grammar; build b8932 supports `maxLength`, `maxItems`, integer bounds, enums and `additionalProperties: false`. Result: **24 of 24 valid**; 121–183 completion tokens; p50 2.9 s, max 3.7 s. Side effect: 22 of 24 summaries stop at exactly 160 characters, mid-word, because the grammar cuts them off rather than the model writing less. See U20.
   - **Person counts.** What the run's 5 valid reports said is unknown, because the run did not record report content. In the reproduction, the one valid unconstrained report said 1. With the schema, reports said 1 on 17 frames, 2 on 3 and 0 on 4. All four zero frames (418, 798, 836, 874) contain a person cut off at the frame edge and/or the small distant person. These are VLM misses, not empty frames. Occupancy never comes from the VLM (D18).
3. **Memory ramp (from the run's `memory.csv`).**
   - Used memory rose by about 0.12 GB per minute, from 4.24 GB at the start of warm-up to 5.30 GB in minute 9 of the 12-minute window. It flattened at 5.32–5.35 GB only when MemFree reached its floor (0.175 GB) and pages began to be swapped out.
   - Over the same period, workload RSS (2.00 → 1.79 GB) and llama-server PSS (about 1.66–1.69 GB) stayed flat.
   - When the workload exited, used memory fell by about 3.2 GB (5.25 → 2.02 GB), far more than its cold loads (+1.24 GB). The growth was therefore most likely GPU-side memory of the workload process: on Jetson, CUDA allocations by torch, TensorRT and Ultralytics do not appear in RSS. This is not attributed yet.
   - **The steady median and p95 describe a ramp, not a plateau**, and the 5.35 GB peak may be where free memory ran out rather than what the workload needs (U21).

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
| `benchmarks/runner/gpu_alloc_probe.py` (session 4) | Check 9 (U18): fixed-size `cudaMalloc` or `cudaMallocManaged` chunks, each written with `cudaMemset`, until failure or a cap; prints one summary line. Refuses unless L4T's libcuda is mapped and CUDA initialises. |

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

## Not run or not established (sessions 1–5)

- GitHub Actions ran only on the pushed commits up to `32985c2` (maintainer report: both workflows passed). Session 3 and 4 commits are unpushed and have not run in CI.
- There was no run on a clean laptop (x86-64 or macOS); all checks ran on this Jetson's aarch64 userspace.
- No lint or type check is configured yet (guide ch. 21 lists both); deferred to keep the slice small.
- There were no camera, decoder, GPU, TensorRT, memory, throughput or latency measurements. This slice establishes no hardware, Gate B or beta-readiness result.
- No v2 code runs on the camera or GPU yet. `EdgeCore`, the incident store and the outbox are exercised only by synthetic replays, mocks and FakeClock smoke runs in `~/onvif_env` (CPU, no camera, no network). No real Telegram message has been sent by v2. The other ch. 18 CLI commands, including `sentinel replay`, were intentionally not added yet.
- The replay regressions use synthetic timelines, not the maintainer's clips. They record the behaviour of the v2 components and of a documented reference model of v1 (D12), not of the running v1 process.
- The mutation sweeps are one-off checks whose scripts are not committed.
- Checks 3a/3b were run by Claude in session 3 (inventory below). Check 8 was run by the maintainer (session 5 log): one run on an unlabelled clip, a provisional profile and not a benchmark; its steady phase was still ramping (U21). Check 9 is only syntax-checked. Claude's session 5 diagnostics (detector, face and VLM reproductions) are observations with dev tools running, not measurements.

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
- **U18. Free memory at GPU load** (new, session 3). On this device, GPU allocations failed whenever they exceeded **MemFree**, although MemAvailable was over 4 GB (inventory below). llama.cpp's own fit check uses MemAvailable, so it does not catch this. How does `sentinel run` (D-1) make GPU loads reliable? Options: (a) evict the model files' page cache with `posix_fadvise(DONTNEED)` (no root) and refuse to load below a MemFree threshold, as `demo_profile.py` does (3.0 GB default); (b) drop caches system-wide before loading (root; a system action); (c) load GPU components first, right after boot. Also open: whether allocations made after start-up (llama.cpp compute buffers, larger images) can fail the same way once the page cache refills. **Check 9** settles whether this depends on unified memory (`cudaMallocManaged`) or also affects `cudaMalloc`; decide after it.
- **U19. Memory left after unload (V2-54, open issue).** After check 8's workload exited and llama-server stopped, used memory was 1.889 GB: **+0.983 GB above the 0.906 GB baseline**, while Cached stayed +2.2 GB above it (0.41 → 2.61 GB). The workload's exit took used memory to 0.612 GB *below* its level with llama-server alone (2.633 → 2.021 GB). Stopping llama-server then freed only 0.13 GB, although its load had added 1.73 GB. Candidate explanations, none tested: page cache that `MemAvailable` does not credit, shared memory, NvMap/CMA pages kept by the driver, or unreclaimable slab. The sampler records none of `Shmem`, `Unevictable`, `Mlocked`, `SUnreclaim`, `KReclaimable` or `CmaFree`. V2-54's unload acceptance cannot pass until this is explained. Next: add those fields to the sampler; after the run, sample again after 60 s, drop the page cache only (`echo 1`), and sample once more; run a second load/unload cycle to see whether the residue accumulates.
- **U20. Constrained scene output (proposal; maintainer decision needed).** Evidence is in the session 5 log (1/24 valid unconstrained, 24/24 with the schema).
  - Proposal: the scene adapter sends the SceneReport JSON Schema, generated from the model class so the two cannot drift, as llama-server's `response_format`. `parse_scene_report` stays the unchanged authority: the grammar helps generation and is never trusted.
  - Ask for brevity in the prompt so that the grammar does not cut summaries mid-word: "summary: one short sentence" and "up to 3 short observations". The schema itself is unchanged (at most 5 observations of at most 80 characters).
  - Treat `finish_reason: length` as its own error ("truncated"). Keep `max_tokens` 200 (at most 183 were used with the schema).
  - The schema feature is tied to llama.cpp build b8932; record the build in the adapter manifest.
  - Not tested yet: the shorter prompt wording. Rejected alternative: loosening the limits or repairing output (ch. 13 forbids repair).
- **U21. Memory ramp in check 8's steady phase.** About +0.12 GB per minute until MemFree ran out; most likely GPU memory in the workload process (session 5 log). Before D-1 relies on the profile, re-run check 8 for at least 30 minutes of steady phase, with attribution (torch `memory_reserved`/`memory_allocated` from the workload; the extra meminfo fields of U19), the D34 cadence and, if U20 is accepted, the schema. Needs script changes first. Not started.

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

**Replay clips.** The maintainer's private recordings stay outside the repository; their location, names and SHA-256 hashes are in the local notes. Probed read-only with ffprobe: `room_static_60s` (renamed 2026-10-03 from `one_person`; same SHA-256; it contains people, see the session 5 log) is 60.0 s, 901 frames, H.264 Main 640×480 at 54060/3601 fps with an AAC-LC 16 kHz mono track; `empty_room` is 455.6 s, 6,832 frames, H.264 Main 640×480 at 204960/13667 (≈15.00) fps, video only. Consent and split manifests belong to V2-07; tests that use them will read a directory from `SENTINEL_REPLAY_CLIPS_DIR` and skip when it is unset.

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
2. **Do allocations beyond MemFree fail with and without it?** **With it: yes**, as far as session 3 shows. Both loads that started with MemFree at or below about 1.7 GB failed, a third failed with MemFree unrecorded, and the load at 3.08 GB MemFree succeeded. **Without it: not tested.** No session 3 run omitted the variable, so whether plain `cudaMalloc` behaves the same is unknown. Check 9 (PENDING) tests both APIs directly with `benchmarks/runner/gpu_alloc_probe.py`, writing every chunk so it is backed, and llama-server without the variable, all with the page cache full and MemFree low. Until check 9 runs, D-1 assumes both fail (U18).
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

Checks 1 (`nvpmodel` only), 3, 5 and 6 were run by Claude in session 3 (inventory above), check 4 was merged into 3b, and check 8 was run by the maintainer on 2026-10-03 (session 5 log). v1 is not running on this device, so no check assumes v1 processes. Nothing below prints the camera URL or any other secret: commands that open the stream redact any `rtsp://…` with `sed` or discard stderr. Paste outputs back; they will be recorded as maintainer measurements. **Demo priority: check 9.** Then the rest of check 2. Check 7 and the `gst-inspect` lines wait for V2-05 proper.

```bash
# 8. DONE by the maintainer, 2026-10-03 (D33). Kept for the U21 re-run, which needs the script
#    changes listed under U21 first. Demo resource profile, about 15 minutes.
#    Headless (D29): best right after a reboot, from a plain SSH session, with VS Code and
#    Claude Code closed and the display manager and any remote-desktop session stopped
#    (the remote-desktop command is in docs/LOCAL_NOTES.md). The script checks all of this
#    and refuses to start otherwise; it also needs MemFree >= 3.0 GB (U18).
sudo systemctl stop display-manager
cd ~/sentinel-surveillance
/usr/bin/python3 benchmarks/runner/demo_profile.py --clip "$SENTINEL_REPLAY_CLIPS_DIR"/room_static_60s_2026-09-29.mp4
#    Paste back the summary it prints (also saved as ~/sentinel-runs/<run id>/summary.txt).
sudo systemctl start display-manager   # afterwards, if you want the desktop back

# 9. NEEDED FOR D-1 (U18). Do GPU allocations beyond MemFree fail with cudaMalloc, with
#    cudaMallocManaged, or both? About 5 minutes; any host state; v1 and llama-server stopped.
#    fill() reads the model files into the page cache, so MemFree is low and MemAvailable high.
L4T=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1
fill() { cat ~/models/lfm2-vl/*.gguf >/dev/null; grep -E '^(MemFree|MemAvailable):' /proc/meminfo | tr -s ' ' | paste -sd' '; }
# 9a. The two allocation APIs directly (128 MiB chunks, each written; stops at the first failure)
for API in device managed; do
  fill
  LD_PRELOAD=$L4T /usr/bin/python3 ~/sentinel-surveillance/benchmarks/runner/gpu_alloc_probe.py --api "$API"
done
# 9b. llama-server with v1's flags, without and with GGML_CUDA_ENABLE_UNIFIED_MEMORY
for UMA in "" 1; do
  echo "== llama-server, GGML_CUDA_ENABLE_UNIFIED_MEMORY=${UMA:-<unset>}"
  fill
  LOG=$(mktemp)
  env LD_PRELOAD=$L4T ${UMA:+GGML_CUDA_ENABLE_UNIFIED_MEMORY=$UMA} ~/llama.cpp/build/bin/llama-server \
    --model ~/models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf --mmproj ~/models/lfm2-vl/mmproj-LFM2-VL-1.6B-Q8_0.gguf \
    --host 127.0.0.1 --port 18081 --n-gpu-layers 999 --ctx-size 2048 --parallel 1 >"$LOG" 2>&1 &
  PID=$!; READY=no
  for i in $(seq 1 45); do
    curl -sf -o /dev/null http://127.0.0.1:18081/health && { READY=yes; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  echo "ready: $READY"
  grep -E 'offloaded|CLIP using|NvMap|cudaMalloc failed|failed to allocate' "$LOG" | sort | uniq -c | head -8
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -f "$LOG"
done
pgrep -a llama-server || echo "no llama-server left"
# 9c. The original v1 OOM: kernel OOM kills and NvMap failures in the retained journal
#     (use sudo if it prints nothing and you are not in the adm/systemd-journal group).
journalctl --list-boots --no-pager 2>/dev/null | head -3
journalctl _TRANSPORT=kernel --no-pager -o short-iso 2>/dev/null \
  | grep -iE 'out of memory|oom-kill|killed process|NvMapMemAlloc' | tail -20

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
