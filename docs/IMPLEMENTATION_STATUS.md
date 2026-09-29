# Sentinel v2 — implementation status

Last updated 2026-09-29, end of session 2 (V2-01 records, V2-03 regressions R1–R3 and the portable `EdgeCore`). Requirements come from the v2 beta implementation guide (V2-01…V2-56 backlog), and corrections and regression cases from the 23 September audit review. Both documents are local-only (see D13).

## Position

| | |
|---|---|
| Branch | `v2-beta`, created from `master`; **not pushed** |
| Base commit | `2b2d639621e8c043cc58a126f47b1b8ab6c22135`, the commit the audit verified, confirmed as HEAD before starting |
| Session 1 commits | `ec6698d` CLAUDE.md · `b52920f` package skeleton and portable tests · `e9f959d` clock · `bc42248` frame identity · `3b47f5f` evidence/track applicability · `ee50275` config and CLI · `6a9e71d` CI workflow · `6578ded` status record |
| Session 2 commits | `4587021` PTS tolerance at ingest · `812b42c` V2-01 records · `e81db81` replay timelines · `86889f8` scene lane (R3) · `432dc69` live state and freshness (R2) · `66937ef` face association and identity (R1) · then the commit that updates this record |
| Working tree | Clean apart from ignored environments/build output and local-only files excluded through `.git/info/exclude` |
| Local-only files | The v2 guide, the audit review and `docs/LOCAL_NOTES.md` (device-specific notes). A fresh clone does not contain them, although CLAUDE.md names the first two. |
| Selected package | C1 is finished except V2-01 (hardware checks PENDING), V2-04's off-device checks and **V2-49** (next). |
| Other branches | `origin/Yogeshvar425-patch-1` (teammate) is **not merged**: a single commit `6755796` that adds @Yogeshvar425 to `.github/CODEOWNERS` (merge base `c66ebde`). `origin/codex/github-audit-fixes-2026-09-19` is already in `master` via PR #4. |
| Effort | No team availability recorded yet (guide ch. 26 kickoff item). The C1 re-estimate is still open: V2-01's B0 run has not happened, so no optimization effort can be re-estimated. |
| Waiting on the maintainer | **U12** (perception stream profile and FPS gate), **U13** (CUDA driver library), **U14** (runtime environment for hardware adapters), and the PENDING hardware commands |

## Next concrete task

1. **V2-49** (C1): versioned extension/adapter manifest with the ch. 27 fields. Unknown major versions and duplicate IDs fail config validation, and disabled adapters import no ML framework (extend `test_portable_imports`). Wire the scene analyzer, detector and face adapters through it as `disabled` by default.
2. Then the portable part of the Oct 20 path (plan below): **V2-13** restricted-zone rule (normalized polygon, bottom-centre anchor, IANA schedule across midnight, persistence), then **V2-14** SQLite incident + evidence + outbox transaction with runtime-ID deduplication, then **V2-15** leased outbox with a stdlib Telegram adapter tested against a mock (HTTP error, `ok=false`, 429, timeout, crash after send).
3. Hardware adapters wait for U12/U14 and the PENDING checks.

## Oct 20 demo milestone: plan and deviations from the guide order

Target (maintainer, 2026-09-29): a demoable end-to-end path on this Jetson by 2026-10-20: camera → detection → tracking → identity → scene/VLM → alert/outbox → dashboard. That is three weeks. In the guide's order it spans C2–C7 (about 12 cycles' worth of dependencies). It is feasible only as a **demo profile**: the v2 core logic (contracts, freshness, identity, scene lane, rules, durable incidents/outbox) plus interim adapters around the existing models. It is not the optimized B2 profile and establishes no gate.

| Week | Work | Status |
|---|---|---|
| 1 (to Oct 6) | V2-49; V2-13 zone rule; V2-14 SQLite incidents/outbox; V2-15 leased outbox + Telegram (mocked). All portable. | V2-49 next |
| 2 (to Oct 13) | Device adapters, after U12/U14: capture (substream, video only, FrameStamper; `gst-launch` pipe or CPU decode); detector + ByteTrack via the existing `yolov8n.engine` as the *legacy parity adapter*; interim face adapter (existing DeepFace/Facenet512 on CPU) feeding v2 association; llama-server scene adapter; `sentinel run` loop around `EdgeCore` | Blocked on decisions and checks |
| 3 (to Oct 20) | Loopback-only, read-only status page (stdlib HTTP server) showing LiveState, incidents and delivery outcomes; end-to-end rehearsal with v1 stopped; demo script including camera loss and recovery | — |

**Deviations from the guide's order (flagged; each needs the maintainer's acceptance):**

1. **Face (V2-25, C7) and VLM (V2-26, C7) come before C4–C6, via interim adapters** around the models v1 already uses: DeepFace/Facenet512 on CPU and llama-server with LFM2-VL-1.6B. Reason: the milestone requires identity and scene. The v2 association, identity and scene-lane logic is final; the adapters are labelled demo-only, uncalibrated and unbenchmarked. Enrollment will be re-created as validated non-executable data from consented photos, never by loading `face_db.pkl`.
2. **Portable C4 packages (V2-13/14/15) before the C2/C3 hardware packages.** Reason: C2/C3 are blocked on the PENDING hardware checks and U12/U14, while C4 is portable and on the milestone path.
3. **No go2rtc relay for the demo (V2-05 deferred).** The v2 runtime opens the substream itself as the only ingest, so v1 must be stopped during demo runs to avoid a second upstream session. Installing go2rtc is a new binary dependency that needs a decision.
4. **Decode may be CPU for the demo** (`~/onvif_env`'s OpenCV has only its bundled FFmpeg). The hardware path (`nvv4l2decoder` in a `gst-launch-1.0` subprocess, U14 option a) is used only if PENDING check 7 passes. CPU decode of a 640×480 at 15 fps stream is expected to be cheap, but this has not been measured.
5. **Dashboard: a loopback-only, read-only status page instead of V2-17/V2-18** (FastAPI, auth, roles, PWA). Access is over an SSH port forward. This avoids new dependencies and does not expose an unauthenticated service. FastAPI is not installed anywhere; adding it is a dependency decision.
6. **Runtime environment for the demo:** `~/onvif_env` read-only, run as `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONPATH=src ~/onvif_env/bin/python`. The portable package passes its suite with that environment's pydantic 2.12.5 (verified below). Nothing will be installed into it; if an adapter needs anything missing, a separate environment will be proposed first.

## Package status

| Package (guide ch. 20) | Status |
|---|---|
| **V2-01** Hardware and v1 timing/memory baseline. *Accept:* sanitized inventory; B0/B1 workload and trace contract; provisional CPU budget; re-estimated optimization effort | **In progress.** Camera substream facts (maintainer-measured) and the `~/onvif_env` diagnostics are recorded in the V2-01 inventory. The device checks are PENDING with exact commands. There is no B0 run, trace contract, CPU budget or re-estimate yet. |
| **V2-02** Config, frame/evidence contracts, fake clock. *Accept:* invalid config fails clearly; epoch/TTL tests pass | **Done (portable).** Session 2 extended the config with `scene`, `hazard` and `identity` sections. |
| **V2-03** Replay fixtures and first identity/empty-scene fixes. *Accept:* multi-person and zero-person regressions recorded | **Done (synthetic replays).** R1, R2 and R3 are recorded, each next to the B0 v1 snapshot. Real-clip replay needs V2-07 consent/split manifests and a decode adapter. |
| **V2-04** Development setup and CI skeleton. *Accept:* clean laptop runs non-GPU checks | **In progress.** No clean-laptop (x86-64/macOS) run and no GitHub Actions run yet (nothing pushed). |
| **V2-49** Versioned extension manifest and evidence validation (C1) | **Next.** |

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
| `src/sentinel/runtime.py` | `EdgeCore`: portable composition of all of the above; `on_frame`, `tick`, `on_scene_outcome`, `request_enrichment`, `diagnostics()` |
| `config/default.yaml` | New `scene`, `hazard` and `identity` sections, with values labelled as proposed or placeholders |
| `tests/replay/` | `test_timeline.py`, `test_r3_stale_scene_results.py` (11), `test_r2_zero_person_scenes.py` (5), `test_r1_face_identity.py` (9); harnesses, builder, fixture and the B0 v1 snapshot |
| `tests/unit/` | New `test_jobs.py`, `test_scene_report.py`, `test_health.py`, `test_scene_hazard.py`, `test_identity.py`; config tests extended |

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

**Every session 2 commit passes its own tests** (the loop from session 1, over `6578ded..HEAD`): `4587021` 48 · `812b42c` 48 · `e81db81` 60 · `86889f8` 87 · `432dc69` 100 · `66937ef` 114 passed.

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

## Not run or not established (both sessions)

- Neither GitHub workflow has run, because nothing was pushed.
- There was no run on a clean laptop (x86-64 or macOS); all checks ran on this Jetson's aarch64 userspace.
- No lint or type check is configured yet (guide ch. 21 lists both); deferred to keep the slice small.
- There were no camera, decoder, GPU, TensorRT, memory, throughput or latency measurements. This slice establishes no hardware, Gate B or beta-readiness result.
- No v2 code runs on the camera or GPU yet. `EdgeCore` is exercised only by synthetic replays and one FakeClock smoke run in `~/onvif_env` (CPU, no camera). The other ch. 18 CLI commands, including `sentinel replay`, were intentionally not added yet.
- The replay regressions use synthetic timelines, not the maintainer's clips. They record the behaviour of the v2 components and of a documented reference model of v1 (D12), not of the running v1 process.
- The mutation sweeps are one-off checks whose scripts are not committed.

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
- **U12. Perception stream profile and frame-rate gate.** The 20–25 fps stretch target exceeds the measured 15 fps substream, and the ≥15 fps gate names a 1080p input. Options A–D and a recommendation are under "Conflict" in the V2-01 inventory. **Maintainer decision needed** before V2-05 fixes the ingest profile.
- **U13. Shadowing CUDA driver library.** `libnvidia-compute-535` hides L4T's `libcuda.so.1`. Keep the `LD_PRELOAD` workaround, or remove the package or fix the loader order (a system change with v1 at stake)? Maintainer decision; nothing was changed.
- **U14. Runtime environment for hardware adapters.** No existing interpreter has both GStreamer bindings and TensorRT. Options (a)–(c) are in the V2-01 inventory; decide when V2-05 starts.
- **U15. Identity calibration.** D19's thresholds are placeholders. Calibrate with consented, session-separated identities (V2-07/V2-25) before any identity is shown as more than context.
- **U16. Face stage cadence.** `EdgeCore.on_frame(..., faces=None)` means the face stage did not run on that frame. Which frames get face analysis, and whether it runs asynchronously in a worker, is V2-25/V2-29 work. The interim demo adapter will run it on sampled frames.

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
| Main stream (`subtype=0`) | **PENDING**; command below |
| Keyframe interval | **PENDING** for both profiles; command below |

Consequences already implemented: `4587021` makes `FrameStamper` keep a missing, repeated or backwards PTS for diagnostics but stamp it `source_time_quality=none`. Identity and ages already came only from ingest (receive) time, so nothing trusts such a PTS as stream time. Regression: `test_unset_and_non_increasing_pts_at_stream_start_fall_back_to_receive_time`.

Consequences for later packages: the capture adapter (V2-05) selects only the video stream; v1's 640×480 `cv2.resize` is a no-op in size on this profile (still a copy); a 640×480 source gives a 640×640 letterboxed detector input 25 % padding **if** the engine is 640×640, which the pending binding check settles (audit "engine shape").

**Replay clips.** The maintainer's private recordings stay outside the repository; their location, names and SHA-256 hashes are in the local notes. Probed read-only with ffprobe: `one_person` is 60.0 s, 901 frames, H.264 Main 640×480 at 54060/3601 fps with an AAC-LC 16 kHz mono track; `empty_room` is 455.6 s, 6,832 frames, H.264 Main 640×480 at 204960/13667 (≈15.00) fps, video only. Consent and split manifests belong to V2-07; tests that use them will read a directory from `SENTINEL_REPLAY_CLIPS_DIR` and skip when it is unset.

**Live runs** read the camera URL from `SENTINEL_RTSP_URL`, which the maintainer exports; Sentinel code and commands never print or log it.

### Conflict: perception frame-rate target versus the substream (U12, needs a maintainer decision)

Guide ch. 22 sets **≥15 unique detected frames/s sustained "in the declared 1080p-input core profile"**, with **20–25 fps as a stretch target**. The measured substream is **640×480 at 15.01 fps**, so:

1. The stretch target cannot be reached from this profile: there are only ~15 unique frames/s to detect.
2. The 15 fps minimum equals the source rate, so any dropped frame fails it. It needs restating as a fraction of source frames for this profile.
3. The substream is not the "1080p-input" profile the gate names. The main stream is not yet measured.

Options (for the maintainer; nothing is chosen yet):

- **A. Substream only (640×480 at 15 fps) for the beta core profile.** Restate the gate as "every unique source frame is detected: ≥ 98 % of source frames at ≥ 14.7 fps sustained, p95 frame age within budget", with 20–25 fps not applicable. Cheapest decode and memory, and it matches the detector input. Risk: small faces at distance reduce identity quality (measure in V2-25).
- **B. Raise the substream frame rate in the camera's encoder settings** (if the camera offers 20/25 fps at 640×480). Keeps option A's costs and restores the stretch target. Needs the camera's encoder options (PENDING) and a re-measured bitrate.
- **C. Main stream for perception** (resolution and rate PENDING). Meets the "1080p-input" wording, and 20–25 fps may be possible. Costs: larger decode and buffers, probably H.265 (the `~/onvif_env` restructure hints at H.265 on the main stream), more letterbox/resize work, and browser codec risk.
- **D. Two profiles:** substream for detection and tracking, main stream for face crops, clips and live view. Best identity detail, but two upstream sessions, which guide ch. 7 requires to be counted and approved, plus cross-stream timestamp mapping.

**Recommendation:** A for the Oct 20 demo and the first Gate B work, because it is the only measured profile and is enough for detection and tracking. Measure the main stream and the camera's encoder options (PENDING commands) before choosing the beta profile between A, B and D. Record the chosen profile and restated gate as a decision; changing it later is a documented gate adjustment (ch. 22) made before held-out evaluation.

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

- `~/onvif_env/bin/activate` works around this by exporting `LD_PRELOAD` of the L4T library. **Invoking `~/onvif_env/bin/python` directly, without `activate`, gets no GPU.** `start_sentinel.sh` does that, so v1's GPU use depends on the environment of the shell that ran the launcher. `llama-server` resolves `libcuda.so.1` the same way, so its `--n-gpu-layers 999` offload is unverified. Checks 3 and 4 below settle both.
- For v2 live GPU runs: `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONPATH=src ~/onvif_env/bin/python …`.
- Removing the package or changing the loader order is a system change for the maintainer to decide (U13). Nothing was changed.

**Consequence for V2-05/V2-09 (U14).** No existing interpreter has both halves of the planned media path. `~/onvif_env` (3.10) has TensorRT 10.3 but no GStreamer in OpenCV and no importable `gi`. The system Python 3.12 imports `gi` with GStreamer 1.24.2 (`python3-gi` 3.48.2, `nvidia-l4t-gstreamer` 36.4.7 installed) but has no TensorRT bindings (not checked for a 3.12 wheel). Options when V2-05 starts: (a) decode in a `gst-launch-1.0` subprocess (`nvv4l2decoder ! nvvidconv ! BGRx ! fdsink`) that any interpreter reads from a pipe, with no new Python dependencies; (b) a new, separate v2 runtime environment, never `~/onvif_env`; (c) interim CPU decode through `~/onvif_env`'s OpenCV for the Oct 20 demo only, labelled as such. Per the maintainer's rules, any new environment or dependency is proposed first.

### Other software facts (read-only, 2026-09-29)

| Item | Observed |
|---|---|
| CUDA toolkit | `/usr/local/cuda-12.6` (`version.json`: CUDA SDK 12.6.11; `libcudart.so.12.6.68`) |
| Release upgrade | `/var/log/dist-upgrade/` shows a release upgrade that started 2026-04-18 12:36 (22.04 → 24.04 by the `.distUpgrade` source backups). This is how the device reached Ubuntu 24.04. |
| NVIDIA apt sources | `repo.download.nvidia.com/jetson/{common,t234}` at **r36.4**. There is also a generic `cuda-ubuntu2404-arm64` CUDA repository, which is not the Jetson repository. Installing from it could replace L4T CUDA components; treat it as a risk to check before any apt operation. |
| Media tools | ffmpeg/ffprobe 6.1.1 (Ubuntu), `gstreamer1.0-tools`, `-plugins-good`, `-plugins-bad` and `-libav` 1.24.x installed |
| Model files | `~/yolov8n.engine` (14,486,949 B, dated 2026-04-18); launcher paths `~/models/lfm2-vl/LFM2-VL-1.6B-Q4_0.gguf` and `mmproj-LFM2-VL-1.6B-Q8_0.gguf`; checksums PENDING |

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

Run from any directory unless stated. Nothing below prints the camera URL: commands that open the stream send errors through `sed` to redact any `rtsp://…`, or discard stderr. Paste outputs back redacted; they will be recorded as maintainer measurements.

```bash
# 1. Hardware decode and conversion elements; power mode (guide ch. 3/7)
gst-inspect-1.0 nvv4l2decoder | sed -n '1,25p'
gst-inspect-1.0 nvvidconv | sed -n '1,25p'
nvpmodel -q            # use sudo if it asks

# 2. Main stream (subtype=0): codec, size, rate. Needs SENTINEL_RTSP_URL (substream) exported.
MAIN_URL="${SENTINEL_RTSP_URL/subtype=1/subtype=0}"
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=index,codec_type,codec_name,profile,level,width,height,pix_fmt,avg_frame_rate,r_frame_rate,sample_rate,channels \
  -of default=noprint_wrappers=1 "$MAIN_URL" 2>&1 | sed -E 's#rtsp://[^[:space:]]+#rtsp://<redacted>#g'
#    60 s of video packets from each profile: frame rate, bitrate, keyframe spacing (no decode)
for URL_VAR in MAIN_URL SENTINEL_RTSP_URL; do
  echo "== $URL_VAR"
  ffprobe -v error -rtsp_transport tcp -select_streams v:0 -read_intervals %+60 \
    -show_entries packet=pts_time,size,flags -of csv=p=0 "${!URL_VAR}" 2>/dev/null \
  | awk -F, '{n++; b+=$2; if(n==1)t0=$1; t1=$1; if($3~/K/){k++; if(kp!="")g=g" "sprintf("%.2f",$1-kp); kp=$1}}
      END {d=t1-t0; printf "packets=%d span_s=%.2f fps=%.2f kbit_s=%.0f keyframes=%d gop_s=%s\n", n, d, (n-1)/d, b*8/d/1000, k, g}'
done
unset MAIN_URL
#    Also note from the camera's web/app settings: model, firmware, and the frame-rate
#    choices offered for each profile (decides option B of U12).

# 3. Which libcuda the running v1 processes actually map (run while v1 is running)
for p in $(pgrep -f surveillance4_1.py) $(pgrep -f llama-server); do
  echo "== $(ps -o comm= -p "$p") $p"; grep -o '/[^ ]*libcuda[^ ]*' /proc/"$p"/maps | sort -u
done

# 4. llama-server offload as logged (run while v1 is running)
tmux capture-pane -p -t llm -S -400 | grep -iE 'ggml_cuda_init|CUDA devices|offloaded|CUDA0|no usable GPU' | head -20

# 5. Detector engine: bindings, shapes, Ultralytics metadata (loads the engine on the GPU)
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 PYTHONDONTWRITEBYTECODE=1 ~/onvif_env/bin/python - <<'EOF'
import json, tensorrt as trt
data = open('/home/villain8001/yolov8n.engine', 'rb').read()
n = int.from_bytes(data[:4], 'little'); meta = None
try:
    meta = json.loads(data[4:4 + n].decode()); data = data[4 + n:]
except Exception:
    pass
print('ultralytics metadata:', {k: meta.get(k) for k in ('version', 'imgsz', 'batch', 'half', 'int8', 'dynamic', 'task')} if meta else None)
eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(data)
print('deserialized:', eng is not None, 'TensorRT', trt.__version__)
for i in range(eng.num_io_tensors):
    t = eng.get_tensor_name(i)
    print(t, eng.get_tensor_mode(t), eng.get_tensor_shape(t), eng.get_tensor_dtype(t))
EOF

# 6. Model checksums
sha256sum ~/yolov8n.engine ~/models/lfm2-vl/*.gguf

# 7. Hardware-decode smoke test on the substream, 30 s (V2-05 discovery; not a benchmark).
#    In a second terminal meanwhile: tegrastats --interval 1000   (save ~10 lines)
timeout -s INT 30 gst-launch-1.0 -e rtspsrc location="$SENTINEL_RTSP_URL" protocols=tcp latency=200 \
  ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw,format=BGRx' \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false -v 2>&1 \
  | sed -E 's#rtsp://[^[:space:]]+#rtsp://<redacted>#g' | grep -E 'last-message|ERROR|WARN' | tail -4
```

Still open from the guide's V2-01 acceptance, after the commands above: the JetPack release that corresponds to L4T 36.4.7 (NVIDIA release notes); PTZ capability response (C9, may stay deferred); B0 run with unique-frame throughput, stage timings, per-process/thread CPU, `MemTotal − MemAvailable`, tegrastats, PSS, clocks, temperature, headless versus desktop (needs the V2-11 trace contract first); B0/B1 workload manifest, provisional CPU budget and re-estimated effort; a known-good backup and restore point before any runtime change.

