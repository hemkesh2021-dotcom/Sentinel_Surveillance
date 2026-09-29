# Sentinel v2 — implementation status

Last updated 2026-09-29, after the V2-01 records and ingest fix of session 2 (see the slice log). Requirements come from the v2 beta implementation guide (V2-01…V2-56 backlog), and corrections and regression cases from the 23 September audit review. Both documents are local-only (see D13).

## Position

| | |
|---|---|
| Branch | `v2-beta`, created from `master`; **not pushed** |
| Base commit | `2b2d639621e8c043cc58a126f47b1b8ab6c22135`, the commit the audit verified, confirmed as HEAD before starting |
| Slice commits | `ec6698d` CLAUDE.md · `b52920f` package skeleton and portable tests · `e9f959d` clock · `bc42248` frame identity · `3b47f5f` evidence/track applicability · `ee50275` config and CLI · `6a9e71d` CI workflow · `6578ded` this record · session 2: `4587021` PTS tolerance at ingest · then the commit that adds the V2-01 records |
| Working tree | Clean apart from ignored environments/build output and local-only files excluded through `.git/info/exclude` |
| Local-only files | The v2 guide, the audit review and `docs/LOCAL_NOTES.md` (device-specific notes). A fresh clone does not contain them, although CLAUDE.md names the first two. |
| Selected package | C1: V2-01 records (portable part), then V2-03 replay regressions. V2-02 portable portion and V2-04 setup are done (session 1). |
| Other branches | `origin/Yogeshvar425-patch-1` (teammate) is **not merged**: a single commit `6755796` that adds @Yogeshvar425 to `.github/CODEOWNERS` (merge base `c66ebde`). `origin/codex/github-audit-fixes-2026-09-19` is already in `master` via PR #4. |
| Effort | No team availability recorded yet (guide ch. 26 kickoff item); re-estimate at the end of C1 |

## C1 map: existing code and status

| Package (guide ch. 20) | Existing code | Status |
|---|---|---|
| **V2-01** Hardware and v1 timing/memory baseline. *Accept:* sanitized inventory; B0/B1 workload and trace contract; provisional CPU budget; re-estimated optimization effort | `surveillance4_1.py` reports a loop-counter FPS (L497–498) that counts the repeated frames `FrameReader.get()` returns, so it is not a throughput baseline. It has no stage timing, CPU or memory telemetry. `start_sentinel.sh` defines the B0 process set (llama-server `--n-gpu-layers 999 --ctx-size 2048 --parallel 1`, engine, dashboard) and stops the display manager. | **In progress.** Camera substream facts measured by the maintainer and the `~/onvif_env` diagnostics are recorded below ("V2-01 inventory so far"). Device checks are PENDING with exact commands. |
| **V2-02** Config, frame/evidence contracts, fake clock. *Accept:* invalid config fails clearly; epoch/TTL tests pass | Config is module constants (engine L29–51, dashboard L28–31) with secrets from `.env`; the only check is an exit when secrets are missing. `FrameReader` (L392–407) keeps a single frame with no sequence, epoch or age, and keeps returning it after a stall. `last_ai_result` (L196, L235) has no source frame or expiry. Intervals and cooldowns use wall-clock `time.time()`. Face results are keyed by bare tracker ID (`verified_faces[tid]`). The dashboard shows `/tmp/surv_state.json` with no age check (L139–145). | **Portable portion implemented and committed.** Its acceptance tests pass on Python 3.10.14 and 3.12.3. Not yet used by any runtime path; GitHub CI has not run. |
| **V2-03** Replay fixtures and first identity/empty-scene fixes. *Accept:* multi-person and zero-person regressions recorded | Defects to regress: the full-frame face fallback shares one face across tracks (L335–382); the empty-track branch skips AI and dashboard updates (L505–513); exhausted retries become "Stranger" and `is_stranger = name in ("Stranger", "Unknown")` (L379–382, L524, L575); `last_ai_result` consumers use unvalidated state (L592–656). No replay harness exists. | **Pending.** Outline below; approach for the v1 side is fixed by D12. |
| **V2-04** Development setup and CI skeleton. *Accept:* clean laptop runs non-GPU checks | `.github/workflows/ci.yml` ("Dashboard checks") runs the four Flask dashboard tests. No packaging. | **In progress.** Setup and workflow are committed and were verified on the Jetson in isolated venvs and a clean archive of HEAD. **Not yet done:** an actual clean-laptop (x86-64/macOS) run and a GitHub Actions run. |

## What this slice added

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

## Verification: exact commands and results

All commands ran on 2026-09-29 on the Jetson (aarch64), from `~/sentinel-surveillance` unless noted. `~/onvif_env` was not used or modified.

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

## Not run or not established

- Neither GitHub workflow has run, because nothing was pushed.
- There was no run on a clean laptop (x86-64 or macOS); all checks ran on this Jetson's aarch64 userspace.
- No lint or type check is configured yet (guide ch. 21 lists both); deferred to keep the slice small.
- There were no camera, decoder, GPU, TensorRT, memory, throughput or latency measurements. This slice establishes no hardware, Gate B or beta-readiness result.
- The contracts are not yet used by the v1 engine or any runtime path. The other ch. 18 CLI commands were intentionally not scaffolded.

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

## Unresolved decisions and semantics

- **U1. Liveness.** Which health state withdraws the live stream? Proposal: `STALE` (no fresh frame for 2 s) already passes `live_stream=None`, so a stalled but connected stream stops current evidence. This matches the guide's "expire confirmation across gaps". Settle it in the zero-person/health slice.
- **U2. Multi-frame evidence.** `Evidence` has one source frame. VLM results citing several frames (ch. 13) need a rule for which frame's time governs age. Proposal: the oldest input frame.
- **U3. Capture time.** All ages are ingest-based. `SourceTimeQuality.CAPTURE_SYNCED` exists but nothing produces it; mapping to capture time is a Gate B / V2-06 decision.
- **U4. Per-rule TTLs.** Evidence TTLs are not configured yet; producers pass `ttl_ns`. Define them with V2-13/V2-16 (rules) and V2-26 (VLM).
- **U5. PTZ pose.** The pose or view generation (ch. 16 and 27) is not in `FrameRef`/`StreamIdentity`. Decide in C9 whether a pose change supersedes evidence the way an epoch change does.
- **U6. Versioning.** Only `Evidence` carries `contract_version: 1`. Decide before the runtime↔core handoff (V2-11) whether every serialized contract carries a version, or whether the V2-49 adapter manifest version suffices.
- **U7. Evidence size.** Evidence `value` has no size bound yet; set one with the VLM adapter.
- **U8. Confidence kinds.** `none`, `detector_score`, `similarity` and `calibrated_probability` are a proposal.
- **U9.** Resolved by D12.
- **U10. Memory units.** Decimal whole-device memory targets (5.0 / 5.4 GB) still need the kickoff confirmation asked for in guide ch. 26.
- **U11. Live-stream announcement.** Consumers outside the runtime (core, UI) must learn the live `StreamIdentity`, including `run_id`, through the V2-11 handoff. Until then, runtime evidence is not current for them. This is the safe default, but the handoff must carry it.
- **U12. Perception stream profile and frame-rate gate.** The 20–25 fps stretch target exceeds the measured 15 fps substream, and the ≥15 fps gate names a 1080p input. Options A–D and a recommendation are under "Conflict" in the V2-01 inventory. **Maintainer decision needed** before V2-05 fixes the ingest profile.
- **U13. Shadowing CUDA driver library.** `libnvidia-compute-535` hides L4T's `libcuda.so.1`. Keep the `LD_PRELOAD` workaround, or remove the package or fix the loader order (a system change with v1 at stake)? Maintainer decision; nothing was changed.
- **U14. Runtime environment for hardware adapters.** No existing interpreter has both GStreamer bindings and TensorRT. Options (a)–(c) are in the V2-01 inventory; decide when V2-05 starts.

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

## Remaining V2-03 regressions: outline of the next replay fixtures

Proposed fixture format: a synthetic JSON-lines timeline with no images or footage, e.g. `{"t_ms": 0, "event": "connect"}`, `{"t_ms": 66, "event": "frame", "persons": [...], "faces": [...]}`, `{"event": "disconnect"}` and `{"event": "ai_complete", ...}`. A loader under `tests/replay/` drives `FakeClock` and `FrameStamper`, and tests assert the resulting observations and state. The v1 side of each regression uses the B0 behaviour snapshot from D12. Consented real clips come with V2-07.

**R1. One face, two overlapping tracks** (audit finding 1; guide ch. 2 row 2 and ch. 21)
- *Setup:* tracks A and B overlap so that one detected face lies in both head regions, and enrollment contains the matching identity. Variants: exact ambiguity; a known and an unknown person with only the known face visible; the pair later separates; a reconnect mid-sequence; an empty enrollment database.
- *Expect:* each face goes to at most one track and each track gets at most one face. Ambiguity leaves both tracks `unresolved`: never `known`, never a face-confirmed stranger. After separation the face attaches to exactly one track. Post-reconnect tracks inherit nothing, because results are keyed by `TrackKey`. An empty database yields no error and no stranger label.
- *Needs:* a `FaceObservation` contract (box, landmarks, quality, `FrameKey`), a global association function, and identity states `unresolved`/`unknown`/`known`/`uncertain`.
- *B0 snapshot:* the full-frame fallback in `face_recognition_worker` (L335–382) labels both tracks.

**R2. Zero-person scenes** (guide ch. 2 row 1 and ch. 21)
- *Setup:* minutes of fresh frames with no detections; a person who leaves; a stub fire/smoke scene candidate with nobody present; then a stall (frames stop) and an outage.
- *Expect:* the published state is empty with a fresh frame age. The departed track expires 1 s after its last detection. The scene lane keeps its interval with zero tracks. The stub candidate yields a zero-person candidate incident, and being VLM-only it is not a critical fire alert. A stall shows stale at 2 s and offline at 10 s, distinct from "empty" and from "detector unavailable".
- *Needs:* freshness classification using `freshness.*` (settles U1), scene-state publication, and a scene-lane scheduler stub.
- *B0 snapshot:* the empty-track branch (L505–513) skips AI analysis and dashboard updates.

**R3. Delayed stale AI results** (audit finding 3; guide ch. 2 row 4 and ch. 21)
- *Setup:* a VLM stub with scripted latency, whose job for frame F (incident A) completes after its TTL. Variants: a reconnect before completion; a new incident B opened meanwhile; malformed JSON; a wrong job or event ID; a timeout.
- *Expect:* the late result is `EXPIRED` or `SUPERSEDED_EPOCH`, so current state does not change and no alert fires. It annotates only A, and B is unchanged. A malformed or mismatched result becomes `error` evidence and a timeout becomes `timeout` evidence; the previous verdict is never reused. Status and dashboard consumers pass through the same applicability gate.
- *Needs:* the ch. 27 job/result contract (job ID, source `FrameRef`, incident ID, epoch, monotonic deadline), a FakeClock-driven stub VLM adapter, and a scene-state holder that accepts only `CURRENT` evidence.
- *B0 snapshot:* the worker and the global `last_ai_result` (L196–259) are read directly by the overlay, intruder and status paths (L592–656).

## Next concrete task

Add the replay loader under `tests/replay/` with the first B0 behaviour snapshot (D12), then implement **R3**. Its core contracts exist, so it needs only the job/result contract and a stub VLM adapter. Follow with R2, which adds freshness classification and settles U1, then R1, which adds face association. Mark V2-03 done only when all three regressions are recorded.
