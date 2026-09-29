# Sentinel v2 — implementation status

Last updated 2026-09-29, after the first C1 foundation slice. Requirements come from the v2 beta implementation guide (V2-01…V2-56 backlog), and corrections and regression cases from the 23 September audit review. Both documents are local-only (see D13).

## Position

| | |
|---|---|
| Branch | `v2-beta`, created from `master`; **not pushed** |
| Base commit | `2b2d639621e8c043cc58a126f47b1b8ab6c22135`, the commit the audit verified, confirmed as HEAD before starting |
| Slice commits | `ec6698d` CLAUDE.md · `b52920f` package skeleton and portable tests · `e9f959d` clock · `bc42248` frame identity · `3b47f5f` evidence/track applicability · `ee50275` config and CLI · `6a9e71d` CI workflow · then the commit that adds this record |
| Working tree | Clean apart from ignored environments/build output and local-only files excluded through `.git/info/exclude` |
| Local-only files | The v2 guide, the audit review and `docs/LOCAL_NOTES.md` (device-specific notes). A fresh clone does not contain them, although CLAUDE.md names the first two. |
| Selected package | C1: V2-02 (portable portion) and V2-04 (portable dev setup and CI skeleton) |
| Other branches | `origin/Yogeshvar425-patch-1` (teammate) is **not merged**: a single commit `6755796` that adds @Yogeshvar425 to `.github/CODEOWNERS` (merge base `c66ebde`). `origin/codex/github-audit-fixes-2026-09-19` is already in `master` via PR #4. |
| Effort | No team availability recorded yet (guide ch. 26 kickoff item); re-estimate at the end of C1 |

## C1 map: existing code and status

| Package (guide ch. 20) | Existing code | Status |
|---|---|---|
| **V2-01** Hardware and v1 timing/memory baseline. *Accept:* sanitized inventory; B0/B1 workload and trace contract; provisional CPU budget; re-estimated optimization effort | `surveillance4_1.py` reports a loop-counter FPS (L497–498) that counts the repeated frames `FrameReader.get()` returns, so it is not a throughput baseline. It has no stage timing, CPU or memory telemetry. `start_sentinel.sh` defines the B0 process set (llama-server `--n-gpu-layers 999 --ctx-size 2048 --parallel 1`, engine, dashboard) and stops the display manager. | **Pending.** Only the read-only software snapshot below was taken. |
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

## Environment observations (read-only snapshot; not the V2-01 inventory)

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

## v1 files outside the repository (read-only; none edited)

| File | Compared with repo at `2b2d639` |
|---|---|
| `~/surveillance4_1.py` | Same code; only trailing whitespace at EOF differs. `start_sentinel.sh` launches this copy. |
| `~/build_face_db.py` | Same; only the final newline differs |
| `~/dashboard.py` | Older revision than the repo file; `start_sentinel.sh` launches this copy. Details are in the local note. |
| `~/dashboard_1.py` | No repo counterpart; historical dashboard revision, not launched by `start_sentinel.sh` |
| `~/onvif_env` restructure | Not a git repository. Contains `src/nano_surveillance/` (capture, classifiers, cli, config, detector, onvif_utils, pipeline, recognizer, tracker), `configs/nano.yaml`, `configs/nano-cloud.yaml`, `scripts/` and `requirements-{base,jetson,cloud-vlm}.txt`. Its pipeline decodes the main stream with `rtph265depay ! avdec_h265` (CPU H.265), while `.env.example` uses `subtype=1`; these are unverified codec hints for Gate B. It references another environment, `~/onvif_env2`. Never copy its configs into the repository. |

No v1 code was migrated in this slice.

## Hardware checks pending (V2-01)

1. Finish the guide ch. 3 inventory and save a sanitized copy: `gst-inspect-1.0 nvv4l2decoder`, `gst-inspect-1.0 nvvidconv`, `nvpmodel -q`, the installed CUDA runtime/toolkit, the JetPack release for L4T 36.4.7, and the origin of the Ubuntu 24.04 userspace.
2. Record the OpenCV build used by `~/onvif_env` (`cv2.getBuildInformation()`: GStreamer/FFmpeg) and the capture backend v1 actually uses; the audit left this unverified.
3. Record the camera model and firmware, main/sub-stream codecs and resolutions, and PTZ capability response, redacted.
4. Record checksums of `yolov8n.engine` and the LFM2 GGUF/mmproj files, plus the engine's bindings and input shape. The audit questioned 640×640 versus 640×480 content.
5. Run B0: unique-frame throughput (not loop FPS), stage timings, per-process/thread CPU, `MemTotal − MemAvailable`, tegrastats, PSS, power mode, clocks, temperature, and headless versus desktop mode.
6. Produce the B0/B1 workload manifest and trace contract, a provisional CPU budget, and a re-estimate of optimization effort.
7. Create a known-good backup and restore point before any runtime change.

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
