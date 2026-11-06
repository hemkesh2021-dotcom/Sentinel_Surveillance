# PR: Add typed core configuration and domain models

## Summary

This refactor PR introduces and integrates a small, tested core layer for
SENTINEL:

- immutable configuration dataclasses and validation;
- reusable typed domain models for bounding boxes, tracking, threat state, and
  system status; and
- unit tests covering the new behaviour; and
- engine integration: startup configuration is loaded through `SentinelConfig`
  and tracked detections are stored as `BoundingBox` objects.

The change preserves the existing deployment defaults while allowing model,
face-database, and intruder-log paths to be overridden through `.env`.

## Why

The engine currently keeps configuration as many module-level constants and
passes bounding boxes as unstructured NumPy arrays. A shared typed core makes
configuration validation explicit and gives future incremental refactors a
single, tested vocabulary for boxes, system state, and threat data.

## Review scope

Please focus on:

- `src/sentinel/core/config.py` for configuration defaults and validation;
- `src/sentinel/core/types.py` for the domain-model API; and
- `tests/` for the expected behaviour.

The runtime process topology is unchanged: llama-server must still be started
before the Python engine.

## Test instructions for the reviewer

### 1. Check out the PR branch

```bash
git fetch origin
git checkout refactor/core-typed-config
```

### 2. Activate the Jetson environment

From the repository root:

```bash
cd ~/Sentinel_Surveillance
source ~/onvif_env/bin/activate
```

Install this branch in editable mode so the engine uses the new `sentinel`
package:

```bash
python -m pip install --no-deps -e .
```

Keep the existing deployment `.env` file unchanged. It must contain valid
`RTSP_URL`, `BOT_TOKEN`, and `CHAT_ID` values. The existing model and face
database paths used by `surveillance4_1.py` must also be present on the Jetson.

### 3. Run the surveillance engine

Start `llama-server` using the usual deployment procedure, then, in a separate
terminal from the repository root, run:

```bash
python surveillance4_1.py
```

Expected result: the process validates its configuration, loads the face
database and YOLO model, connects to the camera, and begins processing frames
without an exception. Confirm that a visible person receives a tracking box
and that the dashboard state continues to update.

Stop the engine cleanly with `Ctrl-C` after the smoke test.

### 4. Run the automated tests

After the engine smoke test, run:

```bash
python -m unittest discover -v
```

Expected result: all tests pass.

## Follow-up work (out of scope for this PR)

Possible follow-up work includes converting the remaining engine state
structures to `SystemState`, `ThreatAssessment`, `TrackedPerson`, and
`FaceIdentity`.

## Acceptance criteria

- [x] New core models and configuration validation are covered by unit tests.
- [x] `surveillance4_1.py` loads its configuration through `SentinelConfig`.
- [x] Tracking uses `BoundingBox` at the engine boundary.
- [x] The package supports standard editable installation.
- [ ] Jetson end-to-end smoke test is completed by the reviewer.
