"""No camera, no credentials, no model: the admission decisions this package relies on, made by the runtime's own
functions on this device's real files. Prints fixed labels; exit 0 only when every line is as the package expects.
Nothing is admitted, changed or bypassed here: a refusal is reported and the launch stops.

usage: PYTHONPATH=src python -B admission_check.py LIVE_YAML IDENTITY_DIR --engine FILE
           --llama-server FILE --scene-model FILE --scene-mmproj FILE --weights-dir DIR
The five paths are this installation's own (the accepted launcher used the same files at their default places).
The detector engine is compared with the accepted build's own pin, as the detector does when it loads: the accepted
launcher left that to the run itself; here it is said before anything starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentinel.config import STREAM_MAIN_SINGLE, load_config
from sentinel.demo_runtime import FaceOptions, SceneOptions, face_admission, scene_admission
from sentinel.identity.retention import reported_numbers
from sentinel.identity.vault import Secret
from sentinel.inference.legacy_ultralytics import LEGACY_ENGINE_SHA256, file_sha256
from sentinel.memory_policy import CANDIDATE_POLICY
from sentinel.scene.chat import CHAT_ADMISSION
from sentinel.scene.models import scene_model_set

SET = "lfm3-vl"
THRESHOLDS = {"match_threshold": 0.70, "margin": 0.05, "min_quality": 0.6, "confirmations": 2, "vote_ttl_s": 30.0}
# Track retention (--identity-retention; PROPOSED, experimental): the rule this launcher's run reports, as the code's own
# defaults give it. A changed value is refused here: this launcher was verified with these.
RETENTION = {"max_since_match_s": 300.0, "contradiction_votes": 10, "contradiction_span_s": 10.0,
             "min_face_width_fraction": 0.03, "conflict_votes": 2, "conflict_window_s": 30.0,
             "max_gap_s": 0.5, "overlap_iou": 0.3, "min_step_iou": 0.4}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("identity_dir", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--scene-model", type=Path, required=True)
    parser.add_argument("--scene-mmproj", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    identity_dir = args.identity_dir
    failed: list[str] = []

    def line(name: str, value: object, ok: bool) -> None:
        print(f"{name}={value}")
        if not ok:
            failed.append(name)

    line("stream_profile", config.capture.stream_profile, config.capture.stream_profile == STREAM_MAIN_SINGLE)
    line("notification_channels", len(config.notifications.channels), not config.notifications.channels)
    numbers = {key: getattr(config.identity, key) for key in THRESHOLDS}
    line("thresholds", "unchanged" if numbers == THRESHOLDS else "CHANGED", numbers == THRESHOLDS)

    try:
        pinned = file_sha256(args.engine) == LEGACY_ENGINE_SHA256
    except OSError:
        pinned = False
    line("detector_engine", "the_pinned_engine" if pinned else "refused", pinned)
    if not pinned:
        print("detector_refusal: engine_hash_mismatch: the accepted build loads only the engine with SHA-256 "
              f"{LEGACY_ENGINE_SHA256}; an engine built on this device is another file", file=sys.stderr)

    chosen = scene_model_set(SET)
    options = SceneOptions(args.llama_server, args.scene_model, args.scene_mmproj, model_set=chosen.key)
    problem, _ = scene_admission(config, options, policy=CANDIDATE_POLICY)
    line("scene_admission", "admitted" if problem is None else "refused", problem is None)
    if problem is not None:
        print(f"scene_refusal: {problem}", file=sys.stderr)

    weights = args.weights_dir
    secret = Secret("not-the-passphrase-no-gallery-is-opened")
    plain = face_admission(config, FaceOptions(identity_dir, secret, validation_run=False), policy=CANDIDATE_POLICY,
                           weights_dir=weights)
    validated = face_admission(config, FaceOptions(identity_dir, secret, validation_run=True), policy=CANDIDATE_POLICY,
                               weights_dir=weights)
    line("face_admission_without_validation_flag", "refused" if plain is not None else "ALLOWED", plain is not None)
    line("face_admission_validation_run", "allowed" if validated is None else "refused", validated is None)
    if validated is not None:
        print(f"face_refusal: {validated}", file=sys.stderr)

    retention = reported_numbers(config.identity.retention_policy(), config.identity.continuity_policy())
    line("identity_retention_rule", "proposed_defaults" if retention == RETENTION else "CHANGED", retention == RETENTION)
    plain_retention = face_admission(config, FaceOptions(identity_dir, secret, validation_run=False, retention=True),
                                     policy=CANDIDATE_POLICY, weights_dir=weights)
    line("identity_retention_without_validation_flag", "refused" if plain_retention is not None else "ALLOWED",
         plain_retention is not None)
    labels = len(config.identity.display_labels)
    line("display_labels", labels, labels == 0)  # nobody is named in chat's answers: "an enrolled person"

    line("chat_admitted", CHAT_ADMISSION.admitted, CHAT_ADMISSION.admitted is False)
    line("chat_validation_run_allowed", CHAT_ADMISSION.validation_allowed, CHAT_ADMISSION.validation_allowed is True)
    print(f"chat_reason={CHAT_ADMISSION.reason}")

    if failed:
        print("admission_check=failed:" + ",".join(failed))
        return 1
    print("admission_check=as_expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
