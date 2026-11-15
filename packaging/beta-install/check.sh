#!/bin/bash
# Sentinel personal beta: everything checked offline. No credentials, no sudo, no camera, no model.
# It is start.sh's own checks and nothing else, so the check and the launch can never disagree:
#   the package's checksums, the Python environment, every asset against the manifest, the llama.cpp build, your
#   enrollment's presence, your configuration, the admission decisions, and the launch command up to the camera URL.
# It ends with BETA_CHECK=passed, or with one STOP line that says what is missing and what to do.
#   bash check.sh                            the default admission (the reference device's own files only)
#   bash check.sh --device-artifacts FILE    the opt-in route: this device's own files, each checked whole against
#                                            your record (record-artifacts.sh); see start.sh and QUICKSTART.md, section 1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$HERE/start.sh" --check "$@"
