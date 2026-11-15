#!/bin/bash
# Sentinel personal beta: show the settings, or change where one asset is.
#
#   bash configure.sh                       show every path in use and whether it is there
#   bash configure.sh --set KEY=/abs/path   use your own file for ENGINE, LLAMA_SERVER, SCENE_MODEL, SCENE_MMPROJ
#                                           or DEEPFACE_HOME (KEY= goes back to the installation's own)
#
# Your settings are in <data dir>/config/: site.conf (asset paths) and live.yaml (the camera's id; check.sh refuses a
# file whose thresholds, stream profile, notifications or labels differ from the accepted ones). It never asks for,
# stores or prints the camera address or the passphrase: start.sh asks for them, hidden, at each launch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
beta_verify_package "$HERE"
SETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --set) [ $# -ge 2 ] || beta_stop "--set needs KEY=/absolute/path"; SETS+=(--set "$2"); shift 2 ;;
    *) echo "usage: bash configure.sh [--set KEY=/absolute/path ...]"; exit 2 ;;
  esac
done
exec python3 -I -B "$HERE/lib/beta.py" configure --package "$HERE" "${SETS[@]}"
