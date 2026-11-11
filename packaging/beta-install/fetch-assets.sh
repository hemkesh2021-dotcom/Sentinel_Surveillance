#!/bin/bash
# Sentinel personal beta: download the model files the asset manifest lists, each checked against its size and SHA-256.
#
#   bash fetch-assets.sh [--only ID]
#
# About 1.35 GB, into <prefix>/assets. A file that is already there and correct is kept; a file that is there and
# differs is never replaced or used; a download that does not match the manifest is deleted. Nothing is run or loaded.
# The addresses and checksums are in manifest/assets.json (upstream source, revision, licence for each).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
ONLY=()
case "${1:-}" in
  "") ;;
  --only) [ $# -ge 2 ] || beta_stop "--only needs an asset id"; ONLY=(--only "$2") ;;
  *) echo "usage: bash fetch-assets.sh [--only ID]"; exit 2 ;;
esac
python3 -I -B "$PKG/lib/beta.py" own --prefix "$PREFIX" assets || exit 1  # recorded before it is made
mkdir -p "$ASSETS" || exit 1
python3 -I -B "$PKG/lib/beta.py" fetch-assets --package "$PKG" "${ONLY[@]}"; CODE=$?
[ "$CODE" = 0 ] && echo "Every listed file is in place and equals its checksum. Next: bash $PKG/build-llama.sh"
exit "$CODE"
