#!/bin/bash
# Sentinel personal beta: record this device's own artifacts, for the opt-in device-artifacts route.
#
#   bash record-artifacts.sh [--accept-unverified-platform]
#
# Writes <data dir>/config/device-artifacts.json: for the detector engine, the llama.cpp build (llama-server and its
# libraries), the scene model, its projector and the two face weight files, the name, size, modification time and
# SHA-256 of the file this installation uses; what this package's own steps observed about it; and this device's
# platform versions. Run it after fetch-assets.sh, build-llama.sh and build-engine.sh. It reads the files whole
# (about 1.6 GB; a minute or two) and changes none of them. No sudo, no network, no camera, no model.
#
# What the record is for, and what it is not (QUICKSTART.md, section 1):
#   - It lets `check.sh --device-artifacts FILE` and `start.sh --device-artifacts FILE` identify YOUR files by your
#     record instead of by the reference device's. For experimental validation runs only.
#   - A built file is recorded as built by this package only when build-llama.sh or build-engine.sh made exactly
#     those bytes. Any other engine or llama.cpp build is "inspected only": a checksum does not show how a file was
#     built, nothing is recorded, and nothing launches with it.
#   - It measures nothing. Resource validation of this device stays PENDING whatever the record says about files.
# A record that is already there is never replaced: move it away yourself to make a new one after a rebuild.
# On a platform the preflight calls NOT VERIFIED it records only with --accept-unverified-platform, and says so.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
ACCEPT=()
case "${1:-}" in
  "") ;;
  --accept-unverified-platform) ACCEPT=(--accept-unverified-platform) ;;
  *) echo "usage: bash record-artifacts.sh [--accept-unverified-platform]"; exit 2 ;;
esac
[ $# -le 1 ] || { echo "usage: bash record-artifacts.sh [--accept-unverified-platform]"; exit 2; }
exec python3 -I -B "$PKG/lib/artifacts.py" record --package "$PKG" "${ACCEPT[@]}"
