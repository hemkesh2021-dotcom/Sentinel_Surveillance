#!/bin/bash
# Sentinel personal beta: one build step under the memory monitor.
#
#   bash guarded-build.sh llama [--jobs N]                         llama.cpp (build-llama.sh), watched
#   bash guarded-build.sh engine [--min-start-free-bytes N]        the detector engine (build-engine.sh), watched
#   bash guarded-build.sh --check llama|engine [the same options]  the start conditions only; starts nothing
#   bash guarded-build.sh --status                                 what the last guarded build wrote
#   bash guarded-build.sh --stop                                   from a second SSH shell: a clean stop
#
# It starts the step's own script, unchanged, and a monitor beside it that reads the device's memory five times a
# second and ends the step (SIGTERM to every process the step owns, then SIGKILL to those left) at these stops:
#   engine   below 1 GiB MemFree, below 2 GiB MemAvailable, or after one hour
#   llama    below 2 GiB MemAvailable, or after three hours
# The stops say when a build is ended. They are not what a build needs: that was not measured (QUICKSTART.md,
# 'Building on this device', has the one device's observations). Before the engine step it also asks for 5 GiB of
# MemFree, a chosen preparation figure and not a measured minimum; --min-start-free-bytes N chooses another.
#
# Preparing memory is YOUR step, by hand, before you run this: closing a desktop session, releasing the page cache.
# Nothing here uses sudo, stops a service, closes a program or frees memory. Ctrl-C once, or --stop from a second
# shell, ends the running step cleanly; a lost SSH connection does not end it. The records of each run are kept in
# <data dir>/builds/. No camera is opened and no model server or runtime is started.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
usage() { sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 2; }
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
CHECK=(); STEP=""
case "${1:-}" in
  --status|--stop)
    [ $# = 1 ] || usage
    beta_load_paths "$HERE"
    exec python3 -I -B "$PKG/lib/guarded_build.py" "${1#--}" --package "$PKG" ;;
  --check) CHECK=(--check); shift ;;
esac
case "${1:-}" in engine|llama) STEP="$1"; shift ;; *) usage ;; esac
REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs|--min-start-free-bytes) [ $# -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]] || beta_stop "$1 needs a number"; REST+=("$1" "$2"); shift 2 ;;
    *) usage ;;
  esac
done
beta_verify_package "$HERE"
beta_load_paths "$HERE"
beta_local_env "$HERE"
exec python3 -I -B "$PKG/lib/guarded_build.py" run --package "$PKG" --step "$STEP" "${CHECK[@]}" "${REST[@]}"
