#!/bin/bash
# Sentinel personal beta: stops the running observation from any SSH shell, the same way Ctrl-C does (use it if the
# launch window was lost). It signals only the guard recorded by start.sh for the latest observation, if that process
# is still that guard, or a runtime started for that observation's own data directory: each is identified by its
# whole arguments, its owner and its start time, never by a pattern. With nothing running it signals nothing.
# It always ends with what the last run recorded about the desktop (stopped, restarted, or left stopped).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_load_paths "$HERE"
exec python3 -I -B "$PKG/lib/beta.py" stop-run --observe-root "$OBS_ROOT" --stop-script "$PKG/stop.sh"
