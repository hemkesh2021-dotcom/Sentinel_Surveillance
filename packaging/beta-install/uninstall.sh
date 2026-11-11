#!/bin/bash
# Sentinel personal beta: remove what the installer made.
#
#   bash uninstall.sh [--keep-assets] [--yes]
#
# Removes <prefix>/package, <prefix>/venv and <prefix>/assets (unless --keep-assets) and the installer's two record
# files: only what the installer's own manifest lists. Your data directory (configuration, enrollment, run records) is
# never removed or changed, and nothing else in the prefix is. No sudo. It refuses while an observation is running.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"

KEEP=(); YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --keep-assets) KEEP=(--keep-assets); shift ;;
    --yes) YES=1; shift ;;
    *) echo "usage: bash uninstall.sh [--keep-assets] [--yes]"; exit 2 ;;
  esac
done
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_load_paths "$HERE"

if pgrep -u "$(id -u)" -f -- "$RT_PY -m sentinel\.cli run" >/dev/null 2>&1; then
  beta_stop "a Sentinel run from this installation is still running. Stop it first: bash $PKG/stop.sh"
fi
echo "This removes the installation at $PREFIX (the package, its Python environment${KEEP[0]:+, NOT its assets}${KEEP[0]:-, its model files and builds})."
echo "It does NOT remove your data at $DATA_DIR."
if [ "$YES" != 1 ]; then
  read -rp 'type UNINSTALL to remove it, or press Enter to keep everything: ' ANSWER
  [ "$ANSWER" = UNINSTALL ] || beta_stop "nothing was removed"
fi
# The helper is copied out first: it removes the package it would otherwise be running from.
TOOL="$(mktemp)" || beta_stop "no temporary file"
cp "$HERE/lib/beta.py" "$TOOL"
cd / && python3 -I -B "$TOOL" uninstall --prefix "$PREFIX" "${KEEP[@]}"; CODE=$?
rm -f -- "$TOOL"
exit "$CODE"
