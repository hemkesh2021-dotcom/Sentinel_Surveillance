#!/bin/bash
# Sentinel personal beta: remove what the installer made.
#
#   bash uninstall.sh [--keep-assets] [--yes]            from an installed package: removes that installation
#   bash uninstall.sh --prefix DIR [--keep-assets] [--yes]   from anywhere, the extracted archive included
#
# Removes only what the installer's own record (<prefix>/install-manifest.txt) names: the package, its Python
# environment, its model files and builds (unless --keep-assets) and its two record files. Your data directory
# (configuration, enrollment, run records) is never removed or changed, and nothing else in the prefix is. No sudo.
# It refuses while anything from the installation is running. It can always be finished: it first leaves
# <prefix>/uninstall.sh, which removes whatever is still the installer's (kept assets, or the rest of an interrupted
# uninstall) and then itself.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
PREFIX=""; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) [ $# -ge 2 ] || beta_stop "--prefix needs a directory"; PREFIX="$2"; shift 2 ;;
    --keep-assets|--yes) REST+=("$1"); shift ;;
    *) echo "usage: bash uninstall.sh [--prefix DIR] [--keep-assets] [--yes]"; exit 2 ;;
  esac
done
if [ -z "$PREFIX" ]; then
  [ "$(basename "$HERE")" = package ] && [ -f "$(dirname "$HERE")/install-manifest.txt" ] \
    || beta_stop "this copy is not installed. Name the installation: bash uninstall.sh --prefix DIR"
  PREFIX="$(dirname "$HERE")"
fi
cd / && exec python3 -I -B "$HERE/lib/beta.py" uninstall --prefix "$PREFIX" "${REST[@]}"
