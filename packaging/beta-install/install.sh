#!/bin/bash
# Sentinel personal beta: install this package.
#
#   bash install.sh [--prefix DIR] [--data-dir DIR] [--accept-unverified-platform]
#
# Copies the package to <prefix>/package and makes the data directory (defaults: ~/sentinel-beta and
# ~/sentinel-beta-data). It uses no network and no sudo, opens no camera, loads no model, installs no system package
# and no service: nothing starts by itself, now or at boot. Installing again refreshes the package only; your
# configuration, your enrollment and your run records are never overwritten.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"

PREFIX="$HOME/sentinel-beta"; DATA=""; ACCEPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) [ $# -ge 2 ] || beta_stop "--prefix needs a directory"; PREFIX="$2"; shift 2 ;;
    --data-dir) [ $# -ge 2 ] || beta_stop "--data-dir needs a directory"; DATA="$2"; shift 2 ;;
    --accept-unverified-platform) ACCEPT=1; shift ;;
    *) echo "usage: bash install.sh [--prefix DIR] [--data-dir DIR] [--accept-unverified-platform]"; exit 2 ;;
  esac
done

[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
if [ "$(basename "$HERE")" = package ] && [ -f "$(dirname "$HERE")/install.conf" ]; then
  beta_stop "this is an installed copy. Run install.sh from the extracted archive"
fi

bash "$HERE/preflight.sh"; PREFLIGHT=$?
case "$PREFLIGHT" in
  0) ;;
  2) [ "$ACCEPT" = 1 ] || beta_stop "this platform was not verified (above); nothing was installed. To install anyway: bash install.sh --accept-unverified-platform"
     echo "Going on with a platform that was NOT verified, as you asked." ;;
  *) beta_stop "the preflight failed (above); nothing was installed" ;;
esac

ROOT="${SENTINEL_BETA_FACTS_ROOT:-}"
PY310="${SENTINEL_BETA_PYTHON310:-$(command -v python3.10 2>/dev/null || true)}"
TRT_PY="${SENTINEL_BETA_TENSORRT_PYTHON:-$ROOT/usr/lib/python3.10/dist-packages}"
python3 -I -B "$HERE/lib/beta.py" install --source "$HERE" --prefix "$PREFIX" --data-dir "$DATA" \
  --python310 "$PY310" --tensorrt-python "$TRT_PY" || exit 1

cat <<NEXT

Installed. Nothing was started, and nothing starts by itself.
NOTE: this package CANNOT LAUNCH ON ANOTHER JETSON YET. The accepted build admits only the reference device's own
detector engine, llama.cpp build and model-file copies; check.sh will say so. QUICKSTART.md, section 1, explains.
Next, in this order (QUICKSTART.md explains each; all run from $PREFIX/package):
  bash setup-env.sh       the Python environment (downloads the pinned packages)
  bash fetch-assets.sh    the model files (downloads, each checked against its checksum)
  bash build-llama.sh     llama.cpp at the pinned commit, built on this device
  bash build-engine.sh    the detector's TensorRT engine, built on this device
  bash identity.sh ...    your own enrollment, with your recorded consent
  bash check.sh           everything checked offline; the camera stays closed
  bash start.sh           the manual launch; the camera opens only after you type START
NEXT
