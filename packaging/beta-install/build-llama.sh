#!/bin/bash
# Sentinel personal beta: build llama.cpp's llama-server on THIS device, at the commit the asset manifest pins.
#
#   bash build-llama.sh
#
# Clones https://github.com/ggml-org/llama.cpp at commit 19cce69 (tag b11938) into <prefix>/assets/llama.cpp and builds
# the llama-server target with the official CUDA options (docs/build.md at that commit) and the reference device's
# explicit ones (manifest/assets.json). About 45 minutes on a Jetson Orin Nano. No model is loaded, no sudo is used.
#
# READ FIRST: the accepted build admits scene analysis and chat only with the llama-server binary it was measured
# with, compared by SHA-256. A build made here has other bytes, so check.sh will refuse it until the maintainer admits
# builds made on other devices (QUICKSTART.md, 'What still blocks another device'). This script exists so that the
# build itself is reproducible and documented; it does not make the build admitted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
asset() { python3 -I -B "$PKG/lib/beta.py" asset --package "$PKG" llama_server "$1"; }
REPO_URL="$(asset repo)" && COMMIT="$(asset commit)" && TARGET="$(asset target)" || exit 1
read -r -a CMAKE_OPTIONS <<<"$(asset cmake)"
SRC="$ASSETS/llama.cpp"
[ -x /usr/local/cuda-12.6/bin/nvcc ] || beta_stop "the CUDA 12.6 compiler is not at /usr/local/cuda-12.6/bin/nvcc (bash $PKG/preflight.sh explains)"
mkdir -p "$ASSETS" && python3 -I -B "$PKG/lib/beta.py" own --prefix "$PREFIX" assets || exit 1
if [ ! -d "$SRC/.git" ]; then
  [ ! -e "$SRC" ] || beta_stop "$SRC exists and is not a clone made by this script; move it away yourself"
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$SRC" || beta_stop "the clone failed (network?)"
fi
git -C "$SRC" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$SRC" -c advice.detachedHead=false checkout --quiet --detach "$COMMIT" || beta_stop "commit $COMMIT is not available from $REPO_URL"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || beta_stop "the checkout is not at $COMMIT"
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || beta_stop "the llama.cpp source tree has local changes; this script builds the pinned commit only"
echo "Configuring llama.cpp ${COMMIT:0:12} with: ${CMAKE_OPTIONS[*]}"
cmake -S "$SRC" -B "$SRC/build" "${CMAKE_OPTIONS[@]}" || beta_stop "cmake could not configure the build (above)"
if [ "${1:-}" = --configure-only ]; then echo "Configured only, as asked; nothing was compiled."; exit 0; fi
echo "Building $TARGET (about 45 minutes; four parallel jobs)."
cmake --build "$SRC/build" --target "$TARGET" -j 4 || beta_stop "the build failed (above)"
"$SRC/build/bin/llama-server" --version 2>&1 | head -n 2
echo "Built: $SRC/build/bin/llama-server"
echo "Remember: the accepted build compares this binary's SHA-256 with the reference device's and will refuse it (see the top of this script)."
