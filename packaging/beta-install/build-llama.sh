#!/bin/bash
# Sentinel personal beta: build llama.cpp's llama-server on THIS device, at the commit the asset manifest pins.
#
#   bash build-llama.sh [--jobs N] [--configure-only]
#
#   --jobs N           compile with N parallel jobs (1 to 8; default 4). Fewer jobs use less memory and take longer:
#                      use 2 when anything else runs on the device. The files built are the same either way.
#   --configure-only   clone and configure; compile nothing
#
# Clones https://github.com/ggml-org/llama.cpp at commit 19cce69 (tag b11938) into <prefix>/assets/llama.cpp and builds
# the llama-server target with the official CUDA options (docs/build.md at that commit) and the reference device's
# explicit ones (manifest/assets.json). About 45 minutes on a Jetson Orin Nano. No model is loaded, no sudo is used.
#
# READ FIRST: the default admission admits scene analysis and chat only with the llama-server binary it was measured
# with on the reference device, compared by SHA-256. A build made here has other bytes, so the default check refuses
# it. The opt-in device-artifacts route (experimental validation runs; QUICKSTART.md, section 1) can use it: when
# the build ends, this script notes what it built (commit, options, tools, the SHA-256 of each file it made) for
# record-artifacts.sh. That note says how the files were built. It does not admit them, and it measures nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
JOBS=4; CONFIGURE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --configure-only) CONFIGURE_ONLY=1; shift ;;
    --jobs) [ $# -ge 2 ] && [[ "$2" =~ ^[1-8]$ ]] || beta_stop "--jobs needs a number from 1 to 8"
            JOBS="$2"; shift 2 ;;
    *) echo "usage: bash build-llama.sh [--jobs N] [--configure-only]"; exit 2 ;;
  esac
done
beta_verify_package "$HERE"
beta_load_paths "$HERE"
asset() { python3 -I -B "$PKG/lib/beta.py" asset --package "$PKG" llama_server "$1"; }
# the build target by its name in the manifest ("cmake_target"; the helper's "target" is where the binary is put)
REPO_URL="$(asset repo)" && COMMIT="$(asset commit)" && TARGET="$(asset cmake_target)" || exit 1
read -r -a CMAKE_OPTIONS <<<"$(asset cmake)"
SRC="$ASSETS/llama.cpp"
[ -x /usr/local/cuda-12.6/bin/nvcc ] || beta_stop "the CUDA 12.6 compiler is not at /usr/local/cuda-12.6/bin/nvcc (bash $PKG/preflight.sh explains)"
# the pinned build uses the compilers' own defaults: flags from the environment would make it another build
for VARIABLE in CFLAGS CXXFLAGS CPPFLAGS CUDAFLAGS LDFLAGS; do
  [ -z "${!VARIABLE:-}" ] || beta_stop "$VARIABLE is set in this shell: unset it first (the pinned build takes no extra compiler or linker flags)"
done
python3 -I -B "$PKG/lib/beta.py" own --prefix "$PREFIX" assets || exit 1  # recorded before it is made
mkdir -p "$ASSETS" || exit 1
if [ ! -d "$SRC/.git" ]; then
  [ ! -e "$SRC" ] || beta_stop "$SRC exists and is not a clone made by this script; move it away yourself"
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$SRC" || beta_stop "the clone failed (network?)"
fi
git -C "$SRC" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$SRC" -c advice.detachedHead=false checkout --quiet --detach "$COMMIT" || beta_stop "commit $COMMIT is not available from $REPO_URL"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || beta_stop "the checkout is not at $COMMIT"
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || beta_stop "the llama.cpp source tree has local changes; this script builds the pinned commit only"
# The build tree is this script's own from its first moment: it is started here, with a marker that names the commit
# and the options, and a tree without that marker (configured by hand, or for another commit) is never built on.
MARKER="$SRC/build/.sentinel-beta-build"
if [ -e "$SRC/build" ]; then
  [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$COMMIT ${CMAKE_OPTIONS[*]}" ] \
    || beta_stop "$SRC/build exists and was not started by this script for this commit and these options: a build tree made another way may hold other settings. Move it away yourself, then run this again"
else
  mkdir "$SRC/build" && printf '%s\n' "$COMMIT ${CMAKE_OPTIONS[*]}" > "$MARKER" || beta_stop "the build directory could not be made"
fi
echo "Configuring llama.cpp ${COMMIT:0:12} with: ${CMAKE_OPTIONS[*]}"
cmake -S "$SRC" -B "$SRC/build" "${CMAKE_OPTIONS[@]}" || beta_stop "cmake could not configure the build (above)"
if [ "$CONFIGURE_ONLY" = 1 ]; then echo "Configured only, as asked; nothing was compiled."; exit 0; fi
# What is hashed and noted afterwards is what THIS run linked: a binary or library left in place by an earlier run
# (or put there some other way) is removed first, so the build command makes it again from the tree's own objects.
read -r -a LIBRARIES <<<"$(asset libraries)"
for NAME in "$TARGET" "${LIBRARIES[@]}"; do rm -f -- "$SRC/build/bin/$NAME"; done
echo "Building $TARGET with $JOBS parallel job(s) (about 45 minutes with four on the reference device; longer with fewer)."
cmake --build "$SRC/build" --target "$TARGET" -j "$JOBS" || beta_stop "the build failed (above)"
"$SRC/build/bin/llama-server" --version 2>&1 | head -n 2
echo "Built: $SRC/build/bin/llama-server"
python3 -I -B "$PKG/lib/artifacts.py" note --package "$PKG" --step build-llama \
  || beta_stop "the build finished but could not be noted (above). Without the note a device-artifacts record cannot say how these files were built, and a launch refuses them. Run this script again"
echo "Remember: the default admission compares this binary's SHA-256 with the reference device's and refuses it; only the opt-in device-artifacts route can use it (see the top of this script)."
