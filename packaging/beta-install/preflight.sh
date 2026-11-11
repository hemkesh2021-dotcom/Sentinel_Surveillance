#!/bin/bash
# Sentinel personal beta: is this device one the package can be installed on?
#
#   bash preflight.sh            reads facts only: no network, no sudo, no camera, no model, nothing written
#
# Exit 0: the one combination this package was verified on (the reference device).
# Exit 2: a nearby combination that was NOT verified; install.sh goes on only with --accept-unverified-platform.
# Exit 1: something is missing or unsupported; each line says what and what to do.
#
# The reference device (the only one the package has run on):
#   NVIDIA Jetson Orin Nano developer kit, 8 GB; L4T R36.4.7 (JetPack 6.2.1 packages); CUDA 12.6; TensorRT 10.3.0;
#   cuDNN 9; Python 3.10; an Ubuntu 24.04 userland. NVIDIA ships JetPack 6.2.1 on Ubuntu 22.04: that combination, the
#   one a tester most likely has, is "not verified" here, not "supported".
set -uo pipefail

# For the package's own tests only: facts are then read from a directory tree that stands in for "/".
ROOT="${SENTINEL_BETA_FACTS_ROOT:-}"
[ -z "$ROOT" ] || echo "TEST FIXTURE: facts are read from $ROOT, not from this device"

FAILED=0; UNVERIFIED=0
ok()         { printf 'ok          %s\n' "$*"; }
fail()       { FAILED=$((FAILED + 1)); printf 'FAIL        %s\n' "$1"; printf '            what to do: %s\n' "$2"; }
unverified() { UNVERIFIED=$((UNVERIFIED + 1)); printf 'NOT VERIFIED %s\n' "$1"; printf '            %s\n' "$2"; }
pkg_version() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true; }

# ---- the device ----
ARCH="$(uname -m 2>/dev/null || true)"
if [ "$ARCH" = aarch64 ]; then ok "architecture aarch64"
else fail "architecture is '$ARCH', not aarch64" "this package runs on an NVIDIA Jetson only"; fi

MODEL="$({ tr -d '\0' < "$ROOT/proc/device-tree/model"; } 2>/dev/null || true)"
case "$MODEL" in
  *"Jetson Orin Nano"*) ok "device: $MODEL" ;;
  *"Jetson Orin"*|*"Jetson AGX Orin"*) unverified "device: $MODEL" "only the Jetson Orin Nano developer kit (8 GB) was used; the memory limits were measured on it" ;;
  "") fail "this is not a Jetson (no /proc/device-tree/model)" "use a Jetson Orin Nano developer kit (8 GB)" ;;
  *) fail "device: $MODEL" "use a Jetson Orin Nano developer kit (8 GB); other boards have another CUDA and TensorRT stack" ;;
esac

MEM_KB="$(awk '$1 == "MemTotal:" {print $2}' "$ROOT/proc/meminfo" 2>/dev/null || true)"
if [ -n "$MEM_KB" ] && [ "$MEM_KB" -ge 7300000 ]; then ok "memory: $MEM_KB kB"
else fail "memory: '${MEM_KB:-unreadable}' kB, below the 8 GB module's" "the launch needs 5,680,693,248 B of free memory; the 4 GB module cannot run it"; fi

# ---- L4T / JetPack ----
L4T="$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1.\2/p' "$ROOT/etc/nv_tegra_release" 2>/dev/null | head -n 1)"
case "$L4T" in
  36.4.7) ok "L4T R$L4T" ;;
  36.4|36.4.*) unverified "L4T R$L4T" "the same JetPack 6.1/6.2 family (CUDA 12.6, TensorRT 10.3) as the reference device's R36.4.7, but not the release it ran" ;;
  "") fail "L4T release unreadable (/etc/nv_tegra_release)" "install NVIDIA JetPack 6.2.1 (L4T R36.4.x)" ;;
  *) fail "L4T R$L4T" "this package needs L4T R36.4.x (JetPack 6.1 to 6.2.1): its GPU libraries are CUDA 12.6 and TensorRT 10.3 builds" ;;
esac

OS="$(. "$ROOT/etc/os-release" 2>/dev/null && printf '%s %s' "${ID:-}" "${VERSION_ID:-}")"
case "$OS" in
  "ubuntu 24.04") ok "userland: Ubuntu 24.04 (as the reference device, which was upgraded from JetPack's 22.04)" ;;
  "ubuntu 22.04") unverified "userland: Ubuntu 22.04 (JetPack 6's own)" "expected to be the easier case (its system Python is 3.10 and JetPack provides python3-libnvinfer), but nothing was run on it" ;;
  *) fail "userland: '${OS:-unreadable}'" "use JetPack 6.2.1's Ubuntu 22.04" ;;
esac

# ---- CUDA, TensorRT, cuDNN (JetPack's own packages; this installer installs no system package) ----
CUDA="$(pkg_version cuda-toolkit-12-6)"
if [ -n "$CUDA" ] && [ -d "$ROOT/usr/local/cuda-12.6" ]; then ok "CUDA toolkit 12.6 ($CUDA)"
else fail "CUDA toolkit 12.6 is not installed (package cuda-toolkit-12-6, /usr/local/cuda-12.6)" "sudo apt install nvidia-jetpack   (JetPack 6.2.1's meta package; run it yourself)"; fi

TRT="$(pkg_version libnvinfer10)"
case "$TRT" in
  10.3.0.*) ok "TensorRT $TRT" ;;
  "") fail "TensorRT 10.3 is not installed (package libnvinfer10)" "sudo apt install nvidia-jetpack   (run it yourself)" ;;
  *) fail "TensorRT $TRT" "this package's detector engine and Python binding are for TensorRT 10.3.0 (JetPack 6.1 to 6.2.1)" ;;
esac

CUDNN="$(pkg_version libcudnn9-cuda-12)"
case "$CUDNN" in
  9.*) ok "cuDNN $CUDNN" ;;
  *) fail "cuDNN 9 is not installed (package libcudnn9-cuda-12; found '${CUDNN:-none}')" "sudo apt install nvidia-jetpack   (run it yourself)" ;;
esac

LIBCUDA="$ROOT/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1"
if [ -e "$LIBCUDA" ]; then ok "L4T CUDA driver library ($LIBCUDA)"
else fail "the L4T CUDA driver library is not at /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1" "it comes with nvidia-l4t-cuda (part of L4T); reinstall JetPack's L4T packages"; fi

# ---- libraries the pinned torch build loads from the system (read from the loader's cache; nothing is loaded) ----
LOADER="$(ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null || true)"
MISSING_LIBS=""
for lib in libcudart.so.12 libcublas.so.12 libcufft.so.11 libcurand.so.10 libcusparse.so.12 libcufile.so.0 libcupti.so.12 libcudnn.so.9; do
  grep -q "^[[:space:]]*$lib " <<<"$LOADER" || MISSING_LIBS="$MISSING_LIBS $lib"
done
if [ -z "$MISSING_LIBS" ]; then ok "CUDA 12.6 and cuDNN 9 libraries are in the loader's cache"
else fail "CUDA libraries missing from the loader's cache:$MISSING_LIBS" "sudo apt install nvidia-jetpack   (run it yourself), then: sudo ldconfig"; fi
if grep -q "^[[:space:]]*libcudss.so.0 " <<<"$LOADER"; then ok "cuDSS (libcudss.so.0)"
else fail "cuDSS (libcudss.so.0) is missing: the pinned torch build loads it and JetPack does not install it" "sudo apt install libcudss0-cuda-12   (from NVIDIA's CUDA apt repository; Ultralytics' NVIDIA Jetson guide gives the repository steps; run it yourself)"; fi
if grep -q "^[[:space:]]*libopenblas.so.0 " <<<"$LOADER"; then ok "OpenBLAS (libopenblas.so.0)"
else fail "OpenBLAS (libopenblas.so.0) is missing: the pinned torch build loads it" "sudo apt install libopenblas0   (run it yourself)"; fi

# ---- the gallery is sealed by the system Python with the system's cryptography package ----
if "$ROOT/usr/bin/python3" -I -c 'import cryptography' 2>/dev/null; then ok "/usr/bin/python3 with python3-cryptography (seals the enrollment)"
else fail "/usr/bin/python3 cannot import cryptography" "sudo apt install python3-cryptography   (run it yourself): the enrollment is sealed with it"; fi

# ---- Python 3.10 (the GPU wheels are built for it) and the tools the scripts call ----
PY310="${SENTINEL_BETA_PYTHON310:-$(command -v python3.10 2>/dev/null || true)}"
if [ -n "$PY310" ] && [ "$("$PY310" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = 3.10 ]; then
  if "$PY310" -c 'import venv, ensurepip' 2>/dev/null; then ok "Python 3.10 with venv ($PY310)"
  else fail "$PY310 has no venv/ensurepip module" "sudo apt install python3.10-venv   (run it yourself)"; fi
else
  fail "no Python 3.10 interpreter (python3.10) was found" "on JetPack's Ubuntu 22.04 it is the system Python: sudo apt install python3.10 python3.10-venv; elsewhere build CPython 3.10 and set SENTINEL_BETA_PYTHON310=/path/to/python3.10"
fi

TOOLS_FAILED=$FAILED
for tool in python3 sha256sum tar awk sed git cmake make g++ ss pgrep systemctl; do
  if command -v "$tool" >/dev/null 2>&1; then :; else
    case "$tool" in
      cmake|make|g++) fail "the build tool '$tool' is missing" "sudo apt install build-essential cmake   (needed once, to build llama.cpp on this device; run it yourself)" ;;
      ss) fail "'ss' is missing" "sudo apt install iproute2   (run it yourself)" ;;
      pgrep) fail "'pgrep' is missing" "sudo apt install procps   (run it yourself)" ;;
      *) fail "'$tool' is missing" "sudo apt install $tool   (run it yourself)" ;;
    esac
  fi
done
[ "$FAILED" -gt "$TOOLS_FAILED" ] || ok "tools: python3 sha256sum tar git cmake make g++ ss pgrep systemctl"

# ---- the TensorRT Python binding for Python 3.10 (JetPack's python3-libnvinfer); setup-env.sh needs it ----
TRT_PY="${SENTINEL_BETA_TENSORRT_PYTHON:-$ROOT/usr/lib/python3.10/dist-packages}"
if [ -f "$TRT_PY/tensorrt/__init__.py" ]; then ok "TensorRT Python binding for 3.10 ($TRT_PY/tensorrt)"
else
  fail "no TensorRT Python binding for Python 3.10 at $TRT_PY/tensorrt" "on JetPack's Ubuntu 22.04: sudo apt install python3-libnvinfer   (run it yourself). If you have the binding elsewhere, set SENTINEL_BETA_TENSORRT_PYTHON to the directory that holds its 'tensorrt' folder"
fi

echo
if [ "$FAILED" -gt 0 ]; then
  echo "PREFLIGHT: $FAILED item(s) must be fixed before installing. Nothing was changed."
  exit 1
fi
if [ "$UNVERIFIED" -gt 0 ]; then
  echo "PREFLIGHT: this device differs from the only one the package was verified on in $UNVERIFIED way(s)."
  echo "It may work; it has not been tried. To install anyway: bash install.sh --accept-unverified-platform"
  exit 2
fi
echo "PREFLIGHT: this is the verified combination."
exit 0
