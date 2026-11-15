#!/bin/bash
# Sentinel personal beta: build the detector's TensorRT engine on THIS device.
#
#   bash build-engine.sh
#
# Exports <prefix>/assets/engine/yolov8n.pt (fetch-assets.sh) to a TensorRT engine with Ultralytics 8.4.25, with the
# reference engine's settings: FP32, 640x640, batch 1, static shape (manifest/assets.json; Ultralytics' Jetson guide:
# `format=engine`). A TensorRT engine belongs to one TensorRT version and one kind of GPU, so it is built where it
# runs. This loads the detector on this device's GPU for a few minutes; no camera is opened and no sudo is used.
#
# READ FIRST: the default admission loads only the engine with the reference device's SHA-256. An engine built here
# is another file: the default check refuses it (engine_hash_mismatch). The opt-in device-artifacts route
# (experimental validation runs; QUICKSTART.md, section 1) can use it: when the export ends, this script notes what
# it exported (the weights' SHA-256, the settings, the tool versions, the engine's SHA-256) for record-artifacts.sh.
# That note says how the engine was built. It does not admit it, and it measures nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
beta_local_env "$HERE"
[ -x "$RT_PY" ] || beta_stop "the Python environment is not there ($VENV). Run: bash $PKG/setup-env.sh"
WEIGHTS="$(python3 -I -B "$PKG/lib/beta.py" asset --package "$PKG" detector_weights target)" || exit 1
# the export's settings: the asset manifest's, given to the export call as they are, and noted as given
EXPORT="$(python3 -I -B "$PKG/lib/beta.py" asset --package "$PKG" detector_engine export)" || exit 1
OUT="$ASSETS/engine/yolov8n.engine"
[ -f "$WEIGHTS" ] || beta_stop "the detector weights are not at $WEIGHTS. Run: bash $PKG/fetch-assets.sh"
[ ! -e "$OUT" ] || beta_stop "$OUT already exists; it is not replaced. Move it away yourself to build again"
if pgrep -u "$(id -u)" -f -- "-m sentinel\.cli run" >/dev/null 2>&1; then beta_stop "a Sentinel runtime is running: stop it first (bash $PKG/stop.sh)"; fi
# What the export needs must be in the environment before any GPU work starts: Ultralytics 8.4.25 exports to ONNX
# and simplifies it first (onnx, onnxslim, onnxruntime-gpu), then TensorRT builds the engine. Offline, it installs
# nothing by itself. Read from the installed metadata; nothing is imported or loaded here.
env PYTHONDONTWRITEBYTECODE=1 "$RT_PY" -B - <<'PY' || beta_stop "the environment lacks what the export needs (above). Run: bash $PKG/setup-env.sh"
import importlib.metadata as metadata, sys
wanted = {"ultralytics": "8.4.25", "onnx": "1.20.1", "onnxslim": "0.1.90", "onnxruntime-gpu": "1.23.0",
          "onnxruntime": "1.23.2", "tensorrt": "10.3.0"}
wrong = []
for name, version in wanted.items():
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        installed = "missing"
    if installed != version:
        wrong.append(f"{name}: needs {version}, installed {installed}")
print("\n".join(wrong))
sys.exit(1 if wrong else 0)
PY
# On a Jetson the GPU's memory is taken out of free memory. An export on the reference device (the installation
# rehearsal, 2026-11-13) failed in its first GPU allocations (NvMapMemAlloc errors, then an unrelated-looking
# PyTorch NVML assertion) with MemFree between 0.08 and 0.84 GB while 4.6 to 5.5 GB were "available" as page cache:
# the downloads, the llama.cpp build and other reading before it had filled the cache. So free memory is checked
# here, before anything is loaded. The minimum is the runtime's own for loading the detector (1,500,000,000 B). What
# the export itself needs was NOT measured and may be more; nothing here frees memory for you.
MEMINFO=/proc/meminfo
MIN_FREE_BYTES=1500000000
memfree() { awk '$1 == "MemFree:" {print $2 * 1024}' "$MEMINFO"; }
FREE_BEFORE="$(memfree)"
[ -n "$FREE_BEFORE" ] || beta_stop "free memory could not be read ($MEMINFO)"
[ "$FREE_BEFORE" -ge "$MIN_FREE_BYTES" ] || beta_stop "MemFree is $FREE_BEFORE B, below the $MIN_FREE_BYTES B needed before the detector is loaded on the GPU. Memory that is only \"available\" (page cache) does not count: on a Jetson the GPU takes free memory. Free some yourself, then run this again: close what is running, or reboot; or, as your own step with sudo: sync; sudo sysctl -w vm.drop_caches=1. Nothing was built"
echo "This loads the detector on this device's GPU to build the engine (a few minutes). The camera is not opened."
echo "MemFree is $FREE_BEFORE B (at least $MIN_FREE_BYTES B is required; what the export needs was not measured)."
read -rp 'type BUILD to build the engine now, or press Enter to stop: ' ANSWER
[ "$ANSWER" = BUILD ] || beta_stop "nothing was built"
python3 -I -B "$PKG/lib/beta.py" own --prefix "$PREFIX" assets || exit 1
cd "$ASSETS/engine" || exit 1
env PYTHONDONTWRITEBYTECODE=1 "LD_PRELOAD=$L4T_LIBCUDA" YOLO_OFFLINE=true "$RT_PY" - "$WEIGHTS" "$EXPORT" <<'PY' || beta_stop "the export failed (above). MemFree was $FREE_BEFORE B before it and is $(memfree) B now. If the lines above show NvMapMemAlloc errors or a PyTorch NVML assertion, the GPU could not get memory: free more (see the top of this script's memory check) and run this again. Nothing was noted, and no engine is in place"
import json, sys
from ultralytics import YOLO
YOLO(sys.argv[1]).export(**json.loads(sys.argv[2]))
PY
[ -f "$OUT" ] || beta_stop "the export ended without writing $OUT"
echo "Built: $OUT ($(stat -c %s "$OUT") bytes, SHA-256 $(sha256sum "$OUT" | cut -d' ' -f1))"
python3 -I -B "$PKG/lib/artifacts.py" note --package "$PKG" --step build-engine --export "$EXPORT" \
  || beta_stop "the engine was built but could not be noted (above). Without the note a device-artifacts record cannot say how it was built, and a launch refuses it. Move $OUT away and run this script again"
echo "Remember: the default admission loads only the reference device's engine and refuses this one; only the opt-in device-artifacts route can use it (see the top of this script)."
