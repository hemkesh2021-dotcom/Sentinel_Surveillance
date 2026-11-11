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
# READ FIRST: the accepted build loads only the engine with the reference device's SHA-256. An engine built here is
# another file: check.sh will refuse it (engine_hash_mismatch) until the maintainer admits engines built on other
# devices (QUICKSTART.md, 'What still blocks another device'). This script documents and reproduces the build; it
# does not make the engine admitted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
[ -x "$RT_PY" ] || beta_stop "the Python environment is not there ($VENV). Run: bash $PKG/setup-env.sh"
WEIGHTS="$(python3 -I -B "$PKG/lib/beta.py" asset --package "$PKG" detector_weights target)" || exit 1
OUT="$ASSETS/engine/yolov8n.engine"
[ -f "$WEIGHTS" ] || beta_stop "the detector weights are not at $WEIGHTS. Run: bash $PKG/fetch-assets.sh"
[ ! -e "$OUT" ] || beta_stop "$OUT already exists; it is not replaced. Move it away yourself to build again"
if pgrep -u "$(id -u)" -f -- "-m sentinel\.cli run" >/dev/null 2>&1; then beta_stop "a Sentinel runtime is running: stop it first (bash $PKG/stop.sh)"; fi
echo "This loads the detector on this device's GPU to build the engine (a few minutes). The camera is not opened."
read -rp 'type BUILD to build the engine now, or press Enter to stop: ' ANSWER
[ "$ANSWER" = BUILD ] || beta_stop "nothing was built"
cd "$ASSETS/engine" || exit 1
env PYTHONDONTWRITEBYTECODE=1 "LD_PRELOAD=$L4T_LIBCUDA" YOLO_OFFLINE=true "$RT_PY" - "$WEIGHTS" <<'PY' || beta_stop "the export failed (above)"
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).export(format="engine", imgsz=640, batch=1, half=False, int8=False, dynamic=False, simplify=True, nms=False, device=0)
PY
[ -f "$OUT" ] || beta_stop "the export ended without writing $OUT"
echo "Built: $OUT ($(stat -c %s "$OUT") bytes, SHA-256 $(sha256sum "$OUT" | cut -d' ' -f1))"
echo "Remember: the accepted build loads only the reference device's engine and will refuse this one (see the top of this script)."
