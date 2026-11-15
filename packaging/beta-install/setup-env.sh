#!/bin/bash
# Sentinel personal beta: make the runtime's Python environment in <prefix>/venv.
#
#   bash setup-env.sh [--no-import-check]
#
# A new virtual environment of Python 3.10 with exactly the pinned packages (manifest/requirements-jetson.txt and
# manifest/requirements-pypi.txt: one version each, every file checked against its SHA-256), plus a link to JetPack's
# own TensorRT Python binding. It downloads about 1.5 GB and installs about 4 GB. No sudo, no system package, no other
# environment is touched; no camera, no model. It ends by importing the GPU and face libraries once (no model is
# loaded) to show that they load on this device; --no-import-check leaves that out.
# pip's download cache and every library's own cache and settings go to <prefix>/cache, temporary files to
# <prefix>/tmp (lib/common.sh, beta_local_env); nothing is written under your home directory. The packages' bytecode
# stays where pip compiles it, inside <prefix>/venv.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
beta_local_env "$HERE"
IMPORT_CHECK=1
case "${1:-}" in "") ;; --no-import-check) IMPORT_CHECK=0 ;; *) echo "usage: bash setup-env.sh [--no-import-check]"; exit 2 ;; esac

[ -x "$PYTHON310" ] || beta_stop "the Python 3.10 interpreter recorded at installation is not there: '$PYTHON310' (bash $PKG/preflight.sh explains; then install again)"
[ -f "$TENSORRT_PYTHON/tensorrt/__init__.py" ] || beta_stop "the TensorRT Python binding recorded at installation is not at $TENSORRT_PYTHON/tensorrt (bash $PKG/preflight.sh explains; then install again)"

# recorded as the installer's before it is made, so that an interruption never leaves an unrecorded environment;
# a directory of that name that the installer did not make is refused here and left alone
python3 -I -B "$PKG/lib/beta.py" own --prefix "$PREFIX" venv || exit 1
if [ ! -x "$RT_PY" ]; then
  env PYTHONDONTWRITEBYTECODE=1 "$PYTHON310" -m venv --clear "$VENV" || beta_stop "the environment could not be made"
fi
PIP=(env PYTHONDONTWRITEBYTECODE=1 "$RT_PY" -m pip --disable-pip-version-check --no-input)
"${PIP[@]}" install --no-deps --require-hashes -r "$PKG/manifest/requirements-jetson.txt" \
  || beta_stop "torch and torchvision could not be installed from the Jetson index (above). Nothing was installed anywhere else"
"${PIP[@]}" install --no-deps --require-hashes -r "$PKG/manifest/requirements-pypi.txt" \
  || beta_stop "a pinned package could not be installed (above). Nothing was installed anywhere else"

# JetPack's TensorRT binding: linked in by name, so that no other system package becomes importable here.
SITE="$("$RT_PY" -B -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" || beta_stop "the environment's package directory could not be found"
for ITEM in "$TENSORRT_PYTHON"/tensorrt "$TENSORRT_PYTHON"/tensorrt-*.dist-info; do
  [ -e "$ITEM" ] || continue
  ln -sfn "$ITEM" "$SITE/$(basename "$ITEM")" || beta_stop "the TensorRT binding could not be linked"
done

"${PIP[@]}" check || beta_stop "the installed packages do not satisfy each other's requirements (above)"
"$RT_PY" -B - "$PKG/manifest/requirements-jetson.txt" "$PKG/manifest/requirements-pypi.txt" <<'PY' || beta_stop "the environment does not hold exactly the pinned versions (above)"
import importlib.metadata as metadata, re, sys
wrong = []
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        found = re.match(r"([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)
        if not found:
            continue
        try:
            installed = metadata.version(found.group(1))
        except metadata.PackageNotFoundError:
            installed = "missing"
        if installed != found.group(2):
            wrong.append(f"{found.group(1)}: pinned {found.group(2)}, installed {installed}")
print("\n".join(wrong) or "every pinned package is installed at its pinned version")
sys.exit(1 if wrong else 0)
PY

if [ "$IMPORT_CHECK" = 1 ]; then
  echo "Importing the GPU and face libraries once (no model is loaded; about a minute)."
  env PYTHONDONTWRITEBYTECODE=1 "LD_PRELOAD=$L4T_LIBCUDA" YOLO_OFFLINE=true "$RT_PY" -B - <<'PY' || beta_stop "a library did not load on this device (above). Send the lines above with your bug report"
import tensorrt, torch, torchvision, cv2, numpy, lap
print(f"tensorrt {tensorrt.__version__}; torch {torch.__version__} (CUDA {torch.version.cuda}); torchvision {torchvision.__version__}; "
      f"OpenCV {cv2.__version__}; NumPy {numpy.__version__}")
assert tensorrt.__version__.startswith("10.3."), "the TensorRT binding is not 10.3"
assert torch.cuda.is_available(), "torch does not see the GPU (CUDA is not available)"
import ultralytics
print(f"ultralytics {ultralytics.__version__}")
import tensorflow, tf_keras, deepface
print(f"tensorflow {tensorflow.__version__}; tf_keras {tf_keras.__version__}; deepface {deepface.__version__}")
assert deepface.__version__ == "0.0.99", "DeepFace is not 0.0.99"
import onnx, onnxslim, onnxruntime
print(f"onnx {onnx.__version__}; onnxslim {onnxslim.__version__}; onnxruntime {onnxruntime.__version__} (for build-engine.sh)")
assert onnxruntime.__version__ == "1.23.2", "the onnxruntime that imports is not PyPI's 1.23.2 (the install order went wrong)"
PY
fi
echo "The Python environment is ready: $VENV"
echo "Next: bash $PKG/fetch-assets.sh"
