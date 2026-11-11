# Shared by the personal-beta scripts. Sourced, never run.
#
# Where things are comes from two plain KEY=VALUE files. They are READ as text by lib/beta.py, never executed:
#   <prefix>/install.conf          written by install.sh (installer-owned): DATA_DIR, PYTHON310, TENSORRT_PYTHON
#   <data dir>/config/site.conf    the tester's own settings, never overwritten: optional asset paths
# Layout after install.sh:
#   <prefix>/package/   this package, checked against its SHA256SUMS before every use
#   <prefix>/venv/      the runtime's Python environment (setup-env.sh)
#   <prefix>/assets/    model files, the llama.cpp build, the detector engine (fetch-assets.sh, build-*.sh)
#   <data dir>/         config/, identity/ (the tester's own enrollment), observe/ (one folder per run)

L4T_LIBCUDA=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1  # decision D27: the L4T driver library, loaded first

beta_stop() { printf '\a' >&2; printf 'STOP: %s\n' "$*" >&2; exit 1; }

# beta_verify_package DIR: every file of the package equals its checksum; nothing is run from a package that differs.
# First with sha256sum alone, so that the helper used next is itself known to be the package's; then the helper
# also refuses any file that is not part of the package (it would be importable beside the checked ones).
beta_verify_package() {
  [ -f "$1/SHA256SUMS" ] && [ -f "$1/VERSION" ] || beta_stop "$1 is not a complete package (SHA256SUMS or VERSION is missing)"
  (cd "$1" && sha256sum --quiet --strict -c SHA256SUMS >/dev/null 2>&1) \
    || beta_stop "this package's files differ from their checksums ($1/SHA256SUMS). Extract the archive again and reinstall; never edit its files"
  python3 -I -B "$1/lib/beta.py" verify-package --package "$1" || exit 1
}

# beta_load_paths PKG: for the scripts that run from an installed package (<prefix>/package). Sets every path they
# use, as lib/beta.py resolves them: PREFIX PKG APP VENV RT_PY ASSETS DATA_DIR SITE CONFIG IDENT OBS_ROOT PYTHON310
# TENSORRT_PYTHON ENGINE LLAMA_SERVER SCENE_MODEL SCENE_MMPROJ DEEPFACE_HOME_DIR WEIGHTS_DIR
beta_load_paths() {
  local line key
  BETA_PATHS="$(python3 -I -B "$1/lib/beta.py" paths --package "$1")" || exit 1
  while IFS= read -r line; do
    key="${line%%=*}"
    case "$key" in
      PREFIX|PKG|APP|VENV|RT_PY|ASSETS|DATA_DIR|SITE|CONFIG|IDENT|OBS_ROOT|PYTHON310|TENSORRT_PYTHON|ENGINE|LLAMA_SERVER|SCENE_MODEL|SCENE_MMPROJ|DEEPFACE_HOME_DIR|WEIGHTS_DIR)
        printf -v "$key" '%s' "${line#*=}" ;;
    esac
  done <<<"$BETA_PATHS"
  [ -d "$DATA_DIR" ] || beta_stop "the data directory recorded at installation is not there: $DATA_DIR"
}
