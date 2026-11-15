#!/bin/bash
# Sentinel personal beta: your own enrollment, with the runtime's existing commands (nothing is enrolled automatically).
#
#   bash identity.sh enroll FOLDER --consent-confirmed --consent-date YYYY-MM-DD [--dry-run]
#   bash identity.sh list
#   bash identity.sh revoke idn-xxxxxxxxxxxx
#   bash identity.sh purge --yes
#
# FOLDER is one folder of 2 to 8 photos of ONE person (you), each showing one face, inside <data dir>/identity/inbox/.
# The passphrase is asked for, hidden, by the command itself (twice for a new gallery; at least 12 characters); it is
# never stored, printed or passed on a command line. No camera is opened. The face model runs on the CPU for a few
# seconds per photo. QUICKSTART.md, 'Your own enrollment', gives the steps and what consent means here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
[ "$(id -u)" != 0 ] || beta_stop "run as your own user, without sudo"
beta_verify_package "$HERE"
beta_load_paths "$HERE"
beta_local_env "$HERE"
[ -x "$RT_PY" ] || beta_stop "the Python environment is not there ($VENV). Run: bash $PKG/setup-env.sh"
[ $# -ge 1 ] || { sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 2; }
COMMAND="$1"; shift
case "$COMMAND" in
  enroll)
    [ $# -ge 1 ] || beta_stop "enroll needs the photo folder (inside $IDENT/inbox/)"
    FOLDER="$1"; shift
    case "$FOLDER" in /*) ;; *) FOLDER="$IDENT/inbox/$FOLDER" ;; esac
    ARGS=(enroll "$CONFIG" "$FOLDER" "$@") ;;
  list|revoke|purge) ARGS=("$COMMAND" "$CONFIG" "$@") ;;
  *) beta_stop "unknown command '$COMMAND' (enroll, list, revoke or purge)" ;;
esac
/usr/bin/python3 -I -c 'import cryptography' 2>/dev/null \
  || beta_stop "the gallery is sealed by /usr/bin/python3 with the system package python3-cryptography, which is missing. Install it yourself: sudo apt install python3-cryptography"
cd "$APP" || beta_stop "the package's application tree is not at $APP"
unset SENTINEL_IDENTITY_PASSPHRASE  # never taken from the environment here: the command asks for it, hidden
exec env PYTHONDONTWRITEBYTECODE=1 "LD_PRELOAD=$L4T_LIBCUDA" PYTHONPATH=src "DEEPFACE_HOME=$DEEPFACE_HOME_DIR" \
  "$RT_PY" -m sentinel.cli identity "${ARGS[@]}" --identity-dir "$IDENT"
