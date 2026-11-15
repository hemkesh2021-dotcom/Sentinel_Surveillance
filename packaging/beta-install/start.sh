#!/bin/bash
# Sentinel personal beta: the manual launch (local only; see QUICKSTART.md). Run as yourself, from a plain SSH shell.
#   bash start.sh           checks, asks for the camera URL and the passphrase (hidden), and opens the camera ONLY after you type START
#   bash start.sh --check   checks only: no credentials, no sudo, no camera, no model; keeps no record outside a temporary directory
#   either, with --device-artifacts FILE   opt-in, experimental: this device's own detector engine, llama.cpp build and
#                           model files, identified by your record FILE (record-artifacts.sh) instead of by the
#                           reference device's records. Every file is checked whole against the record first. For
#                           validation runs only; this device's resources are NOT validated (QUICKSTART.md, section 1).
# It validates and admits nothing: face recognition, identity retention and chat run as experimental validation-run
# features, exactly as in the build this package was made from. Snapshots and external notifications stay off.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
beta_load_paths "$HERE"

# ---- site block ----
MEMINFO=/proc/meminfo
RT=(env PYTHONDONTWRITEBYTECODE=1 "LD_PRELOAD=$L4T_LIBCUDA" PYTHONPATH=src "DEEPFACE_HOME=$DEEPFACE_HOME_DIR" "$RT_PY" -m sentinel.cli)
G=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$RT_PY" benchmarks/runner/step5_guard.py --step4-headroom)
HOSTPREP=(/usr/bin/env python3 -I -B "$PKG/lib/hostprep.py")
# ---- end of site block ----

PORTS='18090|18081'
DURATION_S=3600  # the guard's own maximum
# MemFree needed at launch and memory in use allowed at launch: the accepted launcher's numbers, unchanged. They come
# from one live capped LFM3 run on the reference device (a Jetson Orin Nano, 8 GB): it used 4,573,396,992 B of MemFree
# and rose by 3,472,433,152 B, which must fit under the guard's 4,800,000,000 B stop; plus 32 MiB for the pictures chat
# may hold in memory (arithmetic from array sizes, not a measurement). The guard remains the protection.
HELD_PICTURES_BYTES=33554432
FLOOR=$((5647138816 + HELD_PICTURES_BYTES))
IN_USE_LIMIT=$((1327566848 - HELD_PICTURES_BYTES))
FLAGS=(--engine "$ENGINE" --workload-thp-disable --post-load-release --scene --scene-model-set lfm3-vl
       --llama-server "$LLAMA_SERVER" --scene-model "$SCENE_MODEL" --scene-mmproj "$SCENE_MMPROJ"
       --face --face-validation --identity-retention --identity-dir "$IDENT" --chat --chat-validation --live-preview
       --feature-controls)

CAMERA_STARTED=0; DM_STOPPED=0; KEEP_DESKTOP_DOWN=0; OBS=""; TMPCHECK=""
stop_now() { printf '\a'; echo "STOP: $*"; [ "$CAMERA_STARTED" = 1 ] || echo "The camera was not opened."; exit 1; }
meminfo() { awk -v k="$1:" '$1==k {print $2*1024}' "$MEMINFO"; }
# What happens to the desktop is written to the observation's desktop.txt as it happens, so that stop.sh can still
# say what to do after this window is lost.
desktop_note() { [ -z "$OBS" ] || echo "$1 utc=$(date -u +%FT%TZ)" >> "$OBS/desktop.txt"; }
restore_desktop() {
  [ "$DM_STOPPED" = 1 ] || return 0
  DM_STOPPED=0
  if [ "$KEEP_DESKTOP_DOWN" = 1 ]; then
    desktop_note "restart=left_stopped_something_still_running"
    echo "The desktop was left stopped because something of this run is still running. When it has stopped: sudo systemctl start display-manager"
    return 0
  fi
  echo "Restarting the desktop (sudo may ask your password)."
  if sudo systemctl start display-manager; then desktop_note "restart=ok"
  else desktop_note "restart=FAILED"; printf '\a'; echo "The desktop did NOT restart. Run: sudo systemctl start display-manager (bash $PKG/stop.sh repeats this advice)"; fi
}
on_exit() {
  local code=$?
  trap - EXIT INT HUP QUIT TSTP TERM
  unset SENTINEL_RTSP_URL SENTINEL_IDENTITY_PASSPHRASE
  restore_desktop
  [ -z "$TMPCHECK" ] || rm -rf -- "$TMPCHECK"
  exit "$code"
}
trap on_exit EXIT

MODE=launch; DEVICE_RECORD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check; shift ;;
    --device-artifacts) [ $# -ge 2 ] && [ -n "$2" ] || { echo "usage: bash start.sh [--check] [--device-artifacts FILE]"; exit 2; }
                        DEVICE_RECORD="$2"; shift 2 ;;
    *) echo "usage: bash start.sh [--check] [--device-artifacts FILE]"; exit 2 ;;
  esac
done

# ---- checks, in both modes; none of them opens the camera or loads a model ----
[ "$(id -u)" != 0 ] || stop_now "run as your own user, without sudo"
[ -z "${SENTINEL_TELEGRAM_BOT_TOKEN:-}${SENTINEL_TELEGRAM_CHAT_ID:-}" ] || stop_now "unset the Telegram variables first (external notifications stay off)"
grep -q '^THP_enabled:[[:space:]]*1$' /proc/self/status || stop_now "transparent huge pages are disabled for this shell; the memory policy needs them enabled before launch (open a new plain SSH shell)"
beta_verify_package "$PKG"
# caches, settings and temporary files of everything started below: inside this installation, not under HOME. The
# runtime and the enrollment name the face weights' own place themselves (DEEPFACE_HOME in the site block above).
beta_local_env "$PKG"
[ -x "$RT_PY" ] || stop_now "the Python environment is not there ($VENV). Run: bash $PKG/setup-env.sh"
[ -e "$L4T_LIBCUDA" ] || stop_now "the L4T CUDA driver library is not at $L4T_LIBCUDA (bash $PKG/preflight.sh explains)"
[ -f "$CONFIG" ] || stop_now "your configuration is not at $CONFIG. Run install.sh again from the extracted archive: it writes the file when it is missing"
python3 -I -B "$PKG/lib/beta.py" verify-assets --package "$PKG" \
  || stop_now "an asset is missing or is not the pinned file (above). bash $PKG/configure.sh shows every path"
# the llama.cpp build: its own version line names its commit ("version: ... (build N, commit 19cce6919)"); no model is loaded
WANT="$(python3 -I -B "$PKG/lib/beta.py" asset --package "$PKG" llama_server commit)" || stop_now "the asset manifest names no llama.cpp commit"
BUILT="$("$LLAMA_SERVER" --version 2>&1 | sed -n 's/^version: .*commit \([0-9a-f]\{7,40\}\)).*/\1/p' | head -n 1)"
[ -n "$BUILT" ] && [ "${WANT#"$BUILT"}" != "$WANT" ] \
  || stop_now "$LLAMA_SERVER is not the llama.cpp build this package needs (it reports commit '${BUILT:-none}', the pinned one is ${WANT:0:12}). Run: bash $PKG/build-llama.sh"
[ -f "$IDENT/gallery.sealed" ] || stop_now "there is no enrollment at $IDENT. Enroll yourself first, with your recorded consent: QUICKSTART.md, 'Your own enrollment' (nobody else's gallery comes with this package)"
if pgrep -u "$(id -u)" -x llama-server >/dev/null 2>&1; then stop_now "a llama-server is running: stop it first"; fi
if pgrep -u "$(id -u)" -f -- "-m sentinel\.cli run" >/dev/null 2>&1; then stop_now "a Sentinel runtime is running: stop it first (bash $PKG/stop.sh)"; fi
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq ":($PORTS)\$"; then stop_now "port 18090 or 18081 is already in use"; fi

# ---- the opt-in device-artifacts route: this device's own files, each checked whole against your record ----
# Nothing here admits or measures anything. Without --device-artifacts none of it runs and the default admission
# below is the only one.
ROUTE=(); RUN_ROUTE=(); RECORD_SHA=""; ARTIFACTS=""; PLATFORM_STATE=""; PLATFORM_SAY=""
if [ -n "$DEVICE_RECORD" ]; then
  case "$DEVICE_RECORD" in /*) ;; *) DEVICE_RECORD="$PWD/$DEVICE_RECORD" ;; esac
  [ -f "$DEVICE_RECORD" ] || stop_now "there is no device-artifacts record at $DEVICE_RECORD. Make one with: bash $PKG/record-artifacts.sh"
  ARTIFACTS="$(python3 -I -B "$PKG/lib/artifacts.py" verify --package "$PKG" --record "$DEVICE_RECORD")" \
    || { printf '%s\n' "$ARTIFACTS"; stop_now "this device's files are not as the record says, or the record cannot be used (above). Nothing is bypassed: QUICKSTART.md, section 1"; }
  RECORD_SHA="$(sed -n 's/^device_artifacts_record_sha256=//p' <<<"$ARTIFACTS" | head -n 1)"
  PLATFORM_STATE="$(sed -n 's/^platform_preflight=\([a-z_]*\).*/\1/p' <<<"$ARTIFACTS" | head -n 1)"
  case "$PLATFORM_STATE" in
    verified_combination) PLATFORM_SAY="the verified combination" ;;
    not_verified_acknowledged) PLATFORM_SAY="NOT VERIFIED: nothing was ever run on this combination (you acknowledged that when the record was made)" ;;
    *) stop_now "the record's check did not report the platform's state" ;;
  esac
  [ "${#RECORD_SHA}" = 64 ] || stop_now "the record's check did not report the record's checksum"
  case "$(sed -n 's/^platform_facts=//p' <<<"$ARTIFACTS" | head -n 1)" in
    this_device) ;;
    test_fixture) PLATFORM_STATE="${PLATFORM_STATE}_test_fixture"; PLATFORM_SAY="TEST FIXTURE: the platform facts did not come from this device ($PLATFORM_SAY)" ;;
    *) stop_now "the record's check did not report where the platform facts came from" ;;
  esac
  ROUTE=(--device-artifacts "$DEVICE_RECORD")
fi

cd "$APP" || stop_now "the package's application tree is not at $APP"
VALID="$("${RT[@]}" config validate "$CONFIG" 2>&1)" || { printf '%s\n' "$VALID"; stop_now "the configuration did not validate ($CONFIG)"; }
grep -qxF '  adapter llama-lfm2-vl-scene: enabled (enabled)' <<<"$VALID" || { printf '%s\n' "$VALID"; stop_now "the scene adapter is not enabled in $CONFIG"; }
grep -qxF '  adapter legacy-deepface-facenet512-yunet: enabled (enabled for guarded --face-validation runs only (admission pending validation))' <<<"$VALID" \
  || { printf '%s\n' "$VALID"; stop_now "the face adapter is not as expected in $CONFIG"; }
ADMIT="$(env -u SENTINEL_RTSP_URL -u SENTINEL_IDENTITY_PASSPHRASE PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$RT_PY" -B "$PKG/lib/admission_check.py" \
          "$CONFIG" "$IDENT" --engine "$ENGINE" --llama-server "$LLAMA_SERVER" --scene-model "$SCENE_MODEL" --scene-mmproj "$SCENE_MMPROJ" --weights-dir "$WEIGHTS_DIR" "${ROUTE[@]}")" \
  || { printf '%s\n' "$ADMIT"
       if [ -n "$DEVICE_RECORD" ]; then stop_now "an admission decision is not as the device-artifacts route expects (above). Nothing is bypassed: QUICKSTART.md, section 1"; fi
       stop_now "an admission decision is not as this package expects (above). The default admission admits only the reference device's own files, and nothing is bypassed here. On your own device: the opt-in device-artifacts route, QUICKSTART.md, section 1"; }
if [ "$MODE" = check ]; then
  printf '%s\n' "$ADMIT"
  [ -z "$ARTIFACTS" ] || printf '%s\n' "$ARTIFACTS"
fi

if [ "$MODE" = check ]; then
  TMPCHECK="$(mktemp -d)" || stop_now "no temporary directory"
  LINE="$(env -u SENTINEL_RTSP_URL SENTINEL_IDENTITY_PASSPHRASE=check-only-not-a-passphrase "${RT[@]}" run "$CONFIG" --data-dir "$TMPCHECK/db" "${FLAGS[@]}" "${ROUTE[@]}" 2>&1 >/dev/null | tail -n 1)"
  echo "command_line_check=$LINE"
  [ "$LINE" = "run: rtsp_url_missing" ] || stop_now "the launch command was not accepted up to the camera URL check"
  echo "package=$(head -n 1 "$PKG/VERSION")"
  grep -m 1 '^application=' "$PKG/VERSION" || echo "application=not stated in $PKG/VERSION"
  echo "memfree_bytes=$(meminfo MemFree) launch_floor_bytes=$FLOOR display_manager=$(systemctl is-active display-manager 2>/dev/null || true)"
  # where this installation keeps the libraries' caches, settings and temporary files, as just exported
  sed -n 's/^\([A-Z_][A-Z_]*=..*\)$/local_env \1/p' <<<"$BETA_LOCAL_LINES" | grep -v '^local_env LOCAL_ENV_COUNT='
  echo "local_env_set_in_your_shell_and_not_used=${BETA_LOCAL_REPLACED:-none}"
  if [ -n "$DEVICE_RECORD" ]; then
    echo "BETA_CHECK_ROUTE=device_artifacts (opt-in, experimental validation runs only: this device's files equal its record; its resources are NOT validated, resource_validation=pending; platform $PLATFORM_STATE)"
  fi
  echo "BETA_CHECK=passed (no credentials asked, no sudo, the camera was not opened, no model was loaded)"
  exit 0
fi

# ---- launch mode ----
[ -t 0 ] && [ -t 1 ] || stop_now "run this in an interactive SSH shell"
cat <<'BOX'

=========================== BEFORE YOU LAUNCH ===========================
  What opens the camera : only typing START at the last prompt below.
  How long              : at most 60 minutes after the models are ready
                          (1-2 min to load); it then stops by itself.
  Where to be           : anywhere; ordinary use, nothing is cued.
  To stop early         : Ctrl-C ONCE in this window, then wait for OBS_exit.
  Dashboard             : start the SSH tunnel first (QUICKSTART.md).
  Validates nothing     : face results and identity retention are
                          experimental; this admits nothing.
=========================================================================

BOX
if [ -n "$DEVICE_RECORD" ]; then
  cat <<BOX
================ DEVICE-ARTIFACTS ROUTE (opt-in, experimental) ================
  Files         : this device's own detector engine, llama.cpp build and
                  model files; each equals your record by SHA-256 (checked
                  just now). That is all the record shows.
  NOT validated : this device's memory use, speed and stability. Resource
                  validation is PENDING: nothing was measured on this device.
                  The memory gates and the guard's limits are the reference
                  device's; here they are limits, not evidence.
  Platform      : $PLATFORM_SAY
  Record        : $DEVICE_RECORD
===============================================================================

BOX
fi

OBS="$OBS_ROOT/obs-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p -m 700 "$OBS_ROOT" && chmod 700 "$OBS_ROOT" && mkdir -m 700 "$OBS" "$OBS/db" || stop_now "could not make $OBS"
echo "observation folder: $OBS"

# optional processes: the memory limits were measured without them; closed only on your word
"${HOSTPREP[@]}" scan --out "$OBS/optional-processes.json"; SCAN=$?
case "$SCAN" in
  0) ;;
  3) echo "(Block names such as i-C, i-R or i-H printed above come from this helper's original package: here, rerun start.sh instead.)"
     read -rp 'type CLOSE to close the processes listed under "will be closed", or press Enter to stop without closing them: ' A
     [ "$A" = CLOSE ] || stop_now "nothing was closed: the run needs them closed"
     "${HOSTPREP[@]}" close --plan "$OBS/optional-processes.json" --record "$OBS/optional-processes-closed.txt" || stop_now "closing them failed (the reason is above)" ;;
  *) stop_now "the process scan stopped (the reason is above); use a plain SSH shell, then rerun start.sh" ;;
esac

# memory preparation: the operator's own step, typed
DM_STATE="$(systemctl is-active display-manager 2>/dev/null || true)"
if [ "$DM_STATE" = active ] || [ "$(meminfo MemFree)" -lt "$FLOOR" ]; then
  echo "Memory needs preparing: the desktop is $DM_STATE, MemFree is $(meminfo MemFree) B, the launch floor is $FLOOR B."
  echo "PREPARE stops the desktop (it restarts when this ends) and drops the page cache once: sync; sudo sysctl -w vm.drop_caches=1."
  case "${XDG_SESSION_TYPE:-}" in x11|wayland)
    [ "$DM_STATE" != active ] || stop_now "this shell runs inside the desktop that PREPARE would stop. Use a plain SSH shell from another machine, then rerun start.sh" ;;
  esac
  read -rp 'type PREPARE to do that now (sudo may ask your password), or press Enter to stop: ' A
  [ "$A" = PREPARE ] || stop_now "not prepared; the desktop and the page cache were left as they were"
  sudo -v || stop_now "sudo was not authenticated; the desktop and the page cache were left as they were"
  if [ "$DM_STATE" = active ]; then
    desktop_note "stopped_by_launcher"; DM_STOPPED=1
    sudo systemctl stop display-manager || stop_now "the desktop did not stop"
  fi
  sync && sudo sysctl -w vm.drop_caches=1 >/dev/null || stop_now "the page cache was not dropped"
fi
"${HOSTPREP[@]}" check --record "$OBS/optional-processes-check.txt" \
  || stop_now "not prepared: an optional process is still running or the desktop is not stopped (above); close it, then rerun start.sh"

read -rsp 'camera MAIN-stream URL (hidden, never saved): ' SENTINEL_RTSP_URL; echo; export SENTINEL_RTSP_URL
read -rsp 'gallery passphrase (hidden, never saved): ' SENTINEL_IDENTITY_PASSPHRASE; echo; export SENTINEL_IDENTITY_PASSPHRASE
[ -n "$SENTINEL_IDENTITY_PASSPHRASE" ] || stop_now "the passphrase is empty"
case "$SENTINEL_RTSP_URL" in rtsp://*|rtsps://*) ;; *) stop_now "that is not an rtsp:// URL (it is not shown or saved)" ;; esac

fingerprint() { (cd "$IDENT" && sha256sum gallery.sealed audit.jsonl && stat -c '%a %s %Y %n' . gallery.sealed audit.jsonl inbox); }
PREPARED_UTC="$(date -u +%FT%TZ)"

echo
echo "===================== THE NEXT STEP OPENS THE CAMERA ====================="
echo "main-stream head-window recognition, experimental; your own enrollment; identity retention on a continuous track,"
echo "experimental (PROPOSED values); live view on; capped LFM3 scene analysis; chat, experimental (a validation run;"
echo "not admitted) with recorded-fact answers, follow-ups, refresh, zoom and scan; feature controls; no snapshots;"
echo "no notifications."
[ -z "$DEVICE_RECORD" ] || echo "DEVICE-ARTIFACTS ROUTE: this device's own files, as recorded; this device's resources are NOT validated (pending)."
read -rp 'type START to open the camera now, or press Enter to stop without opening it: ' A
[ "$A" = START ] || stop_now "you did not type START"
START_UTC="$(date -u +%FT%TZ)"

fingerprint > "$OBS/identity-before.txt" || stop_now "the enrollment could not be read"
# the memory last: immediately before the launch
for k in MemTotal MemFree MemAvailable; do [ -n "$(meminfo "$k")" ] || stop_now "memory could not be read"; done
FREE="$(meminfo MemFree)"; AVAIL="$(meminfo MemAvailable)"; IN_USE=$(( $(meminfo MemTotal) - AVAIL ))
[ "$FREE" -ge "$FLOOR" ] || stop_now "MemFree $FREE B is below the launch floor $FLOOR B: rerun start.sh (it offers PREPARE) after closing what is running"
[ "$AVAIL" -ge 4000000000 ] || stop_now "MemAvailable $AVAIL B is below 4,000,000,000 B: close what is running, then rerun start.sh"
[ "$IN_USE" -lt "$IN_USE_LIMIT" ] || stop_now "memory in use $IN_USE B is at or above $IN_USE_LIMIT B: the rise measured on the reference device would reach the guard's 4.8 GB stop; close what is running, then rerun start.sh"
{ echo "kind=live_observation_validates_nothing"; echo "prepared_utc=$PREPARED_UTC"; echo "camera_start_typed_utc=$START_UTC"; echo "package=$(head -n 1 "$PKG/VERSION")"
  echo "memfree_before_launch_bytes=$FREE"; echo "in_use_before_launch_bytes=$IN_USE"; echo "display_manager_stopped_by_launcher=$DM_STOPPED"; } > "$OBS/launch.txt"
if [ -n "$DEVICE_RECORD" ]; then
  # The run reads its own copy of the record, kept with its records: exactly the bytes that were checked above.
  cp -- "$DEVICE_RECORD" "$OBS/device-artifacts.json" && chmod 600 "$OBS/device-artifacts.json" || stop_now "the record could not be copied into $OBS"
  [ "$(sha256sum "$OBS/device-artifacts.json" | cut -d' ' -f1)" = "$RECORD_SHA" ] || stop_now "the device-artifacts record changed after it was checked: rerun start.sh"
  RUN_ROUTE=(--device-artifacts "$OBS/device-artifacts.json")
  { echo "route=device_artifacts_experimental_validation_run"; echo "device_artifacts_record_sha256=$RECORD_SHA"; echo "artifact_verification=passed"
    echo "resource_validation=pending"; echo "platform_preflight=$PLATFORM_STATE"; } >> "$OBS/launch.txt"
fi

CAMERA_STARTED=1
trap '' HUP QUIT TSTP  # inherited: a lost terminal or a stray key must never separate the guard from the camera
trap ':' INT TERM      # this script keeps waiting for the guard; the guard and the runtime keep their own handling
env -u SENTINEL_RTSP_URL -u SENTINEL_IDENTITY_PASSPHRASE /usr/bin/env python3 -I -B "$PKG/lib/features.py" "$OBS" & FEATURES=$!
# which processes this run starts (they descend from its guard), recorded while it is alive: the only proof of
# ownership afterwards, since the runtime and the model server each start a session of their own
env -u SENTINEL_RTSP_URL -u SENTINEL_IDENTITY_PASSPHRASE /usr/bin/env python3 -I -B "$PKG/lib/beta.py" watch-run \
  --guard-pidfile "$OBS/guard.pid" --out "$OBS/guard" --record "$OBS/processes.txt" & WATCH=$!
echo "Starting: the models load first (1-2 minutes), then the camera opens. The dashboard shows 'starting' until then."
echo "To stop early: Ctrl-C ONCE here. Nothing more appears until OBS_exit (up to about 2 minutes); press no other key."
echo "Start nothing else on this device until OBS_exit: extra memory use can trigger the guard's stop."
echo "Feature controls key, once the run is ready (it is deleted at the stop); from another SSH shell: cat $OBS/db/control-key"
bash -c 'printf %s "$$" > "$1"; shift; exec "$@"' observe-guard "$OBS/guard.pid" \
  "${G[@]}" --out "$OBS/guard" --require-ready face_chat_windows --scene-layers 25/25 --duration-s "$DURATION_S" --ready-timeout-s 300 -- \
  "${RT[@]}" run "$CONFIG" --data-dir "$OBS/db" "${FLAGS[@]}" "${RUN_ROUTE[@]}" > "$OBS/guard-console.log" 2>&1
GX=$?
trap '' INT TERM       # nothing cuts the report short
unset SENTINEL_RTSP_URL SENTINEL_IDENTITY_PASSPHRASE
kill "$FEATURES" "$WATCH" 2>/dev/null; wait "$FEATURES" "$WATCH" 2>/dev/null
rm -f -- "$OBS/guard.pid"

RESULT="$(python3 -I -B - "$OBS/guard/result.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("status=no_result_written camera_opened=unknown clean=no"); sys.exit()
child = r.get("child") or {}
ready = r.get("readiness") or {}
guard = r.get("guard") or {}
status = r.get("status") or ""
no_start = ready.get("reason") == "no_starting_line"
during_loading = status == "operator_interrupt" and no_start and child.get("returncode") in (130, -2)
clean = r.get("cleanup_clear") is True and (during_loading or (
    status in ("operator_interrupt", "duration_stop") and child.get("returncode") == 0 and ready.get("met") is True))
opened = "no" if status.startswith("refused:") or no_start else "yes"
print(f"status={status} returncode={child.get('returncode')} ready={ready.get('met')} ready_reason={ready.get('reason')} "
      f"run_error={r.get('run_error_label')} memory_stop={(guard.get('trigger') or {}).get('stop')} "
      f"peak_pressure_bytes={guard.get('peak_pressure_bytes')} cleanup_clear={r.get('cleanup_clear')} camera_opened={opened} "
      f"clean={'yes' if clean else 'no'}")
PY
)"

# The guard is gone: nothing this run started may be left running without it (a refused guard started nothing).
# Only what this run's own record names, or a runtime whose arguments name this run's data directory, is signalled;
# any other process, whatever its name, is reported and left alone. This package's guard (changed after acceptance:
# VERSION) has already stopped the llama-server its own run started, and no other (QUICKSTART.md, 'Operating limits').
RESIDUE=none
case "$RESULT" in
  status=refused:*) ;;
  *)
    python3 -I -B "$PKG/lib/beta.py" after-guard --record "$OBS/processes.txt" --data-dir "$OBS/db" \
      --verdict "$OBS/after-guard.txt" --stop-script "$PKG/stop.sh" || echo "The check for leftover processes failed (above)."
    RESIDUE="$(sed -n 's/^RESIDUE=//p' "$OBS/after-guard.txt" 2>/dev/null | head -n 1)"; RESIDUE="${RESIDUE:-unknown}"
    [ "$RESIDUE" != still_running ] && [ "$RESIDUE" != unknown ] || KEEP_DESKTOP_DOWN=1 ;;
esac

fingerprint > "$OBS/identity-after.txt" 2>/dev/null
if cmp -s "$OBS/identity-before.txt" "$OBS/identity-after.txt"; then ENROLL=unchanged; else ENROLL=CHANGED; fi
case "$RESULT" in *clean=yes) CODE=0 ;; *) CODE=1 ;; esac
case "$RESULT" in
  status=guard_stop*) ENDED_BY="the guard (a memory or sampling limit; see memory_stop above)" ;;
  status=operator_interrupt*) ENDED_BY="you (Ctrl-C or stop.sh)" ;;
  status=duration_stop*) ENDED_BY="the 60-minute limit" ;;
  *) ENDED_BY="see status above" ;;
esac
echo
echo "Observation ended: $RESULT"
echo "ended_by=$ENDED_BY"
echo "runtime_after_guard=$RESIDUE"
echo "enrollment=$ENROLL"
[ -z "$DEVICE_RECORD" ] || echo "route=device artifacts (experimental validation run on this device's own files; resource_validation=pending: this run measured and validated nothing)"
echo "records: $OBS (per-face numbers from the run are in guard/run.jsonl; no pictures; delete the folder when you no longer need it)"
{ echo "$RESULT"; echo "ended_by=$ENDED_BY"; echo "runtime_after_guard=$RESIDUE"; echo "enrollment=$ENROLL"; echo "guard_exit=$GX"; } >> "$OBS/launch.txt"
[ "$ENROLL" = unchanged ] || CODE=1
[ "$RESIDUE" = none ] || CODE=1
echo "OBS_exit=$CODE"
exit "$CODE"
