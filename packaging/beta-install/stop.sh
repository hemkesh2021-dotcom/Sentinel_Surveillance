#!/bin/bash
# Sentinel personal beta: stops the running observation from any SSH shell, the same way Ctrl-C does (use it if the
# launch window was lost). It signals only the guard recorded by start.sh, if that process is still that guard, and
# only a runtime started for that observation's own data directory. With nothing running it signals nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
. "$HERE/lib/common.sh"
beta_load_paths "$HERE"
LATEST="$(ls -d "$OBS_ROOT"/obs-*/ 2>/dev/null | sort | tail -n 1)"; LATEST="${LATEST%/}"
if [ -z "$LATEST" ]; then echo "No observation folder exists; nothing was signalled."; exit 0; fi
PIDFILE="$LATEST/guard.pid"
GUARD=""
if [ -s "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  case "$PID" in ''|*[!0-9]*) echo "STOP: $PIDFILE does not hold a process number; nothing was signalled."; exit 1;; esac
  { CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline")"; } 2>/dev/null || CMD=""
  case "$CMD" in *step5_guard.py*"--out $LATEST/guard"*) GUARD="$PID";; esac
fi
if [ -n "$GUARD" ]; then
  kill -INT "$GUARD" || { echo "STOP: the signal could not be sent."; exit 1; }
  echo "Asked $LATEST to stop; waiting for the clean stop (up to about 2 minutes)."
  for _ in $(seq 1 150); do [ -d "/proc/$GUARD" ] || break; sleep 1; done
  if [ -d "/proc/$GUARD" ]; then echo "It is still stopping after 150 s; the guard forces the stop itself at its limit. Check again with: bash $PKG/stop.sh"; exit 1; fi
  echo "The guard has stopped."
fi
# a runtime left without its guard (the guard was killed): one clean stop request
left_runtime() { pgrep -u "$(id -u)" -f -- "--data-dir $LATEST/db" | tr '\n' ' '; }
RUNTIME="$(left_runtime)"
if [ -n "$RUNTIME" ]; then
  echo "A runtime for this observation is still running without its guard; the camera may still be open. Asking it to stop cleanly, once."
  kill -INT $RUNTIME || { echo "STOP: the signal could not be sent."; exit 1; }
  for _ in $(seq 1 100); do [ -n "$(left_runtime)" ] || break; sleep 1; done
  if [ -n "$(left_runtime)" ]; then echo "STOP: it did not stop within 100 s. Still running: $(left_runtime). Force it with: kill -TERM $(left_runtime)"; exit 1; fi
  echo "The runtime has stopped."
elif [ -z "$GUARD" ]; then
  LLAMA="$(pgrep -u "$(id -u)" -x llama-server | tr '\n' ' ')"
  if [ -n "$LLAMA" ]; then echo "No observation is running, but a llama-server is ($LLAMA). If it was this observation's, stop it with: kill -TERM $LLAMA"
  else echo "No observation is running; nothing was signalled."; fi
  exit 0
fi
if [ "$(systemctl is-active display-manager 2>/dev/null || true)" != active ]; then
  echo "If the launch window is gone and this device has a desktop, restart it with: sudo systemctl start display-manager"
fi
echo "Records: $LATEST"
