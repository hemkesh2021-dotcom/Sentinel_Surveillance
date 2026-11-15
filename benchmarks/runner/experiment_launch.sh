# The launch provenance of the enrollment experiment's evaluation sessions (D91) and of the normal recognition path's
# runs (D102; bash, sourced from the repository root by the packages' launch blocks). face_check reads it against
# face_evidence.F2_PROVENANCE, and the tests write it
# with this same function, so the package's lines and the privacy profile are checked together (session 66: a line
# the package wrote and the profile did not declare failed the capture's reading).

experiment_launch_provenance() {  # PIN BOOT_ID FRAMING LIGHTING F1_GATE_LINES CAPTURE_GATE_LINES: the lines on stdout
  case "$3" in none|changed|unsure) ;; *) echo "STOP: the framing answer is not none, changed or unsure" >&2; return 2;; esac
  case "$4" in daylight|lamps|both|unsure) ;; *) echo "STOP: the room light is not daylight, lamps, both or unsure" >&2; return 2;; esac
  printf '%s\n' "$5" | grep -qx 'f1_gate=passed' || { echo "STOP: the F1 gate's lines did not pass" >&2; return 2; }
  printf '%s\n' "$6" | grep -qx 'capture_gate=passed' || { echo "STOP: the capture gate's lines did not pass" >&2; return 2; }
  printf '%s\n' "f2_launch_utc=$(date -u +%FT%TZ)" "boot_id=$2" "commit=$(git rev-parse HEAD)" "pin_commit=$1" \
    "framing_since_capture=$3" "lighting=$4" "$5" "$6"
}

# D102: the launch provenance of runs on the normal recognition path (production detection, the original gallery
# voting): a non-enrolled participant's session, and F2 and the combined check, whose result holds only for the light
# recorded here. The same lines as above without the capture gate's, since no working gallery is opened.
normal_launch_provenance() {  # PIN BOOT_ID FRAMING LIGHTING F1_GATE_LINES: the lines on stdout
  case "$3" in none|changed|unsure) ;; *) echo "STOP: the framing answer is not none, changed or unsure" >&2; return 2;; esac
  case "$4" in daylight|lamps|both|unsure) ;; *) echo "STOP: the room light is not daylight, lamps, both or unsure" >&2; return 2;; esac
  printf '%s\n' "$5" | grep -qx 'f1_gate=passed' || { echo "STOP: the F1 gate's lines did not pass" >&2; return 2; }
  printf '%s\n' "f2_launch_utc=$(date -u +%FT%TZ)" "boot_id=$2" "commit=$(git rev-parse HEAD)" "pin_commit=$1" \
    "framing_since_capture=$3" "lighting=$4" "$5"
}
