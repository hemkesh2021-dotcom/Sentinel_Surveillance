# Privacy counts for the V2-25 device check's operator blocks (bash; sourced from the repository root).
#
# Every search term reaches face_privacy.py on stdin, written by bash's printf builtin, so no process ever has a
# term in its command line, and nothing here prints a term. Each function writes "<file> <key>=<n>" lines on stdout
# (face_check reads them) and exits 2 with nothing on stdout if a term it needs is missing.
# Variables read: SENTINEL_IDENTITY_PASSPHRASE (F1c/F2p), N1 and N2 (FN), PHOTO_NAMES (F1b), SENTINEL_RTSP_URL (F2b).

FACE_PY=(.venv/bin/python)  # the repository venv (tests point it at their interpreter)

face_counts_f1() {  # F1_DIR AUDIT_FILE: the passphrase, the names and the photo file names in F1's evidence (F1f)
  { printf 'passphrase=%s\0' "${SENTINEL_IDENTITY_PASSPHRASE-}"; printf 'name=%s\0' "${N1-}" "${N2-}"
    printf 'filename=%s\0' "${PHOTO_NAMES[@]}"; } \
    | "${FACE_PY[@]}" benchmarks/runner/face_privacy.py f1 "$1" --audit "$2"
}

face_counts_f2() {  # F2_DIR: the passphrase and the names in F2's evidence (F2e)
  { printf 'passphrase=%s\0' "${SENTINEL_IDENTITY_PASSPHRASE-}"; printf 'name=%s\0' "${N1-}" "${N2-}"; } \
    | "${FACE_PY[@]}" benchmarks/runner/face_privacy.py f2 "$1"
}

face_counts_camera() {  # F2_DIR: the camera URL's userinfo in F2's evidence (F2e)
  printf 'url=%s\0' "${SENTINEL_RTSP_URL-}" | "${FACE_PY[@]}" benchmarks/runner/face_privacy.py camera "$1"
}

face_counts_top() {  # DIR: the camera URL's userinfo in a session's top-level files and f1/ (FH); none without a URL
  [ -z "${SENTINEL_RTSP_URL-}" ] \
    || printf 'url=%s\0' "$SENTINEL_RTSP_URL" | "${FACE_PY[@]}" benchmarks/runner/face_privacy.py top "$1"
}

face_recheck_f1() {  # RUN_DIR IDENTITY_DIR RECHECK_DIR PIN: the offline F1 recheck; only the names are rescanned
  printf 'name=%s\0' "${N1-}" "${N2-}" | "${FACE_PY[@]}" benchmarks/runner/face_recheck.py run "$1" \
    --identity-dir "$2" --out "$3" --pin "$4" --commit "$(git rev-parse HEAD)"
}
