"""Prints, once, which features this observation run actually started with (from the runtime's own starting line in the
guard's run.jsonl). Standard library only; reads no secret; prints fixed labels and counts, never an identity ID.

usage: python3 -I -B features.py OBS_DIR
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WAIT_S = 420.0


def starting_line(run: Path) -> dict | None:
    try:
        lines = run.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("run") == "starting":
            return record.get("startup") or {}
    return None


def main() -> int:
    obs = Path(sys.argv[1])
    run, result = obs / "guard" / "run.jsonl", obs / "guard" / "result.json"
    deadline = time.monotonic() + WAIT_S
    startup = None
    while time.monotonic() < deadline and startup is None and not result.exists():
        startup = starting_line(run)
        if startup is None:
            time.sleep(1.0)
    if startup is None:
        startup = starting_line(run)
    if startup is None:
        print("features: the runtime reported no starting line (still loading, or it stopped before starting)", flush=True)
        return 0
    say = lambda text: print("  " + text, flush=True)  # noqa: E731
    print("\nFeatures this run started with:", flush=True)
    face, problem = startup.get("face"), startup.get("face_problem")
    if problem or not isinstance(face, dict):
        say(f"face recognition: UNAVAILABLE ({problem or 'not started'})")
    else:
        say(f"face recognition: on (main stream, {face.get('detection')}; EXPERIMENTAL: a validation run, admission "
            f"pending; identities enrolled: {face.get('identities_enrolled')})")
    server, problem = startup.get("scene_server"), startup.get("scene_problem")
    if problem or not isinstance(server, dict) or server.get("state") != "ready":
        say(f"scene analysis: UNAVAILABLE ({problem or 'the scene server is not ready'})")
    else:
        say(f"scene analysis: on (capped LFM3-VL, layers {server.get('layers')}, vision on GPU: {server.get('vision_on_gpu')})")
    chat = startup.get("chat")
    if isinstance(chat, dict) and chat.get("state") == "available":
        say(f"chat: on, EXPERIMENTAL ({chat.get('reason')})")
    else:
        say("chat: UNAVAILABLE (" + str((chat or {}).get("reason") if isinstance(chat, dict) else "not started") + ")")
    detector = startup.get("detector_problem")
    say("person detection: " + (f"UNAVAILABLE ({detector})" if detector else "on"))
    say("live view: " + ("on, in the dashboard (it shows its own live, stale or offline state; nothing is saved)"
                         if startup.get("live_preview") else "UNAVAILABLE"))
    controls = startup.get("feature_controls")
    say("feature controls: " + ("on (" + ", ".join(k for k, v in controls.items() if v) + "); the key is in db/control-key"
                                if isinstance(controls, dict) else "UNAVAILABLE"))
    retention = startup.get("identity_retention")
    say("identity retention: " + (
        f"on, EXPERIMENTAL (PROPOSED values, unvalidated): an established identity is kept on a continuously followed "
        f"track for at most {retention.get('max_since_match_s'):g} s after its last match, and says 'retained'"
        if isinstance(retention, dict) else "OFF (the run did not start with --identity-retention)"))
    say("chat extras, EXPERIMENTAL: recorded-fact answers, follow-ups, Analyse again, automatic refresh (off until "
        "started), zoom into a region and scan (held pictures: memory only, never saved)")
    say("incident snapshots: off; external notifications: none configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
