"""The multi-pose comparison's cue schedules and windows (PROPOSED: no session is approved or prepared).

Three sessions, built as face_account.RANGE_CUES is (first walk bell 60 s after T0; 30 s between blocks; the last end
bell, then the stop bell 15 s later and the end 15 s after that):

- **B** (the enrolled person alone; the build): a pose sequence at the seat (face the lens, look down at a dark phone)
  and at the 4.0 and 5.0 m marks (face the lens, the left target, the right target): 8 windows, one prototype each.
- **E_M** (the enrolled person alone; held out, another day): 3 facing holds at the seat, then 3 facing and 3 natural
  holds at each mark (V3's holds).
- **E_P** (the participant alone, never enrolled): the same pose sequences, then 3 natural holds at each mark.

A pose sequence: the walk bell (on the spot, face turned away, within 10 s); pose bells 15 s after it and every 20 s;
the end bell 20 s after the last pose bell; then out of view. A pose window runs from its pose bell to the next bell; the
builder takes faces from POSE_SETTLE_S after the pose bell (the turn takes time: cues are not movement times).

Cue names are letters and digits only. ``schedule_digest`` identifies a session's schedule from its result.json cues
(names and scheduled times), so a reader or the builder refuses a session that is not the one it expects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FIRST_GO_S = 60
APPROACH_S = 15  # walk bell -> first pose bell or hold bell
POSE_S = 20
OUT_S = 15
TRANSITION_S = 30
STOP_S = 15
HOLD_S = {"facing": 30, "natural": 60}
HOLDS = 3
POSE_SETTLE_S = 3.0  # after a pose bell, before the builder takes a face from that window

SEAT_POSES = ("lens", "down")
MARK_POSES = ("lens", "left", "right")
POSITIONS = {"s": "seat", "4": "4.0 m mark", "5": "5.0 m mark"}
SPOT_WORDS = {"s": "your seat and sit", "4": "the 4.0 m mark", "5": "the 5.0 m mark"}
POSE_WORDS = {"lens": "face the camera lens", "down": "look down at the dark phone screen",
              "left": "look at the left target", "right": "look at the right target"}
KIND_WORDS = {"facing": "face the camera lens now and hold still until the next bell",
              "natural": "act naturally on the mark (look around, use the phone with no face on its screen, talk) "
                         "until the next bell"}


@dataclass(frozen=True)
class PoseWindow:
    name: str  # e.g. "bs:lens"
    session: str  # "b" or "p"
    spot: str  # "s", "4" or "5"
    pose: str
    bell: str  # the pose bell's cue name
    until: str  # the next bell's cue name


@dataclass(frozen=True)
class Hold:
    name: str  # e.g. "m4n2"
    who: str  # "m" (the enrolled person) or "p" (the participant)
    spot: str
    kind: str  # "facing" or "natural"
    go: str
    hold: str
    end: str


def _poses(spot: str) -> tuple[str, ...]:
    return SEAT_POSES if spot == "s" else MARK_POSES


def _build(blocks: tuple[tuple, ...]) -> tuple[tuple[tuple[str, int, str], ...], tuple[PoseWindow, ...], tuple[Hold, ...]]:
    cues: list[tuple[str, int, str]] = [("t0", 0, "Session started. Leave the camera's view now and stay out of it "
                                                  "until your next bell (1:00).")]
    windows: list[PoseWindow] = []
    holds: list[Hold] = []
    t = FIRST_GO_S
    last_end = 0
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        if block[0] == "pose":
            _, prefix, spot = block
            poses = _poses(spot)
            go = f"{prefix}{spot}go"
            cues.append((go, t, f"Walk to {SPOT_WORDS[spot]}, face turned away from the camera, within 10 s; wait "
                                f"there turned away"))
            names = []
            for k, pose in enumerate(poses):
                name = f"{prefix}{spot}{pose}"
                cues.append((name, t + APPROACH_S + k * POSE_S, f"Now {POSE_WORDS[pose]} until the next bell"))
                names.append(name)
            end = f"{prefix}{spot}end"
            last_end = t + APPROACH_S + len(poses) * POSE_S
            cues.append((end, last_end, "Turn away from the camera and leave its view; stay out until your next "
                                        "bell"))
            for k, pose in enumerate(poses):
                windows.append(PoseWindow(f"{prefix}{spot}:{pose}", prefix, spot, pose, names[k],
                                          names[k + 1] if k + 1 < len(poses) else end))
            t = last_end + (OUT_S if last else TRANSITION_S)
        else:
            _, who, spot, kind, count = block
            for n in range(1, count + 1):
                stem = f"{who}{spot}{kind[0]}{n}"
                cues.append((f"{stem}go", t, f"{POSITIONS[spot].capitalize()} {kind} {n} of {count}: walk to "
                                             f"{SPOT_WORDS[spot]}, face turned away, within 10 s; wait there turned "
                                             f"away"))
                cues.append((f"{stem}hold", t + APPROACH_S, f"{kind.capitalize()} {n}: {KIND_WORDS[kind]}"))
                last_end = t + APPROACH_S + HOLD_S[kind]
                cues.append((f"{stem}end", last_end, "Turn away from the camera and leave its view; stay out until "
                                                     "your next bell"))
                holds.append(Hold(stem, who, spot, kind, f"{stem}go", f"{stem}hold", f"{stem}end"))
                t = last_end + (OUT_S if n < count or last else TRANSITION_S)
    cues.append(("stop", last_end + OUT_S, "All done. Come back to the Mac and type nothing; the run stops by itself "
                                           "in 15 s."))
    return tuple(cues), tuple(windows), tuple(holds)


B_CUES, B_WINDOWS, _ = _build((("pose", "b", "s"), ("pose", "b", "4"), ("pose", "b", "5")))
EM_CUES, _, EM_HOLDS = _build((("holds", "m", "s", "facing", HOLDS), ("holds", "m", "4", "facing", HOLDS),
                               ("holds", "m", "4", "natural", HOLDS), ("holds", "m", "5", "facing", HOLDS),
                               ("holds", "m", "5", "natural", HOLDS)))
EP_CUES, EP_WINDOWS, EP_HOLDS = _build((("pose", "p", "s"), ("pose", "p", "4"), ("pose", "p", "5"),
                                        ("holds", "p", "4", "natural", HOLDS), ("holds", "p", "5", "natural", HOLDS)))
SESSIONS = {"B": B_CUES, "E_M": EM_CUES, "E_P": EP_CUES}


def duration_s(cues: tuple[tuple[str, int, str], ...]) -> int:
    """The run's length: the stop bell, then STOP_S."""
    return dict((name, at) for name, at, _ in cues)["stop"] + STOP_S


def digest(pairs: list[tuple[str, float]]) -> str:
    """SHA-256 over the (name, scheduled seconds) pairs in order: the schedule's identity."""
    canonical = json.dumps([[name, round(float(at), 3)] for name, at in pairs], separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


SCHEDULE_DIGESTS = {kind: digest([(name, at) for name, at, _ in cues]) for kind, cues in SESSIONS.items()}


def schedule_of(result: dict) -> str | None:
    """Which schedule a guard result.json ran ("B", "E_M", "E_P"), from its cues' names and scheduled times; None for
    any other (a missing, extra, renamed or moved cue)."""
    cues = result.get("cues")
    if not isinstance(cues, list):
        return None
    try:
        pairs = [(c["name"], float(c["at_s"])) for c in cues]
    except (KeyError, TypeError, ValueError):
        return None
    found = digest(pairs)
    return next((kind for kind, value in SCHEDULE_DIGESTS.items() if value == found), None)


def printed_ns(result: dict) -> dict[str, int]:
    """Each cue's printed time on the monotonic clock the frames' times use (result.json's printed_mono_s)."""
    return {c["name"]: round(float(c["printed_mono_s"]) * 1e9) for c in result["cues"]}


ACCOUNT_NAME = "mpe-account.txt"
ACCOUNT_FORMAT = "mpe-account-1"


class AccountError(Exception):
    """A fixed label: account_missing or account_invalid."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


def read_account(session: Path, kind: str, allowed: set[str]) -> dict:
    """The session's account (mpe-account-1), the person's own: one key=value per line, exactly
    format, session (B, E_M or E_P), others_in_view (no, yes or unknown) and not_as_cued (``none``, or a comma-separated
    list of this session's window or hold names, from ``allowed``). AccountError otherwise."""
    path = Path(session) / ACCOUNT_NAME
    if not path.is_file():
        raise AccountError("account_missing")
    fields: dict[str, str] = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key in fields:
            raise AccountError("account_invalid")
        fields[key] = value
    if set(fields) != {"format", "session", "others_in_view", "not_as_cued"} or fields["format"] != ACCOUNT_FORMAT \
            or fields["session"] != kind or fields["others_in_view"] not in ("no", "yes", "unknown"):
        raise AccountError("account_invalid")
    listed = set() if fields["not_as_cued"] == "none" else set(fields["not_as_cued"].split(","))
    if not listed <= allowed or "" in listed:
        raise AccountError("account_invalid")
    return {"others_in_view": fields["others_in_view"], "not_as_cued": listed}
