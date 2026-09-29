"""B0 behaviour snapshot of the v1 engine: a reference model for regressions, NOT production code.

Decision D12 (docs/IMPLEMENTATION_STATUS.md). Each piece extracts only the v1
logic a regression needs, from ``surveillance4_1.py`` at commit 2b2d639, and
cites its line ranges. Tests use it to show v1's defective behaviour next to
v2's corrected behaviour on the same timeline. Behaviour is copied as it was,
defects included; do not fix anything here.
"""

from __future__ import annotations

import json


def parse_lfm2_response(raw: str) -> dict:
    """L172-193, unchanged in behaviour. Malformed output becomes a 'none' verdict."""
    default = {
        "persons": 0,
        "activities": [],
        "postures": ["normal"],
        "harmful": False,
        "threat": "none",
        "fire_smoke": False,
        "description": "",
    }
    raw = raw.replace("```json", "").replace("```", "").strip()
    s = raw.find("{")
    e = raw.rfind("}") + 1
    if s == -1 or e <= s:
        default["description"] = raw[:50]
        return default
    try:
        data = json.loads(raw[s:e])
    except Exception:
        default["description"] = raw[s : s + 50]
        return default
    p = data.get("persons", 0)
    default["persons"] = len(p) if isinstance(p, list) else (1 if isinstance(p, dict) else int(p or 0))
    a = data.get("activities", [])
    default["activities"] = [str(x) for x in a] if isinstance(a, list) else [str(a)]
    pos = data.get("postures", ["normal"])
    default["postures"] = [str(x) for x in pos] if isinstance(pos, list) else [str(pos)]
    t = str(data.get("threat") or "none").lower().strip()
    default["threat"] = t if t in ("none", "low", "medium", "high") else "none"
    for key in ("harmful", "fire_smoke"):
        v = data.get(key, False)
        default[key] = v.lower() == "true" if isinstance(v, str) else bool(v)
    d = data.get("description", "")
    default["description"] = str(d)[:60] if d else ""
    return default


class V1SceneResult:
    """The global ``last_ai_result`` and the code that writes and reads it.

    Declared at L196; written by ``lfm2_worker`` L216-248. Any completion
    replaces the global result, whatever frame it was about and however late
    (L235); an exception keeps the previous result (L240-245). There is no
    source frame, epoch or age. Readers: overlay and fire/threat alerts
    L592-614, intruder policy L616-635, status bar L638-647 and dashboard
    L649-656 read it directly.
    """

    def __init__(self) -> None:
        self.last_ai_result: dict = {}

    def worker_returned(self, raw_text: str) -> None:
        self.last_ai_result = parse_lfm2_response(raw_text)

    def worker_raised(self) -> None:
        if not self.last_ai_result:
            self.last_ai_result = {
                "persons": 0,
                "activities": [],
                "postures": ["normal"],
                "harmful": False,
                "threat": "none",
                "fire_smoke": False,
                "description": "error",
            }

    def fire_alert(self) -> bool:
        """L597, L607-609 (ignoring the 30 s cooldown): a fire flag alerts immediately."""
        return bool(self.last_ai_result.get("fire_smoke", False))

    def status_threat(self) -> str:
        """L653: what the dashboard shows as the current threat."""
        return str(self.last_ai_result.get("threat") or "none").lower()


class V1MainLoop:
    """One pass of the main loop ``while running:`` (L477-656), reduced to what R2 needs.

    ``reader.get()`` (FrameReader L392-407) returns the newest frame and keeps
    returning it after the stream stalls or drops; nothing marks it old. When the
    tracker reports nobody, the empty-track branch (L505-513) ``continue``s before
    the scene request (L565) and the dashboard update (L656), so the published
    state keeps showing whoever was there last. There is no freshness state.
    """

    def __init__(self) -> None:
        self.dashboard_persons: int | None = None  # what /tmp/surv_state.json last said
        self.dashboard_updates = 0
        self.scene_requests = 0

    def iteration(self, persons_in_latest_frame: int) -> None:
        if persons_in_latest_frame == 0:
            return  # L505-513
        self.scene_requests += 1  # L565 (request_ai_analysis applies its own interval, L252-258)
        self.dashboard_persons = persons_in_latest_frame  # L649-656
        self.dashboard_updates += 1


FACE_MATCH_THRESH = 0.40  # L39
FACE_MAX_RETRIES = 15  # L76


def _unit(v: tuple[float, ...]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def v1_identify(faces_in_image: list[tuple[float, ...]], face_db: dict[str, tuple[float, ...]]):
    """``identify_face`` L285-307 on synthetic embeddings: (name, score, face_found).

    Only the first face DeepFace returns is used (``result[0]``). With an empty
    database every face becomes "Stranger". Face-size checks are left out.
    """
    if not faces_in_image:
        return "Unknown", 0.0, False
    emb = _unit(faces_in_image[0])
    best_name, best_score = "Stranger", 0.0
    for name, db_emb in face_db.items():
        score = sum(a * b for a, b in zip(emb, _unit(db_emb)))
        if score > best_score:
            best_score, best_name = score, name
    if best_score < FACE_MATCH_THRESH:
        return "Stranger", best_score, True
    return best_name, best_score, True


class V1FaceWorker:
    """``face_recognition_worker`` L335-382 plus the label sync of L520-527 and L570-576.

    Each unverified track tries its padded crop first, then falls back to the
    full-frame result, which is shared by every track whose crop failed (L346,
    L364-366). After FACE_MAX_RETRIES misses a track becomes "Stranger" (L379-380).
    A track counts as a stranger until verified (TrackState L414) and whenever its
    label is "Stranger" or "Unknown" (L524, L575). Session re-ID is left out.
    """

    def __init__(self, face_db: dict[str, tuple[float, ...]]) -> None:
        self.face_db = face_db
        self.verified: dict[int, str] = {}
        self.retries: dict[int, int] = {}

    def run(self, crops: dict[int, list[tuple[float, ...]]], full_frame: list[tuple[float, ...]]) -> None:
        full_name, full_score, full_found = v1_identify(full_frame, self.face_db)
        for tid, crop_faces in crops.items():
            if tid in self.verified:
                continue
            name, score, found = v1_identify(crop_faces, self.face_db)
            if not found and full_found:
                name, score, found = full_name, full_score, True
            if found:
                self.verified[tid] = name
                self.retries.pop(tid, None)
            else:
                self.retries[tid] = self.retries.get(tid, 0) + 1
                if self.retries[tid] >= FACE_MAX_RETRIES:
                    self.verified[tid] = "Stranger"
                    self.retries.pop(tid, None)

    def is_stranger(self, tid: int) -> bool:
        return self.verified.get(tid, "Stranger") in ("Stranger", "Unknown")
