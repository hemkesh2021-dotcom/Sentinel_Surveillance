"""The landmark diagnostic's reading (session 90; landmark_check.py through ``face_check.py landmarks``): the
predeclared analysis on synthetic schema-7 records, and the whole reading on runs of the real runtime with the real
privacy counts (so the evidence profiles are exercised), including F2's refusal of such a run."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from face_runs import FACE, FACE_BOX, real_face_startup, startup_line
from sentinel.contracts import NormalizedBox
from sentinel.identity import landmarks as lm
from sentinel.identity.association import FaceObservation
from sentinel.identity.legacy_deepface import FaceRun, FullScale

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: the tests' own module is not importable by name
from test_face_check import (  # noqa: E402  (the shared scripted run, its finish, and the fixtures it defines)
    f1_list,  # noqa: F401
    finish,
    mods,  # noqa: F401
    simulate,
)

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"


@pytest.fixture
def lc(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("landmark_check")


ACCOUNT10 = {"stayed_seated": "yes", "worked_as_usual": "yes", "looked_at_camera_on_purpose": "unknown",
             "other_faces_in_view": "no", "notes": ""}


def record(vote: str, lateral: float, position: float, *, status: str = "measured", face_iou: float = 0.95) -> dict:
    landmarks = {**dict.fromkeys(lm.RECORD_KEYS), "status": status}
    if status == "measured":
        landmarks.update(score=0.92, face_iou=face_iou, nose_lateral=lateral, nose_vertical=0.55 * position / 0.5,
                         mouth_vertical=1.1, nose_position=position, mouth_lateral=0.01, eye_line_deg=1.0,
                         eye_spacing_px=22.0, eye_spacing_ratio=0.38)
    return {"identity": "result", "outcome": "applied", "frame_mono_ns": 0,
            "persons": [{"track_id": 1, "vote": vote, "face": {"w_px": 58, "h_px": 80, "score": 0.91} if vote != "none"
                         else None, "landmarks": landmarks}]}


def synthetic(*, miss: str = "outside", rate: float = 0.12, evaluation_shift: float = 0.0, misses: int | None = None,
              seed: int = 3) -> dict[str, list[dict]]:
    """One vote a second over W: matches spread over a small range; non-matching votes in episodes of one to three,
    their landmark numbers ``outside`` the matches' range, ``alike`` to the matches', or none at all (``misses=0``)."""
    rnd = random.Random(seed)
    out, t, placed = [], 15.25, 0
    while t < 615.0:
        block = int((t - 15.0) // 60)
        shift = evaluation_shift if block % 2 else 0.0
        if (misses is None or placed < misses) and rnd.random() < rate / 2:
            for _ in range(rnd.choice((1, 2, 3))):
                lateral = rnd.uniform(0.25, 0.35) if miss == "outside" else rnd.uniform(-0.1, 0.1)
                position = rnd.uniform(0.45, 0.55)
                item = record("unknown", lateral + shift, position)
                item["frame_mono_ns"] = round(t * 1e9)
                out.append(item)
                t += 1.0
            placed += 1
            continue
        lateral = rnd.uniform(-0.1, 0.1) if rnd.random() > 0.05 else rnd.uniform(0.25, 0.35)  # a few turned matches
        item = record("match", lateral + shift, rnd.uniform(0.45, 0.55))
        item["frame_mono_ns"] = round(t * 1e9)
        out.append(item)
        t += 1.0
    return {"result": out, "tick": [], "transition": []}


def analysed(lc, records: dict, account: dict | None = ACCOUNT10, executed: bool = True) -> dict:
    items, _ = lc.observations(records, 0)
    return lc.analyse(items, account, executed)


# ---------------------------------------------------------------- the predeclared analysis


def test_misses_concentrated_outside_the_band_read_as_an_association_with_its_interval(lc) -> None:
    result = analysed(lc, synthetic(miss="outside"))
    figures = result["figures"]
    assert result["outcome"].startswith("association: in the evaluation blocks non-matching votes were more frequent "
                                        "outside the reference band than inside it")
    assert figures["reasons"] == [] and figures["risk_difference"] > 0
    low, high = figures["bootstrap"]["risk_difference_interval"]
    assert 0 < low <= figures["risk_difference"] <= high
    assert figures["effect_sizes"]["nose_lateral"]["distance_auc"] > 0.9  # misses far from the matches' median
    assert 0.8 <= figures["band_share_of_matches"]["reference"] <= 0.9  # the construction's share
    assert figures["episodes"]["outside"] >= lc.MIN_EPISODES
    assert "cause" not in result["outcome"] and "pose" not in result["outcome"]  # an association, worded as one


def test_misses_like_the_matches_read_as_no_association_detected_never_as_absence(lc) -> None:
    result = analysed(lc, synthetic(miss="alike", rate=0.2))
    assert result["outcome"].startswith("no association detected at this precision")
    low, high = result["figures"]["bootstrap"]["risk_difference_interval"]
    assert low <= 0 <= high


@pytest.mark.parametrize(("kwargs", "account", "reason"), [
    ({"misses": 3}, ACCOUNT10, "placed miss episodes"),
    ({}, {**ACCOUNT10, "other_faces_in_view": "unknown"}, "precondition"),
    ({}, {**ACCOUNT10, "stayed_seated": "no"}, "precondition"),
    ({}, None, "precondition"),
    ({"evaluation_shift": 0.2}, ACCOUNT10, "unstable: the band holds"),
])
def test_insufficient_preconditions_and_a_band_that_does_not_carry_over_are_inconclusive(lc, kwargs, account,
                                                                                          reason) -> None:
    result = analysed(lc, synthetic(**kwargs), account)
    assert result["outcome"].startswith("inconclusive: ") and reason in result["outcome"]
    assert "association" not in result["outcome"]


def test_too_few_usable_observations_are_inconclusive_and_nothing_is_extended(lc) -> None:
    records = synthetic()
    for item in records["result"][::2]:  # half the votes without a usable landmark observation
        item["persons"][0]["landmarks"] = {**dict.fromkeys(lm.RECORD_KEYS), "status": "no_candidate"}
    result = analysed(lc, records)
    assert result["outcome"].startswith("inconclusive: ") and "coverage" in result["outcome"]
    assert result["figures"]["not_usable"]["evaluation"] == {"no_candidate": pytest.approx(
        result["figures"]["votes"]["evaluation"]["votes"] - result["figures"]["votes"]["evaluation"]["usable"])}


def test_imprecise_and_unstable_bootstraps_are_inconclusive(lc, monkeypatch) -> None:
    monkeypatch.setattr(lc, "MAX_CI_WIDTH", 0.0001)
    assert "imprecise: the risk difference's interval" in analysed(lc, synthetic())["outcome"]
    monkeypatch.setattr(lc, "MAX_CI_WIDTH", 0.4)
    monkeypatch.setattr(lc, "MAX_UNDEFINED_SHARE", -1.0)
    assert "unstable: 0.000 of the bootstrap replicates" in analysed(lc, synthetic())["outcome"]


def test_an_incomplete_execution_is_never_interpreted(lc) -> None:
    result = analysed(lc, synthetic(), executed=False)
    assert result["outcome"] == "not interpreted: execution incomplete (the judged items above)"


def test_blocks_buffers_and_segments_are_assigned_by_frame_time(lc) -> None:
    records = {"result": [], "tick": [], "transition": []}
    for t in (14.9, 15.0, 19.9, 20.0, 31.0, 75.0, 80.0, 135.0, 140.0, 614.9, 615.0):
        item = record("match", 0.0, 0.5)
        item["frame_mono_ns"] = round(t * 1e9)
        records["result"].append(item)
    two = record("match", 0.0, 0.5)
    two["persons"].append({**two["persons"][0], "track_id": 2})
    two["frame_mono_ns"] = round(50 * 1e9)
    records["result"].append(two)
    items, left_out = lc.observations(records, 0)
    assert [(round(o["t"], 1), o["set"], o["segment"]) for o in items] == [
        (15.0, "buffer", None), (19.9, "buffer", None), (20.0, "reference", (0, 0)), (31.0, "reference", (0, 1)),
        (75.0, "buffer", None), (80.0, "evaluation", (1, 0)), (135.0, "buffer", None), (140.0, "reference", (2, 0)),
        (614.9, "evaluation", (9, 4))]
    assert left_out == {"votes_on_results_with_more_than_one_person": 2}


def test_episodes_merge_close_misses_and_are_placed_by_their_first_usable_vote(lc) -> None:
    limits = {"nose_lateral": (-0.1, 0.1), "nose_position": (0.4, 0.6)}

    def vote(t, kind, lateral=0.0, usable=True):
        return {"t": t, "track": 1, "vote": kind, "category": "usable" if usable else "no_candidate",
                "record": {"nose_lateral": lateral, "nose_position": 0.5}}

    votes = [vote(1, "unknown", 0.3, usable=False), vote(2, "unknown", 0.0), vote(3, "unknown", 0.3),  # one, inside
             vote(4, "match"), vote(5, "unknown", 0.3),  # a match between: a new episode, outside
             vote(9, "unknown", 0.0),  # 4 s later: a new episode, inside
             vote(10, "unknown", 0.0, usable=False)]  # merged into the previous one
    assert lc.episodes(votes, limits) == {"inside": 2, "outside": 1}
    assert lc.episodes([vote(1, "unknown", usable=False)], limits) == {"unplaced": 1}


def test_quantiles_aucs_and_the_bootstrap_are_deterministic(lc) -> None:
    assert lc.quantile([1.0, 2.0, 3.0, 4.0], 0.05) == pytest.approx(1.15)  # type 7: 1 + 0.15 * (2 - 1)
    assert lc.quantile([5.0], 0.95) == 5.0
    assert lc.auc([2.0, 2.0], [1.0, 2.0]) == 0.75 and lc.auc([], [1.0]) is None
    first, second = analysed(lc, synthetic()), analysed(lc, synthetic())
    assert first["figures"]["bootstrap"] == second["figures"]["bootstrap"]


# ---------------------------------------------------------------- the whole reading on the real runtime


LIMITS = {"pressure_stop_bytes": 4_800_000_000, "mem_free_floor_bytes": 1_073_741_824,
          "mem_available_floor_bytes": 2_147_483_648, "swap_counters": "unchanged", "max_sample_gap_s": 0.5}
READINESS = {"required": "landmark_diagnostic", "met": True, "reason": None, "starting_after_s": 70.0}
CUES = {"t0": 0.0, "stop": 620.0}
CAPTURES = ((60.0, "c1"), (315.0, "c2"), (610.0, "end"))


@pytest.fixture
def landmark_startup(tmp_path, accepted_scene) -> dict:
    root = tmp_path / "startup-landmarks"
    root.mkdir()
    return startup_line(real_face_startup(root, accepted_scene, landmarks=True))


def scripted(away: tuple[float, ...]):
    """Production's face and the native pass's candidate on each analysed frame: every fifth result from 30 s on a
    turned presentation that does not match, the others upright matches; landmarks from the face box's centre."""
    def analyze_for(t0: int):
        count = {"n": 0}

        def analyze(frame, image):
            count["n"] += 1
            t = (frame.ingest_mono_ns - t0) / 1e9
            turned = t > 30 and count["n"] % 5 == 0
            x, y = (FACE_BOX.x1 + FACE_BOX.x2) / 2 * 640, (FACE_BOX.y1 + FACE_BOX.y2) / 2 * 480 - 10
            spread = ((count["n"] * 7919) % 101) / 100 - 0.5  # a deterministic spread of the upright presentations
            points = lm.Landmarks((FACE_BOX.x1 * 640, FACE_BOX.y1 * 480, (FACE_BOX.x2 - FACE_BOX.x1) * 640,
                                   (FACE_BOX.y2 - FACE_BOX.y1) * 480),
                                  (x - 11, y), (x + 11, y), (x + (8.0 if turned else 2.0 * spread), y + 12),
                                  (x - 9, y + 24), (x + 9, y + 24))
            embedding = away if turned else FACE
            face = FaceObservation(frame=frame.key, box=FACE_BOX, quality=0.91, embedding=embedding)
            full = FullScale(((0.93, NormalizedBox(x1=FACE_BOX.x1, y1=FACE_BOX.y1, x2=FACE_BOX.x2, y2=FACE_BOX.y2)),),
                             47.0, landmarks=(points,))
            return FaceRun((face,), 1, 0, full_scale=full)

        return analyze

    return analyze_for


def run_session(mods, tmp_path, startup, *, account=ACCOUNT10, readiness=READINESS) -> Path:
    from face_runs import vec

    out = simulate(tmp_path, startup, script=((0, 640, "facing", 1),), duration=630.0, captures=CAPTURES,
                   analyze=scripted(vec(0.0, 0.1, 0.99)), landmarks=True)
    parameters = {"duration_s": 630.0, "from_launch": False, "step4_headroom": True,
                  "cues": [{"at_s": at, "name": name} for name, at in CUES.items()],
                  **({"require_ready": readiness["required"]} if readiness else {})}
    finish(mods, out, account=account, cues=CUES, cued_format=10, duration_s=630.0, captures=CAPTURES,
           readiness=readiness, result_update={"parameters": parameters, "limits": LIMITS})
    return out


def reading(mods, out: Path, f1: Path) -> tuple[bool, str]:
    r = mods.check.read_landmarks(out, f1)
    return r.ok, "\n".join(r.lines)


def test_a_complete_session_passes_execution_and_describes_the_analysis(mods, tmp_path, landmark_startup, f1_list) -> None:
    out = run_session(mods, tmp_path, landmark_startup)
    ok, text = reading(mods, out, f1_list)
    assert ok, "\n".join(line for line in text.splitlines() if not line.startswith(("PASS", "DESCRIPTIVE")))
    assert "PASS camera userinfo counts" in text and "PASS passphrase, name and nonconforming counts" in text
    assert "PASS identity records: schema 7 only" in text and "PASS the measured landmarks fit" in text
    outcome = next(line for line in text.splitlines() if line.startswith("OUTCOME"))
    assert outcome.startswith("OUTCOME (descriptive) association: in the evaluation blocks non-matching votes were more "
                              "frequent outside")
    records = [json.loads(line) for line in (out / "run.jsonl").read_text().splitlines()]
    results = [x for x in records if x.get("identity") == "result"]
    assert results and {x["schema"] for x in results} == {7}
    assert all("similarity" not in json.dumps(x.get("persons", [{}])[0].get("landmarks")) for x in results)
    assert "similarity" not in text.split("PASS passphrase")[1]  # nothing after the judged items prints a similarity
    f2 = mods.check.read_f2(out, f1_list)
    assert not f2.ok  # never an F2 reading: the marker keeps such a run out of it
    assert any(line.startswith("FAIL") and "face admitted for this validation run" in line for line in f2.lines)
    assert mods.check.main(["landmarks", str(out), "--f1", str(f1_list)]) == 0


def test_a_run_without_the_marker_or_its_readiness_fails_execution(mods, tmp_path, f1_list, accepted_scene) -> None:
    root = tmp_path / "plain-startup"
    root.mkdir()
    plain = startup_line(real_face_startup(root, accepted_scene))
    out = run_session(mods, tmp_path, plain, readiness=None)
    ok, text = reading(mods, out, f1_list)
    assert not ok
    assert any(line.startswith("FAIL a landmark diagnostic run") for line in text.splitlines())
    assert any(line.startswith("MISSING readiness required before T0") for line in text.splitlines())  # never passes
    assert "OUTCOME (descriptive) not interpreted: execution incomplete" in text


def test_an_unknown_answer_about_other_faces_leaves_the_analysis_inconclusive(mods, tmp_path, landmark_startup,
                                                                             f1_list) -> None:
    out = run_session(mods, tmp_path, landmark_startup, account={**ACCOUNT10, "other_faces_in_view": "unknown"})
    ok, text = reading(mods, out, f1_list)
    assert ok  # execution is complete: unknown is an accepted answer in format 10
    assert "OUTCOME (descriptive) inconclusive: precondition" in text


def test_misses_exactly_the_episode_gap_apart_are_two_episodes(lc) -> None:
    def vote(t):
        return {"t": t, "track": 1, "vote": "unknown", "category": "usable",
                "record": {"nose_lateral": 0.0, "nose_position": 0.5}}

    limits = {"nose_lateral": (-0.1, 0.1), "nose_position": (0.4, 0.6)}
    assert lc.episodes([vote(1.0), vote(1.0 + lc.EPISODE_GAP_S)], limits) == {"inside": 2}
    assert lc.episodes([vote(1.0), vote(0.99 + lc.EPISODE_GAP_S)], limits) == {"inside": 1}
    assert lc.inside(vote(0.0) | {"record": {"nose_lateral": 0.1, "nose_position": 0.4}}, limits)  # edges inclusive


def test_a_measured_face_is_usable_only_beside_the_production_face_it_voted_with(lc) -> None:
    measured = record("match", 0.0, 0.5)["persons"][0]
    assert lc.category(measured["landmarks"], production_face=True) == "usable"
    low = {**measured["landmarks"], "face_iou": 0.49}  # the native face elsewhere than the voted one
    assert lc.category(low, production_face=True) == "face_mismatch"
    assert lc.category({**measured["landmarks"], "face_iou": 0.5}, production_face=True) == "usable"
    assert lc.category({**measured["landmarks"], "face_iou": None}, production_face=False) \
        == "measured_without_production_face"
    assert lc.category({**dict.fromkeys(lm.RECORD_KEYS), "status": "invalid", "reason": "eye_spacing_too_small"},
                       production_face=True) == "invalid:eye_spacing_too_small"
    records = synthetic()
    for item in records["result"]:
        item["persons"][0]["landmarks"]["face_iou"] = 0.3
    figures = analysed(lc, records)["figures"]
    assert figures["coverage"] == {"reference": 0.0, "evaluation": 0.0}
    assert set(figures["not_usable"]["evaluation"]) == {"face_mismatch"}
