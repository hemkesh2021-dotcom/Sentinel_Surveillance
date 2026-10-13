"""D95: the dashboard's recognition panel, rendered by the server (no browser).

Each tracked person's state as the runtime derived it, with its basis and the time of its newest face result: a
current match is a current claim (D71) and says "was" once the page is not current; a retained identity says it is
retained and on which rule; unknown is never presented as a stranger verdict; unresolved says why in plain words. The
panel is labelled experimental exactly when face recognition runs in a validation run (D92), shows off, unavailable and
not-yet-reported states, marks video that is not fresh, and never shows an identity ID, a similarity or a name.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from status_fixtures import face_row

from sentinel.status_page import build_status, page_document, render_html, status_regions

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
POLICY = {"confirmations": 2, "vote_ttl_s": 30.0, "fresh_window_s": 3.0}


def status(faces: list[dict] | None, *, validation: bool = True, video: str = "fresh", face_state: str = "available",
           problem: str | None = None, more: int = 0) -> dict:
    face = {"state": face_state, "problem": problem}
    live = {"video": video, "detector": "available", "face_recognition": face_state, "scene_analysis": "disabled",
            "occupancy": "occupied" if faces else "empty", "occupancy_reason": "nobody detected on fresh video",
            "people": len(faces or []), "confirmed_people": len(faces or []), "published_utc": NOW.isoformat()}
    if face_state == "available":
        face.update({"validation_run": validation, "identities_enrolled": 1, "policy": POLICY})
        live.update({"faces": faces or [], "faces_not_shown": more})
    snapshot = {"runtime": {"state": "running", "camera_id": "cam-1", "updated_utc": NOW.isoformat()},
                "components": {"capture": {"state": "streaming"}, "detector": {"state": "available"}, "face": face},
                "live": live}
    return build_status(snapshot, {"state": "available"}, NOW)


def panel(document: dict) -> str:
    return status_regions(document)["faces"]


def text(html: str) -> str:
    words = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    return re.sub(r"\s+([.,;)])", r"\1", words)


def test_a_current_match_is_a_current_claim_and_a_retained_one_says_so() -> None:
    html = panel(status([face_row(3, "known", basis="fresh", vote_age_ms=700),
                         face_row(5, "known", basis="retained", vote_age_ms=9400)]))
    fresh, retained = html.split('data-key="p-1-5"')
    assert "<span class=cur><span class=\"badge k-ok\">" in fresh and "Known · current match" in fresh
    assert "Was a current match" in fresh  # what a stale page shows instead (D71)
    assert "Matches an enrolled identity (2 consistent matches; the newest within 3 s)." in text(fresh)
    assert "Known · retained" in retained and "k-ok" not in retained and "class=cur" not in retained
    assert ("Retained from an earlier match with an enrolled identity; no usable face on this track since. Kept on "
            "this track only, until its confirming matches are 30 s old or a different result arrives.") in text(retained)
    assert "Newest face result 0.7 s before this report (about 11:59:59.300 UTC)." in text(fresh)
    assert "Newest face result 9.4 s before this report (about 11:59:50.600 UTC)." in text(retained)


def test_unknown_uncertain_and_unresolved_say_what_they_mean_and_no_more() -> None:
    html = panel(status([face_row(7, "unknown", basis="fresh", vote_age_ms=1200),
                         face_row(8, "uncertain", vote_age_ms=500),
                         face_row(9, "unresolved", reason="no face visible", predicted=True, measured_ms=600),
                         face_row(10, "unresolved", reason="face ownership ambiguous"),
                         face_row(11, "unresolved", reason="face quality too low", status="tentative"),
                         face_row(12, "unresolved", reason="no identities enrolled")]))
    words = text(html)
    assert "Unknown Track" not in words and "Recent faces on this track matched nobody enrolled. Not a stranger verdict." \
        in words
    assert re.findall(r"stranger[^.]*", words) == ["stranger", "stranger verdict"]  # only ever denied
    assert "Recent faces on this track disagree; no identity is shown." in words
    assert "No face is visible on this track. Recognition needs 2 consistent face results." in words
    assert "position predicted, last detected 0.6 s before" in words and "not yet confirmed" in words
    assert "A face could belong to more than one person, so it is given to none." in words
    assert "The face was too small or unclear to compare." in words
    assert "Nobody is enrolled, so no face can be recognized." in words
    assert "No face result on this track yet." in words
    badges = re.findall(r'<span class="badge k-(\w+)"><svg[^>]*><use href="#i-(\w+)"/></svg>([^<]+)</span>', html)
    assert ("off", "help", "Unknown") in badges and ("warn", "warn", "Uncertain") in badges
    assert ("off", "off", "Not recognized") in badges and all(kind != "ok" for kind, _, _ in badges)


def test_the_experimental_label_follows_the_validation_run() -> None:
    assert "Experimental · pending validation" in panel(status([face_row(3, "unresolved")]))
    assert "Validation run: recognition is not validated." in text(panel(status([face_row(3, "unresolved")])))
    admitted = panel(status([face_row(3, "unresolved")], validation=False))
    assert "Experimental" not in admitted and "Validation run" not in admitted


@pytest.mark.parametrize(("face_state", "problem", "expected"), [
    ("disabled", None, "Face recognition is off. This run does not use face recognition."),
    ("unavailable", "no_identities_enrolled", "Face recognition is unavailable: no_identities_enrolled."),
])
def test_off_and_unavailable_are_stated(face_state: str, problem: str | None, expected: str) -> None:
    assert expected in text(panel(status(None, face_state=face_state, problem=problem)))


def test_not_yet_reported_nobody_tracked_stale_video_and_more_people_are_stated() -> None:
    starting = build_status({"runtime": {"state": "starting", "camera_id": "cam-1"}}, {"state": "available"}, NOW)
    assert "Not reported yet. The runtime has not reported face recognition yet." in text(panel(starting))
    assert "Nobody is tracked right now." in text(panel(status([])))
    stale = text(panel(status([face_row(3, "known", basis="retained", vote_age_ms=4000)], video="stale")))
    assert "Video is stale: what is listed is as the runtime last derived it, not the scene now." in stale
    more = text(panel(status([face_row(n, "unresolved") for n in range(8)], more=3)))
    assert "3 more tracked people are not listed (the first 8 by track are)." in more


def test_the_panel_says_not_current_itself_and_carries_no_id_similarity_or_name() -> None:
    row = {**face_row(3, "known", basis="fresh", vote_age_ms=700), "identity_id": "idn-00000000000a",
           "similarity": 0.83, "reason": "<script>alert(1)</script>"}
    html = panel(status([row, face_row(4, "unresolved", reason="<b>Jane</b>")]))
    assert "<div class=context>" in html and "Derived by the runtime at" in html
    assert "NOT CURRENT: last derived at" in html  # the stacked layer a stale page shows (D71)
    assert "idn-" not in html and "0.83" not in html and "<script>" not in html and "<b>" not in html
    assert "Not recognized: &lt;b&gt;Jane&lt;/b&gt;." in html  # an unexpected reason is escaped text, never markup


def test_the_page_embeds_the_same_panel_the_updates_carry() -> None:
    document = status([face_row(3, "known", basis="fresh", vote_age_ms=700)])
    page = render_html(document)
    region = page_document(document, {"instance": "0" * 16, "sequence": 1})["page"]["regions"]["faces"]
    assert region in page and '<section id=faces class="panel faces" aria-labelledby=h-faces>' in page
