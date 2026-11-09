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
from html import unescape as html_unescape

import pytest
from status_fixtures import face_row

from sentinel.status_page import build_status, page_document, render_html, status_regions

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
POLICY = {"confirmations": 2, "vote_ttl_s": 30.0, "fresh_window_s": 3.0}


def status(faces: list[dict] | None, *, validation: bool = True, video: str = "fresh", face_state: str = "available",
           problem: str | None = None, more: int = 0, policy: dict | None = None, identity: dict | None = None) -> dict:
    face = {"state": face_state, "problem": problem}
    live = {"video": video, "detector": "available", "face_recognition": face_state, "scene_analysis": "disabled",
            "occupancy": "occupied" if faces else "empty", "occupancy_reason": "nobody detected on fresh video",
            "people": len(faces or []), "confirmed_people": len(faces or []), "published_utc": NOW.isoformat()}
    if face_state == "available":
        face.update({"validation_run": validation, "identities_enrolled": 1, "policy": POLICY if policy is None else policy})
        live.update({"faces": faces or [], "faces_not_shown": more, **({"identity": identity} if identity else {})})
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
    # session 68: the overflow counts tracks, unconfirmed ones included, so it never says "people"
    assert "3 more tracks are not listed (the first 8 by track number are)." in more


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


# ---------------------------------------------------------------- track retention (session 116, part 28; PROPOSED)

RETENTION_POLICY = {**POLICY, "retention": {"max_since_match_s": 300.0, "contradiction_votes": 10,
                                            "contradiction_span_s": 10.0, "min_face_width_fraction": 0.03,
                                            "conflict_votes": 2, "conflict_window_s": 30.0, "max_gap_s": 0.5,
                                            "overlap_iou": 0.3, "min_step_iou": 0.4, "experimental": True,
                                            "values": "proposed"}}
RETAINED = "Known — retained from an earlier match"


def association(**change) -> dict:
    return {"state": "retained", "last_match_age_ms": 12_345, "evidence": "unknown", "evidence_age_ms": 1_500,
            "contradictions": 2, "contradiction_limit": 10, "conflicts": 0, "conflict_limit": 2,
            "max_since_match_ms": 300_000, **change}


def row(track: int, state: str, **fields) -> dict:
    return {**face_row(track, state, basis=fields.pop("basis", None), vote_age_ms=fields.pop("vote_age_ms", None),
                       reason=fields.pop("reason", None)),
            "association": fields.pop("kept", None), "association_ended": fields.pop("ended", None), **fields}


def test_a_retained_association_says_retained_with_its_age_evidence_counts_and_bound_never_current() -> None:
    html = panel(status([row(3, "unknown", basis="fresh", vote_age_ms=1500, kept=association())], policy=RETENTION_POLICY))
    words = text(html)
    (person,) = re.findall(r"<li class=person.*?</li>", html, re.DOTALL)
    assert RETAINED in words and "k-ok" not in person and "class=cur" not in person
    assert "current match" not in text(person)  # only a current match says current match
    assert ("Kept on this track from an earlier match, not confirmed by the latest faces. "
            "Latest evidence (1.5 s ago): the newest face result matched nobody enrolled. Reliable contradicting results "
            "so far: 2 (it ends at 10 or more spanning at least 10 s). Kept only while this track stays continuous, at "
            "most 5 min after the last match.") in words
    assert "Last successful match 12.3 s before this report (about 11:59:47.655 UTC)." in words
    assert "No face result on this track yet." not in words  # the raw votes may have expired while it is retained
    assert 'data-association="retained"' in html and 'data-evidence="unknown"' in html and 'data-state="unknown"' in html
    assert "stranger" not in text(person)  # matching nobody is never presented as a stranger verdict
    assert "Experimental · track retention" in words and "Experimental · pending validation" in words


def test_a_different_identity_vote_and_the_absence_of_a_face_are_described_as_such() -> None:
    conflicted = association(evidence="different_identity", evidence_age_ms=800, contradictions=1, conflicts=1)
    words = text(panel(status([row(3, "uncertain", vote_age_ms=800, kept=conflicted)], policy=RETENTION_POLICY)))
    assert RETAINED in words and "the newest face result matched a different enrolled identity" in words
    assert ("Reliable contradicting results so far: 1 (it ends at 10 or more spanning at least 10 s). 1 of 2 consecutive "
            "results for a different enrolled identity.") in words
    away = association(evidence="no_face", evidence_age_ms=900, contradictions=0, last_match_age_ms=31_000)
    words = text(panel(status([row(3, "unresolved", kept=away)], policy=RETENTION_POLICY)))
    assert "Last successful match 31.0 s before this report" in words
    assert "no face was visible on this track in the newest result" in words
    assert "contradicting results so far" not in words  # nothing contradicted it
    none = association(evidence="none", evidence_age_ms=None, contradictions=0)
    assert "Latest evidence: no face result since that match." in text(
        panel(status([row(3, "unresolved", kept=none)], policy=RETENTION_POLICY)))


def test_a_current_association_is_the_existing_current_match_and_says_nothing_retained() -> None:
    html = panel(status([row(3, "known", basis="fresh", vote_age_ms=700, kept=association(state="current"))],
                        policy=RETENTION_POLICY))
    assert "Known · current match" in html and RETAINED not in html and 'data-association="current"' in html
    assert "Kept only while" not in text(html)


def test_an_ended_association_says_why_in_one_line_beside_the_raw_state() -> None:
    html = panel(status([row(3, "unresolved", reason="no face visible",
                             ended={"reason": "max_since_match", "age_ms": 5_000})], policy=RETENTION_POLICY))
    words = text(html)
    assert RETAINED not in words and "Not recognized" in words and 'data-association="none"' in html
    assert ("An earlier established identity on this track ended 5.0 s ago: the time limit after its last successful "
            "match was reached.") in words
    assert 'data-association-ended="max_since_match"' in html
    jumped = text(panel(status([row(3, "unresolved", reason="no face visible",
                                     ended={"reason": "continuity_jump", "age_ms": 0})], policy=RETENTION_POLICY)))
    assert "ended 0.0 s ago: the track's box jumped." in html_unescape(jumped)


def test_a_count_at_its_number_with_the_span_not_reached_does_not_read_as_a_limit_reached() -> None:
    words = text(panel(status([row(3, "unknown", basis="fresh", vote_age_ms=500, kept=association(contradictions=14))],
                              policy=RETENTION_POLICY)))
    assert "Reliable contradicting results so far: 14 (it ends at 10 or more spanning at least 10 s)." in words
    assert "14 of 10" not in words and "10 of 10" not in words


def test_a_row_after_an_end_with_no_current_vote_says_so_and_a_plain_row_keeps_its_own_words() -> None:
    words = text(panel(status([row(3, "unresolved", reason="no face visible",
                                   ended={"reason": "continuity_jump", "age_ms": 0})], policy=RETENTION_POLICY)))
    assert "No current face result on this track since that end." in words
    assert "No face result on this track yet." not in words
    newer = text(panel(status([row(3, "uncertain", vote_age_ms=800,
                                   ended={"reason": "continuity_jump", "age_ms": 5_000})], policy=RETENTION_POLICY)))
    assert "Newest face result 0.8 s before this report" in newer and "No current face result" not in newer
    plain = text(panel(status([face_row(3, "unresolved", reason="no face visible")])))
    assert "No face result on this track yet." in plain and "since that end" not in plain


def test_a_raw_known_badge_beside_an_end_says_it_is_the_raw_rule_still_counting_votes_from_before_it() -> None:
    known = text(panel(status([row(3, "known", basis="fresh", vote_age_ms=700,
                                   ended={"reason": "ambiguous_ownership", "age_ms": 100})], policy=RETENTION_POLICY)))
    assert ("An earlier established identity on this track ended 0.1 s ago: a face could have belonged to more than "
            "one person. The state shown is the raw recognition rule, still counting face results from before that "
            "end.") in known
    unresolved = text(panel(status([row(3, "unresolved", reason="no face visible",
                                        ended={"reason": "ambiguous_ownership", "age_ms": 100})],
                                   policy=RETENTION_POLICY)))
    assert "still counting face results" not in unresolved


def test_the_rules_describe_retention_apart_from_recognition_and_state_its_limitation() -> None:
    words = text(panel(status([row(3, "known", basis="fresh", vote_age_ms=700, kept=association(state="current"))],
                              policy=RETENTION_POLICY)))
    assert "Track retention (experimental, separate from the recognition rule above; PROPOSED values, not validated)." in words
    assert "at most 5 min after its last successful match of that identity: this is not whole-track retention." in words
    assert "at least 10 reliable contradicting results spanning at least 10 s" in words
    assert ("a break in tracking (a detection gap over 0.5 s, its box overlapping its previous box by less than 0.4, or "
            "another detection overlapping it by at least 0.3; overlap is intersection over union, geometry only)") in words
    assert "when the video is no longer live (a stall or a stream reconnect)" in words
    assert "2 consecutive results for a different enrolled identity within 30 s (the recognition rule's vote expiry)" in words
    assert "A non-matching face narrower than 3 % of the frame's width" in words
    assert "Nothing carries over to a new track or a restart." in words


def test_a_plain_runs_rows_and_panel_are_unchanged_and_the_counts_keep_associations_apart() -> None:
    html = panel(status([face_row(3, "known", basis="fresh", vote_age_ms=700)]))
    assert "data-association" not in html and "Track retention" not in html and "track retention" not in html
    with_counts = render_html(status([row(3, "unresolved", ended=None)], policy=RETENTION_POLICY, identity={
        "known": 0, "fresh": 0, "retained": 0, "unknown": 0, "uncertain": 0, "unresolved": 1, "associations_retained": 1}))
    assert "associations retained from an earlier match 1" in with_counts
    plain_counts = render_html(status([], identity={"known": 1, "fresh": 1, "retained": 0, "unknown": 0, "uncertain": 0,
                                                    "unresolved": 0}))
    assert "associations retained" not in plain_counts


def test_no_row_carries_an_identity_id_or_a_similarity() -> None:
    html = panel(status([row(3, "unknown", basis="fresh", vote_age_ms=1500,
                             kept={**association(), "identity_id": "idn-00000000000a", "similarity": 0.83})],
                        policy=RETENTION_POLICY))
    assert "idn-" not in html and "0.83" not in html
