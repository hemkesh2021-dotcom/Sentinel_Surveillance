"""Session 115, part 9: the status profile declares every counter the scene worker can produce, each an integer, and
still rejects a counter it does not know.

Session 114 added four counters (a side request waiting for the next gap, or given up) without declaring them, and the
device session fci25-20261026T124349Z's three status captures were nonconforming at components.scene.worker.<key>.
These tests use synthetic statuses and the analyzer's own source, never that session's evidence; the analyzer tests
read the counters a real worker produced through the same profile (evidence_profiles).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from evidence_profiles import evidence, scene_worker_nonconforming

ANALYZER = Path(__file__).resolve().parents[2] / "src" / "sentinel" / "scene" / "analyzer.py"
SESSION_114 = ("side_waits", "side_wait_limit", "side_given_up", "side_plan_failed")


def test_every_counter_the_scene_worker_can_produce_is_declared() -> None:
    source = ANALYZER.read_text()
    names = set(re.findall(r'self\.counters\["([a-z_]+)"\]', source))
    # every counter key in the worker is a literal, so this list is complete
    assert len(re.findall(r"self\.counters\[", source)) == len(re.findall(r'self\.counters\["[a-z_]+"\]', source))
    assert set(SESSION_114) <= names
    vocabulary = evidence().STATUS.keys["components.scene.worker"]
    assert sorted(name for name in names if not vocabulary.fit(name)[0]) == []


def test_the_session_114_counters_fit_beside_the_others(tmp_path) -> None:
    counters = {"submitted": 113, "delivered": 112, "jobs_waited_for_side": 56, "side_grace_started": 56,
                "side_finished_in_grace": 56, "side_waits": 5, "side_wait_limit": 1, "side_given_up": 2,
                "side_plan_failed": 0}
    assert scene_worker_nonconforming(counters, tmp_path) == []


@pytest.mark.parametrize("value", ["5", 5.0, True, None, [5], {"n": 5}])
@pytest.mark.parametrize("counter", SESSION_114)
def test_a_counter_that_is_not_an_integer_is_nonconforming(tmp_path, counter: str, value: Any) -> None:
    assert scene_worker_nonconforming({"submitted": 3, counter: value}, tmp_path) == ["components.scene.worker.*"]


@pytest.mark.parametrize("key", ["side_unknown", "side_waited", "Side_waits", "side waits", ""])
def test_a_counter_the_profile_does_not_know_is_still_nonconforming(tmp_path, key: str) -> None:
    assert scene_worker_nonconforming({"submitted": 3, "side_waits": 1, key: 1}, tmp_path) == [
        "components.scene.worker.<key>"]
