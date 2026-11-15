from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.replay import Timeline

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_timeline():
    def _load(name: str) -> Timeline:
        return Timeline.load(FIXTURES / name)

    return _load
