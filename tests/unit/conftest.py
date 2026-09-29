from __future__ import annotations

import pytest

from sentinel.media.clock import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
