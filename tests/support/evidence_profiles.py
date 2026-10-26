"""The operator's evidence reader (benchmarks/runner/face_evidence.py) on the runtime's own outputs (session 115,
part 9).

A device session failed its privacy and structure counts because the scene worker produced counters that the status
profile did not declare (session 114). These helpers let a runtime test read what it produced through the same
profile, so that a counter added later without its declaration fails here, not on the device.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"


def evidence() -> ModuleType:
    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module("face_evidence")
    finally:
        sys.path.remove(str(RUNNERS))


def scene_worker_nonconforming(counters: dict[str, Any], directory: Path) -> list[str]:
    """The locations face_evidence finds nonconforming in a status capture holding these scene-worker counters."""
    path = directory / "status-end.json"
    path.write_text(json.dumps({"components": {"scene": {"worker": dict(counters)}}}))
    return evidence().read(path, "f2").nonconforming
