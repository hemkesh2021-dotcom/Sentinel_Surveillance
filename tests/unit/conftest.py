from __future__ import annotations

from collections.abc import Callable

import pytest

from sentinel.contracts import FrameRef, PixelFormat
from sentinel.media.clock import FakeClock
from sentinel.media.frames import FrameStamper

FRAME_INTERVAL_S = 1 / 15


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stamper(clock: FakeClock) -> FrameStamper:
    stamper = FrameStamper("cam-1", clock)
    stamper.connect()
    return stamper


@pytest.fixture
def next_frame(clock: FakeClock, stamper: FrameStamper) -> Callable[..., FrameRef]:
    """Advance the clock (default: one 15 fps interval) and ingest a 1080p frame."""

    def _next(*, dt: float = FRAME_INTERVAL_S, pts: int | None = None) -> FrameRef:
        clock.advance(dt)
        return stamper.stamp(
            native_width=1920, native_height=1080, pixel_format=PixelFormat.BGR, source_pts=pts
        )

    return _next


SYNTHETIC_PROFILE_ID = "synthetic-test-cache-off"


@pytest.fixture
def accepted_scene(tmp_path):
    """A SYNTHETIC accepted combined profile plus model files that match it, for tests only.

    It is passed explicitly to ``assemble(..., profiles=...)``/``resolve(..., profiles=...)``
    and never enters ``sentinel.adapters.RESOURCE_PROFILES``: it is not a measurement and
    must never become a runtime acceptance record.
    """
    import dataclasses
    from pathlib import Path
    from types import SimpleNamespace

    from sentinel.adapters import STEP4_CRITERIA_ID, ProfileStatus, ResourceProfile
    from sentinel.demo_runtime import SceneOptions, file_facts
    from sentinel.inference.legacy_ultralytics import LEGACY_ENGINE_SHA256
    from sentinel.scene.llama_server import PROFILED_FLAGS, PROMPT_CACHE_FLAGS

    models = Path(tmp_path) / "synthetic-models"
    models.mkdir(exist_ok=True)
    binary, model, mmproj = models / "llama-server", models / "LFM2-VL-1.6B-Q4_0.gguf", models / "mmproj-LFM2-VL-1.6B-Q8_0.gguf"
    for path, size in ((binary, 8), (model, 64), (mmproj, 32)):
        if not path.exists():
            path.write_bytes(b"\0" * size)

    def make(**overrides):
        profile = ResourceProfile(
            profile_id=SYNTHETIC_PROFILE_ID,
            status=ProfileStatus.ACCEPTED,
            run_dir="demo-profile-20991231T235959Z",
            commit="0" * 40,
            boot_id="00000000-0000-4000-8000-000000000000",
            llama_flags=PROFILED_FLAGS + PROMPT_CACHE_FLAGS,
            cache_ram_mib=0,
            scene_interval_s=4.0,
            llm=file_facts(model),
            mmproj=file_facts(mmproj),
            engine_sha256=LEGACY_ENGINE_SHA256,
            criteria_id=STEP4_CRITERIA_ID,
            criteria_passed=True,
            gpu_guard_ok=True,
            prompt_cache_disabled=True,
            peak_bytes=5_100_000_000,
            steady_p95_bytes=5_000_000_000,
            steady_slope_bytes_per_min=1_000_000,
            scene_errors=0,
            note="SYNTHETIC TEST FIXTURE: not a measurement, never a runtime record",
        )
        profile = dataclasses.replace(profile, **overrides)
        manifest = {"adapter_id": "llama-lfm2-vl-scene", "contract_version": 1, "implementation_revision": "1",
                    "enabled": True, "input_kinds": ["frame"], "output_kinds": ["scene.report"],
                    "model_revision": "lfm2-vl-1.6b-q4_0", "resource_profile_id": profile.profile_id,
                    "timeout_ms": 8000}
        return SimpleNamespace(profile=profile, profiles={profile.profile_id: profile}, manifest=manifest,
                               options=SceneOptions(binary, model, mmproj), model=model, mmproj=mmproj)

    return make
