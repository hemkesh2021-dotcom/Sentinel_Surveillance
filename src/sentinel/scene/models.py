"""The scene model sets (session 115): which GGUF files llama-server loads, and what the requests name.

Scene analysis and chat ask the same loopback llama-server, so one selection serves both: no run can analyse scenes
with one model and answer questions with another. A set pins its model and that model's own projector (never another
model's), the name every scene and chat request carries (part of the request fingerprint, D47), the
``model_revision`` its adapter manifest must declare (it reaches every scene evidence record through the producer
revision), and where the files came from. The launch flags, the request shapes, the sampling values, the token caps,
the deadlines and the memory guards are the same for every set.

- ``lfm2-vl``: v1's LFM2-VL-1.6B, Q4_0 with its Q8_0 projector, the model the accepted resource profile
  step4cand-demo-20261007T090339Z measured (D59). Its files are that profile's; its request is the one the profile
  measured (SCENE_REQUEST_SHA256). The rollback: selecting it restores the admitted configuration exactly.
- ``lfm2.5-vl`` (the default since session 115): LiquidAI/LFM2.5-VL-1.6B-GGUF at a pinned revision, Q4_0 with its
  own Q8_0 projector: the same ``lfm2_vl`` architecture and layer count, served by the same llama-server build (b8932,
  whose ``lfm2`` projector code already cites this model's processor configuration). The step-4 candidate run
  step4cand-lfm25-20261026T094626Z measured these files and this request; the maintainer accepted it as resource
  profile step4cand-lfm25-20261026T094728Z under D105 (2026-10-26), with a kernel-coverage exception for that run
  only. D46 is unchanged: the profile admits scene analysis only for the configuration it measured. It measures
  resources only: not recognition accuracy or chat quality, and it does not admit chat (D77). Measured timings of the
  other model are never its own: chat's pace starts from the fallback in every run (chat_budget).

Portable: nothing here reads a model file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..adapters import RESOURCE_PROFILES, FileFacts
from .llama_server import (
    IMAGE_SIZE,
    JPEG_QUALITY,
    MAX_TOKENS,
    MODEL_NAME,
    SCENE_PROMPT,
    SCENE_SYSTEM,
    TEMPERATURE,
    request_fingerprint,
)

HOME = Path.home()
MODELS_DIR = HOME / "models"
_ACCEPTED = RESOURCE_PROFILES["step4cand-demo-20261007T090339Z"]


@dataclass(frozen=True)
class SceneModelSet:
    """One model with its own projector, as scene analysis and chat use it."""

    key: str  # what --scene-model-set selects
    request_model: str  # the "model" every scene and chat request carries
    model_revision: str  # what the scene adapter's manifest must declare (evidence's producer revision)
    directory: Path  # where the files are by default
    model: FileFacts  # name, size, modification time and SHA-256 of the model file
    mmproj: FileFacts  # the same for its own projector
    source: str  # where the files came from
    profiled: bool  # an accepted resource profile measured these files (D46: scene needs one)

    @property
    def model_path(self) -> Path:
        return self.directory / self.model.name

    @property
    def mmproj_path(self) -> Path:
        return self.directory / self.mmproj.name

    @property
    def scene_request_sha256(self) -> str:
        """The fingerprint of the scene request this set sends (D47): the profiled request, with its model name."""
        return request_fingerprint(
            system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model=self.request_model,
        )


ROLLBACK_SCENE_MODEL_SET = "lfm2-vl"
DEFAULT_SCENE_MODEL_SET = "lfm2.5-vl"  # session 115: the migration; profiled by step4cand-lfm25-20261026T094728Z (D105)
LFM25_REPOSITORY = "LiquidAI/LFM2.5-VL-1.6B-GGUF"
LFM25_REVISION = "36fc16bc95133424921bcc3da009e83b2f23ffb5"  # the Hub's main on 2026-10-26; immutable

SCENE_MODEL_SETS: Mapping[str, SceneModelSet] = MappingProxyType({
    ROLLBACK_SCENE_MODEL_SET: SceneModelSet(
        key=ROLLBACK_SCENE_MODEL_SET,
        request_model=MODEL_NAME,
        model_revision="lfm2-vl-1.6b-q4_0",
        directory=MODELS_DIR / "lfm2-vl",
        model=_ACCEPTED.llm,
        mmproj=_ACCEPTED.mmproj,
        source="v1's LFM2-VL-1.6B files, as resource profile step4cand-demo-20261007T090339Z recorded them",
        profiled=True,
    ),
    DEFAULT_SCENE_MODEL_SET: SceneModelSet(
        key=DEFAULT_SCENE_MODEL_SET,
        request_model="lfm2.5-vl",
        model_revision="lfm2.5-vl-1.6b-q4_0",
        directory=MODELS_DIR / "lfm2.5-vl",
        # Sizes and SHA-256 are the Hub's LFS records at LFM25_REVISION, checked after download; the modification
        # times are this device's downloaded copies (startup compares large files by name, size and time only).
        model=FileFacts("LFM2.5-VL-1.6B-Q4_0.gguf", 695_752_480, "2026-10-25T20:33:16+00:00",
                        "8186364a4e7c3ad30f6dd3d3b7a4e0074c77dd91eed6cad5d8be9090ce285804"),
        mmproj=FileFacts("mmproj-LFM2.5-VL-1.6b-Q8_0.gguf", 583_109_888, "2026-10-25T20:37:24+00:00",
                         "2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d"),
        source=f"{LFM25_REPOSITORY} at {LFM25_REVISION}",
        profiled=True,  # D105: resource profile step4cand-lfm25-20261026T094728Z measured these files and this request
    ),
})


def scene_model_set(key: str) -> SceneModelSet:
    """The set ``key`` names; ValueError for any other."""
    try:
        return SCENE_MODEL_SETS[key]
    except KeyError:
        raise ValueError(f"unknown scene model set {key!r} (known: {', '.join(SCENE_MODEL_SETS)})") from None
