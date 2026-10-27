"""The scene model sets (session 115; LFM3-VL-2B since session 116): which GGUF files llama-server loads, which
llama.cpp build serves them, and what the requests name and carry.

Scene analysis and chat ask the same loopback llama-server, so one selection serves both: no run can analyse scenes
with one model and answer questions with another. A set pins its model and that model's own projector (never another
model's), the llama.cpp build that serves them, the name every scene and chat request carries and the chat-template
arguments the requests send (both part of the request fingerprint, D47), the ``model_revision`` its adapter manifest
must declare (it reaches every scene evidence record through the producer revision), and where the files came from.
The launch flags, the request shapes, the sampling values, the token caps, the deadlines and the memory guards are
the same for every set. A set is never chosen for another: ``--scene-model-set`` selects one, and a set that no
accepted resource profile measured is refused by admission (D46), not replaced.

- ``lfm3-vl`` (the default since session 116): LiquidAI/LFM3-VL-2B-GGUF at a pinned revision, the Quantization-Aware
  Distillation Q4_0 checkpoint with its own Q8_0 projector. The ``lfm3_vl`` architecture (24-layer LFM3 language
  model, a convolution-and-attention vision encoder, one token per 2x2 block of 16-pixel patches) is served by a
  separate pinned llama.cpp build, master at the merge that added it (19cce69, 2026-10-27); the b8932 build of the
  other sets does not know it. The model thinks by default (``<think>`` opens every answer), so every request of this
  set carries ``chat_template_kwargs: {"enable_thinking": false}``, the checkpoint's own template switch (its "no
  think" mode): the template then prefills an empty thinking block and the sampler pays for no reasoning tokens on the
  real-time path. No accepted profile has measured this set, so ``--scene`` refuses it until a step-4 measurement is
  accepted; LFM2.5-VL's profile is never evidence for it.
- ``lfm2.5-vl``: the explicit fallback. LiquidAI/LFM2.5-VL-1.6B-GGUF at a pinned revision, Q4_0 with its own Q8_0
  projector, served by b8932. The maintainer accepted its measurement as resource profile
  step4cand-lfm25-20261026T094728Z (D105, 2026-10-26); the request has changed since (the grammar field, session 115,
  part 20), so that profile refuses it until a new measurement is accepted.
- ``lfm2-vl``: the rollback, v1's LFM2-VL-1.6B, the model the accepted resource profile step4cand-demo-20261007T090339Z
  measured (D59), served by b8932; the same request change applies.

A resource profile measures resources only: not recognition accuracy or chat quality, and it does not admit chat
(D77). Measured timings of another model are never a set's own: chat's pace starts from the fallback in every run
(chat_budget).

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
# The builds. b8932 (98dc1418e) is the build the accepted profiles measured (D36, kept unchanged); the LFM3 build is
# a separate clone at the commit that merged LFM3 support, built with the same options (session 116).
LLAMA_SERVER_B8932 = HOME / "llama.cpp/build/bin/llama-server"
LLAMA_SERVER_LFM3 = HOME / "llama.cpp-lfm3/build/bin/llama-server"
LFM3_LLAMA_CPP_COMMIT = "19cce6919992fdffd92e0ac152caaf1e2b8f254e"  # ggml-org/llama.cpp master, PR #31252 merged
# LFM3-VL-2B's template switch for its "no think" mode (chat_template.jinja: enable_thinking defaults to true and
# opens a <think> block; false prefills an empty one). Sent in every request of the set, as llama-server's
# chat_template_kwargs; a server-side --reasoning flag would do the same but would change the profiled flags.
LFM3_TEMPLATE_KWARGS: Mapping[str, object] = MappingProxyType({"enable_thinking": False})


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
    llama_server: Path = LLAMA_SERVER_B8932  # the llama.cpp build that serves the set (its own directory's libraries)
    llama_build: str = "b8932 (98dc1418e)"  # that build, for the record
    chat_template_kwargs: Mapping[str, object] | None = None  # sent in every request of the set, when not None

    @property
    def model_path(self) -> Path:
        return self.directory / self.model.name

    @property
    def mmproj_path(self) -> Path:
        return self.directory / self.mmproj.name

    @property
    def scene_request_sha256(self) -> str:
        """The fingerprint of the scene request this set sends (D47): the profiled request, with its model name and
        its template arguments."""
        return request_fingerprint(
            system=SCENE_SYSTEM, prompt=SCENE_PROMPT, image_size=IMAGE_SIZE, jpeg_quality=JPEG_QUALITY,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model=self.request_model,
            chat_template_kwargs=self.chat_template_kwargs,
        )


ROLLBACK_SCENE_MODEL_SET = "lfm2-vl"  # the accepted profile step4cand-demo-20261007T090339Z's model (D59)
FALLBACK_SCENE_MODEL_SET = "lfm2.5-vl"  # the explicit fallback (session 116); never selected automatically
DEFAULT_SCENE_MODEL_SET = "lfm3-vl"  # session 116: the migration; no accepted profile yet (D46 refuses --scene with it)
LFM25_REPOSITORY = "LiquidAI/LFM2.5-VL-1.6B-GGUF"
LFM25_REVISION = "36fc16bc95133424921bcc3da009e83b2f23ffb5"  # the Hub's main on 2026-10-26; immutable
LFM3_REPOSITORY = "LiquidAI/LFM3-VL-2B-GGUF"
LFM3_REVISION = "7c3b4a43c9023e80bf234fec5e826a3365c8dda8"  # the Hub's main on 2026-10-28; immutable
LFM3_MODEL_REVISION = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"  # LiquidAI/LFM3-VL-2B (the template, the configs)

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
    FALLBACK_SCENE_MODEL_SET: SceneModelSet(
        key=FALLBACK_SCENE_MODEL_SET,
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
    DEFAULT_SCENE_MODEL_SET: SceneModelSet(
        key=DEFAULT_SCENE_MODEL_SET,
        request_model="lfm3-vl",
        model_revision="lfm3-vl-2b-qad-q4_0",
        directory=MODELS_DIR / "lfm3-vl",
        # Sizes and SHA-256 are the Hub's LFS records at LFM3_REVISION, checked after download (session 116); the
        # modification times are this device's downloaded copies.
        model=FileFacts("LFM3-VL-2B-QAD-Q4_0.gguf", 1_049_615_648, "2026-10-27T20:12:38+00:00",
                        "621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6"),
        mmproj=FileFacts("mmproj-LFM3-VL-2B-Q8_0.gguf", 203_031_872, "2026-10-27T20:21:37+00:00",
                         "c5c5b58d544a18da919172d2d0368090ea1a4a351b6feb7f8a326a264a5918ab"),
        source=f"{LFM3_REPOSITORY} at {LFM3_REVISION}",
        profiled=False,  # D46: no accepted resource profile measures this set yet; --scene refuses it until one does
        llama_server=LLAMA_SERVER_LFM3,
        llama_build=f"master {LFM3_LLAMA_CPP_COMMIT[:9]} (the LFM3 merge)",
        chat_template_kwargs=LFM3_TEMPLATE_KWARGS,
    ),
})


def scene_model_set(key: str) -> SceneModelSet:
    """The set ``key`` names; ValueError for any other."""
    try:
        return SCENE_MODEL_SETS[key]
    except KeyError:
        raise ValueError(f"unknown scene model set {key!r} (known: {', '.join(SCENE_MODEL_SETS)})") from None
