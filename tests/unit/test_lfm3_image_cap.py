"""Session 116, part 7: the LFM3 set's image-token cap, ``--image-max-tokens 256``.

fci3-20261028T133616Z ran with chat's gate closed from readiness (4.75 GB); the largest LFM3-specific allocation found
is the vision encoder's compute buffer, 372.08 MiB, which the pinned build (llama.cpp 19cce69) reserves for a
1,472x1,472 warm-up image. The cap lowers that warm-up to 512x512. What the tests establish, portably:
- **Propagation:** the set declares the cap after its log level; the runtime's launch, the profiler's launch (from the
  operator tool's arguments) and admission's expected flags are the same tuple; malformed caps are refused.
- **Profiles:** a profile matches the capped launch only if it measured exactly these flags; D106's (uncapped), a
  different cap, a missing log level or another order do not; no registry profile matches it, so admission refuses
  the set (D46) until a step-4 run of it is accepted. Criterion C admits the declared pairs and nothing else.
- **Preprocessing:** every path that sends an image to llama-server resizes it to IMAGE_SIZE (480x360) first, and
  the build's resize gives that image 480x352, 165 tokens, with and without the cap: what the model sees does not
  change. An unresized 640x480 frame would change (300 -> 234 tokens), which is why every path is checked.
On this device only (skipped elsewhere): the port of the build's rules is checked against the pinned build's source
and the projector's own metadata (its file header only). Nothing here starts a server or loads a model; the memory
saving is not measured here (an estimate until a step-4 run).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.adapters import RESOURCE_PROFILES
from sentinel.config import parse_config
from sentinel.demo_runtime import SceneOptions, profile_mismatch, scene_admission
from sentinel.memory_policy import CANDIDATE_POLICY
from sentinel.scene import models
from sentinel.scene.llama_server import (
    IMAGE_SIZE,
    PROFILED_FLAGS,
    PROMPT_CACHE_FLAGS,
    image_max_tokens_flags,
    llama_server_command,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / "benchmarks" / "runner"
LFM3 = models.SCENE_MODEL_SETS["lfm3-vl"]
D106 = RESOURCE_PROFILES["step4cand-lfm3-20261028T111716Z"]
CAPPED_FLAGS = PROFILED_FLAGS + PROMPT_CACHE_FLAGS + ("--log-verbosity", "4", "--image-max-tokens", "256")
# The pinned build's rules (llama.cpp 19cce69): tools/mtmd/mtmd-image.cpp mtmd_image_preprocessor_lfm3vl::smart_resize
# and target_size; tools/mtmd/clip.cpp's PROJECTOR_TYPE_LFM3VL branch; tools/mtmd/clip-model.h's
# set_limit_image_tokens and set_warmup_n_tokens. The projector's metadata (clip.vision.*): patch 16, merge 2,
# min 65,536 px, max 16,777,216 px.
PATCH, MERGE, MIN_PIXELS, MAX_PIXELS = 16, 2, 65_536, 16_777_216
PATCH_AREA = PATCH * PATCH * MERGE * MERGE
LFM3_SOURCE = Path.home() / "llama.cpp-lfm3"


def runner(name: str):
    sys.path.insert(0, str(RUNNERS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(RUNNERS))


def limits(cap: int | None) -> tuple[int, int, int]:
    """(min pixels, max pixels, warm-up side) as clip.cpp's LFM3 branch sets them, with --image-max-tokens ``cap``."""
    max_tokens = min(4096, MAX_PIXELS // PATCH_AREA)
    max_pixels = (cap if cap else max_tokens) * PATCH_AREA  # set_limit_image_tokens: the custom max wins
    side = 46  # set_warmup_n_tokens(46*46): never more than the custom max
    if cap and side * side > cap:
        side = max(1, int(math.sqrt(cap)))
    return MIN_PIXELS, max_pixels, side * PATCH * MERGE


def smart_resize(width: int, height: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """mtmd_image_preprocessor_lfm3vl::smart_resize: Python's round is std::nearbyint's half to even."""
    factor = PATCH * MERGE
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar, w_bar = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return max(factor, w_bar), max(factor, h_bar)


def tokens(size: tuple[int, int]) -> int:
    return (size[0] // (PATCH * MERGE)) * (size[1] // (PATCH * MERGE))


def config(profile_id: str = "step4cand-lfm3-20261028T111716Z"):
    return parse_config({"config_version": 1, "camera": {"id": "cam-1"}, "scene": {"interval_s": 4.0}, "adapters": [
        {"adapter_id": "llama-lfm2-vl-scene", "contract_version": 1, "implementation_revision": "1", "enabled": True,
         "input_kinds": ["frame"], "output_kinds": ["scene.report"], "resource_profile_id": profile_id,
         "timeout_ms": 8000, "model_revision": LFM3.model_revision}]})


def options(tmp_path: Path) -> SceneOptions:
    return SceneOptions(tmp_path / "no-server", tmp_path / "no-model", tmp_path / "no-mmproj", model_set=LFM3.key)


# ---------------------------------------------------------------- propagation


def test_the_set_declares_the_cap_after_its_log_level_and_the_other_sets_none() -> None:
    assert models.LFM3_IMAGE_MAX_TOKENS == LFM3.image_max_tokens == 256
    assert LFM3.server_flags == ("--log-verbosity", "4", "--image-max-tokens", "256")
    for key in ("lfm2.5-vl", "lfm2-vl"):
        assert models.SCENE_MODEL_SETS[key].image_max_tokens is None and models.SCENE_MODEL_SETS[key].server_flags == ()
    assert PROFILED_FLAGS + PROMPT_CACHE_FLAGS + LFM3.server_flags == CAPPED_FLAGS
    argv = llama_server_command(Path("llama-server"), Path("m.gguf"), Path("p.gguf"), 18081, LFM3.server_flags)
    assert tuple(argv[-len(CAPPED_FLAGS):]) == CAPPED_FLAGS
    assert LFM3.scene_request_sha256 == D106.scene_request_sha256  # the request itself is unchanged (D47)


@pytest.mark.parametrize("value", [True, 0, -1, 4097, "256", 256.0])
def test_only_a_cap_from_1_to_4096_makes_flags(value) -> None:
    assert image_max_tokens_flags(None) == () and image_max_tokens_flags(1) == ("--image-max-tokens", "1")
    assert image_max_tokens_flags(4096) == ("--image-max-tokens", "4096")
    with pytest.raises(ValueError):
        image_max_tokens_flags(value)


def test_the_operator_tool_and_the_profiler_launch_exactly_the_runtimes_flags() -> None:
    op, profile = runner("operator_check"), runner("demo_profile")
    argv = op.scene_model_args("lfm3-vl")
    assert argv[-2:] == ["--llama-image-max-tokens", "256"]
    args = profile.parse_args([*argv, "--llama-cache-ram", "0"])
    assert tuple(profile.llama_flags(args)) == CAPPED_FLAGS  # the profile then records what admission compares
    assert "--image-max-tokens" not in profile.llama_flags(profile.parse_args([]))  # not passed: nothing added
    for bad in ("0", "4097", "x", "-5"):
        with pytest.raises(SystemExit):
            profile.parse_args(["--llama-image-max-tokens", bad])


def test_the_profilers_log_keeps_the_vision_warmup_size_as_numbers_only() -> None:
    import io

    profile = runner("demo_profile")
    private = "/home/someone/secret prompt text"
    server = SimpleNamespace(stdout=io.BytesIO((
        "clip_model_loader: " + private + "\n"
        "get_dummy_batch: warmup with image size = 512 x 512 " + private + "\n"
        "alloc_compute_meta:      CUDA0 compute buffer size =    45.00 MiB\n").encode()))
    output = io.BytesIO()
    profile.pump_llama_diagnostics(server, output)
    assert output.getvalue() == (b"diagnostic: vision warmup image size = 512 x 512\n"
                                 b"diagnostic: buffer size = 45.00 MiB\n")


# ---------------------------------------------------------------- profiles and admission


def test_a_profile_matches_the_capped_launch_only_if_it_measured_exactly_its_flags(tmp_path) -> None:
    def flags_problem(flags: tuple[str, ...]) -> str:
        profile = dataclasses.replace(D106, llama_flags=flags)
        return profile_mismatch(profile, config(), options(tmp_path), policy=CANDIDATE_POLICY) or ""

    assert "measured llama-server flags" not in flags_problem(CAPPED_FLAGS)  # flags match (identity checks follow)
    base = PROFILED_FLAGS + PROMPT_CACHE_FLAGS
    for flags in (D106.llama_flags,  # D106: measured without the cap
                  base + ("--log-verbosity", "4", "--image-max-tokens", "512"),
                  base + ("--image-max-tokens", "256"),
                  base + ("--image-max-tokens", "256", "--log-verbosity", "4"),
                  base + ("--log-verbosity", "4", "--image-max-tokens", "256", "--image-max-tokens", "256"),
                  base):
        assert "measured llama-server flags" in flags_problem(flags), flags
    assert D106.llama_flags != CAPPED_FLAGS


def test_no_registry_profile_admits_the_capped_launch_so_admission_refuses_it(tmp_path) -> None:
    for profile_id, profile in RESOURCE_PROFILES.items():
        assert tuple(profile.llama_flags) != CAPPED_FLAGS, profile_id
    assert LFM3.profiled is False
    for profile_id in RESOURCE_PROFILES:
        problem, revision = scene_admission(config(profile_id), options(tmp_path), policy=CANDIDATE_POLICY)
        assert problem is not None and revision is None, profile_id


@pytest.mark.parametrize("extra, ok", [
    ((), True),
    (("--log-verbosity", "4"), True),
    (("--image-max-tokens", "256"), True),
    (("--log-verbosity", "4", "--image-max-tokens", "256"), True),
    (("--log-verbosity", "4", "--image-max-tokens", "4096"), True),
    (("--image-max-tokens", "256", "--log-verbosity", "4"), False),  # another order
    (("--log-verbosity", "4", "--image-max-tokens", "256", "--image-max-tokens", "256"), False),
    (("--log-verbosity", "4", "--image-max-tokens"), False),  # no value
    (("--log-verbosity", "4", "--image-max-tokens", "0"), False),
    (("--log-verbosity", "4", "--image-max-tokens", "4097"), False),
    (("--log-verbosity", "4", "--image-max-tokens", "0256"), False),
    (("--log-verbosity", "4", "--image-max-tokens", "-1"), False),
    (("--log-verbosity", "4", "--image-min-tokens", "64"), False),  # anything else
    (("--log-verbosity", "4", "--image-max-tokens", "256", "--no-warmup"), False),
])
def test_criterion_c_admits_the_declared_pairs_in_order_and_nothing_else(extra, ok) -> None:
    criteria = runner("step4_criteria")
    assert criteria.measured_flags_ok(criteria.EXPECTED_LLAMA_FLAGS + extra) is ok
    assert criteria.measured_flags_ok(extra + criteria.EXPECTED_LLAMA_FLAGS) is (extra == ())  # the pairs go last


# ---------------------------------------------------------------- what the model is shown


def test_every_image_submission_path_resizes_to_image_size_before_encoding(monkeypatch) -> None:
    """Every path that sends an image to llama-server: the runtime's scene and chat requests (and incident jobs, the
    same analyzer), the step-4 workload (replay clip and scene-only noise alike) and the operator check tools."""
    resized = []

    def resize(frame, size):
        resized.append(tuple(size))
        return frame

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        resize=resize, IMWRITE_JPEG_QUALITY=1,
        imencode=lambda ext, image, params: (True, SimpleNamespace(tobytes=lambda: b"\xff\xd8x\xff\xd9"))))
    from sentinel import cli
    from sentinel.scene.chat import LlamaChatRequest
    from sentinel.scene.llama_server import LlamaSceneRequest, opencv_jpeg

    assert IMAGE_SIZE == (480, 360)
    opencv_jpeg(object())
    assert resized == [IMAGE_SIZE]
    for cls in (LlamaSceneRequest, LlamaChatRequest):  # the runtime's requests, as the CLI builds them
        assert inspect.signature(cls.__init__).parameters["encode"].default is opencv_jpeg
    assert cli._scene_request(18081, 8.0, LFM3)._encode is opencv_jpeg
    assert cli._chat_request(18081, 8.0, LFM3)._encode is opencv_jpeg
    workload = runner("demo_workload")
    assert workload.VLM_IMAGE_SIZE == IMAGE_SIZE and workload.SYNTHETIC_SHAPE == (480, 640, 3)
    sent = []

    class Response:  # a stand-in llama-server reply: nothing is sent anywhere
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b'{"choices": []}'

    monkeypatch.setattr(workload.urllib.request, "urlopen", lambda request, timeout: sent.append(request) or Response())
    resized.clear()
    workload.scene_request(18081, object())  # the clip's frames and the scene-only noise both come through here
    assert resized == [IMAGE_SIZE] and len(sent) == 1
    for name, function in (("scene_grammar_check", "run"), ("scene_grammar_repro", "run"),
                           ("chat_preempt_check", "run")):
        module = runner(name)
        assert inspect.signature(getattr(module, function)).parameters["encode"].default is opencv_jpeg, name
    smoke = (RUNNERS / "scene_smoke.py").read_text()
    assert "jpeg = opencv_jpeg(image)" in smoke and smoke.count("image_url") == 0  # its bodies are the runtime's


def test_the_builds_resize_leaves_sentinels_image_unchanged_and_only_the_warmup_shrinks() -> None:
    default, capped = limits(None), limits(256)
    assert default == (65_536, 4_194_304, 1_472) and capped == (65_536, 262_144, 512)
    for low, high, _ in (default, capped):
        assert smart_resize(*IMAGE_SIZE, low, high) == (480, 352) and tokens((480, 352)) == 165
    assert (default[2] // PATCH) ** 2 == 8_464 and (capped[2] // PATCH) ** 2 == 1_024  # warm-up patches
    assert 480 * 352 < capped[1]  # below the new largest accepted image as well as the old
    # what the cap would change, and why each path's resize matters: an unresized 640x480 substream frame
    assert tokens(smart_resize(640, 480, *default[:2])) == 300 and tokens(smart_resize(640, 480, *capped[:2])) == 234


# ---------------------------------------------------------------- against the pinned build (this device only)


def gguf_metadata(path: Path) -> dict[str, object]:
    """A GGUF file's key/value header (scalars, strings and short arrays); no tensor is read."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with path.open("rb") as handle:
        def read(fmt: str):
            return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))[0]

        def text() -> str:
            return handle.read(read("Q")).decode()

        def value(kind: int):
            if kind in scalar:
                return read(scalar[kind])
            if kind == 8:
                return text()
            assert kind == 9
            item, count = read("I"), read("Q")
            return [value(item) for _ in range(count)]

        assert handle.read(4) == b"GGUF" and read("I") == 3
        read("Q")
        return {text(): value(read("I")) for _ in range(read("Q"))}


source = pytest.mark.skipif(not (LFM3_SOURCE / "tools/mtmd").is_dir() or not LFM3.mmproj_path.exists(),
                            reason="needs this device's pinned LFM3 build source and projector file")


@source
def test_the_port_is_the_pinned_builds_rules_and_the_projectors_metadata() -> None:
    head = subprocess.run(["git", "-C", str(LFM3_SOURCE), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert head.stdout.strip() == models.LFM3_LLAMA_CPP_COMMIT
    image = (LFM3_SOURCE / "tools/mtmd/mtmd-image.cpp").read_text()
    start = image.index("clip_image_size mtmd_image_preprocessor_lfm3vl::smart_resize(")
    body = image[start:image.index("\n}\n", start)]
    for line in ("int h_bar = static_cast<int>(std::nearbyint(h / factor)) * factor;",
                 "if (static_cast<int64_t>(h_bar) * w_bar > max_pixels) {",
                 "const double beta = std::sqrt(h * w / max_pixels);",
                 "h_bar = static_cast<int>(std::floor(h / beta / factor)) * factor;",
                 "} else if (static_cast<int64_t>(h_bar) * w_bar < min_pixels) {",
                 "h_bar = static_cast<int>(std::ceil(h * beta / factor)) * factor;",
                 "return {std::max(factor, w_bar), std::max(factor, h_bar)};"):
        assert line in body, line
    assert ("return smart_resize(original_size.width, original_size.height, hparams.patch_size * hparams.n_merge,\n"
            "                        hparams.image_min_pixels, hparams.image_max_pixels);") in image
    clip = (LFM3_SOURCE / "tools/mtmd/clip.cpp").read_text()
    branch = clip[clip.index("                case PROJECTOR_TYPE_LFM3VL:\n"):]
    branch = branch[:branch.index("} break;")]
    assert ("hparams.set_limit_image_tokens(min_pixels / patch_area, std::min(4096, max_pixels / patch_area));"
            in branch and "hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup" in branch)
    # the cap reaches the hyperparameters before the LFM3 branch reads it: clip_ctx's constructor sets it, and the
    # loader builds that context before load_hparams
    constructor = clip.index("    clip_ctx(clip_context_params & ctx_params) {")
    assert constructor < clip.index("model.hparams.custom_image_max_tokens = ctx_params.image_max_tokens;",
                                    constructor) < clip.index("\n    }\n", constructor)
    assert ("ctx_vision = new clip_ctx(ctx_params);\n            loader.load_hparams(ctx_vision->model, "
            "CLIP_MODALITY_VISION);") in clip
    model_h = (LFM3_SOURCE / "tools/mtmd/clip-model.h").read_text()
    for line in ("image_max_pixels = (custom_image_max_tokens > 0 ? custom_image_max_tokens : n_tokens_max) * patch_area;",
                 "if (custom_image_max_tokens > 0 && n_tokens > custom_image_max_tokens) {",
                 "n_tok_per_side = std::max(1, static_cast<int>(std::sqrt(custom_image_max_tokens)));",
                 "warmup_image_size = n_tok_per_side * patch_size * n_merge;"):
        assert line in model_h, line
    server = (LFM3_SOURCE / "tools/server/server-context.cpp").read_text()
    assert "mparams.image_max_tokens = params_base.image_max_tokens;" in server
    meta = gguf_metadata(LFM3.mmproj_path)
    assert (meta["clip.vision.projector_type"], meta["clip.vision.patch_size"], meta["clip.vision.spatial_merge_size"],
            meta["clip.vision.image_min_pixels"], meta["clip.vision.image_max_pixels"]) == (
        "lfm3vl", PATCH, MERGE, MIN_PIXELS, MAX_PIXELS)
