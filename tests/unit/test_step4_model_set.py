"""Session 115: step 4 can measure a scene model set (operator_check --scene-model-set), so that LFM2.5-VL-1.6B can be
profiled for admission (D46, unchanged). Without a set, every earlier command line, identity file set and request is
unchanged. Nothing here starts a process: commands are built and parsed, the workload's request is checked with fakes.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.scene import models
from sentinel.scene.completion import scene_report_grammar
from sentinel.scene.llama_server import SCENE_REQUEST_SHA256, scene_request_body

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runner"
OLD, NEW = models.SCENE_MODEL_SETS["lfm2-vl"], models.SCENE_MODEL_SETS["lfm2.5-vl"]


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return SimpleNamespace(profile=importlib.import_module("demo_profile"),
                           workload=importlib.import_module("demo_workload"),
                           operator=importlib.import_module("operator_check"))


def test_the_scripts_set_keys_are_the_runtimes(mods) -> None:
    assert mods.operator.SCENE_MODEL_SET_KEYS == tuple(sorted(models.SCENE_MODEL_SETS))
    assert mods.profile.DEFAULT_SCENE_REQUEST_MODEL == mods.workload.SCENE_REQUEST_MODEL == OLD.request_model


def test_without_a_set_every_step4_command_is_unchanged_and_with_one_it_adds_its_files_and_request(mods) -> None:
    op, out, clip = mods.operator, Path("/out"), Path("/clip.mp4")
    for build in (op.step4_argv, op.step4plr_argv, op.step4cand_argv):
        before = build(out, clip)
        assert not {"--llama", "--model", "--mmproj", "--scene-request-model", "--scene-template-kwargs"} & set(before)
        assert build(out, clip, None) == before
        added = ["--llama", str(NEW.llama_server), "--model", str(NEW.model_path), "--mmproj", str(NEW.mmproj_path),
                 "--scene-request-model", "lfm2.5-vl"]
        after = build(out, clip, "lfm2.5-vl")
        assert [a for a in after if a not in added] == before and len(after) == len(before) + 8
        at = after.index("--llama")
        assert after[at:at + 8] == added and after[at - 2:at] == ["--min-free-gb", "3.5"]
        args = mods.profile.parse_args(after[2:])
        assert (args.llama, args.model, args.mmproj, args.scene_request_model, args.scene_template_kwargs) == (
            NEW.llama_server, NEW.model_path, NEW.mmproj_path, "lfm2.5-vl", None)
        # session 116: a set with its own build and template arguments adds both
        lfm3 = models.SCENE_MODEL_SETS["lfm3-vl"]
        with_lfm3 = build(out, clip, "lfm3-vl")
        assert len(with_lfm3) == len(before) + 10 and with_lfm3[at:at + 2] == ["--llama", str(lfm3.llama_server)]
        assert with_lfm3[at + 8:at + 10] == ["--scene-template-kwargs", '{"enable_thinking": false}']
        args = mods.profile.parse_args(with_lfm3[2:])
        assert (args.llama, args.scene_request_model, args.scene_template_kwargs) == (
            lfm3.llama_server, "lfm3-vl", '{"enable_thinking": false}')
    defaults = mods.profile.parse_args([])
    assert (defaults.model, defaults.mmproj) == (OLD.model_path, OLD.mmproj_path)  # the rollback set's files


def test_the_identity_snapshot_covers_the_selected_sets_files(mods) -> None:
    clip = Path("/clip.mp4")
    before, new = mods.operator.identity_files(clip), mods.operator.identity_files(clip, "lfm2.5-vl")
    assert set(before) == set(new)  # the same labels: a snapshot of one set never matches a run of the other
    assert (new["model:llm"], new["model:mmproj"]) == (NEW.model_path, NEW.mmproj_path)
    assert (before["model:llm"], before["model:mmproj"]) == (OLD.model_path, OLD.mmproj_path)
    assert {k: v for k, v in before.items() if not k.startswith("model:")} == \
        {k: v for k, v in new.items() if not k.startswith("model:")}  # the same clip, build and engine


def test_the_profiler_passes_the_request_model_to_the_workload_only_when_it_is_not_the_default(mods, monkeypatch,
                                                                                               tmp_path) -> None:
    p = mods.profile
    launched = []
    monkeypatch.setattr(p.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))
    p.start_workload(p.parse_args(mods.operator.step4cand_argv(tmp_path, Path("clip.mp4"))[2:]), tmp_path)
    p.start_workload(p.parse_args(mods.operator.step4cand_argv(tmp_path, Path("clip.mp4"), "lfm2.5-vl")[2:]), tmp_path)
    assert "--scene-request-model" not in launched[0]
    at = launched[1].index("--scene-request-model")
    assert launched[1][at + 1] == "lfm2.5-vl" and launched[1][:at] + launched[1][at + 2:] == launched[0]


def test_the_workloads_request_and_fingerprint_are_the_selected_sets(mods, monkeypatch) -> None:
    w = mods.workload
    assert w.scene_request_sha256() == SCENE_REQUEST_SHA256 == OLD.scene_request_sha256
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, limit):
            return json.dumps({"choices": []}).encode()

    fake_cv2 = SimpleNamespace(resize=lambda frame, size: frame, IMWRITE_JPEG_QUALITY=1,
                               imencode=lambda ext, image, params: (True, SimpleNamespace(tobytes=lambda: b"jpeg")))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(w.urllib.request, "urlopen", lambda request, timeout: sent.append(request) or Response())
    monkeypatch.setattr(w, "SCENE_REQUEST_MODEL", NEW.request_model)
    w.scene_request(18081, object())
    assert json.loads(sent[0].data)["model"] == "lfm2.5-vl"
    # session 115, parts 18 and 20: the whole body is the runtime's for the set, its top-level grammar included
    assert json.loads(sent[0].data) == json.loads(scene_request_body(b"jpeg", NEW.request_model))
    assert json.loads(sent[0].data)["grammar"] == scene_report_grammar()
    assert "json_schema" not in json.loads(sent[0].data)
    assert w.scene_request_sha256() == NEW.scene_request_sha256  # what the runtime compares a profile with


def test_the_workload_takes_a_request_model_name_only(mods) -> None:
    with pytest.raises(SystemExit):
        mods.workload.main(["--scene-only", "--port", "18081", "--scene-request-model", "../x y"])


@pytest.mark.parametrize("argv", [
    ["--scene-model-set", "lfm2.5-vl"],
    ["--execute-workload", "check9", "--scene-model-set", "lfm2.5-vl"],
    ["--execute-workload", "ma1wtd", "--scene-model-set", "lfm2.5-vl"],
    ["--execute-workload", "step4cand", "--scene-model-set", "lfm3"],
])
def test_the_set_is_refused_outside_step4_and_its_identity(mods, argv, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        mods.operator.main(argv)
    assert stop.value.code == 2 and "scene-model-set" in capsys.readouterr().err


def test_the_report_names_the_set_only_when_one_is_given(mods, tmp_path) -> None:
    backend = SimpleNamespace(clock=lambda: 0.0)
    plain = mods.operator.execute("step4cand", backend, tmp_path / "a", dropped_caches=False, clip=None)
    with_set = mods.operator.execute("step4cand", backend, tmp_path / "b", dropped_caches=False, clip=None,
                                     scene_model_set="lfm2.5-vl")
    assert "scene_model_set" not in plain and with_set["scene_model_set"] == "lfm2.5-vl"
