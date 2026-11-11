"""The personal-beta installation package (packaging/beta-install), exercised in temporary directories.

Synthetic fixtures only: a directory tree that stands in for "/" and stand-in commands (uname, dpkg-query, ldconfig,
python3.10) answer what the scripts ask about the platform; the package is assembled from the working tree as the
archive builder assembles it. No camera, no model, no network, no real Jetson fact, and nothing outside the test's
own temporary directory is read as data or written. A pass here says the scripts behave as written; it does NOT say
that the package installs on a clean Jetson (that needs a device).

Covered: the preflight's verdicts (the verified combination, nearby unverified ones, and each failure with what to
do); installation into custom directories and nowhere else; refusals that protect existing directories; a repeated
installation keeping the configuration, the enrollment and the run records; settings read as text, never executed;
assets fetched, checked and never replaced; uninstall removing only what the installer owns.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from beta_install_fixture import DPKG_REFERENCE, LOADER_REFERENCE, World, said, tree


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


# ---------------------------------------------------------------- preflight

def test_the_preflight_passes_on_the_verified_combination_and_changes_nothing(world: World) -> None:
    before = tree(world.tmp)
    result = world.run("preflight.sh")
    assert result.returncode == 0, said(result)
    assert "PREFLIGHT: this is the verified combination." in result.stdout
    assert "TEST FIXTURE: facts are read from" in result.stdout  # a stand-in is never mistaken for the device
    assert "FAIL" not in result.stdout and "NOT VERIFIED" not in result.stdout
    assert tree(world.tmp) == before


@pytest.mark.parametrize("change,line", [
    (("etc/os-release", 'ID=ubuntu\nVERSION_ID="22.04"\n'), "NOT VERIFIED userland: Ubuntu 22.04"),
    (("etc/nv_tegra_release", "# R36 (release), REVISION: 4.4, GCID: 1\n"), "NOT VERIFIED L4T R36.4.4"),
    (("proc/device-tree/model", "NVIDIA Jetson Orin NX Engineering Reference Developer Kit\0"), "NOT VERIFIED device:"),
])
def test_a_nearby_platform_is_not_verified_and_is_installed_only_when_the_tester_accepts_that(
        world: World, change: tuple[str, str], line: str) -> None:
    world.fact(*change)
    result = world.run("preflight.sh")
    assert result.returncode == 2 and line in result.stdout, said(result)
    assert "It may work; it has not been tried." in result.stdout and "--accept-unverified-platform" in result.stdout
    refused = world.install()
    assert refused.returncode == 1 and "nothing was installed" in said(refused)
    assert not world.prefix.exists() and not world.data.exists()
    accepted = world.install("--accept-unverified-platform")
    assert accepted.returncode == 0, said(accepted)
    assert "Going on with a platform that was NOT verified, as you asked." in accepted.stdout
    assert (world.installed / "start.sh").is_file()


def broken(world: World, case: str) -> None:
    if case == "not_arm":
        world.fixture("arch", "x86_64\n")
    elif case == "not_a_jetson":
        (world.root / "proc/device-tree/model").unlink()
    elif case == "other_board":
        world.fact("proc/device-tree/model", "NVIDIA Jetson Xavier NX Developer Kit\0")
    elif case == "four_gb":
        world.fact("proc/meminfo", "MemTotal:        3900000 kB\n")
    elif case == "jetpack_5":
        world.fact("etc/nv_tegra_release", "# R35 (release), REVISION: 5.0, GCID: 1\n")
    elif case == "jetpack_6_0":
        world.fact("etc/nv_tegra_release", "# R36 (release), REVISION: 3.0, GCID: 1\n")
    elif case == "other_os":
        world.fact("etc/os-release", 'ID=debian\nVERSION_ID="12"\n')
    elif case == "no_cuda":
        world.dpkg({k: v for k, v in DPKG_REFERENCE.items() if k != "cuda-toolkit-12-6"})
    elif case == "tensorrt_8":
        world.dpkg({**DPKG_REFERENCE, "libnvinfer10": "8.6.2.3-1+cuda12.2"})
    elif case == "no_cudnn":
        world.dpkg({k: v for k, v in DPKG_REFERENCE.items() if k != "libcudnn9-cuda-12"})
    elif case == "no_l4t_libcuda":
        (world.root / "usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1").unlink()
    elif case == "no_cudss":
        world.fixture("loader", "".join(f"{name}\n" for name in LOADER_REFERENCE if name != "libcudss.so.0"))
    elif case == "no_openblas":
        world.fixture("loader", "".join(f"{name}\n" for name in LOADER_REFERENCE if name != "libopenblas.so.0"))
    elif case == "no_cryptography":
        world.fact("usr/bin/python3", "#!/bin/sh\nexit 1\n", mode=0o755)
    elif case == "no_python310":
        (world.bin / "python3.10").unlink()
    elif case == "no_cmake":
        (world.bin / "cmake").unlink()
    elif case == "no_tensorrt_binding":
        shutil.rmtree(world.root / "usr/lib/python3.10/dist-packages/tensorrt")
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case,what,todo", [
    ("not_arm", "architecture is 'x86_64', not aarch64", "runs on an NVIDIA Jetson only"),
    ("not_a_jetson", "this is not a Jetson", "use a Jetson Orin Nano developer kit (8 GB)"),
    ("other_board", "device: NVIDIA Jetson Xavier NX", "other boards have another CUDA and TensorRT stack"),
    ("four_gb", "memory: '3900000' kB, below the 8 GB module's", "the 4 GB module cannot run it"),
    ("jetpack_5", "L4T R35.5.0", "needs L4T R36.4.x"),
    ("jetpack_6_0", "L4T R36.3.0", "needs L4T R36.4.x"),
    ("other_os", "userland: 'debian 12'", "use JetPack 6.2.1's Ubuntu 22.04"),
    ("no_cuda", "CUDA toolkit 12.6 is not installed", "sudo apt install nvidia-jetpack"),
    ("tensorrt_8", "TensorRT 8.6.2.3", "are for TensorRT 10.3.0"),
    ("no_cudnn", "cuDNN 9 is not installed", "sudo apt install nvidia-jetpack"),
    ("no_l4t_libcuda", "the L4T CUDA driver library is not at", "nvidia-l4t-cuda"),
    ("no_cudss", "cuDSS (libcudss.so.0) is missing", "sudo apt install libcudss0-cuda-12"),
    ("no_openblas", "OpenBLAS (libopenblas.so.0) is missing", "sudo apt install libopenblas0"),
    ("no_cryptography", "/usr/bin/python3 cannot import cryptography", "sudo apt install python3-cryptography"),
    ("no_python310", "no Python 3.10 interpreter", "SENTINEL_BETA_PYTHON310"),
    ("no_cmake", "the build tool 'cmake' is missing", "sudo apt install build-essential cmake"),
    ("no_tensorrt_binding", "no TensorRT Python binding for Python 3.10", "sudo apt install python3-libnvinfer"),
])
def test_an_unsupported_platform_or_a_missing_prerequisite_fails_with_what_to_do_and_nothing_is_installed(
        world: World, case: str, what: str, todo: str) -> None:
    broken(world, case)
    before = tree(world.tmp)
    result = world.run("preflight.sh")
    assert result.returncode == 1, said(result)
    assert f"FAIL        {what}" in result.stdout and todo in result.stdout, result.stdout
    assert "must be fixed before installing. Nothing was changed." in result.stdout
    assert "sudo" not in "".join(line for line in result.stdout.splitlines() if "what to do" not in line)  # only as advice
    install = world.install("--accept-unverified-platform")  # accepting "unverified" never overrides a failure
    assert install.returncode == 1 and "the preflight failed" in said(install)
    assert tree(world.tmp) == before


def test_the_tensorrt_binding_and_python_may_be_named_by_the_tester(world: World) -> None:
    shutil.rmtree(world.root / "usr/lib/python3.10/dist-packages/tensorrt")
    (world.bin / "python3.10").rename(world.tmp / "my-python3.10")
    binding = world.tmp / "my binding"
    (binding / "tensorrt").mkdir(parents=True)
    (binding / "tensorrt" / "__init__.py").write_text("")
    result = world.install(SENTINEL_BETA_PYTHON310=str(world.tmp / "my-python3.10"), SENTINEL_BETA_TENSORRT_PYTHON=str(binding))
    assert result.returncode == 0, said(result)
    conf = (world.prefix / "install.conf").read_text()
    assert f"PYTHON310={world.tmp / 'my-python3.10'}\n" in conf and f"TENSORRT_PYTHON={binding}\n" in conf


# ---------------------------------------------------------------- installation

def test_installation_goes_into_the_two_named_directories_and_nowhere_else(world: World) -> None:
    before = tree(world.tmp)
    result = world.install()
    assert result.returncode == 0, said(result)
    after = tree(world.tmp)
    new = {key for key in after if key not in before}
    assert new and all(key.startswith(("opt", "var")) for key in new), sorted(new)[:5]  # only the prefix and the data
    assert {key: after[key] for key in before} == before  # the archive, the home directory and the facts: untouched
    assert tree(world.installed) == tree(world.package)  # the installed package is the extracted one, file for file
    assert (world.prefix / "install-manifest.txt").read_text().splitlines() == [
        "sentinel-beta-install-manifest/1", "package", "install.conf", "install-manifest.txt"]
    assert f"DATA_DIR={world.data}\n" in (world.prefix / "install.conf").read_text()
    for private in (world.data, world.data / "config", world.data / "identity", world.data / "identity/inbox",
                    world.data / "observe"):
        assert private.is_dir() and private.stat().st_mode & 0o777 == 0o700, private
    for name in ("live.yaml", "site.conf"):
        assert (world.data / "config" / name).stat().st_mode & 0o777 == 0o600
    assert (world.data / "config/live.yaml").read_text() == (world.package / "templates/live.yaml").read_text()
    assert not (world.data / "identity/gallery.sealed").exists()  # nobody's enrollment comes with the package
    assert not (world.prefix / "venv").exists() and not (world.prefix / "assets").exists()  # nothing downloaded or built
    assert "Nothing was started, and nothing starts by itself." in result.stdout


def test_the_default_directories_are_the_packages_own_names_under_the_home_directory(world: World) -> None:
    (world.home / "sentinel-data" / "identity").mkdir(parents=True)  # another installation's data: not this package's
    (world.home / "sentinel-data" / "identity" / "gallery.sealed").write_text("someone's enrollment")
    before = tree(world.home / "sentinel-data")
    result = world.run("install.sh")
    assert result.returncode == 0, said(result)
    assert (world.home / "sentinel-beta" / "package" / "start.sh").is_file()
    assert (world.home / "sentinel-beta-data" / "config" / "live.yaml").is_file()
    assert tree(world.home / "sentinel-data") == before


@pytest.mark.parametrize("case", ["tampered", "extra_file", "no_sums"])
def test_a_package_that_is_not_exactly_its_checksummed_files_is_never_installed(world: World, case: str) -> None:
    if case == "tampered":
        (world.package / "start.sh").write_text((world.package / "start.sh").read_text() + "\n# edited\n")
    elif case == "extra_file":
        (world.package / "app" / "src" / "sitecustomize.py").write_text("print('not part of the package')\n")
    else:
        (world.package / "SHA256SUMS").unlink()
    result = world.install()
    assert result.returncode == 1 and "STOP:" in said(result), said(result)
    assert not world.prefix.exists() and not world.data.exists()
    expected = {"tampered": "differ from their checksums", "extra_file": "files that are not part of it: app/src/sitecustomize.py",
                "no_sums": "is not a complete package"}[case]
    assert expected in said(result)


@pytest.mark.parametrize("case,expected", [
    ("prefix_is_home", "choose a directory of its own"),
    ("prefix_holds_other_things", "was not made by this installer"),
    ("data_holds_other_things", "already holds something that this installer did not make"),
    ("data_inside_prefix", "must be a directory of its own, outside"),
    ("prefix_inside_data", "must be a directory of its own, outside"),
    ("relative_prefix", "--prefix must be an absolute path"),
])
def test_installation_refuses_directories_that_are_not_its_own_and_changes_nothing(
        world: World, case: str, expected: str) -> None:
    prefix, data = world.prefix, world.data
    if case == "prefix_is_home":
        prefix = world.home
    elif case == "prefix_holds_other_things":
        prefix.mkdir(parents=True)
        (prefix / "notes.txt").write_text("mine")
    elif case == "data_holds_other_things":
        (data / "identity").mkdir(parents=True)
        (data / "identity" / "gallery.sealed").write_text("another installation's enrollment")
    elif case == "data_inside_prefix":
        data = prefix / "data"
    elif case == "prefix_inside_data":
        prefix = data / "prefix"
    elif case == "relative_prefix":
        prefix = Path("relative/prefix")
    before = tree(world.tmp)
    result = world.run("install.sh", "--prefix", str(prefix), "--data-dir", str(data))
    assert result.returncode == 1 and expected in said(result), said(result)
    assert tree(world.tmp) == before


def test_installing_again_refreshes_the_package_and_keeps_configuration_enrollment_and_records(world: World) -> None:
    assert world.install().returncode == 0
    (world.data / "config/live.yaml").write_text("config_version: 1\ncamera: {id: my-camera}\n")
    (world.data / "config/site.conf").write_text("ENGINE=/somewhere/else/yolov8n.engine\n")
    (world.data / "identity/gallery.sealed").write_text("my sealed enrollment")
    (world.data / "identity/audit.jsonl").write_text("{}\n")
    (world.data / "observe/obs-20260101T000000Z").mkdir()
    (world.data / "observe/obs-20260101T000000Z/launch.txt").write_text("a run's record")
    (world.prefix / "venv").mkdir()
    (world.prefix / "venv/marker").write_text("the environment made by setup-env.sh")
    (world.installed / "stale.py").write_text("left from an older package")  # the old copy is replaced as a whole
    stamps = {path: path.stat().st_mtime_ns for path in world.data.rglob("*") if path.is_file()}
    kept = tree(world.data)
    result = world.install()
    assert result.returncode == 0, said(result)
    assert tree(world.data) == kept
    assert {path: path.stat().st_mtime_ns for path in world.data.rglob("*") if path.is_file()} == stamps  # not rewritten
    assert "kept:" in result.stdout and "your configuration is never overwritten" in result.stdout
    assert "an enrollment is there; the installer never reads or changes it" in result.stdout
    assert tree(world.installed) == tree(world.package) and not (world.installed / "stale.py").exists()
    assert (world.prefix / "venv/marker").read_text() == "the environment made by setup-env.sh"
    again_without_naming_the_data = world.run("install.sh", "--prefix", str(world.prefix))
    assert again_without_naming_the_data.returncode == 0  # the recorded data directory is used, not the default
    assert not (world.home / "sentinel-beta-data").exists()
    other = world.run("install.sh", "--prefix", str(world.prefix), "--data-dir", str(world.tmp / "other data"))
    assert other.returncode == 1 and "would leave that data behind" in said(other)
    assert not (world.tmp / "other data").exists() and tree(world.data) == kept


def test_an_installed_copy_cannot_install_itself(world: World) -> None:
    assert world.install().returncode == 0
    result = world.run("install.sh", "--prefix", str(world.tmp / "second"), package=world.installed)
    assert result.returncode == 1 and "Run install.sh from the extracted archive" in said(result)
    assert not (world.tmp / "second").exists()


# ---------------------------------------------------------------- settings

def test_settings_are_shown_changed_only_when_asked_and_read_as_text_never_executed(world: World) -> None:
    assert world.install().returncode == 0
    shown = world.run("configure.sh", package=world.installed)
    assert shown.returncode == 0, said(shown)
    assert f"configuration file: {world.data / 'config/live.yaml'}" in shown.stdout
    assert f"ENGINE        {world.prefix / 'assets/engine/yolov8n.engine'}  [MISSING]" in shown.stdout
    assert "enrollment    MISSING: enroll yourself first" in shown.stdout
    engine = world.tmp / "my engines" / "yolov8n.engine"
    engine.parent.mkdir()
    engine.write_text("x")
    changed = world.run("configure.sh", "--set", f"ENGINE={engine}", package=world.installed)
    assert changed.returncode == 0 and f"ENGINE        {engine}  [present]" in changed.stdout, said(changed)
    assert f"ENGINE={engine}\n" in (world.data / "config/site.conf").read_text()
    back = world.run("configure.sh", "--set", "ENGINE=", package=world.installed)
    assert back.returncode == 0 and f"ENGINE        {world.prefix / 'assets/engine/yolov8n.engine'}" in back.stdout
    for bad, expected in (("ENGINE=relative/engine", "ENGINE must be an absolute path"),
                          ("PASSPHRASE=/x", "--set takes KEY=/absolute/path with KEY one of"),
                          ("ENGINE", "--set takes KEY=/absolute/path")):
        refused = world.run("configure.sh", "--set", bad, package=world.installed)
        assert refused.returncode == 1 and expected in said(refused), said(refused)
    # A value is text: nothing in the settings file is run by any script that reads it.
    trap = world.tmp / "executed"
    (world.data / "config/site.conf").write_text(f"ENGINE=$(touch {trap})/x\nLLAMA_SERVER=`touch {trap}`\n; touch {trap}\n")
    for script in ("configure.sh", "stop.sh"):
        world.run(script, package=world.installed)
    assert not trap.exists()
    assert f"ENGINE        $(touch {trap})/x  [MISSING]" in world.run("configure.sh", package=world.installed).stdout


# ---------------------------------------------------------------- assets

def manifest(world: World, upstream: Path, **changes: Any) -> Path:
    """A synthetic manifest of the real one's shape: two small files served from a local directory."""
    upstream.mkdir(exist_ok=True)
    assets = []
    for asset_id, relative, content in (("scene_model", "models/lfm3-vl/model.gguf", b"model bytes"),
                                        ("face_detector", "deepface-home/.deepface/weights/detector.onnx", b"detector")):
        (upstream / Path(relative).name).write_bytes(content)
        assets.append({"id": asset_id, "kind": "download", "path": relative, "url": (upstream / Path(relative).name).as_uri(),
                       "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "how": "Run: bash fetch-assets.sh"})
    assets += [{"id": "scene_mmproj", "kind": "download", "path": "models/lfm3-vl/mmproj.gguf",
                "url": (upstream / "absent.gguf").as_uri(), "size": 3, "sha256": "0" * 64, "how": "Run: bash fetch-assets.sh"},
               {"id": "llama_server", "kind": "build", "path": "llama.cpp/build/bin/llama-server", "how": "Run: bash build-llama.sh"},
               {"id": "detector_engine", "kind": "build", "path": "engine/yolov8n.engine", "how": "Run: bash build-engine.sh"}]
    for asset in assets:
        asset.update(changes.get(asset["id"], {}))
    path = world.tmp / "assets.json"
    path.write_text(json.dumps({"schema": "sentinel-beta-assets/1", "assets": assets}))
    return path


def beta(world: World, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(world.bin / "python3"), "-I", "-B", str(world.installed / "lib/beta.py"), *args],
                          env=world.env, capture_output=True, text=True, timeout=120)


def test_assets_are_fetched_checked_never_replaced_and_missing_ones_are_named(world: World) -> None:
    assert world.install().returncode == 0
    listing = manifest(world, world.tmp / "upstream")
    missing = beta(world, "verify-assets", "--package", str(world.installed), "--manifest", str(listing))
    assert missing.returncode == 1
    for asset_id, todo in (("scene_model", "Run: bash fetch-assets.sh"), ("llama_server", "Run: bash build-llama.sh"),
                           ("detector_engine", "Run: bash build-engine.sh")):
        assert f"asset {asset_id}: MISSING" in missing.stdout and todo in missing.stdout, missing.stdout
    fetched = beta(world, "fetch-assets", "--package", str(world.installed), "--manifest", str(listing))
    assert fetched.returncode == 1, said(fetched)  # one of the three is not available upstream
    assert "asset scene_model: ok" in fetched.stdout and "asset face_detector: ok" in fetched.stdout
    assert "asset scene_mmproj: the download failed" in fetched.stdout
    model = world.prefix / "assets/models/lfm3-vl/model.gguf"
    assert model.read_bytes() == b"model bytes"
    assert (world.prefix / "assets/deepface-home/.deepface/weights/detector.onnx").read_bytes() == b"detector"
    assert not list((world.prefix / "assets").rglob("*.part"))  # nothing half-downloaded is left
    again = beta(world, "fetch-assets", "--package", str(world.installed), "--manifest", str(listing), "--only", "scene_model")
    assert again.returncode == 0 and "asset scene_model: ok, already there" in again.stdout
    # A file that is there and differs is neither used nor replaced.
    model.write_bytes(b"another file!")
    differs = beta(world, "fetch-assets", "--package", str(world.installed), "--manifest", str(listing), "--only", "scene_model")
    assert differs.returncode == 1 and "it is NOT replaced" in differs.stdout and model.read_bytes() == b"another file!"
    verify = beta(world, "verify-assets", "--package", str(world.installed), "--manifest", str(listing))
    assert verify.returncode == 1 and "asset scene_model: WRONG SIZE" in verify.stdout
    model.write_bytes(b"model byteZ")  # the same size, another content
    assert "asset scene_model: WRONG CHECKSUM" in beta(world, "verify-assets", "--package", str(world.installed),
                                                      "--manifest", str(listing)).stdout


def test_a_download_that_is_not_the_pinned_file_is_deleted_and_a_place_outside_the_installation_is_not_written(
        world: World) -> None:
    assert world.install().returncode == 0
    listing = manifest(world, world.tmp / "upstream", scene_model={"sha256": "1" * 64})
    result = beta(world, "fetch-assets", "--package", str(world.installed), "--manifest", str(listing), "--only", "scene_model")
    assert result.returncode == 1 and "is not the pinned one (WRONG CHECKSUM); it was deleted" in result.stdout
    assert not (world.prefix / "assets/models/lfm3-vl").exists() or not any((world.prefix / "assets/models/lfm3-vl").iterdir())
    own = world.tmp / "my models" / "model.gguf"
    assert world.run("configure.sh", "--set", f"SCENE_MODEL={own}", package=world.installed).returncode == 0
    listing = manifest(world, world.tmp / "upstream")
    outside = beta(world, "fetch-assets", "--package", str(world.installed), "--manifest", str(listing), "--only", "scene_model")
    assert outside.returncode == 1 and "nothing is downloaded outside the installation" in outside.stdout
    assert not own.exists() and not own.parent.exists()


def test_the_packaged_manifest_names_a_source_a_revision_and_a_checksum_for_every_download(world: World) -> None:
    document = json.loads((world.package / "manifest/assets.json").read_text())
    assets = {asset["id"]: asset for asset in document["assets"]}
    assert set(assets) == {"scene_model", "scene_mmproj", "face_weights", "face_detector", "detector_weights",
                           "llama_server", "detector_engine"}
    for asset in assets.values():
        assert asset["pinned_by_accepted_build"] and asset["how"]
        if asset["kind"] == "download":
            assert asset["url"].startswith("https://") and asset["revision"] in asset["url"]
            assert len(asset["sha256"]) == 64 and asset["size"] > 0 and asset["source"] and asset["license"]
    # What the accepted code itself pins, as the manifest repeats it: the two must never drift apart.
    from sentinel.adapters import FACE_ADMISSIONS, RESOURCE_PROFILES
    from sentinel.inference.legacy_ultralytics import LEGACY_ENGINE_SHA256
    from sentinel.scene.models import LFM3_LLAMA_CPP_COMMIT

    profile = RESOURCE_PROFILES["step4cand-lfm3cap-20261028T174508Z"]
    face = FACE_ADMISSIONS["step4cand-demo-20261007T090339Z"]
    for asset_id, facts in (("scene_model", profile.llm), ("scene_mmproj", profile.mmproj),
                            ("face_weights", face.model_weights), ("face_detector", face.detector_weights)):
        assert (Path(assets[asset_id]["path"]).name, assets[asset_id]["size"], assets[asset_id]["sha256"]) == (
            facts.name, facts.bytes, facts.sha256), asset_id
    assert assets["llama_server"]["commit"] == LFM3_LLAMA_CPP_COMMIT
    assert profile.llama_server.sha256 in assets["llama_server"]["pinned_by_accepted_build"]
    assert LEGACY_ENGINE_SHA256 in assets["detector_engine"]["pinned_by_accepted_build"]
    live = (world.package / "templates/live.yaml").read_text()
    assert "step4cand-lfm3cap-20261028T174508Z" in live and "step4cand-demo-20261007T090339Z" in live
    for secret in ("rtsp://", "passphrase:", "password", "token"):
        assert secret not in live.split("\n", 5)[-1]  # the configuration itself holds no address or credential


# ---------------------------------------------------------------- uninstall

def installed_with_everything(world: World) -> None:
    assert world.install().returncode == 0
    for name in ("venv", "assets"):  # as setup-env.sh and fetch-assets.sh do: recorded first, then made
        assert beta(world, "own", "--prefix", str(world.prefix), name).returncode == 0
        (world.prefix / name / "sub").mkdir(parents=True)
        (world.prefix / name / "sub" / "file").write_text(name)
    (world.data / "identity/gallery.sealed").write_text("my sealed enrollment")
    (world.data / "observe/obs-20260101T000000Z").mkdir()
    (world.data / "observe/obs-20260101T000000Z/launch.txt").write_text("a run's record")


def test_uninstall_removes_what_the_installer_owns_and_never_the_testers_data(world: World) -> None:
    installed_with_everything(world)
    beside = world.prefix.parent / "beside the prefix"
    beside.write_text("not the installer's")
    data, outside = tree(world.data), tree(world.home)
    declined = world.run("uninstall.sh", package=world.installed, stdin="\n")
    assert declined.returncode == 1 and "nothing was removed" in said(declined)
    assert (world.installed / "start.sh").is_file() and (world.prefix / "venv/sub/file").is_file()
    result = world.run("uninstall.sh", package=world.installed, stdin="UNINSTALL\n")
    assert result.returncode == 0, said(result)
    assert not world.prefix.exists()
    assert tree(world.data) == data and tree(world.home) == outside and beside.read_text() == "not the installer's"
    assert f"NOT removed, yours to keep or delete: {world.data}" in result.stdout
    assert "sudo" not in said(result)


def test_uninstall_leaves_what_it_did_not_make_and_kept_assets_can_be_removed_later_as_it_says(world: World) -> None:
    installed_with_everything(world)
    (world.prefix / "my notes.txt").write_text("mine")
    data = tree(world.data)
    result = world.run("uninstall.sh", "--keep-assets", "--yes", package=world.installed)
    assert result.returncode == 0, said(result)
    assert sorted(path.name for path in world.prefix.iterdir()) == [
        "assets", "install-manifest.txt", "my notes.txt", "uninstall-helper.py", "uninstall.sh"]
    assert (world.prefix / "assets/sub/file").read_text() == "assets"
    assert f"left in place, not made by this installer: {world.prefix / 'my notes.txt'}" in result.stdout
    assert "kept at your request" in result.stdout
    later = f"To remove it later: bash {world.prefix / 'uninstall.sh'}"
    assert later in result.stdout  # the command it names exists and is what removes the rest
    final = world.run("uninstall.sh", "--yes", package=world.prefix)
    assert final.returncode == 0, said(final)
    assert sorted(path.name for path in world.prefix.iterdir()) == ["my notes.txt"]
    assert tree(world.data) == data


def test_kept_assets_can_be_installed_over_again_and_stay_the_installers(world: World) -> None:
    installed_with_everything(world)
    assert world.run("uninstall.sh", "--keep-assets", "--yes", package=world.installed).returncode == 0
    again = world.install()
    assert again.returncode == 0, said(again)
    assert (world.prefix / "install-manifest.txt").read_text().splitlines()[1:] == [
        "assets", "package", "install.conf", "install-manifest.txt"]
    assert not (world.prefix / "uninstall.sh").exists() and (world.prefix / "assets/sub/file").read_text() == "assets"
    assert world.run("uninstall.sh", "--yes", package=world.installed).returncode == 0
    assert not world.prefix.exists()


def test_uninstall_removes_a_link_as_a_link_and_refuses_a_manifest_it_did_not_write(world: World) -> None:
    installed_with_everything(world)
    elsewhere = world.tmp / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "big model.gguf").write_text("the tester's own file")
    shutil.rmtree(world.prefix / "assets")
    (world.prefix / "assets").symlink_to(elsewhere)  # the tester keeps the assets on another disk
    manifest_path = world.prefix / "install-manifest.txt"
    good = manifest_path.read_text()
    for foreign in ("../../home", "data", "/etc"):
        manifest_path.write_text(good + foreign + "\n")
        before = tree(world.tmp)
        refused = world.run("uninstall.sh", "--yes", package=world.installed)
        assert refused.returncode == 1 and "nothing was changed" in said(refused), said(refused)
        assert tree(world.tmp) == before
    manifest_path.write_text(good)
    result = world.run("uninstall.sh", "--yes", package=world.installed)
    assert result.returncode == 0, said(result)
    assert not world.prefix.exists()
    assert (elsewhere / "big model.gguf").read_text() == "the tester's own file"


def test_uninstall_is_refused_where_nothing_is_installed(world: World) -> None:
    empty = world.tmp / "empty"
    empty.mkdir()
    result = beta_from_archive(world, "uninstall", "--prefix", str(empty))
    assert result.returncode == 1 and "holds no installation of this package" in result.stderr
    assert empty.is_dir()


def beta_from_archive(world: World, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(world.bin / "python3"), "-I", "-B", str(world.package / "lib/beta.py"), *args],
                          env=world.env, capture_output=True, text=True, timeout=120)
