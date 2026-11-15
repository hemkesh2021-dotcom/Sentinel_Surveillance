"""Device artifacts (session 116, part 41): a tester's own files, identified by a record, for experimental
validation runs only.

The default admission path identifies the detector engine, the llama.cpp build and the model files by records made on
the reference device (the engine's SHA-256, the accepted resource profile's build and file facts, the face
admission's weights). A TensorRT engine and a llama.cpp build made on another device are other bytes, and a
downloaded model file has another modification time, so that path refuses them. It is unchanged.

``sentinel run --device-artifacts FILE`` (only with ``--face --face-validation``) makes those three file comparisons
against FILE instead: a versioned record of this device's own files, written by the personal-beta package's
controlled download and build steps. Everything else admission requires stays as it is: the scene adapter's accepted
resource profile and the configuration it measured (memory policy, server flags, scene interval, request), the face
admission's rules, chat's. The record must also be compatible with this build: the pinned llama.cpp commit and build
options, the pinned export settings and detector weights, the pinned model and weight files (by SHA-256), each built
file made by the package's own step. A file that was only inspected proves nothing about how it was built and is
refused.

What a record is not. It is not a resource profile and it measures nothing: no run on this device has been judged
against any criterion. The accepted profile the configuration names was measured on the reference device with the
reference device's files; with a record, it says which configuration is run, not how this device behaves under it.
``resource_validation`` is therefore always "pending" in a record, and a record that says anything else is refused.
The launch gates and the memory guard's limits remain limits; they are not evidence about this device.

Portable: reads the record and file metadata only; hashing is the caller's (``demo_runtime``), by the same rule as
the default path (files up to STARTUP_HASH_LIMIT_BYTES by SHA-256, larger ones by name, size and modification time
against the record; the package's own check hashes every file before a launch).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .adapters import FileFacts

SCHEMA = "sentinel-device-artifacts/1"
PURPOSE = ("identifies this device's own files for experimental validation runs; it is not a resource profile and "
           "measures nothing")
RESOURCE_VALIDATION_PENDING = "pending"
MAX_RECORD_BYTES = 262_144
# What the package's tooling observed about a file (``provenance.observed``).
BUILT = "built_by_package_step"  # the package's build script made it and hashed what it made
INSPECTED = "inspected_only"  # an existing file was hashed; how it was built was not observed
DOWNLOADED = "downloaded_by_package_step"  # the package's fetch step downloaded it from the pinned address
VERIFIED = "verified_against_pinned_checksum"  # an existing file equals the pinned SHA-256; no download was observed
BUILT_ROLES = ("detector_engine", "llama_server")
PINNED_ROLES = ("scene_model", "scene_mmproj", "face_model_weights", "face_detector_weights")
ROLES = BUILT_ROLES + PINNED_ROLES
PREFLIGHT = ("verified_combination", "not_verified_acknowledged")
PLATFORM_TEXT = ("device_model", "machine", "kernel", "l4t", "os", "cuda_toolkit", "tensorrt", "cudnn", "python")
# The package's own tests read platform facts from a directory that stands in for "/"; such a record says so with
# this entry, and no run uses it.
TEST_FIXTURE = "facts were read from a test fixture, not from a device"
# What the provenance of a file may hold, by what was observed: exactly these keys, nothing else. A record has no
# place for a measurement or for any other claim.
INSPECTED_STATEMENT = ("only this file's checksum was read; how it was built was not observed by this package's "
                       "build step, so nothing is recorded about its source or settings")
VERIFIED_STATEMENT = "this file equals the pinned SHA-256; this package did not observe its download"
PROVENANCE_KEYS: Mapping[tuple[str, str], tuple[str, ...]] = MappingProxyType({
    ("detector_engine", BUILT): ("observed", "step", "utc", "package", "weights_sha256", "export", "tools"),
    ("llama_server", BUILT): ("observed", "step", "utc", "package", "repository", "commit", "cmake", "target",
                              "reports", "tools"),
    ("built", INSPECTED): ("observed", "statement"),
    ("pinned", DOWNLOADED): ("observed", "step", "utc", "url", "revision"),
    ("pinned", VERIFIED): ("observed", "pinned_url", "revision", "statement"),
})
STEPS: Mapping[str, str] = MappingProxyType({"detector_engine": "build-engine.sh", "llama_server": "build-llama.sh"})
FETCH_STEP = "fetch-assets.sh"

# The controlled steps' pinned settings. The package's manifest (packaging/beta-install/manifest/assets.json) states
# the same values and its scripts build with them; a test keeps the two equal.
LLAMA_REPOSITORY = "https://github.com/ggml-org/llama.cpp"
LLAMA_TARGET = "llama-server"
LLAMA_CMAKE_OPTIONS = ("-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=87",
                       "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc")
# The scene model sets this route knows a pinned build for (scene.models.LFM3_LLAMA_CPP_COMMIT; a test keeps it equal).
LLAMA_COMMITS: Mapping[str, str] = MappingProxyType({"lfm3-vl": "19cce6919992fdffd92e0ac152caaf1e2b8f254e"})
# yolov8n.pt as Ultralytics' release v8.4.0 publishes it (GitHub's digest of the release asset). Which weights the
# reference engine was exported from is not recorded; the route pins the file the package's step exports.
DETECTOR_WEIGHTS_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
ENGINE_EXPORT: Mapping[str, Any] = MappingProxyType({
    "format": "engine", "imgsz": 640, "batch": 1, "half": False, "int8": False, "dynamic": False, "simplify": True,
    "nms": False, "device": 0})
ENGINE_TOOLS: Mapping[str, str] = MappingProxyType({"ultralytics": "8.4.25", "tensorrt": "10.3.0"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_MTIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00")  # as demo_runtime.file_facts writes it
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,199}")  # a package's name, a step, a revision: one word
_LOCATION = re.compile(r"[!-~]{1,400}")  # an address: printable, no spaces
_LINE = re.compile(r"[ -~]{0,200}")  # one line a tool printed about itself (may be empty)
_TEXT_LIMIT = 400


class DeviceArtifactsError(Exception):
    """The record cannot be used; ``label`` says why, with field names only (never a path or a file's content)."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


@dataclass(frozen=True)
class DeviceArtifacts:
    """One record, as read. ``files`` holds each role's file facts (with its SHA-256); ``provenance`` what the
    package's tooling observed about each."""

    record_sha256: str  # of the record file's bytes: which record a run used
    created_utc: str
    package: str  # the package whose tooling wrote it
    platform: Mapping[str, Any]
    files: Mapping[str, FileFacts]
    libraries: tuple[FileFacts, ...]  # the llama.cpp build's libraries beside llama-server
    provenance: Mapping[str, Mapping[str, Any]]

    def startup_entry(self, *, scene: bool) -> dict[str, Any]:
        """What the runtime's starting line says about a run on a record: fixed labels and checksums only, and only
        of what this run compared with the record (the llama.cpp build and the scene files only with ``scene``).
        The platform's state is the record's own statement: the runtime does not read the platform (the package's
        check before a launch compares it)."""
        compared = ("detector_engine", "face_weights") + (("llama_build", "scene_model_files") if scene else ())
        return {
            "schema": SCHEMA,
            "record_sha256": self.record_sha256,
            "created_utc": self.created_utc,
            "run": "experimental_validation_run_only",
            "compared_with_the_record": list(compared),
            "resource_validation": RESOURCE_VALIDATION_PENDING,
            "resource_profile_measures_this_device": False,
            "recorded_platform_preflight": self.platform["preflight"],
            "detector_engine_sha256": self.files["detector_engine"].sha256,
            **({"llama_server_sha256": self.files["llama_server"].sha256} if scene else {}),
        }


def _malformed(what: str) -> DeviceArtifactsError:
    return DeviceArtifactsError(f"device_artifacts_malformed: {what}")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise _malformed("a key is given twice")
        found[key] = value
    return found


def _keys(value: Any, where: str, wanted: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _malformed(f"{where} is not an object")
    if sorted(value) != sorted(wanted):
        raise _malformed(f"{where} does not hold exactly: {', '.join(wanted)}")
    return value


def _text(value: Any, where: str, pattern: re.Pattern[str] | None = None) -> str:
    if (not isinstance(value, str) or not value or len(value) > _TEXT_LIMIT or any(ord(c) < 32 for c in value)
            or (pattern is not None and not pattern.fullmatch(value))):
        raise _malformed(f"{where} is not as the schema requires")
    return value


def _time(value: Any, where: str, pattern: re.Pattern[str]) -> str:
    """A time in the schema's one form that is also a real time."""
    text = _text(value, where, pattern)
    try:
        datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise _malformed(f"{where} is not as the schema requires") from None
    return text


def _file(value: Any, where: str) -> FileFacts:
    item = _keys(value, where, ("name", "bytes", "sha256", "mtime_utc"))
    name = _text(item["name"], f"{where}.name")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise _malformed(f"{where}.name is not a file name")
    size = item["bytes"]
    if type(size) is not int or size < 1:
        raise _malformed(f"{where}.bytes is not a positive whole number")
    return FileFacts(name, size, _time(item["mtime_utc"], f"{where}.mtime_utc", _MTIME),
                     _text(item["sha256"], f"{where}.sha256", _SHA256))


def canonical(value: Any) -> str:
    """One text for a settings value, so that two are equal only when they are the same JSON: true is not 1."""
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: plain(inner) for key, inner in item.items()}
        return [plain(inner) for inner in item] if isinstance(item, (list, tuple)) else item
    return json.dumps(plain(value), sort_keys=True)


def _provenance(role: str, value: Any) -> Mapping[str, Any]:
    """What was observed about ``role``'s file: one of the schema's forms, with exactly its keys and their kinds."""
    where = f"artifacts.{role}.provenance"
    seen = _plain(value, where)
    kind = "built" if role in BUILT_ROLES else "pinned"
    allowed = (BUILT, INSPECTED) if kind == "built" else (DOWNLOADED, VERIFIED)
    if not isinstance(seen, Mapping) or seen.get("observed") not in allowed:
        raise _malformed(f"{where}.observed is not one of: {', '.join(allowed)}")
    observed = seen["observed"]
    keys = PROVENANCE_KEYS[(role if observed == BUILT else kind, observed)]
    if sorted(seen) != sorted(keys):
        raise _malformed(f"{where} does not hold exactly: {', '.join(keys)}")

    def words(key: str) -> bool:
        return isinstance(seen[key], Mapping) and all(isinstance(item, str) and _LINE.fullmatch(item)
                                                      for item in seen[key].values())

    checks = {
        "observed": lambda: True,
        "step": lambda: seen["step"] == (STEPS[role] if observed == BUILT else FETCH_STEP),
        "utc": lambda: bool(_time(seen["utc"], f"{where}.utc", _UTC)),
        "package": lambda: isinstance(seen["package"], str) and bool(_NAME.fullmatch(seen["package"])),
        "weights_sha256": lambda: isinstance(seen["weights_sha256"], str) and bool(_SHA256.fullmatch(seen["weights_sha256"])),
        "export": lambda: isinstance(seen["export"], Mapping) and all(
            isinstance(item, (str, int)) for item in seen["export"].values()),
        "tools": lambda: words("tools"),
        "repository": lambda: isinstance(seen["repository"], str) and bool(_LOCATION.fullmatch(seen["repository"])),
        "commit": lambda: isinstance(seen["commit"], str) and bool(_COMMIT.fullmatch(seen["commit"])),
        "cmake": lambda: isinstance(seen["cmake"], tuple) and all(
            isinstance(item, str) and _LOCATION.fullmatch(item) for item in seen["cmake"]),
        "target": lambda: isinstance(seen["target"], str) and bool(_NAME.fullmatch(seen["target"])),
        "reports": lambda: isinstance(seen["reports"], str) and bool(_LINE.fullmatch(seen["reports"])),
        "url": lambda: isinstance(seen["url"], str) and bool(_LOCATION.fullmatch(seen["url"])),
        "pinned_url": lambda: isinstance(seen["pinned_url"], str) and bool(_LOCATION.fullmatch(seen["pinned_url"])),
        "revision": lambda: isinstance(seen["revision"], str) and bool(_NAME.fullmatch(seen["revision"])),
        "statement": lambda: seen["statement"] == (INSPECTED_STATEMENT if observed == INSPECTED else VERIFIED_STATEMENT),
    }
    for key in keys:
        if not checks[key]():
            raise _malformed(f"{where}.{key} is not as the schema requires")
    return seen


def _plain(value: Any, where: str, depth: int = 0) -> Any:
    """Provenance and platform values: text, whole numbers, flags, and one level of lists or objects of those."""
    if value is None or isinstance(value, bool) or type(value) is int:
        return value
    if isinstance(value, str):
        if len(value) > _TEXT_LIMIT or any(ord(c) < 32 for c in value):
            raise _malformed(f"{where} holds text that is too long or has control characters")
        return value
    if depth < 2 and isinstance(value, list) and len(value) <= 64:
        return tuple(_plain(item, where, depth + 1) for item in value)
    if depth < 2 and isinstance(value, dict) and len(value) <= 64:
        return MappingProxyType({_text(key, where): _plain(item, where, depth + 1) for key, item in value.items()})
    raise _malformed(f"{where} holds a value the schema does not allow")


def parse(data: bytes) -> DeviceArtifacts:
    """The record in ``data``, or DeviceArtifactsError. Strict: an unknown key, a missing role or a claim the schema
    does not have (anything but "pending" for resource validation) is refused, never ignored."""
    if len(data) > MAX_RECORD_BYTES:
        raise _malformed("the record is larger than a record can be")

    def constant(_: str) -> Any:
        raise _malformed("a number is not finite")

    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _malformed("not a JSON document") from None
    top = _keys(document, "the record", ("schema", "created_utc", "package", "purpose", "resource_validation",
                                         "platform", "artifacts"))
    if top["schema"] != SCHEMA:
        raise _malformed(f"schema is not {SCHEMA}")
    if top["purpose"] != PURPOSE:
        raise _malformed("purpose is not the schema's own sentence")
    if top["resource_validation"] != RESOURCE_VALIDATION_PENDING:
        raise _malformed("resource_validation is not \"pending\": a record identifies files and validates nothing")
    platform = _plain(top["platform"], "platform")
    if not isinstance(platform, Mapping):
        raise _malformed("platform is not an object")
    for key in PLATFORM_TEXT:
        if not isinstance(platform.get(key), str) or not platform[key] or not _LINE.fullmatch(platform[key]):
            raise _malformed(f"platform.{key} is missing")
    packages = platform.get("packages")
    if not isinstance(packages, Mapping) or not packages or not all(
            isinstance(version, str) and _NAME.fullmatch(version) for version in packages.values()):
        raise _malformed("platform.packages is missing")
    if platform.get("preflight") not in PREFLIGHT:
        raise _malformed("platform.preflight is not one of: " + ", ".join(PREFLIGHT))
    if platform.get("test_fixture", TEST_FIXTURE) != TEST_FIXTURE or not set(platform) <= {
            *PLATFORM_TEXT, "packages", "preflight", "test_fixture"}:
        raise _malformed("platform does not hold exactly its versions, packages and preflight")
    artifacts = _keys(top["artifacts"], "artifacts", ROLES)
    files: dict[str, FileFacts] = {}
    provenance: dict[str, Mapping[str, Any]] = {}
    libraries: tuple[FileFacts, ...] = ()
    for role in ROLES:
        wanted = ("file", "libraries", "provenance") if role == "llama_server" else ("file", "provenance")
        item = _keys(artifacts[role], f"artifacts.{role}", wanted)
        files[role] = _file(item["file"], f"artifacts.{role}.file")
        provenance[role] = _provenance(role, item["provenance"])
        if role == "llama_server":
            listed = item["libraries"]
            if not isinstance(listed, list) or not 1 <= len(listed) <= 32:
                raise _malformed("artifacts.llama_server.libraries is not a list of files")
            libraries = tuple(_file(entry, "artifacts.llama_server.libraries[]") for entry in listed)
            if len({library.name for library in libraries}) != len(libraries):
                raise _malformed("artifacts.llama_server.libraries names a file twice")
    return DeviceArtifacts(
        record_sha256=hashlib.sha256(data).hexdigest(), created_utc=_time(top["created_utc"], "created_utc", _UTC),
        package=_text(top["package"], "package", _NAME), platform=platform, files=MappingProxyType(files),
        libraries=libraries, provenance=MappingProxyType(provenance))


def load(path: Path) -> DeviceArtifacts:
    """The record at ``path``; DeviceArtifactsError when it cannot be read or is not a record."""
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(MAX_RECORD_BYTES + 1)
    except OSError as exc:
        raise DeviceArtifactsError(f"device_artifacts_unreadable: {type(exc).__name__}") from None
    return parse(data)


# ---------------------------------------------------------------- is the record one this build can use?


def engine_problem(record: DeviceArtifacts) -> str | None:
    """None when the recorded detector engine was exported by the package's step from the pinned weights with the
    pinned settings and tool versions; else the first difference."""
    if "test_fixture" in record.platform:
        return "the record's platform facts were read from a test fixture, not from a device: no run uses it"
    seen = record.provenance["detector_engine"]
    if seen["observed"] != BUILT:
        return ("the detector engine was only inspected: its build by the package's step (build-engine.sh) was not "
                "observed, and a checksum does not show how a file was built")
    if seen.get("weights_sha256") != DETECTOR_WEIGHTS_SHA256:
        return "the detector engine was not exported from the pinned yolov8n.pt"
    if canonical(seen.get("export")) != canonical(ENGINE_EXPORT):
        return "the detector engine's export settings are not the pinned ones"
    tools = seen.get("tools")
    for name, version in ENGINE_TOOLS.items():
        if not isinstance(tools, Mapping) or tools.get(name) != version:
            return f"the detector engine was not exported with {name} {version}"
    return None


def llama_problem(record: DeviceArtifacts, *, model_set: str, library_names: tuple[str, ...]) -> str | None:
    """None when the recorded llama.cpp build was made by the package's step at the commit pinned for ``model_set``
    with the pinned options, and consists of ``library_names`` (the accepted profile's build layout)."""
    commit = LLAMA_COMMITS.get(model_set)
    if commit is None:
        return (f"scene model set {model_set} has no llama.cpp build pinned for this route (only: "
                f"{', '.join(LLAMA_COMMITS)})")
    seen = record.provenance["llama_server"]
    if seen["observed"] != BUILT:
        return ("the llama.cpp build was only inspected: its build by the package's step (build-llama.sh) was not "
                "observed, and a checksum does not show how a file was built")
    if seen.get("repository") != LLAMA_REPOSITORY or seen.get("commit") != commit:
        return f"the llama.cpp build is not {LLAMA_REPOSITORY} at the pinned commit {commit}"
    if canonical(seen.get("cmake")) != canonical(LLAMA_CMAKE_OPTIONS) or seen.get("target") != LLAMA_TARGET:
        return "the llama.cpp build's options are not the pinned ones"
    if record.files["llama_server"].name != LLAMA_TARGET:
        return f"the recorded binary is not named {LLAMA_TARGET}"
    if sorted(library.name for library in record.libraries) != sorted(library_names):
        return "the recorded llama.cpp libraries are not the ones the pinned build consists of"
    return None


def pinned_problem(record: DeviceArtifacts, role: str, pinned: FileFacts, label: str) -> str | None:
    """None when the record's file for ``role`` is the pinned one: the same name, size and SHA-256 (its modification
    time is this device's copy's own)."""
    recorded = record.files[role]
    if (recorded.name, recorded.bytes, recorded.sha256) != (pinned.name, pinned.bytes, pinned.sha256):
        return f"the recorded {label} is not the pinned {pinned.name}"
    return None
