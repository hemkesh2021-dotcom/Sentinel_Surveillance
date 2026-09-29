from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import sentinel

# Jetson-only or heavy ML libraries that portable modules must not need at import.
# A future hardware adapter module that legitimately needs one must import it lazily.
BLOCKED = (
    "cv2",
    "gi",
    "tensorrt",
    "pycuda",
    "cuda",
    "torch",
    "tensorflow",
    "ultralytics",
    "deepface",
    "onnxruntime",
    "jtop",
)


def sentinel_module_names() -> set[str]:
    root = Path(sentinel.__file__).parent
    names = set()
    for path in root.rglob("*.py"):
        parts = path.relative_to(root.parent).with_suffix("").parts
        names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def test_every_sentinel_module_imports_with_jetson_and_ml_libraries_blocked() -> None:
    # Blocking (not just checking) matters on the Jetson, where cv2 etc. are installed.
    script = textwrap.dedent(
        f"""
        import importlib, importlib.abc, json, pkgutil, sys

        BLOCKED = {BLOCKED!r}

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in BLOCKED:
                    raise ImportError(f"blocked import of {{name}}")
                return None

        sys.meta_path.insert(0, Block())
        import sentinel
        for module in pkgutil.walk_packages(sentinel.__path__, "sentinel."):
            importlib.import_module(module.name)
        print(json.dumps(sorted(name for name in sys.modules if name.startswith("sentinel"))))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert sentinel_module_names() <= set(json.loads(result.stdout))


def test_resolving_ml_adapters_imports_no_ml_framework_and_a_failed_load_is_contained() -> None:
    # Guide ch. 27 / V2-49: enumerating capabilities must not load frameworks, and
    # an adapter whose framework is missing becomes unavailable instead of crashing.
    script = textwrap.dedent(
        f"""
        import importlib.abc, json, sys

        BLOCKED = {BLOCKED!r}

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in BLOCKED:
                    raise ImportError(f"blocked import of {{name}}")
                return None

        sys.meta_path.insert(0, Block())
        from sentinel.adapters import AdapterManifest, AdapterRole, AdapterSpec, load, resolve

        registry = {{
            name: AdapterSpec(name, AdapterRole.DETECTOR, module, "Adapter",
                              frozenset({{"frame"}}), frozenset({{"detection"}}))
            for name, module in (("trt-detector", "tensorrt"), ("torch-detector", "torch.nn"))
        }}
        manifests = [
            AdapterManifest.model_validate({{
                "adapter_id": name, "contract_version": 1, "implementation_revision": "1",
                "enabled": True, "input_kinds": ["frame"], "output_kinds": ["detection"],
                "resource_profile_id": "measured", "timeout_ms": 100,
            }})
            for name in registry
        ]
        statuses = resolve(manifests, registry=registry, known_profiles=frozenset({{"measured"}}))
        resolved = [s.state.value for s in statuses]
        loaded = [load(s)[1] for s in statuses]
        print(json.dumps({{
            "resolved": resolved,
            "loaded": [[s.state.value, s.reason] for s in loaded],
            "ml_modules": sorted(m for m in sys.modules if m.partition(".")[0] in BLOCKED),
        }}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["resolved"] == ["enabled", "enabled"]
    assert outcome["loaded"] == [
        ["unavailable", "failed to load: ImportError: blocked import of tensorrt"],
        ["unavailable", "failed to load: ImportError: blocked import of torch"],
    ]
    assert outcome["ml_modules"] == []
