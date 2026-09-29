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
