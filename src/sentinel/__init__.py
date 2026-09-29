"""Sentinel v2, built beside the v1 prototype (surveillance4_1.py, dashboard.py).

Modules in this package must import without Jetson-only libraries such as
CUDA, TensorRT, GStreamer or OpenCV. Hardware adapters belong in separate
modules that import those libraries only when the adapter is used.
"""

__version__ = "2.0.0.dev0"
