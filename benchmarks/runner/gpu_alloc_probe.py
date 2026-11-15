#!/usr/bin/env python3
"""Bounded Check 9 API smoke, not an allocation-to-failure or beyond-MemFree test.

Operator entry point: operator_check.py --execute-workload check9. This child
uses at most 256 MiB in chunks no larger than 32 MiB, with projected device
pressure/free-memory checks before every chunk and finally-based release.
CUDA context/native transients are outside that explicit allocation budget.
The parent supplies the execution timeout and owned-process cleanup.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys

import demo_profile
from operator_check import baseline_problem, memory_problem

L4T_LIBCUDA_DIR = "/usr/lib/aarch64-linux-gnu/nvidia/"
DEFAULT_CUDART = "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12"
CUDA_MEM_ATTACH_GLOBAL = 1
MIB = 1 << 20


def meminfo() -> dict[str, int]:
    return {**demo_profile.read_meminfo(), **demo_profile.read_swap_counters()}


def mapped_libcuda() -> list[str]:
    with open("/proc/self/maps") as maps:
        return sorted({line.split()[-1] for line in maps if "libcuda.so" in line})


class CudaAllocator:
    def __init__(self) -> None:
        self.runtime = ctypes.CDLL(DEFAULT_CUDART)
        self.runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.runtime.cudaMallocManaged.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint,
        ]
        self.runtime.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.runtime.cudaFree.argtypes = [ctypes.c_void_p]
        mapped = mapped_libcuda()
        if not mapped or not all(path.startswith(L4T_LIBCUDA_DIR) for path in mapped):
            raise RuntimeError("driver_unavailable")
        if self.runtime.cudaFree(None):
            raise RuntimeError("driver_unavailable")

    def allocate(self, api: str, size: int):
        pointer = ctypes.c_void_p()
        if api == "device":
            code = self.runtime.cudaMalloc(ctypes.byref(pointer), size)
        else:
            code = self.runtime.cudaMallocManaged(ctypes.byref(pointer), size, CUDA_MEM_ATTACH_GLOBAL)
        return code, pointer

    def write(self, pointer, size: int) -> int:
        return self.runtime.cudaMemset(pointer, 0, size) or self.runtime.cudaDeviceSynchronize()

    def free(self, pointer) -> int:
        return self.runtime.cudaFree(pointer)


def allocation_smoke(api: str, chunk_mib: int, cap_mib: int, *, provider=CudaAllocator, telemetry=meminfo) -> dict:
    result = {"api": api, "status": "refused", "allocated_bytes": 0, "cleanup_clear": True}
    if (api not in ("device", "managed") or type(chunk_mib) is not int or type(cap_mib) is not int
            or not (1 <= chunk_mib <= 32 and chunk_mib <= cap_mib <= 256)):
        result["reason"] = "allocation_bounds_refused"
        return result
    pointers = []
    allocator = None
    try:
        start = telemetry()
        problem = baseline_problem(start)
        if problem:
            result["reason"] = problem
            return result
        allocator = provider()
        chunk, cap = chunk_mib * MIB, cap_mib * MIB
        result.update(chunk_bytes=chunk, cap_bytes=cap)
        while result["allocated_bytes"] + chunk <= cap:
            problem = memory_problem(telemetry(), start, projected=chunk)
            if problem:
                result["reason"] = problem
                break
            code, pointer = allocator.allocate(api, chunk)
            if code:
                result.update(status="gpu_error", error_code=int(code), stage="allocate")
                break
            pointers.append(pointer)
            code = allocator.write(pointer, chunk)
            if code:
                result.update(status="gpu_error", error_code=int(code), stage="write")
                break
            result["allocated_bytes"] += chunk
        else:
            result["status"] = "bounded_smoke_complete"
    except Exception:
        result["status"] = "probe_unavailable"
    finally:
        for pointer in pointers:
            try:
                if allocator.free(pointer):
                    result["cleanup_clear"] = False
            except Exception:
                result["cleanup_clear"] = False
        if not result["cleanup_clear"]:
            result["status"] = "cleanup_failed"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", choices=("device", "managed"), required=True)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--cap-mib", type=int, default=256)
    args = parser.parse_args(argv)
    result = allocation_smoke(args.api, args.chunk_mib, args.cap_mib)
    print(json.dumps(result))
    return 0 if result["status"] == "bounded_smoke_complete" else 1


if __name__ == "__main__":
    sys.exit(main())
