#!/usr/bin/env python3
"""Check 9 (V2-01, U18): can GPU allocations go beyond MemFree on this device?

Session 3 saw llama-server's GPU allocations fail with NvMap error 12 whenever
they exceeded MemFree, even with several GB of reclaimable page cache. Every
one of those runs set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, which makes llama.cpp
call cudaMallocManaged instead of cudaMalloc. This probe separates the two:
it allocates fixed-size chunks with one API (``--api device`` = cudaMalloc,
``--api managed`` = cudaMallocManaged), writes each chunk with cudaMemset so it
is backed, and stops at the first failure or at the cap. It then frees
everything and prints one summary line.

Run it with L4T's libcuda preloaded (decision D27); it refuses otherwise.
Standard library only; it reads no credentials and writes no files.
"""

from __future__ import annotations

import argparse
import ctypes
import sys

L4T_LIBCUDA_DIR = "/usr/lib/aarch64-linux-gnu/nvidia/"
DEFAULT_CUDART = "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12"
CUDA_MEM_ATTACH_GLOBAL = 1
MIB = 1 << 20


def meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemFree", "MemAvailable", "Cached"):
                values[key] = int(rest.split()[0]) * 1024
    return values


def mapped_libcuda() -> list[str]:
    with open("/proc/self/maps") as maps:
        return sorted({line.split()[-1] for line in maps if "libcuda.so" in line})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", choices=("device", "managed"), required=True)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--cap-gb", type=float, default=None,
                        help="stop after this many GB (default: MemAvailable at start minus 1.0 GB, at most 4.0)")
    parser.add_argument("--cudart", default=DEFAULT_CUDART)
    args = parser.parse_args(argv)

    rt = ctypes.CDLL(args.cudart)
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    rt.cudaMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    rt.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    rt.cudaFree.argtypes = [ctypes.c_void_p]

    def error(code: int) -> str:
        return f"{code} ({rt.cudaGetErrorString(code).decode(errors='replace')})"

    init = rt.cudaFree(None)  # creates the context, so its cost is not counted as an allocation
    libcuda = mapped_libcuda()
    if init != 0 or not libcuda or not all(path.startswith(L4T_LIBCUDA_DIR) for path in libcuda):
        print(f"refused: CUDA init {error(init)}, libcuda {libcuda}; preload L4T's libcuda (D27)")
        return 2

    start = meminfo()
    cap = args.cap_gb * 1e9 if args.cap_gb is not None else min(start["MemAvailable"] - 1e9, 4e9)
    chunk = args.chunk_mib * MIB
    pointers: list[ctypes.c_void_p] = []
    allocated = 0
    failure = None
    free_at_failure = None
    while allocated + chunk <= cap:
        pointer = ctypes.c_void_p()
        if args.api == "device":
            code = rt.cudaMalloc(ctypes.byref(pointer), chunk)
        else:
            code = rt.cudaMallocManaged(ctypes.byref(pointer), chunk, CUDA_MEM_ATTACH_GLOBAL)
        stage = "alloc"
        if code == 0:
            pointers.append(pointer)
            stage = "memset"
            code = rt.cudaMemset(pointer, 0, chunk) or rt.cudaDeviceSynchronize()
        if code != 0:
            failure = f"{stage} {error(code)}"
            free_at_failure = meminfo()["MemFree"]
            break
        allocated += chunk
    end = meminfo()
    for pointer in pointers:
        rt.cudaFree(pointer)

    beyond = allocated > start["MemFree"]
    print(
        f"api={args.api} chunk_mib={args.chunk_mib} cap_bytes={int(cap):,} "
        f"start_memfree={start['MemFree']:,} start_memavailable={start['MemAvailable']:,} start_cached={start['Cached']:,} "
        f"allocated={allocated:,} beyond_start_memfree={'yes' if beyond else 'no'} "
        f"stopped_by={'failure: ' + failure if failure else 'cap'} "
        f"memfree_at_failure={'n/a' if free_at_failure is None else f'{free_at_failure:,}'} "
        f"end_memfree={end['MemFree']:,} end_cached={end['Cached']:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
