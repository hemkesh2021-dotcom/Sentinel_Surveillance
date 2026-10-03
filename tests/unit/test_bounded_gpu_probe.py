from __future__ import annotations

import importlib
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks/runner"
HEALTHY = {
    "MemTotal": 8_000_000_000, "MemAvailable": 7_000_000_000, "MemFree": 6_000_000_000,
    "pswpin": 0, "pswpout": 0,
}


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    return importlib.import_module("gpu_alloc_probe")


class FakeAllocator:
    def __init__(self):
        self.allocated = []
        self.written = []
        self.freed = []
        self.fail_allocate = None
        self.fail_write = None
        self.raise_write = False
        self.fail_free = False

    def allocate(self, api, size):
        if self.fail_allocate is not None and len(self.allocated) == self.fail_allocate:
            return 2, None
        pointer = object()
        self.allocated.append((api, size, pointer))
        return 0, pointer

    def write(self, pointer, size):
        if self.raise_write:
            raise RuntimeError("secret exception details")
        if self.fail_write is not None and len(self.written) == self.fail_write:
            return 3
        self.written.append(pointer)
        return 0

    def free(self, pointer):
        self.freed.append(pointer)
        return 1 if self.fail_free else 0


@pytest.mark.parametrize("api", ["device", "managed"])
def test_allocation_budget_writes_and_frees_every_chunk(probe, api):
    allocator = FakeAllocator()
    result = probe.allocation_smoke(api, 32, 256, provider=lambda: allocator, telemetry=lambda: HEALTHY)
    assert result["status"] == "bounded_smoke_complete"
    assert result["allocated_bytes"] == 256 * probe.MIB
    assert len(allocator.allocated) == len(allocator.written) == len(allocator.freed) == 8
    assert {size for _, size, _ in allocator.allocated} == {32 * probe.MIB}
    assert {used_api for used_api, _, _ in allocator.allocated} == {api}


@pytest.mark.parametrize("chunk,cap", [(0, 256), (33, 256), (32, 257), (32, 0), (32, 31), (True, 32), (32, 32.0)])
def test_unsafe_bounds_refuse_before_runtime_provider_or_telemetry(probe, chunk, cap):
    fail = lambda: pytest.fail("unsafe probe contacted hardware")
    result = probe.allocation_smoke("device", chunk, cap, provider=fail, telemetry=fail)
    assert result["reason"] == "allocation_bounds_refused"


def test_baseline_refusal_precedes_cuda_context_initialization(probe):
    result = probe.allocation_smoke(
        "device", 32, 256, provider=lambda: pytest.fail("CUDA context created"),
        telemetry=lambda: {**HEALTHY, "MemFree": 1},
    )
    assert result["status"] == "refused"


@pytest.mark.parametrize("changed", [
    {"MemTotal": 8_000_000_000, "MemAvailable": 3_233_554_432},
    {"MemFree": (1 << 30) + 32 * (1 << 20) - 1},
    {"pswpout": 1}, {"MemAvailable": None},
])
def test_per_chunk_projected_guard_refuses_before_another_allocation(probe, changed):
    allocator = FakeAllocator()
    calls = 0

    def telemetry():
        nonlocal calls
        calls += 1
        return HEALTHY if calls <= 2 else {**HEALTHY, **changed}

    result = probe.allocation_smoke("device", 32, 256, provider=lambda: allocator, telemetry=telemetry)
    assert result["status"] == "refused" and result["allocated_bytes"] == 32 * probe.MIB
    assert len(allocator.allocated) == len(allocator.freed) == 1


@pytest.mark.parametrize("failure", ["allocate", "write", "exception", "free"])
def test_gpu_errors_and_exceptions_never_continue_until_failure_and_always_free(probe, failure):
    allocator = FakeAllocator()
    if failure == "allocate":
        allocator.fail_allocate = 2
    elif failure == "write":
        allocator.fail_write = 2
    elif failure == "exception":
        allocator.raise_write = True
    else:
        allocator.fail_free = True
    result = probe.allocation_smoke("device", 32, 256, provider=lambda: allocator, telemetry=lambda: HEALTHY)
    assert result["status"] in ("gpu_error", "probe_unavailable", "cleanup_failed")
    assert len(allocator.freed) == len(allocator.allocated)
    assert "secret" not in str(result)


def test_failing_telemetry_after_allocation_releases_prior_handles(probe):
    allocator = FakeAllocator()
    calls = 0

    def telemetry():
        nonlocal calls
        calls += 1
        if calls > 2:
            raise OSError("private details")
        return HEALTHY

    result = probe.allocation_smoke("managed", 32, 256, provider=lambda: allocator, telemetry=telemetry)
    assert result["status"] == "probe_unavailable"
    assert len(allocator.allocated) == len(allocator.freed) == 1
