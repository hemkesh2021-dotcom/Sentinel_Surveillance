from __future__ import annotations

from datetime import timedelta

import pytest

from sentinel.media.clock import CrossBootComparisonError, FakeClock, SystemClock, read_boot_id


def test_wall_clock_steps_leave_monotonic_time_untouched(clock: FakeClock) -> None:
    start_mono, start_utc = clock.mono(), clock.utc_now()

    clock.step_utc(timedelta(hours=-1))  # NTP correction backwards
    clock.step_utc(timedelta(days=1))  # manual change forwards
    assert clock.mono() == start_mono
    assert clock.utc_now() == start_utc + timedelta(hours=23)

    clock.advance(2.5)
    assert clock.mono().ns_since(start_mono) == 2_500_000_000
    assert clock.utc_now() == start_utc + timedelta(hours=23, seconds=2.5)


def test_monotonic_time_cannot_run_backwards(clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-0.001)


def test_readings_from_before_a_reboot_cannot_be_compared(clock: FakeClock) -> None:
    before = clock.mono()
    clock.reboot("fake-boot-2")
    after = clock.mono()

    # The new boot's counter restarts, so plain subtraction would give a wrong age.
    assert after.ns <= before.ns
    with pytest.raises(CrossBootComparisonError):
        after.ns_since(before)
    with pytest.raises(ValueError, match="new boot ID"):
        clock.reboot("fake-boot-2")


def test_boot_id_comes_from_the_kernel_when_available(tmp_path) -> None:
    boot_file = tmp_path / "boot_id"
    boot_file.write_text("3f1c2a9e-5b7d-4c1e-9a0f-2d8e6b4c7a11\n")
    assert read_boot_id(boot_file) == "3f1c2a9e-5b7d-4c1e-9a0f-2d8e6b4c7a11"


def test_boot_id_fallback_is_shared_within_a_process(tmp_path) -> None:
    # Two clocks in one process must agree, or they could never compare readings.
    missing = tmp_path / "missing"
    assert read_boot_id(missing).startswith("process-")
    assert read_boot_id(missing) == read_boot_id(missing)


def test_system_clock_reads_utc_and_non_decreasing_monotonic_time() -> None:
    clock = SystemClock()
    first, second = clock.mono(), clock.mono()
    assert second.ns_since(first) >= 0
    assert clock.utc_now().utcoffset() == timedelta(0)
    assert first.boot_id == read_boot_id()
