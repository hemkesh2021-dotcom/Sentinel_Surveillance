"""D77: chat's pressure response (ChatGate) and the transport's abort, with deterministic samples and a real socket.

Covered: the guide's initial thresholds on MemTotal - MemAvailable (no new question above 4.7 GB for three samples,
a running one cancelled above 5.0 GB, open again only after 60 s below 4.5 GB with no scene deadline missed), the
proposed "sustained deadline misses" rule, failing closed on unreadable pressure, the /proc/meminfo reading, and a
LoopbackTransport request ended from another thread: the call returns at once and the server sees its connection
closed (what llama-server's cancellation relies on).
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from sentinel.media.clock import NS_PER_SECOND
from sentinel.scene.chat_gate import (
    MISSES_TO_STOP,
    PRESSURE_CANCEL_BYTES,
    PRESSURE_READMIT_BYTES,
    PRESSURE_STOP_NEW_BYTES,
    ChatGate,
    read_pressure,
)
from sentinel.scene.llama_server import AbortHandle, LoopbackTransport, TransportError

S = NS_PER_SECOND


def test_the_thresholds_are_the_guides_in_decimal_bytes() -> None:
    assert (PRESSURE_READMIT_BYTES, PRESSURE_STOP_NEW_BYTES, PRESSURE_CANCEL_BYTES) == (
        4_500_000_000, 4_700_000_000, 5_000_000_000)
    with pytest.raises(ValueError):
        ChatGate(stop_new_bytes=5_100_000_000)  # must order below the cancel threshold


def test_new_questions_wait_after_three_samples_above_4_7_gb_not_after_two() -> None:
    gate = ChatGate()
    for t in (1, 2):
        gate.observe(t * S, 4_700_000_001, 0)
    assert gate.allows_new() == (True, None)
    gate.observe(3 * S, 4_600_000_000, 0)  # a dip resets the count
    for t in (4, 5):
        gate.observe(t * S, 4_800_000_000, 0)
    assert gate.allows_new()[0]
    gate.observe(6 * S, 4_800_000_000, 0)
    open_, why = gate.allows_new()
    assert not open_ and why == "device memory pressure has been above 4.7 GB for 3 samples, so new questions wait"
    assert gate.take_cancel() is None  # not above 5.0 GB: a running question is left to finish


def test_above_5_gb_a_running_question_is_cancelled_once_and_new_ones_wait() -> None:
    gate = ChatGate()
    gate.observe(S, 5_000_000_001, 0)
    assert not gate.allows_new()[0]
    assert gate.take_cancel() == "device memory pressure: monitoring comes first"
    assert gate.take_cancel() is None
    gate.observe(2 * S, 5_200_000_000, 0)  # still above: the request stands until it is taken; no repeats meanwhile
    assert gate.summary()["counters"]["cancel_requests"] == 1


def test_chat_opens_again_only_after_60_s_below_4_5_gb_with_no_deadline_missed() -> None:
    gate = ChatGate()
    gate.observe(0, 5_100_000_000, 0)
    gate.take_cancel()
    gate.observe(1 * S, 4_600_000_000, 0)  # below 4.7 but not below 4.5: still closed
    for t in range(2, 61):
        gate.observe(t * S, 4_400_000_000, 0)
    assert not gate.allows_new()[0]  # 59 s below 4.5 GB
    gate.observe(62 * S, 4_400_000_000, 0)
    assert gate.allows_new() == (True, None) and gate.summary()["counters"]["reopened"] == 1


def test_sustained_deadline_misses_close_chat_and_hold_it_closed_for_60_s_after_the_last() -> None:
    gate = ChatGate()
    gate.observe(0, 3_000_000_000, 0)
    gate.observe(1 * S, 3_000_000_000, 1)  # one miss: not yet "sustained"
    assert gate.allows_new()[0]
    gate.observe(10 * S, 3_000_000_000, MISSES_TO_STOP)
    open_, why = gate.allows_new()
    assert not open_ and why == "scene analysis missed its deadline 2 times within 60 s, so new questions wait"
    for t in range(11, 70):
        gate.observe(t * S, 3_000_000_000, MISSES_TO_STOP)
    assert not gate.allows_new()[0]  # 59 s since the last miss
    gate.observe(71 * S, 3_000_000_000, MISSES_TO_STOP)
    assert gate.allows_new()[0]


def test_unreadable_pressure_closes_chat_until_readings_return_and_hold() -> None:
    gate = ChatGate()
    gate.observe(0, None, 0)
    assert gate.allows_new() == (False, "memory pressure cannot be read, so chat waits until it can")
    for t in range(1, 61):
        gate.observe(t * S, 3_000_000_000, 0)
    assert not gate.allows_new()[0]
    gate.observe(62 * S, 3_000_000_000, 0)
    assert gate.allows_new()[0]


def test_pressure_is_memtotal_minus_memavailable_in_bytes(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:        7990340 kB\nMemFree:          912344 kB\nMemAvailable:    3660000 kB\n")
    assert read_pressure(meminfo) == (7990340 - 3660000) * 1024
    meminfo.write_text("MemTotal:        7990340 kB\n")
    assert read_pressure(meminfo) is None
    meminfo.write_text("MemTotal: 10 kB\nMemAvailable: 20 kB\n")
    assert read_pressure(meminfo) is None  # impossible: never a number
    assert read_pressure(tmp_path / "absent") is None


def test_a_request_ended_from_another_thread_returns_at_once_and_its_connection_closes() -> None:
    """The mechanism D77's cancellation rests on: llama-server notices a closed connection and cancels its task."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    seen: dict[str, object] = {}

    def server() -> None:  # reads the request, never answers, and records when the client's side closed
        conn, _ = listener.accept()
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(4096)
            head, _, body = data.partition(b"\r\n\r\n")
            length = int(next(line.split(b":")[1] for line in head.split(b"\r\n")
                              if line.lower().startswith(b"content-length")))
            while len(body) < length:  # the whole request, so that closing never resets an unread one
                body += conn.recv(4096)
            seen["request"] = head.split(b"\r\n")[0]
            conn.settimeout(10.0)
            try:
                seen["closed"] = conn.recv(1) == b""
            except OSError:
                seen["closed"] = True
            seen["closed_at"] = time.monotonic()

    worker = threading.Thread(target=server, daemon=True)
    worker.start()
    transport = LoopbackTransport(listener.getsockname()[1], timeout_s=20.0)
    handle = AbortHandle()
    outcome: dict[str, object] = {}

    def ask() -> None:
        try:
            transport.post(b"{}", abort=handle)
        except TransportError as exc:
            outcome["error"], outcome["at"] = exc, time.monotonic()

    asking = threading.Thread(target=ask)
    asking.start()
    deadline = time.monotonic() + 5
    while "request" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    ended = time.monotonic()
    assert handle.abort("a scene job is waiting: monitoring comes first")
    asking.join(5.0)
    worker.join(5.0)
    listener.close()
    error = outcome["error"]
    assert isinstance(error, TransportError) and error.aborted and error.label == "aborted"
    assert outcome["at"] - ended < 0.5 and seen["closed"] is True and seen["closed_at"] - ended < 0.5
    assert seen["request"].startswith(b"POST ")
    assert not handle.abort("again") and handle.reason == "a scene job is waiting: monitoring comes first"


def test_a_handle_aborted_first_stops_the_request_from_starting() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    handle = AbortHandle()
    handle.abort("no answer in time")
    with pytest.raises(TransportError) as raised:
        LoopbackTransport(listener.getsockname()[1], timeout_s=5.0).post(b"{}", abort=handle)
    listener.settimeout(0.5)
    conn, _ = listener.accept()  # connected, then closed without a request
    with conn:
        conn.settimeout(2.0)
        assert conn.recv(1024) == b""
    listener.close()
    assert raised.value.aborted
