#!/usr/bin/env python3
"""Steady, repeatable load on the status page during a guarded device check (D80): chat questions and a live viewer.

    alpha_load.py chat OUT.jsonl [--port 18090] [--every-s 6] [--wait-s 600] [--for-s SECONDS] [--grounding G.jsonl]
    alpha_load.py view OUT.jsonl [--port 18090] [--wait-s 600]

Operator tooling, not part of the runtime. Started before the guarded run (D50: nothing under a deadline): each waits
up to ``--wait-s`` for the page on 127.0.0.1, then works until the page has not answered for GONE_S, and exits 0.

- ``chat`` asks every ``--every-s`` seconds through POST /chat, exactly as the page does (its own token, Origin and
  JSON body), and follows each accepted question to its end. It alternates two fixed protocol questions (QUESTIONS):
  a short one, which a question started between scene checks should finish, and a long one, whose answer runs to
  the token bound and so is likely to outlast the gap and be ended for a periodic scene job (D81), so that both paths
  are exercised. Per question it records numbers and fixed labels only: when it was asked (CLOCK_MONOTONIC, the
  guard's and the runtime's clock), which question, the state it ended in, the reason label for a refusal or a
  cancellation, the frame the answer describes (stream epoch, frame, ingest time), its age when asked and answered,
  whether it was historical, the answer's length and whether it was cut short. **Never the answer's text.** With
  ``--for-s`` it stops asking that long after its first accepted question (the page answers while the models load,
  when chat is not yet available) and exits; the operator may then ask their own on the page.
- ``chat --grounding G.jsonl`` (the grounding check G; sessions 65, 105 and 106) also asks the four fixed G questions
  (GROUNDING), each once, with no earlier exchange: each at the first of the regular slots asked at or after its
  offset from A0, in place of that slot's fixed question; the rate, the alternation and every other slot are unchanged.
  A0 is the monotonic time at which the first question was accepted: when the POST that the page answered with 202
  returned, never when its answer finished. A G slot's record in OUT.jsonl has the kind ``grounding`` and its ID, and
  no text. Its record in G.jsonl (created 0600 here, refused if it exists) also holds the question and **the answer's
  text as received**: the only place this tool keeps any answer's text. Nothing is ever retried, and the tool never
  asks longer than it would without G. A slot that ends without an answer keeps its state and the runtime's reason
  (cancelled, unavailable, busy, ...); a slot whose request got no response is ``no_response``; a slot never reached is
  written ``not_asked`` with why, when the tool ends (the page gone, its own end, or SIGTERM). This tool prints
  nothing; G.jsonl is read only by the operator's judging interface (grounding.py judge) and the privacy counts.
- ``view`` pulls live pictures through GET /live/frame as the page's player does (one request at a time, the newest
  newer than the last), and records per second: pictures received, their bytes, and their age when sent (the
  device's clock); never a picture.

Both read the token from the page, as a browser would; it is never written down. Stdlib only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

QUESTIONS = {"short": "What is visible in this picture? Answer in one sentence.",
             "long": "Describe everything visible in this picture in detail, object by object, with its colour and "
                     "where it is."}
GONE_S = 15.0  # the page has not answered for this long: the runtime has stopped
# The grounding check G (session 65's wording, unchanged; offsets from A0 decided in session 105):
# (ID, offset s, question)
GROUNDING = (("G1", 60.0, "Is there a water bottle in this picture?"),
             ("G2", 150.0, "Are there any cosmetic products beside the water bottle?"),
             ("G3", 240.0, "What brand name is printed on the water bottle?"),
             ("G4", 330.0, "What was happening here one minute before this picture was taken?"))
GROUNDING_KIND = "grounding"
GROUNDING_MAX_ANSWER = 600  # scene.chat.CHAT_MAX_ANSWER: the runtime keeps no more of an answer
NO_RESPONSE = "the page did not answer"
NOT_ASKED = {"page_never_answered": "the page never answered", "never_accepted": "chat never accepted a question",
             "page_gone": "the page went away before this slot", "stopped_asking": "the load tool stopped asking "
             "before this slot", "stopped": "the load tool was stopped before this slot"}
TOKEN = re.compile(rb'data-token="([0-9a-f]{32})"')
FRAME = re.compile(r"[0-9]{1,9}-[0-9]{1,12}")
FINAL = {"answered", "error", "timeout", "busy", "cancelled", "unavailable", "paused", "no_fresh_frame", "refused",
         "invalid", "off", "unknown"}


class Page:
    """The loopback page, asked as the browser asks it."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.origin = f"http://127.0.0.1:{port}"
        self.token: str | None = None

    def request(self, method: str, path: str, body: bytes | None = None, *, timeout: float = 10.0,
                **headers: str) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers={"Host": f"127.0.0.1:{self.port}", **headers})
            response = conn.getresponse()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read(2_000_000)
        finally:
            conn.close()

    def refresh_token(self) -> bool:
        status, _, body = self.request("GET", "/")
        match = TOKEN.search(body) if status == 200 else None
        self.token = match.group(1).decode() if match else None
        return self.token is not None

    def json(self, method: str, path: str, document: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        headers = {"X-Sentinel-Token": self.token or "", "Accept": "application/json"}
        body = None
        if document is not None:
            body = json.dumps(document).encode()
            headers.update({"Origin": self.origin, "Content-Type": "application/json"})
        status, _, data = self.request(method, path, body, **headers)
        try:
            parsed = json.loads(data)
        except ValueError:
            parsed = {}
        return status, parsed if isinstance(parsed, dict) else {}


def _wait_for(page: Page, wait_s: float) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            if page.refresh_token():
                return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


def _label(text: Any) -> str | None:
    """A fixed reason label as the runtime gave it (fixed text), bounded; None if absent."""
    return text[:200] if isinstance(text, str) else None


def _record(asked_mono: float, kind: str, document: dict[str, Any], http_status: int) -> dict[str, Any]:
    frame = document.get("frame") if isinstance(document.get("frame"), dict) else {}
    answer = document.get("answer")
    return {
        "asked_mono_s": round(asked_mono, 3), "kind": kind, "http_status": http_status,
        "state": document.get("state") if document.get("state") in FINAL else "unreadable",
        "reason": _label(document.get("reason")) if document.get("state") != "answered" else None,
        "stream_epoch": frame.get("stream_epoch"), "frame_seq": frame.get("frame_seq"),
        "ingest_utc": frame.get("ingest_utc"), "age_ms_when_asked": frame.get("age_ms_when_asked"),
        "age_ms_when_answered": document.get("age_ms_when_answered"), "historical": document.get("historical"),
        "answer_ms": document.get("answer_ms"), "answer_chars": len(answer) if isinstance(answer, str) else None,
        "cut_short": document.get("cut_short"),
    }


def _grounding_record(slot: tuple[str, float, str], a0: float | None, asked: float | None, replaced: str | None,
                      document: dict[str, Any], http_status: int | None, *, state: str | None = None,
                      reason: str | None = None) -> dict[str, Any]:
    """One G slot's record for G.jsonl: _record's numbers and labels, the slot, and the answer's text as received."""
    g, offset, question = slot
    numbers = _record(asked if asked is not None else 0.0, GROUNDING_KIND, document, http_status or 0)
    answer = document.get("answer") if numbers["state"] == "answered" else None
    record = {"g": g, "question": question, "offset_s": offset,
              "a0_mono_s": round(a0, 3) if a0 is not None else None,
              "asked_mono_s": numbers["asked_mono_s"] if asked is not None else None,
              "asked_after_a0_s": round(asked - a0, 3) if asked is not None and a0 is not None else None,
              "replaced_kind": replaced, "http_status": http_status,
              **{k: numbers[k] for k in ("state", "reason", "stream_epoch", "frame_seq", "ingest_utc",
                                         "age_ms_when_asked", "age_ms_when_answered", "historical", "answer_ms",
                                         "answer_chars", "cut_short")},
              "answer": answer[:GROUNDING_MAX_ANSWER] if isinstance(answer, str) else None}
    if state is not None:
        record.update(state=state, reason=reason)
    return record


def _private(path: Path):
    """G.jsonl, created here 0600 (never an existing file: O_EXCL)."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")


def _stopped(signum: int, frame: Any) -> None:
    raise SystemExit(0)  # SIGTERM with G: the slots not reached are still written (chat's finally)


def chat(out: Path, port: int, every_s: float, wait_s: float, for_s: float | None = None,
         grounding: Path | None = None) -> int:
    page = Page(port)
    try:
        g_sink = _private(grounding) if grounding is not None else None
    except OSError as exc:
        print(f"STOP: the grounding record cannot be created ({type(exc).__name__}); nothing was asked",
              file=sys.stderr)
        return 2
    pending, a0, ended = list(GROUNDING) if g_sink is not None else [], None, "stopped"

    def written(record: dict[str, Any]) -> None:
        g_sink.write(json.dumps(record) + "\n")
        g_sink.flush()

    if g_sink is not None:
        signal.signal(signal.SIGTERM, _stopped)
    try:
        with out.open("w") as sink:
            if not _wait_for(page, wait_s):
                ended = "page_never_answered"
                sink.write(json.dumps({"event": "page_never_answered"}) + "\n")
                return 0
            sink.write(json.dumps({"event": "started", "mono_s": round(time.monotonic(), 3), "questions": QUESTIONS,
                                   "every_s": every_s}) + "\n")
            silent_since, n, first = None, 0, None
            while True:
                asked = time.monotonic()
                if for_s is not None and first is not None and asked - first >= for_s:
                    ended = "stopped_asking"
                    sink.write(json.dumps({"event": "stopped_asking", "mono_s": round(asked, 3)}) + "\n")
                    return 0
                kind = "short" if n % 2 == 0 else "long"
                n += 1
                slot = pending.pop(0) if pending and a0 is not None and asked >= a0 + pending[0][1] else None
                try:
                    status, document = page.json("POST", "/chat", {"question": slot[2] if slot else QUESTIONS[kind],
                                                                   "history": []})
                    if status == 202 and first is None:
                        first = asked  # chat is available: --for-s counts from here
                    if status == 202 and a0 is None:
                        a0 = time.monotonic()  # A0: the first acceptance, as its response returned
                    if status == 403:  # a restarted runtime: its new page has a new token
                        page.refresh_token()
                    job = document.get("job") if isinstance(document.get("job"), str) else None
                    while job and document.get("state") in ("sending", "waiting", "answering"):
                        time.sleep(0.5)
                        status, document = page.json("GET", f"/chat/{job}")
                    record = _record(asked, GROUNDING_KIND if slot else kind, document, status)
                    if slot:
                        record["g"] = slot[0]
                        written(_grounding_record(slot, a0, asked, kind, document, status))
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    silent_since = None
                except OSError:
                    if slot:  # asked once: never asked again
                        written(_grounding_record(slot, a0, asked, kind, {}, None, state="no_response",
                                                  reason=NO_RESPONSE))
                    silent_since = silent_since or time.monotonic()
                    if time.monotonic() - silent_since > GONE_S:
                        ended = "page_gone"
                        sink.write(json.dumps({"event": "page_gone", "mono_s": round(time.monotonic(), 3)}) + "\n")
                        return 0
                time.sleep(max(0.0, every_s - (time.monotonic() - asked)))
    finally:
        if g_sink is not None:
            why = NOT_ASKED["never_accepted"] if a0 is None and ended != "page_never_answered" else NOT_ASKED[ended]
            for slot in pending:
                written(_grounding_record(slot, a0, None, None, {}, None, state="not_asked", reason=why))
            g_sink.close()


def view(out: Path, port: int, wait_s: float) -> int:
    page = Page(port)
    with out.open("w") as sink:
        if not _wait_for(page, wait_s):
            sink.write(json.dumps({"event": "page_never_answered"}) + "\n")
            return 0
        sink.write(json.dumps({"event": "started", "mono_s": round(time.monotonic(), 3)}) + "\n")
        after, second, tally, silent_since = None, None, None, None
        while True:
            now = time.monotonic()
            if second is None or now - second >= 1.0:
                if tally is not None:
                    sink.write(json.dumps(tally) + "\n")
                    sink.flush()
                second = now
                tally = {"second_mono_s": round(now, 3), "pictures": 0, "bytes": 0, "max_age_ms_when_sent": None,
                         "states": {}}
            try:
                path = "/live/frame" + (f"?after={after}" if after else "")
                status, headers, body = page.request("GET", path, timeout=10.0,
                                                     **{"X-Sentinel-Token": page.token or ""})
                silent_since = None
                if status == 200 and headers.get("content-type") == "image/jpeg" and FRAME.fullmatch(
                        headers.get("x-frame", "")):
                    after = headers["x-frame"]
                    tally["pictures"] += 1
                    tally["bytes"] += len(body)
                    age = int(headers.get("x-frame-age-ms", "-1"))
                    tally["max_age_ms_when_sent"] = max(age, tally["max_age_ms_when_sent"] or 0)
                else:
                    try:
                        state = json.loads(body).get("state")
                    except ValueError:
                        state = None
                    label = state if isinstance(state, str) and len(state) < 30 else f"http_{status}"
                    tally["states"][label] = tally["states"].get(label, 0) + 1
                    if status == 403:
                        page.refresh_token()
                    time.sleep(0.5)
            except OSError:
                silent_since = silent_since or time.monotonic()
                if time.monotonic() - silent_since > GONE_S:
                    sink.write(json.dumps(tally) + "\n")
                    sink.write(json.dumps({"event": "page_gone", "mono_s": round(time.monotonic(), 3)}) + "\n")
                    return 0
                time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat questions and a live viewer for a guarded device check.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("chat", "view"):
        sub = commands.add_parser(name)
        sub.add_argument("out", type=Path)
        sub.add_argument("--port", type=int, default=18090)
        sub.add_argument("--wait-s", type=float, default=600.0)
        if name == "chat":
            sub.add_argument("--every-s", type=float, default=6.0)
            sub.add_argument("--for-s", type=float, default=None,
                             help="stop asking this long after the first accepted question")
            sub.add_argument("--grounding", type=Path, default=None,
                             help="ask the grounding check's four questions too; their records (with the answers' "
                                  "text) go to this new file, 0600")
    args = parser.parse_args(argv)
    if args.command == "chat":
        return chat(args.out, args.port, args.every_s, args.wait_s, args.for_s, args.grounding)
    return view(args.out, args.port, args.wait_s)


if __name__ == "__main__":
    sys.exit(main())
