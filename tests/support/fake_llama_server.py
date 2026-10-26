"""A loopback stand-in for llama-server's /health, /slots and /v1/chat/completions (session 115, part 11): tests and the
chat preemption check's dry run only, never the model.

A chat request first processes its prompt for ``prompt_s`` (its slot processing, nothing decoded yet), then "decodes"
for the time its question is given (by a word in it), advancing /slots' n_decoded, and stops
as soon as its connection is closed, as llama-server b8932 does when a request is aborted (it notices the closed
connection while its task runs). Otherwise it answers synthetic text with llama-server's timings. A scene request
(``response_format`` set) answers a valid report after ``scene_s``, or the given invalid text. /slots also carries a
``generated`` text, which a reader must drop (the real server sends it only with LLAMA_SERVER_SLOTS_DEBUG).
"""

from __future__ import annotations

import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sentinel.scene.server import ServerState, ServerStatus

ANSWER = "A synthetic answer about labelled shapes."  # never to appear in a check's result
GENERATED = "synthetic text the slots reader must drop"
REPORT = {"persons_visible": 0, "fire_or_smoke": False, "threat": "low", "observations": ["shapes on white"],
          "uncertainty": "low", "summary": "Labelled shapes on a white background."}


class FakeLlamaServer:
    def __init__(self, port: int, *, decode_s: dict[str, float], tokens_per_s: float = 50.0, scene_s: float = 0.2,
                 scene_text: str | None = None, prompt_s: float = 0.0) -> None:
        self.port, self.decode_s, self.tokens_per_s, self.scene_s = port, decode_s, tokens_per_s, scene_s
        self.prompt_s = prompt_s
        self.scene_text = json.dumps(REPORT) if scene_text is None else scene_text
        self._lock = threading.Lock()
        self._slot: dict[str, Any] = {"processing": False, "n_decoded": 0, "n_remain": 0, "task": 0}
        self.closed_by_client = 0  # chat requests whose connection the client closed while decoding
        self.completed = {"chat": 0, "scene": 0}
        self._httpd: ThreadingHTTPServer | None = None

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, code: int, document: Any) -> None:
                data = json.dumps(document).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._json(200, {"status": "ok"})
                elif self.path == "/slots":
                    with server._lock:
                        slot = dict(server._slot)
                    self._json(200, [{"id": 0, "n_ctx": 4096, "is_processing": slot["processing"],
                                      "id_task": slot["task"], "params": {"n_predict": 120},
                                      "next_token": [{"has_next_token": slot["processing"], "has_new_line": False,
                                                      "n_remain": slot["n_remain"], "n_decoded": slot["n_decoded"]}],
                                      "generated": GENERATED}])
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if "response_format" in body:
                    time.sleep(server.scene_s)
                    server.completed["scene"] += 1
                    self._json(200, {"choices": [{"finish_reason": "stop", "index": 0,
                                                  "message": {"role": "assistant", "content": server.scene_text}}]})
                    return
                text = body["messages"][-1]["content"][-1]["text"]
                seconds = next((s for word, s in server.decode_s.items() if word in text), 0.0)
                cap = int(body["max_tokens"])
                with server._lock:
                    server._slot.update(processing=True, n_decoded=0, n_remain=cap, task=server._slot["task"] + 1)
                try:
                    prompt_end = time.monotonic() + (server.prompt_s if seconds > 0 else 0.0)
                    while time.monotonic() < prompt_end:  # the prompt and the picture: processing, nothing decoded
                        if self._closed():
                            server.closed_by_client += 1
                            return
                        time.sleep(0.01)
                    began = time.monotonic()
                    while time.monotonic() - began < seconds:
                        if self._closed():
                            server.closed_by_client += 1
                            return
                        decoded = min(cap, int((time.monotonic() - began) * server.tokens_per_s) + 1)
                        with server._lock:
                            server._slot.update(n_decoded=decoded, n_remain=cap - decoded)
                        time.sleep(0.01)
                    tokens = min(cap, max(1, int(seconds * server.tokens_per_s)))
                    server.completed["chat"] += 1
                    self._json(200, {"choices": [{"finish_reason": "length" if tokens >= cap else "stop", "index": 0,
                                                  "message": {"role": "assistant", "content": ANSWER}}],
                                     "timings": {"prompt_ms": 5.0, "predicted_ms": seconds * 1000.0,
                                                 "predicted_n": tokens}})
                finally:
                    with server._lock:
                        server._slot.update(processing=False)

            def _closed(self) -> bool:
                """True once the client has closed (or shut down) its side of the connection."""
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


class FakeServerProcess:
    """In the place of LlamaServerProcess: start() serves the fake on the port and reports the runtime's ready state."""

    def __init__(self, fake: FakeLlamaServer) -> None:
        self.fake, self._state = fake, ServerState.NOT_STARTED

    def __call__(self, binary: Any, model: Any, mmproj: Any, port: int) -> "FakeServerProcess":
        assert port == self.fake.port
        return self

    def start(self, ready_timeout_s: float) -> ServerStatus:
        self.fake.start()
        self._state = ServerState.READY
        return self.status()

    def stop(self, grace_s: float) -> bool:
        self.fake.stop()
        self._state = ServerState.STOPPED
        return True

    def status(self) -> ServerStatus:
        return ServerStatus(self._state, None, 0 if self._state is ServerState.STOPPED else None, "17/17", True)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
