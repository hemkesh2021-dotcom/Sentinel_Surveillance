"""A loopback stand-in for llama-server's /health, /props, /slots and /v1/chat/completions (session 115, parts 11 and
15): tests and the chat preemption check's dry run only, never the model.

Its one slot follows llama-server b8932 (tools/server/server-context.cpp at 98dc1418e):
- **Each request, chat or scene, is a new task.** When the request is launched, the slot takes the task's id and its
  ``n_predict`` (the request's ``max_tokens``) at once.
- **The counters stay stale through the prompt.** The slot keeps the previous task's ``n_decoded`` and ``n_remain``
  while the new task's prompt is processed (``prompt_s`` for a chat request, ``scene_prompt_s`` for a scene request).
- **Then they restart.** ``n_decoded`` becomes 0 when the prompt is done, and ``n_remain`` is recomputed from the first
  decoded token on.
- **After the task ends,** /slots keeps showing its id and its final counters.

A chat request then "decodes" for the time its question is given (by a word in it), and stops as soon as its connection
is closed, as llama-server does when a request is aborted. Otherwise it answers synthetic text with llama-server's
timings. A scene request (``response_format`` set) decodes ``scene_tokens`` tokens over ``scene_s``, then answers a
valid report, or the given invalid text, with ``finish_reason`` "stop" and its usage.

/slots also carries a ``generated`` text, and /props a template and a model path. A reader must drop them all (the real
server sends slot text only with LLAMA_SERVER_SLOTS_DEBUG).
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
TEMPLATE = "synthetic chat template the props reader must drop"
BUILD = "b8932-98dc1418e"  # the installed build the fake follows
REPORT = {"persons_visible": 0, "fire_or_smoke": False, "threat": "low", "observations": ["shapes on white"],
          "uncertainty": "low", "summary": "Labelled shapes on a white background."}


class FakeLlamaServer:
    def __init__(self, port: int, *, decode_s: dict[str, float], tokens_per_s: float = 50.0, scene_s: float = 0.2,
                 scene_text: str | None = None, prompt_s: float = 0.0, scene_prompt_s: float = 0.0,
                 scene_tokens: int = 60, build: str = BUILD, slot_task_ids: bool = True) -> None:
        self.port, self.decode_s, self.tokens_per_s, self.scene_s = port, decode_s, tokens_per_s, scene_s
        self.prompt_s, self.scene_prompt_s, self.scene_tokens = prompt_s, scene_prompt_s, scene_tokens
        self.build, self.slot_task_ids = build, slot_task_ids
        self.scene_text = json.dumps(REPORT) if scene_text is None else scene_text
        self._lock = threading.Lock()
        # one slot; "task" stays the last task's id once it ends (llama-server's task_prev)
        self._slot: dict[str, Any] = {"processing": False, "task": None, "n_predict": None, "n_decoded": 0,
                                      "n_remain": -1}
        self._tasks = 0
        self.closed_by_client = 0  # chat requests whose connection the client closed while running
        self.completed = {"chat": 0, "scene": 0}
        self._httpd: ThreadingHTTPServer | None = None

    def _launch(self, n_predict: int) -> None:
        """A new task in the slot: its id and n_predict now, the counters left as the previous task left them."""
        with self._lock:
            self._tasks += 1
            self._slot.update(processing=True, task=self._tasks, n_predict=n_predict)

    def _prompt_done(self) -> None:
        with self._lock:
            self._slot.update(n_decoded=0)  # n_remain only follows from the first token

    def _decoded(self, n: int) -> None:
        with self._lock:
            self._slot.update(n_decoded=n, n_remain=self._slot["n_predict"] - n)

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
                elif self.path == "/props":
                    self._json(200, {"build_info": server.build, "chat_template": TEMPLATE,
                                     "model_path": "/synthetic/model.gguf", "total_slots": 1})
                elif self.path == "/slots":
                    with server._lock:
                        slot = dict(server._slot)
                    document: dict[str, Any] = {"id": 0, "n_ctx": 4096, "is_processing": slot["processing"]}
                    if slot["task"] is not None:  # as llama-server: the task fields once the slot has had a task
                        if server.slot_task_ids:
                            document["id_task"] = slot["task"]
                        document["params"] = {"n_predict": slot["n_predict"], "max_tokens": slot["n_predict"]}
                        document["next_token"] = [{"has_next_token": slot["processing"], "has_new_line": False,
                                                   "n_remain": slot["n_remain"], "n_decoded": slot["n_decoded"]}]
                        document["generated"] = GENERATED
                    self._json(200, [document])
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                cap = int(body["max_tokens"])
                server._launch(cap)
                try:
                    if "response_format" in body:
                        self._scene(cap)
                    else:
                        self._chat(body, cap)
                finally:
                    with server._lock:
                        server._slot.update(processing=False)

            def _wait(self, seconds: float) -> bool:
                """Waits; False as soon as the client has closed its side of the connection."""
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    if self._closed():
                        return False
                    time.sleep(0.01)
                return True

            def _scene(self, cap: int) -> None:
                time.sleep(server.scene_prompt_s)
                server._prompt_done()
                tokens = min(cap - 1, server.scene_tokens)
                began = time.monotonic()
                while time.monotonic() - began < server.scene_s:
                    server._decoded(max(1, min(tokens, int((time.monotonic() - began) / server.scene_s * tokens))))
                    time.sleep(0.01)
                server._decoded(tokens)
                server.completed["scene"] += 1
                self._json(200, {"choices": [{"finish_reason": "stop", "index": 0,
                                              "message": {"role": "assistant", "content": server.scene_text}}],
                                 "usage": {"completion_tokens": tokens, "prompt_tokens": 325,
                                           "total_tokens": tokens + 325},
                                 "timings": {"predicted_n": tokens}})

            def _chat(self, body: dict[str, Any], cap: int) -> None:
                text = body["messages"][-1]["content"][-1]["text"]
                seconds = next((s for word, s in server.decode_s.items() if word in text), 0.0)
                if not self._wait(server.prompt_s):  # the prompt and the picture: stale counters, nothing decoded
                    server.closed_by_client += 1
                    return
                server._prompt_done()
                began = time.monotonic()
                while time.monotonic() - began < seconds:
                    if self._closed():
                        server.closed_by_client += 1
                        return
                    server._decoded(min(cap, int((time.monotonic() - began) * server.tokens_per_s) + 1))
                    time.sleep(0.01)
                tokens = min(cap, max(1, int(seconds * server.tokens_per_s)))
                server._decoded(tokens)
                server.completed["chat"] += 1
                self._json(200, {"choices": [{"finish_reason": "length" if tokens >= cap else "stop", "index": 0,
                                              "message": {"role": "assistant", "content": ANSWER}}],
                                 "timings": {"prompt_ms": 5.0, "predicted_ms": seconds * 1000.0,
                                             "predicted_n": tokens}})

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
