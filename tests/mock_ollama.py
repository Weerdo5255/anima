"""A tiny stand-in for an Ollama server, for tests (no model, no network).

Implements GET /api/version, GET /api/tags and POST /api/chat with the same
request/response shapes as Ollama.  The reply content is produced by a
``responder(body) -> str`` callable so tests can simulate good models, slow
models, chatty models, broken models and old servers.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class MockOllama:
    def __init__(self, models: tuple[str, ...] = ("tiny:latest",),
                 responder: Callable[[dict[str, Any]], str] | None = None, delay: float = 0.0,
                 reject_schema: bool = False, port: int = 0) -> None:
        self.models = list(models)
        self.responder = responder or (lambda body: json.dumps(
            {"thought": "watching", "actions": [{"tool": "wait"}]}))
        self.delay = delay
        self.reject_schema = reject_schema
        self.requests: list[dict[str, Any]] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _send(self, code: int, obj: dict[str, Any]) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/api/version":
                    self._send(200, {"version": "0.12.6"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m, "model": m, "size": 1, "digest": "x",
                                                 "details": {"family": "mock"}} for m in mock.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                mock.requests.append(body)
                if self.path != "/api/chat":
                    self._send(404, {"error": "not found"})
                    return
                model = body.get("model", "")
                if model not in mock.models and f"{model}:latest" not in mock.models:
                    self._send(404, {"error": f'model "{model}" not found, try pulling it first'})
                    return
                if mock.reject_schema and isinstance(body.get("format"), dict):
                    self._send(400, {"error": "invalid format: expected \"json\" or a JSON schema"})
                    return
                if mock.delay:
                    time.sleep(mock.delay)
                content = mock.responder(body)
                self._send(200, {"model": model, "created_at": "2026-10-28T00:00:00Z",
                                 "message": {"role": "assistant", "content": content},
                                 "done": True, "done_reason": "stop", "total_duration": 1,
                                 "eval_count": 42})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "MockOllama":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
