"""A local connection for other programs: talk to a living creature over HTTP.

``anima serve`` (or :class:`Bridge` in your own code) runs the creature in
real time and opens a small JSON web service next to it::

    GET  /                  what this is, the endpoints, the creature's programs
    GET  /state             drives, neuromodulators, sleep stage, what it is doing
    GET  /thoughts?n=50&kinds=intention,feeling&since=T     its latest thoughts
    GET  /events?since=SEQ&kinds=utterance,thought&wait=20  events after SEQ (long poll)
    GET  /stream?kinds=...  the same, as server-sent events (curl -N, EventSource)
    GET  /tools             its programs and how they have gone
    GET  /pending           calls to remote programs still waiting for an answer
    POST /hear        {"text": "ba ba :)"}                   speak to it
    POST /reward      {"amount": 0.5, "source": "my-game"}   treat (+) or buzzer (-), |amount| <= 1
    POST /feed        {"where": "ahead"}                     put food near it
    POST /tool_result {"call_id": 12, "answer": {"say": "ding", "reward": 0.3}}
                                                             answer a remote program's call

Events are everything the creature does — ``utterance``, ``syllable``,
``action``, ``hand``, ``ate``, ``bump``, ``stage``, ``replay``, ``memory``,
``heard``, ``tool_call``, ``tool_result``, ``thought`` — each with a sequence
number ``seq``; ask for ``since`` the last one you saw.

Like everything else it never makes the creature wait: requests are served
on their own threads and act on the creature at its next 10 ms tick.

It listens on 127.0.0.1 only.  To listen on another address you must set a
token (then every request needs ``Authorization: Bearer <token>``), because
anyone who can reach it can reward, feed and talk to your creature, and
answer its programs' calls.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_BODY = 64_000
MAX_WAIT_S = 30.0


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, default=_jsonable).encode("utf-8")


class BridgeError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Bridge:
    """Serve a living creature to other programs (see the module docstring).

    agent: the creature (start it yourself, e.g. ``agent.start()``).
    thoughts: a :class:`~anima.mind.ThoughtStream` for ``/thoughts`` (optional).
    belt: a :class:`~anima.tools.ToolBelt` for ``/tools``, ``/pending`` and
        ``/tool_result`` (optional).
    port: 0 picks a free port (see :attr:`port` / :attr:`url`).
    token: required on every request if set; needed for a non-local host.
    """

    def __init__(self, agent: Any, host: str = "127.0.0.1", port: int = 7777, token: str | None = None,
                 thoughts: Any = None, belt: Any = None, max_reward: float = 1.0, keep: int = 2000) -> None:
        if host not in LOCAL_HOSTS and not token:
            raise ValueError(f"refusing to listen on {host!r} without a token: anyone who can reach it "
                             "could reward, feed and talk to the creature (set token=...)")
        self.agent = agent
        self.host = host
        self.token = token
        self.thoughts = thoughts
        self.belt = belt
        self.max_reward = float(max_reward)
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=keep)
        self._seq = 0
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        self.requests = 0
        agent.on("*", self._on_event)
        bridge = self

        class Handler(_Handler):
            pass
        Handler.bridge = bridge
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> "Bridge":
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                            name="anima-bridge", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        with self._cond:
            self._cond.notify_all()
        self.agent.off("*", self._on_event)
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(5.0)
            self._thread = None
        self._server.server_close()

    # ------------------------------------------------------------ events (agent thread)
    def _on_event(self, e: dict[str, Any]) -> None:
        if e.get("kind") == "snapshot":
            return
        with self._cond:
            self._seq += 1
            self._events.append((self._seq, {**e, "seq": self._seq}))
            self._cond.notify_all()

    def events_after(self, since: int, kinds: set[str] | None = None, wait: float = 0.0,
                     limit: int = 500) -> tuple[list[dict[str, Any]], int, bool]:
        """Events with seq > since (waiting up to ``wait`` s for one); also
        returns the latest seq and whether older events were already dropped."""
        deadline = time.monotonic() + max(0.0, min(wait, MAX_WAIT_S))
        with self._cond:
            while True:
                oldest = self._events[0][0] if self._events else self._seq + 1
                items = [e for s, e in self._events if s > since and (not kinds or e.get("kind") in kinds)]
                remaining = deadline - time.monotonic()
                if items or remaining <= 0 or self._stopping.is_set():
                    break
                self._cond.wait(remaining)
            return items[:limit], self._seq, since + 1 < oldest and since > 0

    # ------------------------------------------------------------ the API
    def handle(self, method: str, path: str, query: dict[str, list[str]], body: dict[str, Any]) -> Any:
        a = self.agent
        q = {k: v[-1] for k, v in query.items()}
        if method == "GET":
            if path == "/":
                return {"name": a.name, "anima": __version__, "alive": a.alive,
                        "neurons": int(a.brain.net.N), "t": round(a.t_ms / 1000.0, 2),
                        "thoughts": self.thoughts is not None,
                        "tools": [t.describe() for t in self.belt.tools] if self.belt else [],
                        "endpoints": ["GET /state", "GET /thoughts", "GET /events", "GET /stream", "GET /tools",
                                      "GET /pending", "POST /hear", "POST /reward", "POST /feed",
                                      "POST /tool_result"]}
            if path == "/state":
                snap = dict(a.latest)
                snap.pop("rates_history", None)
                return {"name": a.name, "t": round(a.t_ms / 1000.0, 3), **snap}
            if path == "/thoughts":
                if self.thoughts is None:
                    raise BridgeError(404, "this creature's thoughts are not being read (no ThoughtStream)")
                kinds = tuple(_split(q.get("kinds")))
                since = float(q["since"]) if "since" in q else None
                n = _int(q.get("n"), 50, 0, 1000)
                return {"thoughts": self.thoughts.recent(n, kinds or None, since)}
            if path == "/events":
                since = _int(q.get("since"), 0, 0, None)
                items, last, dropped = self.events_after(since, set(_split(q.get("kinds"))) or None,
                                                         _float(q.get("wait"), 0.0, 0.0, MAX_WAIT_S))
                return {"events": items, "last": last, "dropped": dropped}
            if path == "/tools":
                if self.belt is None:
                    return {"tools": [], "stats": {}}
                return {"tools": [t.info() for t in self.belt.tools], "stats": self.belt.stats,
                        "empty_hands": self.belt.empty_hands}
            if path == "/pending":
                return {"pending": self.belt.pending_remote() if self.belt else []}
            raise BridgeError(404, f"no such endpoint: GET {path}")
        if method == "POST":
            if path == "/hear":
                text = str(body.get("text", ""))[:500]
                if not text.strip():
                    raise BridgeError(400, "say something: {\"text\": \"...\"}")
                a.hear(text)
                return {"ok": True}
            if path == "/reward":
                try:
                    amount = float(body.get("amount"))
                except (TypeError, ValueError):
                    raise BridgeError(400, "amount must be a number") from None
                if not math.isfinite(amount):
                    raise BridgeError(400, "amount must be finite")
                amount = float(np.clip(amount, -self.max_reward, self.max_reward))
                source = "bridge:" + str(body.get("source", "client"))[:40]
                a.call_soon(lambda ag: ag.brain.deliver_reward(amount, source))
                return {"ok": True, "amount": amount}
            if path == "/feed":
                from ..world.foraging import OFFER_PLACES
                where = str(body.get("where", "ahead")).lower()
                if not hasattr(a.world, "offer_food"):
                    raise BridgeError(409, "there is no food in this world")
                if where not in OFFER_PLACES:
                    raise BridgeError(400, f"where must be one of {', '.join(OFFER_PLACES)}")
                a.call_soon(lambda ag: ag.world.offer_food(where))
                return {"ok": True, "where": where}
            if path == "/tool_result":
                if self.belt is None:
                    raise BridgeError(404, "this creature has no programs")
                try:
                    call_id = int(body.get("call_id"))
                except (TypeError, ValueError):
                    raise BridgeError(400, "call_id must be the integer from the tool_call event") from None
                ok = self.belt.complete(call_id, body.get("answer", ""))
                if not ok:
                    raise BridgeError(409, f"call {call_id} is not waiting for an answer "
                                      "(unknown, already answered or timed out)")
                return {"ok": True}
            raise BridgeError(404, f"no such endpoint: POST {path}")
        raise BridgeError(405, f"method {method} not allowed")


def _split(v: str | None) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def _int(v: str | None, default: int, lo: int, hi: int | None) -> int:
    try:
        x = int(v) if v is not None else default
    except ValueError:
        raise BridgeError(400, f"not an integer: {v!r}") from None
    return max(lo, x if hi is None else min(hi, x))


def _float(v: str | None, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v) if v is not None else default
    except ValueError:
        raise BridgeError(400, f"not a number: {v!r}") from None
    return float(min(hi, max(lo, x))) if math.isfinite(x) else default


class _Handler(BaseHTTPRequestHandler):
    bridge: Bridge
    server_version = f"anima/{__version__}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, status: int, payload: Any) -> None:
        data = _dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        tok = self.bridge.token
        if not tok:
            return True
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {tok}" or self.headers.get("X-Anima-Token", "") == tok

    def _route(self, method: str) -> None:
        b = self.bridge
        b.requests += 1
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"
        try:
            if not self._authorized():
                raise BridgeError(401, "missing or wrong token (Authorization: Bearer <token>)")
            body: dict[str, Any] = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise BridgeError(413, "request too large")
                raw = self.rfile.read(n) if n else b""
                if raw:
                    try:
                        body = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        raise BridgeError(400, "the body must be JSON") from None
                    if not isinstance(body, dict):
                        raise BridgeError(400, "the body must be a JSON object")
            if method == "GET" and path == "/stream":
                self._stream(parse_qs(url.query))
                return
            self._send(200, b.handle(method, path, parse_qs(url.query), body))
        except BridgeError as exc:
            self._send(exc.status, {"ok": False, "error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # pragma: no cover - never take the server down
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def do_PUT(self) -> None:
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "use GET or POST"})

    do_DELETE = do_PUT

    def _stream(self, query: dict[str, list[str]]) -> None:
        b = self.bridge
        kinds = set(_split(query.get("kinds", [""])[-1])) or None
        last = _int(query.get("since", [None])[-1], b._seq, 0, None)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while not b._stopping.is_set():
                items, newest, _ = b.events_after(last, kinds, wait=15.0)
                if not items:
                    last = max(last, newest) if kinds is None else last
                    self.wfile.write(b": still here\n\n")
                for e in items:
                    last = e["seq"]
                    self.wfile.write(b"id: %d\nevent: %s\ndata: %s\n\n" % (e["seq"], str(e["kind"]).encode(),
                                                                         _dumps(e)))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
