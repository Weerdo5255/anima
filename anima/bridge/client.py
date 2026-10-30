"""Talk to a creature served by ``anima serve`` from another program.

Only the standard library is needed, so you can copy this file into any
project::

    from anima.bridge import BridgeClient

    anima = BridgeClient("http://127.0.0.1:7777")
    print(anima.state()["drives"])
    anima.hear("ba ba :)")
    for e in anima.follow(kinds=["utterance", "thought"]):
        print(e["kind"], e.get("text"))

and to *be* one of its programs (a ``remote = true`` tool)::

    def bell(call):                       # call: who, when, its drives...
        return {"say": "ding", "reward": 0.3}

    anima.serve({"bell": bell})           # answers every call to "bell"
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Iterator


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, url: str = "http://127.0.0.1:7777", token: str | None = None,
                 timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # a local service: never route it through a proxy
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ------------------------------------------------------------ plumbing
    def request(self, method: str, path: str, body: dict[str, Any] | None = None,
                params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        if params:
            from urllib.parse import urlencode
            path += "?" + urlencode({k: (",".join(v) if isinstance(v, (list, tuple, set)) else v)
                                     for k, v in params.items() if v is not None})
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                msg = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:
                msg = ""
            raise BridgeError(f"{method} {path}: {exc.code} {msg or exc.reason}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise BridgeError(f"cannot reach the creature at {self.url} ({exc}); is `anima serve` running?") \
                from None

    # ------------------------------------------------------------ reading
    def info(self) -> dict[str, Any]:
        return self.request("GET", "/")

    def state(self) -> dict[str, Any]:
        return self.request("GET", "/state")

    def thoughts(self, n: int = 50, kinds: Iterable[str] | None = None, since: float | None = None
                 ) -> list[dict[str, Any]]:
        return self.request("GET", "/thoughts", params={"n": n, "kinds": list(kinds) if kinds else None,
                                                         "since": since})["thoughts"]

    def tools(self) -> dict[str, Any]:
        return self.request("GET", "/tools")

    def pending(self) -> list[dict[str, Any]]:
        return self.request("GET", "/pending")["pending"]

    def events(self, since: int = 0, kinds: Iterable[str] | None = None, wait: float = 0.0) -> dict[str, Any]:
        """``{"events": [...], "last": seq, "dropped": bool}`` — events after ``since``,
        waiting up to ``wait`` seconds (max 30) for the first one."""
        return self.request("GET", "/events", params={"since": since, "kinds": list(kinds) if kinds else None,
                                                       "wait": wait}, timeout=wait + self.timeout)

    def follow(self, kinds: Iterable[str] | None = None, since: int | None = None,
               stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
        """Yield events as they happen (long polling), forever or until ``stop()``."""
        kinds = list(kinds) if kinds else None
        if since is None:
            since = self.events(0, ["-"])["last"]  # start from now
        while stop is None or not stop():
            try:
                res = self.events(since, kinds, wait=10.0)
            except BridgeError:
                if stop is not None and stop():
                    return
                time.sleep(1.0)
                continue
            for e in res["events"]:
                since = max(since, int(e["seq"]))
                yield e
            since = max(since, int(res["last"])) if kinds is None else since

    # ------------------------------------------------------------ acting on it
    def hear(self, text: str) -> None:
        """Speak to it (``:)`` / ``:(`` are a smile / a frown)."""
        self.request("POST", "/hear", {"text": text})

    def reward(self, amount: float, source: str = "client") -> float:
        """A treat (+) or a buzzer (-); clipped to [-1, 1].  Returns what was given."""
        return self.request("POST", "/reward", {"amount": amount, "source": source})["amount"]

    def pleasure(self, amount: float = 0.6, credit: str = "auto", source: str = "client") -> dict[str, Any]:
        """Give it pleasure (0-1) for what it just did (``credit``: auto, speech, movement, now)."""
        return self.request("POST", "/pleasure", {"amount": amount, "for": credit, "source": source})

    def pain(self, amount: float = 0.5, credit: str = "auto", source: str = "client") -> dict[str, Any]:
        """Give it pain (0-1) for what it just did (``credit``: auto, speech, movement, now)."""
        return self.request("POST", "/pain", {"amount": amount, "for": credit, "source": source})

    def feed(self, where: str = "ahead") -> None:
        self.request("POST", "/feed", {"where": where})

    def tool_result(self, call_id: int, answer: Any) -> bool:
        """Answer a remote program's call (text, or {"say", "reward", "food", "ok"})."""
        try:
            return bool(self.request("POST", "/tool_result", {"call_id": call_id, "answer": answer})["ok"])
        except BridgeError as exc:
            if " 409 " in str(exc):
                return False
            raise

    def serve(self, programs: dict[str, Callable[[dict[str, Any]], Any]],
              stop: Callable[[], bool] | None = None, on_call: Callable[[dict[str, Any], Any], None] | None = None
              ) -> None:
        """Be the creature's remote programs: answer every call to these names.

        ``programs`` maps a tool name (a ``remote = true`` tool in its tools
        file) to ``fn(call) -> answer``.  Calls already waiting are answered
        first.  Runs until ``stop()`` returns True (or forever)."""
        seen: set[int] = set()

        def answer(call: dict[str, Any]) -> None:
            cid = int(call["call_id"])
            if cid in seen or call.get("tool") not in programs:
                return
            seen.add(cid)
            try:
                result = programs[call["tool"]](call)
            except Exception as exc:  # a failing program answers with a failure
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.tool_result(cid, result)
            if on_call is not None:
                on_call(call, result)

        for call in self.pending():
            answer(call)
        for e in self.follow(kinds=["tool_call"], stop=stop):
            if e.get("program") == "remote":
                answer(e.get("call") or {"call_id": e["call_id"], "tool": e["tool"]})


__all__ = ["BridgeClient", "BridgeError"]
