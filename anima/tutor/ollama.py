"""A small, dependency-free client for a local Ollama server.

Only the standard library is used (``urllib``), so anima still needs nothing
but NumPy.  Talks to the documented REST API:

* ``GET  /api/version``  -> ``{"version": ...}``
* ``GET  /api/tags``     -> ``{"models": [{"name": ...}, ...]}``
* ``POST /api/chat``     -> ``{"message": {"role", "content", "thinking"?}, "done", ...}``

Structured outputs: ``format`` may be a JSON schema (Ollama >= 0.5); the
model's reply is then a JSON string conforming to it.  Small or "thinking"
models still occasionally wrap JSON in prose or ``<think>`` blocks, so
:meth:`OllamaClient.chat_json` extracts the first JSON object defensively and,
if a server rejects schema-format requests, retries with plain JSON mode.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaError(RuntimeError):
    """Any failure talking to Ollama."""


class OllamaUnavailable(OllamaError):
    """The server is not reachable (not installed, not running, wrong host)."""


class OllamaModelMissing(OllamaError):
    """The server is up but the requested model has not been pulled."""


def resolve_host(host: str | None = None) -> str:
    """Normalise a host the way the ``ollama`` CLI does (``OLLAMA_HOST``)."""
    h = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip()
    if "://" not in h:
        h = "http://" + h
    u = urllib.parse.urlsplit(h)
    hostname = u.hostname or "127.0.0.1"
    if hostname in ("0.0.0.0", "::", ""):
        hostname = "127.0.0.1"
    port = u.port or 11434
    netloc = f"[{hostname}]:{port}" if ":" in hostname else f"{hostname}:{port}"
    return urllib.parse.urlunsplit((u.scheme or "http", netloc, u.path.rstrip("/"), "", ""))


def _is_local(url: str) -> bool:
    h = urllib.parse.urlsplit(url).hostname or ""
    return h in ("localhost", "::1") or h.startswith("127.")


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_json(text: str) -> Any:
    """Parse the first JSON object in ``text`` (tolerates prose, code fences,
    ``<think>`` blocks and trailing junk)."""
    if text is None:
        raise ValueError("empty reply")
    t = _THINK.sub("", text).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise ValueError(f"no JSON object in reply: {text[:120]!r}")


@dataclass
class ChatResult:
    content: str
    thinking: str = ""
    seconds: float = 0.0
    eval_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """Minimal synchronous client.  Thread-safe (no shared mutable state)."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None, timeout: float = 180.0,
                 temperature: float = 0.3, num_ctx: int = 4096, num_predict: int = 1024,
                 keep_alive: str | int = "30m", think: bool | str | None = None) -> None:
        self.model = model
        self.host = resolve_host(host)
        self.timeout = timeout
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}
        self.keep_alive = keep_alive
        self.think = think
        handlers: list[Any] = [urllib.request.ProxyHandler({})] if _is_local(self.host) else []
        self._opener = urllib.request.build_opener(*handlers)
        self._schema_ok = True

    # ------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        url = self.host + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(detail).get("error", detail)
            except Exception:
                msg = detail or str(exc)
            if path == "/api/chat" and (exc.code == 404 or "not found" in str(msg).lower()):
                raise OllamaModelMissing(
                    f"model {self.model!r} is not available on {self.host} ({msg}). "
                    f"Pull it first:  ollama pull {self.model}") from exc
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {msg}") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaUnavailable(
                f"cannot reach Ollama at {self.host} ({reason}). Is it installed and running? "
                "Start it with `ollama serve` (or the Ollama app), or pass --host.") from exc
        try:
            return json.loads(payload) if payload.strip() else {}
        except json.JSONDecodeError:
            # some proxies return NDJSON even for stream=false; take the last object
            lines = [ln for ln in payload.splitlines() if ln.strip()]
            return json.loads(lines[-1])

    # ------------------------------------------------------------ API
    def version(self) -> str:
        return str(self._request("GET", "/api/version", timeout=10).get("version", "?"))

    def models(self) -> list[str]:
        return [m.get("name") or m.get("model", "") for m in
                self._request("GET", "/api/tags", timeout=10).get("models", [])]

    def has_model(self, name: str | None = None) -> bool:
        """True if ``name`` can be used as is ("llama3.2" means "llama3.2:latest")."""
        name = name or self.model
        have = set(self.models())
        return name in have or (":" not in name and f"{name}:latest" in have)

    def similar_models(self, name: str | None = None) -> list[str]:
        base = (name or self.model).split(":")[0]
        return sorted(h for h in self.models() if h.split(":")[0] == base)

    def chat(self, messages: list[dict[str, str]], format: Any = None,
             options: dict[str, Any] | None = None) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                "options": {**self.options, **(options or {})},
                                "keep_alive": self.keep_alive}
        if format is not None:
            body["format"] = format
        if self.think is not None:
            body["think"] = self.think
        t0 = time.perf_counter()
        raw = self._request("POST", "/api/chat", body)
        msg = raw.get("message") or {}
        return ChatResult(content=str(msg.get("content", "")), thinking=str(msg.get("thinking", "") or ""),
                          seconds=time.perf_counter() - t0, eval_count=int(raw.get("eval_count", 0) or 0),
                          raw=raw)

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None,
                  options: dict[str, Any] | None = None) -> tuple[Any, ChatResult]:
        """Chat and parse the reply as JSON (schema-constrained when supported)."""
        fmt: Any = schema if (schema is not None and self._schema_ok) else "json"
        try:
            res = self.chat(messages, format=fmt, options=options)
        except OllamaError as exc:
            if fmt != "json" and "format" in str(exc).lower():
                self._schema_ok = False  # older server: fall back to JSON mode
                res = self.chat(messages, format="json", options=options)
            else:
                raise
        try:
            return extract_json(res.content), res
        except ValueError:
            if res.thinking:
                return extract_json(res.thinking), res
            raise


__all__ = ["OllamaClient", "OllamaError", "OllamaUnavailable", "OllamaModelMissing",
           "ChatResult", "extract_json", "resolve_host", "DEFAULT_MODEL", "DEFAULT_HOST"]
