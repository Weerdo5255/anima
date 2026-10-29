"""Programs the creature can use.

The creature can run external programs — but only programs you list, and
only in ways it can actually produce.  It cannot type a command or invent
arguments; each program is bound to something it *does*:

``say = "b"``  the program runs when it says a syllable starting with "b"
               (``say = "ba"``: exactly that syllable).  Works with any
               creature: speaking is how it acts on the world of programs,
               the way a child's "mama" brings someone.
``hand = 2``   the program runs when it reaches with hand 2.  Hands are a
               small extra motor area; build a creature with them
               (``Agent.create(hands=4)``, ``anima chat --brain pet.npz --new --hands 4``).

What happens next is up to the program.  Its answer comes back as
consequences the creature can feel:

* a **sound** it hears (the program's output, played phoneme by phoneme —
  without the social reward of being talked to);
* **food** placed in front of it;
* a **reward** (+) or punishment (-), like a treat or a buzzer, in [-1, 1];
* for hands, a short signal on ``sense.hands`` saying which hand's program
  answered.

It learns which programs are worth using from these consequences, with the
same dopamine-gated plasticity that shapes its babbling and foraging.

Program kinds (one per tool):

``command = ["{python}", "lamp.py", "{hunger}"]``
    run a local program (never through a shell; ``{python}`` is the Python
    running anima, the other placeholders are filled from the creature's
    state, and the whole call arrives as JSON on stdin); its stdout is the
    answer.
``url = "http://127.0.0.1:8080/lamp"``
    POST the call as JSON to a local web service; the response is the answer.
``function = "mypackage.module:fn"``
    call a Python function ``fn(call) -> answer``.
``remote = true``
    nobody is launched: the call is published (``anima serve`` streams it as
    a ``tool_call`` event) and any program may answer it with
    ``POST /tool_result``.

An answer is either plain text (heard; reward ``reward_ok`` or
``reward_error`` by exit status) or a JSON object with any of
``{"say": "...", "reward": 0.5, "food": "ahead" | "left" | "right" | "near",
"ok": true}``.

Safety: only listed programs run; commands run without a shell, with a
timeout, a cap on output and a cooldown; there is a global limit on calls
per minute; every call and answer is an event (and a line in the thought
log).  Programs run with your user's permissions, so list only programs
you would let a curious animal press the button of.
"""

from __future__ import annotations

import heapq
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

FOOD_PLACES = ("ahead", "left", "right", "near")
#: names usable as {placeholders} in a command's arguments (other braces are left alone)
PLACEHOLDERS = ("python", "tool", "call_id", "what", "t", "clock", "stage", "energy",
                "hunger", "fatigue", "boredom", "loneliness")
_PLACEHOLDER = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")


@dataclass
class ToolResult:
    ok: bool = True
    say: str = ""
    reward: float = 0.0
    food: str | None = None
    error: str = ""
    raw: str = ""
    seconds: float = 0.0

    def summary(self) -> str:
        parts = []
        if self.say:
            parts.append(f"'{self.say[:40]}'")
        if self.food:
            parts.append(f"food {self.food}")
        if self.reward:
            parts.append(f"reward {self.reward:+.2f}")
        if not self.ok:
            parts.append(f"failed{': ' + self.error[:60] if self.error else ''}")
        return ", ".join(parts) or "(nothing)"


def parse_answer(answer: Any, ok: bool = True, reward_ok: float = 0.0, reward_error: float = 0.0,
                 error: str = "") -> ToolResult:
    """Interpret a program's answer (text, JSON text, dict or ToolResult)."""
    if isinstance(answer, ToolResult):
        return answer
    raw = ""
    data: Any = answer
    if isinstance(answer, (bytes, bytearray)):
        answer = answer.decode("utf-8", errors="replace")
    if isinstance(answer, str):
        raw = answer
        text = answer.strip()
        data = None
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError:
                data = None
        if data is None:
            first = " ".join(text.split())
            return ToolResult(ok=ok, say=first, reward=reward_ok if ok else reward_error, error=error, raw=raw)
    if data is None:
        return ToolResult(ok=ok, reward=reward_ok if ok else reward_error, error=error, raw=raw)
    if not isinstance(data, dict):
        return ToolResult(ok=ok, say=" ".join(str(data).split()), reward=reward_ok if ok else reward_error,
                          error=error, raw=raw or json.dumps(data, default=str))
    ok = bool(data.get("ok", ok))
    say = str(data.get("say", data.get("text", "")) or "")
    try:
        reward = float(data["reward"]) if "reward" in data else (reward_ok if ok else reward_error)
    except (TypeError, ValueError):
        reward = 0.0
    food = data.get("food")
    if food is True:
        food = "ahead"
    food = str(food) if food in FOOD_PLACES else None
    return ToolResult(ok=ok, say=" ".join(say.split()), reward=reward, food=food,
                      error=str(data.get("error", error) or ""), raw=raw or json.dumps(data, default=str))


# ===================================================================== tools
@dataclass
class Tool:
    """A program the creature may use.  Subclasses implement :meth:`call`."""

    name: str
    say: str | None = None  # trigger: a syllable ("ba") or its consonant ("b")
    hand: int | None = None  # trigger: a hand
    description: str = ""
    cooldown_s: float = 3.0  # creature seconds before it can run again
    timeout_s: float = 10.0
    speak: bool = True  # play the answer's text into its ears
    max_say: int = 60  # characters of the answer it hears
    reward_ok: float = 0.0  # reward for a plain-text answer / exit status 0
    reward_error: float = 0.0  # ... and for a failure
    kind: str = "tool"

    def call(self, call: dict[str, Any]) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, call: dict[str, Any]) -> ToolResult:
        """Call the program, never raising: failures become results."""
        t0 = time.perf_counter()
        try:
            res = self.call(call)
        except Exception as exc:  # a broken program must not hurt the creature
            res = ToolResult(ok=False, reward=self.reward_error, error=f"{type(exc).__name__}: {exc}")
        res.seconds = time.perf_counter() - t0
        res.reward = float(np.clip(res.reward, -1.0, 1.0))
        res.say = res.say[: self.max_say]
        return res

    def describe(self) -> str:
        trig = f"say '{self.say}'" if self.say else f"hand {self.hand}"
        return f"{self.name} ({self.kind}, {trig})" + (f": {self.description}" if self.description else "")

    def info(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if not callable(v)}
        d.pop("fn", None)
        return d


@dataclass
class CommandTool(Tool):
    """Run a local program (argv list, no shell)."""

    argv: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None
    max_output: int = 4000
    kind: str = "command"

    def __post_init__(self) -> None:
        if not self.argv or not all(isinstance(a, str) for a in self.argv):
            raise ValueError(f"tool {self.name!r}: command must be a non-empty list of strings")

    def call(self, call: dict[str, Any]) -> ToolResult:
        values = _placeholder_values(call)
        argv = [_PLACEHOLDER.sub(lambda m: values[m.group(1)], a) for a in self.argv]
        env = None
        if self.env:
            env = dict(os.environ)
            env.update({str(k): str(v) for k, v in self.env.items()})
        try:
            proc = subprocess.run(argv, input=json.dumps(call), capture_output=True, text=True,
                                  timeout=self.timeout_s, cwd=self.cwd, env=env, shell=False)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, reward=self.reward_error, error=f"timed out after {self.timeout_s:g}s")
        except OSError as exc:
            return ToolResult(ok=False, reward=self.reward_error, error=str(exc))
        out = (proc.stdout or "")[: self.max_output]
        ok = proc.returncode == 0
        err = "" if ok else f"exit {proc.returncode}: {(proc.stderr or '').strip()[:200]}"
        return parse_answer(out, ok, self.reward_ok, self.reward_error, err)


@dataclass
class HTTPTool(Tool):
    """POST the call as JSON to a web service; the response body is the answer."""

    url: str = ""
    kind: str = "http"

    def call(self, call: dict[str, Any]) -> ToolResult:
        req = urllib.request.Request(self.url, data=json.dumps(call).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout_s) as resp:
            body = resp.read(64_000).decode("utf-8", errors="replace")
            ok = 200 <= resp.status < 300
        return parse_answer(body, ok, self.reward_ok, self.reward_error)


@dataclass
class FunctionTool(Tool):
    """Call a Python function ``fn(call) -> answer``.

    ``inline=True`` runs it on the creature's own thread and delivers the
    answer exactly ``latency_s`` creature-seconds later — deterministic, for
    experiments.  Otherwise it runs in a worker thread like any program.
    """

    fn: Callable[[dict[str, Any]], Any] | None = None
    inline: bool = False
    latency_s: float = 0.3
    kind: str = "function"

    def call(self, call: dict[str, Any]) -> ToolResult:
        if self.fn is None:
            raise RuntimeError("no function")
        return parse_answer(self.fn(call), True, self.reward_ok, self.reward_error)


@dataclass
class RemoteTool(Tool):
    """Answered from outside: the call is published and someone replies with
    :meth:`ToolBelt.complete` (``POST /tool_result`` through ``anima serve``)."""

    kind: str = "remote"

    def call(self, call: dict[str, Any]) -> ToolResult:  # never called directly
        raise RuntimeError("remote tools are answered from outside")


def _placeholder_values(call: dict[str, Any]) -> dict[str, str]:
    d = call.get("drives", {})
    vals = {"python": sys.executable or "python3",
            "tool": call.get("tool", ""), "call_id": call.get("call_id", ""), "what": call.get("what", ""),
            "t": f"{call.get('t', 0.0):.1f}", "clock": call.get("clock", ""), "stage": call.get("stage", ""),
            "energy": f"{call.get('energy', 0.0):.3f}"}
    for k in ("hunger", "fatigue", "boredom", "loneliness"):
        vals[k] = f"{float(d.get(k, 0.0)):.3f}"
    return {k: str(v) for k, v in vals.items()}


# ===================================================================== config
def tool_from_dict(d: dict[str, Any], base_dir: Path | None = None) -> Tool:
    d = dict(d)
    name = str(d.pop("name", "")).strip()
    if not name:
        raise ValueError("every tool needs a name")
    common = {k: d.pop(k) for k in ("say", "hand", "description", "cooldown_s", "timeout_s", "speak",
                                    "max_say", "reward_ok", "reward_error") if k in d}
    if "say" in common:
        common["say"] = str(common["say"]).lower()
    if "hand" in common:
        common["hand"] = int(common["hand"])
    kinds = [k for k in ("command", "url", "function", "remote") if d.get(k)]
    if len(kinds) != 1:
        raise ValueError(f"tool {name!r}: give exactly one of command / url / function / remote = true")
    kind = kinds[0]
    if kind == "command":
        cmd = d.pop("command")
        if isinstance(cmd, str):
            raise ValueError(f"tool {name!r}: command must be a list, e.g. [\"python3\", \"lamp.py\"] "
                             "(it never goes through a shell)")
        cwd = d.pop("cwd", None)
        if cwd is None and base_dir is not None:
            cwd = str(base_dir)
        return CommandTool(name=name, argv=[str(a) for a in cmd], cwd=cwd, env=d.pop("env", None),
                           max_output=int(d.pop("max_output", 4000)), **common)
    if kind == "url":
        return HTTPTool(name=name, url=str(d.pop("url")), **common)
    if kind == "function":
        mod, _, attr = str(d.pop("function")).partition(":")
        import importlib
        fn = getattr(importlib.import_module(mod), attr)
        return FunctionTool(name=name, fn=fn, **common)
    return RemoteTool(name=name, **common)


def load_tools(path: str | Path) -> list[Tool]:
    """Read a tool list from TOML (``[[tool]]`` tables) or JSON (``{"tool": [...]}``)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            try:
                import tomli as tomllib  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError("reading TOML needs Python 3.11+ or `pip install tomli`; "
                                   "or write the tools as JSON") from exc
        cfg = tomllib.loads(text)
    items = cfg.get("tool", cfg.get("tools", []))
    return [tool_from_dict(t, path.parent.resolve()) for t in items]


# ===================================================================== the belt
class ToolBelt:
    """Binds programs to what the creature does, runs them, brings the answers back.

    Never blocks the creature: programs run in worker threads (or are answered
    from outside) and their answers are applied on the creature's own thread
    at its next tick.
    """

    def __init__(self, agent: Any, tools: list[Tool], max_calls_per_min: float = 30.0,
                 max_workers: int = 4, feedback_s: float = 0.5) -> None:
        self.agent = agent
        self.tools = list(tools)
        self.max_calls_per_min = max_calls_per_min
        self.feedback_s = feedback_s
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")
        n_hands = len(agent.body.hands.labels) if getattr(agent.body, "hands", None) is not None else 0
        self.say_bindings: list[tuple[str, Tool]] = []
        self.hand_bindings: dict[int, Tool] = {}
        for t in self.tools:
            if (t.say is None) == (t.hand is None):
                raise ValueError(f"tool {t.name!r}: bind it with exactly one of say = ... or hand = ...")
            if t.say is not None:
                if not t.say.isalpha():
                    raise ValueError(f"tool {t.name!r}: say must be letters, e.g. 'b' or 'ba'")
                self.say_bindings.append((t.say, t))
            else:
                if n_hands == 0:
                    raise ValueError(f"tool {t.name!r} is bound to a hand, but this creature has no hands "
                                     "(start a new one with hands, e.g. `--brain pet.npz --new --hands 4`, "
                                     "or bind the program to a syllable: say = \"b\")")
                if not 0 <= t.hand < n_hands:
                    raise ValueError(f"tool {t.name!r}: hand {t.hand} does not exist (it has {n_hands})")
                if t.hand in self.hand_bindings:
                    raise ValueError(f"hand {t.hand} is bound twice")
                self.hand_bindings[t.hand] = t
        self.n_hands = n_hands
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="anima-tool")
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._busy: dict[str, int] = {}
        self._last_call: dict[str, float] = {}
        self._recent: deque[float] = deque()
        self._pending: dict[int, tuple[Tool, dict[str, Any], float]] = {}  # remote calls: deadline
        self._scheduled: list[tuple[float, int, Tool, dict[str, Any], ToolResult]] = []
        self._feedback = np.zeros(n_hands)
        self.history: deque[dict[str, Any]] = deque(maxlen=500)
        self.stats: dict[str, dict[str, Any]] = {t.name: {"calls": 0, "ok": 0, "failed": 0, "skipped": 0,
                                                          "reward": 0.0} for t in self.tools}
        self.empty_hands = 0
        self._closed = False
        agent.on("syllable", self._on_syllable)
        agent.on("hand", self._on_hand)
        agent.add_tick_hook(self._on_tick)

    # ------------------------------------------------------------ triggers (agent thread)
    def _on_syllable(self, e: dict[str, Any]) -> None:
        text = e["text"]
        for trig, tool in self.say_bindings:
            if (len(trig) == 1 and text.startswith(trig)) or text == trig:
                self._trigger(tool, "say", text)

    def _on_hand(self, e: dict[str, Any]) -> None:
        tool = self.hand_bindings.get(int(e["hand"]))
        if tool is None:
            self.empty_hands += 1
            return
        self._trigger(tool, "hand", int(e["hand"]))

    def _context(self, tool: Tool, how: str, what: Any, call_id: int) -> dict[str, Any]:
        a = self.agent
        snap = a.latest
        return {"call_id": call_id, "tool": tool.name, "how": how, "what": what,
                "t": round(a.t_ms / 1000.0, 3), "clock": snap.get("clock", ""),
                "stage": snap.get("stage", ""), "energy": round(float(a.body.energy), 4),
                "drives": {k: round(float(v), 4) for k, v in a.brain.drives.as_dict().items()}}

    def _trigger(self, tool: Tool, how: str, what: Any) -> None:
        if self._closed:
            return
        now = self.agent.t_ms / 1000.0
        reason = None
        if tool.name in self._busy:
            reason = "still running"
        elif now - self._last_call.get(tool.name, -1e9) < tool.cooldown_s:
            reason = "cooling down"
        else:
            while self._recent and now - self._recent[0] > 60.0:
                self._recent.popleft()
            if len(self._recent) >= self.max_calls_per_min:
                reason = "too many calls this minute"
        if reason is not None:
            self.stats[tool.name]["skipped"] += 1
            return
        call_id = next(self._ids)
        call = self._context(tool, how, what, call_id)
        self._busy[tool.name] = call_id
        self._last_call[tool.name] = now
        self._recent.append(now)
        self.stats[tool.name]["calls"] += 1
        self.history.append({"t": now, "event": "call", "tool": tool.name, "call_id": call_id,
                             "how": how, "what": what})
        self.agent._emit("tool_call", {"tool": tool.name, "call_id": call_id, "how": how, "what": what,
                                       "program": tool.kind, "call": call})
        if isinstance(tool, RemoteTool):
            with self._lock:
                self._pending[call_id] = (tool, call, now + tool.timeout_s)
        elif isinstance(tool, FunctionTool) and tool.inline:
            res = tool.run(call)
            heapq.heappush(self._scheduled, (now + max(0.0, tool.latency_s), call_id, tool, call, res))
        else:
            fut = self._executor.submit(tool.run, call)
            fut.add_done_callback(lambda f, tool=tool, call=call: self._from_worker(tool, call, f))

    def _from_worker(self, tool: Tool, call: dict[str, Any], fut: Any) -> None:
        try:
            res = fut.result()
        except Exception as exc:  # pragma: no cover - run() never raises
            res = ToolResult(ok=False, error=str(exc))
        if not self._closed:
            self.agent.call_soon(lambda ag: self._deliver(tool, call, res))

    def complete(self, call_id: int, answer: Any) -> bool:
        """Answer a remote tool's call (any thread).  Returns False if the
        call is unknown or already answered/timed out."""
        with self._lock:
            entry = self._pending.pop(int(call_id), None)
        if entry is None:
            return False
        tool, call, _ = entry
        result = parse_answer(answer, True, tool.reward_ok, tool.reward_error)
        result.reward = float(np.clip(result.reward, -1.0, 1.0))
        result.say = result.say[: tool.max_say]
        self.agent.call_soon(lambda ag: self._deliver(tool, call, result))
        return True

    # ------------------------------------------------------------ answers (agent thread)
    def _deliver(self, tool: Tool, call: dict[str, Any], res: ToolResult) -> None:
        a = self.agent
        call_id = call["call_id"]
        if self._busy.get(tool.name) == call_id:
            del self._busy[tool.name]
        st = self.stats[tool.name]
        st["ok" if res.ok else "failed"] += 1
        st["reward"] += res.reward
        if res.reward:
            a.brain.deliver_reward(res.reward, f"program:{tool.name}")
        if res.food and hasattr(a.world, "offer_food"):
            a.world.offer_food(res.food)
        if res.say and tool.speak:
            a.body.hear_sound(res.say)
        if tool.hand is not None and self.n_hands:
            self._feedback[tool.hand] = 1.0 if res.ok else 0.5
        now = a.t_ms / 1000.0
        self.history.append({"t": now, "event": "result", "tool": tool.name, "call_id": call_id,
                             "ok": res.ok, "reward": res.reward, "say": res.say, "food": res.food,
                             "error": res.error, "seconds": round(res.seconds, 3)})
        a._emit("tool_result", {"tool": tool.name, "call_id": call_id, "ok": res.ok, "reward": res.reward,
                                "say": res.say, "food": res.food, "error": res.error,
                                "seconds": round(res.seconds, 3), "summary": res.summary()})

    def _on_tick(self, agent: Any) -> None:
        now = agent.t_ms / 1000.0
        while self._scheduled and self._scheduled[0][0] <= now:
            _, _, tool, call, res = heapq.heappop(self._scheduled)
            self._deliver(tool, call, res)
        if self._pending:
            with self._lock:
                late = [cid for cid, (_, _, dl) in self._pending.items() if now >= dl]
                expired = [(cid, self._pending.pop(cid)) for cid in late]
            for _, (tool, call, _) in expired:
                self._deliver(tool, call, ToolResult(ok=False, reward=tool.reward_error,
                                                     error="no answer in time"))
        if self.n_hands and "sense.hands" in agent.brain.inputs:
            self._feedback *= np.exp(-agent.tick_ms / 1000.0 / self.feedback_s)
            agent.brain.set_input("sense.hands", np.clip(self._feedback, 0.0, 1.0))

    # ------------------------------------------------------------ views
    def hand_names(self) -> dict[int, str]:
        """hand index -> the program it runs (for the thought log)."""
        return {h: t.name for h, t in self.hand_bindings.items()}

    def pending_remote(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(call) for _, call, _ in self._pending.values()]

    def describe(self) -> str:
        return "; ".join(t.describe() for t in self.tools)

    def report(self) -> str:
        lines = []
        for t in self.tools:
            s = self.stats[t.name]
            lines.append(f"  {t.describe()}: {s['calls']} calls ({s['ok']} ok, {s['failed']} failed, "
                         f"{s['skipped']} skipped), total reward {s['reward']:+.2f}")
        if self.n_hands:
            lines.append(f"  reaches with an empty hand: {self.empty_hands}")
        return "\n".join(lines)

    def close(self) -> None:
        """Stop listening and let running programs finish in the background."""
        self._closed = True
        a = self.agent
        a.off("syllable", self._on_syllable)
        a.off("hand", self._on_hand)
        a.remove_tick_hook(self._on_tick)
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["Tool", "CommandTool", "HTTPTool", "FunctionTool", "RemoteTool", "ToolResult", "ToolBelt",
           "load_tools", "tool_from_dict", "parse_answer", "PLACEHOLDERS", "FOOD_PLACES"]
