"""Where a colony's creatures live: in this process, or in worker processes.

A colony advances in rounds of 10 ms.  In every round each creature

1. is told what happened to it since the last round — food it ate, a feast
   shared, a rival's meal, the change in smell its last movement brought —
   hears the voices that reached it, and senses the arena *as the round
   began*;
2. lives 10 ms (brain, body, voice, thought log);
3. reports the movements it completed, what it said, its energy and
   whether it is awake.

The colony then applies everyone's movements to the arena in a shuffled
order.  Since no creature feels another's movement before the next round,
the creatures can live in separate processes — and the result is the same,
spike for spike, whether they share one process or each has its own
(``workers``).  The consequences of a movement reach a creature one round
(10 ms) after it completes, well within the ~0.3 s its synapses stay
eligible for credit.
"""

from __future__ import annotations

import copy
import os
import signal
import sys
import traceback
from typing import Any

import numpy as np

from ..runtime.agent import Agent
from ..world.base import World


class ProxyView(World):
    """A colony creature's senses and muscles: what the arena reported at the
    start of the round; completed movements are recorded for the colony."""

    name = "foraging"
    deferred = True  # the body leaves outcomes (and the approach reward) to arrive as events

    def __init__(self) -> None:
        self.obs: dict[str, np.ndarray] = {}
        self.smell = 0.0
        self.actions: list[tuple[str, float]] = []
        self.snapshot: dict[str, Any] | None = None
        self._asked = 0.0

    def observe(self) -> dict[str, np.ndarray]:
        return self.obs

    def smell_ahead(self) -> float:
        self._asked = self.smell  # remembered: the smell when the movement began
        return self.smell

    def apply(self, action: str) -> list[tuple[str, dict[str, Any]]]:
        self.actions.append((action, self._asked))
        return []

    def state_dict(self) -> dict[str, Any]:
        return dict(self.snapshot) if self.snapshot else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


def reseed(agent: Agent, seed: int) -> None:
    """New random streams, so a copied brain does not replay its parent's noise."""
    agent.brain.net.reseed(seed)
    agent.brain.rng = np.random.default_rng(seed + 1)


def jitter(agent: Agent, sigma: float, rng: np.random.Generator) -> None:
    """Multiplicative noise on the learned (plastic, non-recurrent) synapses."""
    agent.brain.net.jitter_weights(sigma, rng)


class LocalHost:
    """Creatures living in this process (their brains on ``device``)."""

    def __init__(self, forward_thoughts: bool = False, device: Any = None) -> None:
        self.device = device
        self.agents: dict[int, Agent] = {}
        self.views: dict[int, ProxyView] = {}
        self.minds: dict[int, Any] = {}
        self.forward = forward_thoughts
        self._said: dict[int, list[str]] = {}
        self._thoughts: dict[int, list[dict[str, Any]]] = {}
        self._result: dict[int, dict[str, Any]] | None = None

    # ------------------------------------------------------------ creatures
    def add(self, slot: int, state: dict[str, Any], name: str, world_spec: dict[str, Any],
            seed: int | None = None, jitter_sigma: float = 0.0, jitter_seed: int = 0,
            thoughts_path: str | None = None) -> None:
        state = dict(state)
        state["name"] = name
        view = ProxyView()
        agent = Agent.from_state(state, world=view, device=self.device)
        agent.world_spec = dict(world_spec)
        if seed is not None:
            reseed(agent, seed)
        jitter(agent, jitter_sigma, np.random.default_rng(jitter_seed))
        self._install(slot, agent, view, thoughts_path)

    def load(self, slot: int, path: str, thoughts_path: str | None = None) -> None:
        view = ProxyView()
        self._install(slot, Agent.load(path, world=view, device=self.device), view, thoughts_path)

    def _install(self, slot: int, agent: Agent, view: ProxyView, thoughts_path: str | None) -> None:
        self.agents[slot], self.views[slot] = agent, view
        self._said[slot], self._thoughts[slot] = [], []
        agent.on("utterance", lambda e, s=slot: self._said[s].append(e["text"]))
        mind = None
        if thoughts_path:
            from ..mind import ThoughtStream
            mind = ThoughtStream(agent, thoughts_path)
            mind.subscribe(lambda rec, s=slot: self._thoughts[s].append(rec) if self.forward else None)
        self.minds[slot] = mind

    def remove(self, slot: int, save_to: str | None = None, snapshot: dict[str, Any] | None = None) -> None:
        if save_to:
            self.save(slot, save_to, snapshot)
        if self.minds.get(slot) is not None:
            self.minds[slot].close()
        for d in (self.agents, self.views, self.minds, self._said, self._thoughts):
            d.pop(slot, None)

    def save(self, slot: int, path: str, snapshot: dict[str, Any] | None = None) -> None:
        self.views[slot].snapshot = snapshot
        self.agents[slot].save(path)
        if self.minds.get(slot) is not None:
            self.minds[slot]._flush()

    def state(self, slot: int) -> dict[str, Any]:
        return copy.deepcopy(self.agents[slot].state_dict())

    def info(self, slot: int) -> dict[str, Any]:
        a = self.agents[slot]
        return {"name": a.name, "device": a.device, "energy": float(a.body.energy), "awake": bool(a.brain.awake),
                "stage": a.brain.stage, "clock": a.brain.drives.clock_string(), "t_ms": a.t_ms,
                "neurons": int(a.brain.net.N), "kin": "sense.kin" in a.brain.inputs,
                "stats": dict(a.body.stats)}

    def set_forward(self, on: bool) -> None:
        self.forward = bool(on)

    # ------------------------------------------------------------ living
    def step(self, inputs: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        out = {}
        for slot, inp in inputs.items():
            agent, view = self.agents[slot], self.views[slot]
            view.obs, view.smell, view.actions = inp["obs"], inp["smell"], []
            for text, source, loudness in inp["heard"]:
                agent.hear(text, source=source, loudness=loudness)
            if inp["events"]:
                for kind, data in agent.body.receive(inp["events"]):
                    agent._emit(kind, dict(data))
            self._said[slot], self._thoughts[slot] = [], []
            agent.tick()
            out[slot] = {"actions": view.actions, "said": self._said[slot], "energy": float(agent.body.energy),
                         "awake": bool(agent.brain.awake), "thoughts": self._thoughts[slot]}
        return out

    def step_send(self, inputs: dict[int, dict[str, Any]]) -> None:
        self._result = self.step(inputs)

    def step_recv(self) -> dict[int, dict[str, Any]]:
        out, self._result = self._result or {}, None
        return out

    def close(self) -> None:
        for slot in list(self.agents):
            if self.minds.get(slot) is not None:
                self.minds[slot].close()
        self.agents.clear()


class WorkerError(RuntimeError):
    pass


def _serve(conn: Any, forward_thoughts: bool, device: Any = None) -> None:  # pragma: no cover - a worker
    # Ctrl-C and a service manager's SIGTERM are the colony's to handle: it saves
    # everyone and then closes the workers (which also exit if it disappears)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    host = LocalHost(forward_thoughts, device)
    while True:
        try:
            op, args, kwargs = conn.recv()
        except (EOFError, OSError):
            break
        try:
            conn.send(("ok", getattr(host, op)(*args, **kwargs)))
        except Exception:
            conn.send(("error", traceback.format_exc()))
        if op == "close":
            break


class RemoteHost:
    """Creatures living in a worker process (same interface as LocalHost)."""

    def __init__(self, forward_thoughts: bool = False, ctx: Any = None, device: Any = None) -> None:
        ctx = ctx or mp_context(device)
        self.conn, child = ctx.Pipe()
        dev = None if device is None else str(device)  # (a plain string crosses into a spawned process)
        self.proc = ctx.Process(target=_serve, args=(child, forward_thoughts, dev), daemon=True,
                                name="anima-colony-worker")
        self.proc.start()
        child.close()

    def _call(self, op: str, *args: Any, **kwargs: Any) -> Any:
        self.conn.send((op, args, kwargs))
        return self._answer()

    def _answer(self) -> Any:
        try:
            status, value = self.conn.recv()
        except EOFError:
            raise WorkerError("a colony worker process died") from None
        if status == "error":
            raise WorkerError(f"a colony worker failed:\n{value}")
        return value

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._call("add", *args, **kwargs)

    def load(self, *args: Any, **kwargs: Any) -> None:
        self._call("load", *args, **kwargs)

    def remove(self, *args: Any, **kwargs: Any) -> None:
        self._call("remove", *args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> None:
        self._call("save", *args, **kwargs)

    def state(self, slot: int) -> dict[str, Any]:
        return self._call("state", slot)

    def info(self, slot: int) -> dict[str, Any]:
        return self._call("info", slot)

    def set_forward(self, on: bool) -> None:
        self._call("set_forward", on)

    def step(self, inputs: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        self.step_send(inputs)
        return self.step_recv()

    def step_send(self, inputs: dict[int, dict[str, Any]]) -> None:
        self.conn.send(("step", (inputs,), {}))

    def step_recv(self) -> dict[int, dict[str, Any]]:
        return self._answer()

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self._call("close")
        except (WorkerError, OSError, EOFError):
            pass
        self.proc.join(5)
        if self.proc.is_alive():  # it ignores SIGTERM (see _serve)
            self.proc.kill()
            self.proc.join(2)
        self.conn.close()


def mp_context(device: Any = None) -> Any:
    """How worker processes are started: forked on Linux (fast, and works from
    any script or notebook); spawned elsewhere — and for creatures on a GPU,
    since a GPU library cannot be used in a forked process — where a script
    that runs a colony with workers needs the usual
    ``if __name__ == "__main__":`` guard.  ``ANIMA_MP_START`` overrides it
    (fork, spawn or forkserver)."""
    import multiprocessing as mp
    gpu = device is not None and getattr(device, "library", "numpy") in ("cupy", "torch") \
        and getattr(device, "device", "cpu") != "cpu"
    method = os.environ.get("ANIMA_MP_START") or (
        "fork" if sys.platform.startswith("linux") and not gpu else "spawn")
    return mp.get_context(method)


def make_hosts(workers: int, forward_thoughts: bool = False, device: Any = None) -> list[Any]:
    """``workers`` <= 1: one in-process host; otherwise that many worker
    processes (each creature's brain on ``device``)."""
    if workers <= 1:
        return [LocalHost(forward_thoughts, device)]
    ctx = mp_context(device)
    return [RemoteHost(forward_thoughts, ctx, device) for _ in range(workers)]


__all__ = ["ProxyView", "LocalHost", "RemoteHost", "WorkerError", "make_hosts", "reseed", "jitter"]
