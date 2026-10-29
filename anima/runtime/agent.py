"""The agent: a brain in a body in a world, living on its own clock.

The life loop never blocks.  Each 10 ms tick it

1. drains the inbox (anything you said since the last tick) — if the inbox
   is empty it simply carries on;
2. lets the body sense the world;
3. runs 10 ms of spiking dynamics;
4. updates drives, neuromodulators, sleep, memory;
5. lets the body act, and publishes whatever it did (moved, ate, babbled,
   fell asleep...) as events.

Input is just more sensation.  Output is whatever behaviour the dynamics
produce.  Neither waits for the other.

Typical use::

    agent = Agent.create()            # default infant brain in a foraging world
    agent.on("utterance", lambda e: print(e["text"]))
    agent.start()                     # lives in a background thread, real time
    agent.hear("hello little one")    # any time, from any thread
    ...
    agent.stop(); agent.save("my_anima.npz")
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

from ..body.body import Body, BodyConfig
from ..body.motor import MotorConfig
from ..body.voice import VoiceConfig, parse_social_signals
from ..brain.brain import Brain
from ..brain.builder import build_brain
from ..brain.presets import PRESETS
from ..world import WORLDS
from ..world.base import EmptyWorld, World
from .persistence import load_state, save_state

Listener = Callable[[dict[str, Any]], None]


class Agent:
    def __init__(self, brain: Brain, world: World | None = None,
                 body_config: BodyConfig | None = None, name: str = "anima",
                 world_spec: dict[str, Any] | None = None) -> None:
        self.brain = brain
        self.world = world if world is not None else EmptyWorld()
        self.world_spec = world_spec or {"type": getattr(self.world, "name", "custom"), "kwargs": {}}
        self.body = Body(brain, self.world, body_config)
        self.name = name
        self.tick_ms = 10.0
        self.t_ms = 0.0
        self.log: deque[tuple[float, str, dict[str, Any]]] = deque(maxlen=2000)
        self.latest: dict[str, Any] = {}
        self.speed_measured = 0.0
        self.lagging = False
        self._inbox: queue.Queue = queue.Queue()
        self._listeners: dict[str, list[Listener]] = defaultdict(list)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._snapshot_every = 10  # ticks
        self._ticks = 0
        self._tick_hooks: list[Callable[["Agent"], None]] = []
        self._snapshot()

    # ------------------------------------------------------------ factory
    @classmethod
    def create(cls, preset: str = "infant", world: str = "foraging", seed: int = 7,
               world_kwargs: dict[str, Any] | None = None, body_config: BodyConfig | None = None,
               spec: dict[str, Any] | None = None, **preset_kwargs: Any) -> "Agent":
        """Build an agent from a named preset (or an explicit ``spec``)."""
        if spec is None:
            spec = PRESETS[preset](seed=seed, **preset_kwargs)
        brain = build_brain(spec, seed=seed)
        wk = dict(world_kwargs or {})
        if world == "foraging":
            wk.setdefault("seed", seed + 1)
        w = WORLDS[world](**wk)
        return cls(brain, w, body_config, world_spec={"type": world, "kwargs": wk})

    # ------------------------------------------------------------ I/O
    def hear(self, text: str) -> None:
        """Speak to the agent.  Thread-safe; returns immediately."""
        self._inbox.put(("text", text))

    def call_soon(self, fn: Callable[["Agent"], None]) -> None:
        """Run ``fn(agent)`` on the agent's own thread at the next tick."""
        self._inbox.put(("call", fn))

    def on(self, kind: str, fn: Listener) -> None:
        """Subscribe to events: utterance, syllable, action, ate, bump, stage,
        replay, heard, snapshot — or "*" for everything.  Listeners run on the
        agent thread; keep them quick (e.g. push into a queue)."""
        self._listeners[kind].append(fn)

    def off(self, kind: str, fn: Listener) -> None:
        """Unsubscribe a listener added with :meth:`on`."""
        if fn in self._listeners.get(kind, []):
            self._listeners[kind].remove(fn)

    def add_tick_hook(self, fn: Callable[["Agent"], None]) -> None:
        """Call ``fn(agent)`` at the end of every 10 ms tick, on the agent's
        thread.  For things that must react within milliseconds (e.g. a
        teacher's reflexes); keep it cheap."""
        self._tick_hooks.append(fn)

    def remove_tick_hook(self, fn: Callable[["Agent"], None]) -> None:
        if fn in self._tick_hooks:
            self._tick_hooks.remove(fn)

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        data = dict(data)
        data.setdefault("t", self.t_ms / 1000.0)
        data["kind"] = kind
        if kind not in ("syllable", "snapshot", "action", "thought"):
            self.log.append((self.t_ms / 1000.0, kind, data))
        for fn in self._listeners.get(kind, ()):
            fn(data)
        for fn in self._listeners.get("*", ()):
            fn(data)

    # ------------------------------------------------------------ the loop
    def tick(self) -> None:
        """Advance one tick (10 ms of simulated life)."""
        b, body = self.brain, self.body
        while True:
            try:
                kind, payload = self._inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "text":
                text, valence = parse_social_signals(payload)
                body.hear(text, valence)
                self._emit("heard", {"text": payload, "valence": valence})
            elif kind == "call":
                payload(self)
        body.sense(self.tick_ms)
        b.refresh()
        for _ in range(int(self.tick_ms)):
            b.step()
        b.tick(self.tick_ms, energy=body.energy)
        events = body.act(self.t_ms)
        dt_s = self.tick_ms / 1000.0
        self.world.step(dt_s)
        body.metabolism(dt_s)
        self.t_ms += self.tick_ms
        self._ticks += 1
        for kind, data in events:
            self._emit(kind, data)
        for kind, data in b.pop_events():
            self._emit(kind, data)
        if self._ticks % self._snapshot_every == 0:
            self._snapshot()
            self._emit("snapshot", self.latest)
        for fn in self._tick_hooks:
            fn(self)

    def run(self, seconds: float | None = None, realtime: bool = False, speed: float = 1.0) -> None:
        """Live for ``seconds`` of simulated time (forever if None).

        realtime: pace the simulation to the wall clock (x ``speed``);
            otherwise run as fast as the computer allows.
        """
        self._stop.clear()
        sim0, wall0 = self.t_ms, time.perf_counter()
        pace_sim, pace_wall = sim0, wall0
        end = None if seconds is None else self.t_ms + seconds * 1000.0
        while not self._stop.is_set() and (end is None or self.t_ms < end):
            self.tick()
            if self._ticks % 5:
                continue
            now = time.perf_counter()
            if now > wall0:
                self.speed_measured = (self.t_ms - sim0) / 1000.0 / (now - wall0)
            if realtime:
                target = pace_wall + (self.t_ms - pace_sim) / 1000.0 / speed
                delay = target - now
                if delay > 0:
                    self.lagging = False
                    self._stop.wait(delay)
                elif delay < -0.5:  # cannot keep up: drop the backlog instead of rushing
                    self.lagging = True
                    pace_sim, pace_wall = self.t_ms, now

    def start(self, speed: float = 1.0, realtime: bool = True) -> "Agent":
        """Start living in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(target=self.run, kwargs={"realtime": realtime, "speed": speed},
                                        name=f"{self.name}-life", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ views
    def _snapshot(self) -> None:
        snap = self.brain.snapshot()
        snap["energy"] = self.body.energy
        snap["sim_time_s"] = self.t_ms / 1000.0
        snap["speed"] = self.speed_measured
        snap["speaking"] = self.body.voice.partial() if self.body.voice else ""
        snap["stats"] = dict(self.body.stats)
        snap["action"] = (self.body.motor.labels[self.body.motor.current]
                          if self.body.motor and self.body.motor.current >= 0 else "")
        self.latest = snap

    def describe(self) -> str:
        s = self.latest
        d = s["drives"]
        return (f"[{s['clock']} {s['stage']}] hunger {d['hunger']:.2f} fatigue {d['fatigue']:.2f} "
                f"boredom {d['boredom']:.2f} loneliness {d['loneliness']:.2f} | "
                f"DA {s['modulators']['dopamine_tonic']:.2f} NE {s['modulators']['noradrenaline']:.2f} "
                f"ACh {s['modulators']['acetylcholine']:.2f} 5HT {s['modulators']['serotonin']:.2f}")

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "t_ms": self.t_ms,
            "spec": self.brain.spec,
            "world_spec": self.world_spec,
            "world": self.world.state_dict(),
            "brain": self.brain.state_dict(),
            "body": self.body.state_dict(),
        }

    def snapshot_state(self, timeout: float = 10.0) -> dict[str, Any]:
        """A consistent deep copy of the whole state, taken on the agent's own
        thread if it is alive (so it never sees half a tick)."""
        if not self.alive:
            return copy.deepcopy(self.state_dict())
        done = threading.Event()
        holder: dict[str, Any] = {}

        def grab(agent: "Agent") -> None:
            holder["s"] = copy.deepcopy(agent.state_dict())
            done.set()

        self.call_soon(grab)
        if not done.wait(timeout):
            raise RuntimeError("agent thread did not respond")
        return holder["s"]

    def save(self, path: str | Path) -> Path:
        """Write the whole creature (weights, memories, drives, world) to disk."""
        return save_state(self.snapshot_state(), path)

    def clone(self) -> "Agent":
        """An independent copy of the creature as it is right now — same
        synapses, memories, drives and world — e.g. to test it without
        disturbing it.  Listeners and tick hooks are not copied."""
        return type(self).from_state(self.snapshot_state())

    @classmethod
    def load(cls, path: str | Path, world: World | None = None) -> "Agent":
        return cls.from_state(load_state(path), world)

    @classmethod
    def from_state(cls, s: dict[str, Any], world: World | None = None) -> "Agent":
        brain = build_brain(s["spec"])
        brain.load_state_dict(s["brain"])
        ws = s.get("world_spec", {"type": "void", "kwargs": {}})
        if world is None:
            wcls = WORLDS.get(ws["type"], EmptyWorld)
            world = wcls(**ws.get("kwargs", {})) if wcls is not EmptyWorld else EmptyWorld()
            world.load_state_dict(s["world"])
        cfg = s["body"].get("cfg")
        body_cfg = None
        if cfg:
            cfg = dict(cfg)
            cfg["motor"] = MotorConfig(**cfg.get("motor", {}))
            cfg["voice"] = VoiceConfig(**cfg.get("voice", {}))
            if "hands" in cfg:
                cfg["hands"] = MotorConfig(**cfg["hands"])
            body_cfg = BodyConfig(**cfg)
        agent = cls(brain, world, body_cfg, name=s.get("name", "anima"), world_spec=ws)
        agent.body.load_state_dict(s["body"])
        agent.t_ms = float(s["t_ms"])
        agent._snapshot()
        return agent


__all__ = ["Agent"]
